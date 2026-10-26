import math

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from flow_inference.conditions import BaseCondition
from flow_inference.flow_utils import compute_snr_schedule
from flow_inference.flows import BaseFlow, terminal_draws
from flow_inference.guidance import GuidanceApproximation, _ensemble_logp


class PCNCorrector(nn.Module):
    """Gradient-free preconditioned Crank-Nicolson corrector for the sampling loop.

    A *corrector*, not guidance: it does not contribute to the drift, it moves the state
    itself. After each integration step at time ``t`` it runs Metropolis-Hastings moves
    whose target is the correct intermediate law

        ``p(z_t | y)  ∝  p(z_t) p(y | z_t)``

    using pCN proposals ``z' = sqrt(1 - rho^2) z + rho * xi``, ``xi ~ N(0, I)``. The
    flow's marginals are N(0, I), which the proposal preserves, so the accept ratio
    needs only likelihood estimates; carrying an unbiased estimate with the state makes
    this a valid pseudo-marginal chain for any ``num_mc_samples``.

    Use it to clean up after an approximate guidance term, or on its own
    (``guidance=None``) as a fully gradient-free sampler. Pair with ``sde=True`` if you
    also want the diffusion to rejuvenate the particles.

    Args:
        num_steps: pCN moves per integration step. Cost is
            ``(1 + num_steps) * num_mc_samples`` condition evaluations per step.
        rho: initial step size in (0, 1]. ``rho = 1`` is an independence sampler from the
            prior; small ``rho`` proposes locally. Only a starting point unless
            ``target_accept`` is None — see below.
        num_mc_samples: draws in the likelihood estimate. More reduces the variance of
            the estimate and so raises acceptance, but does not change what is targeted.
        target_accept: adapt ``rho`` toward this acceptance rate, or None to hold it
            fixed. Keep adaptation on: the useful ``rho`` spans orders of magnitude along
            the schedule (the likelihood is nearly flat at ``t = 1`` and sharp as
            ``t -> 0``), so no fixed value works throughout. The default 0.25 is a sharp
            optimum: higher targets push ``rho`` toward the independence sampler, lower
            ones shrink it until the chain stops moving.
    """

    def __init__(
        self,
        flow: BaseFlow,
        condition: BaseCondition,
        num_steps: int = 5,
        rho: float = 0.1,
        num_mc_samples: int = 4,
        target_accept: float | None = 0.25,
        record_history: bool = False,
    ):
        super().__init__()
        if not flow.has_standard_normal_marginals:
            raise ValueError(
                f"{type(flow).__name__} does not have N(0, I) marginals at every t, so "
                "a pCN proposal is not prior-preserving and the accept ratio below "
                "would be wrong (it omits the ratio of flow marginals). PCNCorrector "
                "supports the closed-form Gaussian/GP/increment flows; for a learned "
                "NeuralFlow use guidance instead."
            )
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        self.flow = flow
        self.condition = condition
        condition.bind_output_shape(flow.output_shape)
        self.num_steps = num_steps
        self.num_mc_samples = num_mc_samples
        self.target_accept = target_accept
        self.record_history = record_history
        self._log_rho = math.log(rho)
        self._n_adapt = 0
        self._history: list[dict] = []

    @property
    def rho(self) -> float:
        """Current pCN step size (tracks adaptation)."""
        return math.exp(self._log_rho)

    @property
    def history(self) -> list[dict]:
        """Per-step ``{t, rho, accept_rate, log_lik_mean}`` while ``record_history`` was set.

        A collapsing ``accept_rate`` that adaptation cannot recover means the target is
        sharpening faster than local moves can track — raise ``num_mc_samples`` (a noisy
        likelihood estimate rejects good moves) or ``num_steps`` before reaching for a
        smaller ``rho``.
        """
        return self._history

    def reset_history(self) -> None:
        """Clear recorded diagnostics (call between sampling runs)."""
        self._history = []

    @property
    def acceptance_rate(self) -> float:
        """Mean acceptance over the recorded history (NaN if nothing was recorded)."""
        if not self._history:
            return float("nan")
        return sum(r["accept_rate"] for r in self._history) / len(self._history)

    def _log_lik_hat(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Unbiased-in-``p`` estimate of ``log p(y | z_t)``, shape ``(B,)``.

        ``logsumexp - log S`` is the log of the *arithmetic* mean of the likelihoods, so
        the quantity being averaged is unbiased for ``p(y | z_t)``; that is what the
        pseudo-marginal argument needs (the log itself is of course biased low).

        Fresh noise every call, rather than ``flow.sample_terminal``'s cached noise: this
        differentiates nothing, and a frozen error would be the *same* error at every step
        (see :meth:`~flow_inference.flows.GaussianFlow.sample_terminal`).
        """
        B, C, N = z.shape
        S = self.num_mc_samples
        eps = torch.randn((B, S, C, N), device=z.device, dtype=z.dtype)
        z0 = terminal_draws(self.flow, z, t, eps)
        logp = _ensemble_logp(self.flow, self.condition, z0)  # (B, S)
        return torch.logsumexp(logp, dim=1) - math.log(S)

    @torch.no_grad()
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run the pCN moves at time ``t`` and return the (possibly moved) state."""
        if self.num_steps < 1:
            return z
        log_lik = self._log_lik_hat(z, t)  # carried with the state across moves
        accepted = torch.zeros_like(log_lik)
        for _ in range(self.num_steps):
            rho = self.rho
            xi = torch.randn_like(z)
            z_prop = math.sqrt(max(1.0 - rho * rho, 0.0)) * z + rho * xi
            log_lik_prop = self._log_lik_hat(z_prop, t)
            # log u < log ratio, so an overflowing exp() is never formed. Non-finite
            # ratios (a proposal the condition cannot score) reject.
            log_ratio = torch.nan_to_num(
                log_lik_prop - log_lik, nan=-torch.inf, posinf=0.0
            )
            accept = torch.log(torch.rand_like(log_lik)) < log_ratio
            z = torch.where(accept[:, None, None], z_prop, z)
            log_lik = torch.where(accept, log_lik_prop, log_lik)
            accepted += accept.to(accepted.dtype)

        accept_rate = (accepted / self.num_steps).mean().item()
        if self.target_accept is not None:
            # Robbins-Monro on log rho: nudge up when we are accepting too much, down
            # when too little, with a 1/sqrt(n) gain so it settles rather than chatters.
            # Clamped to keep sqrt(1 - rho^2) real. This makes the chain inhomogeneous,
            # which is harmless here — the target is annealing anyway, so there is no
            # fixed distribution for the adaptation to bias.
            self._n_adapt += 1
            step = (accept_rate - self.target_accept) / math.sqrt(self._n_adapt)
            self._log_rho = min(max(self._log_rho + step, math.log(1e-6)), 0.0)
        if self.record_history:
            self._history.append(
                {
                    "t": float(t),
                    "rho": self.rho,
                    "accept_rate": accept_rate,
                    "log_lik_mean": log_lik.mean().item(),
                    "num_mc_samples": self.num_mc_samples,
                }
            )
        return z


class IntegratorSampler(nn.Module):
    """Integrator for the probability-flow ODE (and SDE variants).

    Integrates  dx/dt = v(x, t)  backwards in time from t=1 (pure noise)
    to t≈0 (data), where v is the sum of the unconditional vector field and
    any optional guidance correction.  Set sde=True to add noise injection
    at each step (Euler-Maruyama for solver="euler"; EDM-style for solver="heun").
    """

    def __init__(self, flow: BaseFlow):
        super().__init__()
        self.flow = flow

    # Read through the flow so they follow .to()/.double() on it.
    @property
    def device(self) -> torch.device:
        return self.flow.device

    @property
    def dtype(self) -> torch.dtype:
        return self.flow.dtype

    def _get_time_schedule(self, num_steps: int, time_schedule: str) -> torch.Tensor:
        """Build the decreasing integration grid t_0 = 1 (noise) … t_{num_steps} ≈ 0 (data).

        Returns a tensor of ``num_steps + 1`` times. Options:
            - "linear"             : uniform spacing in t.
            - "quadratic"          : steps denser near t=0.
            - "snr"                : equal log-SNR increments per step.
        """
        if time_schedule == "linear":
            ts = torch.linspace(1.0, 1e-10, num_steps + 1, device=self.device)
        elif time_schedule == "quadratic":
            # Square the *unit* grid, then rescale, so the endpoint stays at 1e-10
            # rather than becoming 1e-20 (squaring the endpoint too).
            u = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device) ** 2
            ts = 1e-10 + u * (1.0 - 1e-10)
        elif time_schedule == "snr":
            ts = compute_snr_schedule(self.flow.schedule, num_steps, device=self.device)
        else:
            raise ValueError(
                f"Unsupported time schedule: {time_schedule}, Must be 'linear', "
                "'quadratic', or 'snr'."
            )
        return ts

    def _vp_step_coeffs(self, t_curr: torch.Tensor, dt: torch.Tensor):
        """VP Euler-Maruyama step coefficients ``(decay, noise_std)``."""
        beta = self.flow.schedule._beta(t_curr)
        return torch.exp(-0.5 * beta * dt), torch.sqrt(beta * dt)

    def sample(
        self,
        num_samples: int,
        num_steps: int = 1_000,
        guidance: GuidanceApproximation | None = None,
        corrector: PCNCorrector | None = None,
        time_schedule: str = "snr",
        solver: str = "euler",
        sde: bool = False,
        final_denoise: bool = True,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Generate samples by integrating the probability-flow ODE (or SDE).

        Args:
            num_samples:   Number of independent trajectories to simulate.
            num_steps:     Number of integration steps.
            final_denoise: If True (default), replace the final integrated state with
                           the analytic Tweedie denoiser E[x_0 | x_t] = predict_terminal
                           at the last integration time. Near t→0 the probability-flow
                           ODE is stiff (the score ∝ -ε/σ_t blows up as σ_t→0), so the
                           last Euler/Heun steps cannot fully remove the residual noise
                           and it shows up as grain. The analytic denoiser removes it
                           exactly instead of integrating through the stiff tail.
            verbose:       Show tqdm progress bar (default True).
            guidance:      Optional guidance object that adds a correction term
                           to the drift at each step (e.g. classifier-free or
                           observation-conditional guidance).
            corrector:     Optional :class:`PCNCorrector` applied to the *state* after
                           each integration step (predictor-corrector). Gradient-free,
                           and usable with or without ``guidance``.
            time_schedule: Step-size strategy. Options:
                           - "linear"    : uniform spacing in t.
                           - "quadratic" : steps denser near t=0.
                           - "snr"       : equal log-SNR increments per step.
            solver:        Base integration method. Options:
                           - "euler" : 1st-order Forward Euler.
                           - "heun"  : 2nd-order Heun / RK2.
                           - "rk4"   : 4th-order Runge-Kutta (deterministic only).
            sde:           If True, inject noise at each step (stochastic sampling).
                           Compatible with solver="euler" (Euler-Maruyama) and
                           solver="heun" (stochastic Heun for the VP SDE).
        Returns:
            Tensor of shape (num_samples, C, *output_shape) in the data space
            (output_shape defaults to (N,); e.g. (H, W) for grid flows).
        """
        was_training = self.flow.training
        self.flow.eval()

        if corrector is not None and corrector.flow is not self.flow:
            raise ValueError(
                "corrector was built on a different flow than this sampler; its pCN "
                "accept ratio and terminal-posterior draws would use the wrong "
                "schedule and whitening."
            )

        z = torch.randn(
            (num_samples, self.flow.channels, self.flow.latent_dim),
            device=self.device,
            dtype=self.flow.dtype,
        )

        ts = self._get_time_schedule(num_steps, time_schedule)

        # Define drift once — avoids re-creating a closure every step
        def compute_drift(z_eval, t_eval):
            with torch.no_grad():
                dv = self.flow(z_eval, t_eval)
            if guidance is not None:
                dv = dv + guidance(z_eval, t_eval)
            return dv

        if sde:
            # One large randn instead of num_steps small ones — avoids repeated CUDA kernel launches.
            all_noise = torch.randn(
                num_steps, *z.shape, device=self.device, dtype=z.dtype
            )
            noise_iter = iter(all_noise)

        # Select step function once — avoids re-evaluating the solver branch every step
        if solver == "euler":
            if sde:
                # Euler-Maruyama: deterministic drift + VP diffusion noise injection
                def step_fn(z, t_curr, t_next, dt):
                    drift = compute_drift(z, t_curr)
                    noise = next(noise_iter)
                    decay, noise_std = self._vp_step_coeffs(t_curr, dt)
                    return decay * z - dt * 2.0 * drift + noise_std * noise
            else:
                # standard Euler step for the probability flow ODE
                def step_fn(z, t_curr, t_next, dt):
                    return z - dt * compute_drift(z, t_curr)
        elif solver == "heun":
            if sde:
                # Stochastic Heun for the VP SDE, matching the Euler-Maruyama scheme.
                # Predictor mirrors the Euler SDE step; corrector applies the trapezoidal
                # rule to the nonlinear drift while reusing the same noise draw and decay.
                def step_fn(z, t_curr, t_next, dt):
                    noise = next(noise_iter)
                    decay, noise_std = self._vp_step_coeffs(t_curr, dt)
                    k1 = compute_drift(z, t_curr)
                    z_hat = decay * z - dt * 2.0 * k1 + noise_std * noise
                    k2 = compute_drift(z_hat, t_next)
                    return decay * z - dt * (k1 + k2) + noise_std * noise
            else:
                # Heun step for the probability flow ODE
                def step_fn(z, t_curr, t_next, dt):
                    k1 = compute_drift(z, t_curr)
                    k2 = compute_drift(z - dt * k1, t_next)
                    return z - dt * (k1 + k2) / 2

        elif solver == "rk4":
            if sde:
                raise ValueError(
                    "SDE integration only supports Euler or Heun (set solver='euler or solver='heun')"
                )
            else:
                # RK4 step for the probability flow ODE
                def step_fn(z, t_curr, t_next, dt):
                    t_mid = (t_curr + t_next) / 2
                    k1 = compute_drift(z, t_curr)
                    k2 = compute_drift(z - (dt / 2) * k1, t_mid)
                    k3 = compute_drift(z - (dt / 2) * k2, t_mid)
                    k4 = compute_drift(z - dt * k3, t_next)
                    return z - dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise ValueError(
                f"Unknown solver: {solver}. Must be 'euler', 'heun', or 'rk4'."
            )

        # When denoising at the end, stop integrating one step early and replace the
        # final stiff ODE step (t → ts[-1] ≈ 0) with the analytic Tweedie denoiser
        # evaluated at ts[-2]. The denoiser is a no-op as t→0 (α_t→1, σ_t→0), so it
        # must be applied where σ_t still has signal — hence ts[-2], not ts[-1].
        last_step = num_steps - 1 if final_denoise else num_steps
        for i in tqdm(range(last_step), disable=not verbose):
            t_curr = ts[i]
            t_next = ts[i + 1]
            dt = t_curr - t_next  # positive: we step backwards in time
            z = step_fn(z, t_curr, t_next, dt)
            if corrector is not None:
                # Corrector runs at the time the state now sits at, not t_curr.
                z = corrector(z, t_next)

        if final_denoise:
            # E[x_0 | x_t] at the last integrated time removes the residual grain that
            # the stiff tail of the ODE leaves behind.
            with torch.no_grad():
                z = self.flow.predict_terminal(z, ts[num_steps - 1])

        # Map from the internal (e.g. whitened) representation to data space, then
        # reshape to the flow's natural display shape ((B, C, H, W) for grid flows,
        # unchanged (B, C, N) for 1D fields / paths) so callers needn't reshape.
        f = self.flow._to_data_space(z)
        if was_training:  # leave the caller's train/eval mode as we found it
            self.flow.train()
        return self.flow.to_display(f).detach()
