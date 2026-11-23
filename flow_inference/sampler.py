"""Sampling: integrate a flow's probability-flow ODE (or SDE) from noise to data.

:class:`IntegratorSampler` draws Gaussian noise at ``t = 1`` and integrates backwards
to ``t ~ 0`` with the flow's drift, plus an optional guidance correction (see
:mod:`flow_inference.guidance`) that steers the samples towards a condition. Steps can
be fixed (Euler, Heun or RK4 on a chosen time grid) or chosen adaptively from an error
estimate.
"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from flow_inference.flow_utils import compute_snr_schedule
from flow_inference.flows import BaseFlow
from flow_inference.guidance import GuidanceApproximation


class IntegratorSampler(nn.Module):
    """Sampler that integrates a flow's probability-flow ODE or reverse SDE.

    Integrates ``dx/dt = v(x, t)`` backwards in time from ``t = 1`` (pure noise) to
    ``t ~ 0`` (data), where ``v`` is the flow's drift plus an optional guidance
    correction. With ``sde=True`` it integrates the reverse-time SDE instead, injecting
    noise at each step. ``device`` and ``dtype`` are read from the flow.

    Args:
        flow: the flow to sample from.
    """

    def __init__(self, flow: BaseFlow):
        super().__init__()
        self.flow = flow

    # Read from the flow so they follow .to() / .double() on it.
    @property
    def device(self) -> torch.device:
        return self.flow.device

    @property
    def dtype(self) -> torch.dtype:
        return self.flow.dtype

    def _get_time_schedule(self, num_steps: int, time_schedule: str) -> torch.Tensor:
        """Decreasing integration times from 1 (noise) to about 0 (data).

        Args:
            num_steps: number of steps.
            time_schedule: ``"linear"`` (uniform in t), ``"quadratic"`` (denser near
                t = 0) or ``"snr"`` (equal steps in log signal-to-noise ratio).

        Returns:
            ``num_steps + 1`` times, from 1 down to 1e-10.
        """
        if time_schedule == "linear":
            ts = torch.linspace(1.0, 1e-10, num_steps + 1, device=self.device)
        elif time_schedule == "quadratic":
            # Square the unit grid, then rescale, so the endpoint is 1e-10 (squaring
            # it directly would give 1e-20).
            u = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device) ** 2
            ts = 1e-10 + u * (1.0 - 1e-10)
        elif time_schedule == "snr":
            ts = compute_snr_schedule(self.flow.schedule, num_steps, device=self.device)
        else:
            raise ValueError(
                f"Unsupported time schedule: {time_schedule}. Must be 'linear', "
                "'quadratic', or 'snr'."
            )
        return ts

    def _vp_step_coeffs(self, t_curr: torch.Tensor, dt: torch.Tensor):
        """Coefficients ``(decay, noise_std)`` of a reverse VP SDE step.

        A step from ``t`` to ``t - dt`` is ``decay * z - 2 dt v + noise_std * eps``,
        with ``v`` the probability-flow drift. Doubling ``v`` doubles its score term,
        as the reverse SDE requires, and ``decay = exp(-beta dt / 2)`` cancels the
        extra linear term this adds.
        """
        beta = self.flow.schedule._beta(t_curr)
        return torch.exp(-0.5 * beta * dt), torch.sqrt(beta * dt)

    def sample(
        self,
        num_samples: int,
        num_steps: int | None = None,
        guidance: GuidanceApproximation | None = None,
        time_schedule: str = "snr",
        solver: str = "euler",
        sde: bool = False,
        final_denoise: bool = True,
        verbose: bool = True,
        adaptive: bool = False,
        rtol: float = 1e-2,
        atol: float = 1e-2,
        t_min: float = 1e-3,
        max_steps: int = 10_000,
    ) -> torch.Tensor:
        """Draw samples by integrating the probability-flow ODE (or reverse SDE).

        Args:
            num_samples: number of independent samples.
            num_steps: number of integration steps (default 1000). Fixed steps only:
                with ``adaptive=True`` the controller picks the steps, and passing
                this raises.
            guidance: optional guidance (e.g. :class:`~flow_inference.GuidanceMC`)
                whose correction is added to the drift at every step.
            time_schedule: spacing of the fixed steps: ``"linear"`` (uniform in t),
                ``"quadratic"`` (denser near t = 0) or ``"snr"`` (equal steps in log
                signal-to-noise ratio). Ignored when ``adaptive=True``.
            solver: ``"euler"`` (first order), ``"heun"`` (second order) or
                ``"rk4"`` (fourth order, ODE only).
            sde: if True, integrate the reverse SDE, injecting noise at each step.
                Supports ``solver="euler"`` (Euler-Maruyama) and ``"heun"``.
            final_denoise: if True, replace the last integration step by the
                flow's denoiser ``E[x_0 | x_t]`` (``predict_terminal``). Near
                ``t = 0`` the ODE is stiff, because the score grows like
                ``1 / sigma_t``, so the last steps cannot remove the remaining
                noise and it shows up as grain; the denoiser removes it exactly.
            verbose: show a progress bar.
            adaptive: if True, choose step sizes from an embedded Euler/Heun error
                estimate (see :meth:`_integrate_adaptive`). Supports
                ``solver="euler"`` or ``"heun"``, ODE or SDE. Step statistics are
                stored in ``self.adaptive_stats``.
            rtol: adaptive only: relative tolerance; the local error of each
                coordinate is held below ``atol + rtol * |z|``.
            atol: adaptive only: absolute tolerance.
            t_min: adaptive only: integrate from ``t = 1`` down to ``t_min``, where
                ``final_denoise`` takes over.
            max_steps: adaptive only: cap on the number of trial steps (accepted and
                rejected); reaching it raises.

        Returns:
            Samples ``(num_samples, C, *output_shape)`` in data space, e.g.
            ``(num_samples, C, H, W)`` for a grid flow and ``(num_samples, C, N)`` for
            a 1D field.
        """
        was_training = self.flow.training
        self.flow.eval()

        if adaptive and num_steps is not None:
            raise ValueError(
                "num_steps is chosen by the controller when adaptive=True; set "
                "rtol/atol for accuracy, or max_steps to cap the trial steps."
            )
        if num_steps is None:
            num_steps = 1_000

        z = torch.randn(
            (num_samples, self.flow.channels, self.flow.latent_dim),
            device=self.device,
            dtype=self.flow.dtype,
        )

        def compute_drift(z_eval, t_eval):
            # Guidance takes its own gradients; the flow's drift needs none.
            with torch.no_grad():
                dv = self.flow(z_eval, t_eval)
            if guidance is not None:
                dv = dv + guidance(z_eval, t_eval)
            return dv

        if adaptive:
            z, t_last = self._integrate_adaptive(
                z,
                compute_drift,
                solver,
                sde,
                max_steps,
                rtol,
                atol,
                t_min,
                verbose,
            )
        else:
            z, t_last = self._integrate_fixed(
                z,
                compute_drift,
                num_steps,
                time_schedule,
                solver,
                sde,
                final_denoise,
                verbose,
            )

        if final_denoise:
            with torch.no_grad():
                z = self.flow.predict_terminal(z, t_last)

        # From the diffusion state (e.g. whitened) to data space, then to the flow's
        # display shape.
        f = self.flow._to_data_space(z)
        if was_training:
            self.flow.train()
        return self.flow.to_display(f).detach()

    def _integrate_fixed(
        self,
        z,
        compute_drift,
        num_steps,
        time_schedule,
        solver,
        sde,
        final_denoise,
        verbose,
    ):
        """Integrate on a fixed time grid.

        Returns:
            The final state ``z`` and the last time reached, ``t_last``.
        """
        ts = self._get_time_schedule(num_steps, time_schedule)

        if sde:
            # One randn call for all steps, rather than a kernel launch per step.
            all_noise = torch.randn(
                num_steps, *z.shape, device=self.device, dtype=z.dtype
            )
            noise_iter = iter(all_noise)

        if solver == "euler":
            if sde:
                # Euler-Maruyama for the reverse VP SDE (see _vp_step_coeffs).
                def step_fn(z, t_curr, t_next, dt):
                    drift = compute_drift(z, t_curr)
                    noise = next(noise_iter)
                    decay, noise_std = self._vp_step_coeffs(t_curr, dt)
                    return decay * z - dt * 2.0 * drift + noise_std * noise
            else:

                def step_fn(z, t_curr, t_next, dt):
                    return z - dt * compute_drift(z, t_curr)
        elif solver == "heun":
            if sde:
                # Stochastic Heun: an Euler-Maruyama predictor, then the trapezoidal
                # rule on the drift with the same noise draw and decay.
                def step_fn(z, t_curr, t_next, dt):
                    noise = next(noise_iter)
                    decay, noise_std = self._vp_step_coeffs(t_curr, dt)
                    k1 = compute_drift(z, t_curr)
                    z_hat = decay * z - dt * 2.0 * k1 + noise_std * noise
                    k2 = compute_drift(z_hat, t_next)
                    return decay * z - dt * (k1 + k2) + noise_std * noise
            else:

                def step_fn(z, t_curr, t_next, dt):
                    k1 = compute_drift(z, t_curr)
                    k2 = compute_drift(z - dt * k1, t_next)
                    return z - dt * (k1 + k2) / 2

        elif solver == "rk4":
            if sde:
                raise ValueError(
                    "SDE integration only supports Euler or Heun "
                    "(set solver='euler' or solver='heun')"
                )
            else:

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

        # With final_denoise, stop one step early: the caller applies the denoiser at
        # ts[-2] in place of the stiff last step. At ts[-1] ~ 0 the denoiser would do
        # nothing (alpha_t -> 1, sigma_t -> 0).
        last_step = num_steps - 1 if final_denoise else num_steps
        for i in tqdm(range(last_step), disable=not verbose):
            t_curr = ts[i]
            t_next = ts[i + 1]
            dt = t_curr - t_next  # positive: time runs backwards
            z = step_fn(z, t_curr, t_next, dt)
        return z, ts[last_step]

    def _integrate_adaptive(
        self,
        z,
        compute_drift,
        solver,
        sde,
        max_steps,
        rtol,
        atol,
        t_min,
        verbose,
    ):
        """Integrate from t=1 to ``t_min`` with error-controlled step sizes.

        Each trial step forms the Euler (or Euler-Maruyama) state ``z_lo`` and the
        Heun state ``z_hi`` from the same drift evaluations and noise draw; their gap
        estimates the Euler local error, and the noise cancels in it, so the controller
        tracks how fast the drift changes rather than the injected noise. ``solver``
        picks which state to keep:

        - "euler": keep ``z_lo``. ``k2`` was evaluated there, so it is reused as the
          next ``k1`` (first-same-as-last): one drift evaluation per trial step.
        - "heun": keep ``z_hi`` (local extrapolation): two per trial step.

        One ``dt`` is shared across the batch, set by the worst sample. A rejected SDE
        step redraws its noise (Jolicoeur-Martineau et al., 2021), which biases the
        sampler slightly towards low-noise increments; a Brownian bridge would not.

        Step statistics (drift evaluations, accepted and rejected steps, and the
        accepted times) are stored in ``self.adaptive_stats``.

        Returns:
            The final state ``z`` and the time reached, ``t_min``.
        """
        if solver not in ("euler", "heun"):
            raise ValueError(
                f"Adaptive integration supports solver='euler' or 'heun', got {solver}."
            )

        def as_t(t):
            return torch.as_tensor(t, device=self.device, dtype=z.dtype)

        # SDE steps carry a decay, a doubled drift and noise (see _vp_step_coeffs).
        drift_scale = 2.0 if sde else 1.0
        safety, min_factor, max_factor = 0.9, 0.2, 5.0
        t, dt = 1.0, 1e-2
        k1 = compute_drift(z, as_t(t))
        nfe, accepted, rejected = 1, 0, 0
        ts = [t]
        # The step count isn't known up front, so the bar shows the fraction of
        # [t_min, 1] covered, with the controller's state alongside.
        pbar = tqdm(
            total=1.0 - t_min,
            desc=f"adaptive {solver} {'SDE' if sde else 'ODE'}",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}{postfix}",
            disable=not verbose,
        )
        while t > t_min:
            if accepted + rejected >= max_steps:
                raise RuntimeError(
                    f"Adaptive integration hit max_steps={max_steps} trial steps at "
                    f"t={t:.3e}; loosen rtol/atol, raise t_min or raise max_steps."
                )
            dt = min(dt, t - t_min)
            # Snap the last step onto t_min so rounding can't leave a sliver behind.
            t_next = t_min if dt == t - t_min else t - dt
            if sde:
                decay, noise_std = self._vp_step_coeffs(as_t(t), as_t(dt))
                base = decay * z + noise_std * torch.randn_like(z)
            else:
                base = z
            z_lo = base - drift_scale * dt * k1
            k2 = compute_drift(z_lo, as_t(t_next))
            nfe += 1
            z_hi = base - drift_scale * dt * (k1 + k2) / 2

            err = z_hi - z_lo
            scale = atol + rtol * torch.maximum(z.abs(), z_lo.abs())
            err_norm = (err / scale).pow(2).flatten(1).mean(1).sqrt().max().item()

            if err_norm <= 1.0:
                # Set progress from t itself, so summed dt rounding can't overshoot.
                pbar.update(min(1.0 - t_next, pbar.total) - pbar.n)
                t = t_next
                accepted += 1
                ts.append(t)
                if solver == "euler":
                    z, k1 = z_lo, k2
                else:
                    # Heun's state leaves the cached drift stale.
                    z = z_hi
                    if t > t_min:
                        k1 = compute_drift(z, as_t(t))
                        nfe += 1
            else:
                rejected += 1

            # Euler's local error is O(dt^2), hence the 1/2 exponent.
            factor = safety * err_norm**-0.5 if err_norm > 0 else max_factor
            dt *= min(max(factor, min_factor), max_factor)
            # refresh=False: tqdm redraws on the next update(), rate-limited.
            pbar.set_postfix_str(
                f"t={t:.2e}, dt={dt:.1e}, nfe={nfe}, rej={rejected}", refresh=False
            )
        pbar.close()

        self.adaptive_stats = {
            "nfe": nfe,
            "accepted": accepted,
            "rejected": rejected,
            "ts": torch.tensor(ts),
        }
        print(
            f"Adaptive {solver} ({'SDE' if sde else 'ODE'}): {nfe} drift evals, "
            f"{accepted} accepted / {rejected} rejected steps to t={t:.1e}"
        )
        return z, as_t(t)
