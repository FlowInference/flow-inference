import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from flow_inference.conditions import BaseCondition
from flow_inference.flow_utils import clip_vector_field
from flow_inference.flows import BaseFlow


def _snis_weights(logp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-normalised importance weights and their ESS, from ``(B, S)`` log-densities.

    Returns ``(weights, ess)`` with ``weights`` summing to 1 along S and
    ``ess = 1 / sum(w^2)`` in ``[1, S]``.
    """
    log_weights = logp - torch.logsumexp(logp, dim=1, keepdim=True)
    weights = torch.exp(log_weights)
    return weights, 1.0 / (weights**2).sum(dim=1)


def _scrub_logp(logp: torch.Tensor) -> torch.Tensor:
    """Make log-likelihoods safe to weight: NaN and -inf become a huge negative
    (zero weight), +inf becomes 0."""
    return torch.nan_to_num(logp, nan=-1e38, posinf=0.0, neginf=-1e38)


def _ensemble_logp(
    flow: BaseFlow, condition: BaseCondition, z0: torch.Tensor
) -> torch.Tensor:
    """Scrubbed ``log p(condition | x)`` for an ensemble of whitened draws.

    ``z0`` is ``(B, S, C, N)``; each draw is mapped to data space and scored, giving
    ``(B, S)``.
    """
    B, S, C, N = z0.shape
    logp = condition.logp(flow._to_data_space(z0.reshape(B * S, C, N)))
    return _scrub_logp(logp.reshape(B, S))


def _fisher_coef(flow: BaseFlow, t, floor: float) -> torch.Tensor:
    """``sqrt(alpha_t) / (1 - alpha_t)``, the Fisher-identity scale on a posterior-mean
    deviation. Uses the expm1-based ``1 - alpha_t``, which stays accurate as t -> 0."""
    one_minus_alpha_t = flow.schedule._one_minus_alpha_2(t).clamp(min=floor)
    return torch.sqrt(flow.schedule._alpha_2(t)) / one_minus_alpha_t


class GuidanceApproximation(nn.Module, ABC):
    """Base class for guidance term.
    Approximates  (-0.5 * beta_t) * ∇_{x_t} log p(y | x_t)
    """

    def __init__(
        self,
        condition: BaseCondition,
        flow: BaseFlow | None = None,
        max_norm: float | None = 1e2,
    ):
        super().__init__()
        self.condition = condition
        self.flow = flow
        # ``max_norm`` is the norm the tanh squash saturates toward. Treat it as a
        # total path-length budget: once the squash saturates, every step has length
        # max_norm and only the gradient's direction survives. ``history`` reports the
        # attenuation so you can see which regime you are in.
        self.max_norm = max_norm
        # Give grid-aware conditions the flow's output shape, so condition.as_grid()
        # works without passing (H, W) to the condition.
        if flow is not None:
            condition.bind_output_shape(getattr(flow, "output_shape", None))
        # Per-step diagnostics, appended by _to_drift when record_history is on.
        # See :meth:`history` for the recorded fields.
        self.record_history = False
        self._history: list[dict] = []
        self._prev_dir: torch.Tensor | None = None
        self.ess: torch.Tensor | None = None

    def _require_flow_method(self, name: str) -> None:
        """Guard that the flow supports the estimator this guidance is built on."""
        assert hasattr(self.flow, name), (
            f"BaseFlow must implement {name} for {type(self).__name__}"
        )

    def _to_drift(
        self,
        grad_logp: torch.Tensor,
        beta_t: torch.Tensor,
        t: float | None = None,
        ess: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Turn ∇ log p(y | ·) into the detached drift correction.

        Shared tail of every guidance: scrub non-finite entries, scale by −β_t/2, and
        (unless ``max_norm`` is None) squash the norm so the ODE solver sees a
        Lipschitz field. With ``record_history`` set, each call appends the pre- and
        post-squash norms to :attr:`history` so callers can see how hard the squash is
        biting (see :meth:`history`). ``ess`` is the importance-weight effective sample
        size, passed by the Monte-Carlo estimators.
        """
        n_nonfinite = (
            int((~torch.isfinite(grad_logp)).sum()) if self.record_history else 0
        )
        grad_logp = torch.nan_to_num(grad_logp, nan=0.0, posinf=0.0, neginf=0.0)
        dv = (-0.5 * beta_t) * grad_logp
        raw = dv
        if self.max_norm is not None:
            dv = clip_vector_field(dv, self.max_norm)
        if ess is not None:
            self.ess = ess.detach()
        if self.record_history:
            self._record(t, raw, dv, n_nonfinite, ess)
        return dv.detach()

    def _record(self, t, raw, clipped, n_nonfinite, ess=None) -> None:
        """Append one step's guidance-magnitude diagnostics to :attr:`_history`."""
        # Per-(sample, channel) norms over the field axis — the same axis
        # clip_vector_field squashes along, so these are what the squash acts on.
        raw_norm = torch.linalg.vector_norm(raw.detach(), dim=-1)  # (B, C)
        out_norm = torch.linalg.vector_norm(clipped.detach(), dim=-1)  # (B, C)
        # Cosine with the previous step's direction. Persistent (~+1) means the flow is
        # tracking; sustained negatives mean it is chattering across the constraint and
        # burning path budget without net displacement.
        unit = clipped.detach() / out_norm.unsqueeze(-1).clamp(min=1e-30)
        if self._prev_dir is not None and self._prev_dir.shape == unit.shape:
            persistence = (unit * self._prev_dir).sum(-1).mean().item()
        else:
            persistence = float("nan")
        self._prev_dir = unit
        self._history.append(
            {
                "t": float(t) if t is not None else float("nan"),
                "max_norm": self.max_norm,
                "raw_norm_mean": raw_norm.mean().item(),
                "raw_norm_max": raw_norm.max().item(),
                "raw_norm_min": raw_norm.min().item(),
                "out_norm_mean": out_norm.mean().item(),
                "out_norm_max": out_norm.max().item(),
                # Attenuation actually applied: tanh(r)/r with r = raw/max_norm.
                # 1.0 = untouched, 0.5 = halved, -> 0 = saturated.
                "attenuation_mean": (out_norm / raw_norm.clamp(min=1e-12))
                .mean()
                .item(),
                "attenuation_min": (out_norm / raw_norm.clamp(min=1e-12)).min().item(),
                # Fraction of (sample, channel) vectors over the squash threshold —
                # the tanh is smooth, so this is "where it starts to bite", not a
                # hard on/off.
                "frac_over_max": (
                    (raw_norm > self.max_norm).float().mean().item()
                    if self.max_norm is not None
                    else 0.0
                ),
                "n_nonfinite": n_nonfinite,
                "persistence": persistence,
                # Importance-weight effective sample size, in [1, S]. Only the
                # Monte-Carlo estimators supply it; ~1 means a single draw dominates
                # and the estimator has silently degenerated to plain DPS.
                "ess_mean": float("nan") if ess is None else ess.mean().item(),
                "ess_min": float("nan") if ess is None else ess.min().item(),
                "num_mc_samples": getattr(self, "num_mc_samples", None),
            }
        )

    @property
    def history(self) -> list[dict]:
        """Per-step guidance diagnostics recorded while ``record_history`` was set.

        One dict per :meth:`forward` call, in call order (the sampler integrates
        t=1 → 0, so entries run noise → data). Keys:

          ``t``                 integration time of the step
          ``raw_norm_*``        mean/max/min pre-squash ‖dv‖ over (sample, channel)
          ``out_norm_*``        the same after the tanh squash
          ``attenuation_*``     out/raw ratio: 1 = untouched, → 0 = saturated
          ``frac_over_max``     fraction of vectors with ‖dv‖ > ``max_norm``
          ``persistence``       cos(this step's direction, previous step's); ~+1 =
                                tracking, negative = chattering across the constraint
          ``ess_mean``/``ess_min``  importance-weight effective sample size (MC only;
                                NaN otherwise). ~1 means the weights have collapsed.
          ``num_mc_samples``    S for the MC estimators, None otherwise
          ``n_nonfinite``       non-finite entries scrubbed from the raw gradient
          ``max_norm``          the threshold in force (None if unclipped)

        See :func:`plotting.guidance_summary` in ``examples/`` for the arc-length and
        saturation summary computed from these entries.
        """
        return self._history

    def reset_history(self) -> None:
        """Clear recorded diagnostics (call between sampling runs)."""
        self._history = []
        self._prev_dir = None

    @abstractmethod
    def forward(self, f_t: torch.Tensor, t: float) -> torch.Tensor:
        raise NotImplementedError


class GuidanceDPS(GuidanceApproximation):
    """Diffusion Posterior Sampling (DPS) guidance.

    Approximates ∇_{f_t} log p(y | f_t) by differentiating through the
    Tweedie denoiser: grad w.r.t. f_t of log p(y | f_0_hat(f_t)).
    """

    def __init__(
        self,
        flow: BaseFlow,
        condition: BaseCondition,
        max_norm: float = 1e2,
    ):
        super().__init__(condition=condition, flow=flow, max_norm=max_norm)
        self._require_flow_method("predict_terminal")

    def forward(self, f_t: torch.Tensor, t: float) -> torch.Tensor:
        f_t = f_t.detach().requires_grad_(True)
        beta_t = self.flow.schedule._beta(t)
        f_pred = self.flow._to_data_space(self.flow.predict_terminal(f_t, t))
        logp = self.condition.logp(f_pred)

        grad_logp = torch.autograd.grad(
            logp.sum(), f_t, create_graph=False, retain_graph=False
        )[0]
        return self._to_drift(grad_logp, beta_t, t)


class GuidanceMPGD(GuidanceDPS):
    """Manifold-Projected Gradient Descent (MPGD) guidance.

    Like DPS but differentiates log p(y | x_0_hat) w.r.t. x_0_hat rather
    than w.r.t. x_t, projecting the gradient onto the data manifold.
    """

    def forward(self, f_t: torch.Tensor, t: float) -> torch.Tensor:
        f_t = f_t.detach()
        beta_t = self.flow.schedule._beta(t)
        f_0 = self.flow.predict_terminal(f_t, t).requires_grad_(True)
        f_pred = self.flow._to_data_space(f_0)
        logp = self.condition.logp(f_pred)
        grad_logp = torch.autograd.grad(
            logp.sum(), f_0, create_graph=False, retain_graph=False
        )[0]
        return self._to_drift(grad_logp, beta_t, t)


class GuidanceMC(GuidanceApproximation):
    """Monte Carlo guidance via importance-weighted score estimation.

    Approximates ∇ log p(y | x_t) by drawing S samples from p(x_0 | x_t),
    evaluating log p(y | x_0^(s)), and computing a weighted average of the
    per-sample gradients.
    """

    def __init__(
        self,
        flow: BaseFlow,
        condition: BaseCondition,
        num_mc_samples: int = 5,
        max_norm: float = 1e2,
    ):
        super().__init__(condition=condition, flow=flow, max_norm=max_norm)
        self._require_flow_method("sample_terminal")
        self.num_mc_samples = num_mc_samples

    def forward(self, f_t: torch.Tensor, t: float) -> torch.Tensor:
        if getattr(self.flow, "affine_data_space", False):
            return self._forward_affine(f_t, t)

        beta_t = self.flow.schedule._beta(t)
        alpha_t = self.flow.schedule._alpha_2(t)

        # Sample f_0^(s) ~ p(f_0 | f_t) using Tweedie
        f_samples = self.flow.sample_terminal(
            f_t, t, self.num_mc_samples
        )  # (B, S, C, N)
        B, S, C, N = f_samples.shape

        # Flatten and enable gradients w.r.t. f_0 samples
        f_samples_flat = f_samples.reshape(B * S, C, N).detach().requires_grad_(True)

        # Convert to original space for condition evaluation
        f_samples_original = self.flow._to_data_space(f_samples_flat)  # (B*S, C, N)

        # Compute log p(C | f_0^(s)) for importance weights
        logp = self.condition.logp(f_samples_original)  # (B*S,)

        # Compute likelihood scores s^(s) = ∇_{f_0} log p(C | f_0^(s))
        grad_logp_f0 = torch.autograd.grad(
            logp.sum(), f_samples_flat, create_graph=False, retain_graph=False
        )[0]
        grad_logp_f0 = torch.nan_to_num(
            grad_logp_f0, nan=0.0, posinf=0.0, neginf=0.0
        ).reshape(B, S, C, N)  # (B, S, C, N)

        # Self-normalised importance weights; ESS ~ 1 means one draw dominates and the
        # estimate has degenerated to DPS (usually a sigma tighter than the residual
        # can reach).
        weights, ess = _snis_weights(_scrub_logp(logp.detach().reshape(B, S)))

        # Weighted sum of likelihood scores: Σ_s w̄^(s) s^(s)
        weighted_scores = (weights[:, :, None, None] * grad_logp_f0).sum(
            dim=1
        )  # (B, C, N)

        # Chain rule through f_0 = f_t / sqrt(alpha_t) + ...: in whitened space the
        # Jacobian d f_0 / d (score input) is sqrt(alpha_t) * I.
        return self._to_drift(torch.sqrt(alpha_t) * weighted_scores, beta_t, t, ess=ess)

    def _forward_affine(self, f_t: torch.Tensor, t: float) -> torch.Tensor:
        """The same estimator for a flow whose ``_to_data_space`` is affine, ``A z + b``.

        Then ``sum_s w_s A^T g_s = A^T sum_s w_s g_s``, so the scores are taken and
        weighted in data space and pulled back once per particle, and the draws come
        from :meth:`~flow_inference.flows.GaussianFlow.sample_terminal_data`, which
        decodes the posterior mean once per particle. Only the condition itself is
        evaluated on all ``B * S`` draws.
        """
        beta_t = self.flow.schedule._beta(t)
        alpha_t = self.flow.schedule._alpha_2(t)

        f_data = self.flow.sample_terminal_data(f_t, t, self.num_mc_samples)
        B, S, C, M = f_data.shape  # (B, S, C, output_dim)
        f_flat = f_data.reshape(B * S, C, M).detach().requires_grad_(True)

        logp = self.condition.logp(f_flat)  # (B*S,)
        grad_logp = torch.autograd.grad(
            logp.sum(), f_flat, create_graph=False, retain_graph=False
        )[0]
        grad_logp = torch.nan_to_num(
            grad_logp, nan=0.0, posinf=0.0, neginf=0.0
        ).reshape(B, S, C, M)

        weights, ess = _snis_weights(_scrub_logp(logp.detach().reshape(B, S)))
        weighted_scores = self.flow._pullback(
            (weights[:, :, None, None] * grad_logp).sum(dim=1)
        )  # (B, C, N)
        return self._to_drift(torch.sqrt(alpha_t) * weighted_scores, beta_t, t, ess=ess)


class GuidanceMCFisher(GuidanceApproximation):
    """Gradient-free guidance via the Fisher-identity deviation.

    Because ``p(f_0 | f_t) = N(sqrt(alpha_t) f_t, (1 - alpha_t) I)`` is exact for the
    closed-form Gaussian/GP flows, Fisher's identity gives the conditional score in
    closed form up to one expectation::

        grad_{f_t} log p(y | f_t)
            = sqrt(alpha_t) / (1 - alpha_t) * (E[f_0 | f_t, y] - sqrt(alpha_t) f_t)

    so no derivative of the condition is needed.

    Two additions improve the variance of this estimator, all on by default:

    ``antithetic``
        Mirrored ± draws. Without them the estimator reports a random direction of norm
        ~std * sqrt(N / S_eff) even when the likelihood is *flat* across the ensemble;
        with them those contributions cancel and the estimate reduces to a
        finite-difference along each ε, so no-signal really means no drift.
    ``resample_noise``
        Fresh draws per step. With the frozen noise
        :class:`GuidanceMC` needs, the same S directions are reused the whole way down
        and the MC error becomes a coherent bias instead of averaging away over the
        integration.
    """

    def __init__(
        self,
        flow: BaseFlow,
        condition: BaseCondition,
        num_mc_samples: int = 16,
        antithetic: bool = True,
        resample_noise: bool = True,
    ):
        super().__init__(condition=condition, flow=flow, max_norm=None)
        self._require_flow_method("sample_terminal")
        self.num_mc_samples = num_mc_samples
        self.antithetic = antithetic
        self.resample_noise = resample_noise

    def forward(self, f_t: torch.Tensor, t: float) -> torch.Tensor:
        beta_t = self.flow.schedule._beta(t)
        f_t = f_t.detach()
        with torch.no_grad():  # nothing here is differentiated
            draws = self.flow.sample_terminal(
                f_t,
                t,
                self.num_mc_samples,
                resample_noise=self.resample_noise,
                antithetic=self.antithetic,
            )  # (B, S, C, N)
            weights, ess = _snis_weights(
                _ensemble_logp(self.flow, self.condition, draws)
            )
            # E[f_0 | f_t, y] - E[f_0 | f_t]: only the deviation matters.
            deviation = (weights[:, :, None, None] * draws).sum(
                dim=1
            ) - self.flow.predict_terminal(f_t, t)

        coef = _fisher_coef(self.flow, t, floor=1e-6)
        return self._to_drift(coef * deviation, beta_t, t, ess=ess)


class GuidanceEnsembleKalman(GuidanceApproximation):
    """Derivative-free guidance via an ensemble Kalman (EnKG) update.

    Same Fisher-identity skeleton as :class:`GuidanceMCFisher` — the drift is
    ``sqrt(alpha_t)/(1 - alpha_t)`` times a deviation of the terminal posterior mean —
    but the deviation comes from a Kalman analysis step instead of importance weights::

        Delta = C_{f,r} (C_{r,r} + I)^{-1} (0 - rbar)

    with ``C_{f,r}``, ``C_{r,r}`` the ensemble cross-/auto-covariances of the whitened
    draws and their standardised residuals
    (:meth:`~flow_inference.conditions.BaseCondition.standardised_residual`).

    ``num_iterations`` (ESMDA)
        One analysis step linearises the condition once. Splitting the observation precision into
        ``num_iterations`` equal parts and re-evaluating the residual between them is
        Gauss-Newton rather than a single linear solve, and it costs one extra condition
        evaluation per part.
    ``pool_gain``
        The per-particle ensemble has only ``S - 1`` centred members, so the gain is
        confined to an ``(S - 1)``-dimensional subspace of an ``N``-dimensional state.
        Pooling the *deviations* across the sampler's batch estimates the gain from
        up to ``max_pool`` members instead, which is full rank and far less noisy, at no
        extra condition evaluations. Each particle keeps its own residual mean ``rbar``,
        so the only thing shared between particles is a linear operator standing in for
        the Jacobian — no per-particle information is exchanged. Requires a batch: it
        falls back to per-particle gains when ``B == 1``.

    Args:
        num_mc_samples: per-particle ensemble size S. Small is fine when ``pool_gain`` is
            on; raise it to 32+ if you turn pooling off.
        num_iterations: ESMDA sub-iterations per step (1 = a single linear analysis).
            Cost is linear in this.
        pool_gain: estimate the gain from batch-pooled deviations (see above).
        max_pool: cap on the pooled ensemble size, bounding the solve at
            ``max_pool^3``. Values below ``N`` reintroduce the rank deficiency pooling
            exists to remove, so keep it comfortably above the state width. Pooling is
            skipped entirely if this leaves room for fewer than two ensembles.
    """

    def __init__(
        self,
        flow: BaseFlow,
        condition: BaseCondition,
        num_mc_samples: int = 10,
        resample_noise: bool = True,
        num_iterations: int = 4,
        pool_gain: bool = True,
        max_pool: int = 512,
    ):
        super().__init__(condition=condition, flow=flow, max_norm=None)
        self._require_flow_method("sample_terminal")
        if num_mc_samples < 2:
            raise ValueError(
                "GuidanceEnsembleKalman needs num_mc_samples >= 2 to form an ensemble "
                f"covariance, got {num_mc_samples}"
            )
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        self.num_mc_samples = num_mc_samples
        self.resample_noise = resample_noise
        self.num_iterations = num_iterations
        self.pool_gain = pool_gain
        self.max_pool = max_pool

    def _residual(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardised residual of an ensemble, plus its per-particle mean.

        The residual arrives already divided by sigma, which makes the observation
        covariance the identity — that is what lets the solves below use a bare ``I`` and
        keeps components on very different scales commensurate.
        """
        B, S, C, N = state.shape
        f_data = self.flow._to_data_space(state.reshape(B * S, C, N))
        resid = self.condition.standardised_residual(f_data).reshape(B, S, C, -1)
        resid = torch.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
        return resid, resid.mean(1)

    def _analysis(self, state: torch.Tensor, damping: float) -> torch.Tensor:
        """One Kalman analysis increment ``(B, C, N)`` for the ensemble ``state``.

        ``damping`` inflates the observation covariance (ESMDA's ``num_iterations * I``),
        so that ``num_iterations`` successive increments deliver the full update.
        """
        resid, rbar = self._residual(state)
        B, S = state.shape[0], state.shape[1]

        # Centred deviations; S on the -2 axis so the solves batch over (B, C).
        dev_state = state - state.mean(1, keepdim=True)
        dev_resid = resid - resid.mean(1, keepdim=True)

        if self.pool_gain and B > 1 and self.max_pool >= 2 * S:
            # Pool the deviations over the batch into one shared ensemble. Whole
            # ensembles only: each particle's block is centred on its own mean, so this
            # is a pooled *within-particle* covariance — the right estimator for the
            # conditional spread (1 - alpha_t) I, just with many more degrees of freedom.
            # The gram is shared, so this is one factorisation solved against B
            # right-hand sides — cheaper than B separate S x S solves, not dearer.
            groups = min(B, self.max_pool // S)
            M = groups * S
            scale = 1.0 / math.sqrt(groups * (S - 1))  # pooled within-group dof
            # (B, S, C, ·) -> (C, B*S, ·), then keep the first `groups` ensembles.
            X = dev_state.permute(2, 0, 1, 3).flatten(1, 2)[:, :M] * scale  # (C, M, N)
            Y = dev_resid.permute(2, 0, 1, 3).flatten(1, 2)[:, :M] * scale  # (C, M, A)
            gram = Y @ Y.transpose(-1, -2)  # (C, M, M)
            eye = torch.eye(M, device=gram.device, dtype=gram.dtype)
            rhs = Y @ -rbar.permute(1, 2, 0)  # (C, M, B)
            w = torch.linalg.solve(gram + damping * eye, rhs)  # (C, M, B)
            return (X.transpose(-1, -2) @ w).permute(2, 0, 1)  # (B, C, N)

        scale = 1.0 / math.sqrt(S - 1)
        X = dev_state.transpose(1, 2) * scale  # (B, C, S, N)
        Y = dev_resid.transpose(1, 2) * scale  # (B, C, S, A)
        # Delta = X^T (Y Y^T + I)^{-1} Y (-rbar): an S x S solve, never A x A.
        gram = Y @ Y.transpose(-1, -2)  # (B, C, S, S)
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        rhs = -(Y @ rbar.unsqueeze(-1))  # (B, C, S, 1)
        w = torch.linalg.solve(gram + damping * eye, rhs)  # (B, C, S, 1)
        return (X.transpose(-1, -2) @ w).squeeze(-1)  # (B, C, N)

    def forward(self, f_t: torch.Tensor, t: float) -> torch.Tensor:
        beta_t = self.flow.schedule._beta(t)
        f_t = f_t.detach()

        with torch.no_grad():  # no derivative of anything is taken
            state = self.flow.sample_terminal(
                f_t, t, self.num_mc_samples, resample_noise=self.resample_noise
            )  # (B, S, C, N)
            delta = torch.zeros(
                state.shape[0],
                state.shape[2],
                state.shape[3],
                dtype=f_t.dtype,
                device=f_t.device,
            )
            for _ in range(self.num_iterations):
                # Move the whole ensemble by the increment, so the next iteration
                # re-linearises the condition about the updated state.
                step = self._analysis(state, damping=float(self.num_iterations))
                delta = delta + step
                state = state + step.unsqueeze(1)

        coef = _fisher_coef(self.flow, t, floor=1e-12)
        return self._to_drift(coef * delta, beta_t, t)
