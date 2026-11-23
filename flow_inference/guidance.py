"""Guidance: the drift correction that steers a flow's samples towards a condition.

Sampling a flow conditioned on ``y`` needs the score of ``p(x_t | y)``, which is the
unconditional score plus ``grad_{x_t} log p(y | x_t)``. The flow supplies the first;
a guidance object approximates the second and returns the matching correction to the
probability-flow drift, ``(-beta_t / 2) * grad_{x_t} log p(y | x_t)``, which
:class:`~flow_inference.sampler.IntegratorSampler` adds to the flow's drift at every
step. ``log p(y | x_t)`` is intractable, so each class approximates it through the
flow's estimate of the clean sample ``x_0`` given ``x_t``:

- :class:`GuidanceDPS` (Diffusion Posterior Sampling): evaluate the condition at the
  posterior mean ``x0_hat(x_t)`` and differentiate through the denoiser.
- :class:`GuidanceMPGD` (Manifold-Projected Gradient Descent): the gradient of the
  condition at ``x0_hat`` with respect to ``x0_hat`` itself, without differentiating
  through the denoiser.
- :class:`GuidanceMC`: a Monte Carlo estimate from ``S`` draws of ``p(x_0 | x_t)``,
  averaging their likelihood gradients with self-normalised importance weights.
- :class:`GuidancePiGDM`: for Gaussian-residual conditions on flows with ``N(0, I)``
  marginals, the noise-aware score of Pi-GDM, which accounts for the uncertainty of
  ``x_0`` given ``x_t`` and so stays bounded for very sharp conditions; solved
  matrix-free by least squares, optionally with extra Gauss-Newton iterations.

Every guidance takes the same ``condition`` (a
:class:`~flow_inference.conditions.BaseCondition`) and ``flow``, and bounds the size of
its correction with ``max_norm``.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from flow_inference._numerics import bind_condition, lsmr, scrub_grad, scrub_logp
from flow_inference.conditions import BaseCondition
from flow_inference.flow_utils import clip_vector_field
from flow_inference.flows import BaseFlow


def _snis_weights(logp: torch.Tensor) -> torch.Tensor:
    """Self-normalised importance weights from log-likelihoods ``(B, S)``.

    The weights in each row sum to 1.
    """
    return torch.softmax(logp, dim=1)


def _scrub_logp(logp: torch.Tensor) -> torch.Tensor:
    """Give invalid log-likelihoods zero weight.

    The floor is finite so that a softmax over a row of invalid entries is uniform
    rather than NaN.
    """
    return scrub_logp(logp, floor=-1e38)


class GuidanceApproximation(nn.Module, ABC):
    """Abstract base class for guidance.

    Subclasses implement :meth:`forward`, returning an approximation of the drift
    correction ``(-beta_t / 2) * grad_{x_t} log p(y | x_t)``, and finish with
    :meth:`_to_drift`.

    Args:
        condition: the condition ``y`` to steer towards.
        flow: the flow being sampled. When given, the condition is bound to it: it
            receives the flow's ``output_shape`` (see
            :meth:`~flow_inference.conditions.BaseCondition.as_grid`), and a warning
            is issued if it holds float64 tensors while the flow works in float32.
        max_norm: bound on the norm of the correction over each channel's ``N``
            points, applied as a smooth tanh squash, or None for no bound. The sampler
            integrates over unit time, so this is also roughly the furthest guidance
            can move a sample; once the squash saturates, only the gradient's
            direction matters.
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
        self.max_norm = max_norm
        if flow is not None:
            bind_condition(flow, condition)

    def _require_flow_method(self, name: str) -> None:
        """Check that the flow provides the method this estimator needs."""
        assert hasattr(self.flow, name), (
            f"BaseFlow must implement {name} for {type(self).__name__}"
        )

    def _to_drift(self, grad_logp: torch.Tensor, beta_t: torch.Tensor) -> torch.Tensor:
        """Turn a likelihood gradient into the detached drift correction.

        Zeroes non-finite entries, casts to the flow's dtype, scales by ``-beta_t / 2``
        and, unless ``max_norm`` is None, squashes the norm so that the solver sees a
        Lipschitz field.

        Args:
            grad_logp: ``grad log p(y | .)`` with respect to the diffusion state,
                ``(B, C, N)``.
            beta_t: the schedule's ``beta`` at the current time.

        Returns:
            The drift correction ``(B, C, N)``.
        """
        grad_logp = scrub_grad(grad_logp)
        if self.flow is not None:
            grad_logp = grad_logp.to(self.flow.dtype)
        dv = (-0.5 * beta_t) * grad_logp
        if self.max_norm is not None:
            dv = clip_vector_field(dv, self.max_norm)
        return dv.detach()

    @abstractmethod
    def forward(self, f_t: torch.Tensor, t: float) -> torch.Tensor:
        """The drift correction at state ``f_t`` and time ``t``.

        Args:
            f_t: diffusion state ``(B, C, N)``.
            t: current time, from 1 (noise) to 0 (data).

        Returns:
            The drift correction ``(B, C, N)``, detached.
        """
        raise NotImplementedError


class GuidanceDPS(GuidanceApproximation):
    """Diffusion Posterior Sampling (DPS) guidance.

    Approximates ``log p(y | f_t)`` by ``log p(y | f0_hat(f_t))``, where ``f0_hat`` is
    the flow's posterior-mean (Tweedie) estimate of the clean sample, and
    differentiates through the denoiser with respect to ``f_t``. Needs a flow with
    ``predict_terminal``.

    Args:
        flow: the flow being sampled.
        condition: the condition to steer towards.
        max_norm: bound on the correction's norm (see
            :class:`GuidanceApproximation`).
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
        return self._to_drift(grad_logp, beta_t)


class GuidanceMPGD(GuidanceDPS):
    """Manifold-Projected Gradient Descent (MPGD) guidance.

    Like :class:`GuidanceDPS`, but differentiates ``log p(y | f0_hat)`` with respect to
    ``f0_hat`` rather than ``f_t``. Skipping the denoiser's Jacobian is cheaper and
    keeps the update along the data manifold. Takes the same arguments as
    :class:`GuidanceDPS`.
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
        return self._to_drift(grad_logp, beta_t)


class GuidanceMC(GuidanceApproximation):
    """Monte Carlo guidance from draws of the clean sample.

    Writes ``p(y | f_t) = E[p(y | f_0) | f_t]`` and estimates its gradient from ``S``
    draws ``f_0^(s) ~ p(f_0 | f_t)``: the per-draw gradients
    ``grad log p(y | f_0^(s))`` are averaged with self-normalised importance weights
    proportional to ``p(y | f_0^(s))``. When one draw takes almost all the weight, the
    estimate reduces to a single-sample DPS-like gradient; this usually means
    ``sigma`` is tighter than the draws can reach. Needs a flow with
    ``sample_terminal``.

    Args:
        flow: the flow being sampled.
        condition: the condition to steer towards.
        num_mc_samples: number of draws ``S`` per sample.
        max_norm: bound on the correction's norm (see
            :class:`GuidanceApproximation`).
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

        f_samples = self.flow.sample_terminal(
            f_t, t, self.num_mc_samples
        )  # (B, S, C, N)
        B, S, C, N = f_samples.shape
        f_samples_flat = f_samples.reshape(B * S, C, N).detach().requires_grad_(True)
        f_samples_original = self.flow._to_data_space(f_samples_flat)  # (B*S, C, N)
        logp = self.condition.logp(f_samples_original)  # (B*S,)

        grad_logp_f0 = torch.autograd.grad(
            logp.sum(), f_samples_flat, create_graph=False, retain_graph=False
        )[0]
        grad_logp_f0 = scrub_grad(grad_logp_f0).reshape(B, S, C, N)

        weights = _snis_weights(_scrub_logp(logp.detach().reshape(B, S)))
        weights = weights.to(f_t.dtype)  # see _forward_affine
        weighted_scores = (weights[:, :, None, None] * grad_logp_f0).sum(
            dim=1
        )  # (B, C, N)

        # The draws are f_0 = sqrt(alpha_t) * f_t + noise, so d f_0 / d f_t is
        # sqrt(alpha_t) * I.
        return self._to_drift(torch.sqrt(alpha_t) * weighted_scores, beta_t)

    def _forward_affine(self, f_t: torch.Tensor, t: float) -> torch.Tensor:
        """The same estimator, for a flow whose ``_to_data_space`` is affine.

        With ``_to_data_space(z) = A z + b``, ``sum_s w_s A^T g_s`` equals
        ``A^T sum_s w_s g_s``, so the gradients are taken and
        weighted in data space and mapped back to the diffusion state once per sample.
        The draws come from
        :meth:`~flow_inference.flows.GaussianFlow.sample_terminal_data`, which decodes
        the posterior mean once per sample, so only the condition itself is evaluated
        on all ``B * S`` draws.
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
        grad_logp = scrub_grad(grad_logp).reshape(B, S, C, M)

        # Normalise in the condition's own precision, then cast: a condition built
        # from float64 data returns float64 logp even for a float32 flow, and
        # near-hard logps (~-1e10) differ by less than float32 can resolve.
        weights = _snis_weights(_scrub_logp(logp.detach().reshape(B, S)))
        weights = weights.to(f_t.dtype)
        weighted_scores = self.flow._pullback(
            (weights[:, :, None, None] * grad_logp).sum(dim=1)
        )  # (B, C, N)
        return self._to_drift(torch.sqrt(alpha_t) * weighted_scores, beta_t)


class GuidancePiGDM(GuidanceApproximation):
    """Pi-GDM guidance for Gaussian-residual conditions, solved by least squares.

    For flows with ``N(0, I)`` marginals the clean sample given the current state is
    exactly ``z_0 | z_t ~ N(mu_t, v_t I)``, with ``mu_t = alpha_t z_t`` and
    ``v_t = 1 - alpha_t^2``. Linearising the condition's residual ``r`` around
    ``mu_t`` makes ``p(y | z_t)`` Gaussian, and its score is

        ``grad log p(y | z_t) ~= -alpha_t J^T (v_t J J^T + Sigma)^-1 r(mu_t)``,

    where ``J`` is the residual's Jacobian with respect to the latent and ``Sigma`` the
    squared per-component tolerances (from
    :meth:`~flow_inference.conditions.BaseCondition.residual_and_sigma`). This is the
    Pi-GDM guidance of Song et al. (2023), and the noised likelihood of
    Zammit-Mangion et al. (2026), with the denoiser covariance exact rather than
    approximated. Unlike the single-point estimates (:class:`GuidanceDPS`,
    :class:`GuidanceMC`) it accounts for how uncertain ``z_0`` still is, so the drift
    stays bounded however small the tolerance: there is nothing for ``max_norm`` to
    tame, and fewer integration steps suffice. It is exact for conditions linear in
    the latent (point observations, linear operators) and a Gauss-Newton
    approximation otherwise.

    With ``num_iterations > 1`` the linearisation point is moved towards the mode of
    ``p(y | z_0) N(z_0; mu_t, v_t I)`` by further Gauss-Newton steps (a Laplace
    approximation at the mode), which is markedly more accurate for nonlinear
    residuals (ODE and PDE conditions) at proportionally higher cost.

    Each step solves the least-squares problem with the augmented operator
    ``[sqrt(v_t) J, diag(sigma)]`` by LSMR (Fong & Saunders, 2011), using only
    Jacobian-vector products, so ``J`` is never formed and tolerances of exactly 0
    (hard constraints) are allowed. The computation never synchronises with the host,
    so it can be used with ``IntegratorSampler.sample(cuda_graph=True)``.

    Before solving, each row of the system is scaled by its precision relative to the
    tightest tolerance, ``sigma_min / sigma_a`` (rows with ``sigma_a = 0`` get 1). This
    leaves the exact solution unchanged, but makes the fixed number of LSMR
    iterations resolve the tightest constraints first, as the precision-weighted
    gradient of the single-point guidances does. Without it, a condition mixing a
    loose residual with large Jacobian rows (a PDE stencil) and a tight one with small
    rows (boundary values) spends its iterations on the former.

    Args:
        flow: a flow with ``N(0, I)`` marginals (``has_standard_normal_marginals``):
            ``GaussianProcessFlow``, ``BasisGaussianProcessFlow``,
            ``GaussianIncrementFlow`` or ``GaussianFlow``.
        condition: an :class:`~flow_inference.conditions.EqualityCondition`, or a
            :class:`~flow_inference.conditions.ProductCondition` of them.
        num_iterations: Gauss-Newton iterations per step; 1 is Pi-GDM.
        lsmr_iters: LSMR iterations per solve. Fixed (no early stopping, so no host
            sync); about 20 is enough for the stencil residuals of the examples.
        damp_floor: small relative damping of the least-squares solve, for
            conditions whose tolerances are all 0.
        max_norm: optional cap on the drift norm, as for the other guidances; off by
            default, as this guidance's drift is already bounded.

    Example:
        >>> guidance = GuidancePiGDM(flow, pde_condition, num_iterations=3)
        >>> samples = IntegratorSampler(flow).sample(16, guidance=guidance, num_steps=50)
    """

    def __init__(
        self,
        flow: BaseFlow,
        condition: BaseCondition,
        num_iterations: int = 1,
        lsmr_iters: int = 20,
        damp_floor: float = 1e-6,
        max_norm: float | None = None,
    ):
        super().__init__(condition=condition, flow=flow, max_norm=max_norm)
        if not getattr(flow, "has_standard_normal_marginals", False):
            raise ValueError(
                f"{type(flow).__name__} has no N(0, I) marginals; GuidancePiGDM needs "
                "one (use a GaussianFlow subclass)."
            )
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        self.num_iterations = num_iterations
        self.lsmr_iters = lsmr_iters
        self.damp_floor = damp_floor
        self._affine = bool(getattr(flow, "affine_data_space", False))
        z0 = torch.zeros(
            1, flow.channels, flow.latent_dim, device=flow.device, dtype=flow.dtype
        )
        try:
            with torch.no_grad():
                _, sigma = condition.residual_and_sigma(flow._to_data_space(z0))
        except NotImplementedError as error:
            raise TypeError(
                f"GuidancePiGDM needs a condition with Gaussian residuals: {error}"
            ) from error
        # Constant, so computed once (and moved with the module, not per call).
        sigma = sigma.reshape(-1).to(flow.dtype)
        self.register_buffer("sigma", sigma)
        # Row scaling sigma_min / sigma in (0, 1]: precision relative to the tightest
        # positive tolerance; hard (0) and tightest rows get 1, all-hard gives all 1.
        positive = sigma[sigma > 0]
        floor = positive.min() if positive.numel() else sigma.new_tensor(1.0)
        self.register_buffer("row_scale", floor / sigma.clamp(min=floor))

    def _residual(self, f: torch.Tensor) -> torch.Tensor:
        r, _ = self.condition.residual_and_sigma(f)
        return r.reshape(r.shape[0], -1)

    def _products(self, z: torch.Tensor):
        """The residual at latents ``z`` ``(B, C, N)``, flattened to ``(B, m)``, and
        closures for ``J p`` and ``J^T u`` (``J`` w.r.t. the flattened latent).

        For an affine flow (``f = L z + m``) the products go through the residual
        alone, with ``L`` and ``L^T`` applied by the flow's ``_linear_part`` and
        ``_pullback``. Both closures take unit-norm directions and rescale, so very
        sharp conditions cannot overflow float32.
        """
        B, C, N = z.shape
        tiny = torch.finfo(z.dtype).tiny
        with torch.enable_grad():
            if self._affine:
                x = self.flow._to_data_space(z).detach().requires_grad_(True)
                r = self._residual(x)
            else:
                x = z.detach().requires_grad_(True)
                r = self._residual(self.flow._to_data_space(x))
            w = torch.zeros_like(r, requires_grad=True)
            # J p = d/dw <J^T w, p>: a second backward pass gives forward products.
            (jt_w,) = torch.autograd.grad(r, x, w, create_graph=True)

        def unit(v):
            n = v.pow(2).sum(1).sqrt().clamp(min=tiny)
            return v / n[:, None], n

        def J(p):  # (B, CN) -> (B, m)
            p, n = unit(p)
            p = p.view(B, C, N)
            if self._affine:
                p = self.flow._linear_part(p)
            (jp,) = torch.autograd.grad(jt_w, w, p, retain_graph=True)
            return jp * n[:, None]

        def JT(u):  # (B, m) -> (B, CN)
            u, n = unit(u)
            (jtu,) = torch.autograd.grad(r, x, u, retain_graph=True)
            if self._affine:
                jtu = self.flow._pullback(jtu)
            return jtu.reshape(B, C * N) * n[:, None]

        return r.detach(), J, JT

    def forward(self, f_t: torch.Tensor, t) -> torch.Tensor:
        t = torch.as_tensor(t, device=f_t.device, dtype=f_t.dtype)
        schedule = self.flow.schedule
        alpha2 = schedule._alpha_2(t)
        v = (1.0 - alpha2).clamp(min=1e-12)
        sqrt_v, alpha, beta_t = v.sqrt(), alpha2.sqrt(), schedule._beta(t)
        B, C, N = f_t.shape
        D = C * N
        tiny = torch.finfo(f_t.dtype).tiny
        sigma = self.sigma.to(f_t)
        row = self.row_scale.to(f_t)
        with torch.no_grad():
            mu = self.flow.predict_terminal(f_t.detach(), t).reshape(B, D)
        d = torch.zeros_like(mu)  # offset of the linearisation point from mu
        x_star = torch.zeros_like(mu)
        w_prev = None
        for k in range(self.num_iterations):
            r, J, JT = self._products((mu + d).view(B, C, N))

            def A(w, J=J):  # augmented operator [sqrt(v) J, diag(sigma)]
                return sqrt_v * J(w[:, :D]) + sigma * w[:, D:]

            def M(w, A=A):  # its rows scaled by precision (see the class docstring)
                return row * A(w)

            def MT(u, JT=JT):
                u = row * u
                return torch.cat([sqrt_v * JT(u), sigma * u], 1)

            rhs = r if k == 0 else r - J(d)
            if w_prev is not None:  # warm start from the previous iteration
                rhs = rhs - A(w_prev)
            rhs = row * rhs
            scale = rhs.pow(2).sum(1).sqrt().clamp(min=tiny)
            w = lsmr(
                M, MT, rhs / scale[:, None], self.damp_floor * sqrt_v, self.lsmr_iters
            )
            w = w * scale[:, None]
            if w_prev is not None:
                w = w + w_prev
            w_prev = w.detach()
            x_star = w[:, :D]
            d = -sqrt_v * x_star
        grad_logp = (-(alpha / sqrt_v) * x_star).view(B, C, N)
        return self._to_drift(grad_logp, beta_t)
