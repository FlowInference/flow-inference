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

Every guidance takes the same ``condition`` (a
:class:`~flow_inference.conditions.BaseCondition`) and ``flow``, and bounds the size of
its correction with ``max_norm``.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from flow_inference._numerics import bind_condition, scrub_grad, scrub_logp
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
