"""Conditions: the likelihoods ``log p(condition | x)`` that guidance steers towards.

A condition scores how well a field satisfies some requirement. It is an
``nn.Module`` whose :meth:`~BaseCondition.logp` maps a batch of fields to one
log-likelihood per sample. The field always arrives *flat*, with shape ``(B, C, N)``
(batch, channels, points), in the data space of the flow; the result has shape
``(B,)``. Guidance differentiates ``logp`` with respect to the field, so it should be
smooth and finite wherever the sampler may visit.

Building blocks:

- :class:`BinaryCondition`: an inequality, such as ``x > 0`` or monotonicity.
  Subclasses return a margin from :meth:`~BinaryCondition.smooth_eval_condition` that
  is positive where the condition holds, and the likelihood is the smooth relaxation
  ``logsigmoid(margin / v)``; a smaller softness ``v`` is closer to a hard
  constraint.
- :class:`EqualityCondition`: an equality, such as an observation or a differential
  equation. Subclasses return a residual from :meth:`~EqualityCondition.residual` that
  is zero where the condition holds, and the likelihood is Gaussian with tolerance
  ``sigma``.
- :class:`PointWiseCondition`: an :class:`EqualityCondition` for noisy observations of
  the field at a subset of points.
- :class:`ProductCondition`: several conditions imposed together (their log-likelihoods
  add).

To write a new condition, subclass :class:`BinaryCondition` or
:class:`EqualityCondition` and implement the one method returning a ``(B, C, A)``
tensor of ``A`` components, or subclass :class:`BaseCondition` and implement
:meth:`~BaseCondition.logp` directly. A condition that works on an ``H x W`` lattice
(a PDE residual, pooling over regions) can call :meth:`~BaseCondition.as_grid` to view
the flat field as ``(B, C, H, W)``; the grid shape is taken from the flow when the
condition is passed to a guidance object, and is stored as ``output_shape``.

Conditions imposed at every grid cell
-------------------------------------
Both relaxations sum one term per component. That is right when the components are a
fixed set of quantities (observations at given points, region totals, one peak). When
the condition is imposed at *every grid cell* (an ODE or PDE residual, a pointwise
bound, monotonicity), refining the grid adds terms, so the same ``sigma`` enforces the
condition ever more tightly and the posterior changes with resolution. Passing
``weights=`` the cell size (``dx``, or ``dx * dt`` on a space-time grid) turns the sum
into a Riemann sum for an integral over the domain::

    log p = -sum_a w_a r_a^2 / (2 sigma^2)  ~  -(1 / 2 sigma^2) * integral r(x)^2 dx

so ``sigma`` becomes a tolerance on the root-mean-square residual over the domain and
means the same thing at any resolution. For an inequality, write the margin so that it
also has a continuum limit: a slope ``(f[j+1] - f[j]) / dx`` rather than a raw
difference, which shrinks as the grid is refined.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn


class BaseCondition(nn.Module, ABC):
    """Abstract base class for conditions: a likelihood ``log p(condition | x)``.

    Subclasses implement :meth:`logp`, mapping a flat field ``(B, C, N)`` to ``(B,)``.
    Conditions that reason about the field on a lattice call :meth:`as_grid` to view
    it as ``(B, C, *output_shape)``. Physical spacings such as ``dx`` and ``dt`` are
    not inferred from the grid; pass them to the subclass constructor.

    Attributes:
        output_shape: the flow's per-channel grid shape, e.g. ``(H, W)``, or None until
            the condition is passed to a guidance object (see
            :meth:`bind_output_shape`).
    """

    def __init__(self):
        super().__init__()
        self.output_shape: tuple[int, ...] | None = None

    def bind_output_shape(self, output_shape: tuple[int, ...] | None) -> None:
        """Record the flow's per-channel ``output_shape`` (e.g. ``(H, W)``).

        Called by guidance objects, so that grid-aware conditions can use
        :meth:`as_grid` without being given the grid in their constructor. A shape
        already set on a condition is kept, which lets a condition override the flow's
        view if it needs to. The shape is also passed to nested sub-conditions (e.g.
        those of a :class:`ProductCondition`).

        Args:
            output_shape: the flow's per-channel grid shape, or None.
        """
        shape = tuple(output_shape) if output_shape is not None else None
        # modules() yields self and every nested sub-condition.
        for module in self.modules():
            if isinstance(module, BaseCondition) and module.output_shape is None:
                module.output_shape = shape

    def as_grid(self, x: torch.Tensor) -> torch.Tensor:
        """View a flat field ``(B, C, N)`` as ``(B, C, *output_shape)``.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            A view of ``x`` with shape ``(B, C, *output_shape)``.

        Raises:
            RuntimeError: if :attr:`output_shape` has not been set.
            ValueError: if ``N`` does not match the number of grid cells, so that a
                mismatched flow and condition fail loudly rather than mis-tiling the
                field.
        """
        if self.output_shape is None:
            raise RuntimeError(
                f"{type(self).__name__}.as_grid needs output_shape, which is "
                "auto-wired from the flow when the condition is passed to a guidance "
                "object. Construct guidance with this condition first, or set "
                "condition.output_shape = (H, W) manually."
            )
        B, C, N = x.shape
        cells = math.prod(self.output_shape)
        if N != cells:
            raise ValueError(
                f"field width N={N} does not match output_shape "
                f"{self.output_shape} ({cells} cells)"
            )
        return x.view(B, C, *self.output_shape)

    @abstractmethod
    def logp(self, x: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of the condition given each field in the batch.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            ``log p(condition | x)``, shape ``(B,)``.
        """
        raise NotImplementedError

    def standardised_residual(self, x: torch.Tensor) -> torch.Tensor:
        """The condition as a vector of residuals in units of its tolerance.

        Optional: :meth:`logp` is all the library needs. :class:`EqualityCondition`
        and :class:`BinaryCondition` define it; other subclasses may override it.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Residuals ``(B, C, A)``, zero where the condition is satisfied; a value of
            1 means off by one tolerance.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define standardised_residual. Subclass "
            "EqualityCondition / BinaryCondition, or override standardised_residual to "
            "return residual/sigma with shape (B, C, A)."
        )

    def residual_and_sigma(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The condition as Gaussian residuals with a tolerance per component.

        Used by :class:`~flow_inference.guidance.GuidancePiGDM`. Unlike
        :meth:`standardised_residual` it never divides by the tolerance, so a
        tolerance of 0 (a hard constraint) is allowed.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Residuals ``(B, C, A)`` and tolerances ``(C, A)``, such that the
            log-likelihood is ``-0.5 * sum (residual / tolerance)^2``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define residual_and_sigma; it is defined "
            "for EqualityCondition and products of them."
        )


class ProductCondition(BaseCondition):
    """Several conditions imposed together: their log-likelihoods are summed.

    Args:
        conditions: the conditions to combine.
    """

    def __init__(self, conditions: Sequence["BaseCondition"]):
        super().__init__()
        self.conditions = nn.ModuleList(conditions)

    def logp(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of the sub-conditions' log-likelihoods.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Log-likelihood, shape ``(B,)``.
        """
        return torch.stack([cond.logp(x) for cond in self.conditions]).sum(0)

    def standardised_residual(self, x: torch.Tensor) -> torch.Tensor:
        """The sub-conditions' standardised residuals, concatenated along the last axis.

        Each block is in units of its own tolerance, so blocks on very different
        scales (a PDE residual and a boundary value, say) are directly comparable.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Residuals ``(B, C, A_1 + A_2 + ...)``.
        """
        return torch.cat(
            [cond.standardised_residual(x) for cond in self.conditions], dim=-1
        )

    def residual_and_sigma(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The sub-conditions' residuals and tolerances, concatenated along the last
        axis (see :meth:`BaseCondition.residual_and_sigma`).

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Residuals ``(B, C, A_1 + A_2 + ...)`` and tolerances ``(C, A_1 + ...)``.
        """
        parts = [cond.residual_and_sigma(x) for cond in self.conditions]
        return (
            torch.cat([r for r, _ in parts], dim=-1),
            torch.cat([s for _, s in parts], dim=-1),
        )


def _as_weights(weights) -> torch.Tensor | None:
    """``weights`` as a tensor (or None), checked finite and non-negative."""
    if weights is None:
        return None
    w = torch.as_tensor(weights, dtype=torch.get_default_dtype())
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and non-negative")
    return w


def _sum_terms(terms: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Sum per-component terms ``(B, C, A)`` to ``(B,)``, each times its weight."""
    if weights is not None:
        terms = terms * weights.to(terms)
    return terms.sum(dim=[1, 2])


class BinaryCondition(BaseCondition):
    """Smooth relaxation of an inequality (a yes/no condition such as ``x > 0``).

    Subclasses implement :meth:`smooth_eval_condition`, returning a margin
    ``cond_a(x)`` per component that is positive where the condition holds. The
    log-likelihood is ``sum_a w_a logsigmoid(cond_a(x) / v)``: close to 0 where the
    margin is comfortably positive and falling linearly where it is negative.

    Args:
        v: softness of the relaxation; smaller is closer to a hard constraint.
        weights: optional quadrature weights ``w_a``, a scalar (the cell size) or a
            tensor broadcastable to ``(C, A)``. Leave as None (every ``w_a = 1``) for a
            fixed set of margins; set it to the cell size for a constraint imposed at
            every grid cell (see the module docstring).
    """

    def __init__(self, v: float = 1e-4, weights=None):
        super().__init__()
        self.register_buffer("v", torch.tensor(float(v)))
        # Not named `weights`, which subclasses often use for something else.
        self.register_buffer("quadrature_weights", _as_weights(weights))

    def logp(self, x: torch.Tensor) -> torch.Tensor:
        """Log-likelihood ``sum_a w_a logsigmoid(cond_a(x) / v)``.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Log-likelihood, shape ``(B,)``.
        """
        cond = self.smooth_eval_condition(x)
        logp = torch.nn.functional.logsigmoid(cond / self.v)
        return _sum_terms(logp, self.quadrature_weights)

    def standardised_residual(self, x: torch.Tensor) -> torch.Tensor:
        """The violation of each margin in units of ``v``: ``softplus(-cond / v)``.

        Zero once ``cond`` is comfortably positive and growing like ``-cond / v`` where
        it is negative. Scaled by ``sqrt(w_a)`` when weights are set.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Residuals ``(B, C, A)``.
        """
        r = torch.nn.functional.softplus(-self.smooth_eval_condition(x) / self.v)
        w = self.quadrature_weights
        return r if w is None else r * w.to(r).sqrt()

    @abstractmethod
    def smooth_eval_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the margin of each component of the inequality.

        The condition has ``A`` components; monotonicity on ``H`` points, for example,
        has ``A = H - 1``, one per pair of neighbours. The margin should be continuous
        in ``x``.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Margins ``(B, C, A)``: positive where the component is satisfied, negative
            where it is violated.
        """
        raise NotImplementedError


class EqualityCondition(BaseCondition):
    """Gaussian relaxation of an equality constraint ``residual(x) = 0``.

    Subclasses implement :meth:`residual`. The log-likelihood is
    ``-sum_a w_a residual_a(x)^2 / (2 sigma^2)``.

    Args:
        sigma: tolerance on the residual (the observation noise standard deviation,
            for data).
        weights: optional quadrature weights ``w_a``, a scalar (the cell size) or a
            tensor broadcastable to ``(C, A)``. Leave as None (every ``w_a = 1``) when
            the residual has a fixed set of components; set it to the cell size when
            the residual is evaluated at every grid cell, so that ``sigma`` is a
            resolution-independent tolerance on the integrated residual (see the
            module docstring).
    """

    def __init__(self, sigma: float = 1e-4, weights=None):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        # Not named `weights`, which subclasses often use for something else.
        self.register_buffer("quadrature_weights", _as_weights(weights))

    def logp(self, x: torch.Tensor) -> torch.Tensor:
        """Log-likelihood ``-sum_a w_a residual_a(x)^2 / (2 sigma^2)``.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Log-likelihood, shape ``(B,)``.
        """
        diff = self.residual(x)
        logp = -0.5 * (diff / self.sigma) ** 2
        return _sum_terms(logp, self.quadrature_weights)

    def standardised_residual(self, x: torch.Tensor) -> torch.Tensor:
        """The residual in units of ``sigma``, scaled by ``sqrt(w_a)`` when weights
        are set, so that ``logp == -0.5 * ||standardised_residual||^2``.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Residuals ``(B, C, A)``.
        """
        r = self.residual(x) / self.sigma
        w = self.quadrature_weights
        return r if w is None else r * w.to(r).sqrt()

    def residual_and_sigma(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The residual and, per component, the tolerance ``sigma / sqrt(w_a)``, so
        that ``logp == -0.5 * sum (residual / tolerance)^2``. ``sigma = 0`` is allowed
        here and makes the condition a hard constraint for
        :class:`~flow_inference.guidance.GuidancePiGDM` (``logp`` itself is then
        undefined).

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Residuals ``(B, C, A)`` and tolerances ``(C, A)``.
        """
        r = self.residual(x)
        sigma = self.sigma.to(r).expand(r.shape[1:])
        w = self.quadrature_weights
        if w is not None:
            w = torch.broadcast_to(w.to(r), r.shape[1:])
            # A zero weight drops the component: zero residual, unit tolerance
            # (rather than an infinite tolerance, which would poison the solve).
            keep = w > 0
            r = torch.where(keep, r, torch.zeros_like(r))
            sigma = torch.where(keep, sigma / w.sqrt(), torch.ones_like(sigma))
        return r, sigma

    @abstractmethod
    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual of each component of the equality.

        Args:
            x: flat field ``(B, C, N)``.

        Returns:
            Residuals ``(B, C, A)``, zero where the condition holds.
        """
        raise NotImplementedError


class PointWiseCondition(EqualityCondition):
    """Noisy observations of a single-channel field at a subset of its points.

    The residual is ``x[mask] - target_field[mask]``; entries of ``target_field``
    outside the mask are ignored.

    Args:
        target_field: observed values, shape ``(N,)``.
        mask: boolean mask ``(N,)`` selecting the observed points; read once, at
            construction.
        sigma: observation noise standard deviation.
        weights: optional quadrature weights (see :class:`EqualityCondition`).
    """

    def __init__(
        self,
        target_field: torch.Tensor,
        mask: torch.Tensor,
        sigma: float = 0.1,
        weights=None,
    ):
        super().__init__(sigma=sigma, weights=weights)
        self.register_buffer("mask", mask)
        self.register_buffer("target_field", target_field)
        # Integer indices of the observed points, fixed at construction: indexing with
        # them avoids the host sync that boolean-mask indexing costs on the GPU (and so
        # keeps the condition usable inside a CUDA graph).
        index = mask.nonzero(as_tuple=False).squeeze(-1)
        self.register_buffer("observed_index", index)
        self.register_buffer("observed_target", target_field[index].contiguous())

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Difference between the field and the target at the observed points.

        Args:
            x: flat field ``(B, 1, N)``.

        Returns:
            Residuals ``(B, 1, A)``, where ``A`` is the number of observed points.

        Raises:
            ValueError: if ``x`` has more than one channel or its width does not match
                the mask.
        """
        if x.shape[1] != 1:
            raise ValueError(
                f"{type(self).__name__} is single-channel, but got C={x.shape[1]}. "
                "Use one condition per channel, or a custom condition that indexes "
                "the channel axis explicitly."
            )
        if x.shape[-1] != self.mask.shape[0]:
            raise ValueError(
                f"field width N={x.shape[-1]} does not match mask length "
                f"{self.mask.shape[0]}"
            )
        observed = x.squeeze(1).index_select(1, self.observed_index)
        return (observed - self.observed_target.to(observed)).unsqueeze(1)
