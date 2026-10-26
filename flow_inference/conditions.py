import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn


class BaseCondition(nn.Module, ABC):
    """Abstract base class for conditions: the likelihood ``log p(condition | x)``.

    Subclasses implement :meth:`logp`, mapping a field ``(B, C, N)`` to ``(B,)``.

    A condition always receives the *flat* data-space field ``(B, C, N)``. Conditions
    that reason about a field on an ``H × W`` lattice (PDE residuals, pooling, ...)
    call :meth:`as_grid` to view it as ``(B, C, *output_shape)``; ``output_shape`` is
    set from the flow when the condition is handed to a guidance object or corrector
    (see :meth:`bind_output_shape`). Physical spacings such as ``dx``/``dt`` stay
    explicit constructor arguments.
    """

    def __init__(self):
        super().__init__()
        # Per-channel grid shape, auto-wired from the flow at guidance/sampler time.
        # None until bound (or for conditions that don't care about the grid).
        self.output_shape: tuple[int, ...] | None = None

    def bind_output_shape(self, output_shape: tuple[int, ...] | None) -> None:
        """Record the flow's per-channel ``output_shape`` (e.g. ``(H, W)``).

        Called by guidance objects so grid-aware conditions can
        reshape the flat field via :meth:`as_grid` without being told the grid in
        their constructor. An explicit shape already set on the condition is kept
        (so a condition can override the flow's view if it really needs to);
        otherwise the flow's shape is adopted. Propagates into any nested
        sub-conditions (e.g. :class:`ProductCondition`).
        """
        shape = tuple(output_shape) if output_shape is not None else None
        # modules() yields self and every nested sub-condition.
        for module in self.modules():
            if isinstance(module, BaseCondition) and module.output_shape is None:
                module.output_shape = shape

    def as_grid(self, x: torch.Tensor) -> torch.Tensor:
        """View a flat field ``(B, C, N)`` as ``(B, C, *output_shape)``.

        Requires :attr:`output_shape` to have been bound (it is auto-wired from the
        flow). Validates that the flat width matches the grid so a mismatched flow /
        condition pairing fails loudly rather than silently mis-tiling the field.
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
        """
        Compute logprob for input x .

        Returns:
            logprob: Tensor of shape (B)
        """
        raise NotImplementedError

    def standardised_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Residual view of the condition, ``(B, C, A)``.

        The condition is satisfied when this is zero, and a value of 1 means "off by one
        sigma". Concretely it is ``residual(x) / sigma``,

        :meth:`logp` is enough for anything that differentiates or importance-weights,
        but the ensemble methods
        (:class:`~flow_inference.guidance.GuidanceEnsembleKalman`) need the *vector*
        residual, not the scalar log-density: they build a Kalman gain from the
        ensemble cross-covariance between the field and each residual component, which
        is what lets them stand in for a Jacobian without ever differentiating one.
        Collapsing to logp first would throw that structure away.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define standardised_residual, which "
            "ensemble (derivative-free) guidance needs. Either subclass "
            "EqualityCondition / BinaryCondition, override standardised_residual to "
            "return residual/sigma with shape (B, C, A), or use a logp-based guidance "
            "(GuidanceDPS / GuidanceMC / GuidanceMCFisher)."
        )


class ProductCondition(BaseCondition):
    """
    Combines multiple conditions by summing their log-probs.
    """

    def __init__(self, conditions: Sequence["BaseCondition"]):
        super().__init__()
        self.conditions = nn.ModuleList(conditions)

    def logp(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N)
        returns: logprob (B)
        """

        return torch.stack([cond.logp(x) for cond in self.conditions]).sum(0)

    def standardised_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the sub-conditions' standardised residuals along the component axis.

        Each block arrives already divided by its own sigma, which is what makes the
        components commensurate: a PDE residual and a boundary value live on entirely
        different scales, and a single shared sigma would let the loosest one dominate
        the Kalman gain.
        """
        return torch.cat(
            [cond.standardised_residual(x) for cond in self.conditions], dim=-1
        )


class BinaryCondition(BaseCondition):
    """Smooth relaxation of an inequality (a yes/no condition, e.g. ``x > 0``):
    ``log p = sum logsigmoid(cond(x) / v)``."""

    def __init__(self, v: float = 1e-4):
        super().__init__()
        self.register_buffer("v", torch.tensor(float(v)))

    def logp(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N)
        returns logprob: (B)
        """

        cond = self.smooth_eval_condition(x)
        logp = torch.nn.functional.logsigmoid(cond / self.v)
        return logp.sum(dim=[1, 2])  # (B,)

    def standardised_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth hinge on the constraint margin, in units of ``v``: ``softplus(-cond / v)``.

        An inequality has no target to regress onto, so the residual is the *violation*:
        it decays to zero once ``cond`` is comfortably positive (satisfied) and grows
        like ``-cond / v`` when it is negative. The tail shape differs from
        :meth:`logp`'s logsigmoid — quadratic rather than linear far into the violated
        region — but both are minimised on exactly the same set, and the ensemble update
        only uses the residual to decide a direction and a gain.
        """
        return torch.nn.functional.softplus(-self.smooth_eval_condition(x) / self.v)

    @abstractmethod
    def smooth_eval_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N)
        returns logprob: (B, C, A)

        Evaluates the binary condition on input x. Should return a cts tensor of shape (B, C, A) where
        values > 0 indicate condition is satisfied and values < 0 indicate condition
        is not satisfied. Conditions have A components (e.g. monotonicity has A=H-1 components corresponding to each pair of adjacent pixels).
        """
        raise NotImplementedError


class EqualityCondition(BaseCondition):
    """
    Gaussian-likelihood relaxation of an equality constraint: log p(condition | x) = -‖residual(x)‖² / (2σ²).
    """

    def __init__(self, sigma: float = 1e-4):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    def logp(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N)
        returns logprob: (B)
        """

        diff = self.residual(x)  # [B, C, A]
        logp = -0.5 * (diff / self.sigma) ** 2
        return logp.sum(dim=[1, 2])  # (B,)

    def standardised_residual(self, x: torch.Tensor) -> torch.Tensor:
        """The equality residual, expressed in units of its own sigma."""
        return self.residual(x) / self.sigma

    @abstractmethod
    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N)
        returns residual: (B, C, A)

        Evaluates the residual of the condition on x. Should return a cts tensor of shape (B, C, A)
        for conditions with A components.
        """
        raise NotImplementedError


class PointWiseCondition(EqualityCondition):
    """
    Point-wise conditioning. Give whole field x, but only care about specific points, specified by mask.
    """

    def __init__(
        self, target_field: torch.Tensor, mask: torch.Tensor, sigma: float = 0.1
    ):
        super().__init__(sigma=sigma)
        self.register_buffer("mask", mask)  # (N)
        self.register_buffer("target_field", target_field)  # (N)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N) where C=1
        returns residual: (B, C, A) where A is the number of masked points

        Single-channel only: the mask indexes the field axis.
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
        x_2d = x.squeeze(1)  # (B, N)
        # Extract only masked pixels
        diff = x_2d[:, self.mask] - self.target_field[None, self.mask]  # (B, A)
        return diff.unsqueeze(1)  # (B, C=1, A)
