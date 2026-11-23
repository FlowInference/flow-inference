"""Variance-preserving (VP) noise schedules for the diffusion that a flow reverses.

The forward process noises a clean sample ``x_0`` to
``x_t ~ N(alpha(t) x_0, (1 - alpha^2(t)) I)``, with ``alpha^2(1) ~ 0`` so that ``x_1``
is pure noise. A schedule supplies ``alpha^2(t)`` and the noise rate ``beta(t)``, with
``alpha^2(t) = exp(-integral_0^t beta(s) ds)``.
"""

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VPSchedule(nn.Module, ABC):
    """Abstract base class for VP noise schedules.

    Subclasses implement ``_alpha_2(t)`` (the signal power retained at time ``t``)
    and ``_beta(t)`` (the instantaneous noise rate).
    """

    @abstractmethod
    def _alpha_2(self, t: float | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _beta(self, t: float | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _one_minus_alpha_2(self, t: float | torch.Tensor) -> torch.Tensor:
        """``1 - alpha^2(t)``, the noise variance at time ``t``.

        This default subtracts directly, which loses precision as ``alpha^2(t) -> 1``;
        subclasses with ``alpha^2(t) = exp(-Gamma(t))`` in closed form should override
        it with ``-expm1(-Gamma(t))``.
        """
        return 1 - self._alpha_2(t)


class LinearSchedule(VPSchedule):
    """Linear VP schedule, ``beta(t) = beta_0 + (beta_1 - beta_0) t``.

    Args:
        beta_0: noise rate at ``t = 0``.
        beta_1: noise rate at ``t = 1``.
    """

    def __init__(self, beta_0: torch.Tensor, beta_1: torch.Tensor):
        super().__init__()

        self.register_buffer("beta_0", beta_0)
        self.register_buffer("beta_1", beta_1)
        alpha_1 = torch.sqrt(self._alpha_2(torch.ones_like(beta_1))).item()
        logger.info(
            "Using linear schedule. alpha(1) = %.3e "
            "(should be near zero for good denoising at t=1)",
            alpha_1,
        )

    def _gamma(self, t: float | torch.Tensor) -> torch.Tensor:
        # Γ(t) = ∫₀ᵗ β(s) ds
        return self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t * t

    def _alpha_2(self, t: float | torch.Tensor) -> torch.Tensor:
        # α²(t) = exp(-Γ(t))
        return torch.exp(-1 * self._gamma(t))

    def _one_minus_alpha_2(self, t: float | torch.Tensor) -> torch.Tensor:
        # -expm1(-Γ(t)) stays accurate when Γ(t) is tiny (t -> 0).
        return -torch.expm1(-1 * self._gamma(t))

    def _beta(self, t: float | torch.Tensor) -> torch.Tensor:
        return self.beta_0 + (self.beta_1 - self.beta_0) * t
