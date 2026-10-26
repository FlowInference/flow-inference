import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VPSchedule(nn.Module, ABC):
    """Abstract base class for VP noise schedules.

    Subclasses must implement ``_alpha_2(t)`` (signal power retained at time t)
    and ``_beta(t)`` (instantaneous noise coefficient).
    """

    @abstractmethod
    def _alpha_2(self, t: float | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _beta(self, t: float | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _one_minus_alpha_2(self, t: float | torch.Tensor) -> torch.Tensor:
        """1 - alpha^2(t), computed to remain accurate as alpha^2(t) -> 1.

        Default falls back to naive subtraction; subclasses that know
        alpha^2(t) = exp(-Gamma(t)) should override with an expm1-based form.
        """
        return 1 - self._alpha_2(t)


class LinearSchedule(VPSchedule):
    """Linear VP schedule: β(t) = β_0 + (β_1 - β_0) t, giving α²(t) = exp(-∫₀ᵗ β(s) ds)."""

    def __init__(self, beta_0: torch.Tensor, beta_1: torch.Tensor):
        super().__init__()

        self.register_buffer("beta_0", beta_0)
        self.register_buffer("beta_1", beta_1)
        alpha_1 = torch.sqrt(self._alpha_2(torch.ones_like(beta_1))).item()
        logger.info(
            "Using linear schedule. alpha(1) = %.3e (should be near zero for good denoising at t=1)",
            alpha_1,
        )

    def _gamma(self, t: float | torch.Tensor) -> torch.Tensor:
        # Γ(t) = ∫₀ᵗ β(s) ds
        return self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t * t

    def _alpha_2(self, t: float | torch.Tensor) -> torch.Tensor:
        # α²(t) = exp(-Γ(t))
        return torch.exp(-1 * self._gamma(t))

    def _one_minus_alpha_2(self, t: float | torch.Tensor) -> torch.Tensor:
        # 1 - α²(t) = -expm1(-Γ(t)), accurate even when Γ(t) is tiny (t -> 0)
        return -torch.expm1(-1 * self._gamma(t))

    def _beta(self, t: float | torch.Tensor) -> torch.Tensor:
        return self.beta_0 + (self.beta_1 - self.beta_0) * t
