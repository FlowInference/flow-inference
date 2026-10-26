import math

import torch

from flow_inference.schedules import VPSchedule


def build_grid_in_box(
    H: int,
    W: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Unit-box grid coordinates: ``(H, 1)`` for 1D (``W == 1``), else ``(H*W, 2)``.

    ``dtype`` defaults to ``float64``, matching the rest of the package. Coordinates
    feed kernels and finite-difference stencils, where float32 quantises the grid
    (~3e-8 at N=4000) and the kernel then silently promotes it back. Pass ``dtype``
    explicitly for a float32 pipeline (a learned NeuralFlow, say).
    """
    dtype = torch.float64 if dtype is None else dtype
    if W == 1:  # 1D
        xs = torch.arange(H, device=device, dtype=dtype) / H  # 1D Grid coordinates
        return xs.view(-1, 1)  # (H, 1)
    else:  # 2D
        ys = torch.arange(H, device=device, dtype=dtype) / H
        xs = torch.arange(W, device=device, dtype=dtype) / W
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        return torch.stack([yy, xx], dim=-1).view(-1, 2)  # (HW, 2)


def clip_vector_field(dv: torch.Tensor, max_norm: float) -> torch.Tensor:
    # Smooth tanh squashing: saturates to max_norm but has no kink,
    # so the ODE solver sees a Lipschitz-continuous vector field.
    norm = torch.linalg.vector_norm(dv, dim=-1, keepdim=True).clamp(
        min=1e-8
    )  # (B, C, 1)
    scale = max_norm * torch.tanh(norm / max_norm) / norm
    return dv * scale


def compute_snr_schedule(
    schedule: VPSchedule,
    num_steps: int,
    t_min: float = 1e-10,
    t_max: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Uniform spacing in log-SNR space, computed via vectorised inversion."""

    # Log-spaced t grid: gives dense coverage near t_min where log-SNR changes
    # extremely rapidly, and sparser coverage near t_max where it is nearly flat.
    M = max(10_000, num_steps * 10)
    t_grid = torch.exp(
        torch.linspace(math.log(t_min), math.log(t_max), M, device=device)
    )
    alpha_sq = schedule._alpha_2(t_grid)
    log_snr_grid = torch.log(alpha_sq / (1 - alpha_sq + 1e-8))  # decreasing in t

    # Target log-SNR values uniformly spaced from t_max (low SNR) to t_min (high SNR)
    targets = torch.linspace(
        log_snr_grid[-1], log_snr_grid[0], num_steps + 1, device=device
    )

    # searchsorted requires an increasing sequence — flip both arrays
    log_snr_asc = log_snr_grid.flip(0)  # now increasing
    t_asc = t_grid.flip(0)  # t_max … t_min
    indices = torch.searchsorted(log_snr_asc.contiguous(), targets.contiguous()).clamp(
        0, M - 1
    )

    return t_asc[indices]
