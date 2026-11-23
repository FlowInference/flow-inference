"""Small helpers shared across the package: grids, precision and the time grid."""

import copy
import math

import torch

from flow_inference.schedules import VPSchedule


def float64_copy(fn):
    """A float64 copy of a kernel or mean, for one-off setup linear algebra.

    Gram matrices of smooth kernels are badly conditioned, so their factorisations
    (eigen square roots, Choleskys) need float64 even when sampling runs in float32: in
    float32 they reproduce ``K`` only to ~1e-5, which visibly biases large posteriors.
    The package therefore assembles and factorises in float64 and casts the result to
    the working dtype.

    Args:
        fn: a kernel, mean or other callable. ``nn.Module`` objects (gpytorch kernels
            and means, including :class:`~flow_inference.gp_utils.PosteriorKernel`)
            are deep-copied and converted, leaving the caller's object untouched; any
            other callable is returned as is and evaluated at its own precision.

    Returns:
        The float64 copy, or ``fn`` itself if it is not an ``nn.Module``.
    """
    if not isinstance(fn, torch.nn.Module):
        return fn
    # deepcopy refuses tensors that carry an autograd graph (a buffer computed from
    # trainable hyperparameters, a cached evaluation). The copy is only ever evaluated,
    # never differentiated, so hand deepcopy detached clones of those.
    memo = {}
    for module in fn.modules():
        for value in list(vars(module).values()) + list(module._buffers.values()):
            if isinstance(value, torch.Tensor) and not value.is_leaf:
                memo[id(value)] = value.detach().clone()
    return copy.deepcopy(fn, memo).double()


def build_grid_in_box(
    H: int,
    W: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Coordinates of a regular grid in the unit box, in row-major order.

    Point ``i`` along an axis of length ``n`` sits at ``i / n``.

    Args:
        H: number of points along the first axis.
        W: number of points along the second axis; ``W == 1`` gives a 1D grid.
        device: device of the result.
        dtype: dtype of the result; torch's default dtype if None. Kernel matrices
            built from the grid are assembled in float64 regardless (see
            :func:`float64_copy`), so float32 coordinates cost no accuracy there.

    Returns:
        Coordinates ``(H, 1)`` for a 1D grid, else ``(H * W, 2)``.
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if W == 1:
        xs = torch.arange(H, device=device, dtype=dtype) / H
        return xs.view(-1, 1)  # (H, 1)
    else:
        ys = torch.arange(H, device=device, dtype=dtype) / H
        xs = torch.arange(W, device=device, dtype=dtype) / W
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        return torch.stack([yy, xx], dim=-1).view(-1, 2)  # (HW, 2)


def clip_vector_field(dv: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Smoothly bound the norm of a vector field over its last axis.

    Rescales each vector of norm ``n`` to norm ``max_norm * tanh(n / max_norm)``.
    Unlike a hard clip this has no kink, so the ODE solver sees a Lipschitz field.

    Args:
        dv: vector field ``(B, C, N)``.
        max_norm: the norm the result saturates towards.

    Returns:
        The rescaled field, same shape as ``dv``.
    """
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
    """Times from ``t_max`` down to ``t_min`` with equal steps in log-SNR.

    The signal-to-noise ratio is ``alpha^2 / (1 - alpha^2)``. The schedule is
    inverted numerically, by looking the targets up on a fine grid of times.

    Args:
        schedule: the noise schedule.
        num_steps: number of steps.
        t_min: last time.
        t_max: first time.
        device: device of the result.

    Returns:
        ``num_steps + 1`` decreasing times.
    """
    # Log-spaced in t: dense near t_min, where log-SNR changes fastest.
    M = max(10_000, num_steps * 10)
    t_grid = torch.exp(
        torch.linspace(math.log(t_min), math.log(t_max), M, device=device)
    )
    alpha_sq = schedule._alpha_2(t_grid)
    log_snr_grid = torch.log(alpha_sq / (1 - alpha_sq + 1e-8))  # decreasing in t

    # Uniform in log-SNR, from t_max (low SNR) to t_min (high SNR).
    targets = torch.linspace(
        log_snr_grid[-1], log_snr_grid[0], num_steps + 1, device=device
    )

    # searchsorted needs an increasing sequence.
    log_snr_asc = log_snr_grid.flip(0)
    t_asc = t_grid.flip(0)  # t_max ... t_min
    indices = torch.searchsorted(log_snr_asc.contiguous(), targets.contiguous()).clamp(
        0, M - 1
    )

    return t_asc[indices]
