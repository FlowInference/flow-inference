"""Private numerical helpers for the guidance classes, including batched LSMR."""

import math
import warnings
from collections.abc import Callable

import torch


def scrub_logp(logp: torch.Tensor, floor: float = -math.inf) -> torch.Tensor:
    """Replace invalid log-likelihoods by ``floor``, giving them zero weight.

    NaN and ``+inf`` are never valid log-likelihoods (they come from evaluating a
    condition outside its domain), and ``-inf`` is a hard violation; all three become
    ``floor``.

    Args:
        logp: log-likelihoods, any shape.
        floor: replacement value. Pass a large negative finite number when a softmax
            over the result must stay finite even if every entry is invalid.

    Returns:
        ``logp`` with the invalid entries replaced.
    """
    return torch.nan_to_num(logp, nan=floor, posinf=floor, neginf=floor)


def scrub_grad(grad: torch.Tensor) -> torch.Tensor:
    """Zero the non-finite entries of a log-likelihood gradient."""
    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def bind_condition(flow, condition) -> None:
    """Prepare ``condition`` to be evaluated on ``flow``'s samples.

    Gives the condition the flow's ``output_shape``, so that ``condition.as_grid()``
    works without being told the grid. Also warns when a condition holding float64
    tensors will be fed float32 samples: that still works, but every evaluation is
    promoted to float64, which is many times slower on most GPUs.
    """
    condition.bind_output_shape(getattr(flow, "output_shape", None))
    if getattr(flow, "dtype", None) != torch.float32:
        return
    tensors = [*condition.parameters(), *condition.buffers()]
    if any(t.dtype == torch.float64 for t in tensors):
        warnings.warn(
            f"{type(condition).__name__} holds float64 tensors but the flow works in "
            "float32, so the condition will run in float64 (slow on GPUs). Build it "
            "from float32 data or call condition.float().",
            stacklevel=3,
        )


def _sym_ortho(a: torch.Tensor, b: torch.Tensor):
    """A stable Givens rotation: ``(c, s, r)`` with ``r = hypot(a, b)``."""
    r = torch.hypot(a, b)
    safe = torch.where(r > 0, r, torch.ones_like(r))
    c = torch.where(r > 0, a / safe, torch.ones_like(a))
    s = torch.where(r > 0, b / safe, torch.zeros_like(b))
    return c, s, r


def lsmr(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rmatvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    damp: torch.Tensor | float = 0.0,
    iters: int = 20,
) -> torch.Tensor:
    """Solve ``min_x |A x - b|^2 + damp^2 |x|^2`` for every row of ``b`` by LSMR.

    Touches ``A`` only through the products ``x -> A x`` and ``u -> A^T u``. It is the
    Golub-Kahan bidiagonalisation algorithm of Fong & Saunders (2011), the same
    iterates as ``scipy.sparse.linalg.lsmr``. Working with ``A`` and ``A^T``
    separately, rather than with the normal equations ``A^T A``, keeps the effective
    condition number at ``kappa(A)`` instead of ``kappa(A)^2``, which is what makes it
    usable in float32 (Roy, Hauberg & Kramer, 2025). For a wide ``A`` with
    ``damp = 0`` it returns the minimum-norm solution. It runs a fixed number of
    iterations with no data-dependent branching, so it never synchronises with the
    host and can be captured in a CUDA graph. Used by
    :class:`~flow_inference.guidance.GuidancePiGDM`.

    Args:
        matvec: ``x -> A x``, mapping ``(B, n)`` to ``(B, m)``.
        rmatvec: ``u -> A^T u``, mapping ``(B, m)`` to ``(B, n)``.
        b: right-hand sides ``(B, m)``.
        damp: damping, a scalar or one per batch element ``(B,)``.
        iters: number of iterations. Fixed, so that there is no host sync; the
            solution after ``k`` iterations is the best in a ``k``-dimensional
            Krylov subspace, exact once ``k`` reaches the rank of ``A``.

    Returns:
        The solutions ``x``, ``(B, n)``.
    """
    tiny = torch.finfo(b.dtype).tiny
    damp = torch.as_tensor(damp, dtype=b.dtype, device=b.device).expand(b.shape[0])

    def norm(t):
        return t.pow(2).sum(1).sqrt()

    u = b
    beta = norm(u)
    u = u / beta.clamp(min=tiny)[:, None]
    v = rmatvec(u)
    alpha = norm(v)
    v = v / alpha.clamp(min=tiny)[:, None]

    zetabar = alpha * beta
    alphabar = alpha
    rho = torch.ones_like(alpha)
    rhobar = torch.ones_like(alpha)
    cbar = torch.ones_like(alpha)
    sbar = torch.zeros_like(alpha)
    h = v
    hbar = torch.zeros_like(v)
    x = torch.zeros_like(v)
    for _ in range(iters):
        u = matvec(v) - alpha[:, None] * u
        beta = norm(u)
        u = u / beta.clamp(min=tiny)[:, None]
        v = rmatvec(u) - beta[:, None] * v
        alpha = norm(v)
        v = v / alpha.clamp(min=tiny)[:, None]

        # Rotations: first fold in the damping, then eliminate beta.
        _, _, alphahat = _sym_ortho(alphabar, damp)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        thetabar = sbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        hbar = (
            h - (thetabar * rho / (rhoold * rhobarold).clamp(min=tiny))[:, None] * hbar
        )
        x = x + (zeta / (rho * rhobar).clamp(min=tiny))[:, None] * hbar
        h = v - (thetanew / rho.clamp(min=tiny))[:, None] * h
    return torch.nan_to_num(x)
