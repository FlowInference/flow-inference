"""Private numerical helpers for the guidance classes."""

import math
import warnings

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
