"""Tests for the batched matrix-free LSMR solver (``flow_inference._numerics.lsmr``).

Each case is checked against a dense solution: ordinary least squares for tall
systems, the minimum-norm solution for wide ones, and the ridge solution with
damping, in float64 and float32.
"""

import pytest
import torch

from flow_inference._numerics import lsmr

F64 = torch.float64


def _ops(A):
    return (
        lambda x: torch.einsum("bmn,bn->bm", A, x),
        lambda u: torch.einsum("bmn,bm->bn", A, u),
    )


def test_tall_system_matches_least_squares():
    torch.manual_seed(0)
    A, b = torch.randn(3, 30, 10, dtype=F64), torch.randn(3, 30, dtype=F64)
    x = lsmr(*_ops(A), b, iters=50)
    expected = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
    assert torch.allclose(x, expected, atol=1e-10)


def test_wide_system_gives_the_minimum_norm_solution():
    torch.manual_seed(0)
    A, b = torch.randn(2, 6, 15, dtype=F64), torch.randn(2, 6, dtype=F64)
    x = lsmr(*_ops(A), b, iters=50)
    expected = torch.einsum("bnm,bm->bn", torch.linalg.pinv(A), b)
    assert torch.allclose(x, expected, atol=1e-10)


def test_damping_per_batch_element_gives_ridge_solutions():
    torch.manual_seed(0)
    A, b = torch.randn(3, 20, 8, dtype=F64), torch.randn(3, 20, dtype=F64)
    damp = torch.tensor([0.0, 0.5, 5.0], dtype=F64)
    x = lsmr(*_ops(A), b, damp=damp, iters=50)
    eye = torch.eye(8, dtype=F64)
    for i in range(3):
        G = A[i].T @ A[i] + damp[i] ** 2 * eye
        assert torch.allclose(x[i], torch.linalg.solve(G, A[i].T @ b[i]), atol=1e-10)


def test_float32_on_an_ill_conditioned_system():
    # Columns scaled over 4 orders of magnitude: condition number ~1e4, which the
    # normal equations would square to ~1e8, beyond float32. LSMR still converges.
    torch.manual_seed(0)
    A = torch.randn(1, 40, 10, dtype=F64) * torch.logspace(0, 4, 10, dtype=F64)
    x_true = torch.randn(1, 10, dtype=F64)
    b = torch.einsum("bmn,bn->bm", A, x_true)
    x = lsmr(*_ops(A.float()), b.float(), iters=100)
    assert x.dtype == torch.float32
    assert (x.double() - x_true).norm() / x_true.norm() < 1e-3


@pytest.mark.parametrize("iters", [0, 1, 3])
def test_fixed_iterations_return_finite_partial_solutions(iters):
    torch.manual_seed(0)
    A, b = torch.randn(2, 12, 5, dtype=F64), torch.randn(2, 12, dtype=F64)
    x = lsmr(*_ops(A), b, iters=iters)
    assert x.shape == (2, 5) and torch.isfinite(x).all()


def test_zero_right_hand_side_gives_zero():
    A = torch.randn(2, 6, 4, dtype=F64)
    assert torch.equal(
        lsmr(*_ops(A), torch.zeros(2, 6, dtype=F64)), torch.zeros(2, 4, dtype=F64)
    )
