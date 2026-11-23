"""Tests for mixed precision: factorisations in float64, everything else in the
working dtype.

The package works in float32, but GP Gram matrices are too badly conditioned for
float32 factorisations, so those happen in float64. These tests pin the boundary: the
stored factors are as accurate as float64 ones, float32 and float64 runs agree, and
float64 pieces (conditions built from float64 data, float64 posterior weights) work
with float32 flows.
"""

import warnings

import gpytorch
import pytest
import torch

from flow_inference.conditions import PointWiseCondition
from flow_inference.flow_utils import float64_copy
from flow_inference.flows import GaussianIncrementFlow, GaussianProcessFlow
from flow_inference.gp_utils import DecoupledRFFBasis, gp_posterior
from flow_inference.guidance import GuidanceDPS, GuidanceMC
from flow_inference.maps import StudentTProcessMap
from flow_inference.sampler import IntegratorSampler

F32, F64 = torch.float32, torch.float64


def _rbf(lengthscale, dtype=F64):
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(dtype)
    kernel.base_kernel.lengthscale = lengthscale
    return kernel


def _gram(kernel, grid):
    return float64_copy(kernel)(grid.double(), grid.double()).to_dense().detach()


# ---------------------------------------------------------------------------
# float64_copy
# ---------------------------------------------------------------------------


class TestFloat64Copy:
    def test_copy_is_float64_and_caller_object_untouched(self):
        kernel = _rbf(0.2, F32)
        copy = float64_copy(kernel)
        assert copy.base_kernel.raw_lengthscale.dtype == F64
        assert kernel.base_kernel.raw_lengthscale.dtype == F32

    def test_tolerates_tensors_carrying_an_autograd_graph(self):
        kernel = _rbf(0.2)
        kernel.cached = kernel.base_kernel.lengthscale * 2  # a non-leaf attribute
        assert not kernel.cached.is_leaf
        assert float64_copy(kernel).cached.dtype == F64

    def test_plain_callables_pass_through(self):
        def mean(x):
            return x.sum(-1)

        assert float64_copy(mean) is mean


# ---------------------------------------------------------------------------
# GaussianProcessFlow
# ---------------------------------------------------------------------------


class TestGaussianProcessFlowPrecision:
    def test_dtype_argument_sets_the_working_precision(self):
        grid = torch.linspace(0, 1, 40).unsqueeze(-1)
        flow = GaussianProcessFlow(
            _rbf(0.1), gpytorch.means.ZeroMean(), grid, dtype=F32
        )
        assert flow.dtype == F32
        assert flow.L.dtype == F32 and flow.target_mean.dtype == F32

    def test_default_dtype_follows_the_kernel(self):
        # The kernel, not the grid, decides the dtype, as for the maps and bases.
        grid64 = torch.linspace(0, 1, 20).unsqueeze(-1)
        flow32 = GaussianProcessFlow(_rbf(0.1, F32), gpytorch.means.ZeroMean(), grid64)
        flow64 = GaussianProcessFlow(
            _rbf(0.1), gpytorch.means.ZeroMean(), grid64.float()
        )
        assert flow32.dtype == F32 and flow64.dtype == F64

    def test_parameterless_kernel_follows_the_default_dtype(self):
        from tests.conftest import IdentityKernel

        grid32 = torch.linspace(0, 1, 6, dtype=F32).unsqueeze(-1)
        flow = GaussianProcessFlow(IdentityKernel(), gpytorch.means.ZeroMean(), grid32)
        assert flow.dtype == F32

    def test_float32_factor_is_accurate_for_an_ill_conditioned_kernel(self):
        # RBF with lengthscale 0.1 on 200 points: K is singular to working precision.
        n = 200
        grid = torch.linspace(0, 1, n).unsqueeze(-1)
        kernel = _rbf(0.1)
        K = _gram(kernel, grid)
        flow = GaussianProcessFlow(kernel, gpytorch.means.ZeroMean(), grid, dtype=F32)
        L = flow.L[0].double()
        err = (L @ L.T - K).abs().max().item()

        # The same eigen square root computed in float32 throughout.
        lam, U = torch.linalg.eigh(K.float())
        L_naive = (U * lam.clamp(min=1e-12).sqrt()).double()
        err_naive = (L_naive @ L_naive.T - K).abs().max().item()
        assert err < 1e-6
        assert err < err_naive / 10

    def test_float32_kernel_and_grid_give_the_float64_setup(self):
        grid = torch.linspace(0, 1, 30).unsqueeze(-1)
        kernel32 = _rbf(0.15, F32)
        flow32 = GaussianProcessFlow(kernel32, gpytorch.means.ZeroMean(), grid)
        flow64 = GaussianProcessFlow(_rbf(0.15), gpytorch.means.ZeroMean(), grid)
        assert torch.allclose(flow32.L.double(), flow64.L, atol=1e-6)
        assert kernel32.base_kernel.raw_lengthscale.dtype == F32  # not converted

    def test_float32_flow_built_from_a_float32_posterior(self):
        x = torch.linspace(0.05, 0.95, 10).unsqueeze(-1)
        y = torch.sin(6 * x.squeeze(-1))
        grid = torch.linspace(0, 1, 50).unsqueeze(-1)
        mean32, kern32 = gp_posterior(_rbf(0.2, F32), x.float(), y.float(), noise=1e-10)
        mean64, kern64 = gp_posterior(_rbf(0.2), x, y, noise=1e-10)
        flow32 = GaussianProcessFlow(kern32, mean32, grid)
        flow64 = GaussianProcessFlow(kern64, mean64, grid)
        assert flow32.dtype == F32
        assert torch.allclose(
            flow32.target_mean.double(), flow64.target_mean, atol=1e-5
        )


# ---------------------------------------------------------------------------
# gp_posterior with near-noiseless data
# ---------------------------------------------------------------------------


class TestPosteriorPrecision:
    def test_float32_near_noiseless_posterior_factorises(self):
        # K_XX + 1e-10 I is not positive definite in float32; the factor is float64.
        x = torch.linspace(0.05, 0.95, 10, dtype=F32).unsqueeze(-1)
        y = torch.sin(6 * x.squeeze(-1))
        mean, kern = gp_posterior(_rbf(0.2, F32), x, y, noise=1e-10)
        grid = torch.linspace(0, 1, 40, dtype=F32).unsqueeze(-1)
        m = mean(grid)
        K = kern(grid, grid).to_dense()
        assert m.dtype == F32 and K.dtype == F32
        assert torch.isfinite(m).all() and torch.isfinite(K).all()
        # Interpolates the (near-noiseless) data.
        assert torch.allclose(mean(x), y, atol=1e-3)

    def test_float32_posterior_matches_float64(self):
        x = torch.linspace(0.05, 0.95, 10).unsqueeze(-1)
        y = torch.sin(6 * x.squeeze(-1))
        grid = torch.linspace(0, 1, 40).unsqueeze(-1)
        mean32, kern32 = gp_posterior(_rbf(0.2, F32), x.float(), y.float(), noise=1e-6)
        mean64, kern64 = gp_posterior(_rbf(0.2), x, y, noise=1e-6)
        g32, g64 = grid.float(), grid.double()
        # Outputs come back in the inputs' dtype. The factor and solves are float64 in
        # both, so the only difference is float32 kernel evaluation (~1e-4 here).
        assert mean32(g32).dtype == F32 and mean64(g64).dtype == F64
        assert torch.allclose(mean32(g32).double(), mean64(g64), atol=1e-3)
        assert torch.allclose(
            kern32(g32, g32).to_dense().double(), kern64(g64, g64).to_dense(), atol=1e-3
        )

    def test_decoupled_basis_float32_near_noiseless(self):
        x = torch.linspace(0.05, 0.95, 10, dtype=F32).unsqueeze(-1)
        y = torch.sin(6 * x.squeeze(-1))
        basis = DecoupledRFFBasis(64, 1, _rbf(0.2, F32), x, y, noise=1e-10)
        phi = basis.phi(torch.linspace(0, 1, 30, dtype=F32))
        assert phi.dtype == F32 and torch.isfinite(phi).all()


# ---------------------------------------------------------------------------
# float32 and float64 runs agree
# ---------------------------------------------------------------------------


def test_float32_guided_sampler_matches_float64():
    n, noise = 8, 0.3
    grid = torch.linspace(0, 1, n).unsqueeze(-1)
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = 0.05**0.5
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[1, 4, 6]] = True
    target = torch.zeros(n, dtype=F64)
    target[mask] = torch.tensor([1.0, -0.5, 0.8], dtype=F64)
    out = {}
    for dtype in (F32, F64):
        flow = GaussianProcessFlow(kernel, gpytorch.means.ZeroMean(), grid, dtype=dtype)
        cond = PointWiseCondition(target.to(dtype), mask, noise)
        torch.manual_seed(0)
        guidance = GuidanceMC(flow, cond, num_mc_samples=5)
        samples = IntegratorSampler(flow).sample(
            2000, num_steps=200, guidance=guidance, verbose=False
        )
        out[dtype] = samples.reshape(-1, n)
    assert out[F32].dtype == F32 and torch.isfinite(out[F32]).all()
    # Same estimator, same seed: the two precisions agree to Monte Carlo error.
    assert torch.allclose(out[F32].double().mean(0), out[F64].mean(0), atol=0.05)
    assert torch.allclose(out[F32].double().std(0), out[F64].std(0), atol=0.05)


# ---------------------------------------------------------------------------
# Conditions built from float64 data driving a float32 flow
# ---------------------------------------------------------------------------


def _float64_condition(n):
    """A condition whose buffers are float64, as when it is built from float64 data
    (numpy arrays, the default float64 grid) while the flow works in float32."""
    target = torch.linspace(-1, 1, n, dtype=F64)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[::3] = True
    return PointWiseCondition(target, mask, sigma=1e-3)


@pytest.mark.parametrize(
    "make_guidance",
    [
        lambda f, c: GuidanceMC(f, c, num_mc_samples=4),
        lambda f, c: GuidanceDPS(f, c),
    ],
    ids=["mc", "dps"],
)
def test_float32_flow_accepts_float64_condition(make_guidance):
    n = 12
    grid = torch.linspace(0, 1, n).unsqueeze(-1)
    flow = GaussianProcessFlow(_rbf(0.2), gpytorch.means.ZeroMean(), grid, dtype=F32)
    with pytest.warns(UserWarning, match="float64"):
        guidance = make_guidance(flow, _float64_condition(n))
    drift = guidance(torch.randn(3, 1, n, dtype=F32), torch.tensor(0.5))
    assert drift.dtype == F32 and torch.isfinite(drift).all()
    out = IntegratorSampler(flow).sample(
        3, num_steps=20, guidance=guidance, verbose=False
    )
    assert out.dtype == F32 and torch.isfinite(out).all()


def test_float32_increment_flow_accepts_float64_condition():
    # The non-affine GuidanceMC path (decodes every draw through the map).
    kernel = gpytorch.kernels.RBFKernel().float()
    kernel.lengthscale = 0.25
    flow = GaussianIncrementFlow(map=StudentTProcessMap(4, 4, kernel))
    with pytest.warns(UserWarning, match="float64"):
        guidance = GuidanceMC(flow, _float64_condition(16), num_mc_samples=4)
    out = IntegratorSampler(flow).sample(
        3, num_steps=20, guidance=guidance, verbose=False
    )
    assert out.dtype == F32 and torch.isfinite(out).all()


def test_matching_precision_condition_does_not_warn():
    n = 12
    grid = torch.linspace(0, 1, n).unsqueeze(-1)
    flow = GaussianProcessFlow(_rbf(0.2), gpytorch.means.ZeroMean(), grid, dtype=F32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GuidanceMC(flow, _float64_condition(n).float())


def test_decoupled_basis_float32_mean_derivative():
    # The posterior weights are float64; the float32 derivative must come back float32.
    x = torch.linspace(0.05, 0.95, 10, dtype=F32).unsqueeze(-1)
    y = torch.sin(6 * x.squeeze(-1))
    basis = DecoupledRFFBasis(64, 1, _rbf(0.2, F32), x, y, noise=1e-6)
    grid = torch.linspace(0, 1, 30, dtype=F32).unsqueeze(-1)
    d = basis.mean_derivative(grid, (0,), (1,))
    assert d.dtype == F32 and torch.isfinite(d).all()
