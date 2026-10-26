"""Tests for the latent→data-space maps.

Every map takes ``(B, latent_dim)`` (or ``(B, 1, latent_dim)``) iid standard
Gaussians and returns a ``(B, 1, output_dim)`` field. The shared contracts —
output shape, finiteness, channel axis, latent-flattening, and (where the map is
smooth) end-to-end differentiability for gradient guidance — are exercised here,
plus a marginal/structural check specific to each map.
"""

import gpytorch
import pytest
import torch

from flow_inference.maps import (
    BaseMap,
    BrownResnickMap,
    CauchyConvolutionMap,
    LevyNIGMap,
    LogGaussianCoxMap,
    StudentTProcessMap,
)

H = W = 3
GRID = H * W
B = 4


# ---------------------------------------------------------------------------
# Base-class dimension model (the latent_dim / output_shape / channels contract)
# ---------------------------------------------------------------------------


class _PathMap(BaseMap):
    """A non-grid map: an Euler-Maruyama scalar SDE path. latent_dim == output_dim,
    and output_shape is 1D — there is no H×W grid."""

    def __init__(self, num_times):
        super().__init__(latent_dim=num_times, output_shape=(num_times,))
        self.num_times = num_times

    def _map(self, z):
        x = z[..., 0]
        path = [x]
        for k in range(self.num_times - 1):
            x = x + 0.1 * (x - x**3) + 0.3 * z[..., k + 1]
            path.append(x)
        return torch.stack(path, dim=-1)


class _CoupledMap(BaseMap):
    """A C=2 coupled map: the two channels interact, so _map reads across the
    channel axis rather than treating each channel independently."""

    def __init__(self, n):
        super().__init__(latent_dim=n, output_shape=(n,), channels=2)
        self.n = n

    def _map(self, z):
        # couple the channels: output channel 0 depends on both inputs
        x = z[:, 0] + z[:, 1]
        y = z[:, 0] - z[:, 1]
        return torch.stack([x, y], dim=1)  # (B, 2, n)


class TestDimensionModel:
    def test_path_map_has_no_grid(self):
        m = _PathMap(5)
        assert m.latent_dim == 5 and m.output_dim == 5
        assert m.output_shape == (5,)
        assert m.channels == 1
        assert m(torch.randn(B, 5)).shape == (B, 1, 5)

    def test_grid_map_output_shape_is_hw(self, kernel):
        m = StudentTProcessMap(H, W, kernel)
        assert m.output_shape == (H, W)
        assert m.output_dim == GRID

    def test_latent_and_output_dim_can_differ(self, kernel):
        # The Cauchy map consumes a finer latent grid than it outputs.
        m = CauchyConvolutionMap(H, W, kernel, grid_fidelity_ratio=2)
        assert m.latent_dim == GRID * 4
        assert m.output_dim == GRID
        assert m.latent_dim != m.output_dim

    def test_coupled_channels_pass_through_to_map(self):
        m = _CoupledMap(4)
        assert m.channels == 2
        z = torch.randn(B, 2, 4)
        out = m(z)
        assert out.shape == (B, 2, 4)
        # channel coupling: out[:,0] = z[:,0]+z[:,1], out[:,1] = z[:,0]-z[:,1]
        assert torch.allclose(out[:, 0], z[:, 0] + z[:, 1])
        assert torch.allclose(out[:, 1], z[:, 0] - z[:, 1])

    def test_map_must_be_implemented(self):
        # BaseMap._map is abstract, so an incomplete subclass cannot be instantiated.
        class _NoMap(BaseMap):
            pass

        with pytest.raises(TypeError):
            _NoMap(latent_dim=3, output_shape=(3,))


@pytest.fixture
def kernel():
    k = gpytorch.kernels.RBFKernel()
    k.lengthscale = 0.25
    return k


# Builders so each map can be parametrised uniformly. ``differentiable`` flags
# maps that are smooth in *all* latents (so a dense gradient is expected); the
# hard-max / hard-pick maps only receive a sparse subgradient.
def _builders(kernel):
    return {
        "cauchy": (lambda: CauchyConvolutionMap(H, W, kernel), True),
        "brown_resnick": (lambda: BrownResnickMap(H, W, kernel, num_storms=16), False),
        "student_t": (lambda: StudentTProcessMap(H, W, kernel, nu=4), True),
        "lgcp": (lambda: LogGaussianCoxMap(H, W, kernel), True),
        "levy_nig": (lambda: LevyNIGMap(H, W, kernel), True),
    }


ALL_MAPS = ["cauchy", "brown_resnick", "student_t", "lgcp", "levy_nig"]


# ---------------------------------------------------------------------------
# Shared contracts across every map
# ---------------------------------------------------------------------------


class TestMapContracts:
    @pytest.mark.parametrize("name", ALL_MAPS)
    def test_output_shape(self, kernel, name):
        m = _builders(kernel)[name][0]()
        out = m(torch.randn(B, m.latent_dim))
        assert out.shape == (B, 1, GRID)

    @pytest.mark.parametrize("name", ALL_MAPS)
    def test_output_finite(self, kernel, name):
        m = _builders(kernel)[name][0]()
        out = m(torch.randn(B, m.latent_dim))
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("name", ALL_MAPS)
    def test_accepts_channel_axis(self, kernel, name):
        # forward() flattens (B, 1, num_latents) the same as (B, num_latents).
        m = _builders(kernel)[name][0]()
        torch.manual_seed(0)
        z = torch.randn(B, m.latent_dim)
        out_flat = m(z)
        out_chan = m(z.unsqueeze(1))
        assert torch.allclose(out_flat, out_chan)

    @pytest.mark.parametrize("name", ALL_MAPS)
    def test_multichannel_maps_each_channel_independently(self, kernel, name):
        # Every map here is a field map: the channel axis rides along, so channel c of
        # a (B, 3, ·) call equals a single-channel call on that slice.
        m = _builders(kernel)[name][0]()
        torch.manual_seed(0)
        z = torch.randn(B, 3, m.latent_dim)
        out = m(z)
        assert out.shape == (B, 3, GRID)
        single = m(z[:, 1]).squeeze(1)  # (B, GRID)
        assert torch.allclose(out[:, 1], single, atol=1e-5)

    @pytest.mark.parametrize("name", ALL_MAPS)
    def test_batch_independence(self, kernel, name):
        # Row i of the output depends only on row i of the latent.
        m = _builders(kernel)[name][0]()
        torch.manual_seed(0)
        z = torch.randn(3, m.latent_dim)
        full = m(z)
        single = m(z[1:2])
        assert torch.allclose(full[1], single[0], atol=1e-5)

    @pytest.mark.parametrize("name", ALL_MAPS)
    def test_gradient_flows(self, kernel, name):
        # Guidance back-props through the map; the gradient must at least be finite
        # and non-trivial (dense for smooth maps, sparse for the hard-pick ones).
        builder, differentiable = _builders(kernel)[name]
        m = builder()
        z = torch.randn(B, m.latent_dim, requires_grad=True)
        (grad,) = torch.autograd.grad(m(z).sum(), z)
        assert torch.isfinite(grad).all()
        assert (grad != 0).any()
        if differentiable:
            # A smooth map should put gradient on a large fraction of latents.
            assert (grad != 0).float().mean() > 0.25


# ---------------------------------------------------------------------------
# Per-map structural / marginal checks
# ---------------------------------------------------------------------------


class TestCauchyConvolutionMap:
    def test_num_latents_scales_with_fidelity(self, kernel):
        m1 = CauchyConvolutionMap(H, W, kernel, grid_fidelity_ratio=1)
        m2 = CauchyConvolutionMap(H, W, kernel, grid_fidelity_ratio=2)
        assert m1.latent_dim == GRID
        assert m2.latent_dim == GRID * 4

    def test_rejects_bad_fidelity_ratio(self, kernel):
        with pytest.raises(AssertionError):
            CauchyConvolutionMap(H, W, kernel, grid_fidelity_ratio=0)

    def test_approximately_gaussian_marginals(self, kernel):
        # The PIT push makes each marginal standard normal.
        torch.manual_seed(0)
        m = CauchyConvolutionMap(H, W, kernel)
        out = m(torch.randn(20000, m.latent_dim)).squeeze(1)
        assert out.mean().abs() < 0.1
        assert abs(out.std().item() - 1.0) < 0.15


class TestBrownResnickMap:
    def test_latent_layout(self, kernel):
        m = BrownResnickMap(H, W, kernel, num_storms=7)
        assert m.latent_dim == GRID * 7 + 7
        assert m.L == 7

    def test_more_storms_better_max_stable_approx(self, kernel):
        # Both run; sanity that storm count only changes latent size, not shape.
        m = BrownResnickMap(H, W, kernel, num_storms=32)
        out = m(torch.randn(2, m.latent_dim))
        assert out.shape == (2, 1, GRID)


class TestStudentTProcessMap:
    def test_latent_layout(self, kernel):
        m = StudentTProcessMap(H, W, kernel, nu=5)
        assert m.latent_dim == GRID + 5
        assert m.nu == 5

    def test_heavier_tails_than_gaussian(self, kernel):
        # t_4 marginals have excess kurtosis → fatter tails than the Cauchy-PIT map.
        torch.manual_seed(0)
        m = StudentTProcessMap(H, W, kernel, nu=4)
        out = m(torch.randn(20000, m.latent_dim)).squeeze(1)
        # kurtosis well above the Gaussian value of 3
        centered = out - out.mean()
        kurt = (centered**4).mean() / (centered.var() ** 2)
        assert kurt > 3.5


# ---------------------------------------------------------------------------
# Marginal normalisation: the heavy-tail maps divide the GP draw by its pointwise
# std sqrt(diag(K)), so their marginals must not depend on the kernel outputscale.
# A ScaleKernel is the common gpytorch idiom that makes diag(K) != 1 and would
# otherwise silently rescale the marginals.
# ---------------------------------------------------------------------------


def _scaled_rbf(outputscale):
    k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    k.base_kernel.lengthscale = 0.25
    k.outputscale = outputscale
    return k


@pytest.mark.parametrize(
    "build, theory_var",
    [
        # t_nu(0, 1) marginals: var = nu / (nu - 2)
        (lambda k: StudentTProcessMap(H, W, k, nu=6), 6 / (6 - 2)),
        # NIG(alpha, 0, 0, delta) marginals: var = delta * alpha^2 / gamma^3, gamma = alpha
        (lambda k: LevyNIGMap(H, W, k, alpha=2.0, beta=0.0, delta=1.0, mu=0.0), 0.5),
    ],
    ids=["student_t", "levy_nig"],
)
@pytest.mark.parametrize("outputscale", [0.25, 1.0, 3.0])
def test_marginals_invariant_to_outputscale(build, theory_var, outputscale):
    torch.manual_seed(0)
    m = build(_scaled_rbf(outputscale))
    out = m(torch.randn(40000, m.latent_dim, dtype=m.dtype)).squeeze(1)
    assert out.var().item() == pytest.approx(theory_var, rel=0.1)


class TestLogGaussianCoxMap:
    def test_intensity_is_positive(self, kernel):
        m = LogGaussianCoxMap(H, W, kernel)
        out = m(torch.randn(B, m.latent_dim))
        assert (out > 0).all()

    def test_mu_shifts_intensity_up(self, kernel):
        torch.manual_seed(0)
        z = torch.randn(2000, GRID)
        low = LogGaussianCoxMap(H, W, kernel, mu=0.0)(z).mean()
        high = LogGaussianCoxMap(H, W, kernel, mu=2.0)(z).mean()
        assert high > low


class TestLevyNIGMap:
    def test_latent_layout(self, kernel):
        m = LevyNIGMap(H, W, kernel)
        assert m.latent_dim == GRID + 2

    def test_rejects_alpha_le_abs_beta(self, kernel):
        with pytest.raises(AssertionError):
            LevyNIGMap(H, W, kernel, alpha=1.0, beta=2.0)

    def test_positive_beta_induces_skew(self, kernel):
        torch.manual_seed(0)
        m = LevyNIGMap(H, W, kernel, alpha=2.0, beta=1.0, mu=0.0)
        out = m(torch.randn(30000, m.latent_dim)).squeeze(1).reshape(-1)
        centered = out - out.mean()
        skew = (centered**3).mean() / (centered.var() ** 1.5 + 1e-8)
        assert skew > 0.1  # right-skew from positive beta


class TestMapDeviceDtypeTracking:
    def test_dtype_follows_double(self, kernel):
        m = StudentTProcessMap(H, W, kernel).double()
        assert m.dtype == m.L_K.dtype == torch.float64

    def test_map_runs_after_double(self, kernel):
        m = StudentTProcessMap(H, W, kernel).double()
        out = m(torch.randn(2, m.latent_dim, dtype=torch.float64))
        assert out.dtype == torch.float64

    def test_probe_not_in_state_dict(self, kernel):
        assert "_probe" not in StudentTProcessMap(H, W, kernel).state_dict()
