"""Tests for the flow classes (GaussianFlow, GaussianProcessFlow, GaussianIncrementFlow).

The flows are the unconditional generative backbone: a VP-noise process whose
unconditional drift is zero (the prior is already N(0, I) in the working space),
plus a ``_to_data_space`` map and a Tweedie terminal predictor used by guidance.
"""

import warnings

import gpytorch
import pytest
import torch

from flow_inference.conditions import EqualityCondition
from flow_inference.flows import (
    BasisGaussianProcessFlow,
    GaussianFlow,
    GaussianIncrementFlow,
    GaussianProcessFlow,
    NeuralFlow,
)
from flow_inference.maps import BaseMap
from flow_inference.training_samplers import TrainingSampler
from tests.conftest import NF_H, NF_W, TinyVectorField

N = 8
C = 1
B = 4


def _flow_from_matrix(covar):
    """GP flow for a hand-built covariance, via a kernel that returns it verbatim.

    GaussianProcessFlow takes a kernel now; a couple of tests need a specific
    (possibly rank-deficient) matrix rather than one a real kernel would produce.
    """

    class _Fixed(gpytorch.kernels.Kernel):
        is_stationary = False

        def forward(self, x1, x2, diag=False, **params):
            return covar.diagonal() if diag else covar

    grid = torch.linspace(0, 1, covar.shape[0]).unsqueeze(-1)
    return GaussianProcessFlow(_Fixed(), gpytorch.means.ZeroMean(), grid)


# ---------------------------------------------------------------------------
# GaussianFlow
# ---------------------------------------------------------------------------


class TestGaussianFlow:
    def test_dims_and_default_schedule(self, gaussian_flow):
        assert gaussian_flow.channels == 1 and gaussian_flow.latent_dim == N
        # A default VP schedule is installed when none is passed.
        assert gaussian_flow.schedule is not None

    def test_unconditional_drift_is_zero(self, gaussian_flow):
        x = torch.randn(B, C, N)
        out = gaussian_flow(x, torch.tensor(0.5))
        assert out.shape == x.shape and torch.all(out == 0.0)

    def test_drift_zero_for_various_t(self, gaussian_flow):
        x = torch.randn(B, C, N)
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert torch.all(gaussian_flow(x, torch.tensor(t_val)) == 0.0)

    def test_to_data_space_is_identity(self, gaussian_flow):
        x = torch.randn(B, C, N)
        assert torch.equal(gaussian_flow._to_data_space(x), x)

    def test_predict_terminal_shape(self, gaussian_flow):
        x = torch.randn(B, C, N)
        assert gaussian_flow.predict_terminal(x, torch.tensor(0.5)).shape == x.shape

    def test_predict_terminal_at_t0_is_identity(self, gaussian_flow):
        # α²(0) = 1 ⟹ sqrt(α²) x = x
        x = torch.randn(B, C, N)
        out = gaussian_flow.predict_terminal(x, torch.tensor(0.0))
        assert torch.allclose(out, x, atol=1e-5)

    def test_predict_terminal_shrinks_with_t(self, gaussian_flow):
        x = torch.ones(1, C, N)
        small_t = gaussian_flow.predict_terminal(x, torch.tensor(0.1))
        large_t = gaussian_flow.predict_terminal(x, torch.tensor(0.9))
        assert large_t.abs().mean() < small_t.abs().mean()

    # --- sample_terminal (closed-form posterior, GP-specific extra) ---

    def test_sample_terminal_shape(self, gaussian_flow):
        x = torch.randn(B, C, N)
        samples = gaussian_flow.sample_terminal(x, torch.tensor(0.5), num_samples=7)
        assert samples.shape == (B, 7, C, N)

    def test_sample_terminal_reuses_frozen_noise(self, gaussian_flow):
        # The reparameterisation noise is frozen (cached on the flow) across calls
        # at the same batch/sample count, so gradients through guidance flow
        # through mu only.
        x = torch.randn(B, C, N)
        samples_first = gaussian_flow.sample_terminal(
            x, torch.tensor(0.5), num_samples=5
        )
        samples_second = gaussian_flow.sample_terminal(
            x, torch.tensor(0.5), num_samples=5
        )
        assert torch.allclose(samples_first, samples_second)

    def test_sample_terminal_collapses_to_x_at_t0(self, gaussian_flow):
        # At t≈0 the posterior variance vanishes, so every sample ≈ x.
        x = torch.randn(1, C, N)
        samples = gaussian_flow.sample_terminal(x, torch.tensor(1e-6), num_samples=10)
        assert torch.allclose(samples.mean(1), x, atol=1e-3)

    def test_sample_terminal_mean_matches_tweedie(self, gaussian_flow):
        # E[x_0 | x_t] = sqrt(α²) x_t — the sample mean (many draws) tracks it.
        torch.manual_seed(0)
        x = torch.randn(1, C, N)
        t = torch.tensor(0.5)
        samples = gaussian_flow.sample_terminal(x, t, num_samples=4000)
        expected = torch.sqrt(gaussian_flow.schedule._alpha_2(t)) * x
        assert torch.allclose(samples.mean(1), expected, atol=0.05)


# ---------------------------------------------------------------------------
# GaussianProcessFlow
# ---------------------------------------------------------------------------


class TestGaussianProcessFlow:
    def test_buffers_registered(self, gp_flow):
        # The covariance is not stored directly — only its whitening factor L (and
        # inverse) plus the mean. K is recoverable as L Lᵀ (see test below).
        for name in ("target_mean", "L", "L_inv"):
            assert hasattr(gp_flow, name)

    def test_dims_single_channel(self, gp_flow):
        assert gp_flow.channels == 1 and gp_flow.latent_dim == N

    def test_factors_promoted_to_batch(self, gp_flow):
        # A 2-D covariance / 1-D mean are promoted to a leading channel axis.
        assert gp_flow.L.shape == (1, N, N)
        assert gp_flow.target_mean.shape == (1, N)

    def test_L_times_L_inv_is_identity(self, gp_flow):
        product = torch.bmm(gp_flow.L, gp_flow.L_inv)
        eye = torch.eye(N).unsqueeze(0).expand_as(product)
        assert torch.allclose(product, eye, atol=1e-5)

    def test_L_recovers_covariance(self, gp_flow_correlated):
        # L Lᵀ == K, the defining property of the whitening factor. The covariance
        # is no longer stored, so rebuild the reference from the fixture's definition.
        base = torch.linspace(0, 1, N)
        K_ref = torch.exp(-((base[:, None] - base[None, :]) ** 2) / 0.1)
        K = torch.bmm(gp_flow_correlated.L, gp_flow_correlated.L.transpose(-2, -1))
        assert torch.allclose(K[0], K_ref, atol=1e-4)

    def test_drift_is_zero(self, gp_flow):
        x = torch.randn(B, C, N)
        assert torch.all(gp_flow(x, torch.tensor(0.5)) == 0.0)

    def test_to_data_space_identity_covar_zero_mean(self, gp_flow):
        # L = I for identity covariance, so the map is the identity.
        z = torch.randn(B, C, N)
        assert torch.allclose(gp_flow._to_data_space(z), z, atol=1e-5)

    def test_to_data_space_adds_mean(self):
        from tests.conftest import TEST_GRID, IdentityKernel

        const = gpytorch.means.ConstantMean()
        with torch.no_grad():
            const.constant.fill_(3.0)
        flow = GaussianProcessFlow(IdentityKernel(), const, TEST_GRID)
        z = torch.zeros(1, 1, N)
        assert torch.allclose(
            flow._to_data_space(z), torch.full((1, 1, N), 3.0), atol=1e-5
        )

    def test_to_data_space_produces_correct_covariance(self, gp_flow_correlated):
        # Whitened N(0,I) samples mapped through L should have covariance ≈ K. With
        # the covariance buffer gone, the reference is L Lᵀ (its defining identity).
        torch.manual_seed(0)
        z = torch.randn(20000, 1, N)
        f = gp_flow_correlated._to_data_space(z).squeeze(1)  # (S, N)
        emp_cov = torch.cov(f.T)
        K = torch.bmm(gp_flow_correlated.L, gp_flow_correlated.L.transpose(-2, -1))[0]
        assert torch.allclose(emp_cov, K, atol=0.1)

    def test_predict_terminal_at_t0_is_identity(self, gp_flow):
        x = torch.randn(B, C, N)
        assert torch.allclose(
            gp_flow.predict_terminal(x, torch.tensor(0.0)), x, atol=1e-5
        )

    def test_jitter_keeps_psd_covariance_decomposable(self):
        # A rank-deficient covariance must still decompose thanks to the jitter clamp.
        rank1 = torch.ones(N, N)
        flow = _flow_from_matrix(rank1)
        assert torch.isfinite(flow.L).all() and torch.isfinite(flow.L_inv).all()


# ---------------------------------------------------------------------------
# GaussianProcessFlow — multiple channels
# ---------------------------------------------------------------------------


def _rbf(lengthscale, outputscale=1.0, batch_shape=None):
    batch_shape = torch.Size() if batch_shape is None else batch_shape
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape
    )
    kernel.base_kernel.lengthscale = lengthscale
    kernel.outputscale = outputscale
    return kernel


def _const(value):
    mean = gpytorch.means.ConstantMean()
    with torch.no_grad():
        mean.constant.fill_(value)
    return mean


class TestGaussianProcessFlowMultiChannel:
    """The three ways to build independent channels: sequences of kernels/means, a
    batched kernel, and ``channels=C`` replication of a single kernel."""

    @staticmethod
    def _grid():
        return torch.linspace(0, 1, N).unsqueeze(-1)

    def _gram(self, kernel):
        g = self._grid()
        return kernel(g, g).to_dense().detach()

    def test_kernel_list_gives_one_block_per_channel(self):
        k1, k2 = _rbf(0.1), _rbf(0.5, outputscale=2.0)
        flow = GaussianProcessFlow([k1, k2], [_const(0.0), _const(0.0)], self._grid())
        assert flow.channels == 2 and flow.latent_dim == N  # latent_dim is per channel
        assert flow.L.shape == (2, N, N) and flow.L_inv.shape == (2, N, N)
        for c, k in enumerate((k1, k2)):
            LLt = flow.L[c] @ flow.L[c].T
            assert torch.allclose(LLt, self._gram(k), atol=1e-6)

    def test_mean_list_sets_each_channel(self):
        flow = GaussianProcessFlow(
            [_rbf(0.1), _rbf(0.1)], [_const(1.0), _const(-2.0)], self._grid()
        )
        assert torch.allclose(flow.target_mean[0], torch.full((N,), 1.0))
        assert torch.allclose(flow.target_mean[1], torch.full((N,), -2.0))

    def test_single_mean_is_shared_across_kernel_list(self):
        flow = GaussianProcessFlow([_rbf(0.1), _rbf(0.3)], _const(3.0), self._grid())
        assert flow.target_mean.shape == (2, N)
        assert torch.allclose(flow.target_mean, torch.full((2, N), 3.0))

    def test_channels_replicates_a_single_kernel(self):
        k = _rbf(0.2)
        flow = GaussianProcessFlow(k, _const(0.5), self._grid(), channels=3)
        assert flow.channels == 3 and flow.L.shape == (3, N, N)
        assert torch.allclose(flow.target_mean, torch.full((3, N), 0.5))
        for c in range(3):
            assert torch.allclose(flow.L[c], flow.L[0])

    def test_batched_kernel_supplies_its_own_channels(self):
        batch = torch.Size([2])
        k = _rbf(torch.tensor([[[0.1]], [[0.5]]]), batch_shape=batch)
        flow = GaussianProcessFlow(k, gpytorch.means.ZeroMean(), self._grid())
        assert flow.channels == 2
        for c, ls in enumerate((0.1, 0.5)):
            LLt = flow.L[c] @ flow.L[c].T
            assert torch.allclose(LLt, self._gram(_rbf(ls)), atol=1e-6)

    def test_mismatched_channel_counts_raise(self):
        with pytest.raises(ValueError, match="covariance channels"):
            GaussianProcessFlow(
                [_rbf(0.1), _rbf(0.2)],
                [_const(0.0), _const(0.0), _const(0.0)],
                self._grid(),
            )

    def test_channels_are_independent_in_data_space(self):
        # A latent that is zero in channel 1 must decode to channel 1's mean, whatever
        # channel 0 holds: the whitening is block-diagonal across channels.
        flow = GaussianProcessFlow(
            [_rbf(0.1), _rbf(0.4)], [_const(0.0), _const(5.0)], self._grid()
        )
        z = torch.zeros(B, 2, N)
        z[:, 0] = torch.randn(B, N)
        x = flow._to_data_space(z)
        assert torch.allclose(x[:, 1], torch.full((B, N), 5.0))
        assert not torch.allclose(x[:, 0], torch.zeros(B, N))

    def test_samples_have_each_channels_covariance(self):
        from flow_inference.sampler import IntegratorSampler

        k1, k2 = _rbf(0.1), _rbf(0.5, outputscale=2.0)
        flow = GaussianProcessFlow([k1, k2], [_const(0.0), _const(1.0)], self._grid())
        torch.manual_seed(0)
        x = IntegratorSampler(flow).sample(4000, num_steps=100, verbose=False)
        assert x.shape == (4000, 2, N)
        for c, (k, m) in enumerate(((k1, 0.0), (k2, 1.0))):
            assert torch.allclose(x[:, c].mean(0), torch.full((N,), m), atol=0.15)
            assert torch.allclose(torch.cov(x[:, c].T), self._gram(k), atol=0.2)


# ---------------------------------------------------------------------------
# GaussianIncrementFlow
# ---------------------------------------------------------------------------


class TestGaussianIncrementFlow:
    def test_dims_from_map(self, increment_flow):
        # N is the map's latent count; a single channel.
        assert increment_flow.channels == 1
        assert increment_flow.latent_dim == increment_flow.map.latent_dim

    def test_drift_is_zero(self, increment_flow):
        z = torch.randn(B, 1, increment_flow.latent_dim)
        assert torch.all(increment_flow(z, torch.tensor(0.5)) == 0.0)

    def test_to_data_space_applies_map(self, increment_flow):
        z = torch.randn(B, 1, increment_flow.latent_dim)
        out = increment_flow._to_data_space(z)
        # a 3x3 grid map → (B, 1, 9)
        assert out.shape == (B, 1, increment_flow.map.output_dim)
        assert torch.isfinite(out).all()

    def test_to_data_space_matches_raw_map(self, increment_flow):
        z = torch.randn(B, 1, increment_flow.latent_dim)
        assert torch.allclose(increment_flow._to_data_space(z), increment_flow.map(z))

    def test_predict_terminal_shape(self, increment_flow):
        z = torch.randn(B, 1, increment_flow.latent_dim)
        assert increment_flow.predict_terminal(z, torch.tensor(0.5)).shape == z.shape

    def test_dtype_inherited_from_map(self, make_map):
        flow = GaussianIncrementFlow(map=make_map(3, 3))
        assert flow.dtype == flow.map.dtype


# ---------------------------------------------------------------------------
# NeuralFlow
# ---------------------------------------------------------------------------


class TestNeuralFlow:
    def test_dims_and_grid(self, neural_flow):
        assert neural_flow.channels == 1
        assert neural_flow.latent_dim == neural_flow.H * neural_flow.W

    def test_device_dtype_inherited_from_network(self, neural_flow, tiny_vector_field):
        param = next(tiny_vector_field.parameters())
        assert neural_flow.dtype == param.dtype
        assert neural_flow.device == param.device

    def test_float32_network_does_not_warn(self, tiny_vector_field):
        # The intended pairing: float64 everywhere else, float32 for the network.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            NeuralFlow(H=NF_H, W=NF_W, neural_network=tiny_vector_field)

    def test_float64_network_warns(self):
        # A double-precision velocity field costs several times the compute and buys
        # nothing, so it is worth flagging rather than silently accepting.
        net = TinyVectorField(NF_H, NF_W).double()
        with pytest.warns(UserWarning, match="float64"):
            NeuralFlow(H=NF_H, W=NF_W, neural_network=net)

    def test_single_channel(self, neural_flow):
        assert neural_flow.channels == 1

    def test_default_schedule_installed(self, neural_flow):
        assert neural_flow.schedule is not None

    # --- velocity (__call__) ---

    def test_call_shape_scalar_t(self, neural_flow):
        x = torch.randn(B, C, neural_flow.latent_dim, dtype=neural_flow.dtype)
        assert neural_flow(x, 0.5).shape == (B, C, neural_flow.latent_dim)

    def test_call_shape_tensor_t(self, neural_flow):
        # A per-sample (B,) time tensor (the training path) is accepted too.
        x = torch.randn(B, C, neural_flow.latent_dim, dtype=neural_flow.dtype)
        t = torch.full((B,), 0.5)
        assert neural_flow(x, t).shape == (B, C, neural_flow.latent_dim)

    def test_call_is_no_grad(self, neural_flow):
        # __call__ wraps the network in no_grad — the velocity is a leaf w.r.t. autograd.
        x = torch.randn(B, C, neural_flow.latent_dim, dtype=neural_flow.dtype)
        assert not neural_flow(x, 0.5).requires_grad

    def test_predict_velocity_keeps_gradients(self, neural_flow):
        # The internal path used in fit() must be differentiable.
        x = torch.randn(B, C, neural_flow.latent_dim, dtype=neural_flow.dtype)
        t = torch.full((B,), 0.5)
        v = neural_flow._velocity(x, t)
        assert v.requires_grad

    # --- predict_terminal (Tweedie denoiser) ---

    def test_predict_terminal_shape(self, neural_flow):
        x = torch.randn(B, C, neural_flow.latent_dim, dtype=neural_flow.dtype)
        assert neural_flow.predict_terminal(x, 0.5).shape == (
            B,
            C,
            neural_flow.latent_dim,
        )

    def test_predict_terminal_finite(self, neural_flow):
        x = torch.randn(B, C, neural_flow.latent_dim, dtype=neural_flow.dtype)
        assert torch.isfinite(neural_flow.predict_terminal(x, 0.5)).all()

    def test_predict_terminal_inverts_velocity_at_zero_field(self, neural_flow):
        # With v == 0 the Tweedie algebra collapses to x_data = x_t / sqrt(α_t)
        # plus the noise recovered from -x_t. We just assert it is well-defined and
        # finite across a sweep of t (the denoiser is exercised end-to-end in the
        # sampler tests).
        x = torch.randn(B, C, neural_flow.latent_dim, dtype=neural_flow.dtype)
        for t in (0.1, 0.5, 0.9):
            assert torch.isfinite(neural_flow.predict_terminal(x, t)).all()

    # --- fit ---

    def test_fit_runs_and_returns_none(self, neural_flow, training_sampler):
        assert neural_flow.fit(training_sampler, num_steps=3, batch_size=4) is None

    def test_fit_sets_eval_mode(self, neural_flow, training_sampler):
        neural_flow.fit(training_sampler, num_steps=2, batch_size=4)
        assert not neural_flow.neural_network.training

    def test_fit_updates_parameters(self, neural_flow, training_sampler):
        before = [p.detach().clone() for p in neural_flow.neural_network.parameters()]
        neural_flow.fit(training_sampler, num_steps=5, batch_size=4, lr=1e-2)
        after = list(neural_flow.neural_network.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_fit_runs_without_snr_weighting(self, neural_flow, training_sampler):
        neural_flow.fit(
            training_sampler, num_steps=2, batch_size=4, use_snr_weighting=False
        )

    def test_fit_rejects_mismatched_grid(self, tiny_vector_field, make_map):
        # Sampler grid must match the flow grid.
        flow = NeuralFlow(H=4, W=4, neural_network=tiny_vector_field)
        sampler = TrainingSampler(num_samples=12, map=make_map(3, 3), buffer_size=6)
        with pytest.raises(AssertionError, match="grid size"):
            flow.fit(sampler, num_steps=1, batch_size=4)


# ---------------------------------------------------------------------------
# output_shape / to_display — the natural display reshape applied to samples
# ---------------------------------------------------------------------------


class TestDisplayShape:
    """``flow.output_shape`` is the per-channel shape of a returned sample, and
    ``flow.to_display`` reshapes the flat ``(B, C, output_dim)`` field to
    ``(B, C, *output_shape)`` so callers can plot without a manual ``.view``. The
    diffusion and conditions still work on the flat field — only the final sample
    is reshaped (see the sampler)."""

    def test_gp_flow_shape_is_1d(self, gp_flow):
        # A GP flow is a 1D field: output_shape == (N,), so to_display is a no-op.
        assert gp_flow.output_shape == (N,)
        f = torch.randn(B, 1, N)
        assert torch.equal(gp_flow.to_display(f), f)

    def test_increment_flow_grid_shape(self, increment_flow):
        # A 3x3 grid map -> the flat (B, 1, 9) field displays as (B, 1, 3, 3).
        assert increment_flow.output_shape == (3, 3)
        f = torch.randn(B, 1, increment_flow.output_dim)
        d = increment_flow.to_display(f)
        assert d.shape == (B, 1, 3, 3)
        # row-major reshape: flattening the display recovers the flat field
        assert torch.equal(d.reshape(B, 1, -1), f)

    def test_neural_flow_grid_shape(self, neural_flow):
        assert neural_flow.output_shape == (neural_flow.H, neural_flow.W)
        f = torch.randn(B, 1, neural_flow.output_dim)
        assert neural_flow.to_display(f).shape == (B, 1, neural_flow.H, neural_flow.W)

    def test_coupled_path_flow_stays_flat(self):
        # A coupled, non-grid map (e.g. a 2D SDE path) keeps its (output_dim,) shape
        # and channel axis — to_display must NOT collapse or grid-ify the channels.
        class _Coupled2DPath(BaseMap):
            def __init__(self, n):
                super().__init__(latent_dim=n, output_shape=(n,), channels=2)

            def _map(self, z):  # (B, 2, n) -> (B, 2, n)
                return z

        flow = GaussianIncrementFlow(map=_Coupled2DPath(5))
        assert flow.output_shape == (5,)
        f = torch.randn(B, 2, 5)
        assert flow.to_display(f).shape == (B, 2, 5)


# ---------------------------------------------------------------------------
# device / dtype follow .to() — these are properties derived from an empty
# buffer, not attributes snapshotted at construction. Holding them as plain
# attributes meant every tensor the sampler allocated from them (latents, time
# grid, SDE noise) landed on the pre-move device and sampling raised.
# ---------------------------------------------------------------------------


class TestDeviceDtypeTracking:
    def _flow(self):
        from tests.conftest import TEST_GRID, IdentityKernel

        return GaussianProcessFlow(
            IdentityKernel(), gpytorch.means.ZeroMean(), TEST_GRID
        )

    def test_dtype_follows_double(self):
        f = self._flow().double()
        assert f.dtype == f.L.dtype == torch.float64

    def test_dtype_follows_float(self):
        f = self._flow().double().float()
        assert f.dtype == f.L.dtype == torch.float32

    def test_sample_after_double(self):
        from flow_inference.sampler import IntegratorSampler

        f = self._flow().double()
        out = IntegratorSampler(f).sample(num_samples=2, num_steps=3, verbose=False)
        assert out.dtype == torch.float64

    def test_sampler_delegates_not_snapshots(self):
        from flow_inference.sampler import IntegratorSampler

        f = self._flow()
        s = IntegratorSampler(f)  # built *before* the move
        f.double()
        assert s.dtype == torch.float64

    def test_probe_not_in_state_dict(self):
        # persistent=False, so existing checkpoints still load.
        assert "_probe" not in self._flow().state_dict()

    def test_state_dict_round_trips(self):
        f = self._flow()
        self._flow().load_state_dict(f.state_dict())

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs CUDA to test a device move"
    )
    def test_sample_after_cuda_move(self):
        from flow_inference.sampler import IntegratorSampler

        f = self._flow().to("cuda")
        assert f.device == f.L.device
        out = IntegratorSampler(f).sample(num_samples=2, num_steps=3, verbose=False)
        assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# GaussianProcessFlow built from a kernel + mean function on a grid, rather than
# from a precomputed covariance. The matrix route stays available as
# from_covariance(), which is also the only way to get a batched (C, N, N)
# multi-channel flow — a single kernel cannot express that.
# ---------------------------------------------------------------------------


class TestGaussianProcessFlowFromKernel:
    def _kernel_and_grid(self):
        import gpytorch

        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        kernel.base_kernel.lengthscale = 0.3
        grid = torch.linspace(0, 1, N).unsqueeze(-1)
        return kernel, gpytorch.means.ZeroMean(), grid

    def test_dims_from_grid(self):
        kernel, mean, grid = self._kernel_and_grid()
        flow = GaussianProcessFlow(kernel, mean, grid)
        assert flow.latent_dim == N and flow.channels == 1

    def test_recovers_the_kernel_covariance(self):
        # L Lᵀ == k(grid, grid), the defining property of the whitening factor.
        kernel, mean, grid = self._kernel_and_grid()
        flow = GaussianProcessFlow(kernel, mean, grid)
        K = torch.bmm(flow.L, flow.L.transpose(-2, -1))[0]
        assert torch.allclose(K, kernel(grid, grid).to_dense().detach(), atol=1e-4)

    def test_accepts_posterior_mean_and_kernel(self):
        # The pairing gp_posterior() is built for.
        import gpytorch

        from flow_inference.gp_utils import gp_posterior

        base = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
        base.base_kernel.lengthscale = 0.3
        train_x = torch.rand(6, 1, dtype=torch.float64)
        train_y = torch.sin(6 * train_x.squeeze(-1))
        grid = torch.linspace(0, 1, N, dtype=torch.float64).unsqueeze(-1)

        mean, kern = gp_posterior(base, train_x, train_y, noise=1e-4)
        flow = GaussianProcessFlow(kern, mean, grid)
        assert flow.latent_dim == N
        # the flow's mean is the posterior mean on the grid
        assert torch.allclose(flow.target_mean[0], mean(grid), atol=1e-8)

    def test_mean_is_carried_into_data_space(self):
        import gpytorch

        kernel, _, grid = self._kernel_and_grid()
        const = gpytorch.means.ConstantMean()
        with torch.no_grad():
            const.constant.fill_(2.0)
        flow = GaussianProcessFlow(kernel, const, grid)
        out = flow._to_data_space(torch.zeros(1, 1, N))
        assert torch.allclose(out, torch.full((1, 1, N), 2.0), atol=1e-5)

    def test_sampler_runs_on_kernel_built_flow(self):
        from flow_inference.sampler import IntegratorSampler

        kernel, mean, grid = self._kernel_and_grid()
        flow = GaussianProcessFlow(kernel, mean, grid)
        out = IntegratorSampler(flow).sample(num_samples=3, num_steps=4, verbose=False)
        assert out.shape == (3, 1, N) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# BasisGaussianProcessFlow — a GP flow parameterised by basis weights.
#
# The load-bearing design choice: the basis is applied in to_display (end of
# sampling only), NOT in _to_data_space (which guidance goes through). That keeps
# conditions in weight space, where a basis with closed-form derivatives can supply
# exact ones, while the sampler still returns function values.
# ---------------------------------------------------------------------------

BASIS_R = 24


def _rff_basis(n_side=6):
    import gpytorch

    from flow_inference.flow_utils import build_grid_in_box
    from flow_inference.gp_utils import RFFBasis

    grid = build_grid_in_box(n_side, n_side, "cpu").double()
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    kernel.base_kernel.lengthscale = 0.3
    kernel.outputscale = 0.7
    basis = RFFBasis(
        R=BASIS_R,
        input_dim=grid.shape[-1],
        mean=gpytorch.means.ZeroMean(),
        kernel=kernel,
        device="cpu",
    )
    return basis, grid, n_side * n_side


class _WeightCondition(EqualityCondition):
    """A condition in weight space, using the basis's analytic second derivative."""

    def __init__(self, basis, grid, sigma=1.0):
        super().__init__(sigma=sigma)
        self.register_buffer("d2", basis.compute_derivative(grid, (0,), (2,)))

    def residual(self, w):  # w: (B, C, R)
        return w @ self.d2


class TestBasisGaussianProcessFlow:
    def test_is_a_gaussian_flow(self):
        basis, grid, _ = _rff_basis()
        flow = BasisGaussianProcessFlow(basis, grid)
        assert isinstance(flow, GaussianFlow) and flow.has_standard_normal_marginals

    def test_latent_is_R_and_output_is_N(self):
        basis, grid, n = _rff_basis()
        flow = BasisGaussianProcessFlow(basis, grid)
        assert flow.latent_dim == BASIS_R
        assert flow.output_dim == n

    def test_to_data_space_is_the_identity(self):
        # The weight prior is white, so the diffusion runs on the weights themselves.
        basis, grid, _ = _rff_basis()
        flow = BasisGaussianProcessFlow(basis, grid)
        z = torch.randn(3, 1, BASIS_R, dtype=torch.float64)
        assert torch.equal(flow._to_data_space(z), z)

    def test_sampler_returns_function_space(self):
        from flow_inference.sampler import IntegratorSampler

        basis, grid, n = _rff_basis()
        flow = BasisGaussianProcessFlow(basis, grid)
        out = IntegratorSampler(flow).sample(num_samples=3, num_steps=4, verbose=False)
        assert out.shape == (3, 1, n)

    def test_display_shape_applied(self):
        from flow_inference.sampler import IntegratorSampler

        basis, grid, _ = _rff_basis(n_side=6)
        flow = BasisGaussianProcessFlow(basis, grid, output_shape=(6, 6))
        out = IntegratorSampler(flow).sample(num_samples=2, num_steps=4, verbose=False)
        assert out.shape == (2, 1, 6, 6)

    def test_rejects_output_shape_that_is_not_the_grid(self):
        basis, grid, _ = _rff_basis()
        with pytest.raises(ValueError, match="!= the basis grid size"):
            BasisGaussianProcessFlow(basis, grid, output_shape=(5, 5))

    def test_guidance_sees_weights_not_functions(self):
        # The whole reason the basis lives in to_display rather than _to_data_space.
        from flow_inference.guidance import GuidanceMC

        basis, grid, _ = _rff_basis()
        cond = _WeightCondition(basis, grid)
        flow = BasisGaussianProcessFlow(basis, grid)
        seen = {}
        original = cond.residual
        cond.residual = lambda w: (
            seen.setdefault("shape", tuple(w.shape)),
            original(w),
        )[1]
        GuidanceMC(flow, cond, num_mc_samples=2)(
            torch.randn(3, 1, BASIS_R, dtype=torch.float64), 0.5
        )
        assert seen["shape"][-1] == BASIS_R  # weights, not the N grid values

    def test_matches_manual_weight_flow_plus_sample_functions(self):
        # Equivalent to the workflow this class replaces.
        from flow_inference.guidance import GuidanceMC
        from flow_inference.sampler import IntegratorSampler

        basis, grid, _n = _rff_basis()
        cond = _WeightCondition(basis, grid)

        flow = BasisGaussianProcessFlow(basis, grid)
        torch.manual_seed(0)
        auto = IntegratorSampler(flow).sample(
            num_samples=3,
            num_steps=4,
            guidance=GuidanceMC(flow, cond, num_mc_samples=2),
            verbose=False,
        )

        plain = GaussianFlow(latent_dim=BASIS_R, dtype=torch.float64)
        torch.manual_seed(0)
        weights = IntegratorSampler(plain).sample(
            num_samples=3,
            num_steps=4,
            guidance=GuidanceMC(plain, cond, num_mc_samples=2),
            verbose=False,
        )
        assert torch.allclose(auto, basis.sample_functions(weights, grid), atol=1e-12)


# ---------------------------------------------------------------------------
# Affine data space: the hooks GuidanceMC's fast path is built on
# ---------------------------------------------------------------------------


def _two_channel_gp_flow():
    """Two channels with different correlated covariances and non-zero means, so
    neither L nor the mean can hide a channel mix-up or a missing transpose."""
    grid = torch.linspace(0, 1, N).unsqueeze(-1)
    return GaussianProcessFlow(
        [_rbf(0.2), _rbf(0.5, outputscale=2.0)], [_const(1.0), _const(-2.0)], grid
    )


class TestAffineDataSpace:
    def test_flags(self, gaussian_flow, gp_flow, increment_flow):
        assert gaussian_flow.affine_data_space
        assert gp_flow.affine_data_space
        assert not increment_flow.affine_data_space  # the map is nonlinear

    def test_linear_part_plus_mean_is_to_data_space(self):
        flow = _two_channel_gp_flow()
        z = torch.randn(B, 2, N)
        expected = flow._linear_part(z) + flow.target_mean.unsqueeze(0)
        assert torch.allclose(flow._to_data_space(z), expected)

    def test_pullback_is_adjoint_of_linear_part(self):
        # <A z, g> == <z, A^T g>, per channel.
        flow = _two_channel_gp_flow()
        z, g = torch.randn(B, 2, N), torch.randn(B, 2, N)
        lhs = (flow._linear_part(z) * g).sum(-1)
        rhs = (z * flow._pullback(g)).sum(-1)
        assert torch.allclose(lhs, rhs)

    def test_identity_flow_hooks_are_identity(self, gaussian_flow):
        z = torch.randn(B, C, N)
        assert torch.equal(gaussian_flow._linear_part(z), z)
        assert torch.equal(gaussian_flow._pullback(z), z)

    @pytest.mark.parametrize("t", [1e-4, 0.3, 0.9])
    def test_sample_terminal_data_matches_decoded_draws(self, t):
        flow = _two_channel_gp_flow()
        z_t = torch.randn(B, 2, N)
        data = flow.sample_terminal_data(z_t, torch.tensor(t), num_samples=5)
        draws = flow.sample_terminal(z_t, torch.tensor(t), num_samples=5)  # same noise
        decoded = flow._to_data_space(draws.reshape(B * 5, 2, N)).reshape(B, 5, 2, N)
        assert data.shape == (B, 5, 2, N)
        assert torch.allclose(data, decoded, atol=1e-10)

    def test_sample_terminal_data_keeps_gradient_to_z_t(self):
        flow = _two_channel_gp_flow()
        z_t = torch.randn(B, 2, N, requires_grad=True)
        flow.sample_terminal_data(
            z_t, torch.tensor(0.5), num_samples=3
        ).sum().backward()
        assert z_t.grad is not None and torch.isfinite(z_t.grad).all()

    def test_decoded_noise_is_cached(self):
        flow = _two_channel_gp_flow()
        z_t = torch.randn(B, 2, N)
        flow.sample_terminal_data(z_t, torch.tensor(0.5), num_samples=3)
        cached = flow._z_decoded
        flow.sample_terminal_data(z_t, torch.tensor(0.2), num_samples=3)
        assert flow._z_decoded is cached

    def test_decoded_noise_follows_redrawn_noise(self):
        # Anything that redraws the shared noise (here a resample_noise caller) must
        # invalidate the decoded copy, or the draws would silently go stale.
        flow = _two_channel_gp_flow()
        z_t, t = torch.randn(B, 2, N), torch.tensor(0.5)
        flow.sample_terminal_data(z_t, t, num_samples=3)
        flow.sample_terminal(z_t, t, num_samples=3, resample_noise=True)
        data = flow.sample_terminal_data(z_t, t, num_samples=3)
        draws = flow.sample_terminal(z_t, t, num_samples=3)  # the redrawn noise
        decoded = flow._to_data_space(draws.reshape(B * 3, 2, N)).reshape(B, 3, 2, N)
        assert torch.allclose(data, decoded, atol=1e-10)

    @pytest.mark.parametrize("batch, samples", [(B + 1, 3), (B, 4)])
    def test_decoded_noise_redrawn_on_shape_change(self, batch, samples):
        flow = _two_channel_gp_flow()
        flow.sample_terminal_data(torch.randn(B, 2, N), torch.tensor(0.5), 3)
        out = flow.sample_terminal_data(
            torch.randn(batch, 2, N), torch.tensor(0.5), samples
        )
        assert out.shape == (batch, samples, 2, N)
        assert flow._z_decoded.shape == (batch, samples, 2, N)

    def test_increment_flow_refuses_sample_terminal_data(self, increment_flow):
        z_t = torch.randn(B, 1, increment_flow.latent_dim)
        with pytest.raises(NotImplementedError, match="nonlinear"):
            increment_flow.sample_terminal_data(z_t, torch.tensor(0.5), num_samples=2)
