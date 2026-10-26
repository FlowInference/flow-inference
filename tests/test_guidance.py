"""Tests for guidance approximations.

torch.compile is patched out so that tests run fast on CPU without triggering
Triton/inductor compilation. The runtime behaviour of compile is tested
separately (it's a PyTorch concern, not ours).
"""

from unittest.mock import patch

import gpytorch
import pytest
import torch

from flow_inference.conditions import (
    BaseCondition,
    BinaryCondition,
    EqualityCondition,
    PointWiseCondition,
    ProductCondition,
)
from flow_inference.flows import GaussianProcessFlow
from flow_inference.guidance import (
    GuidanceDPS,
    GuidanceEnsembleKalman,
    GuidanceMC,
    GuidanceMCFisher,
    GuidanceMPGD,
    _snis_weights,
)

N = 8
C = 1
B = 3
MC_SAMPLES = 3  # keep low for speed

_NO_COMPILE = {"new": lambda fn, **kw: fn}  # patch torch.compile to be a no-op


@pytest.fixture
def condition():
    target = torch.zeros(N)
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:2] = True
    return PointWiseCondition(target, mask, sigma=0.1)


@pytest.fixture
def guidance_mc(gp_flow, condition):
    with patch(
        "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
    ):
        return GuidanceMC(gp_flow, condition, num_mc_samples=MC_SAMPLES)


@pytest.fixture
def guidance_fisher(gp_flow, condition):
    with patch(
        "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
    ):
        return GuidanceMCFisher(gp_flow, condition, num_mc_samples=MC_SAMPLES)


@pytest.fixture
def guidance_dps(gp_flow, condition):
    with patch(
        "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
    ):
        return GuidanceDPS(gp_flow, condition)


@pytest.fixture
def guidance_mpgd(gp_flow, condition):
    with patch(
        "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
    ):
        return GuidanceMPGD(gp_flow, condition)


# ---------------------------------------------------------------------------
# GuidanceDPS
# ---------------------------------------------------------------------------


class TestGuidanceDPS:
    def test_output_shape(self, guidance_dps):
        x = torch.randn(B, C, N)
        out = guidance_dps(x, torch.tensor(0.5))
        assert out.shape == (B, C, N)

    def test_output_finite(self, guidance_dps):
        x = torch.randn(B, C, N)
        out = guidance_dps(x, torch.tensor(0.5))
        assert torch.isfinite(out).all()

    def test_output_detached(self, guidance_dps):
        # The sampler must be able to use the result without gradient accumulation
        x = torch.randn(B, C, N)
        out = guidance_dps(x, torch.tensor(0.5))
        assert not out.requires_grad

    def test_input_grad_not_modified(self, guidance_dps):
        # DPS detaches x_t internally; the caller's tensor must remain unaffected
        x = torch.randn(B, C, N, requires_grad=True)
        _ = guidance_dps(x, torch.tensor(0.5))
        assert x.requires_grad  # unchanged

    def test_output_clipped_within_max_norm(self, gp_flow, condition):
        max_norm = 3.0
        with patch(
            "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
        ):
            g = GuidanceDPS(gp_flow, condition, max_norm=max_norm)
        out = g(torch.randn(B, C, N), torch.tensor(0.5))
        per_point_norm = out.pow(2).sum(dim=-1).sqrt()  # (B, C)
        assert (per_point_norm <= max_norm + 1e-4).all()

    def test_different_t_gives_different_output(self, guidance_dps):
        torch.manual_seed(0)
        x = torch.randn(B, C, N)
        out1 = guidance_dps(x, torch.tensor(0.2))
        out2 = guidance_dps(x, torch.tensor(0.8))
        assert not torch.allclose(out1, out2)

    def test_zero_output_on_unmasked_points(self, gp_flow, condition):
        # condition only constrains points 0 and 1; gradient w.r.t. x_t
        # should be non-zero only on those points (identity covar, no cross-terms)
        with patch(
            "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
        ):
            g = GuidanceDPS(gp_flow, condition, max_norm=1e6)
        x = torch.randn(1, C, N)
        out = g(x, torch.tensor(0.3))
        assert torch.allclose(out[0, 0, 2:], torch.zeros(N - 2), atol=1e-5)

    def test_guidance_points_toward_condition(self, gp_flow, condition):
        # The sampler computes x_new = x - dt * dv.  A positive dv on points
        # above the target decreases x toward 0.
        with patch(
            "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
        ):
            g = GuidanceDPS(gp_flow, condition, max_norm=1e6)
        x = torch.zeros(1, C, N)
        x[0, 0, :2] = 5.0
        out = g(x, torch.tensor(0.3))
        x_new = x - 0.01 * out
        assert x_new[0, 0, :2].abs().mean() < x[0, 0, :2].abs().mean()

    def test_nan_gradients_replaced_with_zero(self, guidance_dps):
        # The backward pass can return NaN gradients (e.g. log(0) in logp).
        # nan_to_num must replace them with zero so the ODE step is not corrupted.
        nan_grad = torch.full((B, C, N), float("nan"))
        with patch.object(torch.autograd, "grad", return_value=(nan_grad,)):
            out = guidance_dps(torch.randn(B, C, N), torch.tensor(0.5))
        assert torch.all(out == 0.0)


# ---------------------------------------------------------------------------
# GuidanceMPGD
# ---------------------------------------------------------------------------


class TestGuidanceMPGD:
    def test_output_shape(self, guidance_mpgd):
        x = torch.randn(B, C, N)
        out = guidance_mpgd(x, torch.tensor(0.5))
        assert out.shape == (B, C, N)

    def test_output_finite(self, guidance_mpgd):
        x = torch.randn(B, C, N)
        out = guidance_mpgd(x, torch.tensor(0.5))
        assert torch.isfinite(out).all()

    def test_output_detached(self, guidance_mpgd):
        x = torch.randn(B, C, N)
        out = guidance_mpgd(x, torch.tensor(0.5))
        assert not out.requires_grad

    def test_output_clipped_within_max_norm(self, gp_flow, condition):
        max_norm = 3.0
        with patch(
            "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
        ):
            g = GuidanceMPGD(gp_flow, condition, max_norm=max_norm)
        out = g(torch.randn(B, C, N), torch.tensor(0.5))
        per_point_norm = out.pow(2).sum(dim=-1).sqrt()
        assert (per_point_norm <= max_norm + 1e-4).all()

    def test_different_t_gives_different_output(self, guidance_mpgd):
        torch.manual_seed(0)
        x = torch.randn(B, C, N)
        out1 = guidance_mpgd(x, torch.tensor(0.2))
        out2 = guidance_mpgd(x, torch.tensor(0.8))
        assert not torch.allclose(out1, out2)

    def test_guidance_points_toward_condition(self, gp_flow, condition):
        with patch(
            "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
        ):
            g = GuidanceMPGD(gp_flow, condition, max_norm=1e6)
        x = torch.zeros(1, C, N)
        x[0, 0, :2] = 5.0
        out = g(x, torch.tensor(0.3))
        x_new = x - 0.01 * out
        assert x_new[0, 0, :2].abs().mean() < x[0, 0, :2].abs().mean()

    def test_differs_from_dps(self, gp_flow, condition):
        # DPS and MPGD use different gradient paths so must give different results
        with patch(
            "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
        ):
            dps = GuidanceDPS(gp_flow, condition, max_norm=1e6)
            mpgd = GuidanceMPGD(gp_flow, condition, max_norm=1e6)
        torch.manual_seed(42)
        x = torch.randn(B, C, N)
        t = torch.tensor(0.5)
        assert not torch.allclose(dps(x, t), mpgd(x, t))

    def test_nan_gradients_replaced_with_zero(self, guidance_mpgd):
        nan_grad = torch.full((B, C, N), float("nan"))
        with patch.object(torch.autograd, "grad", return_value=(nan_grad,)):
            out = guidance_mpgd(torch.randn(B, C, N), torch.tensor(0.5))
        assert torch.all(out == 0.0)


# ---------------------------------------------------------------------------
# GuidanceMC
# ---------------------------------------------------------------------------


class TestGuidanceMC:
    def test_output_shape(self, guidance_mc):
        x = torch.randn(B, C, N)
        out = guidance_mc(x, torch.tensor(0.5))
        assert out.shape == (B, C, N)

    def test_output_finite(self, guidance_mc):
        x = torch.randn(B, C, N)
        out = guidance_mc(x, torch.tensor(0.5))
        assert torch.isfinite(out).all()

    def test_output_detached(self, guidance_mc):
        x = torch.randn(B, C, N)
        out = guidance_mc(x, torch.tensor(0.5))
        assert not out.requires_grad

    def test_output_clipped_within_max_norm(self, gp_flow, condition):
        max_norm = 3.0
        with patch(
            "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
        ):
            g = GuidanceMC(
                gp_flow, condition, num_mc_samples=MC_SAMPLES, max_norm=max_norm
            )
        x = torch.randn(B, C, N)
        out = g(x, torch.tensor(0.5))
        per_point_norm = out.pow(2).sum(dim=-1).sqrt()  # (B, C)
        assert (per_point_norm <= max_norm + 1e-4).all()

    def test_different_t_gives_different_output(self, guidance_mc):
        torch.manual_seed(0)
        x = torch.randn(B, C, N)
        out1 = guidance_mc(x, torch.tensor(0.2))
        out2 = guidance_mc(x, torch.tensor(0.8))
        assert not torch.allclose(out1, out2)

    def test_guidance_points_toward_condition(self, gp_flow, condition):
        # The sampler applies  x_new = x - dt * (vf + guidance), so a *positive*
        # guidance value on dimensions where x > target correctly decreases x
        # toward the target.  We verify that applying a small gradient step
        # moves x closer to target=0.
        with patch(
            "flow_inference.guidance.torch.compile", side_effect=lambda fn, **kw: fn
        ):
            g = GuidanceMC(gp_flow, condition, num_mc_samples=10, max_norm=1e6)

        x = torch.zeros(1, C, N)
        x[0, 0, :2] = 5.0  # conditioned points far above target 0

        out = g(x, torch.tensor(0.3))
        dt = 0.01
        x_new = x - dt * out
        # After one gradient step the conditioned points should be closer to 0
        assert x_new[0, 0, :2].abs().mean() < x[0, 0, :2].abs().mean()


# ---------------------------------------------------------------------------
# GuidanceMC affine fast path
# ---------------------------------------------------------------------------


class _QuarticCondition(EqualityCondition):
    """Non-linear, every channel: residual (x - 1)^2 - 0.5, so the score depends on
    the draw and the importance weights are not uniform."""

    def residual(self, x):
        return (x - 1.0) ** 2 - 0.5


def _mc_flow():
    """Two channels, correlated covariances and non-zero means: L is not the
    identity and differs per channel, so a missing transpose or channel mix-up in
    the pullback changes the answer."""

    def rbf(lengthscale, outputscale):
        k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        k.base_kernel.lengthscale = lengthscale
        k.outputscale = outputscale
        return k

    def const(value):
        m = gpytorch.means.ConstantMean()
        with torch.no_grad():
            m.constant.fill_(value)
        return m

    grid = torch.linspace(0, 1, N).unsqueeze(-1)
    return GaussianProcessFlow(
        [rbf(0.2, 1.0), rbf(0.5, 2.0)], [const(0.5), const(-1.0)], grid
    )


def _fast_and_slow(flow, condition, x, t, num_mc_samples=7):
    """Run the affine path, then the per-draw path on the same frozen noise."""
    g = GuidanceMC(flow, condition, num_mc_samples=num_mc_samples, max_norm=None)
    fast, fast_ess = g(x, torch.tensor(t)), g.ess
    flow.affine_data_space = False  # instance override: the per-draw path
    slow, slow_ess = g(x, torch.tensor(t)), g.ess
    return fast, slow, fast_ess, slow_ess


class TestGuidanceMCAffinePath:
    @pytest.mark.parametrize("t", [0.05, 0.5, 0.95])
    def test_matches_per_draw_decode_pointwise(self, gp_flow_correlated, t):
        # Weighting scores in data space and pulling back once must agree with
        # decoding and differentiating every draw through the flow.
        mask = torch.zeros(N, dtype=torch.bool)
        mask[::3] = True
        cond = PointWiseCondition(torch.ones(N), mask, 0.5)
        fast, slow, *_ = _fast_and_slow(
            gp_flow_correlated, cond, torch.randn(B, C, N), t
        )
        assert torch.allclose(fast, slow, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("t", [0.05, 0.5, 0.95])
    def test_matches_per_draw_decode_multichannel_nonlinear(self, t):
        torch.manual_seed(0)
        flow = _mc_flow()
        fast, slow, fast_ess, slow_ess = _fast_and_slow(
            flow, _QuarticCondition(sigma=0.5), torch.randn(B, 2, N), t
        )
        assert fast.shape == (B, 2, N)
        assert torch.allclose(fast, slow, rtol=1e-8, atol=1e-10)
        assert torch.allclose(fast_ess, slow_ess)
        assert (fast_ess < 7).any()  # non-uniform weights, so weighting is exercised

    def test_matches_with_single_draw(self, gp_flow_correlated):
        # S = 1 is what the PDE notebooks run with.
        cond = PointWiseCondition(torch.zeros(N), torch.ones(N, dtype=torch.bool), 0.3)
        fast, slow, *_ = _fast_and_slow(
            gp_flow_correlated, cond, torch.randn(B, C, N), 0.4, num_mc_samples=1
        )
        assert torch.allclose(fast, slow, rtol=1e-9, atol=1e-12)

    def test_decodes_once_per_particle(self, gp_flow_correlated, monkeypatch):
        # The point of the fast path: once the decoded noise is cached, a step maps
        # only the B posterior means through L, while the condition still sees all
        # B * S draws.
        flow, S = gp_flow_correlated, 6
        cond = PointWiseCondition(torch.zeros(N), torch.ones(N, dtype=torch.bool), 0.3)
        g = GuidanceMC(flow, cond, num_mc_samples=S)
        x = torch.randn(B, C, N)
        g(x, torch.tensor(0.5))  # warm the decoded-noise cache

        decoded_rows, pulled_rows, scored_rows = [], [], []
        linear_part, pullback, logp = flow._linear_part, flow._pullback, cond.logp
        monkeypatch.setattr(
            flow,
            "_linear_part",
            lambda z: decoded_rows.append(z.shape[0]) or linear_part(z),
        )
        monkeypatch.setattr(
            flow, "_pullback", lambda v: pulled_rows.append(v.shape[0]) or pullback(v)
        )
        monkeypatch.setattr(
            cond, "logp", lambda f: scored_rows.append(f.shape[0]) or logp(f)
        )
        g(x, torch.tensor(0.4))
        assert decoded_rows == [B]
        assert pulled_rows == [B]
        assert scored_rows == [B * S]

    def test_keeps_frozen_noise(self, gp_flow_correlated, pointwise_condition):
        g = GuidanceMC(gp_flow_correlated, pointwise_condition, num_mc_samples=4)
        x = torch.randn(B, C, N)
        g(x, torch.tensor(0.5))
        noise = gp_flow_correlated.z
        g(x, torch.tensor(0.3))
        assert gp_flow_correlated.z is noise

    def test_nonlinear_flow_uses_per_draw_path(self, increment_flow, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("affine path taken for a nonlinear map")

        monkeypatch.setattr(increment_flow, "sample_terminal_data", refuse)
        cond = PointWiseCondition(
            torch.zeros(increment_flow.output_dim),
            torch.ones(increment_flow.output_dim, dtype=torch.bool),
            0.5,
        )
        g = GuidanceMC(increment_flow, cond, num_mc_samples=3)
        out = g(torch.randn(B, 1, increment_flow.latent_dim), torch.tensor(0.5))
        assert out.shape == (B, 1, increment_flow.latent_dim)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# GuidanceMCFisher
# ---------------------------------------------------------------------------


class TestGuidanceMCFisher:
    def test_output_shape(self, guidance_fisher):
        x = torch.randn(B, C, N)
        out = guidance_fisher(x, torch.tensor(0.5))
        assert out.shape == (B, C, N)

    def test_output_finite(self, guidance_fisher):
        x = torch.randn(B, C, N)
        out = guidance_fisher(x, torch.tensor(0.5))
        assert torch.isfinite(out).all()

    def test_output_detached(self, guidance_fisher):
        x = torch.randn(B, C, N)
        out = guidance_fisher(x, torch.tensor(0.5))
        assert not out.requires_grad

    def test_output_finite_near_t_zero(self, guidance_fisher):
        # alpha_t -> 1 as t -> 0, so guidance_coef = sqrt(alpha_t) / (1 - alpha_t)
        # must stay finite instead of dividing by zero.
        x = torch.randn(B, C, N)
        out = guidance_fisher(x, torch.tensor(1e-6))
        assert torch.isfinite(out).all()

    def test_unclipped_by_default_but_clippable(self, gp_flow):
        # This estimator ships unclipped, so its noise floor — roughly constant in t —
        # is unbounded unless the caller opts in via the inherited attribute.
        g = GuidanceMCFisher(gp_flow, _cond(1e-6), num_mc_samples=8)
        assert g.max_norm is None
        g.max_norm = 3.0
        out = g(torch.randn(B, C, N), torch.tensor(0.5))
        assert (out.pow(2).sum(-1).sqrt() <= 3.0 + 1e-4).all()

    def test_antithetic_cancels_the_no_signal_noise_floor(self, gp_flow):
        # Under a flat likelihood the true guidance is exactly zero. Independent draws
        # instead report a random direction of norm ~std*sqrt(N/S_eff) — the failure
        # that made this estimator inject noise all the way down the trajectory.
        # Mirrored pairs get equal weights, so their contributions cancel identically.
        flat = _FlatCondition(torch.zeros(N), torch.zeros(N, dtype=torch.bool))
        common = dict(num_mc_samples=8, resample_noise=True)
        torch.manual_seed(0)
        f_t = torch.randn(B, C, N)
        paired = GuidanceMCFisher(gp_flow, flat, antithetic=True, **common)
        loose = GuidanceMCFisher(gp_flow, flat, antithetic=False, **common)
        for t in (0.9, 0.1, 0.01):
            assert paired(f_t.clone(), torch.tensor(t)).abs().max() < 1e-12
            assert loose(f_t.clone(), torch.tensor(t)).abs().max() > 1e-3

    def test_resample_noise_redraws_the_ensemble(self, gp_flow, condition):
        # Frozen noise reuses the same S directions at every step, so the MC error
        # integrates into a coherent bias instead of averaging away.
        f_t = torch.randn(B, C, N)
        frozen = GuidanceMCFisher(
            gp_flow, condition, num_mc_samples=8, resample_noise=False
        )
        fresh = GuidanceMCFisher(
            gp_flow, condition, num_mc_samples=8, resample_noise=True
        )
        t = torch.tensor(0.5)
        assert torch.allclose(frozen(f_t.clone(), t), frozen(f_t.clone(), t))
        assert not torch.allclose(fresh(f_t.clone(), t), fresh(f_t.clone(), t))


class _FlatCondition(PointWiseCondition):
    """Uninformative condition: logp is constant, so the true guidance is exactly zero."""

    def logp(self, x):
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Weighting helpers
# ---------------------------------------------------------------------------


class TestWeightHelpers:
    def test_snis_weights_normalised_and_ess_bounded(self):
        logp = torch.randn(5, 16) * 10
        w, ess = _snis_weights(logp)
        assert torch.allclose(w.sum(1), torch.ones(5))
        assert ((ess >= 1.0 - 1e-9) & (ess <= 16.0 + 1e-9)).all()


# ---------------------------------------------------------------------------
# GuidanceEnsembleKalman
# ---------------------------------------------------------------------------


class TestGuidanceEnsembleKalman:
    def test_output_shape_finite_detached(self, gp_flow, condition):
        g = GuidanceEnsembleKalman(gp_flow, condition, num_mc_samples=8)
        out = g(torch.randn(B, C, N), torch.tensor(0.5))
        assert out.shape == (B, C, N)
        assert torch.isfinite(out).all() and not out.requires_grad

    def test_output_finite_near_t_zero(self, gp_flow, condition):
        g = GuidanceEnsembleKalman(gp_flow, condition, num_mc_samples=8)
        assert torch.isfinite(g(torch.randn(B, C, N), torch.tensor(1e-8))).all()

    def test_unclipped_by_default_but_clippable(self, gp_flow):
        # Unlike the SNIS estimator this has no constant noise floor to clip away, so it
        # ships unclipped; the inherited attribute still works if a caller wants a cap.
        g = GuidanceEnsembleKalman(gp_flow, _cond(1e-6), num_mc_samples=8)
        assert g.max_norm is None
        g.max_norm = 3.0
        out = g(torch.randn(B, C, N), torch.tensor(0.5))
        assert (out.pow(2).sum(-1).sqrt() <= 3.0 + 1e-4).all()

    def test_needs_at_least_two_ensemble_members(self, gp_flow, condition):
        with pytest.raises(ValueError, match="num_mc_samples >= 2"):
            GuidanceEnsembleKalman(gp_flow, condition, num_mc_samples=1)

    def test_rejects_zero_iterations(self, gp_flow, condition):
        with pytest.raises(ValueError, match="num_iterations must be >= 1"):
            GuidanceEnsembleKalman(gp_flow, condition, num_iterations=0)

    @pytest.mark.parametrize("pool_gain", [True, False])
    @pytest.mark.parametrize("num_iterations", [1, 3])
    def test_all_option_combinations_run(
        self, gp_flow, condition, pool_gain, num_iterations
    ):
        g = GuidanceEnsembleKalman(
            gp_flow,
            condition,
            num_mc_samples=6,
            pool_gain=pool_gain,
            num_iterations=num_iterations,
        )
        out = g(torch.randn(B, C, N), torch.tensor(0.4))
        assert out.shape == (B, C, N) and torch.isfinite(out).all()

    def test_pooling_falls_back_to_per_particle_for_a_single_sample(
        self, gp_flow, condition
    ):
        # Pooling needs a batch to pool over; B=1 must still produce a valid update
        # rather than an empty or degenerate gram.
        g = GuidanceEnsembleKalman(gp_flow, condition, num_mc_samples=6, pool_gain=True)
        out = g(torch.randn(1, C, N), torch.tensor(0.4))
        assert out.shape == (1, C, N) and torch.isfinite(out).all()
        assert out.abs().sum() > 0

    def test_max_pool_caps_the_solve_size(self, gp_flow, condition):
        # max_pool bounds the pooled gram, so a large batch stays affordable.
        g = GuidanceEnsembleKalman(
            gp_flow, condition, num_mc_samples=4, pool_gain=True, max_pool=8
        )
        out = g(torch.randn(32, C, N), torch.tensor(0.4))
        assert out.shape == (32, C, N) and torch.isfinite(out).all()

    def test_iterations_reduce_the_residual_monotonically(self, gp_flow):
        # ESMDA re-linearises between sub-steps, so more iterations should land the
        # ensemble closer to the constraint, not just take a bigger step.
        cond = _cond(0.05)
        torch.manual_seed(0)
        f_t = torch.randn(8, C, N) * 2.0
        resid = []
        for n in (1, 2, 4):
            g = GuidanceEnsembleKalman(
                gp_flow,
                cond,
                num_mc_samples=32,
                num_iterations=n,
                resample_noise=False,
                pool_gain=False,
            )
            torch.manual_seed(1)
            # Follow the update the sampler would apply, then measure the residual.
            f_hat = gp_flow.predict_terminal(f_t, torch.tensor(0.3))
            step = g(f_t.clone(), torch.tensor(0.3))
            moved = gp_flow._to_data_space(f_hat - 1e-3 * step)
            resid.append(cond.residual(moved).abs().mean().item())
        assert resid[1] < resid[0] and resid[2] < resid[1]

    def test_requires_standardised_residual(self, gp_flow):
        # A condition deriving straight from BaseCondition supplies only logp, which
        # cannot build a Kalman gain — the error must say so rather than fail obscurely.
        g = GuidanceEnsembleKalman(gp_flow, _LogpOnlyCondition(), num_mc_samples=4)
        with pytest.raises(NotImplementedError, match="standardised_residual"):
            g(torch.randn(B, C, N), torch.tensor(0.5))

    def test_guidance_points_toward_condition(self, gp_flow, condition):
        g = GuidanceEnsembleKalman(gp_flow, condition, num_mc_samples=64)
        x = torch.zeros(1, C, N)
        x[0, 0, :2] = 5.0  # conditioned points far above target 0
        x_new = x - 0.01 * g(x, torch.tensor(0.3))
        assert x_new[0, 0, :2].abs().mean() < x[0, 0, :2].abs().mean()

    def test_recovers_the_true_gradient_in_the_small_noise_limit(self, gp_flow):
        # As (1 - alpha_t) -> 0 the ensemble spread and the residual spread both vanish
        # like sqrt(1 - alpha_t), so Delta is O(1 - alpha_t), the 1/(1 - alpha_t)
        # coefficient cancels, and the estimator tends to the exact DPS gradient.
        # This is the property that gives it no constant noise floor to clip away.
        torch.manual_seed(0)
        f_t = torch.randn(16, C, N)
        dps = GuidanceDPS(gp_flow, _cond(0.3), max_norm=None)
        enkg = GuidanceEnsembleKalman(gp_flow, _cond(0.3), num_mc_samples=3000)
        a = dps(f_t.clone(), torch.tensor(2e-3))
        b = enkg(f_t.clone(), torch.tensor(2e-3))
        cos = ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))).mean()
        assert cos > 0.99
        assert (b.norm(dim=-1) / a.norm(dim=-1)).mean() == pytest.approx(1.0, abs=0.1)

    def test_does_not_degenerate_as_sigma_tightens(self, gp_flow):
        # Nothing is exponentiated, so a tight sigma sharpens the gain instead of
        # collapsing an importance-weight ESS — the failure mode that limits MCFisher.
        f_t = torch.randn(B, C, N)
        prev = None
        for sigma in (1.0, 1e-2, 1e-4):
            g = GuidanceEnsembleKalman(gp_flow, _cond(sigma), num_mc_samples=32)
            norm = g(f_t.clone(), torch.tensor(0.3)).norm(dim=-1).mean()
            assert torch.isfinite(norm)
            if prev is not None:
                assert norm > prev  # tighter sigma => stronger pull, not noise
            prev = norm

    def test_works_with_product_and_binary_conditions(self, gp_flow):
        prod = ProductCondition([_cond(0.1), _Positivity(v=0.2)])
        g = GuidanceEnsembleKalman(gp_flow, prod, num_mc_samples=8)
        out = g(torch.randn(B, C, N), torch.tensor(0.4))
        assert out.shape == (B, C, N) and torch.isfinite(out).all()


class _Positivity(BinaryCondition):
    """Every entry should be positive — a BinaryCondition with a hinge residual."""

    def smooth_eval_condition(self, x):
        return x


class _LogpOnlyCondition(BaseCondition):
    """A condition that implements only the logp contract, with no residual view."""

    def logp(self, x):
        return -x.pow(2).sum(dim=[1, 2])


# ---------------------------------------------------------------------------
# Diagnostics + the semantics of max_norm.
#
# The key trap these pin down: once the tanh squash is saturated, every step has
# length exactly max_norm, so the gradient's *magnitude* — and hence the
# condition's sigma — is divided out and has no effect on the samples. Nothing in
# the suite caught that before, which is why sigma sat mis-specified by ~3 orders
# of magnitude for a long time.
# ---------------------------------------------------------------------------


def _cond(sigma):
    target = torch.zeros(N)
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:2] = True
    return PointWiseCondition(target, mask, sigma=sigma)


class TestSigmaSemantics:
    def test_sigma_is_inert_when_saturated(self, gp_flow):
        # max_norm far below the gradient scale => saturated => sigma cancels.
        f_t = torch.randn(B, C, N)
        a = GuidanceDPS(gp_flow, _cond(1e-4), max_norm=1e-3)(f_t.clone(), 0.5)
        b = GuidanceDPS(gp_flow, _cond(1e-6), max_norm=1e-3)(f_t.clone(), 0.5)
        # 1e4x change in sigma, identical drift: the magnitude never reaches the output.
        assert torch.allclose(a, b, atol=1e-9)

    def test_sigma_matters_when_unsaturated(self, gp_flow):
        # With max_norm above the gradient scale the squash is ~linear, so sigma
        # scales the drift as it should.
        f_t = torch.randn(B, C, N)
        a = GuidanceDPS(gp_flow, _cond(1.0), max_norm=1e12)(f_t.clone(), 0.5)
        b = GuidanceDPS(gp_flow, _cond(0.1), max_norm=1e12)(f_t.clone(), 0.5)
        assert not torch.allclose(a, b, atol=1e-6)
        # logp ~ 1/sigma^2, so a 10x smaller sigma is ~100x more drift.
        assert b.norm() > 50 * a.norm()

    def test_saturated_output_norm_equals_max_norm(self, gp_flow):
        g = GuidanceDPS(gp_flow, _cond(1e-6), max_norm=7.0)
        dv = g(torch.randn(B, C, N), 0.5)
        assert torch.allclose(dv.norm(dim=-1), torch.full((B, C), 7.0), atol=1e-5)


class TestGuidanceHistory:
    def _run(self, flow, guidance, num_steps):
        from flow_inference.sampler import IntegratorSampler

        guidance.record_history = True
        guidance.reset_history()
        IntegratorSampler(flow).sample(
            num_samples=2, num_steps=num_steps, guidance=guidance, verbose=False
        )
        return guidance.history

    def test_history_off_by_default(self, gp_flow):
        g = GuidanceDPS(gp_flow, _cond(0.1))
        assert g.record_history is False and g.history == []

    def test_arc_length_is_max_norm_not_step_count(self, gp_flow):
        # ||dv|| is pinned at max_norm and the sampler integrates over Δt = 1, so the
        # total path length is max_norm regardless of how finely it is discretised.
        # This is what makes max_norm a budget rather than a per-step size.
        for num_steps in (20, 80):
            h = self._run(
                gp_flow, GuidanceDPS(gp_flow, _cond(1e-6), max_norm=5.0), num_steps
            )
            ts = [r["t"] for r in h]
            arc = sum(
                abs(ts[i] - ts[i + 1]) * h[i]["out_norm_mean"]
                for i in range(len(ts) - 1)
            )
            assert arc == pytest.approx(5.0, rel=0.05)

    def test_records_saturation(self, gp_flow):
        h = self._run(gp_flow, GuidanceDPS(gp_flow, _cond(1e-6), max_norm=1e-3), 10)
        assert all(r["attenuation_mean"] < 0.1 for r in h)
        assert all(r["frac_over_max"] == 1.0 for r in h)

    def test_persistence_recorded(self, gp_flow):
        h = self._run(gp_flow, GuidanceDPS(gp_flow, _cond(0.1), max_norm=1e2), 10)
        assert h[0]["persistence"] != h[0]["persistence"]  # NaN on the first step
        assert all(-1.001 <= r["persistence"] <= 1.001 for r in h[1:])

    @pytest.mark.parametrize("Cls", [GuidanceMC, GuidanceMCFisher])
    def test_ess_recorded_within_bounds(self, gp_flow, Cls):
        with patch("flow_inference.guidance.torch.compile", **_NO_COMPILE):
            g = Cls(gp_flow, _cond(0.5), num_mc_samples=MC_SAMPLES)
        h = self._run(gp_flow, g, 8)
        assert all(1.0 - 1e-6 <= r["ess_mean"] <= MC_SAMPLES + 1e-6 for r in h)
        assert all(r["num_mc_samples"] == MC_SAMPLES for r in h)
        assert g.ess is not None

    def test_ess_absent_for_non_mc_guidance(self, gp_flow):
        h = self._run(gp_flow, GuidanceDPS(gp_flow, _cond(0.1)), 5)
        assert all(r["ess_mean"] != r["ess_mean"] for r in h)  # NaN

    def test_ess_collapses_when_sigma_too_tight(self, gp_flow):
        # A sigma far below what the residual can reach makes one draw dominate the
        # weights, so the estimator silently degenerates to single-sample DPS.
        with patch("flow_inference.guidance.torch.compile", **_NO_COMPILE):
            g = GuidanceMC(gp_flow, _cond(1e-8), num_mc_samples=MC_SAMPLES)
        h = self._run(gp_flow, g, 8)
        assert min(r["ess_mean"] for r in h) == pytest.approx(1.0, abs=0.05)

    def test_reset_clears_history_and_persistence(self, gp_flow):
        g = GuidanceDPS(gp_flow, _cond(0.1))
        self._run(gp_flow, g, 5)
        g.reset_history()
        assert g.history == [] and g._prev_dir is None
