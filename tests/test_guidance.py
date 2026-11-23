"""Tests for the guidance approximations (``flow_inference.guidance``).

DPS, MPGD and Monte Carlo guidance must return a finite, detached, clipped drift that
points toward the condition. The affine path of ``GuidanceMC`` is checked against the
per-draw path on the same frozen noise.
"""

from unittest.mock import patch

import gpytorch
import pytest
import torch

from flow_inference.conditions import (
    EqualityCondition,
    PointWiseCondition,
)
from flow_inference.flows import GaussianProcessFlow
from flow_inference.guidance import (
    GuidanceDPS,
    GuidanceMC,
    GuidanceMPGD,
    _snis_weights,
)

N = 8
C = 1
B = 3
MC_SAMPLES = 3  # keep low for speed


@pytest.fixture
def condition():
    target = torch.zeros(N)
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:2] = True
    return PointWiseCondition(target, mask, sigma=0.1)


@pytest.fixture
def guidance_mc(gp_flow, condition):
    return GuidanceMC(gp_flow, condition, num_mc_samples=MC_SAMPLES)


@pytest.fixture
def guidance_dps(gp_flow, condition):
    return GuidanceDPS(gp_flow, condition)


@pytest.fixture
def guidance_mpgd(gp_flow, condition):
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
        # The sampler uses the result directly, so it must carry no graph.
        x = torch.randn(B, C, N)
        out = guidance_dps(x, torch.tensor(0.5))
        assert not out.requires_grad

    def test_input_grad_not_modified(self, guidance_dps):
        # DPS detaches x_t internally; the caller's tensor is left unaffected.
        x = torch.randn(B, C, N, requires_grad=True)
        _ = guidance_dps(x, torch.tensor(0.5))
        assert x.requires_grad  # unchanged

    def test_output_clipped_within_max_norm(self, gp_flow, condition):
        max_norm = 3.0
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
        # The condition only constrains points 0 and 1, and the identity covariance
        # has no cross-terms, so the drift is non-zero only on those points.
        g = GuidanceDPS(gp_flow, condition, max_norm=1e6)
        x = torch.randn(1, C, N)
        out = g(x, torch.tensor(0.3))
        assert torch.allclose(out[0, 0, 2:], torch.zeros(N - 2), atol=1e-5)

    def test_guidance_points_toward_condition(self, gp_flow, condition):
        # The sampler computes x_new = x - dt * dv, so a positive dv on points above
        # the target moves them toward 0.
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
        g = GuidanceMPGD(gp_flow, condition, max_norm=1e6)
        x = torch.zeros(1, C, N)
        x[0, 0, :2] = 5.0
        out = g(x, torch.tensor(0.3))
        x_new = x - 0.01 * out
        assert x_new[0, 0, :2].abs().mean() < x[0, 0, :2].abs().mean()

    def test_differs_from_dps(self, gp_flow, condition):
        # DPS and MPGD differentiate along different paths, so their drifts differ.
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
        g = GuidanceMC(gp_flow, condition, num_mc_samples=MC_SAMPLES, max_norm=max_norm)
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
        # The sampler computes x_new = x - dt * (vf + guidance), so a positive
        # guidance value where x > target moves x toward the target.
        g = GuidanceMC(gp_flow, condition, num_mc_samples=10, max_norm=1e6)

        x = torch.zeros(1, C, N)
        x[0, 0, :2] = 5.0  # conditioned points far above target 0

        out = g(x, torch.tensor(0.3))
        dt = 0.01
        x_new = x - dt * out
        assert x_new[0, 0, :2].abs().mean() < x[0, 0, :2].abs().mean()


# ---------------------------------------------------------------------------
# GuidanceMC affine path
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
    fast = g(x, torch.tensor(t))
    flow.affine_data_space = False  # instance override: the per-draw path
    slow = g(x, torch.tensor(t))
    return fast, slow


class TestGuidanceMCAffinePath:
    @pytest.mark.parametrize("t", [0.05, 0.5, 0.95])
    def test_matches_per_draw_decode_pointwise(self, gp_flow_correlated, t):
        # Weighting scores in data space and pulling back once must agree with
        # decoding and differentiating every draw through the flow.
        mask = torch.zeros(N, dtype=torch.bool)
        mask[::3] = True
        cond = PointWiseCondition(torch.ones(N), mask, 0.5)
        fast, slow = _fast_and_slow(gp_flow_correlated, cond, torch.randn(B, C, N), t)
        assert torch.allclose(fast, slow, rtol=1e-4, atol=1e-5)  # float32 rounding

    @pytest.mark.parametrize("t", [0.05, 0.5, 0.95])
    def test_matches_per_draw_decode_multichannel_nonlinear(self, t):
        torch.manual_seed(0)
        flow = _mc_flow()
        cond, x = _QuarticCondition(sigma=0.5), torch.randn(B, 2, N)
        fast, slow = _fast_and_slow(flow, cond, x, t)
        assert fast.shape == (B, 2, N)
        assert torch.allclose(fast, slow, rtol=1e-4, atol=1e-5)  # float32 rounding
        # The draws' likelihoods differ, so the importance weighting is exercised.
        draws = flow.sample_terminal(x, torch.tensor(t), 7).reshape(B * 7, 2, N)
        logp = cond.logp(flow._to_data_space(draws)).reshape(B, 7)
        assert (logp.std(1) > 1e-3).all()

    def test_matches_with_single_draw(self, gp_flow_correlated):
        # S = 1 is what the PDE notebooks run with.
        cond = PointWiseCondition(torch.zeros(N), torch.ones(N, dtype=torch.bool), 0.3)
        fast, slow = _fast_and_slow(
            gp_flow_correlated, cond, torch.randn(B, C, N), 0.4, num_mc_samples=1
        )
        assert torch.allclose(fast, slow, rtol=1e-4, atol=1e-5)  # float32 rounding

    def test_decodes_once_per_particle(self, gp_flow_correlated, monkeypatch):
        # The point of the affine path: once the decoded noise is cached, a step maps
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
# Weighting helpers
# ---------------------------------------------------------------------------


class TestWeightHelpers:
    def test_snis_weights_normalised(self):
        w = _snis_weights(torch.randn(5, 16) * 10)
        assert torch.allclose(w.sum(1), torch.ones(5)) and (w >= 0).all()


# ---------------------------------------------------------------------------
# max_norm and sigma
#
# Once the tanh squash saturates, every step has length exactly max_norm, so the
# gradient's magnitude (and hence the condition's sigma) is divided out and has no
# effect on the samples. A mis-specified sigma is invisible in that regime.
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
