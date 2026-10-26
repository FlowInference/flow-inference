"""Tests for the samplers.

``IntegratorSampler`` integrates the probability-flow ODE / VP-SDE, unconditionally
or steered by a gradient ``guidance``.

The flow fixtures (``gp_flow`` etc.) come from conftest; ``pointwise_condition``
pins the first two spatial points to zero, so guidance must pull those points
toward zero relative to the unconditional draw.
"""

import pytest
import torch

from flow_inference.conditions import BaseCondition
from flow_inference.guidance import GuidanceDPS
from flow_inference.sampler import IntegratorSampler, PCNCorrector

N = 8
C = 1
B = 3
STEPS = 5  # keep low so tests run fast


# ---------------------------------------------------------------------------
# IntegratorSampler helpers
# ---------------------------------------------------------------------------


class TestSamplerHelpers:
    def test_inherits_device_and_dtype(self, gp_flow):
        sampler = IntegratorSampler(gp_flow)
        assert sampler.device == gp_flow.device
        assert sampler.dtype == gp_flow.dtype

    @pytest.mark.parametrize("schedule", ["linear", "quadratic", "snr"])
    def test_time_schedule_shape_and_range(self, gp_flow, schedule):
        ts = IntegratorSampler(gp_flow)._get_time_schedule(STEPS, schedule)
        assert ts.shape == (STEPS + 1,)
        assert ts[0] >= ts[-1]  # descending t=1 → t≈0

    def test_invalid_time_schedule_raises(self, gp_flow):
        with pytest.raises(ValueError, match="Unsupported time schedule"):
            IntegratorSampler(gp_flow)._get_time_schedule(STEPS, "nonsense")

    def test_vp_step_coeffs(self, gp_flow):
        sampler = IntegratorSampler(gp_flow)
        t = torch.tensor(0.5)
        dt = torch.tensor(0.1)
        decay, noise_std = sampler._vp_step_coeffs(t, dt)
        beta = gp_flow.schedule._beta(t)
        assert torch.isclose(decay, torch.exp(-0.5 * beta * dt))
        assert torch.isclose(noise_std, torch.sqrt(beta * dt))


# ---------------------------------------------------------------------------
# IntegratorSampler — shapes / schedules / solvers
# ---------------------------------------------------------------------------


class TestIntegratorShapes:
    def test_basic_shape(self, gp_flow):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B, num_steps=STEPS, time_schedule="linear", verbose=False
        )
        assert out.shape == (B, C, N)

    def test_single_sample(self, gp_flow):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=1, num_steps=STEPS, verbose=False
        )
        assert out.shape == (1, C, N)

    def test_increment_flow_shape(self, increment_flow):
        out = IntegratorSampler(increment_flow).sample(
            num_samples=B, num_steps=STEPS, time_schedule="linear", verbose=False
        )
        # A grid-backed increment flow returns the natural (B, C, H, W) display shape
        # (a 3x3 grid map -> (B, 1, 3, 3)), so plotting needs no manual reshape.
        assert out.shape == (B, 1, *increment_flow.output_shape)
        assert increment_flow.output_shape == (
            increment_flow.map.H,
            increment_flow.map.W,
        )

    def test_output_detached(self, gp_flow):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B, num_steps=STEPS, verbose=False
        )
        assert not out.requires_grad

    @pytest.mark.parametrize("schedule", ["linear", "quadratic", "snr"])
    def test_schedule_shape(self, gp_flow, schedule):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B, num_steps=STEPS, time_schedule=schedule, verbose=False
        )
        assert out.shape == (B, C, N)

    @pytest.mark.parametrize("solver", ["euler", "heun", "rk4"])
    def test_solver_shape(self, gp_flow, solver):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            solver=solver,
            time_schedule="linear",
            verbose=False,
        )
        assert out.shape == (B, C, N)

    def test_invalid_solver_raises(self, gp_flow):
        with pytest.raises(ValueError, match="Unknown solver"):
            IntegratorSampler(gp_flow).sample(
                num_samples=1,
                num_steps=STEPS,
                solver="bad",
                time_schedule="linear",
                verbose=False,
            )

    def test_rk4_sde_raises(self, gp_flow):
        with pytest.raises(ValueError, match="SDE integration only supports"):
            IntegratorSampler(gp_flow).sample(
                num_samples=1,
                num_steps=STEPS,
                solver="rk4",
                sde=True,
                time_schedule="linear",
                verbose=False,
            )


# ---------------------------------------------------------------------------
# IntegratorSampler — SDE branch
# ---------------------------------------------------------------------------


class TestIntegratorSDE:
    @pytest.mark.parametrize("solver", ["euler", "heun"])
    def test_sde_shape_and_finite(self, gp_flow, solver):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            sde=True,
            solver=solver,
            time_schedule="linear",
            verbose=False,
        )
        assert out.shape == (B, C, N) and torch.isfinite(out).all()

    def test_sde_differs_from_ode(self, gp_flow):
        torch.manual_seed(0)
        ode = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            sde=False,
            solver="euler",
            time_schedule="linear",
            verbose=False,
        )
        torch.manual_seed(0)
        sde = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            sde=True,
            solver="euler",
            time_schedule="linear",
            verbose=False,
        )
        assert not torch.allclose(ode, sde)


# ---------------------------------------------------------------------------
# IntegratorSampler — final-denoise option
# ---------------------------------------------------------------------------


class TestIntegratorFinalDenoise:
    """The sampler replaces the stiff final ODE step (t→0) with the analytic
    Tweedie denoiser when ``final_denoise=True`` (the default)."""

    @pytest.mark.parametrize("final_denoise", [True, False])
    def test_shape_both_modes(self, gp_flow, final_denoise):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            final_denoise=final_denoise,
            time_schedule="linear",
            verbose=False,
        )
        assert out.shape == (B, C, N) and torch.isfinite(out).all()

    def test_denoise_changes_output(self, gp_flow):
        # The final-step replacement must actually alter the result.
        torch.manual_seed(0)
        denoised = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            final_denoise=True,
            time_schedule="linear",
            verbose=False,
        )
        torch.manual_seed(0)
        plain = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            final_denoise=False,
            time_schedule="linear",
            verbose=False,
        )
        assert not torch.allclose(denoised, plain)

    def test_denoise_default_is_on(self, gp_flow):
        # Calling without the kwarg matches final_denoise=True explicitly.
        torch.manual_seed(0)
        default = IntegratorSampler(gp_flow).sample(
            num_samples=B, num_steps=STEPS, time_schedule="linear", verbose=False
        )
        torch.manual_seed(0)
        explicit = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            final_denoise=True,
            time_schedule="linear",
            verbose=False,
        )
        assert torch.allclose(default, explicit)

    def test_denoise_with_guidance(self, gp_flow, pointwise_condition):
        g = GuidanceDPS(gp_flow, pointwise_condition)
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            guidance=g,
            final_denoise=True,
            time_schedule="linear",
            verbose=False,
        )
        assert out.shape == (B, C, N) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# IntegratorSampler — driving a learned NeuralFlow
# ---------------------------------------------------------------------------


class TestIntegratorNeuralFlow:
    """The IntegratorSampler must drive a NeuralFlow (learned velocity field) just
    like the analytic flows — its predict_terminal feeds the final-denoise step."""

    def test_sample_shape_and_finite(self, neural_flow):
        out = IntegratorSampler(neural_flow).sample(
            num_samples=2, num_steps=STEPS, time_schedule="linear", verbose=False
        )
        assert (
            out.shape == (2, 1, *neural_flow.output_shape) and torch.isfinite(out).all()
        )

    @pytest.mark.parametrize("solver", ["euler", "heun"])
    def test_solvers(self, neural_flow, solver):
        out = IntegratorSampler(neural_flow).sample(
            num_samples=2,
            num_steps=STEPS,
            solver=solver,
            time_schedule="linear",
            verbose=False,
        )
        assert (
            out.shape == (2, 1, *neural_flow.output_shape) and torch.isfinite(out).all()
        )

    def test_sde_branch(self, neural_flow):
        out = IntegratorSampler(neural_flow).sample(
            num_samples=2,
            num_steps=STEPS,
            sde=True,
            solver="euler",
            time_schedule="linear",
            verbose=False,
        )
        assert (
            out.shape == (2, 1, *neural_flow.output_shape) and torch.isfinite(out).all()
        )


# ---------------------------------------------------------------------------
# IntegratorSampler — statistical correctness (unconditional prior recovery)
# ---------------------------------------------------------------------------


class TestIntegratorQuality:
    def test_output_finite(self, gp_flow):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B, num_steps=STEPS, verbose=False
        )
        assert torch.isfinite(out).all()

    def test_output_not_all_zero(self, gp_flow):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B, num_steps=STEPS, verbose=False
        )
        assert out.abs().sum() > 0

    def test_unconditioned_mean_near_zero(self, gp_flow):
        torch.manual_seed(0)
        out = IntegratorSampler(gp_flow).sample(
            num_samples=400, num_steps=20, time_schedule="linear", verbose=False
        )
        assert out.squeeze(1).mean(0).abs().max() < 0.5

    @pytest.mark.slow
    def test_unconditioned_variance_near_one(self, gp_flow):
        torch.manual_seed(0)
        out = IntegratorSampler(gp_flow).sample(
            num_samples=1000,
            num_steps=100,
            time_schedule="snr",
            solver="euler",
            verbose=False,
        )
        assert (out.squeeze(1).var(0) - 1.0).abs().max() < 0.3


# ---------------------------------------------------------------------------
# IntegratorSampler — guidance
# ---------------------------------------------------------------------------


class TestIntegratorGuidance:
    def test_guidance_runs(self, gp_flow, pointwise_condition):
        g = GuidanceDPS(gp_flow, pointwise_condition)
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            guidance=g,
            time_schedule="linear",
            verbose=False,
        )
        assert out.shape == (B, C, N) and torch.isfinite(out).all()

    def test_guidance_pulls_toward_condition(self, gp_flow, pointwise_condition):
        # Condition pins points 0,1 to zero; guided draws should sit closer to 0
        # there than unconditional draws.
        g = GuidanceDPS(gp_flow, pointwise_condition, max_norm=1e3)
        torch.manual_seed(0)
        uncond = IntegratorSampler(gp_flow).sample(
            num_samples=200, num_steps=30, time_schedule="linear", verbose=False
        )
        torch.manual_seed(0)
        guided = IntegratorSampler(gp_flow).sample(
            num_samples=200,
            num_steps=30,
            guidance=g,
            time_schedule="linear",
            verbose=False,
        )
        assert guided[:, 0, :2].abs().mean() < uncond[:, 0, :2].abs().mean()


# ---------------------------------------------------------------------------
# PCNCorrector — gradient-free predictor-corrector
# ---------------------------------------------------------------------------


class FlatCondition(BaseCondition):
    """Uninformative condition: every state is equally likely.

    Makes the pCN accept ratio identically 1, which isolates the *proposal*. Since the
    pCN proposal is reversible w.r.t. N(0, I) — the exact marginal of every Gaussian/GP
    flow at every t — the corrector must then leave that law untouched.
    """

    def logp(self, x):
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


class TestPCNCorrector:
    def test_rejects_flow_without_standard_normal_marginals(self, neural_flow):
        # A learned velocity field has data marginals, not N(0, I), so pCN is not
        # prior-preserving and the accept ratio would be missing a term.
        with pytest.raises(ValueError, match="N\\(0, I\\) marginals"):
            PCNCorrector(neural_flow, FlatCondition())

    def test_accepts_gaussian_family_flows(
        self, gp_flow, gaussian_flow, increment_flow
    ):
        for flow in (gp_flow, gaussian_flow, increment_flow):
            assert PCNCorrector(flow, FlatCondition()).rho == pytest.approx(0.1)

    @pytest.mark.parametrize("rho", [0.0, -0.1, 1.5])
    def test_rejects_rho_outside_unit_interval(self, gp_flow, rho):
        with pytest.raises(ValueError, match="rho must be in"):
            PCNCorrector(gp_flow, FlatCondition(), rho=rho)

    def test_output_shape_and_finite(self, gp_flow, pointwise_condition):
        c = PCNCorrector(gp_flow, pointwise_condition, num_steps=2)
        out = c(torch.randn(B, C, N), torch.tensor(0.5))
        assert out.shape == (B, C, N) and torch.isfinite(out).all()

    def test_zero_steps_is_a_no_op(self, gp_flow, pointwise_condition):
        c = PCNCorrector(gp_flow, pointwise_condition, num_steps=0)
        z = torch.randn(B, C, N)
        assert torch.equal(c(z, torch.tensor(0.5)), z)

    def test_flat_likelihood_accepts_everything(self, gp_flow):
        c = PCNCorrector(gp_flow, FlatCondition(), num_steps=4, record_history=True)
        c(torch.randn(64, C, N), torch.tensor(0.5))
        assert c.history[0]["accept_rate"] == 1.0

    def test_preserves_the_standard_normal_prior(self, gp_flow):
        # The correctness property the whole scheme rests on: with the likelihood flat,
        # the target *is* the prior, and repeated pCN moves must not distort it.
        torch.manual_seed(0)
        c = PCNCorrector(
            gp_flow, FlatCondition(), num_steps=1, rho=0.5, target_accept=None
        )
        z = torch.randn(4000, C, N)
        for _ in range(30):
            z = c(z, torch.tensor(0.5))
        assert z.mean().abs() < 0.05
        assert z.var().item() == pytest.approx(1.0, abs=0.05)

    def test_rho_adapts_upward_when_acceptance_is_too_high(self, gp_flow):
        # A flat likelihood accepts everything, so rho must be driven up toward the
        # independence sampler; the clamp keeps sqrt(1 - rho^2) real.
        c = PCNCorrector(
            gp_flow, FlatCondition(), num_steps=2, rho=0.01, target_accept=0.25
        )
        for _ in range(20):
            c(torch.randn(16, C, N), torch.tensor(0.5))
        assert 0.01 < c.rho <= 1.0

    def test_rho_adapts_downward_when_acceptance_collapses(self, gp_flow):
        # A razor-sharp condition at *small* t is the regime a fixed rho cannot survive:
        # once the chain has found a good state, independent proposals stop beating it
        # and acceptance collapses. Note the state must be carried between calls — with a
        # fresh z each time the chain never accumulates and acceptance sits near 0.5
        # whatever rho is, because the ratio is then two independent noisy estimates.
        # Large t is also inert for the same reason: p(y|z_t) barely depends on z_t there.
        from flow_inference.conditions import PointWiseCondition

        sharp = PointWiseCondition(
            torch.zeros(N), torch.ones(N, dtype=torch.bool), sigma=1e-6
        )
        c = PCNCorrector(
            gp_flow,
            sharp,
            num_steps=2,
            rho=1.0,
            target_accept=0.25,
            record_history=True,
        )
        torch.manual_seed(0)
        z = torch.randn(16, C, N)
        for _ in range(20):
            z = c(z, torch.tensor(0.01))
        assert c.rho < 1.0
        assert c.history[-1]["accept_rate"] < c.history[0]["accept_rate"]

    def test_rho_fixed_when_target_accept_is_none(self, gp_flow, pointwise_condition):
        c = PCNCorrector(gp_flow, pointwise_condition, rho=0.3, target_accept=None)
        for _ in range(5):
            c(torch.randn(B, C, N), torch.tensor(0.5))
        assert c.rho == pytest.approx(0.3)

    def test_history_off_by_default_and_resettable(self, gp_flow, pointwise_condition):
        c = PCNCorrector(gp_flow, pointwise_condition)
        c(torch.randn(B, C, N), torch.tensor(0.5))
        assert c.history == [] and c.acceptance_rate != c.acceptance_rate  # NaN
        c.record_history = True
        c(torch.randn(B, C, N), torch.tensor(0.5))
        assert len(c.history) == 1 and 0.0 <= c.acceptance_rate <= 1.0
        c.reset_history()
        assert c.history == []

    def test_does_not_disturb_a_guidance_noise_cache(
        self, gp_flow, pointwise_condition
    ):
        # The corrector draws its own noise rather than going through
        # flow.sample_terminal, so it must not refresh the frozen-noise cache that
        # GuidanceMC relies on.
        from flow_inference.guidance import GuidanceMC

        g = GuidanceMC(gp_flow, pointwise_condition, num_mc_samples=3)
        g(torch.randn(B, C, N), torch.tensor(0.5))
        cached = gp_flow.z.clone()
        PCNCorrector(gp_flow, pointwise_condition, num_steps=2)(
            torch.randn(B, C, N), torch.tensor(0.5)
        )
        assert torch.equal(gp_flow.z, cached)

    def test_rejects_corrector_built_on_another_flow(
        self, gp_flow, gaussian_flow, pointwise_condition
    ):
        c = PCNCorrector(gaussian_flow, pointwise_condition)
        with pytest.raises(ValueError, match="different flow"):
            IntegratorSampler(gp_flow).sample(
                num_samples=B, num_steps=STEPS, corrector=c, verbose=False
            )

    def test_runs_inside_the_sampler(self, gp_flow, pointwise_condition):
        c = PCNCorrector(gp_flow, pointwise_condition, num_steps=1, record_history=True)
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            corrector=c,
            time_schedule="linear",
            verbose=False,
        )
        assert out.shape == (B, C, N) and torch.isfinite(out).all()
        # final_denoise stops one step early, so the corrector fires on those steps only.
        assert len(c.history) == STEPS - 1

    def test_conditions_without_any_guidance(self, gp_flow, pointwise_condition):
        # The point of the corrector: gradient-free conditioning with guidance=None.
        torch.manual_seed(0)
        uncond = IntegratorSampler(gp_flow).sample(
            num_samples=200, num_steps=30, time_schedule="linear", verbose=False
        )
        torch.manual_seed(0)
        corrected = IntegratorSampler(gp_flow).sample(
            num_samples=200,
            num_steps=30,
            time_schedule="linear",
            verbose=False,
            corrector=PCNCorrector(
                gp_flow, pointwise_condition, num_steps=2, num_mc_samples=4
            ),
        )
        assert corrected[:, 0, :2].abs().mean() < uncond[:, 0, :2].abs().mean()

    def test_composes_with_guidance(self, gp_flow, pointwise_condition):
        out = IntegratorSampler(gp_flow).sample(
            num_samples=B,
            num_steps=STEPS,
            time_schedule="linear",
            verbose=False,
            guidance=GuidanceDPS(gp_flow, pointwise_condition),
            corrector=PCNCorrector(gp_flow, pointwise_condition),
        )
        assert out.shape == (B, C, N) and torch.isfinite(out).all()
