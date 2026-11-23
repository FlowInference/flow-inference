"""Tests for the sampler (``flow_inference.sampler``).

``IntegratorSampler`` integrates the probability-flow ODE or VP-SDE, unconditionally
or steered by a gradient ``guidance``. The ``pointwise_condition`` fixture pins the
first two spatial points to zero, so guidance must pull those points toward zero
relative to the unconditional draw.
"""

import pytest
import torch

from flow_inference.conditions import EqualityCondition, PointWiseCondition
from flow_inference.guidance import GuidanceDPS, GuidanceMC
from flow_inference.sampler import IntegratorSampler

N = 8
C = 1
B = 3
STEPS = 5  # keep low for speed


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
        # A grid-backed flow returns the (B, C, H, W) display shape: a 3x3 grid map
        # gives (B, 1, 3, 3).
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
    """With ``final_denoise=True`` (the default) the stiff final step to t = 0 is
    taken by the Tweedie denoiser rather than the integrator."""

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
        # The denoised final step must actually change the result.
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
    """The sampler drives a NeuralFlow (learned velocity field) just like the
    analytic flows; its predict_terminal feeds the final-denoise step."""

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
# IntegratorSampler — adaptive step sizes
# ---------------------------------------------------------------------------


class TestIntegratorAdaptive:
    @pytest.mark.parametrize("sde", [False, True])
    @pytest.mark.parametrize("solver", ["euler", "heun"])
    def test_shape_and_stats(self, gp_flow, pointwise_condition, solver, sde):
        s = IntegratorSampler(gp_flow)
        g = GuidanceDPS(gp_flow, pointwise_condition)
        out = s.sample(
            num_samples=B,
            guidance=g,
            adaptive=True,
            solver=solver,
            sde=sde,
            t_min=1e-2,
            verbose=False,
        )
        assert out.shape == (B, C, N) and torch.isfinite(out).all()
        st = s.adaptive_stats
        trials = st["accepted"] + st["rejected"]
        if solver == "euler":
            # First-same-as-last: one initial eval plus one per trial step.
            assert st["nfe"] == 1 + trials
        else:
            # Heun re-evaluates at each accepted state, bar the last.
            assert st["nfe"] == trials + st["accepted"]
        assert st["ts"][0] == 1.0 and st["ts"][-1] == 1e-2
        assert (st["ts"].diff() < 0).all()

    @pytest.mark.parametrize("sde", [False, True])
    def test_tighter_tolerance_takes_more_steps(
        self, gp_flow, pointwise_condition, sde
    ):
        s = IntegratorSampler(gp_flow)
        g = GuidanceDPS(gp_flow, pointwise_condition)
        steps = []
        for tol in (1e-1, 1e-3):
            torch.manual_seed(0)
            s.sample(
                num_samples=B,
                rtol=tol,
                atol=tol,
                guidance=g,
                adaptive=True,
                sde=sde,
                verbose=False,
            )
            steps.append(s.adaptive_stats["accepted"])
        assert steps[1] > steps[0]

    def test_step_cap_raises(self, gp_flow, pointwise_condition):
        g = GuidanceDPS(gp_flow, pointwise_condition)
        with pytest.raises(RuntimeError, match="max_steps"):
            IntegratorSampler(gp_flow).sample(
                num_samples=B,
                guidance=g,
                adaptive=True,
                rtol=1e-6,
                atol=1e-6,
                max_steps=3,
                verbose=False,
            )

    def test_num_steps_with_adaptive_raises(self, gp_flow):
        with pytest.raises(ValueError, match="num_steps"):
            IntegratorSampler(gp_flow).sample(
                num_samples=B, num_steps=50, adaptive=True, verbose=False
            )

    def test_rk4_raises(self, gp_flow):
        with pytest.raises(ValueError, match="Adaptive"):
            IntegratorSampler(gp_flow).sample(
                num_samples=B, adaptive=True, solver="rk4", verbose=False
            )

    def test_neural_flow(self, neural_flow):
        out = IntegratorSampler(neural_flow).sample(
            num_samples=2, adaptive=True, sde=True, t_min=1e-2, verbose=False
        )
        assert torch.isfinite(out).all()


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
        # The condition pins points 0 and 1 to zero, so guided draws sit closer to 0
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
# CUDA graph capture of the integration step
# ---------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


class TestCudaGraph:
    def test_ignored_on_cpu(self, gp_flow):
        torch.manual_seed(0)
        plain = IntegratorSampler(gp_flow).sample(4, num_steps=8, verbose=False)
        torch.manual_seed(0)
        graphed = IntegratorSampler(gp_flow).sample(
            4, num_steps=8, verbose=False, cuda_graph=True
        )
        assert torch.equal(plain, graphed)

    @needs_cuda
    @pytest.mark.parametrize("sde", [False, True])
    @pytest.mark.parametrize("solver", ["euler", "heun"])
    def test_matches_the_uncaptured_solve(self, gp_flow, sde, solver):
        flow = gp_flow.to("cuda")
        target = torch.linspace(-1, 1, flow.latent_dim, device="cuda")
        mask = torch.zeros(flow.latent_dim, dtype=torch.bool, device="cuda")
        mask[::4] = True
        guidance = GuidanceMC(flow, PointWiseCondition(target, mask, sigma=0.1))
        runs = []
        for cuda_graph in (False, True):
            torch.manual_seed(0)
            runs.append(
                IntegratorSampler(flow).sample(
                    16,
                    num_steps=20,
                    guidance=guidance,
                    sde=sde,
                    solver=solver,
                    verbose=False,
                    cuda_graph=cuda_graph,
                )
            )
        assert torch.allclose(runs[0], runs[1], atol=1e-6)

    @needs_cuda
    def test_falls_back_with_a_warning_when_the_step_syncs(self, gp_flow):
        class HostSync(EqualityCondition):
            def residual(self, x):
                return x[:, :, :1] * float(x.abs().max())  # .item()-style host sync

        flow = gp_flow.to("cuda")
        guidance = GuidanceMC(flow, HostSync(sigma=1.0))
        with pytest.warns(UserWarning, match="could not be captured"):
            out = IntegratorSampler(flow).sample(
                4, num_steps=5, guidance=guidance, verbose=False, cuda_graph=True
            )
        assert torch.isfinite(out).all()

    @needs_cuda
    @pytest.mark.parametrize("sde", [False, True])
    @pytest.mark.parametrize("solver", ["euler", "heun"])
    def test_adaptive_matches_the_uncaptured_solve(self, gp_flow, sde, solver):
        flow = gp_flow.to("cuda")
        target = torch.linspace(-1, 1, flow.latent_dim, device="cuda")
        mask = torch.zeros(flow.latent_dim, dtype=torch.bool, device="cuda")
        mask[::4] = True
        guidance = GuidanceMC(flow, PointWiseCondition(target, mask, sigma=0.1))
        runs, stats = [], []
        for cuda_graph in (False, True):
            # GuidanceMC's Monte Carlo noise is cached on the flow; clear it so both
            # runs draw it at the same point of the random stream.
            flow.z = None
            torch.manual_seed(0)
            sampler = IntegratorSampler(flow)
            runs.append(
                sampler.sample(
                    16,
                    guidance=guidance,
                    sde=sde,
                    solver=solver,
                    adaptive=True,
                    rtol=0.05,
                    atol=0.05,
                    verbose=False,
                    cuda_graph=cuda_graph,
                )
            )
            stats.append(sampler.adaptive_stats)
        # Same controller decisions, same states.
        assert stats[0]["accepted"] == stats[1]["accepted"]
        assert stats[0]["rejected"] == stats[1]["rejected"]
        assert torch.allclose(runs[0], runs[1], atol=1e-5)
