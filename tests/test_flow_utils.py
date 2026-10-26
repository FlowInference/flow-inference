import pytest
import torch

from flow_inference.flow_utils import (
    build_grid_in_box,
    clip_vector_field,
    compute_snr_schedule,
)

N = 8


# ---------------------------------------------------------------------------
# build_grid_in_box
# ---------------------------------------------------------------------------


class TestBuildGridInBox:
    def test_shape(self):
        assert build_grid_in_box(3, 4).shape == (12, 2)

    def test_defaults_to_float64(self):
        # Coordinates feed kernels and finite-difference stencils, so the grid
        # defaults to double regardless of the ambient default dtype.
        assert build_grid_in_box(3, 4).dtype == torch.float64

    def test_dtype_override(self):
        # The float32 opt-out a learned NeuralFlow pipeline needs.
        assert build_grid_in_box(3, 4, dtype=torch.float32).dtype == torch.float32

    def test_inside_unit_box(self):
        g = build_grid_in_box(5, 5)
        assert g.min() >= 0.0 and g.max() < 1.0


# ---------------------------------------------------------------------------
# clip_vector_field
# ---------------------------------------------------------------------------


class TestClipVectorField:
    def test_output_shape_preserved(self):
        dv = torch.randn(3, 1, N)
        assert clip_vector_field(dv, max_norm=1.0).shape == dv.shape

    def test_zero_input_gives_zero_output(self):
        dv = torch.zeros(3, 1, N)
        assert torch.allclose(clip_vector_field(dv, max_norm=1.0), dv, atol=1e-6)

    def test_large_input_clipped_near_max_norm(self):
        max_norm = 2.0
        dv = torch.zeros(1, 1, 4)
        dv[0, 0, 0] = 1e6
        out = clip_vector_field(dv, max_norm=max_norm)
        norm = out.pow(2).sum(dim=-1).sqrt()
        assert (norm <= max_norm + 1e-4).all()

    def test_small_input_passes_through_unchanged(self):
        dv = torch.randn(5, 1, N) * 0.001
        out = clip_vector_field(dv, max_norm=100.0)
        assert torch.allclose(out, dv, rtol=1e-3, atol=1e-6)

    def test_direction_preserved(self):
        dv = torch.randn(2, 1, N) * 50.0
        out = clip_vector_field(dv, max_norm=1.0)
        cos = (dv * out).sum(-1) / (dv.norm(dim=-1) * out.norm(dim=-1) + 1e-8)
        assert (cos > 0.99).all()

    def test_larger_max_norm_gives_larger_output(self):
        dv = torch.ones(1, 1, N) * 10.0
        out_small = clip_vector_field(dv, max_norm=1.0)
        out_large = clip_vector_field(dv, max_norm=5.0)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_output_finite_for_random_input(self):
        dv = torch.randn(4, 2, N) * 100
        assert torch.isfinite(clip_vector_field(dv, max_norm=10.0)).all()

    def test_clips_per_spatial_point_independently(self):
        # Two points in one (B, C, N) row with very different magnitudes; the small
        # one stays small, the large one saturates — clipping is along the last dim.
        dv = torch.zeros(1, 2, N)
        dv[0, 0] = 1e-3  # tiny channel
        dv[0, 1] = 1e3  # huge channel
        out = clip_vector_field(dv, max_norm=1.0)
        assert out[0, 0].abs().max() < 1e-2
        assert out[0, 1].pow(2).sum().sqrt() <= 1.0 + 1e-4

    def test_tanh_saturation_is_smooth(self):
        # tanh squashing is monotone and never exceeds max_norm even at the knee.
        max_norm = 1.0
        norms = torch.linspace(0.1, 5.0, 20)
        out_norms = []
        for n in norms:
            dv = torch.zeros(1, 1, N)
            dv[0, 0, 0] = n
            out_norms.append(clip_vector_field(dv, max_norm).pow(2).sum().sqrt())
        out_norms = torch.stack(out_norms)
        assert (out_norms[1:] >= out_norms[:-1] - 1e-5).all()  # monotone
        assert (out_norms <= max_norm + 1e-4).all()


# ---------------------------------------------------------------------------
# compute_snr_schedule
# ---------------------------------------------------------------------------


class TestComputeSNRSchedule:
    def test_output_shape(self, gp_flow):
        assert compute_snr_schedule(gp_flow.schedule, num_steps=10).shape == (11,)

    def test_output_shape_larger(self, gp_flow):
        assert compute_snr_schedule(gp_flow.schedule, num_steps=100).shape == (101,)

    def test_starts_near_t_max(self, gp_flow):
        ts = compute_snr_schedule(
            gp_flow.schedule, num_steps=20, t_min=1e-10, t_max=1.0
        )
        assert ts[0].item() == pytest.approx(1.0, abs=1e-2)

    def test_ends_near_t_min(self, gp_flow):
        ts = compute_snr_schedule(
            gp_flow.schedule, num_steps=20, t_min=1e-10, t_max=1.0
        )
        assert ts[-1].item() < 0.01

    def test_monotonically_decreasing(self, gp_flow):
        ts = compute_snr_schedule(gp_flow.schedule, num_steps=20)
        assert (ts[:-1] >= ts[1:]).all()

    def test_all_values_in_range(self, gp_flow):
        ts = compute_snr_schedule(
            gp_flow.schedule, num_steps=20, t_min=1e-10, t_max=1.0
        )
        assert (ts >= 0.0).all() and (ts <= 1.0 + 1e-6).all()

    def test_uniform_log_snr_spacing(self, gp_flow):
        ts = compute_snr_schedule(gp_flow.schedule, num_steps=30)
        alpha_sq = gp_flow.schedule._alpha_2(ts)
        log_snr = torch.log(alpha_sq / (1 - alpha_sq + 1e-8))
        gaps = (log_snr[1:] - log_snr[:-1]).abs()
        interior = gaps[:-5]  # skip the unresolved low-t boundary
        assert interior.numel() > 0
        assert (interior.max() / (interior.min() + 1e-8)) < 1.2

    def test_output_on_device(self, gp_flow):
        ts = compute_snr_schedule(gp_flow.schedule, num_steps=10)
        assert ts.device == gp_flow.device or str(ts.device).startswith("cpu")

    def test_respects_custom_t_max(self, gp_flow):
        ts = compute_snr_schedule(
            gp_flow.schedule, num_steps=20, t_min=1e-10, t_max=0.5
        )
        assert ts.max().item() <= 0.5 + 1e-6

    def test_denser_near_t_min(self, gp_flow):
        # Log-spaced inversion puts more steps at low t (where log-SNR moves fast).
        ts = compute_snr_schedule(gp_flow.schedule, num_steps=40)
        steps = ts[:-1] - ts[1:]  # positive decrements
        first_half, second_half = steps[: len(steps) // 2], steps[len(steps) // 2 :]
        assert first_half.mean() > second_half.mean()
