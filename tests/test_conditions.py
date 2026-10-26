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

N = 16  # spatial points
B = 5  # batch size


# A concrete BinaryCondition for tests: every entry should be positive.
class PositivityCondition(BinaryCondition):
    def smooth_eval_condition(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.fixture
def mask_first4():
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:4] = True
    return mask


@pytest.fixture
def pointwise(mask_first4):
    return PointWiseCondition(torch.zeros(N), mask_first4, sigma=0.1)


# ---------------------------------------------------------------------------
# PointWiseCondition / EqualityCondition
# ---------------------------------------------------------------------------


class TestPointWiseCondition:
    def test_logp_output_shape(self, pointwise):
        assert pointwise.logp(torch.randn(B, 1, N)).shape == (B,)

    def test_residual_output_shape(self, pointwise):
        assert pointwise.residual(torch.randn(B, 1, N)).shape == (B, 1, 4)

    def test_residual_correct_values(self, pointwise):
        x = torch.zeros(1, 1, N)
        x[0, 0, 0] = 3.0
        r = pointwise.residual(x)
        assert float(r[0, 0, 0]) == pytest.approx(3.0)
        assert float(r[0, 0, 1]) == pytest.approx(0.0)

    def test_logp_highest_when_condition_satisfied(self, pointwise):
        x_good = torch.zeros(1, 1, N)
        x_bad = torch.ones(1, 1, N) * 5.0
        assert pointwise.logp(x_good) > pointwise.logp(x_bad)

    def test_logp_symmetric_around_target(self, pointwise):
        x_pos = torch.zeros(1, 1, N)
        x_pos[0, 0, :4] = 1.0
        x_neg = torch.zeros(1, 1, N)
        x_neg[0, 0, :4] = -1.0
        assert torch.isclose(pointwise.logp(x_pos), pointwise.logp(x_neg))

    def test_logp_decreases_as_residual_grows(self, pointwise):
        logps = []
        for offset in [0.0, 1.0, 2.0, 5.0]:
            x = torch.zeros(1, 1, N)
            x[0, 0, :4] = offset
            logps.append(pointwise.logp(x).item())
        assert all(logps[i] > logps[i + 1] for i in range(len(logps) - 1))

    def test_unmasked_points_ignored(self, pointwise):
        x_base = torch.zeros(1, 1, N)
        x_changed = x_base.clone()
        x_changed[0, 0, 8:] = 99.0
        assert torch.isclose(pointwise.logp(x_base), pointwise.logp(x_changed))

    def test_logp_non_positive(self, pointwise):
        # A Gaussian relaxation with no normalising constant is ≤ 0.
        assert (pointwise.logp(torch.randn(B, 1, N)) <= 0.0).all()

    def test_logp_batched_matches_individual(self, pointwise):
        x = torch.randn(3, 1, N)
        batch = pointwise.logp(x)
        for i in range(3):
            assert torch.isclose(batch[i], pointwise.logp(x[i : i + 1])[0])

    def test_tighter_sigma_penalises_more(self):
        target = torch.zeros(N)
        mask = torch.ones(N, dtype=torch.bool)
        tight = PointWiseCondition(target, mask, sigma=1e-4)
        loose = PointWiseCondition(target, mask, sigma=10.0)
        x = torch.ones(1, 1, N) * 0.5
        assert tight.logp(x) < loose.logp(x)

    def test_logp_zero_residual_is_maximal(self):
        mask = torch.ones(N, dtype=torch.bool)
        cond = PointWiseCondition(torch.zeros(N), mask, sigma=0.5)
        # Exact match gives residual 0 ⟹ logp == 0 (the maximum of the relaxation).
        assert torch.allclose(
            cond.logp(torch.zeros(1, 1, N)), torch.zeros(1), atol=1e-6
        )

    def test_gradient_flows_through_logp(self, pointwise):
        x = torch.randn(1, 1, N, requires_grad=True)
        (grad,) = torch.autograd.grad(pointwise.logp(x).sum(), x)
        assert torch.isfinite(grad).all()
        assert (grad[0, 0, :4].abs() > 0).any()  # masked points carry signal
        assert torch.allclose(grad[0, 0, 4:], torch.zeros(N - 4))  # unmasked do not


# ---------------------------------------------------------------------------
# BinaryCondition
# ---------------------------------------------------------------------------


class TestBinaryCondition:
    def test_logp_output_shape(self):
        cond = PositivityCondition(v=0.1)
        assert cond.logp(torch.randn(B, 1, N)).shape == (B,)

    def test_satisfied_scores_higher(self):
        cond = PositivityCondition(v=0.1)
        good, bad = torch.ones(2, 1, N), -torch.ones(2, 1, N)
        assert (cond.logp(good) > cond.logp(bad)).all()

    def test_logp_non_positive(self):
        cond = PositivityCondition(v=0.1)
        assert (cond.logp(torch.randn(B, 1, N)) <= 1e-5).all()

    def test_smaller_v_sharpens_decision(self):
        # Smaller v → steeper relaxation → stronger penalty for a near-boundary value.
        x = -0.05 * torch.ones(1, 1, N)  # slightly violating
        sharp = PositivityCondition(v=0.01)
        soft = PositivityCondition(v=1.0)
        assert sharp.logp(x) < soft.logp(x)

    def test_gradient_finite(self):
        cond = PositivityCondition(v=0.1)
        x = torch.randn(2, 1, N, requires_grad=True)
        (grad,) = torch.autograd.grad(cond.logp(x).sum(), x)
        assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# ProductCondition
# ---------------------------------------------------------------------------


class TestProductCondition:
    @pytest.fixture
    def two_conditions(self):
        target = torch.zeros(N)
        mask_a = torch.zeros(N, dtype=torch.bool)
        mask_a[:4] = True
        mask_b = torch.zeros(N, dtype=torch.bool)
        mask_b[4:8] = True
        return (
            PointWiseCondition(target, mask_a, sigma=0.1),
            PointWiseCondition(target, mask_b, sigma=0.1),
        )

    def test_logp_equals_sum_of_components(self, two_conditions):
        c1, c2 = two_conditions
        prod = ProductCondition([c1, c2])
        x = torch.randn(B, 1, N)
        assert torch.allclose(prod.logp(x), c1.logp(x) + c2.logp(x))

    def test_logp_output_shape(self, two_conditions):
        c1, c2 = two_conditions
        assert ProductCondition([c1, c2]).logp(torch.randn(B, 1, N)).shape == (B,)

    def test_single_condition_passthrough(self, two_conditions):
        c1, _ = two_conditions
        x = torch.randn(B, 1, N)
        assert torch.allclose(ProductCondition([c1]).logp(x), c1.logp(x))

    def test_three_conditions(self, two_conditions):
        c1, c2 = two_conditions
        mask_c = torch.zeros(N, dtype=torch.bool)
        mask_c[8:12] = True
        c3 = PointWiseCondition(torch.zeros(N), mask_c, sigma=0.1)
        prod = ProductCondition([c1, c2, c3])
        x = torch.randn(B, 1, N)
        assert torch.allclose(prod.logp(x), c1.logp(x) + c2.logp(x) + c3.logp(x))

    def test_mixes_binary_and_equality(self, two_conditions):
        c1, _ = two_conditions
        binary = PositivityCondition(v=0.1)
        prod = ProductCondition([c1, binary])
        x = torch.randn(B, 1, N)
        assert torch.allclose(prod.logp(x), c1.logp(x) + binary.logp(x))


# ---------------------------------------------------------------------------
# Abstract base contracts
# ---------------------------------------------------------------------------


class TestAbstractContracts:
    def test_equality_condition_requires_residual(self):
        with pytest.raises(TypeError):
            EqualityCondition(sigma=0.1)  # residual is abstract

    def test_binary_condition_requires_smooth_eval(self):
        with pytest.raises(TypeError):
            BinaryCondition(v=0.1)  # smooth_eval_condition is abstract


# ---------------------------------------------------------------------------
# standardised_residual: the residual-in-units-of-sigma view that derivative-free
# ensemble guidance builds its Kalman gain from. logp alone is not enough there —
# the gain needs the *vector* residual, one component at a time.
# ---------------------------------------------------------------------------


class TestStandardisedResidual:
    def test_base_condition_reports_what_is_missing(self):
        class LogpOnly(BaseCondition):
            def logp(self, x):
                return x.sum(dim=[1, 2])

        with pytest.raises(NotImplementedError, match="standardised_residual"):
            LogpOnly().standardised_residual(torch.randn(B, 1, N))

    def test_equality_residual_is_divided_by_sigma(self, pointwise, mask_first4):
        x = torch.randn(B, 1, N)
        assert torch.allclose(
            pointwise.standardised_residual(x), pointwise.residual(x) / 0.1
        )
        # Zero residual exactly on target, so the implied N(0, sigma^2) model is right.
        assert pointwise.standardised_residual(torch.zeros(B, 1, N)).abs().max() == 0.0

    def test_equality_residual_is_scale_free_in_sigma(self, mask_first4):
        # Standardising is what makes components with different tolerances
        # commensurate: the same physical error reported at half the sigma is twice
        # as many sigmas.
        x = torch.randn(B, 1, N)
        tight = PointWiseCondition(torch.zeros(N), mask_first4, sigma=0.05)
        loose = PointWiseCondition(torch.zeros(N), mask_first4, sigma=0.1)
        assert torch.allclose(
            tight.standardised_residual(x), 2.0 * loose.standardised_residual(x)
        )

    def test_binary_residual_is_a_hinge_on_the_margin(self):
        c = PositivityCondition(v=0.1)
        satisfied = c.standardised_residual(torch.full((1, 1, N), 5.0))
        violated = c.standardised_residual(torch.full((1, 1, N), -5.0))
        assert satisfied.abs().max() < 1e-6  # decays to zero once comfortably positive
        assert violated.min() > 49.0  # grows like -cond / v when violated
        # Monotone decreasing in the margin: more satisfied is never a larger residual.
        xs = torch.linspace(-3, 3, 50).reshape(1, 1, -1)
        r = c.standardised_residual(xs).squeeze()
        assert (r[1:] <= r[:-1] + 1e-12).all()

    def test_binary_residual_vanishes_exactly_where_logp_is_maximal(self):
        c = PositivityCondition(v=0.1)
        xs = torch.linspace(-2, 2, 40).reshape(-1, 1, 1).expand(-1, 1, N).contiguous()
        r = c.standardised_residual(xs).sum(dim=[1, 2])
        # Both are optimised in the same direction, whatever the tail shapes.
        assert torch.all(torch.diff(r) <= 1e-12)
        assert torch.all(torch.diff(c.logp(xs)) >= -1e-12)

    def test_product_concatenates_already_standardised_blocks(
        self, pointwise, mask_first4
    ):
        binary = PositivityCondition(v=0.25)
        prod = ProductCondition([pointwise, binary])
        x = torch.randn(B, 1, N)
        r = prod.standardised_residual(x)
        r_pw = pointwise.standardised_residual(x)
        r_bin = binary.standardised_residual(x)
        assert r.shape == (B, 1, r_pw.shape[-1] + r_bin.shape[-1])
        # Each block arrives already divided by its own sigma, so no full-size sigma
        # tensor is ever materialised alongside the residual in the hot loop.
        assert torch.equal(r, torch.cat([r_pw, r_bin], dim=-1))

    def test_product_raises_when_a_subcondition_has_no_residual(self, pointwise):
        class LogpOnly(BaseCondition):
            def logp(self, x):
                return x.sum(dim=[1, 2])

        prod = ProductCondition([pointwise, LogpOnly()])
        with pytest.raises(NotImplementedError, match="standardised_residual"):
            prod.standardised_residual(torch.randn(B, 1, N))


# ---------------------------------------------------------------------------
# Grid awareness: output_shape auto-wiring + as_grid
# ---------------------------------------------------------------------------


class _GridResidual(EqualityCondition):
    """A grid-aware condition that reshapes via as_grid (no hardcoded H, W)."""

    def residual(self, x):
        u = self.as_grid(x)  # (B, C, H, W) — shape comes from the flow
        return (u[:, :, 1:, :] - u[:, :, :-1, :]).reshape(x.shape[0], 1, -1)


class TestGridAwareConditions:
    def test_as_grid_unbound_raises(self):
        c = _GridResidual(sigma=0.1)
        with pytest.raises(RuntimeError, match="output_shape"):
            c.as_grid(torch.randn(B, 1, N))

    def test_bind_then_as_grid(self):
        c = _GridResidual(sigma=0.1)
        c.bind_output_shape((4, 4))  # 4*4 == N
        u = c.as_grid(torch.randn(B, 1, N))
        assert u.shape == (B, 1, 4, 4)

    def test_bind_keeps_explicit_shape(self):
        c = _GridResidual(sigma=0.1)
        c.output_shape = (2, 8)
        c.bind_output_shape((4, 4))  # must NOT overwrite the explicit (2, 8)
        assert c.output_shape == (2, 8)

    def test_as_grid_rejects_mismatched_width(self):
        c = _GridResidual(sigma=0.1)
        c.bind_output_shape((4, 4))
        with pytest.raises(ValueError, match="does not match output_shape"):
            c.as_grid(torch.randn(B, 1, N + 1))

    def test_bind_propagates_into_product(self):
        a, b = _GridResidual(sigma=0.1), _GridResidual(sigma=0.1)
        prod = ProductCondition([a, b])
        prod.bind_output_shape((4, 4))
        assert a.output_shape == (4, 4) and b.output_shape == (4, 4)

    def test_residual_runs_through_logp(self):
        c = _GridResidual(sigma=0.1)
        c.bind_output_shape((4, 4))
        assert c.logp(torch.randn(B, 1, N)).shape == (B,)

    def test_guidance_autowires_from_flow(self):
        # The flow's output_shape should reach the condition via guidance construction.
        from flow_inference.flows import GaussianProcessFlow
        from flow_inference.guidance import GuidanceMC
        from tests.conftest import IdentityKernel

        grid = torch.linspace(0, 1, N).unsqueeze(-1)  # N == 16 here, so (4, 4) fits
        flow = GaussianProcessFlow(
            IdentityKernel(),
            gpytorch.means.ZeroMean(),
            grid,
            output_shape=(4, 4),
        )
        c = _GridResidual(sigma=0.1)
        assert c.output_shape is None
        GuidanceMC(flow, c, num_mc_samples=1)
        assert c.output_shape == (4, 4)
