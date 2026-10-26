import pytest
import torch

from flow_inference.schedules import LinearSchedule

# ---------------------------------------------------------------------------
# LinearSchedule
# ---------------------------------------------------------------------------


class TestLinearScheduleAlpha2:
    @pytest.fixture
    def schedule(self):
        return LinearSchedule(beta_0=torch.tensor(0.0001), beta_1=torch.tensor(10.0))

    def test_alpha2_at_zero_is_one(self, schedule):
        assert torch.isclose(
            schedule._alpha_2(torch.tensor(0.0)), torch.tensor(1.0), atol=1e-6
        )

    def test_alpha2_strictly_less_than_one_for_positive_t(self, schedule):
        assert schedule._alpha_2(torch.tensor(0.01)) < 1.0

    def test_alpha2_monotonically_decreasing(self, schedule):
        vals = schedule._alpha_2(torch.linspace(0.01, 0.99, 50))
        assert (vals[1:] < vals[:-1]).all()

    def test_alpha2_in_unit_interval(self, schedule):
        vals = schedule._alpha_2(torch.linspace(0.0, 1.0, 100))
        assert (vals >= 0.0).all() and (vals <= 1.0 + 1e-6).all()

    def test_alpha2_accepts_tensor_batch(self, schedule):
        assert schedule._alpha_2(torch.linspace(0.0, 1.0, 20)).shape == (20,)

    def test_alpha2_accepts_python_float(self, schedule):
        # The type hint allows float | Tensor; a bare float must work.
        out = schedule._alpha_2(0.5)
        assert torch.is_tensor(out)

    def test_alpha2_near_zero_at_t1_for_strong_schedule(self, schedule):
        # With beta_1=10 the signal is almost gone by t=1 (the schedule logs this).
        assert schedule._alpha_2(torch.tensor(1.0)) < 1e-2

    def test_alpha2_matches_closed_form(self, schedule):
        t = torch.tensor(0.3)
        gamma = schedule.beta_0 * t + 0.5 * (schedule.beta_1 - schedule.beta_0) * t * t
        assert torch.isclose(schedule._alpha_2(t), torch.exp(-gamma))


class TestLinearScheduleBeta:
    @pytest.fixture
    def schedule(self):
        return LinearSchedule(beta_0=torch.tensor(0.0001), beta_1=torch.tensor(10.0))

    def test_beta_at_zero_equals_beta0(self, schedule):
        assert torch.isclose(schedule._beta(torch.tensor(0.0)), schedule.beta_0)

    def test_beta_at_one_equals_beta1(self, schedule):
        assert torch.isclose(schedule._beta(torch.tensor(1.0)), schedule.beta_1)

    def test_beta_linear_midpoint(self, schedule):
        mid = schedule._beta(torch.tensor(0.5))
        assert torch.isclose(mid, (schedule.beta_0 + schedule.beta_1) / 2)

    def test_beta_non_negative(self, schedule):
        assert (schedule._beta(torch.linspace(0.0, 1.0, 100)) >= 0.0).all()

    def test_beta_monotonically_increasing(self, schedule):
        vals = schedule._beta(torch.linspace(0.0, 1.0, 50))
        assert (vals[1:] >= vals[:-1]).all()

    def test_beta_is_affine_in_t(self, schedule):
        # β is linear: the second difference is zero.
        t = torch.linspace(0.0, 1.0, 11)
        b = schedule._beta(t)
        second_diff = b[2:] - 2 * b[1:-1] + b[:-2]
        assert torch.allclose(second_diff, torch.zeros_like(second_diff), atol=1e-5)


class TestLinearScheduleConsistency:
    @pytest.fixture
    def schedule(self):
        return LinearSchedule(beta_0=torch.tensor(0.0001), beta_1=torch.tensor(10.0))

    def test_dlog_alpha2_dt_equals_minus_beta(self, schedule):
        # α²(t) = exp(-∫β) ⟹ d/dt log α²(t) = -β(t). Check via autograd.
        t = torch.tensor(0.4, requires_grad=True)
        log_alpha2 = torch.log(schedule._alpha_2(t))
        (grad,) = torch.autograd.grad(log_alpha2, t)
        assert torch.isclose(grad, -schedule._beta(t), atol=1e-4)

    def test_smaller_beta1_retains_more_signal(self):
        weak = LinearSchedule(beta_0=torch.tensor(0.0001), beta_1=torch.tensor(1.0))
        strong = LinearSchedule(beta_0=torch.tensor(0.0001), beta_1=torch.tensor(10.0))
        t = torch.tensor(0.5)
        assert weak._alpha_2(t) > strong._alpha_2(t)

    def test_double_dtype_preserved(self):
        sched = LinearSchedule(
            beta_0=torch.tensor(0.1, dtype=torch.float64),
            beta_1=torch.tensor(5.0, dtype=torch.float64),
        )
        out = sched._alpha_2(torch.tensor(0.5, dtype=torch.float64))
        assert out.dtype == torch.float64
