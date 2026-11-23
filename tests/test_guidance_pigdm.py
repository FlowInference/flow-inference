"""Tests for the noise-aware Pi-GDM guidance (``GuidancePiGDM``).

For point observations of a GP the guidance is exact: its drift is checked against
the dense closed-form score, ``-alpha J^T (v J J^T + Sigma)^-1 r``, and sampling with
it against the conjugate posterior. Nonlinear conditions, non-affine flows, hard
constraints (sigma = 0), errors and CUDA-graph use are covered too.
"""

import gpytorch
import pytest
import torch

from flow_inference.conditions import (
    EqualityCondition,
    PointWiseCondition,
    ProductCondition,
)
from flow_inference.flows import GaussianProcessFlow
from flow_inference.guidance import GuidancePiGDM
from flow_inference.sampler import IntegratorSampler
from tests.conftest import PositivityCondition

N = 8
F64 = torch.float64
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _flow(dtype=F64, device="cpu"):
    kernel = gpytorch.kernels.RBFKernel().to(device)
    kernel.lengthscale = 0.05**0.5
    grid = torch.linspace(0, 1, N).unsqueeze(-1)
    return GaussianProcessFlow(
        kernel, gpytorch.means.ZeroMean(), grid, dtype=dtype, device=device
    )


def _observations(sigma, dtype=F64, device="cpu"):
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[[1, 4, 6]] = True
    target = torch.zeros(N, dtype=dtype, device=device)
    target[mask] = torch.tensor([1.0, -0.5, 0.8], dtype=dtype, device=device)
    return PointWiseCondition(target, mask, sigma=sigma), mask, target


def _dense_drift(flow, mask, target, sigma, z, t):
    """The exact noise-aware drift, by dense linear algebra."""
    a2 = flow.schedule._alpha_2(t).to(F64)
    v, alpha, beta = 1 - a2, a2.sqrt(), flow.schedule._beta(t).to(F64)
    J = flow.L[0].to(F64)[mask]  # residual = L z - target at the observed points
    r = z[:, 0].to(F64) @ (alpha * J).T - target[mask].to(F64)
    S = v * J @ J.T + sigma**2 * torch.eye(J.shape[0], dtype=F64)
    grad = -alpha * torch.linalg.solve(S, r.T).T @ J
    return (-0.5 * beta * grad).unsqueeze(1)


class TestExactForPointObservations:
    @pytest.mark.parametrize("sigma", [0.3, 0.01, 0.0])
    @pytest.mark.parametrize("t", [0.9, 0.3, 0.02])
    def test_drift_matches_the_closed_form(self, sigma, t):
        flow = _flow()
        condition, mask, target = _observations(sigma)
        guidance = GuidancePiGDM(flow, condition, lsmr_iters=20)
        torch.manual_seed(0)
        z = torch.randn(4, 1, N, dtype=F64)
        t = torch.tensor(t, dtype=F64)
        expected = _dense_drift(flow, mask, target, sigma, z, t)
        assert torch.allclose(guidance(z, t), expected, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("sigmas", [(0.3, 0.003), (0.3, 0.0)])
    def test_mixed_tolerances_match_the_closed_form(self, sigmas):
        # Rows are rescaled by precision before the solve; the answer must not change.
        flow = _flow()
        loose, tight = (torch.zeros(N, dtype=torch.bool) for _ in range(2))
        loose[[1, 4]], tight[6] = True, True
        target = torch.linspace(-1, 1, N, dtype=F64)
        condition = ProductCondition(
            [
                PointWiseCondition(target, loose, sigma=sigmas[0]),
                PointWiseCondition(target, tight, sigma=sigmas[1]),
            ]
        )
        guidance = GuidancePiGDM(flow, condition)
        torch.manual_seed(0)
        z, t = torch.randn(4, 1, N, dtype=F64), torch.tensor(0.3, dtype=F64)
        a2 = flow.schedule._alpha_2(t)
        v, alpha, beta = 1 - a2, a2.sqrt(), flow.schedule._beta(t)
        mask = torch.cat([loose.nonzero(), tight.nonzero()]).squeeze(-1)
        J = flow.L[0].to(F64)[mask]
        r = z[:, 0] @ (alpha * J).T - target[mask]
        Sigma = torch.diag(torch.tensor([sigmas[0]] * 2 + [sigmas[1]], dtype=F64) ** 2)
        grad = -alpha * torch.linalg.solve(v * J @ J.T + Sigma, r.T).T @ J
        expected = (-0.5 * beta * grad).unsqueeze(1)
        assert torch.allclose(guidance(z, t), expected, rtol=1e-6, atol=1e-8)

    def test_precision_scaling_resolves_tight_rows_first(self):
        # A loose residual with large Jacobian rows next to a tight one with small rows
        # (a PDE stencil next to boundary values): with few LSMR iterations, the tight
        # rows must come out more accurate with the scaling than without it.
        flow = _flow()

        class Loud(EqualityCondition):
            def residual(self, x):
                return 100.0 * x[..., :6]

        tight_mask = torch.zeros(N, dtype=torch.bool)
        tight_mask[6:] = True
        condition = ProductCondition(
            [
                Loud(sigma=1.0),
                PointWiseCondition(torch.ones(N, dtype=F64), tight_mask, sigma=1e-3),
            ]
        )
        torch.manual_seed(0)
        z, t = torch.randn(4, 1, N, dtype=F64), torch.tensor(0.2, dtype=F64)
        exact = GuidancePiGDM(flow, condition, lsmr_iters=50)(z, t)
        scaled = GuidancePiGDM(flow, condition, lsmr_iters=3)
        unscaled = GuidancePiGDM(flow, condition, lsmr_iters=3)
        unscaled.row_scale.fill_(1.0)
        J_tight = flow.L[0].to(F64)[tight_mask]  # the tight rows' directions

        def tight_error(guidance):
            return ((guidance(z, t) - exact)[:, 0] @ J_tight.T).norm()

        assert tight_error(scaled) < 0.1 * tight_error(unscaled)

    def test_extra_iterations_change_nothing_for_a_linear_condition(self):
        flow = _flow()
        condition, *_ = _observations(0.1)
        z, t = torch.randn(4, 1, N, dtype=F64), torch.tensor(0.5, dtype=F64)
        one = GuidancePiGDM(flow, condition, num_iterations=1)(z, t)
        three = GuidancePiGDM(flow, condition, num_iterations=3)(z, t)
        assert torch.allclose(one, three, rtol=1e-6, atol=1e-8)

    def test_sampling_recovers_the_conjugate_posterior(self):
        flow = _flow(dtype=torch.float32)
        condition, mask, target = _observations(0.05, dtype=torch.float32)
        L = flow.L[0].to(F64)
        K = L @ L.T
        K_obs = K[mask][:, mask] + 0.05**2 * torch.eye(3, dtype=F64)
        gain = K[:, mask] @ torch.linalg.inv(K_obs)
        mean = gain @ target[mask].to(F64)
        var = (K - gain @ K[:, mask].T).diagonal()
        torch.manual_seed(0)
        draws = (
            IntegratorSampler(flow)
            .sample(
                4000,
                num_steps=50,
                guidance=GuidancePiGDM(flow, condition),
                verbose=False,
            )
            .reshape(-1, N)
            .to(F64)
        )
        assert torch.allclose(draws.mean(0), mean, atol=0.04)
        assert torch.allclose(draws.var(0), var, rtol=0.15, atol=2e-3)

    def test_hard_constraint_is_met_by_the_samples(self):
        flow = _flow(dtype=torch.float32)
        condition, mask, target = _observations(0.0, dtype=torch.float32)
        torch.manual_seed(0)
        draws = IntegratorSampler(flow).sample(
            64, num_steps=50, guidance=GuidancePiGDM(flow, condition), verbose=False
        )
        assert torch.isfinite(draws).all()
        assert (draws[:, 0, mask] - target[mask]).abs().max() < 1e-3


class Cubic(EqualityCondition):
    """A nonlinear residual: f^3 should equal 0.5 at every point."""

    def residual(self, x):
        return x.pow(3) - 0.5


class TestNonlinearAndGeneral:
    @pytest.mark.parametrize("num_iterations", [1, 3])
    def test_reduces_a_nonlinear_residual(self, num_iterations):
        flow = _flow(dtype=torch.float32)
        condition = Cubic(sigma=1e-3)
        torch.manual_seed(0)
        prior = IntegratorSampler(flow).sample(64, num_steps=30, verbose=False)
        guidance = GuidancePiGDM(flow, condition, num_iterations=num_iterations)
        torch.manual_seed(0)
        guided = IntegratorSampler(flow).sample(
            64, num_steps=30, guidance=guidance, verbose=False
        )
        assert torch.isfinite(guided).all()
        before = condition.residual(prior.reshape(64, 1, -1)).abs().mean()
        after = condition.residual(guided.reshape(64, 1, -1)).abs().mean()
        assert after < 0.05 * before

    def test_non_affine_flow(self, increment_flow):
        width = increment_flow._to_data_space(
            torch.zeros(1, 1, increment_flow.latent_dim)
        ).shape[-1]
        mask = torch.zeros(width, dtype=torch.bool)
        mask[::2] = True
        condition = PointWiseCondition(torch.zeros(width), mask, sigma=0.1)
        guidance = GuidancePiGDM(increment_flow, condition)
        out = IntegratorSampler(increment_flow).sample(
            8, num_steps=10, guidance=guidance, verbose=False
        )
        assert torch.isfinite(out).all()

    def test_residual_and_sigma_agree_with_logp(self):
        torch.manual_seed(0)
        observations, *_ = _observations(0.2, dtype=torch.float32)
        condition = ProductCondition([observations, Cubic(sigma=0.5, weights=0.25)])
        x = torch.randn(5, 1, N)
        r, sigma = condition.residual_and_sigma(x)
        assert r.shape == (5, 1, 3 + N) and sigma.shape == (1, 3 + N)
        assert torch.allclose(
            -0.5 * (r / sigma).pow(2).sum((1, 2)), condition.logp(x), rtol=1e-5
        )

    def test_zero_weights_drop_components(self):
        torch.manual_seed(0)
        weights = torch.ones(N)
        weights[::3] = 0.0
        condition = Cubic(sigma=0.5, weights=weights)
        x = torch.randn(5, 1, N)
        r, sigma = condition.residual_and_sigma(x)
        assert torch.isfinite(sigma).all() and (r[..., ::3] == 0).all()
        assert torch.allclose(
            -0.5 * (r / sigma).pow(2).sum((1, 2)), condition.logp(x), rtol=1e-5
        )
        guidance = GuidancePiGDM(_flow(dtype=torch.float32), condition)
        assert torch.isfinite(guidance(x, torch.tensor(0.5))).all()

    def test_rejects_unsupported_conditions_and_flows(self, neural_flow):
        flow = _flow()
        with pytest.raises(TypeError, match="Gaussian residuals"):
            GuidancePiGDM(flow, PositivityCondition(v=0.1))
        with pytest.raises(ValueError, match="num_iterations"):
            GuidancePiGDM(flow, _observations(0.1)[0], num_iterations=0)
        with pytest.raises(ValueError, match="N\\(0, I\\)"):
            GuidancePiGDM(neural_flow, Cubic(sigma=1.0))

    def test_keeps_the_flow_dtype(self):
        flow = _flow(dtype=torch.float32)
        guidance = GuidancePiGDM(flow, _observations(0.1, dtype=torch.float32)[0])
        out = guidance(torch.randn(3, 1, N), torch.tensor(0.5))
        assert out.dtype == torch.float32 and torch.isfinite(out).all()


@needs_cuda
class TestOnTheGPU:
    # PyTorch flags its sync-debug mode as a prototype; the next test checks capture
    # end to end anyway.
    @pytest.mark.filterwarnings("ignore:Synchronization debug mode")
    def test_never_synchronises_with_the_host(self):
        flow = _flow(dtype=torch.float32, device="cuda")
        condition = ProductCondition(
            [_observations(0.1, torch.float32, "cuda")[0], Cubic(sigma=1e-2)]
        ).to("cuda")
        guidance = GuidancePiGDM(flow, condition, num_iterations=2).to("cuda")
        z, t = torch.randn(8, 1, N, device="cuda"), torch.tensor(0.4, device="cuda")
        guidance(z, t)  # first call may allocate
        torch.cuda.set_sync_debug_mode("error")
        try:
            guidance(z, t)
        finally:
            torch.cuda.set_sync_debug_mode(0)

    def test_matches_under_the_samplers_cuda_graph(self):
        flow = _flow(dtype=torch.float32, device="cuda")
        condition = _observations(0.05, torch.float32, "cuda")[0].to("cuda")
        guidance = GuidancePiGDM(flow, condition).to("cuda")
        runs = []
        for cuda_graph in (False, True):
            torch.manual_seed(0)
            runs.append(
                IntegratorSampler(flow).sample(
                    32,
                    num_steps=20,
                    guidance=guidance,
                    verbose=False,
                    cuda_graph=cuda_graph,
                )
            )
        assert torch.allclose(runs[0], runs[1], atol=1e-5)
