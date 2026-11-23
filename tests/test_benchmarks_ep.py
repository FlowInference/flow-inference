"""Tests for the expectation-propagation baseline (``benchmarks.ep``).

EP is exact for Gaussian sites, so with noisy point observations the evidence and
moments must match the conjugate formulas. For a Poisson likelihood it is a close
approximation, checked against brute-force prior Monte Carlo. Site extraction is
checked by reassembling ``condition.logp`` from the sites.
"""

import math

import gpytorch
import pytest
import torch

from benchmarks.ep import EPResult, Sites, ep, linear_sites
from flow_inference.conditions import (
    BaseCondition,
    BinaryCondition,
    EqualityCondition,
    PointWiseCondition,
    ProductCondition,
)
from flow_inference.flows import GaussianProcessFlow

N = 8
F64 = torch.float64


def _flow():
    grid = torch.linspace(0, 1, N).unsqueeze(-1)
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = 0.05**0.5
    return GaussianProcessFlow(kernel, gpytorch.means.ZeroMean(), grid)


def _observations(flow, sigma):
    mask = torch.zeros(N, dtype=torch.bool)
    mask[[1, 4, 6]] = True
    target = torch.zeros(N, dtype=F64)
    target[mask] = torch.tensor([1.0, -0.5, 0.8], dtype=F64)
    condition = PointWiseCondition(target.float(), mask, sigma=sigma)
    L = flow.L[0].to(F64)
    K = L @ L.T
    K_obs = K[mask][:, mask] + sigma**2 * torch.eye(3, dtype=F64)
    gain = K[:, mask] @ torch.linalg.inv(K_obs)
    mean, var = gain @ target[mask], (K - gain @ K[:, mask].T).diagonal()
    log_evidence = torch.distributions.MultivariateNormal(
        torch.zeros(3, dtype=F64), K_obs
    ).log_prob(target[mask]).item() + 1.5 * math.log(2 * math.pi * sigma**2)
    return condition, mean, var, log_evidence


class Increasing(BinaryCondition):
    """f(x_{j+1}) - f(x_j) > 0 at every adjacent pair: affine in the field."""

    def smooth_eval_condition(self, x):
        return x[:, :, 1:] - x[:, :, :-1]


class Cubic(EqualityCondition):
    """A residual that is not affine in the field."""

    def residual(self, x):
        return x[:, :, :2] ** 3


@pytest.mark.parametrize("sigma", [0.3, 0.02])
def test_exact_for_gaussian_observations(sigma):
    torch.manual_seed(0)
    flow = _flow()
    condition, mean, var, log_evidence = _observations(flow, sigma)
    result = ep(flow, condition, num_samples=20_000)
    assert isinstance(result, EPResult) and result.converged
    assert result.log_evidence == pytest.approx(log_evidence, abs=1e-3)
    draws = result.samples.reshape(-1, N).to(F64)
    assert torch.allclose(draws.mean(0), mean, atol=0.02)
    assert torch.allclose(draws.var(0), var, rtol=0.05, atol=1e-3)


def test_poisson_sites_match_brute_force_evidence():
    flow = _flow()
    idx = torch.tensor([1, 4, 6])
    y = torch.tensor([3.0, 0.0, 2.0], dtype=F64)
    sites = Sites(
        A=flow.L[0].to(F64)[idx],  # s_i = f(x_i) = (L xi)_i
        c=torch.zeros(3, dtype=F64),
        log_site=lambda s: y[:, None] * s - s.exp() - torch.lgamma(y[:, None] + 1),
    )
    torch.manual_seed(0)
    f = flow._to_data_space(torch.randn(1_000_000, 1, N))[:, 0, idx].to(F64)
    ll = (y * f - f.exp() - torch.lgamma(y + 1)).sum(-1)
    brute = (torch.logsumexp(ll, 0) - math.log(len(ll))).item()
    result = ep(flow, sites=sites)
    assert result.converged
    assert result.log_evidence == pytest.approx(brute, abs=0.02)


def test_sites_reassemble_the_condition():
    # sum_i log t_i(a_i . xi + c_i) must equal condition.logp(T(xi)), weights included.
    flow = _flow()
    observations, *_ = _observations(flow, 0.3)
    condition = ProductCondition(
        [Increasing(v=0.1, weights=torch.linspace(0.5, 1.5, N - 1)), observations]
    )
    sites = linear_sites(flow, condition)
    assert sites.A.shape == ((N - 1) + 3, N)  # 7 slopes, 3 observed points
    xi = torch.randn(5, 1, N, dtype=F64)
    s = sites.A @ xi.reshape(5, N).T + sites.c[:, None]  # (m, 5)
    from_sites = sites.log_site(s).sum(0)
    direct = condition.logp(flow._to_data_space(xi.float())).to(F64)
    assert torch.allclose(from_sites, direct, rtol=1e-4, atol=1e-3)


def test_monotonic_condition_runs_and_mostly_satisfies_it():
    torch.manual_seed(0)
    flow = _flow()
    result = ep(flow, Increasing(v=1e-3), num_samples=4000)
    assert result.converged and math.isfinite(result.log_evidence)
    f = result.samples.reshape(-1, N)
    increasing = (f[:, 1:] > f[:, :-1]).all(1).float().mean().item()
    assert increasing > 0.3  # a Gaussian fit to a truncated posterior: not all of it


def test_rejects_non_affine_and_unsupported_conditions():
    flow = _flow()
    with pytest.raises(ValueError, match="not affine"):
        linear_sites(flow, Cubic(sigma=0.1))

    class Opaque(BaseCondition):
        def logp(self, x):
            return x.sum((1, 2))

    with pytest.raises(TypeError, match="build Sites"):
        linear_sites(flow, Opaque())


def test_needs_exactly_one_of_condition_and_sites():
    flow = _flow()
    condition, *_ = _observations(flow, 0.3)
    with pytest.raises(ValueError, match="exactly one"):
        ep(flow)
    sites = linear_sites(flow, condition)
    with pytest.raises(ValueError, match="exactly one"):
        ep(flow, condition, sites=sites)
