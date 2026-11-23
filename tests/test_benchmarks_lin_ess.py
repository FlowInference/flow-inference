"""Tests for the reference sampler for linearly constrained Gaussians
(``benchmarks.lin_ess``).

LIN-ESS draws and subset-simulation evidence are checked against closed forms in two
dimensions and against brute-force prior Monte Carlo on a GP.
"""

import math

import gpytorch
import pytest
import torch

from benchmarks.lin_ess import LinESSResult, _lin_ess_step, _subset_simulation, lin_ess
from flow_inference.conditions import (
    BinaryCondition,
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


class Increasing(BinaryCondition):
    """f(x_{j+1}) - f(x_j) > 0 at every adjacent pair: a set of linear constraints."""

    def smooth_eval_condition(self, x):
        return x[:, :, 1:] - x[:, :, :-1]


def _normal_tail(a):
    return 0.5 * math.erfc(a / math.sqrt(2))


@pytest.mark.parametrize(
    ("B", "d", "log_p"),
    [
        ([[1.0, 0.0]], [-1.0], math.log(_normal_tail(1.0))),  # half-plane x1 >= 1
        ([[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0], math.log(0.25)),  # a 90-degree wedge
        ([[1.0, 0.0]], [-5.0], math.log(_normal_tail(5.0))),  # a 3e-7 tail
    ],
    ids=["half-plane", "wedge", "far-tail"],
)
def test_subset_simulation_matches_closed_form_probabilities(B, d, log_p):
    torch.manual_seed(0)
    B, d = torch.tensor(B, dtype=F64), torch.tensor(d, dtype=F64)
    u, estimate, _, dropped = _subset_simulation(B, d, 4000, 0.5, 10, 100)
    assert estimate == pytest.approx(log_p, abs=0.1)
    assert (u @ B.T + d >= 0).all() and dropped == 0


def test_lin_ess_draws_the_truncated_normal():
    # x1 >= 1 in 2D: x1 is a truncated standard normal, x2 is untouched.
    torch.manual_seed(0)
    B, d = torch.tensor([[1.0, 0.0]], dtype=F64), torch.tensor([-1.0], dtype=F64)
    u = torch.tensor([[2.0, 0.0]], dtype=F64).repeat(4000, 1)
    for _ in range(100):
        u, _ = _lin_ess_step(u, B, d)
    mean = math.exp(-0.5) / math.sqrt(2 * math.pi) / _normal_tail(1.0)
    assert u[:, 0].mean().item() == pytest.approx(mean, abs=0.03)
    assert u[:, 1].var().item() == pytest.approx(1.0, abs=0.06)
    assert (u[:, 0] >= 1).all()


@pytest.fixture(scope="module")
def increasing_prior_draws():
    torch.manual_seed(0)
    prior = _flow()._to_data_space(torch.randn(2_000_000, 1, N))[:, 0].to(F64)
    return prior, (prior[:, 1:] >= prior[:, :-1]).all(1)


def test_matches_brute_force_on_a_monotone_gp(increasing_prior_draws):
    prior, increasing = increasing_prior_draws
    torch.manual_seed(0)
    result = lin_ess(_flow(), Increasing(v=1e-6), num_samples=2000)
    assert isinstance(result, LinESSResult)
    assert result.log_evidence == pytest.approx(
        math.log(increasing.double().mean().item()), abs=0.1
    )
    draws = result.samples.reshape(-1, N).to(F64)
    assert (draws[:, 1:] >= draws[:, :-1] - 1e-5).all()
    assert torch.allclose(draws.mean(0), prior[increasing].mean(0), atol=0.05)
    assert torch.allclose(draws.std(0), prior[increasing].std(0), atol=0.04)
    assert result.stats["inequality_sites"] == N - 1


def test_gaussian_observations_are_integrated_exactly(increasing_prior_draws):
    prior, increasing = increasing_prior_draws
    mask = torch.zeros(N, dtype=torch.bool)
    mask[[1, 6]] = True
    target = torch.zeros(N)
    target[mask] = torch.tensor([-0.3, 0.4])
    observations = PointWiseCondition(target, mask, sigma=0.3)
    ll = observations.logp(prior.unsqueeze(1).float()).to(F64)
    brute = (
        torch.logsumexp(torch.where(increasing, ll, -math.inf), 0) - math.log(len(ll))
    ).item()
    torch.manual_seed(0)
    result = lin_ess(
        _flow(), ProductCondition([Increasing(v=1e-6), observations]), 2000
    )
    assert result.stats["gaussian_sites"] == 2
    assert result.log_evidence == pytest.approx(brute, abs=0.1)


def test_only_gaussian_sites_give_the_exact_evidence():
    flow = _flow()
    mask = torch.zeros(N, dtype=torch.bool)
    mask[[2]] = True
    target = torch.zeros(N)
    target[2] = 0.5
    sigma = 0.3
    K22 = (flow.L[0].to(F64) @ flow.L[0].to(F64).T)[2, 2].item()
    exact = -0.25 / (2 * (K22 + sigma**2)) + 0.5 * math.log(sigma**2 / (K22 + sigma**2))
    result = lin_ess(flow, PointWiseCondition(target, mask, sigma=sigma), 500)
    assert result.num_levels == 0
    assert result.log_evidence == pytest.approx(exact, abs=1e-6)
