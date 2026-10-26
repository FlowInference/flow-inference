"""Tests for the innovation-space baselines: pCN, elliptical slice, HMC and tempered SMC.

The correctness checks use the one case with a closed-form answer: a GP prior observed
at a few points with Gaussian noise. Every sampler targets the latent posterior, whose
pushforward is the GP posterior, so the data-space mean and variance of the draws must
match the conjugate formulas; SMC's evidence estimate must match the Gaussian marginal
likelihood.
"""

import math
from itertools import pairwise

import pytest
import torch

from flow_inference.conditions import PointWiseCondition
from flow_inference.mcmc import (
    MCMCResult,
    SMCResult,
    elliptical_slice,
    hmc,
    pcn,
    tempered_smc,
)

N = 8
NOISE = 0.3
SAMPLERS = {
    "pcn": lambda flow, cond, **kw: pcn(flow, cond, **kw),
    "ess": lambda flow, cond, **kw: elliptical_slice(flow, cond, **kw),
    "hmc": lambda flow, cond, **kw: hmc(flow, cond, **kw),
}


@pytest.fixture
def observed(gp_flow_correlated):
    """(flow, condition, posterior mean, posterior variance, log evidence) for 3 noisy
    point observations."""
    flow = gp_flow_correlated
    mask = torch.zeros(N, dtype=torch.bool)
    mask[[1, 4, 6]] = True
    target = torch.zeros(N)
    target[mask] = torch.tensor([1.0, -0.5, 0.8])
    condition = PointWiseCondition(target, mask, sigma=NOISE)

    K = flow.L[0] @ flow.L[0].T
    K_xo = K[:, mask]
    gain = K_xo @ torch.linalg.inv(
        K[mask][:, mask] + NOISE**2 * torch.eye(int(mask.sum()))
    )
    mean = gain @ target[mask]
    var = (K - gain @ K_xo.T).diagonal()

    # condition.logp is the unnormalised -|r|^2 / (2 sigma^2), so its evidence is the
    # Gaussian marginal likelihood times (2 pi sigma^2)^(m / 2).
    m = int(mask.sum())
    marginal = torch.distributions.MultivariateNormal(
        torch.zeros(m), K[mask][:, mask] + NOISE**2 * torch.eye(m)
    )
    log_evidence = marginal.log_prob(target[mask]).item() + 0.5 * m * math.log(
        2 * math.pi * NOISE**2
    )
    return flow, condition, mean, var, log_evidence


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_matches_conjugate_gp_posterior(observed, name):
    torch.manual_seed(0)
    flow, condition, mean, var, _ = observed
    # The final state of 2000 independent chains: ~independent posterior draws.
    result = SAMPLERS[name](
        flow, condition, num_chains=2000, num_iters=150, burn_in=100, thin=50
    )
    draws = result.samples.reshape(-1, N)
    assert torch.allclose(draws.mean(0), mean, atol=0.08)
    assert torch.allclose(draws.var(0), var, rtol=0.2, atol=0.01)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_result_shapes(increment_flow, name):
    # A grid-backed flow: samples come back in the display shape, like IntegratorSampler.
    flow = increment_flow
    condition = PointWiseCondition(
        torch.zeros(9), torch.ones(9, dtype=torch.bool), sigma=1.0
    )
    result = SAMPLERS[name](
        flow, condition, num_chains=5, num_iters=6, burn_in=2, thin=2
    )
    assert isinstance(result, MCMCResult)
    assert result.samples.shape == (2, 5, 1, 3, 3)
    assert result.latents.shape == (2, 5, 1, flow.latent_dim)
    assert result.log_likelihood.shape == (6, 5)
    assert torch.isfinite(result.log_likelihood).all()
    # The stored latents are the stored samples, before the map.
    mapped = flow.to_display(flow._to_data_space(result.latents.flatten(0, 1)))
    assert torch.allclose(mapped.reshape(result.samples.shape), result.samples)


def test_pcn_counts_one_evaluation_per_chain_per_iteration(observed):
    flow, condition, *_ = observed
    result = pcn(flow, condition, num_chains=7, num_iters=10)
    assert result.num_likelihood_evals == 7 * (10 + 1)  # + the initial state


def test_ess_costs_at_least_pcn(observed):
    flow, condition, *_ = observed
    result = elliptical_slice(flow, condition, num_chains=7, num_iters=10)
    assert result.num_likelihood_evals >= 7 * (10 + 1)
    assert result.stats["mean_shrinks_per_iteration"] >= 0.0


def test_pcn_adapts_only_during_burn_in(observed):
    flow, condition, *_ = observed
    fixed = pcn(
        flow, condition, num_chains=20, num_iters=20, rho=0.3, target_accept=None
    )
    assert fixed.stats["rho"] == pytest.approx(0.3)
    adapted = pcn(flow, condition, num_chains=20, num_iters=40, rho=0.01, burn_in=30)
    assert adapted.stats["rho"] > 0.01  # tiny steps accept too often, so rho grows
    assert 0.0 < adapted.stats["acceptance_rate"] <= 1.0


def test_init_is_used(observed):
    flow, condition, *_ = observed
    init = torch.full((3, 1, N), 0.5)
    # With rho tiny, one pCN step barely moves the chain from its starting point.
    result = pcn(
        flow,
        condition,
        num_chains=3,
        num_iters=1,
        rho=1e-6,
        target_accept=None,
        init=init,
    )
    assert torch.allclose(result.latents[0], init, atol=1e-4)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_rejects_flow_without_gaussian_innovations(neural_flow, name):
    condition = PointWiseCondition(
        torch.zeros(16), torch.ones(16, dtype=torch.bool), sigma=1.0
    )
    with pytest.raises(ValueError, match="N\\(0, I\\)"):
        SAMPLERS[name](neural_flow, condition, num_chains=2, num_iters=2)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_rejects_schedule_with_no_kept_draws(observed, name):
    flow, condition, *_ = observed
    with pytest.raises(ValueError, match="keeps no draws"):
        SAMPLERS[name](flow, condition, num_chains=2, num_iters=5, burn_in=5)


def test_hmc_counts_leapfrog_evaluations(observed):
    flow, condition, *_ = observed
    result = hmc(flow, condition, num_chains=7, num_iters=10, num_leapfrog=4)
    assert result.num_likelihood_evals == 7 * (1 + 10 * 4)  # + the initial state


def test_hmc_adapts_step_size_during_burn_in(observed):
    flow, condition, *_ = observed
    fixed = hmc(
        flow, condition, num_chains=20, num_iters=5, step_size=0.2, target_accept=None
    )
    assert fixed.stats["step_size"] == pytest.approx(0.2)
    adapted = hmc(
        flow, condition, num_chains=50, num_iters=40, step_size=5.0, burn_in=30
    )
    assert (
        adapted.stats["step_size"] < 5.0
    )  # huge steps are rejected, so the step shrinks
    assert 0.0 < adapted.stats["acceptance_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Tempered SMC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["pcn", "ess"])
def test_smc_matches_conjugate_gp_posterior(observed, kernel):
    torch.manual_seed(0)
    flow, condition, mean, var, _ = observed
    result = tempered_smc(
        flow, condition, num_particles=4000, kernel=kernel, num_moves=10
    )
    draws = result.samples.reshape(-1, N)
    assert torch.allclose(draws.mean(0), mean, atol=0.08)
    assert torch.allclose(draws.var(0), var, rtol=0.2, atol=0.01)


@pytest.mark.parametrize("kernel", ["pcn", "ess"])
def test_smc_estimates_the_evidence(observed, kernel):
    torch.manual_seed(0)
    flow, condition, *_, log_evidence = observed
    result = tempered_smc(flow, condition, num_particles=4000, kernel=kernel)
    assert result.log_evidence == pytest.approx(log_evidence, abs=0.1)


def test_smc_schedule_and_result(observed):
    flow, condition, *_ = observed
    result = tempered_smc(flow, condition, num_particles=50, num_moves=2)
    assert isinstance(result, SMCResult)
    assert result.betas[0] == 0.0 and result.betas[-1] == 1.0
    assert all(b0 < b1 for b0, b1 in pairwise(result.betas))
    assert result.stats["num_stages"] == len(result.betas) - 1
    assert result.samples.shape == (50, 1, N)
    assert result.latents.shape == (50, 1, N)
    # One evaluation for the initial particles, then one per particle per pCN move.
    assert result.num_likelihood_evals == 50 * (1 + 2 * result.stats["num_stages"])


def test_smc_with_a_flat_likelihood_returns_the_prior_in_one_stage(gp_flow_correlated):
    condition = PointWiseCondition(
        torch.zeros(N), torch.zeros(N, dtype=torch.bool), sigma=1.0
    )
    result = tempered_smc(gp_flow_correlated, condition, num_particles=20)
    assert result.betas == [0.0, 1.0]
    assert result.log_evidence == pytest.approx(0.0)


def test_smc_rejects_unknown_kernel(observed):
    flow, condition, *_ = observed
    with pytest.raises(ValueError, match="kernel"):
        tempered_smc(flow, condition, num_particles=10, kernel="gibbs")


def test_smc_rejects_flow_without_gaussian_innovations(neural_flow):
    condition = PointWiseCondition(
        torch.zeros(16), torch.ones(16, dtype=torch.bool), sigma=1.0
    )
    with pytest.raises(ValueError, match="N\\(0, I\\)"):
        tempered_smc(neural_flow, condition, num_particles=4)


# ---------------------------------------------------------------------------
# Likelihood-evaluation budgets
# ---------------------------------------------------------------------------


def test_ess_stops_on_budget_and_keeps_the_final_state(observed):
    flow, condition, *_ = observed
    budget = 7 * 50
    result = elliptical_slice(
        flow,
        condition,
        num_chains=7,
        num_iters=10_000,
        thin=10_000,
        max_likelihood_evals=budget,
    )
    assert result.stats["iterations"] < 10_000
    assert result.num_likelihood_evals >= budget
    assert result.samples.shape[:2] == (1, 7)  # just the final state
    assert result.log_likelihood.shape == (result.stats["iterations"], 7)


def test_ess_without_budget_keeps_the_thinned_draws(observed):
    flow, condition, *_ = observed
    result = elliptical_slice(flow, condition, num_chains=3, num_iters=10, thin=3)
    assert result.samples.shape[:2] == (3, 3)  # iterations 3, 6 and 9


def test_smc_spends_its_budget_on_final_moves(observed):
    torch.manual_seed(0)
    flow, condition, mean, var, _ = observed
    budget = 4000 * 100
    result = tempered_smc(
        flow, condition, num_particles=4000, num_moves=2, max_likelihood_evals=budget
    )
    assert result.stats["final_moves"] > 0
    assert budget <= result.num_likelihood_evals < budget + 4000  # pCN costs 1 a move
    draws = result.samples.reshape(-1, N)
    assert torch.allclose(draws.mean(0), mean, atol=0.08)
    assert torch.allclose(draws.var(0), var, rtol=0.2, atol=0.01)
