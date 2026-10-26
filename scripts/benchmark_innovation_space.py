"""Benchmark guided-flow sampling against innovation-space baselines.

Compares LatentFlow (``IntegratorSampler`` + ``GuidanceMC``) with preconditioned
Crank–Nicolson, elliptical slice sampling, Hamiltonian Monte Carlo and likelihood-
tempered SMC (with pCN or elliptical-slice moves), all on the same latent target.
Every method gets the same number of likelihood evaluations per draw as LatentFlow
spends (999 with the default 1,000 steps and one Monte Carlo sample per step).
``compare`` works for any flow with an N(0, I) innovation prior and any condition. The
script's default problem is a 1D GP observed with Gaussian noise, whose exact posterior
and evidence are known, so each method's error is reported too.

Run from the repository root:

    python scripts/benchmark_innovation_space.py --num-samples 500
"""

import argparse
import contextlib
import math
import time

import gpytorch
import torch

from flow_inference.conditions import PointWiseCondition
from flow_inference.flows import GaussianProcessFlow
from flow_inference.guidance import GuidanceMC
from flow_inference.mcmc import elliptical_slice, hmc, pcn, tempered_smc
from flow_inference.sampler import IntegratorSampler


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@contextlib.contextmanager
def counting_evaluations(condition):
    """Count the draws ``condition.logp`` scores inside the block, in ``counter["evals"]``."""
    counter = {"evals": 0}
    original = condition.logp

    def logp(x):
        counter["evals"] += x.shape[0]
        return original(x)

    condition.logp = logp  # an instance attribute shadows the method
    try:
        yield counter
    finally:
        del condition.logp


def run_latentflow(flow, condition, num_samples, num_steps, num_mc_samples):
    """Independent guided samples ``(num_samples, C, *output_shape)``, with their cost."""
    guidance = GuidanceMC(flow, condition, num_mc_samples=num_mc_samples)
    guidance = guidance.to(flow.device)
    sampler = IntegratorSampler(flow)
    _sync(flow.device)
    start = time.perf_counter()
    with counting_evaluations(condition) as counter:
        samples = sampler.sample(
            num_samples, guidance=guidance, num_steps=num_steps, verbose=False, sde=True
        )
    _sync(flow.device)
    return {
        "samples": samples.detach(),
        "seconds": time.perf_counter() - start,
        "evals": counter["evals"],
    }


def _draws(samples, num_samples, discard=0.0):
    """Pool ``num_samples`` draws from ``(num_kept, chains, ...)`` chain output.

    Drops the first ``discard`` fraction of each chain as burn-in, then takes evenly
    spaced draws from the rest, the same number from every chain.
    """
    num_kept, num_chains = samples.shape[:2]
    samples = samples[int(discard * num_kept) :]
    per_chain = math.ceil(num_samples / num_chains)
    idx = torch.linspace(0, samples.shape[0] - 1, per_chain).round().long()
    return samples[idx].flatten(0, 1)[:num_samples]


def run_mcmc(
    sampler, flow, condition, num_samples, num_chains, num_iters, burn_in, **kwargs
):
    """``num_samples`` draws pooled from ``num_chains`` chains of ``num_iters`` iterations.

    The first ``burn_in`` iterations of each chain are discarded and the rest thinned
    evenly. ``sampler`` may stop early on a ``max_likelihood_evals`` budget (elliptical
    slice), in which case pass ``burn_in=0`` and ``discard`` the burn-in afterwards.
    """
    discard = kwargs.pop("discard", 0.0)
    per_chain = math.ceil(num_samples / num_chains)
    thin = max(1, (num_iters - burn_in) // per_chain) if not discard else 1
    _sync(flow.device)
    result = sampler(
        flow,
        condition,
        num_chains=num_chains,
        num_iters=num_iters,
        burn_in=burn_in,
        thin=thin,
        **kwargs,
    )
    _sync(flow.device)
    return {
        "samples": _draws(result.samples, num_samples, discard),
        "seconds": result.wall_time,
        "evals": result.num_likelihood_evals,
        "stats": result.stats,
    }


def run_smc(flow, condition, num_samples, kernel, num_moves, budget):
    _sync(flow.device)
    result = tempered_smc(
        flow,
        condition,
        num_samples,
        kernel=kernel,
        num_moves=num_moves,
        max_likelihood_evals=budget,
    )
    _sync(flow.device)
    return {
        "samples": result.samples,
        "seconds": result.wall_time,
        "evals": result.num_likelihood_evals,
        "log_evidence": result.log_evidence,
        "stats": result.stats,
    }


def compare(
    flow,
    condition,
    num_samples=500,
    num_steps=1_000,
    num_mc_samples=1,
    num_chains=10,
    num_leapfrog=10,
    smc_moves=20,
    smc_ess_moves=3,
    reference=None,
):
    """Run every method on LatentFlow's budget; returns ``{method: run}``.

    LatentFlow runs first, and the ``condition.logp`` evaluations it spends in total
    become every other method's budget. MCMC splits it across ``num_chains`` long chains
    and pools ``num_samples`` thinned draws from their second halves (one fresh chain
    per draw would pay a burn-in for every draw):

    * pCN: one evaluation per iteration, the first half burn-in (``rho`` adapts there);
    * HMC: ``num_leapfrog`` evaluations per iteration, likewise;
    * elliptical slice: iterations until the budget is spent (their cost varies);
    * tempered SMC: ``smc_moves`` pCN moves (or ``smc_ess_moves`` elliptical-slice moves,
      which cost several evaluations each) per stage up to ``beta = 1``, then further
      moves at ``beta = 1`` until the budget is spent. Tempering always finishes, so
      too many moves per stage overrun the budget; the table shows evaluations spent.

    All methods return ``num_samples`` draws; SMC uses ``num_samples`` particles.
    ``reference`` is an optional ``(mean, std, log_evidence)`` of the exact posterior in
    data space (``log_evidence`` may be None); when given, each method's mean absolute
    error in the mean and standard deviation, and SMC's evidence error, are printed.
    """
    runs, per_draw, per_chain = run_all(
        flow,
        condition,
        num_samples,
        num_steps,
        num_mc_samples,
        num_chains,
        num_leapfrog,
        smc_moves,
        smc_ess_moves,
    )
    print(
        f"budget: {per_draw} likelihood evaluations per draw, {num_samples} draws; "
        f"MCMC pools them from {num_chains} chains of {per_chain} evaluations each\n"
    )
    print_table(runs, num_samples, reference)
    return runs


METHODS = (
    "LatentFlow",
    "pCN",
    "elliptical slice",
    "HMC",
    "SMC (pCN moves)",
    "SMC (ESS moves)",
)


def run_all(
    flow,
    condition,
    num_samples,
    num_steps,
    num_mc_samples,
    num_chains,
    num_leapfrog,
    smc_moves,
    smc_ess_moves,
    methods=METHODS,
):
    """Run ``methods`` on LatentFlow's budget (see :func:`compare`), without printing.

    LatentFlow always runs, since it sets the budget. Returns ``(runs, per_draw,
    per_chain)``: the runs by method name and the budget per draw and per MCMC chain.
    """
    runs = {
        "LatentFlow": run_latentflow(
            flow, condition, num_samples, num_steps, num_mc_samples
        )
    }
    per_draw = runs["LatentFlow"]["evals"] // num_samples
    budget = per_draw * num_samples
    per_chain = budget // num_chains
    pcn_iters = per_chain - 1  # the initial state costs one evaluation
    hmc_iters = (per_chain - 1) // num_leapfrog
    mcmc = (flow, condition, num_samples, num_chains)
    runners = {
        "pCN": lambda: run_mcmc(pcn, *mcmc, pcn_iters, pcn_iters // 2),
        "elliptical slice": lambda: run_mcmc(
            elliptical_slice,
            *mcmc,
            per_chain,  # an upper bound: every iteration costs at least one evaluation
            0,
            max_likelihood_evals=budget,
            discard=0.5,  # no adaptation, so burn-in is dropped afterwards
        ),
        "HMC": lambda: run_mcmc(
            hmc, *mcmc, hmc_iters, hmc_iters // 2, num_leapfrog=num_leapfrog
        ),
        "SMC (pCN moves)": lambda: run_smc(
            flow, condition, num_samples, "pcn", smc_moves, budget
        ),
        "SMC (ESS moves)": lambda: run_smc(
            flow, condition, num_samples, "ess", smc_ess_moves, budget
        ),
    }
    for name in methods:
        if name != "LatentFlow":
            runs[name] = runners[name]()
    return runs, per_draw, per_chain


def print_table(runs, num_samples, reference=None):
    """One line per run: wall time, evaluations per draw, errors and sampler stats."""
    header = f"{'method':18s} {'seconds':>8s} {'evals/draw':>11s}"
    if reference is not None:
        header += f" {'|mean err|':>11s} {'|std err|':>10s} {'|logZ err|':>11s}"
    print(header)
    for name, run in runs.items():
        line = f"{name:18s} {run['seconds']:8.2f} {run['evals'] / num_samples:11.1f}"
        if reference is not None:
            mean, std, log_evidence = reference
            flat = run["samples"].reshape(run["samples"].shape[0], -1)
            line += f" {(flat.mean(0) - mean).abs().mean().item():11.4f}"
            line += f" {(flat.std(0) - std).abs().mean().item():10.4f}"
            if "log_evidence" in run and log_evidence is not None:
                line += f" {abs(run['log_evidence'] - log_evidence):11.4f}"
            else:
                line += f" {'':11s}"
        if run.get("stats"):
            line += "   " + ", ".join(f"{k}={v:.3g}" for k, v in run["stats"].items())
        print(line)


def conjugate_gp_problem(
    num_points=100,
    num_obs=10,
    noise=0.01,
    lengthscale=0.1,
    device="cpu",
    seed=0,
    return_cov=False,
):
    """A 1D GP observed at ``num_obs`` points with Gaussian noise.

    Returns ``(flow, condition, (mean, std, log_evidence))``, the last being the exact
    posterior moments in data space and the log evidence of ``condition.logp``. With
    ``return_cov``, the exact posterior covariance follows as a fourth element.
    """
    generator = torch.Generator().manual_seed(seed)
    grid = torch.linspace(0, 1, num_points, dtype=torch.float64, device=device)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
    kernel = kernel.double().to(device)
    kernel.base_kernel.lengthscale = lengthscale
    flow = GaussianProcessFlow(
        kernel, gpytorch.means.ZeroMean(), grid.unsqueeze(-1), device=device
    )

    K = (flow.L[0] @ flow.L[0].T).cpu()
    eye = torch.eye(num_points, dtype=K.dtype)
    mask = torch.zeros(num_points, dtype=torch.bool)
    mask[torch.randperm(num_points, generator=generator)[:num_obs]] = True
    z = torch.randn(num_points, generator=generator, dtype=K.dtype)
    truth = torch.linalg.cholesky(K + 1e-8 * eye) @ z
    noise_draw = torch.randn(num_points, generator=generator, dtype=K.dtype)
    target = truth + noise * noise_draw
    condition = PointWiseCondition(target.to(device), mask.to(device), sigma=noise)

    K_obs = K[mask][:, mask] + noise**2 * torch.eye(num_obs, dtype=K.dtype)
    gain = K[:, mask] @ torch.linalg.inv(K_obs)
    mean = gain @ target[mask]
    cov = K - gain @ K[:, mask].T
    std = cov.diagonal().clamp(min=0).sqrt()
    # condition.logp is the unnormalised -|r|^2 / (2 noise^2): the Gaussian marginal
    # likelihood times (2 pi noise^2)^(num_obs / 2).
    marginal = torch.distributions.MultivariateNormal(
        torch.zeros(num_obs, dtype=K.dtype), K_obs
    )
    log_evidence = (
        marginal.log_prob(target[mask]).item()
        + 0.5 * num_obs * torch.log(torch.tensor(2 * torch.pi * noise**2)).item()
    )
    reference = (mean.to(device), std.to(device), log_evidence)
    if return_cov:
        return flow, condition, reference, cov.to(device)
    return flow, condition, reference


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-samples", type=int, default=500, help="draws per method")
    parser.add_argument("--num-steps", type=int, default=1_000, help="LatentFlow steps")
    parser.add_argument(
        "--num-mc-samples", type=int, default=1, help="LatentFlow MC samples"
    )
    parser.add_argument(
        "--num-chains", type=int, default=10, help="chains per MCMC method"
    )
    parser.add_argument(
        "--num-leapfrog", type=int, default=10, help="HMC leapfrog steps"
    )
    parser.add_argument(
        "--smc-moves", type=int, default=20, help="SMC pCN moves per stage"
    )
    parser.add_argument(
        "--smc-ess-moves", type=int, default=3, help="SMC ESS moves per stage"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    flow, condition, reference = conjugate_gp_problem(device=args.device)
    compare(
        flow,
        condition,
        args.num_samples,
        args.num_steps,
        args.num_mc_samples,
        args.num_chains,
        args.num_leapfrog,
        args.smc_moves,
        args.smc_ess_moves,
        reference=reference,
    )
