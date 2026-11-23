"""Benchmark guided-flow sampling against innovation-space baselines.

Compares LatentFlow (``IntegratorSampler`` with ``GuidanceMC``) and the library's
noise-aware Pi-GDM guidance (``GuidancePiGDM``) with classical methods on the
same latent target: preconditioned Crank–Nicolson and its gradient version pCN-MALA,
elliptical slice sampling, Hamiltonian Monte Carlo, NUTS, likelihood-tempered SMC
(with pCN or elliptical-slice moves), and expectation propagation. The MCMC and SMC
methods get the same number of likelihood evaluations per draw as LatentFlow spends
(999 with the default 1,000 steps and one Monte Carlo sample per step); Pi-GDM
guidance and EP set their own cost.

:func:`compare` works for any flow with an ``N(0, I)`` innovation prior and any
condition. The script's default problem is a 1D GP observed with Gaussian noise,
whose exact posterior and evidence are known, so each method's error is reported
too.

Run from the repository root:

    python -m benchmarks.innovation_space --num-samples 500
    python -m benchmarks.innovation_space --methods NUTS EP   # a subset
"""

import argparse
import contextlib
import math
import time

import gpytorch
import torch

from benchmarks.ep import ep
from benchmarks.mcmc import (
    elliptical_slice,
    hmc,
    nuts,
    pcn,
    pcn_mala,
    tempered_smc,
)
from flow_inference.conditions import PointWiseCondition
from flow_inference.flow_utils import float64_copy
from flow_inference.flows import GaussianProcessFlow
from flow_inference.guidance import GuidanceMC, GuidancePiGDM
from flow_inference.sampler import IntegratorSampler


def _sync(device):
    """Wait for queued CUDA work, so wall-clock timings are accurate."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@contextlib.contextmanager
def counting_evaluations(condition):
    """Count the draws ``condition.logp`` scores inside the block.

    Yields:
        A dict whose ``"evals"`` entry holds the running count.
    """
    counter = {"evals": 0}
    original = condition.logp

    def logp(x):
        counter["evals"] += x.shape[0]
        return original(x)

    condition.logp = logp  # an instance attribute shadows the class's method
    try:
        yield counter
    finally:
        del condition.logp


def run_latentflow(flow, condition, num_samples, num_steps, num_mc_samples):
    """Draw independent guided samples ``(num_samples, C, *output_shape)``.

    Returns:
        A run dict with the ``samples``, wall-clock ``seconds`` and likelihood
        ``evals`` spent.
    """
    guidance = GuidanceMC(flow, condition, num_mc_samples=num_mc_samples)
    guidance = guidance.to(flow.device)
    sampler = IntegratorSampler(flow)
    _sync(flow.device)
    start = time.perf_counter()
    with counting_evaluations(condition) as counter:
        # No CUDA graph: a replayed step would skip the Python counter, and the
        # count sets every other method's budget.
        samples = sampler.sample(
            num_samples,
            guidance=guidance,
            num_steps=num_steps,
            verbose=False,
            sde=True,
            cuda_graph=False,
        )
    _sync(flow.device)
    return {
        "samples": samples.detach(),
        "seconds": time.perf_counter() - start,
        "evals": counter["evals"],
    }


def run_pigdm(flow, condition, num_samples, num_steps, num_iterations=1, lsmr_iters=20):
    """Draw independent samples with :class:`~flow_inference.guidance.GuidancePiGDM`.

    It chooses its own cost: ``num_steps`` ODE steps (it needs far fewer than the
    single-point guidances), each linearising the condition's residual
    ``num_iterations`` times and taking ``2 * lsmr_iters`` Jacobian products per
    linearisation.

    Returns:
        A run dict with the ``samples``, wall-clock ``seconds``, ``evals`` (residual
        evaluations: one per linearisation and draw, the Jacobian products not
        counted) and ``stats``.
    """
    guidance = GuidancePiGDM(
        flow, condition, num_iterations=num_iterations, lsmr_iters=lsmr_iters
    ).to(flow.device)
    counter = {"evals": 0}
    original = condition.residual_and_sigma

    def residual_and_sigma(x):
        counter["evals"] += x.shape[0]
        return original(x)

    condition.residual_and_sigma = residual_and_sigma
    try:
        _sync(flow.device)
        start = time.perf_counter()
        samples = IntegratorSampler(flow).sample(
            num_samples,
            guidance=guidance,
            num_steps=num_steps,
            verbose=False,
            cuda_graph=False,  # a replayed step would skip the Python counter
        )
        _sync(flow.device)
        seconds = time.perf_counter() - start
    finally:
        del condition.residual_and_sigma
    return {
        "samples": samples.detach(),
        "seconds": seconds,
        "evals": counter["evals"],
        "stats": {
            "steps": num_steps,
            "iterations": num_iterations,
            "lsmr_iters": lsmr_iters,
        },
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
    """Pool ``num_samples`` draws from ``num_chains`` chains of ``num_iters`` steps.

    The first ``burn_in`` iterations of each chain are discarded and the rest thinned
    evenly. A sampler that may stop early on a ``max_likelihood_evals`` budget
    (elliptical slice) should get ``burn_in=0`` and a ``discard`` fraction, which
    drops the burn-in afterwards.

    Returns:
        A run dict with the ``samples``, ``seconds``, ``evals`` and sampler ``stats``.
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
    """Tempered SMC with ``num_samples`` particles on a ``budget`` of evaluations."""
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


def run_nuts(flow, condition, num_samples, num_chains, budget, burn_in):
    """NUTS on a budget of likelihood evaluations.

    Each chain runs ``burn_in`` adaptation iterations, then iterations until its share
    of ``budget`` is spent; the draws are pooled and thinned evenly.
    """
    _sync(flow.device)
    result = nuts(
        flow,
        condition,
        num_chains=num_chains,
        num_iters=budget,  # an upper bound: every iteration costs at least one
        burn_in=burn_in,
        max_likelihood_evals=budget,
    )
    _sync(flow.device)
    return {
        "samples": _draws(result.samples, num_samples),
        "seconds": result.wall_time,
        "evals": result.num_likelihood_evals,
        "stats": result.stats,
    }


def run_ep(flow, condition, num_samples):
    """EP, or a ``"skipped"`` entry when the condition's sites are not affine.

    EP's cost is counted in 1D site evaluations, so it is not on the same budget as
    the samplers and is reported with zero likelihood evaluations.
    """
    try:
        result = ep(flow, condition, num_samples)
    except (TypeError, ValueError) as error:
        return {"skipped": str(error)}
    return {
        "samples": result.samples,
        "seconds": result.wall_time,
        "evals": 0,  # deterministic; see EPResult.num_likelihood_evals
        "log_evidence": result.log_evidence,
        "stats": {"sweeps": result.num_sweeps, **result.stats},
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
    nuts_burn_in=200,
    reference=None,
    methods=None,
    pigdm_steps=50,
):
    """Run every method on LatentFlow's budget and print a comparison table.

    LatentFlow runs first, and the ``condition.logp`` evaluations it spends in total
    become every other method's budget. MCMC splits it across ``num_chains`` long
    chains and pools ``num_samples`` thinned draws from their second halves (one fresh
    chain per draw would pay a burn-in for every draw):

    * pCN: one evaluation per iteration, the first half burn-in (``rho`` adapts
      there);
    * pCN-MALA: one evaluation (with gradient) per iteration, likewise;
    * HMC: ``num_leapfrog`` evaluations per iteration, likewise;
    * NUTS: ``nuts_burn_in`` adaptation iterations per chain, then iterations until
      the budget is spent (trajectory lengths vary). Chains run one after another, so
      its wall time is not comparable, but its evaluations are;
    * elliptical slice: iterations until the budget is spent (their cost varies);
    * tempered SMC: ``smc_moves`` pCN moves (or ``smc_ess_moves`` elliptical-slice
      moves, which cost several evaluations each) per stage up to ``beta = 1``, then
      further moves at ``beta = 1`` until the budget is spent. Tempering always
      finishes, so too many moves per stage overrun the budget; the table shows the
      evaluations actually spent;
    * Pi-GDM guidance: sets its own cost, ``pigdm_steps`` ODE steps (it needs far
      fewer than LatentFlow); the table counts its residual evaluations, not the
      Jacobian products each one is followed by;
    * EP: sets its own cost. It is a Gaussian approximation rather than a sampler,
      and applies only to conditions with affine sites (it is skipped otherwise).

    Every method returns ``num_samples`` draws; SMC uses ``num_samples`` particles.

    Args:
        flow: a flow with an ``N(0, I)`` innovation prior.
        condition: the condition to sample under.
        num_samples: draws per method.
        num_steps, num_mc_samples: LatentFlow's integration steps and Monte Carlo
            samples per step, which together set the budget.
        num_chains: chains per MCMC method.
        num_leapfrog: HMC leapfrog steps per iteration.
        smc_moves, smc_ess_moves: SMC moves per stage with the pCN and
            elliptical-slice kernels.
        nuts_burn_in: NUTS adaptation iterations per chain.
        reference: optional ``(mean, std, log_evidence)`` of the exact posterior in
            data space (``log_evidence`` may be None). When given, the table also
            shows each method's mean absolute error in the mean and standard
            deviation, and the evidence error of the methods that estimate it.
        methods: a subset of :data:`METHODS` to run; all by default.
        pigdm_steps: integration steps for Pi-GDM guidance.

    Returns:
        The runs, ``{method: run}``.
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
        methods=METHODS if methods is None else methods,
        nuts_burn_in=nuts_burn_in,
        pigdm_steps=pigdm_steps,
    )
    print(
        f"budget: {per_draw} likelihood evaluations per draw, {num_samples} draws; "
        f"MCMC pools them from {num_chains} chains of {per_chain} evaluations each\n"
    )
    print_table(runs, num_samples, reference)
    return runs


METHODS = (
    "LatentFlow",
    "Pi-GDM guidance",
    "pCN",
    "pCN-MALA",
    "elliptical slice",
    "HMC",
    "NUTS",
    "SMC (pCN moves)",
    "SMC (ESS moves)",
    "EP",
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
    nuts_burn_in=200,
    pigdm_steps=50,
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
        "pCN-MALA": lambda: run_mcmc(pcn_mala, *mcmc, pcn_iters, pcn_iters // 2),
        "NUTS": lambda: run_nuts(
            flow, condition, num_samples, num_chains, budget, nuts_burn_in
        ),
        "EP": lambda: run_ep(flow, condition, num_samples),
        "Pi-GDM guidance": lambda: run_pigdm(flow, condition, num_samples, pigdm_steps),
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
        if "skipped" in run:
            print(f"{name:18s} skipped: {run['skipped']}")
            continue
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

    The prior is a Matérn-5/2 GP on ``num_points`` grid points in ``[0, 1]``; the
    observations are a prior draw plus noise, at random grid points.

    Args:
        num_points: grid size.
        num_obs: number of observed grid points.
        noise: observation noise standard deviation.
        lengthscale: the kernel's lengthscale.
        device: where to build the flow and condition.
        seed: seed for the observation locations and values.
        return_cov: also return the exact posterior covariance.

    Returns:
        ``(flow, condition, (mean, std, log_evidence))``, the last being the exact
        posterior moments in data space and the log evidence of ``condition.logp``;
        with ``return_cov``, the exact posterior covariance follows as a fourth
        element.
    """
    generator = torch.Generator().manual_seed(seed)
    grid = torch.linspace(0, 1, num_points, device=device)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
    kernel = kernel.to(device)
    kernel.base_kernel.lengthscale = lengthscale
    flow = GaussianProcessFlow(
        kernel, gpytorch.means.ZeroMean(), grid.unsqueeze(-1), device=device
    )

    # The exact reference is kernel-matrix work, so it runs in float64 even though
    # the samplers use the flow's float32.
    g64 = grid.unsqueeze(-1).double()
    K = float64_copy(kernel)(g64, g64).to_dense().detach().cpu()
    eye = torch.eye(num_points, dtype=K.dtype)
    mask = torch.zeros(num_points, dtype=torch.bool)
    mask[torch.randperm(num_points, generator=generator)[:num_obs]] = True
    z = torch.randn(num_points, generator=generator, dtype=K.dtype)
    truth = torch.linalg.cholesky(K + 1e-8 * eye) @ z
    noise_draw = torch.randn(num_points, generator=generator, dtype=K.dtype)
    target = truth + noise * noise_draw
    condition = PointWiseCondition(
        target.to(device, flow.dtype), mask.to(device), sigma=noise
    )

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
        "--nuts-burn-in", type=int, default=200, help="NUTS adaptation iterations"
    )
    parser.add_argument(
        "--pigdm-steps", type=int, default=50, help="Pi-GDM guidance steps"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        help="methods to run (LatentFlow always runs: it sets the budget)",
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
        nuts_burn_in=args.nuts_burn_in,
        pigdm_steps=args.pigdm_steps,
        reference=reference,
        methods=args.methods,
    )
