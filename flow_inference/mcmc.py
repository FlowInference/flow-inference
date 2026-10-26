"""Baseline samplers in innovation space, for benchmarking against guided flows.

Every flow with ``has_standard_normal_marginals`` writes a sample as ``f = T(xi)`` with
innovations ``xi ~ N(0, I)`` (``T`` is ``flow._to_data_space``). Conditioning on a
condition with likelihood ``L`` then targets

    ``rho(xi)  ∝  L(T(xi)) N(xi; 0, I)``,

whose pushforward through ``T`` is the conditioned process. The samplers here run
directly on that target:

* :func:`pcn` — preconditioned Crank–Nicolson Metropolis–Hastings (Cotter et al., 2013);
* :func:`elliptical_slice` — elliptical slice sampling (Murray, Adams & MacKay, 2010);
* :func:`hmc` — Hamiltonian Monte Carlo with an identity mass matrix (Neal, 2011);
* :func:`tempered_smc` — likelihood-tempered sequential Monte Carlo (Del Moral, Doucet
  & Jasra, 2006) with pCN or elliptical-slice moves, which also estimates the evidence.

pCN and elliptical slice sampling are prior-reversible for ``N(0, I)``, so they need only
the pulled-back log-likelihood ``condition.logp(T(xi))``; HMC also differentiates it,
through the generator, as :class:`~flow_inference.guidance.GuidanceMC` does.

All of them run a batch of chains or particles in parallel, and their ``samples`` have
the same per-draw shape as ``IntegratorSampler.sample`` output, so the plots and
diagnostics used for the guided samplers apply unchanged.
"""

import math
import time
from dataclasses import dataclass, field

import torch

from flow_inference.conditions import BaseCondition
from flow_inference.flows import BaseFlow


@dataclass
class MCMCResult:
    """Output of an innovation-space MCMC run.

    Attributes:
        samples: kept draws mapped to data space, ``(num_kept, num_chains, C, *output_shape)``.
        latents: the kept innovations, ``(num_kept, num_chains, C, latent_dim)``.
        log_likelihood: pulled-back log-likelihood of every chain after every
            iteration, ``(num_iters, num_chains)`` — a mixing diagnostic.
        num_likelihood_evals: total ``condition.logp`` evaluations, summed over chains;
            the cost measure that is comparable across samplers. For HMC every
            evaluation also includes a gradient.
        wall_time: seconds spent sampling.
        stats: sampler-specific diagnostics (acceptance rate and tuned step size for
            pCN and HMC; mean bracket shrinks per iteration for elliptical slice).
    """

    samples: torch.Tensor
    latents: torch.Tensor
    log_likelihood: torch.Tensor
    num_likelihood_evals: int
    wall_time: float
    stats: dict = field(default_factory=dict)


@dataclass
class SMCResult:
    """Output of a tempered SMC run.

    Attributes:
        samples: the final, equally weighted particles in data space,
            ``(num_particles, C, *output_shape)``.
        latents: the final particles' innovations, ``(num_particles, C, latent_dim)``.
        log_evidence: estimate of ``log E_{xi ~ N(0, I)}[L(T(xi))]``, the log normalising
            constant of the target (unbiased on the evidence scale).
        betas: the adaptive tempering schedule, from 0 to 1.
        num_likelihood_evals: total ``condition.logp`` evaluations over all particles.
        wall_time: seconds spent sampling.
        stats: number of stages, and the move kernel's diagnostics.
    """

    samples: torch.Tensor
    latents: torch.Tensor
    log_evidence: float
    betas: list
    num_likelihood_evals: int
    wall_time: float
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _check_flow(flow: BaseFlow, condition: BaseCondition) -> None:
    if not getattr(flow, "has_standard_normal_marginals", False):
        raise ValueError(
            f"{type(flow).__name__} has no N(0, I) innovation prior; the innovation-space "
            "samplers need one (use a GaussianFlow subclass)."
        )
    # As the guidance classes do: grid-aware conditions reshape with the flow's shape.
    condition.bind_output_shape(getattr(flow, "output_shape", None))


def _scrub(logp: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(logp, nan=-math.inf, posinf=math.inf, neginf=-math.inf)


def _log_likelihood(flow, condition, xi: torch.Tensor) -> torch.Tensor:
    """``condition.logp(T(xi))`` for ``xi`` of shape ``(B, C, latent_dim)``; NaN -> -inf."""
    return _scrub(condition.logp(flow._to_data_space(xi)))


def _log_likelihood_and_grad(flow, condition, xi: torch.Tensor):
    """The pulled-back log-likelihood ``(B,)`` and its gradient in ``xi``."""
    with torch.enable_grad():
        x = xi.detach().requires_grad_(True)
        logp = condition.logp(flow._to_data_space(x))
        (grad,) = torch.autograd.grad(logp.sum(), x)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return _scrub(logp.detach()), grad


def _tempered(beta, ll):
    """``beta * ll`` with ``0 * -inf = 0`` rather than NaN."""
    return torch.zeros_like(ll) if beta == 0 else beta * ll


def _check_schedule(num_iters, burn_in, thin):
    if thin < 1 or num_iters - burn_in < thin:
        raise ValueError(
            f"num_iters={num_iters}, burn_in={burn_in}, thin={thin} keeps no draws; "
            "need num_iters - burn_in >= thin >= 1"
        )


def _initial_state(flow, num_chains, init):
    if init is not None:
        return init.to(device=flow.device, dtype=flow.dtype).clone()
    return torch.randn(
        num_chains, flow.channels, flow.latent_dim, device=flow.device, dtype=flow.dtype
    )


def _to_samples(flow, latents):
    """Innovations ``(..., C, latent_dim)`` -> data-space draws ``(..., C, *output_shape)``."""
    lead = latents.shape[:-2]
    display = flow.to_display(
        flow._to_data_space(latents.reshape(-1, *latents.shape[-2:]))
    )
    return display.reshape(*lead, *display.shape[1:])


def _result(flow, kept, trace, evals, start, stats):
    latents = torch.stack(kept)  # (num_kept, chains, C, latent_dim)
    return MCMCResult(
        _to_samples(flow, latents),
        latents,
        torch.stack(trace),
        evals,
        time.perf_counter() - start,
        stats,
    )


def _kept(iteration, burn_in, thin):
    return iteration >= burn_in and (iteration - burn_in) % thin == thin - 1


def _adapt(log_step, rate, target, iteration):
    """Robbins–Monro on a log step size, towards a target acceptance rate."""
    return log_step + (rate - target) / math.sqrt(iteration + 1)


# ---------------------------------------------------------------------------
# Single-step kernels, each targeting  L(T(xi))^beta N(xi; 0, I)
# ---------------------------------------------------------------------------


def _pcn_step(flow, condition, xi, ll, rho, beta=1.0):
    """One pCN move per chain. Returns ``(xi, ll, accepted)``; costs one evaluation each."""
    proposal = math.sqrt(1.0 - rho**2) * xi + rho * torch.randn_like(xi)
    ll_proposal = _log_likelihood(flow, condition, proposal)
    log_ratio = _tempered(beta, ll_proposal) - _tempered(beta, ll)
    accept = torch.rand_like(ll).log() < log_ratio  # NaN (-inf to -inf) rejects
    xi = torch.where(accept[:, None, None], proposal, xi)
    ll = torch.where(accept, ll_proposal, ll)
    return xi, ll, accept


def _ess_step(flow, condition, xi, ll, beta=1.0, max_shrinks=100):
    """One elliptical-slice move per chain. Returns ``(xi, ll, evaluations, shrinks)``."""
    xi, ll = xi.clone(), ll.clone()
    nu = torch.randn_like(xi)
    log_level = _tempered(beta, ll) + torch.rand_like(ll).log()
    theta = 2 * math.pi * torch.rand_like(ll)
    lower, upper = theta - 2 * math.pi, theta.clone()
    searching = torch.ones_like(ll, dtype=torch.bool)
    evals = shrinks = 0

    for _ in range(max_shrinks + 1):
        idx = searching.nonzero().squeeze(1)
        t = theta[idx][:, None, None]
        proposal = xi[idx] * torch.cos(t) + nu[idx] * torch.sin(t)
        ll_proposal = _log_likelihood(flow, condition, proposal)
        evals += idx.numel()

        accept = _tempered(beta, ll_proposal) > log_level[idx]
        xi[idx[accept]] = proposal[accept]
        ll[idx[accept]] = ll_proposal[accept]
        searching[idx[accept]] = False
        if not searching.any():
            break

        # Shrink the bracket towards theta = 0 and redraw, for the rejected chains.
        rejected = idx[~accept]
        shrinks += rejected.numel()
        below = theta[rejected] < 0
        lower[rejected] = torch.where(below, theta[rejected], lower[rejected])
        upper[rejected] = torch.where(below, upper[rejected], theta[rejected])
        width = upper[rejected] - lower[rejected]
        theta[rejected] = lower[rejected] + width * torch.rand_like(width)
    return xi, ll, evals, shrinks


def _hmc_step(flow, condition, xi, ll, grad, step_size, num_leapfrog):
    """One HMC move per chain with potential ``|xi|^2 / 2 - log L(T(xi))``.

    Returns ``(xi, ll, grad, accept_prob)``; costs ``num_leapfrog`` evaluations each.
    The step size is jittered by +-10% per chain to avoid periodic trajectories.
    """
    eps = step_size * (0.9 + 0.2 * torch.rand_like(ll))[:, None, None]
    momentum = torch.randn_like(xi)
    h0 = 0.5 * xi.pow(2).sum((1, 2)) - ll + 0.5 * momentum.pow(2).sum((1, 2))

    x, x_ll, x_grad = xi, ll, grad
    p = momentum - 0.5 * eps * (x - x_grad)  # grad U = xi - grad log L
    for step in range(num_leapfrog):
        x = x + eps * p
        x_ll, x_grad = _log_likelihood_and_grad(flow, condition, x)
        if step < num_leapfrog - 1:
            p = p - eps * (x - x_grad)
    p = p - 0.5 * eps * (x - x_grad)

    h1 = 0.5 * x.pow(2).sum((1, 2)) - x_ll + 0.5 * p.pow(2).sum((1, 2))
    log_accept = torch.nan_to_num(h0 - h1, nan=-math.inf).clamp(max=0.0)
    accept = torch.rand_like(ll).log() < log_accept
    keep = accept[:, None, None]
    return (
        torch.where(keep, x, xi),
        torch.where(accept, x_ll, ll),
        torch.where(keep, x_grad, grad),
        log_accept.exp(),
    )


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------


@torch.no_grad()
def pcn(
    flow: BaseFlow,
    condition: BaseCondition,
    num_chains: int,
    num_iters: int,
    rho: float = 0.1,
    burn_in: int = 0,
    thin: int = 1,
    target_accept: float | None = 0.25,
    init: torch.Tensor | None = None,
) -> MCMCResult:
    """Preconditioned Crank–Nicolson Metropolis–Hastings on the innovations.

    Proposes ``xi' = sqrt(1 - rho^2) xi + rho eps`` with ``eps ~ N(0, I)``, which leaves
    the ``N(0, I)`` prior invariant, so the acceptance probability is just the likelihood
    ratio ``min(1, L(T(xi')) / L(T(xi)))``.

    Args:
        num_chains: independent chains run in parallel.
        num_iters: iterations per chain, including burn-in.
        rho: step size in (0, 1]; ``rho = 1`` is an independence sampler from the prior.
        burn_in: initial iterations discarded. ``rho`` is adapted only during burn-in,
            so the kept iterations form a valid (non-adaptive) chain.
        thin: keep every ``thin``-th iteration after burn-in.
        target_accept: acceptance rate to adapt ``rho`` towards during burn-in, or None
            to hold ``rho`` fixed.
        init: optional starting innovations ``(num_chains, C, latent_dim)``; prior draws
            by default.
    """
    _check_flow(flow, condition)
    _check_schedule(num_iters, burn_in, thin)
    start = time.perf_counter()
    xi = _initial_state(flow, num_chains, init)
    ll = _log_likelihood(flow, condition, xi)
    evals = num_chains
    log_rho = math.log(rho)
    kept, trace, accepted_after_burn_in = [], [], 0

    for iteration in range(num_iters):
        xi, ll, accept = _pcn_step(flow, condition, xi, ll, math.exp(log_rho))
        evals += num_chains
        if iteration < burn_in:
            if target_accept is not None:  # capped at rho = 1
                rate = accept.double().mean().item()
                log_rho = min(0.0, _adapt(log_rho, rate, target_accept, iteration))
        else:
            accepted_after_burn_in += accept.sum().item()
        trace.append(ll.clone())
        if _kept(iteration, burn_in, thin):
            kept.append(xi.clone())

    stats = {
        "acceptance_rate": accepted_after_burn_in
        / ((num_iters - burn_in) * num_chains),
        "rho": math.exp(log_rho),
    }
    return _result(flow, kept, trace, evals, start, stats)


@torch.no_grad()
def elliptical_slice(
    flow: BaseFlow,
    condition: BaseCondition,
    num_chains: int,
    num_iters: int,
    burn_in: int = 0,
    thin: int = 1,
    max_shrinks: int = 100,
    max_likelihood_evals: int | None = None,
    init: torch.Tensor | None = None,
) -> MCMCResult:
    """Elliptical slice sampling on the innovations.

    Each iteration draws an auxiliary ``nu ~ N(0, I)`` and a slice level
    ``log y = log L(T(xi)) + log u``, then searches the ellipse
    ``xi cos(theta) + nu sin(theta)`` for a point above the level, shrinking the angle
    bracket towards ``theta = 0`` (the current state) after every rejection. Every
    iteration moves; there is no step size to tune. Chains still searching are
    evaluated together, so one iteration costs as many batched likelihood calls as its
    slowest chain needs.

    Args:
        num_chains, num_iters, burn_in, thin, init: as for :func:`pcn`.
        max_shrinks: give up on a chain's move after this many shrinks and keep its
            current state (only reached when the likelihood is degenerate, e.g. -inf).
        max_likelihood_evals: optional budget of evaluations over all chains, including
            the initial state. An iteration costs a variable number of evaluations, so
            this is how to run on a fixed budget: no new iteration starts once it is
            spent (the last one may overrun it), and the chains' final state is kept as
            the last draw.
    """
    _check_flow(flow, condition)
    _check_schedule(num_iters, burn_in, thin)
    start = time.perf_counter()
    xi = _initial_state(flow, num_chains, init)
    ll = _log_likelihood(flow, condition, xi)
    evals = num_chains
    kept, trace, shrinks_total, iterations = [], [], 0, 0
    last_kept = -1

    for iteration in range(num_iters):
        if max_likelihood_evals is not None and evals >= max_likelihood_evals:
            break
        xi, ll, step_evals, shrinks = _ess_step(
            flow, condition, xi, ll, max_shrinks=max_shrinks
        )
        evals += step_evals
        shrinks_total += shrinks
        iterations += 1
        trace.append(ll.clone())
        if _kept(iteration, burn_in, thin):
            kept.append(xi.clone())
            last_kept = iteration

    stopped_by_budget = iterations < num_iters
    if stopped_by_budget and last_kept != iterations - 1:  # keep the final state
        kept.append(xi.clone())
    stats = {
        "iterations": iterations,
        "mean_shrinks_per_iteration": shrinks_total / max(iterations * num_chains, 1),
    }
    return _result(flow, kept, trace, evals, start, stats)


@torch.no_grad()
def hmc(
    flow: BaseFlow,
    condition: BaseCondition,
    num_chains: int,
    num_iters: int,
    step_size: float = 0.1,
    num_leapfrog: int = 10,
    burn_in: int = 0,
    thin: int = 1,
    target_accept: float | None = 0.65,
    init: torch.Tensor | None = None,
) -> MCMCResult:
    """Hamiltonian Monte Carlo on the innovations, with an identity mass matrix.

    The potential is ``U(xi) = |xi|^2 / 2 - log L(T(xi))``; its gradient is taken by
    autograd through the generator. The ``N(0, I)`` reference makes the identity mass
    matrix the natural choice: under a weak likelihood the target is already
    isotropic with unit scale.

    Args:
        num_chains, num_iters, burn_in, thin, init: as for :func:`pcn`.
        step_size: initial leapfrog step size, jittered by +-10% per chain and move.
        num_leapfrog: leapfrog steps per move; each costs one likelihood-and-gradient
            evaluation per chain.
        target_accept: mean acceptance probability to adapt ``step_size`` towards during
            burn-in, or None to hold it fixed.
    """
    _check_flow(flow, condition)
    _check_schedule(num_iters, burn_in, thin)
    start = time.perf_counter()
    xi = _initial_state(flow, num_chains, init)
    ll, grad = _log_likelihood_and_grad(flow, condition, xi)
    evals = num_chains
    log_step = math.log(step_size)
    kept, trace, accept_prob_after_burn_in = [], [], 0.0

    for iteration in range(num_iters):
        xi, ll, grad, accept_prob = _hmc_step(
            flow, condition, xi, ll, grad, math.exp(log_step), num_leapfrog
        )
        evals += num_chains * num_leapfrog
        if iteration < burn_in:
            if target_accept is not None:
                rate = accept_prob.mean().item()
                log_step = _adapt(log_step, rate, target_accept, iteration)
        else:
            accept_prob_after_burn_in += accept_prob.sum().item()
        trace.append(ll.clone())
        if _kept(iteration, burn_in, thin):
            kept.append(xi.clone())

    stats = {
        "acceptance_rate": accept_prob_after_burn_in
        / ((num_iters - burn_in) * num_chains),
        "step_size": math.exp(log_step),
        "num_leapfrog": num_leapfrog,
    }
    return _result(flow, kept, trace, evals, start, stats)


# ---------------------------------------------------------------------------
# Tempered SMC
# ---------------------------------------------------------------------------


def _ess_fraction(log_weights):
    """Effective sample size of unnormalised log-weights, as a fraction of their number."""
    log_ess = 2 * torch.logsumexp(log_weights, 0) - torch.logsumexp(2 * log_weights, 0)
    return (log_ess.exp() / log_weights.numel()).item()


def _next_increment(ll, remaining, target_ess, iterations=50):
    """Largest tempering increment in (0, remaining] keeping the ESS fraction >= target."""
    if _ess_fraction(_tempered(remaining, ll)) >= target_ess:
        return remaining
    low, high = 0.0, remaining
    for _ in range(iterations):  # ESS falls monotonically as the increment grows
        mid = 0.5 * (low + high)
        if _ess_fraction(_tempered(mid, ll)) >= target_ess:
            low = mid
        else:
            high = mid
    return max(low, 1e-12)


def _systematic_resample(log_weights):
    """Indices of a systematic resample of the particles by their log-weights."""
    n = log_weights.numel()
    weights = torch.softmax(log_weights, 0)
    positions = (
        torch.rand((), device=weights.device, dtype=weights.dtype)
        + torch.arange(n, device=weights.device, dtype=weights.dtype)
    ) / n
    return torch.searchsorted(weights.cumsum(0), positions).clamp(max=n - 1)


@torch.no_grad()
def tempered_smc(
    flow: BaseFlow,
    condition: BaseCondition,
    num_particles: int,
    kernel: str = "pcn",
    num_moves: int = 5,
    target_ess: float = 0.5,
    rho: float = 0.5,
    target_accept: float = 0.25,
    max_shrinks: int = 100,
    max_stages: int = 1000,
    max_likelihood_evals: int | None = None,
    init: torch.Tensor | None = None,
) -> SMCResult:
    """Likelihood-tempered SMC on the innovations.

    Bridges from the ``N(0, I)`` prior to the target through
    ``pi_beta(xi) ∝ L(T(xi))^beta N(xi; 0, I)``, ``0 = beta_0 < ... < beta_K = 1``. Each
    stage chooses the next ``beta`` adaptively so the incremental weights keep an
    effective sample size of ``target_ess`` (by bisection), reweights, resamples
    systematically, and then moves every particle with ``num_moves`` steps of a
    ``pi_beta``-invariant kernel. Both kernels offered are prior-reversible, so they are
    exact for every tempered target and need no gradients.

    Args:
        num_particles: particles; also the number of returned draws.
        kernel: ``"pcn"`` or ``"ess"`` (elliptical slice).
        num_moves: kernel steps per stage, to rejuvenate the resampled particles.
        target_ess: effective-sample-size fraction kept by each tempering step.
        rho: initial pCN step size, adapted towards ``target_accept`` at every move
            (the tempered targets sharpen, so the useful step size shrinks with beta).
        max_shrinks: elliptical-slice bracket shrinks before a move is abandoned.
        max_stages: safety cap on the number of tempering stages.
        max_likelihood_evals: optional budget of evaluations over all particles,
            including the initial ones. Tempering always runs to ``beta = 1`` (so it may
            overrun a small budget); whatever remains is then spent on further moves at
            ``beta = 1``, with the pCN step size frozen, which rejuvenates the final
            particles. The number of stages is adaptive, so this is how to run on a
            fixed budget.
        init: optional starting particles ``(num_particles, C, latent_dim)``, which
            should be prior draws; prior draws by default.
    """
    if kernel not in ("pcn", "ess"):
        raise ValueError(f"kernel must be 'pcn' or 'ess', got {kernel!r}")
    _check_flow(flow, condition)
    start = time.perf_counter()
    xi = _initial_state(flow, num_particles, init)
    ll = _log_likelihood(flow, condition, xi)
    if torch.isneginf(ll).all():
        raise ValueError("every initial particle has zero likelihood; SMC cannot start")
    evals = num_particles
    beta, log_evidence, betas = 0.0, 0.0, [0.0]
    log_rho, accepted, proposed, shrinks_total, moves = math.log(rho), 0, 0, 0, 0

    def move(xi, ll, beta, adapt):
        """One kernel step for every particle, targeting the beta-tempered posterior."""
        nonlocal evals, log_rho, accepted, proposed, shrinks_total, moves
        moves += 1
        if kernel == "pcn":
            xi, ll, accept = _pcn_step(flow, condition, xi, ll, math.exp(log_rho), beta)
            evals += num_particles
            accepted += accept.sum().item()
            proposed += num_particles
            if adapt:
                rate = accept.double().mean().item()
                log_rho = min(0.0, log_rho + rate - target_accept)
            return xi, ll
        xi, ll, step_evals, shrinks = _ess_step(
            flow, condition, xi, ll, beta, max_shrinks
        )
        evals += step_evals
        shrinks_total += shrinks
        return xi, ll

    while beta < 1.0:
        if len(betas) > max_stages:
            raise RuntimeError(
                f"tempered SMC did not reach beta = 1 in {max_stages} stages"
            )
        increment = _next_increment(ll, 1.0 - beta, target_ess)
        log_weights = _tempered(increment, ll)
        log_evidence += (
            torch.logsumexp(log_weights, 0) - math.log(num_particles)
        ).item()
        idx = _systematic_resample(log_weights)
        xi, ll = xi[idx], ll[idx]
        beta = 1.0 if increment >= 1.0 - beta else beta + increment
        betas.append(beta)

        for _ in range(num_moves):
            xi, ll = move(xi, ll, beta, adapt=True)

    tempering_moves = moves
    while max_likelihood_evals is not None and evals < max_likelihood_evals:
        xi, ll = move(xi, ll, 1.0, adapt=False)

    stats = {"num_stages": len(betas) - 1, "final_moves": moves - tempering_moves}
    if kernel == "pcn":
        stats |= {
            "acceptance_rate": accepted / max(proposed, 1),
            "rho": math.exp(log_rho),
        }
    else:
        stats["mean_shrinks_per_move"] = shrinks_total / max(moves * num_particles, 1)
    return SMCResult(
        _to_samples(flow, xi),
        xi,
        log_evidence,
        betas,
        evals,
        time.perf_counter() - start,
        stats,
    )
