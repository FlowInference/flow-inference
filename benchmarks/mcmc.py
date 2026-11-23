"""MCMC and SMC baselines in innovation space, for benchmarking guided flows.

Every flow with ``has_standard_normal_marginals`` writes a sample as ``f = T(xi)``
with innovations ``xi ~ N(0, I)`` (``T`` is ``flow._to_data_space``). Conditioning
on a condition with likelihood ``L`` then targets

    ``rho(xi)  ∝  L(T(xi)) N(xi; 0, I)``,

whose pushforward through ``T`` is the conditioned process. The samplers here run
directly on that target:

* :func:`pcn` — preconditioned Crank–Nicolson Metropolis–Hastings (Cotter et al., 2013);
* :func:`elliptical_slice` — elliptical slice sampling (Murray, Adams & MacKay, 2010);
* :func:`pcn_mala` — the gradient-informed pCN, pCN-Langevin / infinity-MALA
  (Beskos et al., 2008; Cotter et al., 2013);
* :func:`hmc` — Hamiltonian Monte Carlo with an identity mass matrix (Neal, 2011);
* :func:`nuts` — the No-U-Turn Sampler (Hoffman & Gelman, 2014), using Pyro's
  implementation with step-size and diagonal mass-matrix adaptation;
* :func:`tempered_smc` — likelihood-tempered sequential Monte Carlo (Del Moral, Doucet
  & Jasra, 2006) with pCN or elliptical-slice moves, which also estimates the evidence.

pCN and elliptical slice sampling are reversible with respect to the ``N(0, I)``
prior, so they need only the pulled-back log-likelihood ``condition.logp(T(xi))``;
pCN-MALA, HMC and NUTS also differentiate it through the generator, as
:class:`~flow_inference.guidance.GuidanceMC` does.

All of them except NUTS run a batch of chains or particles in parallel (NUTS chains
build trees of different sizes, so they run one after another). Their ``samples``
have the same per-draw shape as ``IntegratorSampler.sample`` output, so the plots and
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
        samples: kept draws mapped to data space,
            ``(num_kept, num_chains, C, *output_shape)``.
        latents: the kept innovations, ``(num_kept, num_chains, C, latent_dim)``.
        log_likelihood: pulled-back log-likelihood of every chain after every
            iteration, ``(num_iters, num_chains)``; a mixing diagnostic.
        num_likelihood_evals: total ``condition.logp`` evaluations, summed over
            chains; the cost measure that is comparable across samplers. For the
            gradient-based samplers (pCN-MALA, HMC, NUTS) each evaluation includes
            a gradient.
        wall_time: seconds spent sampling.
        stats: sampler-specific diagnostics, such as the acceptance rate and tuned
            step size, or the mean bracket shrinks per iteration for elliptical slice.
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
        log_evidence: estimate of ``log E_{xi ~ N(0, I)}[L(T(xi))]``, the log
            normalising constant of the target (unbiased on the evidence scale).
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
    """Check that ``flow`` has an ``N(0, I)`` prior, and bind ``condition`` to it."""
    if not getattr(flow, "has_standard_normal_marginals", False):
        raise ValueError(
            f"{type(flow).__name__} has no N(0, I) innovation prior; the innovation-space "
            "samplers need one (use a GaussianFlow subclass)."
        )
    # Grid-aware conditions reshape samples with the flow's output shape.
    condition.bind_output_shape(getattr(flow, "output_shape", None))


def _scrub(logp: torch.Tensor) -> torch.Tensor:
    """Map NaN log-likelihoods to ``-inf`` (zero likelihood)."""
    return torch.nan_to_num(logp, nan=-math.inf, posinf=math.inf, neginf=-math.inf)


def _log_likelihood(flow, condition, xi: torch.Tensor) -> torch.Tensor:
    """The pulled-back log-likelihood ``(B,)`` of ``xi`` ``(B, C, latent_dim)``.

    NaN values become ``-inf``.
    """
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
    """Raise ``ValueError`` unless the schedule keeps at least one draw."""
    if thin < 1 or num_iters - burn_in < thin:
        raise ValueError(
            f"num_iters={num_iters}, burn_in={burn_in}, thin={thin} keeps no draws; "
            "need num_iters - burn_in >= thin >= 1"
        )


def _initial_state(flow, num_chains, init):
    """Starting innovations ``(num_chains, C, latent_dim)``: ``init`` or prior draws."""
    if init is not None:
        return init.to(device=flow.device, dtype=flow.dtype).clone()
    return torch.randn(
        num_chains, flow.channels, flow.latent_dim, device=flow.device, dtype=flow.dtype
    )


def _to_samples(flow, latents):
    """Map innovations ``(..., C, latent_dim)`` to draws ``(..., C, *output_shape)``."""
    lead = latents.shape[:-2]
    display = flow.to_display(
        flow._to_data_space(latents.reshape(-1, *latents.shape[-2:]))
    )
    return display.reshape(*lead, *display.shape[1:])


def _result(flow, kept, trace, evals, start, stats):
    """Assemble an :class:`MCMCResult` from the kept states and the trace."""
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
    """Whether ``iteration`` is past burn-in and on the thinning grid."""
    return iteration >= burn_in and (iteration - burn_in) % thin == thin - 1


def _adapt(log_step, rate, target, iteration):
    """Robbins–Monro on a log step size, towards a target acceptance rate."""
    return log_step + (rate - target) / math.sqrt(iteration + 1)


# ---------------------------------------------------------------------------
# Single-step kernels, each targeting  L(T(xi))^beta N(xi; 0, I)
# ---------------------------------------------------------------------------


def _pcn_step(flow, condition, xi, ll, rho, beta=1.0):
    """One pCN move per chain, costing one evaluation each.

    Returns:
        ``(xi, ll, accepted)``.
    """
    proposal = math.sqrt(1.0 - rho**2) * xi + rho * torch.randn_like(xi)
    ll_proposal = _log_likelihood(flow, condition, proposal)
    log_ratio = _tempered(beta, ll_proposal) - _tempered(beta, ll)
    accept = torch.rand_like(ll).log() < log_ratio  # NaN (-inf to -inf) rejects
    xi = torch.where(accept[:, None, None], proposal, xi)
    ll = torch.where(accept, ll_proposal, ll)
    return xi, ll, accept


def _ess_step(flow, condition, xi, ll, beta=1.0, max_shrinks=100):
    """One elliptical-slice move per chain.

    Returns:
        ``(xi, ll, evaluations, shrinks)``, the last two summed over chains.
    """
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
    """One HMC move per chain, with potential ``|xi|^2 / 2 - log L(T(xi))``.

    Costs ``num_leapfrog`` evaluations per chain. The step size is jittered by +-10%
    per chain to avoid periodic trajectories.

    Returns:
        ``(xi, ll, grad, accept_prob)``.
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

    Proposes ``xi' = sqrt(1 - rho^2) xi + rho eps`` with ``eps ~ N(0, I)``, which
    leaves the ``N(0, I)`` prior invariant, so the acceptance probability is just the
    likelihood ratio ``min(1, L(T(xi')) / L(T(xi)))``.

    Args:
        flow: a flow with an ``N(0, I)`` innovation prior (a ``GaussianFlow``).
        condition: the condition whose ``logp`` is the likelihood ``L``.
        num_chains: independent chains run in parallel.
        num_iters: iterations per chain, including burn-in.
        rho: step size in (0, 1]; ``rho = 1`` is an independence sampler from the prior.
        burn_in: initial iterations discarded. ``rho`` is adapted only during burn-in,
            so the kept iterations form a valid (non-adaptive) chain.
        thin: keep every ``thin``-th iteration after burn-in.
        target_accept: acceptance rate to adapt ``rho`` towards during burn-in, or None
            to hold ``rho`` fixed.
        init: optional starting innovations ``(num_chains, C, latent_dim)``; prior
            draws by default.

    Returns:
        An :class:`MCMCResult`; ``stats`` holds the post-burn-in acceptance rate and
        the final ``rho``.
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
    ``xi cos(theta) + nu sin(theta)`` for a point above the level, shrinking the
    angle bracket towards ``theta = 0`` (the current state) after every rejection.
    Every iteration moves, and there is no step size to tune. Chains still searching
    are evaluated together, so one iteration costs as many batched likelihood calls as
    its slowest chain needs.

    Args:
        flow, condition, num_chains, num_iters, burn_in, thin, init: as for
            :func:`pcn`.
        max_shrinks: give up on a chain's move after this many shrinks and keep its
            current state (only reached when the likelihood is degenerate, e.g.
            ``-inf``).
        max_likelihood_evals: optional budget of evaluations over all chains,
            including the initial state. An iteration costs a variable number of
            evaluations, so this is how to run on a fixed budget: no new iteration
            starts once it is spent (the last one may overrun it), and the chains'
            final state is kept as the last draw.

    Returns:
        An :class:`MCMCResult`; ``stats`` holds the number of iterations run and the
        mean bracket shrinks per chain and iteration.
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

    The potential is ``U(xi) = |xi|^2 / 2 - log L(T(xi))``; its gradient is taken
    by autograd through the generator. The ``N(0, I)`` reference makes the identity
    mass matrix the natural choice: under a weak likelihood the target is already
    isotropic with unit scale.

    Args:
        flow, condition, num_chains, num_iters, burn_in, thin, init: as for
            :func:`pcn`.
        step_size: initial leapfrog step size, jittered by +-10% per chain and move.
        num_leapfrog: leapfrog steps per move; each costs one likelihood-and-gradient
            evaluation per chain.
        target_accept: mean acceptance probability to adapt ``step_size`` towards
            during burn-in, or None to hold it fixed.

    Returns:
        An :class:`MCMCResult`; ``stats`` holds the post-burn-in mean acceptance
        probability, the final step size and ``num_leapfrog``.
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
# pCN-MALA
# ---------------------------------------------------------------------------


def _pcn_mala_step(flow, condition, xi, ll, grad, delta):
    """One pCN-Langevin move per chain.

    The proposal ``xi' = a xi + b grad log L(xi) + sigma eps`` with
    ``a = (2 - delta) / (2 + delta)``, ``b = 2 delta / (2 + delta)``,
    ``sigma^2 = 8 delta / (2 + delta)^2`` is Crank–Nicolson for the Langevin dynamics:
    ``a^2 + sigma^2 = 1``, so without the gradient it is pCN with ``rho = sigma`` and
    leaves the prior invariant. The Metropolis–Hastings ratio is evaluated from the
    explicit Gaussian proposal densities, in float64.

    Returns:
        ``(xi, ll, grad, accepted)``.
    """
    a, b = (2 - delta) / (2 + delta), 2 * delta / (2 + delta)
    sigma2 = 8 * delta / (2 + delta) ** 2
    proposal = a * xi + b * grad + math.sqrt(sigma2) * torch.randn_like(xi)
    ll_p, grad_p = _log_likelihood_and_grad(flow, condition, proposal)

    def log_q(to, frm, g):  # log density of proposing `to` from `frm`, up to a constant
        diff = to.double() - a * frm.double() - b * g.double()
        return -0.5 * diff.pow(2).sum((1, 2)) / sigma2

    def log_target(x, x_ll):
        return x_ll.double() - 0.5 * x.double().pow(2).sum((1, 2))

    log_ratio = (
        log_target(proposal, ll_p)
        - log_target(xi, ll)
        + log_q(xi, proposal, grad_p)
        - log_q(proposal, xi, grad)
    )
    accept = torch.rand_like(log_ratio).log() < torch.nan_to_num(
        log_ratio, nan=-math.inf
    )
    keep = accept[:, None, None]
    return (
        torch.where(keep, proposal, xi),
        torch.where(accept, ll_p, ll),
        torch.where(keep, grad_p, grad),
        accept,
    )


@torch.no_grad()
def pcn_mala(
    flow: BaseFlow,
    condition: BaseCondition,
    num_chains: int,
    num_iters: int,
    delta: float = 0.1,
    burn_in: int = 0,
    thin: int = 1,
    target_accept: float | None = 0.574,
    init: torch.Tensor | None = None,
) -> MCMCResult:
    """pCN-Langevin (infinity-MALA) on the innovations.

    pCN's proposal shifted along the likelihood gradient, discretised in
    Crank–Nicolson style so that it stays well defined as the latent dimension grows
    (pCN's property) while also using the gradient (MALA's). It is the natural
    gradient-based counterpart to pCN: the same prior-preserving structure, with one
    likelihood-and-gradient evaluation per move.

    Args:
        flow, condition, num_chains, num_iters, burn_in, thin, init: as for
            :func:`pcn`.
        delta: step size in ``(0, 2]``; ``delta = 2`` drops the current state from
            the proposal (an independence sampler shifted by the gradient).
        target_accept: acceptance rate to adapt ``delta`` towards during burn-in
            (0.574 is optimal for MALA-type proposals), or None to hold it fixed.

    Returns:
        An :class:`MCMCResult`; ``stats`` holds the post-burn-in acceptance rate and
        the final ``delta``.
    """
    _check_flow(flow, condition)
    _check_schedule(num_iters, burn_in, thin)
    start = time.perf_counter()
    xi = _initial_state(flow, num_chains, init)
    ll, grad = _log_likelihood_and_grad(flow, condition, xi)
    evals = num_chains
    log_delta = math.log(delta)
    kept, trace, accepted_after_burn_in = [], [], 0

    for iteration in range(num_iters):
        xi, ll, grad, accept = _pcn_mala_step(
            flow, condition, xi, ll, grad, math.exp(log_delta)
        )
        evals += num_chains
        if iteration < burn_in:
            if target_accept is not None:  # capped at delta = 2
                rate = accept.double().mean().item()
                log_delta = min(
                    math.log(2.0), _adapt(log_delta, rate, target_accept, iteration)
                )
        else:
            accepted_after_burn_in += accept.sum().item()
        trace.append(ll.clone())
        if _kept(iteration, burn_in, thin):
            kept.append(xi.clone())

    stats = {
        "acceptance_rate": accepted_after_burn_in
        / ((num_iters - burn_in) * num_chains),
        "delta": math.exp(log_delta),
    }
    return _result(flow, kept, trace, evals, start, stats)


# ---------------------------------------------------------------------------
# NUTS (Pyro)
# ---------------------------------------------------------------------------


class _BudgetSpent(Exception):
    """Raised from Pyro's per-iteration hook to end a chain on its budget."""


@torch.no_grad()
def nuts(
    flow: BaseFlow,
    condition: BaseCondition,
    num_chains: int,
    num_iters: int,
    step_size: float = 0.1,
    burn_in: int = 0,
    thin: int = 1,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    adapt_mass_matrix: bool = True,
    max_likelihood_evals: int | None = None,
    init: torch.Tensor | None = None,
) -> MCMCResult:
    """The No-U-Turn Sampler on the innovations, using Pyro's implementation.

    Each iteration simulates a Hamiltonian trajectory, doubling it forwards or
    backwards at random until it starts to turn back on itself, and draws the next
    state from the whole trajectory. There is no trajectory length to tune; during
    burn-in Pyro adapts the step size by dual averaging, and a diagonal mass matrix
    (Hoffman & Gelman, 2014). NUTS is the default sampler of Stan, PyMC and NumPyro,
    so it is what a practitioner would reach for.

    Needs ``pyro-ppl`` (the ``benchmarks`` dependency group). Chains run one after
    another (their trajectories have different lengths), so NUTS is much slower in
    wall-clock time than the batched samplers here; compare it on
    ``num_likelihood_evals`` (one per potential-and-gradient evaluation).

    Args:
        flow, condition, num_chains, num_iters, burn_in, thin, init: as for
            :func:`pcn`; ``burn_in`` is Pyro's warm-up, where the step size and mass
            matrix adapt.
        step_size: initial leapfrog step size.
        target_accept: acceptance probability that dual averaging aims for.
        max_tree_depth: at most ``2**max_tree_depth - 1`` leapfrog steps per
            iteration.
        adapt_mass_matrix: adapt a diagonal mass matrix during warm-up.
        max_likelihood_evals: optional budget over all chains, split equally between
            them. An iteration's cost varies with its trajectory, so this is how to
            run on a fixed budget: once warm-up is over, a chain stops after the
            iteration that spends its share, and every chain is cut to the shortest.

    Returns:
        An :class:`MCMCResult`; ``stats`` holds the mean final step size, the mean
        acceptance probability, the number of divergent transitions after warm-up and
        the mean leapfrog steps per iteration.
    """
    from pyro.infer.mcmc import MCMC, NUTS

    _check_flow(flow, condition)
    _check_schedule(num_iters, burn_in, thin)
    start = time.perf_counter()
    xi0 = _initial_state(flow, num_chains, init)
    evals = {"n": 0}

    def potential(params):
        xi = params["xi"]
        evals["n"] += 1
        ll = condition.logp(flow._to_data_space(xi.unsqueeze(0)))[0]
        return 0.5 * xi.pow(2).sum() - _scrub(ll)

    chain_budget = (
        None if max_likelihood_evals is None else max_likelihood_evals / num_chains
    )
    chains, warmups, step_sizes, accepts, divergences = [], [], [], [], 0
    for c in range(num_chains):
        chain_start, draws, warmup, accept = evals["n"], [], [], [0.0]

        def hook(
            kernel,
            params,
            stage,
            i,
            draws=draws,
            warmup=warmup,
            chain_start=chain_start,
            accept=accept,
        ):
            if stage != "Sample":
                warmup.append(params["xi"].detach().clone())
            else:
                draws.append(params["xi"].detach().clone())
                # Pyro resets this running mean when the run ends, so read it here.
                accept[0] = float(kernel._mean_accept_prob)
                if (
                    chain_budget is not None
                    and evals["n"] - chain_start >= chain_budget
                ):
                    raise _BudgetSpent

        kernel = NUTS(
            potential_fn=potential,
            step_size=step_size,
            adapt_step_size=burn_in > 0,
            adapt_mass_matrix=adapt_mass_matrix and burn_in > 0,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
        )
        mcmc = MCMC(
            kernel,
            num_samples=num_iters - burn_in,
            warmup_steps=burn_in,
            initial_params={"xi": xi0[c]},
            hook_fn=hook,
            disable_progbar=True,
        )
        with torch.enable_grad():
            try:
                mcmc.run()
            except _BudgetSpent:
                pass
        chains.append(torch.stack(draws))  # (iterations kept, C, latent_dim)
        warmups.append(warmup)
        step_sizes.append(float(kernel.step_size))
        accepts.append(accept[0])
        divergences += len(kernel._divergences)

    length = min(len(d) for d in chains)
    draws = torch.stack([d[:length] for d in chains], 1)  # (iters, chains, C, N)
    kept = [draws[i] for i in range(thin - 1, length, thin)]
    # As for the other samplers, the log-likelihood trace includes warm-up.
    path = torch.cat(
        [torch.stack([torch.stack(w) for w in warmups], 1), draws]
        if burn_in
        else [draws]
    )
    ll = _log_likelihood(flow, condition, path.flatten(0, 1))
    trace = list(ll.view(path.shape[0], num_chains))
    sampled_iterations = length * num_chains
    stats = {
        "step_size": sum(step_sizes) / num_chains,
        "acceptance_rate": sum(accepts) / num_chains,
        "divergences": divergences,
        "leapfrogs_per_iteration": evals["n"] / max(sampled_iterations, 1),
    }
    return _result(flow, kept, trace, evals["n"], start, stats)


# ---------------------------------------------------------------------------
# Tempered SMC
# ---------------------------------------------------------------------------


def _ess_fraction(log_weights):
    """Effective sample size of unnormalised log-weights, as a fraction of the count."""
    log_ess = 2 * torch.logsumexp(log_weights, 0) - torch.logsumexp(2 * log_weights, 0)
    return (log_ess.exp() / log_weights.numel()).item()


def _next_increment(ll, remaining, target_ess, iterations=50):
    """Largest tempering increment in ``(0, remaining]`` with ESS fraction >= target."""
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
    ``pi_beta(xi) ∝ L(T(xi))^beta N(xi; 0, I)``, ``0 = beta_0 < ... < beta_K = 1``.
    Each stage chooses the next ``beta`` by bisection, so that the incremental weights
    keep an effective sample size of ``target_ess``; it then reweights, resamples
    systematically, and moves every particle with ``num_moves`` steps of a
    ``pi_beta``-invariant kernel. Both kernels offered are prior-reversible, so they
    are exact for every tempered target and need no gradients. The product of the
    stages' mean incremental weights estimates the evidence.

    Args:
        flow, condition: as for :func:`pcn`.
        num_particles: particles; also the number of returned draws.
        kernel: ``"pcn"`` or ``"ess"`` (elliptical slice).
        num_moves: kernel steps per stage, to rejuvenate the resampled particles.
        target_ess: effective-sample-size fraction kept by each tempering step.
        rho: initial pCN step size, adapted towards ``target_accept`` at every move
            (the tempered targets sharpen, so the useful step size shrinks with beta).
        max_shrinks: elliptical-slice bracket shrinks before a move is abandoned.
        max_stages: safety cap on the number of tempering stages.
        max_likelihood_evals: optional budget of evaluations over all particles,
            including the initial ones. The number of stages is adaptive, so this is
            how to run on a fixed budget. Tempering always runs to ``beta = 1`` (so it
            may overrun a small budget); whatever remains is then spent on further
            moves at ``beta = 1``, with the pCN step size frozen, which rejuvenates the
            final particles.
        init: optional starting particles ``(num_particles, C, latent_dim)``, which
            should be prior draws; prior draws by default.

    Returns:
        An :class:`SMCResult`; ``stats`` holds the number of stages, the number of
        moves made after tempering, and the kernel's acceptance rate and final
        ``rho`` (pCN) or mean shrinks per move (elliptical slice).

    Raises:
        ValueError: for an unknown ``kernel``, or when every initial particle has
            zero likelihood.
        RuntimeError: when tempering does not reach ``beta = 1`` in ``max_stages``.
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
