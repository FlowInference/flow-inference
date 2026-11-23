"""Reference answers for Gaussian processes under linear inequality constraints.

For a condition made of inequalities on linear functionals of the field
(monotonicity, bounds; :class:`BinaryCondition` in its hard limit ``v -> 0``) and,
optionally, Gaussian observations (:class:`EqualityCondition` with affine
residuals), the conditioned latents follow a Gaussian truncated to a polytope:

    ``rho(xi)  ∝  N(xi; 0, I) prod_g exp(-p_g (a_g . xi + c_g)^2 / 2)
    prod_i 1[b_i . xi + d_i >= 0]``.

:func:`lin_ess` samples it and estimates its normalising constant with the method of
Gessner, Kanjilal & Hennig (2020, "Integrals over Gaussians under linear domain
constraints"):

* **Draws: LIN-ESS.** Elliptical slice sampling where the likelihood is the
  indicator of the polytope. On the ellipse ``u cos(t) + nu sin(t)`` each constraint
  is violated on one arc of ``t``, computable in closed form, so the feasible ``t``
  around the current state is known exactly and a move is drawn uniformly from it.
  There are no rejections, no infeasible states, and the chain leaves the truncated
  Gaussian exactly invariant.
* **Evidence: subset simulation** (Au & Beck, 2001). The constraints are relaxed by
  a shift ``gamma`` (``b_i . u + d_i + gamma >= 0``), which is lowered in stages to
  0. Each stage keeps the fraction ``rho`` of the current population that already
  satisfies the next shift, and LIN-ESS moves spread the population back out inside
  it. The product of the kept fractions estimates the probability of the polytope;
  it is consistent (with an ``O(1 / num_samples)`` bias) and stays accurate for
  probabilities far too small for prior Monte Carlo.

The Gaussian observation sites are integrated exactly first, so the polytope is
taken under the Gaussian posterior they give. The inequality sites are treated as
hard constraints, the limit those conditions approximate: at the softness used in
practice (``v`` far below the field's scale) the difference from the logistic
relaxation is negligible. All sites must be affine in the latents, which is checked.
"""

import math
import time
from dataclasses import dataclass, field

import torch

from benchmarks.ep import linear_sites
from benchmarks.mcmc import _check_flow, _to_samples
from flow_inference.conditions import BaseCondition
from flow_inference.flows import BaseFlow

F64 = torch.float64


@dataclass
class LinESSResult:
    """Output of :func:`lin_ess`.

    Attributes:
        samples: draws from the truncated Gaussian in data space,
            ``(num_samples, C, *output_shape)``.
        latents: the same draws as innovations, ``(num_samples, C, latent_dim)``.
        log_evidence: estimate of ``log E_{xi ~ N(0, I)}[L(T(xi))]`` for the
            hard-limit likelihood, on the same scale as the samplers' estimates.
        log_probability: the polytope's part of it, ``log P(constraints)`` under the
            Gaussian posterior of the observation sites.
        num_levels: subset-simulation stages.
        wall_time: seconds spent.
        stats: number of inequality and Gaussian sites, and the fraction of LIN-ESS
            moves dropped for falling outside the polytope by rounding (should be 0).
    """

    samples: torch.Tensor
    latents: torch.Tensor
    log_evidence: float
    log_probability: float
    num_levels: int
    wall_time: float
    stats: dict = field(default_factory=dict)


def _lin_ess_step(u: torch.Tensor, B: torch.Tensor, d: torch.Tensor):
    """One LIN-ESS move for each row of ``u`` ``(n, D)``, inside ``B u + d >= 0``.

    On the ellipse ``u(t) = u cos t + nu sin t`` constraint ``i`` reads
    ``R_i cos(t - phi_i) + d_i >= 0`` and, when ``|d_i| < R_i``, is violated on the
    arc ``t - phi_i in (alpha_i, 2 pi - alpha_i)``, ``alpha_i = arccos(-d_i / R_i)``.
    The feasible interval around ``t = 0`` (the current state) runs from the end of the
    nearest arc below 0 to the start of the nearest arc above it; ``t`` is drawn
    uniformly from it. That interval is the same set of points whichever of its
    points the ellipse is parametrised from, so the move is reversible with respect
    to the truncated Gaussian.

    Returns:
        ``(u, dropped)``: the new states and the number of moves dropped because
        rounding put the proposal outside the polytope.
    """
    nu = torch.randn_like(u)
    p, q = u @ B.T, nu @ B.T  # (n, m)
    R = torch.sqrt(p**2 + q**2)
    phi = torch.atan2(q, p)
    active = R > d.abs()
    alpha = torch.arccos((-d / R.clamp(min=1e-300)).clamp(-1.0, 1.0))
    start = torch.remainder(phi + alpha, 2 * math.pi)  # where the forbidden arc begins
    end = start + 2 * (math.pi - alpha)  # and ends (<= 2 pi, as t = 0 is feasible)
    hi = torch.where(active, start, torch.inf).amin(1)
    lo = torch.where(active, end - 2 * math.pi, -torch.inf).amax(1)
    unconstrained = ~active.any(1)
    hi = torch.where(unconstrained, math.pi, hi).clamp(min=0.0)
    lo = torch.where(unconstrained, -math.pi, lo).clamp(max=0.0)
    t = lo + (hi - lo) * torch.rand_like(hi)
    proposal = u * torch.cos(t)[:, None] + nu * torch.sin(t)[:, None]
    # Rounding can put a proposal a hair outside; keep the current state then.
    inside = (proposal @ B.T + d >= 0).all(1)
    return torch.where(inside[:, None], proposal, u), int((~inside).sum())


def _subset_simulation(B, d, num_samples, rho, num_moves, max_levels):
    """Draws from ``N(0, I)`` restricted to ``B u + d >= 0``, and ``log P`` of that set.

    Returns:
        ``(u, log_probability, num_levels, dropped_moves)``.
    """
    D = B.shape[1]
    u = torch.randn(num_samples, D, dtype=F64, device=B.device)
    log_p, dropped = 0.0, 0
    for level in range(1, max_levels + 1):
        shift = (-(u @ B.T + d)).amax(1)  # the smallest gamma each draw satisfies
        gamma = max(torch.quantile(shift, rho).item(), 0.0)
        inside = shift <= gamma
        log_p += math.log(inside.double().mean().item())
        survivors = inside.nonzero().squeeze(1)
        u = u[survivors[torch.randint(len(survivors), (num_samples,))]]
        for _ in range(num_moves):
            u, n = _lin_ess_step(u, B, d + gamma)
            dropped += n
        if gamma == 0.0:
            return u, log_p, level, dropped
    raise RuntimeError(
        f"subset simulation did not reach the constraints in {max_levels} levels"
    )


@torch.no_grad()
def lin_ess(
    flow: BaseFlow,
    condition: BaseCondition,
    num_samples: int = 1000,
    rho: float = 0.5,
    num_moves: int = 10,
    final_moves: int = 50,
    max_levels: int = 10_000,
) -> LinESSResult:
    """Reference draws and evidence for a linearly constrained Gaussian condition.

    Args:
        flow: a flow with ``N(0, I)`` latents and an affine map to data space (the GP
            flows).
        condition: a :class:`BinaryCondition` or :class:`EqualityCondition`, or a
            :class:`ProductCondition` of them, with affine margins/residuals; the
            inequalities are taken in their hard limit.
        num_samples: population size, and the number of draws returned.
        rho: fraction of the population kept at each subset-simulation stage.
        num_moves: LIN-ESS moves per stage, to spread out the duplicates.
        final_moves: extra moves at the end, so the returned draws are well mixed.
        max_levels: cap on the subset-simulation stages.

    Returns:
        A :class:`LinESSResult`.

    Raises:
        TypeError: for an unsupported condition type.
        ValueError: when ``flow`` has no ``N(0, I)`` latents, or a margin or
            residual is not affine in them.
        RuntimeError: when subset simulation does not finish in ``max_levels``.
    """
    _check_flow(flow, condition)
    start = time.perf_counter()
    sites = linear_sites(flow, condition)
    A, c = sites.A.to(F64), sites.c.to(F64)
    ineq, precision = sites.inequality, sites.precision
    D = A.shape[1]
    eye = torch.eye(D, dtype=F64, device=A.device)

    # Integrate the Gaussian sites exactly: N(0, I) prod_g exp(-p_g s_g^2 / 2) is
    # Z_g N(mu, Lambda^-1), and xi = mu + L^-T u maps u ~ N(0, I) onto it.
    Ag, cg, pg = A[~ineq], c[~ineq], precision[~ineq]
    L = torch.linalg.cholesky(eye + Ag.T @ (pg[:, None] * Ag))
    eta = -Ag.T @ (pg * cg)
    mu = torch.cholesky_solve(eta[:, None], L)[:, 0]
    log_z_gauss = (
        -0.5 * (pg * cg**2).sum()
        + 0.5 * (eta * mu).sum()
        - torch.log(torch.diagonal(L)).sum()
    ).item()

    # The inequalities b . xi + c >= 0 in terms of u.
    B = torch.linalg.solve_triangular(L, A[ineq].T, upper=False).T  # A L^-T
    d = A[ineq] @ mu + c[ineq]
    if B.shape[0] == 0:
        u = torch.randn(num_samples, D, dtype=F64, device=A.device)
        log_p, levels, dropped = 0.0, 0, 0
    else:
        u, log_p, levels, dropped = _subset_simulation(
            B, d, num_samples, rho, num_moves, max_levels
        )
        for _ in range(final_moves):
            u, n = _lin_ess_step(u, B, d)
            dropped += n

    xi = mu + torch.linalg.solve_triangular(L.T, u.T, upper=True).T
    C, N = flow.channels, flow.latent_dim
    latents = xi.reshape(num_samples, C, N).to(flow.dtype)
    total_moves = num_samples * (levels * num_moves + (final_moves if levels else 0))
    return LinESSResult(
        samples=_to_samples(flow, latents),
        latents=latents,
        log_evidence=log_z_gauss + log_p,
        log_probability=log_p,
        num_levels=levels,
        wall_time=time.perf_counter() - start,
        stats={
            "inequality_sites": int(ineq.sum()),
            "gaussian_sites": int((~ineq).sum()),
            "dropped_moves": dropped / max(total_moves, 1),
        },
    )
