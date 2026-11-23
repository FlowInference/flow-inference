"""Expectation propagation (EP) on the innovations, as a deterministic baseline.

EP (Minka, 2001) approximates the target ``rho(xi) ∝ N(xi; 0, I) prod_i t_i(s_i)``
by a Gaussian, when the likelihood factorises into *sites* ``t_i`` that each see one
linear function of the innovations, ``s_i = a_i . xi + c_i``. Each site is replaced
by a Gaussian factor in ``s_i``, fitted so that the approximation matches the mean
and variance of the *tilted* distribution (the approximation with that one site
restored exactly); the fits are repeated until they agree.

That structure covers most constraint-type conditions: point observations with any
1D likelihood (Gaussian, Poisson, Student-t, ...), and inequalities on linear
functionals of the field, such as monotonicity (``f(x_{j+1}) - f(x_j) > 0``) and
bounds. EP is the classic method for exactly these (Riihimäki & Vehtari, 2010, for
monotonic GPs; Rasmussen & Williams, 2006, Sec. 3.6, for classification), where the
Laplace approximation fails. Its answer is a Gaussian and not exact, so it serves as
a speed and accuracy reference point rather than ground truth. It also estimates the
evidence.

:func:`linear_sites` reads the sites off a condition: any :class:`BinaryCondition`
or :class:`EqualityCondition` (or a product of them) whose margins or residuals are
linear in the field. For other likelihoods, build :class:`Sites` directly.

Each sweep updates every site in parallel (damped), then refactorises the ``D x D``
posterior precision once, costing ``O(D^3 + m D^2)`` for ``m`` sites.
"""

import copy
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from benchmarks.mcmc import _check_flow, _to_samples
from flow_inference.conditions import (
    BaseCondition,
    BinaryCondition,
    EqualityCondition,
    ProductCondition,
)
from flow_inference.flows import BaseFlow

F64 = torch.float64


@dataclass
class Sites:
    """A factorised likelihood ``prod_i t_i(s_i)``, ``s = A xi + c``.

    Attributes:
        A: ``(m, D)`` linear maps from the flattened latents ``xi`` (``D =
            channels * latent_dim``) to the site inputs.
        c: ``(m,)`` offsets.
        log_site: maps site inputs ``(m, Q)`` (row ``i`` holds values of ``s_i``) to
            ``log t_i`` elementwise, ``(m, Q)``.
        inequality: ``(m,)`` mask of the sites that relax an inequality ``s_i >= 0``
            (from a :class:`BinaryCondition`); None when unknown.
        precision: ``(m,)`` precision ``w_i / sigma_i^2`` of the Gaussian sites (from an
            :class:`EqualityCondition`), 0 for the others; None when unknown.
    """

    A: torch.Tensor
    c: torch.Tensor
    log_site: Callable[[torch.Tensor], torch.Tensor]
    inequality: torch.Tensor | None = None
    precision: torch.Tensor | None = None


def _components(condition):
    """The pieces of ``condition`` that :func:`linear_sites` needs.

    Returns:
        ``(components, log_t, kind)``: the function giving the per-component margins
        (Binary) or residuals (Equality) ``(B, C, A)``, the matching 1D log-site
        ``log t(s)`` before quadrature weights, and ``(kind, precision)``.
    """
    if isinstance(condition, BinaryCondition):
        v = float(condition.v)
        return (
            condition.smooth_eval_condition,
            lambda s: torch.nn.functional.logsigmoid(s / v),
            ("inequality", None),
        )
    if isinstance(condition, EqualityCondition):
        sigma = float(condition.sigma)
        return (
            condition.residual,
            lambda s: -0.5 * (s / sigma) ** 2,
            ("gaussian", 1.0 / sigma**2),
        )
    raise TypeError(
        f"{type(condition).__name__} is neither a BinaryCondition nor an "
        "EqualityCondition; build Sites for it directly"
    )


def linear_sites(flow, condition: BaseCondition) -> Sites:
    """Read EP sites off ``condition`` evaluated on ``flow``'s samples.

    Supports :class:`BinaryCondition` (a site ``w_a logsigmoid(s_a / v)`` per
    margin), :class:`EqualityCondition` (``-w_a s_a^2 / 2 sigma^2`` per residual) and
    :class:`ProductCondition` of them, provided every margin or residual is an affine
    function of the latents: monotonicity, bounds and point observations on a GP flow
    are; a PDE residual is not. ``A`` is the Jacobian at ``xi = 0`` and ``c`` the
    value there, both computed on a float64 copy of the flow; affinity is checked at
    a random point.

    Args:
        flow: a flow with ``N(0, I)`` latents.
        condition: the condition to read the sites from.

    Returns:
        The :class:`Sites`, in float64.

    Raises:
        TypeError: for an unsupported condition type.
        ValueError: when a margin or residual is not affine in the latents.
    """
    condition.bind_output_shape(getattr(flow, "output_shape", None))
    flow64 = copy.deepcopy(flow).double()
    C, N = flow.channels, flow.latent_dim
    D = C * N
    leaves = (
        list(condition.conditions)
        if isinstance(condition, ProductCondition)
        else [condition]
    )
    device = flow.device
    zero = torch.zeros(D, dtype=F64, device=device)
    probe = torch.randn(D, dtype=F64, device=device)
    As, cs, weights, log_ts, kinds, sizes = [], [], [], [], [], []
    for leaf in leaves:
        components, log_t, kind = _components(leaf)

        def values(xi, components=components):
            x = flow64._to_data_space(xi.view(1, C, N))
            return components(x).reshape(-1).to(F64)

        A = torch.autograd.functional.jacobian(values, zero)  # (m, D)
        c = values(zero).detach()
        with torch.no_grad():
            exact = values(probe)
        err = (exact - (A @ probe + c)).abs().max().item()
        if err > 1e-8 * (1.0 + exact.abs().max().item()):
            raise ValueError(
                f"{type(leaf).__name__}'s margins/residuals are not affine in the "
                f"latents (error {err:.1e}), so EP sites cannot be read off it"
            )
        w = leaf.quadrature_weights
        shape = (c.numel(),) if w is None else None
        if w is not None:
            shape = components(flow64._to_data_space(zero.view(1, C, N))).shape[1:]
            w = torch.broadcast_to(w.to(device), shape)
        w = torch.ones(shape, device=device) if w is None else w
        As.append(A.detach())
        cs.append(c)
        weights.append(w.reshape(-1).to(device=device, dtype=F64))
        log_ts.append(log_t)
        kinds.append(kind)
        sizes.append(c.numel())
    bounds = [0, *torch.tensor(sizes).cumsum(0).tolist()]
    w = torch.cat(weights)
    inequality = torch.cat(
        [
            torch.full((n,), kind == "inequality", device=device)
            for (kind, _), n in zip(kinds, sizes, strict=True)
        ]
    )
    precision = torch.cat(
        [
            torch.full((n,), 0.0 if p is None else p, dtype=F64, device=device)
            for (_, p), n in zip(kinds, sizes, strict=True)
        ]
    )

    def log_site(s):
        parts = [
            log_t(s[lo:hi])
            for log_t, lo, hi in zip(log_ts, bounds, bounds[1:], strict=False)
        ]
        return w[:, None] * torch.cat(parts)

    return Sites(torch.cat(As), torch.cat(cs), log_site, inequality, precision * w)


def tilted_moments(log_site, mu, var, num_points=2001, width=10.0):
    """Log normaliser, mean and variance of each tilted ``N(s; mu, var) t_i(s)``.

    Uses the trapezoid rule on ``num_points`` nodes over ``mu +- width * sd``. A fixed
    grid handles step-like sites that Gauss–Hermite quadrature would miss.

    Args:
        log_site: the sites' log-factors, as :attr:`Sites.log_site`.
        mu, var: the cavity means and variances, ``(m,)``.
        num_points, width: the quadrature grid.

    Returns:
        ``(log_z, mean, var)``, each ``(m,)``.
    """
    sd = var.sqrt()
    u = torch.linspace(-width, width, num_points, dtype=F64, device=mu.device)
    s = mu[:, None] + sd[:, None] * u  # (m, Q)
    log_f = log_site(s) - 0.5 * u**2 - 0.5 * math.log(2 * math.pi)
    log_f = torch.nan_to_num(log_f, nan=-math.inf)
    log_z = torch.logsumexp(log_f, 1) + torch.log(u[1] - u[0])
    p = torch.softmax(log_f, 1)
    mean = (p * s).sum(1)
    return log_z, mean, (p * (s - mean[:, None]) ** 2).sum(1)


@dataclass
class EPFit:
    """Fitted sites and the Gaussian approximation they give.

    Attributes:
        tau, nu: the site precisions and precision-weighted means, ``(m,)``.
        converged: whether the site marginals stopped changing within ``tol``.
        sweeps: sweeps run.
        skipped: site updates skipped in the last sweep (a cavity with non-positive
            precision).
        L: Cholesky factor of the posterior precision ``Lambda``, ``(D, D)``.
        mean: the posterior mean, ``(D,)``.
        eta: ``Lambda mean``, ``(D,)``.
        s_mean, s_var: the posterior marginals of the site inputs ``s = A xi + c``.
    """

    tau: torch.Tensor
    nu: torch.Tensor
    converged: bool
    sweeps: int
    skipped: int
    L: torch.Tensor
    mean: torch.Tensor
    eta: torch.Tensor
    s_mean: torch.Tensor
    s_var: torch.Tensor


@torch.no_grad()
def ep_fit(
    sites: Sites,
    prior_mean: torch.Tensor | None = None,
    prior_var: float = 1.0,
    tau: torch.Tensor | None = None,
    nu: torch.Tensor | None = None,
    max_sweeps: int = 1000,
    damping: float = 0.2,
    tol: float = 1e-4,
    num_points: int = 2001,
    width: float = 10.0,
) -> EPFit:
    """Fit parallel, damped EP to ``N(xi; prior_mean, prior_var I) prod_i t_i(s_i)``.

    Args:
        sites: the factorised likelihood.
        prior_mean, prior_var: the Gaussian prior on ``xi`` (default ``N(0, I)``).
        tau, nu: initial site parameters (a warm start); zeros by default.
        max_sweeps: sweeps at most; each updates every site, then refactorises.
        damping: fraction of each site update applied (1 is undamped). Parallel EP
            needs damping to converge when sites overlap.
        tol: converged once, over a sweep, no site marginal's mean moves by more
            than this many standard deviations and no variance by more than this
            fraction. The marginals are tested rather than the site parameters, which
            can keep jittering for near-hard sites (whose precisions are huge)
            without the posterior changing.
        num_points, width: quadrature for the tilted moments.

    Returns:
        An :class:`EPFit`.
    """
    A, c = sites.A.to(F64), sites.c.to(F64)
    m, D = A.shape
    device = A.device
    m0 = torch.zeros(D, dtype=F64, device=device) if prior_mean is None else prior_mean
    eye = torch.eye(D, dtype=F64, device=device)
    tau = torch.zeros(m, dtype=F64, device=device) if tau is None else tau.clone()
    nu = torch.zeros(m, dtype=F64, device=device) if nu is None else nu.clone()

    def posterior(tau, nu):
        L = torch.linalg.cholesky(eye / prior_var + A.T @ (tau[:, None] * A))
        eta = m0 / prior_var + A.T @ (nu - tau * c)
        mean = torch.cholesky_solve(eta[:, None], L)[:, 0]
        V = torch.linalg.solve_triangular(L, A.T, upper=False)  # (D, m)
        return L, mean, eta, A @ mean + c, (V**2).sum(0)

    converged, sweeps, skipped, previous = False, 0, 0, None
    for sweeps in range(1, max_sweeps + 1):  # noqa: B007
        _, _, _, s_mean, s_var = posterior(tau, nu)
        if previous is not None:
            change = max(
                ((s_mean - previous[0]).abs() / s_var.sqrt()).max().item(),
                torch.log(s_var / previous[1]).abs().max().item(),
            )
            if change < tol:
                converged = True
                break
        previous = (s_mean, s_var)
        tau_cav, nu_cav = 1.0 / s_var - tau, s_mean / s_var - nu
        ok = tau_cav > 0
        skipped = int((~ok).sum())
        mu_cav = torch.where(ok, nu_cav / tau_cav.clamp(min=1e-300), s_mean)
        var_cav = torch.where(ok, 1.0 / tau_cav.clamp(min=1e-300), s_var)
        _, t_mean, t_var = tilted_moments(
            sites.log_site, mu_cav, var_cav, num_points, width
        )
        # Moment matching: the site that turns the cavity into the tilted Gaussian.
        # Clipped at zero precision, which keeps Lambda positive definite (log-concave
        # sites never need negative precisions).
        tau_new = torch.where(ok, (1.0 / t_var - tau_cav).clamp(min=0.0), tau)
        nu_new = torch.where(ok, t_mean / t_var - nu_cav, nu)
        tau = (1 - damping) * tau + damping * tau_new
        nu = (1 - damping) * nu + damping * nu_new
    L, mean, eta, s_mean, s_var = posterior(tau, nu)
    return EPFit(tau, nu, converged, sweeps, skipped, L, mean, eta, s_mean, s_var)


@dataclass
class EPResult:
    """Output of :func:`ep`.

    Attributes:
        samples: draws from the Gaussian approximation in data space,
            ``(num_samples, C, *output_shape)``.
        latents: the same draws as innovations, ``(num_samples, C, latent_dim)``.
        mean: the approximation's mean innovations, ``(C, latent_dim)``.
        scale_tril: Cholesky factor of its covariance, ``(D, D)``.
        log_evidence: EP's estimate of ``log E_{xi ~ N(0, I)}[prod_i t_i(s_i)]``.
        num_sweeps: sweeps run.
        converged: whether the site marginals stopped changing within ``tol``.
        num_likelihood_evals: site evaluations, counted as ``m * Q`` per sweep, so
            ``Q`` 1D evaluations per site; not comparable one-to-one with a full
            ``condition.logp`` call, which evaluates every site at once.
        wall_time: seconds spent.
        stats: number of sites, and how many site updates were skipped (a cavity
            with non-positive precision) in the last sweep.
    """

    samples: torch.Tensor
    latents: torch.Tensor
    mean: torch.Tensor
    scale_tril: torch.Tensor
    log_evidence: float
    num_sweeps: int
    converged: bool
    num_likelihood_evals: int
    wall_time: float
    stats: dict = field(default_factory=dict)


@torch.no_grad()
def ep(
    flow: BaseFlow,
    condition: BaseCondition | None = None,
    num_samples: int = 1000,
    sites: Sites | None = None,
    max_sweeps: int = 1000,
    damping: float = 0.2,
    tol: float = 1e-4,
    num_points: int = 2001,
    width: float = 10.0,
) -> EPResult:
    """Expectation propagation for a factorised likelihood on the innovations.

    Pass exactly one of ``condition`` and ``sites``. A warning is raised if EP does
    not converge within ``max_sweeps``.

    Args:
        flow: a flow with an ``N(0, I)`` innovation prior.
        condition: read the sites off this with :func:`linear_sites`.
        num_samples: draws from the fitted Gaussian to return.
        sites: the factorised likelihood, given directly.
        max_sweeps, damping, tol, num_points, width: as for :func:`ep_fit`: the
            maximum number of sweeps, the fraction of each site update applied, the
            convergence tolerance on the site marginals, and the quadrature for the
            tilted moments.

    Returns:
        An :class:`EPResult`.
    """
    if (condition is None) == (sites is None):
        raise ValueError("pass exactly one of condition and sites")
    start = time.perf_counter()
    if sites is None:
        _check_flow(flow, condition)
        sites = linear_sites(flow, condition)
    fit = ep_fit(
        sites,
        max_sweeps=max_sweeps,
        damping=damping,
        tol=tol,
        num_points=num_points,
        width=width,
    )
    if not fit.converged:
        warnings.warn(
            f"EP did not converge in {max_sweeps} sweeps; lower damping or raise "
            "max_sweeps",
            stacklevel=2,
        )
    A, c = sites.A.to(F64), sites.c.to(F64)
    m, D = A.shape
    tau, nu, L, mean, eta = fit.tau, fit.nu, fit.L, fit.mean, fit.eta

    # Evidence at the fitted sites (Rasmussen & Williams, 2006, eq. 3.65, for sites
    # on linear functionals): the Gaussian integral of the site factors, times the
    # per-site normaliser that makes each cavity-times-site integrate to its tilted Z_i.
    tau_cav = (1.0 / fit.s_var - tau).clamp(min=1e-300)
    nu_cav = fit.s_mean / fit.s_var - nu
    log_z_tilted, _, _ = tilted_moments(
        sites.log_site, nu_cav / tau_cav, 1.0 / tau_cav, num_points, width
    )
    log_gauss = (
        -0.5 * (tau * c**2).sum()
        + (nu * c).sum()
        - torch.log(torch.diagonal(L)).sum()
        + 0.5 * (eta * mean).sum()
    )
    # log of  int N(s; cavity) exp(-tau s^2 / 2 + nu s) ds
    log_site_gauss = (
        0.5 * (nu_cav + nu) ** 2 / (tau_cav + tau)
        - 0.5 * nu_cav**2 / tau_cav
        + 0.5 * torch.log(tau_cav / (tau_cav + tau))
    )
    log_evidence = (log_gauss + (log_z_tilted - log_site_gauss).sum()).item()

    # Draws from N(mean, Lambda^-1): xi = mean + L^-T e.
    e = torch.randn(D, num_samples, dtype=F64, device=A.device)
    xi = mean[:, None] + torch.linalg.solve_triangular(L.T, e, upper=True)
    C, N = flow.channels, flow.latent_dim
    latents = xi.T.reshape(num_samples, C, N).to(flow.dtype)
    return EPResult(
        samples=_to_samples(flow, latents),
        latents=latents,
        mean=mean.reshape(C, N).to(flow.dtype),
        scale_tril=torch.linalg.cholesky(torch.cholesky_inverse(L)),
        log_evidence=log_evidence,
        num_sweeps=fit.sweeps,
        converged=fit.converged,
        num_likelihood_evals=(fit.sweeps + 1) * m * num_points,
        wall_time=time.perf_counter() - start,
        stats={"num_sites": m, "skipped_updates": fit.skipped},
    )
