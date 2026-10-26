"""Gaussian-process helpers — everything in the package that talks to gpytorch.

* :func:`fit_kernel_hyperparameters` — type-II maximum likelihood for a kernel's
  hyperparameters. Fitting and conditioning are separate steps: fit first, then call
  :func:`gp_posterior` to build the posterior, then hand that to
  ``GaussianProcessFlow``.
* :class:`PosteriorKernel` / :class:`PosteriorMean` — the posterior covariance and
  mean wrapped as a callable ``gpytorch.kernels.Kernel`` and ``gpytorch.means.Mean``.
  gpytorch exposes the posterior only as a ``MultivariateNormal`` over a fixed set of
  points; these make it evaluable at arbitrary inputs like any other kernel/mean.
* :func:`gp_posterior` — builds :class:`PosteriorMean` and :class:`PosteriorKernel`
  together, sharing one Cholesky factorisation of the training Gram matrix.
* :class:`AbstractBasis` — finite basis expansions, used to run the diffusion over
  ``R`` weights instead of ``N`` grid values. :class:`RFFBasis` expands a stationary
  prior kernel and has closed-form derivatives of its basis functions;
  :class:`DecoupledRFFBasis` is the conditioned counterpart, giving a GP *posterior*
  as a basis via pathwise conditioning (Wilson et al. 2020).
"""

import logging
import math
from abc import ABC, abstractmethod

import gpytorch
import torch
from gpytorch.kernels import RFFKernel, ScaleKernel
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _train_cholesky(
    base_kernel: gpytorch.kernels.Kernel,
    train_x: torch.Tensor,
    noise: float | torch.Tensor,
) -> torch.Tensor:
    """Lower Cholesky factor of the training Gram matrix ``K_XX + noise I``.

    Shared by :class:`PosteriorKernel` and :class:`PosteriorMean` — both condition on
    the same training set, and this is the only expensive part of doing so.
    """
    eye = torch.eye(train_x.shape[0], dtype=train_x.dtype, device=train_x.device)
    K_xx = base_kernel(train_x, train_x).to_dense() + noise * eye
    return torch.linalg.cholesky(K_xx)


class PosteriorKernel(gpytorch.kernels.Kernel):
    """GP posterior covariance as a callable kernel.

        k_post(x, x') = k(x, x') - k(x, X) [K_XX + noise I]^-1 k(X, x')

    gpytorch has no posterior-kernel object — the posterior is only reachable as a
    ``MultivariateNormal`` over a fixed set of points (``model(X_test).covariance_matrix``).
    This wraps the same quantity as a ``Kernel``, so it can be evaluated at arbitrary
    inputs, composed, and passed anywhere a kernel is expected.

    The Cholesky factor of the training Gram matrix is computed once in ``__init__``
    and registered as a buffer, so repeated evaluation is a triangular solve.

    Note this kernel is **not stationary**: the posterior variance collapses near the
    training inputs and returns to the prior far from them. Anything relying on
    stationarity — Bochner's theorem, and hence :class:`RFFBasis` — does not apply to
    it. For a finite basis of the posterior use :class:`DecoupledRFFBasis`, which
    expands only the stationary prior and applies the data correction exactly.

    Args:
        base_kernel: the prior kernel k.
        train_x: training inputs ``(N, D)``.
        noise: observation noise variance added to the training Gram diagonal.

    Example:
        >>> post = PosteriorKernel(kernel, X_train, noise=1e-4)
        >>> K = post(grid, grid).to_dense()      # (M, M) posterior covariance
        >>> var = post(grid, grid, diag=True)    # (M,) pointwise posterior variance
    """

    is_stationary = False

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        train_x: torch.Tensor,
        noise: float | torch.Tensor = 1e-4,
        _L: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base_kernel = base_kernel
        self.register_buffer("train_x", train_x)
        # _L lets :func:`gp_posterior` hand in a factor it already computed.
        L = _train_cholesky(base_kernel, train_x, noise) if _L is None else _L
        self.register_buffer("L", L)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params
    ) -> torch.Tensor:
        k12 = self.base_kernel(x1, x2).to_dense()
        k1X = self.base_kernel(x1, self.train_x).to_dense()
        kX2 = self.base_kernel(self.train_x, x2).to_dense()
        out = k12 - k1X @ torch.cholesky_solve(kX2, self.L)
        return out.diagonal(dim1=-2, dim2=-1) if diag else out


class PosteriorMean(gpytorch.means.Mean):
    """GP posterior mean as a callable mean function.

        m_post(x) = m(x) + k(x, X) [K_XX + noise I]^-1 (y - m(X))

    The companion to :class:`PosteriorKernel`: gpytorch gives you the posterior mean
    only as ``model(X_test).mean`` over a fixed set of points, whereas this evaluates
    at arbitrary inputs and drops into anything expecting a ``gpytorch.means.Mean``
    (an ``ExactGP``'s ``mean_module``, for instance).

    The representer weights ``alpha = [K_XX + noise I]^-1 (y - m(X))`` are solved once
    in ``__init__`` and stored, so evaluation is a single matmul — the same trick as
    gpytorch's internal ``mean_cache``.

    Args:
        base_kernel: the prior kernel k.
        train_x: training inputs ``(N, D)``.
        train_y: training targets ``(N,)``.
        noise: observation noise variance added to the training Gram diagonal.
        prior_mean: prior mean function m. Defaults to ``ZeroMean``; pass the same
            mean you conditioned on if it is not zero, or the posterior will be
            shifted.

    Example:
        >>> mu = PosteriorMean(kernel, X_train, y_train, noise=1e-4)
        >>> mu(grid)                             # (M,) posterior mean
    """

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        noise: float | torch.Tensor = 1e-4,
        prior_mean: gpytorch.means.Mean | None = None,
        _L: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base_kernel = base_kernel
        self.prior_mean = (
            gpytorch.means.ZeroMean() if prior_mean is None else prior_mean
        )
        self.register_buffer("train_x", train_x)
        # _L lets :func:`gp_posterior` hand in a factor it already computed.
        L = _train_cholesky(base_kernel, train_x, noise) if _L is None else _L
        residual = (train_y - self.prior_mean(train_x)).unsqueeze(-1)  # (N, 1)
        self.register_buffer("alpha", torch.cholesky_solve(residual, L).squeeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k_xX = self.base_kernel(x, self.train_x).to_dense()  # (M, N)
        return self.prior_mean(x) + k_xX @ self.alpha


def gp_posterior(
    base_kernel: gpytorch.kernels.Kernel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    noise: float | torch.Tensor = 1e-4,
    prior_mean: gpytorch.means.Mean | None = None,
) -> tuple["PosteriorMean", "PosteriorKernel"]:
    """Posterior mean *and* kernel, sharing a single Cholesky factorisation.

    :class:`PosteriorMean` and :class:`PosteriorKernel` each condition on the same
    training set, so building them separately factorises ``K_XX + noise I`` twice —
    the only O(N^3) step involved. This does it once and hands the factor to both.

    Returns ``(mean, kernel)``, matching the argument order
    :class:`~flow_inference.flows.GaussianProcessFlow` expects.

    Example:
        >>> mean, kernel = gp_posterior(rbf, X_train, y_train, noise=1e-4)
        >>> flow = GaussianProcessFlow(kernel, mean, grid)
    """
    L = _train_cholesky(base_kernel, train_x, noise)
    mean = PosteriorMean(base_kernel, train_x, train_y, noise, prior_mean, _L=L)
    kernel = PosteriorKernel(base_kernel, train_x, noise, _L=L)
    return mean, kernel


def fit_kernel_hyperparameters(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    kernel: gpytorch.kernels.Kernel,
    noise: torch.Tensor | float = 1e-4,
    num_iters: int = 500,
    lr: float = 0.1,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> gpytorch.kernels.Kernel:
    """Fit a kernel's hyperparameters by maximising the GP marginal likelihood.

    Type-II maximum likelihood (empirical Bayes) for

        f ~ GP(0, k),   y = f(X) + eps,  eps ~ N(0, noise I)

    The mean is held at zero and ``noise`` is held fixed; only the kernel's own
    hyperparameters (lengthscale, outputscale, ...) are optimised.

    ``kernel`` is used as the model's ``covar_module``, so it is fitted **in place**
    and also returned for chaining. Building the posterior is a separate step —
    see :func:`gp_posterior`:

        >>> kernel = fit_kernel_hyperparameters(X, y, kernel, noise=1e-4)
        >>> mean, post_kernel = gp_posterior(kernel, X, y, noise=1e-4)
        >>> flow = GaussianProcessFlow(post_kernel, mean, grid)

    Args:
        X_train: ``(N, D)`` training inputs.
        y_train: ``(N,)`` training targets.
        kernel: the kernel to fit; mutated in place.
        noise: observation noise variance, held fixed during fitting.
        num_iters: Adam steps on the negative marginal log-likelihood.
        lr: Adam learning rate.
        verbose: log the loss every 100 steps and the fitted hyperparameters.

    Returns:
        The same ``kernel`` object, with fitted hyperparameters.
    """

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = kernel

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-20)
    ).to(device)
    likelihood.noise = noise
    likelihood.raw_noise.requires_grad_(False)  # noise is given, not fitted
    model = ExactGPModel(X_train, y_train, likelihood).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    pbar = tqdm(range(num_iters), disable=not verbose, desc="fitting kernel")
    for i in pbar:
        optimizer.zero_grad()
        loss = -mll(model(X_train), y_train)  # negative log marginal likelihood
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            pbar.set_postfix(neg_mll=f"{loss.item():.3f}")

    if verbose:
        # Not every kernel exposes base_kernel/outputscale (e.g. a bare RBFKernel),
        # so report whichever hyperparameters this one actually has.
        for name, obj in [
            ("lengthscale", getattr(kernel, "base_kernel", None)),
            ("outputscale", kernel),
        ]:
            if obj is not None and hasattr(obj, name):
                logger.info("%s is %s", name, getattr(obj, name))

    model.eval()
    likelihood.eval()
    return kernel


class AbstractBasis(ABC):
    """A finite basis expansion of a GP.

    A basis represents draws as ``f(x) = sum_r w_r phi_r(x) + m(x)`` with iid standard
    normal weights ``w``, so the induced covariance is ``Phi(x)^T Phi(x)``. That lets
    the diffusion run over ``R`` weights instead of ``N`` grid values — the point of
    the weight-space formulation.

    The basis is **grid-free**: it holds only what defines the expansion (``R``, the
    kernel, the mean), and every method takes the points to evaluate at. The grid
    belongs to whoever is using the basis —
    :class:`~flow_inference.flows.BasisGaussianProcessFlow` evaluates ``Phi`` once in
    its constructor and stores it, the way ``GaussianProcessFlow`` stores its whitening
    factor.

    Subclasses supply :meth:`_get_basis_params` (whatever the basis precomputes once,
    independent of any grid) and :meth:`phi` (the ``(R, N)`` design matrix at given
    points). Everything downstream follows from ``phi``.

    :meth:`compute_derivative` is *optional*. A Fourier basis (:class:`RFFBasis`) has
    it in closed form; a basis defined only by values at sample points — an
    eigen/Nystrom expansion, say — does not, so the default raises.

    Args:
        R (int): Number of basis functions.
        input_dim (int): Dimension ``D`` of the input space. Needed at construction
            because the expansion itself depends on it (RFF frequencies are
            ``(D, R/2)``), even though no grid is fixed yet.
        mean (gpytorch.means.Mean): Mean function, added back in
            :meth:`sample_functions`. Stored as ``mean_module``; call it through
            :meth:`mean`, which coerces its argument like :meth:`phi` does.
        kernel (gpytorch.kernels.Kernel): The kernel being approximated.
        device (torch.device or str): Device for the stored tensors.
        dtype (torch.dtype): Working precision. Defaults to the kernel's dtype.
    """

    def __init__(self, R, input_dim, mean, kernel, device="cpu", dtype=None):
        self.device = device
        # Follow the kernel unless overridden. The mean and kernel are gpytorch
        # modules the caller built, and calling one at a dtype it was not built at
        # raises inside gpytorch's sq_dist rather than promoting, so pinning a
        # dtype here would break any caller who did not also set the ambient default.
        self.dtype = kernel.dtype if dtype is None else dtype
        self.R = R
        self.input_dim = input_dim
        self.mean_module = mean
        self.kernel = kernel

        # grid-independent precomputation (RFF frequencies, and so on)
        self._get_basis_params()

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """Prior/posterior mean at ``x`` ``(N, D)``, shape ``(N,)``.

        A method rather than the bare gpytorch module, so that ``basis.mean(x)``
        coerces ``x`` exactly like :meth:`phi` does. Called directly, a
        ``gpytorch.means.Mean`` raises on any dtype it was not built at instead of
        promoting — the grid is float64 by default while a kernel built under the
        ambient default is float32, so the raw module fails on the common path.
        """
        return self.mean_module(self._as_points(x)).reshape(-1).to(dtype=self.dtype)

    def _as_points(self, x: torch.Tensor) -> torch.Tensor:
        """Coerce evaluation points to ``(N, D)`` in the basis's device/dtype."""
        x = x.to(device=self.device, dtype=self.dtype)
        return x.unsqueeze(-1) if x.ndim == 1 else x

    @abstractmethod
    def _get_basis_params(self) -> None:
        """Precompute whatever the basis needs, independent of any evaluation grid."""

    @abstractmethod
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Design matrix at ``x`` ``(N, D)``, shape ``(R, N)``."""

    def approximate_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Kernel matrix induced by the basis at ``x``: ``Phi^T Phi``, shape ``(N, N)``.

        How close this is to ``kernel(x, x)`` is what ``R`` buys.
        """
        Phi = self.phi(x)
        return torch.einsum("rn,rm->nm", Phi, Phi)

    def compute_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """Mixed partial derivatives of the basis functions at ``x``, shape ``(R, N)``.

        Optional: only bases whose basis functions have a closed form can provide
        this. Bases defined pointwise cannot.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form derivatives. Use a basis with "
            "analytic basis functions (e.g. RFFBasis), or difference the field."
        )

    def mean_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """Mixed partials of the mean function at ``x``, shape ``(N,)``.

        A sampled function is ``w @ phi(x) + mean(x)``, so a derivative constraint
        needs **both** terms::

            df = w @ basis.compute_derivative(x, dims, orders) \
                 + basis.mean_derivative(x, dims, orders)

        This exists on the base class so the second term can never be silently
        forgotten: it returns zeros for a ``ZeroMean`` (the common case, where dropping
        it happens to be harmless) and raises otherwise, rather than leaving callers to
        notice that their mean has a derivative. :class:`DecoupledRFFBasis` overrides it
        — a posterior mean's derivative is large, not negligible.
        """
        x = self._as_points(x)
        if sum(orders) == 0:
            return self.mean(x)
        if isinstance(self.mean_module, gpytorch.means.ZeroMean):
            return torch.zeros(x.shape[0], dtype=self.dtype, device=self.device)
        raise NotImplementedError(
            f"{type(self).__name__} has a {type(self.mean_module).__name__} mean, whose "
            "derivative is not implemented. A sampled function is w @ phi(x) + "
            "mean(x), so a derivative constraint needs the mean's derivative too — "
            "dropping it silently changes the field."
        )

    def sample_functions(self, weight_samples: torch.Tensor, x: torch.Tensor):
        """Map weights ``(B, C, R)`` to function values ``(B, C, N)`` at ``x``.

        Standard normal weights give prior draws under the basis's approximation; the
        mean function is added back here.
        """
        x = self._as_points(x)
        f = weight_samples @ self.phi(x)  # (B, C, R) @ (R, N)
        return (f + self.mean(x)).detach()


class RFFBasis(AbstractBasis):
    """Random Fourier feature (RFF) basis for a scaled RBF kernel.

    Expands ``k(x, x') ~= Phi(x)^T Phi(x')`` with ``Phi`` built from ``R/2`` random
    frequencies drawn once at construction, so the same basis can be evaluated at any
    points. The basis functions are sinusoids, so their derivatives are available in
    closed form (:meth:`compute_derivative`) — the reason to prefer this over a basis
    defined pointwise when the constraint involves derivatives.
    """

    def _get_basis_params(self):
        """Draw the RFF frequencies. Independent of any evaluation grid.

        Only the *draw* is fixed here; the lengthscale and outputscale are read off the
        kernel (see the properties below), so they can never go stale.
        """
        assert isinstance(self.kernel, gpytorch.kernels.scale_kernel.ScaleKernel)
        assert isinstance(
            self.kernel.base_kernel, gpytorch.kernels.rbf_kernel.RBFKernel
        )
        assert self.R % 2 == 0, "RFF basis requires an even number of features R"

        self.D = self.input_dim
        # gpytorch's RFFKernel owns the frequency draw; phi evaluates the features in
        # closed form (the tests check it against gpytorch's own featurisation).
        rff = ScaleKernel(
            RFFKernel(num_samples=self.R // 2, ard_num_dims=self.D, num_dims=self.D)
        )
        self.omega = rff.base_kernel.randn_weights.to(
            dtype=self.dtype, device=self.device
        )  # (D, R/2)

    @property
    def lengthscale(self) -> torch.Tensor:
        """``(D,)`` lengthscale, read off the kernel."""
        ls = self.kernel.base_kernel.lengthscale.reshape(-1).to(self.device, self.dtype)
        return ls.expand(self.D) if ls.numel() == 1 else ls

    @property
    def outputscale(self) -> torch.Tensor:
        """Kernel outputscale, read off the kernel."""
        return self.kernel.outputscale.to(self.device, self.dtype)

    @property
    def omega_eff(self) -> torch.Tensor:
        """Frequencies rescaled by the lengthscale — what the derivatives bring down."""
        return self.omega / self.lengthscale[:, None]  # (D, R/2)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """RFF design matrix at ``x`` ``(N, D)``, shape ``(R, N)``.

        ``[cos(omega_eff x), sin(omega_eff x)]``, normalised by the number of features
        and scaled by the outputscale, so ``Phi^T Phi`` approximates ``k``.
        """
        arg = torch.einsum("dr,nd->rn", self.omega_eff, self._as_points(x))  # (R/2, N)
        feats = torch.cat([arg.cos(), arg.sin()], dim=0)  # (R, N)
        return torch.sqrt(self.outputscale) / math.sqrt(self.R / 2) * feats

    def compute_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """Mixed partial derivatives of the basis functions at ``x``, shape ``(R, N)``.

        Each dimension in ``dims`` is differentiated to the corresponding order in
        ``orders``.

        Closed form, O(R*N): differentiating the features with nested autograd gives
        the same values but materialises an (N, R, D, ...) Jacobian that does not fit
        in memory at realistic grid sizes.

        Raises:
            ValueError: If ``dims`` and ``orders`` differ in length.
        """
        x = self._as_points(x)
        total = sum(orders)
        if total == 0:
            return self.phi(x)

        # d^n/dx^n of cos and sin cycle with period 4, so the sign and which function
        # appears follow from total % 4 and total % 2.
        cos_sign = 1 if total % 4 == 0 or total % 4 == 3 else -1
        cos_op = torch.cos if total % 2 == 0 else torch.sin
        sin_sign = 1 if total % 4 == 0 or total % 4 == 1 else -1
        sin_op = torch.sin if total % 2 == 0 else torch.cos

        # each derivative brings down a factor of the (lengthscale-scaled) frequency
        prefactor = torch.ones(self.R // 2, device=self.device, dtype=self.dtype)
        for dim, order in zip(dims, orders, strict=True):
            prefactor *= self.omega_eff[dim, :] ** order
        prefactor = torch.cat([prefactor, prefactor], dim=0)  # (R,)

        arg = torch.einsum("dr,nd->rn", self.omega_eff, x)  # (R/2, N)
        all_terms = torch.cat([cos_sign * cos_op(arg), sin_sign * sin_op(arg)], dim=0)

        return (
            torch.sqrt(self.outputscale)
            * (1 / math.sqrt(self.R / 2))
            * prefactor[:, None]
            * all_terms
        )  # (R, N)


def _hermite(u: torch.Tensor, n: int) -> torch.Tensor:
    """Probabilists' Hermite polynomial ``He_n`` evaluated elementwise.

    ``He_0 = 1``, ``He_1 = u``, ``He_{n+1}(u) = u He_n(u) - n He_{n-1}(u)``. These are
    what derivatives of a Gaussian produce, which is why they show up when
    differentiating an RBF kernel (see
    :meth:`DecoupledRFFBasis._prior_kernel_derivative`).
    """
    if n == 0:
        return torch.ones_like(u)
    h_prev, h = torch.ones_like(u), u  # He_0, He_1
    for k in range(1, n):
        h_prev, h = h, u * h - k * h_prev
    return h


class DecoupledRFFBasis(AbstractBasis):
    r"""Pathwise-conditioned RFF basis — a GP *posterior* as a finite basis.

    The conditioned analogue of :class:`PosteriorKernel`. A posterior kernel cannot be
    expanded with random Fourier features directly: RFF rests on Bochner's theorem,
    which needs stationarity, and conditioning destroys it. Pathwise conditioning
    (Wilson et al., *Efficiently Sampling Functions from Gaussian Process Posteriors*,
    ICML 2020) sidesteps that by only ever approximating the **prior**, which is
    stationary, and correcting it exactly::

        (f | y)(.) = Phi(.)^T w  +  k(., X) [K + s2 I]^-1 (y - Phi(X)^T w - eps)
                     \_ RFF prior _/     \_ exact rank-M update in the canonical basis _/

    Collecting terms makes this affine in the random variables, which is what lets it
    be a basis at all::

        f = [ Phi_* - Phi_X A K_X* | -s * A K_X* ]^T [w; eps]  +  m_post
            \________________ Psi, shape (R + M, N) ________/       w, eps ~ N(0, I)

    So :attr:`R` is ``num_features + M`` (M = number of training points): the ``eps``
    block carries the observation noise and is dropped when ``noise == 0``. The mean is
    the exact posterior mean, so :meth:`sample_functions` needs no correction.

    Accuracy, on a grid extending beyond the data: the mean is exact to machine
    precision, and the covariance tracks the true posterior far more closely than
    conditioning the RFF model in weight space, which under-covers away from the data
    (the "variance starvation" the paper is about).

    :meth:`compute_derivative` and :meth:`mean_derivative` are both closed form, so
    this basis can carry derivative constraints (a PDE residual, say) exactly.

    Args:
        num_features (int): RFF features ``R_f`` for the prior. Must be even.
        input_dim (int): input dimension ``D``.
        kernel: the *prior* kernel; must be a ``ScaleKernel(RBFKernel)`` as
            :class:`RFFBasis` requires.
        train_x: ``(M, D)`` training inputs.
        train_y: ``(M,)`` training targets.
        noise: observation noise variance.
        prior_mean: prior mean function; defaults to ``ZeroMean``.
    """

    def __init__(
        self,
        num_features: int,
        input_dim: int,
        kernel: gpytorch.kernels.Kernel,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        noise: float | torch.Tensor = 1e-4,
        prior_mean: gpytorch.means.Mean | None = None,
        device="cpu",
        dtype=None,
    ):
        dtype = kernel.dtype if dtype is None else dtype
        self.train_x = train_x.to(device=device, dtype=dtype)
        self.train_y = train_y.to(device=device, dtype=dtype)
        self.noise = float(noise)
        # sample_functions adds mean(x); for a posterior basis that is the posterior
        # mean, so the deterministic part of Matheron's rule is handled there.
        mean = PosteriorMean(kernel, self.train_x, self.train_y, noise, prior_mean)
        super().__init__(num_features, input_dim, mean, kernel, device, dtype)

    def _get_basis_params(self) -> None:
        # The prior is stationary, so RFF is legitimate here — this is the whole point.
        self.num_features = self.R
        self.prior_basis = RFFBasis(
            R=self.num_features,
            input_dim=self.input_dim,
            mean=gpytorch.means.ZeroMean(),
            kernel=self.kernel,
            device=self.device,
            dtype=self.dtype,
        )
        self.M = self.train_x.shape[0]

        # A = [K_XX + s2 I]^-1, applied via the Cholesky factor.
        L = _train_cholesky(self.kernel, self.train_x, self.noise)
        eye = torch.eye(self.M, dtype=self.dtype, device=self.device)
        self.register_A = torch.cholesky_solve(eye, L)  # (M, M)
        self.Phi_X = self.prior_basis.phi(self.train_x)  # (R_f, M)

        # Total latent width: the eps block is degenerate when there is no noise, so
        # it is dropped rather than diffused over to no effect.
        self.has_noise_block = self.noise > 0.0
        self.R = self.num_features + (self.M if self.has_noise_block else 0)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Conditioned design matrix at ``x``, shape ``(R, N)``.

        Rows are the ``R_f`` corrected prior features followed, when there is
        observation noise, by the ``M`` noise features.
        """
        x = self._as_points(x)
        Phi_x = self.prior_basis.phi(x)  # (R_f, N)
        K_xX = self.kernel(x, self.train_x).to_dense()  # (N, M)
        W = K_xX @ self.register_A  # (N, M)

        blocks = [Phi_x - (W @ self.Phi_X.T).T]  # (R_f, N)
        if self.has_noise_block:
            blocks.append(-(self.noise**0.5) * W.T)  # (M, N)
        return torch.cat(blocks, dim=0)  # (R, N)

    def _prior_kernel_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """``d^n/dx^n k(x, X)`` for the prior kernel, shape ``(N, M)``.

        The only new quantity the conditioned basis needs: A and Phi_X are constants,
        so every derivative of Psi is built from this block.

        For a scaled ARD RBF with ``u_d = (x_d - x'_d) / l_d`` the kernel is separable
        and Gaussian, so mixed partials factorise into Hermite polynomials::

            d^{n_1}_{d_1} d^{n_2}_{d_2} ... k(x, x')
                = k(x, x') * prod_d (-1 / l_d)^{n_d} He_{n_d}(u_d)

        (``n=1`` gives ``-k u / l``; ``n=2`` gives ``k (u^2 - 1) / l^2``.) This assumes
        a ``ScaleKernel(RBFKernel)``, which :class:`RFFBasis` already asserts.
        """
        ls = self.prior_basis.lengthscale  # (D,)
        diff = x[:, None, :] - self.train_x[None, :, :]  # (N, M, D)
        u = diff / ls  # (N, M, D)
        K = self.prior_basis.outputscale * torch.exp(-0.5 * (u**2).sum(-1))  # (N, M)

        factor = torch.ones_like(K)
        for d, n in zip(dims, orders, strict=True):
            factor = factor * ((-1.0 / ls[d]) ** n) * _hermite(u[..., d], n)
        return K * factor

    def compute_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """Mixed partials of the conditioned basis functions at ``x``, shape ``(R, N)``.

        Differentiating the pathwise form term by term, with ``A`` and ``Phi_X``
        constant::

            d Psi = [ d Phi(x) - Phi_X (d k(x, X) A)^T ;  -sigma (d k(x, X) A)^T ]

        so the prior block reuses :meth:`RFFBasis.compute_derivative` and the update
        block reuses :meth:`_prior_kernel_derivative`. Both are closed form.

        This is the derivative of the *basis functions* only. Samples also carry the
        posterior mean, whose derivative is non-zero here — add
        :meth:`mean_derivative` to get the derivative of a sampled function.
        """
        x = self._as_points(x)
        if sum(orders) == 0:
            return self.phi(x)

        dPhi = self.prior_basis.compute_derivative(x, dims, orders)  # (R_f, N)
        dW = self._prior_kernel_derivative(x, dims, orders) @ self.register_A  # (N, M)

        blocks = [dPhi - (dW @ self.Phi_X.T).T]  # (R_f, N)
        if self.has_noise_block:
            blocks.append(-(self.noise**0.5) * dW.T)  # (M, N)
        return torch.cat(blocks, dim=0)  # (R, N)

    def mean_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """Mixed partials of the posterior mean at ``x``, shape ``(N,)``.

        ``m_post = m_prior + k(x, X) alpha``, so with a constant prior mean the
        derivative is ``d k(x, X) alpha`` — the same block :meth:`compute_derivative`
        uses.

        A sampled function is ``w @ phi(x) + mean(x)``, so its derivative is
        ``w @ compute_derivative(...) + mean_derivative(...)``. Forgetting the second
        term silently drops the data-driven part of the field, which for a PDE residual
        is usually most of it.
        """
        x = self._as_points(x)
        if sum(orders) == 0:
            return self.mean(x)
        if not isinstance(self.mean_module.prior_mean, gpytorch.means.ZeroMean):
            raise NotImplementedError(
                "mean_derivative assumes a ZeroMean prior; a non-constant prior mean "
                "would contribute its own derivative term."
            )
        dK = self._prior_kernel_derivative(x, dims, orders)  # (N, M)
        return dK @ self.mean_module.alpha  # (N,)
