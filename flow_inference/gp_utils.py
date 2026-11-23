"""Gaussian-process helpers built on gpytorch.

A typical GP workflow fits the kernel, conditions on data, and hands the posterior to
a flow::

    kernel = fit_kernel_hyperparameters(X, y, kernel, noise=1e-4)
    mean, post_kernel = gp_posterior(kernel, X, y, noise=1e-4)
    flow = GaussianProcessFlow(post_kernel, mean, grid)

* :func:`fit_kernel_hyperparameters`: type-II maximum likelihood for a kernel's
  hyperparameters.
* :func:`gp_posterior`: the posterior mean and kernel, as :class:`PosteriorMean` and
  :class:`PosteriorKernel`. These are an ordinary ``gpytorch.means.Mean`` and
  ``gpytorch.kernels.Kernel``, so the posterior can be evaluated at any inputs.
* :class:`AbstractBasis`: finite basis expansions of a GP, so that
  :class:`~flow_inference.flows.BasisGaussianProcessFlow` can diffuse over ``R``
  weights rather than ``N`` grid values. :class:`RFFBasis` expands a stationary prior
  with closed-form derivatives; :class:`DecoupledRFFBasis` does the same for a GP
  posterior, via pathwise conditioning (Wilson et al. 2020).

Factorisations of the training Gram matrix are done in float64, since it is badly
conditioned; everything else runs in the kernel's dtype.
"""

import logging
import math
from abc import ABC, abstractmethod

import gpytorch
import torch
from gpytorch.kernels import RFFKernel, ScaleKernel
from tqdm.auto import tqdm

from flow_inference.flow_utils import float64_copy

logger = logging.getLogger(__name__)


def _train_cholesky(
    base_kernel: gpytorch.kernels.Kernel,
    train_x: torch.Tensor,
    noise: float | torch.Tensor,
) -> torch.Tensor:
    """Lower Cholesky factor of the training Gram matrix ``K_XX + noise I``.

    The one expensive step in conditioning a GP, shared by :class:`PosteriorKernel`,
    :class:`PosteriorMean` and :class:`DecoupledRFFBasis`. Computed in float64: with
    near-noiseless observations (noise ~1e-10) the matrix is not numerically positive
    definite in float32.
    """
    x = train_x.detach().to(torch.float64)
    eye = torch.eye(x.shape[0], dtype=torch.float64, device=x.device)
    with torch.no_grad():  # a fixed factor: nothing to differentiate
        K_xx = float64_copy(base_kernel)(x, x).to_dense() + noise * eye
        return torch.linalg.cholesky(K_xx)


class PosteriorKernel(gpytorch.kernels.Kernel):
    """GP posterior covariance as a gpytorch kernel.

    Computes

        k_post(x, x') = k(x, x') - k(x, X) [K_XX + noise I]^-1 k(X, x')

    at any inputs, so the posterior can be passed anywhere a kernel is expected, such
    as :class:`~flow_inference.flows.GaussianProcessFlow`. (gpytorch itself gives the
    posterior only as a ``MultivariateNormal`` over fixed test points.) Usually built
    together with :class:`PosteriorMean` by :func:`gp_posterior`.

    The Cholesky factor of ``K_XX + noise I`` is computed once, in float64, so each
    evaluation is a triangular solve; the solve is also float64, and the kernel
    itself is evaluated in the inputs' dtype.

    The posterior kernel is **not stationary**: its variance collapses near the
    training inputs. Random Fourier features (:class:`RFFBasis`) therefore do not
    apply to it; use :class:`DecoupledRFFBasis` for a finite basis of the posterior.

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
        # _L lets gp_posterior pass in a factor it has already computed.
        L = _train_cholesky(base_kernel, train_x, noise) if _L is None else _L
        self.register_buffer("L", L)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params
    ) -> torch.Tensor:
        """Posterior covariance ``(N1, N2)`` between ``x1`` and ``x2``, or its
        diagonal ``(N,)`` when ``diag``."""
        train_x = self.train_x.to(x1.dtype)
        k12 = self.base_kernel(x1, x2).to_dense()
        k1X = self.base_kernel(x1, train_x).to_dense()
        kX2 = self.base_kernel(train_x, x2).to_dense()
        # Only the solve needs float64: it inherits K_XX's conditioning.
        L = self.L.to(torch.float64)
        correction = k1X.to(L.dtype) @ torch.cholesky_solve(kX2.to(L.dtype), L)
        out = k12 - correction.to(k12.dtype)
        return out.diagonal(dim1=-2, dim2=-1) if diag else out


class PosteriorMean(gpytorch.means.Mean):
    """GP posterior mean as a gpytorch mean function.

    Computes

        m_post(x) = m(x) + k(x, X) [K_XX + noise I]^-1 (y - m(X))

    at any inputs, so the posterior mean can be passed anywhere a
    ``gpytorch.means.Mean`` is expected. The companion to :class:`PosteriorKernel`;
    :func:`gp_posterior` builds both.

    The weights ``alpha = [K_XX + noise I]^-1 (y - m(X))`` are solved once, in
    float64, so each evaluation is a single matmul.

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
        # _L lets gp_posterior pass in a factor it has already computed.
        L = _train_cholesky(base_kernel, train_x, noise) if _L is None else _L
        L = L.detach().to(torch.float64)
        x64 = train_x.detach().to(torch.float64)
        with torch.no_grad():
            prior_at_x = float64_copy(self.prior_mean)(x64).to(torch.float64)
            residual = (train_y.detach().to(torch.float64) - prior_at_x).unsqueeze(-1)
            alpha = torch.cholesky_solve(residual, L).squeeze(-1)  # (N,)
        self.register_buffer("alpha", alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior mean ``(M,)`` at ``x`` ``(M, D)``."""
        k_xX = self.base_kernel(x, self.train_x.to(x.dtype)).to_dense()  # (M, N)
        # alpha is applied in float64: it inherits K_XX's conditioning.
        alpha = self.alpha.to(torch.float64)
        return self.prior_mean(x) + (k_xX.to(alpha.dtype) @ alpha).to(k_xX.dtype)


def gp_posterior(
    base_kernel: gpytorch.kernels.Kernel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    noise: float | torch.Tensor = 1e-4,
    prior_mean: gpytorch.means.Mean | None = None,
) -> tuple["PosteriorMean", "PosteriorKernel"]:
    """GP posterior mean and kernel, conditioned on ``(train_x, train_y)``.

    Builds :class:`PosteriorMean` and :class:`PosteriorKernel` with a single shared
    Cholesky factorisation of ``K_XX + noise I``, the only O(N^3) step.

    Args:
        base_kernel: the prior kernel k.
        train_x: training inputs ``(N, D)``.
        train_y: training targets ``(N,)``.
        noise: observation noise variance.
        prior_mean: prior mean function. Defaults to ``ZeroMean``.

    Returns:
        ``(mean, kernel)``, ready to pass to
        :class:`~flow_inference.flows.GaussianProcessFlow`.

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
    hyperparameters (lengthscale, outputscale, ...) are optimised. ``kernel`` is
    fitted **in place** and also returned. Conditioning on the data is a separate
    step, :func:`gp_posterior`:

        >>> kernel = fit_kernel_hyperparameters(X, y, kernel, noise=1e-4)
        >>> mean, post_kernel = gp_posterior(kernel, X, y, noise=1e-4)
        >>> flow = GaussianProcessFlow(post_kernel, mean, grid)

    The optimisation runs on a float64 copy of the kernel, since each step factorises
    ``K_XX + noise I``; the fitted values are then loaded back into ``kernel``.

    Args:
        X_train: ``(N, D)`` training inputs.
        y_train: ``(N,)`` training targets.
        kernel: the kernel to fit; modified in place.
        noise: observation noise variance, held fixed during fitting.
        num_iters: Adam steps on the negative log marginal likelihood.
        lr: Adam learning rate.
        device: device to fit on.
        verbose: show a progress bar with the loss, and log the fitted
            hyperparameters.

    Returns:
        The same ``kernel`` object, with fitted hyperparameters.
    """
    fit_kernel = float64_copy(kernel)
    X_train = X_train.detach().to(device=device, dtype=torch.float64)
    y_train = y_train.detach().to(device=device, dtype=torch.float64)

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = fit_kernel

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-20)
    ).to(device=device, dtype=torch.float64)
    likelihood.noise = noise
    likelihood.raw_noise.requires_grad_(False)  # the noise is given, not fitted
    model = ExactGPModel(X_train, y_train, likelihood).to(
        device=device, dtype=torch.float64
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    pbar = tqdm(range(num_iters), disable=not verbose, desc="fitting kernel")
    for i in pbar:
        optimizer.zero_grad()
        loss = -mll(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            pbar.set_postfix(neg_mll=f"{loss.item():.3f}")

    model.eval()
    likelihood.eval()
    # load_state_dict casts the fitted values to the caller's kernel's dtype.
    kernel.load_state_dict(fit_kernel.state_dict())

    if verbose:
        # Not every kernel has base_kernel/outputscale (e.g. a bare RBFKernel).
        for name, obj in [
            ("lengthscale", getattr(kernel, "base_kernel", None)),
            ("outputscale", kernel),
        ]:
            if obj is not None and hasattr(obj, name):
                logger.info("%s is %s", name, getattr(obj, name))

    return kernel


class AbstractBasis(ABC):
    """A finite basis expansion of a GP.

    A basis writes draws as ``f(x) = sum_r w_r phi_r(x) + m(x)`` with iid standard
    normal weights ``w``, so the induced covariance is ``Phi(x)^T Phi(x)``. This lets
    :class:`~flow_inference.flows.BasisGaussianProcessFlow` diffuse over ``R`` weights
    rather than ``N`` grid values.

    A basis is not tied to a grid: it holds only what defines the expansion (``R``,
    the kernel, the mean), and every method takes the points to evaluate at.

    Subclasses implement :meth:`_get_basis_params` (any grid-independent
    precomputation) and :meth:`phi` (the ``(R, N)`` design matrix at given points).
    They may also implement :meth:`compute_derivative`, when the basis functions have
    closed-form derivatives (as for :class:`RFFBasis`); the default raises.

    Args:
        R (int): number of basis functions.
        input_dim (int): dimension ``D`` of the input space.
        mean (gpytorch.means.Mean): mean function, added in
            :meth:`sample_functions`. Stored as ``mean_module``; evaluate it with
            :meth:`mean`.
        kernel (gpytorch.kernels.Kernel): the kernel being approximated.
        device (torch.device or str): device for the stored tensors.
        dtype (torch.dtype): working precision. Defaults to the kernel's dtype.
    """

    def __init__(self, R, input_dim, mean, kernel, device="cpu", dtype=None):
        self.device = device
        # Default to the kernel's dtype: gpytorch modules raise, rather than promote,
        # when called at a dtype they were not built at.
        self.dtype = kernel.dtype if dtype is None else dtype
        self.R = R
        self.input_dim = input_dim
        self.mean_module = mean
        self.kernel = kernel
        self._get_basis_params()

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """Mean function at ``x`` ``(N, D)``, shape ``(N,)``.

        Coerces ``x`` to the basis's device and dtype first, as :meth:`phi` does;
        the raw gpytorch module raises on a dtype it was not built at.
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

        Approaches ``kernel(x, x)`` as ``R`` grows.
        """
        Phi = self.phi(x)
        return torch.einsum("rn,rm->nm", Phi, Phi)

    def compute_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """Mixed partial derivatives of the basis functions at ``x``, shape ``(R, N)``.

        Only available for bases whose basis functions have a closed form; this
        default raises.

        Args:
            x: evaluation points ``(N, D)``.
            dims: the input dimensions to differentiate along.
            orders: the derivative order for each entry of ``dims``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form derivatives. Use a basis with "
            "analytic basis functions (e.g. RFFBasis), or difference the field."
        )

    def mean_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """Mixed partials of the mean function at ``x``, shape ``(N,)``.

        A sampled function is ``w @ phi(x) + mean(x)``, so its derivative needs both
        terms::

            df = w @ basis.compute_derivative(x, dims, orders) \
                 + basis.mean_derivative(x, dims, orders)

        This default returns zeros for a ``ZeroMean`` and raises for any other mean,
        so a non-zero mean derivative is never silently dropped.
        :class:`DecoupledRFFBasis` overrides it for its posterior mean.

        Args:
            x: evaluation points ``(N, D)``.
            dims: the input dimensions to differentiate along.
            orders: the derivative order for each entry of ``dims``.
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

        Computes ``w @ phi(x) + mean(x)``, so standard normal weights give draws of
        the GP under the basis approximation.
        """
        x = self._as_points(x)
        f = weight_samples @ self.phi(x)  # (B, C, R) @ (R, N)
        return (f + self.mean(x)).detach()


class RFFBasis(AbstractBasis):
    """Random Fourier feature (RFF) basis for a scaled RBF kernel.

    Approximates ``k(x, x') ~= Phi(x)^T Phi(x')`` with sinusoids at ``R/2`` random
    frequencies, drawn once at construction. The basis functions have closed-form
    derivatives (:meth:`compute_derivative`), so conditions can involve derivatives
    of the field, such as a PDE residual.

    Takes the arguments of :class:`AbstractBasis`. The kernel must be a
    ``ScaleKernel(RBFKernel)`` and ``R`` must be even.

    Example:
        >>> basis = RFFBasis(R=256, input_dim=2, mean=ZeroMean(), kernel=kernel)
        >>> f = basis.sample_functions(torch.randn(8, 1, 256), grid)  # (8, 1, N)
    """

    def _get_basis_params(self):
        """Draw the RFF frequencies.

        Only the draw is fixed here; the lengthscale and outputscale are read off the
        kernel when needed, so they track any later change to it.
        """
        assert isinstance(self.kernel, gpytorch.kernels.scale_kernel.ScaleKernel)
        assert isinstance(
            self.kernel.base_kernel, gpytorch.kernels.rbf_kernel.RBFKernel
        )
        assert self.R % 2 == 0, "RFF basis requires an even number of features R"

        self.D = self.input_dim
        # gpytorch's RFFKernel draws the frequencies; phi evaluates the features itself.
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
        """Frequencies divided by the lengthscale, ``(D, R/2)``."""
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

        Closed form, at O(R N) cost; nested autograd would materialise a Jacobian too
        large for realistic grids.

        Args:
            x: evaluation points ``(N, D)``.
            dims: the input dimensions to differentiate along.
            orders: the derivative order for each entry of ``dims``.

        Raises:
            ValueError: if ``dims`` and ``orders`` differ in length.
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

        # Each derivative brings down a factor of the (lengthscale-scaled) frequency.
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

    ``He_0 = 1``, ``He_1 = u``, ``He_{n+1}(u) = u He_n(u) - n He_{n-1}(u)``. They
    appear in the derivatives of a Gaussian, and hence of an RBF kernel (see
    :meth:`DecoupledRFFBasis._prior_kernel_derivative`).
    """
    if n == 0:
        return torch.ones_like(u)
    h_prev, h = torch.ones_like(u), u  # He_0, He_1
    for k in range(1, n):
        h_prev, h = h, u * h - k * h_prev
    return h


class DecoupledRFFBasis(AbstractBasis):
    r"""A GP posterior as a finite basis, by pathwise conditioning of an RFF prior.

    The basis counterpart of :class:`PosteriorKernel`. Random Fourier features need a
    stationary kernel, which a posterior is not, so :class:`RFFBasis` cannot expand
    it directly. Pathwise conditioning (Wilson et al., *Efficiently Sampling Functions
    from Gaussian Process Posteriors*, ICML 2020) approximates only the stationary
    **prior** with RFF and corrects it for the data exactly::

        (f | y)(.) = Phi(.)^T w  +  k(., X) [K + s2 I]^-1 (y - Phi(X)^T w - eps)
                     \_ RFF prior _/     \_ exact rank-M update in the canonical basis _/

    This is affine in the random variables, so collecting terms gives a basis::

        f = [ Phi_* - Phi_X A K_X* | -s * A K_X* ]^T [w; eps]  +  m_post
            \________________ Psi, shape (R + M, N) ________/       w, eps ~ N(0, I)

    So :attr:`R` is ``num_features + M``, with ``M`` the number of training points;
    the ``eps`` block carries the observation noise and is dropped when
    ``noise == 0``. The mean is the exact posterior mean (a :class:`PosteriorMean`).

    The mean is exact, and the covariance is far closer to the true posterior than
    conditioning the RFF weights directly, which under-estimates the variance away
    from the data. :meth:`compute_derivative` and :meth:`mean_derivative` are both
    closed form, so derivative conditions (a PDE residual, say) are exact too.

    Args:
        num_features (int): RFF features ``R_f`` for the prior. Must be even.
        input_dim (int): input dimension ``D``.
        kernel: the *prior* kernel, a ``ScaleKernel(RBFKernel)``.
        train_x: ``(M, D)`` training inputs.
        train_y: ``(M,)`` training targets.
        noise: observation noise variance.
        prior_mean: prior mean function. Defaults to ``ZeroMean``.
        device: device for the stored tensors.
        dtype: working precision. Defaults to the kernel's dtype.
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
        # The posterior mean is the deterministic part of the pathwise update; phi
        # holds the random part.
        mean = PosteriorMean(kernel, self.train_x, self.train_y, noise, prior_mean)
        super().__init__(num_features, input_dim, mean, kernel, device, dtype)

    def _get_basis_params(self) -> None:
        """Build the RFF prior basis and the solve against the training data."""
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

        # A = [K_XX + s2 I]^-1, kept in float64 (it inherits K_XX's conditioning) and
        # applied via _apply_A.
        L = _train_cholesky(self.kernel, self.train_x, self.noise)
        eye = torch.eye(self.M, dtype=torch.float64, device=self.device)
        self.register_A = torch.cholesky_solve(eye, L)  # (M, M)
        self.Phi_X = self.prior_basis.phi(self.train_x)  # (R_f, M)

        # Without noise the eps block is identically zero, so it is dropped.
        self.has_noise_block = self.noise > 0.0
        self.R = self.num_features + (self.M if self.has_noise_block else 0)

    def _apply_A(self, K: torch.Tensor) -> torch.Tensor:
        """``K @ [K_XX + s2 I]^-1``, computed in float64 and returned in the basis
        dtype."""
        return (K.to(torch.float64) @ self.register_A).to(self.dtype)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Conditioned design matrix at ``x``, shape ``(R, N)``.

        Rows are the ``R_f`` corrected prior features followed, when there is
        observation noise, by the ``M`` noise features.
        """
        x = self._as_points(x)
        Phi_x = self.prior_basis.phi(x)  # (R_f, N)
        K_xX = self.kernel(x, self.train_x).to_dense()  # (N, M)
        W = self._apply_A(K_xX)  # (N, M)

        blocks = [Phi_x - (W @ self.Phi_X.T).T]  # (R_f, N)
        if self.has_noise_block:
            blocks.append(-(self.noise**0.5) * W.T)  # (M, N)
        return torch.cat(blocks, dim=0)  # (R, N)

    def _prior_kernel_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """``d^n/dx^n k(x, X)`` for the prior kernel, shape ``(N, M)``.

        ``A`` and ``Phi_X`` are constants, so this block and the prior basis's
        derivatives are all :meth:`compute_derivative` needs.

        For a scaled ARD RBF with ``u_d = (x_d - x'_d) / l_d`` the kernel is separable
        and Gaussian, so mixed partials factorise into Hermite polynomials::

            d^{n_1}_{d_1} d^{n_2}_{d_2} ... k(x, x')
                = k(x, x') * prod_d (-1 / l_d)^{n_d} He_{n_d}(u_d)

        (``n=1`` gives ``-k u / l``; ``n=2`` gives ``k (u^2 - 1) / l^2``.) This assumes
        a ``ScaleKernel(RBFKernel)``, which :class:`RFFBasis` checks.
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

        so the prior block uses :meth:`RFFBasis.compute_derivative` and the update
        block uses :meth:`_prior_kernel_derivative`. Both are closed form.

        This is the derivative of the basis functions only; add
        :meth:`mean_derivative` for the derivative of a sampled function.

        Args:
            x: evaluation points ``(N, D)``.
            dims: the input dimensions to differentiate along.
            orders: the derivative order for each entry of ``dims``.
        """
        x = self._as_points(x)
        if sum(orders) == 0:
            return self.phi(x)

        dPhi = self.prior_basis.compute_derivative(x, dims, orders)  # (R_f, N)
        dW = self._apply_A(self._prior_kernel_derivative(x, dims, orders))  # (N, M)

        blocks = [dPhi - (dW @ self.Phi_X.T).T]  # (R_f, N)
        if self.has_noise_block:
            blocks.append(-(self.noise**0.5) * dW.T)  # (M, N)
        return torch.cat(blocks, dim=0)  # (R, N)

    def mean_derivative(self, x: torch.Tensor, dims, orders) -> torch.Tensor:
        """Mixed partials of the posterior mean at ``x``, shape ``(N,)``.

        ``m_post = m_prior + k(x, X) alpha``, so with a zero prior mean the derivative
        is ``d k(x, X) alpha``, using the same block as :meth:`compute_derivative`.

        A sampled function is ``w @ phi(x) + mean(x)``, so its derivative is
        ``w @ compute_derivative(...) + mean_derivative(...)``. The second term is
        the data-driven part of the field, often most of a PDE residual.

        Args:
            x: evaluation points ``(N, D)``.
            dims: the input dimensions to differentiate along.
            orders: the derivative order for each entry of ``dims``.

        Raises:
            NotImplementedError: if the prior mean is not a ``ZeroMean``.
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
        alpha = self.mean_module.alpha.to(torch.float64)  # as in PosteriorMean.forward
        return (dK.to(torch.float64) @ alpha).to(dK.dtype)  # (N,)
