import math
from abc import ABC, abstractmethod

import gpytorch
import torch
import torch.nn as nn

from flow_inference.flow_utils import build_grid_in_box

_STD_NORMAL = torch.distributions.Normal(0.0, 1.0)
_EPS = 1e-12


def _unit_clamp(u: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Clamp probabilities into (eps, 1 - eps) so CDF round-trips stay finite."""
    return u.clamp(min=eps, max=1.0 - eps)


@torch.no_grad()
def _gp_cholesky(H, W, kernel: gpytorch.kernels.Kernel, jitter, device):
    """Kernel Gram matrix on the H*W grid and its lower Cholesky factor.

    Returns ``(K, L)`` with ``K + jitter*I = L @ L^T``. The grid is built at the
    kernel's dtype: the kernel is the user's object and its precision is the one
    that matters, so following it keeps every buffer on a map coherent.
    """
    coords = build_grid_in_box(H, W, device=device, dtype=kernel.dtype)  # (HW, d)
    K = kernel(coords, coords).evaluate()  # (HW, HW)
    K = K + jitter * torch.eye(H * W, device=device, dtype=K.dtype)
    L = torch.linalg.cholesky(K).contiguous()
    return K, L


def _k_diag_sqrt(K: torch.Tensor) -> torch.Tensor:
    """Pointwise GP std sqrt(diag(K)) — divide a GP draw by this for unit marginals."""
    return K.diagonal().sqrt().clamp(min=_EPS)


class BaseMap(nn.Module, ABC):
    """A deterministic, differentiable map from a latent space (iid isotropic
    Gaussian increments) to a data space (a complicated field, path, or vector).

    A map is described by three numbers:

      - ``latent_dim``    : iid N(0,1) map inputs per channel.
      - ``output_shape``  : per-channel shape of a sample — ``(H, W)`` for a
                            field on a lattice, ``(num_times,)`` for a path.
      - ``channels``      : how many fields are stacked on the channel axis.

    Subclasses implement :meth:`_map`, which receives the full
    ``(B, C, latent_dim)`` tensor and returns ``(B, C, output_dim)``.
    A 2D ``(B, latent_dim)`` input is accepted as a single channel.

    Precomputed tensors should be registered with ``register_buffer`` so they move
    with ``.to(device)`` / ``.double()`` like the rest of the package.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, ...],
        channels: int = 1,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = tuple(output_shape)
        self.output_dim = math.prod(self.output_shape)
        self.channels = channels
        self.normal = _STD_NORMAL
        # Empty buffer carrying only a device and dtype, so :attr:`device`/:attr:`dtype`
        # follow ``.to()``/``.double()``. Non-persistent: not in the state_dict.
        self.register_buffer(
            "_probe", torch.zeros(0, dtype=dtype, device=device), persistent=False
        )

    @property
    def device(self) -> torch.device:
        """Device the map's tensors live on (tracks ``.to()``)."""
        return self._probe.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the map's tensors (tracks ``.double()`` / ``.float()``)."""
        return self._probe.dtype

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent Gaussian increments to a ``(B, C, output_dim)`` data field.

        Accepts ``(B, C, latent_dim)`` or, for a single channel, ``(B, latent_dim)``.
        """
        if z.dim() == 2:  # (B, latent_dim) -> (B, 1, latent_dim)
            z = z.unsqueeze(1)
        return self._map(z)  # (B, C, output_dim)

    @abstractmethod
    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """Map: ``(B, C, latent_dim)`` iid N(0,1) -> ``(B, C, output_dim)``."""
        raise NotImplementedError


class CauchyConvolutionMap(BaseMap):
    """Map iid Gaussian samples to a Cauchy-convolution field with Gaussian marginals.

    Cauchy variables are placed on a finer grid (``grid_fidelity_ratio`` per axis,
    pushed slightly outside the unit box to capture the heavy Cauchy tails) and
    convolved against the kernel; the convolved field is then pushed back to a
    standard-Gaussian marginal via the probability-integral transform.
    """

    def __init__(
        self,
        H,
        W,
        kernel: gpytorch.kernels.Kernel,
        grid_fidelity_ratio: int = 2,
        device="cpu",
    ):
        assert isinstance(grid_fidelity_ratio, int) and grid_fidelity_ratio >= 1, (
            "grid_fidelity_ratio must be a positive integer"
        )
        num_latents = H * W * grid_fidelity_ratio**2
        super().__init__(num_latents, (H, W), device=device, dtype=kernel.dtype)
        self.H, self.W = H, W
        self.grid_fidelity_ratio = (
            grid_fidelity_ratio  # finer Cauchy centres => better convolution approx
        )

        with torch.no_grad():
            coords = build_grid_in_box(H, W, device=device, dtype=kernel.dtype)
            fine_coords = build_grid_in_box(
                H * grid_fidelity_ratio,
                W * grid_fidelity_ratio,
                device=device,
                dtype=kernel.dtype,
            )  # (M, 2)
            # push the fine grid a bit outside the unit box: the Cauchy kernel has heavy
            # tails, so centres beyond the support give a better convolution approximation
            fine_coords = (fine_coords - 0.1 / 3.0) * 3.0
            big_K = kernel(coords, fine_coords).evaluate()  # (HW, M)
            self.register_buffer("K_T", big_K.t().contiguous())  # (M, HW)
            self.register_buffer("inv_norm", 1.0 / big_K.sum(-1))  # (HW,)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """(B, C, HW*ratio^2) Gaussians -> (B, C, HW) Cauchy-convolution field."""
        # inverse probability transform: Gaussian -> Uniform(0,1) -> Cauchy(0, gamma)
        u = self.normal.cdf(z)
        w = torch.tan(math.pi * (u - 0.5))
        S = w @ self.K_T
        # map the convolved field to a uniform marginal, then to a Gaussian marginal
        Z = (1.0 / math.pi) * torch.atan(S * self.inv_norm) + 0.5
        return self.normal.icdf(Z)


class BrownResnickMap(BaseMap):
    """Deterministic map from iid standard-Gaussian samples to a
    Brown-Resnick max-stable field with standard-Gaussian marginals.

    Finite-L truncation of the de Haan spectral representation:

        Z(s) = max_{l=1..L}  xi_l * exp(W_l(s) - gamma(s))

      - xi_l   = 1 / Gamma_l,  Gamma_l = E_1 + ... + E_l   (unit-rate Poisson arrivals)
      - W_l    ~ GP(0, K)  iid "storms", drawn from the full kernel
      - gamma(s) = 0.5 * Var(W(s)) = 0.5 * diag(K)         (normalises each storm to unit mean)

    As L -> infinity, Z(s) has unit-Frechet margins  P(Z <= z) = exp(-1/z).

    We finally push to Gaussian margins via  G = Phi^{-1}(exp(-1/Z))
    """

    def __init__(
        self,
        H,
        W,
        kernel: gpytorch.kernels.Kernel,
        num_storms: int = 500,
        device="cpu",
        jitter=1e-8,
    ):
        num_latents = (
            H * W * num_storms + num_storms
        )  # L storms x N grid points, plus L Poisson arrivals
        super().__init__(num_latents, (H, W), device=device, dtype=kernel.dtype)
        self.H, self.W = H, W
        self.L = num_storms

        # Exact full-rank storm factor: W_l = z_l @ L_K^T draws GP(0, K), since L_K L_K^T = K.
        K, L_K = _gp_cholesky(H, W, kernel, jitter, device)
        self.register_buffer("L_K", L_K)  # (N, N)
        self.register_buffer("gamma", 0.5 * K.diagonal().clone())  # (N,)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, L*(1+N)) iid N(0,1)  ->  (B, C, N)."""
        B, C = z.shape[0], z.shape[1]
        z = z.reshape(B, C, self.L, 1 + self.output_dim)

        z_pois = z[..., 0]  # (B, C, L)
        z_gp = z[..., 1:]  # (B, C, L, N)

        # Poisson arrivals from Gaussians:  U = Phi(z) ~ Unif(0,1),  E = -log U ~ Exp(1)
        u = _unit_clamp(self.normal.cdf(z_pois))
        E = -torch.log(u)
        Gamma = torch.cumsum(E, dim=-1)  # (B, C, L)
        log_xi = -torch.log(Gamma)  # log(1 / Gamma_l)

        # iid exact full-rank storms, per-storm log-intensity  log(xi_l) + W_l(s) - gamma(s)
        storms = torch.einsum("...ln,mn->...lm", z_gp, self.L_K)  # (B, C, L, N)
        log_terms = log_xi[..., None] + storms - self.gamma

        # exact max-stable pooling over storms (hard max, sparse grad)
        log_Z = log_terms.amax(dim=-2)  # (B, C, N)

        # Frechet -> Uniform -> Gaussian marginals:  U = exp(-1/Z) = exp(-exp(-log Z))
        U_out = _unit_clamp(torch.exp(-torch.exp(-log_Z)))
        return self.normal.icdf(U_out)  # (B, C, N)


class StudentTProcessMap(BaseMap):
    """Student-t process — GP with a shared chi-squared mixing variable.

    Representation:  Z(s) = G(s) / sqrt(V / nu)

      G ~ GP(0, K)   spatially correlated Gaussian field (exact full-rank Cholesky)
      V ~ chi^2(nu)  one shared variance multiplier per sample (elliptical mixing)
                     drawn from nu iid N(0,1) latents via  V = sum_k z_k^2

    Marginals are exactly  Z(s) ~ t_nu(0, 1)  (unit scale, after dividing each site
    by its pointwise GP std sqrt(K_ss)).
    """

    def __init__(
        self,
        H,
        W,
        kernel: gpytorch.kernels.Kernel,
        nu: int = 4,
        device="cpu",
        jitter=1e-8,
    ):
        num_latents = (
            H * W + nu
        )  # N grid points for the GP field, plus nu latents for the chi-squared mixer
        super().__init__(num_latents, (H, W), device=device, dtype=kernel.dtype)
        self.H, self.W = H, W
        self.nu = nu

        K, L_K = _gp_cholesky(H, W, kernel, jitter, device)
        self.register_buffer("L_K", L_K)  # (N, N)
        self.register_buffer("k_diag_sqrt", _k_diag_sqrt(K))  # (N,)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, N+nu) iid N(0,1)  ->  (B, C, N), t_nu(0,1) marginals."""
        z_gp = z[..., : self.output_dim]  # (B, C, N)
        z_chi = z[..., self.output_dim :]  # (B, C, nu)

        G = z_gp @ self.L_K.t()  # (B, C, N)  GP(0, K)
        V = (z_chi**2).sum(dim=-1).clamp(min=1e-6)  # (B, C)  chi^2(nu)
        # Normalise so each marginal is exactly t_nu(0, 1)
        return G / (torch.sqrt(V[..., None] / self.nu) * self.k_diag_sqrt)


class LogGaussianCoxMap(BaseMap):
    """Deterministic, differentiable map from iid standard-Gaussian samples to a
    log-Gaussian Cox process (LGCP).

    The latent log-intensity is a Gaussian field, drawn exactly from the kernel:

        G(s)      = mu + (z_gp @ L_K^T)(s),   L_K L_K^T = K          (GP(mu, K))
        Lambda(s) = exp(G(s))                                        (intensity surface)

    By default the map returns the *intensity field* Lambda(s) -- a smooth, positive,
    fully differentiable surface (this is the random object that defines the Cox process).
    """

    def __init__(
        self,
        H,
        W,
        kernel: gpytorch.kernels.Kernel,
        mu: float = 0.0,
        device="cpu",
        jitter=1e-8,
    ):
        super().__init__(H * W, (H, W), device=device, dtype=kernel.dtype)
        self.H, self.W = H, W
        self.mu = mu

        _, L_K = _gp_cholesky(H, W, kernel, jitter, device)
        self.register_buffer("L_K", L_K)  # (grid, grid)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, N) iid N(0,1)  ->  (B, C, N): intensity surface Lambda(s)."""
        log_lambda = self.mu + z @ self.L_K.t()
        return torch.exp(log_lambda)


class LevyNIGMap(BaseMap):
    """Deterministic, differentiable map from iid standard-Gaussian samples to a
    Levy / Normal-Inverse-Gaussian (NIG) random field.

    Uses the normal variance-mean mixture representation of the NIG law: an NIG
    variate is a Gaussian whose variance is itself Inverse-Gaussian (the IG is the
    Levy subordinator). Spatially, we subordinate a *correlated* Gaussian field by a
    single per-sample IG variance W, so each marginal is exactly NIG(alpha,beta,mu,delta)
    while the spatial dependence comes from the kernel:

        W      ~ IG(delta, gamma),   gamma = sqrt(alpha^2 - beta^2)     (one per sample)
        G(s)   ~ GP(0, R),           R = unit-diagonal (correlation) kernel
        Z(s)   = mu + beta * W + sqrt(W) * G(s)

    G is drawn from the raw kernel and divided by its pointwise std sqrt(diag(K)) to
    give the unit-diagonal R — the same normalisation StudentTProcessMap applies.

    Marginally  Z(s) ~ NIG(alpha, beta, mu, delta), heavy-tailed and (for beta != 0) skewed.

    Latents per sample: N grid values for the storm/field G, plus 2 scalars used to
    draw W via the Michael-Schucany-Haas IG sampler (one chi-square magnitude + one
    branch coin).
    """

    def __init__(
        self,
        H,
        W,
        kernel: gpytorch.kernels.Kernel,
        alpha: float = 2.0,
        beta: float = 0.5,
        delta: float = 1.0,
        mu: float = 0.0,
        device="cpu",
        jitter=1e-8,
    ):
        assert alpha > abs(beta), "NIG requires alpha > |beta|"
        # latent layout: H*W for the GP field + 1 (IG magnitude) + 1 (IG branch)
        super().__init__(H * W + 2, (H, W), device=device, dtype=kernel.dtype)
        self.H, self.W = H, W

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu
        self.gamma = math.sqrt(alpha * alpha - beta * beta)
        # IG(delta, gamma) == Wald(mean=delta/gamma, shape=delta**2)
        self.ig_mean = delta / self.gamma
        self.ig_shape = delta * delta

        K, L_K = _gp_cholesky(H, W, kernel, jitter, device)
        self.register_buffer("L_K", L_K)  # (grid, grid)
        # divide by the pointwise GP std so the field has unit marginals (R, not K),
        # making each marginal exactly NIG(alpha, beta, mu, delta)
        self.register_buffer("k_diag_sqrt", _k_diag_sqrt(K))  # (grid,)

    def _sample_ig(self, z_mag: torch.Tensor, z_branch: torch.Tensor) -> torch.Tensor:
        """Michael-Schucany-Haas: draw W ~ IG(mean=ig_mean, shape=ig_shape) from Gaussians."""
        mu_w = self.ig_mean
        lam = self.ig_shape
        y = z_mag * z_mag  # chi-square(1)
        x1 = (
            mu_w
            + (mu_w * mu_w * y) / (2 * lam)
            - (mu_w / (2 * lam)) * torch.sqrt(4 * mu_w * lam * y + mu_w * mu_w * y * y)
        )
        u = self.normal.cdf(z_branch)  # Uniform(0,1)
        accept = u <= mu_w / (mu_w + x1)
        return torch.where(accept, x1, (mu_w * mu_w) / x1)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, N + 2) iid N(0,1)  ->  (B, C, N) NIG field."""
        n = self.output_dim
        z_gp = z[..., :n]  # (B, C, N)
        w = self._sample_ig(z[..., n], z[..., n + 1])  # (B, C)
        G = (z_gp @ self.L_K.t()) / self.k_diag_sqrt  # (B, C, N)  GP(0, R)
        return self.mu + self.beta * w[..., None] + w.sqrt()[..., None] * G
