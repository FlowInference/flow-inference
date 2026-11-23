"""Maps: deterministic, differentiable transforms of iid Gaussians into processes.

A map turns a vector of iid ``N(0, 1)`` latents into a sample of a (typically
non-Gaussian) random field, so that
:class:`~flow_inference.flows.GaussianIncrementFlow` can sample and guide the process
by diffusing over the latents. The maps here produce fields on an ``H x W`` grid in
the unit box, with spatial dependence from a gpytorch kernel:

* :class:`CauchyConvolutionMap`: a Cauchy convolution field with Gaussian marginals.
* :class:`BrownResnickMap`: a Brown-Resnick max-stable field with Gaussian marginals.
* :class:`StudentTProcessMap`: a Student-t process.
* :class:`LogGaussianCoxMap`: the intensity surface of a log-Gaussian Cox process.
* :class:`LevyNIGMap`: a Normal-Inverse-Gaussian field.

Subclass :class:`BaseMap` for a new process.
"""

import math
from abc import ABC, abstractmethod

import gpytorch
import torch
import torch.nn as nn

from flow_inference.flow_utils import build_grid_in_box, float64_copy

_STD_NORMAL = torch.distributions.Normal(0.0, 1.0)
_EPS = 1e-12


def _unit_clamp(u: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Clamp probabilities into ``(eps, 1 - eps)`` so inverse CDFs stay finite."""
    return u.clamp(min=eps, max=1.0 - eps)


@torch.no_grad()
def _gp_cholesky(H, W, kernel: gpytorch.kernels.Kernel, jitter, device):
    """Kernel matrix on the ``H x W`` grid and its lower Cholesky factor.

    Returns ``(K, L)``, where ``K`` is the kernel matrix with ``jitter`` added to its
    diagonal and ``K = L L^T``, both ``(H W, H W)`` in the kernel's dtype. They are
    computed in float64 and then cast, since a float32 Cholesky of a GP kernel matrix
    fails at the default jitter.
    """
    coords = build_grid_in_box(H, W, device=device, dtype=torch.float64)  # (HW, d)
    K = float64_copy(kernel)(coords, coords).to_dense()  # (HW, HW)
    K = K + jitter * torch.eye(H * W, device=device, dtype=K.dtype)
    L = torch.linalg.cholesky(K).contiguous()
    return K.to(kernel.dtype), L.to(kernel.dtype)


def _k_diag_sqrt(K: torch.Tensor) -> torch.Tensor:
    """Pointwise GP standard deviation ``sqrt(diag(K))``.

    Dividing a GP draw by it gives unit-variance marginals.
    """
    return K.diagonal().sqrt().clamp(min=_EPS)


class BaseMap(nn.Module, ABC):
    """Abstract base class for a map from iid Gaussians to a process.

    A map is a deterministic, differentiable function from iid ``N(0, 1)`` latents
    ("increments") to a sample of a field, path or vector. Its dimensions are:

    - ``latent_dim``: latents per channel.
    - ``output_shape``: per-channel shape of a sample, e.g. ``(H, W)`` for a field on
      a grid or ``(num_times,)`` for a path. ``output_dim`` is its product.
    - ``channels``: how many fields are stacked on the channel axis.

    Subclasses implement :meth:`_map`, from ``(B, C, latent_dim)`` to
    ``(B, C, output_dim)``. Register precomputed tensors with ``register_buffer`` so
    they follow ``.to()`` and ``.double()``.

    Args:
        latent_dim: latents per channel.
        output_shape: per-channel shape of a sample.
        channels: number of channels.
        device: device of the map's tensors.
        dtype: working precision of the map.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, ...],
        channels: int = 1,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = tuple(output_shape)
        self.output_dim = math.prod(self.output_shape)
        self.channels = channels
        self.normal = _STD_NORMAL
        # device/dtype are read off an empty buffer so they follow .to()/.double().
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
        """Map latents to a ``(B, C, output_dim)`` sample.

        Accepts ``(B, C, latent_dim)`` or, for a single channel, ``(B, latent_dim)``.
        """
        if z.dim() == 2:
            z = z.unsqueeze(1)
        return self._map(z)

    @abstractmethod
    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, latent_dim)`` iid ``N(0, 1)`` latents to
        ``(B, C, output_dim)``."""
        raise NotImplementedError


class CauchyConvolutionMap(BaseMap):
    """Cauchy convolution field on an ``H x W`` grid, with standard normal marginals.

    Each latent is turned into a Cauchy variable at a centre on a finer grid, and the
    field is their convolution with the kernel. A weighted sum of Cauchy variables is
    Cauchy, so the probability integral transform then gives exactly standard normal
    marginals; the (non-Gaussian) dependence comes from the convolution.

    Args:
        H: grid height.
        W: grid width.
        kernel: the convolution kernel. Sets the map's dtype.
        grid_fidelity_ratio: refinement of the grid of Cauchy centres per axis; finer
            centres give a better approximation of the continuous convolution.
        device: device of the map's tensors.
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
        self.grid_fidelity_ratio = grid_fidelity_ratio

        with torch.no_grad():
            coords = build_grid_in_box(H, W, device=device, dtype=kernel.dtype)
            fine_coords = build_grid_in_box(
                H * grid_fidelity_ratio,
                W * grid_fidelity_ratio,
                device=device,
                dtype=kernel.dtype,
            )  # (M, 2)
            # Spread the centres beyond the unit box: the Cauchy variables are
            # heavy-tailed, so centres outside the grid still contribute.
            fine_coords = (fine_coords - 0.1 / 3.0) * 3.0
            big_K = kernel(coords, fine_coords).to_dense()  # (HW, M)
            self.register_buffer("K_T", big_K.t().contiguous())  # (M, HW)
            self.register_buffer("inv_norm", 1.0 / big_K.sum(-1))  # (HW,)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, C, H W ratio^2)`` latents to a ``(B, C, H W)`` field."""
        # Gaussian -> Uniform(0, 1) -> standard Cauchy.
        u = self.normal.cdf(z)
        w = torch.tan(math.pi * (u - 0.5))
        S = w @ self.K_T
        # S is Cauchy with scale 1 / inv_norm: its CDF gives a uniform, then a normal.
        Z = (1.0 / math.pi) * torch.atan(S * self.inv_norm) + 0.5
        return self.normal.icdf(Z)


class BrownResnickMap(BaseMap):
    """Brown-Resnick max-stable field on an ``H x W`` grid, with standard normal
    marginals.

    Uses the de Haan spectral representation, truncated at ``L`` storms::

        Z(s) = max_{l=1..L} xi_l exp(W_l(s) - gamma(s))

    - ``xi_l = 1 / Gamma_l``, with ``Gamma_l = E_1 + ... + E_l`` the arrival times of
      a unit-rate Poisson process;
    - ``W_l ~ GP(0, K)`` iid "storms";
    - ``gamma(s) = 0.5 diag(K)``, which gives each ``exp(W_l(s) - gamma(s))`` unit
      mean.

    As ``L`` grows, ``Z(s)`` has unit-Frechet marginals ``P(Z <= z) = exp(-1/z)``; the
    map returns ``Phi^{-1}(exp(-1/Z))``, with standard normal marginals.

    Args:
        H: grid height.
        W: grid width.
        kernel: the storm kernel ``K``. Sets the map's dtype.
        num_storms: number of storms ``L``.
        device: device of the map's tensors.
        jitter: added to the diagonal of ``K`` before its Cholesky factorisation.
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
        # One latent per grid point per storm, plus one per Poisson arrival.
        num_latents = H * W * num_storms + num_storms
        super().__init__(num_latents, (H, W), device=device, dtype=kernel.dtype)
        self.H, self.W = H, W
        self.L = num_storms

        # Storms are drawn exactly as W_l = L_K z_l.
        K, L_K = _gp_cholesky(H, W, kernel, jitter, device)
        self.register_buffer("L_K", L_K)  # (N, N)
        self.register_buffer("gamma", 0.5 * K.diagonal().clone())  # (N,)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, C, L (1 + N))`` latents to a ``(B, C, N)`` field."""
        B, C = z.shape[0], z.shape[1]
        z = z.reshape(B, C, self.L, 1 + self.output_dim)

        z_pois = z[..., 0]  # (B, C, L)
        z_gp = z[..., 1:]  # (B, C, L, N)

        # Poisson arrivals: U = Phi(z) ~ Uniform(0, 1), E = -log U ~ Exp(1).
        u = _unit_clamp(self.normal.cdf(z_pois))
        E = -torch.log(u)
        Gamma = torch.cumsum(E, dim=-1)  # (B, C, L)
        log_xi = -torch.log(Gamma)

        # Per-storm log-intensity log(xi_l) + W_l(s) - gamma(s).
        storms = torch.einsum("...ln,mn->...lm", z_gp, self.L_K)  # (B, C, L, N)
        log_terms = log_xi[..., None] + storms - self.gamma

        # Max over storms (a hard max, so the gradient is sparse).
        log_Z = log_terms.amax(dim=-2)  # (B, C, N)

        # Frechet -> Uniform -> Gaussian: U = exp(-1/Z) = exp(-exp(-log Z)).
        U_out = _unit_clamp(torch.exp(-torch.exp(-log_Z)))
        return self.normal.icdf(U_out)  # (B, C, N)


class StudentTProcessMap(BaseMap):
    """Student-t process on an ``H x W`` grid, with ``t_nu(0, 1)`` marginals.

    A GP divided by one shared chi-squared variable per sample::

        Z(s) = G(s) / (sqrt(V / nu) sqrt(K(s, s)))

    - ``G ~ GP(0, K)``, drawn exactly via the Cholesky factor of ``K``;
    - ``V = z_1^2 + ... + z_nu^2 ~ chi^2(nu)``, from ``nu`` further latents.

    Dividing by the pointwise standard deviation ``sqrt(K(s, s))`` gives each site
    exactly a ``t_nu(0, 1)`` marginal.

    Args:
        H: grid height.
        W: grid width.
        kernel: the GP kernel ``K``. Sets the map's dtype.
        nu: degrees of freedom.
        device: device of the map's tensors.
        jitter: added to the diagonal of ``K`` before its Cholesky factorisation.
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
        # One latent per grid point for the GP, plus nu for the chi-squared variable.
        num_latents = H * W + nu
        super().__init__(num_latents, (H, W), device=device, dtype=kernel.dtype)
        self.H, self.W = H, W
        self.nu = nu

        K, L_K = _gp_cholesky(H, W, kernel, jitter, device)
        self.register_buffer("L_K", L_K)  # (N, N)
        self.register_buffer("k_diag_sqrt", _k_diag_sqrt(K))  # (N,)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, C, N + nu)`` latents to a ``(B, C, N)`` field."""
        z_gp = z[..., : self.output_dim]  # (B, C, N)
        z_chi = z[..., self.output_dim :]  # (B, C, nu)

        G = z_gp @ self.L_K.t()  # (B, C, N), GP(0, K)
        V = (z_chi**2).sum(dim=-1).clamp(min=1e-6)  # (B, C), chi^2(nu)
        return G / (torch.sqrt(V[..., None] / self.nu) * self.k_diag_sqrt)


class LogGaussianCoxMap(BaseMap):
    """Intensity surface of a log-Gaussian Cox process on an ``H x W`` grid.

    The log-intensity is a GP, drawn exactly via the Cholesky factor of ``K``::

        G(s) = mu + (L_K z)(s) ~ GP(mu, K)
        Lambda(s) = exp(G(s))

    The map returns the intensity ``Lambda``, a positive, differentiable surface; it
    is the random object that defines the Cox process.

    Args:
        H: grid height.
        W: grid width.
        kernel: the kernel ``K`` of the log-intensity. Sets the map's dtype.
        mu: constant mean of the log-intensity.
        device: device of the map's tensors.
        jitter: added to the diagonal of ``K`` before its Cholesky factorisation.
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
        self.register_buffer("L_K", L_K)  # (N, N)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, C, N)`` latents to a ``(B, C, N)`` intensity surface."""
        log_lambda = self.mu + z @ self.L_K.t()
        return torch.exp(log_lambda)


class LevyNIGMap(BaseMap):
    """Normal-Inverse-Gaussian (NIG) random field on an ``H x W`` grid.

    Uses the normal variance-mean mixture form of the NIG law: a Gaussian whose
    variance is Inverse-Gaussian (IG). A correlated Gaussian field is mixed with one
    IG variable per sample::

        W ~ IG(delta, gamma), gamma = sqrt(alpha^2 - beta^2)
        G ~ GP(0, R), R the correlation kernel (K scaled to unit diagonal)
        Z(s) = mu + beta W + sqrt(W) G(s)

    so each marginal is exactly ``NIG(alpha, beta, mu, delta)``: heavy-tailed, and
    skewed when ``beta != 0``. The latents are one per grid point for ``G``, plus two
    for drawing ``W`` with the Michael-Schucany-Haas sampler.

    Args:
        H: grid height.
        W: grid width.
        kernel: the kernel ``K`` of the Gaussian field. Sets the map's dtype.
        alpha: tail heaviness; must exceed ``|beta|``.
        beta: skewness.
        delta: scale.
        mu: location.
        device: device of the map's tensors.
        jitter: added to the diagonal of ``K`` before its Cholesky factorisation.
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
        # Latents: H*W for the GP field, then the IG magnitude and branch variables.
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
        self.register_buffer("L_K", L_K)  # (N, N)
        # Dividing by the pointwise std turns a GP(0, K) draw into a GP(0, R) one.
        self.register_buffer("k_diag_sqrt", _k_diag_sqrt(K))  # (N,)

    def _sample_ig(self, z_mag: torch.Tensor, z_branch: torch.Tensor) -> torch.Tensor:
        """Draw ``W ~ IG(mean=ig_mean, shape=ig_shape)`` from two standard normals.

        The Michael-Schucany-Haas method: ``z_mag`` gives the chi-squared variable
        and ``z_branch`` the uniform that picks one of the two roots.
        """
        mu_w = self.ig_mean
        lam = self.ig_shape
        y = z_mag * z_mag  # chi^2(1)
        x1 = (
            mu_w
            + (mu_w * mu_w * y) / (2 * lam)
            - (mu_w / (2 * lam)) * torch.sqrt(4 * mu_w * lam * y + mu_w * mu_w * y * y)
        )
        u = self.normal.cdf(z_branch)  # Uniform(0,1)
        accept = u <= mu_w / (mu_w + x1)
        return torch.where(accept, x1, (mu_w * mu_w) / x1)

    def _map(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, C, N + 2)`` latents to a ``(B, C, N)`` field."""
        n = self.output_dim
        z_gp = z[..., :n]  # (B, C, N)
        w = self._sample_ig(z[..., n], z[..., n + 1])  # (B, C)
        G = (z_gp @ self.L_K.t()) / self.k_diag_sqrt  # (B, C, N), GP(0, R)
        return self.mu + self.beta * w[..., None] + w.sqrt()[..., None] * G
