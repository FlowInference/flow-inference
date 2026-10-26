import logging
import math
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from flow_inference.maps import BaseMap
from flow_inference.schedules import LinearSchedule, VPSchedule
from flow_inference.training_samplers import TrainingSampler

logger = logging.getLogger(__name__)


class BaseFlow(nn.Module, ABC):
    """Abstract base class for a flow (as specified by VP SDEs).
    Subclasses define the drift. t=1 (noise) down to t≈0 (data).

    Every flow declares the same dimension model as :class:`~flow_inference.maps.BaseMap`,
    so the sampler can draw the right latent and the caller knows the output width:

      - ``latent_dim``  : iid N(0,1) entries the diffusion runs in, *per channel*.
                          The sampler draws ``(num_samples, channels, latent_dim)``.
      - ``output_dim``  : data-space entries produced *per channel* (after
                          ``_to_data_space``). For the closed-form Gaussian flows the
                          diffusion is already in data space, so this equals
                          ``latent_dim``; for an increment flow it is the map's
                          ``output_dim`` while ``latent_dim`` is the map's latent size.
      - ``channels``    : as for maps — how many fields are stacked on the channel axis.
    """

    # True when p(z_t) = N(0, I) for every t: the closed-form Gaussian/GP flows, not a
    # learned velocity field. PCNCorrector requires it, since its pCN proposal is
    # reversible with respect to N(0, I) specifically.
    has_standard_normal_marginals: bool = False

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        channels: int = 1,
        output_shape: tuple[int, ...] | None = None,
        dtype=torch.float64,
        device="cpu",
        schedule: VPSchedule | None = None,
    ):
        super().__init__()
        # device/dtype are read off an empty buffer so they follow .to()/.double().
        self.register_buffer(
            "_probe", torch.zeros(0, dtype=dtype, device=device), persistent=False
        )
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.channels = channels
        # Natural per-channel shape of a returned sample: (H, W) for a grid field,
        # the flat (output_dim,) otherwise. The diffusion and conditions always work on
        # the flat (B, C, output_dim) field; only to_display applies this shape.
        if output_shape is None:
            output_shape = (output_dim,)
        if math.prod(output_shape) != output_dim:
            raise ValueError(
                f"output_shape {tuple(output_shape)} has "
                f"{math.prod(output_shape)} cells, != output_dim {output_dim}"
            )
        self.output_shape = tuple(output_shape)

        # Default VP schedule (linear beta) if none is provided
        if schedule is None:
            beta_0 = torch.tensor(0.00001, device=device, dtype=dtype)
            beta_1 = torch.tensor(10.0, device=device, dtype=dtype)
            schedule = LinearSchedule(beta_0=beta_0, beta_1=beta_1)
        self.schedule = schedule

    @property
    def device(self) -> torch.device:
        """Device the flow's tensors live on (tracks ``.to()``)."""
        return self._probe.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the flow's tensors (tracks ``.double()`` / ``.float()``)."""
        return self._probe.dtype

    @abstractmethod
    def __call__(self, z: torch.Tensor, t: float) -> torch.Tensor:
        """Return the drift vector v(z, t) for the probability-flow ODE."""
        raise NotImplementedError

    @abstractmethod
    def predict_terminal(self, z_t: torch.Tensor, t: float) -> torch.Tensor:
        """estimate end points z_0 | z_t from the velocity network, for use in conditional sampling."""
        raise NotImplementedError

    def _to_data_space(self, z: torch.Tensor) -> torch.Tensor:
        """Map the diffusion state to the field conditions see. Identity by default;
        flows that diffuse in a latent space (whitened GP, increments) override it."""
        return z

    def to_display(self, f: torch.Tensor) -> torch.Tensor:
        """Reshape a flat ``(B, C, output_dim)`` field to ``(B, C, *output_shape)``.

        A no-op when ``output_shape == (output_dim,)`` (1D fields / paths); reshapes
        to ``(B, C, H, W)`` for grid flows. Applied by the sampler to its returned
        sample so downstream plotting needs no manual ``.view(B, H, W)``.
        """
        return f.reshape(f.shape[0], f.shape[1], *self.output_shape)


def terminal_draws(
    flow: BaseFlow, z_t: torch.Tensor, t: float, noise: torch.Tensor
) -> torch.Tensor:
    """Draws from ``p(z_0 | z_t) = N(sqrt(alpha_t) z_t, (1 - alpha_t) I)``, ``(B, S, C, N)``.

    Takes the standard-normal ``noise`` ``(B, S, C, N)`` rather than drawing it, so each
    caller owns its own freezing policy — the flow caches noise for the frozen-noise
    reparameterisation :class:`~flow_inference.guidance.GuidanceMC` needs, and refreshing
    that cache from elsewhere would silently unfreeze the guidance sharing the flow.

    ``_one_minus_alpha_2``, not ``1 - _alpha_2``: the naive subtraction underflows to
    *exactly zero* in float32 for t <~ 1e-5, which would collapse every draw onto the
    Tweedie mean.
    """
    std = torch.sqrt(flow.schedule._one_minus_alpha_2(t).clamp(min=0.0))
    return flow.predict_terminal(z_t, t).unsqueeze(1) + std * noise


class GaussianFlow(BaseFlow):
    """
    SDE mapping Gaussian to Gaussian via VP noise schedule.
    """

    # The prior is N(0, I) and the unconditional drift is zero, so every marginal is
    # N(0, I). Inherited by the GP / basis / increment flows, all of which diffuse in a
    # whitened space and apply their (possibly nonlinear) map in _to_data_space.
    has_standard_normal_marginals: bool = True
    # True when _to_data_space is affine, f = A z + b (see _linear_part / _pullback).
    # GuidanceMC then decodes once per particle rather than once per Monte Carlo draw.
    # Subclasses with a nonlinear map must set this to False.
    affine_data_space: bool = True

    def __init__(
        self,
        latent_dim: int,
        channels: int = 1,
        output_shape: tuple[int, ...] | None = None,
        device="cpu",
        dtype=torch.float64,
        schedule: VPSchedule | None = None,
        output_dim: int | None = None,
    ):
        # output_dim defaults to latent_dim: the plain Gaussian flow diffuses directly
        # in data space. Subclasses that decode the latent pass their own.
        super().__init__(
            latent_dim=latent_dim,
            output_dim=latent_dim if output_dim is None else output_dim,
            channels=channels,
            output_shape=output_shape,
            device=device,
            dtype=dtype,
            schedule=schedule,
        )
        # Cache for the reparameterisation noise used in sample_terminal;
        # re-sampled lazily when batch size, sample count or pairing changes.
        self.z = None
        self._z_antithetic = False
        # A @ self.z, cached for sample_terminal_data; recomputed when self.z changes.
        self._z_decoded = None
        self._z_decoded_src = None

    def __call__(self, z: torch.Tensor, t: float) -> torch.Tensor:
        # The unconditional drift is zero because
        # the prior is already N(0, I) — the ODE has no curvature.
        return torch.zeros_like(z)

    def predict_terminal(self, z_t: torch.Tensor, t: float) -> torch.Tensor:
        # Tweedie estimate: E[z_0 | z_t] = sqrt(alpha_t) * z_t under z_t ~ N(sqrt(alpha_t) z_0, (1-alpha_t) I).
        return torch.sqrt(self.schedule._alpha_2(t)) * z_t

    def _linear_part(self, z: torch.Tensor) -> torch.Tensor:
        """``A z`` for the affine ``_to_data_space(z) = A z + b``. Identity here."""
        return z

    def _pullback(self, g: torch.Tensor) -> torch.Tensor:
        """``A^T g``: a data-space gradient ``(B, C, output_dim)`` as a latent one."""
        return g

    # -------------------------
    # extra bits, specific for this closed-form GP case, not needed for general flows
    # -------------------------

    def _frozen_noise(
        self,
        batch_size: int,
        num_samples: int,
        resample_noise: bool = False,
        antithetic: bool = False,
    ) -> torch.Tensor:
        """The cached ``(B, S, C, latent_dim)`` reparameterisation noise.

        Redrawn when asked to, or when the batch, sample count or pairing changes.
        """
        if (
            resample_noise
            or self.z is None
            or self.z.shape[1] != num_samples
            or self.z.shape[0] != batch_size
            or self._z_antithetic != antithetic
        ):
            z = torch.randn(
                (batch_size, num_samples, self.channels, self.latent_dim),
                device=self.device,
                dtype=self.dtype,
            )
            if antithetic:
                half = num_samples // 2
                z[:, half : 2 * half] = -z[:, :half]
            self.z = z
            self._z_antithetic = antithetic
        return self.z

    def sample_terminal(
        self,
        z_t: torch.Tensor,
        t: float,
        num_samples: int,
        resample_noise: bool = False,
        antithetic: bool = False,
    ) -> torch.Tensor:
        """Draw samples from the terminal posterior p(z_0 | z_t) via the reparameterisation trick.

        Under the VP forward process, the posterior is Gaussian:
            p(z_0 | z_t) = N(sqrt(α_t) * z_t,  (1 - α_t) * I)

        Args:
            resample_noise: draw fresh noise on every call instead of reusing the
                cached matrix. The default (False) is *frozen-noise*
                reparameterisation, which :class:`~flow_inference.guidance.GuidanceMC`
                needs so that gradients flow through μ only. Estimators that
                differentiate nothing want True: with frozen noise their Monte-Carlo
                error is the *same* error at every step, so it integrates into a
                coherent bias down the whole trajectory instead of averaging away
                like 1/sqrt(num_steps).
            antithetic: mirror the draws in ± pairs (``z[half:] = -z[:half]``, with an
                odd leftover left independent). For a first-moment estimator like
                :class:`~flow_inference.guidance.GuidanceMCFisher` this is a large
                variance reduction: the pair contributions cancel exactly when the
                likelihood is flat across the pair, so the estimator reports zero
                signal rather than a random direction, and reduces to a
                finite-difference along ε when there *is* signal.
        """
        noise = self._frozen_noise(
            z_t.shape[0], num_samples, resample_noise, antithetic
        )
        return terminal_draws(self, z_t, t, noise)  # (B, S, C, N)

    def sample_terminal_data(
        self, z_t: torch.Tensor, t: float, num_samples: int
    ) -> torch.Tensor:
        """Frozen-noise draws of ``p(z_0 | z_t)`` mapped to data space.

        Equal to ``_to_data_space`` applied to :meth:`sample_terminal`'s draws, but
        uses the affine form ``A (mu + s eps) + b = (A mu + b) + s (A eps)``: the
        posterior mean is decoded once per particle, and ``A eps`` is cached alongside
        the frozen noise. For a GP flow that turns ``B * S`` dense ``N x N`` matmuls per
        step into ``B``.

        Returns:
            ``(B, S, C, output_dim)``
        """
        if not self.affine_data_space:
            raise NotImplementedError(
                f"{type(self).__name__} has a nonlinear _to_data_space; "
                "decode sample_terminal's draws instead"
            )
        noise = self._frozen_noise(z_t.shape[0], num_samples)
        if self._z_decoded_src is not noise:
            B, S, C, N = noise.shape
            with torch.no_grad():
                decoded = self._linear_part(noise.reshape(B * S, C, N))
            self._z_decoded = decoded.reshape(B, S, C, -1)
            self._z_decoded_src = noise
        std = torch.sqrt(self.schedule._one_minus_alpha_2(t).clamp(min=0.0))
        mean = self._to_data_space(self.predict_terminal(z_t, t))  # (B, C, output_dim)
        return mean.unsqueeze(1) + std * self._z_decoded


def _whiten_factors(target_covar: torch.Tensor, jitter: float):
    """Eigen square root of a ``(C, N, N)`` covariance: ``(L, L_inv)`` with ``L L^T = K``.

    ``K = U diag(lam) U^T`` (``eigh`` exploits symmetry) gives ``L = U diag(sqrt(lam))``.
    Eigenvalues are clamped at ``jitter``: even after regularising, rounding can leave
    them tiny or slightly negative, which would make ``sqrt_lam`` imaginary and
    ``inv_sqrt_lam`` explode.
    """
    C, N = target_covar.shape[0], target_covar.shape[-1]
    jitter_mat = jitter * torch.eye(
        N, device=target_covar.device, dtype=target_covar.dtype
    ).unsqueeze(0).expand(C, N, N)
    lambdas, U = torch.linalg.eigh(target_covar + jitter_mat)  # (C, N), (C, N, N)
    lambdas = lambdas.clamp(min=jitter)
    sqrt_lam = torch.sqrt(lambdas)
    L = U @ torch.diag_embed(sqrt_lam)  # (C, N, N)
    L_inv = torch.diag_embed(1.0 / sqrt_lam) @ U.transpose(-2, -1)  # (C, N, N)
    return L, L_inv


class GaussianProcessFlow(GaussianFlow):
    """GP diffusion flow with a VP noise schedule.

    The key idea is to work in a *whitened* coordinate system where the GP
    prior becomes a standard normal N(0, I).  The final sample is mapped back to the original GP space via
    the unwhitening transform  f = L @ z + mean,  where  K = L @ L^T  (Cholesky / eigen-sqrt)
    """

    def __init__(
        self,
        kernel,
        mean,
        grid: torch.Tensor,
        channels: int | None = None,
        jitter=1e-12,
        output_shape: tuple[int, ...] | None = None,
        device="cpu",
        schedule: VPSchedule | None = None,
    ):
        """Build the flow from a kernel and mean function evaluated on ``grid``.

        Args:
            kernel: any callable ``k(x, x') -> (N, N)`` — a ``gpytorch.kernels.Kernel``,
                including :class:`~flow_inference.gp_utils.PosteriorKernel`. May also be
                a sequence of kernels, or a batched kernel, for several channels.
            mean: any callable ``m(x) -> (N,)`` — a ``gpytorch.means.Mean``, including
                :class:`~flow_inference.gp_utils.PosteriorMean`. May be a sequence.
            grid: the ``(N, D)`` points the field lives on.
            channels: replicate a single kernel/mean across this many independent
                channels. Leave as None when the kernel already supplies them.

        Pair with :func:`~flow_inference.gp_utils.gp_posterior`, which returns exactly
        the ``(mean, kernel)`` this expects.

        Multiple independent GP channels are supported three ways:

        * pass sequences — ``GaussianProcessFlow([k1, k2], [m1, m2], grid)``;
        * pass a kernel with a gpytorch ``batch_shape``, whose Gram matrix is already
          ``(C, N, N)`` (give a matching batched mean, or one mean for all channels);
        * pass ``channels=C`` with a single kernel to replicate it across channels.
        """
        grid = grid.to(device)
        covar, mu = self._evaluate(kernel, mean, grid, channels)  # (C, N, N), (C, N)
        # The diffusion runs whitened in data space: latent_dim == output_dim == N.
        super().__init__(
            latent_dim=covar.shape[-1],
            channels=covar.shape[0],
            output_shape=output_shape,
            device=device,
            dtype=covar.dtype,
            schedule=schedule,
        )
        self.jitter = jitter
        # .contiguous(): gpytorch's ZeroMean/ConstantMean return an expanded view whose
        # elements share memory, which load_state_dict cannot copy into.
        self.register_buffer("target_mean", mu.contiguous())  # (C, N)
        L, L_inv = _whiten_factors(covar, jitter)
        self.register_buffer("L", L)
        self.register_buffer("L_inv", L_inv)

    @staticmethod
    def _evaluate(kernel, mean, grid, channels):
        """Evaluate kernel(s)/mean(s) on the grid as ``(C, N, N)`` and ``(C, N)``."""

        def dense(k):
            out = k(grid, grid)
            return (out.to_dense() if hasattr(out, "to_dense") else out).detach()

        if isinstance(kernel, (list, tuple)):
            covar = torch.stack([dense(k) for k in kernel])  # (C, N, N)
        else:
            covar = dense(kernel)
            if covar.ndim == 2:  # a single (N, N) kernel
                covar = covar.unsqueeze(0)
            # ndim == 3 already: a batched kernel supplied its own channel axis

        if isinstance(mean, (list, tuple)):
            mu = torch.stack([m(grid).detach() for m in mean])  # (C, N)
        else:
            mu = mean(grid).detach()
            if mu.ndim == 1:
                mu = mu.unsqueeze(0)

        # `channels` replicates a single kernel/mean across C independent channels.
        if channels is not None:
            if covar.shape[0] == 1:
                covar = covar.expand(channels, -1, -1)
            if mu.shape[0] == 1:
                mu = mu.expand(channels, -1)
        if mu.shape[0] == 1 and covar.shape[0] > 1:
            mu = mu.expand(covar.shape[0], -1)  # one mean shared across channels
        if mu.shape[0] != covar.shape[0]:
            raise ValueError(
                f"got {covar.shape[0]} covariance channels but {mu.shape[0]} mean "
                "channels; pass one mean per kernel or a single shared mean"
            )
        return covar, mu

    def _linear_part(self, z: torch.Tensor) -> torch.Tensor:
        """``L @ z`` per channel: (B, C, N) -> (B, C, N)."""
        return torch.einsum("bcn,cmn->bcm", z, self.L)

    def _pullback(self, g: torch.Tensor) -> torch.Tensor:
        """``L^T @ g`` per channel: (B, C, N) -> (B, C, N)."""
        return torch.einsum("bcm,cmn->bcn", g, self.L)

    def _to_data_space(self, z: torch.Tensor) -> torch.Tensor:
        """Unwhiten: z (B, C, N) -> f = L @ z + mean."""
        return self._linear_part(z) + self.target_mean.unsqueeze(0)


class BasisGaussianProcessFlow(GaussianFlow):
    """GP flow that diffuses over the ``R`` weights of a basis expansion.

    A basis (see :class:`~flow_inference.gp_utils.AbstractBasis`) writes draws as
    ``f = Phi^T w + m`` with ``w ~ N(0, I)``, so this is a GP flow — just
    parameterised by ``R``-dimensional weights rather than ``N`` grid values. The
    weight prior is already white, so the diffusion runs directly on the weights and
    ``_to_data_space`` is the identity.

    What this class adds is *where* the basis is applied: in :meth:`to_display`, which
    the sampler calls only on its final output. That placement is deliberate. Guidance
    goes through ``_to_data_space``, so leaving that as the identity keeps conditions in
    **weight** space, where a basis with closed-form derivatives
    (:meth:`~flow_inference.gp_utils.RFFBasis.compute_derivative`) can supply exact ones
    rather than finite differences. Callers still get function values back from the
    sampler and never map by hand.

    If you want conditions in *function* space instead, wrap the basis in a
    :class:`~flow_inference.maps.BaseMap` and use :class:`GaussianIncrementFlow`: that
    applies the map inside ``_to_data_space``, so guidance sees the field — at the cost
    of the analytic derivatives.

    Example:
        >>> basis = RFFBasis(R=256, input_dim=2, mean=mean, kernel=kernel)
        >>> flow = BasisGaussianProcessFlow(basis, coords, output_shape=(H, W))
        >>> samples = IntegratorSampler(flow).sample(B, guidance=weight_guidance)
        >>> samples.shape                      # (B, 1, H, W) — already in function space
    """

    def __init__(
        self,
        basis,
        grid: torch.Tensor,
        output_shape: tuple[int, ...] | None = None,
        device="cpu",
        schedule: VPSchedule | None = None,
    ):
        """Build the flow by evaluating ``basis`` on ``grid``.

        ``Phi`` and the mean are evaluated once and stored as buffers, so decoding is
        a matmul and everything moves with ``.to()``.

        Args:
            basis: an :class:`~flow_inference.gp_utils.AbstractBasis`.
            grid: the ``(N, D)`` points the function samples live on.
            output_shape: per-channel display shape of the *function* samples, e.g.
                ``(H, W)``. Defaults to the flat ``(N,)``.
        """
        grid = grid.to(device=device, dtype=basis.dtype)
        if grid.ndim == 1:
            grid = grid.unsqueeze(-1)

        if output_shape is not None and math.prod(output_shape) != grid.shape[0]:
            raise ValueError(
                f"output_shape {tuple(output_shape)} has {math.prod(output_shape)} "
                f"cells, != the basis grid size {grid.shape[0]}"
            )
        # Latent: R iid weights. Output: the N grid values the basis expands them to.
        super().__init__(
            latent_dim=basis.R,
            output_dim=grid.shape[0],
            output_shape=output_shape,
            device=device,
            dtype=basis.dtype,
            schedule=schedule,
        )
        self.register_buffer("Phi", basis.phi(grid).detach())  # (R, N)
        self.register_buffer(
            "grid_mean", basis.mean(grid).reshape(-1).detach().contiguous()
        )  # (N,)

    def to_display(self, f: torch.Tensor) -> torch.Tensor:
        """Map weights ``(B, C, R)`` to function values ``(B, C, *output_shape)``.

        Called by the sampler on its final output only, which is what keeps guidance in
        weight space (see the class docstring).
        """
        fn = f @ self.Phi + self.grid_mean  # (B, C, R) @ (R, N) + (N,)
        return fn.reshape(fn.shape[0], fn.shape[1], *self.output_shape)


class GaussianIncrementFlow(GaussianFlow):
    """
    SDE mapping Gaussian increments to Gaussian increments via VP noise schedule. A map function is
    required to convert the increments to the data space.
    """

    # The map is nonlinear in general, so GuidanceMC decodes every draw.
    affine_data_space: bool = False

    def __init__(self, map: BaseMap, device="cpu", schedule: VPSchedule | None = None):
        # The diffusion runs in the map's *latent* space (iid Gaussian increments);
        # the map expands those to its own output space, which may be a different width.
        super().__init__(
            latent_dim=map.latent_dim,
            output_dim=map.output_dim,
            output_shape=map.output_shape,
            channels=map.channels,
            device=device,
            schedule=schedule,
            dtype=map.dtype,
        )
        self.map = map

    def _to_data_space(self, z: torch.Tensor) -> torch.Tensor:
        return self.map(z)


class NeuralFlow(BaseFlow):
    """
    A neural flow (i.e. network-based vector field), for VP SDEs.

    Time convention: t=1 is noise, t=0 is data (matches sampler's backward integration)
    The neural network directly predicts the velocity field v(x_t, t) for the ODE.
    """

    def __init__(
        self,
        H: int,
        W: int,
        neural_network: nn.Module,
        schedule: VPSchedule | None = None,
    ):
        param = next(neural_network.parameters())
        dtype = param.dtype
        device = param.device
        if dtype in (torch.float64, torch.complex128):
            warnings.warn(
                f"NeuralFlow was given a {dtype} network. The rest of the package "
                "defaults to float64 for the analytic linear algebra, but a learned "
                "velocity field gains nothing from it and trains/samples several "
                "times slower. Build the network under float32 "
                "(`net.float()`, or construct it before `torch.set_default_dtype"
                "(torch.float64)`) unless you need double precision here.",
                stacklevel=2,
            )
        # The network diffuses directly in image (data) space, so latent and output
        # widths coincide at H*W. Grid shape is a NeuralFlow-specific attribute.
        super().__init__(
            latent_dim=H * W,
            output_dim=H * W,
            output_shape=(H, W),
            channels=1,  # the network maps (B, 1, H, W) images
            device=device,
            dtype=dtype,
            schedule=schedule,
        )
        self.H = H
        self.W = W
        self.neural_network = neural_network

    @torch.no_grad()
    def __call__(self, x_t: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Velocity for the sampler — no_grad, since integration needs no graph."""
        return self._velocity(x_t, t)

    def _velocity(self, x_t: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Network velocity for a ``(B, C, N)`` state, with autograd intact.

        ``t`` is a scalar (sampling) or a per-sample ``(B,)`` tensor (training).
        :meth:`predict_terminal` must call this rather than :meth:`__call__`, whose
        ``no_grad`` would drop the network's Jacobian from DPS/MPGD guidance.
        """
        B = x_t.shape[0]
        if not (isinstance(t, torch.Tensor) and t.ndim > 0):
            t = torch.full((B,), float(t), dtype=x_t.dtype, device=x_t.device)
        t = t.to(device=x_t.device, dtype=x_t.dtype)
        v = self.neural_network(x_t.view(B, self.channels, self.H, self.W), t)
        return v.reshape(B, self.channels, self.output_dim)

    def fit(
        self,
        training_sampler: TrainingSampler,
        num_steps: int = 10_000,
        batch_size: int = 16,
        lr: float = 1e-3,
        use_snr_weighting: bool = True,
        min_snr_gamma: float = 5.0,
        verbose: bool = True,
    ) -> None:
        """Fit the vector field.

        Args:
            use_snr_weighting: Apply min-SNR weighting to balance learning across noise levels
            min_snr_gamma: Maximum SNR value for clipping (default 5.0)
            verbose: Show a tqdm progress bar with the running loss (default True)
        """

        assert self.H == training_sampler.H and self.W == training_sampler.W, (
            "Training sampler grid size must match vector field grid size."
        )

        optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=lr)

        self.neural_network.train()
        pbar = tqdm(range(num_steps), disable=not verbose, desc="training flow")
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            # Sample data from the target distribution (what we want at t=0).
            # The sampler yields (B, C, H*W); reshape to the (B, C, H, W) image
            # layout the network operates on.
            # Cast to the network's dtype: the analytic maps default to float64
            # while a learned velocity field is normally float32, so the training
            # data and the network routinely disagree.
            x_data = (
                training_sampler(batch_size)
                .to(self.dtype)
                .view(batch_size, self.channels, self.H, self.W)
            )

            # Sample noise from N(0, I)
            eps = torch.randn_like(x_data, device=self.device, dtype=self.dtype)

            # Uniform t, so the whole trajectory the sampler traverses is trained —
            # including both ends, where it starts from noise and resolves detail.
            t = torch.rand(batch_size, device=self.device, dtype=self.dtype)
            t = t.clamp(
                min=1e-10, max=1.0 - 1e-10
            )  # clamp to avoid numerical issues at the endpoints

            # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
            t_bc = t.view(batch_size, *([1] * (x_data.ndim - 1)))

            # VP forward process: x_t = sqrt(α_t) * x_data + sqrt(1 - α_t) * ε
            alpha_2 = self.schedule._alpha_2(t_bc)
            alpha_t = torch.sqrt(alpha_2)
            # expm1-based, so sigma_t stays finite for the small t this uniform draw
            # reaches; naive 1 - alpha_2 hits exactly 0 in float32 and the score below
            # would blow up to ~1/1e-8.
            sigma_t = torch.sqrt(self.schedule._one_minus_alpha_2(t_bc))
            x_t = alpha_t * x_data + sigma_t * eps

            # Target velocity for the probability-flow ODE:
            #   v = -0.5 * β(t) * (x_t + score),  score = -ε / sqrt(1 - α_t)
            beta_t = self.schedule._beta(t_bc)
            score = -eps / sigma_t.clamp(min=1e-8)
            v_target = -0.5 * beta_t * (x_t + score)

            # Model prediction via the flow's velocity path (network predicts the
            # velocity). Returns (B, C, output_dim); flatten the target to match.
            v_pred = self._velocity(x_t, t)
            v_target = v_target.reshape(batch_size, self.channels, self.output_dim)

            # Compute loss with optional SNR weighting
            if use_snr_weighting:
                # Compute per-sample MSE loss: (B,)
                mse_loss = ((v_pred - v_target) ** 2).mean(
                    dim=list(range(1, v_pred.ndim))
                )

                # Get alpha^2(t) and compute SNR = alpha^2 / (1 - alpha^2)
                alpha_2 = self.schedule._alpha_2(t)  # (B,)
                snr = alpha_2 / self.schedule._one_minus_alpha_2(t).clamp(min=1e-12)

                # Min-SNR weighting (Hang et al. 2023) for a *velocity* target:
                # min(SNR, γ)/(SNR+1), not the x0-prediction weight min(SNR, γ),
                # which would crush the high-noise end the sampler starts from.
                weights = torch.minimum(snr, torch.full_like(snr, min_snr_gamma)) / (
                    snr + 1.0
                )

                # Weighted loss
                loss = (mse_loss * weights).mean()
            else:
                # Standard uniform loss
                loss = ((v_pred - v_target) ** 2).mean()

            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.neural_network.parameters(), max_norm=1.0
            )

            optimizer.step()

            if step % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        self.neural_network.eval()
        logger.debug("Finished training neural flow.")

    def predict_terminal(self, x_t: torch.Tensor, t: float) -> torch.Tensor:
        """Tweedie estimate ``E[x_0 | x_t]`` from the network's velocity.

        The probability-flow velocity is ``v = -0.5 beta (x_t + score)`` with
        ``score = -(x_t - alpha x_0) / sigma^2``. Solving for ``x_0``, and using
        ``1 - sigma^2 = alpha^2``, gives ``x_0 = (alpha^2 x_t - 2 sigma^2 v / beta) / alpha``.
        """
        v = self._velocity(x_t, t)  # autograd-preserving, so guidance can differentiate
        alpha_sq = self.schedule._alpha_2(t)
        sigma_sq = self.schedule._one_minus_alpha_2(t)
        beta = self.schedule._beta(t)
        alpha = torch.sqrt(alpha_sq).clamp(min=1e-8)
        return (alpha_sq * x_t - 2.0 * sigma_sq * v / beta) / alpha
