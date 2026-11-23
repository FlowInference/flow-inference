"""Flows: the generative dynamics the sampler integrates from noise to a process.

Each flow defines a probability-flow ODE under a VP noise schedule, running from
``t = 1`` (standard normal noise) to ``t ~ 0`` (a sample of the process), together
with the Tweedie estimate ``E[z_0 | z_t]`` that guidance uses to steer it.

* :class:`GaussianProcessFlow`: a GP on a grid, from a kernel and mean.
* :class:`BasisGaussianProcessFlow`: a GP written as a finite basis expansion
  (see :class:`~flow_inference.gp_utils.AbstractBasis`).
* :class:`GaussianIncrementFlow`: a non-Gaussian process given as a deterministic map
  of iid Gaussians (see :mod:`flow_inference.maps`).
* :class:`NeuralFlow`: a learned velocity field, trained with :meth:`NeuralFlow.fit`.

The Gaussian flows are closed form: they diffuse in a *whitened* latent ``z`` whose
marginals are ``N(0, I)`` at every ``t``, and map it to the process (``f = L z + m``
for a GP) only when a condition is evaluated or a sample is returned.
"""

import logging
import math
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from flow_inference.flow_utils import float64_copy
from flow_inference.maps import BaseMap
from flow_inference.schedules import LinearSchedule, VPSchedule
from flow_inference.training_samplers import TrainingSampler

logger = logging.getLogger(__name__)


class BaseFlow(nn.Module, ABC):
    """Abstract base class for a flow under a VP noise schedule.

    Subclasses define the drift of the probability-flow ODE (:meth:`__call__`) and the
    Tweedie estimate of its end point (:meth:`predict_terminal`). Time runs from
    ``t = 1`` (noise) down to ``t ~ 0`` (data).

    A flow has the same dimensions as a :class:`~flow_inference.maps.BaseMap`, so the
    sampler knows what latent to draw and the caller knows what comes back:

    - ``latent_dim``: iid ``N(0, 1)`` entries per channel that the diffusion runs in.
      The sampler draws ``(num_samples, channels, latent_dim)``.
    - ``output_dim``: data-space entries per channel, after :meth:`_to_data_space`.
      Equal to ``latent_dim`` for a GP on a grid; for an increment flow it is the
      map's output width.
    - ``channels``: how many fields are stacked on the channel axis.

    Args:
        latent_dim: latent entries per channel.
        output_dim: data-space entries per channel.
        channels: number of channels.
        output_shape: per-channel shape of a returned sample, e.g. ``(H, W)``.
            Defaults to the flat ``(output_dim,)``.
        dtype: working precision of the flow.
        device: device of the flow's tensors.
        schedule: the VP noise schedule. Defaults to a linear ``beta`` schedule.
    """

    # True when p(z_t) = N(0, I) for every t, as for the closed-form Gaussian flows.
    # The MCMC baselines in benchmarks/ rely on it.
    has_standard_normal_marginals: bool = False

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        channels: int = 1,
        output_shape: tuple[int, ...] | None = None,
        dtype=torch.float32,
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
        # The diffusion and conditions work on the flat (B, C, output_dim) field; only
        # to_display applies this shape.
        if output_shape is None:
            output_shape = (output_dim,)
        if math.prod(output_shape) != output_dim:
            raise ValueError(
                f"output_shape {tuple(output_shape)} has "
                f"{math.prod(output_shape)} cells, != output_dim {output_dim}"
            )
        self.output_shape = tuple(output_shape)

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
        """Drift ``v(z, t)`` of the probability-flow ODE for a ``(B, C, latent_dim)``
        state."""
        raise NotImplementedError

    @abstractmethod
    def predict_terminal(self, z_t: torch.Tensor, t: float) -> torch.Tensor:
        """Tweedie estimate ``E[z_0 | z_t]`` of the end point, used by guidance.

        Must keep the autograd graph back to ``z_t``: gradient-based guidance
        differentiates through it.
        """
        raise NotImplementedError

    def _to_data_space(self, z: torch.Tensor) -> torch.Tensor:
        """Map a diffusion state to the field that conditions are evaluated on.

        The identity by default. Flows that diffuse in a latent space (a whitened GP,
        the increments of a map) override it.
        """
        return z

    def to_display(self, f: torch.Tensor) -> torch.Tensor:
        """Reshape a flat ``(B, C, output_dim)`` field to ``(B, C, *output_shape)``.

        A no-op for 1D fields and paths; gives ``(B, C, H, W)`` for a grid. The
        sampler applies it to the samples it returns.
        """
        return f.reshape(f.shape[0], f.shape[1], *self.output_shape)


def terminal_draws(
    flow: BaseFlow, z_t: torch.Tensor, t: float, noise: torch.Tensor
) -> torch.Tensor:
    """Draws from ``p(z_0 | z_t) = N(sqrt(alpha_t) z_t, (1 - alpha_t) I)``.

    Args:
        flow: a flow with standard normal marginals.
        z_t: current state ``(B, C, N)``.
        t: current time.
        noise: standard normal noise ``(B, S, C, N)``. Passed in rather than drawn,
            so the caller chooses between fresh noise and the flow's frozen noise.

    Returns:
        ``(B, S, C, N)`` draws.
    """
    # _one_minus_alpha_2 rather than 1 - _alpha_2: the subtraction is exactly zero in
    # float32 for t <~ 1e-5, which would collapse every draw onto the mean.
    std = torch.sqrt(flow.schedule._one_minus_alpha_2(t).clamp(min=0.0))
    return flow.predict_terminal(z_t, t).unsqueeze(1) + std * noise


class GaussianFlow(BaseFlow):
    """Closed-form flow whose target is the standard normal ``N(0, I)``.

    Noise and target have the same distribution, so the unconditional drift is zero
    and every marginal is ``N(0, I)``: unguided, the flow returns iid Gaussians.
    Subclasses (the GP, basis and increment flows) diffuse in this whitened latent and
    turn it into a process by overriding :meth:`_to_data_space` (or, for the basis
    flow, :meth:`to_display`).

    The terminal posterior ``p(z_0 | z_t)`` is Gaussian with a known covariance, so
    besides the Tweedie mean the flow can draw from it exactly
    (:meth:`sample_terminal`), which is what
    :class:`~flow_inference.guidance.GuidanceMC` uses.

    Args:
        latent_dim: latent entries per channel.
        channels: number of channels.
        output_shape: per-channel shape of a returned sample.
        device: device of the flow's tensors.
        dtype: working precision of the flow.
        schedule: the VP noise schedule.
        output_dim: data-space entries per channel. Defaults to ``latent_dim``;
            subclasses that decode the latent pass their own.
    """

    has_standard_normal_marginals: bool = True
    # True when _to_data_space is affine, f = A z + b (see _linear_part / _pullback),
    # which lets GuidanceMC decode once per particle rather than once per draw.
    # Subclasses with a nonlinear map set this to False.
    affine_data_space: bool = True

    def __init__(
        self,
        latent_dim: int,
        channels: int = 1,
        output_shape: tuple[int, ...] | None = None,
        device="cpu",
        dtype=torch.float32,
        schedule: VPSchedule | None = None,
        output_dim: int | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            output_dim=latent_dim if output_dim is None else output_dim,
            channels=channels,
            output_shape=output_shape,
            device=device,
            dtype=dtype,
            schedule=schedule,
        )
        # Frozen reparameterisation noise for sample_terminal (see _frozen_noise).
        self.z = None
        # A @ self.z, cached for sample_terminal_data; recomputed when self.z changes.
        self._z_decoded = None
        self._z_decoded_src = None

    def __call__(self, z: torch.Tensor, t: float) -> torch.Tensor:
        """Zero drift: the target is already ``N(0, I)``."""
        return torch.zeros_like(z)

    def predict_terminal(self, z_t: torch.Tensor, t: float) -> torch.Tensor:
        """Tweedie estimate ``E[z_0 | z_t] = sqrt(alpha_t) z_t``."""
        return torch.sqrt(self.schedule._alpha_2(t)) * z_t

    def _linear_part(self, z: torch.Tensor) -> torch.Tensor:
        """``A z`` for the affine ``_to_data_space(z) = A z + b``. Identity here."""
        return z

    def _pullback(self, g: torch.Tensor) -> torch.Tensor:
        """``A^T g``: a data-space gradient ``(B, C, output_dim)`` as a latent one."""
        return g

    # Exact terminal draws: available because p(z_0 | z_t) is known in closed form.

    def _frozen_noise(self, batch_size: int, num_samples: int) -> torch.Tensor:
        """Cached ``(B, S, C, latent_dim)`` reparameterisation noise.

        Redrawn only when the batch size or sample count changes.
        """
        if (
            self.z is None
            or self.z.shape[1] != num_samples
            or self.z.shape[0] != batch_size
        ):
            self.z = torch.randn(
                (batch_size, num_samples, self.channels, self.latent_dim),
                device=self.device,
                dtype=self.dtype,
            )
        return self.z

    def sample_terminal(
        self, z_t: torch.Tensor, t: float, num_samples: int
    ) -> torch.Tensor:
        """Draw from the terminal posterior ``p(z_0 | z_t)`` by reparameterisation.

        Under the VP forward process the posterior is
        ``N(sqrt(alpha_t) z_t, (1 - alpha_t) I)``. The noise is frozen (reused across
        calls), so :class:`~flow_inference.guidance.GuidanceMC` sees gradients through
        the mean only.

        Args:
            z_t: current state ``(B, C, latent_dim)``.
            t: current time.
            num_samples: draws ``S`` per particle.

        Returns:
            ``(B, S, C, latent_dim)`` draws.
        """
        noise = self._frozen_noise(z_t.shape[0], num_samples)
        return terminal_draws(self, z_t, t, noise)

    def sample_terminal_data(
        self, z_t: torch.Tensor, t: float, num_samples: int
    ) -> torch.Tensor:
        """:meth:`sample_terminal`'s draws, mapped to data space.

        Equal to applying ``_to_data_space`` to each draw, but cheaper for an affine
        map: ``A (mu + s eps) + b = (A mu + b) + s (A eps)``, so the posterior mean is
        decoded once per particle and ``A eps`` is cached with the frozen noise. For a
        GP flow that is ``B`` dense ``N x N`` matmuls per step rather than ``B * S``.

        Args:
            z_t: current state ``(B, C, latent_dim)``.
            t: current time.
            num_samples: draws ``S`` per particle.

        Returns:
            ``(B, S, C, output_dim)`` draws.

        Raises:
            NotImplementedError: if the flow's ``_to_data_space`` is not affine.
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


def _whiten_factors(target_covar: torch.Tensor, jitter: float) -> torch.Tensor:
    """Eigen square root ``L`` of a ``(C, N, N)`` covariance ``K``, with ``L L^T = K``.

    ``K + jitter I = U diag(lam) U^T`` gives ``L = U diag(sqrt(lam))``. Eigenvalues are
    clamped at ``jitter``, since rounding can leave them tiny or negative. Pass a
    float64 covariance: a float32 ``eigh`` of a GP Gram matrix reproduces ``K`` only
    to ~1e-5 (see :func:`~flow_inference.flow_utils.float64_copy`).

    Warns if an eigenvalue is negative beyond rounding, i.e. the covariance is not
    positive semidefinite.
    """
    C, N = target_covar.shape[0], target_covar.shape[-1]
    jitter_mat = jitter * torch.eye(
        N, device=target_covar.device, dtype=target_covar.dtype
    ).unsqueeze(0).expand(C, N, N)
    lambdas, U = torch.linalg.eigh(target_covar + jitter_mat)  # (C, N), (C, N, N)
    # Rounding leaves eigenvalues of a PSD matrix at most ~N eps |lam_max| below zero;
    # anything clearly beyond that means the covariance itself is not PSD (an invalid
    # kernel, or a posterior kernel with inconsistent data), and clamping would hide it.
    lam_max = lambdas.amax(-1)
    worst = (lambdas.amin(-1) / lam_max).min().item()
    tolerance = 100 * N * torch.finfo(lambdas.dtype).eps
    if worst < -tolerance:
        warnings.warn(
            f"the covariance on the grid is not positive semidefinite: its smallest "
            f"eigenvalue is {worst:.1e} times its largest. Negative eigenvalues are "
            "clamped to zero, which changes the process; check the kernel.",
            stacklevel=3,
        )
    lambdas = lambdas.clamp(min=jitter)
    return U @ torch.diag_embed(torch.sqrt(lambdas))  # (C, N, N)


def _default_dtype(kernel) -> torch.dtype:
    """The kernel's floating dtype (its first parameter), else torch's default."""
    first = kernel[0] if isinstance(kernel, (list, tuple)) else kernel
    if isinstance(first, nn.Module):
        for p in first.parameters():
            if p.is_floating_point():
                return p.dtype
    return torch.get_default_dtype()


class GaussianProcessFlow(GaussianFlow):
    """Flow for a Gaussian process on a fixed grid.

    The kernel and mean are evaluated on the grid once, at construction. The flow then
    diffuses in *whitened* coordinates ``z``, where the GP is a standard normal, and
    maps to the process with ``f = L z + m``, where ``L L^T = K`` is the kernel matrix
    and ``m`` the mean on the grid. Conditions see ``f``.

    Example:
        >>> mean, kernel = gp_posterior(rbf, X_train, y_train, noise=1e-4)
        >>> flow = GaussianProcessFlow(kernel, mean, grid)
        >>> samples = IntegratorSampler(flow).sample(16, guidance=guidance)

    The kernel matrix is evaluated and factorised in float64 whatever ``dtype`` is,
    since it is badly conditioned; only the stored factor and mean are cast to
    ``dtype``. The whitened dynamics are well conditioned, so sampling in float32 is
    safe.
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
        dtype: torch.dtype | None = None,
    ):
        """Build the flow from a kernel and mean function evaluated on ``grid``.

        For a GP posterior, :func:`~flow_inference.gp_utils.gp_posterior` returns the
        ``(mean, kernel)`` pair this expects.

        Several independent GP channels can be given in three ways:

        * sequences: ``GaussianProcessFlow([k1, k2], [m1, m2], grid)``;
        * a kernel with a gpytorch ``batch_shape``, whose Gram matrix is already
          ``(C, N, N)``, with a matching batched mean or one shared mean;
        * a single kernel and ``channels=C``, which replicates it.

        Args:
            kernel: any callable ``k(x, x') -> (N, N)``, such as a
                ``gpytorch.kernels.Kernel`` or a
                :class:`~flow_inference.gp_utils.PosteriorKernel`; or a sequence of
                them, or a batched kernel, for several channels.
            mean: any callable ``m(x) -> (N,)``, such as a ``gpytorch.means.Mean`` or a
                :class:`~flow_inference.gp_utils.PosteriorMean`; or a sequence.
            grid: the ``(N, D)`` points the field lives on.
            channels: replicate a single kernel and mean across this many channels.
                Leave as None when the kernel already supplies them.
            jitter: added to the diagonal of the kernel matrix, and the floor on its
                eigenvalues.
            output_shape: per-channel shape of a returned sample, e.g. ``(H, W)``.
            device: device of the flow's tensors.
            schedule: the VP noise schedule.
            dtype: working precision of the flow (its buffers, latents and samples).
                Defaults to the kernel's dtype, or torch's default dtype for a kernel
                without parameters.

        Raises:
            ValueError: if the kernel and mean give different numbers of channels.
        """
        grid = grid.to(device)
        dtype = _default_dtype(kernel) if dtype is None else dtype
        # Evaluate on float64 copies; the results are cast to dtype below.
        covar, mu = self._evaluate(
            float64_copy(kernel)
            if not isinstance(kernel, (list, tuple))
            else [float64_copy(k) for k in kernel],
            float64_copy(mean)
            if not isinstance(mean, (list, tuple))
            else [float64_copy(m) for m in mean],
            grid.to(torch.float64),
            channels,
        )  # (C, N, N), (C, N)
        # One whitened latent per grid point: latent_dim == output_dim == N.
        super().__init__(
            latent_dim=covar.shape[-1],
            channels=covar.shape[0],
            output_shape=output_shape,
            device=device,
            dtype=dtype,
            schedule=schedule,
        )
        self.jitter = jitter
        # .contiguous(): gpytorch's ZeroMean/ConstantMean return an expanded view whose
        # elements share memory, which load_state_dict cannot copy into.
        self.register_buffer("target_mean", mu.to(dtype).contiguous())  # (C, N)
        self.register_buffer(
            "L", _whiten_factors(covar.to(torch.float64), jitter).to(dtype)
        )

    @staticmethod
    def _evaluate(kernel, mean, grid, channels):
        """Evaluate the kernel(s) and mean(s) on the grid.

        Returns:
            The covariance ``(C, N, N)`` and mean ``(C, N)``, with a single kernel or
            mean broadcast across channels.
        """

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
        """``L z`` per channel, ``(B, C, N) -> (B, C, N)``."""
        return torch.einsum("bcn,cmn->bcm", z, self.L)

    def _pullback(self, g: torch.Tensor) -> torch.Tensor:
        """``L^T g`` per channel, ``(B, C, N) -> (B, C, N)``."""
        return torch.einsum("bcm,cmn->bcn", g, self.L)

    def _to_data_space(self, z: torch.Tensor) -> torch.Tensor:
        """Unwhiten a ``(B, C, N)`` latent: ``f = L z + m``."""
        return self._linear_part(z) + self.target_mean.unsqueeze(0)


class BasisGaussianProcessFlow(GaussianFlow):
    """GP flow that diffuses over the ``R`` weights of a basis expansion.

    A basis (see :class:`~flow_inference.gp_utils.AbstractBasis`) writes draws as
    ``f = Phi^T w + m`` with ``w ~ N(0, I)``, so this is a GP flow parameterised by
    ``R`` weights rather than ``N`` grid values. The weights are already white, so the
    diffusion runs on them directly.

    Conditions are evaluated in **weight** space: ``_to_data_space`` is the identity,
    and the basis is applied only in :meth:`to_display`, to the samples the sampler
    returns. This lets a condition use a basis's closed-form derivatives (such as
    :meth:`~flow_inference.gp_utils.RFFBasis.compute_derivative`) rather than finite
    differences, while callers still get function values back.

    For conditions on the function values instead, wrap the basis in a
    :class:`~flow_inference.maps.BaseMap` and use :class:`GaussianIncrementFlow`, at
    the cost of the analytic derivatives.

    Example:
        >>> basis = RFFBasis(R=256, input_dim=2, mean=mean, kernel=kernel)
        >>> flow = BasisGaussianProcessFlow(basis, coords, output_shape=(H, W))
        >>> samples = IntegratorSampler(flow).sample(B, guidance=weight_guidance)
        >>> samples.shape  # (B, 1, H, W), function values
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
        a matmul.

        Args:
            basis: an :class:`~flow_inference.gp_utils.AbstractBasis`.
            grid: the ``(N, D)`` points the function samples live on.
            output_shape: per-channel shape of the returned function samples, e.g.
                ``(H, W)``. Defaults to the flat ``(N,)``.
            device: device of the flow's tensors.
            schedule: the VP noise schedule.

        Raises:
            ValueError: if ``output_shape`` does not match the grid size.
        """
        grid = grid.to(device=device, dtype=basis.dtype)
        if grid.ndim == 1:
            grid = grid.unsqueeze(-1)

        if output_shape is not None and math.prod(output_shape) != grid.shape[0]:
            raise ValueError(
                f"output_shape {tuple(output_shape)} has {math.prod(output_shape)} "
                f"cells, != the basis grid size {grid.shape[0]}"
            )
        # Latent: R weights. Output: the N grid values they expand to.
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
    """Flow for a process given as a deterministic map of iid Gaussians.

    The diffusion runs over the map's iid ``N(0, 1)`` inputs (its "increments"), and
    the map (a :class:`~flow_inference.maps.BaseMap`) turns them into the process, so
    conditions see the mapped field. The map may be nonlinear, which gives
    non-Gaussian processes: max-stable, Student-t, Cox and so on.

    Args:
        map: the map from latent increments to the process. Its ``latent_dim``,
            ``output_dim``, ``output_shape``, ``channels`` and dtype set the flow's.
        device: device of the flow's tensors.
        schedule: the VP noise schedule.
    """

    # The map is nonlinear in general, so GuidanceMC decodes every draw.
    affine_data_space: bool = False

    def __init__(self, map: BaseMap, device="cpu", schedule: VPSchedule | None = None):
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
        """Apply the map: ``(B, C, latent_dim) -> (B, C, output_dim)``."""
        return self.map(z)


class NeuralFlow(BaseFlow):
    """Flow with a learned velocity field, for images on an ``H x W`` grid.

    A network predicts the probability-flow velocity ``v(x_t, t)`` directly in data
    space. Train it on samples of the target with :meth:`fit`, then sample and guide
    it like any other flow.

    Example:
        >>> flow = NeuralFlow(H, W, net)
        >>> flow.fit(TrainingSampler(10_000, map), num_steps=5_000)
        >>> samples = IntegratorSampler(flow).sample(16, guidance=guidance)

    Args:
        H: grid height.
        W: grid width.
        neural_network: a module called as ``net(x, t)`` with ``x`` of shape
            ``(B, 1, H, W)`` and ``t`` of shape ``(B,)``, returning the velocity with
            the shape of ``x``. The flow takes its device and dtype from it; float32
            is recommended.
        schedule: the VP noise schedule.
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
                f"NeuralFlow was given a {dtype} network. The package works in "
                "float32 (float64 only inside its kernel-matrix setup), and a learned "
                "velocity field gains nothing from double precision while training "
                "and sampling several times slower. Build the network in float32 "
                "(`net.float()`) unless you need double precision here.",
                stacklevel=2,
            )
        # The network works in data space, so latent and output widths are both H*W.
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
        """Velocity for the sampler, without an autograd graph."""
        return self._velocity(x_t, t)

    def _velocity(self, x_t: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Network velocity for a ``(B, C, N)`` state, with autograd intact.

        ``t`` is a scalar (sampling) or a per-sample ``(B,)`` tensor (training).
        :meth:`predict_terminal` calls this rather than :meth:`__call__`, whose
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
        """Train the network to match the probability-flow velocity of the target.

        Each step draws a batch from ``training_sampler``, noises it to a uniform
        random time under the VP forward process, and regresses the network onto the
        exact velocity for that noise. Uses Adam with gradient-norm clipping at 1. The
        network is left in eval mode.

        Args:
            training_sampler: source of target samples on the same ``H x W`` grid.
            num_steps: optimiser steps.
            batch_size: samples per step.
            lr: Adam learning rate.
            use_snr_weighting: apply min-SNR loss weighting, to balance learning
                across noise levels.
            min_snr_gamma: the SNR at which the weighting is clipped.
            verbose: show a progress bar with the running loss.
        """
        assert self.H == training_sampler.H and self.W == training_sampler.W, (
            "Training sampler grid size must match vector field grid size."
        )

        optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=lr)

        self.neural_network.train()
        pbar = tqdm(range(num_steps), disable=not verbose, desc="training flow")
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            # The sampler yields (B, C, H*W); the network takes (B, C, H, W) images.
            # The cast matters: a map is often built in float64 while the network is
            # float32.
            x_data = (
                training_sampler(batch_size)
                .to(self.dtype)
                .view(batch_size, self.channels, self.H, self.W)
            )
            eps = torch.randn_like(x_data, device=self.device, dtype=self.dtype)

            # Uniform t trains the whole trajectory the sampler traverses, including
            # both ends; the clamp keeps the endpoints themselves out.
            t = torch.rand(batch_size, device=self.device, dtype=self.dtype)
            t = t.clamp(min=1e-10, max=1.0 - 1e-10)
            t_bc = t.view(batch_size, *([1] * (x_data.ndim - 1)))  # (B, 1, ..., 1)

            # VP forward process: x_t = sqrt(α_t) x_data + sqrt(1 - α_t) ε.
            alpha_2 = self.schedule._alpha_2(t_bc)
            alpha_t = torch.sqrt(alpha_2)
            # _one_minus_alpha_2 keeps sigma_t non-zero at small t, where the float32
            # subtraction 1 - alpha_2 is exactly 0 and the score would blow up.
            sigma_t = torch.sqrt(self.schedule._one_minus_alpha_2(t_bc))
            x_t = alpha_t * x_data + sigma_t * eps

            # Probability-flow velocity: v = -0.5 β(t) (x_t + score), with
            # score = -ε / sqrt(1 - α_t).
            beta_t = self.schedule._beta(t_bc)
            score = -eps / sigma_t.clamp(min=1e-8)
            v_target = -0.5 * beta_t * (x_t + score)

            v_pred = self._velocity(x_t, t)  # (B, C, output_dim)
            v_target = v_target.reshape(batch_size, self.channels, self.output_dim)

            if use_snr_weighting:
                mse_loss = ((v_pred - v_target) ** 2).mean(
                    dim=list(range(1, v_pred.ndim))
                )  # (B,)

                alpha_2 = self.schedule._alpha_2(t)  # (B,)
                snr = alpha_2 / self.schedule._one_minus_alpha_2(t).clamp(min=1e-12)

                # Min-SNR weighting (Hang et al. 2023) for a velocity target:
                # min(SNR, γ) / (SNR + 1). The x0-prediction weight min(SNR, γ) would
                # crush the high-noise end the sampler starts from.
                weights = torch.minimum(snr, torch.full_like(snr, min_snr_gamma)) / (
                    snr + 1.0
                )
                loss = (mse_loss * weights).mean()
            else:
                loss = ((v_pred - v_target) ** 2).mean()

            loss.backward()
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
        ``score = -(x_t - alpha x_0) / sigma^2``. Solving for ``x_0`` and using
        ``1 - sigma^2 = alpha^2`` gives
        ``x_0 = (alpha^2 x_t - 2 sigma^2 v / beta) / alpha``.
        """
        v = self._velocity(x_t, t)  # keeps the graph, so guidance can differentiate
        alpha_sq = self.schedule._alpha_2(t)
        sigma_sq = self.schedule._one_minus_alpha_2(t)
        beta = self.schedule._beta(t)
        alpha = torch.sqrt(alpha_sq).clamp(min=1e-8)
        return (alpha_sq * x_t - 2.0 * sigma_sq * v / beta) / alpha
