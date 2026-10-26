import gpytorch
import pytest
import torch
import torch.nn as nn

from flow_inference.conditions import BinaryCondition, PointWiseCondition
from flow_inference.flows import (
    GaussianFlow,
    GaussianIncrementFlow,
    GaussianProcessFlow,
    NeuralFlow,
)
from flow_inference.maps import StudentTProcessMap
from flow_inference.schedules import LinearSchedule
from flow_inference.training_samplers import TrainingSampler

# The package defaults to float64 (see BaseFlow/BaseMap), so the tests run under the
# same ambient default the notebooks use. Without it every torch.randn(...) in a test
# is float32 and collides with the float64 buffers on the flows and maps.
torch.set_default_dtype(torch.float64)

N = 8  # spatial points — small enough to keep tests fast

# NeuralFlow operates on a square H×W grid; keep it tiny so the U-Net / training
# loop run in milliseconds.
NF_H = NF_W = 4


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_schedule():
    return LinearSchedule(
        beta_0=torch.tensor(0.0001),
        beta_1=torch.tensor(10.0),
    )


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


class IdentityKernel(gpytorch.kernels.Kernel):
    """k(x, x') = delta(x, x') — the identity Gram matrix on any grid.

    GaussianProcessFlow builds from a kernel, and no standard kernel gives exactly the
    identity on distinct points. Many tests want an identity covariance so that the
    whitened and data spaces coincide, which makes L and the data-space map trivial to
    assert against.
    """

    is_stationary = False

    def forward(self, x1, x2, diag=False, **params):
        same = (x1.unsqueeze(-2) == x2.unsqueeze(-3)).all(-1).to(x1.dtype)
        return same.diagonal(dim1=-2, dim2=-1) if diag else same


TEST_GRID = torch.linspace(0, 1, N).unsqueeze(-1)  # (N, 1)


@pytest.fixture
def gp_flow():
    """Single-channel GP flow with identity covariance and zero mean (whitened == data space)."""
    return GaussianProcessFlow(IdentityKernel(), gpytorch.means.ZeroMean(), TEST_GRID)


@pytest.fixture
def gp_flow_correlated():
    """Single-channel GP flow whose covariance has off-diagonal structure.

    An RBF with lengthscale sqrt(0.05) reproduces the exp(-d^2 / 0.1) matrix these
    tests were originally written against.
    """
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = 0.05**0.5
    return GaussianProcessFlow(kernel, gpytorch.means.ZeroMean(), TEST_GRID)


@pytest.fixture
def gaussian_flow():
    """Plain GaussianFlow (data space == latent space, zero drift)."""
    return GaussianFlow(latent_dim=N, channels=1)


def _small_map(H, W):
    """A cheap H×W grid map for tests that just need *a* map."""
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = 0.25
    return StudentTProcessMap(H, W, kernel)


@pytest.fixture
def make_map():
    """Factory for a cheap H×W grid map: ``make_map(H, W)``."""
    return _small_map


@pytest.fixture
def increment_flow():
    """GaussianIncrementFlow backed by a small Student-t map (3x3 grid)."""
    return GaussianIncrementFlow(map=_small_map(3, 3))


class TinyVectorField(nn.Module):
    """Minimal stand-in for the notebook U-Net: a single time-conditioned conv on
    the (B, 1, H, W) image. Enough to exercise every NeuralFlow code path (reshape
    in/out, scalar vs per-sample t, gradient flow in ``fit``) without the cost of a
    real U-Net. ``t`` may be a scalar-broadcastable or a (B,) tensor."""

    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W
        # float32, like a real learned velocity field — NeuralFlow warns otherwise.
        self.conv = nn.Conv2d(1, 1, 3, padding=1).float()

    def forward(self, x, t):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1, 1, 1, 1)
        return self.conv(x) + t


@pytest.fixture
def tiny_vector_field():
    return TinyVectorField(NF_H, NF_W)


@pytest.fixture
def neural_flow(tiny_vector_field):
    """NeuralFlow on a 4x4 grid wrapping the tiny conv vector field."""
    return NeuralFlow(H=NF_H, W=NF_W, neural_network=tiny_vector_field)


@pytest.fixture
def training_sampler():
    """TrainingSampler over a small grid map matching the NeuralFlow grid.

    ``buffer_size`` must divide ``num_samples`` and be ≤ it, else the precompute
    loop (range(num_samples // buffer_size)) runs zero times and leaves no samples.
    """
    return TrainingSampler(num_samples=12, map=_small_map(NF_H, NF_W), buffer_size=6)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


@pytest.fixture
def pointwise_condition():
    """Pin the first two points of an N-vector to zero."""
    target = torch.zeros(N)
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:2] = True
    return PointWiseCondition(target, mask, sigma=0.3)


class PositivityCondition(BinaryCondition):
    """Concrete BinaryCondition for tests: every entry should be > 0."""

    def smooth_eval_condition(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.fixture
def positivity_condition():
    return PositivityCondition(v=0.1)


# ---------------------------------------------------------------------------
# Kernels / maps
# ---------------------------------------------------------------------------


@pytest.fixture
def rbf_kernel():
    k = gpytorch.kernels.RBFKernel()
    k.lengthscale = 0.25
    return k
