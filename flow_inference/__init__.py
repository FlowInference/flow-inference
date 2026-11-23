"""Guided flow-based sampling of stochastic processes under user-specified conditions.

A typical workflow has four parts:

- a **flow** (:mod:`flow_inference.flows`) describing the unconditional process, e.g.
  :class:`GaussianProcessFlow` for a Gaussian process on a grid;
- a **condition** (:mod:`flow_inference.conditions`), the likelihood
  ``log p(condition | x)`` of what the samples should satisfy, e.g.
  :class:`PointWiseCondition` for noisy observations;
- a **guidance** (:mod:`flow_inference.guidance`), e.g. :class:`GuidanceMC`, which
  turns the condition into a correction to the flow's drift;
- an :class:`IntegratorSampler` (:mod:`flow_inference.sampler`), which integrates the
  guided probability-flow ODE (or SDE) from noise to samples.

The main entry points are re-exported here, so ``from flow_inference import
GaussianProcessFlow, GuidanceMC, IntegratorSampler`` works; the submodules hold the
rest (maps, bases, schedules, helpers).
"""

from importlib.metadata import PackageNotFoundError, version

from flow_inference.conditions import (
    BaseCondition,
    BinaryCondition,
    EqualityCondition,
    PointWiseCondition,
    ProductCondition,
)
from flow_inference.flow_utils import build_grid_in_box
from flow_inference.flows import (
    BaseFlow,
    BasisGaussianProcessFlow,
    GaussianFlow,
    GaussianIncrementFlow,
    GaussianProcessFlow,
    NeuralFlow,
)
from flow_inference.gp_utils import fit_kernel_hyperparameters, gp_posterior
from flow_inference.guidance import GuidanceDPS, GuidanceMC, GuidanceMPGD
from flow_inference.sampler import IntegratorSampler

try:
    __version__ = version("flow_inference")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "BaseCondition",
    "BaseFlow",
    "BasisGaussianProcessFlow",
    "BinaryCondition",
    "EqualityCondition",
    "GaussianFlow",
    "GaussianIncrementFlow",
    "GaussianProcessFlow",
    "GuidanceDPS",
    "GuidanceMC",
    "GuidanceMPGD",
    "IntegratorSampler",
    "NeuralFlow",
    "PointWiseCondition",
    "ProductCondition",
    "__version__",
    "build_grid_in_box",
    "fit_kernel_hyperparameters",
    "gp_posterior",
]
