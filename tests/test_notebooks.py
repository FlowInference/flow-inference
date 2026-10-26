"""Smoke tests for the example notebooks.

Each notebook is executed top to bottom with a hidden first cell that caps the
expensive library calls (integration steps, training steps, kernel-fitting
iterations, training-set size). That keeps a full run to seconds on a CPU while still
exercising every cell, so API changes that break an example fail here.

These check that the notebooks *run*, not that their results are good. They are
marked ``notebooks`` and excluded from the default run; use ``pytest -m notebooks``.
"""

import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = sorted(EXAMPLES.rglob("*.ipynb"))

# Prepended to every notebook. Patches the library before the notebook imports it.
SMOKE_PREAMBLE = """
import inspect
import functools

import flow_inference.flows as _flows
import flow_inference.gp_utils as _gp_utils
import flow_inference.sampler as _sampler
import flow_inference.training_samplers as _training_samplers


def _cap(owner, name, **caps):
    original = getattr(owner, name)
    signature = inspect.signature(original)

    @functools.wraps(original)
    def capped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for arg, limit in caps.items():
            bound.arguments[arg] = min(bound.arguments[arg], limit)
        return original(*bound.args, **bound.kwargs)

    setattr(owner, name, capped)


_cap(_sampler.IntegratorSampler, "sample", num_steps=10)
_cap(_flows.NeuralFlow, "fit", num_steps=5)
_cap(_gp_utils, "fit_kernel_hyperparameters", num_iters=5)
_cap(_training_samplers.TrainingSampler, "__init__", num_samples=32)
"""


def _run(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(SMOKE_PREAMBLE))
    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        # Run from the notebook's directory so relative data paths resolve.
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "path", NOTEBOOKS, ids=[str(p.relative_to(EXAMPLES)) for p in NOTEBOOKS]
)
def test_notebook_runs(path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")  # no display in CI
    if os.environ.get("NOTEBOOK_SMOKE_GPU") != "1":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # match the CPU-only CI runner
    _run(path)
