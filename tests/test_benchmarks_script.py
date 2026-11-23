"""Smoke test for the benchmark script (``benchmarks.innovation_space``).

Every method runs end to end on a tiny conjugate problem and reports what the results
table needs.
"""

import math

from benchmarks.innovation_space import METHODS, conjugate_gp_problem, run_all


def test_every_method_runs_on_a_tiny_problem():
    flow, condition, (mean, _, log_evidence) = conjugate_gp_problem(
        num_points=10, num_obs=3, noise=0.1
    )
    runs, per_draw, _ = run_all(
        flow,
        condition,
        num_samples=20,
        num_steps=30,
        num_mc_samples=1,
        num_chains=2,
        num_leapfrog=3,
        smc_moves=2,
        smc_ess_moves=1,
        nuts_burn_in=5,
    )
    assert set(runs) == set(METHODS)
    assert per_draw > 0
    for name, run in runs.items():
        assert "skipped" not in run, name
        assert run["samples"].shape[0] == 20, name
        assert run["samples"].reshape(20, -1).shape[1] == mean.numel(), name
    # EP is exact for Gaussian observations.
    assert math.isclose(runs["EP"]["log_evidence"], log_evidence, abs_tol=1e-3)
