"""Classical baselines for benchmarking guided flows, kept outside the library.

Every method here works on the same target as the guided samplers: the whitened
innovations of a flow with an ``N(0, I)`` prior, reweighted by a condition's
likelihood. Nothing in ``flow_inference`` imports from this package.

* :mod:`benchmarks.mcmc` — MCMC and tempered-SMC samplers (pCN, pCN-MALA, elliptical
  slice, HMC, NUTS, likelihood-tempered SMC);
* :mod:`benchmarks.ep` — expectation propagation, a deterministic Gaussian
  approximation;
* :mod:`benchmarks.lin_ess` — exact reference answers for linearly constrained
  Gaussians (LIN-ESS draws, subset-simulation evidence);
* :mod:`benchmarks.innovation_space` — a script comparing them with guided sampling,
  run as ``python -m benchmarks.innovation_space``.
"""
