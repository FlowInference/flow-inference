"""Tests for TrainingSampler.

TrainingSampler pre-generates a fixed pool of unconditional field samples from a
map (drawing iid N(0,1) latents and pushing them through ``map``), then serves
random minibatches from that pool. It is the data source the NeuralFlow trains
against — there is no closed-form density, only these samples.
"""

import pytest
import torch

from flow_inference.training_samplers import TrainingSampler

H, W = 4, 4


class TestTrainingSampler:
    def test_grid_dims_from_map(self, make_map):
        ts = TrainingSampler(num_samples=12, map=make_map(H, W), buffer_size=6)
        assert ts.H == H and ts.W == W

    def test_device_from_map(self, make_map):
        m = make_map(H, W)
        ts = TrainingSampler(num_samples=12, map=m, buffer_size=6)
        assert ts.device == m.device

    def test_pool_size_matches_requested(self, make_map):
        ts = TrainingSampler(num_samples=12, map=make_map(H, W), buffer_size=6)
        assert ts.samples.shape[0] == 12

    def test_pool_exact_when_not_divisible(self, make_map):
        # The final chunk is short rather than dropped: flooring the division used to
        # silently lose the remainder (13 -> 10).
        ts = TrainingSampler(num_samples=13, map=make_map(H, W), buffer_size=5)
        assert ts.samples.shape[0] == 13

    def test_pool_smaller_than_buffer(self, make_map):
        # num_samples < buffer_size used to build an empty list and fail in torch.cat.
        ts = TrainingSampler(num_samples=3, map=make_map(H, W), buffer_size=100)
        assert ts.samples.shape[0] == 3

    def test_rejects_non_positive_num_samples(self, make_map):
        with pytest.raises(ValueError, match="num_samples must be positive"):
            TrainingSampler(num_samples=0, map=make_map(H, W))

    def test_batch_shape(self, make_map):
        # Field samples come back as (B, 1, H*W) — the map's per-sample layout.
        ts = TrainingSampler(num_samples=12, map=make_map(H, W), buffer_size=6)
        batch = ts(5)
        assert batch.shape == (5, 1, H * W)

    def test_batch_drawn_from_pool(self, make_map):
        # Every returned row must be one of the precomputed pool rows.
        ts = TrainingSampler(num_samples=12, map=make_map(H, W), buffer_size=6)
        batch = ts(8).reshape(8, -1)
        pool = ts.samples.reshape(ts.samples.shape[0], -1)
        for row in batch:
            assert (pool == row).all(dim=1).any()

    def test_batch_larger_than_pool_allowed(self, make_map):
        # Sampling is with replacement, so a batch may exceed the pool size.
        ts = TrainingSampler(num_samples=12, map=make_map(H, W), buffer_size=6)
        assert ts(50).shape == (50, 1, H * W)

    def test_batch_is_finite(self, make_map):
        ts = TrainingSampler(num_samples=12, map=make_map(H, W), buffer_size=6)
        assert torch.isfinite(ts(8)).all()

    def test_pool_matches_map_dtype(self, make_map):
        m = make_map(H, W)
        ts = TrainingSampler(num_samples=12, map=m, buffer_size=6)
        assert ts.samples.dtype == m.dtype
