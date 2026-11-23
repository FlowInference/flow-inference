"""A fixed pool of unconditional samples from a map, for training a neural flow."""

import logging

import torch

from flow_inference.maps import BaseMap

logger = logging.getLogger(__name__)


class TrainingSampler:
    """Draws training batches from a pool of samples generated once from a map.

    The pool is generated at construction by pushing standard normal latents through
    ``map``; each call then returns a random batch from it.

    Args:
        num_samples: size of the pool.
        map: the map to sample from; its ``output_shape`` must be a 2D grid
            ``(H, W)``.
        buffer_size: largest number of samples generated in one call to ``map``.
    """

    def __init__(self, num_samples: int, map: BaseMap, buffer_size: int = 100):
        if len(map.output_shape) != 2:
            raise ValueError(
                f"TrainingSampler needs a map on an H×W grid, but "
                f"{type(map).__name__} has output_shape {map.output_shape}"
            )
        self.map = map
        self.H, self.W = map.output_shape
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.num_samples = num_samples
        self.buffer_size = buffer_size
        # In chunks of at most buffer_size; the final chunk may be short.
        samples = []
        drawn = 0
        while drawn < num_samples:
            chunk = min(buffer_size, num_samples - drawn)
            z = torch.randn(chunk, map.latent_dim, device=self.device, dtype=self.dtype)
            logger.debug(
                "precomputing samples %d to %d of %d", drawn, drawn + chunk, num_samples
            )
            samples.append(map(z))
            drawn += chunk
        self.samples = torch.cat(samples, dim=0)

    @property
    def device(self) -> torch.device:
        """The map's device, so it follows a later ``map.to(...)``."""
        return self.map.device

    @property
    def dtype(self) -> torch.dtype:
        """The map's dtype, so latents are drawn in the precision it expects."""
        return self.map.dtype

    def __call__(self, batch_size):
        """A random batch from the pool, drawn with replacement.

        Args:
            batch_size: number of samples ``B``.

        Returns:
            Flat fields ``(B, C, H * W)``.
        """
        random_indices = torch.randint(0, self.samples.shape[0], (batch_size,))
        return self.samples[random_indices].to(self.device)
