import logging

import torch

from flow_inference.maps import BaseMap

logger = logging.getLogger(__name__)


class TrainingSampler:
    def __init__(self, num_samples: int, map: BaseMap, buffer_size: int = 100):
        # The training target is an image on an H×W grid, so the map's output_shape
        # must be 2D.
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
        # Draw in chunks of at most buffer_size; the final chunk may be short.
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
        """Delegated to the map, so it follows a ``map.to(...)`` after construction."""
        return self.map.device

    @property
    def dtype(self) -> torch.dtype:
        """Delegated to the map, so latents are drawn in the precision it expects."""
        return self.map.dtype

    def __call__(self, batch_size):
        """
        Sample a batch of functions on an HxW grid.

        Args:
            batch_size: number of samples
        Returns:
            f: (B, C, H*W) — the flat field; callers reshape to (B, C, H, W)
        """
        random_indices = torch.randint(0, self.samples.shape[0], (batch_size,))
        return self.samples[random_indices].to(self.device)
