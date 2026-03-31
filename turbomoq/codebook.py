"""Lloyd-Max optimal codebook for value cache quantization.

Standard quantization uses uniformly spaced levels. Lloyd-Max iterates
k-means to place codebook entries where data density is highest,
minimizing reconstruction error for the actual data distribution.

For transformer value caches, distributions are typically peaked near zero
with long tails — Lloyd-Max concentrates quantization levels at the center
where most values live, yielding 1-2% cosine improvement over uniform grids.
"""

import numpy as np

__all__ = ["LloydMaxCodebook", "lloyd_max_quantize", "lloyd_max_dequantize"]


class LloydMaxCodebook:
    """Lloyd-Max optimal scalar quantizer.

    Usage:
        cb = LloydMaxCodebook(bits=4, iterations=5)
        cb.fit(calibration_data)

        indices = cb.quantize(values)
        reconstructed = cb.dequantize(indices)
    """

    def __init__(self, bits: int = 4, iterations: int = 5):
        self.bits = bits
        self.n_levels = 2 ** bits
        self.iterations = iterations
        self.centroids: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "LloydMaxCodebook":
        """Fit codebook to calibration data via iterative k-means.

        Args:
            data: 1D array of sample values from the distribution to quantize.
        """
        flat = data.ravel().astype(np.float32)
        if len(flat) == 0:
            self.centroids = np.zeros(self.n_levels, dtype=np.float32)
            return self

        # Initialize centroids with uniform spacing
        self.centroids = np.linspace(flat.min(), flat.max(), self.n_levels).astype(np.float32)

        for _ in range(self.iterations):
            # Assign to nearest centroid
            dists = np.abs(flat[:, None] - self.centroids[None, :])
            assignments = dists.argmin(axis=1)
            # Update centroids
            for i in range(self.n_levels):
                mask = assignments == i
                if mask.any():
                    self.centroids[i] = flat[mask].mean()

        # Sort centroids for monotonic quantization
        self.centroids.sort()
        return self

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize values to nearest centroid indices."""
        if self.centroids is None:
            raise RuntimeError("Codebook not fitted. Call fit() first.")
        flat = values.ravel().astype(np.float32)
        dists = np.abs(flat[:, None] - self.centroids[None, :])
        indices = dists.argmin(axis=1).astype(np.uint8)
        return indices.reshape(values.shape)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Reconstruct values from centroid indices."""
        if self.centroids is None:
            raise RuntimeError("Codebook not fitted. Call fit() first.")
        return self.centroids[indices.astype(int)]


def lloyd_max_quantize(values: np.ndarray, bits: int, iterations: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """One-shot quantize: returns (indices, centroids)."""
    cb = LloydMaxCodebook(bits, iterations)
    cb.fit(values)
    return cb.quantize(values), cb.centroids


def lloyd_max_dequantize(indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Dequantize from indices + centroids."""
    return centroids[indices.astype(int)]
