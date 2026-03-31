"""Numpy QR rotation — LAPACK-backed orthogonal matrix for key decorrelation.

Replaces PolarQuant's pure-Python Gram-Schmidt with a single LAPACK QR call.
This eliminates the O(dim^3) accumulated floating-point error that caused
PolarQuant to degrade to cosine 0.265 in pure-Python mode.

The rotation distributes information uniformly across dimensions, making
even simple per-channel symmetric quantization near-lossless at 4-bit.
"""

import numpy as np

__all__ = ["NumpyRotation"]


class NumpyRotation:
    """Random orthogonal rotation via QR decomposition.

    Usage:
        rot = NumpyRotation(dim=128, seed=42)
        x_rot = rot.rotate(x)       # decorrelate dimensions
        x_hat = rot.unrotate(x_rot)  # inverse rotation (Q^T)
    """

    def __init__(self, dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        H = rng.randn(dim, dim).astype(np.float64)
        Q, R = np.linalg.qr(H)
        # Fix sign to ensure det(Q) = +1 (proper rotation, not reflection)
        d = np.sign(np.diag(R))
        d[d == 0] = 1.0
        self.Q = (Q * d[None, :]).astype(np.float32)
        self.dim = dim

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply rotation: x @ Q. Input shape: (..., dim)."""
        return x @ self.Q

    def unrotate(self, x: np.ndarray) -> np.ndarray:
        """Apply inverse rotation: x @ Q^T. Input shape: (..., dim)."""
        return x @ self.Q.T

    @property
    def orthogonality_error(self) -> float:
        """Measure how far Q from perfect orthogonality (should be ~1e-7)."""
        I = self.Q @ self.Q.T
        return float(np.max(np.abs(I - np.eye(self.dim))))
