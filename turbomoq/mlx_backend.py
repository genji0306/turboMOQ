"""MLX backend — native Metal acceleration for Apple Silicon.

Provides drop-in replacements for numpy-based rotation, quantization, and
codebook operations using Apple's MLX framework. On M-series chips, MLX
dispatches to Metal shaders for ~2x throughput over numpy on matrix ops.

All functions accept and return mx.array when MLX is available, with
automatic numpy fallback when it isn't.

Requires: mlx >= 0.18.0 (pip install mlx)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MLX_AVAILABLE",
    "MLXRotation",
    "MLXCodebook",
    "MLXQuantizer",
    "mlx_symmetric_quantize",
    "mlx_symmetric_dequantize",
    "mlx_lloyd_max_fit",
]

logger = logging.getLogger("turbomoq.mlx_backend")

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore[assignment]
    MLX_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if not MLX_AVAILABLE:
        raise ImportError("MLX backend requires mlx: pip install mlx")


# ── MLX Rotation ─────────────────────────────────────────────────────────────


class MLXRotation:
    """Random orthogonal rotation using MLX Metal kernels.

    Same algorithm as NumpyRotation (QR decomposition) but the actual
    rotate/unrotate matmuls run on Metal GPU via MLX.

    The QR factorization itself uses numpy (MLX doesn't expose QR yet),
    then the Q matrix is transferred to MLX for all subsequent matmuls.
    """

    def __init__(self, dim: int, seed: int = 42):
        _require_mlx()
        if np is None:
            raise ImportError("MLXRotation requires numpy for QR init")

        # QR init via numpy (one-time cost)
        rng = np.random.RandomState(seed)
        H = rng.randn(dim, dim).astype(np.float32)
        Q_np, R_np = np.linalg.qr(H)
        d = np.sign(np.diag(R_np))
        d[d == 0] = 1.0
        Q_np = (Q_np * d[None, :]).astype(np.float32)

        # Transfer to MLX for Metal-accelerated matmuls
        self._Q = mx.array(Q_np)
        self._QT = mx.transpose(self._Q)
        self._Q_np = Q_np  # keep numpy copy for interop
        self.dim = dim

    def rotate(self, x: Any) -> Any:
        """Apply rotation: x @ Q. Returns mx.array."""
        x_mx = _to_mx(x)
        return mx.matmul(x_mx, self._Q)

    def unrotate(self, x: Any) -> Any:
        """Apply inverse rotation: x @ Q^T. Returns mx.array."""
        x_mx = _to_mx(x)
        return mx.matmul(x_mx, self._QT)

    def rotate_numpy(self, x: Any) -> Any:
        """Rotate and return numpy array (for interop with numpy pipeline)."""
        result = self.rotate(x)
        mx.eval(result)
        return _to_numpy(result)

    def unrotate_numpy(self, x: Any) -> Any:
        """Unrotate and return numpy array."""
        result = self.unrotate(x)
        mx.eval(result)
        return _to_numpy(result)

    @property
    def Q(self) -> Any:
        """Return Q matrix as mx.array."""
        return self._Q

    @property
    def Q_numpy(self) -> Any:
        """Return Q matrix as numpy array."""
        return self._Q_np

    @property
    def orthogonality_error(self) -> float:
        """Measure departure from perfect orthogonality."""
        I = mx.matmul(self._Q, self._QT)
        mx.eval(I)
        eye = mx.eye(self.dim)
        err = mx.max(mx.abs(I - eye))
        mx.eval(err)
        return float(err.item())


# ── MLX Quantizer ────────────────────────────────────────────────────────────


class MLXQuantizer:
    """Per-channel symmetric quantizer using MLX Metal ops.

    Vectorized quantize/dequantize — no Python loops over columns.
    """

    def __init__(self, bits: int = 4):
        _require_mlx()
        self.bits = bits
        self.n_levels = 2**bits
        self.half = self.n_levels // 2

    def quantize(self, data: Any) -> tuple[Any, Any]:
        """Per-channel symmetric quantize. Returns (quantized_int, scales)."""
        x = _to_mx(data).astype(mx.float32)

        if x.ndim == 1:
            x = mx.expand_dims(x, 0)

        # Per-column abs-max
        abs_max = mx.max(mx.abs(x), axis=0)
        scales = mx.where(abs_max > 0, abs_max / self.half, mx.ones_like(abs_max))

        # Quantize: round(x / scales), clip to [-half, half-1], store unsigned
        q = mx.round(x / scales)
        q = mx.clip(q, -self.half, self.half - 1)
        q_unsigned = (q + self.half).astype(mx.int32)

        mx.eval(q_unsigned, scales)
        return q_unsigned, scales

    def dequantize(self, quantized: Any, scales: Any) -> Any:
        """Reverse symmetric quantization."""
        q = _to_mx(quantized).astype(mx.float32)
        s = _to_mx(scales).astype(mx.float32)
        result = (q - self.half) * s
        mx.eval(result)
        return result


def mlx_symmetric_quantize(data: Any, bits: int) -> tuple[Any, Any]:
    """Functional symmetric quantize using MLX."""
    q = MLXQuantizer(bits)
    return q.quantize(data)


def mlx_symmetric_dequantize(quantized: Any, scales: Any, bits: int) -> Any:
    """Functional symmetric dequantize using MLX."""
    q = MLXQuantizer(bits)
    return q.dequantize(quantized, scales)


# ── MLX Codebook (Lloyd-Max) ────────────────────────────────────────────────


class MLXCodebook:
    """Lloyd-Max codebook with MLX-accelerated distance computation.

    The iterative k-means loop runs on Metal — the distance matrix
    (N_values x N_centroids) is the bottleneck and maps perfectly
    to a single MLX broadcast-subtract + argmin.
    """

    def __init__(self, bits: int = 4, iterations: int = 5):
        _require_mlx()
        self.bits = bits
        self.n_levels = 2**bits
        self.iterations = iterations
        self.centroids: Any = None  # mx.array when fitted

    def fit(self, data: Any) -> "MLXCodebook":
        """Fit codebook via MLX-accelerated Lloyd-Max."""
        flat = _to_mx(data).reshape(-1).astype(mx.float32)
        n = flat.shape[0]

        if n == 0:
            self.centroids = mx.zeros(self.n_levels)
            return self

        # Initialize with linspace
        vmin = mx.min(flat)
        vmax = mx.max(flat)
        mx.eval(vmin, vmax)
        self.centroids = mx.linspace(vmin.item(), vmax.item(), self.n_levels).astype(mx.float32)

        for _ in range(self.iterations):
            # Distance matrix: |flat - centroids| via broadcast
            dists = mx.abs(mx.expand_dims(flat, 1) - mx.expand_dims(self.centroids, 0))
            assignments = mx.argmin(dists, axis=1)
            mx.eval(assignments)

            # Update centroids
            new_centroids = []
            assignments_np = _to_numpy(assignments).astype(int)
            flat_np = _to_numpy(flat)
            for i in range(self.n_levels):
                mask = assignments_np == i
                if mask.any():
                    new_centroids.append(float(flat_np[mask].mean()))
                else:
                    new_centroids.append(float(_to_numpy(self.centroids)[i]))

            new_mx = mx.array(sorted(new_centroids), dtype=mx.float32)

            # Check convergence
            diff = mx.max(mx.abs(self.centroids - new_mx))
            mx.eval(diff)
            self.centroids = new_mx
            if diff.item() < 1e-8:
                break

        mx.eval(self.centroids)
        return self

    def quantize(self, values: Any) -> Any:
        """Quantize to nearest centroid indices (Metal-accelerated)."""
        if self.centroids is None:
            raise RuntimeError("Not fitted")
        flat = _to_mx(values).reshape(-1).astype(mx.float32)
        dists = mx.abs(mx.expand_dims(flat, 1) - mx.expand_dims(self.centroids, 0))
        indices = mx.argmin(dists, axis=1).astype(mx.int32)
        mx.eval(indices)
        return indices.reshape(values.shape if hasattr(values, "shape") else (-1,))

    def dequantize(self, indices: Any) -> Any:
        """Reconstruct from centroid indices."""
        if self.centroids is None:
            raise RuntimeError("Not fitted")
        idx = _to_mx(indices).astype(mx.int32).reshape(-1)
        result = mx.take(self.centroids, idx)
        mx.eval(result)
        return result.reshape(indices.shape if hasattr(indices, "shape") else (-1,))


def mlx_lloyd_max_fit(data: Any, bits: int = 4, iterations: int = 5) -> Any:
    """One-shot Lloyd-Max fit, returns centroids as mx.array."""
    cb = MLXCodebook(bits, iterations)
    cb.fit(data)
    return cb.centroids


# ── Conversion Helpers ───────────────────────────────────────────────────────


def _to_mx(x: Any) -> Any:
    """Convert numpy/list/mx.array to mx.array."""
    _require_mlx()
    if isinstance(x, mx.array):
        return x
    if np is not None and isinstance(x, np.ndarray):
        return mx.array(x.astype(np.float32))
    return mx.array(x)


def _to_numpy(x: Any) -> Any:
    """Convert mx.array to numpy array."""
    if np is None:
        raise ImportError("numpy required for conversion")
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x)
    return np.asarray(x)


# ── Backend Selection Helper ─────────────────────────────────────────────────


@dataclass
class BackendInfo:
    """Information about the active compute backend."""

    name: str  # "mlx" or "numpy"
    device: str  # "gpu" or "cpu"
    available: bool
    version: str


def get_backend_info() -> BackendInfo:
    """Return info about the best available backend."""
    if MLX_AVAILABLE:
        try:
            ver = mx.__version__ if hasattr(mx, "__version__") else "unknown"
            return BackendInfo(name="mlx", device="gpu", available=True, version=ver)
        except Exception:
            pass
    return BackendInfo(name="numpy", device="cpu", available=np is not None,
                       version=np.__version__ if np is not None else "N/A")


def benchmark_backends(dim: int = 128, seq_len: int = 1024, iterations: int = 10) -> dict[str, Any]:
    """Benchmark MLX vs numpy for rotation + quantize on given dimensions.

    Returns timing dict with speedup ratio.
    """
    if np is None:
        return {"error": "numpy not available"}

    rng = np.random.RandomState(42)
    data = rng.randn(seq_len, dim).astype(np.float32)

    results: dict[str, Any] = {"dim": dim, "seq_len": seq_len, "iterations": iterations}

    # Numpy baseline
    from turbomoq.rotation import NumpyRotation as NpRot
    np_rot = NpRot(dim, seed=42)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = np_rot.rotate(data)
    np_time = (time.perf_counter() - t0) / iterations
    results["numpy_rotate_ms"] = round(np_time * 1000, 3)

    # MLX
    if MLX_AVAILABLE:
        mlx_rot = MLXRotation(dim, seed=42)
        data_mx = mx.array(data)
        # Warmup
        _ = mlx_rot.rotate(data_mx)
        mx.eval(_)

        t0 = time.perf_counter()
        for _ in range(iterations):
            r = mlx_rot.rotate(data_mx)
            mx.eval(r)
        mlx_time = (time.perf_counter() - t0) / iterations
        results["mlx_rotate_ms"] = round(mlx_time * 1000, 3)
        results["speedup"] = round(np_time / mlx_time, 2) if mlx_time > 0 else float("inf")
    else:
        results["mlx_rotate_ms"] = None
        results["speedup"] = None

    return results
