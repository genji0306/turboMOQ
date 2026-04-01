"""Tests for MLX backend — Metal-accelerated rotation, quantization, codebook."""

import numpy as np
import pytest

# MLX is optional — skip tests if not installed
mlx_available = False
try:
    import mlx.core as mx
    mlx_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not mlx_available, reason="mlx not installed")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ── MLXRotation ──────────────────────────────────────────────────────────────

class TestMLXRotation:
    def test_orthogonality(self):
        from turbomoq.mlx_backend import MLXRotation
        rot = MLXRotation(64, seed=42)
        assert rot.orthogonality_error < 1e-5

    def test_roundtrip(self, rng):
        from turbomoq.mlx_backend import MLXRotation
        rot = MLXRotation(32, seed=42)
        x = rng.standard_normal((10, 32)).astype(np.float32)
        x_mx = mx.array(x)
        x_rot = rot.rotate(x_mx)
        x_hat = rot.unrotate(x_rot)
        mx.eval(x_hat)
        np.testing.assert_allclose(np.array(x_hat), x, atol=1e-4)

    def test_deterministic(self):
        from turbomoq.mlx_backend import MLXRotation
        r1 = MLXRotation(16, seed=123)
        r2 = MLXRotation(16, seed=123)
        np.testing.assert_array_equal(r1.Q_numpy, r2.Q_numpy)

    def test_numpy_interop(self, rng):
        """rotate_numpy / unrotate_numpy should accept numpy and return numpy."""
        from turbomoq.mlx_backend import MLXRotation
        rot = MLXRotation(32, seed=42)
        x = rng.standard_normal((5, 32)).astype(np.float32)
        x_rot = rot.rotate_numpy(x)
        assert isinstance(x_rot, np.ndarray)
        x_hat = rot.unrotate_numpy(x_rot)
        np.testing.assert_allclose(x_hat, x, atol=1e-4)

    def test_matches_numpy_rotation(self, rng):
        """MLX and numpy rotations with same seed should produce same result."""
        from turbomoq.mlx_backend import MLXRotation
        from turbomoq.rotation import NumpyRotation
        dim = 64
        mlx_rot = MLXRotation(dim, seed=42)
        np_rot = NumpyRotation(dim, seed=42)
        x = rng.standard_normal((8, dim)).astype(np.float32)

        mlx_result = mlx_rot.rotate_numpy(x)
        np_result = np_rot.rotate(x)
        np.testing.assert_allclose(mlx_result, np_result, atol=1e-4)


# ── MLXQuantizer ─────────────────────────────────────────────────────────────

class TestMLXQuantizer:
    def test_roundtrip(self, rng):
        from turbomoq.mlx_backend import MLXQuantizer
        q = MLXQuantizer(bits=4)
        data = rng.standard_normal((16, 64)).astype(np.float32)
        data_mx = mx.array(data)
        quantized, scales = q.quantize(data_mx)
        recon = q.dequantize(quantized, scales)
        mx.eval(recon)
        recon_np = np.array(recon)
        cos = _cosine(data.ravel(), recon_np.ravel())
        assert cos > 0.95, f"4-bit roundtrip cosine {cos} too low"

    def test_2bit_compression(self, rng):
        from turbomoq.mlx_backend import MLXQuantizer
        q = MLXQuantizer(bits=2)
        data = rng.standard_normal((16, 64)).astype(np.float32)
        quantized, scales = q.quantize(mx.array(data))
        recon = q.dequantize(quantized, scales)
        mx.eval(recon)
        recon_np = np.array(recon)
        cos = _cosine(data.ravel(), recon_np.ravel())
        assert cos > 0.60, f"2-bit roundtrip cosine {cos} too low"

    def test_1d_input(self, rng):
        from turbomoq.mlx_backend import MLXQuantizer
        q = MLXQuantizer(bits=4)
        data = rng.standard_normal(64).astype(np.float32)
        quantized, scales = q.quantize(mx.array(data))
        assert quantized.shape[1] == 64


# ── MLXCodebook ──────────────────────────────────────────────────────────────

class TestMLXCodebook:
    def test_fit_and_quantize(self, rng):
        from turbomoq.mlx_backend import MLXCodebook
        data = rng.standard_normal(1000).astype(np.float32)
        cb = MLXCodebook(bits=4, iterations=5)
        cb.fit(mx.array(data))
        assert cb.centroids is not None
        assert cb.centroids.shape[0] == 16

    def test_roundtrip_quality(self, rng):
        from turbomoq.mlx_backend import MLXCodebook
        data = rng.standard_normal(1000).astype(np.float32)
        cb = MLXCodebook(bits=4)
        cb.fit(mx.array(data))
        indices = cb.quantize(mx.array(data))
        recon = cb.dequantize(indices)
        mx.eval(recon)
        mse = np.mean((data - np.array(recon).ravel()) ** 2)
        assert mse < 0.1

    def test_empty_data(self):
        from turbomoq.mlx_backend import MLXCodebook
        cb = MLXCodebook(bits=2)
        cb.fit(mx.array(np.array([], dtype=np.float32)))
        assert cb.centroids is not None


# ── Backend Info ─────────────────────────────────────────────────────────────

class TestBackendInfo:
    def test_get_backend_info(self):
        from turbomoq.mlx_backend import get_backend_info
        info = get_backend_info()
        assert info.name == "mlx"
        assert info.device == "gpu"
        assert info.available is True


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
