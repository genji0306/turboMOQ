"""Tests for QJL residual correction — sub-2-bit recovery."""

import numpy as np
import pytest

from turbomoq.qjl import QJLCorrector, QJLResidual, QJLConfig


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def corrector():
    return QJLCorrector(QJLConfig(head_dim=64, jl_dim=16, use_mlx=False))


# ── Basic Encode/Decode ──────────────────────────────────────────────────────

class TestQJLBasic:
    def test_encode_shape(self, corrector, rng):
        residual = rng.standard_normal((10, 64)).astype(np.float32)
        packed = corrector.encode(residual)
        assert packed.n_vectors == 10
        assert packed.jl_dim == 16
        assert packed.original_dim == 64

    def test_decode_shape(self, corrector, rng):
        residual = rng.standard_normal((10, 64)).astype(np.float32)
        packed = corrector.encode(residual)
        correction = corrector.decode(packed)
        assert correction.shape == (10, 64)

    def test_empty_input(self, corrector):
        packed = corrector.encode(np.zeros((0, 64), dtype=np.float32))
        assert packed.n_vectors == 0
        result = corrector.decode(packed)
        assert result.shape == (0, 64)

    def test_1d_input(self, corrector, rng):
        residual = rng.standard_normal(64).astype(np.float32)
        packed = corrector.encode(residual)
        assert packed.n_vectors == 1


# ── Quality Recovery ─────────────────────────────────────────────────────────

class TestQJLQuality:
    def test_correction_at_full_jl_dim(self, rng):
        """At jl_dim == head_dim, QJL recovers cosine for sign-quantized data.

        The sign-encoded JL projection recovers the residual direction best
        when jl_dim approaches head_dim (full-rank projection).
        We test with pure sign quantization (no scaling) where residual
        is the magnitude component.
        """
        head_dim = 64
        corrector = QJLCorrector(QJLConfig(
            head_dim=head_dim, jl_dim=head_dim, use_mlx=False))

        original = rng.standard_normal((50, head_dim)).astype(np.float32)

        # Pure sign quantization (no magnitude) — maximizes residual energy
        quantized = np.sign(original).astype(np.float32)
        residual = original - quantized

        correction = corrector.encode_decode(residual)
        corrected = quantized + correction

        cos_before = _cosine(original.ravel(), quantized.ravel())
        cos_after = _cosine(original.ravel(), corrected.ravel())
        assert cos_after > cos_before, (
            f"QJL@full_dim should improve cosine: {cos_before:.4f} -> {cos_after:.4f}"
        )

    def test_cosine_recovery_at_1bit(self, rng):
        """At 1-bit base quantization, QJL is most effective."""
        head_dim = 64
        corrector = QJLCorrector(QJLConfig(
            head_dim=head_dim, jl_dim=head_dim, use_mlx=False))

        original = rng.standard_normal((100, head_dim)).astype(np.float32)
        # Real 1-bit: sign only (no magnitude)
        quantized = np.sign(original).astype(np.float32)

        cos_before = _cosine(original.ravel(), quantized.ravel())
        residual = original - quantized
        correction = corrector.encode_decode(residual)
        corrected = quantized + correction
        cos_after = _cosine(original.ravel(), corrected.ravel())

        assert cos_after > cos_before, (
            f"Expected improvement: before={cos_before:.4f}, after={cos_after:.4f}"
        )

    def test_projection_preserves_direction(self, rng):
        """Even at small jl_dim, encode_decode should preserve residual direction."""
        head_dim = 64
        corrector = QJLCorrector(QJLConfig(
            head_dim=head_dim, jl_dim=32, use_mlx=False))

        residual = rng.standard_normal((50, head_dim)).astype(np.float32)
        reconstruction = corrector.encode_decode(residual)

        # The reconstruction should be positively correlated with the residual
        cos = _cosine(residual.ravel(), reconstruction.ravel())
        assert cos > 0.3, f"Direction should be preserved, got cosine {cos:.4f}"


# ── Memory Efficiency ────────────────────────────────────────────────────────

class TestQJLMemory:
    def test_memory_overhead(self, corrector, rng):
        """QJL should add ~1 bit per projected dimension."""
        residual = rng.standard_normal((100, 64)).astype(np.float32)
        packed = corrector.encode(residual)

        # 1 bit per jl_dim per vector + 4 bytes per scale
        expected_bits = 100 * 16  # n_vectors * jl_dim
        expected_bytes = expected_bits // 8 + 100 * 4  # + scales
        assert packed.memory_bytes <= expected_bytes * 1.1  # 10% tolerance

    def test_bits_per_dim(self, corrector, rng):
        residual = rng.standard_normal((100, 64)).astype(np.float32)
        packed = corrector.encode(residual)
        # jl_dim=16, head_dim=64 → ~0.25 bits/dim + scale overhead
        assert packed.bits_per_dim < 1.0


# ── Per-Vector Scale ─────────────────────────────────────────────────────────

class TestScaleCorrection:
    def test_per_vector_vs_global(self, rng):
        """Per-vector scale should be better than global scale."""
        head_dim = 64
        original = rng.standard_normal((50, head_dim)).astype(np.float32)
        # Make some vectors much larger than others
        original[0:10] *= 10.0

        corrector_pv = QJLCorrector(QJLConfig(
            head_dim=head_dim, jl_dim=16, use_mlx=False, scale_correction=True))
        corrector_gs = QJLCorrector(QJLConfig(
            head_dim=head_dim, jl_dim=16, use_mlx=False, scale_correction=False))

        pv_result = corrector_pv.encode_decode(original)
        gs_result = corrector_gs.encode_decode(original)

        pv_cos = _cosine(original.ravel(), pv_result.ravel())
        gs_cos = _cosine(original.ravel(), gs_result.ravel())
        # Per-vector should be at least as good
        assert pv_cos >= gs_cos - 0.01


# ── MLX Backend ──────────────────────────────────────────────────────────────

class TestQJLMLX:
    @pytest.fixture
    def mlx_corrector(self):
        try:
            import mlx.core as mx
            return QJLCorrector(QJLConfig(head_dim=64, jl_dim=16, use_mlx=True))
        except ImportError:
            pytest.skip("mlx not installed")

    def test_mlx_encode_decode(self, mlx_corrector, rng):
        residual = rng.standard_normal((10, 64)).astype(np.float32)
        packed = mlx_corrector.encode(residual)
        assert packed.n_vectors == 10

    def test_mlx_matches_numpy(self, rng):
        """MLX and numpy backends should produce compatible results."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not installed")

        head_dim = 64
        np_corrector = QJLCorrector(QJLConfig(head_dim=head_dim, jl_dim=16, use_mlx=False))
        mlx_corrector = QJLCorrector(QJLConfig(head_dim=head_dim, jl_dim=16, use_mlx=True))

        residual = rng.standard_normal((20, head_dim)).astype(np.float32)

        np_packed = np_corrector.encode(residual)
        mlx_packed = mlx_corrector.encode(residual)

        # Sign bits should match (same projection matrix)
        assert np_packed.sign_bits == mlx_packed.sign_bits

    def test_mlx_decode_correction(self, mlx_corrector, rng):
        original = rng.standard_normal((30, 64)).astype(np.float32)
        quantized = np.round(original * 2) / 2
        residual = original - quantized

        correction = mlx_corrector.encode_decode(residual)
        try:
            import mlx.core as mx
            if isinstance(correction, mx.array):
                mx.eval(correction)
                correction = np.array(correction)
        except ImportError:
            pass

        corrected = quantized + correction
        cos_before = _cosine(original.ravel(), quantized.ravel())
        cos_after = _cosine(original.ravel(), corrected.ravel())
        assert cos_after >= cos_before


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
