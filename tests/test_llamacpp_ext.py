"""Tests for llama.cpp C extension — topology-aware allocation at kernel level."""

import numpy as np
import pytest

from turbomoq.llamacpp_ext.wrapper import (
    LlamaCppExtension,
    EXTENSION_AVAILABLE,
    compile_extension,
)


@pytest.fixture(scope="module")
def ext():
    """Compile and load the C extension for this test module."""
    if not compile_extension(force=True):
        pytest.skip("C extension compilation failed")
    try:
        return LlamaCppExtension(n_layers=4, n_heads=4, head_dim=64)
    except RuntimeError:
        pytest.skip("C extension not available")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ── Compilation ──────────────────────────────────────────────────────────────

class TestCompilation:
    def test_compile(self):
        assert compile_extension(force=True) is True

    def test_version(self, ext):
        ver = ext.version()
        assert ver == "0.2.0"


# ── Scoring ──────────────────────────────────────────────────────────────────

class TestScoring:
    def test_score_synthetic(self, ext):
        scores = ext.score_synthetic(seed=42)
        assert len(scores) == 16  # 4 layers * 4 heads
        for s in scores:
            assert 0 <= s.combined <= 1.5  # weighted sum can exceed 1.0

    def test_position_importance(self, ext):
        mid = ext.position_importance(2)  # middle of 4 layers
        edge = ext.position_importance(0)
        assert mid > edge

    def test_deterministic(self, ext):
        s1 = ext.score_synthetic(seed=42)
        s2 = ext.score_synthetic(seed=42)
        for a, b in zip(s1, s2):
            assert abs(a.combined - b.combined) < 1e-5


# ── Bit Allocation ───────────────────────────────────────────────────────────

class TestAllocation:
    def test_allocate_bits(self, ext):
        ext.score_synthetic(seed=42)
        alloc = ext.allocate_bits()
        assert len(alloc) == 16
        for (l, h), (kb, vb) in alloc.items():
            assert 1 <= kb <= 8
            assert 1 <= vb <= 8

    def test_get_bits(self, ext):
        ext.score_synthetic(seed=42)
        ext.allocate_bits()
        k, v = ext.get_bits(0, 0)
        assert 1 <= k <= 8
        assert 1 <= v <= 8

    def test_non_uniform_allocation(self, ext):
        """Different heads should get different bit allocations."""
        ext.score_synthetic(seed=42)
        alloc = ext.allocate_bits()
        bits_set = set(kb for kb, vb in alloc.values())
        # With 16 heads, we expect at least 2 different bit allocations
        assert len(bits_set) >= 2


# ── Symmetric Quantize / Dequantize ─────────────────────────────────────────

class TestQuantization:
    def test_roundtrip_4bit(self, ext, rng):
        data = rng.standard_normal((16, 64)).astype(np.float32)
        q, scales = ext.symmetric_quantize(data, bits=4)
        recon = ext.symmetric_dequantize(q, scales, bits=4)
        cos = _cosine(data.ravel(), recon.ravel())
        assert cos > 0.95, f"4-bit cosine {cos}"

    def test_roundtrip_2bit(self, ext, rng):
        data = rng.standard_normal((16, 64)).astype(np.float32)
        q, scales = ext.symmetric_quantize(data, bits=2)
        recon = ext.symmetric_dequantize(q, scales, bits=2)
        cos = _cosine(data.ravel(), recon.ravel())
        assert cos > 0.50, f"2-bit cosine {cos}"

    def test_1d_input(self, ext, rng):
        data = rng.standard_normal(64).astype(np.float32)
        q, scales = ext.symmetric_quantize(data, bits=4)
        assert q.shape == (1, 64)


# ── Rotation ─────────────────────────────────────────────────────────────────

class TestRotation:
    def test_generate_rotation(self, ext):
        Q = ext.generate_rotation(32, seed=42)
        assert Q.shape == (32, 32)
        # Check approximate orthogonality
        I = Q @ Q.T
        err = np.max(np.abs(I - np.eye(32)))
        assert err < 0.1  # C MGS is less precise than LAPACK QR

    def test_rotate_roundtrip(self, ext, rng):
        dim = 32
        Q = ext.generate_rotation(dim, seed=42)
        data = rng.standard_normal((8, dim)).astype(np.float32)
        rotated = ext.rotate(data, Q)
        # Unrotate = rotate with Q^T
        # The C ext has unrotate function, test via numpy
        unrotated = rotated @ Q.T
        np.testing.assert_allclose(unrotated, data, atol=0.05)

    def test_rotation_deterministic(self, ext):
        Q1 = ext.generate_rotation(16, seed=123)
        Q2 = ext.generate_rotation(16, seed=123)
        np.testing.assert_array_equal(Q1, Q2)


# ── Integration: Score → Allocate → Quantize ────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self, ext, rng):
        """Score → allocate → quantize with per-head bits."""
        ext.score_synthetic(seed=42)
        alloc = ext.allocate_bits()

        data = rng.standard_normal((32, 64)).astype(np.float32)

        # Use allocation for head (0, 0)
        k_bits, v_bits = alloc[(0, 0)]
        q, scales = ext.symmetric_quantize(data, bits=k_bits)
        recon = ext.symmetric_dequantize(q, scales, bits=k_bits)

        cos = _cosine(data.ravel(), recon.ravel())
        assert cos > 0.40  # Even at low bits should have some quality


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
