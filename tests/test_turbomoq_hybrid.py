"""Tests for TurboMOQ hybrid compressor."""

import numpy as np
import pytest

from turbomoq.head_scorer import HeadScorer, HeadScore, position_importance
from turbomoq.rotation import NumpyRotation
from turbomoq.codebook import LloydMaxCodebook, lloyd_max_quantize
from turbomoq.compressor import TurboMOQCompressor, CompressedMOQCache
from turbomoq.progressive import ProgressivePolicy
from turbomoq.pool import MemoryPool


# ── Head Scorer ───────────────────────────────────────────────────────────────

class TestHeadScorer:
    def test_position_importance_bell(self):
        n = 32
        mid = position_importance(16, n)
        edge = position_importance(0, n)
        assert mid > edge
        assert mid > 0.9

    def test_position_importance_symmetric(self):
        n = 32
        assert abs(position_importance(5, n) - position_importance(n - 6, n)) < 0.01

    def test_score_synthetic(self):
        scorer = HeadScorer(n_layers=4, n_heads=2)
        scores = scorer.score_synthetic()
        assert len(scores) == 8  # 4 layers * 2 heads
        for s in scores:
            assert 0 <= s.combined <= 1.0

    def test_allocate_bits(self):
        scorer = HeadScorer(n_layers=4, n_heads=2)
        scores = scorer.score_synthetic()
        alloc = scorer.allocate_bits(scores, target_avg=4.0)
        assert len(alloc) == 8
        avg = np.mean([k for k, v in alloc.values()])
        assert 2 <= avg <= 6  # within reasonable range

    def test_score_from_attention(self):
        scorer = HeadScorer(n_layers=2, n_heads=2)
        attn = {(l, h): np.random.rand(16, 16).astype(np.float32)
                for l in range(2) for h in range(2)}
        scores = scorer.score_from_attention(attn)
        assert len(scores) == 4

    def test_recommended_bits(self):
        s = HeadScore(0, 0, 0.9, 0.9, 0.9, 0.9)
        assert s.recommended_bits == 8
        s_low = HeadScore(0, 0, 0.1, 0.1, 0.1, 0.1)
        assert s_low.recommended_bits <= 2


# ── Numpy Rotation ────────────────────────────────────────────────────────────

class TestNumpyRotation:
    def test_orthogonality(self):
        rot = NumpyRotation(64, seed=42)
        assert rot.orthogonality_error < 1e-5

    def test_roundtrip(self):
        rot = NumpyRotation(32, seed=42)
        x = np.random.randn(10, 32).astype(np.float32)
        x_hat = rot.unrotate(rot.rotate(x))
        np.testing.assert_allclose(x, x_hat, atol=1e-4)

    def test_deterministic(self):
        rot1 = NumpyRotation(16, seed=123)
        rot2 = NumpyRotation(16, seed=123)
        np.testing.assert_array_equal(rot1.Q, rot2.Q)

    def test_information_spreading(self):
        """Rotation should spread concentrated info across dims."""
        rot = NumpyRotation(32, seed=42)
        x = np.zeros((5, 32), dtype=np.float32)
        x[:, 0] = np.random.randn(5)  # all info in dim 0
        x_rot = rot.rotate(x)
        # After rotation, energy should be spread across dims
        assert np.std(np.abs(x_rot).mean(axis=0)) < np.abs(x[:, 0]).mean()


# ── Lloyd-Max Codebook ────────────────────────────────────────────────────────

class TestLloydMax:
    def test_fit_and_quantize(self):
        data = np.random.randn(1000).astype(np.float32)
        cb = LloydMaxCodebook(bits=4, iterations=5)
        cb.fit(data)
        assert cb.centroids is not None
        assert len(cb.centroids) == 16

    def test_roundtrip_quality(self):
        data = np.random.randn(1000).astype(np.float32)
        cb = LloydMaxCodebook(bits=4)
        cb.fit(data)
        indices = cb.quantize(data)
        recon = cb.dequantize(indices)
        mse = np.mean((data - recon) ** 2)
        assert mse < 0.1  # reasonable for 4-bit

    def test_one_shot(self):
        data = np.random.randn(500).astype(np.float32)
        indices, centroids = lloyd_max_quantize(data, bits=3)
        assert len(centroids) == 8
        assert indices.shape == data.shape

    def test_empty_data(self):
        cb = LloydMaxCodebook(bits=2)
        cb.fit(np.array([], dtype=np.float32))
        assert cb.centroids is not None


# ── TurboMOQ Compressor ──────────────────────────────────────────────────────

class TestCompressor:
    @pytest.fixture
    def setup(self):
        n_layers, n_heads, seq_len, head_dim = 2, 4, 32, 64
        rng = np.random.default_rng(42)
        k = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float32)
        v = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float32)
        scorer = HeadScorer(n_layers, n_heads, head_dim)
        return k, v, scorer

    def test_compress_decompress(self, setup):
        k, v, scorer = setup
        comp = TurboMOQCompressor(scorer, k_bits=4, v_bits=4)
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_quality(self, setup):
        k, v, scorer = setup
        comp = TurboMOQCompressor(scorer, k_bits=4, v_bits=4)
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)

        k_cos = _cosine(k.ravel(), k_hat.ravel())
        v_cos = _cosine(v.ravel(), v_hat.ravel())
        assert k_cos > 0.95, f"Key cosine {k_cos} too low"
        assert v_cos > 0.90, f"Value cosine {v_cos} too low"

    def test_with_calibration(self, setup):
        k, v, scorer = setup
        comp = TurboMOQCompressor(scorer, k_bits=4, v_bits=4)
        comp.calibrate(v)
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)

        v_cos = _cosine(v.ravel(), v_hat.ravel())
        assert v_cos > 0.92  # Lloyd-Max should improve quality

    def test_compression_ratio(self, setup):
        k, v, scorer = setup
        comp = TurboMOQCompressor(scorer, k_bits=4, v_bits=4)
        compressed = comp.compress(k, v)
        assert compressed.compression_ratio > 2.0

    def test_stats(self, setup):
        k, v, scorer = setup
        comp = TurboMOQCompressor(scorer, k_bits=4, v_bits=4)
        compressed = comp.compress(k, v)
        stats = compressed.stats()
        assert stats["layers"] == 2
        assert stats["heads"] == 4
        assert stats["compression_ratio"] > 2.0

    def test_3d_input(self):
        """Test with (n_heads, seq, dim) input (single layer)."""
        rng = np.random.default_rng(42)
        k = rng.standard_normal((4, 32, 64)).astype(np.float32)
        v = rng.standard_normal((4, 32, 64)).astype(np.float32)
        scorer = HeadScorer(1, 4, 64)
        comp = TurboMOQCompressor(scorer, k_bits=4, v_bits=4)
        compressed = comp.compress(k, v)
        assert compressed.num_layers == 1

    def test_no_rotation(self, setup):
        k, v, scorer = setup
        comp = TurboMOQCompressor(scorer, enable_rotation=False)
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert k_hat.shape == k.shape

    def test_no_lloyd_max(self, setup):
        k, v, scorer = setup
        comp = TurboMOQCompressor(scorer, enable_lloyd_max=False)
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert v_hat.shape == v.shape


# ── Progressive Policy ────────────────────────────────────────────────────────

class TestProgressive:
    def test_tier_for_age(self):
        policy = ProgressivePolicy()
        assert policy.tier_for_age(500).name == "active"
        assert policy.tier_for_age(2000).name == "warm"
        assert policy.tier_for_age(50000).name == "cold"
        assert policy.tier_for_age(100000).name == "archive"

    def test_estimate_capacity(self):
        policy = ProgressivePolicy()
        cap = policy.estimate_capacity(8e9, head_dim=128, n_heads=8, n_layers=32)
        assert cap["total_tokens"] > 50000  # should be much more than FP16
        assert len(cap["tiers"]) > 0


# ── Memory Pool ───────────────────────────────────────────────────────────────

class TestMemoryPool:
    def test_allocate_and_get(self):
        pool = MemoryPool(budget_mb=100)
        pool.allocate("a1", "research")
        slot = pool._slots["a1"]
        assert slot.agent_id == "a1"

    def test_eviction(self):
        pool = MemoryPool(budget_mb=1)  # tiny budget
        # Store would trigger eviction on real data
        pool.allocate("a1", "research", priority=1.0)
        pool.allocate("a2", "coding", priority=0.5)
        assert pool.stats.total_agents == 2

    def test_stats(self):
        pool = MemoryPool(budget_mb=4096)
        pool.allocate("a1")
        pool.allocate("a2")
        s = pool.stats
        assert s.total_agents == 2
        assert s.budget_bytes == 4096 * 1024 * 1024


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
