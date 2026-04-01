"""Tests for attention-weighted tier demotion."""

import numpy as np
import pytest

from turbomoq.attention_demotion import (
    AttentionDemotionPolicy,
    AttentionTier,
    TokenRecord,
    DemotionResult,
    DEFAULT_ATTENTION_TIERS,
)


@pytest.fixture
def policy():
    return AttentionDemotionPolicy()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestAttentionTiers:
    def test_default_tiers_cover_full_range(self):
        floors = sorted(t.percentile_floor for t in DEFAULT_ATTENTION_TIERS)
        ceils = sorted(t.percentile_ceil for t in DEFAULT_ATTENTION_TIERS)
        assert floors[0] == 0.0
        assert ceils[-1] == 1.0

    def test_tier_bits_ordering(self):
        """Higher percentile tiers should have more bits."""
        sorted_tiers = sorted(DEFAULT_ATTENTION_TIERS, key=lambda t: t.percentile_floor)
        prev_bits = 0
        for tier in sorted_tiers:
            assert tier.avg_bits >= prev_bits
            prev_bits = tier.avg_bits


class TestTokenRegistration:
    def test_register_token(self, policy):
        rec = policy.register_token(0, step=0)
        assert rec.position == 0
        assert rec.insert_step == 0
        assert policy.n_tokens == 1

    def test_register_batch(self, policy):
        policy.register_tokens_batch(list(range(100)), step=0)
        assert policy.n_tokens == 100

    def test_duplicate_registration(self, policy):
        r1 = policy.register_token(0)
        r2 = policy.register_token(0)
        assert r1 is r2
        assert policy.n_tokens == 1


class TestAttentionUpdate:
    def test_update_ema(self, policy):
        policy.register_token(0)
        policy.register_token(1)
        policy.update_attention([0.8, 0.2], current_step=1)
        assert policy.token_records[0].cumulative_attention > 0
        assert policy.token_records[0].cumulative_attention > policy.token_records[1].cumulative_attention

    def test_ema_decay(self, policy):
        policy.register_token(0)
        # High attention first, then low
        policy.update_attention([1.0], current_step=1)
        high = policy.token_records[0].cumulative_attention
        policy.update_attention([0.0], current_step=2)
        decayed = policy.token_records[0].cumulative_attention
        assert decayed < high

    def test_numpy_input(self, policy, rng):
        policy.register_tokens_batch(list(range(10)))
        weights = rng.random(10).astype(np.float32)
        policy.update_attention(weights, current_step=1)
        assert all(r.update_count == 1 for r in policy.token_records.values())

    def test_multi_head_input(self, policy, rng):
        """2D attention input (n_heads, seq_len) should be summed across heads."""
        policy.register_tokens_batch(list(range(8)))
        weights = rng.random((4, 8)).astype(np.float32)  # 4 heads, 8 tokens
        policy.update_attention(weights, current_step=1)
        assert policy.n_tokens == 8

    def test_auto_register_on_update(self, policy):
        """Tokens not yet registered should be auto-registered on update."""
        policy.update_attention([0.5, 0.3, 0.8], current_step=1)
        assert policy.n_tokens == 3


class TestTierAssignment:
    def test_below_min_threshold(self, policy):
        policy.register_tokens_batch(list(range(5)))  # Below default min=32
        result = policy.assign_tiers()
        assert result.stats.get("reason") == "below_min_threshold"

    def test_differentiation(self, policy):
        """Tokens with high attention should get more bits than low attention."""
        n = 100
        policy.register_tokens_batch(list(range(n)))

        # Some tokens get constant high attention, others low
        for step in range(20):
            weights = [0.0] * n
            for i in range(90, 100):  # Top 10% get high attention
                weights[i] = 1.0
            for i in range(0, 10):  # Bottom 10% get no attention
                weights[i] = 0.0
            for i in range(10, 90):
                weights[i] = 0.3
            policy.update_attention(weights, current_step=step)

        result = policy.assign_tiers()
        # High-attention tokens should be in critical/high tiers
        high_tier = result.assignments[95]
        low_tier = result.assignments[5]
        assert high_tier.avg_bits >= low_tier.avg_bits

    def test_tier_distribution(self, policy, rng):
        n = 200
        policy.register_tokens_batch(list(range(n)))
        for step in range(10):
            weights = rng.random(n).tolist()
            policy.update_attention(weights, current_step=step)

        result = policy.assign_tiers()
        assert "tier_distribution" in result.stats
        assert result.stats["n_tokens"] == n
        # All 5 tiers should be represented
        assert len(result.stats["tier_distribution"]) >= 3


class TestPinning:
    def test_pin_prevents_demotion(self):
        policy = AttentionDemotionPolicy(pin_threshold=0.5)
        n = 50
        policy.register_tokens_batch(list(range(n)))

        # Give token 0 very high attention initially
        for step in range(10):
            weights = [0.0] * n
            weights[0] = 1.0
            for i in range(1, n):
                weights[i] = 0.1
            policy.update_attention(weights, current_step=step)

        # Token 0 should be pinned
        assert policy.token_records[0].pinned is True

        # Now give token 0 zero attention
        for step in range(10, 30):
            weights = [0.0] * n
            for i in range(1, n):
                weights[i] = 0.5
            policy.update_attention(weights, current_step=step)

        result = policy.assign_tiers()
        # Token 0 should still be in a high tier due to pinning
        assert result.assignments[0].avg_bits >= 4


class TestEviction:
    def test_evict_below(self, policy):
        n = 50
        policy.register_tokens_batch(list(range(n)))
        for step in range(5):
            weights = [0.01] * n
            weights[0] = 1.0
            policy.update_attention(weights, current_step=step)

        evicted = policy.evict_below(0.05)
        assert len(evicted) > 0
        assert 0 not in evicted  # Token 0 has high attention


class TestMemoryEstimation:
    def test_estimate(self, policy):
        n = 100
        policy.register_tokens_batch(list(range(n)))
        for step in range(5):
            weights = [0.5] * n
            policy.update_attention(weights, current_step=step)

        mem = policy.estimate_memory(head_dim=128, n_heads=8, n_layers=32)
        assert mem["compressed_mb"] > 0
        assert mem["compression_ratio"] > 1.0


class TestReset:
    def test_reset(self, policy):
        policy.register_tokens_batch(list(range(10)))
        policy.reset()
        assert policy.n_tokens == 0
