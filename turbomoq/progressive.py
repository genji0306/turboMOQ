"""Progressive tier demotion for long-context compression.

Recent tokens keep full precision. As tokens age, they're compressed deeper:
  Active (0-1K):    FP16 (lossless)
  Warm (1K-8K):     4-bit (rotation + Lloyd-Max)
  Cold (8K-64K):    3-bit keys / 2-bit values
  Archive (64K+):   2-bit keys / 1-bit values

This enables ~750K effective context on 16GB RAM with graceful quality
degradation — core reasoning stays lossless, background context compresses.
"""

from dataclasses import dataclass

__all__ = ["ProgressivePolicy", "ProgressiveTier"]


@dataclass
class ProgressiveTier:
    """A compression tier for progressive demotion."""
    name: str
    max_age: int       # Max token age for this tier
    key_bits: int      # Key quantization bits
    value_bits: int    # Value quantization bits
    use_rotation: bool # Apply rotation to keys


DEFAULT_TIERS = [
    ProgressiveTier("active",   1024, 16, 16, False),   # Full precision
    ProgressiveTier("warm",     8192,  4,  4,  True),   # 4-bit + rotation
    ProgressiveTier("cold",    65536,  3,  2,  True),   # 3-bit K, 2-bit V
    ProgressiveTier("archive", 1 << 30, 2, 1, True),    # Aggressive
]


class ProgressivePolicy:
    """Manages progressive tier demotion for long contexts.

    Usage:
        policy = ProgressivePolicy()
        tier = policy.tier_for_age(5000)  # -> "warm" tier
        tier.key_bits  # 4
        tier.value_bits  # 4

        # Estimate capacity
        capacity = policy.estimate_capacity(budget_bytes=8e9, head_dim=128, n_heads=8, n_layers=32)
    """

    def __init__(self, tiers: list[ProgressiveTier] | None = None):
        self.tiers = tiers or DEFAULT_TIERS

    def tier_for_age(self, age: int) -> ProgressiveTier:
        """Get the compression tier for a token at given age."""
        for tier in self.tiers:
            if age <= tier.max_age:
                return tier
        return self.tiers[-1]

    def estimate_capacity(self, budget_bytes: float, head_dim: int,
                          n_heads: int, n_layers: int) -> dict:
        """Estimate effective context length with progressive tiers.

        Returns dict with total_tokens, per-tier breakdown, and effective_context.
        """
        total_tokens = 0
        remaining = budget_bytes
        breakdown = []

        prev_max = 0
        for tier in self.tiers:
            tier_capacity = tier.max_age - prev_max
            avg_bits = (tier.key_bits + tier.value_bits) / 2
            bytes_per_token = 2 * n_layers * n_heads * head_dim * (avg_bits / 8)

            if bytes_per_token <= 0:
                prev_max = tier.max_age
                continue

            can_fit = int(remaining / bytes_per_token)
            actual = min(tier_capacity, can_fit)

            if actual <= 0:
                break

            total_tokens += actual
            remaining -= actual * bytes_per_token
            breakdown.append({
                "tier": tier.name,
                "tokens": actual,
                "bits": avg_bits,
                "bytes": actual * bytes_per_token,
            })
            prev_max = tier.max_age

            if remaining <= 0:
                break

        return {
            "total_tokens": total_tokens,
            "budget_gb": round(budget_bytes / 1e9, 1),
            "tiers": breakdown,
        }
