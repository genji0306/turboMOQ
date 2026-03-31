"""MOQ Head Scorer — topology-aware per-head importance scoring.

Three signals combined:
  1. Position importance: bell curve across layers (middle layers most important)
  2. Topological persistence: spectral gap of attention weight matrix
  3. Gradient sensitivity: Fisher information proxy via attention variance

Each head gets a combined score in [0, 1] that determines bit allocation.
Hub heads (global attention, induction heads) score high -> get more bits.
Peripheral heads (local attention, positional) score low -> aggressive compression.
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["HeadScorer", "HeadScore"]


@dataclass
class HeadScore:
    """Importance score for a single attention head."""
    layer: int
    head: int
    topo_score: float     # Topological persistence [0, 1]
    grad_score: float     # Gradient sensitivity [0, 1]
    position_score: float # Layer position bell curve [0, 1]
    combined: float       # Weighted combination [0, 1]

    @property
    def recommended_bits(self) -> int:
        """Map score to bit allocation (1-8)."""
        if self.combined > 0.8:
            return 8
        elif self.combined > 0.6:
            return 6
        elif self.combined > 0.4:
            return 4
        elif self.combined > 0.2:
            return 2
        return 1


def position_importance(layer_idx: int, n_layers: int) -> float:
    """Bell curve: middle layers are most important, edges least.

    This follows the empirical finding that middle transformer layers
    contain the most critical attention patterns (induction heads,
    composition heads), while early layers do embedding-like operations
    and late layers do projection-like operations.
    """
    if n_layers <= 1:
        return 1.0
    x = (layer_idx - (n_layers - 1) / 2) / ((n_layers - 1) / 2)
    return float(math.exp(-2 * x * x))


def compute_persistence(attn_weights: np.ndarray) -> float:
    """Topological persistence via spectral gap of attention matrix.

    Higher spectral gap = more structured attention = more important head.
    Falls back to variance-based estimate if eigendecomp fails.
    """
    try:
        if attn_weights.ndim == 1:
            return float(np.std(attn_weights))
        sym = (attn_weights + attn_weights.T) / 2
        eigs = np.linalg.eigvalsh(sym)
        eigs = np.sort(eigs)[::-1]
        if len(eigs) >= 2:
            gap = float(eigs[0] - eigs[1])
            return min(1.0, gap / (abs(eigs[0]) + 1e-10))
        return float(np.std(attn_weights))
    except Exception:
        return float(np.std(attn_weights))


def compute_gradient_sensitivity(attn_weights: np.ndarray) -> float:
    """Fisher information proxy via variance of attention weights.

    Higher variance = more informative attention = more sensitive to quantization.
    """
    return float(np.var(attn_weights))


class HeadScorer:
    """Score all attention heads for bit allocation.

    Usage:
        scorer = HeadScorer(n_layers=32, n_heads=8, head_dim=128)

        # From real attention weights (best)
        scores = scorer.score_from_attention(attention_maps)

        # From synthetic data (no model needed)
        scores = scorer.score_synthetic(seed=42)

        # Get bit allocation dict
        allocation = scorer.allocate_bits(scores, target_avg=4.0)
    """

    def __init__(self, n_layers: int, n_heads: int, head_dim: int = 128,
                 weights: tuple[float, float, float] = (0.4, 0.3, 0.3)):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.w_topo, self.w_grad, self.w_pos = weights

    def score_from_attention(self, attention_maps: dict[tuple[int, int], np.ndarray]) -> list[HeadScore]:
        """Score heads from real attention weight matrices.

        Args:
            attention_maps: {(layer, head): attn_matrix} where attn_matrix is (seq, seq) or (seq,)
        """
        scores = []
        max_topo = max_grad = 1e-10

        raw = {}
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                attn = attention_maps.get((l, h))
                if attn is None:
                    attn = np.eye(4, dtype=np.float32)
                topo = compute_persistence(attn)
                grad = compute_gradient_sensitivity(attn)
                raw[(l, h)] = (topo, grad)
                max_topo = max(max_topo, topo)
                max_grad = max(max_grad, grad)

        for l in range(self.n_layers):
            for h in range(self.n_heads):
                topo, grad = raw[(l, h)]
                topo_n = topo / max_topo
                grad_n = grad / max_grad
                pos = position_importance(l, self.n_layers)
                combined = self.w_topo * topo_n + self.w_grad * grad_n + self.w_pos * pos
                scores.append(HeadScore(l, h, topo_n, grad_n, pos, combined))

        return scores

    def score_synthetic(self, seed: int = 42) -> list[HeadScore]:
        """Generate scores without real attention data (uses position + random)."""
        rng = np.random.RandomState(seed)
        scores = []
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                pos = position_importance(l, self.n_layers)
                topo = rng.beta(2, 2)
                grad = rng.beta(2, 2)
                combined = self.w_topo * topo + self.w_grad * grad + self.w_pos * pos
                scores.append(HeadScore(l, h, topo, grad, pos, combined))
        return scores

    def allocate_bits(self, scores: list[HeadScore], target_avg: float = 4.0,
                      min_bits: int = 1, max_bits: int = 8) -> dict[tuple[int, int], tuple[int, int]]:
        """Allocate bits per head to hit target average.

        Returns: {(layer, head): (key_bits, value_bits)}
        """
        total_heads = len(scores)
        total_budget = target_avg * total_heads

        sorted_scores = sorted(scores, key=lambda s: s.combined, reverse=True)
        allocation = {}
        remaining = total_budget

        for i, hs in enumerate(sorted_scores):
            heads_left = total_heads - i
            avg_left = remaining / heads_left if heads_left > 0 else target_avg

            if hs.combined > 0.75:
                bits = min(max_bits, max(min_bits, round(avg_left * 1.5)))
            elif hs.combined > 0.5:
                bits = min(max_bits, max(min_bits, round(avg_left)))
            elif hs.combined > 0.25:
                bits = max(min_bits, round(avg_left * 0.7))
            else:
                bits = max(min_bits, round(avg_left * 0.5))

            bits = max(min_bits, min(max_bits, bits))
            allocation[(hs.layer, hs.head)] = (bits, bits)
            remaining -= bits

        return allocation
