"""TurboMOQ Compressor — hybrid KV cache compression.

Split K/V strategy:
  - Keys:   Numpy QR rotation -> per-channel symmetric quantize (decorrelation)
  - Values: Lloyd-Max codebook grids (hub-preservation)

Both paths use MOQ topology-aware bit allocation: hub heads get more bits,
peripheral heads get aggressive compression.
"""

import numpy as np
from dataclasses import dataclass, field

from turbomoq.rotation import NumpyRotation
from turbomoq.codebook import LloydMaxCodebook
from turbomoq.head_scorer import HeadScorer, HeadScore

__all__ = ["TurboMOQCompressor", "CompressedMOQCache"]


@dataclass
class CompressedHead:
    """Compressed KV data for a single head."""
    k_quantized: np.ndarray    # quantized key data
    k_scales: np.ndarray       # per-channel scales for key dequant
    v_quantized: np.ndarray    # quantized value indices
    v_centroids: np.ndarray    # Lloyd-Max centroids for value dequant
    k_bits: int
    v_bits: int
    seq_len: int
    rotated: bool = True

    @property
    def memory_bytes(self) -> int:
        k_bytes = int(np.ceil(self.k_quantized.size * self.k_bits / 8))
        v_bytes = int(np.ceil(self.v_quantized.size * self.v_bits / 8))
        scale_bytes = self.k_scales.nbytes + self.v_centroids.nbytes
        return k_bytes + v_bytes + scale_bytes


@dataclass
class CompressedMOQCache:
    """Container for a full compressed KV cache."""
    heads: dict[tuple[int, int], CompressedHead] = field(default_factory=dict)
    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0

    @property
    def memory_bytes(self) -> int:
        return sum(h.memory_bytes for h in self.heads.values())

    @property
    def fp16_bytes(self) -> int:
        return self.num_layers * self.num_heads * self.seq_len * self.head_dim * 2 * 2

    @property
    def compression_ratio(self) -> float:
        mem = self.memory_bytes
        return self.fp16_bytes / mem if mem > 0 else 0.0

    def stats(self) -> dict:
        avg_k = np.mean([h.k_bits for h in self.heads.values()]) if self.heads else 0
        avg_v = np.mean([h.v_bits for h in self.heads.values()]) if self.heads else 0
        return {
            "layers": self.num_layers,
            "heads": self.num_heads,
            "seq_len": self.seq_len,
            "head_dim": self.head_dim,
            "memory_mb": round(self.memory_bytes / 1e6, 2),
            "fp16_mb": round(self.fp16_bytes / 1e6, 2),
            "compression_ratio": round(self.compression_ratio, 2),
            "avg_k_bits": round(float(avg_k), 1),
            "avg_v_bits": round(float(avg_v), 1),
        }


class TurboMOQCompressor:
    """Hybrid KV cache compressor: rotation for keys, Lloyd-Max for values.

    Usage:
        scorer = HeadScorer(n_layers=32, n_heads=8, head_dim=128)
        compressor = TurboMOQCompressor(scorer, k_bits=4, v_bits=4)

        # Compress KV cache
        compressed = compressor.compress(k_cache, v_cache)

        # Decompress
        k_hat, v_hat = compressor.decompress(compressed)

        # Check quality
        print(compressed.stats())
    """

    def __init__(self, scorer: HeadScorer, k_bits: int = 4, v_bits: int = 4,
                 seed: int = 42, enable_rotation: bool = True,
                 enable_lloyd_max: bool = True, target_avg_bits: float | None = None):
        self.scorer = scorer
        self.default_k_bits = k_bits
        self.default_v_bits = v_bits
        self.seed = seed
        self.enable_rotation = enable_rotation
        self.enable_lloyd_max = enable_lloyd_max

        # Per-head rotations (lazy init)
        self._rotations: dict[tuple[int, int], NumpyRotation] = {}
        # Per-head value codebooks (lazy init)
        self._v_codebooks: dict[tuple[int, int], LloydMaxCodebook] = {}
        # Bit allocation (set via calibrate or manual)
        self._allocation: dict[tuple[int, int], tuple[int, int]] | None = None
        self._target_avg = target_avg_bits

    def calibrate(self, v_cache: np.ndarray, attention_maps: dict | None = None) -> None:
        """Calibrate bit allocation and value codebooks from sample data.

        Args:
            v_cache: Value cache (n_layers, n_heads, seq, dim) or dict
            attention_maps: Optional {(layer, head): attn_weights} for scoring
        """
        if attention_maps is not None:
            scores = self.scorer.score_from_attention(attention_maps)
        else:
            scores = self.scorer.score_synthetic(seed=self.seed)

        target = self._target_avg or (self.default_k_bits + self.default_v_bits) / 2
        self._allocation = self.scorer.allocate_bits(scores, target_avg=target)

        # Fit Lloyd-Max codebooks per head
        if self.enable_lloyd_max:
            if isinstance(v_cache, np.ndarray) and v_cache.ndim == 4:
                n_layers, n_heads = v_cache.shape[:2]
                for l in range(n_layers):
                    for h in range(n_heads):
                        cb = LloydMaxCodebook(bits=self._get_v_bits(l, h))
                        cb.fit(v_cache[l, h])
                        self._v_codebooks[(l, h)] = cb
            elif isinstance(v_cache, dict):
                for key, data in v_cache.items():
                    l, h = key
                    cb = LloydMaxCodebook(bits=self._get_v_bits(l, h))
                    cb.fit(data)
                    self._v_codebooks[key] = cb

    def compress(self, k_cache: np.ndarray, v_cache: np.ndarray) -> CompressedMOQCache:
        """Compress full KV cache.

        Args:
            k_cache: (n_layers, n_heads, seq_len, head_dim) or (n_heads, seq_len, head_dim)
            v_cache: same shape as k_cache

        Returns:
            CompressedMOQCache with per-head compressed data.
        """
        if k_cache.ndim == 3:
            k_cache = k_cache[None, :]
            v_cache = v_cache[None, :]

        n_layers, n_heads, seq_len, head_dim = k_cache.shape
        result = CompressedMOQCache(
            num_layers=n_layers, num_heads=n_heads,
            seq_len=seq_len, head_dim=head_dim,
        )

        for l in range(n_layers):
            for h in range(n_heads):
                k_bits = self._get_k_bits(l, h)
                v_bits = self._get_v_bits(l, h)

                k_data = k_cache[l, h].astype(np.float32)
                v_data = v_cache[l, h].astype(np.float32)

                # Keys: rotation + symmetric quantize
                rot = self._get_rotation(l, h, head_dim)
                if self.enable_rotation and rot is not None:
                    k_rot = rot.rotate(k_data)
                else:
                    k_rot = k_data

                k_q, k_scales = _symmetric_quantize(k_rot, k_bits)

                # Values: Lloyd-Max codebook or symmetric
                cb = self._v_codebooks.get((l, h))
                if self.enable_lloyd_max and cb is not None:
                    v_q = cb.quantize(v_data)
                    v_cents = cb.centroids.copy()
                else:
                    v_q, v_cents = _symmetric_quantize(v_data, v_bits)

                result.heads[(l, h)] = CompressedHead(
                    k_quantized=k_q, k_scales=k_scales,
                    v_quantized=v_q, v_centroids=v_cents,
                    k_bits=k_bits, v_bits=v_bits,
                    seq_len=seq_len,
                    rotated=self.enable_rotation,
                )

        return result

    def decompress(self, cache: CompressedMOQCache) -> tuple[np.ndarray, np.ndarray]:
        """Decompress full KV cache.

        Returns:
            (k_hat, v_hat) each shaped (n_layers, n_heads, seq_len, head_dim)
        """
        k_out = np.zeros((cache.num_layers, cache.num_heads, cache.seq_len, cache.head_dim),
                         dtype=np.float32)
        v_out = np.zeros_like(k_out)

        for (l, h), ch in cache.heads.items():
            # Keys: dequant + inverse rotation
            k_deq = _symmetric_dequantize(ch.k_quantized, ch.k_scales, ch.k_bits)
            if ch.rotated and self.enable_rotation:
                rot = self._get_rotation(l, h, cache.head_dim)
                if rot is not None:
                    k_deq = rot.unrotate(k_deq)
            k_out[l, h] = k_deq

            # Values: codebook lookup or symmetric dequant
            cb = self._v_codebooks.get((l, h))
            if self.enable_lloyd_max and cb is not None:
                v_out[l, h] = cb.dequantize(ch.v_quantized)
            else:
                v_out[l, h] = _symmetric_dequantize(ch.v_quantized, ch.v_centroids, ch.v_bits)

        return k_out, v_out

    def _get_rotation(self, layer: int, head: int, dim: int) -> NumpyRotation | None:
        key = (layer, head)
        if key not in self._rotations:
            self._rotations[key] = NumpyRotation(dim, seed=self.seed + layer * 1000 + head)
        return self._rotations[key]

    def _get_k_bits(self, layer: int, head: int) -> int:
        if self._allocation and (layer, head) in self._allocation:
            return self._allocation[(layer, head)][0]
        return self.default_k_bits

    def _get_v_bits(self, layer: int, head: int) -> int:
        if self._allocation and (layer, head) in self._allocation:
            return self._allocation[(layer, head)][1]
        return self.default_v_bits


def _symmetric_quantize(data: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel symmetric quantization."""
    qmax = (1 << bits) - 1
    half = qmax // 2

    if data.ndim == 1:
        abs_max = np.abs(data).max()
        scale = abs_max / half if abs_max > 0 else 1.0
        q = np.round(data / scale).clip(-half, half).astype(np.int8)
        return q, np.array([scale], dtype=np.float32)

    # Per-column scales for 2D
    abs_max = np.abs(data).max(axis=0)
    scales = np.where(abs_max > 0, abs_max / half, 1.0).astype(np.float32)
    q = np.round(data / scales[None, :]).clip(-half, half).astype(np.int8)
    return q, scales


def _symmetric_dequantize(quantized: np.ndarray, scales: np.ndarray, bits: int) -> np.ndarray:
    """Reverse symmetric quantization."""
    if quantized.ndim == 1 or scales.ndim == 0 or len(scales) == 1:
        return quantized.astype(np.float32) * float(scales.ravel()[0])
    return quantized.astype(np.float32) * scales[None, :]
