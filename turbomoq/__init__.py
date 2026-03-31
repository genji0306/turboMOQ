"""TurboMOQ: Hybrid KV cache compression combining TurboQuant rotation with MOQ topology.

Extends turboquant_plus with:
  - Topology-aware per-head bit allocation (MOQ head scoring)
  - Numpy QR rotation (LAPACK-backed, replaces pure-Python Gram-Schmidt)
  - Lloyd-Max codebook grids for value cache
  - Split K/V strategy: rotation for keys, grids for values
  - Progressive tier demotion for long contexts
  - Multi-agent memory pool with priority eviction

Usage:
    from turbomoq import TurboMOQCompressor, HeadScorer

    scorer = HeadScorer(n_layers=32, n_heads=8, head_dim=128)
    compressor = TurboMOQCompressor(scorer, k_bits=4, v_bits=4)

    compressed = compressor.compress(k_cache, v_cache)
    k_hat, v_hat = compressor.decompress(compressed)
"""

from turbomoq.head_scorer import HeadScorer, HeadScore
from turbomoq.rotation import NumpyRotation
from turbomoq.codebook import LloydMaxCodebook
from turbomoq.compressor import TurboMOQCompressor, CompressedMOQCache
from turbomoq.progressive import ProgressivePolicy, ProgressiveTier
from turbomoq.pool import MemoryPool

__all__ = [
    "HeadScorer",
    "HeadScore",
    "NumpyRotation",
    "LloydMaxCodebook",
    "TurboMOQCompressor",
    "CompressedMOQCache",
    "ProgressivePolicy",
    "ProgressiveTier",
    "MemoryPool",
]

__version__ = "0.1.0"
