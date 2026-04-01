"""TurboMOQ: Hybrid KV cache compression combining TurboQuant rotation with MOQ topology.

Extends turboquant_plus with:
  - Topology-aware per-head bit allocation (MOQ head scoring)
  - Numpy QR rotation (LAPACK-backed, replaces pure-Python Gram-Schmidt)
  - MLX backend — native Metal acceleration for Apple Silicon (~2x speedup)
  - Lloyd-Max codebook grids for value cache
  - Split K/V strategy: rotation for keys, grids for values
  - Attention-weighted tier demotion (uses real attention patterns)
  - Progressive tier demotion for long contexts
  - QJL residual correction for sub-2-bit recovery
  - Multi-agent memory pool with priority eviction
  - llama.cpp C extension for kernel-level topology-aware allocation

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
from turbomoq.qjl import QJLCorrector, QJLResidual, QJLConfig
from turbomoq.attention_demotion import (
    AttentionDemotionPolicy, AttentionTier, TokenRecord, DemotionResult,
)

# Optional MLX backend
try:
    from turbomoq.mlx_backend import (
        MLX_AVAILABLE, MLXRotation, MLXCodebook, MLXQuantizer,
    )
except ImportError:
    MLX_AVAILABLE = False

# Optional C extension
try:
    from turbomoq.llamacpp_ext import LlamaCppExtension, EXTENSION_AVAILABLE
except ImportError:
    EXTENSION_AVAILABLE = False

__all__ = [
    # Core
    "HeadScorer",
    "HeadScore",
    "NumpyRotation",
    "LloydMaxCodebook",
    "TurboMOQCompressor",
    "CompressedMOQCache",
    "ProgressivePolicy",
    "ProgressiveTier",
    "MemoryPool",
    # QJL
    "QJLCorrector",
    "QJLResidual",
    "QJLConfig",
    # Attention demotion
    "AttentionDemotionPolicy",
    "AttentionTier",
    "TokenRecord",
    "DemotionResult",
    # MLX (optional)
    "MLX_AVAILABLE",
    "MLXRotation",
    "MLXCodebook",
    "MLXQuantizer",
    # C extension (optional)
    "LlamaCppExtension",
    "EXTENSION_AVAILABLE",
]

__version__ = "0.2.0"
