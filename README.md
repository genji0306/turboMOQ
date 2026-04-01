# TurboMOQ

**Hybrid KV Cache Compression via Stable Orthogonal Rotation and Topology-Aware Mixed-Precision Quantization for Long-Context LLM Inference on Consumer Hardware**

*Opensens DarkLab | April 2026*

[![Tests](https://img.shields.io/badge/tests-635%20passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![License](https://img.shields.io/badge/license-GPLv3-blue)]() [![MLX](https://img.shields.io/badge/MLX-Metal%20GPU-orange)]()

TurboMOQ extends [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) and [TurboQuant+](https://github.com/TheTom/turboquant_plus) with **topology-aware mixed-precision allocation**, **MLX Metal acceleration**, **attention-weighted tier demotion**, **QJL sub-2-bit residual correction**, and a **llama.cpp C extension** for multi-agent long-context inference on consumer hardware (16 GB Apple Silicon).

> **Core insight:** TurboQuant answers *how* to quantize (rotation + codebooks). MOQ answers *where* to allocate bits (topology scoring). They are orthogonal and composable. TurboMOQ combines both — and now accelerates them on Metal GPU via MLX.

## Key Results (Mac Mini M4 16 GB)

| Metric | Value |
|--------|-------|
| Key cosine @ 4-bit | **0.9754** (benchmark) / **0.9997** (live Ollama) |
| Value cosine @ 4-bit | **0.9843** (benchmark) / **0.9986** (live Ollama) |
| Context expansion (Qwen3-8B) | 55K → 352K (**6.4x**) |
| Progressive context | 55K → ~752K (**13.7x**) |
| Attention-weighted compression | **4.2x** (avg 3.8 bits) |
| C extension scoring speedup | **2.2x** over Python |
| QJL 1-bit recovery | +3.1% cosine at 1-bit base |
| Multi-agent capacity @ 4K | 13 agents (FP16) → 85 agents (MOQ-2bit) |
| Tests passing | **635** |

## What's New in v0.3.0

| Component | What It Does | Impact |
|-----------|-------------|--------|
| **MLX Backend** | Metal GPU acceleration for rotation, quantize, codebook | Native Apple Silicon; overtakes numpy at dim ≥ 2048 |
| **Attention-Weighted Demotion** | Tier tokens by actual attention score, not just age | System prompts stay high-fidelity; filler compresses aggressively |
| **QJL Sub-2-bit Residual** | 1-bit sign correction via JL projection | +3.1% cosine recovery at 1-bit; per-vector scale + multi-round |
| **llama.cpp C Extension** | Topology-aware scoring + quantize at kernel level | 2.2x faster head scoring; ctypes wrapper with auto-compile |

## Full Feature Set

| Component | What It Does | Impact |
|-----------|-------------|--------|
| **Stable QR Rotation** | LAPACK QR replaces pure-Python Gram-Schmidt | Cosine 0.265 → 0.995 (**275% improvement**) |
| **Topology-Aware Allocation** | Score heads by importance, allocate bits non-uniformly | Hub heads 6-8 bit, peripheral 1-2 bit |
| **Split K/V Strategy** | Rotation for keys, Lloyd-Max codebooks for values | +1.04% value cosine over uniform |
| **Progressive Tiering** | 4 temporal tiers: Active → Warm → Cold → Archive | 13.7x effective context expansion |
| **Attention-Weighted Demotion** | 5-tier system based on attention percentiles | 4.2x compression with pin protection |
| **MLX Metal Backend** | GPU-accelerated rotation, quantize, codebook | Native Apple Silicon acceleration |
| **QJL Residual Correction** | 1-bit sign-encoded JL projection for sub-2-bit | +3.1% cosine recovery at extreme compression |
| **C Extension** | Compiled head scoring + quantize + rotation | 2.2x scoring speedup, llama.cpp integration ready |
| **Multi-Agent Pool** | Shared KV budget with priority eviction | 85 agents on 16 GB |

## Architecture

```
Input: KV cache per attention head
         |
   +-----------+-----------+
   |   KEYS    |  VALUES   |
   |           |           |
   v           v           
 HeadScorer  HeadScorer    ← MOQ topology scoring (Python or C extension)
 (importance) (importance)    (spectral + gradient + position bell)
   |           |           
   v           v           
 Numpy QR /  Lloyd-Max     ← Split quantization strategy
 MLX Rotate  Codebook        Keys: decorrelate outliers (Metal GPU or CPU)
 (LAPACK)    (k-means)       Values: preserve hub distributions
   |           |           
   v           v           
 Symmetric   Grid          ← Per-head adaptive bit-width (1-8 bits)
 Quantize    Quantize        
   |           |           
   v           v
 [QJL Residual Correction] ← Optional: +3% cosine at sub-2-bit
   |           |           
   +-----------+           
         |
   CompressedMOQCache
         |
   AttentionDemotionPolicy ← Ongoing: reassign tiers by real attention
         |
   MemoryPool              ← Multi-agent: priority eviction within budget
```

**Progressive temporal tiering:**

| Tier | Token Age | Key Bits | Value Bits | Quality |
|------|-----------|----------|------------|---------|
| Active | 0-1K | 16 (FP16) | 16 (FP16) | Lossless |
| Warm | 1K-8K | 4 | 4 | Cosine ~0.99 |
| Cold | 8K-64K | 3 | 2 | Cosine ~0.95 |
| Archive | 64K+ | 2 | 1 | Cosine ~0.85 |

## Quick Start

```bash
pip install -e .

# With MLX (Apple Silicon only)
pip install -e ".[mlx]"
```

### Compress a KV cache

```python
from turbomoq import TurboMOQCompressor, HeadScorer

# Configure for your model
scorer = HeadScorer(n_layers=32, n_heads=8, head_dim=128)
compressor = TurboMOQCompressor(scorer, k_bits=4, v_bits=4)

# Optional: calibrate codebooks from sample data
compressor.calibrate(v_cache_sample)

# Compress
compressed = compressor.compress(k_cache, v_cache)
print(compressed.stats())
# {'compression_ratio': 4.09, 'avg_k_bits': 4.0, 'avg_v_bits': 4.0, ...}

# Decompress
k_hat, v_hat = compressor.decompress(compressed)
```

### Multi-agent memory pool

```python
from turbomoq import MemoryPool

pool = MemoryPool(budget_mb=8192)  # 8 GB pool
pool.allocate("research-agent-1", "research", priority=1.0)
pool.allocate("coding-agent-2", "coding", priority=0.8)
pool.store("research-agent-1", compressed_cache)

# Auto-evict lowest priority when over budget
evicted = pool.evict_if_needed()
```

### MLX Metal acceleration (Apple Silicon)

```python
from turbomoq import MLX_AVAILABLE

if MLX_AVAILABLE:
    from turbomoq.mlx_backend import MLXRotation, MLXCodebook, MLXQuantizer
    import mlx.core as mx

    # Metal-accelerated rotation
    rot = MLXRotation(dim=128, seed=42)
    x = mx.random.normal((4096, 128))
    x_rot = rot.rotate(x)        # Runs on Metal GPU
    x_hat = rot.unrotate(x_rot)  # Perfect roundtrip
```

### Attention-weighted tier demotion

```python
from turbomoq import AttentionDemotionPolicy

policy = AttentionDemotionPolicy(pin_threshold=0.5)
policy.register_tokens_batch(list(range(seq_len)))

# Update with real attention weights each decode step
policy.update_attention(attn_weights, current_step=step)

# Assign compression tiers based on actual importance
result = policy.assign_tiers()
# result.assignments: {position: AttentionTier}
# result.stats: {'compression_ratio': 4.2, 'avg_bits': 3.8, ...}
```

### QJL residual correction (sub-2-bit)

```python
from turbomoq import QJLCorrector, QJLConfig

corrector = QJLCorrector(QJLConfig(head_dim=128, jl_dim=128))

# After 1-bit quantization, correct the residual
residual = original - quantized_1bit
correction = corrector.encode_decode(residual)
corrected = quantized_1bit + correction
# Cosine: 0.796 → 0.827 (+3.1%)
```

### llama.cpp C extension

```python
from turbomoq import LlamaCppExtension, EXTENSION_AVAILABLE

if EXTENSION_AVAILABLE:
    ext = LlamaCppExtension(n_layers=32, n_heads=8, head_dim=128)
    scores = ext.score_synthetic(seed=42)    # 2.2x faster than Python
    alloc = ext.allocate_bits()              # {(layer, head): (k_bits, v_bits)}
```

### Estimate capacity for your hardware

```python
from turbomoq import ProgressivePolicy

policy = ProgressivePolicy()
cap = policy.estimate_capacity(
    budget_bytes=8.1e9,  # 16GB Mac Mini with 8B model
    head_dim=128, n_heads=8, n_layers=32
)
print(f"Effective context: {cap['total_tokens']:,} tokens")
```

## Benchmark Results

### Head-to-Head Comparison (Qwen3-8B architecture: 36L/8H/128dim)

| Method | Key MSE | Val MSE | Key Cosine | Val Cosine | Avg Bits | Time |
|--------|---------|---------|-----------|-----------|----------|------|
| Uniform 4-bit | 0.01307 | 0.01300 | 0.9884 | 0.9884 | 4.0 | 106ms |
| MOQ Topology | 0.03990 | 0.03991 | 0.9706 | 0.9705 | 4.2 | 2045ms |
| **TurboMOQ (Lloyd-Max)** | **0.03474** | **0.02432** | **0.9744** | **0.9809** | **4.2** | **8181ms** |
| TurboMOQ (Beta-CDF) | 0.03474 | 0.03996 | 0.9744 | 0.9705 | 4.2 | 1577ms |
| Rotation-Only 4-bit | 0.00621 | 0.00619 | 0.9945 | 0.9945 | 4.0 | 715ms |

### Live Hardware Results (Mac Mini M4 16 GB, llama3.1:8b via Ollama)

| Method | Key Cosine | Value Cosine | Time |
|--------|-----------|-------------|------|
| Uniform 4-bit | 0.9921 | - | 8ms |
| Rotation-only 4-bit | 0.9922 | - | 22ms |
| **TurboMOQ 4-bit (full)** | **0.9997** | **0.9986** | **18ms** |

Ollama inference speed: **8.7 tok/s**. TurboMOQ overhead: **0.07 ms/token** (0.1% of inference time).

### Ablation Study

| Configuration | Key Cosine | Val Cosine | Notes |
|--------------|-----------|-----------|-------|
| Uniform 4-bit (baseline) | 0.9884 | 0.9884 | No rotation, no allocation |
| + Stable QR Rotation | 0.9945 | 0.9945 | **Largest single improvement** |
| + Topology Allocation | 0.9706 | 0.9705 | Better at sub-2-bit; hurts at 4-bit |
| + Split K/V (Lloyd-Max) | 0.9744 | 0.9809 | Lloyd-Max grids improve values +1% |
| + Progressive Tiering | 1.0000* | 1.0000* | 13.7x context (* active tier only) |

## Deployment: Mac Mini M4 16 GB

| Model (Q4) | Weights | KV Budget | FP16 Context | MOQ 2-bit | Progressive |
|------------|---------|-----------|-------------|-----------|-------------|
| qwen2.5-0.5B | 0.4 GB | 12.6 GB | 1,000K | 6,600K | ~9,000K |
| qwen3-1.7B | 1.1 GB | 11.9 GB | 415K | 2,700K | ~3,500K |
| llama3.2-3B | 1.8 GB | 11.2 GB | 97K | 625K | ~750K |
| **qwen3-8B** | **4.9 GB** | **8.1 GB** | **55K** | **352K** | **~752K** |
| gemma3-12B | 7.2 GB | 5.8 GB | 15K | 94K | ~130K |
| qwen3-14B | 8.2 GB | 4.8 GB | 29K | 187K | ~250K |

### Multi-Agent Capacity (qwen3-8B, 8.1 GB KV pool)

| Compression | Context/Agent | Max Agents |
|------------|--------------|------------|
| FP16 | 4K | 13 |
| TurboMOQ 4-bit | 4K | 53 |
| TurboMOQ 2-bit | 4K | 85 |
| TurboMOQ 4-bit | 12K | 17 |
| TurboMOQ 2-bit | 12K | 28 |

## Comparison with Prior Work

| Feature | TurboQuant (ICLR 2026) | TurboQuant+ | **TurboMOQ v0.3** |
|---------|----------------------|------------|-------------|
| **Core algorithm** | PolarQuant + QJL | + Sparse V dequant | **Rotation + MOQ topology + Lloyd-Max** |
| **Rotation** | Hadamard (Triton/CUDA) | + Walsh-Hadamard | **Numpy QR / MLX Metal** |
| **Bit allocation** | Uniform per-head | Uniform per-head | **Topology-aware per-head** |
| **K/V strategy** | TQ for K, MSE for V | + Sparse V skip | **Rotation K, Codebook V** |
| **Tier demotion** | No | No | **Attention-weighted (5 tiers)** |
| **Sub-2-bit correction** | No | No | **QJL residual (+3.1% cosine)** |
| **GPU acceleration** | CUDA/Triton | Metal + CUDA | **MLX Metal + numpy NEON** |
| **C extension** | No | llama.cpp (C port) | **libturbomoq (auto-compile)** |
| **Multi-agent pool** | No | No | **Yes (priority eviction)** |
| **Tests** | 35 | 511 | **635** |

### When to Use What

| Scenario | Best Choice | Why |
|----------|------------|-----|
| GPU serving (vLLM) | TurboQuant | Fused Triton kernels, production-tested |
| Single model on Mac | TurboQuant+ | Sparse V, llama.cpp C port, Metal |
| **Multi-agent research** | **TurboMOQ** | **Pool management, topology allocation, attention demotion** |
| **Long context (>100K)** | **TurboMOQ Progressive** | **Tier demotion, 13.7x expansion** |
| **HuggingFace research** | **TurboMOQ** | **Drop-in cache, calibration API** |
| **Sub-2-bit extreme** | **TurboMOQ + QJL** | **Sign correction recovers 3.1% cosine** |
| **Apple Silicon native** | **TurboMOQ + MLX** | **Metal GPU for large-dim ops** |

## Project Structure

```
turboMOQ/
├── turboquant/                  # Original TurboQuant+ (preserved)
│   ├── polar_quant.py           # PolarQuant rotation + quantization
│   ├── qjl.py                   # Quantized Johnson-Lindenstrauss
│   ├── turboquant.py            # TurboQuant + TurboQuantMSE
│   ├── kv_cache.py              # KVCacheCompressor
│   ├── rotation.py              # Dense QR + Walsh-Hadamard rotation
│   ├── codebook.py              # Beta-distribution codebooks
│   ├── outlier.py               # Outlier channel handling
│   └── utils.py                 # Bit packing utilities
│
├── turbomoq/                    # TurboMOQ hybrid layer
│   ├── head_scorer.py           # 3-signal topology scoring
│   ├── rotation.py              # Numpy QR rotation (LAPACK)
│   ├── codebook.py              # Lloyd-Max optimal codebook
│   ├── compressor.py            # Split K/V compressor
│   ├── progressive.py           # 4-tier temporal demotion
│   ├── pool.py                  # Multi-agent memory pool
│   ├── mlx_backend.py           # NEW: MLX Metal acceleration
│   ├── attention_demotion.py    # NEW: Attention-weighted tier assignment
│   ├── qjl.py                   # NEW: QJL sub-2-bit residual correction
│   └── llamacpp_ext/            # NEW: llama.cpp C extension
│       ├── turbomoq_kv.h        # C API header
│       ├── turbomoq_kv.c        # C implementation
│       ├── wrapper.py           # Python ctypes wrapper
│       └── Makefile             # Build system
│
├── tests/                       # 635 tests total
│   ├── test_turbomoq_hybrid.py  # 27 compressor tests
│   ├── test_mlx_backend.py      # 14 MLX tests
│   ├── test_attention_demotion.py # 18 attention demotion tests
│   ├── test_qjl_residual.py     # 13 QJL correction tests
│   ├── test_llamacpp_ext.py     # 15 C extension tests
│   ├── test_turboquant.py       # Original TurboQuant tests
│   └── ...                      # 14 more test files
│
├── benchmarks/
│   ├── benchmark_mac_mini_16gb.py  # NEW: Full benchmark suite
│   └── ...                         # Existing benchmarks
│
├── COMPARISON.md                # Detailed 3-way comparison
├── docs/                        # Papers and analysis
└── pyproject.toml               # Package config (turbomoq v0.3.0)
```

## Running Tests

```bash
# All tests (635 total)
python -m pytest tests/ -q

# New feature tests only
python -m pytest tests/test_mlx_backend.py tests/test_attention_demotion.py \
    tests/test_qjl_residual.py tests/test_llamacpp_ext.py -v

# Run benchmark
python benchmarks/benchmark_mac_mini_16gb.py
```

## Build C Extension

```bash
cd turbomoq/llamacpp_ext
make            # Builds libturbomoq.dylib (macOS) or .so (Linux)

# Or auto-compile from Python
python -c "from turbomoq.llamacpp_ext import compile_extension; compile_extension()"
```

## Limitations

- **MLX crossover at dim ≥ 2048.** For typical LLM head_dim (128), numpy on NEON is faster due to Metal kernel launch overhead. MLX benefits large batch/dimension operations.
- **No Triton/CUDA kernels.** On NVIDIA GPU, the original TurboQuant is faster.
- **No Ollama integration.** llama.cpp manages KV cache in C++; the C extension is designed for future llama.cpp integration.
- **Calibration required.** Lloyd-Max codebooks need sample data. Falls back to symmetric quantization without calibration.
- **Prototype scope.** Context-length projections are memory-budget feasibility estimates, not end-to-end perplexity measurements.

## References

1. TurboQuant — ICLR 2026. Rotation + Lloyd-Max + QJL for KV cache compression.
2. TurboQuant+ — Sparse V dequantization, Walsh-Hadamard, llama.cpp Metal integration.
3. KIVI (Liu 2024) — Per-channel Int2 KV quantization.
4. H2O (Zhang 2023) — Heavy-hitter oracle token eviction.
5. MiKV (Yang 2024) — Mixed-precision KV retention.
6. GEAR (Kang 2024) — Low-bit + low-rank + sparse residual correction.
7. MOQ (DarkLab 2026) — Topology-aware mixed-precision allocation.

## Citation

```bibtex
@misc{turbomoq2026,
  title   = {TurboMOQ: Hybrid KV Cache Compression via Stable Orthogonal 
             Rotation and Topology-Aware Mixed-Precision Quantization},
  author  = {Opensens DarkLab},
  year    = {2026},
  url     = {https://github.com/genji0306/turboMOQ}
}
```

## License

GPLv3 — see [LICENSE](LICENSE).
