# TurboMOQ

**Hybrid KV Cache Compression via Stable Orthogonal Rotation and Topology-Aware Mixed-Precision Quantization for Long-Context LLM Inference on Consumer Hardware**

*Opensens DarkLab | March 2026*

[![Tests](https://img.shields.io/badge/tests-584%20passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![License](https://img.shields.io/badge/license-GPLv3-blue)]()

TurboMOQ extends [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) and [TurboQuant+](https://github.com/TheTom/turboquant_plus) with **topology-aware mixed-precision allocation** and **progressive temporal tiering** for multi-agent long-context inference on consumer hardware (16 GB Apple Silicon).

> **Core insight:** TurboQuant answers *how* to quantize (rotation + codebooks). MOQ answers *where* to allocate bits (topology scoring). They are orthogonal and composable. TurboMOQ combines both.

## Key Results (Mac Mini M4 16 GB)

| Metric | Value |
|--------|-------|
| Key cosine @ 4-bit | **0.9997** (live hardware) |
| Value cosine @ 4-bit | **0.9986** (live hardware) |
| Context expansion (Qwen3-8B) | 55K -> 352K (**6.4x**) |
| Progressive context | 55K -> ~752K (**13.7x**) |
| Compression overhead | 0.07 ms/token (**0.1%** of inference) |
| Multi-agent capacity @ 4K | 13 agents (FP16) -> 85 agents (MOQ-2bit) |

## What TurboMOQ Adds

| Component | What It Does | Impact |
|-----------|-------------|--------|
| **Stable QR Rotation** | LAPACK QR replaces pure-Python Gram-Schmidt | Cosine 0.265 -> 0.995 (**275% improvement**) |
| **Topology-Aware Allocation** | Score heads by importance, allocate bits non-uniformly | Hub heads 6-8 bit, peripheral 1-2 bit |
| **Split K/V Strategy** | Rotation for keys, Lloyd-Max codebooks for values | +1.04% value cosine over uniform |
| **Progressive Tiering** | 4 temporal tiers: Active -> Warm -> Cold -> Archive | 13.7x effective context expansion |
| **Multi-Agent Pool** | Shared KV budget with priority eviction | 85 agents on 16 GB |

## Architecture

```
Input: KV cache per attention head
         |
   +-----------+-----------+
   |   KEYS    |  VALUES   |
   |           |           |
   v           v           
 HeadScorer  HeadScorer    <- MOQ topology scoring
 (importance) (importance)    (spectral + gradient + position bell)
   |           |           
   v           v           
 Numpy QR    Lloyd-Max     <- Split quantization strategy
 Rotation    Codebook         Keys: decorrelate outliers
 (LAPACK)    (k-means)        Values: preserve hub distributions
   |           |           
   v           v           
 Symmetric   Grid          <- Per-head adaptive bit-width
 Quantize    Quantize         (1-8 bits based on score)
   |           |           
   +-----------+           
         |
   CompressedMOQCache
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
# {'compression_ratio': 4.2, 'avg_k_bits': 4.0, 'avg_v_bits': 4.0, ...}

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

| Feature | TurboQuant (ICLR 2026) | TurboQuant+ | **TurboMOQ** |
|---------|----------------------|------------|-------------|
| **Core algorithm** | PolarQuant + QJL | + Sparse V dequant | **Rotation + MOQ topology + Lloyd-Max** |
| **Rotation** | Hadamard (Triton/CUDA) | + Walsh-Hadamard | **Numpy QR (LAPACK)** |
| **Bit allocation** | Uniform per-head | Uniform per-head | **Topology-aware per-head** |
| **K/V strategy** | TQ for K, MSE for V | + Sparse V skip | **Rotation K, Codebook V** |
| **Progressive tiers** | No | No | **Yes (4 tiers)** |
| **Multi-agent pool** | No | No | **Yes (priority eviction)** |
| **Hardware** | CUDA/Triton | Metal + CUDA | **numpy/Apple Silicon** |
| **Integration** | vLLM | llama.cpp (C port) | **HuggingFace + standalone** |
| **Tests** | 35 | 511 | **584** |

### When to Use What

| Scenario | Best Choice | Why |
|----------|------------|-----|
| GPU serving (vLLM) | TurboQuant | Fused Triton kernels, production-tested |
| Single model on Mac | TurboQuant+ | Sparse V, llama.cpp C port, Metal |
| **Multi-agent research** | **TurboMOQ** | **Pool management, topology allocation** |
| **Long context (>100K)** | **TurboMOQ Progressive** | **Tier demotion, 13.7x expansion** |
| **HuggingFace research** | **TurboMOQ** | **Drop-in cache, calibration API** |
| Sub-2-bit extreme | MOQ | Topology preserves quality at 1-2 bit |

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
├── turbomoq/                    # NEW: TurboMOQ hybrid layer
│   ├── head_scorer.py           # 3-signal topology scoring
│   ├── rotation.py              # Numpy QR rotation (LAPACK)
│   ├── codebook.py              # Lloyd-Max optimal codebook
│   ├── compressor.py            # Split K/V compressor
│   ├── progressive.py           # 4-tier temporal demotion
│   └── pool.py                  # Multi-agent memory pool
│
├── tests/                       # 584 tests total
│   ├── test_turbomoq_hybrid.py  # 27 TurboMOQ tests
│   ├── test_turboquant.py       # Original TurboQuant tests
│   └── ...                      # 14 test files
│
├── COMPARISON.md                # Detailed 3-way comparison
├── benchmarks/                  # Speed and compression benchmarks
├── docs/                        # Papers and analysis
└── pyproject.toml               # Package config (turbomoq v0.2.0)
```

## Running Tests

```bash
# All tests (584 total)
python -m pytest tests/ -q

# TurboMOQ tests only (27)
python -m pytest tests/test_turbomoq_hybrid.py -v

# Original TurboQuant tests (557)
python -m pytest tests/ --ignore=tests/test_turbomoq_hybrid.py -q
```

## Limitations

- **No Triton/CUDA kernels.** TurboMOQ is numpy-only. On GPU, the original TurboQuant is faster.
- **No Ollama integration.** llama.cpp manages KV cache in C++; Python-level compression cannot be injected.
- **Calibration required.** Lloyd-Max codebooks need sample data. Falls back to symmetric quantization without calibration.
- **Prototype scope.** Context-length projections are memory-budget feasibility estimates, not end-to-end perplexity measurements.

## Future Work

- **MLX backend** — native Metal acceleration for Apple Silicon (~2x speedup over numpy)
- **Attention-weighted tier demotion** — use actual attention patterns, not just token age
- **QJL residual for sub-2-bit** — sign correction could recover 0.5-1% cosine at extreme compression
- **llama.cpp C port** — bring topology-aware allocation to the metal kernel level

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
