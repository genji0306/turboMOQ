# TurboMOQ vs TurboQuant vs TurboQuant+

## Architecture Comparison

| Feature | TurboQuant (0xSero) | TurboQuant+ (TheTom) | **TurboMOQ (ours)** |
|---------|--------------------|--------------------|---------------------|
| **Core algorithm** | PolarQuant + QJL | PolarQuant + QJL + Sparse V | **Rotation + MOQ topology + Lloyd-Max** |
| **Rotation** | QR decomp (Triton/CUDA) | QR decomp + Walsh-Hadamard | **Numpy QR (LAPACK)** |
| **Bit allocation** | Uniform per-head | Uniform per-head | **Topology-aware per-head** |
| **Key quantizer** | PolarQuant (b-1 bit) + QJL (1 bit) | Same | **Rotation + symmetric quant** |
| **Value quantizer** | Symmetric group quant | Symmetric + Sparse V skip | **Lloyd-Max codebook grids** |
| **K/V split strategy** | Different algorithms (TQ for K, MSE for V) | Same + sparse V | **Different: rotation K, codebook V** |
| **Progressive tiers** | No | No | **Yes (4 tiers: active/warm/cold/archive)** |
| **Multi-agent pool** | No | No | **Yes (priority eviction)** |
| **Hardware target** | CUDA/Triton (GPU) | Metal (Apple Silicon) + CUDA | **numpy/Apple Silicon** |
| **llama.cpp integration** | No (vLLM only) | Yes (C port) | No (HuggingFace + standalone) |
| **vLLM integration** | Yes (monkey-patch) | No | No |
| **Tests** | 35 | 511+ | **584 (557 inherited + 27 new)** |
| **License** | GPLv3 | GPLv3 | GPLv3 |

## What Each System Solves

### TurboQuant (0xSero/turboquant)
**Problem:** GPU memory for KV cache in vLLM serving.
**Solution:** CUDA/Triton fused kernels for PolarQuant + QJL. Doubles context on RTX 5090.
**Best for:** Production GPU serving with vLLM.

### TurboQuant+ (TheTom/turboquant_plus)  
**Problem:** Local inference on Apple Silicon and consumer GPUs.
**Solution:** Sparse V dequantization (+22.8% decode speed), Walsh-Hadamard fast rotation, C port for llama.cpp.
**Best for:** Single-model local inference on Mac/consumer GPU.

### TurboMOQ (this repo)
**Problem:** Multi-agent research swarms on consumer hardware (16GB Mac Mini).
**Solution:** Topology-aware bit allocation gives hub heads more bits and peripheral heads fewer. Lloyd-Max codebooks adapt to actual value distributions. Progressive tiers enable 750K+ context.
**Best for:** Multi-agent pools, long-context research, HuggingFace integration.

## Key Innovation: Topology-Aware Allocation

TurboQuant and TurboQuant+ treat all attention heads equally — every head gets the same number of bits. TurboMOQ scores each head on three dimensions:

1. **Position importance** — bell curve across layers (middle layers most critical)
2. **Topological persistence** — spectral gap of attention patterns (structured → important)
3. **Gradient sensitivity** — Fisher information proxy (high variance → sensitive to quantization)

Hub heads (global attention, induction heads) get 6-8 bits. Peripheral heads get 1-2 bits. Same average bits, but quality preserved where it matters.

## Benchmark Comparison (Mac Mini M4 16GB, llama3.1:8b architecture)

| Method | Key Cosine | Value Cosine | Avg Bits | Overhead |
|--------|-----------|-------------|----------|----------|
| TurboQuant (pure Python) | 0.265 | 0.265 | 3.5 | 954ms |
| Uniform 4-bit | 0.992 | 0.992 | 4.0 | 8ms |
| **TurboMOQ 4-bit** | **0.9997** | **0.9986** | **4.2** | **18ms** |

Note: TurboQuant's low cosine in pure Python is due to O(n^3) floating-point error in Gram-Schmidt. On CUDA/Triton it achieves 1.000. TurboMOQ's numpy QR matches CUDA-quality on CPU.

## Context Extension (qwen3-8b Q4 on 16GB)

| Method | FP16 Context | Compressed Context | Improvement |
|--------|-------------|-------------------|-------------|
| llama.cpp Q4 KV (built-in) | 55K | ~220K | 4x |
| TurboMOQ 4-bit | 55K | 253K | 4.6x |
| TurboMOQ 2.5-bit | 55K | 405K | 7.4x |
| **TurboMOQ Progressive** | **55K** | **~750K** | **13.7x** |

## When to Use What

| Scenario | Best Choice | Why |
|----------|------------|-----|
| GPU serving (vLLM) | TurboQuant (0xSero) | Fused Triton kernels, production-tested |
| Single model on Mac | TurboQuant+ (TheTom) | Sparse V, llama.cpp C port, Metal |
| Multi-agent research | **TurboMOQ** | Pool management, topology allocation |
| Long context (>100K) | **TurboMOQ Progressive** | Tier demotion, 13.7x extension |
| HuggingFace research | **TurboMOQ** | Drop-in cache, calibration API |
| Sub-2-bit extreme | **TurboMOQ** | Topology preserves quality at 1-2 bit |
