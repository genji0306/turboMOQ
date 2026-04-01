#!/usr/bin/env python3
"""TurboMOQ benchmark for Mac mini 16GB (Apple Silicon).

Tests all 4 new features:
  1. MLX backend vs numpy (rotation, quantize, codebook)
  2. Attention-weighted tier demotion
  3. QJL residual correction at sub-2-bit
  4. C extension (llama.cpp port)

Usage:
    python benchmarks/benchmark_mac_mini_16gb.py
"""

import json
import platform
import sys
import time
from dataclasses import asdict

import numpy as np

# ── System Info ──────────────────────────────────────────────────────────────

def system_info():
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        info["ram_gb"] = round(int(result.stdout.strip()) / 1e9, 1)
    except Exception:
        info["ram_gb"] = "unknown"
    return info


# ── 1. MLX Backend Benchmark ────────────────────────────────────────────────

def benchmark_mlx(dim=128, seq_len=1024, iterations=50):
    """Compare MLX Metal vs numpy for rotation + quantize."""
    print("\n" + "=" * 60)
    print("1. MLX BACKEND BENCHMARK")
    print("=" * 60)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((seq_len, dim)).astype(np.float32)
    results = {"dim": dim, "seq_len": seq_len, "iterations": iterations}

    # Numpy rotation
    from turbomoq.rotation import NumpyRotation
    np_rot = NumpyRotation(dim, seed=42)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = np_rot.rotate(data)
    np_time = (time.perf_counter() - t0) / iterations
    results["numpy_rotate_ms"] = round(np_time * 1000, 3)
    print(f"  Numpy rotation: {np_time*1000:.3f} ms/iter")

    # Numpy codebook
    from turbomoq.codebook import LloydMaxCodebook
    flat = data.ravel()[:10000]
    t0 = time.perf_counter()
    for _ in range(iterations):
        cb = LloydMaxCodebook(bits=4, iterations=3)
        cb.fit(flat)
    np_cb_time = (time.perf_counter() - t0) / iterations
    results["numpy_codebook_ms"] = round(np_cb_time * 1000, 3)
    print(f"  Numpy codebook:  {np_cb_time*1000:.3f} ms/iter")

    # MLX
    try:
        import mlx.core as mx
        from turbomoq.mlx_backend import MLXRotation, MLXCodebook, MLXQuantizer, benchmark_backends

        mlx_rot = MLXRotation(dim, seed=42)
        data_mx = mx.array(data)

        # Warmup
        r = mlx_rot.rotate(data_mx)
        mx.eval(r)

        t0 = time.perf_counter()
        for _ in range(iterations):
            r = mlx_rot.rotate(data_mx)
            mx.eval(r)
        mlx_time = (time.perf_counter() - t0) / iterations
        results["mlx_rotate_ms"] = round(mlx_time * 1000, 3)
        results["rotation_speedup"] = round(np_time / mlx_time, 2) if mlx_time > 0 else 0
        print(f"  MLX rotation:    {mlx_time*1000:.3f} ms/iter ({results['rotation_speedup']}x speedup)")

        # MLX codebook
        flat_mx = mx.array(flat)
        # Warmup
        cb_mlx = MLXCodebook(bits=4, iterations=3)
        cb_mlx.fit(flat_mx)

        t0 = time.perf_counter()
        for _ in range(iterations):
            cb_mlx = MLXCodebook(bits=4, iterations=3)
            cb_mlx.fit(flat_mx)
        mlx_cb_time = (time.perf_counter() - t0) / iterations
        results["mlx_codebook_ms"] = round(mlx_cb_time * 1000, 3)
        results["codebook_speedup"] = round(np_cb_time / mlx_cb_time, 2) if mlx_cb_time > 0 else 0
        print(f"  MLX codebook:    {mlx_cb_time*1000:.3f} ms/iter ({results['codebook_speedup']}x speedup)")

        # MLX quantizer
        mlx_q = MLXQuantizer(bits=4)
        t0 = time.perf_counter()
        for _ in range(iterations):
            q, s = mlx_q.quantize(data_mx)
            mx.eval(q, s)
        mlx_q_time = (time.perf_counter() - t0) / iterations
        results["mlx_quantize_ms"] = round(mlx_q_time * 1000, 3)
        print(f"  MLX quantize:    {mlx_q_time*1000:.3f} ms/iter")

        # Quality check
        r_np = mlx_rot.rotate_numpy(data)
        q_np = np_rot.rotate(data)
        cos = float(np.dot(r_np.ravel(), q_np.ravel()) / (np.linalg.norm(r_np.ravel()) * np.linalg.norm(q_np.ravel())))
        results["rotation_agreement_cosine"] = round(cos, 6)
        print(f"  MLX/numpy agreement: cosine={cos:.6f}")

    except ImportError:
        print("  [MLX not available — Apple Silicon required]")
        results["mlx_rotate_ms"] = None
        results["rotation_speedup"] = None

    return results


# ── 2. Attention-Weighted Demotion Benchmark ────────────────────────────────

def benchmark_attention_demotion(n_tokens=10000, n_steps=100):
    """Benchmark attention-weighted tier assignment."""
    print("\n" + "=" * 60)
    print("2. ATTENTION-WEIGHTED TIER DEMOTION BENCHMARK")
    print("=" * 60)

    from turbomoq.attention_demotion import AttentionDemotionPolicy

    rng = np.random.default_rng(42)
    policy = AttentionDemotionPolicy()

    # Register tokens
    t0 = time.perf_counter()
    policy.register_tokens_batch(list(range(n_tokens)))
    reg_time = time.perf_counter() - t0
    print(f"  Register {n_tokens} tokens: {reg_time*1000:.1f} ms")

    # Simulate attention updates
    t0 = time.perf_counter()
    for step in range(n_steps):
        weights = rng.random(n_tokens).astype(np.float32)
        # Simulate: system prompt tokens always high attention
        weights[:50] = 0.8 + rng.random(50).astype(np.float32) * 0.2
        # Recent tokens also high
        recent = max(0, n_tokens - 200)
        weights[recent:] *= 2.0
        policy.update_attention(weights, current_step=step)
    update_time = (time.perf_counter() - t0) / n_steps
    print(f"  Update attention ({n_tokens} tokens): {update_time*1000:.2f} ms/step")

    # Tier assignment
    t0 = time.perf_counter()
    result = policy.assign_tiers()
    assign_time = time.perf_counter() - t0
    print(f"  Assign tiers: {assign_time*1000:.2f} ms")
    print(f"  Tier distribution: {result.stats['tier_distribution']}")
    print(f"  Avg bits: {result.stats['avg_bits']}")
    print(f"  Promoted: {result.stats['n_promoted']}, Demoted: {result.stats['n_demoted']}")

    # Memory estimate
    mem = policy.estimate_memory(head_dim=128, n_heads=8, n_layers=32)
    print(f"  Estimated memory: {mem['compressed_mb']:.1f} MB (FP16: {mem['fp16_mb']:.1f} MB)")
    print(f"  Compression ratio: {mem['compression_ratio']:.1f}x")

    return {
        "n_tokens": n_tokens,
        "n_steps": n_steps,
        "register_ms": round(reg_time * 1000, 1),
        "update_ms_per_step": round(update_time * 1000, 2),
        "assign_ms": round(assign_time * 1000, 2),
        "tier_distribution": result.stats["tier_distribution"],
        "avg_bits": result.stats["avg_bits"],
        "compression_ratio": mem["compression_ratio"],
    }


# ── 3. QJL Residual Benchmark ──────────────────────────────────────────────

def benchmark_qjl(head_dim=128, seq_len=512):
    """Benchmark QJL residual correction quality and speed."""
    print("\n" + "=" * 60)
    print("3. QJL RESIDUAL CORRECTION BENCHMARK")
    print("=" * 60)

    from turbomoq.qjl import QJLCorrector, QJLConfig

    rng = np.random.default_rng(42)
    original = rng.standard_normal((seq_len, head_dim)).astype(np.float32)

    results = {"head_dim": head_dim, "seq_len": seq_len}

    for base_bits in [2, 1]:
        n_levels = 2 ** base_bits
        half = n_levels // 2
        per_col = np.abs(original).max(axis=0)
        scales = np.where(per_col > 0, per_col / half, 1.0)
        if base_bits == 1:
            quantized = np.sign(original).astype(np.float32)
        else:
            quantized = np.round(original / scales).clip(-half, half - 1) * scales

        residual = original - quantized
        cos_base = _cosine(original.ravel(), quantized.ravel())
        print(f"\n  {base_bits}-bit base: cosine={cos_base:.4f}")

        for jl_frac in [0.25, 0.5, 0.75, 1.0]:
            jl_dim = max(4, int(head_dim * jl_frac))
            config = QJLConfig(head_dim=head_dim, jl_dim=jl_dim, use_mlx=False)
            corrector = QJLCorrector(config)

            # Speed
            t0 = time.perf_counter()
            for _ in range(20):
                packed = corrector.encode(residual)
                correction = corrector.decode(packed)
            speed = (time.perf_counter() - t0) / 20

            correction = corrector.encode_decode(residual)
            corrected = quantized + correction
            cos_after = _cosine(original.ravel(), corrected.ravel())
            improvement = cos_after - cos_base
            overhead_bits = packed.bits_per_dim

            key = f"{base_bits}bit_jl{jl_dim}"
            results[key] = {
                "base_cosine": round(cos_base, 4),
                "corrected_cosine": round(cos_after, 4),
                "improvement": round(improvement, 4),
                "overhead_bits_per_dim": round(overhead_bits, 3),
                "encode_decode_ms": round(speed * 1000, 2),
            }
            sign = "+" if improvement > 0 else ""
            print(f"    jl_dim={jl_dim:3d}: cosine={cos_after:.4f} ({sign}{improvement:.4f}), "
                  f"+{overhead_bits:.2f} bits/dim, {speed*1000:.2f} ms")

    # MLX comparison
    try:
        import mlx.core as mx
        config = QJLConfig(head_dim=head_dim, jl_dim=head_dim, use_mlx=True)
        corrector_mlx = QJLCorrector(config)

        residual_1bit = original - np.sign(original)
        t0 = time.perf_counter()
        for _ in range(20):
            packed = corrector_mlx.encode(residual_1bit)
            _ = corrector_mlx.decode(packed)
        mlx_speed = (time.perf_counter() - t0) / 20
        results["mlx_encode_decode_ms"] = round(mlx_speed * 1000, 2)
        print(f"\n  MLX QJL encode+decode: {mlx_speed*1000:.2f} ms")
    except ImportError:
        print("\n  [MLX QJL: not available]")

    return results


# ── 4. C Extension Benchmark ──────────────────────────────────────────────

def benchmark_c_extension(n_layers=32, n_heads=8, head_dim=128):
    """Benchmark C extension vs Python for scoring + quantize."""
    print("\n" + "=" * 60)
    print("4. C EXTENSION (llama.cpp PORT) BENCHMARK")
    print("=" * 60)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((64, head_dim)).astype(np.float32)
    results = {"n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim}

    # Python scoring
    from turbomoq.head_scorer import HeadScorer
    scorer_py = HeadScorer(n_layers, n_heads, head_dim)

    t0 = time.perf_counter()
    for _ in range(100):
        scores = scorer_py.score_synthetic(seed=42)
    py_score_time = (time.perf_counter() - t0) / 100
    results["python_score_ms"] = round(py_score_time * 1000, 3)
    print(f"  Python score_synthetic: {py_score_time*1000:.3f} ms")

    # Python quantize
    from turbomoq.compressor import _symmetric_quantize
    t0 = time.perf_counter()
    for _ in range(100):
        q, s = _symmetric_quantize(data, 4)
    py_quant_time = (time.perf_counter() - t0) / 100
    results["python_quantize_ms"] = round(py_quant_time * 1000, 3)
    print(f"  Python quantize:        {py_quant_time*1000:.3f} ms")

    # C extension
    try:
        from turbomoq.llamacpp_ext import LlamaCppExtension, compile_extension
        compile_extension()
        ext = LlamaCppExtension(n_layers, n_heads, head_dim)

        t0 = time.perf_counter()
        for _ in range(100):
            ext.score_synthetic(seed=42)
        c_score_time = (time.perf_counter() - t0) / 100
        results["c_score_ms"] = round(c_score_time * 1000, 3)
        results["score_speedup"] = round(py_score_time / c_score_time, 2) if c_score_time > 0 else 0
        print(f"  C score_synthetic:      {c_score_time*1000:.3f} ms ({results['score_speedup']}x)")

        t0 = time.perf_counter()
        for _ in range(100):
            q, s = ext.symmetric_quantize(data, bits=4)
        c_quant_time = (time.perf_counter() - t0) / 100
        results["c_quantize_ms"] = round(c_quant_time * 1000, 3)
        results["quantize_speedup"] = round(py_quant_time / c_quant_time, 2) if c_quant_time > 0 else 0
        print(f"  C quantize:             {c_quant_time*1000:.3f} ms ({results['quantize_speedup']}x)")

        # Bit allocation
        ext.score_synthetic(seed=42)
        alloc = ext.allocate_bits()
        bits_values = [k for k, v in alloc.values()]
        print(f"  Allocation range: {min(bits_values)}-{max(bits_values)} bits")
        print(f"  Avg bits: {np.mean(bits_values):.1f}")
        results["allocation_range"] = f"{min(bits_values)}-{max(bits_values)}"
        results["avg_allocated_bits"] = round(float(np.mean(bits_values)), 1)

    except Exception as e:
        print(f"  [C extension error: {e}]")
        results["c_score_ms"] = None

    return results


# ── 5. Full Pipeline: Compress + Quality ─────────────────────────────────

def benchmark_full_pipeline(n_layers=4, n_heads=8, seq_len=2048, head_dim=128):
    """Full TurboMOQ compress → decompress with quality metrics."""
    print("\n" + "=" * 60)
    print("5. FULL PIPELINE BENCHMARK")
    print("=" * 60)

    from turbomoq import TurboMOQCompressor, HeadScorer

    rng = np.random.default_rng(42)
    k = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float32)
    v = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float32)

    scorer = HeadScorer(n_layers, n_heads, head_dim)
    results = {}

    for bits in [4, 3, 2]:
        comp = TurboMOQCompressor(scorer, k_bits=bits, v_bits=bits)
        comp.calibrate(v)

        t0 = time.perf_counter()
        compressed = comp.compress(k, v)
        compress_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        k_hat, v_hat = comp.decompress(compressed)
        decompress_time = time.perf_counter() - t0

        stats = compressed.stats()
        k_cos = _cosine(k.ravel(), k_hat.ravel())
        v_cos = _cosine(v.ravel(), v_hat.ravel())

        key = f"{bits}bit"
        results[key] = {
            "compress_ms": round(compress_time * 1000, 1),
            "decompress_ms": round(decompress_time * 1000, 1),
            "compression_ratio": stats["compression_ratio"],
            "key_cosine": round(k_cos, 4),
            "value_cosine": round(v_cos, 4),
            "memory_mb": stats["memory_mb"],
        }
        print(f"\n  {bits}-bit:")
        print(f"    Compress:   {compress_time*1000:.1f} ms")
        print(f"    Decompress: {decompress_time*1000:.1f} ms")
        print(f"    Ratio:      {stats['compression_ratio']}x")
        print(f"    Key cos:    {k_cos:.4f}")
        print(f"    Value cos:  {v_cos:.4f}")
        print(f"    Memory:     {stats['memory_mb']:.2f} MB (FP16: {stats['fp16_mb']:.2f} MB)")

    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("TurboMOQ Benchmark — Mac mini 16GB")
    print("=" * 60)

    info = system_info()
    print(f"Python:    {info['python'].split()[0]}")
    print(f"Platform:  {info['platform']}")
    print(f"Machine:   {info['machine']}")
    print(f"RAM:       {info['ram_gb']} GB")

    all_results = {"system": info}
    all_results["mlx"] = benchmark_mlx()
    all_results["attention_demotion"] = benchmark_attention_demotion()
    all_results["qjl"] = benchmark_qjl()
    all_results["c_extension"] = benchmark_c_extension()
    all_results["full_pipeline"] = benchmark_full_pipeline()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    mlx_speedup = all_results["mlx"].get("rotation_speedup")
    if mlx_speedup:
        print(f"  MLX rotation speedup:    {mlx_speedup}x over numpy")
    else:
        print(f"  MLX: not available (Intel Mac)")

    c_speedup = all_results["c_extension"].get("score_speedup")
    if c_speedup:
        print(f"  C ext scoring speedup:   {c_speedup}x over Python")

    attn_ratio = all_results["attention_demotion"].get("compression_ratio")
    if attn_ratio:
        print(f"  Attention demotion:      {attn_ratio}x compression")

    for bits in [4, 3, 2]:
        key = f"{bits}bit"
        if key in all_results["full_pipeline"]:
            d = all_results["full_pipeline"][key]
            print(f"  {bits}-bit pipeline:        cos(K)={d['key_cosine']}, cos(V)={d['value_cosine']}, {d['compression_ratio']}x")

    # Save results
    out_path = "benchmark-results-raw/mac_mini_16gb_results.json"
    try:
        import os
        os.makedirs("benchmark-results-raw", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")
    except Exception as e:
        print(f"\n  Could not save results: {e}")

    return all_results


if __name__ == "__main__":
    main()
