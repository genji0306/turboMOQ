/**
 * turbomoq_kv.h — TurboMOQ KV cache compression C API
 *
 * Topology-aware per-head bit allocation for llama.cpp integration.
 * Compiles to libturbomoq.dylib (macOS) or libturbomoq.so (Linux).
 *
 * Thread-safe: all functions are reentrant. The context struct holds
 * all mutable state.
 */

#ifndef TURBOMOQ_KV_H
#define TURBOMOQ_KV_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Configuration ─────────────────────────────────────────────────────── */

#define TURBOMOQ_MAX_LAYERS 128
#define TURBOMOQ_MAX_HEADS  128
#define TURBOMOQ_VERSION    "0.2.0"

typedef struct {
    int n_layers;
    int n_heads;
    int head_dim;
    int default_k_bits;
    int default_v_bits;
    float target_avg_bits;      /* Target average bits across all heads */
    float topo_weight;          /* Weight for topological score [0,1] */
    float grad_weight;          /* Weight for gradient score [0,1] */
    float position_weight;      /* Weight for position score [0,1] */
} turbomoq_config_t;

/* ── Head Score ────────────────────────────────────────────────────────── */

typedef struct {
    int layer;
    int head;
    float topo_score;           /* Topological persistence [0,1] */
    float grad_score;           /* Gradient sensitivity [0,1] */
    float position_score;       /* Layer position bell [0,1] */
    float combined;             /* Weighted combination [0,1] */
    int allocated_k_bits;       /* Assigned key bits */
    int allocated_v_bits;       /* Assigned value bits */
} turbomoq_head_score_t;

/* ── Context ───────────────────────────────────────────────────────────── */

typedef struct {
    turbomoq_config_t config;
    turbomoq_head_score_t scores[TURBOMOQ_MAX_LAYERS * TURBOMOQ_MAX_HEADS];
    int n_scores;
    int calibrated;             /* 1 if scores have been computed */
} turbomoq_ctx_t;

/* ── Quantized Tensor ──────────────────────────────────────────────────── */

typedef struct {
    int8_t *data;               /* Quantized values (caller-allocated) */
    float  *scales;             /* Per-channel scales (caller-allocated) */
    int     rows;
    int     cols;
    int     bits;
} turbomoq_qtensor_t;

/* ── API Functions ─────────────────────────────────────────────────────── */

/**
 * Initialize a TurboMOQ context with default config.
 * Returns 0 on success, -1 on error.
 */
int turbomoq_init(turbomoq_ctx_t *ctx, const turbomoq_config_t *config);

/**
 * Compute position importance for a layer (bell curve).
 * Returns value in [0, 1].
 */
float turbomoq_position_importance(int layer_idx, int n_layers);

/**
 * Compute topological persistence score from attention weights.
 * attn: (seq_len x seq_len) row-major attention matrix.
 * Returns score in [0, 1].
 */
float turbomoq_compute_persistence(const float *attn, int seq_len);

/**
 * Compute gradient sensitivity (variance of attention weights).
 * Returns score in [0, 1] (normalized by caller).
 */
float turbomoq_compute_gradient(const float *attn, int n_elements);

/**
 * Score all heads using synthetic data (no model needed).
 * Fills ctx->scores[] and sets ctx->calibrated = 1.
 * Returns number of heads scored.
 */
int turbomoq_score_synthetic(turbomoq_ctx_t *ctx, uint32_t seed);

/**
 * Score heads from real attention weight matrices.
 * attn_maps: array of (seq_len x seq_len) matrices, one per (layer, head).
 * n_maps: number of maps (should be n_layers * n_heads).
 * seq_len: attention matrix dimension.
 * Returns number of heads scored.
 */
int turbomoq_score_from_attention(turbomoq_ctx_t *ctx,
                                   const float **attn_maps,
                                   int n_maps, int seq_len);

/**
 * Allocate bits per head to hit target average.
 * Must be called after scoring. Updates allocated_k_bits / allocated_v_bits
 * in ctx->scores[].
 * Returns 0 on success, -1 if not calibrated.
 */
int turbomoq_allocate_bits(turbomoq_ctx_t *ctx);

/**
 * Get bit allocation for a specific (layer, head).
 * Sets *k_bits and *v_bits. Returns 0 on success.
 */
int turbomoq_get_bits(const turbomoq_ctx_t *ctx, int layer, int head,
                       int *k_bits, int *v_bits);

/**
 * Per-channel symmetric quantize.
 * src: (rows x cols) row-major float32 input.
 * dst: pre-allocated turbomoq_qtensor_t with data[rows*cols] and scales[cols].
 * Returns 0 on success.
 */
int turbomoq_symmetric_quantize(const float *src, turbomoq_qtensor_t *dst,
                                 int rows, int cols, int bits);

/**
 * Per-channel symmetric dequantize.
 * src: turbomoq_qtensor_t with quantized data.
 * dst: (rows x cols) row-major float32 output (caller-allocated).
 * Returns 0 on success.
 */
int turbomoq_symmetric_dequantize(const turbomoq_qtensor_t *src, float *dst);

/**
 * Apply random orthogonal rotation (QR-based).
 * src: (rows x dim) row-major float32 input.
 * dst: (rows x dim) row-major float32 output (caller-allocated).
 * Q: (dim x dim) orthogonal matrix (row-major).
 * Returns 0 on success.
 */
int turbomoq_rotate(const float *src, float *dst, const float *Q,
                     int rows, int dim);

/**
 * Apply inverse rotation (Q^T).
 */
int turbomoq_unrotate(const float *src, float *dst, const float *Q,
                       int rows, int dim);

/**
 * Generate random orthogonal matrix via QR decomposition.
 * Q: (dim x dim) output (caller-allocated).
 * Uses a simple LCG PRNG seeded by `seed`.
 * Returns 0 on success.
 */
int turbomoq_generate_rotation(float *Q, int dim, uint32_t seed);

/**
 * Get version string.
 */
const char *turbomoq_version(void);

/**
 * Free context resources (currently a no-op since all stack-allocated).
 */
void turbomoq_free(turbomoq_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* TURBOMOQ_KV_H */
