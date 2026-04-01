/**
 * turbomoq_kv.c — TurboMOQ KV cache compression C implementation
 *
 * Topology-aware per-head bit allocation + symmetric quantization.
 * Designed for integration with llama.cpp at the Metal kernel level.
 *
 * Compilation:
 *   macOS:  clang -O2 -shared -fPIC -o libturbomoq.dylib turbomoq_kv.c -lm
 *   Linux:  gcc   -O2 -shared -fPIC -o libturbomoq.so    turbomoq_kv.c -lm
 */

#include "turbomoq_kv.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── LCG PRNG (deterministic, no external deps) ────────────────────────── */

static inline uint32_t lcg_next(uint32_t *state) {
    *state = (*state) * 1664525u + 1013904223u;
    return *state;
}

static inline float lcg_randn(uint32_t *state) {
    /* Box-Muller transform from uniform LCG */
    float u1 = (float)(lcg_next(state) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    float u2 = (float)(lcg_next(state) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ── Init ──────────────────────────────────────────────────────────────── */

int turbomoq_init(turbomoq_ctx_t *ctx, const turbomoq_config_t *config) {
    if (!ctx || !config) return -1;
    if (config->n_layers > TURBOMOQ_MAX_LAYERS) return -1;
    if (config->n_heads > TURBOMOQ_MAX_HEADS) return -1;

    memset(ctx, 0, sizeof(*ctx));
    ctx->config = *config;
    ctx->n_scores = 0;
    ctx->calibrated = 0;
    return 0;
}

/* ── Position Importance ───────────────────────────────────────────────── */

float turbomoq_position_importance(int layer_idx, int n_layers) {
    if (n_layers <= 1) return 1.0f;
    float mid = (float)(n_layers - 1) / 2.0f;
    float x = ((float)layer_idx - mid) / mid;
    return expf(-2.0f * x * x);
}

/* ── Topological Persistence ───────────────────────────────────────────── */

float turbomoq_compute_persistence(const float *attn, int seq_len) {
    if (!attn || seq_len <= 0) return 0.0f;

    /*
     * Approximate spectral gap via power iteration on symmetric part.
     * Full eigendecomp is too expensive for a C kernel — 2 rounds of
     * power iteration gives a good enough top-eigenvalue estimate.
     */
    int n = seq_len;
    int nn = n * n;

    /* Compute variance as a simpler proxy when seq_len is small */
    if (n <= 4) {
        float mean = 0.0f;
        for (int i = 0; i < nn; i++) mean += attn[i];
        mean /= (float)nn;
        float var = 0.0f;
        for (int i = 0; i < nn; i++) {
            float d = attn[i] - mean;
            var += d * d;
        }
        var /= (float)nn;
        return fminf(1.0f, sqrtf(var));
    }

    /* Power iteration for top eigenvalue of symmetric part */
    float *v = (float *)malloc(n * sizeof(float));
    float *w = (float *)malloc(n * sizeof(float));
    if (!v || !w) {
        free(v); free(w);
        return 0.0f;
    }

    /* Init v = [1, 0, 0, ...] */
    memset(v, 0, n * sizeof(float));
    v[0] = 1.0f;

    float lambda1 = 0.0f;
    for (int iter = 0; iter < 20; iter++) {
        /* w = (A + A^T)/2 * v */
        for (int i = 0; i < n; i++) {
            float s = 0.0f;
            for (int j = 0; j < n; j++) {
                float sym = 0.5f * (attn[i * n + j] + attn[j * n + i]);
                s += sym * v[j];
            }
            w[i] = s;
        }
        /* Normalize */
        float norm = 0.0f;
        for (int i = 0; i < n; i++) norm += w[i] * w[i];
        norm = sqrtf(norm);
        if (norm < 1e-10f) break;
        lambda1 = norm;
        for (int i = 0; i < n; i++) v[i] = w[i] / norm;
    }

    /* Deflate for second eigenvalue */
    float lambda2 = 0.0f;
    /* Reset v for second eigenvector (orthogonal start) */
    float *v1 = (float *)malloc(n * sizeof(float));
    if (v1) {
        memcpy(v1, v, n * sizeof(float));
        memset(v, 0, n * sizeof(float));
        v[n > 1 ? 1 : 0] = 1.0f;

        for (int iter = 0; iter < 20; iter++) {
            for (int i = 0; i < n; i++) {
                float s = 0.0f;
                for (int j = 0; j < n; j++) {
                    float sym = 0.5f * (attn[i * n + j] + attn[j * n + i]);
                    s += sym * v[j];
                }
                w[i] = s;
            }
            /* Deflate: w = w - (w . v1) * v1 */
            float dot = 0.0f;
            for (int i = 0; i < n; i++) dot += w[i] * v1[i];
            for (int i = 0; i < n; i++) w[i] -= dot * v1[i];

            float norm = 0.0f;
            for (int i = 0; i < n; i++) norm += w[i] * w[i];
            norm = sqrtf(norm);
            if (norm < 1e-10f) break;
            lambda2 = norm;
            for (int i = 0; i < n; i++) v[i] = w[i] / norm;
        }
        free(v1);
    }

    free(v);
    free(w);

    float gap = lambda1 - lambda2;
    float result = (fabsf(lambda1) > 1e-10f) ? gap / fabsf(lambda1) : 0.0f;
    return fminf(1.0f, fmaxf(0.0f, result));
}

/* ── Gradient Sensitivity ──────────────────────────────────────────────── */

float turbomoq_compute_gradient(const float *attn, int n_elements) {
    if (!attn || n_elements <= 0) return 0.0f;
    float mean = 0.0f;
    for (int i = 0; i < n_elements; i++) mean += attn[i];
    mean /= (float)n_elements;
    float var = 0.0f;
    for (int i = 0; i < n_elements; i++) {
        float d = attn[i] - mean;
        var += d * d;
    }
    return var / (float)n_elements;
}

/* ── Synthetic Scoring ─────────────────────────────────────────────────── */

int turbomoq_score_synthetic(turbomoq_ctx_t *ctx, uint32_t seed) {
    if (!ctx) return -1;
    int n_layers = ctx->config.n_layers;
    int n_heads = ctx->config.n_heads;
    int idx = 0;

    uint32_t state = seed;

    for (int l = 0; l < n_layers; l++) {
        float pos = turbomoq_position_importance(l, n_layers);
        for (int h = 0; h < n_heads; h++) {
            /* Beta(2,2) approximation via averaged uniforms */
            float u1 = (float)(lcg_next(&state) & 0xFFFF) / 65535.0f;
            float u2 = (float)(lcg_next(&state) & 0xFFFF) / 65535.0f;
            float topo = (u1 + u2) / 2.0f;  /* rough Beta(2,2) */

            u1 = (float)(lcg_next(&state) & 0xFFFF) / 65535.0f;
            u2 = (float)(lcg_next(&state) & 0xFFFF) / 65535.0f;
            float grad = (u1 + u2) / 2.0f;

            float combined = ctx->config.topo_weight * topo
                           + ctx->config.grad_weight * grad
                           + ctx->config.position_weight * pos;

            ctx->scores[idx].layer = l;
            ctx->scores[idx].head = h;
            ctx->scores[idx].topo_score = topo;
            ctx->scores[idx].grad_score = grad;
            ctx->scores[idx].position_score = pos;
            ctx->scores[idx].combined = combined;
            ctx->scores[idx].allocated_k_bits = ctx->config.default_k_bits;
            ctx->scores[idx].allocated_v_bits = ctx->config.default_v_bits;
            idx++;
        }
    }

    ctx->n_scores = idx;
    ctx->calibrated = 1;
    return idx;
}

/* ── Score from Real Attention ─────────────────────────────────────────── */

int turbomoq_score_from_attention(turbomoq_ctx_t *ctx,
                                   const float **attn_maps,
                                   int n_maps, int seq_len) {
    if (!ctx || !attn_maps) return -1;
    int n_layers = ctx->config.n_layers;
    int n_heads = ctx->config.n_heads;
    int expected = n_layers * n_heads;
    if (n_maps < expected) return -1;

    float max_topo = 1e-10f;
    float max_grad = 1e-10f;

    /* First pass: compute raw scores */
    for (int i = 0; i < expected; i++) {
        float topo = turbomoq_compute_persistence(attn_maps[i], seq_len);
        float grad = turbomoq_compute_gradient(attn_maps[i], seq_len * seq_len);
        ctx->scores[i].topo_score = topo;
        ctx->scores[i].grad_score = grad;
        if (topo > max_topo) max_topo = topo;
        if (grad > max_grad) max_grad = grad;
    }

    /* Second pass: normalize and combine */
    int idx = 0;
    for (int l = 0; l < n_layers; l++) {
        float pos = turbomoq_position_importance(l, n_layers);
        for (int h = 0; h < n_heads; h++) {
            ctx->scores[idx].layer = l;
            ctx->scores[idx].head = h;
            ctx->scores[idx].topo_score /= max_topo;
            ctx->scores[idx].grad_score /= max_grad;
            ctx->scores[idx].position_score = pos;
            ctx->scores[idx].combined =
                ctx->config.topo_weight * ctx->scores[idx].topo_score
              + ctx->config.grad_weight * ctx->scores[idx].grad_score
              + ctx->config.position_weight * pos;
            ctx->scores[idx].allocated_k_bits = ctx->config.default_k_bits;
            ctx->scores[idx].allocated_v_bits = ctx->config.default_v_bits;
            idx++;
        }
    }

    ctx->n_scores = idx;
    ctx->calibrated = 1;
    return idx;
}

/* ── Bit Allocation ────────────────────────────────────────────────────── */

/* qsort comparator — descending by combined score */
static int cmp_score_desc(const void *a, const void *b) {
    float sa = ((const turbomoq_head_score_t *)a)->combined;
    float sb = ((const turbomoq_head_score_t *)b)->combined;
    if (sa > sb) return -1;
    if (sa < sb) return  1;
    return 0;
}

int turbomoq_allocate_bits(turbomoq_ctx_t *ctx) {
    if (!ctx || !ctx->calibrated) return -1;

    int n = ctx->n_scores;
    float target = ctx->config.target_avg_bits;
    float total_budget = target * (float)n;

    /* Sort by combined score descending (in-place) */
    qsort(ctx->scores, n, sizeof(turbomoq_head_score_t), cmp_score_desc);

    float remaining = total_budget;
    for (int i = 0; i < n; i++) {
        int heads_left = n - i;
        float avg_left = remaining / (float)heads_left;
        float score = ctx->scores[i].combined;
        int bits;

        if (score > 0.75f)
            bits = (int)roundf(avg_left * 1.5f);
        else if (score > 0.5f)
            bits = (int)roundf(avg_left);
        else if (score > 0.25f)
            bits = (int)roundf(avg_left * 0.7f);
        else
            bits = (int)roundf(avg_left * 0.5f);

        if (bits < 1) bits = 1;
        if (bits > 8) bits = 8;

        ctx->scores[i].allocated_k_bits = bits;
        ctx->scores[i].allocated_v_bits = bits;
        remaining -= (float)bits;
    }

    return 0;
}

int turbomoq_get_bits(const turbomoq_ctx_t *ctx, int layer, int head,
                       int *k_bits, int *v_bits) {
    if (!ctx || !ctx->calibrated) return -1;
    for (int i = 0; i < ctx->n_scores; i++) {
        if (ctx->scores[i].layer == layer && ctx->scores[i].head == head) {
            *k_bits = ctx->scores[i].allocated_k_bits;
            *v_bits = ctx->scores[i].allocated_v_bits;
            return 0;
        }
    }
    /* Not found — use defaults */
    *k_bits = ctx->config.default_k_bits;
    *v_bits = ctx->config.default_v_bits;
    return 0;
}

/* ── Symmetric Quantize / Dequantize ───────────────────────────────────── */

int turbomoq_symmetric_quantize(const float *src, turbomoq_qtensor_t *dst,
                                 int rows, int cols, int bits) {
    if (!src || !dst || !dst->data || !dst->scales) return -1;

    int qmax = (1 << bits) - 1;
    int half = qmax / 2;

    dst->rows = rows;
    dst->cols = cols;
    dst->bits = bits;

    /* Per-column abs-max → scale */
    for (int c = 0; c < cols; c++) {
        float amax = 0.0f;
        for (int r = 0; r < rows; r++) {
            float a = fabsf(src[r * cols + c]);
            if (a > amax) amax = a;
        }
        dst->scales[c] = (amax > 0.0f) ? amax / (float)half : 1.0f;
    }

    /* Quantize */
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float q = roundf(src[r * cols + c] / dst->scales[c]);
            if (q < -(float)half) q = -(float)half;
            if (q > (float)(half - 1)) q = (float)(half - 1);
            dst->data[r * cols + c] = (int8_t)q;
        }
    }

    return 0;
}

int turbomoq_symmetric_dequantize(const turbomoq_qtensor_t *src, float *dst) {
    if (!src || !dst || !src->data || !src->scales) return -1;

    for (int r = 0; r < src->rows; r++) {
        for (int c = 0; c < src->cols; c++) {
            dst[r * src->cols + c] = (float)src->data[r * src->cols + c] * src->scales[c];
        }
    }
    return 0;
}

/* ── Rotation ──────────────────────────────────────────────────────────── */

int turbomoq_rotate(const float *src, float *dst, const float *Q,
                     int rows, int dim) {
    if (!src || !dst || !Q) return -1;
    /* dst = src @ Q  (row-major matmul) */
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < dim; c++) {
            float s = 0.0f;
            for (int k = 0; k < dim; k++) {
                s += src[r * dim + k] * Q[k * dim + c];
            }
            dst[r * dim + c] = s;
        }
    }
    return 0;
}

int turbomoq_unrotate(const float *src, float *dst, const float *Q,
                       int rows, int dim) {
    if (!src || !dst || !Q) return -1;
    /* dst = src @ Q^T  (Q^T[i][j] = Q[j][i]) */
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < dim; c++) {
            float s = 0.0f;
            for (int k = 0; k < dim; k++) {
                s += src[r * dim + k] * Q[c * dim + k];
            }
            dst[r * dim + c] = s;
        }
    }
    return 0;
}

int turbomoq_generate_rotation(float *Q, int dim, uint32_t seed) {
    if (!Q || dim <= 0) return -1;

    uint32_t state = seed;

    /* Generate random matrix */
    float *H = (float *)malloc(dim * dim * sizeof(float));
    if (!H) return -1;

    for (int i = 0; i < dim * dim; i++) {
        H[i] = lcg_randn(&state);
    }

    /*
     * Modified Gram-Schmidt QR decomposition.
     * For dim <= 256 this is fast enough; for larger dims,
     * the caller should use LAPACK and pass Q directly.
     */
    float *R_diag = (float *)calloc(dim, sizeof(float));
    if (!R_diag) { free(H); return -1; }

    /* Copy H → Q (in-place MGS) */
    memcpy(Q, H, dim * dim * sizeof(float));

    for (int j = 0; j < dim; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < dim; i++) {
            norm += Q[i * dim + j] * Q[i * dim + j];
        }
        norm = sqrtf(norm);
        if (norm < 1e-10f) { norm = 1e-10f; }
        R_diag[j] = norm;
        for (int i = 0; i < dim; i++) {
            Q[i * dim + j] /= norm;
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < dim; k++) {
            float dot = 0.0f;
            for (int i = 0; i < dim; i++) {
                dot += Q[i * dim + j] * Q[i * dim + k];
            }
            for (int i = 0; i < dim; i++) {
                Q[i * dim + k] -= dot * Q[i * dim + j];
            }
        }
    }

    /* Fix sign: multiply each column by sign(R_diag) */
    for (int j = 0; j < dim; j++) {
        if (R_diag[j] < 0.0f) {
            for (int i = 0; i < dim; i++) {
                Q[i * dim + j] = -Q[i * dim + j];
            }
        }
    }

    free(H);
    free(R_diag);
    return 0;
}

/* ── Utility ───────────────────────────────────────────────────────────── */

const char *turbomoq_version(void) {
    return TURBOMOQ_VERSION;
}

void turbomoq_free(turbomoq_ctx_t *ctx) {
    if (ctx) {
        memset(ctx, 0, sizeof(*ctx));
    }
}
