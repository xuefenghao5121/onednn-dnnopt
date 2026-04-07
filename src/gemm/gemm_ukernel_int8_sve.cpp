/// @file gemm_ukernel_int8_sve.cpp
/// SVE INT8 GEMM microkernel using svmmla_s32 (SMMLA).
///
/// VLA design: Mr=8, Nr=2*svcntw() (VLA)
/// On SVE-128: equivalent to NEON SMMLA 8x8
/// On SVE-256+: wider tile for more parallelism
///
/// Kgroup=8 (each svmmla/SMMLA processes 8 INT8 elements per K-step)

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_MATMUL_INT8)
#include <arm_sve.h>
#include <arm_neon.h>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace dnnopt {

// ============================================================
// INT8 quantization helpers (same as NEON version)
// ============================================================

static float compute_quant_scale_sve(const float* data, int rows, int cols, int ld) {
    float32x4_t vmax = vdupq_n_f32(0);
    for (int i = 0; i < rows; ++i) {
        const float* row = data + i * ld;
        int j = 0;
        for (; j + 3 < cols; j += 4) {
            float32x4_t v = vld1q_f32(row + j);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        for (; j < cols; ++j) {
            float av = std::fabs(row[j]);
            vmax = vmaxq_f32(vmax, vdupq_n_f32(av));
        }
    }
    float amax = vmaxvq_f32(vmax);
    if (amax == 0.0f) return 1.0f;
    return amax / 127.0f;
}

static inline int8_t quantize_s8_sve(float val, float inv_scale) {
    int32_t q = (int32_t)lrintf(val * inv_scale);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

// ============================================================
// SVE INT8 packing: FP32 → INT8 quantization + pack
// ============================================================

static void pack_a_int8_sve(int m_len, int k_len, const float* A, int lda,
                             void* packed_out, int Mr, float* scale_out) {
    int8_t* packed = static_cast<int8_t*>(packed_out);
    constexpr int Kgroup = 8;
    int k_padded = ((k_len + Kgroup - 1) / Kgroup) * Kgroup;

    *scale_out = compute_quant_scale_sve(A, m_len, k_len, lda);
    float inv_scale = 1.0f / *scale_out;

    for (int i = 0; i < m_len; i += Mr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            for (int rp = 0; rp < Mr; rp += 2) {
                int r0 = i + rp;
                int r1 = i + rp + 1;
                for (int kk = 0; kk < Kgroup; ++kk) {
                    float v0 = 0, v1 = 0;
                    if (r0 < m_len && (k + kk) < k_len) v0 = A[r0 * lda + k + kk];
                    if (r1 < m_len && (k + kk) < k_len) v1 = A[r1 * lda + k + kk];
                    packed[kk]          = quantize_s8_sve(v0, inv_scale);
                    packed[Kgroup + kk] = quantize_s8_sve(v1, inv_scale);
                }
                packed += 16;
            }
        }
    }
}

static void pack_b_int8_sve(int k_len, int n_len, const float* B, int ldb,
                             void* packed_out, int Nr, float* scale_out) {
    int8_t* packed = static_cast<int8_t*>(packed_out);
    constexpr int Kgroup = 8;
    int k_padded = ((k_len + Kgroup - 1) / Kgroup) * Kgroup;

    *scale_out = compute_quant_scale_sve(B, k_len, n_len, ldb);
    float inv_scale = 1.0f / *scale_out;

    for (int j = 0; j < n_len; j += Nr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            for (int cp = 0; cp < Nr; cp += 2) {
                int c0 = j + cp;
                int c1 = j + cp + 1;
                for (int kk = 0; kk < Kgroup; ++kk) {
                    float v0 = 0, v1 = 0;
                    if (c0 < n_len && (k + kk) < k_len) v0 = B[(k + kk) * ldb + c0];
                    if (c1 < n_len && (k + kk) < k_len) v1 = B[(k + kk) * ldb + c1];
                    packed[kk]          = quantize_s8_sve(v0, inv_scale);
                    packed[Kgroup + kk] = quantize_s8_sve(v1, inv_scale);
                }
                packed += 16;
            }
        }
    }
}

// ============================================================
// SVE INT8 VLA microkernel
// ============================================================

static void gemm_ukernel_int8_sve_vla(int K,
                                       const int8_t* packed_A,
                                       const int8_t* packed_B,
                                       float* C, int ldc,
                                       float alpha, float beta,
                                       float dequant_scale) {
    const int vl = (int)svcntw();  // FP32 elements per SVE reg
    const int Nr = 2 * vl;
    constexpr int Kgroup = 8;
    constexpr int Mr = 8;
    const int n_col_pairs = Nr / 2;
    const int n_accs = 4 * n_col_pairs;

    // Dynamic accumulator buffer
    int32_t acc_buf[4 * 32 * 4] = {};  // Max SVE-512: 4 rp × 32 cp × 4 elements
    memset(acc_buf, 0, n_accs * 4 * sizeof(int32_t));

    // K-loop
    for (int k = 0; k < K; k += Kgroup) {
        // Load A row-pairs: 4 × int8x16_t (128-bit each, 2×8 INT8)
        int8x16_t a0 = vld1q_s8(packed_A + 0);
        int8x16_t a1 = vld1q_s8(packed_A + 16);
        int8x16_t a2 = vld1q_s8(packed_A + 32);
        int8x16_t a3 = vld1q_s8(packed_A + 48);
        packed_A += 64;

        for (int cp = 0; cp < n_col_pairs; ++cp) {
            int8x16_t b = vld1q_s8(packed_B);
            packed_B += 16;

            int base = cp * 4;
            int32x4_t c0 = vld1q_s32(&acc_buf[(base + 0) * 4]);
            int32x4_t c1 = vld1q_s32(&acc_buf[(base + 1) * 4]);
            int32x4_t c2 = vld1q_s32(&acc_buf[(base + 2) * 4]);
            int32x4_t c3 = vld1q_s32(&acc_buf[(base + 3) * 4]);

            c0 = vmmlaq_s32(c0, a0, b);
            c1 = vmmlaq_s32(c1, a1, b);
            c2 = vmmlaq_s32(c2, a2, b);
            c3 = vmmlaq_s32(c3, a3, b);

            vst1q_s32(&acc_buf[(base + 0) * 4], c0);
            vst1q_s32(&acc_buf[(base + 1) * 4], c1);
            vst1q_s32(&acc_buf[(base + 2) * 4], c2);
            vst1q_s32(&acc_buf[(base + 3) * 4], c3);
        }
    }

    // Epilogue: INT32 → FP32, scale, store
    float scale = dequant_scale * alpha;

    for (int cp = 0; cp < n_col_pairs; ++cp) {
        int col = cp * 2;
        for (int rp = 0; rp < 4; ++rp) {
            int row = rp * 2;
            int idx = (cp * 4 + rp) * 4;
            int32x4_t block_i = vld1q_s32(&acc_buf[idx]);
            float32x4_t block_f = vcvtq_f32_s32(block_i);
            float32x4_t scale_v = vdupq_n_f32(scale);
            block_f = vmulq_f32(block_f, scale_v);

            float r0c0 = vgetq_lane_f32(block_f, 0);
            float r0c1 = vgetq_lane_f32(block_f, 1);
            float r1c0 = vgetq_lane_f32(block_f, 2);
            float r1c1 = vgetq_lane_f32(block_f, 3);

            float* Cr0 = &C[row * ldc + col];
            float* Cr1 = &C[(row + 1) * ldc + col];

            if (beta == 0.0f) {
                Cr0[0] = r0c0;  Cr0[1] = r0c1;
                Cr1[0] = r1c0;  Cr1[1] = r1c1;
            } else {
                Cr0[0] = r0c0 + beta * Cr0[0];
                Cr0[1] = r0c1 + beta * Cr0[1];
                Cr1[0] = r1c0 + beta * Cr1[0];
                Cr1[1] = r1c1 + beta * Cr1[1];
            }
        }
    }
}

// ============================================================
// Registry wrappers
// ============================================================

static void pack_a_int8_sve_wrap(int m_len, int k_len, const float* A, int lda,
                                  void* packed_A, int Mr, float* scale_out) {
    pack_a_int8_sve(m_len, k_len, A, lda, packed_A, Mr, scale_out);
}

static void pack_b_int8_sve_wrap(int k_len, int n_len, const float* B, int ldb,
                                  void* packed_B, int Nr, float* scale_out) {
    pack_b_int8_sve(k_len, n_len, B, ldb, packed_B, Nr, scale_out);
}

static void ukernel_int8_sve_wrap(int K, const void* packed_A,
                                   const void* packed_B,
                                   float* C, int ldc, float alpha,
                                   float beta, float dequant_scale) {
    gemm_ukernel_int8_sve_vla(K,
                               static_cast<const int8_t*>(packed_A),
                               static_cast<const int8_t*>(packed_B),
                               C, ldc, alpha, beta, dequant_scale);
}

// SVE INT8 VLA: only on wide SVE (256+)
static const GemmMicrokernelDesc sve_int8_vla_desc = {
    "sve_int8_8xVL",
    GemmDataType::kINT8,
    kSVE | kSVEI8MM | kI8MM,
    8,                    // Mr
    0,                    // Nr (computed)
    8,                    // Kgroup
    true,                 // nr_is_vla
    200,                  // priority
    sizeof(int8_t),
    sizeof(int8_t),
    256,                  // min_sve_bits: only on wide SVE
    ukernel_int8_sve_wrap,
    pack_a_int8_sve_wrap,
    pack_b_int8_sve_wrap,
};

static RegisterKernel reg_sve_int8_vla(sve_int8_vla_desc);

}  // namespace dnnopt

#endif  // __ARM_FEATURE_SVE && __ARM_FEATURE_SVE_MATMUL_INT8
