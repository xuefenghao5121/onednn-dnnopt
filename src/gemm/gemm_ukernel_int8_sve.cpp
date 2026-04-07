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
    // SVE path: wider vectors + svmaxv for fast horizontal reduction
    svfloat32_t vmax_sve = svdup_f32(0);
    for (int i = 0; i < rows; ++i) {
        const float* row = data + i * ld;
        int j = 0;
        for (; j < cols; ) {
            svbool_t pg = svwhilelt_b32(j, cols);
            svfloat32_t v = svld1_f32(pg, row + j);
            vmax_sve = svmax_f32_m(svptrue_b32(), vmax_sve, svabs_f32_x(pg, v));
            j += (int)svcntw();
        }
    }
    float amax = svmaxv_f32(svptrue_b32(), vmax_sve);
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

/// INT8 SMMLA VLA microkernel: Mr=8, Nr=2*VL_f32.
/// SMMLA operates on 128-bit segments, so we use NEON vmmlaq_s32 intrinsics
/// with all accumulators kept in registers (no stack spills).
///
/// SVE-256 (Nr=16): 8 col-pairs, 4×8=32 register accumulators
/// SVE-512 (Nr=32): 16 col-pairs — uses 8-way col-pair chunks
///
/// K-loop processes Kgroup=8 INT8 elements per iteration.
static void gemm_ukernel_int8_sve_vla(int K,
                                       const int8_t* packed_A,
                                       const int8_t* packed_B,
                                       float* C, int ldc,
                                       float alpha, float beta,
                                       float dequant_scale) {
    const int vl = (int)svcntw();
    const int Nr = 2 * vl;
    const int n_col_pairs = Nr / 2;
    constexpr int Kgroup = 8;

    // Process col-pairs in chunks of 8 to cap register pressure
    constexpr int CP_CHUNK = 8;
    float scale = dequant_scale * alpha;

    for (int cp_base = 0; cp_base < n_col_pairs; cp_base += CP_CHUNK) {
        int cp_end = std::min(cp_base + CP_CHUNK, n_col_pairs);
        int cp_count = cp_end - cp_base;

        // 32 INT32 accumulators: 4 row-pairs × up to 8 col-pairs
        int32x4_t c00={}, c01={}, c02={}, c03={}, c04={}, c05={}, c06={}, c07={};
        int32x4_t c10={}, c11={}, c12={}, c13={}, c14={}, c15={}, c16={}, c17={};
        int32x4_t c20={}, c21={}, c22={}, c23={}, c24={}, c25={}, c26={}, c27={};
        int32x4_t c30={}, c31={}, c32={}, c33={}, c34={}, c35={}, c36={}, c37={};
        c00 = c01 = c02 = c03 = c04 = c05 = c06 = c07 = vdupq_n_s32(0);
        c10 = c11 = c12 = c13 = c14 = c15 = c16 = c17 = vdupq_n_s32(0);
        c20 = c21 = c22 = c23 = c24 = c25 = c26 = c27 = vdupq_n_s32(0);
        c30 = c31 = c32 = c33 = c34 = c35 = c36 = c37 = vdupq_n_s32(0);

        const int8_t* pA = packed_A;
        const int8_t* pB = packed_B + cp_base * 16;  // 16 bytes per col-pair
        const int b_stride = n_col_pairs * 16;  // B stride per K-group

        // K-loop
        for (int k = 0; k < K; k += Kgroup) {
            int8x16_t a0 = vld1q_s8(pA);
            int8x16_t a1 = vld1q_s8(pA + 16);
            int8x16_t a2 = vld1q_s8(pA + 32);
            int8x16_t a3 = vld1q_s8(pA + 48);
            pA += 64;

            const int8_t* pb = pB;

#define INT8_VLA_COLPAIR(idx)                       \
    if ((idx) < cp_count) {                         \
        int8x16_t b = vld1q_s8(pb + (idx)*16);     \
        c0##idx = vmmlaq_s32(c0##idx, a0, b);       \
        c1##idx = vmmlaq_s32(c1##idx, a1, b);       \
        c2##idx = vmmlaq_s32(c2##idx, a2, b);       \
        c3##idx = vmmlaq_s32(c3##idx, a3, b);       \
    }

            INT8_VLA_COLPAIR(0); INT8_VLA_COLPAIR(1);
            INT8_VLA_COLPAIR(2); INT8_VLA_COLPAIR(3);
            INT8_VLA_COLPAIR(4); INT8_VLA_COLPAIR(5);
            INT8_VLA_COLPAIR(6); INT8_VLA_COLPAIR(7);
#undef INT8_VLA_COLPAIR

            pB += b_stride;
        }

        // Epilogue: INT32 → FP32, scale, store
#define STORE_INT8_VLA_PAIR(rp_row, a0, a1, a2, a3, a4, a5, a6, a7) \
    do {                                                              \
        int32x4_t* accs[] = {&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7}; \
        for (int ci = 0; ci < cp_count; ++ci) {                       \
            int col = (cp_base + ci) * 2;                             \
            int32x4_t block_i = *accs[ci];                            \
            float32x4_t block_f = vcvtq_f32_s32(block_i);            \
            float r0c0 = scale * vgetq_lane_f32(block_f, 0);         \
            float r0c1 = scale * vgetq_lane_f32(block_f, 1);         \
            float r1c0 = scale * vgetq_lane_f32(block_f, 2);         \
            float r1c1 = scale * vgetq_lane_f32(block_f, 3);         \
            float* Cr0 = &C[(rp_row) * ldc + col];                   \
            float* Cr1 = &C[((rp_row)+1) * ldc + col];               \
            if (beta == 0.0f) {                                       \
                Cr0[0] = r0c0; Cr0[1] = r0c1;                        \
                Cr1[0] = r1c0; Cr1[1] = r1c1;                        \
            } else {                                                  \
                Cr0[0] = r0c0 + beta * Cr0[0];                        \
                Cr0[1] = r0c1 + beta * Cr0[1];                        \
                Cr1[0] = r1c0 + beta * Cr1[0];                        \
                Cr1[1] = r1c1 + beta * Cr1[1];                        \
            }                                                         \
        }                                                             \
    } while(0)

        STORE_INT8_VLA_PAIR(0, c00,c01,c02,c03,c04,c05,c06,c07);
        STORE_INT8_VLA_PAIR(2, c10,c11,c12,c13,c14,c15,c16,c17);
        STORE_INT8_VLA_PAIR(4, c20,c21,c22,c23,c24,c25,c26,c27);
        STORE_INT8_VLA_PAIR(6, c30,c31,c32,c33,c34,c35,c36,c37);
#undef STORE_INT8_VLA_PAIR
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

// ============================================================
// SVE-128 INT8: same 8x8 SMMLA tile as NEON, priority=120
// ============================================================
// On SVE-128, SMMLA intrinsics are identical to NEON. The benefit
// comes from SVE-accelerated packing with vectorized quantization.

// Reuse the NEON SMMLA microkernel (declared in gemm_ukernel_int8_neon.cpp)
extern void gemm_ukernel_int8_8x8(int K, const int8_t* packed_A,
                                    const int8_t* packed_B,
                                    float* C, int ldc, float alpha,
                                    float beta, float dequant_scale);

static void ukernel_int8_sve128_wrap(int K, const void* packed_A,
                                      const void* packed_B,
                                      float* C, int ldc, float alpha,
                                      float beta, float dequant_scale) {
    gemm_ukernel_int8_8x8(K,
                            static_cast<const int8_t*>(packed_A),
                            static_cast<const int8_t*>(packed_B),
                            C, ldc, alpha, beta, dequant_scale);
}

static const GemmMicrokernelDesc sve128_int8_desc = {
    "sve128_int8_8x8",
    GemmDataType::kINT8,
    kSVE | kI8MM,             // required_hwcaps
    8,                        // Mr
    8,                        // Nr (fixed)
    8,                        // Kgroup
    false,                    // nr_is_vla: NO
    120,                      // priority: higher than NEON 100
    sizeof(int8_t),
    sizeof(int8_t),
    0,                        // min_sve_bits: any SVE
    ukernel_int8_sve128_wrap,
    pack_a_int8_sve_wrap,     // SVE packing (vectorized quantization)
    pack_b_int8_sve_wrap,
};

static RegisterKernel reg_sve128_int8(sve128_int8_desc);

}  // namespace dnnopt

#endif  // __ARM_FEATURE_SVE && __ARM_FEATURE_SVE_MATMUL_INT8
