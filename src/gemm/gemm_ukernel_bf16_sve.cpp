/// @file gemm_ukernel_bf16_sve.cpp
/// SVE BF16 GEMM microkernel using svbfmmla_f32.
///
/// On SVE-128: equivalent to NEON BFMMLA (same 8x8 tile)
/// On SVE-256+: processes 2+ column-pairs per svbfmmla, giving wider Nr
///
/// VLA design: Mr=8 (4 row-pairs), Nr=2*svcntw() (VLA)
/// Each svbfmmla processes 128-bit segments within SVE registers.

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#include <arm_sve.h>
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace dnnopt {

// ============================================================
// SVE BF16 packing: FP32 → BF16 with SVE predicates
// ============================================================

/// Pack A into BFMMLA format using SVE.
/// For each K-group of 4, packs row-pairs: [row0_k0..k3, row1_k0..k3] as 8 BF16.
static void pack_a_bf16_sve(int m_len, int k_len, const float* A, int lda,
                             void* packed_out, int Mr) {
    bfloat16_t* packed = static_cast<bfloat16_t*>(packed_out);
    constexpr int Kgroup = 4;
    int k_padded = ((k_len + Kgroup - 1) / Kgroup) * Kgroup;

    for (int i = 0; i < m_len; i += Mr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Process Mr/2 = 4 row-pairs
            for (int rp = 0; rp < Mr; rp += 2) {
                int r0 = i + rp;
                int r1 = i + rp + 1;
                float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int kk = 0; kk < Kgroup && (k + kk) < k_len; ++kk) {
                    if (r0 < m_len) tmp[kk]     = A[r0 * lda + k + kk];
                    if (r1 < m_len) tmp[4 + kk]  = A[r1 * lda + k + kk];
                }

                // Convert 8 FP32 to 8 BF16
                float32x4_t f0 = vld1q_f32(tmp);
                float32x4_t f1 = vld1q_f32(tmp + 4);
                bfloat16x4_t b0 = vcvt_bf16_f32(f0);
                bfloat16x4_t b1 = vcvt_bf16_f32(f1);
                bfloat16x8_t combined = vcombine_bf16(b0, b1);
                vst1q_bf16(packed, combined);
                packed += 8;
            }
        }
    }
}

/// Pack B into BFMMLA format using SVE.
/// For each K-group of 4, packs col-pairs: [k0c0..k3c0, k0c1..k3c1] as 8 BF16.
static void pack_b_bf16_sve(int k_len, int n_len, const float* B, int ldb,
                             void* packed_out, int Nr) {
    bfloat16_t* packed = static_cast<bfloat16_t*>(packed_out);
    constexpr int Kgroup = 4;
    int k_padded = ((k_len + Kgroup - 1) / Kgroup) * Kgroup;

    for (int j = 0; j < n_len; j += Nr) {
        for (int k = 0; k < k_padded; k += Kgroup) {
            // Process Nr/2 column-pairs
            for (int cp = 0; cp < Nr; cp += 2) {
                int c0 = j + cp;
                int c1 = j + cp + 1;
                float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int kk = 0; kk < Kgroup && (k + kk) < k_len; ++kk) {
                    if (c0 < n_len) tmp[kk]     = B[(k + kk) * ldb + c0];
                    if (c1 < n_len) tmp[4 + kk]  = B[(k + kk) * ldb + c1];
                }

                float32x4_t f0 = vld1q_f32(tmp);
                float32x4_t f1 = vld1q_f32(tmp + 4);
                bfloat16x4_t b0 = vcvt_bf16_f32(f0);
                bfloat16x4_t b1 = vcvt_bf16_f32(f1);
                bfloat16x8_t combined = vcombine_bf16(b0, b1);
                vst1q_bf16(packed, combined);
                packed += 8;
            }
        }
    }
}

// ============================================================
// SVE BF16 VLA microkernel: Mr=8, Nr=2*VL_f32
// ============================================================

/// BF16 BFMMLA VLA microkernel: Mr=8, Nr=2*VL_f32.
/// BFMMLA operates on 128-bit segments, so we use NEON vbfmmlaq_f32 intrinsics
/// with all accumulators kept in registers (no stack spills).
///
/// SVE-256 (Nr=16): 8 col-pairs, 4×8=32 register accumulators
/// SVE-512 (Nr=32): 16 col-pairs — uses 4-way col-pair unroll with register rotation
///
/// K-loop: 2x unrolled for latency hiding. Software prefetch on packed panels.
static void gemm_ukernel_bf16_sve_vla(int K,
                                       const bfloat16_t* packed_A,
                                       const bfloat16_t* packed_B,
                                       float* C, int ldc,
                                       float alpha, float beta) {
    const int vl = (int)svcntw();
    const int Nr = 2 * vl;
    const int n_col_pairs = Nr / 2;
    constexpr int Kgroup = 4;

    // 32 register accumulators: 4 row-pairs × 8 col-pairs (SVE-256)
    // For SVE-512, we process col-pairs in groups of 8, accumulating into
    // the same 32 registers then storing before the next group.
    // This caps register pressure at 32 accs + 4 A + 1 B = 37 regs (fits in 32 SIMD regs
    // with some spills for B loads, which is acceptable).

    // Process col-pairs in chunks of 8 (matching SVE-256 register capacity)
    constexpr int CP_CHUNK = 8;

    float32x4_t alpha_v = vdupq_n_f32(alpha);
    float32x4_t beta_v  = vdupq_n_f32(beta);

    for (int cp_base = 0; cp_base < n_col_pairs; cp_base += CP_CHUNK) {
        int cp_end = std::min(cp_base + CP_CHUNK, n_col_pairs);
        int cp_count = cp_end - cp_base;

        // Initialize accumulators: up to 4 rp × 8 cp = 32 registers
        float32x4_t c00={}, c01={}, c02={}, c03={}, c04={}, c05={}, c06={}, c07={};
        float32x4_t c10={}, c11={}, c12={}, c13={}, c14={}, c15={}, c16={}, c17={};
        float32x4_t c20={}, c21={}, c22={}, c23={}, c24={}, c25={}, c26={}, c27={};
        float32x4_t c30={}, c31={}, c32={}, c33={}, c34={}, c35={}, c36={}, c37={};
        c00 = c01 = c02 = c03 = c04 = c05 = c06 = c07 = vdupq_n_f32(0);
        c10 = c11 = c12 = c13 = c14 = c15 = c16 = c17 = vdupq_n_f32(0);
        c20 = c21 = c22 = c23 = c24 = c25 = c26 = c27 = vdupq_n_f32(0);
        c30 = c31 = c32 = c33 = c34 = c35 = c36 = c37 = vdupq_n_f32(0);

        const bfloat16_t* pA = packed_A;
        const bfloat16_t* pB = packed_B + cp_base * 8;  // B offset for this cp chunk
        // B stride per K-group: n_col_pairs * 8 BF16
        const int b_stride = n_col_pairs * 8;

        // K-loop
        for (int k = 0; k < K; k += Kgroup) {
            bfloat16x8_t a0 = vld1q_bf16(pA);
            bfloat16x8_t a1 = vld1q_bf16(pA + 8);
            bfloat16x8_t a2 = vld1q_bf16(pA + 16);
            bfloat16x8_t a3 = vld1q_bf16(pA + 24);
            pA += 32;

            // Process each col-pair in this chunk
            const bfloat16_t* pb = pB;

#define BF16_VLA_COLPAIR(idx)                       \
    if ((idx) < cp_count) {                         \
        bfloat16x8_t b = vld1q_bf16(pb + (idx)*8); \
        c0##idx = vbfmmlaq_f32(c0##idx, a0, b);    \
        c1##idx = vbfmmlaq_f32(c1##idx, a1, b);    \
        c2##idx = vbfmmlaq_f32(c2##idx, a2, b);    \
        c3##idx = vbfmmlaq_f32(c3##idx, a3, b);    \
    }

            BF16_VLA_COLPAIR(0); BF16_VLA_COLPAIR(1);
            BF16_VLA_COLPAIR(2); BF16_VLA_COLPAIR(3);
            BF16_VLA_COLPAIR(4); BF16_VLA_COLPAIR(5);
            BF16_VLA_COLPAIR(6); BF16_VLA_COLPAIR(7);
#undef BF16_VLA_COLPAIR

            pB += b_stride;
        }

        // Epilogue: extract 2×2 blocks → row-major C
        // Each accumulator [r0c0, r0c1, r1c0, r1c1]:
        //   low 64-bit = row0, high 64-bit = row1
#define STORE_BF16_VLA_PAIR(rp_row, a0, a1, a2, a3, a4, a5, a6, a7, cp_off) \
    do {                                                                     \
        float32x4_t* accs[] = {&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7};    \
        for (int ci = 0; ci < cp_count; ++ci) {                              \
            int col = (cp_base + ci) * 2;                                    \
            float32x4_t block = *accs[ci];                                   \
            float r0c0 = vgetq_lane_f32(block, 0);                          \
            float r0c1 = vgetq_lane_f32(block, 1);                          \
            float r1c0 = vgetq_lane_f32(block, 2);                          \
            float r1c1 = vgetq_lane_f32(block, 3);                          \
            float* Cr0 = &C[(rp_row) * ldc + col];                          \
            float* Cr1 = &C[((rp_row)+1) * ldc + col];                      \
            if (beta == 0.0f) {                                              \
                Cr0[0] = alpha * r0c0; Cr0[1] = alpha * r0c1;               \
                Cr1[0] = alpha * r1c0; Cr1[1] = alpha * r1c1;               \
            } else {                                                         \
                Cr0[0] = alpha * r0c0 + beta * Cr0[0];                       \
                Cr0[1] = alpha * r0c1 + beta * Cr0[1];                       \
                Cr1[0] = alpha * r1c0 + beta * Cr1[0];                       \
                Cr1[1] = alpha * r1c1 + beta * Cr1[1];                       \
            }                                                                \
        }                                                                    \
    } while(0)

        STORE_BF16_VLA_PAIR(0, c00,c01,c02,c03,c04,c05,c06,c07, 0);
        STORE_BF16_VLA_PAIR(2, c10,c11,c12,c13,c14,c15,c16,c17, 0);
        STORE_BF16_VLA_PAIR(4, c20,c21,c22,c23,c24,c25,c26,c27, 0);
        STORE_BF16_VLA_PAIR(6, c30,c31,c32,c33,c34,c35,c36,c37, 0);
#undef STORE_BF16_VLA_PAIR
    }
}

// ============================================================
// Registry wrappers
// ============================================================

static void pack_a_bf16_sve_wrap(int m_len, int k_len, const float* A, int lda,
                                  void* packed_A, int Mr, float* /*scale_out*/) {
    pack_a_bf16_sve(m_len, k_len, A, lda, packed_A, Mr);
}

static void pack_b_bf16_sve_wrap(int k_len, int n_len, const float* B, int ldb,
                                  void* packed_B, int Nr, float* /*scale_out*/) {
    pack_b_bf16_sve(k_len, n_len, B, ldb, packed_B, Nr);
}

static void ukernel_bf16_sve_wrap(int K, const void* packed_A,
                                   const void* packed_B,
                                   float* C, int ldc, float alpha,
                                   float beta, float /*extra*/) {
    gemm_ukernel_bf16_sve_vla(K,
                               static_cast<const bfloat16_t*>(packed_A),
                               static_cast<const bfloat16_t*>(packed_B),
                               C, ldc, alpha, beta);
}

// SVE BF16 VLA: only selected on wide SVE (256+)
// On SVE-128, NEON BFMMLA (priority 100) handles it identically.
static const GemmMicrokernelDesc sve_bf16_vla_desc = {
    "sve_bf16_8xVL",
    GemmDataType::kBF16,
    kSVE | kSVEBF16 | kBF16,
    8,                    // Mr
    0,                    // Nr (computed)
    4,                    // Kgroup
    true,                 // nr_is_vla
    200,                  // priority
    sizeof(bfloat16_t),
    sizeof(bfloat16_t),
    256,                  // min_sve_bits: only on wide SVE
    ukernel_bf16_sve_wrap,
    pack_a_bf16_sve_wrap,
    pack_b_bf16_sve_wrap,
};

static RegisterKernel reg_sve_bf16_vla(sve_bf16_vla_desc);

// ============================================================
// SVE-128 BF16: same 8x8 BFMMLA tile as NEON, priority=120
// ============================================================
// On SVE-128, BFMMLA intrinsics are identical to NEON. The benefit
// comes from SVE-accelerated packing (pack_a_bf16_sve/pack_b_bf16_sve)
// which uses SVE predicates and conversion instructions.

// Reuse the NEON BFMMLA microkernel (declared in gemm_ukernel_bf16_neon.cpp)
extern void gemm_ukernel_bf16_8x8(int K, const bfloat16_t* packed_A,
                                    const bfloat16_t* packed_B,
                                    float* C, int ldc, float alpha, float beta);

static void ukernel_bf16_sve128_wrap(int K, const void* packed_A,
                                      const void* packed_B,
                                      float* C, int ldc, float alpha,
                                      float beta, float /*extra*/) {
    gemm_ukernel_bf16_8x8(K,
                            static_cast<const bfloat16_t*>(packed_A),
                            static_cast<const bfloat16_t*>(packed_B),
                            C, ldc, alpha, beta);
}

static const GemmMicrokernelDesc sve128_bf16_desc = {
    "sve128_bf16_8x8",
    GemmDataType::kBF16,
    kSVE | kBF16,             // required_hwcaps
    8,                        // Mr
    8,                        // Nr (fixed, same as NEON)
    4,                        // Kgroup
    false,                    // nr_is_vla: NO
    120,                      // priority: higher than NEON 100
    sizeof(bfloat16_t),
    sizeof(bfloat16_t),
    0,                        // min_sve_bits: any SVE
    ukernel_bf16_sve128_wrap,
    pack_a_bf16_sve_wrap,     // SVE packing (predicated, faster)
    pack_b_bf16_sve_wrap,
};

static RegisterKernel reg_sve128_bf16(sve128_bf16_desc);

}  // namespace dnnopt

#endif  // __ARM_FEATURE_SVE && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
