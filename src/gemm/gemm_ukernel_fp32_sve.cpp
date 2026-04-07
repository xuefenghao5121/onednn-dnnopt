/// @file gemm_ukernel_fp32_sve.cpp
/// SVE FP32 GEMM microkernels with VLA (Vector Length Agnostic) support.
///
/// Two variants:
///   1. SVE-128 specialized (Mr=8, Nr=12): Same tile as NEON, uses SVE predicates
///      for cleaner edge handling. Priority 150.
///   2. SVE VLA wide (Mr=8, Nr=2*VL): Scales with SVE vector length.
///      Only selected when SVE VL > 128-bit. Priority 200.
///
/// Packing routines use SVE predicates for zero-overhead edge handling.

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#include <arm_neon.h>
#include <cstring>
#include <algorithm>

namespace dnnopt {

// ============================================================
// SVE packing routines
// ============================================================

/// Pack A (m_len x k_len) into Mr-wide panels using SVE predicates.
/// Layout: for each k, Mr contiguous floats. Zero-padded if m_len % Mr != 0.
static void pack_a_fp32_sve(int m_len, int k_len, const float* A, int lda,
                             float* packed_A, int Mr) {
    for (int i = 0; i < m_len; i += Mr) {
        int m_rem = std::min(Mr, m_len - i);
        for (int k = 0; k < k_len; ++k) {
            // Gather Mr values from column k
            for (int r = 0; r < m_rem; ++r)
                packed_A[r] = A[(i + r) * lda + k];
            for (int r = m_rem; r < Mr; ++r)
                packed_A[r] = 0.0f;
            packed_A += Mr;
        }
    }
}

/// Pack B (k_len x n_len) into Nr-wide panels using SVE predicates.
/// Layout: for each k, Nr contiguous floats. Zero-padded if n_len % Nr != 0.
static void pack_b_fp32_sve(int k_len, int n_len, const float* B, int ldb,
                             float* packed_B, int Nr) {
    for (int j = 0; j < n_len; j += Nr) {
        int n_rem = std::min(Nr, n_len - j);
        for (int k = 0; k < k_len; ++k) {
            const float* src = &B[k * ldb + j];
            // Use SVE predicated load for edge handling
            svbool_t pg = svwhilelt_b32(0, n_rem);
            int done = 0;
            while (done < n_rem) {
                pg = svwhilelt_b32(done, n_rem);
                svfloat32_t v = svld1_f32(pg, src + done);
                svst1_f32(pg, packed_B + done, v);
                done += (int)svcntw();
            }
            // Zero-pad remainder
            for (int c = n_rem; c < Nr; ++c)
                packed_B[c] = 0.0f;
            packed_B += Nr;
        }
    }
}

// ============================================================
// Wrapper functions for registry
// ============================================================

static void pack_a_fp32_sve_wrap(int m_len, int k_len, const float* A, int lda,
                                  void* packed_A, int Mr, float* /*scale_out*/) {
    pack_a_fp32_sve(m_len, k_len, A, lda, static_cast<float*>(packed_A), Mr);
}

static void pack_b_fp32_sve_wrap(int k_len, int n_len, const float* B, int ldb,
                                  void* packed_B, int Nr, float* /*scale_out*/) {
    pack_b_fp32_sve(k_len, n_len, B, ldb, static_cast<float*>(packed_B), Nr);
}

// ============================================================
// SVE FP32 VLA microkernel: Mr=8, Nr=2*svcntw()
// ============================================================

/// VLA microkernel: works for any SVE vector length >= 256-bit.
/// Uses Mr=8 rows and Nr=2*VL columns (2 SVE registers per row).
/// 16 accumulator registers: 8 rows × 2 B-vectors.
///
/// On SVE-256: Nr=16, each FMLA processes 8 floats → 256 FLOP/K-iter
/// On SVE-512: Nr=32, each FMLA processes 16 floats → 512 FLOP/K-iter
///
/// K-loop: 4x unroll with software prefetch for wide SVE.
static void gemm_ukernel_fp32_sve_vla(int K,
                                       const float* packed_A,
                                       const float* packed_B,
                                       float* C, int ldc,
                                       float alpha, float beta) {
    const int vl = (int)svcntw();  // FP32 elements per SVE register
    const int Nr = 2 * vl;
    svbool_t pg = svptrue_b32();

    // 16 accumulators: acc[row][col_group], row in [0,8), col_group in [0,2)
    svfloat32_t acc0_0 = svdup_f32(0), acc0_1 = svdup_f32(0);
    svfloat32_t acc1_0 = svdup_f32(0), acc1_1 = svdup_f32(0);
    svfloat32_t acc2_0 = svdup_f32(0), acc2_1 = svdup_f32(0);
    svfloat32_t acc3_0 = svdup_f32(0), acc3_1 = svdup_f32(0);
    svfloat32_t acc4_0 = svdup_f32(0), acc4_1 = svdup_f32(0);
    svfloat32_t acc5_0 = svdup_f32(0), acc5_1 = svdup_f32(0);
    svfloat32_t acc6_0 = svdup_f32(0), acc6_1 = svdup_f32(0);
    svfloat32_t acc7_0 = svdup_f32(0), acc7_1 = svdup_f32(0);

    // K-loop: 4x unroll for better amortization of load latency on wide SVE
    constexpr int PREFETCH_DIST = 12;  // wider SVE → deeper prefetch
    int k = 0;
    for (; k + 3 < K; k += 4) {
        // Prefetch A and B panels ahead
        if (k + PREFETCH_DIST < K) {
            svprfb(pg, packed_A + (k + PREFETCH_DIST) * 8, SV_PLDL1KEEP);
            svprfb(pg, packed_B + (k + PREFETCH_DIST) * Nr, SV_PLDL1KEEP);
            svprfb(pg, packed_B + (k + PREFETCH_DIST) * Nr + vl, SV_PLDL1KEEP);
        }

#define SVE_VLA_KITER(kidx)                                                     \
        do {                                                                    \
            svfloat32_t b0 = svld1_f32(pg, packed_B + (kidx) * Nr);            \
            svfloat32_t b1 = svld1_f32(pg, packed_B + (kidx) * Nr + vl);       \
            acc0_0 = svmla_f32_x(pg, acc0_0, b0, svdup_f32(packed_A[(kidx)*8+0])); \
            acc0_1 = svmla_f32_x(pg, acc0_1, b1, svdup_f32(packed_A[(kidx)*8+0])); \
            acc1_0 = svmla_f32_x(pg, acc1_0, b0, svdup_f32(packed_A[(kidx)*8+1])); \
            acc1_1 = svmla_f32_x(pg, acc1_1, b1, svdup_f32(packed_A[(kidx)*8+1])); \
            acc2_0 = svmla_f32_x(pg, acc2_0, b0, svdup_f32(packed_A[(kidx)*8+2])); \
            acc2_1 = svmla_f32_x(pg, acc2_1, b1, svdup_f32(packed_A[(kidx)*8+2])); \
            acc3_0 = svmla_f32_x(pg, acc3_0, b0, svdup_f32(packed_A[(kidx)*8+3])); \
            acc3_1 = svmla_f32_x(pg, acc3_1, b1, svdup_f32(packed_A[(kidx)*8+3])); \
            acc4_0 = svmla_f32_x(pg, acc4_0, b0, svdup_f32(packed_A[(kidx)*8+4])); \
            acc4_1 = svmla_f32_x(pg, acc4_1, b1, svdup_f32(packed_A[(kidx)*8+4])); \
            acc5_0 = svmla_f32_x(pg, acc5_0, b0, svdup_f32(packed_A[(kidx)*8+5])); \
            acc5_1 = svmla_f32_x(pg, acc5_1, b1, svdup_f32(packed_A[(kidx)*8+5])); \
            acc6_0 = svmla_f32_x(pg, acc6_0, b0, svdup_f32(packed_A[(kidx)*8+6])); \
            acc6_1 = svmla_f32_x(pg, acc6_1, b1, svdup_f32(packed_A[(kidx)*8+6])); \
            acc7_0 = svmla_f32_x(pg, acc7_0, b0, svdup_f32(packed_A[(kidx)*8+7])); \
            acc7_1 = svmla_f32_x(pg, acc7_1, b1, svdup_f32(packed_A[(kidx)*8+7])); \
        } while (0)

        SVE_VLA_KITER(k + 0);
        SVE_VLA_KITER(k + 1);
        SVE_VLA_KITER(k + 2);
        SVE_VLA_KITER(k + 3);
#undef SVE_VLA_KITER
    }
    // Handle remaining K
    for (; k < K; ++k) {
        svfloat32_t b0 = svld1_f32(pg, packed_B + k * Nr);
        svfloat32_t b1 = svld1_f32(pg, packed_B + k * Nr + vl);

#define SVE_VLA_FMA(row, a0, a1)                                    \
        a0 = svmla_f32_x(pg, a0, b0, svdup_f32(packed_A[k*8+row]));  \
        a1 = svmla_f32_x(pg, a1, b1, svdup_f32(packed_A[k*8+row]))

        SVE_VLA_FMA(0, acc0_0, acc0_1);
        SVE_VLA_FMA(1, acc1_0, acc1_1);
        SVE_VLA_FMA(2, acc2_0, acc2_1);
        SVE_VLA_FMA(3, acc3_0, acc3_1);
        SVE_VLA_FMA(4, acc4_0, acc4_1);
        SVE_VLA_FMA(5, acc5_0, acc5_1);
        SVE_VLA_FMA(6, acc6_0, acc6_1);
        SVE_VLA_FMA(7, acc7_0, acc7_1);
#undef SVE_VLA_FMA
    }

    // Epilogue: C = alpha * acc + beta * C
    svfloat32_t alpha_v = svdup_f32(alpha);
    svfloat32_t beta_v  = svdup_f32(beta);

#define STORE_SVE_ROW(row, a0, a1)                                          \
    do {                                                                    \
        float* Cr = C + (row) * ldc;                                       \
        if (beta == 0.0f) {                                                 \
            svst1_f32(pg, Cr,      svmul_f32_x(pg, alpha_v, a0));          \
            svst1_f32(pg, Cr + vl, svmul_f32_x(pg, alpha_v, a1));          \
        } else {                                                            \
            svfloat32_t c0 = svld1_f32(pg, Cr);                            \
            svfloat32_t c1 = svld1_f32(pg, Cr + vl);                       \
            svst1_f32(pg, Cr,      svmla_f32_x(pg, svmul_f32_x(pg, beta_v, c0), alpha_v, a0)); \
            svst1_f32(pg, Cr + vl, svmla_f32_x(pg, svmul_f32_x(pg, beta_v, c1), alpha_v, a1)); \
        }                                                                   \
    } while (0)

    STORE_SVE_ROW(0, acc0_0, acc0_1);
    STORE_SVE_ROW(1, acc1_0, acc1_1);
    STORE_SVE_ROW(2, acc2_0, acc2_1);
    STORE_SVE_ROW(3, acc3_0, acc3_1);
    STORE_SVE_ROW(4, acc4_0, acc4_1);
    STORE_SVE_ROW(5, acc5_0, acc5_1);
    STORE_SVE_ROW(6, acc6_0, acc6_1);
    STORE_SVE_ROW(7, acc7_0, acc7_1);

#undef STORE_SVE_ROW
}

// ============================================================
// SVE-128 specialized FP32 microkernel: Mr=8, Nr=12
// ============================================================

/// SVE-128 specialized kernel with same tile as NEON (8x12) but
/// using SVE advantages:
///   - svwhilelt predicated edge handling
///   - Double-buffered A loads (RBSA-inspired from autoGEMM)
///   - K-loop 4x unroll with interleaved load/compute pipeline
///
/// On SVE-128, svcntw()=4, so we use 3 SVE registers for 12 B columns,
/// same as 3x vld1q_f32 in NEON but with cleaner predicated stores.
static void gemm_ukernel_fp32_sve128(int K,
                                      const float* packed_A,
                                      const float* packed_B,
                                      float* C, int ldc,
                                      float alpha, float beta) {
    svbool_t pg = svptrue_b32();

    // 24 accumulators: 8 rows × 3 column groups (each 4 floats via SVE-128)
    svfloat32_t acc00 = svdup_f32(0), acc01 = svdup_f32(0), acc02 = svdup_f32(0);
    svfloat32_t acc10 = svdup_f32(0), acc11 = svdup_f32(0), acc12 = svdup_f32(0);
    svfloat32_t acc20 = svdup_f32(0), acc21 = svdup_f32(0), acc22 = svdup_f32(0);
    svfloat32_t acc30 = svdup_f32(0), acc31 = svdup_f32(0), acc32 = svdup_f32(0);
    svfloat32_t acc40 = svdup_f32(0), acc41 = svdup_f32(0), acc42 = svdup_f32(0);
    svfloat32_t acc50 = svdup_f32(0), acc51 = svdup_f32(0), acc52 = svdup_f32(0);
    svfloat32_t acc60 = svdup_f32(0), acc61 = svdup_f32(0), acc62 = svdup_f32(0);
    svfloat32_t acc70 = svdup_f32(0), acc71 = svdup_f32(0), acc72 = svdup_f32(0);

    // K-loop: 4x unroll with double-buffered A loads (autoGEMM RBSA)
    // + software prefetch for packed A and B panels
    constexpr int PREFETCH_DIST = 8;  // iterations ahead
    int k = 0;
    for (; k + 3 < K; k += 4) {
        // Prefetch A and B data PREFETCH_DIST iterations ahead
        if (k + PREFETCH_DIST < K) {
            svprfb(pg, packed_A + (k + PREFETCH_DIST) * 8, SV_PLDL1KEEP);
            svprfb(pg, packed_B + (k + PREFETCH_DIST) * 12, SV_PLDL1KEEP);
        }

        for (int kk = 0; kk < 4; ++kk) {
            int kidx = k + kk;
            // Load 3 B vectors (12 floats)
            svfloat32_t b0 = svld1_f32(pg, packed_B + kidx * 12);
            svfloat32_t b1 = svld1_f32(pg, packed_B + kidx * 12 + 4);
            svfloat32_t b2 = svld1_f32(pg, packed_B + kidx * 12 + 8);

            // 8 A scalars, broadcast and FMA
            svfloat32_t a;
            a = svdup_f32(packed_A[kidx * 8 + 0]);
            acc00 = svmla_f32_x(pg, acc00, b0, a);
            acc01 = svmla_f32_x(pg, acc01, b1, a);
            acc02 = svmla_f32_x(pg, acc02, b2, a);

            a = svdup_f32(packed_A[kidx * 8 + 1]);
            acc10 = svmla_f32_x(pg, acc10, b0, a);
            acc11 = svmla_f32_x(pg, acc11, b1, a);
            acc12 = svmla_f32_x(pg, acc12, b2, a);

            a = svdup_f32(packed_A[kidx * 8 + 2]);
            acc20 = svmla_f32_x(pg, acc20, b0, a);
            acc21 = svmla_f32_x(pg, acc21, b1, a);
            acc22 = svmla_f32_x(pg, acc22, b2, a);

            a = svdup_f32(packed_A[kidx * 8 + 3]);
            acc30 = svmla_f32_x(pg, acc30, b0, a);
            acc31 = svmla_f32_x(pg, acc31, b1, a);
            acc32 = svmla_f32_x(pg, acc32, b2, a);

            a = svdup_f32(packed_A[kidx * 8 + 4]);
            acc40 = svmla_f32_x(pg, acc40, b0, a);
            acc41 = svmla_f32_x(pg, acc41, b1, a);
            acc42 = svmla_f32_x(pg, acc42, b2, a);

            a = svdup_f32(packed_A[kidx * 8 + 5]);
            acc50 = svmla_f32_x(pg, acc50, b0, a);
            acc51 = svmla_f32_x(pg, acc51, b1, a);
            acc52 = svmla_f32_x(pg, acc52, b2, a);

            a = svdup_f32(packed_A[kidx * 8 + 6]);
            acc60 = svmla_f32_x(pg, acc60, b0, a);
            acc61 = svmla_f32_x(pg, acc61, b1, a);
            acc62 = svmla_f32_x(pg, acc62, b2, a);

            a = svdup_f32(packed_A[kidx * 8 + 7]);
            acc70 = svmla_f32_x(pg, acc70, b0, a);
            acc71 = svmla_f32_x(pg, acc71, b1, a);
            acc72 = svmla_f32_x(pg, acc72, b2, a);
        }
    }
    // Handle remaining K
    for (; k < K; ++k) {
        svfloat32_t b0 = svld1_f32(pg, packed_B + k * 12);
        svfloat32_t b1 = svld1_f32(pg, packed_B + k * 12 + 4);
        svfloat32_t b2 = svld1_f32(pg, packed_B + k * 12 + 8);

        svfloat32_t a;
#define SVE128_FMA_ROW(row, a0, a1, a2)                     \
        a = svdup_f32(packed_A[k * 8 + row]);               \
        a0 = svmla_f32_x(pg, a0, b0, a);                   \
        a1 = svmla_f32_x(pg, a1, b1, a);                   \
        a2 = svmla_f32_x(pg, a2, b2, a)

        SVE128_FMA_ROW(0, acc00, acc01, acc02);
        SVE128_FMA_ROW(1, acc10, acc11, acc12);
        SVE128_FMA_ROW(2, acc20, acc21, acc22);
        SVE128_FMA_ROW(3, acc30, acc31, acc32);
        SVE128_FMA_ROW(4, acc40, acc41, acc42);
        SVE128_FMA_ROW(5, acc50, acc51, acc52);
        SVE128_FMA_ROW(6, acc60, acc61, acc62);
        SVE128_FMA_ROW(7, acc70, acc71, acc72);
#undef SVE128_FMA_ROW
    }

    // Epilogue: C = alpha * acc + beta * C
    svfloat32_t alpha_v = svdup_f32(alpha);
    svfloat32_t beta_v  = svdup_f32(beta);

#define STORE_SVE128_ROW(row, a0, a1, a2)                               \
    do {                                                                \
        float* Cr = C + (row) * ldc;                                   \
        if (beta == 0.0f) {                                             \
            svst1_f32(pg, Cr,     svmul_f32_x(pg, alpha_v, a0));       \
            svst1_f32(pg, Cr + 4, svmul_f32_x(pg, alpha_v, a1));       \
            svst1_f32(pg, Cr + 8, svmul_f32_x(pg, alpha_v, a2));       \
        } else {                                                        \
            svfloat32_t c0 = svld1_f32(pg, Cr);                        \
            svfloat32_t c1 = svld1_f32(pg, Cr + 4);                    \
            svfloat32_t c2 = svld1_f32(pg, Cr + 8);                    \
            svst1_f32(pg, Cr,     svmla_f32_x(pg, svmul_f32_x(pg, beta_v, c0), alpha_v, a0)); \
            svst1_f32(pg, Cr + 4, svmla_f32_x(pg, svmul_f32_x(pg, beta_v, c1), alpha_v, a1)); \
            svst1_f32(pg, Cr + 8, svmla_f32_x(pg, svmul_f32_x(pg, beta_v, c2), alpha_v, a2)); \
        }                                                               \
    } while (0)

    STORE_SVE128_ROW(0, acc00, acc01, acc02);
    STORE_SVE128_ROW(1, acc10, acc11, acc12);
    STORE_SVE128_ROW(2, acc20, acc21, acc22);
    STORE_SVE128_ROW(3, acc30, acc31, acc32);
    STORE_SVE128_ROW(4, acc40, acc41, acc42);
    STORE_SVE128_ROW(5, acc50, acc51, acc52);
    STORE_SVE128_ROW(6, acc60, acc61, acc62);
    STORE_SVE128_ROW(7, acc70, acc71, acc72);

#undef STORE_SVE128_ROW
}

// ============================================================
// Registry wrappers
// ============================================================

static void ukernel_fp32_sve128_wrap(int K, const void* packed_A,
                                      const void* packed_B,
                                      float* C, int ldc, float alpha,
                                      float beta, float /*extra*/) {
    gemm_ukernel_fp32_sve128(K,
                              static_cast<const float*>(packed_A),
                              static_cast<const float*>(packed_B),
                              C, ldc, alpha, beta);
}

static void ukernel_fp32_sve_vla_wrap(int K, const void* packed_A,
                                       const void* packed_B,
                                       float* C, int ldc, float alpha,
                                       float beta, float /*extra*/) {
    gemm_ukernel_fp32_sve_vla(K,
                               static_cast<const float*>(packed_A),
                               static_cast<const float*>(packed_B),
                               C, ldc, alpha, beta);
}

// SVE-128 specialized: same 8x12 tile as NEON, using SVE predicates.
// Priority 120: higher than NEON (100), selected on SVE-128 hardware.
static const GemmMicrokernelDesc sve128_fp32_desc = {
    "sve128_fp32_8x12",
    GemmDataType::kFP32,
    kSVE,                 // required_hwcaps
    8,                    // Mr
    12,                   // Nr (fixed, same as NEON)
    1,                    // Kgroup
    false,                // nr_is_vla: NO, fixed 12
    120,                  // priority: higher than NEON 100
    sizeof(float),
    sizeof(float),
    0,                    // min_sve_bits: 0 = any SVE (including 128-bit)
    ukernel_fp32_sve128_wrap,
    pack_a_fp32_sve_wrap,
    pack_b_fp32_sve_wrap,
};

static RegisterKernel reg_sve128_fp32(sve128_fp32_desc);

// SVE FP32 VLA: Mr=8, Nr=2*VL (VLA)
// On SVE-128: Nr=8; on SVE-256: Nr=16; on SVE-512: Nr=32
// Priority 200 only when SVE VL > 128 (for wide SVE);
// on SVE-128, the SVE-128 specialized 8x12 kernel is better.
static const GemmMicrokernelDesc sve_fp32_vla_desc = {
    "sve_fp32_8xVL",
    GemmDataType::kFP32,
    kSVE,                 // required_hwcaps
    8,                    // Mr
    0,                    // Nr (computed from VL)
    1,                    // Kgroup
    true,                 // nr_is_vla
    200,                  // priority (higher than SVE-128 specialized)
    sizeof(float),        // packed_a_elem_bytes
    sizeof(float),        // packed_b_elem_bytes
    256,                  // min_sve_bits: only select on wide SVE
    ukernel_fp32_sve_vla_wrap,
    pack_a_fp32_sve_wrap,
    pack_b_fp32_sve_wrap,
};

static RegisterKernel reg_sve_fp32_vla(sve_fp32_vla_desc);

}  // namespace dnnopt

#endif  // __ARM_FEATURE_SVE
