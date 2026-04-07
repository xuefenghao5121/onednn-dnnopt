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

/// VLA microkernel: works for any SVE vector length.
/// Uses Mr=8 rows and Nr=2*VL columns (2 SVE registers per row).
/// 16 accumulator registers: 8 rows × 2 B-vectors.
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

    // K-loop: 2x unroll for better instruction scheduling
    int k = 0;
    for (; k + 1 < K; k += 2) {
        // Iteration 0
        svfloat32_t b0_0 = svld1_f32(pg, packed_B + k * Nr);
        svfloat32_t b0_1 = svld1_f32(pg, packed_B + k * Nr + vl);

        acc0_0 = svmla_f32_x(pg, acc0_0, b0_0, svdup_f32(packed_A[k * 8 + 0]));
        acc0_1 = svmla_f32_x(pg, acc0_1, b0_1, svdup_f32(packed_A[k * 8 + 0]));
        acc1_0 = svmla_f32_x(pg, acc1_0, b0_0, svdup_f32(packed_A[k * 8 + 1]));
        acc1_1 = svmla_f32_x(pg, acc1_1, b0_1, svdup_f32(packed_A[k * 8 + 1]));
        acc2_0 = svmla_f32_x(pg, acc2_0, b0_0, svdup_f32(packed_A[k * 8 + 2]));
        acc2_1 = svmla_f32_x(pg, acc2_1, b0_1, svdup_f32(packed_A[k * 8 + 2]));
        acc3_0 = svmla_f32_x(pg, acc3_0, b0_0, svdup_f32(packed_A[k * 8 + 3]));
        acc3_1 = svmla_f32_x(pg, acc3_1, b0_1, svdup_f32(packed_A[k * 8 + 3]));
        acc4_0 = svmla_f32_x(pg, acc4_0, b0_0, svdup_f32(packed_A[k * 8 + 4]));
        acc4_1 = svmla_f32_x(pg, acc4_1, b0_1, svdup_f32(packed_A[k * 8 + 4]));
        acc5_0 = svmla_f32_x(pg, acc5_0, b0_0, svdup_f32(packed_A[k * 8 + 5]));
        acc5_1 = svmla_f32_x(pg, acc5_1, b0_1, svdup_f32(packed_A[k * 8 + 5]));
        acc6_0 = svmla_f32_x(pg, acc6_0, b0_0, svdup_f32(packed_A[k * 8 + 6]));
        acc6_1 = svmla_f32_x(pg, acc6_1, b0_1, svdup_f32(packed_A[k * 8 + 6]));
        acc7_0 = svmla_f32_x(pg, acc7_0, b0_0, svdup_f32(packed_A[k * 8 + 7]));
        acc7_1 = svmla_f32_x(pg, acc7_1, b0_1, svdup_f32(packed_A[k * 8 + 7]));

        // Iteration 1
        svfloat32_t b1_0 = svld1_f32(pg, packed_B + (k + 1) * Nr);
        svfloat32_t b1_1 = svld1_f32(pg, packed_B + (k + 1) * Nr + vl);

        acc0_0 = svmla_f32_x(pg, acc0_0, b1_0, svdup_f32(packed_A[(k + 1) * 8 + 0]));
        acc0_1 = svmla_f32_x(pg, acc0_1, b1_1, svdup_f32(packed_A[(k + 1) * 8 + 0]));
        acc1_0 = svmla_f32_x(pg, acc1_0, b1_0, svdup_f32(packed_A[(k + 1) * 8 + 1]));
        acc1_1 = svmla_f32_x(pg, acc1_1, b1_1, svdup_f32(packed_A[(k + 1) * 8 + 1]));
        acc2_0 = svmla_f32_x(pg, acc2_0, b1_0, svdup_f32(packed_A[(k + 1) * 8 + 2]));
        acc2_1 = svmla_f32_x(pg, acc2_1, b1_1, svdup_f32(packed_A[(k + 1) * 8 + 2]));
        acc3_0 = svmla_f32_x(pg, acc3_0, b1_0, svdup_f32(packed_A[(k + 1) * 8 + 3]));
        acc3_1 = svmla_f32_x(pg, acc3_1, b1_1, svdup_f32(packed_A[(k + 1) * 8 + 3]));
        acc4_0 = svmla_f32_x(pg, acc4_0, b1_0, svdup_f32(packed_A[(k + 1) * 8 + 4]));
        acc4_1 = svmla_f32_x(pg, acc4_1, b1_1, svdup_f32(packed_A[(k + 1) * 8 + 4]));
        acc5_0 = svmla_f32_x(pg, acc5_0, b1_0, svdup_f32(packed_A[(k + 1) * 8 + 5]));
        acc5_1 = svmla_f32_x(pg, acc5_1, b1_1, svdup_f32(packed_A[(k + 1) * 8 + 5]));
        acc6_0 = svmla_f32_x(pg, acc6_0, b1_0, svdup_f32(packed_A[(k + 1) * 8 + 6]));
        acc6_1 = svmla_f32_x(pg, acc6_1, b1_1, svdup_f32(packed_A[(k + 1) * 8 + 6]));
        acc7_0 = svmla_f32_x(pg, acc7_0, b1_0, svdup_f32(packed_A[(k + 1) * 8 + 7]));
        acc7_1 = svmla_f32_x(pg, acc7_1, b1_1, svdup_f32(packed_A[(k + 1) * 8 + 7]));
    }
    // Handle odd K
    for (; k < K; ++k) {
        svfloat32_t b0 = svld1_f32(pg, packed_B + k * Nr);
        svfloat32_t b1 = svld1_f32(pg, packed_B + k * Nr + vl);

        acc0_0 = svmla_f32_x(pg, acc0_0, b0, svdup_f32(packed_A[k * 8 + 0]));
        acc0_1 = svmla_f32_x(pg, acc0_1, b1, svdup_f32(packed_A[k * 8 + 0]));
        acc1_0 = svmla_f32_x(pg, acc1_0, b0, svdup_f32(packed_A[k * 8 + 1]));
        acc1_1 = svmla_f32_x(pg, acc1_1, b1, svdup_f32(packed_A[k * 8 + 1]));
        acc2_0 = svmla_f32_x(pg, acc2_0, b0, svdup_f32(packed_A[k * 8 + 2]));
        acc2_1 = svmla_f32_x(pg, acc2_1, b1, svdup_f32(packed_A[k * 8 + 2]));
        acc3_0 = svmla_f32_x(pg, acc3_0, b0, svdup_f32(packed_A[k * 8 + 3]));
        acc3_1 = svmla_f32_x(pg, acc3_1, b1, svdup_f32(packed_A[k * 8 + 3]));
        acc4_0 = svmla_f32_x(pg, acc4_0, b0, svdup_f32(packed_A[k * 8 + 4]));
        acc4_1 = svmla_f32_x(pg, acc4_1, b1, svdup_f32(packed_A[k * 8 + 4]));
        acc5_0 = svmla_f32_x(pg, acc5_0, b0, svdup_f32(packed_A[k * 8 + 5]));
        acc5_1 = svmla_f32_x(pg, acc5_1, b1, svdup_f32(packed_A[k * 8 + 5]));
        acc6_0 = svmla_f32_x(pg, acc6_0, b0, svdup_f32(packed_A[k * 8 + 6]));
        acc6_1 = svmla_f32_x(pg, acc6_1, b1, svdup_f32(packed_A[k * 8 + 6]));
        acc7_0 = svmla_f32_x(pg, acc7_0, b0, svdup_f32(packed_A[k * 8 + 7]));
        acc7_1 = svmla_f32_x(pg, acc7_1, b1, svdup_f32(packed_A[k * 8 + 7]));
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
// Registry wrapper
// ============================================================

static void ukernel_fp32_sve_vla_wrap(int K, const void* packed_A,
                                       const void* packed_B,
                                       float* C, int ldc, float alpha,
                                       float beta, float /*extra*/) {
    gemm_ukernel_fp32_sve_vla(K,
                               static_cast<const float*>(packed_A),
                               static_cast<const float*>(packed_B),
                               C, ldc, alpha, beta);
}

// SVE FP32 VLA: Mr=8, Nr=2*VL (VLA)
// On SVE-128: Nr=8; on SVE-256: Nr=16; on SVE-512: Nr=32
// Priority 200 only when SVE VL > 128 (for wide SVE);
// on SVE-128, NEON 8x12 is better due to larger Nr.
static const GemmMicrokernelDesc sve_fp32_vla_desc = {
    "sve_fp32_8xVL",
    GemmDataType::kFP32,
    kSVE,                 // required_hwcaps
    8,                    // Mr
    0,                    // Nr (computed from VL)
    1,                    // Kgroup
    true,                 // nr_is_vla
    200,                  // priority (higher than NEON 100)
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
