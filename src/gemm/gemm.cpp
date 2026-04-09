/// @file gemm.cpp
/// Top-level GEMM dispatch with registry-based adaptive kernel selection.
/// Falls back to legacy drivers when no registry kernel matches.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"
#include "dnnopt/gemm/gemm_driver_generic.h"
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/cpu_tuning_profile.h"

namespace dnnopt {

// Legacy drivers (preserved for fallback)
void gemm_driver_fp32(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc);
void gemm_smallm_driver_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc);
void gemm_driver_bf16(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc);
void gemm_driver_int8(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc);

// Tiny GEMM kernels (defined in gemm_tiny_fp32.cpp)
bool gemm_tiny_dispatch_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc);

namespace {

/// Naive FP32 GEMM: C = alpha * A * B + beta * C
void gemm_naive_fp32(int M, int N, int K,
                     float alpha, const float* A, int lda,
                     const float* B, int ldb,
                     float beta, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
        }
    }
}

/// Dispatch via registry + generic driver.
/// Returns true if handled, false if caller should use legacy path.
bool dispatch_via_registry(GemmDataType dtype,
                           int M, int N, int K,
                           float alpha, const float* A, int lda,
                           const float* B, int ldb,
                           float beta, float* C, int ldc) {
    const auto& hw = detect_arm_hwcaps();
    const auto& profile = get_autotuned_profile();
    const auto* desc = GemmUkernelRegistry::instance().select(dtype, hw);
    if (!desc) return false;

    int Nr = desc->nr_is_vla ? desc->compute_nr(hw.sve_vector_bits) : desc->Nr;
    int Mr = desc->Mr;

    // Small-M: fall back to FP32 small-M driver (no packing overhead justified)
    if (M < Mr) return false;

    auto bp = compute_blocking_params(hw, profile, Mr, Nr, desc->Kgroup,
                                      desc->packed_a_elem_bytes,
                                      desc->packed_b_elem_bytes,
                                      M, N, K);

    GemmDriverConfig cfg;
    cfg.Mr = Mr;
    cfg.Nr = Nr;
    cfg.Kgroup = desc->Kgroup;
    cfg.Mc = bp.Mc;
    cfg.Nc = bp.Nc;
    cfg.Kc = bp.Kc;
    cfg.packed_a_elem_bytes = desc->packed_a_elem_bytes;
    cfg.packed_b_elem_bytes = desc->packed_b_elem_bytes;
    cfg.dtype = dtype;
    cfg.ukernel = desc->ukernel;
    cfg.pack_a = desc->pack_a;
    cfg.pack_b = desc->pack_b;

    // Threading config from tuning profile
    cfg.threading_min_flops = profile.threading_min_flops;
    cfg.prefer_2d_threading = profile.prefer_2d_threading;
    cfg.shape = classify_shape(M, N, K);

    gemm_driver_generic(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg);
    return true;
}

}  // namespace

// ============================================================
// Public API
// ============================================================

void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, GemmAlgo::kAuto);
}

void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc,
               GemmAlgo algo) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    if (algo == GemmAlgo::kNaive) {
        gemm_naive_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // Auto: try registry dispatch first
    if (algo == GemmAlgo::kAuto) {
        // Tiny shapes: N=1, M=1, or M,N ≤ 4 get specialized kernels
#ifdef __ARM_NEON
        if (gemm_tiny_dispatch_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)) {
            return;
        }
#endif

        // Small-M uses dedicated fast path (no packing)
        if (M < kGemmMrFp32) {
#ifdef __ARM_NEON
            gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
#endif
        }
        if (dispatch_via_registry(GemmDataType::kFP32, M, N, K,
                                  alpha, A, lda, B, ldb, beta, C, ldc))
            return;
    }

    // Explicit NEON or fallback from registry
#ifdef __ARM_NEON
    if (algo == GemmAlgo::kNeonFp32 || algo == GemmAlgo::kAuto) {
        if (M < kGemmMrFp32)
            gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            gemm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif

    gemm_naive_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_bf16(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    // Small-M: memory-bound, FP32 small-M is better
    if (M <= kGemmMrBf16 / 2) {
#ifdef __ARM_NEON
        gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
#endif
    }

    // Try registry dispatch
    if (dispatch_via_registry(GemmDataType::kBF16, M, N, K,
                              alpha, A, lda, B, ldb, beta, C, ldc))
        return;

    // Legacy fallback
#ifdef __ARM_NEON
    const auto& hw = detect_arm_hwcaps();
    if (hw.hwcaps & static_cast<uint64_t>(HwCap::kBF16)) {
        gemm_driver_bf16(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_int8(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    // Small-M: quantization overhead not worth it
    if (M <= kGemmMrInt8 / 2) {
#ifdef __ARM_NEON
        gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
#endif
    }

    // Try registry dispatch
    if (dispatch_via_registry(GemmDataType::kINT8, M, N, K,
                              alpha, A, lda, B, ldb, beta, C, ldc))
        return;

    // Legacy fallback
#ifdef __ARM_NEON
    const auto& hw = detect_arm_hwcaps();
    if (hw.hwcaps & static_cast<uint64_t>(HwCap::kI8MM)) {
        gemm_driver_int8(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace dnnopt
