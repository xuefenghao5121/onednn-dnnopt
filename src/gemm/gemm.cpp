/// @file gemm.cpp
/// Top-level GEMM dispatch and naive fallback.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/arm_hwcaps.h"

namespace dnnopt {

// BLIS driver (defined in gemm_driver_fp32.cpp)
void gemm_driver_fp32(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc);

// Small-M driver (defined in gemm_smallm_fp32.cpp)
void gemm_smallm_driver_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc);

// BF16 BLIS driver (defined in gemm_driver_bf16.cpp)
void gemm_driver_bf16(int M, int N, int K,
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

}  // namespace

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

    // Auto or explicit NEON
#ifdef __ARM_NEON
    if (algo == GemmAlgo::kAuto || algo == GemmAlgo::kNeonFp32) {
        if (M < kGemmMrFp32)
            gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            gemm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif

    // Fallback
    gemm_naive_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_bf16(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;

#ifdef __ARM_NEON
    const auto& hw = detect_arm_hwcaps();
    if (hw.hwcaps & static_cast<uint64_t>(HwCap::kBF16)) {
        // Small-M is memory-bound (arithmetic intensity < 1 FLOP/byte).
        // For M <= 4 (≤50% tile utilization), FP32 small-M avoids packing
        // overhead and is faster. For M=5-7, BFMMLA's compute density wins
        // despite partial tile utilization.
        if (M <= kGemmMrBf16 / 2)
            gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            gemm_driver_bf16(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif

    // Fallback to FP32 if BF16 not available
    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace dnnopt
