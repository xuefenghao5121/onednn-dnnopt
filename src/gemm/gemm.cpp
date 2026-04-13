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
void gemm_smallm_wide_driver_fp32(int M, int N, int K,
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

    // Try all matching kernels in priority order until one fits M.
    auto candidates = GemmUkernelRegistry::instance().select_all(dtype, hw);
    for (const auto* desc : candidates) {
        int Nr = desc->nr_is_vla ? desc->compute_nr(hw.sve_vector_bits) : desc->Nr;
        int Mr = desc->Mr;

        // Small-M: try next smaller kernel
        if (M < Mr) continue;

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
    return false;
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
        // Phase B: M=2-7 with large N use wide driver (48-col macro-tiling + Kc blocking).
        // M=1 uses dedicated GEMV path. M=4-7 with small N falls through to adaptive tile.
        // Phase 13: When N*K is very large, prefer packed path with threading instead
        // of small-M path — packing overhead is amortized and threading is beneficial.
        if (M < 8) {
#ifdef __ARM_NEON
            if (M == 1) {
                gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            }
            // Phase 13: For M=4-7 with very large N*K, use packed registry path.
            // This enables 2D threading + huge pages, which outperforms small-M
            // wide driver for shapes like batch4-LLM (4×4096×4096).
            // Threshold: N*K > 4M (e.g., N=4096, K>=1024 or N>=1024, K=4096)
            // Only when M matches a registry kernel Mr (4 or 8) to avoid tail waste.
            // M=5,6,7 use adaptive tile kernels (5x16, 6x16, 7x16) instead.
            constexpr int64_t kLargeNKThreshold = 4 * 1024 * 1024;
            if (M == 4 && (int64_t)N * K > kLargeNKThreshold) {
                // Fall through to registry dispatch (packed + threaded)
                goto registry_dispatch;
            }
            // M=2-7: use wide driver for N >= 48 (macro-tiling benefit).
            // M=2-3: always use wide driver (was original routing).
            // M=4-7: only for N >= 48 where 48-col panels amortize B loads.
            // For tiny N, fall through to adaptive tile (asm kernels better).
            if (M >= 2 && (M < 4 || N >= 48)) {
                gemm_smallm_wide_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            }
#endif
        }

#ifdef __ARM_NEON
        // Phase 7D: small-K fast path (K ≤ 16): preloads B, shares across rows
        if (K <= 16) {
            gemm_smallK_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }

        // Phase 8/10/11: adaptive tile GEMM (autoGEMM-style, unpacked + Kc blocking)
        // Use for: (a) small vol shapes (packing overhead > compute), OR
        // (b) M=4-7 — asm kernels outperform packed 8x12 which zero-pads M to 8.
        if (M >= 4 && (
            (int64_t)M * N * K < kUnpackedFlopsThreshold ||
            M < 8
        )) {
            gemm_adaptive_tile_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
#endif

        registry_dispatch:
        if (dispatch_via_registry(GemmDataType::kFP32, M, N, K,
                                  alpha, A, lda, B, ldb, beta, C, ldc))
            return;
    }

    // Explicit NEON or fallback from registry
#ifdef __ARM_NEON
    if (algo == GemmAlgo::kNeonFp32 || algo == GemmAlgo::kAuto) {
        if (M == 1) {
            gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (M >= 2 && M < 8 && (M < 4 || N >= 48)) {
            gemm_smallm_wide_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (K <= 16) {
            gemm_smallK_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (M >= 4 && (
                   (int64_t)M * N * K < kUnpackedFlopsThreshold ||
                   M < 8)) {
            gemm_adaptive_tile_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else {
            gemm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
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
