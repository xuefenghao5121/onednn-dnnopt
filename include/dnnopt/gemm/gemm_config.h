#pragma once
/// @file gemm_config.h
/// Platform-specific and adaptive cache blocking parameters for GEMM.
///
/// Blocking strategy inspired by autoGEMM (SC'24):
///   - Per-CPU tuning profiles replace hardcoded ratios
///   - Shape-aware adjustments for irregular matrices
///   - Dynamic bounds instead of fixed Mc/Nc limits

#include "dnnopt/gemm/gemm_types.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/cpu_tuning_profile.h"

#include <algorithm>

namespace dnnopt {

// FP32 NEON microkernel tile dimensions.
constexpr int kGemmMrFp32 = 8;
constexpr int kGemmNrFp32 = 12;

// Threshold for unpacked fast path: use when M*N*K < this and M <= 32
constexpr int64_t kUnpackedFlopsThreshold = 4 * 1024 * 1024;  // ~8M FLOPs

// BF16 BFMMLA microkernel tile dimensions.
constexpr int kGemmMrBf16 = 8;
constexpr int kGemmNrBf16 = 8;

// INT8 SMMLA microkernel tile dimensions.
constexpr int kGemmMrInt8 = 8;
constexpr int kGemmNrInt8 = 8;

/// Compute cache blocking parameters using CpuTuningProfile + ShapeClass.
///
/// Algorithm:
///   1. Classify the matrix shape (square, tall-skinny, short-wide, etc.)
///   2. Apply shape-specific multipliers to base cache utilization ratios
///   3. Compute Kc/Mc/Nc from adjusted ratios and actual cache sizes
///   4. Clamp to shape-adjusted bounds
///
/// This replaces the old hardcoded 40%/40%/30% approach with per-CPU,
/// per-shape adaptive parameters.
inline GemmBlockingParams compute_blocking_params(
    const ArmHwProfile& hw,
    const CpuTuningProfile& profile,
    int Mr, int Nr, int Kgroup,
    int packed_a_elem_bytes, int packed_b_elem_bytes,
    int M, int N, int K) {

    GemmBlockingParams p;
    p.Mr = Mr;
    p.Nr = Nr;

    // --- Get base cache sizes ---
    uint32_t l1d_bytes = hw.l1d.size_bytes;
    uint32_t l2_bytes  = hw.l2.size_bytes;
    uint32_t l3_bytes  = hw.l3.size_bytes;

    if (l1d_bytes == 0) l1d_bytes = 64 * 1024;
    if (l2_bytes  == 0) l2_bytes  = 1024 * 1024;

    // --- Classify shape and get adjustments ---
    ShapeClass sc = classify_shape(M, N, K);

    float l1d_util = profile.l1d_util;
    float l2_util  = profile.l2_util;
    float l3_util  = profile.l3_util;
    float mc_max_f = (float)profile.mc_max;
    float nc_max_f = (float)profile.nc_max;

    switch (sc) {
    case ShapeClass::kTallSkinny: {
        const auto& a = profile.shape_adj_tall_skinny;
        l1d_util *= a.l1d_mult;
        l2_util  *= a.l2_mult;
        l3_util  *= a.l3_mult;
        mc_max_f *= a.mc_mult;
        nc_max_f *= a.nc_mult;
        break;
    }
    case ShapeClass::kShortWide: {
        const auto& a = profile.shape_adj_short_wide;
        l1d_util *= a.l1d_mult;
        l2_util  *= a.l2_mult;
        l3_util  *= a.l3_mult;
        mc_max_f *= a.mc_mult;
        nc_max_f *= a.nc_mult;
        break;
    }
    case ShapeClass::kSmallGemm: {
        const auto& a = profile.shape_adj_small;
        l1d_util *= a.l1d_mult;
        l2_util  *= a.l2_mult;
        l3_util  *= a.l3_mult;
        mc_max_f *= a.mc_mult;
        nc_max_f *= a.nc_mult;
        break;
    }
    case ShapeClass::kBertLike: {
        const auto& a = profile.shape_adj_bert;
        l1d_util *= a.l1d_mult;
        l2_util  *= a.l2_mult;
        l3_util  *= a.l3_mult;
        mc_max_f *= a.mc_mult;
        nc_max_f *= a.nc_mult;
        break;
    }
    case ShapeClass::kSquare:
    default:
        break;
    }

    int mc_max = std::max((int)mc_max_f, Mr);
    int nc_max = std::max((int)nc_max_f, Nr);

    // --- Kc: fit one Mr-panel of A + one Nr-panel of B in L1D ---
    int bytes_per_k = Mr * packed_a_elem_bytes + Nr * packed_b_elem_bytes;
    if (bytes_per_k <= 0) bytes_per_k = 1;
    int Kc = (int)(l1d_bytes * l1d_util) / bytes_per_k;
    if (Kgroup > 1) Kc = (Kc / Kgroup) * Kgroup;
    Kc = std::max(Kc, Kgroup);
    Kc = std::min(Kc, K);

    // --- Mc: fit packed A (Mc x Kc) in L2 ---
    int a_panel_bytes_per_m = Kc * packed_a_elem_bytes;
    if (a_panel_bytes_per_m <= 0) a_panel_bytes_per_m = 1;
    int Mc = (int)(l2_bytes * l2_util) / a_panel_bytes_per_m;
    Mc = (Mc / Mr) * Mr;
    Mc = std::max(Mc, Mr);
    Mc = std::min(Mc, std::min(M, mc_max));

    // --- Nc: fit packed B (Kc x Nc) in L3 (or L2 if no L3) ---
    uint32_t nc_cache = (l3_bytes > 0) ? l3_bytes : l2_bytes;
    int b_panel_bytes_per_n = Kc * packed_b_elem_bytes;
    if (b_panel_bytes_per_n <= 0) b_panel_bytes_per_n = 1;
    int Nc = (int)(nc_cache * l3_util) / b_panel_bytes_per_n;
    Nc = (Nc / Nr) * Nr;
    Nc = std::max(Nc, Nr);
    Nc = std::min(Nc, std::min(N, nc_max));

    p.Mc = Mc;
    p.Nc = Nc;
    p.Kc = Kc;
    return p;
}

/// Backward-compatible overload: uses detected hardware + lookup profile.
inline GemmBlockingParams compute_blocking_params(
    const ArmHwProfile& hw,
    int Mr, int Nr, int Kgroup,
    int packed_a_elem_bytes, int packed_b_elem_bytes,
    int M, int N, int K) {

    const auto& profile = lookup_tuning_profile(hw);
    return compute_blocking_params(hw, profile, Mr, Nr, Kgroup,
                                   packed_a_elem_bytes, packed_b_elem_bytes,
                                   M, N, K);
}

/// Legacy: select cache blocking parameters based on detected CPU part number.
/// Preserved for backward compatibility with old drivers.
inline GemmBlockingParams get_gemm_blocking_params() {
    const auto& hw = detect_arm_hwcaps();
    GemmBlockingParams p;
    p.Mr = kGemmMrFp32;
    p.Nr = kGemmNrFp32;

    switch (hw.part_number) {
    case 0xd40:  // Neoverse V1
    case 0xd4f:  // Neoverse V2
        p.Mc = 128;  p.Nc = 2048;  p.Kc = 384;
        break;
    case 0xd48:  // Cortex-X2
        p.Mc = 64;   p.Nc = 1024;  p.Kc = 256;
        break;
    case 0xd0c:  // Neoverse N1
    case 0xd49:  // Neoverse N2
    default:
        p.Mc = 128;  p.Nc = 2048;  p.Kc = 512;
        break;
    }
    return p;
}

// ============================================================
// v2 optimized functions (prefetch + software pipelining)
// ============================================================

/// v1 small-M driver (baseline).
void gemm_smallm_driver_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc);

/// v2 small-M driver with prefetch optimizations.
void gemm_smallm_driver_fp32_v2(int M, int N, int K,
                                 float alpha, const float* A, int lda,
                                 const float* B, int ldb,
                                 float beta, float* C, int ldc);

/// Wide-panel small-M driver (48-col panels, optimized for M=2-7).
void gemm_smallm_wide_driver_fp32(int M, int N, int K,
                                   float alpha, const float* A, int lda,
                                   const float* B, int ldb,
                                   float beta, float* C, int ldc);

/// Unpacked driver for M=8-32 (skips packing, reads A/B directly).
void gemm_driver_unpacked_fp32(int M, int N, int K,
                                float alpha, const float* A, int lda,
                                const float* B, int ldb,
                                float beta, float* C, int ldc);

/// Small-K GEMM (K ≤ 16): preloads B, shares across M rows.
void gemm_smallK_fp32(int M, int N, int K,
                       float alpha, const float* A, int lda,
                       const float* B, int ldb,
                       float beta, float* C, int ldc);

/// Mx1 GEMM (matrix-vector, defined in gemm_tiny_fp32.cpp).
void gemm_mx1_fp32(int M, int K,
                    float alpha, const float* A, int lda,
                    const float* B, int ldb,
                    float beta, float* C, int ldc);

// ============================================================
// Phase 8: Adaptive tile GEMM (autoGEMM-style dynamic assembly)
// ============================================================

/// Tile configuration for adaptive kernel selection.
struct TileConfig {
    int Mr, Nr;
};

/// Select optimal tile (Mr, Nr) for given shape and hardware.
TileConfig select_tile_fp32(int M, int N, int K, uint32_t l1d_bytes);

/// Adaptive tile GEMM driver (unpacked, no packing).
void gemm_adaptive_tile_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc);

// ============================================================
// Phase 9: Inline assembly micro-kernels (autoGEMM-style)
// ============================================================
#ifdef __aarch64__

void gemm_kernel_4x16_asm(int K, const float* A, int lda,
                           const float* B, int ldb,
                           float* C, int ldc,
                           float alpha, float beta);

void gemm_kernel_6x16_asm(int K, const float* A, int lda,
                           const float* B, int ldb,
                           float* C, int ldc,
                           float alpha, float beta);

void gemm_kernel_3x16_asm(int K, const float* A, int lda,
                           const float* B, int ldb,
                           float* C, int ldc,
                           float alpha, float beta);

void gemm_kernel_5x16_asm(int K, const float* A, int lda,
                           const float* B, int ldb,
                           float* C, int ldc,
                           float alpha, float beta);

#endif  // __aarch64__

}  // namespace dnnopt
