#pragma once
/// @file gemm_config.h
/// Platform-specific and adaptive cache blocking parameters for GEMM.

#include "dnnopt/gemm/gemm_types.h"
#include "dnnopt/arm_hwcaps.h"

#include <algorithm>

namespace dnnopt {

// FP32 NEON microkernel tile dimensions.
constexpr int kGemmMrFp32 = 8;
constexpr int kGemmNrFp32 = 12;

// BF16 BFMMLA microkernel tile dimensions.
constexpr int kGemmMrBf16 = 8;
constexpr int kGemmNrBf16 = 8;

// INT8 SMMLA microkernel tile dimensions.
constexpr int kGemmMrInt8 = 8;
constexpr int kGemmNrInt8 = 8;

/// Compute cache blocking parameters from actual cache sizes.
///
/// Algorithm:
///   Kc: sized so one Mr-panel of A and one Nr-panel of B fit in ~40% of L1D.
///   Mc: sized so packed A (Mc x Kc) fits in ~40% of L2.
///   Nc: sized so packed B (Kc x Nc) fits in ~30% of L3 (or L2 if no L3).
///
/// All results are aligned down to the appropriate tile/group boundary.
inline GemmBlockingParams compute_blocking_params(
    const ArmHwProfile& hw,
    int Mr, int Nr, int Kgroup,
    int packed_a_elem_bytes, int packed_b_elem_bytes,
    int M, int N, int K) {

    GemmBlockingParams p;
    p.Mr = Mr;
    p.Nr = Nr;

    uint32_t l1d_bytes = hw.l1d.size_bytes;
    uint32_t l2_bytes  = hw.l2.size_bytes;
    uint32_t l3_bytes  = hw.l3.size_bytes;

    // Sensible defaults if sysfs detection failed
    if (l1d_bytes == 0) l1d_bytes = 64 * 1024;    // 64 KB
    if (l2_bytes  == 0) l2_bytes  = 1024 * 1024;   // 1 MB

    // --- Kc: fit one Mr-panel and one Nr-panel in ~40% of L1D ---
    // Per K-element: Mr * a_bytes + Nr * b_bytes
    int bytes_per_k = Mr * packed_a_elem_bytes + Nr * packed_b_elem_bytes;
    if (bytes_per_k <= 0) bytes_per_k = 1;
    int Kc = (int)(l1d_bytes * 0.4) / bytes_per_k;
    // Align down to Kgroup
    if (Kgroup > 1) Kc = (Kc / Kgroup) * Kgroup;
    Kc = std::max(Kc, Kgroup);  // at least one K-group
    Kc = std::min(Kc, K);       // don't exceed problem size

    // --- Mc: fit packed A (Mc x Kc) in ~40% of L2 ---
    int a_panel_bytes_per_m = Kc * packed_a_elem_bytes;
    if (a_panel_bytes_per_m <= 0) a_panel_bytes_per_m = 1;
    int Mc = (int)(l2_bytes * 0.4) / a_panel_bytes_per_m;
    Mc = (Mc / Mr) * Mr;          // align down to Mr
    Mc = std::max(Mc, Mr);        // at least one Mr-tile
    Mc = std::min(Mc, std::min(M, 512));  // upper bound

    // --- Nc: fit packed B (Kc x Nc) in ~30% of L3 (or L2) ---
    uint32_t nc_cache = (l3_bytes > 0) ? l3_bytes : l2_bytes;
    int b_panel_bytes_per_n = Kc * packed_b_elem_bytes;
    if (b_panel_bytes_per_n <= 0) b_panel_bytes_per_n = 1;
    int Nc = (int)(nc_cache * 0.3) / b_panel_bytes_per_n;
    Nc = (Nc / Nr) * Nr;          // align down to Nr
    Nc = std::max(Nc, Nr);        // at least one Nr-tile
    Nc = std::min(Nc, std::min(N, 8192));  // upper bound

    p.Mc = Mc;
    p.Nc = Nc;
    p.Kc = Kc;
    return p;
}

/// Legacy: select cache blocking parameters based on detected CPU part number.
/// Preserved for backward compatibility.
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

}  // namespace dnnopt
