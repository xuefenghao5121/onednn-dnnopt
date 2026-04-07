#pragma once
/// @file gemm_ukernel_registry.h
/// Microkernel registry for adaptive GEMM dispatch.
/// Kernels self-register at static-init time; the registry selects the
/// highest-priority implementation whose hardware requirements are met.

#include "dnnopt/arm_hwcaps.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dnnopt {

/// Data type tag for GEMM dispatch.
enum class GemmDataType { kFP32, kBF16, kINT8 };

// ============================================================
// Unified microkernel function signatures
// ============================================================

/// Microkernel: compute a Mr x Nr tile from packed A and packed B.
/// packed_A/packed_B are typed as void* to unify FP32/BF16/INT8.
/// For INT8 kernels, `extra` carries the dequant_scale (as float, bit-cast via memcpy).
/// For FP32/BF16 kernels, `extra` is unused (0.0f).
using UkernelFn = void (*)(int K, const void* packed_A, const void* packed_B,
                           float* C, int ldc, float alpha, float beta,
                           float extra);

/// Pack A: (m_len x k_len) block of FP32 input -> packed format.
/// For BF16/INT8, includes type conversion (FP32->BF16 or FP32->INT8).
/// For INT8, `scale_out` receives the quantization scale; otherwise ignored.
using PackAFn = void (*)(int m_len, int k_len, const float* A, int lda,
                         void* packed_A, int Mr, float* scale_out);

/// Pack B: (k_len x n_len) block of FP32 input -> packed format.
using PackBFn = void (*)(int k_len, int n_len, const float* B, int ldb,
                         void* packed_B, int Nr, float* scale_out);

// ============================================================
// Microkernel descriptor
// ============================================================

struct GemmMicrokernelDesc {
    const char* name;           // e.g. "neon_fp32_8x12"
    GemmDataType dtype;
    uint64_t required_hwcaps;   // bitmask of HwCap flags
    int Mr;                     // row tile (fixed)
    int Nr;                     // column tile (fixed for NEON, base for VLA)
    int Kgroup;                 // K granularity (1 for FP32, 4 for BF16, 8 for INT8)
    bool nr_is_vla;             // true if Nr scales with SVE vector length
    int priority;               // higher = preferred (NEON=100, SVE128=150, SVE_wide=200, SME=300)
    int packed_a_elem_bytes;    // bytes per element in packed A
    int packed_b_elem_bytes;    // bytes per element in packed B
    int min_sve_bits;           // minimum SVE VL in bits (0 = no requirement)

    UkernelFn ukernel;
    PackAFn pack_a;
    PackBFn pack_b;

    /// Compute actual Nr for VLA kernels. Returns Nr as-is for non-VLA.
    int compute_nr(int sve_bits) const {
        if (!nr_is_vla) return Nr;
        // Nr = 2 * (sve_bits / 32) for FP32
        // For BF16/INT8, Nr = 2 * (sve_bits / 32) as well (2 col-pairs per SVE reg)
        int vl_words = sve_bits / 32;
        return 2 * vl_words;
    }

    /// Packed A panel stride (bytes) for a given Kc.
    size_t a_panel_stride(int kc_padded) const {
        if (dtype == GemmDataType::kFP32) {
            return (size_t)Mr * kc_padded * packed_a_elem_bytes;
        }
        // BF16/INT8: (kc_padded / Kgroup) * (Mr/2) * 2*Kgroup * elem_bytes
        // Simplified: (kc_padded / Kgroup) * Mr * Kgroup * elem_bytes
        return (size_t)(kc_padded / Kgroup) * Mr * Kgroup * packed_a_elem_bytes;
    }

    /// Packed B panel stride (bytes) for a given Kc and actual Nr.
    size_t b_panel_stride(int kc_padded, int actual_nr) const {
        if (dtype == GemmDataType::kFP32) {
            return (size_t)actual_nr * kc_padded * packed_b_elem_bytes;
        }
        return (size_t)(kc_padded / Kgroup) * actual_nr * Kgroup * packed_b_elem_bytes;
    }
};

// ============================================================
// Registry
// ============================================================

class GemmUkernelRegistry {
public:
    static GemmUkernelRegistry& instance();

    void register_kernel(const GemmMicrokernelDesc& desc);

    /// Select the best kernel for the given dtype and hardware.
    /// Returns nullptr if no kernel matches.
    const GemmMicrokernelDesc* select(GemmDataType dtype,
                                      const ArmHwProfile& hw) const;

    /// Return all matching kernels, sorted by priority (descending).
    std::vector<const GemmMicrokernelDesc*> select_all(
        GemmDataType dtype, const ArmHwProfile& hw) const;

private:
    GemmUkernelRegistry() = default;
    std::vector<GemmMicrokernelDesc> kernels_;
};

// ============================================================
// Auto-registration helper
// ============================================================

/// Place a file-scope `static RegisterKernel reg_(desc);` to auto-register.
struct RegisterKernel {
    RegisterKernel(const GemmMicrokernelDesc& desc) {
        GemmUkernelRegistry::instance().register_kernel(desc);
    }
};

}  // namespace dnnopt
