#pragma once
/// @file gemm_types.h
/// Shared types for GEMM operations.

#include <cstdint>

namespace dnnopt {

/// bfloat16 storage type: 16-bit with 7-bit mantissa.
/// Binary-compatible with oneDNN's bfloat16_t and arm_neon.h's internal type.
struct bfloat16_t {
    uint16_t raw_bits;

    bfloat16_t() = default;
    constexpr explicit bfloat16_t(uint16_t raw, bool) : raw_bits(raw) {}

    /// Construct from float (round-to-nearest-even).
    explicit bfloat16_t(float f) {
        union { float f32; uint32_t u32; } u;
        u.f32 = f;
        // Round-to-nearest-even: add 0x7FFF + (bit 15 of mantissa)
        uint32_t round = 0x7FFF + ((u.u32 >> 16) & 1);
        raw_bits = static_cast<uint16_t>((u.u32 + round) >> 16);
    }

    /// Convert to float.
    operator float() const {
        union { float f32; uint32_t u32; } u;
        u.u32 = static_cast<uint32_t>(raw_bits) << 16;
        return u.f32;
    }

    bfloat16_t& operator=(float f) {
        *this = bfloat16_t(f);
        return *this;
    }
};

static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

/// Algorithm selection for GEMM dispatch.
enum class GemmAlgo {
    kAuto,        // Automatic: pick best for current hardware + shape
    kNaive,       // Scalar reference (for testing only)
    kNeonFp32,    // NEON 8x12 FP32 microkernel + BLIS blocking
    kBf16Bfmmla,  // BF16 BFMMLA microkernel
    kInt8Smmla,   // INT8 SMMLA microkernel
    kInt8Sdot,    // INT8 SDOT microkernel (future)
    kSveFp32,     // SVE FP32 microkernel
    kSveBf16,     // SVE BF16 microkernel
    kSveInt8,     // SVE INT8 microkernel
    kSmeFp32,     // SME FP32 microkernel (future)
};

/// Cache blocking parameters for BLIS-style GEMM.
struct GemmBlockingParams {
    int Mr;   // Microkernel row tile
    int Nr;   // Microkernel column tile
    int Mc;   // L2 blocking on M
    int Nc;   // L3 blocking on N
    int Kc;   // L2 blocking on K
};

}  // namespace dnnopt
