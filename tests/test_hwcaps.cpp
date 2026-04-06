/// @file test_hwcaps.cpp
/// Test ARM hardware capability detection.

#include "dnnopt/arm_hwcaps.h"
#include "test_utils.h"

#include <cstdio>

int main() {
    printf("=== test_hwcaps ===\n");

    const auto& hw = dnnopt::detect_arm_hwcaps();

    // Basic sanity: we must be on AArch64 if this compiled
    TEST_ASSERT(hw.has(dnnopt::kNEON), "NEON must be available on AArch64");
    TEST_ASSERT(hw.num_cores > 0, "Must detect at least 1 core");
    TEST_ASSERT(hw.cpu_name.size() > 0, "CPU name must not be empty");

    // Cache detection
    TEST_ASSERT(hw.l1d.size_bytes > 0, "L1D cache size must be > 0");
    TEST_ASSERT(hw.l1d.line_size >= 32 && hw.l1d.line_size <= 256,
                "L1D line size must be reasonable (32-256)");
    TEST_ASSERT(hw.l2.size_bytes > hw.l1d.size_bytes,
                "L2 must be larger than L1D");

    // Consistency checks
    if (hw.has(dnnopt::kSVE2)) {
        TEST_ASSERT(hw.has(dnnopt::kSVE), "SVE2 implies SVE");
    }
    if (hw.has(dnnopt::kSME2)) {
        TEST_ASSERT(hw.has(dnnopt::kSME), "SME2 implies SME");
    }

    // SVE vector length
    if (hw.has(dnnopt::kSVE)) {
        TEST_ASSERT(hw.sve_vector_bits >= 128, "SVE VL must be >= 128");
        TEST_ASSERT(hw.sve_vector_bits % 128 == 0, "SVE VL must be multiple of 128");
    }

    // Peak performance should be computed if freq is known
    if (hw.freq_mhz > 0) {
        TEST_ASSERT(hw.fp32_gflops_per_core > 0, "FP32 peak must be > 0");
    }

    // Platform tag
    std::string tag = dnnopt::platform_tag(hw);
    TEST_ASSERT(tag.size() > 0, "Platform tag must not be empty");
    printf("  Platform tag: %s\n", tag.c_str());

    // Print full report for visual inspection
    dnnopt::print_hwcaps_summary(hw);

    TEST_SUMMARY();
}
