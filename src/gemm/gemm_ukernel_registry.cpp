/// @file gemm_ukernel_registry.cpp
/// Microkernel registry implementation.

#include "dnnopt/gemm/gemm_ukernel_registry.h"

#include <algorithm>

namespace dnnopt {

GemmUkernelRegistry& GemmUkernelRegistry::instance() {
    static GemmUkernelRegistry reg;
    return reg;
}

void GemmUkernelRegistry::register_kernel(const GemmMicrokernelDesc& desc) {
    kernels_.push_back(desc);
}

const GemmMicrokernelDesc* GemmUkernelRegistry::select(
    GemmDataType dtype, const ArmHwProfile& hw) const {

    const GemmMicrokernelDesc* best = nullptr;
    int best_priority = -1;

    for (const auto& k : kernels_) {
        if (k.dtype != dtype) continue;
        // Check all required hwcaps are present
        if ((hw.hwcaps & k.required_hwcaps) != k.required_hwcaps) continue;
        // Check SVE vector length requirement
        if (k.min_sve_bits > 0 && (int)hw.sve_vector_bits < k.min_sve_bits)
            continue;
        // For VLA kernels on narrow SVE, ensure computed Nr is reasonable
        if (k.nr_is_vla && hw.sve_vector_bits > 0) {
            int nr = k.compute_nr(hw.sve_vector_bits);
            if (nr < 4) continue;  // Too narrow to be useful
        }
        if (k.priority > best_priority) {
            best = &k;
            best_priority = k.priority;
        }
    }
    return best;
}

std::vector<const GemmMicrokernelDesc*> GemmUkernelRegistry::select_all(
    GemmDataType dtype, const ArmHwProfile& hw) const {

    std::vector<const GemmMicrokernelDesc*> result;
    for (const auto& k : kernels_) {
        if (k.dtype != dtype) continue;
        if ((hw.hwcaps & k.required_hwcaps) != k.required_hwcaps) continue;
        if (k.min_sve_bits > 0 && (int)hw.sve_vector_bits < k.min_sve_bits)
            continue;
        if (k.nr_is_vla && hw.sve_vector_bits > 0) {
            int nr = k.compute_nr(hw.sve_vector_bits);
            if (nr < 4) continue;
        }
        result.push_back(&k);
    }
    std::sort(result.begin(), result.end(),
              [](const GemmMicrokernelDesc* a, const GemmMicrokernelDesc* b) {
                  return a->priority > b->priority;
              });
    return result;
}

}  // namespace dnnopt
