/// @file hwcaps_report.cpp
/// Standalone tool: print ARM hardware capability report.

#include "dnnopt/arm_hwcaps.h"

int main() {
    const auto& profile = dnnopt::detect_arm_hwcaps();
    dnnopt::print_hwcaps_summary(profile);
    printf("\nPlatform tag: %s\n", dnnopt::platform_tag(profile).c_str());
    return 0;
}
