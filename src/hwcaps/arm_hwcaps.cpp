/// @file arm_hwcaps.cpp
/// ARM hardware capability detection implementation.

#include "dnnopt/arm_hwcaps.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#ifdef __linux__
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

namespace dnnopt {
namespace {

// ============================================================
// CPU part number → human-readable name
// ============================================================
const char* cpu_part_name(uint32_t implementer, uint32_t part) {
    if (implementer == 0x41) {  // ARM Ltd
        switch (part) {
        case 0xd03: return "Cortex-A53";
        case 0xd04: return "Cortex-A35";
        case 0xd05: return "Cortex-A55";
        case 0xd07: return "Cortex-A57";
        case 0xd08: return "Cortex-A72";
        case 0xd09: return "Cortex-A73";
        case 0xd0a: return "Cortex-A75";
        case 0xd0b: return "Cortex-A76";
        case 0xd0c: return "Neoverse N1";
        case 0xd0d: return "Cortex-A77";
        case 0xd40: return "Neoverse V1";
        case 0xd41: return "Cortex-A78";
        case 0xd44: return "Cortex-X1";
        case 0xd46: return "Cortex-A510";
        case 0xd47: return "Cortex-A710";
        case 0xd48: return "Cortex-X2";
        case 0xd49: return "Neoverse N2";
        case 0xd4a: return "Neoverse E1";
        case 0xd4b: return "Cortex-A78C";
        case 0xd4c: return "Cortex-X1C";
        case 0xd4d: return "Cortex-A715";
        case 0xd4e: return "Cortex-X3";
        case 0xd4f: return "Neoverse V2";
        case 0xd80: return "Cortex-A520";
        case 0xd81: return "Cortex-A720";
        case 0xd82: return "Cortex-X4";
        case 0xd84: return "Neoverse V3";
        default:    return "Unknown ARM";
        }
    }
    if (implementer == 0x48) {  // HiSilicon
        switch (part) {
        case 0xd01: return "Kunpeng 920 (TSV110)";
        default:    return "Unknown HiSilicon";
        }
    }
    if (implementer == 0xc0) {  // Ampere
        switch (part) {
        case 0xac3: return "Ampere Altra";
        case 0xac4: return "Ampere AmpereOne";
        default:    return "Unknown Ampere";
        }
    }
    return "Unknown";
}

// ============================================================
// Parse /proc/cpuinfo for CPU identification
// ============================================================
void parse_cpuinfo(ArmHwProfile& p) {
    std::ifstream f("/proc/cpuinfo");
    if (!f.is_open()) return;

    std::string line;
    bool first_core = true;
    while (std::getline(f, line)) {
        if (!first_core) continue;  // Only parse first core
        if (line.find("CPU implementer") != std::string::npos) {
            sscanf(line.c_str(), "CPU implementer\t: %x", &p.implementer);
        } else if (line.find("CPU part") != std::string::npos) {
            sscanf(line.c_str(), "CPU part\t: %x", &p.part_number);
        } else if (line.find("CPU variant") != std::string::npos) {
            sscanf(line.c_str(), "CPU variant\t: %x", &p.variant);
        } else if (line.find("CPU revision") != std::string::npos) {
            sscanf(line.c_str(), "CPU revision\t: %u", &p.revision);
            first_core = false;  // Done with first core
        } else if (line.find("CPU MHz") != std::string::npos) {
            float mhz = 0;
            sscanf(line.c_str(), "CPU MHz\t\t: %f", &mhz);
            p.freq_mhz = static_cast<uint32_t>(mhz);
        }
    }
    p.cpu_name = cpu_part_name(p.implementer, p.part_number);

    // Fallback: try lscpu or sysfs for frequency
    if (p.freq_mhz == 0) {
        std::ifstream fmax("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
        if (fmax.is_open()) {
            uint32_t khz = 0;
            fmax >> khz;
            p.freq_mhz = khz / 1000;
        }
    }
    // Second fallback: read BogoMIPS as rough estimate
    if (p.freq_mhz == 0) {
        std::ifstream f2("/proc/cpuinfo");
        if (f2.is_open()) {
            std::string line2;
            while (std::getline(f2, line2)) {
                if (line2.find("BogoMIPS") != std::string::npos) {
                    float bogo = 0;
                    sscanf(line2.c_str(), "BogoMIPS\t: %f", &bogo);
                    if (bogo > 0) p.freq_mhz = static_cast<uint32_t>(bogo * 30);
                    break;
                }
            }
        }
    }
}

// ============================================================
// Detect capabilities via getauxval (Linux)
// ============================================================
void detect_hwcaps_auxval(ArmHwProfile& p) {
#ifdef __linux__
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    // HWCAP flags
    if (hwcap & HWCAP_ASIMD)    p.hwcaps |= kNEON;
    if (hwcap & HWCAP_FPHP)     p.hwcaps |= kFP16;
    if (hwcap & HWCAP_ASIMDDP)  p.hwcaps |= kDotProd;
    if (hwcap & HWCAP_SVE)      p.hwcaps |= kSVE;
    if (hwcap & HWCAP_ATOMICS)  p.hwcaps |= kAtomics;
    if (hwcap & HWCAP_AES)      p.hwcaps |= kAES;
    if (hwcap & HWCAP_SHA2)     p.hwcaps |= kSHA256;

    // HWCAP2 flags
#ifdef HWCAP2_SVE2
    if (hwcap2 & HWCAP2_SVE2)     p.hwcaps |= kSVE2;
#endif
#ifdef HWCAP2_SVEBF16
    if (hwcap2 & HWCAP2_SVEBF16)  p.hwcaps |= kSVEBF16;
#endif
#ifdef HWCAP2_SVEI8MM
    if (hwcap2 & HWCAP2_SVEI8MM)  p.hwcaps |= kSVEI8MM;
#endif
#ifdef HWCAP2_BF16
    if (hwcap2 & HWCAP2_BF16)     p.hwcaps |= kBF16;
#endif
#ifdef HWCAP2_I8MM
    if (hwcap2 & HWCAP2_I8MM)     p.hwcaps |= kI8MM;
#endif
#ifdef HWCAP2_SME
    if (hwcap2 & HWCAP2_SME)      p.hwcaps |= kSME;
#endif
#ifdef HWCAP2_SME2
    if (hwcap2 & HWCAP2_SME2)     p.hwcaps |= kSME2;
#endif
#ifdef HWCAP2_FRINT
    if (hwcap2 & HWCAP2_FRINT)    p.hwcaps |= kFRINT;
#endif
#endif  // __linux__
}

// ============================================================
// Detect SVE vector length
// ============================================================
void detect_sve_vl(ArmHwProfile& p) {
    if (!p.has(kSVE)) return;

#ifdef __ARM_FEATURE_SVE
    uint64_t vl_bytes;
    asm volatile("rdvl %0, #1" : "=r"(vl_bytes));
    p.sve_vector_bits = static_cast<uint32_t>(vl_bytes * 8);
#else
    // Fallback: try reading from prctl or sysfs
    // rdvl is only available if compiled with SVE support
    p.sve_vector_bits = 0;  // Unknown
#endif
}

// ============================================================
// Detect cache hierarchy from sysfs
// ============================================================
uint32_t read_sysfs_uint(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) return 0;
    std::string val;
    std::getline(f, val);
    if (val.empty()) return 0;
    // Handle K/M suffix
    uint32_t n = 0;
    char suffix = 0;
    sscanf(val.c_str(), "%u%c", &n, &suffix);
    if (suffix == 'K' || suffix == 'k') n *= 1024;
    else if (suffix == 'M' || suffix == 'm') n *= 1024 * 1024;
    return n;
}

void detect_cache_info(CacheInfo& ci, int index) {
    char path[256];
    const char* base = "/sys/devices/system/cpu/cpu0/cache/index";

    snprintf(path, sizeof(path), "%s%d/size", base, index);
    ci.size_bytes = read_sysfs_uint(path);

    snprintf(path, sizeof(path), "%s%d/coherency_line_size", base, index);
    ci.line_size = read_sysfs_uint(path);

    snprintf(path, sizeof(path), "%s%d/ways_of_associativity", base, index);
    ci.ways = read_sysfs_uint(path);

    snprintf(path, sizeof(path), "%s%d/number_of_sets", base, index);
    ci.sets = read_sysfs_uint(path);
}

void detect_caches(ArmHwProfile& p) {
    // index0 = L1D, index1 = L1I, index2 = L2, index3 = L3
    detect_cache_info(p.l1d, 0);
    detect_cache_info(p.l1i, 1);
    detect_cache_info(p.l2,  2);
    detect_cache_info(p.l3,  3);
}

// ============================================================
// Compute theoretical peak performance
// ============================================================
void compute_peak_perf(ArmHwProfile& p) {
    if (p.freq_mhz == 0) return;
    double ghz = p.freq_mhz / 1000.0;

    // NEON: 2 FMLA units × 4 FP32/unit × 2 (FMA) = 16 FLOP/cycle (typical N2)
    // Conservative: assume 2 FMLA/cycle for NEON 128-bit
    double neon_flops_per_cycle = 2.0 * 4.0 * 2.0;  // 16 FLOP/cycle
    p.fp32_gflops_per_core = ghz * neon_flops_per_cycle;

    if (p.has(kSVE) && p.sve_vector_bits > 0) {
        // SVE: scale by vector width ratio
        double sve_ratio = static_cast<double>(p.sve_vector_bits) / 128.0;
        // Neoverse V1 (256-bit SVE): 2 × 8 FP32 × 2 = 32 FLOP/cycle
        // Neoverse N2 (128-bit SVE2): similar to NEON
        p.fp32_gflops_per_core = ghz * neon_flops_per_cycle * sve_ratio;
    }

    // BF16: BFMMLA does 2x4 × 4x2 = 16 BF16 MACs → 32 FLOP per instruction
    // Roughly 2× FP32 throughput
    if (p.has(kBF16)) {
        p.bf16_gflops_per_core = p.fp32_gflops_per_core * 2.0;
    }

    // INT8: SDOT/SMMLA, roughly 4× FP32
    if (p.has(kI8MM)) {
        p.int8_gops_per_core = p.fp32_gflops_per_core * 4.0;
    } else if (p.has(kDotProd)) {
        p.int8_gops_per_core = p.fp32_gflops_per_core * 4.0;
    }
}

}  // anonymous namespace

// ============================================================
// Public API
// ============================================================

const ArmHwProfile& detect_arm_hwcaps() {
    static ArmHwProfile profile = []() {
        ArmHwProfile p;
        p.num_cores = std::thread::hardware_concurrency();
        parse_cpuinfo(p);
        detect_hwcaps_auxval(p);
        detect_sve_vl(p);
        detect_caches(p);
        compute_peak_perf(p);
        return p;
    }();
    return profile;
}

void print_hwcaps_summary(const ArmHwProfile& profile) {
    const auto& p = profile;

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║          ARM Hardware Capability Report             ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║ CPU: %-20s  Cores: %-3u  MHz: %-5u ║\n",
           p.cpu_name.c_str(), p.num_cores, p.freq_mhz);
    printf("║ Implementer: 0x%02x  Part: 0x%03x  Rev: r%up%u        ║\n",
           p.implementer, p.part_number, p.variant, p.revision);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║ SIMD / Vector Extensions:                           ║\n");
    printf("║   NEON (ASIMD) : %-3s                                ║\n",
           p.has(kNEON) ? "YES" : "NO");
    printf("║   FP16         : %-3s                                ║\n",
           p.has(kFP16) ? "YES" : "NO");
    printf("║   DotProd      : %-3s  (SDOT/UDOT)                  ║\n",
           p.has(kDotProd) ? "YES" : "NO");
    printf("║   SVE          : %-3s",
           p.has(kSVE) ? "YES" : "NO");
    if (p.sve_vector_bits > 0)
        printf("  (%u-bit)", p.sve_vector_bits);
    printf("                       ║\n");
    printf("║   SVE2         : %-3s                                ║\n",
           p.has(kSVE2) ? "YES" : "NO");
    printf("║   BF16         : %-3s  (BFMMLA/BFDOT)               ║\n",
           p.has(kBF16) ? "YES" : "NO");
    printf("║   I8MM         : %-3s  (SMMLA/UMMLA)                ║\n",
           p.has(kI8MM) ? "YES" : "NO");
    printf("║   SME          : %-3s                                ║\n",
           p.has(kSME) ? "YES" : "NO");
    printf("║   SME2         : %-3s                                ║\n",
           p.has(kSME2) ? "YES" : "NO");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║ Cache Hierarchy:                                    ║\n");
    printf("║   L1D: %6u KB  line=%uB  %u-way                  ║\n",
           p.l1d.size_bytes/1024, p.l1d.line_size, p.l1d.ways);
    printf("║   L1I: %6u KB  line=%uB  %u-way                  ║\n",
           p.l1i.size_bytes/1024, p.l1i.line_size, p.l1i.ways);
    printf("║   L2:  %6u KB  line=%uB  %u-way                  ║\n",
           p.l2.size_bytes/1024, p.l2.line_size, p.l2.ways);
    if (p.l3.size_bytes > 0)
        printf("║   L3:  %6u KB  line=%uB  %u-way                  ║\n",
               p.l3.size_bytes/1024, p.l3.line_size, p.l3.ways);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║ Theoretical Peak (per core):                        ║\n");
    printf("║   FP32:  %7.2f GFLOPS                              ║\n",
           p.fp32_gflops_per_core);
    if (p.has(kBF16))
        printf("║   BF16:  %7.2f GFLOPS                              ║\n",
               p.bf16_gflops_per_core);
    if (p.int8_gops_per_core > 0)
        printf("║   INT8:  %7.2f GOPS                                ║\n",
               p.int8_gops_per_core);
    printf("╚══════════════════════════════════════════════════════╝\n");
}

std::string platform_tag(const ArmHwProfile& profile) {
    std::string tag;
    // Lowercase CPU name, replace spaces with underscores
    for (char c : profile.cpu_name) {
        if (c == ' ' || c == '-') tag += '_';
        else tag += static_cast<char>(tolower(c));
    }
    if (profile.has(kSVE2))     tag += "_sve2";
    else if (profile.has(kSVE)) tag += "_sve";
    else                        tag += "_neon";

    if (profile.sve_vector_bits > 0)
        tag += "_" + std::to_string(profile.sve_vector_bits);

    if (profile.has(kBF16)) tag += "_bf16";
    if (profile.has(kI8MM)) tag += "_i8mm";

    return tag;
}

}  // namespace dnnopt
