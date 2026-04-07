# DNN-Opt: ARM Platform Deep Learning Optimization Library

**Version: 0.3.0** (Phase 3A: Hardware-Adaptive Tuning Infrastructure)

ARM 平台高性能深度学习推理优化库，充分利用 NEON/SVE/SVE2/SME 指令集和微架构特征，在 ARM CPU 环境下实现极致推理性能。

## Performance Highlights

| Precision | Microkernel | Peak % (Neoverse N2) | vs FP32 |
|-----------|-------------|----------------------|---------|
| **FP32** | NEON 8x12 FMLA | 93% (44.6 / 48 GFLOPS) | 1.0x |
| **BF16** | BFMMLA 8x8 | 86.5% (166 / 192 GFLOPS) | 3.67x |
| **INT8** | SMMLA 8x8 | 70.8% (272 / 384 GOPS) | 6.1x |

## Supported ARM Platforms

| CPU Core | Architecture | Vector Extension | ML Features |
|----------|-------------|-----------------|-------------|
| Neoverse N1 | ARMv8.2 | 128-bit NEON | DotProd |
| Neoverse N2 | ARMv9.0 | 128-bit SVE2 | BF16, I8MM, SVE2 |
| Neoverse V1 | ARMv8.4+ | 256-bit SVE | BF16, SVE |
| Neoverse V2 | ARMv9.0 | 128-bit SVE2 | BF16, I8MM, SME |
| Cortex-A78 | ARMv8.2 | 128-bit NEON | DotProd |
| Cortex-X2/X3 | ARMv9.0 | 128-bit SVE2 | BF16, I8MM |
| Cortex-A55/A510 | ARMv8.2/v9 | NEON/SVE2 | Efficiency cores |
| A64FX | ARMv8.2-SVE | 512-bit SVE | HBM2, wide SVE |
| Kunpeng 920 | ARMv8.2 | 128-bit NEON | DotProd |

## Key Features

- **Hardware-Adaptive**: Automatically detects CPU capabilities and selects optimal kernel + blocking parameters
- **Per-CPU Tuning Profiles**: Built-in profiles for 11 ARM CPU families, auto-tuning fallback for unknown CPUs
- **Shape-Aware Blocking**: Matrix shape classification (square/tall-skinny/short-wide/BERT-like) with per-class cache utilization adjustments
- **Multi-Precision**: FP32, BF16 (BFMMLA), INT8 (SMMLA) with transparent quantization
- **Microkernel Registry**: Priority-based auto-dispatch (NEON=100 < SVE-128=120 < SVE-wide=200 < SME=300)
- **Generic BLIS Driver**: Parameterized 5-loop implementation with OpenMP threading

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run tests
./tests/test_gemm_correctness

# Run benchmarks
./benchmarks/bench_gemm
./benchmarks/hwcaps_report
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `DNNOPT_NATIVE_ARCH` | ON | Use `-march=native` for host CPU |
| `DNNOPT_USE_OPENMP` | ON | Enable OpenMP multi-threading |
| `DNNOPT_ENABLE_SME` | OFF | Enable SME kernel compilation |
| `DNNOPT_BUILD_TESTS` | ON | Build correctness tests |
| `DNNOPT_BUILD_BENCHMARKS` | ON | Build performance benchmarks |

## Project Structure

```
onednn-arm-opt/
├── include/dnnopt/
│   ├── arm_hwcaps.h              # Hardware capability detection
│   ├── cpu_tuning_profile.h      # Per-CPU tuning profiles [NEW v0.3.0]
│   ├── timer.h                   # High-resolution benchmarking
│   ├── aligned_alloc.h           # Cache-aligned memory allocation
│   └── gemm/
│       ├── gemm.h                # Public GEMM API
│       ├── gemm_types.h          # Data types and enums
│       ├── gemm_config.h         # Adaptive cache blocking
│       ├── gemm_autotune.h       # Runtime auto-tuning [NEW v0.3.0]
│       ├── gemm_driver_generic.h # Generic BLIS 5-loop driver
│       ├── gemm_threading.h      # Thread control
│       └── gemm_ukernel_registry.h # Microkernel registry
├── src/
│   ├── hwcaps/arm_hwcaps.cpp     # CPU detection (/proc/cpuinfo + sysfs)
│   ├── cpu_tuning_profiles.cpp   # Built-in tuning database [NEW v0.3.0]
│   ├── utils/                    # Timer, aligned alloc
│   └── gemm/
│       ├── gemm.cpp              # Top-level dispatch
│       ├── gemm_autotune.cpp     # Auto-tuning engine [NEW v0.3.0]
│       ├── gemm_driver_generic.cpp # BLIS 5-loop with OpenMP
│       ├── gemm_ukernel_registry.cpp
│       ├── gemm_ukernel_fp32_neon.cpp  # NEON FP32 8x12
│       ├── gemm_ukernel_bf16_neon.cpp  # BFMMLA BF16 8x8
│       ├── gemm_ukernel_int8_neon.cpp  # SMMLA INT8 8x8
│       ├── gemm_ukernel_fp32_sve.cpp   # SVE FP32 VLA
│       ├── gemm_ukernel_bf16_sve.cpp   # SVE BF16 VLA
│       ├── gemm_ukernel_int8_sve.cpp   # SVE INT8 VLA
│       ├── gemm_ukernel_fp32_sme.cpp   # SME FP32 stub
│       ├── gemm_pack_fp32.cpp
│       ├── gemm_pack_bf16.cpp
│       ├── gemm_pack_int8.cpp
│       └── gemm_smallm_fp32.cpp  # Small-M optimized path
├── tests/                        # Correctness tests (74 cases)
├── benchmarks/                   # Performance benchmarks + hwcaps report
└── docs/                         # Design documentation
```

## Development Log

### v0.3.0 — Phase 3A: Hardware-Adaptive Tuning Infrastructure (2026-04-07)

New: Hardware-adaptive GEMM framework for cross-platform portability.

- **CpuTuningProfile system**: Built-in tuning profiles for 11 ARM CPU families
  (N1, N2, V1, V2, A78, X2, X3, A55, A510, A64FX, Kunpeng920)
  with per-CPU cache utilization ratios, blocking bounds, prefetch distances
- **Shape-aware blocking**: Matrix shape classification (Square/TallSkinny/ShortWide/SmallGemm/BertLike)
  with per-class cache utilization multipliers. Replaces hardcoded 40%/40%/30% ratios
- **Runtime auto-tuning**: Micro-benchmark for unknown CPUs, 3-candidate search in <5ms,
  results cached for session lifetime
- **Adaptive dispatch**: `compute_blocking_params()` now uses profile + shape class
  instead of fixed parameters

Design inspired by [autoGEMM (SC'24)](https://github.com/wudu98/autoGEMM) dynamic tiling approach.

### v0.2.3 — Generic BLIS Driver + Microkernel Registry (2026-04-07)

- Generic parameterized BLIS 5-loop driver (replaces per-type drivers)
- Microkernel registry with priority-based auto-selection
- SVE FP32/BF16/INT8 VLA microkernel skeletons
- SME FP32 FMOPA compilation stub
- OpenMP threading with dynamic schedule

### v0.2.2 — Phase 2C: INT8 SMMLA GEMM (2026-04-07)

- 8x8 SMMLA microkernel: 70.8% INT8 peak (272 / 384 GOPS)
- 6.1x speedup over FP32, 1.66x over BF16
- Symmetric per-tensor FP32-to-INT8 quantization during packing

### v0.2.1 — Phase 2B: BF16 BFMMLA GEMM (2026-04-07)

- 8x8 BFMMLA microkernel: 86.5% BF16 peak (166 / 192 GFLOPS)
- 3.67x speedup over FP32
- FP32-to-BF16 conversion during packing

### v0.2.0 — Phase 2A: FP32 NEON GEMM (2026-04-06)

- 8x12 NEON FMLA microkernel: 93% FP32 peak
- BLIS-style cache blocking (L1/L2/L3 aware)
- Small-M specialized path for batch-1 inference (M<8)

### v0.1.0 — Phase 1: Infrastructure (2026-04-06)

- Build system (CMake, ARM intrinsics detection)
- Hardware capability detection (getauxval + sysfs + /proc/cpuinfo)
- Benchmark framework with CSV export
- Correctness test framework

## Roadmap

### Phase 3B: SVE/SVE2 Full Optimization (Next)
- SVE-128 specialized microkernels (beat NEON via predicates + RBSA register scrolling)
- SVE vectorized packing (critical for INT8 quantization acceleration)
- SVE prefetch integration with per-CPU prefetch distance tuning

### Phase 3C: Advanced Multi-threading
- M x N 2D thread decomposition (replace M-only parallelism)
- big.LITTLE core topology awareness
- NUMA-aware buffer allocation, huge pages

### Phase 3D: SME Framework
- Complete FMOPA/BFMOPA/SMOPA microkernels with proper ZA tile management
- Streaming mode transitions (SMSTART/SMSTOP)
- Compile-time gating + runtime hwcap detection

### Phase 4: Convolution Operators (Lower Priority)
- im2col + GEMM convolution
- Winograd F(2x2,3x3) / F(4x4,3x3)
- Direct convolution for 1x1 and depthwise
- Post-ops fusion (Conv + Bias + ReLU)

## References

- [autoGEMM: Pushing the Limits of Irregular Matrix Multiplication on Arm Architectures (SC'24)](https://github.com/wudu98/autoGEMM)
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://github.com/flame/blis)
- [ARM Neoverse N2 Technical Reference Manual](https://developer.arm.com/documentation/102099/latest)

## License

MIT
