# DNN-Opt: ARM Platform Deep Learning Optimization Library

**Version: 0.4.0** (Phase 4: Convolution Operators)

ARM 平台高性能深度学习推理优化库，充分利用 NEON/SVE/SVE2/SME 指令集和微架构特征，在 ARM CPU 环境下实现极致推理性能。

## Performance Highlights

### GEMM Microkernels

| Precision | Microkernel | Peak % (Neoverse N2) | vs FP32 |
|-----------|-------------|----------------------|---------|
| **FP32** | NEON 8x12 FMLA | 93% (44.6 / 48 GFLOPS) | 1.0x |
| **BF16** | BFMMLA 8x8 | 86.5% (166 / 192 GFLOPS) | 3.67x |
| **INT8** | SMMLA 8x8 | 70.8% (272 / 384 GOPS) | 6.1x |

### Conv2D (im2col + Optimized GEMM)

| Layer | Naive | dnnopt | Speedup | GFLOPS |
|-------|-------|--------|---------|--------|
| ResNet-Conv1 (7x7 s2) | 91.2 ms | 6.7 ms | **13.7x** | 35.4 |
| ResNet-3x3-64 | 64.3 ms | 4.3 ms | **15.0x** | 53.8 |
| ResNet-3x3-128 | 63.4 ms | 5.9 ms | **10.7x** | 39.1 |
| ResNet-3x3-256 | 61.0 ms | 6.7 ms | **9.2x** | 34.7 |
| ResNet-1x1-256 | 26.6 ms | 1.9 ms | **14.2x** | 54.8 |
| ResNet-1x1-64 | 27.9 ms | 1.8 ms | **15.5x** | 57.0 |
| MBNet-DW-128 | 259.9 ms | 14.7 ms | **17.7x** | 62.9 |

Peak Conv2D throughput: **62.9 GFLOPS** (Neoverse N2 @ 3GHz, 2 cores).

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
- **SVE VLA Kernels**: Register-resident accumulators for SVE-256/512, 4x K-loop unroll, col-pair chunking for BF16/INT8
- **SME Framework**: Complete FMOPA/BFMOPA/SMOPA microkernels with ZA tile management
- **2D Thread Decomposition**: Adaptive M×N parallelism with shape-aware scheduling
- **Big.LITTLE Awareness**: Core topology detection, performance-core-first scheduling, thread affinity
- **Huge Page Allocation**: MAP_HUGETLB + MADV_HUGEPAGE for large packing buffers
- **Conv2D Operator**: im2col + optimized GEMM, direct 1×1 fast path, fused post-ops (Bias/ReLU/ReLU6)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run tests
./tests/test_gemm_correctness
./tests/test_conv_correctness

# Run benchmarks
./benchmarks/bench_gemm
./benchmarks/bench_conv
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
│   ├── cpu_tuning_profile.h      # Per-CPU tuning profiles
│   ├── timer.h                   # High-resolution benchmarking
│   ├── aligned_alloc.h           # Cache-aligned + huge page allocation
│   ├── conv/
│   │   └── conv.h                # Public Conv2D API [NEW v0.4.0]
│   └── gemm/
│       ├── gemm.h                # Public GEMM API
│       ├── gemm_types.h          # Data types and enums
│       ├── gemm_config.h         # Adaptive cache blocking
│       ├── gemm_autotune.h       # Runtime auto-tuning
│       ├── gemm_driver_generic.h # Generic BLIS 5-loop driver
│       ├── gemm_threading.h      # Thread control + big.LITTLE affinity
│       ├── gemm_thread_decomp.h  # 2D M×N thread decomposition
│       └── gemm_ukernel_registry.h # Microkernel registry
├── src/
│   ├── hwcaps/arm_hwcaps.cpp     # CPU detection (/proc/cpuinfo + sysfs)
│   ├── cpu_tuning_profiles.cpp   # Built-in tuning database
│   ├── utils/                    # Timer, aligned alloc
│   ├── conv/                     # [NEW v0.4.0]
│   │   ├── conv2d.cpp            # Conv dispatch: 1×1 direct + im2col+GEMM
│   │   ├── im2col.cpp            # NHWC im2col with NEON acceleration
│   │   └── conv_postops.cpp      # Fused bias + ReLU/ReLU6 (NEON)
│   └── gemm/
│       ├── gemm.cpp              # Top-level dispatch
│       ├── gemm_autotune.cpp     # Auto-tuning engine
│       ├── gemm_driver_generic.cpp # BLIS 5-loop with OpenMP 2D threading
│       ├── gemm_ukernel_registry.cpp
│       ├── gemm_ukernel_fp32_neon.cpp  # NEON FP32 8x12
│       ├── gemm_ukernel_bf16_neon.cpp  # BFMMLA BF16 8x8
│       ├── gemm_ukernel_int8_neon.cpp  # SMMLA INT8 8x8
│       ├── gemm_ukernel_fp32_sve.cpp   # SVE FP32 VLA
│       ├── gemm_ukernel_bf16_sve.cpp   # SVE BF16 VLA
│       ├── gemm_ukernel_int8_sve.cpp   # SVE INT8 VLA
│       ├── gemm_ukernel_fp32_sme.cpp   # SME FP32 FMOPA
│       ├── gemm_ukernel_bf16_sme.cpp   # SME BF16 BFMOPA
│       ├── gemm_ukernel_int8_sme.cpp   # SME INT8 SMOPA
│       ├── gemm_pack_fp32.cpp
│       ├── gemm_pack_bf16.cpp
│       ├── gemm_pack_int8.cpp
│       └── gemm_smallm_fp32.cpp  # Small-M optimized path
├── tests/                        # Correctness tests (105 cases)
├── benchmarks/                   # Performance benchmarks + hwcaps report
└── docs/                         # Design documentation
```

## Development Log

### v0.4.0 — Phase 4: Convolution Operators (2026-04-07)

New: Conv2D operator with im2col + optimized GEMM, achieving up to 17.7x speedup over naive.

- **Public Conv2D API**: `conv2d_fp32()` with `Conv2DParams` and `ConvPostOp` enum.
  NHWC data layout throughout (input, filter, output).
- **im2col + optimized GEMM**: NHWC im2col rearranges input patches into column matrix,
  then dispatches to the Phase 2/3 optimized GEMM (NEON/SVE/multi-threaded).
  Filter transposed once [OC, K] → [K, OC] for GEMM B convention.
- **Direct 1×1 convolution**: Fast path for 1×1 stride=1 pad=0 — no im2col needed.
  Input [N*H*W, IC] is already in GEMM-ready NHWC layout. Up to 57 GFLOPS.
- **Fused post-ops**: Bias, ReLU, ReLU6, BiasRelu applied in-place after GEMM.
  NEON vectorized (4 channels/iteration) with scalar tail.
- **NEON im2col**: Contiguous channel copies accelerated with `vld1q_f32`/`vst1q_f32`,
  zero-fill for padded regions via `memset`.
- **31 correctness tests**: 10 im2col_naive vs ref + 10 optimized vs ref + 4 bias +
  4 bias+relu + 3 relu6 tests, all passing.
- **13-shape benchmark suite**: ResNet-50 + MobileNet layers with naive/im2col/dnnopt comparison.

### v0.3.3 — Phase 3C: Advanced Multi-threading (2026-04-07)

New: 2D thread decomposition, big.LITTLE awareness, and memory optimization.

- **2D M×N thread decomposition**: Replaces M-only parallelism. Thread team factored
  into (mt, nt) with shape-aware bias: tall-skinny → more mt, short-wide → more nt.
- **Core topology detection**: Reads per-CPU max frequency from sysfs, clusters cores
  by frequency, detects big.LITTLE heterogeneous configurations (>30% freq delta).
- **Big.LITTLE scheduling**: Performance cores used first via `pthread_setaffinity_np`.
- **Huge page allocation**: `MAP_HUGETLB` for packed B buffers >2MB with transparent
  fallback to `mmap` + `MADV_HUGEPAGE`, then `posix_memalign`.

### v0.3.2 — Phase 3B+/3D: SVE VLA + SME Framework (2026-04-07)

New: Optimized SVE-256/512 VLA kernels and complete SME FMOPA/BFMOPA/SMOPA framework.

- **SVE FP32/BF16/INT8 VLA kernel rewrite**: Register-resident accumulators, col-pair chunking
- **SME FP32 FMOPA + BF16 BFMOPA + INT8 SMOPA**: Complete ZA tile management with streaming mode

### v0.3.1 — Phase 3B: SVE/SVE2 Full Optimization (2026-04-07)

- SVE-128 FP32 8x12 microkernel with predicates and software prefetch
- SVE-128 BF16/INT8 wrappers with SVE packing
- SVE-accelerated INT8 quantization

### v0.3.0 — Phase 3A: Hardware-Adaptive Tuning Infrastructure (2026-04-07)

- CpuTuningProfile system for 11 ARM CPU families
- Shape-aware blocking with per-class cache utilization
- Runtime auto-tuning for unknown CPUs

### v0.2.x — Phase 2: GEMM Microkernels (2026-04-06~07)

- FP32 93% / BF16 86.5% / INT8 70.8% peak
- BLIS-style cache blocking, small-M path, generic driver + registry

### v0.1.0 — Phase 1: Infrastructure (2026-04-06)

- Build system, hwcaps detection, benchmark + test frameworks

## Roadmap

### Completed
- Phase 1: Infrastructure
- Phase 2: GEMM Microkernels (FP32/BF16/INT8)
- Phase 3: Hardware Adaptation (Tuning, SVE VLA, SME, Multi-threading)
- Phase 4: Convolution Operators

### Future
- Winograd F(2x2,3x3) / F(4x4,3x3) for additional conv speedup
- Depthwise separable convolution
- Pooling and elementwise operator optimization
- End-to-end model inference pipeline

## References

- [autoGEMM: Pushing the Limits of Irregular Matrix Multiplication on Arm Architectures (SC'24)](https://github.com/wudu98/autoGEMM)
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://github.com/flame/blis)
- [ARM Neoverse N2 Technical Reference Manual](https://developer.arm.com/documentation/102099/latest)

## License

MIT
