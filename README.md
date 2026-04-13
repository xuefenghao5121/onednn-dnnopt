# DNN-Opt: oneDNN Supplementary Patch for ARM Inference

**Version: 0.9.13** | **Status: Production-ready patch for oneDNN**

ARM 平台推理优化库，以 **oneDNN 补丁** 的形式提供，专注于补齐 oneDNN 在小型矩阵和奇异形状上的性能弱点。无需替换 oneDNN，只需 patch 即可自动获得加速。

## Design Philosophy

dnnopt **不是** oneDNN 的替代品，而是 **supplementary patch**：

- oneDNN 在大型规则矩阵 (M>=64, N/K 对齐) 上已经接近峰值性能，无需优化
- 真正的弱点在 **M=1~7 的小矩阵**、**非对齐 N（质数维度）**、**高瘦矩阵** 等形状
- dnnopt 通过 oneDNN 的 `dnnl_sgemm` 接口注入，只在弱点形状上激活，强项形状 fallback 给 oneDNN

## Performance

### oneDNN+dnnopt vs oneDNN-native (Neoverse N2, 2 cores @ 3GHz)

**55 shapes tested: 54 wins / 1 loss**

| 类别 | 示例形状 | 加速倍数 |
|------|---------|---------|
| M=1 GEMV | 推理 batch=1 | **4~89x** |
| M=2-7 小矩阵 | 小 batch 推理 | **3~190x** |
| 高瘦矩阵 | M=128, N=2~7 | **2.1~4.2x** |
| 质数/不规则 N | N=17,37,53 | **1.5~2.6x** |

### 推理工作负载端到端

| 模型 | oneDNN | +dnnopt | 加速 |
|------|--------|---------|------|
| CVR model batch=1 | 691 us | 52 us | **13.3x** |
| CVR model batch=4 | 1861 us | 85 us | **21.9x** |
| BERT-small batch=1 | 7698 us | 1145 us | **6.7x** |
| BERT-small batch=4 | 27215 us | 2365 us | **11.5x** |
| LLM inference batch=1 | 260927 us | 44016 us | **5.9x** |
| LLM inference batch=4 | 531784 us | 148018 us | **3.6x** |

### FP32 Peak Performance (Large Regular Shapes)

| Shape | dnnopt GFLOPS | Peak % |
|-------|--------------|--------|
| 512×512×512 | 44.6 | 93% |
| 1024×1024×1024 | 45.1 | 94% |
| 2048×2048×2048 | 45.2 | 94% |

## Key Techniques

### Small / Irregular Matrix Optimizations
- **Clang `.s[N]` fused FMLA**: `vfmaq_laneq_f32` → 单条 `fmla v.4s, v.4s, v.s[N]` 指令
- **npo2 内核**: M=3,5,7 专用内核，向量 A 加载 + `.s[0..3]` 提取
- **Tall-skinny 内核**: N=2~7 模板化内核，4x K-unrolling，无 packing 开销
- **Small-M wide driver**: 48 列 macro-tiling，M=2~7 专用路径
- **Tiny shape kernels**: M×1 GEMV, M,N≤8 微型矩阵

### General GEMM Optimizations
- **autoGEMM-style tile selection**: 形状评分自动选择最优 (Mr, Nr) 组合
- **Per-CPU tuning profiles**: 11 个 ARM CPU 家族内置调优参数
- **Shape-aware blocking**: 矩阵形状分类 + 自适应 cache 利用率
- **OpenMP 2D threading**: M×N 并行分解，大页分配
- **Kc blocking**: L1D 自适应 K 分块

### Multi-Precision Support
- **FP32**: NEON 8x16/6x16/4x16 + SVE VLA
- **BF16**: BFMMLA 8x8 (192 GFLOPS peak)
- **INT8**: SMMLA 8x8 (384 GOPS peak)
- **SME**: FMOPA/BFMOPA/SMOPA (opt-in)

## Integration

### as oneDNN Patch (Recommended)

```bash
# 1. Build dnnopt
cd onednn-arm-opt && mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 2. Patch oneDNN
cd /path/to/onednn && mkdir build && cd build
cmake .. -DDNNL_AARCH64_USE_DNNOPT=ON \
         -DCMAKE_PREFIX_PATH=/path/to/dnnopt/build
cmake --build . -j$(nproc)

# 3. Use patched oneDNN — zero code changes needed
LD_LIBRARY_PATH=./src/libdnnl.so python your_model.py
```

### as BLAS Drop-in Replacement

```bash
# LD_PRELOAD for any BLAS consumer
LD_PRELOAD=./src/libdnnopt_blas.so python your_model.py
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `DNNOPT_NATIVE_ARCH` | ON | `-march=native` |
| `DNNOPT_USE_OPENMP` | ON | OpenMP 多线程 |
| `DNNOPT_ENABLE_SME` | OFF | SME 内核编译 |
| `DNNOPT_BUILD_TESTS` | ON | 正确性测试 |
| `DNNOPT_BUILD_BLAS` | ON | BLAS 兼容库 |

## Project Structure

```
onednn-arm-opt/
├── include/dnnopt/
│   ├── arm_hwcaps.h              # CPU 能力检测
│   ├── cpu_tuning_profile.h      # Per-CPU 调优参数
│   ├── gemm/
│   │   ├── gemm.h                # 公共 GEMM API
│   │   ├── gemm_config.h         # 自适应 cache blocking + tile 配置
│   │   ├── gemm_types.h          # 类型定义
│   │   └── gemm_ukernel_registry.h
│   └── blas/cblas.h              # CBLAS 标准接口
├── src/
│   ├── hwcaps/arm_hwcaps.cpp
│   ├── cpu_tuning_profiles.cpp
│   ├── gemm/
│   │   ├── gemm.cpp              # 顶层 dispatch（形状分类路由）
│   │   ├── gemm_tiny_fp32.cpp    # M×1 GEMV + M,N≤8 + tall-skinny N=2~7
│   │   ├── gemm_smallm_fp32.cpp  # M=1~7 专用路径
│   │   ├── gemm_smallm_wide_fp32.cpp  # 48 列 macro-tiling
│   │   ├── gemm_adaptive_tile_fp32.cpp # autoGEMM 动态 tile
│   │   ├── gemm_ukernel_fp32_8x16.cpp  # Clang .s[N] 8x16 packed
│   │   ├── gemm_ukernel_fp32_npo2.cpp  # Clang .s[N] M=3,5,7 npo2
│   │   ├── gemm_ukernel_fp32_asm.cpp   # 内联汇编内核
│   │   ├── gemm_driver_generic.cpp     # BLIS 5-loop + OpenMP
│   │   ├── gemm_threading.cpp          # 2D 线程分解
│   │   └── ... (BF16, INT8, SVE, SME)
│   ├── conv/
│   │   ├── conv2d.cpp            # Conv2D: im2col + GEMM
│   │   └── im2col.cpp
│   └── blas/                     # CBLAS/BLAS 接口
├── tests/
│   ├── test_gemm_correctness.cpp # 74 正确性测试
│   ├── bench_onednn_sgemm.cpp    # oneDNN 对比 benchmark
│   └── bench_inference_workload.cpp # 推理工作负载 benchmark
└── benchmarks/
    └── bench_gemm.cpp            # 峰值性能 benchmark
```

## Supported Platforms

| CPU | Arch | Vector | ML Features |
|-----|------|--------|-------------|
| Neoverse N1/N2 | ARMv8.2/v9.0 | NEON/SVE2 | BF16, I8MM |
| Neoverse V1/V2 | ARMv8.4/v9.0 | SVE/SVE2 | BF16, SME |
| Cortex-A78/X2/X3 | ARMv8.2/v9.0 | NEON/SVE2 | BF16, I8MM |
| Kunpeng 920 | ARMv8.2 | NEON | DotProd |
| A64FX | ARMv8.2-SVE | 512-bit SVE | HBM2 |

## Development Log

### v0.9.13 — Phase 13I+C: M=6 Packed Path + Vectorized N-Tail (2026-04-13)

M=6 大形状 packed 路径 + 不规则 N-tail 全面向量化。

- **Packed 6x16 内核**: 注册到 kernel registry，M=6 形状可用 packed+threaded 路径
- **M=6 大形状路由**: N*K > 4M 时通过 8x16 M-padding 走 packed 路径（cache-friendly B 访问）
- **N-tail 向量化**: gemm_tile_tail 全面向量化，消除 scalar K-loop fallback
- **edge_buf 优化**: 移除 generic driver 中不必要的 memset
- **结果**: 54/1 wins vs oneDNN, M16_N23 1.28→1.68x, M32_N47 1.63→1.95x

### v0.9.12 — Phase 13H: Small/Irregular Matrix Optimization (2026-04-13)

Clang 编译器迁移 + 小型奇异矩阵全面优化，作为 oneDNN 补丁。

- **Clang-15 编译器**: 启用 `.s[N]` 融合 FMLA 指令
- **npo2 内核**: M=3,5,7 专用 `.s[N]` 内核（12/20/28 FMLAs/K）
- **Tall-skinny 内核**: N=2~7 模板化内核，M 任意值
- **OpenMP N-parallelism**: adaptive tile 大形状自动 N 维并行
- **oneDNN patch 集成**: 通过 `dnnl_sgemm` 注入，54/1 wins vs oneDNN-native
- **推理工作负载**: CVR 13~22x, BERT 7~12x, LLM 4~6x 加速

### v0.9.0 — Phase 12: autoGEMM Integration (2026-04-12)

- 6x16 2x K-unrolling, prefetch 优化, 8x16 packed kernel
- Batch GEMM dispatch: M=4-7 大 N*K 走 packed+threaded path

### v0.8.0 — Phase 8-11: ASM Kernels + Kc Blocking (2026-04-11)

- 内联汇编 4x16/6x16 内核, autoGEMM 动态 tile 选择
- Kc blocking 优化小型 shape, vs oneDNN 35/17 wins

### v0.5.0 — Phase 5A: CBLAS/BLAS Interface (2026-04-07)

- Drop-in BLAS 替换, LD_PRELOAD 支持
- vs OpenBLAS 1.5~1.6x on large matrices

### v0.4.0 — Phase 4: Convolution (2026-04-07)

- Conv2D: im2col + GEMM, up to 17.7x speedup over naive

### v0.3.0 — Phase 3: Hardware Adaptation (2026-04-07)

- Per-CPU tuning, SVE/SME, 2D threading, huge pages

### v0.2.0 — Phase 2: GEMM Microkernels (2026-04-06)

- FP32 93% / BF16 86.5% / INT8 70.8% peak

### v0.1.0 — Phase 1: Infrastructure (2026-04-06)

- Build system, hwcaps, test + benchmark frameworks

## References

- [autoGEMM: Pushing the Limits of Irregular Matrix Multiplication on Arm Architectures (SC'24)](https://github.com/wudu98/autoGEMM)
- [IAAT: Input-Adaptive Auto-Tuning for Small GEMM](https://arxiv.org/abs/2405.05636)
- [LibShalom: Shape-aware GEMM optimization](https://github.com/IsShalom/LibShalom)
- [LBBGEMM: Load-Balanced Batch GEMM](https://github.com/MuspiMerol/LBBGEMM)
- [BLIS: Framework for Rapidly Instantiating BLAS Functionality](https://github.com/flame/blis)

## License

MIT
