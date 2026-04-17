# ACL GEMM 集成方案分析

## 背景

oneDNN aarch64 没有 `gemm_driver`（只有 x64/ppc64 有），导致大矩阵 fallback 到 `ref_gemm` 性能很差。需要集成 ACL (Arm Compute Library) 的大矩阵 GEMM 作为 fallback。

## 现状分析

### oneDNN 已有 ACL 集成

oneDNN 已支持 `DNNL_AARCH64_USE_ACL=ON`，集成包括：
- `acl_inner_product.hpp` - 使用 ACL CpuFullyConnected (matmul)
- `acl_gemm_convolution.hpp` - 使用 ACL CpuGemmConv2d
- 其他：pooling, eltwise, batchnorm 等

**问题**: 当前 `gemm.cpp` 的 `extended_sgemm` 不使用 ACL，即使启用 ACL 也走不到。

### ACL 的 GEMM 层次

```
ACL 库结构：
├── arm_compute/runtime/NEON/NEFunctions.h  ← 高层 API (NEGEMM)
│   └── NEGEMM::configure(src0, src1, dst, alpha, beta)
│       └── 需要 Tensor 对象，有内存管理开销
│
└── src/cpu/kernels/assembly/  ← 底层 kernel (arm_gemm)
    ├── arm_gemm.hpp           ← kernel registry
    ├── gemm_common.hpp        ← GemmCommon 基类
    ├── gemm_interleaved.hpp   ← interleaved 算法
    └── kernels/
        ├── a64_sgemm_8x12.hpp ← 8x12 microkernel
        ├── a64_sgemm_8x6.hpp
        ├── a64_hybrid_fp32_mla_6x16.hpp  ← hybrid kernel
        └── ...
```

## 方案对比

| 方案 | 集成复杂度 | 依赖大小 | 性能 | 维护成本 |
|------|-----------|---------|------|---------|
| A: 完整 ACL | 低 | ~10MB | 高 | 低 |
| B: arm_gemm 模块 | 中 | ~1MB | 高 | 中 |
| C: 复制关键 kernel | 高 | ~100KB | 高 | 高 |
| D: 优化 dnnopt 大矩阵 | 中 | 0 | 中 | 低 |

### 方案 A: 完整 ACL 集成 (推荐)

**方式**: 启用 oneDNN 的 `DNNL_AARCH64_USE_ACL=ON`

**修改点**:
1. 在 `gemm.cpp` 添加 ACL gemm fallback：
```cpp
#if DNNL_AARCH64 && defined(DNNL_AARCH64_USE_ACL)
    // Large M: use ACL's gemm
    if (*M >= 64) {
        return acl_gemm_sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
#endif
#if DNNL_AARCH64 && defined(DNNL_USE_DNNOPT)
    // Small M: use dnnopt
    {
        auto status = aarch64::dnnopt_sgemm(...);
        if (status != status::unimplemented) return status;
    }
#endif
```

2. 实现 `acl_gemm_sgemm()` wrapper，调用 ACL 的 NEGEMM

**优点**:
- oneDNN 已有 ACL infrastructure
- 性能最优 (ACL 经过高度优化)
- 维护成本低 (上游更新)

**缺点**:
- 需要完整 ACL 库依赖
- 编译复杂度增加

### 方案 B: arm_gemm 模块 (最小推荐)

**方式**: 只使用 ACL 的 arm_gemm kernel 模块

**arm_gemm 特点**:
- 独立的 kernel registry
- 直接调用 `gemm_fp32.cpp` 中的方法
- 不需要完整 ACL runtime

**修改点**:
1. 复制 arm_gemm 相关文件到 oneDNN
2. 实现 wrapper 调用 arm_gemm

**依赖文件** (约 50 个):
```
arm_gemm.hpp, gemm_common.hpp, gemm_interleaved.hpp
gemm_fp32.cpp, kernels/a64_sgemm_8x12.hpp, ...
cpuinfo.cpp (CPU 特性检测)
```

**优点**:
- 比完整 ACL 轻量 (~1MB vs ~10MB)
- 直接 kernel 调用，效率高

**缺点**:
- 需要手动集成 kernel registry
- CPUInfo 需要适配

### 方案 C: 复制关键 kernel

**方式**: 直接复制 ACL 的 microkernel 实现到 dnnopt

**文件**:
- `kernels/a64_sgemm_8x12.hpp` (核心 8x12 kernel)
- 相关 transpose/pack 函数

**优点**:
- 最小集成 (~100KB)
- 完全自主可控

**缺点**:
- 维护成本高 (需跟进 ACL 更新)
- License 需注意 (MIT)
- 缺少 kernel selection logic

### 方案 D: 优化 dnnopt 大矩阵 (当前可行)

**方式**: 在 dnnopt 中优化 M≥64 的场景

**修改点**:
1. 改进 dnnopt 的 cache blocking 策略
2. 针对大矩阵使用更大的 tile size

**优点**:
- 无外部依赖
- 与项目目标一致

**缺点**:
- 需要大量调优工作
- 可能无法达到 ACL 水平

## 建议方案

**短期**: 方案 D - 优化 dnnopt 大矩阵 kernel
- 当前已无条件 dispatch 到 dnnopt
- 可以先优化 dnnopt 的 M≥64 性能

**中期**: 方案 A - 完整 ACL 集成
- 用户如果已有 ACL 环境可以直接启用
- oneDNN 已有 ACL infrastructure，改动最小

**关键**: 在 gemm.cpp 中添加形状条件：
```cpp
#if DNNL_AARCH64 && defined(DNNL_AARCH64_USE_ACL)
    // Large M: ACL gemm (best performance)
    if (*M >= 64) {
        return acl_sgemm_wrapper(...);
    }
#endif
#if DNNL_AARCH64 && defined(DNNL_USE_DNNOPT)
    // Small M: dnnopt (optimized for batch inference)
    {
        auto status = aarch64::dnnopt_sgemm(...);
        if (status != status::unimplemented) return status;
    }
#endif
    return ref_gemm<float>(...);
```

## 下一步行动

1. **验证 ACL 是否可用**: 检查 TensorFlow bazel 缓存中的 ACL 是否可链接
2. **实现 acl_sgemm_wrapper**: 简化版 wrapper 调用 ACL NEGEMM
3. **测试性能**: 验证 ACL 大矩阵性能
4. **条件 dispatch**: 根据形状选择 dnnopt/ACL