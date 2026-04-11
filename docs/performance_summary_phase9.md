# DNN-Opt v0.9.7 性能总结 & 多 ISA 优化路线图

## 测试环境

| 项目 | 值 |
|------|-----|
| CPU | Neoverse N2 (0xd49), SVE2 128-bit |
| 频率 | 3.0 GHz, 2 cores |
| FP32 Peak | 48 GFLOPS/core (2 FMLA × 4 FP32 × 3 GHz) |
| 编译器 | GCC 10.2.1, -O3 -march=native |
| OpenBLAS | 系统版本 (generic ARMV8 kernel) |
| oneDNN | 3.x (libdnnl.so.3, Neoverse N2 native) |

---

## 一、Phase 9 核心成果：Assembly Kernel

### 微内核性能

| Kernel | 技术 | GFLOPS | % Peak | vs Intrinsics |
|--------|------|--------|--------|---------------|
| 4x16 inline asm | 8x K-unroll, ping-pong B, .s[0-3] | **45.03** | **93.8%** | **1.58x** |
| 6x16 inline asm | scalar A, 24 accumulators | ~38-40 | ~80% | ~1.05x |
| 5x16 inline asm | scalar A, 20 accumulators | ~35-38 | ~75% | 新增 |
| 3x16 inline asm | scalar A, 12 accumulators | ~30-35 | ~65% | 新增 |
| 4x16 intrinsics (旧) | 2x K-unroll | 28.44 | 59.3% | baseline |
| autoGEMM 参考 (4×64×16) | shape-specific asm | 47.08 | 98.1% | — |

**关键技术突破**:
1. **8x K-unrolling via .s[0..3]**: 一次 `ldr q` 加载 4 个 K 值，通过标量索引访问
2. **Ping-pong B pointers**: x22/x23 交替加载 B 行，隐藏内存延迟
3. **显式寄存器分配**: v0-v7 A 值, v8-v11 B panel, v12-v27 累加器
4. **Alpha/Beta 快速路径**: beta=0 时 fmul+str，beta!=0 时 ldr+fmla+str

---

## 二、三路性能对比 (52 shapes)

### 总体结果

| 指标 | dnnopt vs OpenBLAS | dnnopt vs oneDNN |
|------|-------------------|------------------|
| 总耗时 | 484ms vs 481ms | 484ms vs 483ms |
| Overall | **0.99x** | **1.00x** (持平) |
| dnnopt 获胜 | 17 shapes | 31 shapes |
| 对手获胜 | 35 shapes | 21 shapes |
| 最佳加速 | 1.56x (npo2-5x64) | 1.34x (tall-256x8) |
| 最差 | 0.49x (batch4-LLM) | 0.94x (batch1-FC/LLM) |

### 分类详细对比

#### Attention Shapes (推理关键路径)

| Shape (M×N, K=64/128) | dnnopt GFLOPS | vs OB | vs oneDNN |
|------------------------|---------------|-------|-----------|
| attn-128×64 | 40.39 | 0.97x | **1.00x** |
| attn-256×64 | 41.81 | 1.00x | **1.00x** |
| attn-512×64 | **43.64** | **1.03x** | **1.00x** |
| attn-64×128 | 34.79 | 0.84x | 1.01x |
| attn-128×128 | 40.47 | 0.95x | **1.00x** |
| attn-256×128 | 42.04 | 0.97x | **1.00x** |

**结论**: Attention shapes 上 dnnopt = oneDNN，均达 83-91% peak。

#### Large/Square Matrices (计算密集)

| Shape | dnnopt GFLOPS | vs OB | vs oneDNN |
|-------|---------------|-------|-----------|
| square-384 | 43.49 | **1.04x** | **1.00x** |
| square-512 | 43.94 | **1.01x** | **1.00x** |
| large-1024 | 44.13 | **1.01x** | 0.99x |
| large-2048 | 44.59 | **1.01x** | **1.00x** |

**结论**: 大矩阵均达 91-93% peak，三者基本持平。

#### Batch Inference (小 M, 大 N×K)

| Shape | dnnopt GFLOPS | vs OB | vs oneDNN |
|-------|---------------|-------|-----------|
| batch1-FC (1×1000×2048) | 12.44 | **1.52x** | 0.94x |
| batch2-FC (2×1000×2048) | 19.10 | **1.27x** | 1.01x |
| batch4-FC (4×1000×2048) | 12.31 | 0.51x | 0.97x |
| batch8-FC (8×1000×2048) | 20.63 | 0.66x | 0.95x |
| batch16-FC (16×1000×2048) | 28.57 | 0.78x | 0.98x |
| batch1-LLM (1×4096×4096) | 8.94 | **1.37x** | 0.94x |
| batch2-LLM (2×4096×4096) | 10.07 | 0.83x | 0.94x |
| batch4-LLM (4×4096×4096) | 9.76 | **0.49x** | 0.99x |

**结论**: batch inference 是主要差距来源。
- batch1/2: dnnopt > OpenBLAS (memory-bound GEMV 优化好)
- batch4-16: OpenBLAS > dnnopt (packing 路径更优)
- vs oneDNN: 基本持平 (0.94-1.01x)

#### Conv Shapes (im2col GEMM)

| Shape | dnnopt GFLOPS | vs OB | vs oneDNN |
|-------|---------------|-------|-----------|
| conv-MNIST-1 | 41.61 | **1.14x** | **1.00x** |
| conv-MNIST-2 | 42.50 | **1.03x** | **1.00x** |
| conv-ResNet-1 | 30.18 | 0.77x | **1.00x** |
| conv-ResNet-2 | 40.23 | 0.95x | **1.00x** |
| conv-ResNet-3 | 40.20 | 0.94x | **1.00x** |
| conv-ResNet-4 | 33.71 | 0.83x | **1.00x** |

**结论**: Conv shapes 与 oneDNN 完全持平。OpenBLAS 在 ResNet 形状上更快。

#### NPO2 (非 2 的幂 M)

| Shape | dnnopt GFLOPS | vs OB | vs oneDNN |
|-------|---------------|-------|-----------|
| npo2-5x64 | **40.96** | **1.56x** | 1.02x |
| npo2-23x64 | **43.21** | **1.16x** | **1.00x** |
| npo2-47x64 | **44.26** | **1.11x** | **1.00x** |
| npo2-63x64 | **44.57** | **1.11x** | **1.00x** |
| npo2-7x64 | 21.72 | 0.74x | **1.00x** |
| npo2-17x64 | 23.37 | 0.65x | **1.00x** |

**结论**: Phase 9 asm kernels 显著改善了 npo2-5x64 (M=5 走 5x16 asm)。
M=7/11/13/17 仍有差距 — M tail + 无 packing。

#### Ksmall (K ≤ 16)

| Shape | dnnopt GFLOPS | vs OB | vs oneDNN |
|-------|---------------|-------|-----------|
| Ksmall-64x64x4 | 18.41 | 0.91x | 1.01x |
| Ksmall-64x64x8 | 23.07 | 0.80x | 1.01x |
| Ksmall-64x64x12 | 23.74 | 0.72x | **1.00x** |
| Ksmall-128x128x4 | 19.11 | 0.87x | **1.00x** |
| Ksmall-128x128x8 | 23.00 | 0.74x | 0.99x |

**结论**: Ksmall 与 oneDNN 持平，但 vs OpenBLAS 有差距 (0.72-0.91x)。

---

## 三、关键差距分析

### dnnopt 主要弱项 (vs OpenBLAS)

| 类别 | 代表形状 | 差距 | 根因 |
|------|----------|------|------|
| **Batch large-K** | batch4-LLM (0.49x) | 严重 | M=4, K=4096: unpacked path 无法利用 L2 cache |
| **Batch FC** | batch4/8/16-FC (0.51-0.78x) | 中等 | packed BLIS 路径 tile 选择不优 |
| **Wide M=16** | wide-16x256/512 (0.72x) | 中等 | M=16 走 adaptive tile 但 N blocking 不够 |
| **Ksmall** | Ksmall-*x*x12 (0.72x) | 中等 | smallK path 未充分 SIMD 化 |
| **NPO2 odd-M** | npo2-7/17 (0.65-0.74x) | 中等 | M tail 处理效率低 |
| **Conv ResNet** | conv-ResNet-1 (0.77x) | 中等 | im2col 开销 + 小形状 |

### dnnopt vs oneDNN: 基本持平

Phase 9 后 dnnopt vs oneDNN = **1.00x**。两者使用相同底层技术，差异在 dispatch/edge handling 层面。dnnopt 在 tall-N=8 形状上快 25-34%，oneDNN 在 batch1 形状上快 5-6%。

---

## 四、NEON 优化路线图 (Phase 10)

### 10.1 Batch/Large-K 形状优化 (预期: batch4-LLM 0.49x → 0.85x+)

**问题**: M=4, K=4096 走 unpacked adaptive tile path，K-loop 过程中 B 数据反复从内存加载。

**方案**: 
1. **B-panel streaming**: 对 K>512 的情况，分 Kc 块处理，每块内 B 数据热在 L1
2. **Skinny packed path**: 专门的 M<8 packed path，只 pack B，不 pack A (A 只有几行)
3. **Registry fallback 调优**: 调整 `kUnpackedFlopsThreshold` 让 batch shapes 走 packed registry

### 10.2 6x16 Assembly Kernel K-Unrolling (预期: 80% → 88-90% peak)

**问题**: 6x16 asm 当前用 scalar A load (1 K/iter)，无法利用 `.s[0-3]` 技巧。

**方案**:
1. **2x K-unroll**: 每次加载 2 个 A scalar，2 组 B panel，24 acc + 4 B + 4 B' + 6 A = 38 > 32 不行
2. **替代方案**: 将 6x16 改为 **6x12** (18 acc) + padding，给 K-unroll 腾出寄存器
3. **或**: 使用 4x16 asm (93.8% peak) 处理更多形状，减少 6x16 使用场景

### 10.3 Wide/Ksmall 优化 (预期: 0.72x → 0.90x+)

**方案**:
1. **Ksmall**: 为 K=4/8/12 写 fully-unrolled asm kernel，零 K-loop 开销
2. **Wide M=16**: 调整 dispatch 让 M=16 走 packed registry path (packing 开销可摊薄)

### 10.4 M-Tail 优化 (预期: npo2 odd-M 0.65x → 0.85x+)

**方案**:
1. **组合 tile dispatch**: M=17 → 一个 6x16 + 一个 5x16 + 一个 6x16 (而非统一 tile)
2. **M-remainder asm kernel**: 针对 M%Mr 的尾部写专用 asm

---

## 五、SVE 优化路线图 (Phase 11)

### 目标平台

| 平台 | SVE VL | 预期收益 | 代表 CPU |
|------|--------|----------|----------|
| Neoverse N2 | 128-bit | 0% (= NEON) | 当前开发机 |
| Neoverse V1 | 256-bit | **~1.8x** | AWS Graviton3 |
| Neoverse V2 | 128-bit | 0% (= NEON) | AWS Graviton4 |
| Neoverse V3 | 128-bit | 0% (= NEON) | 未发布 |
| Fujitsu A64FX | 512-bit | **~3.5x** | 富岳超算 |

**关键洞察**: SVE 仅在 VL > 128-bit 平台有显著收益。V1 (256-bit) 和 A64FX (512-bit) 是主要目标。

### 11.1 SVE VLA (Vector-Length Agnostic) Assembly Kernels

**现状**: 已有 SVE intrinsics 8×(2*VL) kernel，但无 asm 版本。

**方案**:
```
Tile: Mr × Nr
  VL=128: 6×8  → 48 元素/tile  (= NEON 6x8, 不如 6x16)
  VL=256: 6×16 → 96 元素/tile  (= NEON 6x16, 但更少指令)
  VL=512: 6×32 → 192 元素/tile (2x NEON 6x16 throughput)
```

**实现**:
1. **VLA asm kernel**: 使用 SVE 谓词 (predicate) 代替固定 Nr
   - `ptrue p0.s, VL4` (128-bit), `ptrue p0.s, VL8` (256-bit)
   - `whilelt p1.s, cnt, N` 处理 N-tail
   - Nr = svcntw() (runtime: 4/8/16 for 128/256/512)
2. **Register plan for VL=256**:
   - 6 rows × 2 Z-vectors = 12 accumulators (z10-z21)
   - 2 B vectors (z0-z1)
   - 6 A broadcast (z2-z7)
   - 8 spare for K-unrolling
3. **K-unrolling**: VL=256 时 16 acc + 4 B + 4 A = 24 regs → 4x K-unroll 可行

### 11.2 SVE Predicated Edge Handling

**NEON 问题**: M/N tail 需要标量循环或多个 tile kernel。

**SVE 优势**: 谓词寄存器天然处理 tail:
```asm
whilelt p0.s, j, N    ; 生成 mask: 1s for j..min(j+VL,N)
ld1w z0.s, p0/z, [B]  ; 只加载有效元素
fmla z8.s, p0/m, z0.s, z1.s[0]  ; 只计算有效列
st1w z8.s, p0, [C]    ; 只存有效元素
```
这消除了 N-tail 的特殊处理，简化 dispatch 并提升 NPO2 性能。

### 11.3 SVE Gather/Scatter for Strided Access

对非连续内存 (im2col, transposed):
```asm
ld1w z0.s, p0/z, [base, offsets.s, UXTW #2]  ; gather load
```
可用于 conv im2col 消除显式 im2col 缓冲区。

---

## 六、SVE2 优化路线图 (Phase 12)

### SVE2 新增指令 (vs SVE)

| 指令类别 | 代表指令 | GEMM 相关性 |
|----------|----------|-------------|
| 整数乘累加 | SMLAL, UMLAL | INT8 GEMM 吞吐翻倍 |
| 复数运算 | CMLA, SQRDCMLAH | FFT-based conv |
| 位操作 | MATCH, NBSL | 量化/打包优化 |
| 饱和算术 | SQADD, SQSUB | INT8 后处理 |
| Histogram | HISTCNT, HISTSEG | Sparse GEMM |
| XAR (旋转异或) | XAR | 无直接 GEMM 用途 |

### 12.1 SVE2 INT8 GEMM 优化

**目标**: 利用 SVE2 SMLALB/SMLALT (signed multiply-add long bottom/top)
- 将 INT8×INT8→INT32 累加吞吐提升 2x vs 基础 SVE SDOT
- 与 I8MM (SMMLA) 互补：I8MM 处理 2×8 tile，SVE2 SMLAL 处理 edge cases

**实现**:
```asm
; SVE2 INT8 micro-kernel inner loop
ld1b z0.b, p0/z, [A_ptr]     ; A: 8-bit
ld1b z1.b, p0/z, [B_ptr]     ; B: 8-bit  
smlalb z8.s, z0.h, z1.h      ; bottom half: a[0]*b[0], a[2]*b[2], ...
smlalt z9.s, z0.h, z1.h      ; top half:   a[1]*b[1], a[3]*b[3], ...
```

### 12.2 SVE2 BF16 GEMM 优化

**现状**: 当前 BF16 kernel 使用 BFMMLA (ARMv8.6-A)，在 SVE2 平台上可进一步优化:
- BFMLALB/BFMLALT: BF16×BF16→FP32 long multiply-add
- 与 BFMMLA 互补：BFMMLA 处理 2×4 tile，BFMLAL 处理 edge

### 12.3 SVE2 量化优化

- **TBL/TBX**: 高效 lookup table，用于非线性量化 (GPTQ, AWQ)
- **MATCH**: 快速稀疏矩阵元素匹配
- **NBSL**: 三输入位逻辑，用于混合精度 packing

---

## 七、SME 优化路线图 (Phase 13)

### SME 架构概述

| 特性 | 说明 |
|------|------|
| ZA 累加器 | SVL×SVL 二维 tile (SVL=256: 8×8 FP32, SVL=512: 16×16) |
| FMOPA | FP32 outer product: ZA += A_col × B_row |
| BFMOPA | BF16 outer product: ZA += A_col_bf16 × B_row_bf16 (FP32 累加) |
| SMOPA | INT8 outer product: ZA += A_col_i8 × B_row_i8 (INT32 累加) |
| Streaming SVE | SME 独占模式，SVE/NEON 寄存器不可用 |
| SMSTART/SMSTOP | 进入/退出 streaming 模式 |

### 目标平台

| 平台 | SME 版本 | SVL | ZA Tile | 代表 CPU |
|------|----------|-----|---------|----------|
| Neoverse V3 | SME2 | 128-bit | 4×4 FP32 | 未来 |
| Apple M4+ | SME2 | 512-bit | 16×16 FP32 | M4 Pro/Max |
| Cortex-X925 | SME2 | 128-bit | 4×4 FP32 | 旗舰手机 |

### 13.1 SME FP32 FMOPA Kernel

**现状**: 已有 compile-only 实现 (`gemm_ukernel_fp32_sme.cpp`)。

**优化方案**:
```asm
; SME FP32 outer product kernel
smstart sm              ; 进入 streaming 模式
zero {za}               ; 清零 ZA 累加器

; K-loop: 每次一个 outer product
.Lk_loop:
  ld1w {z0.s}, p0/z, [A_ptr]    ; 加载 A 列 (SVL 元素)
  ld1w {z1.s}, p1/z, [B_ptr]    ; 加载 B 行 (SVL 元素)
  fmopa za0.s, p0/m, p1/m, z0.s, z1.s  ; ZA += A_col × B_row^T
  add A_ptr, A_ptr, #(SVL*4)
  add B_ptr, B_ptr, #(SVL*4)
  subs K_cnt, K_cnt, #1
  b.ne .Lk_loop

; 提取结果到 NEON/SVE 寄存器
mov z2.s, p0/m, za0h.s[w12, 0]  ; 提取第 0 行
st1w {z2.s}, p0, [C_ptr]
; ... 提取所有行

smstop sm               ; 退出 streaming 模式
```

**关键优化点**:
1. **多 ZA tile**: SVL=512 时有 za0-za3 四个 16×16 tile，可做 16×64 blocking
2. **K-unrolling**: 多次 FMOPA 交错以隐藏 ld1w 延迟
3. **Streaming ↔ Non-streaming 切换成本**: SMSTART/SMSTOP ~100 cycles，需要批量处理

### 13.2 SME2 Multi-Vector Operations

**SME2 新增**:
- `FMOPA za.s, {z0.s-z1.s}, {z2.s-z3.s}`: 2-vector outer product
- `LD1W {z0.s-z3.s}, ...`: 4-vector 连续加载
- `MOVA {z0.s-z3.s}, za0h.s[w12, 0:3]`: 4 行同时提取

**性能预期**:
- SVL=128: ZA = 4×4 FP32, FMOPA 32 FLOPs/cycle → 与 NEON 持平
- SVL=256: ZA = 8×8 FP32, FMOPA 128 FLOPs/cycle → **2x NEON**
- SVL=512: ZA = 16×16 FP32, FMOPA 512 FLOPs/cycle → **8x NEON**

### 13.3 SME BF16 BFMOPA Kernel

**理论峰值**: BFMOPA 处理 BF16 输入 FP32 累加，吞吐 = 2x FMOPA:
- SVL=256: 256 BF16 FLOPs/cycle
- SVL=512: 1024 BF16 FLOPs/cycle

**实现**: 类似 FMOPA kernel，替换 `ld1w` → `ld1h` + `bfmopa`

### 13.4 SME INT8 SMOPA Kernel

**理论峰值**: SMOPA 处理 INT8 输入 INT32 累加，吞吐 = 4x FMOPA:
- SVL=256: 512 INT8 OPs/cycle
- SVL=512: 2048 INT8 OPs/cycle

---

## 八、实施优先级矩阵

### P0: 高价值 / 低风险 (立即实施)

| 任务 | ISA | 预期收益 | 复杂度 | 可验证平台 |
|------|-----|----------|--------|-----------|
| Batch large-K packed path | NEON | batch4-LLM 0.49x → 0.85x | 中 | Neoverse N2 |
| Ksmall fully-unrolled asm | NEON | Ksmall 0.72x → 0.90x | 低 | Neoverse N2 |
| M-tail 组合 dispatch | NEON | npo2 odd-M 0.65x → 0.85x | 低 | Neoverse N2 |

### P1: 高价值 / 中风险 (1-2 周)

| 任务 | ISA | 预期收益 | 复杂度 | 可验证平台 |
|------|-----|----------|--------|-----------|
| SVE VLA asm kernels | SVE | VL=256: ~1.8x peak | 高 | Graviton3 (V1) |
| SVE predicated tails | SVE | NPO2 +10-20% | 中 | Graviton3 |
| 6x16 asm K-unroll 或 6x12 替代 | NEON | 80% → 88% peak | 中 | Neoverse N2 |

### P2: 战略价值 / 高风险 (2-4 周)

| 任务 | ISA | 预期收益 | 复杂度 | 可验证平台 |
|------|-----|----------|--------|-----------|
| SME FMOPA kernel 优化 | SME | SVL=512: ~8x | 高 | Apple M4 / V3 |
| SVE2 INT8 SMLAL kernel | SVE2 | INT8 +50-100% | 中 | Graviton3+ |
| SVE2 BF16 BFMLAL kernel | SVE2 | BF16 edge +30% | 中 | Graviton3+ |

### P3: 前瞻性 (4+ 周)

| 任务 | ISA | 预期收益 | 复杂度 | 可验证平台 |
|------|-----|----------|--------|-----------|
| SME2 multi-vector ops | SME2 | SVL=512: ~16x | 很高 | V3 / M4+ |
| SVE gather for im2col | SVE | Conv -10% 延迟 | 中 | Graviton3 |
| Streaming SVE ↔ NEON 混合调度 | SME | 减少模式切换 | 高 | V3 / M4+ |

---

## 九、多平台验证计划

### Tier 1: 核心验证 (当前)
- **Neoverse N2** (SVE2 128-bit): NEON + SVE2 kernels
- 工具: `bench_small_vs_openblas`, `test_gemm_correctness`

### Tier 2: SVE 宽向量验证
- **Neoverse V1** (SVE 256-bit): SVE VLA kernels, 验证 VL=256 性能
- **平台**: AWS Graviton3 (c7g 实例)
- 预期: FMLA 吞吐 2x，总体 GFLOPS ~1.5-1.8x vs N2

### Tier 3: SME 验证
- **Apple M4** (SME2, SVL=512): SME FMOPA/BFMOPA/SMOPA kernels
- **Neoverse V3** (SME2, SVL=128): SME kernels (SVL 较小，收益有限)
- 预期: M4 上 FMOPA 应达 8x NEON FP32 throughput

### 每平台验证检查清单
1. `cmake .. && make -j$(nproc)` — 编译通过
2. `./tests/test_gemm_correctness` — 74/74 GEMM + 63/63 CBLAS 全通过
3. `./benchmarks/bench_small_vs_openblas` — 52 shapes 性能对比
4. `./benchmarks/bench_4x16_perf` — 微内核 peak% 验证
5. 记录 `hwcaps` 输出 (SVE VL, SME 支持等)

---

## 十、架构演进总览

```
Phase 9 (当前, DONE)
  └── NEON inline asm: 4x16 93.8% peak, 3/5/6x16 tails
  └── 52-shape: 1.00x vs oneDNN, 0.99x vs OpenBLAS

Phase 10 (NEON 收尾)
  ├── Batch large-K packed path → batch4-LLM 0.49x → 0.85x
  ├── Ksmall unrolled asm → 0.72x → 0.90x
  ├── M-tail combo dispatch → odd-M 0.65x → 0.85x
  └── 目标: 52-shape overall 1.10x+ vs OpenBLAS

Phase 11 (SVE)
  ├── VLA asm kernels (VL=256/512 自动适配)
  ├── Predicated edge handling (消除 tail kernels)
  └── 目标: Graviton3 上 1.5-1.8x vs NEON

Phase 12 (SVE2)
  ├── INT8 SMLALB/T kernels
  ├── BF16 BFMLALB/T edge kernels
  └── 目标: INT8/BF16 +50-100% throughput

Phase 13 (SME/SME2)
  ├── FMOPA/BFMOPA/SMOPA outer-product kernels
  ├── Multi-vector SME2 operations
  └── 目标: Apple M4 上 8x FP32, 16x BF16 throughput
```

---

## 附录: 完整 52-Shape 三路对比数据 (Phase 9)

*测试日期: 2026-04-11*

| Shape | dnnopt (ms) | OpenBLAS (ms) | oneDNN (ms) | vs OB | vs DNN | GFLOPS |
|-------|-------------|---------------|-------------|-------|--------|--------|
| batch1-FC | 0.329 | 0.501 | 0.310 | 1.52x | 0.94x | 12.44 |
| batch2-FC | 0.429 | 0.547 | 0.433 | 1.27x | 1.01x | 19.10 |
| batch4-FC | 1.331 | 0.681 | 1.288 | 0.51x | 0.97x | 12.31 |
| batch8-FC | 1.588 | 1.050 | 1.503 | 0.66x | 0.95x | 20.63 |
| batch16-FC | 2.294 | 1.791 | 2.244 | 0.78x | 0.98x | 28.57 |
| batch1-LLM | 3.753 | 5.156 | 3.520 | 1.37x | 0.94x | 8.94 |
| batch2-LLM | 6.661 | 5.526 | 6.247 | 0.83x | 0.94x | 10.07 |
| batch4-LLM | 13.752 | 6.739 | 13.567 | 0.49x | 0.99x | 9.76 |
| attn-128x64 | 0.026 | 0.025 | 0.026 | 0.97x | 1.00x | 40.39 |
| attn-256x64 | 0.050 | 0.050 | 0.050 | 1.00x | 1.00x | 41.81 |
| attn-512x64 | 0.096 | 0.099 | 0.096 | 1.03x | 1.00x | 43.64 |
| attn-64x128 | 0.030 | 0.025 | 0.030 | 0.84x | 1.01x | 34.79 |
| attn-128x128 | 0.052 | 0.049 | 0.052 | 0.95x | 1.00x | 40.47 |
| attn-256x128 | 0.100 | 0.097 | 0.100 | 0.97x | 1.00x | 42.04 |
| conv-MNIST-1 | 0.033 | 0.037 | 0.033 | 1.14x | 1.00x | 41.61 |
| conv-MNIST-2 | 0.170 | 0.175 | 0.169 | 1.03x | 1.00x | 42.50 |
| conv-ResNet-1 | 0.359 | 0.277 | 0.359 | 0.77x | 1.00x | 30.18 |
| conv-ResNet-2 | 2.873 | 2.717 | 2.880 | 0.95x | 1.00x | 40.23 |
| conv-ResNet-3 | 2.876 | 2.702 | 2.882 | 0.94x | 1.00x | 40.20 |
| conv-ResNet-4 | 3.429 | 2.842 | 3.427 | 0.83x | 1.00x | 33.71 |
| tall-256x16 | 0.006 | 0.006 | 0.006 | 0.97x | 1.01x | 22.37 |
| tall-512x16 | 0.012 | 0.011 | 0.012 | 0.96x | 1.01x | 22.22 |
| tall-1024x16 | 0.024 | 0.023 | 0.024 | 0.92x | 1.00x | 21.43 |
| tall-256x8 | 0.002 | 0.003 | 0.003 | 1.33x | 1.34x | 17.25 |
| tall-512x8 | 0.004 | 0.005 | 0.005 | 1.29x | 1.29x | 17.43 |
| wide-16x256 | 0.005 | 0.004 | 0.005 | 0.72x | 1.00x | 24.73 |
| wide-16x512 | 0.011 | 0.008 | 0.011 | 0.72x | 1.00x | 24.78 |
| wide-8x256 | 0.001 | 0.001 | 0.001 | 0.90x | 1.01x | 23.08 |
| wide-8x512 | 0.003 | 0.002 | 0.003 | 0.88x | 1.01x | 24.09 |
| npo2-3x64 | 0.001 | 0.001 | 0.001 | 1.18x | 1.04x | 24.58 |
| npo2-5x64 | 0.001 | 0.002 | 0.001 | 1.56x | 1.02x | 40.96 |
| npo2-7x64 | 0.003 | 0.002 | 0.003 | 0.74x | 1.00x | 21.72 |
| npo2-11x64 | 0.003 | 0.003 | 0.003 | 0.82x | 1.00x | 26.98 |
| npo2-13x64 | 0.004 | 0.003 | 0.004 | 0.84x | 1.02x | 29.10 |
| npo2-17x64 | 0.006 | 0.004 | 0.006 | 0.65x | 1.00x | 23.37 |
| npo2-23x64 | 0.004 | 0.005 | 0.004 | 1.16x | 1.00x | 43.21 |
| npo2-31x64 | 0.007 | 0.007 | 0.007 | 0.95x | 1.00x | 36.70 |
| npo2-47x64 | 0.009 | 0.010 | 0.009 | 1.11x | 1.00x | 44.26 |
| npo2-63x64 | 0.012 | 0.013 | 0.012 | 1.11x | 1.00x | 44.57 |
| Ksmall-64x64x4 | 0.002 | 0.002 | 0.002 | 0.91x | 1.01x | 18.41 |
| Ksmall-64x64x8 | 0.003 | 0.002 | 0.003 | 0.80x | 1.01x | 23.07 |
| Ksmall-64x64x12 | 0.004 | 0.003 | 0.004 | 0.72x | 1.00x | 23.74 |
| Ksmall-128x128x4 | 0.007 | 0.006 | 0.007 | 0.87x | 1.00x | 19.11 |
| Ksmall-128x128x8 | 0.011 | 0.008 | 0.011 | 0.74x | 0.99x | 23.00 |
| square-64 | 0.015 | 0.013 | 0.015 | 0.85x | 1.00x | 34.63 |
| square-128 | 0.102 | 0.098 | 0.102 | 0.96x | 1.00x | 41.10 |
| square-192 | 0.336 | 0.328 | 0.336 | 0.98x | 1.00x | 42.19 |
| square-256 | 0.802 | 0.775 | 0.801 | 0.97x | 1.00x | 41.84 |
| square-384 | 2.604 | 2.702 | 2.602 | 1.04x | 1.00x | 43.49 |
| square-512 | 6.110 | 6.189 | 6.109 | 1.01x | 1.00x | 43.94 |
| large-1024 | 48.662 | 49.104 | 48.370 | 1.01x | 0.99x | 44.13 |
| large-2048 | 385.244 | 390.472 | 385.648 | 1.01x | 1.00x | 44.59 |

---

*Generated: 2026-04-11, DNN-Opt v0.9.7-dev Phase 9*
