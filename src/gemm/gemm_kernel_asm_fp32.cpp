/// @file gemm_kernel_asm_fp32.cpp
/// Phase 9: Inline assembly GEMM micro-kernels based on autoGEMM techniques.
///
/// Key optimizations over C intrinsics:
///   1. Scalar broadcast via v[i].s[0..3] — 4 K values from one ldr q
///   2. Ping-pong B pointers (x22/x23) to hide load latency
///   3. FMLA interleaved with ldr for compute/memory overlap
///   4. Explicit register allocation — zero stack spills
///
/// Reference: autoGEMM (SC'24), https://github.com/wudu98/autoGEMM

#include "dnnopt/gemm/gemm_config.h"

#ifdef __aarch64__

#include <arm_neon.h>

namespace dnnopt {

// ============================================================
// 4×16 inline assembly kernel
// ============================================================
//
// Register allocation:
//   v0-v3:   A rows 0-3, K values [0..3]  (ping-pong set 0)
//   v4-v7:   A rows 0-3, K values [4..7]  (ping-pong set 1)
//   v8-v11:  B panel (4 × float32x4 = 16 cols), alternates x22/x23
//   v12-v27: 16 accumulators (row0: v12-v15, row1: v16-v19,
//                              row2: v20-v23, row3: v24-v27)
//
//   x6-x9:   C row pointers
//   x11-x14: A row pointers (post-increment)
//   x21:     saved A base
//   x22,x23: B ping-pong pointers (each advances by ldb per K step)
//   x24:     saved C base
//   x27:     K loop counter (counts groups of 8)
//
// K-loop: processes 8 K values per iteration (2 × 4 via .s[0..3])
// K-tail: handles K%8 with a 1-at-a-time scalar loop

void gemm_kernel_4x16_asm(int K,
                           const float* __restrict__ A, int lda,
                           const float* __restrict__ B, int ldb,
                           float* __restrict__ C, int ldc,
                           float alpha, float beta) {
    // Convert K to loop counts
    int k_main = K / 8;   // number of 8-element groups
    int k_tail = K % 8;   // remainder

    // Byte strides (will be computed in asm, but we pass element strides)
    int lda_val = lda;
    int ldb_val = ldb;
    int ldc_val = ldc;

    asm volatile(
    // === Prologue: convert strides to bytes ===
    "lsl     %w[lda], %w[lda], #2              \n"  // lda *= 4 (float bytes)
    "lsl     %w[ldb], %w[ldb], #2              \n"  // ldb *= 4 (float bytes)
    "lsl     %w[ldc], %w[ldc], #2              \n"  // ldc *= 4

    // Sign-extend to 64-bit for address arithmetic
    "sxtw    x15, %w[lda]                      \n"
    "sxtw    x16, %w[ldb]                      \n"
    "sxtw    x17, %w[ldc]                      \n"

    // === Setup A row pointers ===
    "mov     x11, %[A]                         \n"  // A row 0
    "add     x12, x11, x15                     \n"  // A row 1
    "add     x13, x11, x15, lsl #1             \n"  // A row 2
    "add     x14, x12, x15, lsl #1             \n"  // A row 3

    // === Setup C row pointers ===
    "mov     x6,  %[C]                         \n"  // C row 0
    "add     x7,  x6, x17                      \n"  // C row 1
    "add     x8,  x6, x17, lsl #1              \n"  // C row 2
    "add     x9,  x7, x17, lsl #1              \n"  // C row 3

    // === Setup B ping-pong pointers ===
    "mov     x22, %[B]                         \n"  // B[k=0]
    "add     x23, x22, x16                     \n"  // B[k=1]

    // === Zero accumulators ===
    "movi    v12.4s, #0                        \n"
    "movi    v13.4s, #0                        \n"
    "movi    v14.4s, #0                        \n"
    "movi    v15.4s, #0                        \n"
    "movi    v16.4s, #0                        \n"
    "movi    v17.4s, #0                        \n"
    "movi    v18.4s, #0                        \n"
    "movi    v19.4s, #0                        \n"
    "movi    v20.4s, #0                        \n"
    "movi    v21.4s, #0                        \n"
    "movi    v22.4s, #0                        \n"
    "movi    v23.4s, #0                        \n"
    "movi    v24.4s, #0                        \n"
    "movi    v25.4s, #0                        \n"
    "movi    v26.4s, #0                        \n"
    "movi    v27.4s, #0                        \n"

    // Compute 2*ldb for advancing B by 2 rows
    "lsl     x20, x16, #1                      \n"  // x20 = 2*ldb_bytes

    // === Check if we have main loop iterations ===
    "cbz     %w[k_main], 20f                   \n"  // skip main loop if k_main==0

    // === Preload first A vectors and first B panel ===
    "ldr     q0, [x11], #16                    \n"  // A row0 [k0..k3]
    "ldr     q1, [x12], #16                    \n"  // A row1 [k0..k3]
    "ldr     q2, [x13], #16                    \n"  // A row2 [k0..k3]
    "ldr     q3, [x14], #16                    \n"  // A row3 [k0..k3]
    "ldr     q8,  [x22]                        \n"  // B[k0] cols 0-3
    "ldr     q9,  [x22, #16]                   \n"  // B[k0] cols 4-7
    "ldr     q10, [x22, #32]                   \n"  // B[k0] cols 8-11
    "ldr     q11, [x22, #48]                   \n"  // B[k0] cols 12-15
    "add     x22, x22, x20                     \n"  // x22 -> B[k=2]

    "sub     %w[k_main], %w[k_main], #1        \n"
    "cbz     %w[k_main], 12f                   \n"  // only 1 group, skip to tail of main

    // ================================================================
    // Main K loop: processes 8 K values per iteration
    // Pattern: .s[0] uses B from x23(odd), .s[1] uses B from x22(even),
    //          .s[2] uses B from x23(odd), .s[3] uses B from x22(even)
    //          Then swap A ping-pong and repeat.
    // ================================================================
    ".p2align 4                                \n"
    "10:                                       \n"

    // --- K+0: A set 0, .s[0], B from x23 (k=1) ---
    "fmla    v12.4s, v8.4s,  v0.s[0]           \n"
    "fmla    v13.4s, v9.4s,  v0.s[0]           \n"
    "fmla    v14.4s, v10.4s, v0.s[0]           \n"
    "fmla    v15.4s, v11.4s, v0.s[0]           \n"
    "fmla    v16.4s, v8.4s,  v1.s[0]           \n"
    "fmla    v17.4s, v9.4s,  v1.s[0]           \n"
    "fmla    v18.4s, v10.4s, v1.s[0]           \n"
    "fmla    v19.4s, v11.4s, v1.s[0]           \n"
    "fmla    v20.4s, v8.4s,  v2.s[0]           \n"
    "fmla    v21.4s, v9.4s,  v2.s[0]           \n"
    "fmla    v22.4s, v10.4s, v2.s[0]           \n"
    "fmla    v23.4s, v11.4s, v2.s[0]           \n"
    "fmla    v24.4s, v8.4s,  v3.s[0]           \n"
    "fmla    v25.4s, v9.4s,  v3.s[0]           \n"
    "fmla    v26.4s, v10.4s, v3.s[0]           \n"
    "fmla    v27.4s, v11.4s, v3.s[0]           \n"

    // Load B[k=1] from x23
    "ldr     q8,  [x23]                        \n"
    "ldr     q9,  [x23, #16]                   \n"
    "ldr     q10, [x23, #32]                   \n"
    "ldr     q11, [x23, #48]                   \n"
    "add     x23, x23, x20                     \n"  // x23 -> B[k=3]

    // --- K+1: A set 0, .s[1] ---
    "fmla    v12.4s, v8.4s,  v0.s[1]           \n"
    "fmla    v13.4s, v9.4s,  v0.s[1]           \n"
    "fmla    v14.4s, v10.4s, v0.s[1]           \n"
    "fmla    v15.4s, v11.4s, v0.s[1]           \n"
    "fmla    v16.4s, v8.4s,  v1.s[1]           \n"
    "fmla    v17.4s, v9.4s,  v1.s[1]           \n"
    "fmla    v18.4s, v10.4s, v1.s[1]           \n"
    "fmla    v19.4s, v11.4s, v1.s[1]           \n"
    "fmla    v20.4s, v8.4s,  v2.s[1]           \n"
    "fmla    v21.4s, v9.4s,  v2.s[1]           \n"
    "fmla    v22.4s, v10.4s, v2.s[1]           \n"
    "fmla    v23.4s, v11.4s, v2.s[1]           \n"
    "fmla    v24.4s, v8.4s,  v3.s[1]           \n"
    "fmla    v25.4s, v9.4s,  v3.s[1]           \n"
    "fmla    v26.4s, v10.4s, v3.s[1]           \n"
    "fmla    v27.4s, v11.4s, v3.s[1]           \n"

    // Load B[k=2] from x22
    "ldr     q8,  [x22]                        \n"
    "ldr     q9,  [x22, #16]                   \n"
    "ldr     q10, [x22, #32]                   \n"
    "ldr     q11, [x22, #48]                   \n"
    "add     x22, x22, x20                     \n"  // x22 -> B[k=4]

    // --- K+2: A set 0, .s[2] ---
    "fmla    v12.4s, v8.4s,  v0.s[2]           \n"
    "fmla    v13.4s, v9.4s,  v0.s[2]           \n"
    "fmla    v14.4s, v10.4s, v0.s[2]           \n"
    "fmla    v15.4s, v11.4s, v0.s[2]           \n"
    "fmla    v16.4s, v8.4s,  v1.s[2]           \n"
    "fmla    v17.4s, v9.4s,  v1.s[2]           \n"
    "fmla    v18.4s, v10.4s, v1.s[2]           \n"
    "fmla    v19.4s, v11.4s, v1.s[2]           \n"
    "fmla    v20.4s, v8.4s,  v2.s[2]           \n"
    "fmla    v21.4s, v9.4s,  v2.s[2]           \n"
    "fmla    v22.4s, v10.4s, v2.s[2]           \n"
    "fmla    v23.4s, v11.4s, v2.s[2]           \n"
    "fmla    v24.4s, v8.4s,  v3.s[2]           \n"
    "fmla    v25.4s, v9.4s,  v3.s[2]           \n"
    "fmla    v26.4s, v10.4s, v3.s[2]           \n"
    "fmla    v27.4s, v11.4s, v3.s[2]           \n"

    // Load B[k=3] from x23
    "ldr     q8,  [x23]                        \n"
    "ldr     q9,  [x23, #16]                   \n"
    "ldr     q10, [x23, #32]                   \n"
    "ldr     q11, [x23, #48]                   \n"
    "add     x23, x23, x20                     \n"  // x23 -> B[k=5]

    // --- K+3: A set 0, .s[3] ---
    "fmla    v12.4s, v8.4s,  v0.s[3]           \n"
    "fmla    v13.4s, v9.4s,  v0.s[3]           \n"
    "fmla    v14.4s, v10.4s, v0.s[3]           \n"
    "fmla    v15.4s, v11.4s, v0.s[3]           \n"
    "fmla    v16.4s, v8.4s,  v1.s[3]           \n"
    "fmla    v17.4s, v9.4s,  v1.s[3]           \n"
    "fmla    v18.4s, v10.4s, v1.s[3]           \n"
    "fmla    v19.4s, v11.4s, v1.s[3]           \n"
    "fmla    v20.4s, v8.4s,  v2.s[3]           \n"
    "fmla    v21.4s, v9.4s,  v2.s[3]           \n"
    "fmla    v22.4s, v10.4s, v2.s[3]           \n"
    "fmla    v23.4s, v11.4s, v2.s[3]           \n"
    "fmla    v24.4s, v8.4s,  v3.s[3]           \n"
    "fmla    v25.4s, v9.4s,  v3.s[3]           \n"
    "fmla    v26.4s, v10.4s, v3.s[3]           \n"
    "fmla    v27.4s, v11.4s, v3.s[3]           \n"

    // Load next A ping-pong set (q4-q7) and B[k=4] from x22
    "ldr     q4, [x11], #16                    \n"  // A row0 [k4..k7]
    "ldr     q5, [x12], #16                    \n"  // A row1
    "ldr     q6, [x13], #16                    \n"  // A row2
    "ldr     q7, [x14], #16                    \n"  // A row3
    "ldr     q8,  [x22]                        \n"
    "ldr     q9,  [x22, #16]                   \n"
    "ldr     q10, [x22, #32]                   \n"
    "ldr     q11, [x22, #48]                   \n"
    "add     x22, x22, x20                     \n"  // x22 -> B[k=6]

    // --- K+4: A set 1, .s[0] ---
    "fmla    v12.4s, v8.4s,  v4.s[0]           \n"
    "fmla    v13.4s, v9.4s,  v4.s[0]           \n"
    "fmla    v14.4s, v10.4s, v4.s[0]           \n"
    "fmla    v15.4s, v11.4s, v4.s[0]           \n"
    "fmla    v16.4s, v8.4s,  v5.s[0]           \n"
    "fmla    v17.4s, v9.4s,  v5.s[0]           \n"
    "fmla    v18.4s, v10.4s, v5.s[0]           \n"
    "fmla    v19.4s, v11.4s, v5.s[0]           \n"
    "fmla    v20.4s, v8.4s,  v6.s[0]           \n"
    "fmla    v21.4s, v9.4s,  v6.s[0]           \n"
    "fmla    v22.4s, v10.4s, v6.s[0]           \n"
    "fmla    v23.4s, v11.4s, v6.s[0]           \n"
    "fmla    v24.4s, v8.4s,  v7.s[0]           \n"
    "fmla    v25.4s, v9.4s,  v7.s[0]           \n"
    "fmla    v26.4s, v10.4s, v7.s[0]           \n"
    "fmla    v27.4s, v11.4s, v7.s[0]           \n"

    // Load B[k=5] from x23
    "ldr     q8,  [x23]                        \n"
    "ldr     q9,  [x23, #16]                   \n"
    "ldr     q10, [x23, #32]                   \n"
    "ldr     q11, [x23, #48]                   \n"
    "add     x23, x23, x20                     \n"  // x23 -> B[k=7]

    // --- K+5: A set 1, .s[1] ---
    "fmla    v12.4s, v8.4s,  v4.s[1]           \n"
    "fmla    v13.4s, v9.4s,  v4.s[1]           \n"
    "fmla    v14.4s, v10.4s, v4.s[1]           \n"
    "fmla    v15.4s, v11.4s, v4.s[1]           \n"
    "fmla    v16.4s, v8.4s,  v5.s[1]           \n"
    "fmla    v17.4s, v9.4s,  v5.s[1]           \n"
    "fmla    v18.4s, v10.4s, v5.s[1]           \n"
    "fmla    v19.4s, v11.4s, v5.s[1]           \n"
    "fmla    v20.4s, v8.4s,  v6.s[1]           \n"
    "fmla    v21.4s, v9.4s,  v6.s[1]           \n"
    "fmla    v22.4s, v10.4s, v6.s[1]           \n"
    "fmla    v23.4s, v11.4s, v6.s[1]           \n"
    "fmla    v24.4s, v8.4s,  v7.s[1]           \n"
    "fmla    v25.4s, v9.4s,  v7.s[1]           \n"
    "fmla    v26.4s, v10.4s, v7.s[1]           \n"
    "fmla    v27.4s, v11.4s, v7.s[1]           \n"

    // Load B[k=6] from x22
    "ldr     q8,  [x22]                        \n"
    "ldr     q9,  [x22, #16]                   \n"
    "ldr     q10, [x22, #32]                   \n"
    "ldr     q11, [x22, #48]                   \n"
    "add     x22, x22, x20                     \n"  // x22 -> B[k=8]

    // --- K+6: A set 1, .s[2] ---
    "fmla    v12.4s, v8.4s,  v4.s[2]           \n"
    "fmla    v13.4s, v9.4s,  v4.s[2]           \n"
    "fmla    v14.4s, v10.4s, v4.s[2]           \n"
    "fmla    v15.4s, v11.4s, v4.s[2]           \n"
    "fmla    v16.4s, v8.4s,  v5.s[2]           \n"
    "fmla    v17.4s, v9.4s,  v5.s[2]           \n"
    "fmla    v18.4s, v10.4s, v5.s[2]           \n"
    "fmla    v19.4s, v11.4s, v5.s[2]           \n"
    "fmla    v20.4s, v8.4s,  v6.s[2]           \n"
    "fmla    v21.4s, v9.4s,  v6.s[2]           \n"
    "fmla    v22.4s, v10.4s, v6.s[2]           \n"
    "fmla    v23.4s, v11.4s, v6.s[2]           \n"
    "fmla    v24.4s, v8.4s,  v7.s[2]           \n"
    "fmla    v25.4s, v9.4s,  v7.s[2]           \n"
    "fmla    v26.4s, v10.4s, v7.s[2]           \n"
    "fmla    v27.4s, v11.4s, v7.s[2]           \n"

    // Load B[k=7] from x23
    "ldr     q8,  [x23]                        \n"
    "ldr     q9,  [x23, #16]                   \n"
    "ldr     q10, [x23, #32]                   \n"
    "ldr     q11, [x23, #48]                   \n"
    "add     x23, x23, x20                     \n"  // x23 -> B[k=9]

    // --- K+7: A set 1, .s[3] ---
    "fmla    v12.4s, v8.4s,  v4.s[3]           \n"
    "fmla    v13.4s, v9.4s,  v4.s[3]           \n"
    "fmla    v14.4s, v10.4s, v4.s[3]           \n"
    "fmla    v15.4s, v11.4s, v4.s[3]           \n"
    "fmla    v16.4s, v8.4s,  v5.s[3]           \n"
    "fmla    v17.4s, v9.4s,  v5.s[3]           \n"
    "fmla    v18.4s, v10.4s, v5.s[3]           \n"
    "fmla    v19.4s, v11.4s, v5.s[3]           \n"
    "fmla    v20.4s, v8.4s,  v6.s[3]           \n"
    "fmla    v21.4s, v9.4s,  v6.s[3]           \n"
    "fmla    v22.4s, v10.4s, v6.s[3]           \n"
    "fmla    v23.4s, v11.4s, v6.s[3]           \n"
    "fmla    v24.4s, v8.4s,  v7.s[3]           \n"
    "fmla    v25.4s, v9.4s,  v7.s[3]           \n"
    "fmla    v26.4s, v10.4s, v7.s[3]           \n"
    "fmla    v27.4s, v11.4s, v7.s[3]           \n"

    // Reload A set 0 and B for next iteration
    "ldr     q0, [x11], #16                    \n"
    "ldr     q1, [x12], #16                    \n"
    "ldr     q2, [x13], #16                    \n"
    "ldr     q3, [x14], #16                    \n"
    "ldr     q8,  [x22]                        \n"
    "ldr     q9,  [x22, #16]                   \n"
    "ldr     q10, [x22, #32]                   \n"
    "ldr     q11, [x22, #48]                   \n"
    "add     x22, x22, x20                     \n"

    "subs    %w[k_main], %w[k_main], #1        \n"
    "bne     10b                               \n"

    // ================================================================
    // Last main-loop group (no need to reload A/B at end)
    // ================================================================
    "12:                                       \n"
    // --- K+0: .s[0] ---
    "fmla    v12.4s, v8.4s,  v0.s[0]           \n"
    "fmla    v13.4s, v9.4s,  v0.s[0]           \n"
    "fmla    v14.4s, v10.4s, v0.s[0]           \n"
    "fmla    v15.4s, v11.4s, v0.s[0]           \n"
    "fmla    v16.4s, v8.4s,  v1.s[0]           \n"
    "fmla    v17.4s, v9.4s,  v1.s[0]           \n"
    "fmla    v18.4s, v10.4s, v1.s[0]           \n"
    "fmla    v19.4s, v11.4s, v1.s[0]           \n"
    "fmla    v20.4s, v8.4s,  v2.s[0]           \n"
    "fmla    v21.4s, v9.4s,  v2.s[0]           \n"
    "fmla    v22.4s, v10.4s, v2.s[0]           \n"
    "fmla    v23.4s, v11.4s, v2.s[0]           \n"
    "fmla    v24.4s, v8.4s,  v3.s[0]           \n"
    "fmla    v25.4s, v9.4s,  v3.s[0]           \n"
    "fmla    v26.4s, v10.4s, v3.s[0]           \n"
    "fmla    v27.4s, v11.4s, v3.s[0]           \n"
    "ldr     q8,  [x23]                        \n"
    "ldr     q9,  [x23, #16]                   \n"
    "ldr     q10, [x23, #32]                   \n"
    "ldr     q11, [x23, #48]                   \n"
    "add     x23, x23, x20                     \n"

    // --- K+1: .s[1] ---
    "fmla    v12.4s, v8.4s,  v0.s[1]           \n"
    "fmla    v13.4s, v9.4s,  v0.s[1]           \n"
    "fmla    v14.4s, v10.4s, v0.s[1]           \n"
    "fmla    v15.4s, v11.4s, v0.s[1]           \n"
    "fmla    v16.4s, v8.4s,  v1.s[1]           \n"
    "fmla    v17.4s, v9.4s,  v1.s[1]           \n"
    "fmla    v18.4s, v10.4s, v1.s[1]           \n"
    "fmla    v19.4s, v11.4s, v1.s[1]           \n"
    "fmla    v20.4s, v8.4s,  v2.s[1]           \n"
    "fmla    v21.4s, v9.4s,  v2.s[1]           \n"
    "fmla    v22.4s, v10.4s, v2.s[1]           \n"
    "fmla    v23.4s, v11.4s, v2.s[1]           \n"
    "fmla    v24.4s, v8.4s,  v3.s[1]           \n"
    "fmla    v25.4s, v9.4s,  v3.s[1]           \n"
    "fmla    v26.4s, v10.4s, v3.s[1]           \n"
    "fmla    v27.4s, v11.4s, v3.s[1]           \n"
    "ldr     q8,  [x22]                        \n"
    "ldr     q9,  [x22, #16]                   \n"
    "ldr     q10, [x22, #32]                   \n"
    "ldr     q11, [x22, #48]                   \n"
    "add     x22, x22, x20                     \n"

    // --- K+2: .s[2] ---
    "fmla    v12.4s, v8.4s,  v0.s[2]           \n"
    "fmla    v13.4s, v9.4s,  v0.s[2]           \n"
    "fmla    v14.4s, v10.4s, v0.s[2]           \n"
    "fmla    v15.4s, v11.4s, v0.s[2]           \n"
    "fmla    v16.4s, v8.4s,  v1.s[2]           \n"
    "fmla    v17.4s, v9.4s,  v1.s[2]           \n"
    "fmla    v18.4s, v10.4s, v1.s[2]           \n"
    "fmla    v19.4s, v11.4s, v1.s[2]           \n"
    "fmla    v20.4s, v8.4s,  v2.s[2]           \n"
    "fmla    v21.4s, v9.4s,  v2.s[2]           \n"
    "fmla    v22.4s, v10.4s, v2.s[2]           \n"
    "fmla    v23.4s, v11.4s, v2.s[2]           \n"
    "fmla    v24.4s, v8.4s,  v3.s[2]           \n"
    "fmla    v25.4s, v9.4s,  v3.s[2]           \n"
    "fmla    v26.4s, v10.4s, v3.s[2]           \n"
    "fmla    v27.4s, v11.4s, v3.s[2]           \n"
    "ldr     q8,  [x23]                        \n"
    "ldr     q9,  [x23, #16]                   \n"
    "ldr     q10, [x23, #32]                   \n"
    "ldr     q11, [x23, #48]                   \n"
    "add     x23, x23, x20                     \n"

    // --- K+3: .s[3] ---
    "fmla    v12.4s, v8.4s,  v0.s[3]           \n"
    "fmla    v13.4s, v9.4s,  v0.s[3]           \n"
    "fmla    v14.4s, v10.4s, v0.s[3]           \n"
    "fmla    v15.4s, v11.4s, v0.s[3]           \n"
    "fmla    v16.4s, v8.4s,  v1.s[3]           \n"
    "fmla    v17.4s, v9.4s,  v1.s[3]           \n"
    "fmla    v18.4s, v10.4s, v1.s[3]           \n"
    "fmla    v19.4s, v11.4s, v1.s[3]           \n"
    "fmla    v20.4s, v8.4s,  v2.s[3]           \n"
    "fmla    v21.4s, v9.4s,  v2.s[3]           \n"
    "fmla    v22.4s, v10.4s, v2.s[3]           \n"
    "fmla    v23.4s, v11.4s, v2.s[3]           \n"
    "fmla    v24.4s, v8.4s,  v3.s[3]           \n"
    "fmla    v25.4s, v9.4s,  v3.s[3]           \n"
    "fmla    v26.4s, v10.4s, v3.s[3]           \n"
    "fmla    v27.4s, v11.4s, v3.s[3]           \n"

    // Load A set 1 and B
    "ldr     q4, [x11], #16                    \n"
    "ldr     q5, [x12], #16                    \n"
    "ldr     q6, [x13], #16                    \n"
    "ldr     q7, [x14], #16                    \n"
    "ldr     q8,  [x22]                        \n"
    "ldr     q9,  [x22, #16]                   \n"
    "ldr     q10, [x22, #32]                   \n"
    "ldr     q11, [x22, #48]                   \n"
    "add     x22, x22, x20                     \n"

    // --- K+4: .s[0] ---
    "fmla    v12.4s, v8.4s,  v4.s[0]           \n"
    "fmla    v13.4s, v9.4s,  v4.s[0]           \n"
    "fmla    v14.4s, v10.4s, v4.s[0]           \n"
    "fmla    v15.4s, v11.4s, v4.s[0]           \n"
    "fmla    v16.4s, v8.4s,  v5.s[0]           \n"
    "fmla    v17.4s, v9.4s,  v5.s[0]           \n"
    "fmla    v18.4s, v10.4s, v5.s[0]           \n"
    "fmla    v19.4s, v11.4s, v5.s[0]           \n"
    "fmla    v20.4s, v8.4s,  v6.s[0]           \n"
    "fmla    v21.4s, v9.4s,  v6.s[0]           \n"
    "fmla    v22.4s, v10.4s, v6.s[0]           \n"
    "fmla    v23.4s, v11.4s, v6.s[0]           \n"
    "fmla    v24.4s, v8.4s,  v7.s[0]           \n"
    "fmla    v25.4s, v9.4s,  v7.s[0]           \n"
    "fmla    v26.4s, v10.4s, v7.s[0]           \n"
    "fmla    v27.4s, v11.4s, v7.s[0]           \n"
    "ldr     q8,  [x23]                        \n"
    "ldr     q9,  [x23, #16]                   \n"
    "ldr     q10, [x23, #32]                   \n"
    "ldr     q11, [x23, #48]                   \n"
    "add     x23, x23, x20                     \n"

    // --- K+5: .s[1] ---
    "fmla    v12.4s, v8.4s,  v4.s[1]           \n"
    "fmla    v13.4s, v9.4s,  v4.s[1]           \n"
    "fmla    v14.4s, v10.4s, v4.s[1]           \n"
    "fmla    v15.4s, v11.4s, v4.s[1]           \n"
    "fmla    v16.4s, v8.4s,  v5.s[1]           \n"
    "fmla    v17.4s, v9.4s,  v5.s[1]           \n"
    "fmla    v18.4s, v10.4s, v5.s[1]           \n"
    "fmla    v19.4s, v11.4s, v5.s[1]           \n"
    "fmla    v20.4s, v8.4s,  v6.s[1]           \n"
    "fmla    v21.4s, v9.4s,  v6.s[1]           \n"
    "fmla    v22.4s, v10.4s, v6.s[1]           \n"
    "fmla    v23.4s, v11.4s, v6.s[1]           \n"
    "fmla    v24.4s, v8.4s,  v7.s[1]           \n"
    "fmla    v25.4s, v9.4s,  v7.s[1]           \n"
    "fmla    v26.4s, v10.4s, v7.s[1]           \n"
    "fmla    v27.4s, v11.4s, v7.s[1]           \n"
    "ldr     q8,  [x22]                        \n"
    "ldr     q9,  [x22, #16]                   \n"
    "ldr     q10, [x22, #32]                   \n"
    "ldr     q11, [x22, #48]                   \n"
    "add     x22, x22, x20                     \n"

    // --- K+6: .s[2] ---
    "fmla    v12.4s, v8.4s,  v4.s[2]           \n"
    "fmla    v13.4s, v9.4s,  v4.s[2]           \n"
    "fmla    v14.4s, v10.4s, v4.s[2]           \n"
    "fmla    v15.4s, v11.4s, v4.s[2]           \n"
    "fmla    v16.4s, v8.4s,  v5.s[2]           \n"
    "fmla    v17.4s, v9.4s,  v5.s[2]           \n"
    "fmla    v18.4s, v10.4s, v5.s[2]           \n"
    "fmla    v19.4s, v11.4s, v5.s[2]           \n"
    "fmla    v20.4s, v8.4s,  v6.s[2]           \n"
    "fmla    v21.4s, v9.4s,  v6.s[2]           \n"
    "fmla    v22.4s, v10.4s, v6.s[2]           \n"
    "fmla    v23.4s, v11.4s, v6.s[2]           \n"
    "fmla    v24.4s, v8.4s,  v7.s[2]           \n"
    "fmla    v25.4s, v9.4s,  v7.s[2]           \n"
    "fmla    v26.4s, v10.4s, v7.s[2]           \n"
    "fmla    v27.4s, v11.4s, v7.s[2]           \n"
    "ldr     q8,  [x23]                        \n"
    "ldr     q9,  [x23, #16]                   \n"
    "ldr     q10, [x23, #32]                   \n"
    "ldr     q11, [x23, #48]                   \n"
    "add     x23, x23, x20                     \n"

    // --- K+7: .s[3] (final of this group) ---
    "fmla    v12.4s, v8.4s,  v4.s[3]           \n"
    "fmla    v13.4s, v9.4s,  v4.s[3]           \n"
    "fmla    v14.4s, v10.4s, v4.s[3]           \n"
    "fmla    v15.4s, v11.4s, v4.s[3]           \n"
    "fmla    v16.4s, v8.4s,  v5.s[3]           \n"
    "fmla    v17.4s, v9.4s,  v5.s[3]           \n"
    "fmla    v18.4s, v10.4s, v5.s[3]           \n"
    "fmla    v19.4s, v11.4s, v5.s[3]           \n"
    "fmla    v20.4s, v8.4s,  v6.s[3]           \n"
    "fmla    v21.4s, v9.4s,  v6.s[3]           \n"
    "fmla    v22.4s, v10.4s, v6.s[3]           \n"
    "fmla    v23.4s, v11.4s, v6.s[3]           \n"
    "fmla    v24.4s, v8.4s,  v7.s[3]           \n"
    "fmla    v25.4s, v9.4s,  v7.s[3]           \n"
    "fmla    v26.4s, v10.4s, v7.s[3]           \n"
    "fmla    v27.4s, v11.4s, v7.s[3]           \n"

    // ================================================================
    // K tail: process remaining K%8 values one at a time
    // ================================================================
    "20:                                       \n"
    "cbz     %w[k_tail], 30f                   \n"  // skip tail if none

    // x22 and x23 have been ping-ponging. Compute tail B pointer.
    // After main loop, B pointer is at B + k_main*8*ldb.
    // But x22/x23 already point past the last used row, so
    // we need to figure out where the tail starts.
    // Actually, x22 points to B[k_main*8] (even) and
    // x23 points to B[k_main*8+1] (odd), but they've been
    // advancing by 2*ldb. Let's use a fresh pointer.
    // Recompute: tail_B = B + (K - k_tail) * ldb_bytes
    // But we don't have original B anymore. Use x22 which is correct
    // for even-indexed tail start since main loop leaves x22 at the
    // right position (B + k_main*8 * ldb, but each x22 step is +2*ldb).
    // Simpler: just use x22 as single B pointer for tail, advance by ldb.

    "21:                                       \n"
    // Load one B row
    "ldr     q8,  [x22]                        \n"
    "ldr     q9,  [x22, #16]                   \n"
    "ldr     q10, [x22, #32]                   \n"
    "ldr     q11, [x22, #48]                   \n"

    // Load one A scalar per row
    "ldr     s0, [x11], #4                     \n"  // A[row0][k]
    "ldr     s1, [x12], #4                     \n"  // A[row1][k]
    "ldr     s2, [x13], #4                     \n"  // A[row2][k]
    "ldr     s3, [x14], #4                     \n"  // A[row3][k]

    "fmla    v12.4s, v8.4s,  v0.s[0]           \n"
    "fmla    v13.4s, v9.4s,  v0.s[0]           \n"
    "fmla    v14.4s, v10.4s, v0.s[0]           \n"
    "fmla    v15.4s, v11.4s, v0.s[0]           \n"
    "fmla    v16.4s, v8.4s,  v1.s[0]           \n"
    "fmla    v17.4s, v9.4s,  v1.s[0]           \n"
    "fmla    v18.4s, v10.4s, v1.s[0]           \n"
    "fmla    v19.4s, v11.4s, v1.s[0]           \n"
    "fmla    v20.4s, v8.4s,  v2.s[0]           \n"
    "fmla    v21.4s, v9.4s,  v2.s[0]           \n"
    "fmla    v22.4s, v10.4s, v2.s[0]           \n"
    "fmla    v23.4s, v11.4s, v2.s[0]           \n"
    "fmla    v24.4s, v8.4s,  v3.s[0]           \n"
    "fmla    v25.4s, v9.4s,  v3.s[0]           \n"
    "fmla    v26.4s, v10.4s, v3.s[0]           \n"
    "fmla    v27.4s, v11.4s, v3.s[0]           \n"

    "add     x22, x22, x16                     \n"  // advance by ldb (single row)
    "subs    %w[k_tail], %w[k_tail], #1        \n"
    "bne     21b                               \n"

    // ================================================================
    // Epilogue: store C = alpha * acc + beta * C
    // ================================================================
    "30:                                       \n"
    // Load alpha into v0
    "ld1r    {v0.4s}, [%[alpha_ptr]]           \n"

    // Check beta
    "ldr     s1, [%[beta_ptr]]                 \n"
    "fcmp    s1, #0.0                          \n"
    "beq     40f                               \n"  // beta == 0 fast path

    // --- beta != 0: C = alpha*acc + beta*C ---
    "dup     v1.4s, v1.s[0]                    \n"  // broadcast beta

    // Row 0
    "ldr     q2, [x6]                          \n"
    "ldr     q3, [x6, #16]                     \n"
    "ldr     q4, [x6, #32]                     \n"
    "ldr     q5, [x6, #48]                     \n"
    "fmul    v12.4s, v12.4s, v0.4s             \n"
    "fmul    v13.4s, v13.4s, v0.4s             \n"
    "fmul    v14.4s, v14.4s, v0.4s             \n"
    "fmul    v15.4s, v15.4s, v0.4s             \n"
    "fmla    v12.4s, v2.4s, v1.4s              \n"
    "fmla    v13.4s, v3.4s, v1.4s              \n"
    "fmla    v14.4s, v4.4s, v1.4s              \n"
    "fmla    v15.4s, v5.4s, v1.4s              \n"
    "str     q12, [x6]                         \n"
    "str     q13, [x6, #16]                    \n"
    "str     q14, [x6, #32]                    \n"
    "str     q15, [x6, #48]                    \n"

    // Row 1
    "ldr     q2, [x7]                          \n"
    "ldr     q3, [x7, #16]                     \n"
    "ldr     q4, [x7, #32]                     \n"
    "ldr     q5, [x7, #48]                     \n"
    "fmul    v16.4s, v16.4s, v0.4s             \n"
    "fmul    v17.4s, v17.4s, v0.4s             \n"
    "fmul    v18.4s, v18.4s, v0.4s             \n"
    "fmul    v19.4s, v19.4s, v0.4s             \n"
    "fmla    v16.4s, v2.4s, v1.4s              \n"
    "fmla    v17.4s, v3.4s, v1.4s              \n"
    "fmla    v18.4s, v4.4s, v1.4s              \n"
    "fmla    v19.4s, v5.4s, v1.4s              \n"
    "str     q16, [x7]                         \n"
    "str     q17, [x7, #16]                    \n"
    "str     q18, [x7, #32]                    \n"
    "str     q19, [x7, #48]                    \n"

    // Row 2
    "ldr     q2, [x8]                          \n"
    "ldr     q3, [x8, #16]                     \n"
    "ldr     q4, [x8, #32]                     \n"
    "ldr     q5, [x8, #48]                     \n"
    "fmul    v20.4s, v20.4s, v0.4s             \n"
    "fmul    v21.4s, v21.4s, v0.4s             \n"
    "fmul    v22.4s, v22.4s, v0.4s             \n"
    "fmul    v23.4s, v23.4s, v0.4s             \n"
    "fmla    v20.4s, v2.4s, v1.4s              \n"
    "fmla    v21.4s, v3.4s, v1.4s              \n"
    "fmla    v22.4s, v4.4s, v1.4s              \n"
    "fmla    v23.4s, v5.4s, v1.4s              \n"
    "str     q20, [x8]                         \n"
    "str     q21, [x8, #16]                    \n"
    "str     q22, [x8, #32]                    \n"
    "str     q23, [x8, #48]                    \n"

    // Row 3
    "ldr     q2, [x9]                          \n"
    "ldr     q3, [x9, #16]                     \n"
    "ldr     q4, [x9, #32]                     \n"
    "ldr     q5, [x9, #48]                     \n"
    "fmul    v24.4s, v24.4s, v0.4s             \n"
    "fmul    v25.4s, v25.4s, v0.4s             \n"
    "fmul    v26.4s, v26.4s, v0.4s             \n"
    "fmul    v27.4s, v27.4s, v0.4s             \n"
    "fmla    v24.4s, v2.4s, v1.4s              \n"
    "fmla    v25.4s, v3.4s, v1.4s              \n"
    "fmla    v26.4s, v4.4s, v1.4s              \n"
    "fmla    v27.4s, v5.4s, v1.4s              \n"
    "str     q24, [x9]                         \n"
    "str     q25, [x9, #16]                    \n"
    "str     q26, [x9, #32]                    \n"
    "str     q27, [x9, #48]                    \n"
    "b       50f                               \n"

    // --- beta == 0: C = alpha * acc ---
    "40:                                       \n"
    "fmul    v12.4s, v12.4s, v0.4s             \n"
    "fmul    v13.4s, v13.4s, v0.4s             \n"
    "fmul    v14.4s, v14.4s, v0.4s             \n"
    "fmul    v15.4s, v15.4s, v0.4s             \n"
    "str     q12, [x6]                         \n"
    "str     q13, [x6, #16]                    \n"
    "str     q14, [x6, #32]                    \n"
    "str     q15, [x6, #48]                    \n"

    "fmul    v16.4s, v16.4s, v0.4s             \n"
    "fmul    v17.4s, v17.4s, v0.4s             \n"
    "fmul    v18.4s, v18.4s, v0.4s             \n"
    "fmul    v19.4s, v19.4s, v0.4s             \n"
    "str     q16, [x7]                         \n"
    "str     q17, [x7, #16]                    \n"
    "str     q18, [x7, #32]                    \n"
    "str     q19, [x7, #48]                    \n"

    "fmul    v20.4s, v20.4s, v0.4s             \n"
    "fmul    v21.4s, v21.4s, v0.4s             \n"
    "fmul    v22.4s, v22.4s, v0.4s             \n"
    "fmul    v23.4s, v23.4s, v0.4s             \n"
    "str     q20, [x8]                         \n"
    "str     q21, [x8, #16]                    \n"
    "str     q22, [x8, #32]                    \n"
    "str     q23, [x8, #48]                    \n"

    "fmul    v24.4s, v24.4s, v0.4s             \n"
    "fmul    v25.4s, v25.4s, v0.4s             \n"
    "fmul    v26.4s, v26.4s, v0.4s             \n"
    "fmul    v27.4s, v27.4s, v0.4s             \n"
    "str     q24, [x9]                         \n"
    "str     q25, [x9, #16]                    \n"
    "str     q26, [x9, #32]                    \n"
    "str     q27, [x9, #48]                    \n"

    "50:                                       \n"

    : [k_main] "+r" (k_main),
      [k_tail] "+r" (k_tail),
      [lda] "+r" (lda_val),
      [ldb] "+r" (ldb_val),
      [ldc] "+r" (ldc_val)
    : [A] "r" (A),
      [B] "r" (B),
      [C] "r" (C),
      [alpha_ptr] "r" (&alpha),
      [beta_ptr] "r" (&beta)
    : "cc", "memory",
      "x6", "x7", "x8", "x9",
      "x11", "x12", "x13", "x14", "x15", "x16", "x17",
      "x20", "x22", "x23",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
      "v8", "v9", "v10", "v11",
      "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
      "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27"
    );
}

// ============================================================
// 6×16 inline assembly kernel
// ============================================================
//
// Register allocation (very tight — all 32 SIMD regs used):
//   v0-v3:   A rows 0-3, 4 K values via .s[0..3]
//   v4-v5:   A rows 4-5
//   v6-v9:   B panel (4 vectors = 16 cols)
//   v10-v31: 22 accumulators... NOT ENOUGH for 24.
//
// Solution: use 4x K-unrolling (not 8x). Load A one scalar at a time.
//   v0-v5:   A scalars (6 rows, broadcast)
//   v6-v9:   B panel (4 vectors)
//   v10-v31: 22 regs. We need 24 acc (6×4).
//
// Better solution: same pattern as 4x16 but with 4x K-unroll.
//   v0-v2:   A rows 0-2, 4 K values (ping-pong set 0, only 3 regs)
//   v3-v5:   A rows 3-5, 4 K values
//   v6-v9:   B panel
//   v10-v31: 22 acc. Still need 24!
//
// Final approach: 6x16 with scalar A loads (1 K at a time):
//   v0:      temp A scalar broadcast
//   v1-v4:   B panel (4 vectors)
//   v5-v28:  24 accumulators (6 rows × 4 B-quads)
//   v29-v31: spare
//
// This is less efficient than the 4x16 approach but still avoids
// compiler-generated spills.

void gemm_kernel_6x16_asm(int K,
                           const float* __restrict__ A, int lda,
                           const float* __restrict__ B, int ldb,
                           float* __restrict__ C, int ldc,
                           float alpha, float beta) {
    int lda_val = lda;
    int ldb_val = ldb;
    int ldc_val = ldc;

    asm volatile(
    // === Convert strides to bytes ===
    "lsl     %w[lda], %w[lda], #2              \n"
    "lsl     %w[ldb], %w[ldb], #2              \n"
    "lsl     %w[ldc], %w[ldc], #2              \n"
    "sxtw    x15, %w[lda]                      \n"
    "sxtw    x16, %w[ldb]                      \n"
    "sxtw    x17, %w[ldc]                      \n"

    // === Setup A row pointers (6 rows) ===
    "mov     x6,  %[A]                         \n"  // A row 0
    "add     x7,  x6,  x15                     \n"  // A row 1
    "add     x8,  x6,  x15, lsl #1             \n"  // A row 2
    "add     x9,  x7,  x15, lsl #1             \n"  // A row 3
    "add     x10, x8,  x15, lsl #1             \n"  // A row 4
    "add     x11, x9,  x15, lsl #1             \n"  // A row 5

    // === Setup C row pointers (6 rows) ===
    "mov     x12, %[C]                         \n"  // C row 0
    "add     x13, x12, x17                     \n"  // C row 1
    "add     x14, x12, x17, lsl #1             \n"  // C row 2
    "add     x19, x13, x17, lsl #1             \n"  // C row 3
    "add     x20, x14, x17, lsl #1             \n"  // C row 4
    "add     x21, x19, x17, lsl #1             \n"  // C row 5

    // === Setup B pointer ===
    "mov     x22, %[B]                         \n"

    // === Zero 24 accumulators (v5-v28) ===
    "movi    v5.4s,  #0 \n"  "movi    v6.4s,  #0 \n"
    "movi    v7.4s,  #0 \n"  "movi    v8.4s,  #0 \n"
    "movi    v9.4s,  #0 \n"  "movi    v10.4s, #0 \n"
    "movi    v11.4s, #0 \n"  "movi    v12.4s, #0 \n"
    "movi    v13.4s, #0 \n"  "movi    v14.4s, #0 \n"
    "movi    v15.4s, #0 \n"  "movi    v16.4s, #0 \n"
    "movi    v17.4s, #0 \n"  "movi    v18.4s, #0 \n"
    "movi    v19.4s, #0 \n"  "movi    v20.4s, #0 \n"
    "movi    v21.4s, #0 \n"  "movi    v22.4s, #0 \n"
    "movi    v23.4s, #0 \n"  "movi    v24.4s, #0 \n"
    "movi    v25.4s, #0 \n"  "movi    v26.4s, #0 \n"
    "movi    v27.4s, #0 \n"  "movi    v28.4s, #0 \n"

    // === K loop (1 K value at a time) ===
    "mov     w23, %w[K]                        \n"
    "cbz     w23, 30f                          \n"

    ".p2align 4                                \n"
    "10:                                       \n"
    // Load B row (4 vectors = 16 cols)
    "ldr     q1, [x22]                         \n"
    "ldr     q2, [x22, #16]                    \n"
    "ldr     q3, [x22, #32]                    \n"
    "ldr     q4, [x22, #48]                    \n"
    "add     x22, x22, x16                     \n"  // B += ldb

    // Load A scalars and compute
    "ldr     s0, [x6], #4                      \n"  // A[row0][k]
    "fmla    v5.4s,  v1.4s, v0.s[0]            \n"
    "fmla    v6.4s,  v2.4s, v0.s[0]            \n"
    "fmla    v7.4s,  v3.4s, v0.s[0]            \n"
    "fmla    v8.4s,  v4.4s, v0.s[0]            \n"

    "ldr     s0, [x7], #4                      \n"  // A[row1][k]
    "fmla    v9.4s,  v1.4s, v0.s[0]            \n"
    "fmla    v10.4s, v2.4s, v0.s[0]            \n"
    "fmla    v11.4s, v3.4s, v0.s[0]            \n"
    "fmla    v12.4s, v4.4s, v0.s[0]            \n"

    "ldr     s0, [x8], #4                      \n"  // A[row2][k]
    "fmla    v13.4s, v1.4s, v0.s[0]            \n"
    "fmla    v14.4s, v2.4s, v0.s[0]            \n"
    "fmla    v15.4s, v3.4s, v0.s[0]            \n"
    "fmla    v16.4s, v4.4s, v0.s[0]            \n"

    "ldr     s0, [x9], #4                      \n"  // A[row3][k]
    "fmla    v17.4s, v1.4s, v0.s[0]            \n"
    "fmla    v18.4s, v2.4s, v0.s[0]            \n"
    "fmla    v19.4s, v3.4s, v0.s[0]            \n"
    "fmla    v20.4s, v4.4s, v0.s[0]            \n"

    "ldr     s0, [x10], #4                     \n"  // A[row4][k]
    "fmla    v21.4s, v1.4s, v0.s[0]            \n"
    "fmla    v22.4s, v2.4s, v0.s[0]            \n"
    "fmla    v23.4s, v3.4s, v0.s[0]            \n"
    "fmla    v24.4s, v4.4s, v0.s[0]            \n"

    "ldr     s0, [x11], #4                     \n"  // A[row5][k]
    "fmla    v25.4s, v1.4s, v0.s[0]            \n"
    "fmla    v26.4s, v2.4s, v0.s[0]            \n"
    "fmla    v27.4s, v3.4s, v0.s[0]            \n"
    "fmla    v28.4s, v4.4s, v0.s[0]            \n"

    "subs    w23, w23, #1                      \n"
    "bne     10b                               \n"

    // ================================================================
    // Epilogue: store C = alpha * acc + beta * C
    // ================================================================
    "30:                                       \n"
    "ld1r    {v0.4s}, [%[alpha_ptr]]           \n"  // broadcast alpha
    "ldr     s29, [%[beta_ptr]]                \n"
    "fcmp    s29, #0.0                         \n"
    "beq     40f                               \n"  // beta == 0

    // --- beta != 0 ---
    "dup     v29.4s, v29.s[0]                  \n"  // broadcast beta

    // Row 0 (acc in v5-v8, C at x12)
    "ldr     q1, [x12]     \n"  "ldr     q2, [x12, #16] \n"
    "ldr     q3, [x12, #32]\n"  "ldr     q4, [x12, #48] \n"
    "fmul    v5.4s,  v5.4s,  v0.4s \n"  "fmla    v5.4s,  v1.4s, v29.4s \n"
    "fmul    v6.4s,  v6.4s,  v0.4s \n"  "fmla    v6.4s,  v2.4s, v29.4s \n"
    "fmul    v7.4s,  v7.4s,  v0.4s \n"  "fmla    v7.4s,  v3.4s, v29.4s \n"
    "fmul    v8.4s,  v8.4s,  v0.4s \n"  "fmla    v8.4s,  v4.4s, v29.4s \n"
    "str     q5,  [x12]     \n"  "str     q6,  [x12, #16] \n"
    "str     q7,  [x12, #32]\n"  "str     q8,  [x12, #48] \n"

    // Row 1 (acc in v9-v12, C at x13)
    "ldr     q1, [x13]     \n"  "ldr     q2, [x13, #16] \n"
    "ldr     q3, [x13, #32]\n"  "ldr     q4, [x13, #48] \n"
    "fmul    v9.4s,  v9.4s,  v0.4s \n"  "fmla    v9.4s,  v1.4s, v29.4s \n"
    "fmul    v10.4s, v10.4s, v0.4s \n"  "fmla    v10.4s, v2.4s, v29.4s \n"
    "fmul    v11.4s, v11.4s, v0.4s \n"  "fmla    v11.4s, v3.4s, v29.4s \n"
    "fmul    v12.4s, v12.4s, v0.4s \n"  "fmla    v12.4s, v4.4s, v29.4s \n"
    "str     q9,  [x13]     \n"  "str     q10, [x13, #16] \n"
    "str     q11, [x13, #32]\n"  "str     q12, [x13, #48] \n"

    // Row 2 (acc in v13-v16, C at x14)
    "ldr     q1, [x14]     \n"  "ldr     q2, [x14, #16] \n"
    "ldr     q3, [x14, #32]\n"  "ldr     q4, [x14, #48] \n"
    "fmul    v13.4s, v13.4s, v0.4s \n"  "fmla    v13.4s, v1.4s, v29.4s \n"
    "fmul    v14.4s, v14.4s, v0.4s \n"  "fmla    v14.4s, v2.4s, v29.4s \n"
    "fmul    v15.4s, v15.4s, v0.4s \n"  "fmla    v15.4s, v3.4s, v29.4s \n"
    "fmul    v16.4s, v16.4s, v0.4s \n"  "fmla    v16.4s, v4.4s, v29.4s \n"
    "str     q13, [x14]     \n"  "str     q14, [x14, #16] \n"
    "str     q15, [x14, #32]\n"  "str     q16, [x14, #48] \n"

    // Row 3 (acc in v17-v20, C at x19)
    "ldr     q1, [x19]     \n"  "ldr     q2, [x19, #16] \n"
    "ldr     q3, [x19, #32]\n"  "ldr     q4, [x19, #48] \n"
    "fmul    v17.4s, v17.4s, v0.4s \n"  "fmla    v17.4s, v1.4s, v29.4s \n"
    "fmul    v18.4s, v18.4s, v0.4s \n"  "fmla    v18.4s, v2.4s, v29.4s \n"
    "fmul    v19.4s, v19.4s, v0.4s \n"  "fmla    v19.4s, v3.4s, v29.4s \n"
    "fmul    v20.4s, v20.4s, v0.4s \n"  "fmla    v20.4s, v4.4s, v29.4s \n"
    "str     q17, [x19]     \n"  "str     q18, [x19, #16] \n"
    "str     q19, [x19, #32]\n"  "str     q20, [x19, #48] \n"

    // Row 4 (acc in v21-v24, C at x20)
    "ldr     q1, [x20]     \n"  "ldr     q2, [x20, #16] \n"
    "ldr     q3, [x20, #32]\n"  "ldr     q4, [x20, #48] \n"
    "fmul    v21.4s, v21.4s, v0.4s \n"  "fmla    v21.4s, v1.4s, v29.4s \n"
    "fmul    v22.4s, v22.4s, v0.4s \n"  "fmla    v22.4s, v2.4s, v29.4s \n"
    "fmul    v23.4s, v23.4s, v0.4s \n"  "fmla    v23.4s, v3.4s, v29.4s \n"
    "fmul    v24.4s, v24.4s, v0.4s \n"  "fmla    v24.4s, v4.4s, v29.4s \n"
    "str     q21, [x20]     \n"  "str     q22, [x20, #16] \n"
    "str     q23, [x20, #32]\n"  "str     q24, [x20, #48] \n"

    // Row 5 (acc in v25-v28, C at x21)
    "ldr     q1, [x21]     \n"  "ldr     q2, [x21, #16] \n"
    "ldr     q3, [x21, #32]\n"  "ldr     q4, [x21, #48] \n"
    "fmul    v25.4s, v25.4s, v0.4s \n"  "fmla    v25.4s, v1.4s, v29.4s \n"
    "fmul    v26.4s, v26.4s, v0.4s \n"  "fmla    v26.4s, v2.4s, v29.4s \n"
    "fmul    v27.4s, v27.4s, v0.4s \n"  "fmla    v27.4s, v3.4s, v29.4s \n"
    "fmul    v28.4s, v28.4s, v0.4s \n"  "fmla    v28.4s, v4.4s, v29.4s \n"
    "str     q25, [x21]     \n"  "str     q26, [x21, #16] \n"
    "str     q27, [x21, #32]\n"  "str     q28, [x21, #48] \n"
    "b       50f                               \n"

    // --- beta == 0 ---
    "40:                                       \n"
    "fmul    v5.4s,  v5.4s,  v0.4s \n"  "fmul    v6.4s,  v6.4s,  v0.4s \n"
    "fmul    v7.4s,  v7.4s,  v0.4s \n"  "fmul    v8.4s,  v8.4s,  v0.4s \n"
    "str     q5,  [x12]     \n"  "str     q6,  [x12, #16] \n"
    "str     q7,  [x12, #32]\n"  "str     q8,  [x12, #48] \n"

    "fmul    v9.4s,  v9.4s,  v0.4s \n"  "fmul    v10.4s, v10.4s, v0.4s \n"
    "fmul    v11.4s, v11.4s, v0.4s \n"  "fmul    v12.4s, v12.4s, v0.4s \n"
    "str     q9,  [x13]     \n"  "str     q10, [x13, #16] \n"
    "str     q11, [x13, #32]\n"  "str     q12, [x13, #48] \n"

    "fmul    v13.4s, v13.4s, v0.4s \n"  "fmul    v14.4s, v14.4s, v0.4s \n"
    "fmul    v15.4s, v15.4s, v0.4s \n"  "fmul    v16.4s, v16.4s, v0.4s \n"
    "str     q13, [x14]     \n"  "str     q14, [x14, #16] \n"
    "str     q15, [x14, #32]\n"  "str     q16, [x14, #48] \n"

    "fmul    v17.4s, v17.4s, v0.4s \n"  "fmul    v18.4s, v18.4s, v0.4s \n"
    "fmul    v19.4s, v19.4s, v0.4s \n"  "fmul    v20.4s, v20.4s, v0.4s \n"
    "str     q17, [x19]     \n"  "str     q18, [x19, #16] \n"
    "str     q19, [x19, #32]\n"  "str     q20, [x19, #48] \n"

    "fmul    v21.4s, v21.4s, v0.4s \n"  "fmul    v22.4s, v22.4s, v0.4s \n"
    "fmul    v23.4s, v23.4s, v0.4s \n"  "fmul    v24.4s, v24.4s, v0.4s \n"
    "str     q21, [x20]     \n"  "str     q22, [x20, #16] \n"
    "str     q23, [x20, #32]\n"  "str     q24, [x20, #48] \n"

    "fmul    v25.4s, v25.4s, v0.4s \n"  "fmul    v26.4s, v26.4s, v0.4s \n"
    "fmul    v27.4s, v27.4s, v0.4s \n"  "fmul    v28.4s, v28.4s, v0.4s \n"
    "str     q25, [x21]     \n"  "str     q26, [x21, #16] \n"
    "str     q27, [x21, #32]\n"  "str     q28, [x21, #48] \n"

    "50:                                       \n"

    : [lda] "+r" (lda_val),
      [ldb] "+r" (ldb_val),
      [ldc] "+r" (ldc_val)
    : [A] "r" (A),
      [B] "r" (B),
      [C] "r" (C),
      [K] "r" (K),
      [alpha_ptr] "r" (&alpha),
      [beta_ptr] "r" (&beta)
    : "cc", "memory",
      "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
      "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
      "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
      "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
      "v24", "v25", "v26", "v27", "v28", "v29"
    );
}

// ============================================================
// 3×16 and 5×16 tail kernels (simpler, use scalar A loads)
// ============================================================

void gemm_kernel_3x16_asm(int K,
                           const float* __restrict__ A, int lda,
                           const float* __restrict__ B, int ldb,
                           float* __restrict__ C, int ldc,
                           float alpha, float beta) {
    int lda_val = lda, ldb_val = ldb, ldc_val = ldc;

    asm volatile(
    "lsl     %w[lda], %w[lda], #2              \n"
    "lsl     %w[ldb], %w[ldb], #2              \n"
    "lsl     %w[ldc], %w[ldc], #2              \n"
    "sxtw    x15, %w[lda]                      \n"
    "sxtw    x16, %w[ldb]                      \n"
    "sxtw    x17, %w[ldc]                      \n"

    // A row pointers
    "mov     x6,  %[A]                         \n"
    "add     x7,  x6, x15                      \n"
    "add     x8,  x6, x15, lsl #1              \n"

    // C row pointers
    "mov     x9,  %[C]                         \n"
    "add     x10, x9, x17                      \n"
    "add     x11, x9, x17, lsl #1              \n"

    // B pointer
    "mov     x22, %[B]                         \n"

    // Zero 12 accumulators (v5-v16)
    "movi    v5.4s,  #0 \n"  "movi    v6.4s,  #0 \n"
    "movi    v7.4s,  #0 \n"  "movi    v8.4s,  #0 \n"
    "movi    v9.4s,  #0 \n"  "movi    v10.4s, #0 \n"
    "movi    v11.4s, #0 \n"  "movi    v12.4s, #0 \n"
    "movi    v13.4s, #0 \n"  "movi    v14.4s, #0 \n"
    "movi    v15.4s, #0 \n"  "movi    v16.4s, #0 \n"

    "mov     w23, %w[K]                        \n"
    "cbz     w23, 30f                          \n"

    ".p2align 4                                \n"
    "10:                                       \n"
    "ldr     q1, [x22]      \n"  "ldr     q2, [x22, #16] \n"
    "ldr     q3, [x22, #32] \n"  "ldr     q4, [x22, #48] \n"
    "add     x22, x22, x16                     \n"

    "ldr     s0, [x6], #4                      \n"
    "fmla    v5.4s,  v1.4s, v0.s[0]            \n"
    "fmla    v6.4s,  v2.4s, v0.s[0]            \n"
    "fmla    v7.4s,  v3.4s, v0.s[0]            \n"
    "fmla    v8.4s,  v4.4s, v0.s[0]            \n"

    "ldr     s0, [x7], #4                      \n"
    "fmla    v9.4s,  v1.4s, v0.s[0]            \n"
    "fmla    v10.4s, v2.4s, v0.s[0]            \n"
    "fmla    v11.4s, v3.4s, v0.s[0]            \n"
    "fmla    v12.4s, v4.4s, v0.s[0]            \n"

    "ldr     s0, [x8], #4                      \n"
    "fmla    v13.4s, v1.4s, v0.s[0]            \n"
    "fmla    v14.4s, v2.4s, v0.s[0]            \n"
    "fmla    v15.4s, v3.4s, v0.s[0]            \n"
    "fmla    v16.4s, v4.4s, v0.s[0]            \n"

    "subs    w23, w23, #1                      \n"
    "bne     10b                               \n"

    "30:                                       \n"
    "ld1r    {v0.4s}, [%[alpha_ptr]]           \n"
    "ldr     s29, [%[beta_ptr]]                \n"
    "fcmp    s29, #0.0                         \n"
    "beq     40f                               \n"

    "dup     v29.4s, v29.s[0]                  \n"

    // Row 0
    "ldr     q1,[x9] \n" "ldr     q2,[x9,#16] \n" "ldr     q3,[x9,#32] \n" "ldr     q4,[x9,#48] \n"
    "fmul    v5.4s,v5.4s,v0.4s \n" "fmla    v5.4s,v1.4s,v29.4s \n"
    "fmul    v6.4s,v6.4s,v0.4s \n" "fmla    v6.4s,v2.4s,v29.4s \n"
    "fmul    v7.4s,v7.4s,v0.4s \n" "fmla    v7.4s,v3.4s,v29.4s \n"
    "fmul    v8.4s,v8.4s,v0.4s \n" "fmla    v8.4s,v4.4s,v29.4s \n"
    "str     q5,[x9] \n" "str     q6,[x9,#16] \n" "str     q7,[x9,#32] \n" "str     q8,[x9,#48] \n"

    // Row 1
    "ldr     q1,[x10] \n" "ldr     q2,[x10,#16] \n" "ldr     q3,[x10,#32] \n" "ldr     q4,[x10,#48] \n"
    "fmul    v9.4s,v9.4s,v0.4s \n" "fmla    v9.4s,v1.4s,v29.4s \n"
    "fmul    v10.4s,v10.4s,v0.4s \n" "fmla    v10.4s,v2.4s,v29.4s \n"
    "fmul    v11.4s,v11.4s,v0.4s \n" "fmla    v11.4s,v3.4s,v29.4s \n"
    "fmul    v12.4s,v12.4s,v0.4s \n" "fmla    v12.4s,v4.4s,v29.4s \n"
    "str     q9,[x10] \n" "str     q10,[x10,#16] \n" "str     q11,[x10,#32] \n" "str     q12,[x10,#48] \n"

    // Row 2
    "ldr     q1,[x11] \n" "ldr     q2,[x11,#16] \n" "ldr     q3,[x11,#32] \n" "ldr     q4,[x11,#48] \n"
    "fmul    v13.4s,v13.4s,v0.4s \n" "fmla    v13.4s,v1.4s,v29.4s \n"
    "fmul    v14.4s,v14.4s,v0.4s \n" "fmla    v14.4s,v2.4s,v29.4s \n"
    "fmul    v15.4s,v15.4s,v0.4s \n" "fmla    v15.4s,v3.4s,v29.4s \n"
    "fmul    v16.4s,v16.4s,v0.4s \n" "fmla    v16.4s,v4.4s,v29.4s \n"
    "str     q13,[x11] \n" "str     q14,[x11,#16] \n" "str     q15,[x11,#32] \n" "str     q16,[x11,#48] \n"
    "b       50f                               \n"

    "40:                                       \n"
    "fmul    v5.4s,v5.4s,v0.4s \n" "fmul    v6.4s,v6.4s,v0.4s \n"
    "fmul    v7.4s,v7.4s,v0.4s \n" "fmul    v8.4s,v8.4s,v0.4s \n"
    "str     q5,[x9] \n" "str     q6,[x9,#16] \n" "str     q7,[x9,#32] \n" "str     q8,[x9,#48] \n"
    "fmul    v9.4s,v9.4s,v0.4s \n" "fmul    v10.4s,v10.4s,v0.4s \n"
    "fmul    v11.4s,v11.4s,v0.4s \n" "fmul    v12.4s,v12.4s,v0.4s \n"
    "str     q9,[x10] \n" "str     q10,[x10,#16] \n" "str     q11,[x10,#32] \n" "str     q12,[x10,#48] \n"
    "fmul    v13.4s,v13.4s,v0.4s \n" "fmul    v14.4s,v14.4s,v0.4s \n"
    "fmul    v15.4s,v15.4s,v0.4s \n" "fmul    v16.4s,v16.4s,v0.4s \n"
    "str     q13,[x11] \n" "str     q14,[x11,#16] \n" "str     q15,[x11,#32] \n" "str     q16,[x11,#48] \n"

    "50:                                       \n"
    : [lda] "+r" (lda_val), [ldb] "+r" (ldb_val), [ldc] "+r" (ldc_val)
    : [A] "r" (A), [B] "r" (B), [C] "r" (C), [K] "r" (K),
      [alpha_ptr] "r" (&alpha), [beta_ptr] "r" (&beta)
    : "cc", "memory",
      "x6", "x7", "x8", "x9", "x10", "x11", "x15", "x16", "x17", "x22", "x23",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
      "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v29"
    );
}

void gemm_kernel_5x16_asm(int K,
                           const float* __restrict__ A, int lda,
                           const float* __restrict__ B, int ldb,
                           float* __restrict__ C, int ldc,
                           float alpha, float beta) {
    int lda_val = lda, ldb_val = ldb, ldc_val = ldc;

    asm volatile(
    "lsl     %w[lda], %w[lda], #2              \n"
    "lsl     %w[ldb], %w[ldb], #2              \n"
    "lsl     %w[ldc], %w[ldc], #2              \n"
    "sxtw    x15, %w[lda]                      \n"
    "sxtw    x16, %w[ldb]                      \n"
    "sxtw    x17, %w[ldc]                      \n"

    // A row pointers (5 rows)
    "mov     x6,  %[A]                         \n"
    "add     x7,  x6,  x15                     \n"
    "add     x8,  x6,  x15, lsl #1             \n"
    "add     x9,  x7,  x15, lsl #1             \n"
    "add     x10, x8,  x15, lsl #1             \n"

    // C row pointers (5 rows)
    "mov     x11, %[C]                         \n"
    "add     x12, x11, x17                     \n"
    "add     x13, x11, x17, lsl #1             \n"
    "add     x14, x12, x17, lsl #1             \n"
    "add     x19, x13, x17, lsl #1             \n"

    // B pointer
    "mov     x22, %[B]                         \n"

    // Zero 20 accumulators (v5-v24)
    "movi    v5.4s,  #0 \n"  "movi    v6.4s,  #0 \n"
    "movi    v7.4s,  #0 \n"  "movi    v8.4s,  #0 \n"
    "movi    v9.4s,  #0 \n"  "movi    v10.4s, #0 \n"
    "movi    v11.4s, #0 \n"  "movi    v12.4s, #0 \n"
    "movi    v13.4s, #0 \n"  "movi    v14.4s, #0 \n"
    "movi    v15.4s, #0 \n"  "movi    v16.4s, #0 \n"
    "movi    v17.4s, #0 \n"  "movi    v18.4s, #0 \n"
    "movi    v19.4s, #0 \n"  "movi    v20.4s, #0 \n"
    "movi    v21.4s, #0 \n"  "movi    v22.4s, #0 \n"
    "movi    v23.4s, #0 \n"  "movi    v24.4s, #0 \n"

    "mov     w23, %w[K]                        \n"
    "cbz     w23, 30f                          \n"

    ".p2align 4                                \n"
    "10:                                       \n"
    "ldr     q1, [x22]      \n"  "ldr     q2, [x22, #16] \n"
    "ldr     q3, [x22, #32] \n"  "ldr     q4, [x22, #48] \n"
    "add     x22, x22, x16                     \n"

    "ldr     s0, [x6], #4 \n"
    "fmla    v5.4s,v1.4s,v0.s[0] \n" "fmla    v6.4s,v2.4s,v0.s[0] \n"
    "fmla    v7.4s,v3.4s,v0.s[0] \n" "fmla    v8.4s,v4.4s,v0.s[0] \n"

    "ldr     s0, [x7], #4 \n"
    "fmla    v9.4s,v1.4s,v0.s[0] \n" "fmla    v10.4s,v2.4s,v0.s[0] \n"
    "fmla    v11.4s,v3.4s,v0.s[0] \n" "fmla    v12.4s,v4.4s,v0.s[0] \n"

    "ldr     s0, [x8], #4 \n"
    "fmla    v13.4s,v1.4s,v0.s[0] \n" "fmla    v14.4s,v2.4s,v0.s[0] \n"
    "fmla    v15.4s,v3.4s,v0.s[0] \n" "fmla    v16.4s,v4.4s,v0.s[0] \n"

    "ldr     s0, [x9], #4 \n"
    "fmla    v17.4s,v1.4s,v0.s[0] \n" "fmla    v18.4s,v2.4s,v0.s[0] \n"
    "fmla    v19.4s,v3.4s,v0.s[0] \n" "fmla    v20.4s,v4.4s,v0.s[0] \n"

    "ldr     s0, [x10], #4 \n"
    "fmla    v21.4s,v1.4s,v0.s[0] \n" "fmla    v22.4s,v2.4s,v0.s[0] \n"
    "fmla    v23.4s,v3.4s,v0.s[0] \n" "fmla    v24.4s,v4.4s,v0.s[0] \n"

    "subs    w23, w23, #1                      \n"
    "bne     10b                               \n"

    "30:                                       \n"
    "ld1r    {v0.4s}, [%[alpha_ptr]]           \n"
    "ldr     s29, [%[beta_ptr]]                \n"
    "fcmp    s29, #0.0                         \n"
    "beq     40f                               \n"

    "dup     v29.4s, v29.s[0]                  \n"

    // Rows 0-4 beta!=0
    "ldr q1,[x11] \n" "ldr q2,[x11,#16] \n" "ldr q3,[x11,#32] \n" "ldr q4,[x11,#48] \n"
    "fmul v5.4s,v5.4s,v0.4s \n" "fmla v5.4s,v1.4s,v29.4s \n"
    "fmul v6.4s,v6.4s,v0.4s \n" "fmla v6.4s,v2.4s,v29.4s \n"
    "fmul v7.4s,v7.4s,v0.4s \n" "fmla v7.4s,v3.4s,v29.4s \n"
    "fmul v8.4s,v8.4s,v0.4s \n" "fmla v8.4s,v4.4s,v29.4s \n"
    "str q5,[x11] \n" "str q6,[x11,#16] \n" "str q7,[x11,#32] \n" "str q8,[x11,#48] \n"

    "ldr q1,[x12] \n" "ldr q2,[x12,#16] \n" "ldr q3,[x12,#32] \n" "ldr q4,[x12,#48] \n"
    "fmul v9.4s,v9.4s,v0.4s \n" "fmla v9.4s,v1.4s,v29.4s \n"
    "fmul v10.4s,v10.4s,v0.4s \n" "fmla v10.4s,v2.4s,v29.4s \n"
    "fmul v11.4s,v11.4s,v0.4s \n" "fmla v11.4s,v3.4s,v29.4s \n"
    "fmul v12.4s,v12.4s,v0.4s \n" "fmla v12.4s,v4.4s,v29.4s \n"
    "str q9,[x12] \n" "str q10,[x12,#16] \n" "str q11,[x12,#32] \n" "str q12,[x12,#48] \n"

    "ldr q1,[x13] \n" "ldr q2,[x13,#16] \n" "ldr q3,[x13,#32] \n" "ldr q4,[x13,#48] \n"
    "fmul v13.4s,v13.4s,v0.4s \n" "fmla v13.4s,v1.4s,v29.4s \n"
    "fmul v14.4s,v14.4s,v0.4s \n" "fmla v14.4s,v2.4s,v29.4s \n"
    "fmul v15.4s,v15.4s,v0.4s \n" "fmla v15.4s,v3.4s,v29.4s \n"
    "fmul v16.4s,v16.4s,v0.4s \n" "fmla v16.4s,v4.4s,v29.4s \n"
    "str q13,[x13] \n" "str q14,[x13,#16] \n" "str q15,[x13,#32] \n" "str q16,[x13,#48] \n"

    "ldr q1,[x14] \n" "ldr q2,[x14,#16] \n" "ldr q3,[x14,#32] \n" "ldr q4,[x14,#48] \n"
    "fmul v17.4s,v17.4s,v0.4s \n" "fmla v17.4s,v1.4s,v29.4s \n"
    "fmul v18.4s,v18.4s,v0.4s \n" "fmla v18.4s,v2.4s,v29.4s \n"
    "fmul v19.4s,v19.4s,v0.4s \n" "fmla v19.4s,v3.4s,v29.4s \n"
    "fmul v20.4s,v20.4s,v0.4s \n" "fmla v20.4s,v4.4s,v29.4s \n"
    "str q17,[x14] \n" "str q18,[x14,#16] \n" "str q19,[x14,#32] \n" "str q20,[x14,#48] \n"

    "ldr q1,[x19] \n" "ldr q2,[x19,#16] \n" "ldr q3,[x19,#32] \n" "ldr q4,[x19,#48] \n"
    "fmul v21.4s,v21.4s,v0.4s \n" "fmla v21.4s,v1.4s,v29.4s \n"
    "fmul v22.4s,v22.4s,v0.4s \n" "fmla v22.4s,v2.4s,v29.4s \n"
    "fmul v23.4s,v23.4s,v0.4s \n" "fmla v23.4s,v3.4s,v29.4s \n"
    "fmul v24.4s,v24.4s,v0.4s \n" "fmla v24.4s,v4.4s,v29.4s \n"
    "str q21,[x19] \n" "str q22,[x19,#16] \n" "str q23,[x19,#32] \n" "str q24,[x19,#48] \n"
    "b       50f                               \n"

    "40:                                       \n"
    "fmul v5.4s,v5.4s,v0.4s \n" "fmul v6.4s,v6.4s,v0.4s \n"
    "fmul v7.4s,v7.4s,v0.4s \n" "fmul v8.4s,v8.4s,v0.4s \n"
    "str q5,[x11] \n" "str q6,[x11,#16] \n" "str q7,[x11,#32] \n" "str q8,[x11,#48] \n"
    "fmul v9.4s,v9.4s,v0.4s \n" "fmul v10.4s,v10.4s,v0.4s \n"
    "fmul v11.4s,v11.4s,v0.4s \n" "fmul v12.4s,v12.4s,v0.4s \n"
    "str q9,[x12] \n" "str q10,[x12,#16] \n" "str q11,[x12,#32] \n" "str q12,[x12,#48] \n"
    "fmul v13.4s,v13.4s,v0.4s \n" "fmul v14.4s,v14.4s,v0.4s \n"
    "fmul v15.4s,v15.4s,v0.4s \n" "fmul v16.4s,v16.4s,v0.4s \n"
    "str q13,[x13] \n" "str q14,[x13,#16] \n" "str q15,[x13,#32] \n" "str q16,[x13,#48] \n"
    "fmul v17.4s,v17.4s,v0.4s \n" "fmul v18.4s,v18.4s,v0.4s \n"
    "fmul v19.4s,v19.4s,v0.4s \n" "fmul v20.4s,v20.4s,v0.4s \n"
    "str q17,[x14] \n" "str q18,[x14,#16] \n" "str q19,[x14,#32] \n" "str q20,[x14,#48] \n"
    "fmul v21.4s,v21.4s,v0.4s \n" "fmul v22.4s,v22.4s,v0.4s \n"
    "fmul v23.4s,v23.4s,v0.4s \n" "fmul v24.4s,v24.4s,v0.4s \n"
    "str q21,[x19] \n" "str q22,[x19,#16] \n" "str q23,[x19,#32] \n" "str q24,[x19,#48] \n"

    "50:                                       \n"
    : [lda] "+r" (lda_val), [ldb] "+r" (ldb_val), [ldc] "+r" (ldc_val)
    : [A] "r" (A), [B] "r" (B), [C] "r" (C), [K] "r" (K),
      [alpha_ptr] "r" (&alpha), [beta_ptr] "r" (&beta)
    : "cc", "memory",
      "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
      "x15", "x16", "x17", "x19", "x22", "x23",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
      "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
      "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v29"
    );
}

}  // namespace dnnopt

#endif  // __aarch64__
