/// @file gemm_adaptive_tile_fp32.cpp
/// Phase 8: Adaptive tile GEMM — autoGEMM-style dynamic tile assembly.
///
/// Key insight: fixed 8x12 tile wastes N-tail for N=64/128 shapes.
/// Nr=16 gives zero N-tail for N=64 (64/16=4) and N=128 (128/16=8).
/// Hand-specialized kernels for critical (Mr,Nr) pairs with named
/// accumulators to force register allocation (no stack spills).
///
/// Selected at runtime by scoring register utilization, shape
/// divisibility, and L1 cache fit.

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/arm_hwcaps.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// ============================================================
// Tile selection (autoGEMM scoring)
// ============================================================

TileConfig select_tile_fp32(int M, int N, int K, uint32_t l1d_bytes) {
    struct Candidate { int Mr, Nr; };
    static const Candidate candidates[] = {
        {4, 16}, {6, 16}, {4, 12}, {8, 12}, {8, 8}, {12, 8}, {16, 4}
    };

    TileConfig best = {8, 12};
    float best_score = -1.0f;

    for (const auto& c : candidates) {
        int Nr4 = c.Nr / 4;
        if (c.Mr * Nr4 + 3 > 30) continue;
        if (l1d_bytes > 0 && (int64_t)c.Mr * K * 4 > (int64_t)l1d_bytes * 4 / 5)
            continue;

        float reg_util = (float)(c.Mr * Nr4) / 27.0f;
        int m_full = (M / c.Mr) * c.Mr;
        int n_full = (N / c.Nr) * c.Nr;
        float m_eff = (float)m_full / (float)M;
        float n_eff = (float)n_full / (float)N;
        float n_bonus = (N % c.Nr == 0) ? 1.0f : 0.7f;

        float score = reg_util * m_eff * n_eff * n_bonus;
        if (score > best_score) {
            best_score = score;
            best = {c.Mr, c.Nr};
        }
    }
    return best;
}

// ============================================================
// Hand-specialized kernels with named accumulators
// ============================================================

/// Helper: store Mr rows × Nr4 quads, beta=0 fast path.
/// Accumulators passed by reference — GCC keeps them in registers.
template<int Mr, int Nr4>
static inline void store_acc_beta0(
    float* C, int ldc,
    float32x4_t av,
    const float32x4_t* acc) {
    for (int i = 0; i < Mr; i++) {
        float* cr = C + i * ldc;
        for (int j = 0; j < Nr4; j++) {
            vst1q_f32(cr + j * 4, vmulq_f32(av, acc[i * Nr4 + j]));
        }
    }
}

template<int Mr, int Nr4>
static inline void store_acc_beta1(
    float* C, int ldc,
    float32x4_t av, float32x4_t bvv,
    const float32x4_t* acc) {
    for (int i = 0; i < Mr; i++) {
        float* cr = C + i * ldc;
        for (int j = 0; j < Nr4; j++) {
            float32x4_t s = vmulq_f32(av, acc[i * Nr4 + j]);
            s = vfmaq_f32(s, bvv, vld1q_f32(cr + j * 4));
            vst1q_f32(cr + j * 4, s);
        }
    }
}

// ------------------------------------------------------------------
// 4x16 kernel: 16 acc (4 rows × 4 quads), fits in 19 registers
// ------------------------------------------------------------------
static void gemm_kernel_4x16(int K,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float* C, int ldc,
                              float alpha, float beta) {
    float32x4_t a00 = vdupq_n_f32(0), a01 = vdupq_n_f32(0),
                 a02 = vdupq_n_f32(0), a03 = vdupq_n_f32(0);
    float32x4_t a10 = vdupq_n_f32(0), a11 = vdupq_n_f32(0),
                 a12 = vdupq_n_f32(0), a13 = vdupq_n_f32(0);
    float32x4_t a20 = vdupq_n_f32(0), a21 = vdupq_n_f32(0),
                 a22 = vdupq_n_f32(0), a23 = vdupq_n_f32(0);
    float32x4_t a30 = vdupq_n_f32(0), a31 = vdupq_n_f32(0),
                 a32 = vdupq_n_f32(0), a33 = vdupq_n_f32(0);

    const float *a0 = A, *a1 = A + lda, *a2 = A + 2*lda, *a3 = A + 3*lda;

    for (int k = 0; k < K; ++k) {
        const float* bk = B + k * ldb;
        float32x4_t b0 = vld1q_f32(bk);
        float32x4_t b1 = vld1q_f32(bk + 4);
        float32x4_t b2 = vld1q_f32(bk + 8);
        float32x4_t b3 = vld1q_f32(bk + 12);

        a00 = vfmaq_n_f32(a00, b0, a0[k]);
        a01 = vfmaq_n_f32(a01, b1, a0[k]);
        a02 = vfmaq_n_f32(a02, b2, a0[k]);
        a03 = vfmaq_n_f32(a03, b3, a0[k]);

        a10 = vfmaq_n_f32(a10, b0, a1[k]);
        a11 = vfmaq_n_f32(a11, b1, a1[k]);
        a12 = vfmaq_n_f32(a12, b2, a1[k]);
        a13 = vfmaq_n_f32(a13, b3, a1[k]);

        a20 = vfmaq_n_f32(a20, b0, a2[k]);
        a21 = vfmaq_n_f32(a21, b1, a2[k]);
        a22 = vfmaq_n_f32(a22, b2, a2[k]);
        a23 = vfmaq_n_f32(a23, b3, a2[k]);

        a30 = vfmaq_n_f32(a30, b0, a3[k]);
        a31 = vfmaq_n_f32(a31, b1, a3[k]);
        a32 = vfmaq_n_f32(a32, b2, a3[k]);
        a33 = vfmaq_n_f32(a33, b3, a3[k]);
    }

    float32x4_t av = vdupq_n_f32(alpha);
    float *c0 = C, *c1 = C + ldc, *c2 = C + 2*ldc, *c3 = C + 3*ldc;

    if (beta == 0.0f) {
        vst1q_f32(c0,    vmulq_f32(av, a00)); vst1q_f32(c0+4,  vmulq_f32(av, a01));
        vst1q_f32(c0+8,  vmulq_f32(av, a02)); vst1q_f32(c0+12, vmulq_f32(av, a03));
        vst1q_f32(c1,    vmulq_f32(av, a10)); vst1q_f32(c1+4,  vmulq_f32(av, a11));
        vst1q_f32(c1+8,  vmulq_f32(av, a12)); vst1q_f32(c1+12, vmulq_f32(av, a13));
        vst1q_f32(c2,    vmulq_f32(av, a20)); vst1q_f32(c2+4,  vmulq_f32(av, a21));
        vst1q_f32(c2+8,  vmulq_f32(av, a22)); vst1q_f32(c2+12, vmulq_f32(av, a23));
        vst1q_f32(c3,    vmulq_f32(av, a30)); vst1q_f32(c3+4,  vmulq_f32(av, a31));
        vst1q_f32(c3+8,  vmulq_f32(av, a32)); vst1q_f32(c3+12, vmulq_f32(av, a33));
    } else {
        float32x4_t bvv = vdupq_n_f32(beta);
        #define STORE_R(r, a0n, a1n, a2n, a3n) do { \
            float* cr = c##r; \
            float32x4_t s0 = vfmaq_f32(vmulq_f32(av, a0n), bvv, vld1q_f32(cr)); \
            float32x4_t s1 = vfmaq_f32(vmulq_f32(av, a1n), bvv, vld1q_f32(cr+4)); \
            float32x4_t s2 = vfmaq_f32(vmulq_f32(av, a2n), bvv, vld1q_f32(cr+8)); \
            float32x4_t s3 = vfmaq_f32(vmulq_f32(av, a3n), bvv, vld1q_f32(cr+12)); \
            vst1q_f32(cr, s0); vst1q_f32(cr+4, s1); \
            vst1q_f32(cr+8, s2); vst1q_f32(cr+12, s3); \
        } while(0)
        STORE_R(0, a00, a01, a02, a03);
        STORE_R(1, a10, a11, a12, a13);
        STORE_R(2, a20, a21, a22, a23);
        STORE_R(3, a30, a31, a32, a33);
        #undef STORE_R
    }
}

// ------------------------------------------------------------------
// 6x16 kernel: 24 acc (6 rows × 4 quads), fits in 27 registers
// ------------------------------------------------------------------
static void gemm_kernel_6x16(int K,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float* C, int ldc,
                              float alpha, float beta) {
    // 24 accumulators — 6 rows × 4 quads
    float32x4_t a00=vdupq_n_f32(0),a01=vdupq_n_f32(0),a02=vdupq_n_f32(0),a03=vdupq_n_f32(0);
    float32x4_t a10=vdupq_n_f32(0),a11=vdupq_n_f32(0),a12=vdupq_n_f32(0),a13=vdupq_n_f32(0);
    float32x4_t a20=vdupq_n_f32(0),a21=vdupq_n_f32(0),a22=vdupq_n_f32(0),a23=vdupq_n_f32(0);
    float32x4_t a30=vdupq_n_f32(0),a31=vdupq_n_f32(0),a32=vdupq_n_f32(0),a33=vdupq_n_f32(0);
    float32x4_t a40=vdupq_n_f32(0),a41=vdupq_n_f32(0),a42=vdupq_n_f32(0),a43=vdupq_n_f32(0);
    float32x4_t a50=vdupq_n_f32(0),a51=vdupq_n_f32(0),a52=vdupq_n_f32(0),a53=vdupq_n_f32(0);

    const float *a0=A, *a1=A+lda, *a2=A+2*lda, *a3=A+3*lda,
                *a4=A+4*lda, *a5=A+5*lda;

    for (int k = 0; k < K; ++k) {
        const float* bk = B + k * ldb;
        float32x4_t b0 = vld1q_f32(bk);
        float32x4_t b1 = vld1q_f32(bk + 4);
        float32x4_t b2 = vld1q_f32(bk + 8);
        float32x4_t b3 = vld1q_f32(bk + 12);

        a00=vfmaq_n_f32(a00,b0,a0[k]); a01=vfmaq_n_f32(a01,b1,a0[k]);
        a02=vfmaq_n_f32(a02,b2,a0[k]); a03=vfmaq_n_f32(a03,b3,a0[k]);

        a10=vfmaq_n_f32(a10,b0,a1[k]); a11=vfmaq_n_f32(a11,b1,a1[k]);
        a12=vfmaq_n_f32(a12,b2,a1[k]); a13=vfmaq_n_f32(a13,b3,a1[k]);

        a20=vfmaq_n_f32(a20,b0,a2[k]); a21=vfmaq_n_f32(a21,b1,a2[k]);
        a22=vfmaq_n_f32(a22,b2,a2[k]); a23=vfmaq_n_f32(a23,b3,a2[k]);

        a30=vfmaq_n_f32(a30,b0,a3[k]); a31=vfmaq_n_f32(a31,b1,a3[k]);
        a32=vfmaq_n_f32(a32,b2,a3[k]); a33=vfmaq_n_f32(a33,b3,a3[k]);

        a40=vfmaq_n_f32(a40,b0,a4[k]); a41=vfmaq_n_f32(a41,b1,a4[k]);
        a42=vfmaq_n_f32(a42,b2,a4[k]); a43=vfmaq_n_f32(a43,b3,a4[k]);

        a50=vfmaq_n_f32(a50,b0,a5[k]); a51=vfmaq_n_f32(a51,b1,a5[k]);
        a52=vfmaq_n_f32(a52,b2,a5[k]); a53=vfmaq_n_f32(a53,b3,a5[k]);
    }

    float32x4_t av = vdupq_n_f32(alpha);
    #define STORE_ROW6(r, a0n,a1n,a2n,a3n) do { \
        float* cr = C + (r)*ldc; \
        float32x4_t s0=vmulq_f32(av,a0n), s1=vmulq_f32(av,a1n), \
                    s2=vmulq_f32(av,a2n), s3=vmulq_f32(av,a3n); \
        if (beta != 0.0f) { \
            float32x4_t bv=vdupq_n_f32(beta); \
            s0=vfmaq_f32(s0,bv,vld1q_f32(cr)); s1=vfmaq_f32(s1,bv,vld1q_f32(cr+4)); \
            s2=vfmaq_f32(s2,bv,vld1q_f32(cr+8)); s3=vfmaq_f32(s3,bv,vld1q_f32(cr+12)); \
        } \
        vst1q_f32(cr,s0); vst1q_f32(cr+4,s1); vst1q_f32(cr+8,s2); vst1q_f32(cr+12,s3); \
    } while(0)
    STORE_ROW6(0, a00,a01,a02,a03);
    STORE_ROW6(1, a10,a11,a12,a13);
    STORE_ROW6(2, a20,a21,a22,a23);
    STORE_ROW6(3, a30,a31,a32,a33);
    STORE_ROW6(4, a40,a41,a42,a43);
    STORE_ROW6(5, a50,a51,a52,a53);
    #undef STORE_ROW6
}

// ------------------------------------------------------------------
// Template fallback for non-critical tiles (using array accumulators)
// ------------------------------------------------------------------
template<int Mr, int Nr>
static void gemm_tile_kernel_full(int K,
                                   const float* A, int lda,
                                   const float* B, int ldb,
                                   float* C, int ldc,
                                   float alpha, float beta) {
    constexpr int Nr4 = Nr / 4;
    constexpr int total_acc = Mr * Nr4;

    float32x4_t acc[total_acc];
    for (int x = 0; x < total_acc; x++) acc[x] = vdupq_n_f32(0);

    const float* a_row[Mr];
    for (int i = 0; i < Mr; i++) a_row[i] = A + i * lda;

    for (int k = 0; k < K; ++k) {
        const float* bk = B + k * ldb;
        for (int j = 0; j < Nr4; j++) {
            float32x4_t bv = vld1q_f32(bk + j * 4);
            for (int i = 0; i < Mr; i++) {
                acc[i * Nr4 + j] = vfmaq_n_f32(
                    acc[i * Nr4 + j], bv, a_row[i][k]);
            }
        }
    }

    float32x4_t av = vdupq_n_f32(alpha);
    if (beta == 0.0f) {
        for (int i = 0; i < Mr; i++) {
            float* cr = C + i * ldc;
            for (int j = 0; j < Nr4; j++)
                vst1q_f32(cr + j * 4, vmulq_f32(av, acc[i * Nr4 + j]));
        }
    } else {
        float32x4_t bvv = vdupq_n_f32(beta);
        for (int i = 0; i < Mr; i++) {
            float* cr = C + i * ldc;
            for (int j = 0; j < Nr4; j++) {
                float32x4_t s = vmulq_f32(av, acc[i * Nr4 + j]);
                s = vfmaq_f32(s, bvv, vld1q_f32(cr + j * 4));
                vst1q_f32(cr + j * 4, s);
            }
        }
    }
}

// Template instantiations for fallback tiles
template void gemm_tile_kernel_full<4, 12>(int, const float*, int, const float*, int,
                                           float*, int, float, float);
template void gemm_tile_kernel_full<8, 12>(int, const float*, int, const float*, int,
                                           float*, int, float, float);
template void gemm_tile_kernel_full<8,  8>(int, const float*, int, const float*, int,
                                           float*, int, float, float);
template void gemm_tile_kernel_full<12, 8>(int, const float*, int, const float*, int,
                                           float*, int, float, float);
template void gemm_tile_kernel_full<16, 4>(int, const float*, int, const float*, int,
                                           float*, int, float, float);

// ============================================================
// Dispatch table
// ============================================================

using TileFn = void(*)(int, const float*, int, const float*, int,
                       float*, int, float, float);

struct TileDispatch {
    int Mr, Nr;
    TileFn fn;
};

static const TileDispatch kTileDispatch[] = {
    { 4, 16, gemm_kernel_4x16 },
    { 6, 16, gemm_kernel_6x16 },
    { 4, 12, gemm_tile_kernel_full<4, 12> },
    { 8, 12, gemm_tile_kernel_full<8, 12> },
    { 8,  8, gemm_tile_kernel_full<8,  8> },
    {12,  8, gemm_tile_kernel_full<12, 8> },
    {16,  4, gemm_tile_kernel_full<16, 4> },
};

static constexpr int kNumTileDispatch =
    sizeof(kTileDispatch) / sizeof(kTileDispatch[0]);

// ============================================================
// M-tail handler (stack-backed accumulators for <Mr rows)
// ============================================================

static void gemm_tile_tail(int M, int N, int K,
                            float alpha, const float* A, int lda,
                            const float* B, int ldb,
                            float beta, float* C, int ldc) {
    int sub_full4 = (N / 4) * 4;
    int n_acc = (N + 3) / 4;
    float32x4_t* acc = reinterpret_cast<float32x4_t*>(
        __builtin_alloca(M * n_acc * sizeof(float32x4_t)));
    for (int x = 0; x < M * n_acc; x++) acc[x] = vdupq_n_f32(0);

    for (int k = 0; k < K; ++k) {
        const float* bk = B + k * ldb;
        for (int j = 0; j < sub_full4; j += 4) {
            float32x4_t bv = vld1q_f32(bk + j);
            int aidx = j / 4;
            for (int i = 0; i < M; ++i) {
                acc[i * n_acc + aidx] = vfmaq_n_f32(
                    acc[i * n_acc + aidx], bv, A[i * lda + k]);
            }
        }
    }

    float32x4_t av = vdupq_n_f32(alpha);
    bool beta_zero = (beta == 0.0f);
    for (int i = 0; i < M; ++i) {
        float* cr = C + i * ldc;
        for (int j = 0; j < sub_full4; j += 4) {
            int aidx = j / 4;
            float32x4_t s = vmulq_f32(av, acc[i * n_acc + aidx]);
            if (!beta_zero) {
                float32x4_t bvv = vdupq_n_f32(beta);
                s = vfmaq_f32(s, bvv, vld1q_f32(cr + j));
            }
            vst1q_f32(cr + j, s);
        }
        for (int j = sub_full4; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * lda + k] * B[k * ldb + j];
            cr[j] = beta_zero ? alpha * sum : alpha * sum + beta * cr[j];
        }
    }
}

// ============================================================
// Public driver
// ============================================================

void gemm_adaptive_tile_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc) {
    const auto& hw = detect_arm_hwcaps();
    auto tile = select_tile_fp32(M, N, K, hw.l1d.size_bytes);

    // Find dispatch function
    TileFn kernel_fn = nullptr;
    for (int d = 0; d < kNumTileDispatch; d++) {
        if (kTileDispatch[d].Mr == tile.Mr && kTileDispatch[d].Nr == tile.Nr) {
            kernel_fn = kTileDispatch[d].fn;
            break;
        }
    }
    if (!kernel_fn) kernel_fn = gemm_tile_kernel_full<8, 12>;

    int Mr = tile.Mr;
    int Nr = tile.Nr;

    // N outer loop: full Nr-wide panels
    int j_full = (N / Nr) * Nr;
    for (int j0 = 0; j0 < j_full; j0 += Nr) {
        int i = 0;
        for (; i + Mr - 1 < M; i += Mr) {
            kernel_fn(K, A + i * lda, lda,
                      B + j0, ldb,
                      C + i * ldc + j0, ldc,
                      alpha, beta);
        }
        if (i < M) {
            gemm_tile_tail(M - i, Nr, K, alpha,
                           A + i * lda, lda,
                           B + j0, ldb,
                           beta, C + i * ldc + j0, ldc);
        }
    }

    // N tail: partial panel
    if (j_full < N) {
        int n_len = N - j_full;
        int i = 0;
        for (; i + Mr - 1 < M; i += Mr) {
            gemm_tile_tail(Mr, n_len, K, alpha,
                           A + i * lda, lda,
                           B + j_full, ldb,
                           beta, C + i * ldc + j_full, ldc);
        }
        if (i < M) {
            gemm_tile_tail(M - i, n_len, K, alpha,
                           A + i * lda, lda,
                           B + j_full, ldb,
                           beta, C + i * ldc + j_full, ldc);
        }
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
