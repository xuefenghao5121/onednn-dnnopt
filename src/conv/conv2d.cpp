/// @file conv2d.cpp
/// Conv2D dispatch: im2col + GEMM for general convolutions,
/// direct GEMM for 1×1 pointwise convolutions.

#include "dnnopt/conv/conv.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"

#include <cstring>

namespace dnnopt {

// Forward declarations
void im2col_nhwc(const Conv2DParams& p, const float* input, float* col);
void apply_conv_postops(float* output, int num_rows, int OC,
                        const float* bias, ConvPostOp op);

/// Transpose filter from [OC, K] to [K, OC] for GEMM B matrix.
static void transpose_filter(const float* filter, float* filter_T,
                              int OC, int K) {
    for (int oc = 0; oc < OC; ++oc) {
        for (int k = 0; k < K; ++k) {
            filter_T[k * OC + oc] = filter[oc * K + k];
        }
    }
}

/// Direct 1×1 convolution: input [N*H*W, IC] × filter^T [IC, OC] → output [N*H*W, OC]
/// No im2col needed — NHWC input is already in GEMM-ready layout.
static void conv2d_1x1_direct(const Conv2DParams& p,
                               const float* input,
                               const float* filter,
                               const float* bias,
                               float* output,
                               ConvPostOp post_op) {
    const int M = p.N * p.IH * p.IW;  // For 1×1 s1 p0: OH=IH, OW=IW
    const int K = p.IC;
    const int N_gemm = p.OC;

    // Transpose filter once: [OC, IC] → [IC, OC]
    auto filter_T = aligned_array<float>((size_t)K * N_gemm);
    transpose_filter(filter, filter_T.get(), N_gemm, K);

    // GEMM: C[M, OC] = A[M, IC] × B[IC, OC]
    gemm_fp32(M, N_gemm, K,
              1.0f, input, K,
              filter_T.get(), N_gemm,
              0.0f, output, N_gemm);

    // Apply bias + post-ops
    if (bias || post_op != ConvPostOp::kNone) {
        apply_conv_postops(output, M, N_gemm, bias, post_op);
    }
}

/// im2col + GEMM convolution for general kernels.
static void conv2d_im2col_gemm(const Conv2DParams& p,
                                const float* input,
                                const float* filter,
                                const float* bias,
                                float* output,
                                ConvPostOp post_op) {
    const int OH = p.OH(), OW = p.OW();
    const int M = p.N * OH * OW;           // Output spatial elements
    const int K = p.IC * p.KH * p.KW;     // Flattened receptive field
    const int N_gemm = p.OC;

    // Allocate im2col buffer
    auto col = aligned_array<float>((size_t)M * K);

    // im2col: input [N,IH,IW,IC] → col [M, K]
    im2col_nhwc(p, input, col.get());

    // Transpose filter: [OC, K] → [K, OC]
    auto filter_T = aligned_array<float>((size_t)K * N_gemm);
    transpose_filter(filter, filter_T.get(), N_gemm, K);

    // GEMM: output[M, OC] = col[M, K] × filter_T[K, OC]
    gemm_fp32(M, N_gemm, K,
              1.0f, col.get(), K,
              filter_T.get(), N_gemm,
              0.0f, output, N_gemm);

    // Apply bias + post-ops
    if (bias || post_op != ConvPostOp::kNone) {
        apply_conv_postops(output, M, N_gemm, bias, post_op);
    }
}

void conv2d_fp32(const Conv2DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 ConvPostOp post_op) {
    // Fast path: 1×1 conv with stride=1, no padding
    if (p.KH == 1 && p.KW == 1 &&
        p.stride_h == 1 && p.stride_w == 1 &&
        p.pad_h == 0 && p.pad_w == 0) {
        conv2d_1x1_direct(p, input, filter, bias, output, post_op);
        return;
    }

    // General path: im2col + GEMM
    conv2d_im2col_gemm(p, input, filter, bias, output, post_op);
}

}  // namespace dnnopt
