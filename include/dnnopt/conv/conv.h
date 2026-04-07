#pragma once
/// @file conv.h
/// Public Conv2D API for DNN-Opt.
///
/// Data layouts:
///   input:  [N, IH, IW, IC] (NHWC)
///   filter: [OC, KH, KW, IC] (OIHW with IC innermost)
///   bias:   [OC] or nullptr
///   output: [N, OH, OW, OC] (NHWC)

#include <cstddef>

namespace dnnopt {

/// Conv2D parameters.
struct Conv2DParams {
    int N;             // Batch size
    int IC, IH, IW;   // Input: channels, height, width
    int OC;            // Output channels
    int KH, KW;        // Kernel height, width
    int stride_h, stride_w;
    int pad_h, pad_w;

    int OH() const { return (IH + 2 * pad_h - KH) / stride_h + 1; }
    int OW() const { return (IW + 2 * pad_w - KW) / stride_w + 1; }
};

/// Post-operation applied after convolution.
enum class ConvPostOp {
    kNone,       // No post-op
    kRelu,       // max(0, x)
    kRelu6,      // min(6, max(0, x))
    kBiasRelu,   // max(0, x + bias)
};

/// FP32 Conv2D with im2col + optimized GEMM.
/// Automatically selects optimal path:
///   - 1×1 stride=1 pad=0: direct GEMM (no im2col)
///   - Otherwise: im2col + GEMM
///
/// @param p      Convolution parameters
/// @param input  Input tensor [N, IH, IW, IC] (NHWC)
/// @param filter Filter tensor [OC, KH, KW, IC]
/// @param bias   Bias vector [OC], or nullptr for no bias
/// @param output Output tensor [N, OH, OW, OC] (NHWC)
/// @param post_op Post-operation to apply
void conv2d_fp32(const Conv2DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 ConvPostOp post_op = ConvPostOp::kNone);

}  // namespace dnnopt
