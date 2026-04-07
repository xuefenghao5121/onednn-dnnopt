/// @file conv_postops.cpp
/// Fused post-operations for Conv2D output: bias, ReLU, ReLU6.
/// Applied in-place on NHWC output tensor after GEMM.

#include "dnnopt/conv/conv.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <algorithm>

namespace dnnopt {

/// Apply post-ops in-place on output [num_rows, OC] (row-major).
/// num_rows = N * OH * OW. Each row has OC elements.
void apply_conv_postops(float* output, int num_rows, int OC,
                        const float* bias, ConvPostOp op) {
    if (op == ConvPostOp::kNone && !bias) return;

    const bool has_bias = (bias != nullptr);
    const bool has_relu = (op == ConvPostOp::kRelu || op == ConvPostOp::kBiasRelu);
    const bool has_relu6 = (op == ConvPostOp::kRelu6);

    // If kBiasRelu, bias is applied regardless of bias pointer
    // (kBiasRelu implies bias is non-null)

#ifdef __ARM_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vsix  = vdupq_n_f32(6.0f);
#endif

    for (int row = 0; row < num_rows; ++row) {
        float* out_row = output + (size_t)row * OC;
        int oc = 0;

#ifdef __ARM_NEON
        for (; oc + 3 < OC; oc += 4) {
            float32x4_t v = vld1q_f32(out_row + oc);

            if (has_bias) {
                float32x4_t b = vld1q_f32(bias + oc);
                v = vaddq_f32(v, b);
            }

            if (has_relu || op == ConvPostOp::kBiasRelu) {
                v = vmaxq_f32(v, vzero);
            } else if (has_relu6) {
                v = vmaxq_f32(v, vzero);
                v = vminq_f32(v, vsix);
            }

            vst1q_f32(out_row + oc, v);
        }
#endif

        // Scalar tail
        for (; oc < OC; ++oc) {
            float v = out_row[oc];
            if (has_bias) v += bias[oc];
            if (has_relu || op == ConvPostOp::kBiasRelu) {
                v = std::max(0.0f, v);
            } else if (has_relu6) {
                v = std::min(6.0f, std::max(0.0f, v));
            }
            out_row[oc] = v;
        }
    }
}

}  // namespace dnnopt
