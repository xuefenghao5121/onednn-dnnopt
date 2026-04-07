/// @file im2col.cpp
/// NHWC im2col: rearranges input patches into column matrix for GEMM.
///
/// Output: col_matrix [N*OH*OW, IC*KH*KW] row-major
/// Each row is one flattened receptive field patch.

#include "dnnopt/conv/conv.h"
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

/// NHWC im2col with fast-path for contiguous channel copies.
void im2col_nhwc(const Conv2DParams& p, const float* input, float* col) {
    const int OH = p.OH(), OW = p.OW();
    const int K = p.IC * p.KH * p.KW;  // Column width

    for (int n = 0; n < p.N; ++n) {
        const float* in_batch = input + (size_t)n * p.IH * p.IW * p.IC;

        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float* dst = col + ((size_t)(n * OH + oh) * OW + ow) * K;
                int col_idx = 0;

                for (int kh = 0; kh < p.KH; ++kh) {
                    int ih = oh * p.stride_h - p.pad_h + kh;
                    if (ih < 0 || ih >= p.IH) {
                        // Entire row is zero-padded
                        memset(dst + col_idx, 0, (size_t)p.KW * p.IC * sizeof(float));
                        col_idx += p.KW * p.IC;
                        continue;
                    }

                    for (int kw = 0; kw < p.KW; ++kw) {
                        int iw = ow * p.stride_w - p.pad_w + kw;
                        if (iw < 0 || iw >= p.IW) {
                            // Zero-padded column
                            memset(dst + col_idx, 0, (size_t)p.IC * sizeof(float));
                        } else {
                            // Copy IC channels contiguously (NHWC advantage)
                            const float* src = in_batch + ((size_t)ih * p.IW + iw) * p.IC;
#ifdef __ARM_NEON
                            int ic = 0;
                            for (; ic + 3 < p.IC; ic += 4) {
                                vst1q_f32(dst + col_idx + ic, vld1q_f32(src + ic));
                            }
                            for (; ic < p.IC; ++ic) {
                                dst[col_idx + ic] = src[ic];
                            }
#else
                            memcpy(dst + col_idx, src, (size_t)p.IC * sizeof(float));
#endif
                        }
                        col_idx += p.IC;
                    }
                }
            }
        }
    }
}

}  // namespace dnnopt
