#pragma once
/// @file gemm_driver_generic.h
/// Generic parameterized BLIS-style GEMM driver.
/// Works with any microkernel via function pointers from the registry.

#include "dnnopt/gemm/gemm_ukernel_registry.h"
#include "dnnopt/gemm/gemm_types.h"

namespace dnnopt {

/// Configuration bundle for the generic GEMM driver.
/// Populated from a GemmMicrokernelDesc + computed blocking params.
struct GemmDriverConfig {
    int Mr, Nr, Kgroup;
    int Mc, Nc, Kc;
    int packed_a_elem_bytes;
    int packed_b_elem_bytes;
    GemmDataType dtype;
    UkernelFn ukernel;
    PackAFn pack_a;
    PackBFn pack_b;
};

/// Generic BLIS 5-loop GEMM driver.
/// Handles any data type through the function pointers in cfg.
void gemm_driver_generic(int M, int N, int K,
                         float alpha, const float* A, int lda,
                         const float* B, int ldb,
                         float beta, float* C, int ldc,
                         const GemmDriverConfig& cfg);

}  // namespace dnnopt
