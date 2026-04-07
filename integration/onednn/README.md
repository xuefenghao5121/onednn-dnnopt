# oneDNN Integration Patch

This patch integrates DNN-Opt as the FP32 GEMM backend into oneDNN,
replacing the slow `ref_gemm` fallback on AArch64.

## How to apply

```bash
# 1. Clone oneDNN (tested with commit db17ac9)
git clone --depth=1 https://github.com/oneapi-src/oneDNN.git onednn-dnnopt
cd onednn-dnnopt

# 2. Apply the patch
git apply /path/to/0001-dnnopt-integration.patch

# 3. Build DNN-Opt first (if not already built)
cd /path/to/DNN-Opt && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

# 4. Build oneDNN with DNN-Opt
cd onednn-dnnopt && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DDNNL_AARCH64_USE_ACL=OFF \
  -DDNNL_BLAS_VENDOR=NONE \
  -DDNNL_AARCH64_USE_DNNOPT=ON \
  -DDNNOPT_ROOT=/path/to/DNN-Opt \
  -DDNNL_CPU_RUNTIME=OMP
make -j
```

## What it does

- Adds `dnnopt_sgemm()` dispatch in `extended_sgemm()` before `ref_gemm`
- Handles col-major to row-major duality conversion
- Automatically accelerates `dnnl_sgemm()`, GEMM-based convolution, and matmul

## Performance (Neoverse N2)

| Size       | ref_gemm | DNN-Opt | Speedup |
|------------|----------|---------|---------|
| 512x512    | 9.8      | 43.2    | 4.4x    |
| 1024x1024  | 9.8      | 43.7    | 4.5x    |
| 1024x1024 (4T) | 18.8 | 66.7   | 3.6x    |

## Files modified

- `cmake/dnnopt.cmake` (new) — CMake module to find DNN-Opt
- `cmake/options.cmake` — add `DNNL_AARCH64_USE_DNNOPT` option
- `CMakeLists.txt` — include dnnopt.cmake
- `src/cpu/aarch64/dnnopt_gemm_wrapper.hpp` (new) — col→row duality adapter
- `src/cpu/gemm/gemm.cpp` — dispatch to DNN-Opt in extended_sgemm
