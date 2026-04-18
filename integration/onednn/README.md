# oneDNN Integration

Dnnopt integrates as a **supplementary patch** for oneDNN via the `dnnl_sgemm` interface.

## Design Rationale

Dnnopt does **not** replace oneDNN. It accelerates the shapes where oneDNN underperforms:

- M=1 GEMV (inference batch=1)
- M=2--7 small matrices (small-batch inference)
- Irregular/prime N (N=17, 37, 53...)
- Tall-skinny matrices (M=128, N=2--7)

For large regular matrices (M>=32), dnnopt falls back to OpenBLAS (cblas_sgemm), matching upstream oneDNN's behavior.

## Dispatch Strategy

```
dnnl_sgemm() call
       |
       v
 Shape Classification (M, N, K, transa, transb)
       |
       +-- Small-M (M < 32) or irregular (N/K < 16) --> dnnopt kernels
       |
       +-- Large regular (M >= 32) --> OpenBLAS cblas_sgemm
       |
       +-- Final fallback --> ref_gemm
```

## Build Instructions

### Prerequisites

- **dnnopt**: Clang-15 (`/usr/bin/clang++-15`)
- **OpenBLAS**: `yum install openblas-devel` (for large matrix fallback)
- **oneDNN source**: `git clone https://github.com/oneapi-src/oneDNN`

### Step 1: Build dnnopt

```bash
cd onednn-arm-opt && mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Step 2: Apply Patch to oneDNN

```bash
cd /path/to/onednn
git apply /path/to/onednn-arm-opt/integration/onednn/0001-dnnopt-integration.patch
```

### Step 3: Build oneDNN with dnnopt + OpenBLAS

```bash
cd /path/to/onednn && mkdir build && cd build
cmake .. \
    -DDNNL_AARCH64_USE_DNNOPT=ON \
    -DDNNOPT_ROOT=/path/to/onednn-arm-opt \
    -DDNNL_BLAS_VENDOR=OPENBLAS \
    -DBLAS_INCLUDE_DIR=/usr/include/openblas \
    -DCMAKE_BUILD_TYPE=Release \
    -DDNNL_CPU_RUNTIME=OMP
cmake --build . -j$(nproc)
```

**Note**: OpenBLAS is required for large matrix fallback. Without it, large matrices would fall back to slow `ref_gemm`.

### Quick Build Script

```bash
./scripts/build_onednn_with_dnnopt.sh [/path/to/onednn]
```

### Verify

```bash
# Check library built
ls -lh build/src/libdnnl.so.*

# Run benchmark
LD_LIBRARY_PATH=build/src ./build/tests/bench_onednn_sgemm
```

## Performance (Neoverse N2, 2 cores @ 3GHz)

### dnnopt + OpenBLAS vs upstream oneDNN

| Shape | dnnopt+OpenBLAS | upstream | Speedup |
|-------|-----------------|----------|---------|
| CVR embedding b1 | 11.91 GF | 4.36 GF | **2.7x** |
| CVR embedding b4 | 28.63 GF | 4.69 GF | **6.1x** |
| LLM qkv b1 | 19.71 GF | 10.54 GF | **1.87x** |
| LLM qkv b4 | 38.70 GF | 29.53 GF | **1.31x** |
| BERT qkv b128 | 43.15 GF | 62.94 GF | 0.69x* |
| **Average** | **31.18 GF** | ~25 GF | **1.24x** |

*Large-M shapes use OpenBLAS fallback, which matches upstream behavior.

### End-to-End Inference

| Model | oneDNN | +dnnopt | Speedup |
|-------|--------|---------|---------|
| CVR batch=1 | 691 us | 52 us | **13.3x** |
| CVR batch=4 | 1861 us | 85 us | **21.9x** |
| BERT-small batch=1 | 7698 us | 1145 us | **6.7x** |
| BERT-small batch=4 | 27215 us | 2365 us | **11.5x** |
| LLM batch=1 | 260927 us | 44016 us | **5.9x** |
| LLM batch=4 | 531784 us | 148018 us | **3.6x** |

## Files Modified in oneDNN

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Include dnnopt.cmake |
| `cmake/dnnopt.cmake` | Find and link dnnopt library |
| `cmake/options.cmake` | Add DNNL_AARCH64_USE_DNNOPT option |
| `src/cpu/aarch64/dnnopt_gemm_wrapper.hpp` | sgemm dispatch adapter |
| `src/cpu/aarch64/brgemm_sgemm_wrapper.hpp` | BRGEMM fallback (experimental) |
| `src/cpu/gemm/gemm.cpp` | Shape-based dispatch logic |

## Key Design Decisions

### Why OpenBLAS Fallback?

oneDNN aarch64 lacks `gemm_driver` (only x64/ppc64 have it). Without OpenBLAS, large matrices would fall back to slow `ref_gemm`. Using OpenBLAS matches upstream oneDNN's behavior.

### Why Not ACL?

TensorFlow's bazel cache contains ACL v31.0.1, but oneDNN requires ACL v52.4+ for `CpuActivation.h` API. Version mismatch prevents integration.

### Why Not BRGEMM?

oneDNN's BRGEMM has JIT compilation overhead per call, making it unsuitable for single-batch GEMM calls.

## Troubleshooting

### `DNNOPT_ROOT not found`

```bash
# Set DNNOPT_ROOT in cmake command
cmake .. -DDNNOPT_ROOT=/path/to/onednn-arm-opt ...
```

### `OpenBLAS not found`

```bash
# Install OpenBLAS
yum install openblas-devel  # CentOS/RHEL
apt-get install libopenblas-dev  # Ubuntu/Debian

# Specify include path
cmake .. -DBLAS_INCLUDE_DIR=/usr/include/openblas ...
```

### Patch conflicts

```bash
# Check oneDNN version (patch targets upstream oneDNN v3.x)
cd /path/to/onednn
git log --oneline -1

# If conflict, manually apply key changes:
# 1. Add cmake/dnnopt.cmake
# 2. Add include in CMakeLists.txt
# 3. Add dnnopt_gemm_wrapper.hpp
# 4. Modify gemm.cpp dispatch logic
```

## References

- [oneDNN GitHub](https://github.com/oneapi-src/oneDNN)
- [OpenBLAS GitHub](https://github.com/OpenMathLib/OpenBLAS)
- [autoGEMM Paper (SC'24)](https://github.com/wudu98/autoGEMM)