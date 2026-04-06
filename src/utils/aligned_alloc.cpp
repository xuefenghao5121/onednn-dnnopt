/// @file aligned_alloc.cpp
/// Cache-line-aligned memory allocation.

#include "dnnopt/aligned_alloc.h"

#include <cstdlib>
#include <cstdio>

namespace dnnopt {

void* aligned_malloc(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        fprintf(stderr, "aligned_malloc failed: size=%zu align=%zu\n",
                size, alignment);
        return nullptr;
    }
    return ptr;
}

void aligned_free(void* ptr) {
    free(ptr);
}

}  // namespace dnnopt
