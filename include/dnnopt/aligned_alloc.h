#pragma once
/// @file aligned_alloc.h
/// Cache-line-aligned memory allocation utilities.

#include <cstddef>
#include <cstdlib>
#include <memory>

namespace dnnopt {

/// Default alignment: 64 bytes (cache line on most ARM cores).
constexpr size_t kCacheLineSize = 64;

/// Allocate `size` bytes aligned to `alignment`.
/// Returns nullptr on failure. Must be freed with aligned_free().
void* aligned_malloc(size_t size, size_t alignment = kCacheLineSize);

/// Free memory allocated by aligned_malloc().
void aligned_free(void* ptr);

/// RAII wrapper for aligned memory.
template<typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const { aligned_free(ptr); }
};

template<typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>;

/// Allocate an aligned array of `count` elements of type T.
template<typename T>
AlignedPtr<T> aligned_array(size_t count, size_t alignment = kCacheLineSize) {
    void* p = aligned_malloc(count * sizeof(T), alignment);
    return AlignedPtr<T>(static_cast<T*>(p));
}

}  // namespace dnnopt
