add_library(dnnopt_core STATIC
    hwcaps/arm_hwcaps.cpp
    utils/timer.cpp
    utils/aligned_alloc.cpp
)

target_include_directories(dnnopt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(dnnopt_core PUBLIC OpenMP::OpenMP_CXX)
endif()
