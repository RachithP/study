cmake_minimum_required(VERSION 3.14)
project(mmul_benchmark)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add custom search paths for Google Benchmark
list(APPEND CMAKE_PREFIX_PATH
    /Users/rachithp/code/github/benchmark/build
)

# Find Google Benchmark
find_package(benchmark REQUIRED)

# Add executable for baseline benchmark
add_executable(mmul_baseline_bench baseline/benchmark.cpp)
add_executable(mmul_blocked_bench blocked/benchmark.cpp)

# Link libraries
target_link_libraries(mmul_baseline_bench 
    benchmark::benchmark
    pthread
)

target_link_libraries(mmul_blocked_bench 
    benchmark::benchmark
    pthread
)


# Set build type to Release by default for better performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optional: Add custom target for easy building
add_custom_target(benchmark_all
    DEPENDS mmul_baseline_bench mmul_blocked_bench
    COMMENT "Building all matrix multiplication benchmarks"
)
