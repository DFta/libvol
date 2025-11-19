cmake_minimum_required(VERSION 3.20)
project(libvol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
find_package(Threads REQUIRED)

# Dependencies
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0)
FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4)
FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.4)
set(BENCHMARK_ENABLE_TESTING Off CACHE BOOL "" FORCE)

# Google Benchmark config
set(BENCHMARK_ENABLE_WERROR  Off CACHE BOOL "" FORCE)


if(DEFINED PYTHON_EXECUTABLE)
    set(PYBIND11_PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "" FORCE)
endif()


FetchContent_MakeAvailable(pybind11 catch2 benchmark)

# Library
add_library(vol STATIC
    src/models/black_scholes.cpp
    src/models/implied_vol.cpp
    src/models/gbm.cpp
    src/models/binom.cpp
    src/models/heston.cpp
    src/models/svi.cpp
    src/math/quadrature.cpp
    src/calib/svi_slice.cpp
    src/calib/least_squares.cpp
    )
target_include_directories(vol PUBLIC include)
target_compile_features(vol PUBLIC cxx_std_20)

if (MSVC)
    target_compile_options(vol PRIVATE /W4 /permissive- /EHsc /bigobj)
    target_compile_definitions(vol PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(vol PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
endif()

# Python bindings
pybind11_add_module(volpy bindings/python_bindings.cpp)
target_link_libraries(volpy PRIVATE vol)

# Tests
enable_testing()
add_executable(vol_tests
    tests/test_black_scholes.cpp
    tests/test_implied_vol.cpp
    tests/test_binom.cpp
    tests/test_svi_slice.cpp
    tests/test_heston.cpp
    )
target_link_libraries(vol_tests PRIVATE vol Catch2::Catch2WithMain)
add_test(NAME vol_tests COMMAND vol_tests)

# Examples
function(add_vol_example target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE vol)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endfunction()

add_vol_example(svi_slice_demo examples/svi_slice_demo.cpp)
add_vol_example(bs_demo examples/bs_demo.cpp)
add_vol_example(demo examples/demo.cpp)


# Benchmarks
add_executable(bs_bench bench/bench_black_scholes.cpp)
target_link_libraries(bs_bench PRIVATE vol benchmark::benchmark Threads::Threads)
add_executable(binom_bench bench/bench_binom.cpp)
target_link_libraries(binom_bench PRIVATE vol benchmark::benchmark Threads::Threads)
add_executable(svi_slice_bench bench/bench_svi_slice.cpp)
target_link_libraries(svi_slice_bench PRIVATE vol benchmark::benchmark Threads::Threads)
add_executable(heston_bench bench/bench_heston.cpp)
target_link_libraries(heston_bench PRIVATE vol benchmark::benchmark Threads::Threads)

