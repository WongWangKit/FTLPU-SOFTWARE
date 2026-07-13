cmake_minimum_required(VERSION 3.16)

project(FTLPU_SOFTWARE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FTLPU_CMODEL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../FTLPU-CMODEL"
    CACHE PATH
    "Path to the FTLPU-CMODEL repository")

add_library(ftlpu_software
    src/runtime/icu_program.cpp)

target_include_directories(ftlpu_software
    PUBLIC
        include
        "${FTLPU_CMODEL_DIR}/include")

add_executable(icu_program_smoke_test
    tests/runtime/icu_program_smoke_test.cpp)

target_link_libraries(icu_program_smoke_test PRIVATE ftlpu_software)

add_executable(mem_mxm_gemm_320_test
    tests/runtime/mem_mxm_gemm_320_test.cpp)

target_link_libraries(mem_mxm_gemm_320_test PRIVATE ftlpu_software)

enable_testing()
add_test(NAME icu_program_smoke_test COMMAND icu_program_smoke_test)
add_test(NAME mem_mxm_gemm_320_test COMMAND mem_mxm_gemm_320_test)
