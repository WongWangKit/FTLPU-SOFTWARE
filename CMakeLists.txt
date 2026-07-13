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
    src/runtime/binary.cpp
    src/runtime/cmodel_runtime.cpp
    src/runtime/icu_program.cpp
    src/runtime/lpu_ir.cpp)

target_include_directories(ftlpu_software
    PUBLIC
        include
        "${FTLPU_CMODEL_DIR}/include")

add_executable(icu_program_smoke_test
    tests/runtime/icu_program_smoke_test.cpp)

target_link_libraries(icu_program_smoke_test PRIVATE ftlpu_software)

add_executable(lpu_ir_binary_runtime_test
    tests/runtime/lpu_ir_binary_runtime_test.cpp)

target_link_libraries(lpu_ir_binary_runtime_test PRIVATE ftlpu_software)

enable_testing()
add_test(NAME icu_program_smoke_test COMMAND icu_program_smoke_test)
add_test(NAME lpu_ir_binary_runtime_test COMMAND lpu_ir_binary_runtime_test)
