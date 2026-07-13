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

find_package(Python3 COMPONENTS Interpreter)

enable_testing()
add_test(NAME icu_program_smoke_test COMMAND icu_program_smoke_test)
add_test(NAME lpu_ir_binary_runtime_test COMMAND lpu_ir_binary_runtime_test)

if(Python3_Interpreter_FOUND)
    add_test(
        NAME iree_frontend_stage_common_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ftlpu-iree-import.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/simple_stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/iree_frontend/simple_stablehlo.common.mlir
            --input-format stablehlo
            --mode stage-common-ir)

    add_test(
        NAME iree_frontend_onnx_import_dry_run_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ftlpu-iree-import.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/model.onnx
            --output ${CMAKE_CURRENT_BINARY_DIR}/iree_frontend/model.mlir
            --input-format onnx
            --mode import-onnx
            --onnx-opset-version 17
            --dry-run)
endif()
