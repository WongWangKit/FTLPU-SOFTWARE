cmake_minimum_required(VERSION 3.16)

project(FTLPU_SOFTWARE LANGUAGES CXX)

option(FTLPU_ENABLE_ADDRESS_SANITIZER
    "Instrument FTLPU compiler and runtime targets with AddressSanitizer" OFF)
option(FTLPU_BUILD_COMPILER
    "Build the MLIR-based FTLPU compiler" OFF)
option(FTLPU_BUILD_LEGACY_CMODEL_RUNTIME
    "Build the direct-SRAM CModel runtime adapter and its tests" OFF)
option(FTLPU_CMODEL_TRANSFORMER_EVAL
    "Build against the TransformerEval CModel hardware configuration" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FTLPU_CMODEL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../FTLPU-CMODEL"
    CACHE PATH
    "Path to the FTLPU-CMODEL repository")

if(NOT EXISTS "${FTLPU_CMODEL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "FTLPU_CMODEL_DIR does not point to an FTLPU-CMODEL checkout: "
        "${FTLPU_CMODEL_DIR}")
endif()

set(FTLPU_CMODEL_REQUIRED_HEADERS
    include/ftlpu/core/hardware_params.hpp
    include/ftlpu/dma/engine.hpp
    include/ftlpu/program/program_image.hpp
    include/ftlpu/system/tsp_slice_system.hpp)

foreach(required_header IN LISTS FTLPU_CMODEL_REQUIRED_HEADERS)
    if(NOT EXISTS "${FTLPU_CMODEL_DIR}/${required_header}")
        message(FATAL_ERROR
            "Selected CModel does not provide ${required_header}. "
            "Use the feat/transformer-datapath branch.")
    endif()
endforeach()

set(FTLPU_CMODEL_BUILD_TESTS OFF CACHE BOOL
    "Build embedded CModel tests" FORCE)
set(FTLPU_CMODEL_BUILD_EXAMPLES OFF CACHE BOOL
    "Build embedded CModel examples" FORCE)

add_subdirectory(
    "${FTLPU_CMODEL_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/_deps/ftlpu-cmodel"
    EXCLUDE_FROM_ALL)

if(TARGET ftlpu_cmodel AND NOT TARGET ftlpu::cmodel)
    add_library(ftlpu::cmodel ALIAS ftlpu_cmodel)
endif()

if(NOT TARGET ftlpu::cmodel)
    message(FATAL_ERROR
        "FTLPU-CMODEL did not define the ftlpu::cmodel target")
endif()

add_library(ftlpu_cmodel_selected_config INTERFACE)
add_library(
    ftlpu::cmodel_selected_config
    ALIAS ftlpu_cmodel_selected_config)
target_link_libraries(
    ftlpu_cmodel_selected_config
    INTERFACE ftlpu::cmodel)
if(FTLPU_CMODEL_TRANSFORMER_EVAL)
    target_compile_definitions(
        ftlpu_cmodel_selected_config
        INTERFACE FTLPU_TRANSFORMER_EVAL_CONFIG)
endif()

enable_testing()

add_subdirectory(runtime)

if(FTLPU_BUILD_COMPILER)
    add_subdirectory(compiler)
endif()
