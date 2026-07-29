cmake_minimum_required(VERSION 3.16)

project(FTLPU_SOFTWARE LANGUAGES CXX)

option(FTLPU_ENABLE_ADDRESS_SANITIZER
    "Instrument FTLPU compiler and runtime targets with AddressSanitizer" OFF)
option(FTLPU_BUILD_COMPILER
    "Build the MLIR-based FTLPU compiler" OFF)
option(FTLPU_BUILD_LEGACY_CMODEL_RUNTIME
    "Build the direct-SRAM CModel runtime adapter and its tests" OFF)

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

if(NOT TARGET ftlpu::cmodel)
    message(FATAL_ERROR
        "FTLPU-CMODEL did not define the ftlpu::cmodel target")
endif()

enable_testing()

add_subdirectory(runtime)

if(FTLPU_BUILD_COMPILER)
    if(NOT TARGET ftlpu_software)
        message(FATAL_ERROR
            "The compiler still depends on the legacy ftlpu_software target. "
            "Keep FTLPU_BUILD_COMPILER=OFF until the new backend runtime "
            "provides the binary support library.")
    endif()
    add_subdirectory(compiler)
endif()