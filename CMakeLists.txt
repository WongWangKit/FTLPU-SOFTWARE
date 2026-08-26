cmake_minimum_required(VERSION 3.20)

project(FTLPU_SOFTWARE LANGUAGES CXX)

option(FTLPU_ENABLE_ADDRESS_SANITIZER
    "Instrument FTLPU compiler and runtime targets with AddressSanitizer" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Some LLVM/IREE cache configurations intentionally clear CMake's default
# MSVC Release flags. Keep FTLPU's cycle-accurate CModel optimized even when
# this project is configured through one of those caches.
if(MSVC)
    add_compile_options(
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Ob2>)
endif()

set(FTLPU_CMODEL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../FTLPU-CMODEL"
    CACHE PATH
    "Path to the FTLPU-CMODEL repository")

set(FTLPU_HARDWARE_CONFIG
    "${FTLPU_CMODEL_DIR}/config/ftlpu-lpu32.json"
    CACHE FILEPATH
    "Shared hardware target JSON used by FTLPU-SOFTWARE and FTLPU-CMODEL")
set(FTLPU_GENERATED_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
include("${FTLPU_CMODEL_DIR}/cmake/FTLPUHardwareConfig.cmake")
ftlpu_generate_hardware_config(
    "${FTLPU_HARDWARE_CONFIG}"
    "${FTLPU_GENERATED_INCLUDE_DIR}/ftlpu/core/hardware_config.hpp")

enable_testing()

add_subdirectory(runtime)
add_subdirectory(compiler)
