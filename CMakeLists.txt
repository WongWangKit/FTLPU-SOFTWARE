cmake_minimum_required(VERSION 3.16)

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

enable_testing()

add_subdirectory(runtime)
add_subdirectory(compiler)
