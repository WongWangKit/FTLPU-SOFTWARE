cmake_minimum_required(VERSION 3.16)

project(FTLPU_SOFTWARE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FTLPU_CMODEL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../FTLPU-CMODEL"
    CACHE PATH
    "Path to the FTLPU-CMODEL repository")

enable_testing()

add_subdirectory(runtime)
add_subdirectory(compiler)
