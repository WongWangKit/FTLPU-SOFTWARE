add_library(ftlpu_cmodel_static STATIC
    src/cmodel/cmodel_static.cpp)

target_include_directories(ftlpu_cmodel_static
    PUBLIC
        include
        "${FTLPU_CMODEL_DIR}/include")

add_library(ftlpu_software
    src/runtime/binary.cpp
    src/runtime/cmodel_runtime.cpp
    src/runtime/icu_program.cpp
    src/runtime/lpu_ir.cpp)

target_include_directories(ftlpu_software
    PUBLIC
        include)

target_link_libraries(ftlpu_software
    PUBLIC
        ftlpu_cmodel_static)

add_executable(icu_program_smoke_test
    tests/runtime/icu_program_smoke_test.cpp)

target_link_libraries(icu_program_smoke_test PRIVATE ftlpu_software)

add_executable(lpu_ir_binary_runtime_test
    tests/runtime/lpu_ir_binary_runtime_test.cpp)

target_link_libraries(lpu_ir_binary_runtime_test PRIVATE ftlpu_software)

add_executable(cmodel_gemm_320_runtime_test
    tests/runtime/cmodel_gemm_320_runtime_test.cpp)

target_link_libraries(cmodel_gemm_320_runtime_test PRIVATE ftlpu_software)

add_test(NAME icu_program_smoke_test COMMAND icu_program_smoke_test)
add_test(NAME lpu_ir_binary_runtime_test COMMAND lpu_ir_binary_runtime_test)
add_test(NAME cmodel_gemm_320_runtime_test COMMAND cmodel_gemm_320_runtime_test)
