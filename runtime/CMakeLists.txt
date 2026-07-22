add_library(ftlpu_cmodel_static STATIC
    src/cmodel/cmodel_static.cpp)

target_include_directories(ftlpu_cmodel_static
    PUBLIC
        include
        "${FTLPU_CMODEL_DIR}/include")

add_library(ftlpu_software
    src/runtime/binary.cpp
    src/runtime/cmodel_runtime.cpp
    src/runtime/icu_program.cpp)

target_include_directories(ftlpu_software
    PUBLIC
        include)

target_link_libraries(ftlpu_software
    PUBLIC
        ftlpu_cmodel_static)

add_executable(command_vxm_binary_runtime_test
    tests/runtime/command_vxm_binary_runtime_test.cpp)

target_link_libraries(command_vxm_binary_runtime_test PRIVATE ftlpu_software)

add_executable(sxm_binary_roundtrip_test
    tests/runtime/sxm_binary_roundtrip_test.cpp)

target_link_libraries(sxm_binary_roundtrip_test PRIVATE ftlpu_software)

add_executable(compiled_sxm_command_runtime_test
    tests/runtime/compiled_sxm_command_runtime_test.cpp)

target_link_libraries(compiled_sxm_command_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_queue_program_runtime_test
    tests/runtime/compiled_queue_program_runtime_test.cpp)

target_link_libraries(compiled_queue_program_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_smollm2_135m_ffn_runtime_test
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp)

target_link_libraries(compiled_smollm2_135m_ffn_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_smollm2_135m_ffn_seq128_runtime_test
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp)

target_compile_definitions(compiled_smollm2_135m_ffn_seq128_runtime_test
    PRIVATE FTLPU_TEST_SEQUENCE_LENGTH=128)
target_link_libraries(compiled_smollm2_135m_ffn_seq128_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_smollm2_attention_projection_runtime_test
    tests/runtime/compiled_smollm2_attention_projection_runtime_test.cpp)

target_link_libraries(compiled_smollm2_attention_projection_runtime_test PRIVATE ftlpu_software)
