add_library(ftlpu_cmodel_static STATIC
    src/cmodel/cmodel_static.cpp
    "${FTLPU_CMODEL_DIR}/src/mem/sram.cpp"
    "${FTLPU_CMODEL_DIR}/src/mxm/accumulator.cpp"
    "${FTLPU_CMODEL_DIR}/src/mxm/block_accumulator.cpp")

file(GLOB_RECURSE FTLPU_CMODEL_PUBLIC_HEADERS CONFIGURE_DEPENDS
    "${FTLPU_CMODEL_DIR}/include/ftlpu/*.hpp")
file(GLOB_RECURSE FTLPU_RUNTIME_PUBLIC_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ftlpu/*.hpp")
set_source_files_properties(${FTLPU_CMODEL_PUBLIC_HEADERS}
    PROPERTIES HEADER_FILE_ONLY TRUE)
target_sources(ftlpu_cmodel_static PRIVATE ${FTLPU_CMODEL_PUBLIC_HEADERS})

# Ninja cannot reliably parse localized MSVC /showIncludes output. Keep an
# explicit dependency edge so every CModel ABI change rebuilds its consumers.
set(FTLPU_CMODEL_CONSUMER_SOURCES
    src/cmodel/cmodel_static.cpp
    "${FTLPU_CMODEL_DIR}/src/mem/sram.cpp"
    "${FTLPU_CMODEL_DIR}/src/mxm/accumulator.cpp"
    "${FTLPU_CMODEL_DIR}/src/mxm/block_accumulator.cpp"
    src/runtime/binary.cpp
    src/runtime/cmodel_runtime.cpp
    src/runtime/icu_program.cpp
    src/runtime/model_package.cpp
    src/runtime/model_session.cpp
    src/runtime/performance.cpp
    src/runtime/schedule_trace.cpp
    tests/runtime/command_mxm_column_binary_test.cpp
    tests/runtime/command_vxm_binary_runtime_test.cpp
    tests/runtime/sxm_binary_roundtrip_test.cpp
    tests/runtime/compiled_sxm_command_runtime_test.cpp
    tests/runtime/compiled_queue_program_runtime_test.cpp
    tests/runtime/compiled_rmsnorm_runtime_test.cpp
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp
    tests/runtime/compiled_smollm2_lm_head_runtime_test.cpp
    tests/runtime/compiled_smollm2_attention_runtime_test.cpp
    tests/runtime/compiled_smollm2_decoder_layer_runtime_test.cpp
    tests/runtime/hf_decoder_layer_model_session_test.cpp
    tests/runtime/hf_two_decoder_layers_model_session_test.cpp
    tests/runtime/hf_model_boundaries_package_test.cpp
    tests/runtime/hf_model_boundaries_session_test.cpp
    tests/runtime/smollm2_full_prefill_lm_head_e2e_test.cpp
    tests/runtime/host_lm_head_test.cpp
    tests/runtime/model_package_roundtrip_test.cpp
    tests/runtime/model_session_embedding_test.cpp
    tests/runtime/resident_weight_planner_test.cpp)
set_source_files_properties(${FTLPU_CMODEL_CONSUMER_SOURCES}
    PROPERTIES OBJECT_DEPENDS
        "${FTLPU_CMODEL_PUBLIC_HEADERS};${FTLPU_RUNTIME_PUBLIC_HEADERS}")

target_include_directories(ftlpu_cmodel_static
    PUBLIC
        include
        "${FTLPU_CMODEL_DIR}/include")

add_library(ftlpu_software
    src/runtime/binary.cpp
    src/runtime/cmodel_runtime.cpp
    src/runtime/icu_program.cpp
    src/runtime/model_package.cpp
    src/runtime/model_session.cpp
    src/runtime/performance.cpp
    src/runtime/schedule_trace.cpp
    src/runtime/session_memory_planner.cpp)

if(FTLPU_ENABLE_ADDRESS_SANITIZER AND MSVC)
    target_compile_options(ftlpu_software PRIVATE /fsanitize=address)
    target_compile_definitions(ftlpu_software PRIVATE
        _DISABLE_VECTOR_ANNOTATION
        _DISABLE_STRING_ANNOTATION
        _DISABLE_OPTIONAL_ANNOTATION)
endif()

target_include_directories(ftlpu_software
    PUBLIC
        include)

target_link_libraries(ftlpu_software
    PUBLIC
        ftlpu_cmodel_static)

add_executable(command_vxm_binary_runtime_test
    tests/runtime/command_vxm_binary_runtime_test.cpp)

target_link_libraries(command_vxm_binary_runtime_test PRIVATE ftlpu_software)

add_executable(command_mxm_column_binary_test
    tests/runtime/command_mxm_column_binary_test.cpp)

target_link_libraries(command_mxm_column_binary_test PRIVATE ftlpu_software)

add_executable(command_bf16_binary_test
    tests/runtime/command_bf16_binary_test.cpp)

target_link_libraries(command_bf16_binary_test PRIVATE ftlpu_software)

add_executable(command_cmodel_compat_binary_test
    tests/runtime/command_cmodel_compat_binary_test.cpp)
target_link_libraries(command_cmodel_compat_binary_test
    PRIVATE ftlpu_software)
add_executable(block8_linear_projection_runtime_test
    tests/runtime/block8_linear_projection_runtime_test.cpp)
target_link_libraries(block8_linear_projection_runtime_test
    PRIVATE ftlpu_software)

add_executable(sxm_binary_roundtrip_test
    tests/runtime/sxm_binary_roundtrip_test.cpp)

target_link_libraries(sxm_binary_roundtrip_test PRIVATE ftlpu_software)
add_test(
    NAME sxm_binary_roundtrip_test
    COMMAND $<TARGET_FILE:sxm_binary_roundtrip_test>
        ${CMAKE_CURRENT_BINARY_DIR}/test-output/sxm-v5.ftlpu)

add_executable(model_package_roundtrip_test
    tests/runtime/model_package_roundtrip_test.cpp)

target_link_libraries(model_package_roundtrip_test PRIVATE ftlpu_software)
add_test(
    NAME model_package_roundtrip_test
    COMMAND $<TARGET_FILE:model_package_roundtrip_test>
        ${CMAKE_CURRENT_BINARY_DIR}/test-output/roundtrip.ftlpum)

add_executable(model_session_embedding_test
    tests/runtime/model_session_embedding_test.cpp)
target_link_libraries(model_session_embedding_test PRIVATE ftlpu_software)
add_test(
    NAME model_session_embedding_test
    COMMAND $<TARGET_FILE:model_session_embedding_test>)

add_executable(host_lm_head_test
    tests/runtime/host_lm_head_test.cpp)
target_link_libraries(host_lm_head_test PRIVATE ftlpu_software)
add_test(
    NAME host_lm_head_test
    COMMAND $<TARGET_FILE:host_lm_head_test>)

add_executable(resident_weight_planner_test
    tests/runtime/resident_weight_planner_test.cpp)
target_link_libraries(resident_weight_planner_test PRIVATE ftlpu_software)
add_test(
    NAME resident_weight_planner_test
    COMMAND $<TARGET_FILE:resident_weight_planner_test>)

add_executable(hf_model_boundaries_package_test
    tests/runtime/hf_model_boundaries_package_test.cpp)
target_link_libraries(hf_model_boundaries_package_test PRIVATE ftlpu_software)
add_executable(hf_model_boundaries_session_test
    tests/runtime/hf_model_boundaries_session_test.cpp)
target_link_libraries(hf_model_boundaries_session_test PRIVATE ftlpu_software)

add_executable(smollm2_full_prefill_lm_head_e2e_test
    tests/runtime/smollm2_full_prefill_lm_head_e2e_test.cpp)
target_link_libraries(smollm2_full_prefill_lm_head_e2e_test
    PRIVATE ftlpu_software)

add_executable(compiled_sxm_command_runtime_test
    tests/runtime/compiled_sxm_command_runtime_test.cpp)

target_link_libraries(compiled_sxm_command_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_queue_program_runtime_test
    tests/runtime/compiled_queue_program_runtime_test.cpp)

target_link_libraries(compiled_queue_program_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_rmsnorm_runtime_test
    tests/runtime/compiled_rmsnorm_runtime_test.cpp)

target_link_libraries(compiled_rmsnorm_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_smollm2_135m_ffn_runtime_test
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp)

target_link_libraries(compiled_smollm2_135m_ffn_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_smollm2_lm_head_runtime_test
    tests/runtime/compiled_smollm2_lm_head_runtime_test.cpp)

target_link_libraries(compiled_smollm2_lm_head_runtime_test
    PRIVATE ftlpu_software)

add_executable(compiled_smollm2_135m_ffn_seq128_runtime_test
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp)

target_compile_definitions(compiled_smollm2_135m_ffn_seq128_runtime_test
    PRIVATE FTLPU_TEST_SEQUENCE_LENGTH=128)
target_link_libraries(compiled_smollm2_135m_ffn_seq128_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_smollm2_attention_runtime_test
    tests/runtime/compiled_smollm2_attention_runtime_test.cpp)

target_link_libraries(compiled_smollm2_attention_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_smollm2_decoder_layer_runtime_test
    tests/runtime/compiled_smollm2_decoder_layer_runtime_test.cpp)

target_link_libraries(compiled_smollm2_decoder_layer_runtime_test
    PRIVATE ftlpu_software)

add_executable(hf_decoder_layer_model_session_test
    tests/runtime/hf_decoder_layer_model_session_test.cpp)

target_link_libraries(hf_decoder_layer_model_session_test
    PRIVATE ftlpu_software)

add_executable(hf_two_decoder_layers_model_session_test
    tests/runtime/hf_two_decoder_layers_model_session_test.cpp)

target_link_libraries(hf_two_decoder_layers_model_session_test
    PRIVATE ftlpu_software)

add_executable(hf_decoder_stack_checkpoint_test
    tests/runtime/hf_decoder_stack_checkpoint_test.cpp)

target_link_libraries(hf_decoder_stack_checkpoint_test
    PRIVATE ftlpu_software)
