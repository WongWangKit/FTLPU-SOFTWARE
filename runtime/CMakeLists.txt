add_library(ftlpu_cmodel_static STATIC
    src/cmodel/cmodel_static.cpp
    "${FTLPU_CMODEL_DIR}/src/mem/sram.cpp"
    "${FTLPU_CMODEL_DIR}/src/mxm/accumulator.cpp")

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
    src/runtime/binary.cpp
    src/runtime/c2c_weight_pager.cpp
    src/runtime/cmodel_runtime.cpp
    src/runtime/icu_program.cpp
    src/runtime/model_package.cpp
    src/runtime/model_session.cpp
    src/runtime/performance.cpp
    src/runtime/schedule_trace.cpp
    src/runtime/session_memory_planner.cpp
    src/runtime/weight_page_builder.cpp
    src/runtime/weight_prefetch_plan.cpp
    examples/binary_inspect.cpp
    examples/binary_repack.cpp
    examples/pack_model_weights.cpp
    tests/runtime/command_mxm_column_binary_test.cpp
    tests/runtime/command_vxm_binary_runtime_test.cpp
    tests/runtime/sxm_binary_roundtrip_test.cpp
    tests/runtime/compiled_sxm_command_runtime_test.cpp
    tests/runtime/compiled_queue_program_runtime_test.cpp
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp
    tests/runtime/compiled_qwen_real_ffn_runtime_test.cpp
    tests/runtime/compiled_qwen_real_decoder_layer_runtime_test.cpp
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
    tests/runtime/resident_weight_planner_test.cpp
    tests/runtime/c2c_weight_ping_pong_test.cpp
    tests/runtime/weight_page_planner_test.cpp
    tests/runtime/model_session_c2c_weight_pages_test.cpp
    tests/runtime/model_session_c2c_io_test.cpp
    tests/runtime/model_session_executable_c2c_weight_test.cpp
    tests/runtime/weight_page_builder_test.cpp
    tests/runtime/qwen_weight_page_builder_test.cpp)
set_source_files_properties(${FTLPU_CMODEL_CONSUMER_SOURCES}
    PROPERTIES OBJECT_DEPENDS
        "${FTLPU_CMODEL_PUBLIC_HEADERS};${FTLPU_RUNTIME_PUBLIC_HEADERS}")

target_include_directories(ftlpu_cmodel_static
    PUBLIC
        include
        "${FTLPU_GENERATED_INCLUDE_DIR}"
        "${FTLPU_CMODEL_DIR}/include")

add_library(ftlpu_software
    src/runtime/binary.cpp
    src/runtime/c2c_weight_pager.cpp
    src/runtime/cmodel_runtime.cpp
    src/runtime/icu_program.cpp
    src/runtime/model_package.cpp
    src/runtime/model_session.cpp
    src/runtime/performance.cpp
    src/runtime/schedule_trace.cpp
    src/runtime/session_memory_planner.cpp
    src/runtime/weight_page_builder.cpp
    src/runtime/weight_prefetch_plan.cpp)

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

add_executable(ftlpu_binary_inspect examples/binary_inspect.cpp)
target_link_libraries(ftlpu_binary_inspect PRIVATE ftlpu_software)

add_executable(ftlpu_binary_repack examples/binary_repack.cpp)
target_link_libraries(ftlpu_binary_repack PRIVATE ftlpu_software)

add_executable(schedule_trace_weight_page_test
    tests/runtime/schedule_trace_weight_page_test.cpp)
target_link_libraries(schedule_trace_weight_page_test PRIVATE ftlpu_software)
add_test(
    NAME schedule_trace_weight_page_test
    COMMAND $<TARGET_FILE:schedule_trace_weight_page_test>)

add_executable(model_session_executable_c2c_weight_test
    tests/runtime/model_session_executable_c2c_weight_test.cpp)
target_link_libraries(model_session_executable_c2c_weight_test
    PRIVATE ftlpu_software)
add_test(
    NAME model_session_executable_c2c_weight_test
    COMMAND $<TARGET_FILE:model_session_executable_c2c_weight_test>)

add_executable(ftlpu_pack_model_weights
    examples/pack_model_weights.cpp)
set_target_properties(ftlpu_pack_model_weights PROPERTIES
    OUTPUT_NAME "ftlpu-pack-model-weights")
target_link_libraries(ftlpu_pack_model_weights PRIVATE ftlpu_software)

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
add_executable(sxm_binary_roundtrip_test
    tests/runtime/sxm_binary_roundtrip_test.cpp)

target_link_libraries(sxm_binary_roundtrip_test PRIVATE ftlpu_software)
add_test(
    NAME sxm_binary_roundtrip_test
    COMMAND $<TARGET_FILE:sxm_binary_roundtrip_test>
        ${CMAKE_CURRENT_BINARY_DIR}/test-output/sxm-v16.ftlpu)

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

add_executable(c2c_weight_ping_pong_test
    tests/runtime/c2c_weight_ping_pong_test.cpp)
target_link_libraries(c2c_weight_ping_pong_test PRIVATE ftlpu_software)
add_test(
    NAME c2c_weight_ping_pong_test
    COMMAND $<TARGET_FILE:c2c_weight_ping_pong_test>)

add_executable(c2c_weight_page_package_roundtrip_test
    tests/runtime/c2c_weight_page_package_roundtrip_test.cpp)
target_link_libraries(c2c_weight_page_package_roundtrip_test
    PRIVATE ftlpu_software)

add_executable(weight_page_planner_test
    tests/runtime/weight_page_planner_test.cpp)
target_link_libraries(weight_page_planner_test PRIVATE ftlpu_software)
add_test(
    NAME weight_page_planner_test
    COMMAND $<TARGET_FILE:weight_page_planner_test>)

add_executable(model_session_c2c_weight_pages_test
    tests/runtime/model_session_c2c_weight_pages_test.cpp)
target_link_libraries(model_session_c2c_weight_pages_test
    PRIVATE ftlpu_software)
add_test(
    NAME model_session_c2c_weight_pages_test
    COMMAND $<TARGET_FILE:model_session_c2c_weight_pages_test>)

add_executable(model_session_c2c_io_test
    tests/runtime/model_session_c2c_io_test.cpp)
target_link_libraries(model_session_c2c_io_test PRIVATE ftlpu_software)
add_test(
    NAME model_session_c2c_io_test
    COMMAND $<TARGET_FILE:model_session_c2c_io_test>)

add_executable(weight_page_builder_test
    tests/runtime/weight_page_builder_test.cpp)
target_link_libraries(weight_page_builder_test PRIVATE ftlpu_software)
add_test(
    NAME weight_page_builder_test
    COMMAND $<TARGET_FILE:weight_page_builder_test>)

add_executable(qwen_weight_page_builder_test
    tests/runtime/qwen_weight_page_builder_test.cpp)
target_link_libraries(qwen_weight_page_builder_test PRIVATE ftlpu_software)
add_test(
    NAME qwen_weight_page_builder_test
    COMMAND $<TARGET_FILE:qwen_weight_page_builder_test>)

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

add_executable(compiled_smollm2_135m_ffn_runtime_test
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp)

target_link_libraries(compiled_smollm2_135m_ffn_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_qwen2_5_1_5b_ffn_seq32_runtime_test
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp)
target_compile_definitions(compiled_qwen2_5_1_5b_ffn_seq32_runtime_test
    PRIVATE
    FTLPU_TEST_SEQUENCE_LENGTH=32
    FTLPU_TEST_HIDDEN_SIZE=1536
    FTLPU_TEST_INTERMEDIATE_SIZE=8960
    FTLPU_TEST_MODEL_NAME="Qwen2.5-1.5B-seq32")
target_link_libraries(compiled_qwen2_5_1_5b_ffn_seq32_runtime_test
    PRIVATE ftlpu_software)

add_executable(compiled_qwen_real_ffn_runtime_test
    tests/runtime/compiled_qwen_real_ffn_runtime_test.cpp)
target_link_libraries(compiled_qwen_real_ffn_runtime_test
    PRIVATE ftlpu_software)

add_executable(compiled_qwen_real_decoder_layer_runtime_test
    tests/runtime/compiled_qwen_real_decoder_layer_runtime_test.cpp)
target_link_libraries(compiled_qwen_real_decoder_layer_runtime_test
    PRIVATE ftlpu_software)

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

add_executable(compiled_qwen2_5_1_5b_decoder_layer_runtime_test
    tests/runtime/compiled_smollm2_decoder_layer_runtime_test.cpp)
target_compile_definitions(
    compiled_qwen2_5_1_5b_decoder_layer_runtime_test PRIVATE
    FTLPU_TEST_SEQUENCE_LENGTH=128
    FTLPU_TEST_HIDDEN_SIZE=1536
    FTLPU_TEST_INTERMEDIATE_SIZE=8960
    FTLPU_TEST_QUERY_HEADS=12
    FTLPU_TEST_KV_HEADS=2
    FTLPU_TEST_HEAD_DIM=128
    FTLPU_TEST_RMS_EPSILON=1.0e-6f
    FTLPU_TEST_ROPE_THETA=1000000.0f
    FTLPU_TEST_MODEL_NAME=\"Qwen2.5-1.5B\")
target_link_libraries(compiled_qwen2_5_1_5b_decoder_layer_runtime_test
    PRIVATE ftlpu_software)

add_executable(compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test
    tests/runtime/compiled_smollm2_decoder_layer_runtime_test.cpp)
target_compile_definitions(
    compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test PRIVATE
    FTLPU_TEST_SEQUENCE_LENGTH=32
    FTLPU_TEST_HIDDEN_SIZE=1536
    FTLPU_TEST_INTERMEDIATE_SIZE=8960
    FTLPU_TEST_QUERY_HEADS=12
    FTLPU_TEST_KV_HEADS=2
    FTLPU_TEST_HEAD_DIM=128
    FTLPU_TEST_RMS_EPSILON=1.0e-6f
    FTLPU_TEST_ROPE_THETA=1000000.0f
    FTLPU_TEST_MODEL_NAME="Qwen2.5-1.5B-seq32")
target_link_libraries(
    compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test
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
