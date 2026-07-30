add_executable(cmodel_dependency_smoke_test
    tests/runtime/cmodel_dependency_smoke_test.cpp)

target_link_libraries(cmodel_dependency_smoke_test
    PRIVATE
        ftlpu::cmodel_selected_config)

add_test(
    NAME cmodel_dependency_smoke_test
    COMMAND cmodel_dependency_smoke_test)

add_library(ftlpu_device_backend
    src/runtime/cmodel_device_backend.cpp)

add_library(ftlpu::device_backend ALIAS ftlpu_device_backend)

target_include_directories(ftlpu_device_backend
    PUBLIC
        include)

target_link_libraries(ftlpu_device_backend
    PUBLIC
        ftlpu::cmodel_selected_config)

add_library(ftlpu_runtime_adapter
    src/runtime/binary.cpp
    src/runtime/binary_program_adapter.cpp
    src/runtime/binding_transfer.cpp)

add_library(ftlpu::runtime_adapter ALIAS ftlpu_runtime_adapter)

target_include_directories(ftlpu_runtime_adapter
    PUBLIC
        include)

target_link_libraries(ftlpu_runtime_adapter
    PUBLIC
        ftlpu::device_backend)

add_executable(binary_program_adapter_test
    tests/runtime/binary_program_adapter_test.cpp)

target_link_libraries(binary_program_adapter_test
    PRIVATE
        ftlpu::runtime_adapter)

add_test(
    NAME binary_program_adapter_test
    COMMAND binary_program_adapter_test)

add_executable(compiled_binary_device_backend_test
    tests/runtime/compiled_binary_device_backend_test.cpp)

target_link_libraries(compiled_binary_device_backend_test
    PRIVATE
        ftlpu::runtime_adapter)

add_test(
    NAME compiled_binary_device_backend_test
    COMMAND compiled_binary_device_backend_test)

if(NOT FTLPU_BUILD_LEGACY_CMODEL_RUNTIME)
    return()
endif()

set(FTLPU_RUNTIME_SOURCES
    src/runtime/binary.cpp
    src/runtime/cmodel_runtime.cpp
    src/runtime/icu_program.cpp
    src/runtime/model_package.cpp
    src/runtime/model_session.cpp
    src/runtime/performance.cpp
    src/runtime/schedule_trace.cpp
    src/runtime/session_memory_planner.cpp)

add_library(ftlpu_software ${FTLPU_RUNTIME_SOURCES})

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
        ftlpu::cmodel)

add_executable(model_package_roundtrip_test
    tests/runtime/model_package_roundtrip_test.cpp)

target_link_libraries(model_package_roundtrip_test PRIVATE ftlpu_software)
add_test(
    NAME model_package_roundtrip_test
    COMMAND $<TARGET_FILE:model_package_roundtrip_test>
        ${CMAKE_CURRENT_BINARY_DIR}/test-output/roundtrip.ftlpum)

add_executable(command_vxm_binary_runtime_test
    tests/runtime/command_vxm_binary_runtime_test.cpp)

target_link_libraries(command_vxm_binary_runtime_test PRIVATE ftlpu_software)

add_executable(sxm_binary_roundtrip_test
    tests/runtime/sxm_binary_roundtrip_test.cpp)

target_link_libraries(sxm_binary_roundtrip_test PRIVATE ftlpu_software)
add_test(
    NAME sxm_binary_roundtrip_test
    COMMAND $<TARGET_FILE:sxm_binary_roundtrip_test>
        ${CMAKE_CURRENT_BINARY_DIR}/test-output/sxm-v5.ftlpu)

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
