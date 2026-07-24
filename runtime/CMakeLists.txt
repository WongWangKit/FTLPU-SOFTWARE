add_library(ftlpu_cmodel_static STATIC
    src/cmodel/cmodel_static.cpp)

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
    src/runtime/binary.cpp
    src/runtime/cmodel_runtime.cpp
    src/runtime/icu_program.cpp
    src/runtime/performance.cpp
    src/runtime/schedule_trace.cpp
    tests/runtime/command_vxm_binary_runtime_test.cpp
    tests/runtime/sxm_binary_roundtrip_test.cpp
    tests/runtime/compiled_sxm_command_runtime_test.cpp
    tests/runtime/compiled_queue_program_runtime_test.cpp
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp
    tests/runtime/compiled_smollm2_attention_softmax_runtime_test.cpp)
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
    src/runtime/performance.cpp
    src/runtime/schedule_trace.cpp)

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

add_executable(compiled_smollm2_135m_ffn_runtime_test
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp)

target_link_libraries(compiled_smollm2_135m_ffn_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_smollm2_135m_ffn_seq128_runtime_test
    tests/runtime/compiled_smollm2_135m_ffn_runtime_test.cpp)

target_compile_definitions(compiled_smollm2_135m_ffn_seq128_runtime_test
    PRIVATE FTLPU_TEST_SEQUENCE_LENGTH=128)
target_link_libraries(compiled_smollm2_135m_ffn_seq128_runtime_test PRIVATE ftlpu_software)

add_executable(compiled_smollm2_attention_softmax_runtime_test
    tests/runtime/compiled_smollm2_attention_softmax_runtime_test.cpp)

target_link_libraries(compiled_smollm2_attention_softmax_runtime_test PRIVATE ftlpu_software)
