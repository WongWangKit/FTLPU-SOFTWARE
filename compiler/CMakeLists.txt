find_package(Python3 COMPONENTS Interpreter)

# The compiler owns its LLVM/MLIR and StableHLO revisions directly. IREE is a
# reference compiler and is deliberately not part of this dependency chain.
set(FTLPU_LLVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project" CACHE PATH
    "Pinned LLVM project source directory")
set(FTLPU_STABLEHLO_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/stablehlo" CACHE PATH
    "Pinned StableHLO source directory")
set(FTLPU_MLIR_DIR "${CMAKE_SOURCE_DIR}/build-mlir/lib/cmake/mlir" CACHE PATH
    "Directory containing MLIRConfig.cmake")

if(MSVC AND MSVC_VERSION LESS 1936)
    message(FATAL_ERROR
        "FTLPU's pinned LLVM/MLIR requires MSVC 19.36 or newer. "
        "Use Visual Studio 2026, or a current Visual Studio 2022 toolset.")
endif()

find_package(MLIR REQUIRED CONFIG HINTS "${FTLPU_MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(TableGen)
include(HandleLLVMOptions)
include_directories(${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "Build StableHLO in the FTLPU MLIR process" FORCE)
add_subdirectory(
    "${FTLPU_STABLEHLO_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/stablehlo"
    EXCLUDE_FROM_ALL)

set(LLVM_TARGET_DEFINITIONS include/ftlpu/compiler/Dialect/Kernel/IR/KernelOps.td)
mlir_tablegen(ftlpu/compiler/Dialect/Kernel/IR/KernelOps.h.inc -gen-op-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Kernel/IR/KernelOps.cpp.inc -gen-op-defs)
mlir_tablegen(ftlpu/compiler/Dialect/Kernel/IR/KernelOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Kernel/IR/KernelOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(FTLPUKernelOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/ftlpu/compiler/Dialect/Tensor/IR/TensorOps.td)
mlir_tablegen(ftlpu/compiler/Dialect/Tensor/IR/TensorOps.h.inc -gen-op-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Tensor/IR/TensorOps.cpp.inc -gen-op-defs)
mlir_tablegen(ftlpu/compiler/Dialect/Tensor/IR/TensorOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Tensor/IR/TensorOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(FTLPUTensorOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/ftlpu/compiler/Dialect/Stream/IR/StreamOps.td)
mlir_tablegen(ftlpu/compiler/Dialect/Stream/IR/StreamOps.h.inc -gen-op-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Stream/IR/StreamOps.cpp.inc -gen-op-defs)
mlir_tablegen(ftlpu/compiler/Dialect/Stream/IR/StreamOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Stream/IR/StreamOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(FTLPUStreamOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/ftlpu/compiler/Dialect/Schedule/IR/ScheduleOps.td)
mlir_tablegen(ftlpu/compiler/Dialect/Schedule/IR/ScheduleOps.h.inc -gen-op-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Schedule/IR/ScheduleOps.cpp.inc -gen-op-defs)
mlir_tablegen(ftlpu/compiler/Dialect/Schedule/IR/ScheduleOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Schedule/IR/ScheduleOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(FTLPUScheduleOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/ftlpu/compiler/Dialect/Command/IR/CommandOps.td)
mlir_tablegen(ftlpu/compiler/Dialect/Command/IR/CommandOps.h.inc -gen-op-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Command/IR/CommandOps.cpp.inc -gen-op-defs)
mlir_tablegen(ftlpu/compiler/Dialect/Command/IR/CommandOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ftlpu/compiler/Dialect/Command/IR/CommandOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(FTLPUCommandOpsIncGen)

# Ninja cannot recover this cross-subdirectory ABI dependency from localized
# MSVC /showIncludes output. Binding and queue layout changes must always
# rebuild the binary translator and its CLI entry point.
file(GLOB_RECURSE FTLPU_RUNTIME_ABI_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/runtime/include/ftlpu/*.hpp")

add_library(ftlpu_compiler
    src/Dialect/Kernel/IR/KernelDialect.cpp
    src/Dialect/Kernel/Analysis/AttentionGraph.cpp
    src/Dialect/Kernel/Analysis/FfnGraph.cpp
    src/Dialect/Kernel/Transforms/StableHloToKernel.cpp
    src/Dialect/Tensor/IR/TensorDialect.cpp
    src/Dialect/Tensor/Analysis/AttentionTaskGraph.cpp
    src/Dialect/Tensor/Analysis/PhysicalMemoryAllocator.cpp
    src/Dialect/Tensor/Transforms/AttentionToTensor.cpp
    src/Dialect/Tensor/Transforms/FfnToTensor.cpp
    src/Dialect/Tensor/Transforms/KernelToTensorLowering.cpp
    src/Dialect/Tensor/Transforms/KernelToTensor.cpp
    src/Dialect/Tensor/Transforms/MatmulToTensor.cpp
    src/Dialect/Tensor/Transforms/RmsNormToTensor.cpp
    src/Dialect/Tensor/Transforms/SwigluToTensor.cpp
    src/Dialect/Stream/IR/StreamDialect.cpp
    src/Dialect/Stream/Analysis/StreamAllocator.cpp
    src/Dialect/Stream/Analysis/StreamRoutePlan.cpp
    src/Dialect/Stream/Transforms/AttentionToStream.cpp
    src/Dialect/Stream/Transforms/FfnToStream.cpp
    src/Dialect/Stream/Transforms/ElementwiseToStream.cpp
    src/Dialect/Stream/Transforms/MatmulToStream.cpp
    src/Dialect/Stream/Transforms/RmsNormToStream.cpp
    src/Dialect/Stream/Transforms/SwigluToStream.cpp
    src/Dialect/Stream/Transforms/TensorToStream.cpp
    src/Dialect/Schedule/IR/ScheduleDialect.cpp
    src/Dialect/Schedule/Analysis/AttentionWorkPlanner.cpp
    src/Dialect/Schedule/Analysis/AttentionProjectionPlanner.cpp
    src/Dialect/Schedule/Analysis/AttentionSoftmaxPlanner.cpp
    src/Dialect/Schedule/Analysis/AttentionProjectionStagePlanner.cpp
    src/Dialect/Schedule/Analysis/AttentionRopeStagePlanner.cpp
    src/Dialect/Schedule/Analysis/AttentionSoftmaxStagePlanner.cpp
    src/Dialect/Schedule/Analysis/AttentionPvStagePlanner.cpp
    src/Dialect/Schedule/Analysis/AttentionOutputProjectionStagePlanner.cpp
    src/Dialect/Schedule/Analysis/AttentionStagePlan.cpp
    src/Dialect/Schedule/Analysis/FfnSchedulePlanner.cpp
    src/Dialect/Schedule/Analysis/FfnScheduleBuilders.cpp
    src/Dialect/Schedule/Analysis/FfnWeightLoadScheduleBuilder.cpp
    src/Dialect/Schedule/Analysis/FfnProjectionScheduleBuilder.cpp
    src/Dialect/Schedule/Analysis/FfnProjectionTimelinePlanner.cpp
    src/Dialect/Schedule/Analysis/FfnBlock8ProjectionPlanner.cpp
    src/Dialect/Schedule/Analysis/FfnSwishScheduleBuilder.cpp
    src/Dialect/Schedule/Analysis/FfnSwishPlanner.cpp
    src/Dialect/Schedule/Analysis/FfnDownProjectionScheduleBuilder.cpp
    src/Dialect/Schedule/Analysis/FfnDownProjectionTimelinePlanner.cpp
    src/Dialect/Schedule/Analysis/LPUResourceModel.cpp
    src/Dialect/Schedule/Analysis/ResourceScheduler.cpp
    src/Dialect/Schedule/Analysis/StreamFabricScheduler.cpp
    src/Dialect/Schedule/Analysis/ScheduleGraph.cpp
    src/Dialect/Schedule/Analysis/SchedulePlan.cpp
    src/Dialect/Schedule/Analysis/AttentionTaskGraph.cpp
    src/Dialect/Schedule/Analysis/AttentionMemoryLayout.cpp
    src/Dialect/Schedule/Transforms/AttentionEmitterUtils.cpp
    src/Dialect/Schedule/Transforms/SxmWavefrontEmitterUtils.cpp
    src/Dialect/Schedule/Transforms/AttentionOutputProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionBlock8ProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionBlock8OutputProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionPvEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionRopeEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionSoftmaxEmitter.cpp
    src/Dialect/Schedule/Transforms/AssignWeightBank.cpp
    src/Dialect/Schedule/Transforms/FfnDownProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnBlock8ProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnBlock8DownProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnEmitterUtils.cpp
    src/Dialect/Schedule/Transforms/FfnProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnStageEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnSwishEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnSwishStageEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnWeightLoadEmitter.cpp
    src/Dialect/Schedule/Transforms/ElementwiseScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/LinearProjectionScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/MatmulScheduleEmitter.cpp
      src/Dialect/Schedule/Transforms/RmsNormScheduleEmitter.cpp
      src/Dialect/Schedule/Transforms/ScheduleCompression.cpp
      src/Dialect/Schedule/Transforms/ScheduleEmitterUtils.cpp
    src/Dialect/Schedule/Transforms/StreamToSchedule.cpp
    src/Dialect/Schedule/Transforms/SwigluScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/VerifySchedule.cpp
    src/Dialect/Command/IR/CommandDialect.cpp
    src/Dialect/Command/Transforms/ScheduleToCommand.cpp
    src/Target/LPUTargetModel.cpp
    src/Target/MxmExecutionStrategy.cpp
    src/Target/CommandBinary.cpp)

if(FTLPU_ENABLE_ADDRESS_SANITIZER AND MSVC)
    target_compile_options(ftlpu_compiler PRIVATE /fsanitize=address)
    target_compile_definitions(ftlpu_compiler PRIVATE
        _DISABLE_VECTOR_ANNOTATION
        _DISABLE_STRING_ANNOTATION
        _DISABLE_OPTIONAL_ANNOTATION)
endif()

# Ninja cannot reliably parse localized MSVC /showIncludes output. Keep
# explicit dependency edges for all compiler sources: LPUTargetModel is also
# exposed transitively through allocator and planner headers, and has inline
# accessors whose offsets change as architecture parameters evolve.
get_target_property(FTLPU_COMPILER_LIBRARY_SOURCES ftlpu_compiler SOURCES)
set(FTLPU_TARGET_MODEL_CONSUMER_SOURCES
    ${FTLPU_COMPILER_LIBRARY_SOURCES}
    tools/ftlpu_opt.cpp
    tools/ftlpu_translate.cpp
    tools/ftlpu_compile.cpp
    tests/attention_projection_planner_test.cpp
    tests/attention_target_layout_test.cpp
    tests/attention_work_planner_test.cpp
    tests/physical_memory_allocator_test.cpp)
set_source_files_properties(${FTLPU_TARGET_MODEL_CONSUMER_SOURCES}
    PROPERTIES OBJECT_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ftlpu/compiler/Target/lpu_target_model.hpp;${CMAKE_SOURCE_DIR}/runtime/include/ftlpu/software/runtime/target_abi.hpp")
set_source_files_properties(
    src/Target/CommandBinary.cpp
    tools/ftlpu_translate.cpp
    PROPERTIES OBJECT_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ftlpu/compiler/Target/lpu_target_model.hpp;${FTLPU_RUNTIME_ABI_HEADERS}")

add_dependencies(ftlpu_compiler FTLPUKernelOpsIncGen FTLPUTensorOpsIncGen
    FTLPUStreamOpsIncGen FTLPUScheduleOpsIncGen FTLPUCommandOpsIncGen)

target_include_directories(ftlpu_compiler
    PUBLIC
        include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS})
target_compile_definitions(ftlpu_compiler PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(ftlpu_compiler
    PUBLIC
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRFuncDialect
        MLIRSupport
        StablehloOps
        ftlpu_software)

add_executable(ftlpu_opt tools/ftlpu_opt.cpp)
target_link_libraries(ftlpu_opt PRIVATE ftlpu_compiler)
add_executable(ftlpu_translate tools/ftlpu_translate.cpp)
set_target_properties(ftlpu_translate PROPERTIES OUTPUT_NAME "ftlpu-translate")
target_link_libraries(ftlpu_translate PRIVATE ftlpu_compiler)
add_executable(ftlpu_compile tools/ftlpu_compile.cpp)
set_target_properties(ftlpu_compile PROPERTIES OUTPUT_NAME "ftlpu-compile")
target_link_libraries(ftlpu_compile PRIVATE ftlpu_compiler)
if(FTLPU_ENABLE_ADDRESS_SANITIZER AND MSVC)
    target_compile_options(ftlpu_translate PRIVATE /fsanitize=address)
    target_compile_definitions(ftlpu_translate PRIVATE
        _DISABLE_VECTOR_ANNOTATION
        _DISABLE_STRING_ANNOTATION
        _DISABLE_OPTIONAL_ANNOTATION)
    target_link_options(ftlpu_translate PRIVATE /INCREMENTAL:NO)
endif()

add_executable(attention_work_planner_test tests/attention_work_planner_test.cpp)
target_link_libraries(attention_work_planner_test PRIVATE ftlpu_compiler)
add_test(NAME attention_work_planner_test COMMAND $<TARGET_FILE:attention_work_planner_test>)
add_executable(attention_projection_planner_test tests/attention_projection_planner_test.cpp)
target_link_libraries(attention_projection_planner_test PRIVATE ftlpu_compiler)
add_test(NAME attention_projection_planner_test COMMAND $<TARGET_FILE:attention_projection_planner_test>)
add_executable(attention_stage_plan_test tests/attention_stage_plan_test.cpp)
target_link_libraries(attention_stage_plan_test PRIVATE ftlpu_compiler)
add_test(NAME attention_stage_plan_test COMMAND $<TARGET_FILE:attention_stage_plan_test>)
add_executable(attention_rope_emitter_test tests/attention_rope_emitter_test.cpp)
target_include_directories(attention_rope_emitter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Dialect/Schedule/Transforms)
target_link_libraries(attention_rope_emitter_test PRIVATE ftlpu_compiler)
add_test(NAME attention_rope_emitter_test COMMAND $<TARGET_FILE:attention_rope_emitter_test>)
add_executable(attention_target_layout_test tests/attention_target_layout_test.cpp)
target_link_libraries(attention_target_layout_test PRIVATE ftlpu_compiler)
add_test(NAME attention_target_layout_test COMMAND $<TARGET_FILE:attention_target_layout_test>)
add_executable(schedule_graph_test tests/schedule_graph_test.cpp)
target_link_libraries(schedule_graph_test PRIVATE ftlpu_compiler)
add_test(NAME schedule_graph_test COMMAND $<TARGET_FILE:schedule_graph_test>)
add_executable(schedule_plan_test tests/schedule_plan_test.cpp)
target_link_libraries(schedule_plan_test PRIVATE ftlpu_compiler)
add_test(NAME schedule_plan_test COMMAND $<TARGET_FILE:schedule_plan_test>)
add_executable(stream_fabric_scheduler_test
    tests/stream_fabric_scheduler_test.cpp)
target_link_libraries(stream_fabric_scheduler_test PRIVATE ftlpu_compiler)
add_test(NAME stream_fabric_scheduler_test
    COMMAND $<TARGET_FILE:stream_fabric_scheduler_test>)
add_executable(stream_route_plan_test tests/stream_route_plan_test.cpp)
target_link_libraries(stream_route_plan_test PRIVATE ftlpu_compiler)
add_test(NAME stream_route_plan_test COMMAND $<TARGET_FILE:stream_route_plan_test>)
add_executable(ffn_schedule_builders_test tests/ffn_schedule_builders_test.cpp)
target_link_libraries(ffn_schedule_builders_test PRIVATE ftlpu_compiler)
add_test(NAME ffn_schedule_builders_test COMMAND $<TARGET_FILE:ffn_schedule_builders_test>)
add_executable(mxm_execution_strategy_test
    tests/mxm_execution_strategy_test.cpp)
target_link_libraries(mxm_execution_strategy_test PRIVATE ftlpu_compiler)
add_test(NAME mxm_execution_strategy_test
    COMMAND $<TARGET_FILE:mxm_execution_strategy_test>)
add_executable(ffn_swish_emitter_test tests/ffn_swish_emitter_test.cpp)
target_include_directories(ffn_swish_emitter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Dialect/Schedule/Transforms)
target_link_libraries(ffn_swish_emitter_test PRIVATE ftlpu_compiler)
add_test(NAME ffn_swish_emitter_test COMMAND $<TARGET_FILE:ffn_swish_emitter_test>)
add_executable(physical_memory_allocator_test tests/physical_memory_allocator_test.cpp)
target_link_libraries(physical_memory_allocator_test PRIVATE ftlpu_compiler)
add_test(NAME physical_memory_allocator_test COMMAND $<TARGET_FILE:physical_memory_allocator_test>)
if(MSVC)
    target_compile_options(ftlpu_opt PRIVATE /EHsc)
    target_compile_options(ftlpu_translate PRIVATE /EHsc)
endif()

if(Python3_Interpreter_FOUND)
    add_test(
        NAME qwen2_5_1_5b_decoder_layer_stablehlo_to_stream_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_1_5b_decoder_layer_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq128.stablehlo.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen2_5_1_5b_decoder_layer)
    add_test(
        NAME qwen2_5_1_5b_paged_weight_layout_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_1_5b_decoder_layer_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq128.stablehlo.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/examples/targets/cmodel_large_sram.json
            --weight-bank 1
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen2_5_1_5b_paged)
    add_custom_target(
        qwen2_5_1_5b_decoder_layer_pipeline
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_1_5b_decoder_layer_pipeline_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq128.stablehlo.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen2_5_1_5b_decoder_layer
        DEPENDS ftlpu_opt ftlpu_translate
        USES_TERMINAL)
    add_test(
        NAME lm_head_target_aware_tensor_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/lm_head_tensor_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/lm_head/lm_head_128x576x4096.stablehlo.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/examples/targets/cmodel_large_sram.json
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/lm_head/lm_head.tensor.mlir)
    if(EXISTS "${CMAKE_SOURCE_DIR}/.cache/hf/SmolLM2-135M/model.safetensors")
        add_test(
            NAME smollm2_135m_real_int8_lm_head_cmodel_test
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_lm_head_binary_runtime_test.py
                --opt $<TARGET_FILE:ftlpu_opt>
                --translate $<TARGET_FILE:ftlpu_translate>
                --runtime-test $<TARGET_FILE:compiled_smollm2_lm_head_runtime_test>
                --model-dir ${CMAKE_SOURCE_DIR}/.cache/hf/SmolLM2-135M
                --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
                --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_lm_head)
        set_tests_properties(
            smollm2_135m_real_int8_lm_head_cmodel_test
            PROPERTIES TIMEOUT 900)
        add_custom_target(
            smollm2_135m_lm_head_pipeline
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_lm_head_binary_runtime_test.py
                --opt $<TARGET_FILE:ftlpu_opt>
                --translate $<TARGET_FILE:ftlpu_translate>
                --runtime-test $<TARGET_FILE:compiled_smollm2_lm_head_runtime_test>
                --model-dir ${CMAKE_SOURCE_DIR}/.cache/hf/SmolLM2-135M
                --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
                --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_lm_head
            DEPENDS
                ftlpu_opt
                ftlpu_translate
                compiled_smollm2_lm_head_runtime_test
            USES_TERMINAL)
    endif()
    add_test(
        NAME schedule_verifier_overlap_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/schedule_verifier_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/schedule_resource_overlap.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/schedule_verifier/invalid.mlir)
    add_test(
        NAME smollm2_135m_ffn_stablehlo_to_stream_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_ffn_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_ffn/ffn.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_ffn)
    add_test(
        NAME generic_w8a16_bf16_ffn_lowering_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/generic_w8a16_ffn_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/w8a16_ffn/ffn_32x64x128x64_bf16.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/generic_w8a16_bf16_ffn_32x64x128x64)
    add_test(
        NAME lpu_target_configuration_lowering_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/target_configuration_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/w8a16_ffn/ffn_32x64x128x64_bf16.stablehlo.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/examples/targets/exploration_40_streams.json
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/lpu_target_configuration)
    add_test(
        NAME w8a16_ffn_mtile_weight_reuse_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/w8a16_ffn_mtile_weight_reuse_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/w8a16_ffn/ffn_128x64x128x64.stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/w8a16_ffn_mtile_weight_reuse/ffn.schedule.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_legacy_mxm.json)
    add_test(
        NAME smollm2_135m_ffn_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_ffn_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_135m_ffn_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_ffn/ffn.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_ffn_binary_runtime
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_legacy_mxm.json)
    add_test(
        NAME smollm2_135m_ffn_seq128_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_ffn_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_135m_ffn_seq128_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_ffn/ffn_seq128.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_ffn_seq128_binary_runtime
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_legacy_mxm.json)
    add_test(
        NAME smollm2_135m_ffn_seq128_block8_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_ffn_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_135m_ffn_seq128_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_ffn/ffn_seq128.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_ffn_seq128_block8_binary_runtime
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
            --mxm-execution block8)
    add_test(
        NAME smollm2_135m_ffn_seq128_block8_fused_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_ffn_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_135m_ffn_seq128_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_ffn/ffn_seq128.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_ffn_seq128_block8_fused_binary_runtime
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
            --mxm-execution block8
            --ffn-schedule fused)
    add_test(
        NAME smollm2_135m_attention_stablehlo_to_kernel_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_attention_kernel_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_attention/attention_seq128.stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_attention/kernel/attention.kernel.mlir)
    add_test(
        NAME smollm2_135m_attention_stablehlo_to_tensor_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_attention_tensor_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_attention/attention_seq128.stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_attention/tensor/attention.tensor.mlir
            --legacy-target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_legacy_mxm.json)
    add_test(
        NAME tensor_attention_graph_verifier_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/tensor_attention_graph_verifier_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/tensor_attention_incomplete.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/tensor_attention_graph_verifier/invalid.mlir)
    add_test(
        NAME smollm2_135m_attention_stablehlo_to_stream_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_attention_stream_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_attention/attention_seq128.stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_attention/stream/attention.stream.mlir)
    add_test(
        NAME smollm2_135m_attention_stablehlo_to_schedule_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_attention_schedule_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_attention/attention_seq128.stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_attention/schedule/attention.schedule.mlir)
    add_test(
        NAME smollm2_attention_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_attention_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_attention_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_attention/attention_seq128.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_attention_binary_runtime
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
            --attention-schedule fused)
    add_test(
        NAME rmsnorm_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rmsnorm_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_rmsnorm_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/rmsnorm/rmsnorm_128x576.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/rmsnorm_binary_runtime
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json)
    add_test(
        NAME rmsnorm_feedback_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rmsnorm_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_rmsnorm_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/rmsnorm/rmsnorm_128x576.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/rmsnorm_feedback_binary_runtime
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
            --strategy vxm-feedback)
    add_test(
        NAME smollm2_decoder_layer_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_decoder_layer_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_decoder_layer_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_decoder_layer/decoder_layer_seq128.stablehlo.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/examples/targets/cmodel_large_sram.json
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_decoder_layer_binary_runtime)
    add_test(
        NAME smollm2_decoder_layer_block8_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_decoder_layer_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_decoder_layer_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_decoder_layer/decoder_layer_seq128.stablehlo.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_decoder_layer_block8_binary_runtime
            --mxm-execution block8)
    add_custom_target(
        smollm2_decoder_layer_pipeline
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_decoder_layer_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_decoder_layer_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_decoder_layer/decoder_layer_seq128.stablehlo.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/examples/targets/cmodel_large_sram.json
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_decoder_layer_pipeline
            --trace
            --render
        DEPENDS
            ftlpu_opt
            ftlpu_translate
            compiled_smollm2_decoder_layer_runtime_test
        USES_TERMINAL)
    add_test(
        NAME schedule_wave_compression_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/schedule_wave_compression_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/schedule_wave_compression.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/examples/targets/cmodel_large_sram.json
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/schedule_wave_compression/command.mlir)
    add_test(
        NAME stablehlo_matmul_320_to_ftlpu_kernel_mlir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/stablehlo_matmul_mlir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/simple_stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/stablehlo_matmul_320_to_ftlpu_kernel/matmul_320.kernel.mlir)
    add_test(
        NAME stablehlo_matmul_320_to_ftlpu_tensor_mlir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/stablehlo_matmul_tensor_mlir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/simple_stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/stablehlo_matmul_320_to_ftlpu_tensor/matmul_320.tensor.mlir)
    add_test(
        NAME ftlpu_tensor_lifetime_reuse_mlir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/tensor_lifetime_reuse_mlir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/kernel_matmul_lifetime.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/ftlpu_tensor_lifetime_reuse/matmul_lifetime.tensor.mlir)
    add_test(
        NAME stablehlo_matmul_320_to_ftlpu_stream_mlir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/stablehlo_matmul_stream_mlir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/simple_stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/stablehlo_matmul_320_to_ftlpu_stream/matmul_320.stream.mlir)
    add_test(
        NAME ftlpu_stream_allocator_reuse_mlir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/stream_allocator_reuse_mlir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/kernel_matmul_lifetime.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/ftlpu_stream_allocator_reuse/matmul_lifetime.stream.mlir)
    add_test(
        NAME ftlpu_schedule_resource_and_dependency_mlir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/schedule_resource_dependency_mlir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/kernel_matmul_lifetime.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/ftlpu_schedule_resource_and_dependency/matmul_lifetime.schedule.mlir)
    add_test(
        NAME ftlpu_command_vxm_binary_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/command_vxm_binary_runtime_test.py
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:command_vxm_binary_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/command_vxm.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/ftlpu_command_vxm_binary_runtime/command_vxm.ftlpu)
    add_test(
        NAME ftlpu_command_mxm_column_binary_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/command_vxm_binary_runtime_test.py
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:command_mxm_column_binary_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/command_mxm_column.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/ftlpu_command_mxm_column/command_mxm_column.ftlpu)
    add_test(
        NAME ftlpu_command_bf16_binary_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/command_vxm_binary_runtime_test.py
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:command_bf16_binary_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/command_bf16.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/ftlpu_command_bf16/command_bf16.ftlpu)
    add_test(
        NAME ftlpu_command_cmodel_compat_binary_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/command_vxm_binary_runtime_test.py
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:command_cmodel_compat_binary_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/command_cmodel_compat.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/ftlpu_command_cmodel_compat/command_cmodel_compat.ftlpu)
    add_test(
        NAME block8_linear_projection_e2e_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/block8_linear_projection_e2e_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:block8_linear_projection_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/block8_linear_projection.stablehlo.mlir
            --block8-target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_block8_1mxm.json
            --legacy-target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_legacy_mxm.json
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/block8_linear_projection_e2e)
    add_test(
        NAME direct_binary_compile_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/direct_binary_compile_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --compile $<TARGET_FILE:ftlpu_compile>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/w8a16_ffn/ffn_32x64x128x64_bf16.stablehlo.mlir
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/examples/targets/cmodel_large_sram.json
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/direct_binary_compile)
endif()
