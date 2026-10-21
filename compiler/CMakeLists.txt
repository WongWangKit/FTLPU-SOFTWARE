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
    src/Dialect/Kernel/Analysis/DecoderGraph.cpp
    src/Dialect/Kernel/Transforms/StableHloToKernel.cpp
    src/Dialect/Kernel/Transforms/ConnectedKernelGraphToSchedule.cpp
    src/Dialect/Kernel/Transforms/LogicalKernelGraphToSchedule.cpp
    src/Dialect/Tensor/IR/TensorDialect.cpp
    src/Dialect/Tensor/Analysis/AttentionTaskGraph.cpp
    src/Dialect/Tensor/Analysis/AttentionWeightTilePlan.cpp
    src/Dialect/Tensor/Analysis/PhysicalMemoryAllocator.cpp
    src/Dialect/Tensor/Analysis/FfnWeightTilePlan.cpp
    src/Dialect/Tensor/Transforms/AttentionToTensor.cpp
      src/Dialect/Tensor/Transforms/DepthwiseConv1DToTensor.cpp
      src/Dialect/Tensor/Transforms/GatedDeltaRuleToTensor.cpp
    src/Dialect/Tensor/Transforms/FfnToTensor.cpp
    src/Dialect/Tensor/Transforms/KernelToTensorLowering.cpp
    src/Dialect/Tensor/Transforms/KernelToTensor.cpp
    src/Dialect/Tensor/Transforms/MatmulToTensor.cpp
    src/Dialect/Tensor/Transforms/L2NormToTensor.cpp
    src/Dialect/Tensor/Transforms/RmsNormToTensor.cpp
    src/Dialect/Tensor/Transforms/SwigluToTensor.cpp
    src/Dialect/Stream/IR/StreamDialect.cpp
    src/Dialect/Stream/Analysis/StreamAllocator.cpp
    src/Dialect/Stream/Analysis/StreamRoutePlan.cpp
    src/Dialect/Stream/Transforms/AttentionToStream.cpp
      src/Dialect/Stream/Transforms/DepthwiseConv1DToStream.cpp
      src/Dialect/Stream/Transforms/GatedDeltaRuleToStream.cpp
    src/Dialect/Stream/Transforms/FfnToStream.cpp
    src/Dialect/Stream/Transforms/ElementwiseToStream.cpp
    src/Dialect/Stream/Transforms/MatmulToStream.cpp
    src/Dialect/Stream/Transforms/L2NormToStream.cpp
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
    src/Dialect/Schedule/Analysis/FfnUp3DLowering.cpp
    src/Dialect/Schedule/Analysis/FfnWeightTilePlanner.cpp
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
    src/Dialect/Schedule/Analysis/PagedWeightResidency.cpp
    src/Dialect/Schedule/Transforms/AttentionEmitterUtils.cpp
    src/Dialect/Schedule/Transforms/SxmWavefrontEmitterUtils.cpp
    src/Dialect/Schedule/Transforms/AttentionOutputProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionPvEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionRopeEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionSoftmaxEmitter.cpp
    src/Dialect/Schedule/Transforms/AssignWeightBank.cpp
    src/Dialect/Schedule/Transforms/FfnDownProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnEmitterUtils.cpp
    src/Dialect/Schedule/Transforms/FfnProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnStageEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnSwishEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnSwishStageEmitter.cpp
    src/Dialect/Schedule/Transforms/FfnWeightLoadEmitter.cpp
    src/Dialect/Schedule/Transforms/ElementwiseScheduleEmitter.cpp
      src/Dialect/Schedule/Transforms/DepthwiseConv1DScheduleEmitter.cpp
      src/Dialect/Schedule/Transforms/GatedDeltaRuleScheduleEmitter.cpp
      src/Dialect/Schedule/Transforms/VectorAluScheduleEmitter.cpp
      src/Dialect/Schedule/Transforms/LinearProjectionScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/MatmulScheduleEmitter.cpp
      src/Dialect/Schedule/Transforms/RmsNormScheduleEmitter.cpp
      src/Dialect/Schedule/Transforms/ScheduleCompression.cpp
      src/Dialect/Schedule/Transforms/ScheduleEmitterUtils.cpp
    src/Dialect/Schedule/Transforms/StreamToSchedule.cpp
    src/Dialect/Schedule/Transforms/SwigluScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/VerifySchedule.cpp
    src/Dialect/Command/IR/CommandDialect.cpp
    src/Dialect/Command/Transforms/Fu3DCommandMaterializer.cpp
    src/Dialect/Command/Transforms/StandaloneFfnUp3DToCommand.cpp
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
    tools/ftlpu_graph_compile.cpp
    tests/attention_projection_planner_test.cpp
    tests/attention_target_layout_test.cpp
    tests/attention_work_planner_test.cpp
    tests/physical_memory_allocator_test.cpp)
set_source_files_properties(${FTLPU_TARGET_MODEL_CONSUMER_SOURCES}
    PROPERTIES OBJECT_DEPENDS
        "${FTLPU_GENERATED_HARDWARE_CONFIG};${FTLPU_CMODEL_DIR}/include/ftlpu/core/hardware_params.hpp;${CMAKE_CURRENT_SOURCE_DIR}/include/ftlpu/compiler/Target/lpu_target_model.hpp;${CMAKE_SOURCE_DIR}/runtime/include/ftlpu/software/runtime/target_abi.hpp")
set(FTLPU_ATTENTION_EMITTER_CONSUMER_SOURCES
    src/Dialect/Schedule/Transforms/AttentionEmitterUtils.cpp
    src/Dialect/Schedule/Transforms/AttentionProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionOutputProjectionEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionPvEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionRopeEmitter.cpp
    src/Dialect/Schedule/Transforms/AttentionSoftmaxEmitter.cpp
    src/Dialect/Schedule/Transforms/ElementwiseScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/LinearProjectionScheduleEmitter.cpp
    src/Dialect/Schedule/Transforms/RmsNormScheduleEmitter.cpp)
set_property(SOURCE ${FTLPU_ATTENTION_EMITTER_CONSUMER_SOURCES}
    APPEND PROPERTY OBJECT_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/Dialect/Schedule/Transforms/AttentionEmitterUtils.hpp")
set_property(SOURCE src/Dialect/Schedule/IR/ScheduleDialect.cpp
    APPEND PROPERTY OBJECT_DEPENDS
        "${CMAKE_CURRENT_BINARY_DIR}/ftlpu/compiler/Dialect/Schedule/IR/ScheduleOps.cpp.inc")
file(GLOB_RECURSE FTLPU_CMODEL_COMPILER_HEADERS CONFIGURE_DEPENDS
    "${FTLPU_CMODEL_DIR}/include/ftlpu/*.hpp")
set(FTLPU_CMODEL_COMPILER_CONSUMER_SOURCES
    src/Dialect/Schedule/Analysis/FfnUp3DLowering.cpp
    src/Dialect/Command/IR/CommandDialect.cpp
    src/Dialect/Command/Transforms/Fu3DCommandMaterializer.cpp
    src/Dialect/Command/Transforms/StandaloneFfnUp3DToCommand.cpp
    src/Target/CommandBinary.cpp
    tools/ftlpu_translate.cpp
    tools/ftlpu_compile.cpp
    tests/ffn_up_3d_lowering_test.cpp)
list(APPEND FTLPU_CMODEL_COMPILER_CONSUMER_SOURCES
    tests/vxm_run_2d_command_test.cpp)
set_property(SOURCE ${FTLPU_CMODEL_COMPILER_CONSUMER_SOURCES}
    APPEND PROPERTY OBJECT_DEPENDS
        "${FTLPU_GENERATED_HARDWARE_CONFIG};${FTLPU_CMODEL_COMPILER_HEADERS};${CMAKE_CURRENT_SOURCE_DIR}/include/ftlpu/compiler/Target/lpu_target_model.hpp;${FTLPU_RUNTIME_ABI_HEADERS}")

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
add_executable(ftlpu_graph_compile tools/ftlpu_graph_compile.cpp)
set_target_properties(ftlpu_graph_compile PROPERTIES OUTPUT_NAME "ftlpu-graph-compile")
target_link_libraries(ftlpu_graph_compile PRIVATE ftlpu_compiler)
if(MSVC)
    target_compile_options(ftlpu_graph_compile PRIVATE /EHsc)
endif()
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    # Large command streams such as Qwen seq128 exceed the default 2 GiB
    # user-mode address-space ceiling of a 32-bit Windows executable.
    target_link_options(ftlpu_compile PRIVATE /LARGEADDRESSAWARE)
    target_link_options(ftlpu_opt PRIVATE /LARGEADDRESSAWARE)
endif()
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
add_executable(attention_memory_layout_test tests/attention_memory_layout_test.cpp)
target_link_libraries(attention_memory_layout_test PRIVATE ftlpu_compiler)
add_test(NAME attention_memory_layout_test COMMAND $<TARGET_FILE:attention_memory_layout_test>)
add_executable(attention_weight_tile_planner_test
    tests/attention_weight_tile_planner_test.cpp)
target_link_libraries(attention_weight_tile_planner_test PRIVATE ftlpu_compiler)
add_test(NAME attention_weight_tile_planner_test
    COMMAND $<TARGET_FILE:attention_weight_tile_planner_test>)
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
add_test(NAME ffn_projection_affine_test
    COMMAND $<TARGET_FILE:ffn_schedule_builders_test> --affine-only)
add_executable(elementwise_vxm_sr_direction_test tests/elementwise_vxm_sr_direction_test.cpp)
target_link_libraries(elementwise_vxm_sr_direction_test PRIVATE ftlpu_compiler)
add_test(NAME elementwise_vxm_sr_direction_test COMMAND $<TARGET_FILE:elementwise_vxm_sr_direction_test>)
add_executable(ffn_up_3d_lowering_test tests/ffn_up_3d_lowering_test.cpp)
target_link_libraries(ffn_up_3d_lowering_test PRIVATE ftlpu_compiler)
add_test(NAME ffn_up_3d_lowering_test COMMAND $<TARGET_FILE:ffn_up_3d_lowering_test>)
add_executable(vxm_run_2d_command_test tests/vxm_run_2d_command_test.cpp)
target_link_libraries(vxm_run_2d_command_test PRIVATE ftlpu_compiler)
add_test(NAME vxm_run_2d_command_test COMMAND $<TARGET_FILE:vxm_run_2d_command_test>)
add_executable(ffn_weight_tile_planner_test
    tests/ffn_weight_tile_planner_test.cpp)
target_link_libraries(ffn_weight_tile_planner_test PRIVATE ftlpu_compiler)
add_test(NAME ffn_weight_tile_planner_test
    COMMAND $<TARGET_FILE:ffn_weight_tile_planner_test>)
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
    add_test(NAME smollm2_seq128_compile_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_seq128_compile_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --inspect $<TARGET_FILE:ftlpu_binary_inspect>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_decoder_layer/decoder_layer_seq128.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/seq128_compile)
    add_test(NAME direct_raw_macro_cli_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/direct_raw_macro_cli_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --translate $<TARGET_FILE:ftlpu_translate>
            --inspect $<TARGET_FILE:ftlpu_binary_inspect>
            --runtime-test $<TARGET_FILE:raw_macro_cli_replay_test>
            --numeric-test $<TARGET_FILE:compiled_vxm_sr_ffn_runtime_test>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/direct_raw_macro_cli)
    add_test(NAME macro_rectangle_folding_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/macro_rectangle_folding_test.py
            --translate $<TARGET_FILE:ftlpu_translate>
            --inspect $<TARGET_FILE:ftlpu_binary_inspect>
            --runtime-test $<TARGET_FILE:macro_bitstream_test>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/macro_rectangle_folding)
    # Keep the native compiler usable without the optional HF/XLA environment.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pytest, numpy"
        RESULT_VARIABLE FTLPU_FRONTEND_TEST_DEPS
        OUTPUT_QUIET ERROR_QUIET)
    if(FTLPU_FRONTEND_TEST_DEPS EQUAL 0)
        add_test(NAME hf_frontend_protocol_test
            COMMAND ${Python3_EXECUTABLE} -B -m pytest -q -p no:cacheprovider
                --basetemp ${CMAKE_CURRENT_BINARY_DIR}/hf_frontend_protocol_tmp
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/frontend_bundle_test.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/frontend_entry_test.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/frontend_weights_test.py
                -k "not xla")
    endif()
    add_test(NAME icu_short_lowering_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/icu_short_lowering_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --export $<TARGET_FILE:ftlpu_icu_program_export>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/icu_short)
    add_test(
        NAME model_test_entrypoint_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_SOURCE_DIR}/tools/tests/test_run_model_tests.py)
    add_test(
        NAME attention_probability_layout_test
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/attention_probability_layout_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/attention_probability_layout)
    add_test(
        NAME standalone_ffn_up_3d_compile_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/standalone_ffn_up_3d_compile_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --inspect $<TARGET_FILE:ftlpu_binary_inspect>
            --runtime-test $<TARGET_FILE:compiled_queue_program_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/qwen2_5_up_direct_3d.stream.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/direct_ffn_up_3d)
    add_test(
        NAME import_hf_decoder_layer_math_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/import_hf_decoder_layer_math_test.py)
    add_test(
        NAME qwen2_5_1_5b_single_layer_decode_reference_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_1_5b_decode_reference_test.py)
    add_test(
        NAME qwen3_0_6b_single_layer_decode_reference_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen3_0_6b_decode_reference_test.py)
    add_test(
        NAME qwen3_0_6b_importer_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen3_0_6b_importer_test.py)
    add_test(
        NAME qwen3_5_gated_deltanet_reference_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen3_5_gated_deltanet_reference_test.py)
    add_test(
        NAME qwen3_5_importer_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen3_5_importer_test.py)
    add_test(
        NAME qwen3_moe_importer_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen3_moe_importer_test.py)
    add_test(
        NAME dense_decoder_family_importer_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/dense_decoder_family_importer_test.py)
    add_test(
        NAME dense_attention_ping_pong_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/dense_attention_ping_pong_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --target-config ${FTLPU_HARDWARE_CONFIG})
    add_test(
        NAME official_dense_reference_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/official_dense_reference_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --target-config ${FTLPU_HARDWARE_CONFIG})
    add_test(
        NAME build_hf_decoder_stack_paging_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/build_hf_decoder_stack_paging_test.py)
    add_test(
        NAME qwen2_5_1_5b_decoder_layer_seq32_vector_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_1_5b_decoder_layer_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --weight-bank 0
            --mxm-execution vector
            --seq-len 32
            --kv-cache-capacity 256
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen2_5_1_5b_decoder_layer)
    add_test(
        NAME qwen2_5_1_5b_decoder_layer_seq32_vector_weight_bank1_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_1_5b_decoder_layer_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --weight-bank 1
            --mxm-execution vector
            --seq-len 32
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen2_5_1_5b_paged)
    add_test(
        NAME qwen2_5_projection_rope_overlap_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_projection_rope_overlap_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen2_5_projection_rope_overlap)
    add_test(
        NAME qwen2_5_o_projection_direct_3d_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_o_projection_direct_3d_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen2_5_o_projection_direct_3d)
    add_custom_target(
        qwen2_5_1_5b_decoder_layer_pipeline
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_1_5b_decoder_layer_pipeline_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --weight-bank 0
            --kv-cache-capacity 256
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen2_5_1_5b_decoder_layer
        DEPENDS ftlpu_opt ftlpu_compile
        USES_TERMINAL)
    add_custom_target(
        qwen2_5_1_5b_decoder_layer_decode_cmodel_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_5_1_5b_decode_cmodel_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --runtime-test $<TARGET_FILE:compiled_qwen_real_decoder_layer_runtime_test>
            --icu-export $<TARGET_FILE:ftlpu_icu_program_export>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --weight-bank 1
            --past-len 32
            --current-len 1
            --kv-cache-capacity 256
            --output-dir ${CMAKE_SOURCE_DIR}/results/qwen2_5_decoder_layer_decode
        DEPENDS
            ftlpu_opt
            ftlpu_compile
            compiled_qwen_real_decoder_layer_runtime_test
            ftlpu_icu_program_export
        USES_TERMINAL)
    add_custom_target(
        qwen3_0_6b_decoder_layer_prefill_pipeline
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen3_0_6b_decoder_layer_pipeline_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen3_0_6b_decoder_layer/decoder_layer_seq32.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --weight-bank 1
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen3_0_6b_decoder_layer_prefill
        DEPENDS ftlpu_opt ftlpu_compile
        USES_TERMINAL)
    add_test(
        NAME lm_head_target_aware_tensor_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/lm_head_tensor_ir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/lm_head/lm_head_128x576x4096.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
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
                --target-config ${FTLPU_HARDWARE_CONFIG}
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
                --target-config ${FTLPU_HARDWARE_CONFIG}
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
            --hardware-config ${FTLPU_HARDWARE_CONFIG}
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/examples/targets/exploration_40_streams.json
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/lpu_target_configuration)
    add_test(
        NAME mem_group_configuration_lowering_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/mem_group_configuration_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/w8a16_ffn/ffn_32x64x128x64_bf16.stablehlo.mlir
            --hardware-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/mem_group_configuration)
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
        NAME vxm_sr_ffn_binary_runtime_test
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/vxm_sr_ffn_binary_runtime_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --runtime-test $<TARGET_FILE:compiled_vxm_sr_ffn_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/w8a16_ffn/ffn_32x64x128x64_bf16.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/vxm_sr_ffn
            --target-config ${FTLPU_HARDWARE_CONFIG})
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
        NAME qwen3_moe_expert_ffn_seq32_cmodel_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_ffn_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_qwen3_moe_expert_ffn_seq32_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen3_moe_30b_a3b/expert_ffn_seq32.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen3_moe_expert_ffn_seq32
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --mxm-execution vector)
    set_tests_properties(qwen3_moe_expert_ffn_seq32_cmodel_test
        PROPERTIES TIMEOUT 900)
    add_test(
        NAME smollm2_135m_ffn_vector_macro_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_135m_ffn_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_135m_ffn_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_ffn/ffn.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_135m_ffn_vector_macro_binary_runtime
            --target-config ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/cmodel_legacy_mxm.json
            --icu-macro-schedule)
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
        NAME smollm2_attention_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_attention_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_attention_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_attention/attention_seq128.stablehlo.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_attention_binary_runtime
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --attention-schedule fused)
    add_test(
        NAME smollm2_decoder_layer_binary_cmodel_runtime_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_decoder_layer_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_decoder_layer_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_decoder_layer/decoder_layer_seq128.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/smollm2_decoder_layer_binary_runtime)
    add_custom_target(
        smollm2_decoder_layer_pipeline
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/smollm2_decoder_layer_binary_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --translate $<TARGET_FILE:ftlpu_translate>
            --runtime-test $<TARGET_FILE:compiled_smollm2_decoder_layer_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/smollm2_135m_decoder_layer/decoder_layer_seq128.stablehlo.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
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
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/schedule_wave_compression/command.mlir)
    add_test(
        NAME mem_3d_blocked_outer_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/mem_3d_blocked_outer_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/mem_3d_blocked_outer.schedule.mlir
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/mem_3d_blocked_outer)
    add_test(
        NAME mem_slice_program_ab_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/mem_slice_program_ab_test.py
            --translate $<TARGET_FILE:ftlpu_translate>
            --inspect $<TARGET_FILE:ftlpu_binary_inspect>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/command_mem_slice_ab.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/mem_slice_program_ab)
    add_test(
        NAME stablehlo_matmul_320_to_ftlpu_kernel_mlir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/stablehlo_matmul_mlir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/simple_stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/stablehlo_matmul_320_to_ftlpu_kernel/matmul_320.kernel.mlir)
      add_test(
          NAME depthwise_conv1d_direct_lowering_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/depthwise_conv1d_direct_lowering_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/depthwise_conv1d_kernel4.kernel.mlir
              --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/depthwise_conv1d_direct)
    add_test(
        NAME depthwise_conv1d_binary_runtime_test
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/depthwise_conv1d_binary_runtime_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --runtime-test $<TARGET_FILE:compiled_depthwise_conv1d_runtime_test>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/depthwise_conv1d_kernel4.kernel.mlir
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/depthwise_conv1d_runtime)
      add_test(
          NAME gated_delta_rule_ir_lowering_test
          COMMAND
              ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/gated_delta_rule_ir_lowering_test.py
              --tool $<TARGET_FILE:ftlpu_opt>
              --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/Inputs/gated_delta_rule.kernel.mlir
              --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/gated_delta_rule)
        add_test(NAME qwen3_5_compiled_decoder_layer_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen3_5_compiled_decoder_layer_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen3_5_compiled_decoder_layer)
        add_test(NAME resident_layout_bridge_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/resident_layout_bridge_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/resident_layout_bridge)
        add_test(NAME resident_sr_copy_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/resident_sr_copy_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/resident_sr_copy)
        add_test(NAME depthwise_conv_mirrored_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/depthwise_conv_mirrored_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/depthwise_conv_mirrored)
        add_test(NAME l2norm_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/l2norm_runtime_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/l2norm)
        add_test(NAME vector_nonlinear_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/vector_nonlinear_runtime_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/vector_nonlinear)
        add_test(NAME vector_token_parallel_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/vector_token_parallel_runtime_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --graph-compile $<TARGET_FILE:ftlpu_graph_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/vector_token_parallel)
        add_test(NAME shared_rmsnorm_resident_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/shared_rmsnorm_resident_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/shared_rmsnorm_resident)
        add_test(NAME shared_rmsnorm_partitioned_resident_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/shared_rmsnorm_resident_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --target-config ${CMAKE_SOURCE_DIR}/../FTLPU-CMODEL/config/ftlpu-lpu32.json
            --direct-initial-input
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/shared_rmsnorm_partitioned_resident)
        add_test(NAME qwen3_5_resident_decoder_layer_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen3_5_resident_decoder_layer_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/qwen3_5_resident_decoder_layer)
        add_test(NAME decoder_graph_patterns_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/decoder_graph_patterns_test.py
            --graph-compile $<TARGET_FILE:ftlpu_graph_compile>
            --opt $<TARGET_FILE:ftlpu_opt>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/decoder_graph_patterns)
        add_test(NAME connected_ffn_memory_lifetime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/connected_ffn_memory_lifetime_test.py
            --graph-compile $<TARGET_FILE:ftlpu_graph_compile>
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/connected_ffn_memory_lifetime)
        add_test(NAME gated_attention_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/gated_attention_runtime_test.py
            --hidden-size 256
            --compile $<TARGET_FILE:ftlpu_compile>
            --graph-compile $<TARGET_FILE:ftlpu_graph_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/gated_attention_runtime)
        add_test(NAME native_dense_gated_decoder_layer_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/native_dense_gated_decoder_layer_runtime_test.py
            --opt $<TARGET_FILE:ftlpu_opt>
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/native_dense_gated_decoder_layer)
        add_test(NAME gated_attention_decoder_layer_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/gated_attention_runtime_test.py
            --hidden-size 256 --full-layer
            --compile $<TARGET_FILE:ftlpu_compile>
            --graph-compile $<TARGET_FILE:ftlpu_graph_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --target-config ${FTLPU_HARDWARE_CONFIG}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/gated_attention_decoder_layer_runtime)
        foreach(native_vector_test packed_vector_graph_runtime_test vxm_queue_group_runtime_test vector_memory_plan_runtime_test)
          add_test(NAME ${native_vector_test}
            COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/${native_vector_test}.py
              --compile $<TARGET_FILE:ftlpu_compile>
              --runner $<TARGET_FILE:compiled_stage_file_runner>
              --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/${native_vector_test})
        endforeach()
        add_test(NAME vector_alu_binary_runtime_test
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/vector_alu_binary_runtime_test.py
            --compile $<TARGET_FILE:ftlpu_compile>
            --runner $<TARGET_FILE:compiled_stage_file_runner>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/vector_alu_binary_runtime)
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
        NAME stablehlo_matmul_320_to_ftlpu_command_mlir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/stablehlo_matmul_command_mlir_test.py
            --tool $<TARGET_FILE:ftlpu_opt>
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/simple_stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/stablehlo_matmul_320_to_ftlpu_command/matmul_320.command.mlir)
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
endif()
