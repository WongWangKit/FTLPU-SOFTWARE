find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    add_test(
        NAME iree_frontend_stage_common_ir_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ftlpu-iree-import.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/simple_stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/iree_frontend/simple_stablehlo.common.mlir
            --input-format stablehlo
            --mode stage-common-ir)

    add_test(
        NAME iree_frontend_onnx_import_dry_run_test
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ftlpu-iree-import.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/model.onnx
            --output ${CMAKE_CURRENT_BINARY_DIR}/iree_frontend/model.mlir
            --input-format onnx
            --mode import-onnx
            --onnx-opset-version 17
            --dry-run)
endif()
