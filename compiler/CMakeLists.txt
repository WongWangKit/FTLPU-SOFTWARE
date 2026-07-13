find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    set(FTLPU_COMPILER_PYTHON "${Python3_EXECUTABLE}")
    if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
        set(FTLPU_COMPILER_PYTHON "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
    endif()

    find_program(FTLPU_IREE_IMPORT_ONNX
        NAMES iree-import-onnx iree-import-onnx.exe
        HINTS "${CMAKE_SOURCE_DIR}/.venv/Scripts")

    find_program(FTLPU_IREE_COMPILE
        NAMES iree-compile iree-compile.exe
        HINTS "${CMAKE_SOURCE_DIR}/.venv/Scripts")

    add_test(
        NAME iree_frontend_stage_common_ir_test
        COMMAND
            ${FTLPU_COMPILER_PYTHON}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ftlpu-iree-import.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/simple_stablehlo.mlir
            --output ${CMAKE_CURRENT_BINARY_DIR}/iree_frontend/simple_stablehlo.common.mlir
            --input-format stablehlo
            --mode stage-common-ir)

    add_test(
        NAME iree_frontend_onnx_import_dry_run_test
        COMMAND
            ${FTLPU_COMPILER_PYTHON}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ftlpu-iree-import.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/model.onnx
            --output ${CMAKE_CURRENT_BINARY_DIR}/iree_frontend/model.mlir
            --input-format onnx
            --mode import-onnx
            --onnx-opset-version 17
            --dry-run)

    add_test(
        NAME stablehlo_matmul_320_to_ftlpu_ir_layers_test
        COMMAND
            ${FTLPU_COMPILER_PYTHON}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/stablehlo_matmul_lowering_test.py
            --tool ${CMAKE_CURRENT_SOURCE_DIR}/tools/ftlpu-lower.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/iree_frontend/simple_stablehlo.mlir
            --work-dir ${CMAKE_CURRENT_BINARY_DIR}/ftlpu_lower/matmul_320)

    if(FTLPU_IREE_IMPORT_ONNX AND FTLPU_IREE_COMPILE)
        add_test(
            NAME iree_frontend_onnx_matmul_320_to_flow_ir_test
            COMMAND
                ${FTLPU_COMPILER_PYTHON}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/onnx_matmul_320_to_iree_ir_test.py
                --importer ${CMAKE_CURRENT_SOURCE_DIR}/tools/ftlpu-iree-import.py
                --work-dir ${CMAKE_CURRENT_BINARY_DIR}/iree_frontend/matmul_320
                --iree-import-onnx ${FTLPU_IREE_IMPORT_ONNX}
                --iree-compile ${FTLPU_IREE_COMPILE})
    endif()
endif()
