@echo off
call "D:\Visual Studio\Visual Studio 2022\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "CMAKE=D:\Visual Studio\Visual Studio 2022\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "ROOT=E:/Fudan_University/internship/timesintelli/LPU/work/FTLPU-SOFTWARE"
"%CMAKE%" -S "%ROOT%" -B "%ROOT%/out/main-qctl-integration/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DFTLPU_CMODEL_DIR="%ROOT%/../FTLPU-CMODEL" -DFTLPU_MLIR_DIR="%ROOT%/out/main-common-barrier-toolchain/lib/cmake/mlir" -DMLIR_DIR="%ROOT%/out/main-common-barrier-toolchain/lib/cmake/mlir" -DLLVM_DIR="%ROOT%/out/main-common-barrier-toolchain/lib/cmake/llvm"
if errorlevel 1 exit /b 1
"%CMAKE%" --build "%ROOT%/out/main-qctl-integration/build" --target ftlpu_compile ftlpu_binary_inspect binary_queue_control_test binary_barrier_test macro_bitstream_test fu_3d_macro_lowering_test runtime_common_barrier_test issue_inspector_test model_session_executable_c2c_weight_test c2c_weight_ping_pong_test program_load_runtime_test compiled_smollm2_common_barrier_decoder_test compiled_smollm2_common_barrier_seq128_decoder_test --parallel 3
exit /b %errorlevel%
