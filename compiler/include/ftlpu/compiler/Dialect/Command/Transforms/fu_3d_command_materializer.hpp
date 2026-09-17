#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_up_3d_lowering.hpp"
#include "ftlpu/icu/fu_3d_instruction.hpp"
#include "ftlpu/icu/vxm_run_2d.hpp"
#include "ftlpu/icu/sxm_run_2d.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace ftlpu::compiler::command {

mlir::LogicalResult materializeMem3DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MemIcuInstruction& instruction,
    int64_t addressBinding = -1,
    llvm::StringRef addressBindingAccess = {},
    std::string* error = nullptr);

mlir::LogicalResult materializeMemWriteRead2DCommand(
    mlir::OpBuilder& builder, mlir::Location location,
    std::size_t cycle, std::size_t queue,
    const MemIcuWriteRead2DInstruction& instruction,
    std::string* error = nullptr);

mlir::LogicalResult materializeMxmLoad3DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MxmLoadIcuInstruction& instruction,
    std::string* error = nullptr);

mlir::LogicalResult materializeMxmDequant3DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MxmDequantIcuInstruction& instruction,
    int64_t scaleBinding = -1,
    std::string* error = nullptr);

mlir::LogicalResult materializeMxmCompute3DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MxmComputeIcuInstruction& instruction,
    std::string* error = nullptr);

// Materializes already-lowered hardware instructions as raw Command IR
// packets. This boundary accepts no Schedule operations or fine issue list.
mlir::LogicalResult materializeFfnUp3DCommands(mlir::OpBuilder& builder,
    mlir::Location location,
    const schedule::FfnUp3DLoweringResult& lowering,
    std::string* error = nullptr);

// Emits one hardware-visible VXM RUN_2D descriptor. The compact VXM
// instruction owns the contiguous datapath run; the ICU descriptor owns the
// two outer launch counters. No per-cycle Schedule events are constructed.
mlir::LogicalResult materializeVxmRun2DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const VxmIcuRun2DInstruction& instruction,
    std::string* error = nullptr,
    int64_t scaleBinding = -1);

mlir::LogicalResult materializeSxmRun2DCommand(mlir::OpBuilder& builder,
    mlir::Location location, bool transpose, std::size_t queue,
    const SxmIcuRun2DInstruction& instruction,
    std::string* error = nullptr);

} // namespace ftlpu::compiler::command
