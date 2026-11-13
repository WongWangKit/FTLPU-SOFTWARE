#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_up_3d_lowering.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/affine_domains.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

#include <string>
#include "ftlpu/software/runtime/icu_program.hpp"

namespace ftlpu::compiler::command {

software::runtime::QueueCommand readMacroCommand(mlir::Operation* op);
std::vector<software::runtime::QueueCommand> lowerMacroDomain(
    std::size_t start, software::runtime::QueueCommand native, IcuInductionTarget target,
    std::array<std::size_t,3> counts, std::array<std::size_t,3> intervals,
    std::array<std::int64_t,3> strides);


mlir::LogicalResult materializeMemMacroCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MemAddressDomain& instruction,
    int64_t addressBinding = -1,
    llvm::StringRef addressBindingAccess = {},
    std::string* error = nullptr);

mlir::LogicalResult materializeMemTransferPair(
    mlir::OpBuilder& builder, mlir::Location location,
    std::size_t cycle, std::size_t queue,
    const MemTransferPairDomain& instruction,
    std::string* error = nullptr);

mlir::LogicalResult materializeMxmLoadMacroCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MxmLoadDomain& instruction,
    std::string* error = nullptr);

mlir::LogicalResult materializeMxmDequantMacroCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MxmDequantDomain& instruction,
    int64_t scaleBinding = -1,
    std::string* error = nullptr);

mlir::LogicalResult materializeMxmComputeMacroCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MxmComputeDomain& instruction,
    std::string* error = nullptr);

// Materializes already-lowered hardware instructions as raw Command IR
// packets. This boundary accepts no Schedule operations or fine issue list.
mlir::LogicalResult materializeFfnUpMacroCommands(mlir::OpBuilder& builder,
    mlir::Location location,
    const schedule::FfnUp3DLoweringResult& lowering,
    std::string* error = nullptr);


mlir::LogicalResult materializeSxmMacroCommand(mlir::OpBuilder& builder,
    mlir::Location location, bool transpose, std::size_t queue,
    const SxmLaunchDomain& instruction,
    std::string* error = nullptr);

} // namespace ftlpu::compiler::command
