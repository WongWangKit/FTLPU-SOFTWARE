#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include "mlir/IR/BuiltinOps.h"

namespace ftlpu::compiler::target {

software::runtime::BinaryProgram translate_command_module(mlir::ModuleOp module);

} // namespace ftlpu::compiler::target
