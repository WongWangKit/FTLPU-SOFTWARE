#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ftlpu/compiler/Dialect/Command/IR/CommandOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "ftlpu/compiler/Dialect/Command/IR/CommandOps.h.inc"
