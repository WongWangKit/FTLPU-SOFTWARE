#pragma once

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Types.h"

namespace ftlpu::compiler {

inline bool is_lpu_16bit_float(mlir::Type type)
{
    return type.isF16() || type.isBF16();
}

inline llvm::StringRef lpu_16bit_data_format(mlir::Type type)
{
    return type.isBF16() ? "bf16" : "fp16";
}

inline llvm::StringRef lpu_16bit_stream_kind(mlir::Type type)
{
    return type.isBF16() ? "stream_bf16" : "stream_f16";
}

} // namespace ftlpu::compiler
