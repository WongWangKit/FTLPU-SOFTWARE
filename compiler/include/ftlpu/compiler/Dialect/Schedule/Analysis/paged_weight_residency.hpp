#pragma once

#include "mlir/IR/BuiltinAttributes.h"

namespace ftlpu::compiler::schedule {

// Matches the runtime weight-prefetch planner's physical residency test.
// Logical bindings that occupy disjoint bank/hemisphere/slice/row regions can
// be loaded before execution and do not require an in-program refill gap.
bool pagedWeightResidencyOverlaps(
    mlir::DictionaryAttr lhs, mlir::DictionaryAttr rhs);
bool pagedWeightResidencyOverlaps(mlir::DictionaryAttr lhs, int64_t lhsBank,
    mlir::DictionaryAttr rhs, int64_t rhsBank);

} // namespace ftlpu::compiler::schedule
