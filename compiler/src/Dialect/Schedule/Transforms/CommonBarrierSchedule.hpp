#pragma once

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallSet.h"

#include <algorithm>
#include <optional>

namespace ftlpu::compiler::schedule {

inline bool requestsRawMacroBarriers(mlir::Operation* operation, const target::LPUTargetModel& target)
{
    const auto module = operation->getParentOfType<mlir::ModuleOp>();
    // Keep the transport protocol independent of computation encoding so
    // Native and Raw Macro can be evaluated with identical stop boundaries.
    const auto common = module ? module->getAttrOfType<mlir::BoolAttr>("ftlpu.common_barrier_protocol") : mlir::BoolAttr{};
    if (common && common.getValue()) return true;
    const auto mode = module ? module->getAttrOfType<mlir::StringAttr>("ftlpu.icu_compression") : mlir::StringAttr{};
    return mode && mode.getValue() == "raw-macro"
        && target.icu_memory().queue_control_version == 0;
}

inline int64_t commonBarrierDrainCycles(const target::LPUTargetModel& target)
{
    const auto& t = target.throughput();
    return target.streams().system_register_columns
        + target.streams().mem_boundary_register_columns + t.mxm_pipeline_rows + t.tile_rows
        + std::max({t.mem_to_sxm_latency, t.mem_to_mxm_latency,
            t.mxm0_accumulator_latency, t.mxm1_accumulator_latency,
            t.accumulator_to_vxm_latency, t.accumulator_read_to_vxm_latency,
            t.swiglu_write_latency, t.qk_iw_to_compute_latency, t.vxm_weight_to_iw_latency});
}

inline std::optional<int64_t> commonBarrierInputBinding(mlir::Value value)
{
    llvm::SmallVector<mlir::Value> pending {value};
    llvm::SmallDenseSet<mlir::Value, 16> visited;
    std::optional<int64_t> result;
    while (!pending.empty()) {
        const auto current = pending.pop_back_val();
        if (!visited.insert(current).second) continue;
        if (const auto argument = llvm::dyn_cast<mlir::BlockArgument>(current)) {
            if (result && *result != argument.getArgNumber()) return std::nullopt;
            result = argument.getArgNumber();
            continue;
        }
        const auto defining = current.getDefiningOp();
        if (!defining) continue;
        if (auto binding = llvm::dyn_cast<BindingOp>(defining)) {
            if (binding.getAccess() != "input") continue;
            if (result && *result != binding.getIndex()) return std::nullopt;
            result = binding.getIndex();
            continue;
        }
        for (const auto operand : defining->getOperands()) pending.push_back(operand);
    }
    return result;
}

inline void emitCommonBarrierCandidate(mlir::OpBuilder& builder, mlir::Location location,
    int64_t binding, int64_t page, int64_t stableBegin, int64_t stop)
{
    mlir::OperationState state(location, TimelineOp::getOperationName());
    state.addAttributes({builder.getNamedAttr("name", builder.getStringAttr(
        "c2c.window.binding:" + std::to_string(binding) + ".page:" + std::to_string(page))),
        builder.getNamedAttr("start", builder.getI64IntegerAttr(stableBegin)),
        builder.getNamedAttr("end", builder.getI64IntegerAttr(stop))});
    builder.create(state);
}

} // namespace ftlpu::compiler::schedule
