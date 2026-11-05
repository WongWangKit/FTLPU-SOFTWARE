#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include <stdexcept>

namespace ftlpu::compiler::target {

// Policy for connected graphs. Native operator adapters still describe their
// physical weight ABI; they use weight_page_plan.hpp for capacity/residency.
// No model name, sequence length or test stage participates in this decision.
struct GraphWeightPaging {
    bool overlap{};
    int64_t base_row{}, rows{}, slice_base{}, slice_count{};

    static GraphWeightPaging resolve(const LPUTargetModel& target,
                                    mlir::DictionaryAttr attributes) {
        const auto& m = target.memory();
        GraphWeightPaging result{false, m.sram_depth_rows / 2,
            m.sram_depth_rows / 2, m.w8a16_weight_slice_base,
            m.w8a16_weight_slice_count};
        bool requested = true;
        if (attributes) {
            auto policy = attributes.getAs<mlir::StringAttr>("ftlpu.weight_paging_policy");
            auto legacy = attributes.getAs<mlir::BoolAttr>("ftlpu.weight_page_ping_pong");
            if (policy) {
                if (policy.getValue() != "auto" && policy.getValue() != "serialized")
                    throw std::runtime_error("weight_paging_policy must be auto or serialized");
                requested = policy.getValue() == "auto";
                if (legacy && legacy.getValue() != requested)
                    throw std::runtime_error("conflicting weight paging policies");
            } else if (legacy) requested = legacy.getValue();
            if (auto transport = attributes.getAs<mlir::BoolAttr>("ftlpu.common_transport_overlap"))
                requested &= transport.getValue();
            if (auto rows = attributes.getAs<mlir::IntegerAttr>("ftlpu.weight_page_rows")) {
                if (rows.getInt() <= 0 || rows.getInt() > m.sram_depth_rows / 2)
                    throw std::runtime_error("weight_page_rows exceeds graph weight capacity");
                result.rows = rows.getInt();
                result.base_row = m.sram_depth_rows - result.rows;
            }
        }
        // The lower half is the native operator scratch arena. Weight slots
        // are reserved before allocating graph values, not after FU lowering.
        result.overlap = requested && m.banks_per_slice >= 2
            && result.rows > 0 && result.slice_count > 0
            && result.slice_base >= 0
            && result.slice_base + result.slice_count <= m.slices_per_hemisphere
            && target.icu_memory().common_transport_overlap_version == 1;
        return result;
    }

    mlir::DictionaryAttr attributes(mlir::MLIRContext* context,
                                   mlir::DictionaryAttr original) const {
        mlir::NamedAttrList attrs(original);
        mlir::Builder b(context);
        attrs.set("ftlpu.weight_paging_policy", b.getStringAttr(overlap ? "auto" : "serialized"));
        attrs.set("ftlpu.weight_page_ping_pong", b.getBoolAttr(overlap));
        attrs.set("ftlpu.common_transport_overlap", b.getBoolAttr(overlap));
        if (overlap) attrs.set("ftlpu.common_barrier_protocol", b.getBoolAttr(true));
        attrs.set("ftlpu.weight_page_rows", b.getI64IntegerAttr(rows));
        return attrs.getDictionary(context);
    }

    // A native operator supplies its scratch width and preferred location.
    // Avoid a reserved page queue regardless of disjoint row addresses:
    // WRITE_SYNC cannot interrupt an executing READ3D on the same MEM ICU.
    int64_t scratchSlice(int64_t preferred, int64_t width,
                         int64_t limit) const {
        if (width <= 0 || width > limit) throw std::runtime_error("invalid scratch group");
        const auto available = [&](int64_t first) {
            return first >= 0 && first + width <= limit
                && (!overlap || first + width <= slice_base
                    || first >= slice_base + slice_count);
        };
        if (available(preferred)) return preferred;
        for (int64_t first = preferred; first + width <= limit; ++first)
            if (available(first)) return first;
        for (int64_t first = 0; first < preferred; ++first)
            if (available(first)) return first;
        throw std::runtime_error("no independent scratch queues outside weight slots");
    }
};

} // namespace ftlpu::compiler::target
