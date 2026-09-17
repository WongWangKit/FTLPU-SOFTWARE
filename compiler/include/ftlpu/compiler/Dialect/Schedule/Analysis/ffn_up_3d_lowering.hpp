#pragma once

#include "ftlpu/icu/fu_3d_instruction.hpp"

#include "mlir/Support/LogicalResult.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ftlpu::compiler::schedule {

// Shape consumed by standalone Up lowering.  It deliberately has no Down
// projection dimension: the direct path does not pretend to lower a complete
// FFN in order to select an Up command sequence.
struct FfnUp3DShape {
    std::int64_t m{0};
    std::int64_t k{0};
    std::int64_t n{0};
};

// These are compiler-side placement and timeline records. The lowering result
// contains only commands understood by a physical FU ICU; there is no hardware
// "projection" instruction or program object.
struct FfnUpWeightRegion3DPlacement {
    std::int64_t first_pair{0};
    std::int64_t pair_count{0};
    std::int64_t bank{0};
    std::int64_t first_slice{0};
    std::int64_t base_address{0};
};

struct FfnUpHemisphere3DPlacement {
    std::vector<FfnUpWeightRegion3DPlacement> weight_regions;
    std::array<std::int64_t,
        2 * hw::kLanesPerTile> activation_slices{};
    std::array<std::int64_t, sizeof(std::uint16_t)> result_slices{};
    // Dense executable-local queue id. Runtime maps this logical topology
    // onto the physical CModel MXM stride when fewer MXMs are enabled.
    std::int64_t mxm_queue{-1};
    std::int64_t result_stream_base{-1};
};

struct FfnUp3DPlacement {
    std::int64_t mem_slices_per_hemisphere{hw::kMemSliceColumns};
    std::int64_t mem_banks_per_slice{hw::kMemBanksPerSlice};
    std::int64_t mxms_per_hemisphere{hw::kMxmsPerHemisphere};
    std::int64_t weight_stream_base{8};
    std::int64_t activation_stream_base{16};
    std::int64_t activation_bank{1};
    std::int64_t activation_base_address{0};
    std::int64_t result_bank{0};
    std::int64_t result_base_address{0};
    std::int64_t accumulator_address_base{32};
    // Consecutive output pairs share one address group. The current W8A16
    // layout stores two pairs as adjacent four-row regions.
    std::int64_t weight_outer_group_size{2};
    std::vector<FfnUpHemisphere3DPlacement> hemispheres;
};

// Closed-form consumer/producer anchors for the three loop counters. Counter 0
// is the innermost row/pulse counter, counter 1 is K reduction, and counter 2
// is the output pair. MEM-local issue cycles are derived from these anchors and
// the physical slice route latency without materializing any per-cycle reads.
struct FfnUp3DTimeline {
    std::int64_t mxm_load_start_cycle{0};
    std::int64_t mxm_dequant_start_cycle{0};
    std::int64_t mxm_compute_start_cycle{0};
    // Cycle at which row zero of the final reduction first enters the west
    // result stream, before its MXM-to-MEM transport latency.
    std::int64_t mxm_result_start_cycle{0};
    std::int64_t reduction_cycle_stride{0};
    std::int64_t pair_cycle_stride{0};
};

// All arrays are indexed by a queue-local MEM slice. They keep topology
// knowledge outside the FU command format while allowing the direct lowering
// to calculate the issue cycle of each physical MEM ICU independently.
struct FfnUp3DRouteLatencies {
    std::array<std::int64_t, hw::kMemSliceColumns>
        mem_to_mxm_weight_cycles{};
    std::array<std::int64_t, hw::kMemSliceColumns>
        mem_to_mxm_activation_cycles{};
    std::array<std::int64_t, hw::kMemSliceColumns>
        mxm_result_to_mem_cycles{};
};

template <typename Instruction>
struct PhysicalIcu3DCommand {
    // Queue ids use the executable's logical topology. Cycle is compiler-only
    // absolute schedule metadata; the hardware instruction itself always uses
    // a zero origin and is never expanded by the compiler.
    std::size_t queue{0};
    std::size_t cycle{0};
    Instruction instruction{};
};

using PhysicalMemIcu3DCommand = PhysicalIcu3DCommand<MemIcuInstruction>;
using PhysicalMxmLoadIcu3DCommand =
    PhysicalIcu3DCommand<MxmLoadIcuInstruction>;
using PhysicalMxmDequantIcu3DCommand =
    PhysicalIcu3DCommand<MxmDequantIcuInstruction>;
using PhysicalMxmComputeIcu3DCommand =
    PhysicalIcu3DCommand<MxmComputeIcuInstruction>;

// A compiler lowering result grouped by the independently addressed physical
// ICU families. Every element is already a hardware FU-specific 3-D command.
struct FfnUp3DLoweringResult {
    std::vector<PhysicalMemIcu3DCommand> mem_commands;
    std::vector<PhysicalMxmLoadIcu3DCommand> mxm_load_commands;
    std::vector<PhysicalMxmDequantIcu3DCommand> mxm_dequant_commands;
    std::vector<PhysicalMxmComputeIcu3DCommand> mxm_compute_commands;

    std::size_t hardware_command_count() const noexcept
    {
        return mem_commands.size() + mxm_load_commands.size()
            + mxm_dequant_commands.size() + mxm_compute_commands.size();
    }
};

// Lowers directly from shape, physical placement, and a closed-form timeline.
// This API deliberately cannot consume Schedule dialect fine events, so it
// cannot implement the result by discovering/compressing an event pattern.
mlir::FailureOr<FfnUp3DLoweringResult> lowerFfnUpToFu3D(
    FfnUp3DShape shape, const FfnUp3DPlacement& placement,
    const FfnUp3DTimeline& timeline,
    const FfnUp3DRouteLatencies& routeLatencies,
    std::uint16_t dequantScaleBf16, std::string* error = nullptr);

} // namespace ftlpu::compiler::schedule
