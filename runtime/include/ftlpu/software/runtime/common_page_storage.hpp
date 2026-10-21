#pragma once

#include "ftlpu/software/runtime/common_page_range.hpp"
#include "ftlpu/software/runtime/common_transport_validation.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu::software::runtime {

// Build only after all runtime page uses have final barrier/event links. The
// page builder supplies the actual populated segments; a placement's row
// envelope is insufficient because unpopulated rows do not become visible.
inline std::vector<CommonPageLoadRange>
build_common_page_load_ranges(const BinaryProgram& program)
{
    validate_binary_barriers(program);
    if (!program.common_barrier_protocol) return {};
    const auto fail = [](const char* reason) { throw std::invalid_argument(reason); };
    std::map<std::uint32_t, const BinaryBinding*> inputs;
    for (const auto& binding : program.bindings) {
        if (binding.access != BindingAccess::Input) continue;
        if (!inputs.emplace(binding.index, &binding).second)
            fail("common storage proof has duplicate Input binding indices");
    }
    std::map<std::uint32_t, const BinaryBarrierEvent*> events;
    for (const auto& event : program.barrier_events)
        events.emplace(event.event_id, &event);
    std::map<std::uint32_t, std::vector<std::uint8_t>> zeros;
    std::vector<CommonPageLoadRange> ranges;
    for (const auto& barrier : program.barriers) {
        const auto event = events.find(barrier.event_id);
        if (event == events.end()) fail("common storage proof has an unlinked event");
        for (const auto instance : event->second->page_use_indices) {
            if (instance >= program.weight_page_uses.size())
                fail("common storage proof has an invalid page-use instance");
            const auto& use = program.weight_page_uses[instance];
            const auto input = inputs.find(use.binding_index);
            if (input == inputs.end() || !input->second->paged_weight
                || input->second->role != "weight" || use.transport_launch_event_id == 0
                || use.page_index >= input->second->page_count
                || use.bank >= program.hardware.banks_per_slice
                || use.bank >= hw::kMemBanksPerSlice
                || use.ready_cycle < barrier.stop_cycle
                || use.release_cycle < use.ready_cycle
                || use.ready_cycle > std::numeric_limits<std::uint32_t>::max()
                || use.release_cycle > std::numeric_limits<std::uint32_t>::max())
                fail("common storage proof has an invalid final page-use domain");
            const auto& binding = *input->second;
            if (resolve_weight_page_placement(binding, use.page_index).bank != use.bank)
                fail("common storage proof page bank differs from final residency");
            auto [zero, inserted] = zeros.try_emplace(use.binding_index);
            if (inserted) {
                if (binding.byte_size > std::numeric_limits<std::size_t>::max()
                    || binding.byte_size > zero->second.max_size())
                    fail("common storage proof logical weight size is not representable");
                zero->second.resize(static_cast<std::size_t>(binding.byte_size), 0);
            }
            const auto image = pack_weight_binding_page(binding, use.page_index,
                zero->second, program.hardware);
            if (image.segments.empty())
                fail("common storage proof page has no actual MEM segments");
            if (image.segments.size() > kBinaryCommonBarrierMaxCompletionInstances - ranges.size())
                fail("common storage proof exceeds 65535 actual MEM segments");
            std::map<std::size_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> spans;
            for (const auto& segment : image.segments) {
                const auto rowEnd = static_cast<std::uint64_t>(segment.base_row) + segment.vector_count;
                const auto bytes = static_cast<std::uint64_t>(segment.vector_count) * hw::kPhysicalVectorBytes;
                if (segment.hemisphere >= program.hardware.hemispheres
                    || segment.hemisphere >= hw::kHemispheres
                    || segment.slice >= program.hardware.slices_per_hemisphere
                    || segment.slice >= hw::kMemSliceColumns
                    || segment.vector_count == 0 || rowEnd > program.hardware.sram_depth_rows
                    || segment.byte_offset > image.data.size()
                    || bytes > image.data.size() - segment.byte_offset)
                    fail("common storage proof actual MEM segment exceeds its SRAM or packed image");
                const auto queue = InstructionControlUnit::mem_queue(
                    static_cast<Hemisphere>(segment.hemisphere), segment.slice, use.bank);
                if (queue >= InstructionControlUnit::kMemQueues)
                    fail("common storage proof has an invalid physical MEM queue");
                for (const auto& [begin, end] : spans[queue])
                    if (segment.base_row < end && rowEnd > begin)
                        fail("common storage proof page contains overlapping actual MEM segments");
                spans[queue].emplace_back(segment.base_row, rowEnd);
                ranges.push_back({queue, segment.base_row, segment.vector_count,
                    use.binding_index, instance, barrier.stop_cycle,
                    use.ready_cycle, use.release_cycle});
            }
        }
    }
    return ranges;
}

} // namespace ftlpu::software::runtime
