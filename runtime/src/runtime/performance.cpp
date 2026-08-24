#include "ftlpu/software/runtime/performance.hpp"

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <stdexcept>

namespace ftlpu::software::runtime {
namespace {

struct QueueGroupStats {
    const char* name;
    std::size_t capacity;
    std::size_t active_queues{0};
    std::size_t issued{0};
};

std::size_t group_index(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem: return 0;
    case QueueKind::MxmLoad: return 1;
    case QueueKind::MxmCompute: return 2;
    case QueueKind::MxmDequant: return 3;
    case QueueKind::Vxm: return 4;
    case QueueKind::SxmTranspose: return 5;
    case QueueKind::SxmPermute: return 6;
    }
    throw std::logic_error("unknown ICU queue kind in runtime performance report");
}

std::size_t issued_commands(const QueueProgram& queue)
{
    std::size_t issued = 0;
    std::size_t prior_instructions = 0;
    for (const auto& command : queue.commands) {
        if (is_macro_schedule_command(command)) {
            const auto macro = decode_macro_schedule_command(command);
            issued += macro.inner_count * macro.outer_count;
            continue;
        }
        if (is_repeat_2d_command(command)) {
            if (prior_instructions == 0)
                throw std::logic_error(
                    "runtime performance found Repeat2D without instruction");
            const auto repeat = decode_repeat_2d_command(command);
            issued += repeat.inner_count * repeat.outer_count - 1;
            continue;
        }
        const auto opcode = isa::decode_icu_command_opcode(command.command);
        if (opcode == isa::IcuCommandOpcode::Instruction) {
            ++issued;
            ++prior_instructions;
            continue;
        }
        if (opcode == isa::IcuCommandOpcode::Nop) continue;
        if (opcode == isa::IcuCommandOpcode::Loop) {
            const auto loop = isa::decode_icu_loop(command.command);
            if (loop.window_size > prior_instructions)
                throw std::logic_error(
                    "runtime performance found Loop with an invalid window");
            issued += loop.window_size * loop.count;
            continue;
        }
        if (prior_instructions == 0)
            throw std::logic_error("runtime performance found Repeat without instruction");
        issued += isa::decode_icu_repeat(command.command).count;
    }
    return issued;
}

} // namespace

void print_runtime_performance(
    const BinaryProgram& program, std::size_t executed_cycles, std::ostream& os)
{
    const std::size_t cycles = executed_cycles == 0
        ? program.max_cycle + 1 : executed_cycles;
    const std::size_t logical_mxm_queues =
        program.hardware.hemispheres
        * program.hardware.mxms_per_hemisphere;
    std::array<QueueGroupStats, 7> groups {{
        {"MEM", program.hardware.hemispheres
                * program.hardware.slices_per_hemisphere},
        {"MXM.load", logical_mxm_queues},
        {"MXM.compute", logical_mxm_queues},
        {"MXM.dequant", logical_mxm_queues},
        {"VXM", program.hardware.vxm_alus},
        {"SXM.transpose", program.hardware.hemispheres},
        {"SXM.permute", program.hardware.hemispheres},
    }};

    for (const auto& queue : program.queues) {
        auto& group = groups[group_index(queue.kind)];
        const auto issued = issued_commands(queue);
        group.issued += issued;
        if (issued != 0) ++group.active_queues;
    }

    const auto old_flags = os.flags();
    const auto old_precision = os.precision();
    os << std::fixed << std::setprecision(2);
    for (const auto& group : groups) {
        const auto slots = cycles * group.capacity;
        const double utilization = slots == 0 ? 0.0
            : static_cast<double>(group.issued) / static_cast<double>(slots);
        os << "runtime perf resource=" << group.name
           << " cycles=" << cycles
           << " active_queues=" << group.active_queues << "/" << group.capacity
           << " issued=" << group.issued
           << " issue_util=" << utilization * 100.0 << "%\n";
    }
    os.flags(old_flags);
    os.precision(old_precision);
}

void DatapathPerformanceMonitor::reset()
{
    sampled_cycles_ = 0;
    mxm_ = {};
    vxm_ = {};
    vxm_useful_slots_ = 0;
    sr_ = {};
    sr_east_ = {};
    sr_west_ = {};
}

void DatapathPerformanceMonitor::sample(const TspSliceSystem& system,
    std::size_t mxms_per_hemisphere, std::size_t vxm_alus)
{
    ++sampled_cycles_;
    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres;
         ++hemisphere) {
        for (std::size_t local = 0; local < mxms_per_hemisphere; ++local) {
            const std::size_t physical =
                hemisphere * hw::kMxmsPerHemisphere + local;
            std::size_t active = 0;
            for (std::size_t tile = 0;
                 tile < hw::kMxmSupercellsPerPlane; ++tile)
                for (std::size_t column = 0;
                     column < hw::kMxmSupercellsPerPlane; ++column)
                    active += system.mxm_unit(physical).computing_cell(
                        tile, column) ? 1 : 0;
            auto& stats = mxm_[physical];
            stats.active_slots += active;
            stats.non_idle_cycles += active != 0 ? 1 : 0;
            stats.peak_active_slots = std::max<std::uint64_t>(
                stats.peak_active_slots, active);
        }

        const auto& fabric = system.stream_fabric(
            static_cast<Hemisphere>(hemisphere));
        const auto& activity = fabric.last_cycle_activity();
        auto& stats = sr_[hemisphere];
        stats.active_slots += activity.staged_writes;
        stats.non_idle_cycles += activity.staged_writes != 0 ? 1 : 0;
        stats.peak_active_slots = std::max<std::uint64_t>(
            stats.peak_active_slots, activity.staged_writes);
        sr_east_[hemisphere] += activity.east_staged_writes;
        sr_west_[hemisphere] += activity.west_staged_writes;
    }

    const auto& vxm = system.vxm_unit();
    std::size_t active_rows = 0;
    std::size_t useful_rows = 0;
    for (std::size_t tile = 0; tile < VxmSlice::kTileCount; ++tile) {
        const auto& timeline =
            vxm.superlane(tile).lane(0).statistics().timeline;
        if (timeline.empty()) continue;
        active_rows += timeline.back().active_slots();
        useful_rows += timeline.back().useful_slots();
    }
    const std::size_t vxm_active = active_rows * hw::kLanesPerTile;
    const std::size_t vxm_useful = useful_rows * hw::kLanesPerTile;
    const auto vxm_capacity =
        hw::kTileRows * hw::kLanesPerTile * VxmLane::kAluCount;
    if (vxm_active > vxm_capacity)
        throw std::logic_error(
            "VXM activity exceeds the executable hardware capacity");
    vxm_.active_slots += vxm_active;
    vxm_.non_idle_cycles += vxm_active != 0 ? 1 : 0;
    vxm_.peak_active_slots = std::max<std::uint64_t>(
        vxm_.peak_active_slots, vxm_active);
    vxm_useful_slots_ += vxm_useful;
}

void DatapathPerformanceMonitor::print(const TspSliceSystem& system,
    std::size_t mxms_per_hemisphere, std::size_t vxm_alus,
    std::ostream& os) const
{
    const auto percent = [](std::uint64_t value, std::uint64_t capacity) {
        return capacity == 0 ? 0.0
            : 100.0 * static_cast<double>(value)
                / static_cast<double>(capacity);
    };
    constexpr std::uint64_t mxm_cells =
        hw::kMxmSupercellsPerPlane * hw::kMxmSupercellsPerPlane;
    const auto old_flags = os.flags();
    const auto old_precision = os.precision();
    os << std::fixed << std::setprecision(2);
    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres;
         ++hemisphere) {
        for (std::size_t local = 0; local < mxms_per_hemisphere; ++local) {
            const std::size_t physical =
                hemisphere * hw::kMxmsPerHemisphere + local;
            const auto& stats = mxm_[physical];
            os << "runtime datapath resource=MXM."
               << (hemisphere == 0 ? "E" : "W") << local
               << " cycles=" << sampled_cycles_
               << " active_cycles=" << stats.non_idle_cycles
               << " active_cell_cycles=" << stats.active_slots
               << " array_util="
               << percent(stats.active_slots,
                      sampled_cycles_ * mxm_cells)
               << "% active_density="
               << percent(stats.active_slots,
                      stats.non_idle_cycles * mxm_cells)
               << "% peak=" << stats.peak_active_slots << "/"
               << mxm_cells << '\n';
        }
    }

    const std::uint64_t vxm_capacity =
        hw::kTileRows * hw::kLanesPerTile * vxm_alus;
    os << "runtime datapath resource=VXM cycles=" << sampled_cycles_
       << " active_cycles=" << vxm_.non_idle_cycles
       << " executed_slots=" << vxm_.active_slots
       << " useful_slots=" << vxm_useful_slots_
       << " full_util="
       << percent(vxm_.active_slots, sampled_cycles_ * vxm_capacity)
       << "% active_density="
       << percent(vxm_.active_slots,
              vxm_.non_idle_cycles * vxm_capacity)
       << "% useful_util="
       << percent(vxm_useful_slots_, sampled_cycles_ * vxm_capacity)
       << "% peak=" << vxm_.peak_active_slots << "/"
       << vxm_capacity << '\n';

    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres;
         ++hemisphere) {
        const auto columns = system.stream_fabric(
            static_cast<Hemisphere>(hemisphere)).column_count();
        const std::uint64_t capacity = (columns - 1) * 2
            * hw::kTileRows * hw::kLanesPerTile
            * hw::kStreamsPerDirection;
        const auto& stats = sr_[hemisphere];
        os << "runtime datapath resource=SR."
           << (hemisphere == 0 ? "E" : "W")
           << " cycles=" << sampled_cycles_
           << " active_cycles=" << stats.non_idle_cycles
           << " staged_writes=" << stats.active_slots
           << " east_writes=" << sr_east_[hemisphere]
           << " west_writes=" << sr_west_[hemisphere]
           << " full_util="
           << percent(stats.active_slots, sampled_cycles_ * capacity)
           << "% active_density="
           << percent(stats.active_slots,
                  stats.non_idle_cycles * capacity)
           << "% peak=" << stats.peak_active_slots << "/"
           << capacity << '\n';
    }
    os.flags(old_flags);
    os.precision(old_precision);
}

} // namespace ftlpu::software::runtime
