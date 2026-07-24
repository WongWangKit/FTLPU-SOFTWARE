#include "ftlpu/software/runtime/performance.hpp"

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/icu.hpp"

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
    case QueueKind::Vxm: return 3;
    case QueueKind::SxmTranspose: return 4;
    case QueueKind::SxmPermute: return 5;
    }
    throw std::logic_error("unknown ICU queue kind in runtime performance report");
}

std::size_t issued_commands(const QueueProgram& queue)
{
    std::size_t issued = 0;
    bool has_previous_instruction = false;
    for (const auto& command : queue.commands) {
        const auto opcode = isa::decode_icu_command_opcode(command.command);
        if (opcode == isa::IcuCommandOpcode::Instruction) {
            ++issued;
            has_previous_instruction = true;
            continue;
        }
        if (opcode == isa::IcuCommandOpcode::Nop) continue;
        if (!has_previous_instruction)
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
    std::array<QueueGroupStats, 6> groups {{
        {"MEM", InstructionControlUnit::kMemQueues},
        {"MXM.load", InstructionControlUnit::kMxmQueues},
        {"MXM.compute", InstructionControlUnit::kMxmQueues},
        {"VXM", InstructionControlUnit::kVxmQueues},
        {"SXM.transpose", hw::kHemispheres},
        {"SXM.permute", hw::kHemispheres},
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

} // namespace ftlpu::software::runtime
