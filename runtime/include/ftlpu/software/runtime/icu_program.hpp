#pragma once

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/icu.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {

enum class QueueKind : std::uint16_t {
    Mem = 0,
    MxmLoad = 1,
    MxmCompute = 2,
    Vxm = 4,
    SxmTranspose = 5,
    SxmPermute = 6,
};

enum class InstructionKind : std::uint16_t {
    None = 0,
    Mem = 1,
    Mxm = 2,
    Vxm = 3,
    Sxm = 4,
};

struct QueueCommand {
    isa::EncodedIcuCommand command{0};
    InstructionKind instruction_kind{InstructionKind::None};
    std::uint16_t word_count{0};
    std::array<std::uint32_t, 4> words{};
    // MEM/MXM/VXM fit in the fixed header. SXM carries variable stream lists
    // and a 32-lane map in this trailing payload.
    std::vector<std::uint32_t> extension_words{};
};

struct QueueProgram {
    QueueKind kind{QueueKind::Mem};
    std::size_t index{0};
    std::vector<QueueCommand> commands{};
};

class IcuProgram {
public:
    void emit_mem(std::size_t cycle, std::size_t column, MemInstruction instruction);
    void emit_mxm_load(std::size_t cycle, std::size_t mxm, MxmControlInstruction instruction);
    void emit_mxm_compute(std::size_t cycle, std::size_t mxm, MxmControlInstruction instruction);
    void emit_vxm(std::size_t cycle, std::size_t alu, VxmLaneAluInstruction instruction);
    void emit_sxm_transpose(std::size_t cycle, Hemisphere hemisphere, SxmInstruction instruction);
    void emit_sxm_permute(std::size_t cycle, Hemisphere hemisphere, SxmInstruction instruction);

    std::vector<QueueProgram> encode_queues() const;
    void load_into(InstructionControlUnit& icu) const;

    std::size_t last_cycle() const;
    bool empty() const;

private:
    template <typename Instruction>
    struct ScheduledInstruction {
        std::size_t cycle{0};
        Instruction instruction{};
    };

    using MemQueue = std::vector<ScheduledInstruction<MemInstruction>>;
    using MxmQueue = std::vector<ScheduledInstruction<MxmControlInstruction>>;
    using VxmQueue = std::vector<ScheduledInstruction<VxmLaneAluInstruction>>;
    using SxmQueue = std::vector<ScheduledInstruction<SxmInstruction>>;

    void check_mem_column(std::size_t column) const;
    void check_mxm(std::size_t mxm) const;
    void check_vxm_alu(std::size_t alu) const;

    template <typename Instruction, typename EncodeFn>
    static std::vector<QueueCommand> encode_scheduled_queue(
        std::vector<ScheduledInstruction<Instruction>> events,
        const std::string& queue_name,
        EncodeFn encode);

    template <typename Instruction, typename NopFn, typename EmitFn>
    static void load_scheduled_queue(
        std::vector<ScheduledInstruction<Instruction>> events,
        const std::string& queue_name,
        NopFn nop,
        EmitFn emit);

    std::array<MemQueue, InstructionControlUnit::kMemQueues> mem_{};
    std::array<MxmQueue, InstructionControlUnit::kMxmQueues> mxm_load_{};
    std::array<MxmQueue, InstructionControlUnit::kMxmQueues> mxm_compute_{};
    std::array<VxmQueue, InstructionControlUnit::kVxmQueues> vxm_{};
    std::array<SxmQueue, hw::kHemispheres> sxm_transpose_{};
    std::array<SxmQueue, hw::kHemispheres> sxm_permute_{};
    std::size_t last_cycle_{0};
};

const char* queue_kind_name(QueueKind kind);
void load_queue_programs_into_icu(const std::vector<QueueProgram>& queues, InstructionControlUnit& icu);

} // namespace ftlpu::software::runtime
