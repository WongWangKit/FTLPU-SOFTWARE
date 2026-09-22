#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    if (argc != 2)
        throw std::runtime_error(
            "usage: command_mxm_decode_binary_test program.ftlpu");

    const auto program = read_binary_program(std::filesystem::path(argv[1]));
    const auto findQueue = [&](QueueKind kind) -> const QueueProgram& {
        for (const auto& queue : program.queues)
            if (queue.kind == kind && queue.index == 0) return queue;
        throw std::logic_error("decode workload is missing an MXM queue");
    };
    const auto& load = findQueue(QueueKind::MxmLoad);
    const auto& compute = findQueue(QueueKind::MxmCompute);
    const auto& dequant = findQueue(QueueKind::MxmDequant);
    require(load.packed_macro_imem.has_value()
            && compute.packed_macro_imem.has_value()
            && dequant.packed_macro_imem.has_value(),
        "decode workload did not use packed Macro queues");

    InstructionControlUnit icu;
    const auto prefill = load_queue_programs_into_icu(program.queues, icu);
    require(prefill != 0, "decode Macro workload bypassed raw ICU prefill");
    require(icu.mxm_load_iq(0).raw_macro_mode()
            && icu.mxm_compute_iq(0).raw_macro_mode(),
        "decode workload did not enter the raw Macro frontend");

    bool sawLoad = false;
    std::size_t streamComputeCount = 0;
    bool sawAccumulatorRead = false;
    for (std::size_t cycle = 0; cycle <= program.max_cycle; ++cycle) {
        if (const auto instruction = icu.mxm_load_iq(0).tick()) {
            require(cycle == 8
                    && instruction->opcode == MxmControlOpcode::Decode
                    && instruction->decode_operation
                        == MxmDecodeOperation::LoadActivation
                    && instruction->weight_buffer == 1
                    && instruction->decode_layout
                        == MxmDecodeLayout::Native4x4,
                "compiler changed Decode.LoadActivation");
            sawLoad = true;
        }
        if (const auto instruction = icu.mxm_compute_iq(0).tick()) {
            if (instruction->opcode == MxmControlOpcode::Decode) {
                require(cycle == 16 + streamComputeCount
                        && instruction->decode_operation
                            == MxmDecodeOperation::StreamCompute
                        && instruction->accumulator_address == 64
                        && instruction->weight_column == 2
                        && instruction->decode_layout
                            == MxmDecodeLayout::Native4x4,
                    "compiler changed Decode.StreamCompute");
                ++streamComputeCount;
            } else if (instruction->opcode
                == MxmControlOpcode::AccumulatorRead) {
                require(cycle == 24 && instruction->accumulator_address == 64,
                    "decode workload lost its AccumulatorRead");
                sawAccumulatorRead = true;
            } else {
                throw std::logic_error(
                    "decode workload emitted an unexpected compute operation");
            }
        }
        (void)icu.mxm_dequant_iq(0).tick();
    }
    require(sawLoad && streamComputeCount == 2 && sawAccumulatorRead,
        "compiler-generated decode workload did not issue the new MXM ISA");

    std::cout << "command_mxm_decode_binary_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "command_mxm_decode_binary_test failed: "
              << error.what() << '\n';
    return 1;
}
