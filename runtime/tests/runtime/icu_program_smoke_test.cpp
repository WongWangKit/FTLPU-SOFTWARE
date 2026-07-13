#include "ftlpu/software/runtime/icu_program.hpp"

#include <cassert>
#include <cstddef>
#include <iostream>

int main()
{
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;

    auto program = IcuProgram {};
    assert(program.empty());

    program.emit_mem(3, 32, MemInstruction::Read(0, 0));
    program.emit_mxm_load(5, 0, MxmControlInstruction::IW(0, 0));
    program.emit_mxm_compute(7, 0, MxmControlInstruction::Compute(0));
    program.emit_mxm_output(9, 0, MxmControlInstruction::Output(32));
    program.emit_vxm(11, 0, VxmLaneAluInstruction {
                            VxmAluOpcode::Cast,
                            VxmLaneOperand::StreamInt32(32),
                            VxmLaneOperand::Imm(0.0f),
                            1.0f,
                            0,
                            VxmCastTarget::Float32,
                        });

    const auto queues = program.encode_queues();
    assert(!program.empty());
    assert(program.last_cycle() == 11);
    assert(queues.size() == 66);

    std::size_t non_empty_queues = 0;
    for (const auto& queue : queues) {
        if (!queue.commands.empty()) {
            ++non_empty_queues;
        }
    }
    assert(non_empty_queues == 5);

    auto icu = InstructionControlUnit {};
    program.load_into(icu);

    std::cout << "icu_program_smoke_test passed\n";
    return 0;
}
