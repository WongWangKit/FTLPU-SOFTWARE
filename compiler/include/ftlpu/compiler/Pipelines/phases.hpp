#pragma once

namespace ftlpu::compiler::pipeline {

enum class Phase {
    Start,
    StableHloInput,
    Kernel,
    Tensor,
    Stream,
    Schedule,
    ExecutableBinary,
    End,
};

const char* phase_name(Phase phase);
Phase parse_phase(const char* value);

} // namespace ftlpu::compiler::pipeline
