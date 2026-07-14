#include "ftlpu/compiler/Pipelines/phases.hpp"

#include <stdexcept>
#include <string>

namespace ftlpu::compiler::pipeline {

const char* phase_name(Phase phase)
{
    switch (phase) {
    case Phase::Start:
        return "start";
    case Phase::StableHloInput:
        return "stablehlo-input";
    case Phase::Kernel:
        return "kernel";
    case Phase::Tensor:
        return "tensor";
    case Phase::Stream:
        return "stream";
    case Phase::Schedule:
        return "schedule";
    case Phase::ExecutableBinary:
        return "executable-binary";
    case Phase::End:
        return "end";
    }
    return "unknown";
}

Phase parse_phase(const char* value)
{
    const auto phase = std::string {value};
    if (phase == "start") {
        return Phase::Start;
    }
    if (phase == "stablehlo-input" || phase == "input") {
        return Phase::StableHloInput;
    }
    if (phase == "kernel") {
        return Phase::Kernel;
    }
    if (phase == "tensor") {
        return Phase::Tensor;
    }
    if (phase == "stream") {
        return Phase::Stream;
    }
    if (phase == "schedule") {
        return Phase::Schedule;
    }
    if (phase == "executable-binary" || phase == "binary") {
        return Phase::ExecutableBinary;
    }
    if (phase == "end") {
        return Phase::End;
    }
    throw std::runtime_error("unknown compiler phase: " + phase);
}

} // namespace ftlpu::compiler::pipeline
