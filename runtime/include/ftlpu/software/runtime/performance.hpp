#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <cstddef>
#include <ostream>

namespace ftlpu::software::runtime {

// Reports ICU queue issue utilization from the loaded binary timeline.
// Datapath state remains owned by the CModel; aggregation and presentation
// belong to the runtime.
void print_runtime_performance(
    const BinaryProgram& program, std::size_t executed_cycles, std::ostream& os);

} // namespace ftlpu::software::runtime
