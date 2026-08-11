#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>

namespace ftlpu {
class TspSliceSystem;
}

namespace ftlpu::software::runtime {

// Reports ICU queue issue utilization from the loaded binary timeline.
// Datapath state remains owned by the CModel; aggregation and presentation
// belong to the runtime.
void print_runtime_performance(
    const BinaryProgram& program, std::size_t executed_cycles, std::ostream& os);

class DatapathPerformanceMonitor {
public:
    void reset();
    void sample(const TspSliceSystem& system,
        std::size_t mxms_per_hemisphere, std::size_t vxm_alus);
    void print(const TspSliceSystem& system,
        std::size_t mxms_per_hemisphere, std::size_t vxm_alus,
        std::ostream& os) const;

private:
    struct Utilization {
        std::uint64_t active_slots{0};
        std::uint64_t non_idle_cycles{0};
        std::uint64_t peak_active_slots{0};
    };

    std::uint64_t sampled_cycles_{0};
    std::array<Utilization, hw::kMxmCount> mxm_{};
    Utilization vxm_{};
    std::uint64_t vxm_useful_slots_{0};
    std::array<Utilization, hw::kHemispheres> sr_{};
    std::array<std::uint64_t, hw::kHemispheres> sr_east_{};
    std::array<std::uint64_t, hw::kHemispheres> sr_west_{};
};

} // namespace ftlpu::software::runtime
