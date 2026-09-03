#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <cstdint>
#include <string_view>
#include <utility>

namespace ftlpu::software::runtime {

inline constexpr std::string_view kLpu32StreamTargetName = hw::kTargetName;

// Self-contained hardware contract embedded in every executable. Structural
// fields describe the instruction and transport geometry; capacity fields may
// select a logical subset of a larger CModel instance.
struct ExecutableHardwareConfig {
    std::uint32_t hemispheres{hw::kHemispheres};
    std::uint32_t slices_per_hemisphere{hw::kMemSliceColumns};
    std::uint32_t banks_per_slice{hw::kMemBanksPerSlice};
    std::uint32_t words_per_bank{hw::kSramDepthRows};
    std::uint32_t bytes_per_word{hw::kSramRowBytes};
    std::uint32_t sram_depth_rows{hw::kSramDepthRows};
    std::uint32_t sram_read_ports_per_slice{1};
    std::uint32_t sram_write_ports_per_slice{1};
    std::uint32_t streams_per_direction{hw::kStreamsPerDirection};
    std::uint32_t encoded_streams{hw::kStreams};
    std::uint32_t c2c_streams_per_direction{hw::kC2cStreamsPerDirection};
    std::uint32_t c2c_bytes_per_stream_per_cycle{hw::kPhysicalVectorBytes};
    std::uint32_t mem_boundary_register_columns{hw::kMemBoundaryStreamRegisterColumns};
    std::uint32_t system_register_columns{hw::kSystemStreamRegisterColumns};
    std::uint32_t mem_slices_per_register_group{hw::kMemSlicesPerGroup};
    std::uint32_t tile_rows{hw::kTileRows};
    std::uint32_t lanes_per_tile{hw::kLanesPerTile};
    std::uint32_t mem_read_bytes_per_cycle{hw::kMemReadBytesPerCycle};
    std::uint32_t mem_write_bytes_per_cycle{hw::kMemWriteBytesPerCycle};
    std::uint32_t mxm_rows{hw::kMxmRows};
    std::uint32_t mxm_columns{hw::kMxmColumns};
    std::uint32_t mxm_load_streams_per_cycle{hw::kMxmLoadStreamsPerCycle};
    std::uint32_t mxm_int8_load_streams_per_cycle{hw::kMxmInt8LoadStreamsPerCycle};
    std::uint32_t mxm_load_bytes_per_cycle{hw::kMxmLoadBytesPerCycle};
    std::uint32_t mxm_activation_streams{4};
    std::uint32_t mxm_result_streams{4};
    std::uint32_t mxm_pipeline_rows{4};
    std::uint32_t mxm_block_rows{8};
    std::uint32_t mxm_local_dequant_enabled{1};
    std::uint32_t mxm_block_compute_enabled{0};
    std::uint32_t mxm_weight_activation_overlap_enabled{1};
    std::uint32_t mxm_local_load_to_compute_latency{4};
    std::uint32_t mxm_block_group_interval{8};
    std::uint32_t mxm_earliest_iw_cycle{2};
    std::uint32_t qk_iw_to_compute_latency{24};
    std::uint32_t mxms_per_hemisphere{hw::kMxmsPerHemisphere};
    std::uint32_t mxm_weight_buffers{2};
    std::uint32_t vxm_alus{hw::kVxmAluCount};
    std::uint32_t vxm_cross_hemisphere_streams_enabled{1};
    std::uint32_t vxm_fma_enabled{1};
    std::uint32_t vxm_weight_to_iw_latency{16};
    std::uint32_t mem_to_sxm_latency{14};
    std::uint32_t mem_to_mxm_latency{16};
    std::uint32_t mxm0_accumulator_latency{6};
    std::uint32_t mxm1_accumulator_latency{5};
    std::uint32_t accumulator_to_vxm_latency{19};
    std::uint32_t accumulator_read_to_vxm_latency{16};
    std::uint32_t swiglu_write_latency{13};
    std::uint32_t mxm_accumulator_blocks{hw::kMxmAccumulatorBlockCount};
    std::uint32_t lpu_clock_mhz{500};
    std::uint32_t ddr_peak_bandwidth_mbytes_per_second{51200};
    std::uint32_t ddr_scheduling_efficiency_percent{90};
    std::uint32_t ddr_read_latency_cycles{35};
    std::uint32_t ddr_write_latency_cycles{25};
    std::uint32_t ddr_read_latency_jitter_cycles{15};
    std::uint32_t ddr_write_latency_jitter_cycles{10};
    std::uint32_t ddr_request_queue_depth{256};
    std::uint32_t ddr_latency_random_seed{0x46544c50};
    // Per-ICU physical instruction-memory geometry. The current CModel uses
    // one decoded QueueCommand per slot; instruction_bits therefore describes
    // the nominal physical storage cost of that abstract slot.
    std::uint32_t icu_mem_instruction_bits{hw::kIcuMemInstructionBits};
    std::uint32_t icu_mem_imem_depth{hw::kIcuMemImemDepth};
    std::uint32_t icu_mxm_instruction_bits{hw::kIcuMxmInstructionBits};
    std::uint32_t icu_mxm_imem_depth{hw::kIcuMxmImemDepth};
    std::uint32_t icu_vxm_instruction_bits{hw::kIcuVxmInstructionBits};
    std::uint32_t icu_vxm_imem_depth{hw::kIcuVxmImemDepth};
    std::uint32_t icu_sxm_instruction_bits{hw::kIcuSxmInstructionBits};
    std::uint32_t icu_sxm_imem_depth{hw::kIcuSxmImemDepth};
    std::uint32_t icu_macro_encoding_version{1};
    std::uint32_t icu_mem_macro_contexts{hw::kIcuMemMacroContextDepth};
    std::uint32_t icu_mxm_macro_contexts{hw::kIcuMxmMacroContextDepth};
    std::uint32_t icu_mem_macro_context_bits{hw::kIcuMemMacroContextBits};
    std::uint32_t icu_mxm_macro_context_bits{hw::kIcuMxmMacroContextBits};

    template <typename Visitor>
    constexpr void visit(Visitor&& visitor)
    {
        visit_impl(*this, std::forward<Visitor>(visitor));
    }

    template <typename Visitor>
    constexpr void visit(Visitor&& visitor) const
    {
        visit_impl(*this, std::forward<Visitor>(visitor));
    }

    template <typename Visitor>
    constexpr void visit_pre_v19(Visitor&& visitor)
    {
        visit_pre_v19_impl(*this, std::forward<Visitor>(visitor));
    }

    template <typename Visitor>
    constexpr void visit_pre_v20(Visitor&& visitor)
    {
        visit_impl(*this, std::forward<Visitor>(visitor), true, false, false,
                   false, false, false, false);
    }

    template <typename Visitor>
    constexpr void visit_pre_v23(Visitor&& visitor)
    {
        visit_impl(*this, std::forward<Visitor>(visitor), true, true, false,
                   false, false, false, false);
    }

    template <typename Visitor>
    constexpr void visit_pre_v24(Visitor&& visitor)
    {
        visit_impl(*this, std::forward<Visitor>(visitor), true, true, true,
                   false, false, false, false);
    }

    template <typename Visitor>
    constexpr void visit_pre_v26(Visitor&& visitor)
    {
        visit_impl(*this, std::forward<Visitor>(visitor), true, true, true,
                   true, false, false, false);
    }

    template <typename Visitor>
    constexpr void visit_v26(Visitor&& visitor)
    {
        visit_impl(*this, std::forward<Visitor>(visitor), true, true, true,
                   true, true, false, false);
    }

    template <typename Visitor>
    constexpr void visit_legacy_v27(Visitor&& visitor)
    {
        visit_impl(*this, std::forward<Visitor>(visitor), true, true, true,
                   true, false, true, false);
    }

    template <typename Visitor>
    constexpr void visit_legacy_v28(Visitor&& visitor)
    {
        visit_impl(*this, std::forward<Visitor>(visitor), true, true, true,
                   true, false, true, true);
    }

    template <typename Visitor>
    constexpr void visit_command_abi(Visitor&& visitor) const
    {
        visit_impl(*this, std::forward<Visitor>(visitor), true, true, true,
                   true, true, false, false);
    }

private:
    template <typename Self, typename Visitor>
    static constexpr void visit_pre_v19_impl(Self& self, Visitor&& visitor)
    {
        visit_impl(self, std::forward<Visitor>(visitor), false, false, false,
                   false, false, false, false);
    }

    template <typename Self, typename Visitor>
    static constexpr void visit_impl(Self& self, Visitor&& visitor)
    {
        visit_impl(self, std::forward<Visitor>(visitor), true, true, true,
                   true, true, true, true);
    }

    template <typename Self, typename Visitor>
    static constexpr void visit_impl(Self& self, Visitor&& visitor,
                                     bool include_accumulator_capacity,
                                     bool include_c2c_fabric,
                                     bool include_external_memory,
                                     bool include_ddr_scheduling_efficiency,
                                     bool include_vxm_stream_fma,
                                     bool include_imem_geometry,
                                     bool include_macro_geometry)
    {
        visitor(self.hemispheres); visitor(self.slices_per_hemisphere);
        visitor(self.banks_per_slice); visitor(self.words_per_bank);
        visitor(self.bytes_per_word); visitor(self.sram_depth_rows);
        visitor(self.sram_read_ports_per_slice);
        visitor(self.sram_write_ports_per_slice);
        visitor(self.streams_per_direction); visitor(self.encoded_streams);
        if (include_c2c_fabric) {
            visitor(self.c2c_streams_per_direction);
            visitor(self.c2c_bytes_per_stream_per_cycle);
        }
        visitor(self.mem_boundary_register_columns);
        visitor(self.system_register_columns);
        visitor(self.mem_slices_per_register_group); visitor(self.tile_rows);
        visitor(self.lanes_per_tile); visitor(self.mem_read_bytes_per_cycle);
        visitor(self.mem_write_bytes_per_cycle); visitor(self.mxm_rows);
        visitor(self.mxm_columns); visitor(self.mxm_load_streams_per_cycle);
        visitor(self.mxm_int8_load_streams_per_cycle);
        visitor(self.mxm_load_bytes_per_cycle);
        visitor(self.mxm_activation_streams);
        visitor(self.mxm_result_streams); visitor(self.mxm_pipeline_rows);
        visitor(self.mxm_block_rows); visitor(self.mxm_local_dequant_enabled);
        visitor(self.mxm_block_compute_enabled);
        visitor(self.mxm_weight_activation_overlap_enabled);
        visitor(self.mxm_local_load_to_compute_latency);
        visitor(self.mxm_block_group_interval);
        visitor(self.mxm_earliest_iw_cycle);
        visitor(self.qk_iw_to_compute_latency);
        visitor(self.mxms_per_hemisphere);
        visitor(self.mxm_weight_buffers); visitor(self.vxm_alus);
        if (include_vxm_stream_fma) {
            visitor(self.vxm_cross_hemisphere_streams_enabled);
            visitor(self.vxm_fma_enabled);
        }
        visitor(self.vxm_weight_to_iw_latency);
        visitor(self.mem_to_sxm_latency); visitor(self.mem_to_mxm_latency);
        visitor(self.mxm0_accumulator_latency);
        visitor(self.mxm1_accumulator_latency);
        visitor(self.accumulator_to_vxm_latency);
        visitor(self.accumulator_read_to_vxm_latency);
        visitor(self.swiglu_write_latency);
        if (include_accumulator_capacity)
            visitor(self.mxm_accumulator_blocks);
        if (include_external_memory) {
            visitor(self.lpu_clock_mhz);
            visitor(self.ddr_peak_bandwidth_mbytes_per_second);
            if (include_ddr_scheduling_efficiency)
                visitor(self.ddr_scheduling_efficiency_percent);
            visitor(self.ddr_read_latency_cycles);
            visitor(self.ddr_write_latency_cycles);
            visitor(self.ddr_read_latency_jitter_cycles);
            visitor(self.ddr_write_latency_jitter_cycles);
            visitor(self.ddr_request_queue_depth);
            visitor(self.ddr_latency_random_seed);
        }
        if (include_imem_geometry) {
            visitor(self.icu_mem_instruction_bits);
            visitor(self.icu_mem_imem_depth);
            visitor(self.icu_mxm_instruction_bits);
            visitor(self.icu_mxm_imem_depth);
            visitor(self.icu_vxm_instruction_bits);
            visitor(self.icu_vxm_imem_depth);
            visitor(self.icu_sxm_instruction_bits);
            visitor(self.icu_sxm_imem_depth);
        }
        if (include_macro_geometry) {
            visitor(self.icu_macro_encoding_version);
            visitor(self.icu_mem_macro_contexts);
            visitor(self.icu_mxm_macro_contexts);
            visitor(self.icu_mem_macro_context_bits);
            visitor(self.icu_mxm_macro_context_bits);
        }
    }
};

class TargetAbiHasher {
public:
    constexpr void add(std::int64_t value)
    {
        const auto bits = static_cast<std::uint64_t>(value);
        for (unsigned shift = 0; shift < 64; shift += 8) {
            value_ ^= static_cast<std::uint8_t>(bits >> shift);
            value_ *= 1099511628211ULL;
        }
    }

    constexpr std::uint64_t value() const { return value_; }

private:
    std::uint64_t value_ = 14695981039346656037ULL;
};

// This list is the command ABI contract for the CModel-backed 32-stream LPU.
// LPUTargetModel::abi_fingerprint() hashes fields in exactly this order.
constexpr std::uint64_t lpu_32stream_target_abi(
    std::int64_t mxms_per_hemisphere = hw::kMxmsPerHemisphere)
{
    ExecutableHardwareConfig config;
    config.mxms_per_hemisphere =
        static_cast<std::uint32_t>(mxms_per_hemisphere);
    TargetAbiHasher hash;
    hash.add(17); // Hardware ABI schema version (VXM stream/FMA capability).
    // i-MEM capacity is a deployment constraint rather than a command-bit ABI
    // change. Keep legacy Schedule attributes valid and validate geometry
    // explicitly when loading an executable into a CModel.
    config.visit_command_abi([&](std::uint32_t value) { hash.add(value); });
    return hash.value();
}

constexpr std::uint64_t executable_target_abi(
    const ExecutableHardwareConfig& config)
{
    TargetAbiHasher hash;
    hash.add(17); // Hardware ABI schema version (VXM stream/FMA capability).
    config.visit_command_abi([&](std::uint32_t value) { hash.add(value); });
    return hash.value();
}

inline constexpr std::uint64_t kLpu32StreamTargetAbi =
    lpu_32stream_target_abi();

} // namespace ftlpu::software::runtime
