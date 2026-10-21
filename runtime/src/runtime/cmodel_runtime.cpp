#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"

#include "ftlpu/system/c2c_dma_system.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"
#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"
#include "ftlpu/software/runtime/common_page_storage.hpp"

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

// Keep this translation unit rebuilt when BinaryProgram ABI evolves.
#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ftlpu::software::runtime {
namespace {

const char* macro_queue_kind_name(IcuMacroQueueKind kind)
{
    switch (kind) {
    case IcuMacroQueueKind::Mem: return "mem";
    case IcuMacroQueueKind::MxmLoad: return "mxm_load";
    case IcuMacroQueueKind::MxmCompute: return "mxm_compute";
    case IcuMacroQueueKind::MxmDequant: return "mxm_dequant";
    }
    return "unknown";
}


template <typename System>
std::size_t stage_and_prime_icu_programs_if_supported(
    System& system, std::uint64_t imageBase)
{
    if constexpr (requires(System& candidate) {
                      candidate.stage_icu_programs_to_ddr(imageBase);
                      candidate.prime_icu_instruction_queues();
                  }) {
        static_cast<void>(system.stage_icu_programs_to_ddr(imageBase));
        return system.prime_icu_instruction_queues();
    }
    return 0;
}

void write_binding_sram_byte(TspSliceSystem& system,
    const BinaryBinding& binding, Hemisphere hemisphere, std::size_t slice,
    std::size_t address, std::size_t column, std::uint8_t value)
{
    system.initialize_mem_sram_lane_byte(hemisphere, slice, binding.bank,
        column / hw::kLanesPerTile, address,
        column % hw::kLanesPerTile, value);
}

std::uint8_t read_binding_sram_byte(const TspSliceSystem& system,
    const BinaryBinding& binding, Hemisphere hemisphere, std::size_t slice,
    std::size_t address, std::size_t column)
{
    return system.read_mem_sram_lane_byte(hemisphere, slice, binding.bank,
        column / hw::kLanesPerTile, address,
        column % hw::kLanesPerTile);
}

template <typename Fn>
void for_each_binding_hemisphere(const BinaryBinding& binding, Fn&& fn)
{
    if ((binding.hemisphere_mask & 1) != 0) fn(Hemisphere::East);
    if ((binding.hemisphere_mask & 2) != 0) fn(Hemisphere::West);
}

Hemisphere single_binding_hemisphere(const BinaryBinding& binding)
{
    return (binding.hemisphere_mask & 2) != 0
        && (binding.hemisphere_mask & 1) == 0
        ? Hemisphere::West : Hemisphere::East;
}

void require_matrix_binding(const BinaryBinding& binding)
{
    if (binding.shape.size() != 2 || binding.base_row < 0
        || binding.address_stride == 0 || binding.slices.empty())
        throw std::logic_error("runtime currently requires a valid rank-2 physical binding");
}

bool is_16bit_float(BindingElementType type)
{
    return type == BindingElementType::F16
        || type == BindingElementType::BF16;
}

template <typename Fn>
std::vector<VxmLutEntry> make_vxm_lut(
    float input_min, float segment_width, std::size_t count, Fn fn)
{
    std::vector<VxmLutEntry> entries;
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float x0 = input_min
            + static_cast<float>(index) * segment_width;
        const float y0 = fn(x0);
        entries.push_back(VxmLutEntry::from_float(
            (fn(x0 + segment_width) - y0) / segment_width, y0));
    }
    return entries;
}

void initialize_vxm_luts(TspSliceSystem& system)
{
    constexpr std::size_t entries = 256;
    constexpr float ln2 = 0.6931471805599453f;
    const float expWidth = ln2 / static_cast<float>(entries);
    system.initialize_vxm_lut(VxmSpecialAluOpcode::Exp,
        {-ln2 / 2.0f, expWidth},
        make_vxm_lut(-ln2 / 2.0f, expWidth, entries,
            [](float x) { return std::exp(x); }));

    const float reciprocalWidth = 1.0f / static_cast<float>(entries);
    system.initialize_vxm_lut(VxmSpecialAluOpcode::Reciprocal,
        {1.0f, reciprocalWidth},
        make_vxm_lut(1.0f, reciprocalWidth, entries,
            [](float x) { return 1.0f / x; }));

    const float rsqrtWidth = 3.0f / static_cast<float>(entries);
    system.initialize_vxm_lut(VxmSpecialAluOpcode::Rsqrt,
        {1.0f, rsqrtWidth},
        make_vxm_lut(1.0f, rsqrtWidth, entries,
            [](float x) { return 1.0f / std::sqrt(x); }));
}

void validate_cmodel_hardware_config(const BinaryProgram& program)
{
    const auto& requested = program.hardware;
    const auto physical = ExecutableHardwareConfig {};
    if (program.target_abi != executable_target_abi(requested)) {
        std::ostringstream message;
        message << "binary target '" << program.target_name
                << "' ABI does not match its embedded hardware configuration"
                << ": encoded=" << program.target_abi
                << " computed=" << executable_target_abi(requested);
        throw std::invalid_argument(message.str());
    }

    const auto require_exact = [&](const char* field, std::uint32_t value,
                                   std::uint32_t available) {
        if (value == available) return;
        std::ostringstream message;
        message << "CModel hardware mismatch for " << field
                << ": executable requires " << value
                << ", CModel implements " << available;
        throw std::invalid_argument(message.str());
    };
    const auto require_capacity = [&](const char* field, std::uint32_t value,
                                      std::uint32_t available) {
        if (value != 0 && value <= available) return;
        std::ostringstream message;
        message << "CModel capacity mismatch for " << field
                << ": executable requires " << value
                << ", CModel provides " << available;
        throw std::invalid_argument(message.str());
    };
    const auto require_capability = [&](const char* field, std::uint32_t value,
                                        std::uint32_t available) {
        if (value <= available) return;
        std::ostringstream message;
        message << "CModel capability mismatch for " << field
                << ": executable requires " << value
                << ", CModel provides " << available;
        throw std::invalid_argument(message.str());
    };

#define REQUIRE_EXACT(field) \
    require_exact(#field, requested.field, physical.field)
    REQUIRE_EXACT(hemispheres);
    REQUIRE_EXACT(slices_per_hemisphere);
    REQUIRE_EXACT(banks_per_slice);
    REQUIRE_EXACT(bytes_per_word);
    REQUIRE_EXACT(sram_read_ports_per_slice);
    REQUIRE_EXACT(sram_write_ports_per_slice);
    REQUIRE_EXACT(streams_per_direction);
    REQUIRE_EXACT(encoded_streams);
    REQUIRE_EXACT(c2c_bytes_per_stream_per_cycle);
    REQUIRE_EXACT(mem_boundary_register_columns);
    REQUIRE_EXACT(system_register_columns);
    REQUIRE_EXACT(mem_slices_per_register_group);
    REQUIRE_EXACT(tile_rows);
    REQUIRE_EXACT(lanes_per_tile);
    REQUIRE_EXACT(mem_read_bytes_per_cycle);
    REQUIRE_EXACT(mem_write_bytes_per_cycle);
    REQUIRE_EXACT(mxm_rows);
    REQUIRE_EXACT(mxm_columns);
    REQUIRE_EXACT(mxm_load_streams_per_cycle);
    REQUIRE_EXACT(mxm_int8_load_streams_per_cycle);
    REQUIRE_EXACT(mxm_load_bytes_per_cycle);
    REQUIRE_EXACT(mxm_activation_streams);
    REQUIRE_EXACT(mxm_result_streams);
    REQUIRE_EXACT(mxm_pipeline_rows);
    REQUIRE_EXACT(mxm_block_rows);
    REQUIRE_EXACT(mxm_local_load_to_compute_latency);
    REQUIRE_EXACT(mxm_block_group_interval);
    REQUIRE_EXACT(mxm_earliest_iw_cycle);
    REQUIRE_EXACT(qk_iw_to_compute_latency);
    REQUIRE_EXACT(vxm_weight_to_iw_latency);
    REQUIRE_EXACT(mem_to_sxm_latency);
    REQUIRE_EXACT(mem_to_mxm_latency);
    REQUIRE_EXACT(mxm0_accumulator_latency);
    REQUIRE_EXACT(mxm1_accumulator_latency);
    REQUIRE_EXACT(accumulator_to_vxm_latency);
    REQUIRE_EXACT(accumulator_read_to_vxm_latency);
    REQUIRE_EXACT(swiglu_write_latency);
    REQUIRE_EXACT(icu_mem_instruction_bits);
    REQUIRE_EXACT(icu_mxm_instruction_bits);
    REQUIRE_EXACT(icu_vxm_instruction_bits);
    REQUIRE_EXACT(icu_sxm_instruction_bits);
    REQUIRE_EXACT(icu_macro_encoding_version);
    REQUIRE_EXACT(icu_mem_macro_context_bits);
    REQUIRE_EXACT(icu_mxm_macro_context_bits);
    if (program.common_barrier_protocol)
        REQUIRE_EXACT(common_barrier_encoding_version);
#undef REQUIRE_EXACT

    require_capacity("sram_depth_rows", requested.sram_depth_rows,
        physical.sram_depth_rows);
    require_capacity("words_per_bank", requested.words_per_bank,
        physical.words_per_bank);
    require_capacity("mxms_per_hemisphere", requested.mxms_per_hemisphere,
        physical.mxms_per_hemisphere);
    require_capacity("mxm_weight_buffers", requested.mxm_weight_buffers,
        physical.mxm_weight_buffers);
    require_capacity("mxm_accumulator_blocks",
        requested.mxm_accumulator_blocks,
        physical.mxm_accumulator_blocks);
    require_capacity("icu_mem_imem_depth", requested.icu_mem_imem_depth,
        physical.icu_mem_imem_depth);
    require_capacity("icu_mxm_imem_depth", requested.icu_mxm_imem_depth,
        physical.icu_mxm_imem_depth);
    require_capacity("icu_vxm_imem_depth", requested.icu_vxm_imem_depth,
        physical.icu_vxm_imem_depth);
    require_capacity("icu_sxm_imem_depth", requested.icu_sxm_imem_depth,
        physical.icu_sxm_imem_depth);
    require_capacity("icu_mem_macro_contexts",
        requested.icu_mem_macro_contexts,
        physical.icu_mem_macro_contexts);
    require_capacity("icu_mxm_macro_contexts",
        requested.icu_mxm_macro_contexts,
        physical.icu_mxm_macro_contexts);
    if (program.common_barrier_protocol)
        require_capacity("common_barrier_capacity", requested.common_barrier_capacity,
            physical.common_barrier_capacity);
    require_capacity("vxm_alus", requested.vxm_alus, physical.vxm_alus);
    require_capacity("c2c_streams_per_direction",
        requested.c2c_streams_per_direction,
        physical.c2c_streams_per_direction);
    require_capability("mxm_local_dequant_enabled",
        requested.mxm_local_dequant_enabled,
        physical.mxm_local_dequant_enabled);
    require_capability("mxm_block_compute_enabled",
        requested.mxm_block_compute_enabled,
        physical.mxm_block_compute_enabled);
    require_capability("mxm_weight_activation_overlap_enabled",
        requested.mxm_weight_activation_overlap_enabled,
        physical.mxm_weight_activation_overlap_enabled);
    require_capability("vxm_cross_hemisphere_streams_enabled",
        requested.vxm_cross_hemisphere_streams_enabled,
        physical.vxm_cross_hemisphere_streams_enabled);
}

std::uint16_t encode_16bit_float(float value, BindingElementType type)
{
    if (type == BindingElementType::BF16)
        return Bf16::from_float(value).bits();
    if (type == BindingElementType::F16)
        return Fp16::from_float(value).bits();
    throw std::logic_error("binding is not a supported 16-bit float");
}

std::uint16_t read_16bit_element(const TspSliceSystem& system,
    const BinaryBinding& binding, std::size_t row, std::size_t column,
    std::size_t mxms_per_hemisphere,
    const ExecutableHardwareConfig& hardware)
{
    const std::size_t rows = static_cast<std::size_t>(binding.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(binding.shape[1]);
    if (binding.layout == BindingLayout::Fp16PairPlanar
        && binding.slices.size() == 4) {
        const std::size_t width = hardware.bytes_per_word;
        const std::size_t local_mxm = (column % (2 * width)) / width;
        const bool dual_hemisphere = (binding.hemisphere_mask & 3) == 3;
        const auto hemisphere = dual_hemisphere
            ? static_cast<Hemisphere>((column / (2 * width)) % 2)
            : single_binding_hemisphere(binding);
        const std::size_t address = static_cast<std::size_t>(binding.base_row)
            + (dual_hemisphere ? column / (4 * width)
                               : column / (2 * width)) * rows + row;
        return static_cast<std::uint16_t>(read_binding_sram_byte(system, binding, hemisphere,
                   binding.slices[local_mxm * 2], address, column % width))
            | (static_cast<std::uint16_t>(read_binding_sram_byte(system, binding, hemisphere,
                   binding.slices[local_mxm * 2 + 1], address, column % width))
                << 8);
    }
    if (binding.layout == BindingLayout::Fp16PairPlanar
        && binding.slices.size() == 2) {
        const std::size_t width = hardware.bytes_per_word;
        const std::size_t address = static_cast<std::size_t>(binding.base_row)
            + (column / width) * rows + row;
        const auto hemisphere = single_binding_hemisphere(binding);
        return static_cast<std::uint16_t>(read_binding_sram_byte(system, binding,
                   hemisphere, binding.slices[0], address, column % width))
            | (static_cast<std::uint16_t>(read_binding_sram_byte(system, binding,
                   hemisphere, binding.slices[1], address, column % width))
                << 8);
    }
    if ((binding.layout == BindingLayout::Fp16MxmDistributed16
            || binding.layout
                == BindingLayout::Fp16MxmBlock8Distributed16)
        && binding.slices.size() == 16) {
        const std::size_t hidden_blocks = columns / hardware.bytes_per_word;
        const std::size_t token_block = row / hardware.mxm_rows;
        const std::size_t token_wave = (row % hardware.mxm_rows) / hardware.lanes_per_tile;
        const std::size_t token_lane = row % hardware.lanes_per_tile;
        const std::size_t hidden_block = column / hardware.bytes_per_word;
        const std::size_t feature_lane = column % hardware.bytes_per_word;
        const std::size_t address = static_cast<std::size_t>(binding.base_row)
            + (token_block * hidden_blocks + hidden_block) * hardware.tile_rows
            + token_wave;
        const auto hemisphere = binding.layout
                == BindingLayout::Fp16MxmBlock8Distributed16
            ? static_cast<Hemisphere>(
                (hidden_block
                    % (hw::kHemispheres * mxms_per_hemisphere))
                / mxms_per_hemisphere)
            : Hemisphere::East;
        return static_cast<std::uint16_t>(read_binding_sram_byte(system, binding,
                   hemisphere, binding.slices[2 * token_lane], address,
                   feature_lane))
            | (static_cast<std::uint16_t>(read_binding_sram_byte(system, binding,
                   hemisphere, binding.slices[2 * token_lane + 1],
                   address, feature_lane))
                << 8);
    }
    throw std::logic_error(
        "device copy does not support the source 16-bit float layout");
}

void write_16bit_element(TspSliceSystem& system,
    const BinaryBinding& binding, std::size_t row, std::size_t column,
    std::uint16_t value, const ExecutableHardwareConfig& hardware)
{
    const std::size_t columns = static_cast<std::size_t>(binding.shape[1]);
    if (binding.layout == BindingLayout::Fp16MxmDistributed16
        && binding.slices.size() == 16) {
        const std::size_t hidden_blocks = columns / hardware.bytes_per_word;
        const std::size_t token_block = row / hardware.mxm_rows;
        const std::size_t token_wave = (row % hardware.mxm_rows) / hardware.lanes_per_tile;
        const std::size_t token_lane = row % hardware.lanes_per_tile;
        const std::size_t hidden_block = column / hardware.bytes_per_word;
        const std::size_t feature_lane = column % hardware.bytes_per_word;
        const std::size_t address = static_cast<std::size_t>(binding.base_row)
            + (token_block * hidden_blocks + hidden_block) * hardware.tile_rows
            + token_wave;
        for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
            write_binding_sram_byte(system, binding, hemisphere,
                binding.slices[2 * token_lane], address,
                feature_lane,
                static_cast<std::uint8_t>(value));
            write_binding_sram_byte(system, binding, hemisphere,
                binding.slices[2 * token_lane + 1], address,
                feature_lane,
                static_cast<std::uint8_t>(value >> 8));
        });
        return;
    }
    throw std::logic_error(
        "device copy does not support the destination 16-bit float layout");
}

} // namespace

CModelRuntime::CModelRuntime(TspSliceSystem& system)
    : system_(system)
    , tick_([&system](TspSliceSystem::LogSinks sinks) {
        system.tick(sinks);
    })
{
}

CModelRuntime::CModelRuntime(TspSliceSystem& system,
    std::function<void(TspSliceSystem::LogSinks)> tick)
    : system_(system)
    , tick_(std::move(tick))
{
}

CModelRuntime::CModelRuntime(C2cDmaSystem& system,
    std::function<void(TspSliceSystem::LogSinks)> tick)
    : system_(system.chip())
    , c2c_system_(&system)
    , tick_(std::move(tick))
{
}

void CModelRuntime::load(
    const BinaryProgram& program, bool initializeInternalData)
{
    validate_cmodel_hardware_config(program);
    const auto imem = analyze_physical_imem(program);
    if (!imem.fits()) {
        std::ostringstream detail;
        detail << "linked ICU program exceeds local i-MEM capacity:";
        for (const auto& queue : imem.queues) {
            if (!queue.deployment_overflow()) continue;
            detail << ' ' << queue_kind_name(queue.kind) << '[' << queue.index
                   << "]=" << queue.physical_slots << '/' << queue.depth;
        }
        detail << "; segmented i-MEM loading is required for this program";
        throw std::runtime_error(detail.str());
    }
    validate_binary_barriers(program);
    if (program.common_barrier_protocol) {
        // KV page-out control remains in the saved image and executes later.
        // It must never block the independent page-in transport port.
        const auto filtered = without_compiled_mem_read_sync(program);
        if (filtered.transport_queues.size() != program.transport_queues.size()
            || !std::equal(filtered.transport_queues.begin(), filtered.transport_queues.end(),
                program.transport_queues.begin(), [](const auto& a, const auto& b) {
                    return a.commands.size() == b.commands.size(); })) {
            load(filtered, initializeInternalData);
            return;
        }
    }
    if (program.common_barrier_protocol && !program.weight_page_uses.empty()
        && c2c_system_ == nullptr)
        throw std::invalid_argument("common-barrier paging requires the real C2C transport runtime");
    configure_common_transport(program);
    SystemHardwareConfiguration hardware;
    hardware.sram_depth_rows = program.hardware.sram_depth_rows;
    hardware.mxms_per_hemisphere = program.hardware.mxms_per_hemisphere;
    hardware.mxm_weight_buffers = program.hardware.mxm_weight_buffers;
    hardware.vxm_alus = program.hardware.vxm_alus;
    hardware.c2c_streams_per_direction =
        program.hardware.c2c_streams_per_direction;
    hardware.mxm_local_dequant_enabled =
        program.hardware.mxm_local_dequant_enabled != 0;
    hardware.mxm_weight_activation_overlap_enabled =
        program.hardware.mxm_weight_activation_overlap_enabled != 0;
    system_.configure_hardware(hardware);
    if (c2c_system_ != nullptr)
        c2c_system_->reset_execution_state();
    else
        system_.reset_execution_state();
    static std::atomic<std::uint64_t> nextGeneration{1};
    execution_generation_ = nextGeneration.fetch_add(1);
    if (execution_generation_ == 0)
        throw std::overflow_error("execution generation exhausted");
    common_barrier_protocol_ = program.common_barrier_protocol;
    if (common_barrier_protocol_) {
        std::vector<IcuCommonBarrierEntry> entries;
        std::vector<IcuCommonEventBinding> events;
        for (const auto& barrier : program.barriers)
            entries.push_back({barrier.stop_cycle, barrier.event_id});
        for (const auto& event : program.barrier_events)
            events.push_back({event.event_id, event.page_use_indices});
        system_.icu().configure_common_barriers(
            execution_generation_, std::move(entries), std::move(events));
    }
    load_transport_queue_programs_into_icu(program.transport_queues, system_.icu());
    initialize_vxm_luts(system_);
    for (std::size_t group = 0; group < 16; ++group)
        system_.configure_vxm_input_group_source(group,
            group < 8 ? Hemisphere::East : Hemisphere::West);
    for (std::size_t block = 0; block < 8; ++block)
        system_.configure_vxm_output_block_destination(block,
            block < 4 ? Hemisphere::West : Hemisphere::East);
    instruction_prefill_cycles_ = load_queue_programs_into_icu(program.queues, system_.icu(),
        program.hardware.mxms_per_hemisphere);
    if (c2c_system_ != nullptr) {
        constexpr auto kInstructionImageBase =
            std::uint64_t {0x1'0000'0000};
        instruction_prefill_cycles_ +=
            stage_and_prime_icu_programs_if_supported(
                *c2c_system_, kInstructionImageBase);
    }
    if (common_barrier_protocol_) {
        // Configure/launch before any cycle-zero event can be delivered.
        // Lazy launch clears notification latches as part of reset_execution.
        for (const auto& queue : program.queues) {
            if (queue.kind == QueueKind::C2cDma)
                system_.icu().c2c_dma_iq(static_cast<Hemisphere>(queue.index)).launch();
            else if (queue.kind == QueueKind::C2cRx)
                system_.icu().c2c_rx_iq(static_cast<Hemisphere>(queue.index)).launch();
        }
        for (const auto& queue : program.transport_queues)
            system_.icu().mem_transport_iq(queue.index).launch();
    }
    loaded_max_cycle_ = program.max_cycle;
    loaded_mxms_per_hemisphere_ = program.hardware.mxms_per_hemisphere;
    loaded_vxm_alus_ = program.hardware.vxm_alus;
    datapath_performance_.reset();
    bindings_ = program.bindings;
    hardware_ = program.hardware;
    // Common-protocol pages become visible only through actual MEM commits.
    weight_page_uses_ = common_barrier_protocol_
        ? std::vector<BinaryWeightPageUse>{} : program.weight_page_uses;
    std::sort(weight_page_uses_.begin(), weight_page_uses_.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.ready_cycle != rhs.ready_cycle
                ? lhs.ready_cycle < rhs.ready_cycle
                : lhs.binding_index != rhs.binding_index
                ? lhs.binding_index < rhs.binding_index
                : lhs.page_index < rhs.page_index;
        });
    paged_weight_data_.clear();
    next_weight_page_use_ = 0;
    executed_cycles_ = 0;
    physical_cycles_ = 0;
    system_.icu().set_program_issue_enabled(true);
    if (execution_trace_enabled_)
        execution_trace_.begin_segment(program,
            execution_trace_cycle_offset_, execution_trace_append_on_load_);
    if (mem_execution_trace_enabled_)
        mem_execution_trace_.begin_segment(
            execution_trace_cycle_offset_, execution_trace_append_on_load_);
    if (initializeInternalData)
        initialize_internal_data(program);
}

void CModelRuntime::initialize_internal_data(const BinaryProgram& program)
{
    for (const BinaryBinding& binding : program.bindings) {
        if (binding.access != BindingAccess::Internal) continue;
        if (binding.initializer == BindingInitializer::None) continue;
        if (binding.initializer == BindingInitializer::Zero) {
            if (binding.base_row < 0 || binding.instruction_count <= 0)
                throw std::logic_error("invalid zero-initialized internal binding");
            for (std::int64_t row = 0; row < binding.instruction_count; ++row)
                for (std::uint16_t slice : binding.slices)
                    for (std::size_t column = 0; column < 32; ++column)
                        for_each_binding_hemisphere(binding,
                            [&](Hemisphere hemisphere) {
                                write_binding_sram_byte(system_, binding, hemisphere, slice,
                                    static_cast<std::size_t>(binding.base_row + row),
                                    column, 0);
                            });
            continue;
        }
        if (binding.initializer == BindingInitializer::CausalMask) {
            require_matrix_binding(binding);
            const bool fp32Mask =
                binding.layout == BindingLayout::Fp32CausalMaskTile
                && binding.element_type == BindingElementType::F32
                && binding.slices.size() == sizeof(float);
            const bool fp16Mask =
                binding.layout == BindingLayout::Fp16CausalMaskTile
                && is_16bit_float(binding.element_type)
                && binding.slices.size() == sizeof(std::uint16_t);
            if ((!fp32Mask && !fp16Mask)
                || binding.shape[0] + 1 != binding.shape[1])
                throw std::logic_error("invalid internal causal-mask binding");
            const std::size_t rows = static_cast<std::size_t>(binding.shape[0]);
            const std::size_t columns = static_cast<std::size_t>(binding.shape[1]);
            const std::size_t stride =
                static_cast<std::size_t>(std::abs(binding.address_stride));
            for (std::size_t row = 0; row < rows; ++row) {
                const std::size_t local_key = row + 1;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row) + row * stride;
                for (std::size_t query_lane = 0; query_lane < columns; ++query_lane) {
                    const float mask = local_key <= query_lane ? 0.0f : -1.0e9f;
                    const std::uint32_t bits = fp16Mask
                        ? encode_16bit_float(mask, binding.element_type)
                        : std::bit_cast<std::uint32_t>(mask);
                    for (std::size_t byte = 0; byte < binding.slices.size(); ++byte)
                        for_each_binding_hemisphere(binding,
                            [&](Hemisphere hemisphere) {
                                write_binding_sram_byte(system_, binding, hemisphere,
                                    binding.slices[byte], address, query_lane,
                                    static_cast<std::uint8_t>(bits >> (8 * byte)));
                            });
                }
            }
            continue;
        }
        if (binding.initializer == BindingInitializer::RopeTable) {
            const bool decodeCompact = binding.layout
                == BindingLayout::Fp16RopeTableDecodeCompact;
            if ((binding.layout != BindingLayout::Fp16RopeTable && !decodeCompact)
                || !is_16bit_float(binding.element_type)
                || binding.shape.size() != 3 || binding.shape[2] != 2
                || binding.slices.size() != 4 || binding.base_row < 0
                || binding.address_stride == 0
                || binding.rope_head_dim != 2 * binding.shape[1]
                || (binding.rope_rotary_dim && (binding.rope_rotary_dim > binding.rope_head_dim
                    || binding.rope_rotary_dim % 2))
                || !std::isfinite(binding.rope_attention_factor) || binding.rope_attention_factor <= 0.0f
                || !std::isfinite(binding.rope_theta)
                || binding.rope_theta <= 1.0f)
                throw std::logic_error("invalid internal RoPE-table binding");
            const auto rotaryWidth = binding.rope_rotary_dim
                ? binding.rope_rotary_dim : binding.rope_head_dim;
            if (!binding.rope_inverse_frequencies.empty()
                && (binding.rope_inverse_frequencies.size() != rotaryWidth / 2
                    || !std::ranges::all_of(binding.rope_inverse_frequencies,
                        [](float frequency) { return std::isfinite(frequency) && frequency > 0.0f; })))
                throw std::logic_error("invalid explicit RoPE inverse frequencies");
            const std::size_t sequence =
                static_cast<std::size_t>(binding.shape[0]);
            const std::size_t dimensions =
                static_cast<std::size_t>(binding.shape[1]);
            const std::size_t stride =
                static_cast<std::size_t>(std::abs(binding.address_stride));
            const std::size_t lanes =
                static_cast<std::size_t>(program.hardware.mxm_rows);
            if (lanes == 0 || dimensions % lanes != 0)
                throw std::logic_error(
                    "RoPE-table dimensions must be a multiple of MXM rows");
            for (std::size_t token = 0; token < sequence; ++token) {
                for (std::size_t dimension = 0;
                     dimension < dimensions; ++dimension) {
                    const std::size_t frequencyBlock = dimension / lanes;
                    const std::size_t localDimension = dimension % lanes;
                    const std::size_t address =
                        static_cast<std::size_t>(binding.base_row)
                        + (decodeCompact
                            ? token + frequencyBlock
                            : frequencyBlock * sequence + token) * stride;
                    const auto rotary = binding.rope_rotary_dim
                        ? binding.rope_rotary_dim : binding.rope_head_dim;
                    const bool rotated = dimension < rotary / 2;
                    const float inverse = rotated && !binding.rope_inverse_frequencies.empty()
                        ? binding.rope_inverse_frequencies[dimension]
                        : rotated ? 1.0f / std::pow(
                        binding.rope_theta,
                        static_cast<float>(2 * dimension)
                            / rotary) : 0.0f;
                    const float angle = static_cast<float>(token) * inverse;
                    const auto cosine = encode_16bit_float(
                        std::cos(angle) * (rotated ? binding.rope_attention_factor : 1.0f), binding.element_type);
                    const auto sine = encode_16bit_float(
                        std::sin(angle) * (rotated ? binding.rope_attention_factor : 1.0f), binding.element_type);
                    for_each_binding_hemisphere(binding,
                        [&](Hemisphere hemisphere) {
                            write_binding_sram_byte(system_, binding, hemisphere,
                                binding.slices[0], address, localDimension,
                                static_cast<std::uint8_t>(cosine));
                            write_binding_sram_byte(system_, binding, hemisphere,
                                binding.slices[1], address, localDimension,
                                static_cast<std::uint8_t>(cosine >> 8));
                            write_binding_sram_byte(system_, binding, hemisphere,
                                binding.slices[2], address, localDimension,
                                static_cast<std::uint8_t>(sine));
                            write_binding_sram_byte(system_, binding, hemisphere,
                                binding.slices[3], address, localDimension,
                                static_cast<std::uint8_t>(sine >> 8));
                        });
                }
            }
            continue;
        }
        throw std::logic_error("unsupported internal binding initializer");
    }
}

const BinaryBinding& CModelRuntime::find_binding(BindingAccess access, std::size_t index) const
{
    for (const auto& binding : bindings_) {
        if (binding.access == access && binding.index == index) return binding;
    }
    throw std::out_of_range("FTLPU binary does not contain the requested runtime binding");
}

void CModelRuntime::upload_input(std::size_t index, std::span<const std::uint8_t> data)
{
    const auto& binding = find_binding(BindingAccess::Input, index);
    if (binding.paged_weight) {
        if (data.size() != binding.byte_size)
            throw std::invalid_argument(
                "paged input byte size does not match binding");
        paged_weight_data_[binding.index] =
            std::vector<std::uint8_t>(data.begin(), data.end());
        return;
    }
    upload_binding(binding, data);
}

void CModelRuntime::load_ready_weight_pages()
{
    while (next_weight_page_use_ < weight_page_uses_.size()
        && weight_page_uses_[next_weight_page_use_].ready_cycle
            <= executed_cycles_) {
        const auto& use = weight_page_uses_[next_weight_page_use_];
        const auto& binding = find_binding(
            BindingAccess::Input, use.binding_index);
        const auto logical = paged_weight_data_.find(use.binding_index);
        if (logical == paged_weight_data_.end())
            throw std::logic_error(
                "paged weight was not uploaded before its ready cycle");
        if (use.bank >= hardware_.banks_per_slice
            || use.bank >= binding.page_bank_count)
            throw std::logic_error(
                "binary weight-page bank is outside the binding policy");
        if (resolve_weight_page_placement(binding, use.page_index).bank
            != use.bank)
            throw std::logic_error(
                "binary weight-page use disagrees with binding placement");
        const auto image = pack_weight_binding_page(binding,
            use.page_index, logical->second, hardware_);
        for (const auto& segment : image.segments) {
            const auto hemisphere = static_cast<Hemisphere>(
                segment.hemisphere);
            for (std::uint32_t row = 0;
                 row < segment.vector_count; ++row) {
                const std::size_t offset =
                    static_cast<std::size_t>(segment.byte_offset)
                    + static_cast<std::size_t>(row)
                        * hw::kPhysicalVectorBytes;
                for (std::size_t column = 0;
                     column < hw::kPhysicalVectorBytes; ++column)
                    system_.initialize_mem_sram_lane_byte(hemisphere,
                        segment.slice, use.bank,
                        column / hw::kLanesPerTile,
                        segment.base_row + row,
                        column % hw::kLanesPerTile,
                        image.data[offset + column]);
            }
        }
        ++next_weight_page_use_;
    }
}

void CModelRuntime::enable_execution_trace(bool enabled) noexcept
{
    execution_trace_enabled_ = enabled;
}

void CModelRuntime::configure_execution_trace_segment(
    std::int64_t cycleOffset, bool append) noexcept
{
    execution_trace_cycle_offset_ = cycleOffset;
    execution_trace_append_on_load_ = append;
}

void CModelRuntime::record_execution_trace_interval(
    std::int64_t startCycle, std::int64_t endCycle,
    std::string resource, std::string detail, std::size_t issueCount)
{
    if (!execution_trace_enabled_) return;
    execution_trace_.record_interval(startCycle, endCycle,
        std::move(resource), std::move(detail), issueCount);
}

void CModelRuntime::write_execution_trace_csv(
    const std::filesystem::path& path) const
{
    if (!execution_trace_enabled_)
        throw std::logic_error(
            "runtime execution trace was not enabled before program load");
    execution_trace_.write_csv(path);
}

void CModelRuntime::begin_external_execution_trace_segment(
    const BinaryProgram& program, std::int64_t cycleOffset, bool append)
{
    if (execution_trace_enabled_)
        execution_trace_.begin_segment(program, cycleOffset, append);
    if (mem_execution_trace_enabled_)
        mem_execution_trace_.begin_segment(cycleOffset, append);
}

void CModelRuntime::sample_external_execution_trace_cycle(
    std::uint64_t physicalCycle, bool programIssueEnabled)
{
    if (execution_trace_enabled_)
        execution_trace_.sample(
            system_, physicalCycle, programIssueEnabled, nullptr);
    if (mem_execution_trace_enabled_)
        mem_execution_trace_.sample(
            system_, physicalCycle, programIssueEnabled);
}

void CModelRuntime::enable_mem_execution_trace(bool enabled) noexcept
{
    mem_execution_trace_enabled_ = enabled;
}

void CModelRuntime::stream_mem_execution_trace_csv(
    const std::filesystem::path& path)
{
    mem_execution_trace_.stream_csv(path);
    mem_execution_trace_enabled_ = true;
}

void CModelRuntime::write_mem_execution_trace_csv(
    const std::filesystem::path& path) const
{
    if (!mem_execution_trace_enabled_)
        throw std::logic_error(
            "MEM execution trace was not enabled before program load");
    mem_execution_trace_.write_csv(path);
}

void CModelRuntime::upload_binding(
    const BinaryBinding& binding, std::span<const std::uint8_t> data)
{
    if (binding.role == "weight" && binding.rope_rotary_dim
        && binding.rope_rotary_dim < binding.rope_head_dim) {
        auto physical = binding;
        physical.rope_rotary_dim = 0;
        upload_binding(physical, pack_rope_projection_data(binding, data));
        return;
    }
    if ((binding.shape.empty() || binding.shape.size() > 3)
        || binding.base_row < 0 || binding.address_stride == 0
        || binding.slices.empty())
        throw std::logic_error(
            "runtime requires a valid rank-1 through rank-3 physical binding");
    if (data.size() != binding.byte_size)
        throw std::invalid_argument("input byte size or element type does not match binding");
    const bool vector = binding.shape.size() == 1;
    const std::size_t rows =
        vector ? 1 : static_cast<std::size_t>(binding.shape[0]);
    std::size_t columns = static_cast<std::size_t>(
        vector ? binding.shape[0] : binding.shape[1]);
    if (binding.shape.size() == 3)
        columns *= static_cast<std::size_t>(binding.shape[2]);
    const std::size_t row_stride = static_cast<std::size_t>(std::abs(binding.address_stride));
    if (binding.layout == BindingLayout::TokenPartitionPlanar) {
        const std::size_t bytes = binding.element_type == BindingElementType::F32 ? 4 : 2;
        if (binding.slices.size() != bytes || binding.hemisphere_mask != 3 || rows == 0)
            throw std::invalid_argument("invalid token-partitioned planar binding");
        const auto half = (rows + 1) / 2;
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column) {
                const auto hemisphere = row < half ? Hemisphere::West : Hemisphere::East;
                const auto address = static_cast<std::size_t>(binding.base_row)
                    + ((column / hardware_.bytes_per_word) * half + row % half) * row_stride;
                for (std::size_t byte = 0; byte < bytes; ++byte)
                    write_binding_sram_byte(system_, binding, hemisphere, binding.slices[byte],
                        address, column % hardware_.bytes_per_word, data[(row * columns + column) * bytes + byte]);
            }
        return;
    }

    if (binding.layout == BindingLayout::Vector) {
        if (binding.slices.size() != 1)
            throw std::logic_error("vector binding requires exactly one MEM slice");
        for (std::size_t row = 0; row < rows; ++row) {
            const auto address = static_cast<std::size_t>(binding.base_row) + row * row_stride;
            for (std::size_t column = 0; column < columns; ++column)
                for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                    write_binding_sram_byte(system_, binding, hemisphere, binding.slices[0], address,
                        column, data[row * columns + column]);
                });
        }
        return;
    }

    if (binding.layout == BindingLayout::MxmWeightStriped) {
        constexpr std::size_t mxm_k = 320;
        const std::size_t column_blocks =
            (columns + binding.slices.size() - 1) / binding.slices.size();
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t column = 0; column < columns; ++column) {
                const auto slice = binding.slices[column % binding.slices.size()];
                const auto address = static_cast<std::size_t>(binding.base_row)
                    + ((k / mxm_k) * column_blocks
                        + column / binding.slices.size()) * row_stride;
                for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                    write_binding_sram_byte(system_, binding, hemisphere, slice, address, k % mxm_k,
                        data[k * columns + column]);
                });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16MxmActivationPlanar
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 4) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t k = 0; k < columns; ++k) {
                const std::size_t address = static_cast<std::size_t>(binding.base_row)
                    + (k / 32) * rows + row;
                const std::size_t offset = (row * columns + k) * 2;
                for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                    write_binding_sram_byte(system_, binding, hemisphere, binding.slices[0], address,
                        k % 32, data[offset]);
                    write_binding_sram_byte(system_, binding, hemisphere, binding.slices[1], address,
                        k % 32, data[offset + 1]);
                    write_binding_sram_byte(system_, binding, hemisphere, binding.slices[2], address,
                        k % 32, data[offset]);
                    write_binding_sram_byte(system_, binding, hemisphere, binding.slices[3], address,
                        k % 32, data[offset + 1]);
                });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16ProjectionBiasX4
        && is_16bit_float(binding.element_type)
        && (binding.slices.size() == 4 || binding.slices.size() == 8)) {
        const std::size_t blockColumns = hw::kPhysicalVectorBytes;
        const std::size_t pairs = binding.slices.size() / 2;
        if (!vector || columns % blockColumns != 0)
            throw std::logic_error(
                "projection bias requires a physical-row-aligned rank-1 tensor");
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t block = column / blockColumns;
            const std::size_t pair = (block / 2) % pairs;
            const std::size_t address =
                static_cast<std::size_t>(binding.base_row)
                + (block / (2 * pairs)) * 2 + block % 2;
            const std::size_t offset = column * 2;
            for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                write_binding_sram_byte(system_, binding, hemisphere,
                    binding.slices[2 * pair], address, column % blockColumns,
                    data[offset]);
                write_binding_sram_byte(system_, binding, hemisphere,
                    binding.slices[2 * pair + 1], address, column % blockColumns,
                    data[offset + 1]);
            });
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16HemispherePairPlanar
        && is_16bit_float(binding.element_type) && binding.slices.size() == 2
        && (binding.hemisphere_mask & 3) == 3) {
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < columns; ++c) {
                const auto hemisphere = static_cast<Hemisphere>((c / 32) % 2);
                const std::size_t address = binding.base_row + (c / 64) * rows + r;
                for (std::size_t byte = 0; byte < 2; ++byte)
                    write_binding_sram_byte(system_, binding, hemisphere,
                        binding.slices[byte], address, c % 32, data[(r * columns + c) * 2 + byte]);
            }
        return;
    }
    if (binding.layout == BindingLayout::Fp16PairPlanar
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 2) {
        const std::size_t width = hardware_.bytes_per_word;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (column / width) * rows + row;
                const std::size_t offset = (row * columns + column) * 2;
                for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                    write_binding_sram_byte(system_, binding, hemisphere, binding.slices[0],
                        address, column % width, data[offset]);
                    write_binding_sram_byte(system_, binding, hemisphere, binding.slices[1],
                        address, column % width, data[offset + 1]);
                });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16SxmDistributed16
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (vector) {
            if (columns % 32 != 0)
                throw std::logic_error(
                    "distributed16 16-bit float vector requires "
                    "32-aligned length");
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hiddenBlock = column / 32;
                const std::size_t featureWave = (column % 32) / 8;
                const std::size_t featureLane = column % 8;
                const std::size_t offset = column * 2;
                for (std::size_t tokenWave = 0;
                     tokenWave < 4; ++tokenWave) {
                    const std::size_t destinationTile =
                        (tokenWave + 4 - featureWave) % 4;
                    const std::size_t address =
                        static_cast<std::size_t>(binding.base_row)
                        + hiddenBlock * 4 + tokenWave;
                    for_each_binding_hemisphere(
                        binding, [&](Hemisphere hemisphere) {
                            for (std::size_t lane = 0; lane < 8; ++lane) {
                                write_binding_sram_byte(system_, binding, hemisphere,
                                    binding.slices[2 * featureLane],
                                    address, destinationTile * 8 + lane,
                                    data[offset]);
                                write_binding_sram_byte(system_, binding, hemisphere,
                                    binding.slices[2 * featureLane + 1],
                                    address, destinationTile * 8 + lane,
                                    data[offset + 1]);
                            }
                        });
                }
            }
            return;
        }
        if (rows % 32 != 0 || columns % 32 != 0)
            throw std::logic_error(
                "distributed16 16-bit float input requires "
                "32-aligned dimensions");
        const std::size_t hidden_blocks = columns / 32;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / 32;
            const std::size_t token_wave = (row % 32) / 8;
            const std::size_t token_lane = row % 8;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t destination_tile =
                    (token_wave + 4 - feature_wave) % 4;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + token_wave;
                const std::size_t offset = (row * columns + column) * 2;
                for_each_binding_hemisphere(
                    binding, [&](Hemisphere hemisphere) {
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * feature_lane], address,
                            destination_tile * 8 + token_lane,
                            data[offset]);
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * feature_lane + 1], address,
                            destination_tile * 8 + token_lane,
                            data[offset + 1]);
                    });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16VxmDistributed16
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (columns % 32 != 0 || (!vector && rows % 32 != 0))
            throw std::logic_error(
                "VXM distributed16 16-bit float input requires "
                "32-aligned dimensions");
        const std::size_t hidden_blocks = columns / 32;
        const std::size_t stored_rows = vector ? 32 : rows;
        for (std::size_t row = 0; row < stored_rows; ++row) {
            const std::size_t logical_row = vector ? 0 : row;
            const std::size_t token_block = row / 32;
            const std::size_t lane = row % 32;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + feature_wave;
                const std::size_t offset =
                    (logical_row * columns + column) * 2;
                for_each_binding_hemisphere(
                    binding, [&](Hemisphere hemisphere) {
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * feature_lane], address,
                            lane, data[offset]);
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * feature_lane + 1], address,
                            lane, data[offset + 1]);
                    });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16VxmRowParallel8
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (columns == 0 || (!vector && rows % 32 != 0))
            throw std::logic_error(
                "VXM row-parallel input requires 32-aligned rows");
        const std::size_t storedRows = vector ? 32 : rows;
        for (std::size_t row = 0; row < storedRows; ++row) {
            const std::size_t logicalRow = vector ? 0 : row;
            const std::size_t tokenBlock = row / 32;
            const std::size_t pair = vector ? 0 : tokenBlock % 8;
            const std::size_t batch = vector ? 0 : tokenBlock / 8;
            const std::size_t lane = row % 32;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + batch * columns + column;
                const std::size_t offset =
                    (logicalRow * columns + column) * 2;
                for_each_binding_hemisphere(binding,
                    [&](Hemisphere hemisphere) {
                        const std::size_t firstPair = vector ? 0 : pair;
                        const std::size_t lastPair = vector ? 8 : pair + 1;
                        for (std::size_t targetPair = firstPair;
                             targetPair < lastPair; ++targetPair) {
                            write_binding_sram_byte(system_, binding,
                                hemisphere,
                                binding.slices[2 * targetPair], address,
                                lane, data[offset]);
                            write_binding_sram_byte(system_, binding,
                                hemisphere,
                                binding.slices[2 * targetPair + 1], address,
                                lane, data[offset + 1]);
                        }
                    });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16VxmGammaBroadcast
        && is_16bit_float(binding.element_type)
        && (binding.slices.size() == 2 || binding.slices.size() == 4)) {
        if (!vector)
            throw std::logic_error(
                "VXM gamma broadcast input requires a rank-1 tensor");
        const std::size_t segmentRows = binding.instruction_count > 0
            ? binding.instruction_count : hw::kSramDepthRows;
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t offset = column * 2;
            const std::size_t segment = column / segmentRows;
            const std::size_t address =
                static_cast<std::size_t>(binding.base_row)
                + column % segmentRows;
            if (2 * segment + 1 >= binding.slices.size())
                throw std::logic_error("segmented gamma exceeds binding slices");
            for_each_binding_hemisphere(binding,
                [&](Hemisphere hemisphere) {
                    for (std::size_t lane = 0;
                         lane < hw::kPhysicalVectorBytes; ++lane) {
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * segment], address, lane, data[offset]);
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * segment + 1], address, lane,
                            data[offset + 1]);
                    }
                });
        }
        return;
    }
    if ((binding.layout == BindingLayout::Fp16MxmDistributed16
            || binding.layout
                == BindingLayout::Fp16MxmBlock8Distributed16)
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (rows == 0 || columns % hardware_.bytes_per_word != 0)
            throw std::logic_error(
                "MXM distributed16 16-bit float input requires "
                "non-zero rows and physical-vector-aligned columns");
        const std::size_t hidden_blocks = columns / hardware_.bytes_per_word;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / hardware_.mxm_rows;
            const std::size_t token_wave = (row % hardware_.mxm_rows) / hardware_.lanes_per_tile;
            const std::size_t token_lane = row % hardware_.lanes_per_tile;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / hardware_.bytes_per_word;
                const std::size_t feature_lane = column % hardware_.bytes_per_word;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * hardware_.tile_rows
                    + token_wave;
                const std::size_t offset = (row * columns + column) * 2;
                const auto writeHemisphere = [&](Hemisphere hemisphere) {
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * token_lane], address,
                            feature_lane, data[offset]);
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * token_lane + 1], address,
                            feature_lane, data[offset + 1]);
                    };
                if (binding.layout
                    == BindingLayout::Fp16MxmBlock8Distributed16) {
                    const auto hemisphere = static_cast<Hemisphere>(
                        (hidden_block % (hw::kHemispheres
                            * loaded_mxms_per_hemisphere_))
                        / loaded_mxms_per_hemisphere_);
                    writeHemisphere(hemisphere);
                } else {
                    for_each_binding_hemisphere(binding, writeHemisphere);
                }
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::W8A16Native4Weight
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == 32) {
        if (rows % 32 || columns % 64)
            throw std::logic_error(
                "Native4 weight binding must be K32/N64 aligned");
        const std::size_t reductions = rows / 32;
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t n = 0; n < columns; ++n) {
                const auto hemisphere = static_cast<Hemisphere>(
                    (n / 32) % 2);
                const std::size_t localWave = n / 64;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + localWave * reductions + k / 32;
                write_binding_sram_byte(system_, binding, hemisphere,
                    binding.slices[n % 32], address, k % 32,
                    data[k * columns + n]);
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp32QuadPlanar
        && binding.element_type == BindingElementType::F32
        && binding.slices.size() == 4) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (column / hardware_.bytes_per_word) * rows + row;
                const std::size_t offset = (row * columns + column) * 4;
                for_each_binding_hemisphere(binding,
                    [&](Hemisphere hemisphere) {
                        for (std::size_t byte = 0; byte < 4; ++byte)
                            write_binding_sram_byte(system_, binding,
                                hemisphere, binding.slices[byte], address,
                                column % hardware_.bytes_per_word, data[offset + byte]);
                    });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16ScalarBroadcastPlanar
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 2) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + row * columns + column;
                const std::size_t offset = (row * columns + column) * 2;
                for_each_binding_hemisphere(binding,
                    [&](Hemisphere hemisphere) {
                        for (std::size_t lane = 0; lane < hardware_.bytes_per_word; ++lane) {
                            write_binding_sram_byte(system_, binding,
                                hemisphere, binding.slices[0], address,
                                lane, data[offset]);
                            write_binding_sram_byte(system_, binding,
                                hemisphere, binding.slices[1], address,
                                lane, data[offset + 1]);
                        }
                    });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp32ScalarBroadcastPlanar
        && binding.element_type == BindingElementType::F32
        && binding.slices.size() == 4) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + row * columns + column;
                const std::size_t offset = (row * columns + column) * 4;
                for_each_binding_hemisphere(binding,
                    [&](Hemisphere hemisphere) {
                        for (std::size_t byte = 0; byte < 4; ++byte)
                            for (std::size_t lane = 0; lane < hardware_.bytes_per_word; ++lane)
                                write_binding_sram_byte(system_, binding,
                                    hemisphere, binding.slices[byte],
                                    address, lane, data[offset + byte]);
                    });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16DepthwiseKernel4
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 8) {
        if (rows == 0 || columns != 4)
            throw std::logic_error(
                "depthwise kernel-4 input requires shape [channels, 4]");
        const std::size_t channelBlocks = (rows + 31) / 32;
        for (std::size_t channel = 0; channel < rows; ++channel) {
            const std::size_t block = channel / 32;
            const std::size_t lane = channel % 32;
            for (std::size_t tap = 0; tap < 4; ++tap) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + tap * channelBlocks + block;
                const std::size_t offset = (channel * 4 + tap) * 2;
                for_each_binding_hemisphere(binding,
                    [&](Hemisphere hemisphere) {
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * tap], address, lane,
                            data[offset]);
                        write_binding_sram_byte(system_, binding, hemisphere,
                            binding.slices[2 * tap + 1], address, lane,
                            data[offset + 1]);
                    });
            }
        }
        return;
    }
    if ((binding.layout == BindingLayout::W8A16MxmWeightStriped
            || binding.layout == BindingLayout::W8A16MxmWeightWaveStriped)
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == 8) {
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t local = n % 32;
                const std::size_t pulse = 3 - local / 8;
                const std::size_t stream = local % 8;
                std::size_t address = static_cast<std::size_t>(binding.base_row);
                Hemisphere hemisphere = Hemisphere::East;
                if (binding.layout == BindingLayout::W8A16MxmWeightWaveStriped) {
                    hemisphere = static_cast<Hemisphere>((n / 64) % 2);
                    address += ((n / 128) * (rows / 32) + k / 32) * 8
                        + ((n % 64) / 32) * 4 + pulse;
                } else {
                    hemisphere = ((n / 32) % 2) == 0
                        ? Hemisphere::East : Hemisphere::West;
                    address += ((n / 64) * (rows / 32) + k / 32) * 4 + pulse;
                }
                write_binding_sram_byte(system_, binding, hemisphere, binding.slices[stream], address,
                    k % 32, data[k * columns + n]);
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::W8A16MxmWeightReplicated
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == 8) {
        if (rows % 32 != 0 || columns % 32 != 0)
            throw std::logic_error(
                "replicated W8A16 weight must be K32/N32 aligned");
        const std::size_t reductionBlocks = rows / 32;
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t local = n % 32;
                const std::size_t pulse = 3 - local / 8;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + ((n / 32) * reductionBlocks + k / 32) * 4
                    + pulse;
                for_each_binding_hemisphere(binding,
                    [&](Hemisphere hemisphere) {
                        write_binding_sram_byte(system_, binding,
                            hemisphere, binding.slices[local % 8],
                            address, k % 32,
                            data[k * columns + n]);
                    });
            }
        }
        return;
    }
    if (binding.layout
            == BindingLayout::W8A16Block8WeightWaveStriped
        && binding.element_type == BindingElementType::I8
        && binding.slices.size()
            == loaded_mxms_per_hemisphere_ * 8) {
        const std::size_t columnsPerHemisphere =
            loaded_mxms_per_hemisphere_ * 32;
        const std::size_t columnsPerWave =
            hw::kHemispheres * columnsPerHemisphere;
        if (rows % 32 != 0 || columns % columnsPerWave != 0)
            throw std::logic_error(
                "Block8 weight binding must align to the physical MXM wave");
        const std::size_t reductionBlocks = rows / 32;
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t localColumn = n % 32;
                const std::size_t pulse = 3 - localColumn / 8;
                const std::size_t localMxm =
                    (n % columnsPerHemisphere) / 32;
                const std::size_t stream = localColumn % 8;
                const auto hemisphere =
                    static_cast<Hemisphere>(
                        (n / columnsPerHemisphere) % hw::kHemispheres);
                const std::size_t wave = n / columnsPerWave;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (wave * reductionBlocks + k / 32) * 4
                    + pulse;
                write_binding_sram_byte(system_, binding, hemisphere,
                    binding.slices[localMxm * 8 + stream],
                    address, k % 32, data[k * columns + n]);
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::W8A16AttentionWeightStriped
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == 8) {
        if (rows % 32 || columns % 64)
            throw std::logic_error("attention weight binding must be K32/N64 aligned");
        const std::size_t reduction_blocks = rows / 32;
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t local_column = n % 32;
                const std::size_t pulse = 3 - local_column / 8;
                const std::size_t stream = local_column % 8;
                const auto hemisphere = static_cast<Hemisphere>((n / 64) % 2);
                const std::size_t head_group = n / 128;
                const std::size_t local_mxm = (n % 64) / 32;
                const std::size_t address = static_cast<std::size_t>(binding.base_row)
                    + (head_group * reduction_blocks + k / 32) * 8
                    + local_mxm * 4 + pulse;
                write_binding_sram_byte(system_, binding, hemisphere, binding.slices[stream], address,
                    k % 32, data[k * columns + n]);
            }
        }
        return;
    }
    throw std::logic_error("unsupported input binding layout");
}

std::vector<std::uint8_t> CModelRuntime::download_output(std::size_t index) const
{
    return download_binding(find_binding(BindingAccess::Output, index));
}

std::vector<std::uint8_t> CModelRuntime::download_binding(
    const BinaryBinding& binding) const
{
    require_matrix_binding(binding);
    const std::size_t rows = static_cast<std::size_t>(binding.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(binding.shape[1]);
    const std::size_t row_stride = static_cast<std::size_t>(std::abs(binding.address_stride));
    if (binding.layout == BindingLayout::TokenPartitionPlanar) {
        const std::size_t bytes = binding.element_type == BindingElementType::F32 ? 4 : 2;
        if (binding.slices.size() != bytes || binding.hemisphere_mask != 3 || rows == 0)
            throw std::invalid_argument("invalid token-partitioned planar binding");
        const auto half = (rows + 1) / 2;
        std::vector<std::uint8_t> result(binding.byte_size);
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column) {
                const auto hemisphere = row < half ? Hemisphere::West : Hemisphere::East;
                const auto address = static_cast<std::size_t>(binding.base_row)
                    + ((column / hardware_.bytes_per_word) * half + row % half) * row_stride;
                for (std::size_t byte = 0; byte < bytes; ++byte)
                    result[(row * columns + column) * bytes + byte] = read_binding_sram_byte(
                        system_, binding, hemisphere, binding.slices[byte], address, column % hardware_.bytes_per_word);
            }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16MxmActivationPlanar
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 4) {
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (column / 32) * rows + row;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_binding_sram_byte(system_, binding, Hemisphere::East,
                    binding.slices[0], address, column % 32);
                result[offset + 1] = read_binding_sram_byte(system_, binding, Hemisphere::East,
                    binding.slices[1], address, column % 32);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Vector
        && binding.element_type == BindingElementType::I8 && binding.slices.size() == 1) {
        std::vector<std::uint8_t> result(static_cast<std::size_t>(binding.byte_size));
        for (std::size_t row = 0; row < rows; ++row) {
            const auto address = static_cast<std::size_t>(binding.base_row) + row * row_stride;
            for (std::size_t column = 0; column < columns; ++column)
                result[row * columns + column] =
                    read_binding_sram_byte(system_, binding, Hemisphere::East,
                        binding.slices[0], address, column);
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16HemispherePairPlanar
        && is_16bit_float(binding.element_type) && binding.slices.size() == 2
        && (binding.hemisphere_mask & 3) == 3) {
        std::vector<std::uint8_t> result(static_cast<std::size_t>(binding.byte_size));
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < columns; ++c) {
                const auto hemisphere = static_cast<Hemisphere>((c / 32) % 2);
                const std::size_t address = binding.base_row + (c / 64) * rows + r;
                for (std::size_t byte = 0; byte < 2; ++byte)
                    result[(r * columns + c) * 2 + byte] = read_binding_sram_byte(
                        system_, binding, hemisphere, binding.slices[byte], address, c % 32);
            }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16PairPlanar
        && is_16bit_float(binding.element_type)
        && (binding.slices.size() == 2 || binding.slices.size() == 4)) {
        const std::size_t width = hardware_.bytes_per_word;
        std::vector<std::uint8_t> result(static_cast<std::size_t>(binding.byte_size));
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t local_mxm = binding.slices.size() == 2
                    ? 0 : (column % (2 * width)) / width;
                const bool dual_hemisphere = binding.slices.size() == 4
                    && (binding.hemisphere_mask & 3) == 3;
                const auto hemisphere = dual_hemisphere
                    ? static_cast<Hemisphere>((column / (2 * width)) % 2)
                    : single_binding_hemisphere(binding);
                const std::size_t columnBlock = binding.slices.size() == 2
                    ? column / width
                    : (dual_hemisphere ? column / (4 * width)
                                       : column / (2 * width));
                const std::size_t address = static_cast<std::size_t>(binding.base_row)
                    + columnBlock * rows + row;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_binding_sram_byte(system_, binding, hemisphere,
                    binding.slices[local_mxm * 2], address, column % width);
                result[offset + 1] = read_binding_sram_byte(system_, binding, hemisphere,
                    binding.slices[local_mxm * 2 + 1], address, column % width);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp32QuadPlanar
        && binding.element_type == BindingElementType::F32
        && binding.slices.size() == 4) {
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        const auto hemisphere = single_binding_hemisphere(binding);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (column / hardware_.bytes_per_word) * rows + row;
                const std::size_t offset = (row * columns + column) * 4;
                for (std::size_t byte = 0; byte < 4; ++byte)
                    result[offset + byte] = read_binding_sram_byte(
                        system_, binding, hemisphere,
                        binding.slices[byte], address, column % hardware_.bytes_per_word);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16ScalarBroadcastPlanar
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 2) {
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        const auto hemisphere = single_binding_hemisphere(binding);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + row * columns + column;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_binding_sram_byte(system_, binding,
                    hemisphere, binding.slices[0], address, 0);
                result[offset + 1] = read_binding_sram_byte(system_, binding,
                    hemisphere, binding.slices[1], address, 0);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp32ScalarBroadcastPlanar
        && binding.element_type == BindingElementType::F32
        && binding.slices.size() == 4) {
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        const auto hemisphere = single_binding_hemisphere(binding);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + row * columns + column;
                const std::size_t offset = (row * columns + column) * 4;
                for (std::size_t byte = 0; byte < 4; ++byte)
                    result[offset + byte] = read_binding_sram_byte(
                        system_, binding, hemisphere,
                        binding.slices[byte], address, 0);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16SxmDistributed16
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (rows % 32 != 0 || columns % 32 != 0)
            throw std::logic_error(
                "distributed16 16-bit float output requires "
                "32-aligned dimensions");
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        const std::size_t hidden_blocks = columns / 32;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / 32;
            const std::size_t token_wave = (row % 32) / 8;
            const std::size_t token_lane = row % 8;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t source_tile =
                    (token_wave + 4 - feature_wave) % 4;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + token_wave;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_binding_sram_byte(system_, binding,
                    Hemisphere::East,
                    binding.slices[2 * feature_lane], address,
                    source_tile * 8 + token_lane);
                result[offset + 1] = read_binding_sram_byte(system_, binding,
                    Hemisphere::East,
                    binding.slices[2 * feature_lane + 1], address,
                    source_tile * 8 + token_lane);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16VxmDistributed16
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (rows % 32 != 0 || columns % 32 != 0)
            throw std::logic_error(
                "VXM distributed16 16-bit float output requires "
                "32-aligned dimensions");
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        const std::size_t hidden_blocks = columns / 32;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / 32;
            const std::size_t lane = row % 32;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + feature_wave;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_binding_sram_byte(system_, binding, Hemisphere::East,
                    binding.slices[2 * feature_lane], address, lane);
                result[offset + 1] = read_binding_sram_byte(system_, binding, Hemisphere::East,
                    binding.slices[2 * feature_lane + 1], address, lane);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16VxmRowParallel8
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (rows % 32 != 0 || columns == 0)
            throw std::logic_error(
                "VXM row-parallel output requires 32-aligned rows");
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t tokenBlock = row / 32;
            const std::size_t pair = tokenBlock % 8;
            const std::size_t batch = tokenBlock / 8;
            const std::size_t lane = row % 32;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + batch * columns + column;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_binding_sram_byte(system_, binding,
                    Hemisphere::East, binding.slices[2 * pair], address,
                    lane);
                result[offset + 1] = read_binding_sram_byte(system_, binding,
                    Hemisphere::East, binding.slices[2 * pair + 1], address,
                    lane);
            }
        }
        return result;
    }
    if ((binding.layout == BindingLayout::Fp16MxmDistributed16
            || binding.layout
                == BindingLayout::Fp16MxmBlock8Distributed16)
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (rows == 0 || columns % hardware_.bytes_per_word != 0)
            throw std::logic_error(
                "MXM distributed16 16-bit float output requires "
                "non-zero rows and physical-vector-aligned columns");
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        const std::size_t hidden_blocks = columns / hardware_.bytes_per_word;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / hardware_.mxm_rows;
            const std::size_t token_wave = (row % hardware_.mxm_rows) / hardware_.lanes_per_tile;
            const std::size_t token_lane = row % hardware_.lanes_per_tile;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / hardware_.bytes_per_word;
                const std::size_t feature_lane = column % hardware_.bytes_per_word;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * hardware_.tile_rows
                    + token_wave;
                const std::size_t offset = (row * columns + column) * 2;
                const auto hemisphere = binding.layout
                        == BindingLayout::Fp16MxmBlock8Distributed16
                    ? static_cast<Hemisphere>(
                        (hidden_block
                            % (hw::kHemispheres
                                * loaded_mxms_per_hemisphere_))
                        / loaded_mxms_per_hemisphere_)
                    : Hemisphere::East;
                result[offset] = read_binding_sram_byte(system_, binding, hemisphere,
                    binding.slices[2 * token_lane], address,
                    feature_lane);
                result[offset + 1] = read_binding_sram_byte(system_, binding, hemisphere,
                    binding.slices[2 * token_lane + 1], address,
                    feature_lane);
            }
        }
        return result;
    }
    if (binding.layout != BindingLayout::Int32BytePlanar
        || binding.element_type != BindingElementType::I32 || binding.slices.size() != 4)
        throw std::logic_error("unsupported output binding layout");
    std::vector<std::uint8_t> result(static_cast<std::size_t>(binding.byte_size));
    for (std::size_t row = 0; row < rows; ++row) {
        const auto address = static_cast<std::size_t>(binding.base_row) + row * row_stride;
        for (std::size_t column = 0; column < columns; ++column) {
            const auto offset = (row * columns + column) * 4;
            for (std::size_t byte = 0; byte < 4; ++byte)
                result[offset + byte] = read_binding_sram_byte(
                    system_, binding, Hemisphere::East, binding.slices[byte], address, column);
        }
    }
    return result;
}

void CModelRuntime::copy_binding(
    const BinaryBinding& source, const BinaryBinding& destination)
{
    require_matrix_binding(source);
    require_matrix_binding(destination);
    if (!is_16bit_float(source.element_type)
        || source.element_type != destination.element_type
        || source.shape != destination.shape
        || source.byte_size != destination.byte_size)
        throw std::invalid_argument(
            "device binding copy requires matching 16-bit float tensors");
    if (source.layout == destination.layout
        && source.bank == destination.bank
        && source.base_row == destination.base_row
        && source.slices == destination.slices
        && source.hemisphere_mask == destination.hemisphere_mask)
        return;

    const std::size_t rows = static_cast<std::size_t>(source.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(source.shape[1]);
    std::vector<std::uint16_t> staging(rows * columns);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            staging[row * columns + column] =
                read_16bit_element(system_, source, row, column,
                    loaded_mxms_per_hemisphere_, hardware_);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            write_16bit_element(system_, destination, row, column,
                staging[row * columns + column], hardware_);
}

void CModelRuntime::load_file(const std::filesystem::path& path)
{
    load(read_binary_program(path));
}

void CModelRuntime::configure_common_transport(const BinaryProgram& program)
{
    common_transports_.clear();
    if (!program.common_barrier_protocol) return;
    std::map<std::uint32_t, std::size_t> owners;
    for (const auto& barrier : program.barriers) {
        const auto event = std::find_if(program.barrier_events.begin(), program.barrier_events.end(),
            [&](const auto& e) { return e.event_id == barrier.event_id; });
        for (const auto instance : event->page_use_indices) {
            const auto launch = program.weight_page_uses.at(instance).transport_launch_event_id;
            if (launch == 0) throw std::invalid_argument("common page has no final transport launch association");
            auto [it, inserted] = owners.emplace(launch, common_transports_.size());
            if (inserted) common_transports_.push_back(CommonTransportPage{launch, barrier.stop_cycle});
            auto& page = common_transports_[it->second];
            if (page.stop_cycle != barrier.stop_cycle)
                throw std::invalid_argument("one common transport launch belongs to different barriers");
            page.instances.push_back(instance);
        }
    }
    // A compiler candidate is deliberately serializable, but is not executable
    // until every runtime residency instance has a real transport association.
    const auto pagePlans = plan_weight_prefetches(program);
    for (std::size_t planIndex = 0; planIndex < pagePlans.size(); ++planIndex) {
        const auto& plan = pagePlans[planIndex];
        if (plan.pre_execution) continue;
        std::uint64_t reuseCycle = 0;
        for (std::size_t previous = 0; previous < planIndex; ++previous)
            if (weight_prefetch_plans_overlap(plan, pagePlans[previous]))
                reuseCycle = std::max(reuseCycle, pagePlans[previous].release_cycle);
        for (const auto instance : plan.use_indices) {
            const auto launch = program.weight_page_uses.at(instance).transport_launch_event_id;
            if (!owners.contains(launch)
                || std::find(common_transports_[owners.at(launch)].instances.begin(),
                    common_transports_[owners.at(launch)].instances.end(), instance)
                    == common_transports_[owners.at(launch)].instances.end())
                throw std::invalid_argument("common paging control table is not fully linked before launch");
            if (common_transports_[owners.at(launch)].stop_cycle < reuseCycle)
                throw std::invalid_argument("common page load precedes physical SRAM reuse or the preceding page's release");
        }
    }
    const auto packetLaunches = validate_common_transport_packet_links(program);
    const auto pageLoads = build_common_page_load_ranges(program);
    struct Segment {
        std::size_t queue, address, count, target;
        std::uint32_t launch;
        bool assigned{false};
    };
    std::vector<Segment> segments;
    for (const auto& queue : program.transport_queues) {
        std::size_t target = 0;
        for (std::size_t pc = 0; pc < queue.commands.size();) {
            if (!is_mem_synchronized_raw_packet_header(queue.commands[pc])) { ++pc; continue; }
            const auto packet = decode_mem_synchronized_icu_packet(queue, pc);
            const auto& header = packet[0].lanes;
            const auto& native = packet[1].lanes;
            const auto bits = ((static_cast<isa::EncodedMemInstruction>(native[0])
                | (static_cast<isa::EncodedMemInstruction>(native[1]) << 32)) >> 2)
                | (static_cast<isa::EncodedMemInstruction>(native[2]) << 62);
            const auto instruction = isa::decode_mem_instruction(bits);
            if (instruction.opcode != MemOpcode::Write)
                throw std::invalid_argument("common page-in contains a non-write transport packet");
            const auto count = ((header[0] >> 2) & 0xffffU) + 1;
            target += count;
            segments.push_back({queue.index, instruction.address, count, target,
                packetLaunches.at({queue.index, pc})});
            pc += InstructionControlUnit::MemIcu::synchronized_packet_word_count;
        }
    }
    std::unordered_map<std::uint32_t, std::vector<std::uint8_t>> layoutData;
    for (auto& page : common_transports_) {
        std::map<std::size_t, std::size_t> targets;
        for (const auto instance : page.instances) {
            const auto& use = program.weight_page_uses.at(instance);
            const auto binding = std::find_if(program.bindings.begin(), program.bindings.end(),
                [&](const auto& b) { return b.access == BindingAccess::Input && b.index == use.binding_index; });
            if (binding == program.bindings.end())
                throw std::invalid_argument("common transport has no weight binding");
            auto [data, inserted] = layoutData.try_emplace(use.binding_index);
            if (inserted) data->second.resize(static_cast<std::size_t>(binding->byte_size), 0);
            const auto image = pack_weight_binding_page(*binding, use.page_index, data->second, program.hardware);
            for (const auto& expected : image.segments) {
                const auto queue = InstructionControlUnit::mem_queue(
                    static_cast<Hemisphere>(expected.hemisphere), expected.slice, use.bank);
                const std::array<std::uint32_t, 1> ownerBinding {use.binding_index};
                validate_common_page_storage(program, page.stop_cycle, queue,
                    expected.base_row, expected.vector_count, ownerBinding, use.release_cycle, pageLoads);
                auto found = std::find_if(segments.begin(), segments.end(), [&](const auto& actual) {
                    return !actual.assigned && actual.launch == page.launch_event
                        && actual.queue == queue && actual.address == expected.base_row && actual.count == expected.vector_count;
                });
                if (found == segments.end())
                    throw std::invalid_argument("common page dependency has no matching physical MEM commit packet");
                found->assigned = true;
                targets[queue] = std::max(targets[queue], found->target);
                page.hemisphere_mask |= static_cast<std::uint16_t>(1U << expected.hemisphere);
            }
        }
        if (targets.empty()) throw std::invalid_argument("common transport page has no SRAM writes");
        page.mem_issue_targets.assign(targets.begin(), targets.end());
    }
    if (std::any_of(segments.begin(), segments.end(), [](const auto& s) { return !s.assigned; }))
        throw std::invalid_argument("common page-in MEM packet has no linked completion instance");
    // Preserve the physical DMA/RX queue order, including same-C all-of pages.
    std::vector<std::set<std::size_t>> predecessors(common_transports_.size());
    for (const auto& queue : program.queues) {
        if (queue.kind != QueueKind::C2cDma && queue.kind != QueueKind::C2cRx) continue;
        std::optional<std::size_t> previous;
        std::set<std::uint32_t> seen;
        for (const auto& command : queue.commands) {
            if (!is_icu_control_raw_word_command(command)) continue;
            const auto control = decode_icu_control_raw_word(command);
            if (control.opcode != IcuControlOpcode::WaitEvent) continue;
            const auto launch = static_cast<std::uint32_t>(control.event_tag);
            if (!owners.contains(launch) || !seen.insert(launch).second)
                throw std::invalid_argument("common C2C launch event is orphaned or repeated");
            const auto index = owners.at(launch);
            if (previous) predecessors[index].insert(*previous);
            previous = index;
        }
        for (const auto& page : common_transports_)
            if ((page.hemisphere_mask & (1U << queue.index)) && !seen.contains(page.launch_event))
                throw std::invalid_argument("common page missing physical DMA/RX launch control");
    }
    for (const auto& page : common_transports_)
        for (std::size_t side = 0; side < hw::kHemispheres; ++side)
            if (page.hemisphere_mask & (1U << side))
                for (const auto kind : {QueueKind::C2cDma, QueueKind::C2cRx})
                    if (std::none_of(program.queues.begin(), program.queues.end(), [&](const auto& q) {
                        return q.kind == kind && q.index == side;
                    })) throw std::invalid_argument("common page has no DMA/RX control queue");
    // MEM bank ownership order is part of the same transport dependency
    // graph. An opposite MEM/RX ordering would otherwise deadlock in HOLD.
    for (const auto& queue : program.transport_queues) {
        std::optional<std::size_t> previous;
        for (const auto& [location, launch] : packetLaunches) {
            if (location.first != queue.index) continue;
            const auto index = owners.at(launch);
            if (previous && *previous != index) predecessors[index].insert(*previous);
            previous = index;
        }
    }
    std::vector<CommonTransportPage> ordered;
    std::set<std::size_t> done;
    while (ordered.size() != common_transports_.size()) {
        std::optional<std::size_t> next;
        for (std::size_t i = 0; i < common_transports_.size(); ++i) {
            if (done.contains(i) || !std::all_of(predecessors[i].begin(), predecessors[i].end(),
                [&](auto p) { return done.contains(p); })) continue;
            if (!next || std::tie(common_transports_[i].stop_cycle, common_transports_[i].launch_event)
                < std::tie(common_transports_[*next].stop_cycle, common_transports_[*next].launch_event)) next = i;
        }
        if (!next || (!ordered.empty() && ordered.back().stop_cycle > common_transports_[*next].stop_cycle))
            throw std::invalid_argument("common C2C transport order depends on a future frozen cycle");
        done.insert(*next);
        ordered.push_back(std::move(common_transports_[*next]));
    }
    common_transports_ = std::move(ordered);
}

bool CModelRuntime::common_transport_launched(std::uint32_t launch) const
{
    const auto page = std::find_if(common_transports_.begin(), common_transports_.end(),
        [&](const auto& p) { return p.launch_event == launch; });
    return page != common_transports_.end() && page->launched;
}

bool CModelRuntime::common_transport_committed(std::uint32_t launch) const
{
    const auto page = std::find_if(common_transports_.begin(), common_transports_.end(),
        [&](const auto& p) { return p.launch_event == launch; });
    return page != common_transports_.end() && page->committed;
}

void CModelRuntime::advance_common_transport()
{
    if (!common_barrier_protocol_) return;
    for (auto& page : common_transports_) {
        if (page.committed) continue;
        if (!page.launched) {
            if (executed_cycles_ < page.stop_cycle) return;
            for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
                if (!(page.hemisphere_mask & (1U << side))) continue;
                const auto hemisphere = static_cast<Hemisphere>(side);
                system_.icu().notify_tagged(IcuLocation::C2cDma(hemisphere), page.launch_event);
                system_.icu().notify_tagged(IcuLocation::C2cRx(hemisphere), page.launch_event);
            }
            page.launched = true;
            return;
        }
        for (const auto& [queue, target] : page.mem_issue_targets)
            if (system_.icu().mem_transport_iq(queue).synchronized_issued_count() < target) return;
        if (!page.writes_issued_cycle) page.writes_issued_cycle = physical_cycles_;
        if (physical_cycles_ - *page.writes_issued_cycle + 1 < hw::kTileRows
            || !system_.common_barrier_resources_drained()) return;
        // All requested vectors have reached the final tile and transient
        // transport data has drained before the next compute-enabled edge.
        for (const auto instance : page.instances)
            notify_common_page_commit(execution_generation_, instance);
        page.committed = true;
    }
}

void CModelRuntime::notify_common_page_commit(std::uint32_t instance)
{
    if (!common_barrier_protocol_)
        throw std::logic_error("common commit notification without a common-barrier task");
    notify_common_page_commit(execution_generation_, instance);
}

void CModelRuntime::notify_common_page_commit(std::uint64_t generation, std::uint32_t instance)
{
    system_.icu().notify_common_mem_commit(generation, instance);
}

void CModelRuntime::run_logical_cycles(
    std::size_t cycles, TspSliceSystem::LogSinks sinks)
{
    sinks.capture_mem_trace = mem_execution_trace_enabled_;
    const auto count = cycles == 0 ? loaded_max_cycle_ + 1 : cycles;
    system_.icu().set_program_issue_enabled(true);
    std::size_t heldCycles = 0;
    constexpr std::size_t kPhysicalWaitWatchdog = 10'000'000;
    for (std::size_t advanced = 0; advanced < count;) {
        // P advances on every tick; L advances only on an enabled dispatch.
        load_ready_weight_pages();
        advance_common_transport();
        tick_(sinks);
        const bool enabled = !common_barrier_protocol_ || system_.icu().compute_enabled();
        if (execution_trace_enabled_)
            execution_trace_.sample(
                system_, physical_cycles_, enabled, nullptr);
        if (mem_execution_trace_enabled_)
            mem_execution_trace_.sample(system_, physical_cycles_, enabled);
        datapath_performance_.sample(
            system_, loaded_mxms_per_hemisphere_, loaded_vxm_alus_);
        ++physical_cycles_;
        advance_common_transport();
        if (enabled) {
            ++executed_cycles_;
            ++advanced;
            heldCycles = 0;
        } else if (++heldCycles > kPhysicalWaitWatchdog) {
            throw std::runtime_error("common barrier physical wait timed out before SRAM commit");
        }
    }
}

void CModelRuntime::dispatch_icu_cycles(std::size_t cycles, std::ostream* log)
{
    run_logical_cycles(cycles, log != nullptr
        ? TspSliceSystem::LogSinks {log, log, log, log, log}
        : TspSliceSystem::LogSinks {});
}

void CModelRuntime::run_cycles(std::size_t cycles, std::ostream* log)
{
    auto sinks = TspSliceSystem::LogSinks {};
    if (log != nullptr) {
        sinks = TspSliceSystem::LogSinks {log, log, log, log, log};
        sinks.sxm = log;
    }
    try {
        run_logical_cycles(cycles, sinks);
    } catch (const std::exception& ex) {
        std::ostringstream message;
        message << "CModel global_cycle=" << system_.cycle()
                << " logical_cycle=" << executed_cycles_
                << " physical_cycle=" << physical_cycles_
                << ": " << ex.what();
        throw std::logic_error(message.str());
    }
}

IcuFrontendStatistics CModelRuntime::icu_frontend_statistics() const noexcept
{
    return system_.icu().frontend_statistics();
}

std::vector<IcuMacroQueueFrontendStatistics>
CModelRuntime::icu_macro_queue_frontend_statistics() const
{
    return system_.icu().macro_queue_frontend_statistics();
}

void CModelRuntime::print_icu_frontend_performance(std::ostream& os) const
{
    const auto statistics = icu_frontend_statistics();
    os << "runtime perf resource=ICU.frontend"
       << " imem_entries=" << statistics.imem_entries
       << " fetched_entries=" << statistics.fetched_entries
       << " issued_instructions=" << statistics.issued_instructions
       << " macro_queues=" << statistics.macro_queues
       << " peak_macro_contexts_per_queue="
       << statistics.peak_macro_contexts_per_queue
       << " decoded_macro_contexts="
       << statistics.decoded_macro_contexts
       << " macro_decoder_bit_wait_cycles="
       << statistics.macro_decoder_bit_wait_cycles
       << " macro_decoder_context_stall_cycles="
       << statistics.macro_decoder_context_stall_cycles
       << " peak_macro_reservoir_bits="
       << statistics.peak_macro_reservoir_bits
       << " macro_frontend_cycles="
       << statistics.macro_frontend_cycles
       << " macro_fetched_words="
       << statistics.macro_fetched_words
       << " macro_payload_bits_consumed="
       << statistics.macro_payload_bits_consumed
       << " macro_reservoir_empty_cycles="
       << statistics.macro_reservoir_empty_cycles
       << " macro_reservoir_full_cycles="
       << statistics.macro_reservoir_full_cycles
       << " macro_reservoir_occupancy_bit_samples="
       << statistics.macro_reservoir_occupancy_bit_samples
       << " macro_descriptor_count="
       << statistics.macro_descriptor_count
       << " macro_mem_descriptor_count="
       << statistics.macro_mem_descriptor_count
       << " macro_mxm_descriptor_count="
       << statistics.macro_mxm_descriptor_count
       << " macro_descriptor_decode_cycles="
       << statistics.macro_descriptor_decode_cycles
       << " macro_max_descriptor_decode_cycles="
       << statistics.macro_max_descriptor_decode_cycles
       << " macro_decoder_active_cycles="
       << statistics.macro_decoder_active_cycles
       << " macro_decoder_starvation_cycles="
       << statistics.macro_decoder_starvation_cycles
       << " macro_decoder_ddb_stall_cycles="
       << statistics.macro_decoder_ddb_stall_cycles
       << " macro_ddb_entries=" << statistics.macro_ddb_entries
       << " peak_macro_ddb_occupancy="
       << statistics.peak_macro_ddb_occupancy
       << " macro_ddb_full_cycles="
       << statistics.macro_ddb_full_cycles
       << " macro_ddb_residency_cycles="
       << statistics.macro_ddb_residency_cycles
       << " macro_ddb_occupancy_samples="
       << statistics.macro_ddb_occupancy_samples
       << " macro_max_ddb_residency_cycles="
       << statistics.macro_max_ddb_residency_cycles
       << " macro_contexts_generated="
       << statistics.macro_contexts_generated
       << " macro_timing_window_block_cycles="
       << statistics.macro_timing_window_block_cycles
       << " macro_active_ram_full_cycles="
       << statistics.macro_active_ram_full_cycles
       << " macro_capacity_feasible_wait_cycles="
       << statistics.macro_capacity_feasible_wait_cycles
       << " macro_admission_count="
       << statistics.macro_admission_count
       << " macro_admission_stall_cycles="
       << statistics.macro_admission_stall_cycles
       << " macro_admission_to_start_cycles="
       << statistics.macro_admission_to_start_cycles
       << " macro_max_admission_to_start_cycles="
       << statistics.macro_max_admission_to_start_cycles
       << " macro_active_occupancy_samples="
       << statistics.macro_active_occupancy_samples
       << " macro_context_active_lifetime_cycles="
       << statistics.macro_context_active_lifetime_cycles
       << " macro_max_context_active_lifetime_cycles="
       << statistics.macro_max_context_active_lifetime_cycles
       << " instruction_prefill_cycles="
       << instruction_prefill_cycles_ << '\n';

    for (const auto& queue : icu_macro_queue_frontend_statistics()) {
        const auto& decoder = queue.decoder;
        os << "runtime perf resource=ICU.macro_frontend.queue"
           << " queue_kind=" << macro_queue_kind_name(queue.kind)
           << " queue_index=" << queue.queue_index
           << " imem_entries=" << queue.imem_entries
           << " fetched_entries=" << queue.fetched_entries
           << " issued_instructions=" << queue.issued_instructions
           << " peak_active_contexts=" << queue.peak_active_contexts
           << " active_context_capacity=" << queue.active_context_capacity
           << " active_context_bits=" << queue.active_context_bits
           << " instruction_bits=" << queue.instruction_bits
           << " imem_read_latency=" << queue.imem_read_latency
           << " imem_request_ii="
           << queue.imem_request_initiation_interval
           << " reservoir_words=" << queue.reservoir_words
           << " decode_window_bits=" << queue.decode_window_bits
           << " ddb_depth=" << queue.ddb_depth
           << " ddb_run_capacity=" << queue.ddb_run_capacity
           << " ddb_entry_bits=" << queue.ddb_entry_bits
           << " admission_lookahead=" << queue.admission_lookahead
           << " context_expand_width=" << queue.context_expand_width
           << " frontend_cycles=" << decoder.cycles
           << " fetched_words=" << decoder.fetched_words
           << " decoded_contexts=" << decoder.decoded_contexts
           << " bit_wait_cycles=" << decoder.bit_wait_cycles
           << " context_stall_cycles=" << decoder.context_stall_cycles
           << " payload_bits_consumed="
           << decoder.payload_bits_consumed
           << " reservoir_empty_cycles="
           << decoder.reservoir_empty_cycles
           << " reservoir_full_cycles="
           << decoder.reservoir_full_cycles
           << " reservoir_occupancy_bit_samples="
           << decoder.reservoir_occupancy_bit_samples
           << " peak_reservoir_bits=" << decoder.peak_reservoir_bits
           << " decoder_active_cycles=" << decoder.decoder_active_cycles
           << " decoder_starvation_cycles="
           << decoder.decoder_starvation_cycles
           << " descriptor_count=" << decoder.descriptor_count
           << " descriptor_decode_cycles="
           << decoder.descriptor_decode_cycles
           << " max_descriptor_decode_cycles="
           << decoder.max_descriptor_decode_cycles
           << " decoder_ddb_stall_cycles="
           << decoder.decoder_ddb_stall_cycles
           << " ddb_entries=" << decoder.ddb_entries_committed
           << " ddb_full_cycles=" << decoder.ddb_full_cycles
           << " ddb_occupancy_samples="
           << decoder.ddb_occupancy_samples
           << " peak_ddb_occupancy=" << decoder.peak_ddb_occupancy
           << " ddb_residency_cycles=" << decoder.ddb_residency_cycles
           << " max_ddb_residency_cycles="
           << decoder.max_ddb_residency_cycles
           << " contexts_generated=" << decoder.contexts_generated
           << " timing_window_block_cycles="
           << decoder.timing_window_block_cycles
           << " active_ram_full_cycles="
           << decoder.active_ram_full_cycles
           << " capacity_feasible_wait_cycles="
           << decoder.capacity_feasible_wait_cycles
           << " admission_count=" << decoder.admission_count
           << " admission_stall_cycles="
           << decoder.admission_stall_cycles
           << " admission_to_start_cycles="
           << decoder.admission_to_start_cycles
           << " max_admission_to_start_cycles="
           << decoder.max_admission_to_start_cycles
           << " active_occupancy_samples="
           << decoder.active_occupancy_samples
           << " peak_active_occupancy="
           << decoder.peak_active_occupancy
           << " context_active_lifetime_cycles="
           << decoder.context_active_lifetime_cycles
           << " max_context_active_lifetime_cycles="
           << decoder.max_context_active_lifetime_cycles
           << '\n';
    }
}

void CModelRuntime::print_datapath_performance(std::ostream& os) const
{
    datapath_performance_.print(
        system_, loaded_mxms_per_hemisphere_, loaded_vxm_alus_, os);
}

} // namespace ftlpu::software::runtime
