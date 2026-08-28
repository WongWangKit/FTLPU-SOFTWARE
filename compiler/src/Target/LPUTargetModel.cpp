#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "ftlpu/software/runtime/target_abi.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "mlir/IR/Builders.h"

#include <algorithm>
#include <charconv>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace ftlpu::compiler::target {
namespace {

int64_t divide_ceil(int64_t numerator, int64_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

template <typename Struct>
void read_json_integer(
    const llvm::json::Object* object, llvm::StringRef name, int64_t Struct::*field,
    Struct& value)
{
    if (!object) return;
    if (const auto integer = object->getInteger(name)) value.*field = *integer;
}

template <typename Struct>
void read_attr_integer(mlir::DictionaryAttr dictionary, llvm::StringRef name,
    int64_t Struct::*field, Struct& value)
{
    if (!dictionary) return;
    if (const auto integer = dictionary.getAs<mlir::IntegerAttr>(name))
        value.*field = integer.getInt();
}

template <size_t Size>
void read_json_array(const llvm::json::Object* object, llvm::StringRef name,
    std::array<int64_t, Size>& values)
{
    if (!object) return;
    const auto* array = object->getArray(name);
    if (!array || array->size() != Size) return;
    for (size_t index = 0; index < Size; ++index)
        if (const auto integer = (*array)[index].getAsInteger())
            values[index] = *integer;
}

template <size_t Size>
void read_attr_array(mlir::DictionaryAttr dictionary, llvm::StringRef name,
    std::array<int64_t, Size>& values)
{
    if (!dictionary) return;
    const auto array = dictionary.getAs<mlir::ArrayAttr>(name);
    if (!array || array.size() != Size) return;
    for (size_t index = 0; index < Size; ++index)
        if (const auto integer = llvm::dyn_cast<mlir::IntegerAttr>(array[index]))
            values[index] = integer.getInt();
}

template <size_t Size>
mlir::ArrayAttr make_i64_array(
    mlir::Builder& builder, const std::array<int64_t, Size>& values)
{
    llvm::SmallVector<mlir::Attribute, Size> attributes;
    for (int64_t value : values)
        attributes.push_back(builder.getI64IntegerAttr(value));
    return builder.getArrayAttr(attributes);
}

} // namespace

// Keep physical-layout defaults in one compiled model shared by all stages.
LPUTargetModel::LPUTargetModel() = default;

LPUTargetModel::LPUTargetModel(MemoryTopology memory, StreamTopology streams,
    ThroughputModel throughput)
    : memory_(std::move(memory))
    , streams_(std::move(streams))
    , throughput_(std::move(throughput))
{
}

mlir::FailureOr<LPUTargetModel> LPUTargetModel::from_json(
    llvm::StringRef json, std::string& error)
{
    auto parsed = llvm::json::parse(json);
    if (!parsed) {
        error = llvm::toString(parsed.takeError());
        return mlir::failure();
    }
    const auto* root = parsed->getAsObject();
    if (!root) {
        error = "target configuration root must be a JSON object";
        return mlir::failure();
    }
    const auto* memory = root->getObject("memory");
    const auto* streams = root->getObject("streams");
    const auto* throughput = root->getObject("throughput");
    const auto* externalMemory = root->getObject("external_memory");
    LPUTargetModel model;

    // Schema v1 is the shared physical-target format consumed by both the
    // CModel CMake build and FTLPU-SOFTWARE. Legacy compiler exploration
    // files keep using the memory/streams/throughput overlay below.
    if (const auto schema = root->getInteger("schema_version")) {
        if (*schema != 1) {
            error = "unsupported hardware configuration schema_version";
            return mlir::failure();
        }
        const auto require_object = [&](llvm::StringRef name) {
            const auto* object = root->getObject(name);
            if (!object && error.empty())
                error = ("missing hardware configuration object: " + name).str();
            return object;
        };
        const auto* target = require_object("target");
        const auto* topology = require_object("topology");
        const auto* mem = require_object("mem");
        const auto* sr = require_object("sr");
        const auto* mxm = require_object("mxm");
        const auto* vxm = require_object("vxm");
        if (!error.empty()) return mlir::failure();

        const auto read_required = [&](const llvm::json::Object* object,
                                       llvm::StringRef section,
                                       llvm::StringRef field,
                                       int64_t& value) {
            const auto integer = object->getInteger(field);
            if (!integer) {
                error = ("missing integer hardware configuration field: "
                    + section + "." + field).str();
                return false;
            }
            value = *integer;
            return true;
        };

        const auto target_name = target->getString("name");
        if (!target_name || target_name->empty()) {
            error = "target.name must be a non-empty string";
            return mlir::failure();
        }
        model.name_ = target_name->str();

        int64_t bytes_per_lane = 0;
        int64_t registers_per_lane = 0;
        int64_t bytes_per_stream_per_lane = 0;
        int64_t vxm_alus_per_direction = 0;
        if (!read_required(topology, "topology", "hemispheres",
                model.memory_.hemispheres)
            || !read_required(topology, "topology", "tiles_per_slice",
                model.throughput_.tile_rows)
            || !read_required(topology, "topology", "lanes_per_tile",
                model.throughput_.lanes_per_tile)
            || !read_required(mem, "mem", "slices_per_hemisphere",
                model.memory_.slices_per_hemisphere)
            || !read_required(mem, "mem", "banks_per_slice",
                model.memory_.banks_per_slice)
            || !read_required(mem, "mem", "rows_per_bank",
                model.memory_.sram_depth_rows)
            || !read_required(mem, "mem", "bytes_per_lane",
                bytes_per_lane)
            || !read_required(sr, "sr", "registers_per_lane",
                registers_per_lane)
            || !read_required(sr, "sr", "bytes_per_stream_per_lane",
                bytes_per_stream_per_lane)
            || !read_required(mxm, "mxm", "accum_contexts",
                model.throughput_.mxm_accumulator_blocks)
            || !read_required(vxm, "vxm", "alus",
                vxm_alus_per_direction))
            return mlir::failure();

        if (model.memory_.hemispheres <= 0
            || model.throughput_.tile_rows <= 0
            || model.throughput_.lanes_per_tile <= 0
            || model.memory_.slices_per_hemisphere <= 0
            || model.memory_.banks_per_slice <= 0
            || model.memory_.sram_depth_rows <= 0
            || bytes_per_lane <= 0
            || registers_per_lane <= 0
            || bytes_per_stream_per_lane <= 0
            || model.throughput_.mxm_accumulator_blocks <= 0
            || vxm_alus_per_direction <= 0) {
            error = "hardware configuration integer fields must be positive";
            return mlir::failure();
        }
        if (registers_per_lane != 64) {
            error = "sr.registers_per_lane must be 64 for the current 6-bit stream ISA";
            return mlir::failure();
        }
        if (bytes_per_lane != 1 || bytes_per_stream_per_lane != 1) {
            error = "the current byte-stream datapath requires both byte-width fields to be 1";
            return mlir::failure();
        }
        if (model.throughput_.mxm_accumulator_blocks > 256) {
            error = "mxm.accum_contexts exceeds the 13-bit accumulator address capacity";
            return mlir::failure();
        }
        const auto rows_per_bank = model.memory_.sram_depth_rows;
        if (rows_per_bank < 64 || rows_per_bank > 32768
            || (rows_per_bank & (rows_per_bank - 1)) != 0) {
            error = "mem.rows_per_bank must be a power of two in [64, 32768]";
            return mlir::failure();
        }

        model.memory_.words_per_bank = model.memory_.sram_depth_rows;
        model.memory_.bytes_per_word = model.throughput_.tile_rows
            * model.throughput_.lanes_per_tile * bytes_per_lane;
        model.memory_.attention_mask_base_row =
            model.memory_.sram_depth_rows - 64;
        model.streams_.encoded_streams = registers_per_lane;
        model.streams_.streams_per_direction = registers_per_lane / 2;
        model.streams_.c2c_streams_per_direction =
            model.throughput_.lanes_per_tile;
        model.streams_.c2c_bytes_per_stream_per_cycle =
            model.memory_.bytes_per_word;
        model.streams_.mem_slices_per_register_group =
            model.throughput_.tile_rows;
        model.streams_.mem_boundary_register_columns =
            model.memory_.slices_per_hemisphere
                / model.streams_.mem_slices_per_register_group
            + 1;
        model.streams_.system_register_columns =
            model.streams_.mem_boundary_register_columns + 2;
        model.throughput_.mem_read_bytes_per_cycle =
            model.throughput_.lanes_per_tile * bytes_per_lane;
        model.throughput_.mem_write_bytes_per_cycle =
            model.throughput_.mem_read_bytes_per_cycle;
        model.throughput_.mxm_rows = model.memory_.bytes_per_word;
        model.throughput_.mxm_columns = model.memory_.bytes_per_word;
        model.throughput_.mxm_load_streams_per_cycle =
            2 * model.throughput_.lanes_per_tile;
        model.throughput_.mxm_int8_load_streams_per_cycle =
            model.throughput_.lanes_per_tile;
        model.throughput_.mxm_load_bytes_per_cycle =
            model.throughput_.lanes_per_tile
            * model.throughput_.mxm_load_streams_per_cycle
            * bytes_per_stream_per_lane;
        model.throughput_.mxm_pipeline_rows =
            model.throughput_.tile_rows;
        model.throughput_.vxm_alus = 2 * vxm_alus_per_direction;
        model.throughput_.mxm_block_compute_enabled = 0;
        model.throughput_.mem_to_sxm_latency =
            model.streams_.system_register_columns - 2;
        model.throughput_.mem_to_mxm_latency =
            model.streams_.system_register_columns;

        const auto* supported_modes = mxm->getArray("supported_modes");
        if (!supported_modes || supported_modes->empty()) {
            error = "mxm.supported_modes must be a non-empty array";
            return mlir::failure();
        }
        bool native4x4 = false;
        bool linear1x16 = false;
        for (const auto& value : *supported_modes) {
            const auto mode = value.getAsString();
            if (!mode) {
                error = "mxm.supported_modes entries must be strings";
                return mlir::failure();
            }
            if (*mode == "native4x4") {
                if (native4x4) {
                    error = "mxm.supported_modes contains native4x4 twice";
                    return mlir::failure();
                }
                native4x4 = true;
            } else if (*mode == "linear1x16") {
                if (linear1x16) {
                    error = "mxm.supported_modes contains linear1x16 twice";
                    return mlir::failure();
                }
                linear1x16 = true;
            }
            else {
                error = ("unsupported MXM mode: " + *mode).str();
                return mlir::failure();
            }
        }
        if (native4x4 != hw::kMxmSupportsNative4x4
            || linear1x16 != hw::kMxmSupportsLinear1x16) {
            error = "mxm.supported_modes differs from the software build configuration";
            return mlir::failure();
        }
    }
    if (const auto name = root->getString("name")) model.name_ = name->str();

#define READ_MEMORY(field) \
    read_json_integer(memory, #field, &MemoryTopology::field, model.memory_)
    READ_MEMORY(hemispheres);
    READ_MEMORY(slices_per_hemisphere);
    READ_MEMORY(banks_per_slice);
    READ_MEMORY(words_per_bank);
    READ_MEMORY(bytes_per_word);
    READ_MEMORY(sram_depth_rows);
    READ_MEMORY(sram_read_ports_per_slice);
    READ_MEMORY(sram_write_ports_per_slice);
    READ_MEMORY(dedicated_slice_roles);
    READ_MEMORY(w8a16_weight_slice_count);
    READ_MEMORY(w8a16_weight_slice_base);
    READ_MEMORY(w8a16_weight_slice_stride);
    READ_MEMORY(w8a16_activation_slice_base);
    READ_MEMORY(w8a16_hidden_slice_base);
    READ_MEMORY(w8a16_hidden_base_row);
    READ_MEMORY(attention_mask_base_row);
    READ_MEMORY(w8a16_result_slice_base);
    READ_MEMORY(matmul_result_base_row);
#undef READ_MEMORY
    read_json_array(memory, "w8a16_fused_gate_temp_slices",
        model.memory_.w8a16_fused_gate_temp_slices);
    read_json_array(memory, "w8a16_fused_up_temp_slices",
        model.memory_.w8a16_fused_up_temp_slices);
    read_json_array(memory, "w8a16_output_weight_spare_slices",
        model.memory_.w8a16_output_weight_spare_slices);

#define READ_STREAM(field) \
    read_json_integer(streams, #field, &StreamTopology::field, model.streams_)
    READ_STREAM(streams_per_direction);
    READ_STREAM(encoded_streams);
    READ_STREAM(c2c_streams_per_direction);
    READ_STREAM(c2c_bytes_per_stream_per_cycle);
    READ_STREAM(mem_boundary_register_columns);
    READ_STREAM(system_register_columns);
    READ_STREAM(mem_slices_per_register_group);
#undef READ_STREAM

#define READ_THROUGHPUT(field) \
    read_json_integer(throughput, #field, &ThroughputModel::field, model.throughput_)
    READ_THROUGHPUT(icu_repeat_2d_enabled);
    READ_THROUGHPUT(tile_rows);
    READ_THROUGHPUT(lanes_per_tile);
    READ_THROUGHPUT(mem_read_bytes_per_cycle);
    READ_THROUGHPUT(mem_write_bytes_per_cycle);
    READ_THROUGHPUT(mxm_rows);
    READ_THROUGHPUT(mxm_columns);
    READ_THROUGHPUT(mxm_load_streams_per_cycle);
    READ_THROUGHPUT(mxm_int8_load_streams_per_cycle);
    READ_THROUGHPUT(mxm_load_bytes_per_cycle);
    READ_THROUGHPUT(mxm_activation_streams);
    READ_THROUGHPUT(mxm_result_streams);
    READ_THROUGHPUT(mxm_pipeline_rows);
    READ_THROUGHPUT(mxm_block_rows);
    READ_THROUGHPUT(mxm_local_dequant_enabled);
    READ_THROUGHPUT(mxm_block_compute_enabled);
    READ_THROUGHPUT(mxm_weight_activation_overlap_enabled);
    READ_THROUGHPUT(mxm_local_load_to_compute_latency);
    READ_THROUGHPUT(mxm_block_group_interval);
    READ_THROUGHPUT(mxm_earliest_iw_cycle);
    READ_THROUGHPUT(qk_iw_to_compute_latency);
    READ_THROUGHPUT(mxms_per_hemisphere);
    READ_THROUGHPUT(mxm_weight_buffers);
    READ_THROUGHPUT(mxm_accumulator_blocks);
    READ_THROUGHPUT(vxm_alus);
    READ_THROUGHPUT(vxm_weight_to_iw_latency);
    READ_THROUGHPUT(mem_to_sxm_latency);
    READ_THROUGHPUT(mem_to_mxm_latency);
    READ_THROUGHPUT(mxm0_accumulator_latency);
    READ_THROUGHPUT(mxm1_accumulator_latency);
    READ_THROUGHPUT(accumulator_to_vxm_latency);
    READ_THROUGHPUT(accumulator_read_to_vxm_latency);
    READ_THROUGHPUT(swiglu_write_latency);
#undef READ_THROUGHPUT

#define READ_EXTERNAL(field) \
    read_json_integer(externalMemory, #field, &ExternalMemoryModel::field, \
        model.external_memory_)
    READ_EXTERNAL(lpu_clock_mhz);
    READ_EXTERNAL(ddr_peak_bandwidth_mbytes_per_second);
    READ_EXTERNAL(ddr_scheduling_efficiency_percent);
    READ_EXTERNAL(ddr_read_latency_cycles);
    READ_EXTERNAL(ddr_write_latency_cycles);
    READ_EXTERNAL(ddr_read_latency_jitter_cycles);
    READ_EXTERNAL(ddr_write_latency_jitter_cycles);
    READ_EXTERNAL(ddr_request_queue_depth);
    READ_EXTERNAL(ddr_latency_random_seed);
#undef READ_EXTERNAL

    if (mlir::failed(model.validate(&error))) return mlir::failure();
    return model;
}

mlir::FailureOr<LPUTargetModel> LPUTargetModel::from_operation(
    mlir::Operation* operation)
{
    mlir::DictionaryAttr target;
    for (mlir::Operation* current = operation; current;
         current = current->getParentOp()) {
        target = current->getAttrOfType<mlir::DictionaryAttr>("ftlpu.target");
        if (target) break;
    }
    if (!target) return LPUTargetModel{};

    // Operation verifiers call this for every emitted Schedule/Command op.
    // DictionaryAttr is immutable and uniqued within its MLIRContext, so the
    // last parsed target remains valid until either the context or attribute
    // changes. This avoids reparsing and rehashing the same target hundreds
    // of thousands of times for model-scale schedules.
    struct ParsedTargetCache {
        mlir::MLIRContext* context;
        mlir::DictionaryAttr attribute;
        LPUTargetModel model;
    };
    static thread_local std::optional<ParsedTargetCache> cache;
    if (cache && cache->context == operation->getContext()
        && cache->attribute == target)
        return cache->model;

    LPUTargetModel model;
    if (const auto name = target.getAs<mlir::StringAttr>("name"))
        model.name_ = name.str();
    const auto memory = target.getAs<mlir::DictionaryAttr>("memory");
    const auto streams = target.getAs<mlir::DictionaryAttr>("streams");
    const auto throughput = target.getAs<mlir::DictionaryAttr>("throughput");
    const auto externalMemory =
        target.getAs<mlir::DictionaryAttr>("external_memory");
#define READ_MEMORY(field) \
    read_attr_integer(memory, #field, &MemoryTopology::field, model.memory_)
    READ_MEMORY(hemispheres);
    READ_MEMORY(slices_per_hemisphere);
    READ_MEMORY(banks_per_slice);
    READ_MEMORY(words_per_bank);
    READ_MEMORY(bytes_per_word);
    READ_MEMORY(sram_depth_rows);
    READ_MEMORY(sram_read_ports_per_slice);
    READ_MEMORY(sram_write_ports_per_slice);
    READ_MEMORY(dedicated_slice_roles);
    READ_MEMORY(w8a16_weight_slice_count);
    READ_MEMORY(w8a16_weight_slice_base);
    READ_MEMORY(w8a16_weight_slice_stride);
    READ_MEMORY(w8a16_activation_slice_base);
    READ_MEMORY(w8a16_hidden_slice_base);
    READ_MEMORY(w8a16_hidden_base_row);
    READ_MEMORY(attention_mask_base_row);
    READ_MEMORY(w8a16_result_slice_base);
    READ_MEMORY(matmul_result_base_row);
#undef READ_MEMORY
    read_attr_array(memory, "w8a16_fused_gate_temp_slices",
        model.memory_.w8a16_fused_gate_temp_slices);
    read_attr_array(memory, "w8a16_fused_up_temp_slices",
        model.memory_.w8a16_fused_up_temp_slices);
    read_attr_array(memory, "w8a16_output_weight_spare_slices",
        model.memory_.w8a16_output_weight_spare_slices);
#define READ_STREAM(field) \
    read_attr_integer(streams, #field, &StreamTopology::field, model.streams_)
    READ_STREAM(streams_per_direction);
    READ_STREAM(encoded_streams);
    READ_STREAM(c2c_streams_per_direction);
    READ_STREAM(c2c_bytes_per_stream_per_cycle);
    READ_STREAM(mem_boundary_register_columns);
    READ_STREAM(system_register_columns);
    READ_STREAM(mem_slices_per_register_group);
#undef READ_STREAM
#define READ_THROUGHPUT(field) \
    read_attr_integer(throughput, #field, &ThroughputModel::field, model.throughput_)
    READ_THROUGHPUT(icu_repeat_2d_enabled);
    READ_THROUGHPUT(tile_rows);
    READ_THROUGHPUT(lanes_per_tile);
    READ_THROUGHPUT(mem_read_bytes_per_cycle);
    READ_THROUGHPUT(mem_write_bytes_per_cycle);
    READ_THROUGHPUT(mxm_rows);
    READ_THROUGHPUT(mxm_columns);
    READ_THROUGHPUT(mxm_load_streams_per_cycle);
    READ_THROUGHPUT(mxm_int8_load_streams_per_cycle);
    READ_THROUGHPUT(mxm_load_bytes_per_cycle);
    READ_THROUGHPUT(mxm_activation_streams);
    READ_THROUGHPUT(mxm_result_streams);
    READ_THROUGHPUT(mxm_pipeline_rows);
    READ_THROUGHPUT(mxm_block_rows);
    READ_THROUGHPUT(mxm_local_dequant_enabled);
    READ_THROUGHPUT(mxm_block_compute_enabled);
    READ_THROUGHPUT(mxm_weight_activation_overlap_enabled);
    READ_THROUGHPUT(mxm_local_load_to_compute_latency);
    READ_THROUGHPUT(mxm_block_group_interval);
    READ_THROUGHPUT(mxm_earliest_iw_cycle);
    READ_THROUGHPUT(qk_iw_to_compute_latency);
    READ_THROUGHPUT(mxms_per_hemisphere);
    READ_THROUGHPUT(mxm_weight_buffers);
    READ_THROUGHPUT(mxm_accumulator_blocks);
    READ_THROUGHPUT(vxm_alus);
    READ_THROUGHPUT(vxm_weight_to_iw_latency);
    READ_THROUGHPUT(mem_to_sxm_latency);
    READ_THROUGHPUT(mem_to_mxm_latency);
    READ_THROUGHPUT(mxm0_accumulator_latency);
    READ_THROUGHPUT(mxm1_accumulator_latency);
    READ_THROUGHPUT(accumulator_to_vxm_latency);
    READ_THROUGHPUT(accumulator_read_to_vxm_latency);
    READ_THROUGHPUT(swiglu_write_latency);
#undef READ_THROUGHPUT
#define READ_EXTERNAL(field) \
    read_attr_integer(externalMemory, #field, &ExternalMemoryModel::field, \
        model.external_memory_)
    READ_EXTERNAL(lpu_clock_mhz);
    READ_EXTERNAL(ddr_peak_bandwidth_mbytes_per_second);
    READ_EXTERNAL(ddr_scheduling_efficiency_percent);
    READ_EXTERNAL(ddr_read_latency_cycles);
    READ_EXTERNAL(ddr_write_latency_cycles);
    READ_EXTERNAL(ddr_read_latency_jitter_cycles);
    READ_EXTERNAL(ddr_write_latency_jitter_cycles);
    READ_EXTERNAL(ddr_request_queue_depth);
    READ_EXTERNAL(ddr_latency_random_seed);
#undef READ_EXTERNAL
    std::string error;
    if (mlir::failed(model.validate(&error))) {
        operation->emitError("invalid ftlpu.target configuration: ") << error;
        return mlir::failure();
    }
    if (const auto abi = target.getAs<mlir::StringAttr>("abi")) {
        std::uint64_t serialized = 0;
        const std::string value = abi.str();
        const auto* begin = value.data();
        const auto* end = begin + value.size();
        if (value.starts_with("0x")) begin += 2;
        const auto result = std::from_chars(begin, end, serialized, 16);
        if (result.ec != std::errc{} || result.ptr != end
            || serialized != model.abi_fingerprint()) {
            operation->emitError(
                "ftlpu.target ABI fingerprint does not match its parameters");
            return mlir::failure();
        }
    }
    cache = ParsedTargetCache {
        operation->getContext(), target, model};
    return model;
}

mlir::DictionaryAttr LPUTargetModel::to_attribute(
    mlir::MLIRContext* context) const
{
    mlir::Builder builder(context);
    std::ostringstream abi;
    abi << "0x" << std::hex << std::setfill('0') << std::setw(16)
        << abi_fingerprint();
#define I64(object, field) \
    builder.getNamedAttr(#field, builder.getI64IntegerAttr(object.field))
    const auto memory = builder.getDictionaryAttr({
        I64(memory_, hemispheres),
        I64(memory_, slices_per_hemisphere),
        I64(memory_, banks_per_slice),
        I64(memory_, words_per_bank),
        I64(memory_, bytes_per_word),
        I64(memory_, sram_depth_rows),
        I64(memory_, sram_read_ports_per_slice),
        I64(memory_, sram_write_ports_per_slice),
        I64(memory_, dedicated_slice_roles),
        I64(memory_, w8a16_weight_slice_count),
        I64(memory_, w8a16_weight_slice_base),
        I64(memory_, w8a16_weight_slice_stride),
        I64(memory_, w8a16_activation_slice_base),
        I64(memory_, w8a16_hidden_slice_base),
        I64(memory_, w8a16_hidden_base_row),
        I64(memory_, attention_mask_base_row),
        I64(memory_, w8a16_result_slice_base),
        I64(memory_, matmul_result_base_row),
        builder.getNamedAttr("w8a16_fused_gate_temp_slices",
            make_i64_array(builder, memory_.w8a16_fused_gate_temp_slices)),
        builder.getNamedAttr("w8a16_fused_up_temp_slices",
            make_i64_array(builder, memory_.w8a16_fused_up_temp_slices)),
        builder.getNamedAttr("w8a16_output_weight_spare_slices",
            make_i64_array(
                builder, memory_.w8a16_output_weight_spare_slices)),
    });
    const auto streams = builder.getDictionaryAttr({
        I64(streams_, streams_per_direction),
        I64(streams_, encoded_streams),
        I64(streams_, c2c_streams_per_direction),
        I64(streams_, c2c_bytes_per_stream_per_cycle),
        I64(streams_, mem_boundary_register_columns),
        I64(streams_, system_register_columns),
        I64(streams_, mem_slices_per_register_group),
    });
    const auto throughput = builder.getDictionaryAttr({
        I64(throughput_, icu_repeat_2d_enabled),
        I64(throughput_, tile_rows),
        I64(throughput_, lanes_per_tile),
        I64(throughput_, mem_read_bytes_per_cycle),
        I64(throughput_, mem_write_bytes_per_cycle),
        I64(throughput_, mxm_rows),
        I64(throughput_, mxm_columns),
        I64(throughput_, mxm_load_streams_per_cycle),
        I64(throughput_, mxm_int8_load_streams_per_cycle),
        I64(throughput_, mxm_load_bytes_per_cycle),
        I64(throughput_, mxm_activation_streams),
        I64(throughput_, mxm_result_streams),
        I64(throughput_, mxm_pipeline_rows),
        I64(throughput_, mxm_block_rows),
        I64(throughput_, mxm_local_dequant_enabled),
        I64(throughput_, mxm_block_compute_enabled),
        I64(throughput_, mxm_weight_activation_overlap_enabled),
        I64(throughput_, mxm_local_load_to_compute_latency),
        I64(throughput_, mxm_block_group_interval),
        I64(throughput_, mxm_earliest_iw_cycle),
        I64(throughput_, qk_iw_to_compute_latency),
        I64(throughput_, mxms_per_hemisphere),
        I64(throughput_, mxm_weight_buffers),
        I64(throughput_, mxm_accumulator_blocks),
        I64(throughput_, vxm_alus),
        I64(throughput_, vxm_weight_to_iw_latency),
        I64(throughput_, mem_to_sxm_latency),
        I64(throughput_, mem_to_mxm_latency),
        I64(throughput_, mxm0_accumulator_latency),
        I64(throughput_, mxm1_accumulator_latency),
        I64(throughput_, accumulator_to_vxm_latency),
        I64(throughput_, accumulator_read_to_vxm_latency),
        I64(throughput_, swiglu_write_latency),
    });
    const auto externalMemory = builder.getDictionaryAttr({
        I64(external_memory_, lpu_clock_mhz),
        I64(external_memory_, ddr_peak_bandwidth_mbytes_per_second),
        I64(external_memory_, ddr_scheduling_efficiency_percent),
        I64(external_memory_, ddr_read_latency_cycles),
        I64(external_memory_, ddr_write_latency_cycles),
        I64(external_memory_, ddr_read_latency_jitter_cycles),
        I64(external_memory_, ddr_write_latency_jitter_cycles),
        I64(external_memory_, ddr_request_queue_depth),
        I64(external_memory_, ddr_latency_random_seed),
    });
#undef I64
    return builder.getDictionaryAttr({
        builder.getNamedAttr("name", builder.getStringAttr(name_)),
        builder.getNamedAttr("abi", builder.getStringAttr(abi.str())),
        builder.getNamedAttr("memory", memory),
        builder.getNamedAttr("streams", streams),
        builder.getNamedAttr("throughput", throughput),
        builder.getNamedAttr("external_memory", externalMemory),
    });
}

mlir::LogicalResult LPUTargetModel::validate(std::string* error) const
{
    const auto fail = [&](llvm::StringRef message) {
        if (error) *error = message.str();
        return mlir::failure();
    };
    const auto positive = [](std::initializer_list<int64_t> values) {
        return llvm::all_of(values, [](int64_t value) { return value > 0; });
    };
    if (name_.empty()) return fail("target name must not be empty");
    if (!positive({memory_.hemispheres, memory_.slices_per_hemisphere,
            memory_.banks_per_slice, memory_.words_per_bank,
            memory_.bytes_per_word, memory_.sram_depth_rows,
            memory_.sram_read_ports_per_slice,
            memory_.sram_write_ports_per_slice,
            streams_.streams_per_direction,
            streams_.encoded_streams, streams_.c2c_streams_per_direction,
            streams_.c2c_bytes_per_stream_per_cycle,
            streams_.mem_slices_per_register_group,
            throughput_.tile_rows, throughput_.lanes_per_tile,
            throughput_.mxm_rows, throughput_.mxm_columns,
            throughput_.mxm_load_streams_per_cycle,
            throughput_.mxm_int8_load_streams_per_cycle,
            throughput_.mxm_activation_streams,
            throughput_.mxm_result_streams, throughput_.mxms_per_hemisphere,
            throughput_.mxm_weight_buffers,
            throughput_.mxm_accumulator_blocks, throughput_.vxm_alus,
            throughput_.qk_iw_to_compute_latency,
            throughput_.mxm_block_rows,
            throughput_.mxm_local_load_to_compute_latency,
            throughput_.mxm_block_group_interval,
            external_memory_.lpu_clock_mhz,
            external_memory_.ddr_peak_bandwidth_mbytes_per_second,
            external_memory_.ddr_scheduling_efficiency_percent,
            external_memory_.ddr_read_latency_cycles,
            external_memory_.ddr_write_latency_cycles,
            external_memory_.ddr_request_queue_depth}))
        return fail("topology dimensions and throughput values must be positive");
    if (external_memory_.ddr_read_latency_jitter_cycles < 0
        || external_memory_.ddr_write_latency_jitter_cycles < 0
        || external_memory_.ddr_latency_random_seed < 0)
        return fail("external-memory jitter and seed must be non-negative");
    if (external_memory_.ddr_scheduling_efficiency_percent > 100)
        return fail("DDR scheduling efficiency must be in [1, 100]");
    if ((throughput_.mxm_local_dequant_enabled != 0
            && throughput_.mxm_local_dequant_enabled != 1)
        || (throughput_.mxm_block_compute_enabled != 0
            && throughput_.mxm_block_compute_enabled != 1)
        || (throughput_.mxm_weight_activation_overlap_enabled != 0
            && throughput_.mxm_weight_activation_overlap_enabled != 1)
        || (throughput_.icu_repeat_2d_enabled != 0
            && throughput_.icu_repeat_2d_enabled != 1))
        return fail("target feature switches must be zero or one");
    if (throughput_.mxm_accumulator_blocks > 256)
        return fail(
            "mxm_accumulator_blocks exceeds the 13-bit address encoding");
    if (streams_.encoded_streams
        < 2 * streams_.streams_per_direction)
        return fail("encoded_streams must cover east and west stream ranges");
    if (streams_.c2c_bytes_per_stream_per_cycle != memory_.bytes_per_word)
        return fail(
            "a dedicated C2C lane must carry one complete SRAM row per cycle");
    if (streams_.system_register_columns
        < streams_.mem_boundary_register_columns)
        return fail("system register columns must cover MEM boundary columns");
    if (throughput_.mem_to_sxm_latency
            != streams_.system_register_columns - 2
        || throughput_.mem_to_mxm_latency
            != streams_.system_register_columns)
        return fail("MEM transport latencies must match the stream-register topology");
    if (throughput_.mxm_rows % throughput_.tile_rows != 0
        || throughput_.mxm_columns % throughput_.lanes_per_tile != 0
        || throughput_.mxm_rows % throughput_.mxm_block_rows != 0)
        return fail("MXM dimensions must be divisible by tile geometry");
    if (throughput_.mxm_load_streams_per_cycle
            > streams_.streams_per_direction
        || throughput_.mxm_int8_load_streams_per_cycle
            > streams_.streams_per_direction
        || throughput_.mxm_activation_streams
            > streams_.streams_per_direction
        || throughput_.mxm_result_streams
            > streams_.streams_per_direction)
        return fail("functional-unit stream width exceeds directional streams");
    const auto valid_slice = [&](int64_t slice) {
        return slice >= 0 && slice < memory_.slices_per_hemisphere;
    };
    if (!valid_slice(memory_.w8a16_activation_slice_base)
        || !valid_slice(memory_.w8a16_hidden_slice_base)
        || !valid_slice(memory_.w8a16_result_slice_base))
        return fail("configured MEM slice base is outside a hemisphere");
    if (memory_.w8a16_hidden_base_row < 0
        || memory_.w8a16_hidden_base_row >= memory_.sram_depth_rows)
        return fail("configured W8A16 hidden base row is outside SRAM");
    if (memory_.attention_mask_base_row < 0
        || memory_.attention_mask_base_row >= memory_.sram_depth_rows)
        return fail("configured attention mask base row is outside SRAM");
    if (!valid_slice(memory_.w8a16_weight_slice_base)
        || !valid_slice(memory_.w8a16_weight_slice_base
            + (memory_.w8a16_weight_slice_count - 1)
                * memory_.w8a16_weight_slice_stride))
        return fail("configured W8A16 weight slices are outside a hemisphere");
    if (uses_dedicated_slice_roles()
        && activation_storage_slices().size() < 16)
        return fail(
            "dedicated activation storage requires at least 16 MEM slices for the SXM plane");
    for (int64_t slice : memory_.w8a16_fused_gate_temp_slices)
        if (!valid_slice(slice)) return fail("gate temporary slice is invalid");
    for (int64_t slice : memory_.w8a16_fused_up_temp_slices)
        if (!valid_slice(slice)) return fail("up temporary slice is invalid");
    for (int64_t slice : memory_.w8a16_output_weight_spare_slices)
        if (!valid_slice(slice))
            return fail("output weight spare slice is invalid");
    llvm::SmallDenseSet<int64_t, 8> fusedTempSlices;
    for (int64_t slice : memory_.w8a16_fused_gate_temp_slices)
        if (!fusedTempSlices.insert(slice).second)
            return fail("fused temporary slices must be unique");
    for (int64_t slice : memory_.w8a16_fused_up_temp_slices)
        if (!fusedTempSlices.insert(slice).second)
            return fail("fused temporary slices must be unique");
    if (ffn_hidden_slices().size()
        != static_cast<std::size_t>(throughput_.mxm_activation_streams))
        return fail("not enough independent MEM slices for fused hidden data");
    const auto gateWeightSlices =
        ffn_projection_weight_slices(FfnProjectionKind::Gate);
    const auto upWeightSlices =
        ffn_projection_weight_slices(FfnProjectionKind::Up);
    if (gateWeightSlices.size()
            != static_cast<std::size_t>(memory_.w8a16_weight_slice_count)
        || upWeightSlices.size()
            != static_cast<std::size_t>(memory_.w8a16_weight_slice_count))
        return fail("not enough independent MEM slices for W8A16 FFN");
    llvm::SmallDenseSet<int64_t, 16> gateWeightSet(
        gateWeightSlices.begin(), gateWeightSlices.end());
    for (int64_t slice : upWeightSlices) {
        if (gateWeightSet.contains(slice))
            return fail("Gate and Up weights must use independent slices");
    }
    return mlir::success();
}

std::uint64_t LPUTargetModel::abi_fingerprint() const
{
    software::runtime::TargetAbiHasher hash;
    hash.add(16);
#define HASH(field) hash.add(memory_.field)
    HASH(hemispheres);
    HASH(slices_per_hemisphere);
    HASH(banks_per_slice);
    HASH(words_per_bank);
    HASH(bytes_per_word);
    HASH(sram_depth_rows);
    HASH(sram_read_ports_per_slice);
    HASH(sram_write_ports_per_slice);
#undef HASH
#define HASH(field) hash.add(streams_.field)
    HASH(streams_per_direction);
    HASH(encoded_streams);
    HASH(c2c_streams_per_direction);
    HASH(c2c_bytes_per_stream_per_cycle);
    HASH(mem_boundary_register_columns);
    HASH(system_register_columns);
    HASH(mem_slices_per_register_group);
#undef HASH
#define HASH(field) hash.add(throughput_.field)
    HASH(tile_rows);
    HASH(lanes_per_tile);
    HASH(mem_read_bytes_per_cycle);
    HASH(mem_write_bytes_per_cycle);
    HASH(mxm_rows);
    HASH(mxm_columns);
    HASH(mxm_load_streams_per_cycle);
    HASH(mxm_int8_load_streams_per_cycle);
    HASH(mxm_load_bytes_per_cycle);
    HASH(mxm_activation_streams);
    HASH(mxm_result_streams);
    HASH(mxm_pipeline_rows);
    HASH(mxm_block_rows);
    HASH(mxm_local_dequant_enabled);
    HASH(mxm_block_compute_enabled);
    HASH(mxm_weight_activation_overlap_enabled);
    HASH(mxm_local_load_to_compute_latency);
    HASH(mxm_block_group_interval);
    HASH(mxm_earliest_iw_cycle);
    HASH(qk_iw_to_compute_latency);
    HASH(mxms_per_hemisphere);
    HASH(mxm_weight_buffers);
    HASH(vxm_alus);
    HASH(vxm_weight_to_iw_latency);
    HASH(mem_to_sxm_latency);
    HASH(mem_to_mxm_latency);
    HASH(mxm0_accumulator_latency);
    HASH(mxm1_accumulator_latency);
    HASH(accumulator_to_vxm_latency);
    HASH(accumulator_read_to_vxm_latency);
    HASH(swiglu_write_latency);
    HASH(mxm_accumulator_blocks);
#undef HASH
#define HASH(field) hash.add(external_memory_.field)
    HASH(lpu_clock_mhz);
    HASH(ddr_peak_bandwidth_mbytes_per_second);
    HASH(ddr_scheduling_efficiency_percent);
    HASH(ddr_read_latency_cycles);
    HASH(ddr_write_latency_cycles);
    HASH(ddr_read_latency_jitter_cycles);
    HASH(ddr_write_latency_jitter_cycles);
    HASH(ddr_request_queue_depth);
    HASH(ddr_latency_random_seed);
#undef HASH
    return hash.value();
}

int64_t LPUTargetModel::external_read_transfer_cycles(int64_t bytes) const
{
    if (bytes <= 0) return 0;
    const int64_t c2cBytesPerCycle = memory_.hemispheres
        * streams_.c2c_streams_per_direction
        * streams_.c2c_bytes_per_stream_per_cycle;
    const int64_t c2cCycles = divide_ceil(bytes, c2cBytesPerCycle);
    const int64_t ddrCycles = divide_ceil(
        bytes * external_memory_.lpu_clock_mhz * 100,
        external_memory_.ddr_peak_bandwidth_mbytes_per_second
            * external_memory_.ddr_scheduling_efficiency_percent);
    return std::max(c2cCycles, ddrCycles)
        + external_memory_.ddr_read_latency_cycles
        + external_memory_.ddr_read_latency_jitter_cycles;
}

bool LPUTargetModel::supports_route(StreamEndpoint source, StreamEndpoint destination,
    StreamDirection direction) const
{
    if (source == StreamEndpoint::Mem
        && destination == StreamEndpoint::MxmActivation)
        return direction == StreamDirection::East;
    if (source == StreamEndpoint::Mem
        && destination == StreamEndpoint::MxmWeight)
        return direction == StreamDirection::East;
    if (source == StreamEndpoint::Mem && destination == StreamEndpoint::VxmInput)
        return direction == StreamDirection::East || direction == StreamDirection::West;
    if (source == StreamEndpoint::Mem && destination == StreamEndpoint::SxmInput)
        return direction == StreamDirection::East;
    if (source == StreamEndpoint::MxmResult
        && destination == StreamEndpoint::VxmInput)
        return direction == StreamDirection::West;
    if (source == StreamEndpoint::VxmResult && destination == StreamEndpoint::MxmWeight)
        return direction == StreamDirection::East;
    if (source == StreamEndpoint::MxmResult && destination == StreamEndpoint::Mem)
        return direction == StreamDirection::West;
    if ((source == StreamEndpoint::VxmResult
            || source == StreamEndpoint::VxmBridgeResult)
        && destination == StreamEndpoint::Mem)
        return direction == StreamDirection::East;
    if (source == StreamEndpoint::SxmResult && destination == StreamEndpoint::Mem)
        return direction == StreamDirection::West;
    return false;
}

std::optional<int64_t> LPUTargetModel::route_stream_count(StreamEndpoint source,
    StreamEndpoint destination, StreamDirection direction) const
{
    if (!supports_route(source, destination, direction)) return std::nullopt;
    if (source == StreamEndpoint::Mem && destination == StreamEndpoint::VxmInput)
        return throughput_.lanes_per_tile;
    if ((source == StreamEndpoint::Mem
            && destination == StreamEndpoint::SxmInput)
        || (source == StreamEndpoint::SxmResult
            && destination == StreamEndpoint::Mem))
        return 2;
    if (source == StreamEndpoint::MxmResult
        && destination == StreamEndpoint::VxmInput)
        return throughput_.mxm_result_streams;
    if (destination == StreamEndpoint::MxmWeight)
        return throughput_.mxm_load_streams_per_cycle;
    if (destination == StreamEndpoint::MxmActivation)
        return throughput_.mxm_activation_streams;
    return throughput_.mxm_result_streams;
}

std::optional<int64_t> LPUTargetModel::stream_register_id(StreamEndpoint source,
    StreamEndpoint destination, StreamDirection direction, int64_t mem_slice) const
{
    if (!supports_route(source, destination, direction)
        || mem_slice < 0 || mem_slice >= memory_.slices_per_hemisphere)
        return std::nullopt;
    return mem_slice / streams_.mem_slices_per_register_group + 1;
}

std::optional<int64_t> LPUTargetModel::stream_source_column(
    StreamEndpoint source, StreamDirection direction, int64_t mem_slice) const
{
    if (mem_slice < 0 || mem_slice >= memory_.slices_per_hemisphere)
        return std::nullopt;
    const int64_t group =
        mem_slice / streams_.mem_slices_per_register_group;
    switch (source) {
    case StreamEndpoint::Mem:
        // MEM group g lies between boundary columns g and g+1.
        return direction == StreamDirection::East ? group + 1 : group;
    case StreamEndpoint::MxmResult:
        return streams_.system_register_columns - 1;
    case StreamEndpoint::VxmResult:
    case StreamEndpoint::VxmBridgeResult:
        return 0;
    case StreamEndpoint::SxmResult:
        return direction == StreamDirection::East
            ? streams_.system_register_columns - 1
            : streams_.system_register_columns - 2;
    case StreamEndpoint::MxmActivation:
    case StreamEndpoint::MxmWeight:
    case StreamEndpoint::VxmInput:
    case StreamEndpoint::SxmInput:
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<int64_t> LPUTargetModel::stream_destination_column(
    StreamEndpoint destination, StreamDirection direction,
    int64_t mem_slice) const
{
    if (mem_slice < 0 || mem_slice >= memory_.slices_per_hemisphere)
        return std::nullopt;
    const int64_t group =
        mem_slice / streams_.mem_slices_per_register_group;
    switch (destination) {
    case StreamEndpoint::Mem:
        return direction == StreamDirection::East ? group : group + 1;
    case StreamEndpoint::MxmActivation:
    case StreamEndpoint::MxmWeight:
        return streams_.system_register_columns - 1;
    case StreamEndpoint::VxmInput:
        return 0;
    case StreamEndpoint::SxmInput:
        return direction == StreamDirection::East
            ? streams_.system_register_columns - 2
            : streams_.system_register_columns - 1;
    case StreamEndpoint::MxmResult:
    case StreamEndpoint::VxmResult:
    case StreamEndpoint::VxmBridgeResult:
    case StreamEndpoint::SxmResult:
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<int64_t> LPUTargetModel::route_issue_cycles(StreamEndpoint source,
    StreamEndpoint destination, int64_t bytes) const
{
    if (bytes <= 0) return std::nullopt;
    const int64_t vector_bytes = throughput_.tile_rows * throughput_.lanes_per_tile;
    if (source == StreamEndpoint::Mem && destination == StreamEndpoint::VxmInput)
        return divide_ceil(bytes, throughput_.mxm_rows * throughput_.lanes_per_tile);
    if (source == StreamEndpoint::Mem && destination == StreamEndpoint::MxmWeight)
        return divide_ceil(bytes, vector_bytes * throughput_.mxm_load_streams_per_cycle);
    if (source == StreamEndpoint::Mem && destination == StreamEndpoint::MxmActivation)
        return divide_ceil(bytes, vector_bytes);
    if (source == StreamEndpoint::Mem && destination == StreamEndpoint::SxmInput)
        return divide_ceil(bytes, 2 * vector_bytes);
    if (source == StreamEndpoint::SxmResult && destination == StreamEndpoint::Mem)
        return divide_ceil(bytes, 2 * vector_bytes);
    if (source == StreamEndpoint::MxmResult && destination == StreamEndpoint::Mem)
        return divide_ceil(bytes, vector_bytes * throughput_.mxm_result_streams);
    if ((source == StreamEndpoint::VxmResult
            || source == StreamEndpoint::VxmBridgeResult)
        && destination == StreamEndpoint::Mem)
        return divide_ceil(bytes, vector_bytes);
    return std::nullopt;
}

int64_t LPUTargetModel::mxm_compute_issue_cycles(int64_t rows) const
{
    return rows;
}

int64_t LPUTargetModel::mxm_first_result_latency() const
{
    return throughput_.mxm_pipeline_rows - 1;
}

int64_t LPUTargetModel::mxm_result_window_cycles(int64_t rows) const
{
    return rows + mxm_first_result_latency();
}

int64_t LPUTargetModel::mxm_block_issue_interval() const
{
    // The current MXM model wraps and clears physical accumulator rows as a
    // new 32-row block enters, so blocks can be issued back-to-back.
    return throughput_.mxm_rows;
}

llvm::SmallVector<int64_t> LPUTargetModel::weight_storage_slices() const
{
    llvm::SmallVector<int64_t> slices;
    if (!uses_dedicated_slice_roles()) {
        for (int64_t slice = 0;
             slice < memory_.slices_per_hemisphere; ++slice)
            slices.push_back(slice);
        return slices;
    }
    for (int64_t slice = memory_.w8a16_weight_slice_base;
         slice < memory_.slices_per_hemisphere; ++slice)
        slices.push_back(slice);
    return slices;
}

llvm::SmallVector<int64_t> LPUTargetModel::activation_storage_slices() const
{
    llvm::SmallVector<int64_t> slices;
    if (!uses_dedicated_slice_roles()) {
        for (int64_t slice = 0;
             slice < memory_.slices_per_hemisphere; ++slice)
            slices.push_back(slice);
        return slices;
    }
    for (int64_t slice = 0;
         slice < memory_.w8a16_weight_slice_base; ++slice)
        slices.push_back(slice);
    return slices;
}

bool LPUTargetModel::is_weight_storage_slice(int64_t slice) const
{
    return llvm::is_contained(weight_storage_slices(), slice);
}

bool LPUTargetModel::is_activation_storage_slice(int64_t slice) const
{
    return llvm::is_contained(activation_storage_slices(), slice);
}

const std::array<int64_t, 16>& LPUTargetModel::attention_query_iw_slices(
    int64_t reduction_block) const
{
    static constexpr std::array<int64_t, 16> kDedicatedSlices {{
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15}};
    static constexpr std::array<std::array<int64_t, 16>, 2> kSingleMxmSlices {{
        {{0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33}},
        {{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35}},
    }};
    static constexpr std::array<std::array<int64_t, 16>, 2> kDualMxmSlices {{
        {{44, 45, 46, 47, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 17, 32, 33}},
        {{18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 34, 35}},
    }};
    if (uses_dedicated_slice_roles()) {
        if (reduction_block < 0)
            throw std::out_of_range("attention query IW reduction block");
        return kDedicatedSlices;
    }
    const auto& slices = throughput_.mxms_per_hemisphere == 1
        ? kSingleMxmSlices : kDualMxmSlices;
    if (reduction_block < 0)
        throw std::out_of_range("attention query IW reduction block");
    return slices[static_cast<std::size_t>(
        reduction_block % static_cast<int64_t>(slices.size()))];
}

llvm::SmallVector<int64_t> LPUTargetModel::attention_weight_slices() const
{
    llvm::SmallVector<int64_t> slices;
    for (int64_t index = 0; index < memory_.w8a16_weight_slice_count; ++index)
        slices.push_back(memory_.w8a16_weight_slice_base
            + index * memory_.w8a16_weight_slice_stride);
    return slices;
}

llvm::SmallVector<int64_t>
LPUTargetModel::attention_output_weight_slices() const
{
    llvm::SmallVector<int64_t> slices = attention_weight_slices();
    if (slices.size() >= 6) {
        slices[4] = memory_.w8a16_output_weight_spare_slices[0];
        slices[5] = memory_.w8a16_output_weight_spare_slices[1];
    }
    llvm::SmallDenseSet<int64_t, 8> unique;
    for (int64_t slice : slices)
        if (!unique.insert(slice).second)
            throw std::logic_error(
                "output weight spare slices must preserve unique lanes");
    return slices;
}

llvm::SmallVector<int64_t>
LPUTargetModel::attention_activation_slices() const
{
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        return llvm::SmallVector<int64_t>(storage.begin(),
            storage.begin() + throughput_.mxm_activation_streams);
    }
    llvm::SmallVector<int64_t> slices;
    for (int64_t index = 0; index < throughput_.mxm_activation_streams; ++index)
        slices.push_back(memory_.w8a16_activation_slice_base + index);
    return slices;
}

llvm::SmallVector<int64_t>
LPUTargetModel::mxm_distributed_activation_slices() const
{
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        return llvm::SmallVector<int64_t>(
            storage.begin(), storage.begin() + 16);
    }
    llvm::SmallDenseSet<int64_t, 16> reserved;
    for (int64_t slice : attention_weight_slices())
        reserved.insert(slice);
    for (int64_t slice : attention_output_weight_slices())
        reserved.insert(slice);
    for (int64_t slice : attention_result_slices())
        reserved.insert(slice);
    for (int64_t slice : attention_activation_slices())
        reserved.insert(slice);
    for (int64_t index = 0;
         index < throughput_.mxm_result_streams; ++index)
        reserved.insert(memory_.w8a16_result_slice_base + index);

    llvm::SmallVector<int64_t> slices;
    for (int64_t slice = memory_.slices_per_hemisphere - 1;
         slice >= 0 && slices.size() < 16; --slice) {
        if (!reserved.contains(slice)) slices.push_back(slice);
    }
    llvm::reverse(slices);
    return slices;
}

llvm::SmallVector<int64_t>
LPUTargetModel::ffn_projection_weight_slices(
    FfnProjectionKind kind) const
{
    if (kind == FfnProjectionKind::Gate)
        return attention_weight_slices();

    llvm::SmallDenseSet<int64_t, 32> reserved;
    for (int64_t slice : mxm_distributed_activation_slices())
        reserved.insert(slice);
    for (int64_t slice : attention_weight_slices())
        reserved.insert(slice);
    for (int64_t slice : memory_.w8a16_fused_gate_temp_slices)
        reserved.insert(slice);
    for (int64_t slice : memory_.w8a16_fused_up_temp_slices)
        reserved.insert(slice);
    llvm::SmallVector<int64_t> slices;
    const auto candidates = weight_storage_slices();
    for (int64_t slice : candidates) {
        if (slices.size()
            >= static_cast<std::size_t>(
                memory_.w8a16_weight_slice_count))
            break;
        if (!reserved.contains(slice)) slices.push_back(slice);
    }
    return slices;
}

llvm::SmallVector<int64_t> LPUTargetModel::ffn_hidden_slices() const
{
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        return llvm::SmallVector<int64_t>(storage.begin(),
            storage.begin() + throughput_.mxm_activation_streams);
    }
    llvm::SmallDenseSet<int64_t, 32> reserved;
    for (int64_t slice : attention_weight_slices())
        reserved.insert(slice);
    for (int64_t slice : attention_output_weight_slices())
        reserved.insert(slice);
    for (int64_t slice : attention_activation_slices())
        reserved.insert(slice);
    for (int64_t slice : mxm_distributed_activation_slices())
        reserved.insert(slice);
    for (int64_t index = 0; index < throughput_.mxm_result_streams; ++index)
        reserved.insert(memory_.w8a16_result_slice_base + index);
    for (int64_t slice : memory_.w8a16_fused_gate_temp_slices)
        reserved.insert(slice);
    for (int64_t slice : memory_.w8a16_fused_up_temp_slices)
        reserved.insert(slice);

    llvm::SmallVector<int64_t> slices;
    for (int64_t offset = 0;
         offset < memory_.slices_per_hemisphere
         && slices.size()
             < static_cast<std::size_t>(
                 throughput_.mxm_activation_streams);
         ++offset) {
        const int64_t slice =
            (memory_.w8a16_hidden_slice_base + offset)
            % memory_.slices_per_hemisphere;
        if (!reserved.contains(slice)) slices.push_back(slice);
    }
    return slices;
}

llvm::SmallVector<int64_t>
LPUTargetModel::attention_projection_output_slices() const
{
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        return llvm::SmallVector<int64_t>(storage.begin(),
            storage.begin() + throughput_.mxm_result_streams);
    }
    llvm::SmallVector<int64_t> slices;
    for (int64_t index = 0; index < throughput_.mxm_result_streams; ++index)
        slices.push_back(index);
    return slices;
}

llvm::SmallVector<int64_t> LPUTargetModel::ffn_gate_temp_slices() const
{
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        return llvm::SmallVector<int64_t>(storage.begin(),
            storage.begin() + std::min<std::size_t>(8, storage.size()));
    }
    return llvm::SmallVector<int64_t>(
        memory_.w8a16_fused_gate_temp_slices.begin(),
        memory_.w8a16_fused_gate_temp_slices.end());
}

llvm::SmallVector<int64_t> LPUTargetModel::ffn_up_temp_slices() const
{
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        const std::size_t begin = std::min<std::size_t>(8, storage.size());
        const std::size_t end = std::min<std::size_t>(16, storage.size());
        return llvm::SmallVector<int64_t>(
            storage.begin() + begin, storage.begin() + end);
    }
    return llvm::SmallVector<int64_t>(
        memory_.w8a16_fused_up_temp_slices.begin(),
        memory_.w8a16_fused_up_temp_slices.end());
}

llvm::SmallVector<int64_t> LPUTargetModel::attention_qk_key_slices() const
{
    llvm::SmallDenseSet<int64_t, 32> queryIwSlices;
    for (int64_t reduction = 0; reduction < 2; ++reduction) {
        const auto& slices = attention_query_iw_slices(reduction);
        for (int64_t slice : slices) queryIwSlices.insert(slice);
    }

    const int64_t count = throughput_.mxm_result_streams;
    const auto candidates = uses_dedicated_slice_roles()
        ? activation_storage_slices()
        : llvm::to_vector(llvm::seq<int64_t>(
              0, memory_.slices_per_hemisphere));
    for (auto begin = candidates.begin(); begin != candidates.end(); ++begin) {
        if (std::distance(begin, candidates.end()) < count) break;
        bool available = true;
        for (int64_t offset = 0; offset < count; ++offset) {
            available &= begin[offset] == begin[0] + offset;
            available &= !queryIwSlices.contains(begin[offset]);
        }
        if (!available) continue;

        llvm::SmallVector<int64_t> slices;
        for (int64_t offset = 0; offset < count; ++offset)
            slices.push_back(begin[offset]);
        return slices;
    }
    throw std::logic_error(
        "target cannot allocate QK key slices disjoint from query IW");
}

llvm::SmallVector<int64_t> LPUTargetModel::attention_value_slices() const
{
    llvm::SmallVector<int64_t> slices;
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        slices.append(storage.begin(), storage.begin() + 16);
        slices.append(storage.begin(), storage.begin() + 16);
        return slices;
    }
    if (throughput_.mxms_per_hemisphere == 1) {
        const auto& shared = attention_query_iw_slices(1);
        slices.append(shared.begin(), shared.end());
        slices.append(shared.begin(), shared.end());
        return slices;
    }
    static constexpr std::array<std::array<int64_t, 16>, 2>
        kDualMxmValueSlices {{
            {{0, 1, 2, 3, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 32, 33}},
            {{18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 34, 35}},
    }};
    for (const auto& blockSlices : kDualMxmValueSlices)
        slices.append(blockSlices.begin(), blockSlices.end());
    return slices;
}

llvm::SmallVector<int64_t>
LPUTargetModel::page_resident_attention_weight_slices() const
{
    llvm::SmallDenseSet<int64_t, 64> reserved;
    for (FfnProjectionKind kind : {
             FfnProjectionKind::Gate, FfnProjectionKind::Up})
        for (int64_t slice : ffn_projection_weight_slices(kind))
            reserved.insert(slice);
    for (int64_t slice : ffn_down_projection_weight_slices())
        reserved.insert(slice);

    llvm::SmallVector<int64_t> slices;
    const auto candidates = weight_storage_slices();
    for (int64_t slice : candidates) {
        if (slices.size()
            >= static_cast<std::size_t>(
                memory_.w8a16_weight_slice_count))
            break;
        if (!reserved.contains(slice)) slices.push_back(slice);
    }
    if (slices.size()
        != static_cast<std::size_t>(memory_.w8a16_weight_slice_count))
        throw std::logic_error(
            "target cannot allocate a page-resident attention weight plane");
    return slices;
}

llvm::SmallVector<int64_t>
LPUTargetModel::ffn_down_projection_weight_slices() const
{
    llvm::SmallDenseSet<int64_t, 32> reserved;
    for (FfnProjectionKind kind : {
             FfnProjectionKind::Gate, FfnProjectionKind::Up})
        for (int64_t slice : ffn_projection_weight_slices(kind))
            reserved.insert(slice);

    const int64_t count = memory_.w8a16_weight_slice_count;
    llvm::SmallVector<int64_t> slices;
    const auto candidates = weight_storage_slices();
    for (int64_t slice : candidates) {
        if (static_cast<int64_t>(slices.size()) >= count) break;
        if (!reserved.contains(slice)) slices.push_back(slice);
    }
    if (static_cast<int64_t>(slices.size()) != count)
        throw std::logic_error(
            "target cannot allocate an independent FFN down-weight plane");
    return slices;
}

llvm::SmallVector<int64_t> LPUTargetModel::attention_rope_slices() const
{
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        return llvm::SmallVector<int64_t>(storage.end() - 4, storage.end());
    }
    const auto result = attention_result_slices();
    return {memory_.w8a16_output_weight_spare_slices[0],
        memory_.w8a16_output_weight_spare_slices[1],
        result[result.size() - 2], result.back()};
}

llvm::SmallVector<int64_t>
LPUTargetModel::attention_rope_staging_slices() const
{
    if (uses_dedicated_slice_roles())
        return mxm_distributed_activation_slices();
    llvm::SmallDenseSet<int64_t, 64> reserved;
    const auto reserve = [&](llvm::ArrayRef<int64_t> slices) {
        reserved.insert(slices.begin(), slices.end());
    };
    reserve(attention_weight_slices());
    reserve(attention_output_weight_slices());
    reserve(attention_rope_slices());
    reserve(attention_result_slices());
    reserve(attention_activation_slices());
    reserve(mxm_distributed_activation_slices());
    for (int64_t stream = 0; stream < throughput_.mxm_result_streams;
         ++stream)
        reserved.insert(memory_.w8a16_result_slice_base + stream);

    llvm::SmallVector<int64_t> slices;
    for (int64_t slice = 0;
         slice < memory_.slices_per_hemisphere
         && slices.size() < 16;
         ++slice) {
        if (!reserved.contains(slice)) slices.push_back(slice);
    }
    if (slices.size() != 16)
        throw std::logic_error(
            "target cannot allocate a 16-slice RoPE staging FIFO");
    return slices;
}

llvm::SmallVector<int64_t> LPUTargetModel::attention_context_slices() const
{
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        return llvm::SmallVector<int64_t>(storage.end() - 8, storage.end());
    }
    if (memory_.slices_per_hemisphere < 8)
        throw std::logic_error(
            "target cannot allocate the attention context layout");
    llvm::SmallVector<int64_t> slices;
    for (int64_t slice = memory_.slices_per_hemisphere - 8;
         slice < memory_.slices_per_hemisphere; ++slice)
        slices.push_back(slice);
    return slices;
}

llvm::SmallVector<int64_t>
LPUTargetModel::attention_output_activation_slices(
    bool page_resident_weights) const
{
    if (uses_dedicated_slice_roles())
        return mxm_distributed_activation_slices();
    // RoPE staging is dead before O projection. Reusing its low slices lets
    // westbound context streams tap local MEM and continue to the passive
    // inter-hemisphere bridge without a VXM pass.
    if (!page_resident_weights) return attention_rope_staging_slices();

    llvm::SmallDenseSet<int64_t, 16> reserved;
    for (int64_t slice : page_resident_attention_weight_slices())
        reserved.insert(slice);
    llvm::SmallVector<int64_t> slices;
    const int64_t contextBegin = memory_.slices_per_hemisphere - 8;
    for (int64_t slice = 0;
         slice < contextBegin && slices.size() < 16; ++slice)
        if (!reserved.contains(slice)) slices.push_back(slice);
    if (slices.size() != 16)
        throw std::logic_error(
            "target cannot allocate paged O-projection activation slices disjoint from weights");
    return slices;
}

llvm::SmallVector<int64_t> LPUTargetModel::attention_result_slices() const
{
    if (uses_dedicated_slice_roles()) {
        const auto storage = activation_storage_slices();
        return llvm::SmallVector<int64_t>(storage.begin(),
            storage.begin() + throughput_.mxm_result_streams);
    }
    return {28, 29, 30, 31};
}

int64_t LPUTargetModel::attention_query_iw_base_row() const
{
    return 7600;
}

int64_t LPUTargetModel::attention_score_base_row() const
{
    return 3000;
}

int64_t LPUTargetModel::attention_probability_pack_base_row() const
{
    return 6000;
}

int64_t LPUTargetModel::attention_probability_diagonal_base_row() const
{
    return 7000;
}

int64_t LPUTargetModel::attention_value_base_row() const
{
    return 7800;
}

int64_t LPUTargetModel::attention_mask_base_row() const
{
    return memory_.attention_mask_base_row;
}

int64_t LPUTargetModel::attention_context_base_row() const
{
    return 2000;
}

bool LPUTargetModel::supports_w8a16_ffn_shape(
    int64_t m, int64_t k, int64_t hidden, int64_t n) const
{
    const int64_t tile = throughput_.mxm_rows;
    const int64_t output_pair = tile * throughput_.mxms_per_hemisphere;
    return m > 0 && m % tile == 0 && k > 0 && hidden > 0 && n > 0
        && k % tile == 0
        && hidden % output_pair == 0
        && n % output_pair == 0;
}

std::optional<int64_t> LPUTargetModel::transport_latency(StreamEndpoint source,
    StreamEndpoint destination, StreamDirection direction, int64_t mem_slice) const
{
    if (!supports_route(source, destination, direction)
        || mem_slice < 0 || mem_slice >= memory_.slices_per_hemisphere)
        return std::nullopt;
    const int64_t group = mem_slice / streams_.mem_slices_per_register_group;
    if (source == StreamEndpoint::MxmResult
        && destination == StreamEndpoint::VxmInput)
        return throughput_.accumulator_to_vxm_latency;
    if (source == StreamEndpoint::Mem
        && destination == StreamEndpoint::SxmInput)
        return throughput_.mem_to_sxm_latency - group;
    if (source == StreamEndpoint::VxmResult && destination == StreamEndpoint::MxmWeight)
        return 1;
    if (source == StreamEndpoint::VxmBridgeResult
        && destination == StreamEndpoint::Mem)
        return group;
    if (source == StreamEndpoint::SxmResult
        && destination == StreamEndpoint::Mem)
        return streams_.system_register_columns - 2 - group;
    if (source == StreamEndpoint::Mem && direction == StreamDirection::West)
        return group + 1;
    if (source == StreamEndpoint::Mem
        && (destination == StreamEndpoint::MxmActivation
            || destination == StreamEndpoint::MxmWeight)) {
        const auto sourceColumn =
            stream_source_column(source, direction, mem_slice);
        const auto destinationColumn =
            stream_destination_column(destination, direction, mem_slice);
        if (!sourceColumn || !destinationColumn) return std::nullopt;
        return std::max(*destinationColumn, *sourceColumn)
            - std::min(*destinationColumn, *sourceColumn) + 1;
    }
    if (source == StreamEndpoint::Mem)
        return streams_.system_register_columns - group;
    if (source == StreamEndpoint::VxmResult)
        return group + 1;
    return streams_.system_register_columns - 1 - group;
}

std::string_view LPUTargetModel::direction_name(StreamDirection direction)
{
    return direction == StreamDirection::East ? "east" : "west";
}

std::string_view LPUTargetModel::endpoint_name(StreamEndpoint endpoint)
{
    switch (endpoint) {
    case StreamEndpoint::Mem: return "MEM";
    case StreamEndpoint::MxmActivation: return "MXM.activation";
    case StreamEndpoint::MxmWeight: return "MXM.weight";
    case StreamEndpoint::MxmResult: return "MXM.result";
    case StreamEndpoint::VxmInput: return "VXM.input";
    case StreamEndpoint::VxmResult: return "VXM.result";
    case StreamEndpoint::VxmBridgeResult: return "VXM.bridge_result";
    case StreamEndpoint::SxmInput: return "SXM.input";
    case StreamEndpoint::SxmResult: return "SXM.result";
    }
    return "unknown";
}

} // namespace ftlpu::compiler::target
