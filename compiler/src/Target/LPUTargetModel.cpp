#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/Support/JSON.h"
#include "mlir/IR/Builders.h"

#include <algorithm>
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
    LPUTargetModel model;

#define READ_MEMORY(field) \
    read_json_integer(memory, #field, &MemoryTopology::field, model.memory_)
    READ_MEMORY(hemispheres);
    READ_MEMORY(slices_per_hemisphere);
    READ_MEMORY(banks_per_slice);
    READ_MEMORY(words_per_bank);
    READ_MEMORY(bytes_per_word);
    READ_MEMORY(accumulator_slice_base);
    READ_MEMORY(accumulator_slices_per_mxm);
    READ_MEMORY(w8a16_weight_slice_count);
    READ_MEMORY(w8a16_weight_slice_stride);
    READ_MEMORY(w8a16_activation_slice_base);
    READ_MEMORY(w8a16_hidden_slice_base);
    READ_MEMORY(w8a16_result_slice_base);
    READ_MEMORY(accumulator_scratch_base_row);
#undef READ_MEMORY
    read_json_array(memory, "w8a16_fused_gate_temp_slices",
        model.memory_.w8a16_fused_gate_temp_slices);
    read_json_array(memory, "w8a16_fused_up_temp_slices",
        model.memory_.w8a16_fused_up_temp_slices);

#define READ_STREAM(field) \
    read_json_integer(streams, #field, &StreamTopology::field, model.streams_)
    READ_STREAM(streams_per_direction);
    READ_STREAM(encoded_streams);
    READ_STREAM(mem_boundary_register_columns);
    READ_STREAM(system_register_columns);
    READ_STREAM(mem_slices_per_register_group);
#undef READ_STREAM

#define READ_THROUGHPUT(field) \
    read_json_integer(throughput, #field, &ThroughputModel::field, model.throughput_)
    READ_THROUGHPUT(tile_rows);
    READ_THROUGHPUT(lanes_per_tile);
    READ_THROUGHPUT(mem_read_bytes_per_cycle);
    READ_THROUGHPUT(mem_write_bytes_per_cycle);
    READ_THROUGHPUT(mxm_rows);
    READ_THROUGHPUT(mxm_columns);
    READ_THROUGHPUT(mxm_load_streams_per_cycle);
    READ_THROUGHPUT(mxm_load_bytes_per_cycle);
    READ_THROUGHPUT(mxm_activation_streams);
    READ_THROUGHPUT(mxm_result_streams);
    READ_THROUGHPUT(mxm_pipeline_rows);
    READ_THROUGHPUT(mxm_earliest_iw_cycle);
    READ_THROUGHPUT(mxms_per_hemisphere);
    READ_THROUGHPUT(mxm_weight_buffers);
    READ_THROUGHPUT(vxm_alus);
    READ_THROUGHPUT(vxm_weight_to_iw_latency);
    READ_THROUGHPUT(mem_to_sxm_latency);
    READ_THROUGHPUT(mem_to_mxm_latency);
    READ_THROUGHPUT(mxm0_accumulator_latency);
    READ_THROUGHPUT(mxm1_accumulator_latency);
    READ_THROUGHPUT(accumulator_to_vxm_latency);
    READ_THROUGHPUT(swiglu_write_latency);
#undef READ_THROUGHPUT

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

    LPUTargetModel model;
    const auto memory = target.getAs<mlir::DictionaryAttr>("memory");
    const auto streams = target.getAs<mlir::DictionaryAttr>("streams");
    const auto throughput = target.getAs<mlir::DictionaryAttr>("throughput");
#define READ_MEMORY(field) \
    read_attr_integer(memory, #field, &MemoryTopology::field, model.memory_)
    READ_MEMORY(hemispheres);
    READ_MEMORY(slices_per_hemisphere);
    READ_MEMORY(banks_per_slice);
    READ_MEMORY(words_per_bank);
    READ_MEMORY(bytes_per_word);
    READ_MEMORY(accumulator_slice_base);
    READ_MEMORY(accumulator_slices_per_mxm);
    READ_MEMORY(w8a16_weight_slice_count);
    READ_MEMORY(w8a16_weight_slice_stride);
    READ_MEMORY(w8a16_activation_slice_base);
    READ_MEMORY(w8a16_hidden_slice_base);
    READ_MEMORY(w8a16_result_slice_base);
    READ_MEMORY(accumulator_scratch_base_row);
#undef READ_MEMORY
    read_attr_array(memory, "w8a16_fused_gate_temp_slices",
        model.memory_.w8a16_fused_gate_temp_slices);
    read_attr_array(memory, "w8a16_fused_up_temp_slices",
        model.memory_.w8a16_fused_up_temp_slices);
#define READ_STREAM(field) \
    read_attr_integer(streams, #field, &StreamTopology::field, model.streams_)
    READ_STREAM(streams_per_direction);
    READ_STREAM(encoded_streams);
    READ_STREAM(mem_boundary_register_columns);
    READ_STREAM(system_register_columns);
    READ_STREAM(mem_slices_per_register_group);
#undef READ_STREAM
#define READ_THROUGHPUT(field) \
    read_attr_integer(throughput, #field, &ThroughputModel::field, model.throughput_)
    READ_THROUGHPUT(tile_rows);
    READ_THROUGHPUT(lanes_per_tile);
    READ_THROUGHPUT(mem_read_bytes_per_cycle);
    READ_THROUGHPUT(mem_write_bytes_per_cycle);
    READ_THROUGHPUT(mxm_rows);
    READ_THROUGHPUT(mxm_columns);
    READ_THROUGHPUT(mxm_load_streams_per_cycle);
    READ_THROUGHPUT(mxm_load_bytes_per_cycle);
    READ_THROUGHPUT(mxm_activation_streams);
    READ_THROUGHPUT(mxm_result_streams);
    READ_THROUGHPUT(mxm_pipeline_rows);
    READ_THROUGHPUT(mxm_earliest_iw_cycle);
    READ_THROUGHPUT(mxms_per_hemisphere);
    READ_THROUGHPUT(mxm_weight_buffers);
    READ_THROUGHPUT(vxm_alus);
    READ_THROUGHPUT(vxm_weight_to_iw_latency);
    READ_THROUGHPUT(mem_to_sxm_latency);
    READ_THROUGHPUT(mem_to_mxm_latency);
    READ_THROUGHPUT(mxm0_accumulator_latency);
    READ_THROUGHPUT(mxm1_accumulator_latency);
    READ_THROUGHPUT(accumulator_to_vxm_latency);
    READ_THROUGHPUT(swiglu_write_latency);
#undef READ_THROUGHPUT
    std::string error;
    if (mlir::failed(model.validate(&error))) {
        operation->emitError("invalid ftlpu.target configuration: ") << error;
        return mlir::failure();
    }
    return model;
}

mlir::DictionaryAttr LPUTargetModel::to_attribute(
    mlir::MLIRContext* context) const
{
    mlir::Builder builder(context);
#define I64(object, field) \
    builder.getNamedAttr(#field, builder.getI64IntegerAttr(object.field))
    const auto memory = builder.getDictionaryAttr({
        I64(memory_, hemispheres),
        I64(memory_, slices_per_hemisphere),
        I64(memory_, banks_per_slice),
        I64(memory_, words_per_bank),
        I64(memory_, bytes_per_word),
        I64(memory_, accumulator_slice_base),
        I64(memory_, accumulator_slices_per_mxm),
        I64(memory_, w8a16_weight_slice_count),
        I64(memory_, w8a16_weight_slice_stride),
        I64(memory_, w8a16_activation_slice_base),
        I64(memory_, w8a16_hidden_slice_base),
        I64(memory_, w8a16_result_slice_base),
        I64(memory_, accumulator_scratch_base_row),
        builder.getNamedAttr("w8a16_fused_gate_temp_slices",
            make_i64_array(builder, memory_.w8a16_fused_gate_temp_slices)),
        builder.getNamedAttr("w8a16_fused_up_temp_slices",
            make_i64_array(builder, memory_.w8a16_fused_up_temp_slices)),
    });
    const auto streams = builder.getDictionaryAttr({
        I64(streams_, streams_per_direction),
        I64(streams_, encoded_streams),
        I64(streams_, mem_boundary_register_columns),
        I64(streams_, system_register_columns),
        I64(streams_, mem_slices_per_register_group),
    });
    const auto throughput = builder.getDictionaryAttr({
        I64(throughput_, tile_rows),
        I64(throughput_, lanes_per_tile),
        I64(throughput_, mem_read_bytes_per_cycle),
        I64(throughput_, mem_write_bytes_per_cycle),
        I64(throughput_, mxm_rows),
        I64(throughput_, mxm_columns),
        I64(throughput_, mxm_load_streams_per_cycle),
        I64(throughput_, mxm_load_bytes_per_cycle),
        I64(throughput_, mxm_activation_streams),
        I64(throughput_, mxm_result_streams),
        I64(throughput_, mxm_pipeline_rows),
        I64(throughput_, mxm_earliest_iw_cycle),
        I64(throughput_, mxms_per_hemisphere),
        I64(throughput_, mxm_weight_buffers),
        I64(throughput_, vxm_alus),
        I64(throughput_, vxm_weight_to_iw_latency),
        I64(throughput_, mem_to_sxm_latency),
        I64(throughput_, mem_to_mxm_latency),
        I64(throughput_, mxm0_accumulator_latency),
        I64(throughput_, mxm1_accumulator_latency),
        I64(throughput_, accumulator_to_vxm_latency),
        I64(throughput_, swiglu_write_latency),
    });
#undef I64
    return builder.getDictionaryAttr({
        builder.getNamedAttr("memory", memory),
        builder.getNamedAttr("streams", streams),
        builder.getNamedAttr("throughput", throughput),
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
    if (!positive({memory_.hemispheres, memory_.slices_per_hemisphere,
            memory_.banks_per_slice, memory_.words_per_bank,
            memory_.bytes_per_word, streams_.streams_per_direction,
            streams_.encoded_streams, streams_.mem_slices_per_register_group,
            throughput_.tile_rows, throughput_.lanes_per_tile,
            throughput_.mxm_rows, throughput_.mxm_columns,
            throughput_.mxm_load_streams_per_cycle,
            throughput_.mxm_activation_streams,
            throughput_.mxm_result_streams, throughput_.mxms_per_hemisphere,
            throughput_.mxm_weight_buffers, throughput_.vxm_alus}))
        return fail("topology dimensions and throughput values must be positive");
    if (streams_.encoded_streams
        < 2 * streams_.streams_per_direction)
        return fail("encoded_streams must cover east and west stream ranges");
    if (streams_.system_register_columns
        < streams_.mem_boundary_register_columns)
        return fail("system register columns must cover MEM boundary columns");
    if (throughput_.mxm_rows % throughput_.tile_rows != 0
        || throughput_.mxm_columns % throughput_.lanes_per_tile != 0)
        return fail("MXM dimensions must be divisible by tile geometry");
    if (throughput_.mxm_load_streams_per_cycle
            > streams_.streams_per_direction
        || throughput_.mxm_activation_streams
            > streams_.streams_per_direction
        || throughput_.mxm_result_streams
            > streams_.streams_per_direction)
        return fail("functional-unit stream width exceeds directional streams");
    const auto valid_slice = [&](int64_t slice) {
        return slice >= 0 && slice < memory_.slices_per_hemisphere;
    };
    if (!valid_slice(memory_.accumulator_slice_base)
        || !valid_slice(memory_.w8a16_activation_slice_base)
        || !valid_slice(memory_.w8a16_hidden_slice_base)
        || !valid_slice(memory_.w8a16_result_slice_base))
        return fail("configured MEM slice base is outside a hemisphere");
    for (int64_t slice : memory_.w8a16_fused_gate_temp_slices)
        if (!valid_slice(slice)) return fail("gate temporary slice is invalid");
    for (int64_t slice : memory_.w8a16_fused_up_temp_slices)
        if (!valid_slice(slice)) return fail("up temporary slice is invalid");
    return mlir::success();
}

bool LPUTargetModel::supports_route(StreamEndpoint source, StreamEndpoint destination,
    StreamDirection direction) const
{
    const bool mxm_input = destination == StreamEndpoint::MxmActivation
        || destination == StreamEndpoint::MxmWeight;
    if (source == StreamEndpoint::Mem && mxm_input)
        return direction == StreamDirection::East;
    if (source == StreamEndpoint::Mem && destination == StreamEndpoint::VxmInput)
        return direction == StreamDirection::East || direction == StreamDirection::West;
    if (source == StreamEndpoint::VxmResult && destination == StreamEndpoint::MxmWeight)
        return direction == StreamDirection::East;
    if (source == StreamEndpoint::MxmResult && destination == StreamEndpoint::Mem)
        return direction == StreamDirection::West;
    if (source == StreamEndpoint::VxmResult && destination == StreamEndpoint::Mem)
        return direction == StreamDirection::East;
    return false;
}

std::optional<int64_t> LPUTargetModel::route_stream_count(StreamEndpoint source,
    StreamEndpoint destination, StreamDirection direction) const
{
    if (!supports_route(source, destination, direction)) return std::nullopt;
    if (source == StreamEndpoint::Mem && destination == StreamEndpoint::VxmInput)
        return throughput_.lanes_per_tile;
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
    if (source == StreamEndpoint::MxmResult && destination == StreamEndpoint::Mem)
        return divide_ceil(bytes, vector_bytes * throughput_.mxm_result_streams);
    if (source == StreamEndpoint::VxmResult && destination == StreamEndpoint::Mem)
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

const std::array<int64_t, 16>& LPUTargetModel::attention_query_iw_slices(
    int64_t reduction_block) const
{
    static constexpr std::array<std::array<int64_t, 16>, 2> kSlices {{
        {{0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33}},
        {{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35}},
    }};
    if (reduction_block < 0 || reduction_block >= static_cast<int64_t>(kSlices.size()))
        throw std::out_of_range("attention query IW reduction block");
    return kSlices[static_cast<std::size_t>(reduction_block)];
}

int64_t LPUTargetModel::attention_query_iw_base_row() const
{
    return 7600;
}

int64_t LPUTargetModel::attention_score_base_row() const
{
    return 3000;
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
    if (source == StreamEndpoint::VxmResult && destination == StreamEndpoint::MxmWeight)
        return 1;
    if (source == StreamEndpoint::Mem && direction == StreamDirection::West)
        return group + 2;
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
    }
    return "unknown";
}

} // namespace ftlpu::compiler::target
