#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include <stdexcept>

namespace ftlpu::compiler::target {
namespace {

static_assert(ThroughputModel{}.mxm_activation_streams == 4,
    "compiler target must match the 4x8 W8A16 CModel");

int64_t divide_ceil(int64_t numerator, int64_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

} // namespace

// Keep physical-layout defaults in one compiled model shared by all stages.
LPUTargetModel::LPUTargetModel() = default;

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
    return throughput_.mxm_rows + 2 * (throughput_.mxm_pipeline_rows - 1);
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
