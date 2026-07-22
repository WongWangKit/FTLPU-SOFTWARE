#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ftlpu::compiler::target {

enum class StreamDirection {
    East,
    West,
};

enum class StreamEndpoint {
    Mem,
    MxmActivation,
    MxmWeight,
    MxmResult,
    VxmInput,
    VxmResult,
};

struct MemoryTopology {
    int64_t hemispheres = 2;
    int64_t slices_per_hemisphere = 44;
    int64_t banks_per_slice = 2;
    int64_t words_per_bank = 4096;
    int64_t bytes_per_word = 16;
    int64_t accumulator_slice_base = 36;
    int64_t accumulator_slices_per_mxm = 4;
    int64_t w8a16_weight_slice_count = 8;
    int64_t w8a16_weight_slice_stride = 4;
    int64_t w8a16_activation_slice_base = 32;
    int64_t w8a16_hidden_slice_base = 21;
    int64_t w8a16_result_slice_base = 24;
    int64_t accumulator_scratch_base_row = 1600;
};

struct StreamTopology {
    int64_t streams_per_direction = 32;
    int64_t encoded_streams = 64;
    int64_t mem_boundary_register_columns = 12;
    int64_t system_register_columns = 13;
    int64_t mem_slices_per_register_group = 4;
};

struct ThroughputModel {
    int64_t tile_rows = 4;
    int64_t lanes_per_tile = 8;
    int64_t mem_read_bytes_per_cycle = 8;
    int64_t mem_write_bytes_per_cycle = 8;
    int64_t mxm_rows = 32;
    int64_t mxm_columns = 32;
    int64_t mxm_load_streams_per_cycle = 16;
    int64_t mxm_load_bytes_per_cycle = 128;
    int64_t mxm_activation_streams = 4;
    int64_t mxm_result_streams = 4;
    int64_t mxm_pipeline_rows = 4;
    int64_t mxm_earliest_iw_cycle = 2;
    int64_t mxms_per_hemisphere = 2;
    int64_t mxm_weight_buffers = 2;
    int64_t vxm_weight_to_iw_latency = 14;
    int64_t mem_to_sxm_latency = 12;
    int64_t mem_to_mxm_latency = 13;
    int64_t mxm0_accumulator_latency = 6;
    int64_t mxm1_accumulator_latency = 5;
    int64_t accumulator_to_vxm_latency = 16;
    int64_t swiglu_write_latency = 13;
};

class LPUTargetModel {
public:
    LPUTargetModel();
    const MemoryTopology& memory() const { return memory_; }
    const StreamTopology& streams() const { return streams_; }
    const ThroughputModel& throughput() const { return throughput_; }

    bool supports_route(StreamEndpoint source, StreamEndpoint destination,
        StreamDirection direction) const;
    std::optional<int64_t> stream_register_id(StreamEndpoint source,
        StreamEndpoint destination, StreamDirection direction, int64_t mem_slice) const;
    std::optional<int64_t> transport_latency(StreamEndpoint source,
        StreamEndpoint destination, StreamDirection direction, int64_t mem_slice) const;
    std::optional<int64_t> route_stream_count(StreamEndpoint source,
        StreamEndpoint destination, StreamDirection direction) const;
    std::optional<int64_t> route_issue_cycles(StreamEndpoint source,
        StreamEndpoint destination, int64_t bytes) const;
    int64_t mxm_compute_issue_cycles(int64_t rows) const;
    int64_t mxm_first_result_latency() const;
    int64_t mxm_result_window_cycles(int64_t rows) const;
    int64_t mxm_block_issue_interval() const;
    // Attention QK uses two physical 16-stream IW source layouts. Keep this
    // target-specific routing outside MemoryTopology so it cannot alter ABI.
    const std::array<int64_t, 16>& attention_query_iw_slices(
        int64_t reduction_block) const;
    int64_t attention_query_iw_base_row() const;
    int64_t attention_score_base_row() const;
    bool supports_w8a16_ffn_shape(int64_t m, int64_t k,
        int64_t hidden, int64_t n) const;
    int64_t mxm_earliest_iw_cycle() const { return throughput_.mxm_earliest_iw_cycle; }
    bool is_valid_mxm_unit(int64_t unit_id) const
    {
        return unit_id >= 0
            && unit_id < memory_.hemispheres * throughput_.mxms_per_hemisphere;
    }
    bool is_valid_weight_buffer(int64_t buffer) const
    {
        return buffer >= 0 && buffer < throughput_.mxm_weight_buffers;
    }
    bool is_valid_vxm_alu(int64_t alu) const { return alu >= 0 && alu < 16; }

    static std::string_view direction_name(StreamDirection direction);
    static std::string_view endpoint_name(StreamEndpoint endpoint);

private:
    MemoryTopology memory_;
    StreamTopology streams_;
    ThroughputModel throughput_;
};

} // namespace ftlpu::compiler::target
