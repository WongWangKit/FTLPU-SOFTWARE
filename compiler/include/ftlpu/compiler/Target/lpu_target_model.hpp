#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
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
    SxmInput,
    SxmResult,
};

enum class FfnProjectionKind {
    Gate,
    Up,
};

struct MemoryTopology {
    int64_t hemispheres = 2;
    int64_t slices_per_hemisphere = 52;
    int64_t banks_per_slice = 2;
    int64_t words_per_bank = 32768;
    int64_t bytes_per_word = 32;
    int64_t sram_depth_rows = 32768;
    int64_t sram_read_ports_per_slice = 1;
    int64_t sram_write_ports_per_slice = 1;
    // When enabled, slices below w8a16_weight_slice_base are activation-only
    // and slices at or above that base are weight-only. MXM accumulators are
    // functional-unit-local storage and are not MEM slices.
    int64_t dedicated_slice_roles = 0;
    int64_t w8a16_weight_slice_count = 8;
    int64_t w8a16_weight_slice_base = 0;
    int64_t w8a16_weight_slice_stride = 4;
    std::array<int64_t, 2> w8a16_output_weight_spare_slices{{2, 18}};
    int64_t w8a16_activation_slice_base = 32;
    int64_t w8a16_hidden_slice_base = 21;
    int64_t w8a16_hidden_base_row = 0;
    int64_t attention_mask_base_row = 8128;
    int64_t w8a16_result_slice_base = 24;
    int64_t matmul_result_base_row = 1600;
    std::array<int64_t, 4> w8a16_fused_gate_temp_slices{{1, 5, 9, 29}};
    std::array<int64_t, 4> w8a16_fused_up_temp_slices{{2, 6, 10, 30}};
};

struct StreamTopology {
    int64_t streams_per_direction = 32;
    int64_t encoded_streams = 64;
    int64_t c2c_streams_per_direction = 8;
    int64_t c2c_bytes_per_stream_per_cycle = 32;
    int64_t mem_boundary_register_columns = 14;
    int64_t system_register_columns = 16;
    int64_t mem_slices_per_register_group = 4;
};

struct ThroughputModel {
    int64_t icu_repeat_2d_enabled = 1;
    int64_t tile_rows = 4;
    int64_t lanes_per_tile = 8;
    int64_t mem_read_bytes_per_cycle = 8;
    int64_t mem_write_bytes_per_cycle = 8;
    int64_t mxm_rows = 32;
    int64_t mxm_columns = 32;
    int64_t mxm_load_streams_per_cycle = 16;
    int64_t mxm_int8_load_streams_per_cycle = 8;
    int64_t mxm_load_bytes_per_cycle = 128;
    int64_t mxm_activation_streams = 4;
    int64_t mxm_result_streams = 4;
    int64_t mxm_pipeline_rows = 4;
    int64_t mxm_block_rows = 8;
    int64_t mxm_local_dequant_enabled = 1;
    int64_t mxm_block_compute_enabled = 1;
    int64_t mxm_weight_activation_overlap_enabled = 1;
    int64_t mxm_local_load_to_compute_latency = 4;
    int64_t mxm_block_group_interval = 8;
    int64_t mxm_earliest_iw_cycle = 2;
    int64_t qk_iw_to_compute_latency = 24;
    int64_t mxms_per_hemisphere = 1;
    int64_t mxm_weight_buffers = 2;
    int64_t mxm_accumulator_blocks = 256;
    int64_t vxm_alus = 16;
    int64_t vxm_weight_to_iw_latency = 16;
    int64_t mem_to_sxm_latency = 15;
    int64_t mem_to_mxm_latency = 16;
    int64_t mxm0_accumulator_latency = 6;
    int64_t mxm1_accumulator_latency = 5;
    int64_t accumulator_to_vxm_latency = 19;
    int64_t accumulator_read_to_vxm_latency = 16;
    int64_t swiglu_write_latency = 13;
};

class LPUTargetModel {
public:
    LPUTargetModel();
    LPUTargetModel(MemoryTopology memory, StreamTopology streams,
        ThroughputModel throughput);

    static mlir::FailureOr<LPUTargetModel> from_json(
        llvm::StringRef json, std::string& error);
    static mlir::FailureOr<LPUTargetModel> from_operation(
        mlir::Operation* operation);
    mlir::DictionaryAttr to_attribute(mlir::MLIRContext* context) const;
    mlir::LogicalResult validate(std::string* error = nullptr) const;

    const std::string& name() const { return name_; }
    std::uint64_t abi_fingerprint() const;
    const MemoryTopology& memory() const { return memory_; }
    const StreamTopology& streams() const { return streams_; }
    const ThroughputModel& throughput() const { return throughput_; }

    bool supports_route(StreamEndpoint source, StreamEndpoint destination,
        StreamDirection direction) const;
    std::optional<int64_t> stream_register_id(StreamEndpoint source,
        StreamEndpoint destination, StreamDirection direction, int64_t mem_slice) const;
    std::optional<int64_t> stream_source_column(StreamEndpoint source,
        StreamDirection direction, int64_t mem_slice) const;
    std::optional<int64_t> stream_destination_column(StreamEndpoint destination,
        StreamDirection direction, int64_t mem_slice) const;
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
    bool supports_mxm_local_dequant() const
    {
        return throughput_.mxm_local_dequant_enabled != 0;
    }
    bool supports_mxm_block8_compute() const
    {
        return throughput_.mxm_block_compute_enabled != 0;
    }
    bool supports_mxm_weight_activation_overlap() const
    {
        return throughput_.mxm_weight_activation_overlap_enabled != 0;
    }
    bool uses_dedicated_slice_roles() const
    {
        return memory_.dedicated_slice_roles != 0;
    }
    llvm::SmallVector<int64_t> weight_storage_slices() const;
    llvm::SmallVector<int64_t> activation_storage_slices() const;
    bool is_weight_storage_slice(int64_t slice) const;
    bool is_activation_storage_slice(int64_t slice) const;
    // Attention QK uses two physical 16-stream IW source layouts. Keep this
    // target-specific routing outside MemoryTopology so it cannot alter ABI.
    const std::array<int64_t, 16>& attention_query_iw_slices(
        int64_t reduction_block) const;
    llvm::SmallVector<int64_t> attention_weight_slices() const;
    llvm::SmallVector<int64_t> page_resident_attention_weight_slices(
        bool block8) const;
    llvm::SmallVector<int64_t> attention_output_weight_slices() const;
    llvm::SmallVector<int64_t> attention_activation_slices() const;
    llvm::SmallVector<int64_t> mxm_distributed_activation_slices() const;
    llvm::SmallVector<int64_t> ffn_projection_weight_slices(
        FfnProjectionKind kind) const;
    llvm::SmallVector<int64_t> ffn_down_projection_weight_slices(
        bool block8) const;
    llvm::SmallVector<int64_t> ffn_block8_result_slices() const;
    llvm::SmallVector<int64_t> ffn_block8_input_slices() const;
    llvm::SmallVector<int64_t> ffn_hidden_slices() const;
    llvm::SmallVector<int64_t> ffn_gate_temp_slices() const;
    llvm::SmallVector<int64_t> ffn_up_temp_slices() const;
    llvm::SmallVector<int64_t> attention_projection_output_slices() const;
    llvm::SmallVector<int64_t> attention_qk_key_slices() const;
    llvm::SmallVector<int64_t> attention_value_slices() const;
    llvm::SmallVector<int64_t> attention_rope_slices() const;
    llvm::SmallVector<int64_t> attention_rope_staging_slices() const;
    llvm::SmallVector<int64_t> attention_context_slices() const;
    llvm::SmallVector<int64_t> attention_output_activation_slices(
        bool page_resident_weights = false) const;
    llvm::SmallVector<int64_t> attention_block8_result_slices(
        bool page_resident_weights = false) const;
    llvm::SmallVector<int64_t> attention_result_slices() const;
    int64_t attention_query_iw_base_row() const;
    int64_t attention_score_base_row() const;
    int64_t attention_probability_pack_base_row() const;
    int64_t attention_probability_diagonal_base_row() const;
    int64_t attention_value_base_row() const;
    int64_t attention_mask_base_row() const;
    int64_t attention_context_base_row() const;
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
    bool is_valid_vxm_alu(int64_t alu) const
    {
        return alu >= 0 && alu < throughput_.vxm_alus;
    }

    static std::string_view direction_name(StreamDirection direction);
    static std::string_view endpoint_name(StreamEndpoint endpoint);

private:
    std::string name_{"lpu_32stream_v1"};
    MemoryTopology memory_;
    StreamTopology streams_;
    ThroughputModel throughput_;
};

} // namespace ftlpu::compiler::target
