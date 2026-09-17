module {
  func.func @qwen2_5_up_prefill_3d(
      %activation: tensor<32x1536xbf16>,
      %up_weight: tensor<1536x8960xi8>) -> tensor<32x8960xbf16> {
    %activation_route = ftlpu.stream.route %activation {
      stream_base = 16 : i64,
      stream_count = 4 : i64,
      register_id = 1 : i64,
      direction = "east",
      source = "MEM",
      destination = "MXM.activation",
      source_unit_id = -1 : i64,
      destination_unit_id = 0 : i64,
      address = {bank = 1 : i64, byte = 0 : i64, device = 0 : i64,
        hemisphere = "east", slice = 0 : i64, word = 0 : i64},
      placement = {address_stride = 1 : i64, bank = 1 : i64,
        base_row = 0 : i64, hemisphere = "both",
        instruction_count = 192 : i64,
        kind = "fp16_mxm_distributed_16",
        slices = [0, 1, 2, 3, 4, 5, 6, 7,
                  8, 9, 10, 11, 12, 13, 14, 15]},
      bytes = 98304 : i64,
      transport_latency = 15 : i64
    } : tensor<32x1536xbf16>
    %weight_route = ftlpu.stream.route %up_weight {
      stream_base = 8 : i64,
      stream_count = 8 : i64,
      register_id = 10 : i64,
      direction = "east",
      source = "MEM",
      destination = "MXM.weight",
      source_unit_id = -1 : i64,
      destination_unit_id = 0 : i64,
      address = {bank = 1 : i64, byte = 0 : i64, device = 0 : i64,
        hemisphere = "east", slice = 36 : i64, word = 0 : i64},
      placement = {address_stride = 1 : i64, bank = 1 : i64,
        base_row = 0 : i64, hemisphere = "both",
        instruction_count = 430080 : i64,
        kind = "w8a16_direct_3d",
        slices = [36, 37, 38, 39, 40, 41, 42, 43,
                  44, 45, 46, 47, 48, 49, 50, 51]},
      bytes = 13762560 : i64,
      transport_latency = 6 : i64
    } : tensor<1536x8960xi8>
    %up = ftlpu.stream.matmul_task[%activation_route], [%weight_route] {
      config = {
        projection_kind = "up",
        rhs_scale = 7.812500e-3 : f32,
        direct_3d = {
          accumulator_address_base = 32 : i64,
          result_stream_bases = [12, 20],
          weight_outer_group_size = 2 : i64,
          weight_regions = [
            {base_address = 0 : i64, bank = 1 : i64,
             first_pair = 0 : i64, first_slice = 36 : i64,
             pair_count = 42 : i64},
            {base_address = 0 : i64, bank = 1 : i64,
             first_pair = 42 : i64, first_slice = 44 : i64,
             pair_count = 42 : i64},
            {base_address = 0 : i64, bank = 0 : i64,
             first_pair = 84 : i64, first_slice = 36 : i64,
             pair_count = 42 : i64},
            {base_address = 0 : i64, bank = 0 : i64,
             first_pair = 126 : i64, first_slice = 44 : i64,
             pair_count = 14 : i64}
          ],
          timeline = {
            mxm_load_start_cycle = 32 : i64,
            mxm_dequant_start_cycle = 32 : i64,
            mxm_compute_start_cycle = 64 : i64,
            mxm_result_start_cycle = 1584 : i64,
            reduction_cycle_stride = 32 : i64,
            pair_cycle_stride = 1536 : i64
          }
        }
      },
      k = 1536 : i64,
      m = 32 : i64,
      n = 8960 : i64,
      result_allocations = [{
        address = {bank = 0 : i64, byte = 0 : i64, device = 0 : i64,
          hemisphere = "east", slice = 8 : i64, word = 0 : i64},
        bytes = 573440 : i64,
        placement = {address_stride = 1 : i64, bank = 0 : i64,
          base_row = 0 : i64, hemisphere = "both",
          instruction_count = 8960 : i64,
          kind = "fp16_pair_planar", slices = [8, 9]}
      }],
      result_stream_bases = [12],
      result_stream_counts = [4],
      unit_ids = [0],
      weight_buffers = [0]
    } : (tensor<32x1536xbf16>, tensor<1536x8960xi8>)
        -> tensor<32x8960xbf16>
    return %up : tensor<32x8960xbf16>
  }
}
