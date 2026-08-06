module {
  func.func @main() {
    ftlpu.command.binding {
      access = "input",
      bytes = 64 : i64,
      element_type = "bf16",
      index = 0 : i64,
      initializer = "none",
      initializer_config = {},
      name = "activation",
      placement = {
        address_stride = 1 : i64,
        base_row = 0 : i64,
        hemisphere = "east",
        instruction_count = 1 : i64,
        kind = "fp16_pair_planar",
        slices = [0, 1]
      },
      ready_cycle = 0 : i64,
      role = "activation",
      shape = [1, 32]
    }
    ftlpu.command.mxm {
      accumulator_address = 0 : i64,
      accumulator_clear = true,
      accumulator_destination = "sram",
      accumulator_row_stride = 1 : i64,
      activation_stream_base = 0 : i64,
      cycle = 0 : i64,
      data_format = "bf16",
      opcode = "compute",
      output_stream_base = 0 : i64,
      queue = 0 : i64,
      repeat_count = 1 : i64,
      repeat_interval = 1 : i64,
      weight_buffer = 0 : i64,
      weight_column = 0 : i64
    }
    ftlpu.command.vxm {
      cast_target = "bf16",
      cycle = 0 : i64,
      input_hemisphere = "east",
      lhs_immediate = 0.0 : f32,
      lhs_index = 32 : i64,
      lhs_kind = "stream_bf16",
      opcode = "cast",
      output_hemisphere = "east",
      output_stream = 40 : i64,
      queue = 0 : i64,
      repeat_count = 1 : i64,
      repeat_interval = 1 : i64,
      rhs_immediate = 0.0 : f32,
      rhs_index = 0 : i64,
      rhs_kind = "immediate"
    }
    return
  }
}
