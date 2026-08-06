module {
  func.func @main() {
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 0 : i64, opcode = "read",
      address = 10 : i64, packed_stream = 0 : i64,
      repeat_count = 3 : i64, repeat_interval = 1 : i64,
      address_stride = 1 : i64
    }
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 0 : i64, opcode = "write",
      address = 20 : i64, packed_stream = 32 : i64,
      repeat_count = 3 : i64, repeat_interval = 1 : i64,
      address_stride = 2 : i64
    }
    ftlpu.command.mxm_dequant {
      cycle = 0 : i64, queue = 0 : i64, scale = 0.125 : f32,
      repeat_count = 1 : i64, repeat_interval = 1 : i64
    }
    ftlpu.command.mxm {
      cycle = 0 : i64, queue = 0 : i64, opcode = "iw",
      weight_buffer = 0 : i64, weight_column = 0 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 0 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      accumulator_address = 0 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram", accumulator_clear = false,
      weight_input_mode = "int8_dequant_bf16"
    }
    ftlpu.command.mxm {
      cycle = 1 : i64, queue = 0 : i64, opcode = "compute",
      weight_buffer = 0 : i64, weight_column = 0 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 0 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      accumulator_address = 0 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram", accumulator_clear = false,
      data_format = "bf16", compute_mode = "block8"
    }
    return
  }
}
