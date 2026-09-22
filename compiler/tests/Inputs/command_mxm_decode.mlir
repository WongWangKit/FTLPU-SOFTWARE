module attributes {ftlpu.icu_compression = "macro"} {
  func.func @main() {
    // Keep a real program origin at cycle zero. The raw Macro frontend needs
    // decode lookahead before the first MXM operation becomes executable.
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 0 : i64, opcode = "read",
      address = 0 : i64, packed_stream = 0 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      address_stride = 0 : i64
    }
    ftlpu.command.mxm {
      cycle = 8 : i64, queue = 0 : i64,
      opcode = "decode_load_activation",
      weight_buffer = 1 : i64, weight_column = 0 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 0 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      accumulator_address = 0 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram", accumulator_clear = false,
      data_format = "bf16", decode_layout = "native_4x4"
    }
    ftlpu.command.mxm_dequant {
      cycle = 16 : i64, queue = 0 : i64, scale = 0.125 : f32,
      repeat_count = 2 : i64, repeat_interval = 1 : i64
    }
    ftlpu.command.mxm {
      cycle = 16 : i64, queue = 0 : i64,
      opcode = "decode_stream_compute",
      weight_buffer = 1 : i64, weight_column = 2 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 2 : i64,
      repeat_count = 2 : i64, repeat_interval = 1 : i64,
      accumulator_address = 64 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram", accumulator_clear = false,
      data_format = "bf16", decode_layout = "native_4x4"
    }
    ftlpu.command.mxm {
      cycle = 24 : i64, queue = 0 : i64,
      opcode = "accumulator_read",
      weight_buffer = 0 : i64, weight_column = 0 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 0 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      accumulator_address = 64 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "stream", accumulator_clear = true,
      data_format = "bf16", accumulator_output_format = "bf16"
    }
    return
  }
}
