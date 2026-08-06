module {
  func.func @main() {
    ftlpu.command.mxm {
      cycle = 0 : i64,
      queue = 1 : i64,
      opcode = "iw",
      weight_buffer = 1 : i64,
      weight_column = 2 : i64,
      activation_stream_base = 0 : i64,
      output_stream_base = 0 : i64,
      repeat_count = 1 : i64,
      repeat_interval = 1 : i64,
      accumulator_address = 0 : i64,
      accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram",
      accumulator_clear = false,
      weight_load_mode = "column",
      weight_inner_column = 6 : i64
    }
    return
  }
}
