module {
  func.func @main() {
    ftlpu.command.vxm {
      cycle = 0 : i64,
      queue = 3 : i64,
      opcode = "multiply",
      lhs_kind = "stream_i32",
      lhs_index = 32 : i64,
      lhs_immediate = 0.0 : f32,
      rhs_kind = "immediate",
      rhs_index = 0 : i64,
      rhs_immediate = 5.000000e-01 : f32,
      cast_target = "fp32",
      output_stream = 40 : i64,
      repeat_count = 2 : i64,
      repeat_interval = 1 : i64,
      input_hemisphere = "east",
      output_hemisphere = "east"
    }
    return
  }
}
