module {
  func.func @main() {
    ftlpu.command.vxm {
      cycle = 0 : i64,
      queue = 1 : i64,
      chain_depth = 2 : i64,
      opcode = "negate",
      lhs_kind = "previous",
      lhs_index = 0 : i64,
      lhs_immediate = 2.0 : f32,
      rhs_kind = "immediate",
      rhs_index = 0 : i64,
      rhs_immediate = 0.0 : f32,
      cast_target = "bf16",
      output_stream = -1 : i64,
      repeat_count = 2 : i64,
      repeat_interval = 1 : i64,
      input_hemisphere = "east",
      output_hemisphere = "east"
    }
    return
  }
}
