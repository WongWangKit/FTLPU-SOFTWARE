module {
  func.func @main() {
    // These two operations share the same launch domain and interleave on one
    // MEM queue, so macro lowering may share their descriptor header.
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 0 : i64, opcode = "read",
      address = 100 : i64, packed_stream = 0 : i64,
      repeat_count = 3 : i64, repeat_interval = 2 : i64,
      address_stride = 1 : i64
    }
    ftlpu.command.mem {
      cycle = 1 : i64, queue = 0 : i64, opcode = "write",
      address = 200 : i64, packed_stream = 1 : i64,
      repeat_count = 3 : i64, repeat_interval = 2 : i64,
      address_stride = 2 : i64
    }
    return
  }
}
