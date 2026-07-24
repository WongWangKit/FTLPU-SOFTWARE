module {
  func.func @overlap() {
    ftlpu.schedule.mem_transfer {
      accumulator_destination = "sram",
      address = 0 : i64,
      address_stride = 0 : i64,
      cycle = 4 : i64,
      hemisphere = 0 : i64,
      opcode = "read",
      packed_stream = 0 : i64,
      repeat_count = 2 : i64,
      repeat_interval = 1 : i64,
      slice = 0 : i64
    }
    ftlpu.schedule.mem_transfer {
      accumulator_destination = "sram",
      address = 32 : i64,
      address_stride = 0 : i64,
      cycle = 5 : i64,
      hemisphere = 0 : i64,
      opcode = "write",
      packed_stream = 1 : i64,
      repeat_count = 1 : i64,
      repeat_interval = 1 : i64,
      slice = 0 : i64
    }
    return
  }
}
