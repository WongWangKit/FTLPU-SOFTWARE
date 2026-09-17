module {
  func.func @mem_3d_group() {
    ftlpu.schedule.mem_transfer {
      address = 100 : i64,
      address_stride = 1 : i64,
      cycle = 10 : i64,
      group_address_stride = 32 : i64,
      group_count = 2 : i64,
      group_interval = 40 : i64,
      hemisphere = 0 : i64,
      opcode = "read",
      packed_stream = 0 : i64,
      repeat_count = 4 : i64,
      repeat_interval = 1 : i64,
      slice = 0 : i64,
      wave_address_stride = 8 : i64,
      wave_count = 3 : i64,
      wave_interval = 8 : i64
    }
    return
  }
}
