module {
  func.func @nested_mem_repeat() {
    ftlpu.schedule.mem_transfer {accumulator_destination = "sram", address = 100 : i64, address_stride = 1 : i64, cycle = 10 : i64, hemisphere = 0 : i64, opcode = "read", packed_stream = 0 : i64, repeat_count = 4 : i64, repeat_interval = 1 : i64, slice = 0 : i64}
    ftlpu.schedule.mem_transfer {accumulator_destination = "sram", address = 104 : i64, address_stride = 1 : i64, cycle = 138 : i64, hemisphere = 0 : i64, opcode = "read", packed_stream = 0 : i64, repeat_count = 4 : i64, repeat_interval = 1 : i64, slice = 0 : i64}
    ftlpu.schedule.mem_transfer {accumulator_destination = "sram", address = 108 : i64, address_stride = 1 : i64, cycle = 266 : i64, hemisphere = 0 : i64, opcode = "read", packed_stream = 0 : i64, repeat_count = 4 : i64, repeat_interval = 1 : i64, slice = 0 : i64}
    return
  }
}
