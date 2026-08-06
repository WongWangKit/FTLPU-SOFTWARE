module {
  func.func @nested_mem_repeat() {
    ftlpu.schedule.mem_transfer {accumulator_destination = "sram", address = 100 : i64, address_stride = 1 : i64, cycle = 10 : i64, hemisphere = 0 : i64, opcode = "read", packed_stream = 0 : i64, repeat_count = 4 : i64, repeat_interval = 1 : i64, slice = 0 : i64}
    ftlpu.schedule.mem_transfer {accumulator_destination = "sram", address = 104 : i64, address_stride = 1 : i64, cycle = 138 : i64, hemisphere = 0 : i64, opcode = "read", packed_stream = 0 : i64, repeat_count = 4 : i64, repeat_interval = 1 : i64, slice = 0 : i64}
    ftlpu.schedule.mem_transfer {accumulator_destination = "sram", address = 108 : i64, address_stride = 1 : i64, cycle = 266 : i64, hemisphere = 0 : i64, opcode = "read", packed_stream = 0 : i64, repeat_count = 4 : i64, repeat_interval = 1 : i64, slice = 0 : i64}
    return
  }
  func.func @sxm_repeat() {
    ftlpu.schedule.sxm {cycle = 20 : i64, destination_streams = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], hemisphere = 0 : i64, opcode = "transpose", permute_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], source_streams = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], weight_layout = "vector_columns"}
    ftlpu.schedule.sxm {cycle = 22 : i64, destination_streams = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], hemisphere = 0 : i64, opcode = "transpose", permute_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], source_streams = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], weight_layout = "vector_columns"}
    ftlpu.schedule.sxm {cycle = 24 : i64, destination_streams = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], hemisphere = 0 : i64, opcode = "transpose", permute_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], source_streams = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], weight_layout = "vector_columns"}
    return
  }
}
