module {
  func.func @mem_3d_blocked_outer() {
    ftlpu.schedule.mem_transfer {
      address = 100 : i64,
      address_stride = 1 : i64,
      cycle = 10 : i64,
      group_address_stride = 0 : i64,
      group_count = 5 : i64,
      group_interval = 40 : i64,
      hemisphere = 0 : i64,
      opcode = "read",
      outer_group_size = 2 : i64,
      outer_group_stride = 32 : i64,
      outer_inner_stride = 4 : i64,
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
  func.func @mem_read_blocked_outer(%arg0: tensor<256xi8>) {
    %0 = ftlpu.schedule.mem_read %arg0 {
      address = {hemisphere = "east"},
      bytes = 32 : i64,
      cycle = 10 : i64,
      direction = "east",
      duration = 4 : i64,
      group_address_stride = 0 : i64,
      group_count = 5 : i64,
      group_interval = 40 : i64,
      outer_group_size = 2 : i64,
      outer_group_stride = 32 : i64,
      outer_inner_stride = 4 : i64,
      placement = {address_stride = 1 : i64, bank = 0 : i64,
        base_row = 100 : i64, hemisphere = "east",
        instruction_count = 4 : i64, kind = "schedule_slice",
        slices = [0]},
      register_id = 0 : i64,
      role = "weight_i8",
      stream_base = 0 : i64,
      stream_count = 1 : i64,
      wave_address_stride = 8 : i64,
      wave_count = 3 : i64,
      wave_interval = 8 : i64
    } : (tensor<256xi8>) -> tensor<256xi8>
    return
  }
}
