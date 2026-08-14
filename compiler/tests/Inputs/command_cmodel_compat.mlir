module {
  func.func @main() {
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 0 : i64, opcode = "read",
      address = 10 : i64, packed_stream = 0 : i64,
      repeat_count = 3 : i64, repeat_interval = 1 : i64,
      address_stride = 1 : i64
    }
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 1 : i64, opcode = "write",
      address = 20 : i64, packed_stream = 32 : i64,
      repeat_count = 3 : i64, repeat_interval = 1 : i64,
      address_stride = 2 : i64
    }
    ftlpu.command.mxm_dequant {
      cycle = 0 : i64, queue = 0 : i64, scale = 0.125 : f32,
      repeat_count = 1 : i64, repeat_interval = 1 : i64
    }
    ftlpu.command.mxm {
      cycle = 0 : i64, queue = 0 : i64, opcode = "iw",
      weight_buffer = 0 : i64, weight_column = 0 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 0 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      accumulator_address = 0 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram", accumulator_clear = false,
      weight_input_mode = "int8_dequant_bf16"
    }
    ftlpu.command.mxm {
      cycle = 1 : i64, queue = 0 : i64, opcode = "compute",
      weight_buffer = 0 : i64, weight_column = 0 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 0 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      accumulator_address = 0 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram", accumulator_clear = false,
      data_format = "bf16", compute_mode = "block8"
    }
    ftlpu.command.mxm {
      cycle = 16 : i64, queue = 1 : i64, opcode = "compute",
      weight_buffer = 0 : i64, weight_column = 0 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 0 : i64,
      repeat_count = 4 : i64, repeat_interval = 1 : i64,
      accumulator_address = 4 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram", accumulator_clear = false,
      data_format = "bf16", compute_mode = "block8",
      wave_count = 3 : i64, wave_interval = 8 : i64,
      wave_accumulator_address_stride = 4 : i64
    }
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 2 : i64, opcode = "read",
      address = 100 : i64, packed_stream = 0 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      address_stride = 0 : i64
    }
    ftlpu.command.mem {
      cycle = 1 : i64, queue = 2 : i64, opcode = "read",
      address = 101 : i64, packed_stream = 1 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      address_stride = 0 : i64
    }
    ftlpu.command.mem {
      cycle = 2 : i64, queue = 2 : i64, opcode = "read",
      address = 102 : i64, packed_stream = 2 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      address_stride = 0 : i64
    }
    ftlpu.command.loop {
      cycle = 3 : i64, queue_kind = "mem", queue = 2 : i64,
      window_size = 3 : i64, count = 2 : i64,
      interval = 5 : i64, address_stride = 16 : i64
    }
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 4 : i64, opcode = "read",
      address = 200 : i64, packed_stream = 3 : i64,
      repeat_count = 4 : i64, repeat_interval = 1 : i64,
      address_stride = -1 : i64,
      wave_count = 3 : i64, wave_interval = 8 : i64,
      wave_address_stride = 16 : i64
    }
    // The event at cycle 4 interleaves the outer wave. The binary emitter
    // must expand the outer dimension because Repeat2D is blocking.
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 6 : i64, opcode = "read",
      address = 300 : i64, packed_stream = 4 : i64,
      repeat_count = 4 : i64, repeat_interval = 1 : i64,
      address_stride = 1 : i64,
      wave_count = 3 : i64, wave_interval = 8 : i64,
      wave_address_stride = 16 : i64
    }
    ftlpu.command.mem {
      cycle = 4 : i64, queue = 6 : i64, opcode = "read",
      address = 999 : i64, packed_stream = 5 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      address_stride = 0 : i64
    }
    // A repeated candidate is not a legal round of a single-issue Loop
    // window. Folding it would silently drop address 402.
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 8 : i64, opcode = "read",
      address = 400 : i64, packed_stream = 6 : i64,
      repeat_count = 1 : i64, repeat_interval = 1 : i64,
      address_stride = 0 : i64
    }
    ftlpu.command.mem {
      cycle = 1 : i64, queue = 8 : i64, opcode = "read",
      address = 401 : i64, packed_stream = 6 : i64,
      repeat_count = 2 : i64, repeat_interval = 1 : i64,
      address_stride = 1 : i64
    }
    return
  }
}
