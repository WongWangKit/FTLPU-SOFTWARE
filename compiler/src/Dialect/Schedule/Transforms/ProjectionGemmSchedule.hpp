#pragma once

#include "AttentionEmitterUtils.hpp"

namespace ftlpu::compiler::schedule::projection_detail {

// Closed-form single-token-tile projection kernel. Layout/page planning owns
// absolute starts and MEM addresses; this kernel owns the shared M/K/N loop,
// weight-buffer rotation and final-K accumulator protocol. No FU compression.
struct ProjectionGemmSchedule {
  int64_t rows;
  int64_t reductions;
  int64_t groups;
  int64_t reductionInterval;
  int64_t groupInterval;

  struct Binding {
    int64_t queue = 0;
    int64_t weightBuffer = 0;
    int64_t activationStream = 0;
    int64_t outputStream = 0;
    int64_t accumulatorAddress = 0;
    llvm::StringRef format;
  };

  void emitLoad(mlir::IRRewriter &rewriter, mlir::Location loc,
                int64_t cycle, const Binding &binding, int64_t columns,
                bool columnLoad, llvm::StringRef inputMode,
                int64_t weightStream = -1,
                llvm::StringRef destination = "stream") const {
    attention_detail::emitMxm3D(rewriter, loc, cycle, binding.queue, "iw",
        binding.weightBuffer, columnLoad ? 0 : columns - 1, 0, 0, 0, 1,
        destination, true, columnLoad ? "vector" : "block", 0,
        binding.format, inputMode, {}, load(columns, columnLoad), weightStream);
  }

  void emitCompute(mlir::IRRewriter &rewriter, mlir::Location loc,
                   int64_t cycle, const Binding &binding,
                   bool finalOnly = false,
                   llvm::StringRef partialFormat = "fp32") const {
    attention_detail::emitMxm3D(rewriter, loc, cycle, binding.queue, "compute",
        binding.weightBuffer, 0, binding.activationStream, binding.outputStream,
        binding.accumulatorAddress, 1, finalOnly ? "stream" : "sram",
        finalOnly, "block", 0, binding.format, {},
        finalOnly ? binding.format : partialFormat, compute(binding.format));
  }

  attention_detail::MxmDomain3D load(int64_t columns,
                                   bool columnLoad = false) const {
    auto domain = loop(columns);
    domain.repeat_weight_column_stride = columnLoad ? 1 : -1;
    return domain;
  }

  attention_detail::MxmDomain3D compute(llvm::StringRef format) const {
    auto domain = loop(rows);
    // Every output group drains on its own last K block, not on last N.
    domain.terminal_dimension = 1;
    domain.terminal_accumulator_destination = "stream";
    domain.terminal_accumulator_clear = true;
    domain.terminal_accumulator_output_format = format;
    return domain;
  }

private:
  attention_detail::MxmDomain3D loop(int64_t inner) const {
    attention_detail::MxmDomain3D domain;
    domain.repeat_count = inner;
    domain.wave_count = reductions;
    domain.wave_interval = reductionInterval;
    domain.group_count = groups;
    domain.group_interval = groupInterval;
    domain.weight_buffer_mode = "toggle_dim1";
    return domain;
  }
};

} // namespace ftlpu::compiler::schedule::projection_detail
