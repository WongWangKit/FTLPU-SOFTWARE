#pragma once

#include "ftlpu/vxm/compact_instruction.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ftlpu::compiler {

// Experimental unit-lower solve mapping. All memory operands are BF16;
// products and the C1/C9 accumulator are FP32. This is deliberately separate
// from the FP32 graph baseline: converting its operands is a precision choice.
// A word contains 32 value channels across the slice's four 8-lane tiles.
// Mirrored datapaths process two consecutive value words concurrently.
// L is supplied as replicated coefficient words, never dynamic immediates.
struct Vector32SolvePacket {
    std::size_t queue;
    VxmCompactInstruction instruction;
};

struct Vector32SolveRow {
    std::size_t row;
    std::size_t block;
    std::size_t value_blocks;

    std::size_t terms() const { return row + 1; }
    std::size_t output_word() const { return row * value_blocks + block; }
    std::size_t solved_word(std::size_t j) const {
        if (j >= row) throw std::out_of_range("solve reads an unsolved row");
        return j * value_blocks + block;
    }
    // One broadcast word per strict-lower entry, shared across value blocks.
    std::size_t coefficient_word(std::size_t j) const {
        if (j >= row) throw std::out_of_range("solve coefficient is not lower triangular");
        return row * (row - 1) / 2 + j;
    }

    std::vector<Vector32SolvePacket> packets() const {
        std::vector<Vector32SolvePacket> result;
        auto emit = [&](std::size_t queue, const VxmLaneAluInstruction& i) {
            result.push_back({queue, VxmCompactInstructionCodec::encode(
                queue, VxmChainDepth::Two, i)});
        };
        VxmLaneAluInstruction multiply{VxmAluOpcode::Multiply,
            VxmLaneOperand::StreamBFloat16(), VxmLaneOperand::StreamBFloat16()};
        multiply.precision = VxmAluPrecision::Float32;
        multiply.repeat_count = terms();
        emit(0, multiply);
        auto accumulate = [&](std::size_t count, bool reset, bool last) {
            VxmLaneAluInstruction add{VxmAluOpcode::Add,
                VxmLaneOperand::Previous(), VxmLaneOperand::Acc()};
            add.precision = VxmAluPrecision::Float32;
            add.accumulator_write = true;
            add.accumulator_reset = reset;
            add.accumulator_emit = last;
            add.repeat_count = count;
            if (last) {
                add.output_stream = 0;
                add.output_type = VxmCastTarget::BFloat16;
            }
            emit(1, add);
        };
        // Feed (-L_ij, D_j) followed by (1, B_i). No partial RHS is stored.
        if (row == 0) accumulate(1, true, true);
        else {
            accumulate(1, true, false);
            if (row > 1) accumulate(row - 1, false, false);
            accumulate(1, false, true);
        }
        return result;
    }
};

inline std::vector<Vector32SolveRow> plan_vector32_triangular_solve(
    std::size_t rows, std::size_t value_dim) {
    if (rows != 32 || value_dim == 0 || value_dim % 32)
        throw std::invalid_argument("vector32 solve requires chunk32 and a 32-aligned value dimension");
    std::vector<Vector32SolveRow> result;
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t block = 0; block < value_dim / 32; block += 2)
            result.push_back({row, block, value_dim / 32});
    return result;
}

} // namespace ftlpu::compiler
