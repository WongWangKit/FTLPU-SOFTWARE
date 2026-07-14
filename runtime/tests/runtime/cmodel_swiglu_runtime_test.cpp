#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"

#include "ftlpu/vxm/alu.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>

namespace {

constexpr std::size_t kRows = 160;
constexpr std::size_t kColumns = ftlpu::hw::kMxmColumns;
constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;
constexpr std::size_t kGateMatrix = 0;
constexpr std::size_t kUpMatrix = 1;
constexpr std::size_t kActivationColumn = 32;
constexpr std::size_t kOutputColumn = 40;
constexpr std::size_t kActivationStream = 16;
constexpr std::size_t kGateWestStreamBase = 0;
constexpr std::size_t kUpWestStreamBase = 4;
constexpr std::size_t kGateStreamOperand = ftlpu::hw::kEastStreams + kGateWestStreamBase;
constexpr std::size_t kUpStreamOperand = ftlpu::hw::kEastStreams + kUpWestStreamBase;
constexpr std::size_t kSwigluOutputStream = 31;
constexpr std::size_t kIwStart = 20;
constexpr std::size_t kGemmStart = kIwStart + 2 * kBlocks;
constexpr std::size_t kVxmStartOffset = (kBlocks - 1) + ftlpu::hw::kStreamRegisterColumns;
constexpr std::size_t kSwigluLatency = 9;
constexpr std::size_t kOutputWriteOffset = kVxmStartOffset + kSwigluLatency + kOutputColumn / ftlpu::hw::kSlicesPerGroup + 2;
constexpr float kInputScale = 1.0f / 2048.0f;
constexpr float kOutputScale = 1.0f / 128.0f;

class NullBuffer : public std::streambuf {
public:
    int overflow(int c) override
    {
        return c;
    }
};

std::ostream& null_stream()
{
    static NullBuffer buffer;
    static std::ostream stream(&buffer);
    return stream;
}

std::int8_t activation_value(std::size_t row, std::size_t k)
{
    const auto mixed = row * 13 + k * 7 + ((row + 3) * (k + 5)) % 29;
    return static_cast<std::int8_t>(static_cast<int>(mixed % 25) - 12);
}

std::int8_t weight_value(std::size_t matrix, std::size_t k, std::size_t column)
{
    const auto mixed = matrix * 37 + k * 11 + column * 17 + (k * column + matrix * 5) % 23;
    return static_cast<std::int8_t>(static_cast<int>(mixed % 31) - 15);
}

std::size_t vector_address(std::size_t row, std::size_t lane)
{
    return row * kLanes + lane;
}

std::size_t weight_address(std::size_t matrix, std::size_t column_block)
{
    return matrix * kBlocks * kLanes + column_block * kLanes;
}

std::int32_t expected_projection(std::size_t matrix, std::size_t row, std::size_t column)
{
    std::int32_t sum = 0;
    for (std::size_t k = 0; k < kColumns; ++k) {
        sum += static_cast<std::int32_t>(activation_value(row, k))
            * static_cast<std::int32_t>(weight_value(matrix, k, column));
    }
    return sum;
}

std::int8_t expected_swiglu(std::size_t row, std::size_t column)
{
    const auto gate = static_cast<float>(expected_projection(kGateMatrix, row, column)) * kInputScale;
    const auto up = static_cast<float>(expected_projection(kUpMatrix, row, column)) * kInputScale;
    const auto sigmoid = 1.0f / (1.0f + std::exp(-gate));
    return ftlpu::VxmAlu::quantize_scalar(gate * sigmoid * up, kOutputScale, 0);
}

std::size_t east_stream_cycles_to_sreg11(std::size_t column)
{
    return (ftlpu::hw::kStreamRegisterColumns - 1) - column / ftlpu::hw::kSlicesPerGroup;
}

void stage_mem(ftlpu::TileArrayModel& mem)
{
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t k = 0; k < kColumns; ++k) {
            mem.set_sram_byte(
                kActivationColumn,
                k / kLanes,
                vector_address(row, k % kLanes),
                static_cast<std::uint8_t>(activation_value(row, k)));
        }
    }

    for (std::size_t matrix = 0; matrix < 2; ++matrix) {
        const auto stream_base = matrix == kGateMatrix ? std::size_t {0} : kLoadStreams;
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
                for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
                    const auto column = column_block * kLoadStreams + stream;
                    for (std::size_t lane = 0; lane < kLanes; ++lane) {
                        mem.set_sram_byte(
                            stream_base + stream,
                            tile,
                            weight_address(matrix, column_block) + lane,
                            static_cast<std::uint8_t>(weight_value(matrix, tile * kLanes + lane, column)));
                    }
                }
            }
        }
    }
}

void emit_weight_load(ftlpu::software::runtime::IcuProgram& program, std::size_t mxm, std::size_t matrix)
{
    const auto stream_base = mxm * kLoadStreams;
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = stream_base + stream;
        const auto first_cycle = kIwStart - east_stream_cycles_to_sreg11(mem_column) - 1;
        for (std::size_t block = 0; block < kBlocks; ++block) {
            const auto column_block = kBlocks - 1 - block;
            program.emit_mem(
                first_cycle + block,
                mem_column,
                ftlpu::MemInstruction::Read(weight_address(matrix, column_block), stream_base + stream));
        }
    }

    for (std::size_t block = 0; block < kBlocks; ++block) {
        program.emit_mxm_load(kIwStart + block, mxm, ftlpu::MxmControlInstruction::IW(0));
    }
}

void emit_swiglu_stage(
    ftlpu::software::runtime::IcuProgram& program,
    std::size_t cycle,
    std::size_t stage)
{
    using ftlpu::VxmAluOpcode;
    using ftlpu::VxmCastTarget;
    using ftlpu::VxmLaneAluInstruction;
    using ftlpu::VxmLaneOperand;

    auto emit = [&](std::size_t alu, VxmLaneAluInstruction instruction) {
        program.emit_vxm(cycle, alu, instruction);
    };

    switch (stage) {
    case 0:
        emit(0, {VxmAluOpcode::Cast, VxmLaneOperand::StreamInt32(kGateStreamOperand), VxmLaneOperand::Imm(0.0f), 1.0f, 0, VxmCastTarget::Float32});
        emit(1, {VxmAluOpcode::Cast, VxmLaneOperand::StreamInt32(kUpStreamOperand), VxmLaneOperand::Imm(0.0f), 1.0f, 0, VxmCastTarget::Float32});
        break;
    case 1:
        emit(2, {VxmAluOpcode::Multiply, VxmLaneOperand::Alu(0), VxmLaneOperand::Imm(kInputScale)});
        emit(3, {VxmAluOpcode::Multiply, VxmLaneOperand::Alu(1), VxmLaneOperand::Imm(kInputScale)});
        break;
    case 2:
        emit(4, {VxmAluOpcode::Multiply, VxmLaneOperand::Alu(2), VxmLaneOperand::Alu(3)});
        emit(5, {VxmAluOpcode::Negate, VxmLaneOperand::Alu(2), VxmLaneOperand::Imm(0.0f)});
        break;
    case 3:
        emit(6, {VxmAluOpcode::Exp, VxmLaneOperand::Alu(5), VxmLaneOperand::Imm(0.0f)});
        emit(9, {VxmAluOpcode::Pass, VxmLaneOperand::Alu(4), VxmLaneOperand::Imm(0.0f)});
        break;
    case 4:
        emit(7, {VxmAluOpcode::Add, VxmLaneOperand::Alu(6), VxmLaneOperand::Imm(1.0f)});
        emit(10, {VxmAluOpcode::Pass, VxmLaneOperand::Alu(9), VxmLaneOperand::Imm(0.0f)});
        break;
    case 5:
        emit(8, {VxmAluOpcode::Divide, VxmLaneOperand::Imm(1.0f), VxmLaneOperand::Alu(7)});
        emit(11, {VxmAluOpcode::Pass, VxmLaneOperand::Alu(10), VxmLaneOperand::Imm(0.0f)});
        break;
    case 6:
        emit(12, {VxmAluOpcode::Multiply, VxmLaneOperand::Alu(11), VxmLaneOperand::Alu(8)});
        break;
    case 7:
        emit(13, {VxmAluOpcode::Multiply, VxmLaneOperand::Alu(12), VxmLaneOperand::Imm(1.0f / kOutputScale)});
        break;
    case 8:
        emit(14, {VxmAluOpcode::Add, VxmLaneOperand::Alu(13), VxmLaneOperand::Imm(0.0f)});
        break;
    case 9:
        emit(15, {VxmAluOpcode::Cast, VxmLaneOperand::Alu(14), VxmLaneOperand::Imm(0.0f), 1.0f, 0, VxmCastTarget::Int8, kSwigluOutputStream});
        break;
    default:
        throw std::out_of_range("SwiGLU stage is outside the VXM pipeline");
    }
}

void emit_swiglu(ftlpu::software::runtime::IcuProgram& program)
{
    for (std::size_t row = 0; row < kRows; ++row) {
        const auto activation_read_cycle = kGemmStart + row - east_stream_cycles_to_sreg11(kActivationColumn);
        program.emit_mem(
            activation_read_cycle,
            kActivationColumn,
            ftlpu::MemInstruction::Read(vector_address(row, 0), kActivationStream));

        program.emit_mxm_compute(
            kGemmStart + row,
            0,
            ftlpu::MxmControlInstruction::Compute(0, kActivationStream, kGateWestStreamBase));
        program.emit_mxm_compute(
            kGemmStart + row,
            1,
            ftlpu::MxmControlInstruction::Compute(0, kActivationStream, kUpWestStreamBase));

        const auto vxm_start = kGemmStart + row + kVxmStartOffset;
        for (std::size_t stage = 0; stage < 10; ++stage) {
            emit_swiglu_stage(program, vxm_start + stage, stage);
        }

        program.emit_mem(
            kGemmStart + row + kOutputWriteOffset,
            kOutputColumn,
            ftlpu::MemInstruction::Write(vector_address(row, 0), kSwigluOutputStream));
    }
}

std::int8_t load_output(const ftlpu::TileArrayModel& mem, std::size_t row, std::size_t column)
{
    return static_cast<std::int8_t>(
        mem.sram_byte(kOutputColumn, column / kLanes, vector_address(row, column % kLanes)));
}

void run_cmodel(ftlpu::TspSliceSystem& system, std::size_t cycles)
{
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        try {
            system.dispatch_icu_only(nullptr);
            system.tick_mxm_controls_only(ftlpu::TspSliceSystem::LogSinks {});
            system.mem().tick(null_stream());
            system.tick_mxm_datapaths_only(ftlpu::TspSliceSystem::LogSinks {});
            system.tick_vxm_stream_bridge(ftlpu::TspSliceSystem::LogSinks {});
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "cycle " << cycle << ": " << ex.what();
            throw std::logic_error(os.str());
        }
    }
}

void verify_all_outputs(const ftlpu::TileArrayModel& mem)
{
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kColumns; ++column) {
            const auto actual = load_output(mem, row, column);
            const auto expected = expected_swiglu(row, column);
            if (actual != expected) {
                std::ostringstream os;
                os << "SwiGLU mismatch row=" << row
                   << " column=" << column
                   << " actual=" << static_cast<int>(actual)
                   << " expected=" << static_cast<int>(expected);
                throw std::logic_error(os.str());
            }
        }
    }
}

} // namespace

int main()
{
    try {
        using namespace ftlpu::software::runtime;

        auto program = IcuProgram {};
        emit_weight_load(program, 0, kGateMatrix);
        emit_weight_load(program, 1, kUpMatrix);
        emit_swiglu(program);

        const auto binary_path = std::filesystem::path("cmodel_swiglu_runtime_test.ftlpu");
        write_binary_program(BinaryProgram {program.last_cycle(), program.encode_queues()}, binary_path);

        auto system = std::make_unique<ftlpu::TspSliceSystem>();
        stage_mem(system->mem());

        auto runtime = CModelRuntime(*system);
        runtime.load_file(binary_path);

        run_cmodel(*system, program.last_cycle() + 2 * kBlocks + 8);
        verify_all_outputs(system->mem());

        std::cout << "cmodel_swiglu_runtime_test passed: verified 160x320 SwiGLU against CPU baseline\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cmodel_swiglu_runtime_test failed: " << ex.what() << '\n';
        return 1;
    }
}
