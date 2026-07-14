#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"

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

constexpr std::size_t kRows = ftlpu::hw::kMxmRows;
constexpr std::size_t kColumns = ftlpu::hw::kMxmColumns;
constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;
constexpr std::size_t kActivationColumn = 32;
constexpr std::size_t kActivationStream = 16;
constexpr std::size_t kResultColumnBase = 40;
constexpr std::size_t kResultWestStreamBase = 0;
constexpr std::size_t kResultStreamBase = ftlpu::hw::kEastStreams + kResultWestStreamBase;
constexpr std::size_t kIwStart = 20;
constexpr std::size_t kGemmStart = kIwStart + 2 * kBlocks;

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

std::int8_t lhs_value(std::size_t row, std::size_t k)
{
    const auto mixed = row * 13 + k * 7 + ((row + 5) * (k + 3)) % 31;
    return static_cast<std::int8_t>(static_cast<int>(mixed % 17) - 8);
}

std::int8_t rhs_value(std::size_t k, std::size_t column)
{
    const auto mixed = k * 11 + column * 5 + ((k + 7) * (column + 9)) % 29;
    return static_cast<std::int8_t>(static_cast<int>(mixed % 15) - 7);
}

std::int32_t expected_result(std::size_t row, std::size_t column)
{
    std::int32_t sum = 0;
    for (std::size_t k = 0; k < kRows; ++k) {
        sum += static_cast<std::int32_t>(lhs_value(row, k))
            * static_cast<std::int32_t>(rhs_value(k, column));
    }
    return sum;
}

std::size_t vector_address(std::size_t row, std::size_t lane)
{
    return row * kLanes + lane;
}

std::size_t weight_address(std::size_t column_block)
{
    return column_block * kLanes;
}

std::size_t east_stream_cycles_to_sreg11(std::size_t column)
{
    return (ftlpu::hw::kStreamRegisterColumns - 1) - column / ftlpu::hw::kSlicesPerGroup;
}

void stage_matrices(ftlpu::TileArrayModel& mem)
{
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t k = 0; k < kRows; ++k) {
            mem.set_sram_byte(
                kActivationColumn,
                k / kLanes,
                vector_address(row, k % kLanes),
                static_cast<std::uint8_t>(lhs_value(row, k)));
        }
    }

    for (std::size_t tile = 0; tile < kBlocks; ++tile) {
        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
                const auto column = column_block * kLoadStreams + stream;
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    mem.set_sram_byte(
                        stream,
                        tile,
                        weight_address(column_block) + lane,
                        static_cast<std::uint8_t>(rhs_value(tile * kLanes + lane, column)));
                }
            }
        }
    }
}

void emit_weight_load(ftlpu::software::runtime::IcuProgram& program)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto first_cycle = kIwStart - east_stream_cycles_to_sreg11(stream) - 1;
        for (std::size_t block = 0; block < kBlocks; ++block) {
            const auto column_block = kBlocks - 1 - block;
            program.emit_mem(
                first_cycle + block,
                stream,
                ftlpu::MemInstruction::Read(weight_address(column_block), stream));
        }
    }

    for (std::size_t block = 0; block < kBlocks; ++block) {
        program.emit_mxm_load(kIwStart + block, 0, ftlpu::MxmControlInstruction::IW(0));
    }
}

void emit_gemm(ftlpu::software::runtime::IcuProgram& program)
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
            ftlpu::MxmControlInstruction::Compute(0, kActivationStream, kResultWestStreamBase));

        const auto write_issue_cycle = kGemmStart + row + kBlocks + 1;
        for (std::size_t byte = 0; byte < 4; ++byte) {
            program.emit_mem(
                write_issue_cycle,
                kResultColumnBase + byte,
                ftlpu::MemInstruction::Write(vector_address(row, 0), kResultStreamBase + byte));
        }
    }
}

std::int32_t load_result(const ftlpu::TileArrayModel& mem, std::size_t row, std::size_t column)
{
    const auto tile = column / kLanes;
    const auto lane = column % kLanes;
    const auto address = vector_address(row, lane);
    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < 4; ++byte) {
        raw |= static_cast<std::uint32_t>(mem.sram_byte(kResultColumnBase + byte, tile, address)) << (8 * byte);
    }
    return static_cast<std::int32_t>(raw);
}

void run_cmodel(ftlpu::TspSliceSystem& system, std::size_t cycles)
{
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        try {
            system.dispatch_icu_only(nullptr);
            system.tick_mxm_controls_only(ftlpu::TspSliceSystem::LogSinks {});
            system.mem().tick(null_stream());
            system.tick_mxm_datapaths_only(ftlpu::TspSliceSystem::LogSinks {});
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "cycle " << cycle << ": " << ex.what();
            throw std::logic_error(os.str());
        }
    }
}

void verify_all_results(const ftlpu::TileArrayModel& mem)
{
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kColumns; ++column) {
            const auto actual = load_result(mem, row, column);
            const auto expected = expected_result(row, column);
            if (actual != expected) {
                std::ostringstream os;
                os << "GEMM mismatch row=" << row
                   << " column=" << column
                   << " actual=" << actual
                   << " expected=" << expected;
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
        emit_weight_load(program);
        emit_gemm(program);

        const auto binary_path = std::filesystem::path("cmodel_gemm_320_runtime_test.ftlpu");
        write_binary_program(BinaryProgram {program.last_cycle(), program.encode_queues()}, binary_path);

        auto system = std::make_unique<ftlpu::TspSliceSystem>();
        stage_matrices(system->mem());

        auto runtime = CModelRuntime(*system);
        runtime.load_file(binary_path);

        const auto total_cycles = program.last_cycle() + 2 * kBlocks + 8;
        run_cmodel(*system, total_cycles);
        verify_all_results(system->mem());

        std::cout << "cmodel_gemm_320_runtime_test passed: verified 320x320 int8 GEMM against CPU baseline\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cmodel_gemm_320_runtime_test failed: " << ex.what() << '\n';
        return 1;
    }
}
