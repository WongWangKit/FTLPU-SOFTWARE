#include "ftlpu/core/topology.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/gemm_engine.hpp"
#include "ftlpu/mxm/mxm.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
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
constexpr std::size_t kTargetSreg = ftlpu::hw::kStreamRegisterColumns - 1;

constexpr std::size_t kWeightSramBaseAddress = 6144;
constexpr std::size_t kActivationColumn = 16;
constexpr std::size_t kActivationStream = 0;
constexpr std::size_t kResultColumnBase = 40;
constexpr std::size_t kResultWestStreamBase = 0;
constexpr std::size_t kResultCombinedStreamBase = ftlpu::hw::kEastStreams + kResultWestStreamBase;

constexpr std::size_t kReadExecuteBaseCycle = 6;
constexpr std::size_t kMxmHandoffBaseCycle = 17;
constexpr std::size_t kTile0WeightsReadyPhaseCycle = kMxmHandoffBaseCycle + kBlocks - 1;
constexpr std::size_t kComputeStartPhaseCycle = kTile0WeightsReadyPhaseCycle + 1;
constexpr std::size_t kLastLoadPhaseCycle = kMxmHandoffBaseCycle + kBlocks - 1;
constexpr std::size_t kComputeCycles = kRows;
constexpr std::size_t kComputeFlushCycles = 2 * (kBlocks - 1);
constexpr std::size_t kLastComputePhaseCycle = kComputeStartPhaseCycle + kComputeCycles + kComputeFlushCycles;

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

std::int8_t weight_value(std::size_t k, std::size_t n)
{
    return static_cast<std::int8_t>(static_cast<int>((k * 3 + n * 5) % 7) - 3);
}

std::int8_t activation_value(std::size_t row, std::size_t k)
{
    return static_cast<std::int8_t>(static_cast<int>((row * 2 + k * 3) % 5) - 2);
}

std::int32_t expected_result(std::size_t row, std::size_t column)
{
    std::int32_t sum = 0;
    for (std::size_t k = 0; k < kRows; ++k) {
        sum += static_cast<std::int32_t>(activation_value(row, k))
            * static_cast<std::int32_t>(weight_value(k, column));
    }
    return sum;
}

std::size_t weight_address(std::size_t column_block)
{
    return kWeightSramBaseAddress + column_block * kLanes;
}

std::size_t activation_address(std::size_t row)
{
    return row * kLanes;
}

std::size_t result_address(std::size_t row_group, std::size_t column_block)
{
    return (row_group * kBlocks + column_block) * kLanes;
}

void preload_weight_matrix(ftlpu::TileArrayModel& mem)
{
    for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
        for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
            const auto mem_column = stream;
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    const auto k = tile * kLanes + lane;
                    const auto n = column_block * kLoadStreams + stream;
                    mem.set_sram_byte(
                        mem_column,
                        tile,
                        weight_address(column_block) + lane,
                        static_cast<std::uint8_t>(weight_value(k, n)));
                }
            }
        }
    }
}

void preload_activation_matrix(ftlpu::TileArrayModel& mem)
{
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto k = tile * kLanes + lane;
                mem.set_sram_byte(
                    kActivationColumn,
                    tile,
                    activation_address(row) + lane,
                    static_cast<std::uint8_t>(activation_value(row, k)));
            }
        }
    }
}

void issue_weight_reads_for_cycle(
    ftlpu::TileArrayModel& mem,
    std::size_t column_block,
    std::size_t cycle,
    std::size_t column_issue_cycle)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = stream;
        if (cycle == column_issue_cycle + ftlpu::stream_register_before_slice(mem_column)) {
            mem.enqueue_instruction(
                mem_column,
                ftlpu::MemInstruction::Read(weight_address(column_block), stream));
        }
    }
}

void issue_pipelined_weight_reads_for_cycle(ftlpu::TileArrayModel& mem, std::size_t cycle)
{
    for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
        issue_weight_reads_for_cycle(mem, column_block, cycle, kReadExecuteBaseCycle + column_block);
    }
}

void issue_activation_read_for_cycle(ftlpu::TileArrayModel& mem, std::size_t phase_cycle)
{
    const auto handoff_sreg = ftlpu::stream_register_after_slice(kActivationColumn);
    const auto read_to_handoff_cycles = kTargetSreg - handoff_sreg + 1;
    if (phase_cycle + read_to_handoff_cycles < kComputeStartPhaseCycle) {
        return;
    }

    const auto row = phase_cycle + read_to_handoff_cycles - kComputeStartPhaseCycle;
    if (row >= kRows) {
        return;
    }

    mem.enqueue_instruction(
        kActivationColumn,
        ftlpu::MemInstruction::Read(activation_address(row), kActivationStream));
}

ftlpu::MxmControlSlice::WeightInput collect_weight_input(const ftlpu::TileArrayModel& mem, std::size_t tile)
{
    ftlpu::MxmControlSlice::WeightInput input {};
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
            const auto& slot = mem.east_register(tile, lane, kTargetSreg, stream);
            if (!slot.has_value()) {
                throw std::logic_error("weight handoff reached an empty MEM east stream slot");
            }
            input[lane][stream] = ftlpu::MxmArray::Supercell::InputWord {
                static_cast<std::int8_t>(slot->data),
                stream + 1 == kLoadStreams,
            };
        }
    }
    return input;
}

ftlpu::MxmGemmEngine::ActivationVector collect_activation_input(
    const ftlpu::TileArrayModel& mem,
    std::size_t tile)
{
    ftlpu::MxmGemmEngine::ActivationVector input {};
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        const auto& slot = mem.east_register(tile, lane, kTargetSreg, kActivationStream);
        if (!slot.has_value()) {
            throw std::logic_error("activation handoff reached an empty MEM east stream slot");
        }
        input[lane] = ftlpu::MxmGemmEngine::ActivationWord {
            static_cast<std::int8_t>(slot->data),
            lane + 1 == kLanes,
        };
    }
    return input;
}

void set_result_streams_for_tile(
    ftlpu::TileArrayModel& mem,
    const ftlpu::MxmGemmEngine& gemm,
    std::size_t row_group,
    std::size_t column_block,
    std::size_t tile)
{
    const auto row = row_group * kBlocks + tile;
    const auto& result = gemm.accumulator_row(row);
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        const auto value = static_cast<std::uint32_t>(
            result[column_block * kLanes + lane]);
        for (std::size_t byte = 0; byte < 4; ++byte) {
            mem.set_west_stream_input(
                tile,
                lane,
                kResultWestStreamBase + byte,
                ftlpu::TileArrayModel::DataWord {
                    static_cast<std::uint8_t>((value >> (8 * byte)) & 0xffu),
                    lane + 1 == kLanes,
                });
        }
    }
}

void run_mem_to_mxm_gemm(ftlpu::TileArrayModel& mem, ftlpu::Mxm& mxm)
{
    auto gemm = std::make_unique<ftlpu::MxmGemmEngine>(mxm.array());
    auto& control = mxm.control();
    bool gemm_started = false;

    for (std::size_t phase_cycle = 0; phase_cycle <= kLastComputePhaseCycle; ++phase_cycle) {
        if (phase_cycle <= kLastLoadPhaseCycle) {
            issue_pipelined_weight_reads_for_cycle(mem, phase_cycle);
        }
        issue_activation_read_for_cycle(mem, phase_cycle);
        mem.tick(null_stream());

        if (phase_cycle >= kMxmHandoffBaseCycle && phase_cycle < kMxmHandoffBaseCycle + kBlocks) {
            const auto column_block = phase_cycle - kMxmHandoffBaseCycle;
            control.issue_south(ftlpu::MxmControlInstruction::IW(column_block, 0));
        }
        if (phase_cycle >= kComputeStartPhaseCycle && phase_cycle < kComputeStartPhaseCycle + kComputeCycles) {
            control.issue_south(ftlpu::MxmControlInstruction::Compute(0));
        }
        if (phase_cycle == kComputeStartPhaseCycle) {
            gemm->start_compute(kComputeCycles, 0);
            gemm_started = true;
        }

        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            if (phase_cycle < kMxmHandoffBaseCycle + tile) {
                continue;
            }
            const auto column_block = phase_cycle - kMxmHandoffBaseCycle - tile;
            if (column_block < kBlocks) {
                control.set_weight_input(tile, collect_weight_input(mem, tile));
            }
        }
        control.tick(null_stream(), false);

        if (!gemm_started) {
            continue;
        }

        const auto compute_cycle = phase_cycle - kComputeStartPhaseCycle;
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            if (compute_cycle >= tile && compute_cycle - tile < kComputeCycles) {
                gemm->set_activation_input(tile, collect_activation_input(mem, tile));
            }
        }
        gemm->tick(null_stream(), false, false);
    }

    for (std::size_t row = 0; row < kRows; ++row) {
        const auto& result = gemm->accumulator_row(row);
        for (std::size_t column = 0; column < kColumns; ++column) {
            if (result[column] != expected_result(row, column)) {
                std::ostringstream os;
                os << "MXM accumulator mismatch row=" << row
                   << " column=" << column
                   << " actual=" << result[column]
                   << " expected=" << expected_result(row, column);
                throw std::logic_error(os.str());
            }
        }
    }

    for (std::size_t row_group = 0; row_group < kRows / kBlocks; ++row_group) {
        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            const auto address = result_address(row_group, column_block);

            set_result_streams_for_tile(mem, *gemm, row_group, column_block, 0);
            mem.tick(null_stream());

            for (std::size_t byte = 0; byte < 4; ++byte) {
                mem.enqueue_instruction(
                    kResultColumnBase + byte,
                    ftlpu::MemInstruction::Write(address, kResultCombinedStreamBase + byte));
            }

            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                if (tile + 1 < kBlocks) {
                    set_result_streams_for_tile(mem, *gemm, row_group, column_block, tile + 1);
                }
                mem.tick(null_stream());
            }
        }
    }
}

std::int32_t load_result_from_mem(
    const ftlpu::TileArrayModel& mem,
    std::size_t row,
    std::size_t column)
{
    const auto tile = row % kBlocks;
    const auto row_group = row / kBlocks;
    const auto column_block = column / kLanes;
    const auto lane = column % kLanes;
    const auto address = result_address(row_group, column_block) + lane;

    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < 4; ++byte) {
        raw |= static_cast<std::uint32_t>(
                   mem.sram_byte(kResultColumnBase + byte, tile, address))
            << (8 * byte);
    }
    return static_cast<std::int32_t>(raw);
}

void verify_result_mem(const ftlpu::TileArrayModel& mem)
{
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kColumns; ++column) {
            const auto actual = load_result_from_mem(mem, row, column);
            const auto expected = expected_result(row, column);
            if (actual != expected) {
                std::ostringstream os;
                os << "result MEM mismatch row=" << row
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
        auto mem = std::make_unique<ftlpu::TileArrayModel>();
        auto mxm = std::make_unique<ftlpu::Mxm>();

        preload_weight_matrix(*mem);
        preload_activation_matrix(*mem);
        run_mem_to_mxm_gemm(*mem, *mxm);
        verify_result_mem(*mem);

        std::cout << "mem_mxm_gemm_320_test passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "mem_mxm_gemm_320_test failed: " << ex.what() << '\n';
        return 1;
    }
}
