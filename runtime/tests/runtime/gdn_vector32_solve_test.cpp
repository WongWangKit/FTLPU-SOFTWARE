#include "ftlpu/compiler/Target/vector32_triangular_solve.hpp"
#include "ftlpu/vxm/superlane.hpp"

#include <cmath>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ftlpu;
constexpr std::size_t rows = 32, width = 128;
float bf16(float x) {
    return VxmDataFormat::bf16_bits_to_float(VxmDataFormat::float_to_bf16_bits(x));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void put(VxmSuperlane::StreamMatrix& streams, std::size_t lane,
         std::size_t stage, bool rhs, float value) {
    const auto address = VxmLane::fixed_input_group_for_stage(stage, rhs) * 2;
    const auto bits = VxmDataFormat::float_to_bf16_bits(value);
    streams[lane][address] = static_cast<std::uint8_t>(bits);
    streams[lane][address+1] = static_cast<std::uint8_t>(bits >> 8);
}
double error(const std::vector<float>& a, const std::vector<float>& b) {
    double diff=0, norm=0;
    for (std::size_t i=0; i<a.size(); ++i) {
        diff += double(a[i]-b[i])*(a[i]-b[i]); norm += double(b[i])*b[i];
    }
    return std::sqrt(diff / std::max(norm, 1e-30));
}
} // namespace

// A native FU microtest, not an end-to-end ICU/MEM/C2C run. The harness feeds
// prepacked words at the VXM boundary and models SRAM as a vector. Only the
// hardware model performs the solve arithmetic. No model operation is added
// to CModel. Trace timestamps exclude memory access and coefficient packing.
int main(int argc, char** argv) try {
    using namespace ftlpu;
    static_assert(hw::kTileRows * VxmSuperlane::kLaneCount == 32,
        "this mapping uses 32 lanes across all tiles");
    const auto out = std::filesystem::path(argc > 1 ? argv[1] : "gdn_vector32_solve");
    std::filesystem::create_directories(out);
    std::ofstream trace(out / "vxm_fu.pipeline.csv");
    trace << "start,end,resource,detail\n";
    std::mt19937 rng(2364);
    std::normal_distribution<float> normal(0, 1);
    const auto plan = compiler::plan_vector32_triangular_solve(rows, width);
    std::size_t cycle=0, packets=0, words_read=0, words_written=0;
    double worst=0;
    // Ordinary, identity, and strongly cancelling lower-triangular cases.
    for (int test=0; test<3; ++test) {
        std::vector<float> lower(rows*rows), rhs(rows*width), actual(rows*width),
            expected(rows*width), ideal(rows*width);
        for (std::size_t i=0; i<rows; ++i) {
            for (std::size_t j=0; j<i; ++j)
                lower[i*rows+j] = bf16(test == 1 ? 0 : normal(rng)*(test == 2 ? .15f : .025f));
            for (std::size_t c=0; c<width; ++c) {
                rhs[i*width+c] = bf16(normal(rng)*.2f);
                float sum=0, full=0;
                for (std::size_t j=0; j<i; ++j) {
                    sum += -lower[i*rows+j]*expected[j*width+c];
                    full += -lower[i*rows+j]*ideal[j*width+c];
                }
                expected[i*width+c] = bf16(sum + rhs[i*width+c]);
                ideal[i*width+c] = full + rhs[i*width+c];
            }
        }
        // Expanded coefficients are explicit input storage, not free lane
        // broadcasts or compiler-generated immediates dependent on data.
        std::vector<float> coefficients(rows*(rows-1)/2*32);
        for (std::size_t i=1; i<rows; ++i)
            for (std::size_t j=0; j<i; ++j)
                for (std::size_t lane=0; lane<32; ++lane)
                    coefficients[(i*(i-1)/2+j)*32+lane] = -lower[i*rows+j];
        for (const auto& task : plan) {
            std::array<VxmSuperlane,hw::kTileRows> units;
            for (const auto& packet : task.packets()) {
                for (auto& unit : units)
                    unit.enqueue_compact_instruction(packet.queue, packet.instruction);
                ++packets;
            }
            for (auto& unit : units) unit.tick();
            ++cycle; // compact decode start
            std::size_t outputs=0;
            auto tick = [&] {
                for (auto& unit : units) unit.tick();
                for (auto stage : {0u,1u,8u,9u})
                    if (units[0].lane(0).last_trace()[stage].state == VxmLaneAluTraceState::Executed)
                        trace << cycle << ',' << cycle+1 << ",VXM.C" << stage
                              << "," << (stage%8 == 0 ? "multiply" : "add_acc")
                              << " case=" << test << " row=" << task.row
                              << " block=" << task.block << '\n';
                for (std::size_t tile=0; tile<units.size(); ++tile)
                  for (const auto& value : units[tile].outputs()) {
                    const auto half = value.stream == 0 ? 0u : 1u;
                    require(value.stream == 0 || value.stream == 8, "unexpected VXM output group");
                    require(value.byte_count == 2, "solve output must be BF16");
                    for (std::size_t lane=0; lane<VxmSuperlane::kLaneCount; ++lane) {
                        const auto bits = std::uint16_t(value.byte_values[lane][0]) |
                            (std::uint16_t(value.byte_values[lane][1]) << 8);
                        if (task.block+half < task.value_blocks)
                        actual[(task.output_word()+half)*32+tile*VxmSuperlane::kLaneCount+lane] =
                            VxmDataFormat::bf16_bits_to_float(bits);
                    }
                    ++outputs;
                }
                ++cycle;
            };
            for (std::size_t j=0; j<task.terms(); ++j) {
              for (std::size_t tile=0; tile<units.size(); ++tile) {
                VxmSuperlane::StreamMatrix streams{};
                for (std::size_t half=0; half<2; ++half)
                    for (std::size_t lane=0; lane<VxmSuperlane::kLaneCount; ++lane) {
                        const auto c=tile*VxmSuperlane::kLaneCount+lane;
                        const auto blockOffset=task.block+half<task.value_blocks ? half : 0;
                        put(streams,lane,half*8,false, j<task.row
                            ? actual[(task.solved_word(j)+blockOffset)*32+c]
                            : rhs[(task.output_word()+blockOffset)*32+c]);
                        put(streams,lane,half*8,true, j<task.row
                            ? coefficients[task.coefficient_word(j)*32+c] : 1.0f);
                    }
                units[tile].set_stream_inputs(streams);
              }
                tick();
                words_read += j<task.row ? 4 : 2;
            }
            std::size_t drain=0;
            while (!units[0].idle() && drain++<128) tick();
            for (auto& unit : units) require(unit.idle(), "VXM solve did not drain");
            require(outputs == 2*units.size(), "partial accumulation leaked or final output missing");
            words_written += 2;
        }
        require(actual == expected, "native VXM differs from BF16 storage / FP32 accumulate reference");
        const auto relative = error(actual,ideal);
        require(relative < .01, "BF16 solve error exceeded 1% for test fixture");
        worst = std::max(worst,relative);
        std::cout << "case=" << test << " nrmse=" << relative << '\n';
    }
    std::ofstream summary(out / "summary.json");
    summary << "{\n  \"status\": \"passed\",\n"
        << "  \"backend\": \"native VXM compact decode and FU; host-fed streams, no MEM/ICU\",\n"
        << "  \"chunk_size\": 32, \"value_dim\": 128, \"cases\": 3,\n"
        << "  \"operand_storage\": \"BF16\", \"accumulator\": \"FP32\",\n"
        << "  \"coefficient_expansion_bytes_per_case\": 31744,\n"
        << "  \"fu_only_cycles_all_cases\": " << cycle << ",\n"
        << "  \"compact_packets_all_cases\": " << packets << ",\n"
        << "  \"modeled_word_reads_all_cases\": " << words_read << ",\n"
        << "  \"modeled_word_writes_all_cases\": " << words_written << ",\n"
        << "  \"worst_nrmse_vs_unrounded_solve\": " << worst << "\n}\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
}
