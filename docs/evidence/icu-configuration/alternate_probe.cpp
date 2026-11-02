#include "ftlpu/system/icu.hpp"
#include <iostream>
using namespace ftlpu;
using Mem = InstructionControlUnit::MemIcu;
using ProbeMxm = InstructionControlUnit::MxmComputeIcu;
static_assert(Mem::imem_depth == 512 && Mem::iq_depth == 8);
static_assert(ProbeMxm::imem_depth == 256 && ProbeMxm::iq_depth == 8);
static_assert(InstructionControlUnit::C2cDmaIcu::imem_depth == 256);
static_assert(Mem::macro_context_depth == 16 && ProbeMxm::macro_context_depth == 24);
static_assert(Mem::fetch_latency == 2 && Mem::macro_imem_request_initiation_interval == 2);
static_assert(Mem::macro_reservoir_words == 4 && Mem::macro_decode_window_bits == 32);
static_assert(Mem::macro_ddb_depth == 4 && Mem::macro_admission_lookahead == 6);
static_assert(Mem::macro_ddb_run_capacity == 4);
int main() {
    Mem queue;
    Mem::RawImemWord word{};
    word.lanes[0] = 0x1234;
    queue.write_raw_imem(511, word);
    if (queue.read_raw_imem(511).lanes[0] != 0x1234) return 1;
    try { queue.write_raw_imem(512, word); }
    catch (const StaticScheduleError&) {
        std::cout << "alternate profile passed: actual ICU types use changed geometry; last i-MEM word roundtrips and overflow is rejected\n";
        return 0;
    }
    return 2;
}
