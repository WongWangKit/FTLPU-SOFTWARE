#include "ftlpu/icu/macro_bitstream_decoder.hpp"
#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rt = ftlpu::software::runtime;

namespace {

template <typename RawWord>
std::vector<RawWord> raw_words(const rt::PackedMacroImemImage& image)
{
    if (image.word_bits != RawWord::bit_count)
        throw std::runtime_error("packed image has the wrong i-MEM word width");
    constexpr std::size_t wordBytes = RawWord::bit_count / 8;
    if (image.bytes.size() % wordBytes != 0)
        throw std::runtime_error("packed image is not word aligned");

    std::vector<RawWord> result(image.bytes.size() / wordBytes);
    for (std::size_t address = 0; address < result.size(); ++address) {
        for (std::size_t lane = 0; lane < RawWord::lane_count; ++lane) {
            const auto offset = address * wordBytes
                + lane * sizeof(std::uint32_t);
            result[address].lanes[lane] =
                static_cast<std::uint32_t>(image.bytes[offset])
                | (static_cast<std::uint32_t>(image.bytes[offset + 1]) << 8)
                | (static_cast<std::uint32_t>(image.bytes[offset + 2]) << 16)
                | (static_cast<std::uint32_t>(image.bytes[offset + 3]) << 24);
        }
    }
    return result;
}

struct SweepResult {
    std::size_t prime_cycles{0};
    std::size_t ddb_entry_bits{0};
    std::uint64_t semantic_hash{1469598103934665603ull};
    ftlpu::IcuMacroDecoderStatistics statistics{};
};

void hash_value(std::uint64_t& hash, std::uint64_t value)
{
    for (unsigned byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<std::uint8_t>(value >> (byte * 8));
        hash *= 1099511628211ull;
    }
}

template <typename Instruction>
std::uint64_t native_bits(const Instruction& instruction)
{
    if constexpr (std::is_same_v<Instruction, ftlpu::MemInstruction>)
        return ftlpu::isa::encode_mem_instruction(instruction);
    else if constexpr (std::is_same_v<Instruction,
                           ftlpu::MxmControlInstruction>)
        return ftlpu::isa::encode_mxm_instruction(instruction);
    else
        return ftlpu::isa::encode_mxm_dequant_instruction(instruction);
}

template <typename Instruction, std::size_t WordBits,
          std::size_t FetchLatency, std::size_t RequestIi,
          std::size_t ReservoirWords, std::size_t DecodeWindowBits,
          std::size_t DdbDepth, std::size_t AdmissionLookahead,
          std::size_t DdbRunCapacity>
SweepResult simulate(ftlpu::IcuMacroQueueKind kind,
    const std::vector<ftlpu::IcuRawImemWord<WordBits>>& image,
    std::size_t expectedContexts)
{
    using Decoder = ftlpu::IcuMacroV1Decoder<Instruction, WordBits,
        FetchLatency, ReservoirWords, DecodeWindowBits, DdbDepth,
        AdmissionLookahead, DdbRunCapacity, RequestIi>;

    Decoder primeDecoder(kind, image);
    std::size_t primeCycles = 0;
    const auto guard = std::max<std::size_t>(4096,
        expectedContexts * 64 + image.size() * 64);
    while (!primeDecoder.primed()) {
        // A full Active RAM forces priming to stop only at the architectural
        // decode-ahead boundary: DDB full or end of image.
        static_cast<void>(primeDecoder.tick(0, 1, 1));
        if (++primeCycles > guard)
            throw std::runtime_error("Macro frontend priming did not converge");
    }

    Decoder decoder(kind, image);
    std::size_t contexts = 0;
    std::size_t cycles = 0;
    std::uint64_t semanticHash = 1469598103934665603ull;
    while (!decoder.done()) {
        if (auto context = decoder.tick(true)) {
            ++contexts;
            hash_value(semanticHash, context->record_index);
            hash_value(semanticHash, native_bits(context->instruction));
            hash_value(semanticHash, context->schedule.start_cycle);
            hash_value(semanticHash, context->schedule.inner_count);
            hash_value(semanticHash, context->schedule.inner_interval);
            hash_value(semanticHash,
                static_cast<std::uint64_t>(context->schedule.inner_stride));
            hash_value(semanticHash, context->schedule.outer_count);
            hash_value(semanticHash, context->schedule.outer_interval);
            hash_value(semanticHash,
                static_cast<std::uint64_t>(context->schedule.outer_stride));
            hash_value(semanticHash, static_cast<std::uint64_t>(
                context->schedule.induction_target));
        }
        if (++cycles > guard)
            throw std::runtime_error("Macro frontend decode did not converge");
    }
    if (contexts != expectedContexts)
        throw std::runtime_error("Macro frontend changed the context count");
    return {primeCycles, decoder.ddb_layout().entry_bits,
        semanticHash, decoder.statistics()};
}

template <typename Instruction, std::size_t WordBits,
          std::size_t FetchLatency, std::size_t RequestIi,
          std::size_t ReservoirWords, std::size_t DecodeWindowBits,
          std::size_t DdbDepth, std::size_t AdmissionLookahead,
          std::size_t DdbRunCapacity>
SweepResult simulate_timed(ftlpu::IcuMacroQueueKind kind,
    const std::vector<ftlpu::IcuRawImemWord<WordBits>>& image,
    std::size_t expectedContexts, std::size_t activeCapacity,
    std::size_t maximumIssueCycle)
{
    using Decoder = ftlpu::IcuMacroV1Decoder<Instruction, WordBits,
        FetchLatency, ReservoirWords, DecodeWindowBits, DdbDepth,
        AdmissionLookahead, DdbRunCapacity, RequestIi>;

    Decoder decoder(kind, image);
    std::size_t primeCycles = 0;
    const auto guard = std::max({std::size_t{4096},
        expectedContexts * 64 + image.size() * 64,
        maximumIssueCycle + image.size() * 64 + 4096});
    while (!decoder.primed()) {
        static_cast<void>(decoder.tick(0, 1, 1));
        if (++primeCycles > guard)
            throw std::runtime_error("Macro frontend priming did not converge");
    }
    const auto primeStatistics = decoder.statistics();

    std::multimap<std::size_t, std::size_t> activeReleases;
    std::size_t peakActiveOccupancy = 0;
    std::size_t contexts = 0;
    std::size_t cycle = 0;
    std::uint64_t semanticHash = 1469598103934665603ull;
    while (!decoder.done()) {
        while (!activeReleases.empty()
            && activeReleases.begin()->first <= cycle)
            activeReleases.erase(activeReleases.begin());
        const auto earliestRelease = activeReleases.empty()
            ? std::optional<std::size_t>{}
            : std::optional<std::size_t>{activeReleases.begin()->first};
        if (auto context = decoder.tick(cycle, activeReleases.size(),
                activeCapacity, earliestRelease)) {
            if (context->schedule.start_cycle < cycle)
                throw std::runtime_error(
                    "Macro context missed its issue deadline");
            ++contexts;
            hash_value(semanticHash, context->record_index);
            hash_value(semanticHash, native_bits(context->instruction));
            hash_value(semanticHash, context->schedule.start_cycle);
            hash_value(semanticHash, context->schedule.inner_count);
            hash_value(semanticHash, context->schedule.inner_interval);
            hash_value(semanticHash,
                static_cast<std::uint64_t>(context->schedule.inner_stride));
            hash_value(semanticHash, context->schedule.outer_count);
            hash_value(semanticHash, context->schedule.outer_interval);
            hash_value(semanticHash,
                static_cast<std::uint64_t>(context->schedule.outer_stride));
            hash_value(semanticHash, static_cast<std::uint64_t>(
                context->schedule.induction_target));
            const std::size_t finalCycle = context->schedule.start_cycle
                + (context->schedule.outer_count - 1)
                    * context->schedule.outer_interval
                + (context->schedule.inner_count - 1)
                    * context->schedule.inner_interval;
            activeReleases.emplace(finalCycle + 1, context->admission_cycle);
            peakActiveOccupancy = std::max(
                peakActiveOccupancy, activeReleases.size());
            decoder.record_context_completion(
                context->admission_cycle, finalCycle);
        }
        if (++cycle > guard)
            throw std::runtime_error(
                "timed Macro frontend simulation did not converge");
    }
    if (contexts != expectedContexts)
        throw std::runtime_error("Macro frontend changed the context count");
    auto statistics = decoder.statistics();
    statistics.cycles -= primeStatistics.cycles;
    statistics.timing_window_block_cycles -=
        primeStatistics.timing_window_block_cycles;
    statistics.active_ram_full_cycles -=
        primeStatistics.active_ram_full_cycles;
    statistics.capacity_feasible_wait_cycles -=
        primeStatistics.capacity_feasible_wait_cycles;
    statistics.admission_stall_cycles -=
        primeStatistics.admission_stall_cycles;
    statistics.context_stall_cycles -=
        primeStatistics.context_stall_cycles;
    statistics.peak_active_occupancy = peakActiveOccupancy;
    return {primeCycles, decoder.ddb_layout().entry_bits,
        semanticHash, statistics};
}

struct Aggregate {
    std::size_t queue_count{0};
    std::size_t imem_words{0};
    std::size_t command_count{0};
    std::size_t prime_cycles_max{0};
    std::size_t frontend_cycles{0};
    std::size_t fetched_words{0};
    std::size_t bit_wait_cycles{0};
    std::size_t reservoir_empty_cycles{0};
    std::size_t reservoir_full_cycles{0};
    std::size_t decoder_active_cycles{0};
    std::size_t decoder_starvation_cycles{0};
    std::size_t decoder_ddb_stall_cycles{0};
    std::size_t descriptor_count{0};
    std::size_t descriptor_decode_cycles{0};
    std::size_t max_descriptor_decode_cycles{0};
    std::size_t ddb_entries_committed{0};
    std::size_t peak_ddb_occupancy{0};
    std::size_t ddb_full_cycles{0};
    std::size_t ddb_entry_bits{0};
    std::size_t timing_window_block_cycles{0};
    std::size_t active_ram_full_cycles{0};
    std::size_t capacity_feasible_wait_cycles{0};
    std::size_t admission_stall_cycles{0};
    std::size_t max_admission_to_start_cycles{0};
    std::size_t peak_active_occupancy{0};
    std::size_t contexts_generated{0};
    std::uint64_t semantic_hash{0};

    void add(std::size_t words, std::size_t commands,
        const SweepResult& result)
    {
        const auto& stats = result.statistics;
        ++queue_count;
        imem_words += words;
        command_count += commands;
        prime_cycles_max = std::max(prime_cycles_max, result.prime_cycles);
        frontend_cycles += stats.cycles;
        fetched_words += stats.fetched_words;
        bit_wait_cycles += stats.bit_wait_cycles;
        reservoir_empty_cycles += stats.reservoir_empty_cycles;
        reservoir_full_cycles += stats.reservoir_full_cycles;
        decoder_active_cycles += stats.decoder_active_cycles;
        decoder_starvation_cycles += stats.decoder_starvation_cycles;
        decoder_ddb_stall_cycles += stats.decoder_ddb_stall_cycles;
        descriptor_count += stats.descriptor_count;
        descriptor_decode_cycles += stats.descriptor_decode_cycles;
        max_descriptor_decode_cycles = std::max(
            max_descriptor_decode_cycles,
            stats.max_descriptor_decode_cycles);
        ddb_entries_committed += stats.ddb_entries_committed;
        peak_ddb_occupancy = std::max(
            peak_ddb_occupancy, stats.peak_ddb_occupancy);
        ddb_full_cycles += stats.ddb_full_cycles;
        ddb_entry_bits = std::max(ddb_entry_bits, result.ddb_entry_bits);
        timing_window_block_cycles += stats.timing_window_block_cycles;
        active_ram_full_cycles += stats.active_ram_full_cycles;
        capacity_feasible_wait_cycles += stats.capacity_feasible_wait_cycles;
        admission_stall_cycles += stats.admission_stall_cycles;
        max_admission_to_start_cycles = std::max(
            max_admission_to_start_cycles,
            stats.max_admission_to_start_cycles);
        peak_active_occupancy = std::max(
            peak_active_occupancy, stats.peak_active_occupancy);
        contexts_generated += stats.contexts_generated;
        semantic_hash ^= result.semantic_hash;
    }
};

std::string_view kind_name(rt::QueueKind kind)
{
    switch (kind) {
    case rt::QueueKind::Mem: return "mem";
    case rt::QueueKind::MxmLoad: return "mxm_load";
    case rt::QueueKind::MxmCompute: return "mxm_compute";
    case rt::QueueKind::MxmDequant: return "mxm_dequant";
    default: return "unsupported";
    }
}

ftlpu::IcuMacroQueueKind cmodel_kind(rt::QueueKind kind)
{
    switch (kind) {
    case rt::QueueKind::Mem: return ftlpu::IcuMacroQueueKind::Mem;
    case rt::QueueKind::MxmLoad: return ftlpu::IcuMacroQueueKind::MxmLoad;
    case rt::QueueKind::MxmCompute: return ftlpu::IcuMacroQueueKind::MxmCompute;
    case rt::QueueKind::MxmDequant: return ftlpu::IcuMacroQueueKind::MxmDequant;
    default: throw std::invalid_argument("queue kind has no Macro v1 frontend");
    }
}

template <std::size_t Latency, std::size_t Ii,
          std::size_t ReservoirWords = 3,
          std::size_t DecodeWindowBits = 64,
          std::size_t DdbDepth = 8,
          std::size_t AdmissionLookahead = 8,
          std::size_t DdbRunCapacity = 8,
          bool TimingAware = false>
void simulate_queue(const rt::QueueProgram& queue, Aggregate& aggregate)
{
    if (!queue.packed_macro_imem.has_value()) return;
    // Re-encode semantic commands with the current packed-v1 implementation.
    // Stored .ftlpu fixtures may predate a physical-field-width correction;
    // reusing their cached raw words would mix an old ABI into this timing
    // experiment.
    rt::PackedMacroImemImage packed;
    if (queue.kind == rt::QueueKind::Mem) {
        packed = rt::pack_mem_macro_imem(
            rt::encode_mem_macro_bitstream(queue));
    } else if (queue.kind == rt::QueueKind::MxmLoad
        || queue.kind == rt::QueueKind::MxmCompute
        || queue.kind == rt::QueueKind::MxmDequant) {
        packed = rt::pack_mxm_macro_imem(
            rt::encode_mxm_macro_bitstream(queue));
    } else {
        return;
    }
    SweepResult result;
    const std::size_t activeCapacity = queue.kind == rt::QueueKind::Mem
        ? 32 : 40;
    std::size_t maximumIssueCycle = 0;
    for (const rt::QueueCommand& command : queue.commands) {
        if (!rt::is_macro_schedule_command(command)) continue;
        const auto schedule = rt::decode_macro_schedule_command(command);
        maximumIssueCycle = std::max(maximumIssueCycle,
            schedule.start_cycle
                + (schedule.outer_count - 1) * schedule.outer_interval
                + (schedule.inner_count - 1) * schedule.inner_interval);
    }
    switch (queue.kind) {
    case rt::QueueKind::Mem: {
        const auto words = raw_words<ftlpu::IcuRawImemWord96>(packed);
        if constexpr (TimingAware)
            result = simulate_timed<ftlpu::MemInstruction, 96, Latency, Ii,
                ReservoirWords, DecodeWindowBits, DdbDepth,
                AdmissionLookahead, DdbRunCapacity>(cmodel_kind(queue.kind),
                words, queue.commands.size(), activeCapacity,
                maximumIssueCycle);
        else
            result = simulate<ftlpu::MemInstruction, 96, Latency, Ii,
                ReservoirWords, DecodeWindowBits, DdbDepth,
                AdmissionLookahead, DdbRunCapacity>(cmodel_kind(queue.kind),
                words, queue.commands.size());
        break;
    }
    case rt::QueueKind::MxmLoad:
    case rt::QueueKind::MxmCompute: {
        const auto words = raw_words<ftlpu::IcuRawImemWord128>(packed);
        if constexpr (TimingAware)
            result = simulate_timed<ftlpu::MxmControlInstruction, 128,
                Latency, Ii, ReservoirWords, DecodeWindowBits, DdbDepth,
                AdmissionLookahead, DdbRunCapacity>(cmodel_kind(queue.kind),
                words, queue.commands.size(), activeCapacity,
                maximumIssueCycle);
        else
            result = simulate<ftlpu::MxmControlInstruction, 128, Latency, Ii,
                ReservoirWords, DecodeWindowBits, DdbDepth,
                AdmissionLookahead, DdbRunCapacity>(cmodel_kind(queue.kind),
                words, queue.commands.size());
        break;
    }
    case rt::QueueKind::MxmDequant: {
        const auto words = raw_words<ftlpu::IcuRawImemWord128>(packed);
        if constexpr (TimingAware)
            result = simulate_timed<ftlpu::MxmDequantInstruction, 128,
                Latency, Ii, ReservoirWords, DecodeWindowBits, DdbDepth,
                AdmissionLookahead, DdbRunCapacity>(cmodel_kind(queue.kind),
                words, queue.commands.size(), activeCapacity,
                maximumIssueCycle);
        else
            result = simulate<ftlpu::MxmDequantInstruction, 128, Latency, Ii,
                ReservoirWords, DecodeWindowBits, DdbDepth,
                AdmissionLookahead, DdbRunCapacity>(cmodel_kind(queue.kind),
                words, queue.commands.size());
        break;
    }
    default: return;
    }
    aggregate.add(packed.word_count(), queue.commands.size(), result);
}

template <std::size_t Latency, std::size_t Ii,
          std::size_t ReservoirWords = 3,
          std::size_t DecodeWindowBits = 64,
          std::size_t DdbDepth = 8,
          std::size_t AdmissionLookahead = 8,
          std::size_t DdbRunCapacity = 8,
          bool TimingAware = false>
void run_configuration(std::string_view experiment,
    const std::filesystem::path& path,
    const rt::BinaryProgram& program)
{
    std::map<rt::QueueKind, Aggregate> aggregates;
    for (const auto& queue : program.queues) {
        try {
            simulate_queue<Latency, Ii, ReservoirWords, DecodeWindowBits,
                DdbDepth, AdmissionLookahead, DdbRunCapacity, TimingAware>(
                queue, aggregates[queue.kind]);
        } catch (const std::exception& error) {
            throw std::runtime_error("queue " + std::string(kind_name(queue.kind))
                + "[" + std::to_string(queue.index) + "] at latency="
                + std::to_string(Latency) + ", II=" + std::to_string(Ii)
                + ": " + error.what());
        }
    }

    for (const auto& [kind, result] : aggregates) {
        if (result.queue_count == 0) continue;
        std::cout << path.generic_string() << ',' << experiment << ','
                  << kind_name(kind) << ','
                  << (kind == rt::QueueKind::Mem ? 96 : 128) << ','
                  << Latency << ',' << Ii << ',' << ReservoirWords << ','
                  << DecodeWindowBits << ',' << DdbDepth << ','
                  << AdmissionLookahead << ',' << DdbRunCapacity << ",1,"
                  << result.queue_count << ',' << result.imem_words << ','
                  << result.command_count << ',' << result.prime_cycles_max
                  << ',' << result.frontend_cycles << ','
                  << result.fetched_words << ',' << result.bit_wait_cycles
                  << ',' << result.reservoir_empty_cycles << ','
                  << result.reservoir_full_cycles << ','
                  << result.decoder_active_cycles << ','
                  << result.decoder_starvation_cycles << ','
                  << result.decoder_ddb_stall_cycles << ','
                  << result.descriptor_count << ','
                  << result.descriptor_decode_cycles << ','
                  << result.max_descriptor_decode_cycles << ','
                  << result.ddb_entries_committed << ','
                  << result.peak_ddb_occupancy << ','
                  << result.ddb_full_cycles << ','
                  << result.ddb_entry_bits << ','
                  << result.queue_count * DdbDepth * result.ddb_entry_bits
                  << ','
                  << result.timing_window_block_cycles << ','
                  << result.active_ram_full_cycles << ','
                  << result.capacity_feasible_wait_cycles << ','
                  << result.admission_stall_cycles << ','
                  << result.max_admission_to_start_cycles << ','
                  << result.peak_active_occupancy << ','
                  << result.contexts_generated << ','
                  << result.semantic_hash << '\n';
    }
}

template <std::size_t Latency>
void run_latency(const std::filesystem::path& path,
    const rt::BinaryProgram& program)
{
    run_configuration<Latency, 1>("fetch", path, program);
    run_configuration<Latency, 2>("fetch", path, program);
    run_configuration<Latency, 4>("fetch", path, program);
}

void run_structure_sweep(const std::filesystem::path& path,
    const rt::BinaryProgram& program)
{
    // Effective latency=3/II=2 represents a two-cycle SRAM path plus one
    // registered ECC stage. Structural candidates are one-factor changes from
    // the current R3/W64/D8/A8/C8 baseline, plus one low-area combination.
    run_configuration<3, 2, 3, 64, 8, 8, 8>(
        "baseline", path, program);
    run_configuration<3, 2, 2, 64, 8, 8, 8>(
        "reservoir_2", path, program);
    run_configuration<3, 2, 4, 64, 8, 8, 8>(
        "reservoir_4", path, program);
    run_configuration<3, 2, 3, 64, 1, 8, 8>(
        "ddb_1", path, program);
    run_configuration<3, 2, 3, 64, 2, 8, 8>(
        "ddb_2", path, program);
    run_configuration<3, 2, 3, 64, 4, 8, 8>(
        "ddb_4", path, program);
    run_configuration<3, 2, 3, 64, 16, 8, 8>(
        "ddb_16", path, program);
    run_configuration<3, 2, 3, 64, 8, 8, 4>(
        "run_4", path, program);
    run_configuration<3, 2, 3, 64, 8, 8, 16>(
        "run_16", path, program);
    run_configuration<3, 2, 3, 64, 4, 8, 4>(
        "selected_d4_run4", path, program);
    run_configuration<3, 2, 3, 64, 2, 8, 4>(
        "aggressive_d2_run4", path, program);
    // Kept last because some workloads require parser reads wider than these
    // windows. Failure is an expected functional rejection of the candidate.
    run_configuration<3, 2, 3, 48, 8, 8, 8>(
        "window_48", path, program);
    run_configuration<3, 2, 3, 32, 8, 8, 8>(
        "window_32", path, program);
}

void run_admission_sweep(const std::filesystem::path& path,
    const rt::BinaryProgram& program)
{
    run_configuration<3, 2, 3, 64, 4, 16, 4, true>(
        "lookahead_16", path, program);
    run_configuration<3, 2, 3, 64, 4, 8, 4, true>(
        "lookahead_8", path, program);
    run_configuration<3, 2, 3, 64, 4, 4, 4, true>(
        "lookahead_4", path, program);
    run_configuration<3, 2, 3, 64, 4, 2, 4, true>(
        "lookahead_2", path, program);
}

void run_file(const std::filesystem::path& path, bool structureSweep,
    bool admissionSweep)
{
    const auto program = rt::read_binary_program(path);
    if (admissionSweep) {
        run_admission_sweep(path, program);
        return;
    }
    if (structureSweep) {
        run_structure_sweep(path, program);
        return;
    }
    run_latency<1>(path, program);
    run_latency<2>(path, program);
    run_latency<3>(path, program);
    run_latency<4>(path, program);
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc < 2)
        throw std::runtime_error(
            "usage: macro_frontend_fetch_sweep [--structure|--admission] "
            "program.ftlpu [...]");

    int firstPath = 1;
    bool structureSweep = false;
    bool admissionSweep = false;
    if (std::string_view(argv[firstPath]) == "--structure") {
        structureSweep = true;
        ++firstPath;
    } else if (std::string_view(argv[firstPath]) == "--admission") {
        admissionSweep = true;
        ++firstPath;
    }
    if (firstPath == argc)
        throw std::runtime_error("missing program.ftlpu input");

    std::cout
        << "model,experiment,queue_kind,word_bits,fetch_latency,request_ii,"
           "reservoir_words,decode_window_bits,ddb_depth,"
           "admission_lookahead,ddb_run_capacity,context_expand_width,queue_count,"
           "imem_words,command_count,prime_cycles_max,frontend_cycles,"
           "fetched_words,bit_wait_cycles,reservoir_empty_cycles,"
           "reservoir_full_cycles,decoder_active_cycles,"
           "decoder_starvation_cycles,decoder_ddb_stall_cycles,descriptor_count,"
           "descriptor_decode_cycles,max_descriptor_decode_cycles,"
           "ddb_entries_committed,peak_ddb_occupancy,ddb_full_cycles,"
           "ddb_entry_bits,ddb_storage_bits,"
           "timing_window_block_cycles,active_ram_full_cycles,"
           "capacity_feasible_wait_cycles,admission_stall_cycles,"
           "max_admission_to_start_cycles,peak_active_occupancy,"
           "contexts_generated,semantic_hash\n";
    for (int index = firstPath; index < argc; ++index)
        run_file(std::filesystem::path(argv[index]), structureSweep,
            admissionSweep);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "macro_frontend_fetch_sweep failed: "
              << error.what() << '\n';
    return 1;
}
