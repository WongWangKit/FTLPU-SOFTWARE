#include "ftlpu/software/runtime/binary.hpp"

#include "ftlpu/icu/fu_3d_codec.hpp" // Includes the shared wait_cycle field.
#include "ftlpu/icu/sxm_run_2d.hpp"
#include "ftlpu/icu/vxm_run_2d.hpp"
#include "ftlpu/system/icu.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ftlpu::software::runtime::InstructionKind;
using ftlpu::software::runtime::QueueCommand;
using ftlpu::software::runtime::QueueKind;
using ftlpu::software::runtime::QueueProgram;

struct IcuSpec {
    QueueKind kind{QueueKind::Mem};
    std::size_t index{0};
    std::string filename{};
    std::string location{};
};

struct InstructionRow {
    std::size_t ordinal{0};
    std::size_t pc{0};
    std::size_t words{0};
    std::string opcode{};
    ftlpu::IcuLoop3D loop{};
    std::size_t wait_cycle{0};
    std::size_t expanded{0};
    std::optional<std::size_t> end_cycle{};
    std::string raw{};
    std::string details{};
};

struct ExportSummary {
    std::size_t files{0};
    std::size_t active{0};
    std::size_t coarse_instructions{0};
    std::size_t imem_words{0};
    std::size_t expanded_fu_issues{0};
};

std::string csv_field(std::string_view value)
{
    std::string output(1, '"');
    for (const char ch : value) {
        if (ch == '"') output += '"';
        output += ch;
    }
    output += '"';
    return output;
}

std::string hemisphere_name(std::size_t hemisphere)
{
    return hemisphere == 0 ? "E" : "W";
}

std::string hex32(std::uint32_t value)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(8) << value;
    return output.str();
}

std::size_t physical_word_lanes(QueueKind kind)
{
    switch (kind) {
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
    case QueueKind::MxmDequant:
        return 4;
    default:
        return 3;
    }
}

std::string raw_words(
    const QueueProgram& queue, std::size_t pc, std::size_t wordCount)
{
    std::ostringstream output;
    for (std::size_t word = 0; word < wordCount; ++word) {
        if (word != 0) output << ';';
        const auto& command = queue.commands.at(pc + word);
        output << "0x";
        const auto lanes = command.word_count == 0
            ? physical_word_lanes(queue.kind) : command.word_count;
        for (std::size_t lane = lanes; lane != 0; --lane) {
            output << hex32(command.word_count == 0 && lane == 1
                    ? command.command : command.words[lane - 1]);
        }
    }
    return output.str();
}

template <typename Packet>
Packet packet_at(const QueueProgram& queue, std::size_t pc)
{
    if (pc > queue.commands.size()
        || Packet::kWordCount > queue.commands.size() - pc)
        throw std::logic_error("truncated ICU packet at pc="
            + std::to_string(pc));
    Packet packet{};
    for (std::size_t word = 0; word < Packet::kWordCount; ++word) {
        const auto& command = queue.commands[pc + word];
        if (command.word_count != Packet::kLanesPerWord)
            throw std::logic_error("ICU packet has the wrong physical width");
        for (std::size_t lane = 0; lane < Packet::kLanesPerWord; ++lane)
            packet.words[word].lanes[lane] = command.words[lane];
    }
    return packet;
}

std::size_t expanded_issues(const ftlpu::IcuLoop3D& loop)
{
    return loop.counts[0] * loop.counts[1] * loop.counts[2];
}

std::size_t final_cycle(const ftlpu::IcuLoop3D& loop)
{
    std::size_t cycle = loop.start_cycle;
    for (std::size_t dimension = 0;
         dimension < ftlpu::IcuLoop3D::kDimensions; ++dimension)
        cycle += (loop.counts[dimension] - 1)
            * loop.cycle_strides[dimension];
    return cycle;
}

std::size_t final_cycle(const InstructionRow& row)
{
    return row.end_cycle.value_or(final_cycle(row.loop));
}

std::string stream_description(std::size_t packed)
{
    const auto stream = ftlpu::StreamId::from_packed(packed);
    return std::string("stream=")
        + (stream.direction() == ftlpu::StreamDirection::East ? "E" : "W")
        + std::to_string(stream.index());
}

InstructionRow decode_3d(
    const QueueProgram& queue, std::size_t pc, std::size_t ordinal)
{
    InstructionRow row;
    row.ordinal = ordinal;
    row.pc = pc;
    row.words = ftlpu::software::runtime::
        fu_3d_raw_packet_word_count(queue.kind);
    row.raw = raw_words(queue, pc, row.words);
    std::ostringstream details;
    switch (queue.kind) {
    case QueueKind::Mem: {
        const auto& header = queue.commands.at(pc);
        if (((header.words[2] >> 24) & 0xfU) == 7
            && ((header.words[0] >> 4) & 0x3U) == 3) {
            const auto instruction =
                ftlpu::isa::decode_mem_icu_write_read_2d_instruction(
                    packet_at<ftlpu::isa::EncodedMemIcuWriteRead2DPacket>(
                        queue, pc));
            row.opcode = "WRITE_READ_2D";
            row.loop.counts = {instruction.counts[0],
                instruction.counts[1], 1};
            row.loop.cycle_strides = {instruction.write_cycle_strides[0],
                instruction.write_cycle_strides[1], 1};
            row.end_cycle = ftlpu::detail::
                mem_icu_write_read_2d_last_issue_cycle(instruction);
            row.expanded = 2 * instruction.counts[0]
                * instruction.counts[1];
            details << "start_wait=" << instruction.start_wait
                    << " write_cycle_strides="
                    << instruction.write_cycle_strides[0] << ':'
                    << instruction.write_cycle_strides[1]
                    << " read_start_offset="
                    << instruction.read_start_offset
                    << " read_cycle_strides="
                    << instruction.read_cycle_strides[0] << ':'
                    << instruction.read_cycle_strides[1]
                    << " address_base=" << instruction.base_address
                    << " address_strides="
                    << instruction.address_strides[0] << ':'
                    << instruction.address_strides[1]
                    << " write_" << stream_description(
                           instruction.write_stream)
                    << " read_stream_base="
                    << instruction.read_stream_base
                    << " read_stream_outer_stride="
                    << instruction.read_stream_outer_stride;
            row.details = details.str();
            return row;
        }
        const auto instruction = ftlpu::isa::decode_mem_icu_3d_instruction(
            packet_at<ftlpu::isa::EncodedMemIcu3DPacket>(queue, pc));
        row.loop = instruction.loop;
        switch (instruction.opcode) {
        case ftlpu::MemIcuOpcode::Read3D: row.opcode = "READ_3D"; break;
        case ftlpu::MemIcuOpcode::Write3D: row.opcode = "WRITE_3D"; break;
        case ftlpu::MemIcuOpcode::WriteTap3D:
            row.opcode = "WRITE_TAP_3D";
            break;
        }
        details << "address_base=" << instruction.address.base_address
                << " address_strides=" << instruction.address.inner_stride
                << ':' << instruction.address.middle_stride << ':'
                << instruction.address.outer_inner_stride
                << " outer_group_size="
                << instruction.address.outer_group_size
                << " outer_group_stride="
                << instruction.address.outer_group_stride << ' '
                << stream_description(instruction.stream);
        break;
    }
    case QueueKind::MxmLoad: {
        const auto instruction =
            ftlpu::isa::decode_mxm_load_icu_3d_instruction(
                packet_at<ftlpu::isa::EncodedMxmLoadIcu3DPacket>(queue, pc));
        row.loop = instruction.loop;
        row.opcode = "LOAD_3D";
        details << "weight_buffer_base=" << instruction.weight_buffer_base
                << " weight_buffer_mode="
                << static_cast<unsigned>(instruction.weight_buffer_mode)
                << " weight_column_base=" << instruction.weight_column_base
                << " weight_column_strides="
                << instruction.weight_column_strides[0] << ':'
                << instruction.weight_column_strides[1] << ':'
                << instruction.weight_column_strides[2]
                << " weight_stream_base=" << instruction.weight_stream_base
                << " weight_input_mode="
                << static_cast<unsigned>(instruction.weight_input_mode);
        break;
    }
    case QueueKind::MxmDequant: {
        const auto instruction =
            ftlpu::isa::decode_mxm_dequant_icu_3d_instruction(
                packet_at<ftlpu::isa::EncodedMxmDequantIcu3DPacket>(queue, pc));
        row.loop = instruction.loop;
        row.opcode = "DEQUANT_3D";
        details << "dequant_word=0x"
                << hex32(ftlpu::isa::encode_mxm_dequant_instruction(
                       instruction.instruction));
        break;
    }
    case QueueKind::MxmCompute: {
        const auto instruction =
            ftlpu::isa::decode_mxm_compute_icu_3d_instruction(
                packet_at<ftlpu::isa::EncodedMxmComputeIcu3DPacket>(queue, pc));
        row.loop = instruction.loop;
        row.opcode = instruction.opcode
                    == ftlpu::MxmComputeIcuOpcode::Compute3D
            ? "COMPUTE_3D" : "ACCUMULATOR_READ_3D";
        details << "weight_buffer_base=" << instruction.weight_buffer_base
                << " weight_buffer_mode="
                << static_cast<unsigned>(instruction.weight_buffer_mode)
                << " activation_stream_base="
                << instruction.activation_stream_base
                << " result_stream_base=" << instruction.result_stream_base
                << " accumulator_address_base="
                << instruction.accumulator_address_base
                << " accumulator_address_strides="
                << instruction.accumulator_address_strides[0] << ':'
                << instruction.accumulator_address_strides[1] << ':'
                << instruction.accumulator_address_strides[2]
                << " accumulator_row_stride="
                << instruction.accumulator_row_stride
                << " data_format="
                << static_cast<unsigned>(instruction.data_format)
                << " regular_destination="
                << static_cast<unsigned>(
                       instruction.regular_mode.accumulator_destination)
                << " regular_clear="
                << instruction.regular_mode.accumulator_clear
                << " regular_output_format="
                << static_cast<unsigned>(
                       instruction.regular_mode.accumulator_output_format)
                << " terminal_dimension=" << instruction.terminal_dimension
                << " terminal_destination="
                << static_cast<unsigned>(
                       instruction.terminal_mode.accumulator_destination)
                << " terminal_clear="
                << instruction.terminal_mode.accumulator_clear
                << " terminal_output_format="
                << static_cast<unsigned>(
                       instruction.terminal_mode.accumulator_output_format);
        break;
    }
    case QueueKind::Vxm: {
        const auto instruction =
            ftlpu::isa::decode_vxm_icu_run_2d_instruction(
                packet_at<ftlpu::isa::EncodedVxmIcuRun2DPacket>(queue, pc));
        row.loop = instruction.loop;
        row.opcode = "RUN_2D";
        details << "compact_control=0x" << std::hex << std::setfill('0')
                << std::setw(16) << instruction.instruction.control
                << " immediate_bits=0x" << std::setw(8)
                << instruction.instruction.immediate_bits;
        break;
    }
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute: {
        const auto instruction =
            ftlpu::isa::decode_sxm_icu_run_2d_instruction(
                packet_at<ftlpu::isa::EncodedSxmIcuRun2DPacket>(queue, pc));
        row.loop = instruction.loop;
        row.opcode = "RUN_2D";
        const auto encoded =
            ftlpu::isa::encode_sxm_instruction(instruction.instruction);
        details << "sxm_opcode="
                << static_cast<unsigned>(instruction.instruction.opcode)
                << " permute_map_stride="
                << instruction.permute_map_stride << " sxm_words=";
        for (std::size_t word = 0; word < encoded.words.size(); ++word) {
            if (word != 0) details << ':';
            details << hex32(encoded.words[word]);
        }
        break;
    }
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        throw std::logic_error("C2C is not an FU 3-D packet");
    }
    row.expanded = expanded_issues(row.loop);
    row.details = details.str();
    return row;
}

InstructionRow decode_mem_write_sync(
    const QueueProgram& queue, std::size_t pc, std::size_t ordinal,
    std::size_t startCycle)
{
    const auto packet = ftlpu::software::runtime::
        decode_mem_synchronized_icu_packet(queue, pc);
    const auto& header = packet[0].lanes;
    const auto& native = packet[1].lanes;
    const std::size_t count = ((header[0] >> 2) & 0xffffU) + 1;
    const std::size_t synchronizationTag =
        ((header[0] >> 18) | (header[1] << 14)) & 0xffffU;
    const std::size_t transportDelay = (header[1] >> 2) & 0xffffU;
    constexpr std::uint32_t kStrideMask = (1U << 14) - 1;
    constexpr std::uint32_t kStrideSign = 1U << 13;
    const std::uint32_t strideBits = (header[1] >> 18) & kStrideMask;
    const std::int64_t addressStride = (strideBits & kStrideSign) == 0
        ? static_cast<std::int64_t>(strideBits)
        : -static_cast<std::int64_t>(
              ((~strideBits) & kStrideMask) + 1);
    const std::size_t reservationCycles =
        (header[2] & 0x00ffffffU) + 1;
    const auto encoded = static_cast<ftlpu::isa::EncodedMemInstruction>(
                             native[0])
        | (static_cast<ftlpu::isa::EncodedMemInstruction>(native[1]) << 32);
    const auto instruction = ftlpu::isa::decode_mem_instruction(encoded);

    InstructionRow row;
    row.ordinal = ordinal;
    row.pc = pc;
    row.words = ftlpu::InstructionControlUnit::MemIcu::
        synchronized_packet_word_count;
    row.opcode = "MEM_WRITE_SYNC";
    row.loop.start_cycle = startCycle;
    row.loop.counts = {1, 1, 1};
    row.loop.cycle_strides = {1, 1, 1};
    row.expanded = count;
    row.end_cycle = startCycle + reservationCycles - 1;
    row.raw = raw_words(queue, pc, row.words);
    std::ostringstream details;
    details << "count=" << count
            << " synchronization_tag=" << synchronizationTag
            << " transport_delay=" << transportDelay
            << " reservation_cycles=" << reservationCycles
            << " address_base=" << instruction.address
            << " address_stride=" << addressStride << ' '
            << stream_description(instruction.stream_id().packed());
    row.details = details.str();
    return row;
}

const char* envelope_name(ftlpu::isa::IcuCommandOpcode opcode)
{
    switch (opcode) {
    case ftlpu::isa::IcuCommandOpcode::Instruction: return "INSTRUCTION";
    case ftlpu::isa::IcuCommandOpcode::Nop: return "NOP";
    case ftlpu::isa::IcuCommandOpcode::Repeat: return "REPEAT";
    case ftlpu::isa::IcuCommandOpcode::Extended: return "EXTENDED";
    }
    return "UNKNOWN";
}

std::vector<InstructionRow> decode_queue(const QueueProgram& queue)
{
    std::vector<InstructionRow> rows;
    std::size_t cursor = 0;
    for (std::size_t pc = 0; pc < queue.commands.size();) {
        const auto& command = queue.commands[pc];
        if (ftlpu::software::runtime::
                is_mem_synchronized_raw_packet_header(command)) {
            auto row = decode_mem_write_sync(
                queue, pc, rows.size(), cursor);
            cursor += (queue.commands[pc].words[2] & 0x00ffffffU) + 1;
            pc += row.words;
            rows.push_back(std::move(row));
            continue;
        }
        if (ftlpu::software::runtime::
                is_fu_3d_raw_packet_header(command)) {
            auto row = decode_3d(queue, pc, rows.size());
            row.wait_cycle = row.loop.wait_cycle;
            row.loop.start_cycle += cursor + row.wait_cycle;
            if (row.end_cycle)
                *row.end_cycle += cursor;
            cursor = final_cycle(row) + 1;
            pc += row.words;
            rows.push_back(std::move(row));
            continue;
        }
        InstructionRow row;
        row.ordinal = rows.size();
        row.pc = pc;
        row.words = 1;
        row.opcode = envelope_name(
            ftlpu::isa::decode_icu_command_opcode(command.command));
        row.loop.start_cycle = cursor;
        row.loop.counts = {1, 1, 1};
        row.loop.cycle_strides = {1, 1, 1};
        const auto opcode =
            ftlpu::isa::decode_icu_command_opcode(command.command);
        if (opcode == ftlpu::isa::IcuCommandOpcode::Nop) {
            const auto cycles =
                ftlpu::isa::decode_icu_nop_cycles(command.command);
            row.expanded = 0;
            row.end_cycle = cursor + (cycles == 0 ? 0 : cycles - 1);
            row.details = "nop_cycles=" + std::to_string(cycles);
            cursor += cycles;
        } else {
            row.expanded = 1;
            ++cursor;
        }
        row.raw = raw_words(queue, pc, 1);
        if (!row.details.empty()) row.details += ' ';
        row.details += "instruction_kind="
            + std::to_string(static_cast<unsigned>(command.instruction_kind));
        rows.push_back(std::move(row));
        ++pc;
    }
    return rows;
}

std::vector<IcuSpec> physical_icus(
    const ftlpu::software::runtime::BinaryProgram& program)
{
    std::vector<IcuSpec> result;
    for (std::size_t index = 0;
         index < ftlpu::InstructionControlUnit::kMemQueues; ++index) {
        const auto hemisphere = index
            / ftlpu::InstructionControlUnit::kMemQueuesPerHemisphere;
        const auto local = index
            % ftlpu::InstructionControlUnit::kMemQueuesPerHemisphere;
        const auto slice = local / ftlpu::hw::kMemBanksPerSlice;
        const auto bank = local % ftlpu::hw::kMemBanksPerSlice;
        const auto side = hemisphere_name(hemisphere);
        std::ostringstream file;
        file << "mem_" << side << "_slice" << std::setfill('0')
             << std::setw(2) << slice << "_bank" << bank
             << ".icu.csv";
        std::ostringstream location;
        location << "hemisphere=" << side << " slice=" << slice
                 << " bank=" << bank;
        result.push_back({QueueKind::Mem, index,
            file.str(), location.str()});
    }
    const auto mxmCount = ftlpu::hw::kHemispheres
        * program.hardware.mxms_per_hemisphere;
    for (const auto kind : {QueueKind::MxmLoad, QueueKind::MxmCompute,
             QueueKind::MxmDequant}) {
        for (std::size_t index = 0; index < mxmCount; ++index) {
            const auto hemisphere = index
                / program.hardware.mxms_per_hemisphere;
            const auto local = index
                % program.hardware.mxms_per_hemisphere;
            const auto side = hemisphere_name(hemisphere);
            const std::string resource =
                ftlpu::software::runtime::queue_kind_name(kind);
            result.push_back({kind, index,
                resource + '_' + side + "_mxm" + std::to_string(local)
                    + ".icu.csv",
                "hemisphere=" + side + " mxm=" + std::to_string(local)});
        }
    }
    for (std::size_t index = 0;
         index < ftlpu::InstructionControlUnit::kVxmQueues; ++index)
        result.push_back({QueueKind::Vxm, index,
            "vxm_alu" + std::to_string(index) + ".icu.csv",
            "alu_queue=" + std::to_string(index)});
    for (const auto kind : {QueueKind::SxmTranspose, QueueKind::SxmPermute}) {
        for (std::size_t hemisphere = 0;
             hemisphere < ftlpu::hw::kHemispheres; ++hemisphere) {
            const auto side = hemisphere_name(hemisphere);
            const std::string resource =
                ftlpu::software::runtime::queue_kind_name(kind);
            result.push_back({kind, hemisphere,
                resource + '_' + side + ".icu.csv",
                "hemisphere=" + side});
        }
    }
    for (const auto kind : {
             QueueKind::C2cDma, QueueKind::C2cTx, QueueKind::C2cRx}) {
        for (std::size_t hemisphere = 0;
             hemisphere < ftlpu::hw::kHemispheres; ++hemisphere) {
            const auto side = hemisphere_name(hemisphere);
            const std::string resource =
                ftlpu::software::runtime::queue_kind_name(kind);
            result.push_back({kind, hemisphere,
                resource + '_' + side + ".icu.csv",
                "hemisphere=" + side});
        }
    }
    return result;
}

void write_icu_file(const std::filesystem::path& path,
    const ftlpu::software::runtime::BinaryProgram& program,
    const IcuSpec& spec, const std::vector<InstructionRow>& rows,
    std::size_t imemWords)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot create ICU file: " + path.string());
    output << "# target=" << program.target_name << '\n'
           << "# max_cycle=" << program.max_cycle << '\n'
           << "# resource="
           << ftlpu::software::runtime::queue_kind_name(spec.kind) << '\n'
           << "# queue_index=" << spec.index << '\n'
           << "# physical_location=" << spec.location << '\n'
           << "# coarse_instructions=" << rows.size() << '\n'
           << "# mem_write_sync="
           << std::count_if(rows.begin(), rows.end(),
                  [](const InstructionRow& row) {
                      return row.opcode == "MEM_WRITE_SYNC";
                  }) << '\n'
           << "# imem_words=" << imemWords << '\n'
           << "instruction,pc_word,imem_words,opcode,start_cycle,wait_cycle,end_cycle,"
              "count0,count1,count2,cycle_stride0,cycle_stride1,"
              "cycle_stride2,expanded_fu_issues,raw_words,details\n";
    for (const auto& row : rows)
        output << row.ordinal << ',' << row.pc << ',' << row.words << ','
               << row.opcode << ',' << row.loop.start_cycle << ','
               << row.wait_cycle << ','
               << final_cycle(row) << ',' << row.loop.counts[0] << ','
               << row.loop.counts[1] << ',' << row.loop.counts[2] << ','
               << row.loop.cycle_strides[0] << ','
               << row.loop.cycle_strides[1] << ','
               << row.loop.cycle_strides[2] << ',' << row.expanded << ','
               << csv_field(row.raw) << ',' << csv_field(row.details) << '\n';
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 3)
        throw std::runtime_error(
            "usage: ftlpu_icu_program_export program.ftlpu output_directory");
    const auto input = std::filesystem::absolute(argv[1]).lexically_normal();
    const auto outputDirectory =
        std::filesystem::absolute(argv[2]).lexically_normal();
    const auto program =
        ftlpu::software::runtime::read_binary_program(input);
    std::filesystem::create_directories(outputDirectory);

    std::map<std::pair<QueueKind, std::size_t>, const QueueProgram*> queues;
    for (const auto& queue : program.queues) {
        const auto key = std::pair {queue.kind, queue.index};
        if (!queues.emplace(key, &queue).second)
            throw std::logic_error("duplicate ICU queue in binary");
    }
    const auto specs = physical_icus(program);
    for (const auto& [key, queue] : queues) {
        (void)queue;
        const bool mapped = std::any_of(specs.begin(), specs.end(),
            [&](const IcuSpec& spec) {
                return spec.kind == key.first && spec.index == key.second;
            });
        if (!mapped)
            throw std::logic_error(
                "binary queue has no physical ICU in the current topology");
    }

    // The physical topology can change between exporter versions. Remove only
    // files owned by this exporter so obsolete per-port files (for example
    // *_read.icu.csv and *_write.icu.csv) cannot survive beside the current
    // one-file-per-ICU image.
    std::size_t staleFilesRemoved = 0;
    for (const auto& entry :
         std::filesystem::directory_iterator(outputDirectory)) {
        if (!entry.is_regular_file()) continue;
        const auto filename = entry.path().filename().string();
        if (!filename.ends_with(".icu.csv")) continue;
        if (std::filesystem::remove(entry.path())) ++staleFilesRemoved;
    }

    std::ofstream index(outputDirectory / "index.csv", std::ios::trunc);
    if (!index) throw std::runtime_error("cannot create ICU index.csv");
    index << "file,resource,queue_index,physical_location,active,"
             "coarse_instructions,imem_words,expanded_fu_issues,"
             "first_cycle,last_cycle\n";

    std::size_t active = 0;
    std::size_t coarse = 0;
    std::size_t imem = 0;
    std::size_t expanded = 0;
    std::size_t memWriteSync = 0;
    std::map<QueueKind, ExportSummary> resourceSummaries;
    for (const auto& spec : specs) {
        const auto found = queues.find({spec.kind, spec.index});
        std::vector<InstructionRow> rows;
        std::size_t queueWords = 0;
        if (found != queues.end()) {
            rows = decode_queue(*found->second);
            queueWords = found->second->commands.size();
        }
        write_icu_file(outputDirectory / spec.filename,
            program, spec, rows, queueWords);
        std::size_t queueExpanded = 0;
        for (const auto& row : rows) {
            queueExpanded += row.expanded;
            if (row.opcode == "MEM_WRITE_SYNC") ++memWriteSync;
        }
        std::optional<std::size_t> firstCycle;
        std::optional<std::size_t> lastCycle;
        for (const auto& row : rows) {
            firstCycle = std::min(firstCycle.value_or(row.loop.start_cycle),
                row.loop.start_cycle);
            lastCycle = std::max(lastCycle.value_or(final_cycle(row)),
                final_cycle(row));
        }
        index << csv_field(spec.filename) << ','
              << ftlpu::software::runtime::queue_kind_name(spec.kind) << ','
              << spec.index << ',' << csv_field(spec.location) << ','
              << (!rows.empty()) << ',' << rows.size() << ',' << queueWords
              << ',' << queueExpanded << ',';
        if (firstCycle) index << *firstCycle;
        index << ',';
        if (lastCycle) index << *lastCycle;
        index << '\n';
        auto& resourceSummary = resourceSummaries[spec.kind];
        ++resourceSummary.files;
        if (!rows.empty()) ++resourceSummary.active;
        resourceSummary.coarse_instructions += rows.size();
        resourceSummary.imem_words += queueWords;
        resourceSummary.expanded_fu_issues += queueExpanded;
        if (!rows.empty()) ++active;
        coarse += rows.size();
        imem += queueWords;
        expanded += queueExpanded;
    }

    std::ofstream readme(outputDirectory / "README.txt", std::ios::trunc);
    readme << "Source: " << input.string() << '\n'
           << "Target: " << program.target_name << '\n'
           << "Qwen2.5 prefill horizon: 0.." << program.max_cycle << '\n'
           << "Physical ICU files: " << specs.size() << '\n'
           << "Active ICU files: " << active << '\n'
           << "ICU coarse instructions: " << coarse << '\n'
           << "MEM_WRITE_SYNC instructions: " << memWriteSync << '\n'
           << "Physical i-MEM words: " << imem << '\n'
           << "Expanded FU issues: " << expanded << "\n\n"
           << "Each *.icu.csv is one physical ICU program. One data row is "
              "one coarse ICU instruction; pc_word addresses physical i-MEM "
              "and raw_words preserves every encoded 96/128-bit word. Empty "
              "files represent physical ICUs with no static prefill work. "
              "start_cycle/end_cycle are reconstructed by replaying the "
              "preceding NOP durations; work packets encode only relative "
              "loop offsets.\n\n"
           << "A compiler .ftlpu image precedes dynamic C2C weight-page "
              "linking and normally contains no MEM_WRITE_SYNC packets. "
              "Export the ModelSession linked .ftlpu image to inspect the "
              "actual MEM ICU queues loaded into CModel.\n\n"
           << "C2C ICU programs in this binary:\n";
    for (const auto kind : {
             QueueKind::C2cDma, QueueKind::C2cRx, QueueKind::C2cTx}) {
        const auto& summary = resourceSummaries.at(kind);
        readme << "  "
               << ftlpu::software::runtime::queue_kind_name(kind)
               << ": files=" << summary.files
               << " active=" << summary.active
               << " coarse_instructions=" << summary.coarse_instructions
               << " imem_words=" << summary.imem_words
               << " expanded_fu_issues=" << summary.expanded_fu_issues
               << '\n';
    }
    readme << "Empty C2C files represent physical endpoints with no commands "
              "in this .ftlpu instruction image.\n";

    std::cout << "ICU export written: directory=" << outputDirectory.string()
              << " files=" << specs.size() << " active=" << active
              << " stale_files_removed=" << staleFilesRemoved
              << " coarse_instructions=" << coarse
              << " mem_write_sync=" << memWriteSync
              << " imem_words=" << imem
              << " expanded_fu_issues=" << expanded << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu_icu_program_export failed: " << ex.what() << '\n';
    return 1;
}
