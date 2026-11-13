#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"

#include "ftlpu/c2c/icu_instruction.hpp"
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
#include <set>
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

// Export-only repetition coordinates; these are not a hardware instruction.
struct IssuePattern {
    std::size_t start_cycle{0};
    std::array<std::size_t,3> counts{1,1,1};
    std::array<std::size_t,3> cycle_strides{1,1,1};
};

struct InstructionRow {
    std::size_t ordinal{0};
    std::string phase{"execution"};
    std::optional<std::size_t> load_id{};
    std::size_t pc{0};
    std::size_t words{0};
    std::string opcode{};
    IssuePattern loop{};
    std::size_t wait_cycle{0};
    std::size_t expanded{0};
    std::optional<std::size_t> end_cycle{};
    std::string raw{};
    std::string details{};
    bool packed_macro{false};
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
        if (queue.kind == QueueKind::Vxm
            && command.command == 0 && command.word_count == 3
            && command.extension_words.empty()) {
            const auto physical = ftlpu::isa::encode_vxm_icu_instruction({
                std::uint64_t(command.words[0]) | (std::uint64_t(command.words[1]) << 32),
                command.words[2]});
            for (std::size_t lane = 3; lane != 0; --lane)
                output << hex32(physical.lanes[lane - 1]);
            continue;
        }
        const auto lanes = command.word_count == 0
            ? physical_word_lanes(queue.kind) : command.word_count;
        for (std::size_t lane = lanes; lane != 0; --lane) {
            output << hex32(command.word_count == 0 && lane == 1
                    ? command.command : command.words[lane - 1]);
        }
    }
    return output.str();
}

std::size_t expanded_issues(const IssuePattern& loop)
{
    return loop.counts[0] * loop.counts[1] * loop.counts[2];
}

std::size_t final_cycle(const IssuePattern& loop)
{
    std::size_t cycle = loop.start_cycle;
    for (std::size_t dimension = 0;
         dimension < 3; ++dimension)
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

InstructionRow decode_mem_sync(
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
    const auto encoded =
        (static_cast<ftlpu::isa::EncodedMemInstruction>(native[0])
         | (static_cast<ftlpu::isa::EncodedMemInstruction>(native[1]) << 32))
            >> 2
        | (static_cast<ftlpu::isa::EncodedMemInstruction>(native[2]) << 62);
    const auto instruction = ftlpu::isa::decode_mem_instruction(encoded);

    InstructionRow row;
    row.ordinal = ordinal;
    row.pc = pc;
    row.words = ftlpu::InstructionControlUnit::MemIcu::
        synchronized_packet_word_count;
    row.opcode = instruction.opcode == ftlpu::MemOpcode::Read
        ? "MEM_READ_SYNC" : "MEM_WRITE_SYNC";
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
        if ((queue.kind == QueueKind::Mem || queue.kind == QueueKind::MxmLoad ||
             queue.kind == QueueKind::MxmCompute || queue.kind == QueueKind::MxmDequant ||
             queue.kind == QueueKind::SxmTranspose || queue.kind == QueueKind::SxmPermute) &&
            ftlpu::software::runtime::is_macro_schedule_command(command)) {
            // Packed instructions cross fetch-word boundaries. The
            // container reconstructs semantic runs, not per-command iMEM PCs.
            // Export their real domains and retain the complete physical image.
            const auto schedule=ftlpu::software::runtime::decode_macro_schedule_command(command);
            const auto native=static_cast<std::uint64_t>(command.words[0])
                | (static_cast<std::uint64_t>(command.words[1]) << 32);
            InstructionRow row;
            row.ordinal=rows.size();row.packed_macro=true;
            row.loop.start_cycle=schedule.start_cycle;
            row.loop.counts={schedule.inner_count,schedule.outer_count,1};
            row.loop.cycle_strides={schedule.inner_interval,schedule.outer_interval,1};
            row.wait_cycle=schedule.start_cycle>=cursor?schedule.start_cycle-cursor:0;
            row.expanded=schedule.inner_count*schedule.outer_count;
            row.details="encoding=packed_macro command_ordinal="+std::to_string(pc)
                +" induction_target="+std::to_string(static_cast<unsigned>(schedule.induction_target))
                +" induction_stride0="+std::to_string(schedule.inner_stride)
                +" induction_stride1="+std::to_string(schedule.outer_stride)
                +" wait_cycle=derived_idle_gap";
            if(queue.kind==QueueKind::Mem) {
                const auto instruction=ftlpu::isa::decode_mem_instruction(native);
                row.opcode=instruction.opcode==ftlpu::MemOpcode::Read?"MEM_READ_MACRO"
                    : instruction.preserve_stream?"MEM_WRITE_TAP_MACRO":"MEM_WRITE_MACRO";
                row.details+=" address_base="+std::to_string(instruction.address)
                    +" "+stream_description(instruction.stream_id().packed());
            } else if(queue.kind==QueueKind::SxmTranspose || queue.kind==QueueKind::SxmPermute) {
                const auto instruction=ftlpu::software::runtime::decode_sxm_command(command);
                row.opcode=queue.kind==QueueKind::SxmTranspose?"SXM_TRANSPOSE_MACRO":"SXM_PERMUTE_MACRO";
                row.details+=" source_stream_base="+std::to_string(instruction.src_streams[0].stream)
                    +" destination_stream_base="+std::to_string(instruction.dst_streams[0].stream);
            } else if(queue.kind==QueueKind::MxmDequant) {
                row.opcode="MXM_DEQUANT_MACRO";
                row.details+=" scale_bf16="+std::to_string(command.words[0]&0xffffU);
            } else {
                const auto instruction=ftlpu::isa::decode_mxm_instruction(native);
                switch(instruction.opcode) {
                case ftlpu::MxmControlOpcode::IW: row.opcode="MXM_LOAD_MACRO";break;
                case ftlpu::MxmControlOpcode::Compute: row.opcode="MXM_COMPUTE_MACRO";break;
                case ftlpu::MxmControlOpcode::AccumulatorRead: row.opcode="MXM_ACC_READ_MACRO";break;
                case ftlpu::MxmControlOpcode::IS: row.opcode="MXM_IS_MACRO";break;
                }
                row.details+=" native_control="+std::to_string(native);
                if(instruction.opcode==ftlpu::MxmControlOpcode::IW) {
                    row.details+=" weight_buffer="+std::to_string(instruction.weight_buffer)
                        +" weight_column="+std::to_string(instruction.weight_column)
                        +" weight_inner_column="+std::to_string(instruction.weight_inner_column)
                        +" weight_stream_base="+std::to_string(instruction.weight_stream_base)
                        +" weight_load_mode="+std::to_string(static_cast<unsigned>(instruction.weight_load_mode))
                        +" weight_input_mode="+std::to_string(static_cast<unsigned>(instruction.weight_input_mode));
                } else {
                    row.details+=" weight_buffer="+std::to_string(instruction.weight_buffer)
                        +" accumulator_address="+std::to_string(instruction.accumulator_address)
                        +" accumulator_clear="+std::to_string(instruction.accumulator_clear)
                        +" accumulator_destination="+std::to_string(static_cast<unsigned>(instruction.accumulator_destination))
                        +" output_format="+std::to_string(static_cast<unsigned>(instruction.accumulator_output_format))
                        +" activation_format="+std::to_string(static_cast<unsigned>(instruction.data_format));
                }
            }
            cursor=final_cycle(row)+1;
            rows.push_back(std::move(row));++pc;continue;
        }
        if (ftlpu::software::runtime::
                is_mem_synchronized_raw_packet_header(command)) {
            auto row = decode_mem_sync(
                queue, pc, rows.size(), cursor);
            cursor += (queue.commands[pc].words[2] & 0x00ffffffU) + 1;
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
        if (ftlpu::software::runtime::
                is_icu_control_raw_word_command(command)
            && [] (ftlpu::IcuControlOpcode controlOpcode) {
                return controlOpcode == ftlpu::IcuControlOpcode::Sync
                    || controlOpcode == ftlpu::IcuControlOpcode::WaitEvent
                    || controlOpcode == ftlpu::IcuControlOpcode::Notify;
            }(ftlpu::software::runtime::
                    decode_icu_control_raw_word(command).opcode)) {
            const auto control = ftlpu::software::runtime::
                decode_icu_control_raw_word(command);
            if (control.opcode == ftlpu::IcuControlOpcode::Sync)
                row.opcode = "SYNC";
            else if (control.opcode
                == ftlpu::IcuControlOpcode::WaitEvent) {
                const bool c2cProducer = queue.kind == QueueKind::C2cDma
                    || queue.kind == QueueKind::C2cRx
                    || queue.kind == QueueKind::C2cTx;
                row.opcode = c2cProducer ? "SYNC_LAUNCH" : "SYNC_PAGE";
            }
            else
                row.opcode = "NOTIFY";
            row.expanded = 0;
            row.end_cycle = cursor;
            row.details = "actual_wait=runtime_dependent";
            if (control.opcode == ftlpu::IcuControlOpcode::WaitEvent)
                row.details += " event_tag="
                    + std::to_string(control.event_tag);
            ++cursor;
        } else if (opcode == ftlpu::isa::IcuCommandOpcode::Nop) {
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
        if (command.instruction_kind == InstructionKind::C2cEndpoint
            && (queue.kind == QueueKind::C2cRx
                || queue.kind == QueueKind::C2cTx)) {
            const auto packet = ftlpu::software::runtime::decode_c2c_raw_word(
                command, InstructionKind::C2cEndpoint);
            std::ostringstream c2c;
            if (queue.kind == QueueKind::C2cRx) {
                const auto instruction =
                    ftlpu::C2cIcuPacketCodec::decode_rx(packet);
                row.opcode = "RECEIVE";
                c2c << "lane=" << instruction.lane
                    << " fabric_stream=W" << instruction.fabric_stream
                    << " vectors=" << instruction.vector_count
                    << " synchronization_tag=" << instruction.sync_tag;
            } else {
                const auto instruction =
                    ftlpu::C2cIcuPacketCodec::decode_tx(packet);
                row.opcode = "SEND";
                c2c << "lane=" << instruction.lane
                    << " fabric_stream=W" << instruction.fabric_stream
                    << " vectors=" << instruction.vector_count
                    << " synchronization_tag=" << instruction.sync_tag;
            }
            row.details = c2c.str();
        }
        row.raw = raw_words(queue, pc, 1);
        if (queue.kind == QueueKind::Vxm
            && opcode == ftlpu::isa::IcuCommandOpcode::Instruction) {
            if (command.word_count != 3 || !command.extension_words.empty())
                throw std::invalid_argument("invalid sequential VXM configuration");
            const auto decoded = ftlpu::isa::decode_vxm_instruction(queue.index, {
                std::uint64_t(command.words[0]) | (std::uint64_t(command.words[1]) << 32),
                command.words[2]});
            const auto& config = decoded.instruction;
            row.opcode = "CONFIG";
            row.end_cycle = row.loop.start_cycle + config.repeat_count - 1;
            row.details = std::string("operation=") + ftlpu::VxmLane::operation_name(config.operation)
                + " repeat_count=" + std::to_string(config.repeat_count)
                + " chain_depth=" + std::to_string(static_cast<unsigned>(decoded.chain_depth))
                + " accumulator_reset=" + std::to_string(config.accumulator_reset)
                + " accumulator_write=" + std::to_string(config.accumulator_write)
                + " accumulator_emit=" + std::to_string(config.accumulator_emit)
                + " precision=" + std::to_string(static_cast<unsigned>(config.precision))
                + " output_type=" + std::to_string(static_cast<unsigned>(config.output_type));
        }
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
           << "# mem_read_sync="
           << std::count_if(rows.begin(), rows.end(),
                  [](const InstructionRow& row) {
                      return row.opcode == "MEM_READ_SYNC";
                  }) << '\n'
           << "# mem_write_sync="
           << std::count_if(rows.begin(), rows.end(),
                  [](const InstructionRow& row) {
                      return row.opcode == "MEM_WRITE_SYNC";
                  }) << '\n'
           << "# sync="
           << std::count_if(rows.begin(), rows.end(),
                  [](const InstructionRow& row) {
                      return row.opcode == "SYNC";
                  }) << '\n'
           << "# page_sync="
           << std::count_if(rows.begin(), rows.end(),
                  [](const InstructionRow& row) {
                      return row.opcode == "SYNC_PAGE";
                  }) << '\n'
           << "# launch_sync="
           << std::count_if(rows.begin(), rows.end(),
                  [](const InstructionRow& row) {
                      return row.opcode == "SYNC_LAUNCH";
                  }) << '\n'
           << "# imem_words=" << imemWords << '\n'
           << "instruction,phase,load_id,pc_word,imem_words,opcode,start_cycle,wait_cycle,end_cycle,"
              "count0,count1,count2,cycle_stride0,cycle_stride1,"
              "cycle_stride2,expanded_fu_issues,raw_words,details\n";
    for (const auto& row : rows) {
        output << row.ordinal << ',' << row.phase << ',';
        if (row.load_id) output << *row.load_id;
        output << ',';
        if(!row.packed_macro)output << row.pc;
        output << ',';
        if(!row.packed_macro)output << row.words;
        output << ',' << row.opcode << ',' << row.loop.start_cycle << ','
               << row.wait_cycle << ','
               << final_cycle(row) << ',' << row.loop.counts[0] << ','
               << row.loop.counts[1] << ',' << row.loop.counts[2] << ','
               << row.loop.cycle_strides[0] << ','
               << row.loop.cycle_strides[1] << ','
               << row.loop.cycle_strides[2] << ',' << row.expanded << ','
               << csv_field(row.raw) << ',' << csv_field(row.details) << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc < 3)
        throw std::runtime_error(
            "usage: ftlpu_icu_program_export program.ftlpu output_directory "
            "[--pre-execution-dir directory] [--compute-imem-segments]");
    std::optional<std::filesystem::path> preExecutionDirectory;
    bool exportComputeSegments = false;
    for (int argument = 3; argument < argc; ++argument) {
        const std::string_view option = argv[argument];
        if (option == "--pre-execution-dir" && argument + 1 < argc)
            preExecutionDirectory = argv[++argument];
        else if (option == "--compute-imem-segments")
            exportComputeSegments = true;
        else
            throw std::runtime_error("unknown or incomplete export option: " + std::string(option));
    }
    const auto input = std::filesystem::absolute(argv[1]).lexically_normal();
    const auto outputDirectory =
        std::filesystem::absolute(argv[2]).lexically_normal();
    const auto program =
        ftlpu::software::runtime::read_binary_program(input);
    std::vector<ftlpu::software::runtime::BinaryProgram> preExecution;
    if (preExecutionDirectory) {
        std::vector<std::pair<std::size_t, std::filesystem::path>> files;
        for (const auto& entry :
             std::filesystem::directory_iterator(*preExecutionDirectory)) {
            if (!entry.is_regular_file()) continue;
            const auto name = entry.path().filename().string();
            if (!name.starts_with("pre_execution_")
                || !name.ends_with(".ftlpu")) continue;
            const auto id = name.substr(14, name.size() - 20);
            if (id.empty() || !std::all_of(id.begin(), id.end(),
                    [](char ch) { return ch >= '0' && ch <= '9'; }))
                throw std::runtime_error(
                    "invalid pre-execution image name: " + name);
            files.emplace_back(std::stoull(id), entry.path());
        }
        std::ranges::sort(files);
        for (std::size_t index = 0; index < files.size(); ++index) {
            if (files[index].first != index)
                throw std::runtime_error(
                    "pre-execution image IDs must be contiguous from zero");
            preExecution.push_back(
                ftlpu::software::runtime::read_binary_program(
                    files[index].second));
        }
    }
    std::filesystem::create_directories(outputDirectory);

    // Reproduce the exact segmentation/relative waits used by CModelRuntime.
    // This is an export of the linked image, not a new lowering or schedule.
    // MEM/MXM queues remain resident; only the runtime's overflowed compute
    // queues are reloaded at compiler-declared drain boundaries.
    std::vector<ftlpu::software::runtime::BinaryProgram> computeSegments;
    std::set<std::pair<QueueKind, std::size_t>> reloadedQueues;
    if (exportComputeSegments) {
        computeSegments = ftlpu::software::runtime::plan_compute_imem_segments(program);
        std::ofstream segments(outputDirectory / "compute_imem_segments.csv", std::ios::trunc);
        if (!segments) throw std::runtime_error("cannot create compute iMEM segment index");
        segments << "load_id,logical_start,logical_end_exclusive,queue_count,used_slots,capacity_checked\n";
        std::size_t begin = 0;
        for (std::size_t load = 0; load < computeSegments.size(); ++load) {
            const auto& segment = computeSegments[load];
            const auto report = ftlpu::software::runtime::analyze_physical_imem(segment);
            if (!report.fits()) throw std::logic_error("exported iMEM segment exceeds physical capacity");
            segments << load << ',' << begin << ',' << begin + segment.max_cycle
                     << ',' << segment.queues.size() << ',' << report.used_slots << ",1\n";
            begin += segment.max_cycle;
            for (const auto& queue : segment.queues)
                reloadedQueues.emplace(queue.kind, queue.index);
        }
    } else
        std::filesystem::remove(outputDirectory / "compute_imem_segments.csv");

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
        if (!mapped) {
            auto message = std::string {
                "binary queue has no physical ICU in the current topology: "};
            message += ftlpu::software::runtime::queue_kind_name(key.first);
            message += " index=";
            message += std::to_string(key.second);
            throw std::logic_error(message);
        }
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
        if (!filename.ends_with(".icu.csv") && !filename.ends_with(".imem.bin")) continue;
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
    std::size_t memReadSync = 0;
    std::size_t memWriteSync = 0;
    std::size_t ordinarySync = 0;
    std::size_t pageSync = 0;
    std::size_t launchSync = 0;
    std::map<QueueKind, ExportSummary> resourceSummaries;
    for (const auto& spec : specs) {
        const auto found = queues.find({spec.kind, spec.index});
        std::vector<InstructionRow> rows;
        std::size_t queueWords = 0;
        const auto exportImage = [&](const QueueProgram& queue,
                                     std::vector<InstructionRow>& decodedRows,
                                     const std::string& suffix) -> std::size_t {
            using namespace ftlpu::software::runtime;
            if (queue.commands.empty()) return 0;
            const auto images = prepare_program_images({queue}, 0, program.hardware.mxms_per_hemisphere);
            if (images.size() != 1) throw std::logic_error("one ICU queue must produce one physical image");
            const auto& bytes = images.front().bytes;
            const auto name=spec.filename+suffix+".imem.bin";
            std::ofstream image(outputDirectory/name,std::ios::binary|std::ios::trunc);
            if(!image)throw std::runtime_error("cannot write physical iMEM image");
            image.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
            if(!image)throw std::runtime_error("cannot finish physical iMEM image");
            for(auto& row:decodedRows) {
                if(row.packed_macro) row.raw="file:"+name;
                else row.details += " imem_file="+name;
            }
            const bool mxm=queue.kind==QueueKind::MxmLoad || queue.kind==QueueKind::MxmCompute || queue.kind==QueueKind::MxmDequant;
            const auto wordBits=mxm ? program.hardware.icu_mxm_instruction_bits : program.hardware.icu_mem_instruction_bits;
            if (!wordBits || bytes.size()*8%wordBits) throw std::logic_error("physical image has a partial word");
            return bytes.size()*8/wordBits;
        };

        for (std::size_t load = 0; load < preExecution.size(); ++load) {
            for (const auto& queue : preExecution[load].queues) {
                if (queue.kind != spec.kind || queue.index != spec.index)
                    continue;
                auto pageRows = decode_queue(queue);
                queueWords += exportImage(queue,pageRows,".pre_execution_"+std::to_string(load));
                for (auto& row : pageRows) {
                    row.ordinal = rows.size();
                    row.phase = "pre_execution";
                    row.load_id = load;
                    rows.push_back(std::move(row));
                }
            }
        }
        if (reloadedQueues.contains({spec.kind, spec.index})) {
            std::size_t begin = 0;
            for (std::size_t load = 0; load < computeSegments.size(); ++load) {
                for (const auto& queue : computeSegments[load].queues) {
                    if (queue.kind != spec.kind || queue.index != spec.index) continue;
                    auto segmentRows = decode_queue(queue);
                    queueWords += exportImage(queue,segmentRows,".execution_segment_"+std::to_string(load));
                    for (auto& row : segmentRows) {
                        row.ordinal = rows.size();
                        row.phase = "execution_segment";
                        row.load_id = load;
                        // Cycle columns stay on the common logical timeline;
                        // raw words, wait_cycle and pc_word stay segment-local.
                        row.loop.start_cycle += begin;
                        if (row.end_cycle) *row.end_cycle += begin;
                        row.details += " imem_segment_start=" + std::to_string(begin);
                        rows.push_back(std::move(row));
                    }
                }
                begin += computeSegments[load].max_cycle;
            }
        } else if (found != queues.end()) {
            auto executionRows = decode_queue(*found->second);
            queueWords += exportImage(*found->second,executionRows,"");
            for (auto& row : executionRows) {
                row.ordinal = rows.size();
                rows.push_back(std::move(row));
            }
        }
        // Common-barrier MEM transport storage belongs to the same physical
        // ICU. Export it beside its compute Macro image, without hiding SYNC.
        for (const auto& transport : program.transport_queues) {
            if (transport.kind != spec.kind || transport.index != spec.index) continue;
            auto transportRows = decode_queue(transport);
            queueWords += exportImage(transport, transportRows, ".transport");
            for (auto& row : transportRows) {
                row.ordinal = rows.size();
                row.phase = "execution_transport";
                row.details += " cycle_domain=transport_local common_barrier_controlled=1";
                rows.push_back(std::move(row));
            }
        }
        write_icu_file(outputDirectory / spec.filename,
            program, spec, rows, queueWords);
        std::size_t queueExpanded = 0;
        for (const auto& row : rows) {
            queueExpanded += row.expanded;
            if (row.opcode == "MEM_READ_SYNC") ++memReadSync;
            if (row.opcode == "MEM_WRITE_SYNC") ++memWriteSync;
            if (row.opcode == "SYNC") ++ordinarySync;
            if (row.opcode == "SYNC_PAGE") ++pageSync;
            if (row.opcode == "SYNC_LAUNCH") ++launchSync;
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

    const bool overlapBudgetFallback = std::any_of(program.timelines.begin(),
        program.timelines.end(), [](const auto& timeline) {
            return timeline.name == "c2c.serial_fallback.proof_work_bound";
        });
    // These are properties of the executed/linked image, not the source
    // compiler's requested policy. Preserve safe-fallback provenance in CSV
    // exports so a requested auto policy is not reported as actual overlap.
    std::ofstream metadata(outputDirectory / "program.json", std::ios::trunc);
    metadata << std::boolalpha
        << "{\n  \"common_barrier_protocol\": " << program.common_barrier_protocol
        << ",\n  \"transport_overlap_protocol\": " << program.transport_overlap_protocol
        << ",\n  \"overlap_analysis_fallback\": " << overlapBudgetFallback
        << ",\n  \"compute_imem_segment_export\": " << exportComputeSegments
        << ",\n  \"compute_imem_segment_count\": " << computeSegments.size()
        << ",\n  \"weight_page_uses\": " << program.weight_page_uses.size() << "\n}\n";
    std::ofstream readme(outputDirectory / "README.txt", std::ios::trunc);
    readme << "Source: " << input.string() << '\n'
           << "Target: " << program.target_name << '\n'
           << "Executable prefill horizon: 0.." << program.max_cycle << '\n'
           << "Pre-execution ICU loads: " << preExecution.size() << '\n'
           << "Compute iMEM segments (runtime planner): " << computeSegments.size() << '\n'
           << "Physical ICU files: " << specs.size() << '\n'
           << "Active ICU files: " << active << '\n'
           << "ICU coarse instructions: " << coarse << '\n'
           << "MEM_READ_SYNC instructions: " << memReadSync << '\n'
           << "MEM_WRITE_SYNC instructions: " << memWriteSync << '\n'
           << "Ordinary SYNC instructions: " << ordinarySync << '\n'
           << "Page SYNC instructions: " << pageSync << '\n'
           << "C2C launch SYNC instructions: " << launchSync << '\n'
           << "Total loaded i-MEM words: " << imem << '\n'
           << "Expanded FU issues: " << expanded << "\n\n"
           << "Each *.icu.csv is one physical ICU program. One data row is "
              "one coarse ICU instruction; phase and load_id identify "
              "pre-execution page loads, whose pc_word restarts at zero. "
              "pc_word addresses physical i-MEM within its load, "
              "imem_words sums words loaded across phases rather than "
              "simultaneous i-MEM capacity, "
              "and raw_words preserves every encoded 96/128-bit word. Empty "
              "files represent physical ICUs with no static prefill work. "
              "start_cycle/end_cycle replay NOP durations, packet wait_cycle, "
              "and synchronized MEM reservations. SYNC_PAGE is a tagged "
              "WAIT_EVENT placed at a static execution-epoch boundary; "
              "all pages required by that epoch share its event_tag. Ordinary "
              "SYNC and SYNC_PAGE mark their fetch cycles; their actual wait "
              "is dynamic. SYNC_LAUNCH is the corresponding tagged wait in "
              "a C2C producer queue.\n\n"
           << "Packed MEM/MXM macro rows retain the decoded domain and command ordinal. "
              "Their pc_word/imem_words cells are empty because runs can cross "
              "physical word boundaries; index.csv counts the actual packed words. "
              "raw_words links the complete relocated *.imem.bin image used by the "
              "raw loader. wait_cycle is the derived idle gap for these absolute-start "
              "domains, not an extra encoded wait field.\n\n"
           << "With --compute-imem-segments, execution_segment rows reproduce "
              "the runtime planner's per-drain compute images. load_id and "
              "pc_word identify the segment and its local iMEM PC; raw_words "
              "and wait_cycle preserve the segment-local encoding. Cycle "
              "columns include the segment's logical start offset, not dynamic "
              "C2C waits; use runtime.pipeline.csv for actual FU cycles. "
              "compute_imem_segments.csv lists checked segment capacities. "
              "Segmentation is reconstructed from the linked image, not a "
              "separate hardware instruction-fetch trace.\n\n"
           << "Compiler-authored KV-cache MEM_READ_SYNC and executable "
              "weight-page MEM_WRITE_SYNC packets appear in the MEM ICU "
              "files. Ordinary SYNC appears "
              "only when encoded in an ICU queue. Use --pre-execution-dir "
              "with the ModelSession snapshots to include the standalone "
              "C2C RX/DMA queues and their ordinary SYNC instructions.\n\n"
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
              << " mem_read_sync=" << memReadSync
              << " mem_write_sync=" << memWriteSync
              << " sync=" << ordinarySync
              << " page_sync=" << pageSync
              << " launch_sync=" << launchSync
              << " imem_words=" << imem
              << " expanded_fu_issues=" << expanded << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu_icu_program_export failed: " << ex.what() << '\n';
    return 1;
}
