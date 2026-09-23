#include "ftlpu/software/runtime/runtime_execution_trace.hpp"

#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace ftlpu::software::runtime {
namespace {

std::string csv_field(const std::string& value)
{
    std::string result = "\"";
    for (const char ch : value) {
        if (ch == '"') result += '"';
        result += ch;
    }
    return result + '"';
}

bool issued(IcuQueueAction action)
{
    switch (action) {
    case IcuQueueAction::FunctionalIssue:
    case IcuQueueAction::RepeatIssue:
    case IcuQueueAction::Repeat2DIssue:
    case IcuQueueAction::MacroIssue:
    case IcuQueueAction::MemStreamNdIssue:
    case IcuQueueAction::MemSliceProgramIssue:
    case IcuQueueAction::MxmStreamNdIssue:
    case IcuQueueAction::VxmStreamNdIssue:
    case IcuQueueAction::SxmTileProgramIssue:
    case IcuQueueAction::VxmRun2DIssue:
    case IcuQueueAction::SxmRun2DIssue:
    case IcuQueueAction::Mem3DIssue:
    case IcuQueueAction::MxmLoad3DIssue:
    case IcuQueueAction::MxmDequant3DIssue:
    case IcuQueueAction::MxmCompute3DIssue:
    case IcuQueueAction::SynchronizedIssue:
        return true;
    default:
        return false;
    }
}

std::string stream_name(StreamId stream)
{
    return std::string(stream.direction() == StreamDirection::East ? "E" : "W")
        + std::to_string(stream.index());
}

const char* mem_opcode_name(const MemInstruction& instruction)
{
    if (instruction.preserve_stream) return "WriteTap";
    switch (instruction.opcode) {
    case MemOpcode::Read: return "Read";
    case MemOpcode::Write: return "Write";
    case MemOpcode::Gather: return "Gather";
    case MemOpcode::Scatter: return "Scatter";
    }
    return "Unknown";
}

const char* icu_action_name(IcuQueueAction action)
{
    switch (action) {
    case IcuQueueAction::Idle: return "idle";
    case IcuQueueAction::WaitingForStart: return "waiting_for_start";
    case IcuQueueAction::PrefetchOnly: return "prefetch";
    case IcuQueueAction::FunctionalIssue: return "issue";
    case IcuQueueAction::Nop: return "nop";
    case IcuQueueAction::NopWait: return "nop_wait";
    case IcuQueueAction::RepeatIssue: return "repeat_issue";
    case IcuQueueAction::RepeatWait: return "repeat_wait";
    case IcuQueueAction::Repeat2DIssue: return "repeat2d_issue";
    case IcuQueueAction::Repeat2DWait: return "repeat2d_wait";
    case IcuQueueAction::MacroIssue: return "macro_issue";
    case IcuQueueAction::MacroWait: return "macro_wait";
    case IcuQueueAction::MemStreamNdIssue: return "stream_nd_issue";
    case IcuQueueAction::MemStreamNdWait: return "stream_nd_wait";
    case IcuQueueAction::MemSliceProgramIssue: return "slice_program_issue";
    case IcuQueueAction::MemSliceProgramWait: return "slice_program_wait";
    case IcuQueueAction::MxmStreamNdIssue: return "mxm_stream_nd_issue";
    case IcuQueueAction::MxmStreamNdWait: return "mxm_stream_nd_wait";
    case IcuQueueAction::VxmStreamNdIssue: return "vxm_stream_nd_issue";
    case IcuQueueAction::VxmStreamNdWait: return "vxm_stream_nd_wait";
    case IcuQueueAction::VxmRun2DIssue: return "vxm_run2d_issue";
    case IcuQueueAction::VxmRun2DWait: return "vxm_run2d_wait";
    case IcuQueueAction::SxmRun2DIssue: return "sxm_run2d_issue";
    case IcuQueueAction::SxmRun2DWait: return "sxm_run2d_wait";
    case IcuQueueAction::SxmTileProgramIssue: return "sxm_tile_issue";
    case IcuQueueAction::SxmTileProgramWait: return "sxm_tile_wait";
    case IcuQueueAction::Mem3DIssue: return "mem3d_issue";
    case IcuQueueAction::Mem3DWait: return "mem3d_wait";
    case IcuQueueAction::MxmLoad3DIssue: return "mxm_load3d_issue";
    case IcuQueueAction::MxmLoad3DWait: return "mxm_load3d_wait";
    case IcuQueueAction::MxmDequant3DIssue: return "mxm_dequant3d_issue";
    case IcuQueueAction::MxmDequant3DWait: return "mxm_dequant3d_wait";
    case IcuQueueAction::MxmCompute3DIssue: return "mxm_compute3d_issue";
    case IcuQueueAction::MxmCompute3DWait: return "mxm_compute3d_wait";
    case IcuQueueAction::ThreeDDecodeWait: return "decode3d_wait";
    case IcuQueueAction::ThreeDContextFullWait: return "context_full_wait";
    case IcuQueueAction::SynchronizedWait: return "sync_data_wait";
    case IcuQueueAction::SynchronizedDelay: return "sync_transport_delay";
    case IcuQueueAction::SynchronizedIssue: return "sync_issue";
    case IcuQueueAction::SyncWait: return "barrier_wait";
    case IcuQueueAction::SyncRelease: return "barrier_release";
    case IcuQueueAction::EventWait: return "event_wait";
    case IcuQueueAction::EventRelease: return "event_release";
    case IcuQueueAction::Notify: return "notify";
    case IcuQueueAction::Underflow: return "underflow";
    }
    return "unknown";
}

std::string transfer_bytes_hex(
    const MemArrayModel::MemTransfer& transfer)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto byte : transfer.bytes)
        output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}

std::pair<std::string, std::string> describe_mem(
    std::size_t queue, const MemInstruction& instruction,
    IcuQueueAction action)
{
    const bool east =
        queue < InstructionControlUnit::kMemQueuesPerHemisphere;
    const std::size_t local =
        queue % InstructionControlUnit::kMemQueuesPerHemisphere;
    const std::size_t slice = local / hw::kMemBanksPerSlice;
    const std::size_t bank = local % hw::kMemBanksPerSlice;
    const bool write = instruction.opcode == MemOpcode::Write
        || instruction.opcode == MemOpcode::Scatter;
    std::ostringstream detail;
    detail << "slice=" << slice << " bank=" << bank
           << " operation=" << (write ? "write" : "read")
           << " stream=" << stream_name(instruction.stream_id());
    std::string opcode = mem_opcode_name(instruction);
    if (action == IcuQueueAction::Mem3DIssue)
        opcode += "3D";
    else if (action == IcuQueueAction::SynchronizedIssue)
        opcode += "Sync";
    return {std::string("MEM.") + (east ? "E." : "W.")
            + opcode, detail.str()};
}

std::pair<std::string, std::string> describe_mxm(std::size_t queue,
    std::size_t per_hemisphere, QueueKind kind,
    const MxmControlInstruction& instruction)
{
    const bool east = queue < per_hemisphere;
    std::ostringstream detail;
    if (instruction.opcode == MxmControlOpcode::IW) {
        detail << "IW buffer=" << instruction.weight_buffer
               << " mode="
               << (instruction.weight_input_mode
                           == MxmWeightInputMode::Int8DequantBf16
                       ? "int8_dequant_bf16" : "direct16");
    } else if (instruction.opcode == MxmControlOpcode::Compute) {
        detail << "Compute buffer=" << instruction.weight_buffer
               << " format=" << mxm_data_format_name(instruction.data_format)
               << " destination="
               << (instruction.accumulator_destination
                           == MxmAccumulatorDestination::Stream
                       ? "stream" : "sram")
               << " clear=" << (instruction.accumulator_clear ? 1 : 0);
    } else if (instruction.opcode == MxmControlOpcode::AccumulatorRead) {
        detail << "AccumulatorRead clear="
               << (instruction.accumulator_clear ? 1 : 0);
    } else {
        detail << "Decode";
    }
    return {std::string("MXM.") + (east ? "E" : "W")
            + std::to_string(queue % per_hemisphere)
            + (kind == QueueKind::MxmLoad ? ".Load" : ".Compute"),
        detail.str()};
}

template <typename Queue>
const typename Queue::FunctionalInstruction* last_issued(const Queue& queue)
{
    if (!issued(queue.last_trace().action)
        || !queue.last_dispatched().has_value())
        return nullptr;
    return &*queue.last_dispatched();
}

template <typename Queue>
std::string with_issue_pc(std::string detail, const Queue& queue)
{
    if (queue.last_trace().issue_pc.has_value())
        detail += " pc=" + std::to_string(*queue.last_trace().issue_pc);
    return detail;
}

} // namespace

void RuntimeExecutionTrace::reset(const BinaryProgram& program)
{
    begin_segment(program, 0, false);
}

void RuntimeExecutionTrace::begin_segment(const BinaryProgram& program,
    std::int64_t cycleOffset, bool append)
{
    queues_.clear();
    last_event_by_resource_.clear();
    if (!append) {
        events_.clear();
        sequence_ = 0;
    }
    cycle_offset_ = cycleOffset;
    mxms_per_hemisphere_ = std::max<std::size_t>(
        1, program.hardware.mxms_per_hemisphere);
    for (const QueueProgram& queue : program.queues)
        queues_.push_back({queue.kind, queue.index});
}

void RuntimeExecutionTrace::record_interval(std::int64_t startCycle,
    std::int64_t endCycle, std::string resource, std::string detail,
    std::size_t issueCount)
{
    startCycle += cycle_offset_;
    endCycle += cycle_offset_;
    if (endCycle <= startCycle) return;
    // One display resource can contain many independently issuing queues.
    // Keep a separate run for each instruction detail so repeated MEM
    // slice/bank issues collapse into actual continuous intervals.
    const std::string mergeKey = resource + '\0' + detail;
    const auto previous = last_event_by_resource_.find(mergeKey);
    if (previous != last_event_by_resource_.end()) {
        Event& event = events_[previous->second];
        if (event.repeat_count == 1 && startCycle <= event.end_cycle) {
            event.end_cycle = std::max(event.end_cycle, endCycle);
            event.issue_count += issueCount;
            return;
        }
        const std::int64_t duration = event.end_cycle - event.start_cycle;
        const std::int64_t lastStart = event.start_cycle
            + static_cast<std::int64_t>(event.repeat_count - 1)
                * event.repeat_interval;
        const std::int64_t interval = startCycle - lastStart;
        if (interval > 0 && endCycle - startCycle == duration
            && (event.repeat_count == 1
                || interval == event.repeat_interval)) {
            if (event.repeat_count == 1)
                event.repeat_interval = interval;
            ++event.repeat_count;
            event.issue_count += issueCount;
            return;
        }
    }
    last_event_by_resource_[mergeKey] = events_.size();
    events_.push_back(Event {startCycle, endCycle, std::move(resource),
        std::move(detail), issueCount, 1, 0, 1, 0, sequence_++});
}

void RuntimeExecutionTrace::sample(TspSliceSystem& system,
    std::uint64_t physicalCycle, bool programIssueEnabled,
    const BinaryWeightPageUse* waitingPage)
{
    auto& icu = system.icu();
    const auto record = [&](std::string resource, std::string detail,
                            std::size_t duration = 1) {
        record_interval(static_cast<std::int64_t>(physicalCycle),
            static_cast<std::int64_t>(physicalCycle + duration),
            std::move(resource), std::move(detail));
    };

    if (!programIssueEnabled) {
        std::ostringstream detail;
        detail << "wait=page_ready";
        if (waitingPage != nullptr)
            detail << " binding=" << waitingPage->binding_index
                   << " page=" << waitingPage->page_index
                   << " bank=" << waitingPage->bank
                   << " consumer_cycle=" << waitingPage->ready_cycle;
        record("ICU.PageReadyWait", detail.str());
    }

    for (const QueueRef& ref : queues_) {
        // During a page-ready hold, all ordinary queues retain their previous
        // trace state because they are not ticked. MEM queues are different:
        // transport-only mode ticks them so an in-order MEM_WRITE_SYNC can
        // keep draining the arriving page. Sample only those MEM queues while
        // the ordinary program is paused; otherwise stale MXM/VXM/SXM state
        // would look like repeated FU issues.
        if (!programIssueEnabled && ref.kind != QueueKind::Mem)
            continue;
        {
            // Binary MXM queue ids are dense in the executable's logical
            // topology. The ICU arrays use the CModel's physical stride.
            const auto physicalMxmIndex = [&] {
                const auto hemisphere =
                    ref.index / mxms_per_hemisphere_;
                const auto localMxm =
                    ref.index % mxms_per_hemisphere_;
                return hemisphere * hw::kMxmsPerHemisphere + localMxm;
            };
            switch (ref.kind) {
            case QueueKind::Mem: {
                const auto& queue = icu.mem_iq(ref.index);
                const auto* instruction = last_issued(queue);
                if (instruction == nullptr) break;
                auto [resource, detail] = describe_mem(ref.index,
                    *instruction, queue.last_trace().action);
                record(std::move(resource),
                    with_issue_pc(std::move(detail), queue));
                break;
            }
            case QueueKind::MxmLoad:
            case QueueKind::MxmCompute: {
                const auto sampleMxm = [&](const auto& queue) {
                    const auto* instruction = last_issued(queue);
                    if (instruction == nullptr) return;
                    auto [resource, detail] = describe_mxm(ref.index,
                        mxms_per_hemisphere_, ref.kind, *instruction);
                    record(std::move(resource),
                        with_issue_pc(std::move(detail), queue));
                };
                if (ref.kind == QueueKind::MxmLoad)
                    sampleMxm(icu.mxm_load_iq(physicalMxmIndex()));
                else
                    sampleMxm(icu.mxm_compute_iq(physicalMxmIndex()));
                break;
            }
            case QueueKind::MxmDequant: {
                const auto& queue =
                    icu.mxm_dequant_iq(physicalMxmIndex());
                const auto* instruction = last_issued(queue);
                if (instruction == nullptr) break;
                const bool east = ref.index < mxms_per_hemisphere_;
                record(std::string("MXM.") + (east ? "E" : "W")
                        + std::to_string(ref.index % mxms_per_hemisphere_)
                        + ".Dequant",
                    with_issue_pc("dequant", queue));
                break;
            }
            case QueueKind::Vxm: {
                const auto& queue = icu.vxm_iq(ref.index);
                const auto* packet = last_issued(queue);
                if (packet == nullptr) break;
                const auto decoded =
                    VxmCompactInstructionCodec::decode(ref.index, *packet);
                std::ostringstream detail;
                detail << VxmLane::operation_name(
                              decoded.instruction.operation)
                       << " depth="
                       << static_cast<std::size_t>(decoded.chain_depth);
                record("VXM.C" + std::to_string(ref.index),
                    with_issue_pc(detail.str(), queue),
                    decoded.instruction.repeat_count);
                break;
            }
            case QueueKind::SxmTranspose:
            case QueueKind::SxmPermute: {
                const auto side = static_cast<Hemisphere>(ref.index);
                const auto& queue = ref.kind == QueueKind::SxmTranspose
                    ? icu.sxm_transpose_iq(side)
                    : icu.sxm_permute_iq(side);
                if (last_issued(queue) == nullptr) break;
                record(std::string("SXM.")
                        + (side == Hemisphere::East ? "E." : "W.")
                        + (ref.kind == QueueKind::SxmTranspose
                                  ? "Transpose" : "Permute"),
                    with_issue_pc(ref.kind == QueueKind::SxmTranspose
                            ? "transpose" : "permute", queue));
                break;
            }
            case QueueKind::C2cDma:
            case QueueKind::C2cTx:
            case QueueKind::C2cRx:
                // Sampled once below, independent of the program hold.
                break;
            }
        }
    }

    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        const char* sideName = side == 0 ? "E" : "W";
        const auto& dmaQueue = icu.c2c_dma_iq(hemisphere);
        if (const auto* instruction = last_issued(dmaQueue)) {
            std::ostringstream detail;
            detail << "stream=" << instruction->stream_index
                   << " vectors=" << instruction->vector_count
                   << " ddr4=" << instruction->ddr4_address;
            record(std::string("C2C.") + sideName + ".DMA",
                with_issue_pc(detail.str(), dmaQueue));
        }
        const auto& rxQueue = icu.c2c_rx_iq(hemisphere);
        if (const auto* instruction = last_issued(rxQueue)) {
            std::ostringstream detail;
            detail << "stream=" << instruction->stream_index
                   << " vectors=" << instruction->vector_count
                   << " target_slice=" << instruction->consumer.mem_slice
                   << " bank=" << instruction->consumer.mem_bank;
            record(std::string("C2C.") + sideName + ".RX",
                with_issue_pc(detail.str(), rxQueue));
        }
    }
}

void RuntimeExecutionTrace::write_csv(
    const std::filesystem::path& path) const
{
    std::ofstream output(path, std::ios::trunc);
    if (!output)
        throw std::runtime_error(
            "cannot open runtime execution trace: " + path.string());
    output << "start,end,resource,detail,pattern,inner_count,inner_interval,"
              "inner_stride,outer_count,outer_interval,outer_stride,skip_first,"
              "induction,base_delta\n";
    auto events = events_;
    std::ranges::sort(events, [](const Event& left, const Event& right) {
        if (left.start_cycle != right.start_cycle)
            return left.start_cycle < right.start_cycle;
        if (left.resource != right.resource)
            return left.resource < right.resource;
        return left.sequence < right.sequence;
    });
    std::vector<Event> compacted;
    compacted.reserve(events.size());
    std::unordered_map<std::string, std::size_t> lastPattern;
    for (Event& event : events) {
        const std::int64_t duration = event.end_cycle - event.start_cycle;
        const std::string key = event.resource + '\0' + event.detail + '\0'
            + std::to_string(duration) + ':'
            + std::to_string(event.repeat_count) + ':'
            + std::to_string(event.repeat_interval);
        const auto previous = lastPattern.find(key);
        if (previous != lastPattern.end()) {
            Event& pattern = compacted[previous->second];
            const std::int64_t lastOuterStart = pattern.start_cycle
                + static_cast<std::int64_t>(pattern.outer_count - 1)
                    * pattern.outer_interval;
            const std::int64_t interval = event.start_cycle - lastOuterStart;
            if (interval > 0
                && (pattern.outer_count == 1
                    || interval == pattern.outer_interval)) {
                if (pattern.outer_count == 1)
                    pattern.outer_interval = interval;
                ++pattern.outer_count;
                pattern.issue_count += event.issue_count;
                continue;
            }
        }
        lastPattern[key] = compacted.size();
        compacted.push_back(std::move(event));
    }
    std::ranges::sort(compacted, [](const Event& left, const Event& right) {
        if (left.start_cycle != right.start_cycle)
            return left.start_cycle < right.start_cycle;
        if (left.resource != right.resource)
            return left.resource < right.resource;
        return left.sequence < right.sequence;
    });
    for (const Event& event : compacted) {
        std::ostringstream detail;
        detail << event.detail << " source=runtime issues="
               << event.issue_count;
        output << event.start_cycle << ',' << event.end_cycle << ','
               << csv_field(event.resource) << ','
               << csv_field(detail.str())
               << ',' << csv_field(event.outer_count > 1
                        ? "repeat2d"
                        : event.repeat_count > 1 ? "repeat" : "single")
               << ',' << event.repeat_count << ','
               << event.repeat_interval
               << ",0," << event.outer_count << ','
               << event.outer_interval << ",0,0,\"none\",0\n";
    }
}

void MemExecutionTrace::begin_segment(
    std::int64_t cycleOffset, bool append)
{
    if (!append) events_.clear();
    cycle_offset_ = cycleOffset;
}

void MemExecutionTrace::write_header(std::ostream& output)
{
    output << "cycle,hemisphere,slice,bank,port,tile,stage,action,opcode,"
              "address,stream_direction,stream_index,sr_column,vector_tag,"
              "data_hex,source,pc,iq_before,iq_after\n";
}

void MemExecutionTrace::write_event(
    std::ostream& output, const Event& event)
{
    output << event.cycle << ','
           << (event.hemisphere == 0 ? "E" : "W") << ','
           << event.slice << ',' << event.bank << ','
           << (event.port == 0 ? "read" : "write") << ',';
    if (event.tile >= 0) output << event.tile;
    output << ',' << csv_field(event.stage)
           << ',' << csv_field(event.action)
           << ',' << csv_field(event.opcode) << ',';
    if (event.address >= 0) output << event.address;
    output << ',' << event.stream_direction << ',';
    if (event.stream_index >= 0) output << event.stream_index;
    output << ',';
    if (event.sr_column >= 0) output << event.sr_column;
    output << ',';
    if (event.has_vector_tag) output << event.vector_tag;
    output << ',' << csv_field(event.data_hex)
           << ',' << csv_field(event.source) << ',';
    if (event.pc >= 0) output << event.pc;
    output << ',' << event.iq_before << ',' << event.iq_after << '\n';
}

void MemExecutionTrace::stream_csv(const std::filesystem::path& path)
{
    if (stream_output_.is_open()) stream_output_.close();
    stream_output_.clear();
    stream_buffer_.resize(8 * 1024 * 1024);
    stream_output_.rdbuf()->pubsetbuf(
        stream_buffer_.data(),
        static_cast<std::streamsize>(stream_buffer_.size()));
    stream_output_.open(path, std::ios::trunc);
    if (!stream_output_)
        throw std::runtime_error(
            "cannot open streaming MEM execution trace: " + path.string());
    stream_path_ = path;
    events_.clear();
    write_header(stream_output_);
}

void MemExecutionTrace::record(Event event)
{
    if (stream_output_.is_open())
        write_event(stream_output_, event);
    else
        events_.push_back(std::move(event));
}

void MemExecutionTrace::sample(TspSliceSystem& system,
    std::uint64_t physicalCycle, bool programIssueEnabled)
{
    const auto cycle = cycle_offset_
        + static_cast<std::int64_t>(physicalCycle);
    auto& icu = system.icu();

    const auto addIcu = [&](std::size_t queueIndex,
                            const auto& queue,
                            const char* source,
                            bool gated) {
        const auto side = queueIndex
            / InstructionControlUnit::kMemQueuesPerHemisphere;
        const auto local = queueIndex
            % InstructionControlUnit::kMemQueuesPerHemisphere;
        const auto slice = local / hw::kMemBanksPerSlice;
        const auto bank = local % hw::kMemBanksPerSlice;
        const auto& trace = queue.last_trace();
        const bool pending = !queue.done();
        if (gated && !pending) return;
        if (!gated && trace.action == IcuQueueAction::Idle) return;

        Event event;
        event.cycle = cycle;
        event.hemisphere = static_cast<std::uint16_t>(side);
        event.slice = static_cast<std::uint16_t>(slice);
        event.bank = static_cast<std::uint16_t>(bank);
        event.port = 0;
        event.stage = "icu";
        event.action = gated ? "program_gated"
                             : icu_action_name(trace.action);
        event.source = source;
        event.pc = trace.issue_pc.has_value()
            ? static_cast<std::int64_t>(*trace.issue_pc) : -1;
        event.iq_before = trace.iq_before;
        event.iq_after = trace.iq_after;
        if (!gated) {
            if (const auto* instruction = last_issued(queue)) {
                event.opcode = mem_opcode_name(*instruction);
                event.port = static_cast<std::uint16_t>(
                    instruction->opcode == MemOpcode::Write
                    || instruction->opcode == MemOpcode::Scatter);
                event.address = static_cast<std::int64_t>(
                    instruction->address);
                event.stream_direction =
                    instruction->stream_id().direction()
                            == StreamDirection::East ? "E" : "W";
                event.stream_index = static_cast<std::int32_t>(
                    instruction->stream_id().index());
            }
        }
        record(std::move(event));
    };

    for (std::size_t queue = 0;
         queue < InstructionControlUnit::kMemQueues; ++queue) {
        const auto& memQueue = icu.mem_iq(queue);
        const bool gated = !programIssueEnabled
            && memQueue.last_trace().action
                == IcuQueueAction::ProgramPaused;
        addIcu(queue, memQueue, "program", gated);
    }

    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        const auto& mem = system.mem_array(hemisphere);
        for (const auto& trace : mem.executed_instructions()) {
            Event event;
            event.cycle = cycle;
            event.hemisphere = static_cast<std::uint16_t>(side);
            event.slice = static_cast<std::uint16_t>(trace.mem_slice);
            event.bank = static_cast<std::uint16_t>(trace.bank);
            event.port = static_cast<std::uint16_t>(
                trace.instruction.opcode == MemOpcode::Write
                    || trace.instruction.opcode == MemOpcode::Scatter);
            event.tile = static_cast<std::int16_t>(trace.tile);
            event.stage = "pipeline";
            event.action = "execute";
            event.opcode = mem_opcode_name(trace.instruction);
            event.address = static_cast<std::int64_t>(
                trace.instruction.address);
            event.stream_direction =
                trace.instruction.stream_id().direction()
                        == StreamDirection::East ? "E" : "W";
            event.stream_index = static_cast<std::int32_t>(
                trace.instruction.stream_id().index());
            event.source = "mem_fu";
            record(std::move(event));
        }
        for (const auto& transfer : mem.executed_transfers()) {
            Event event;
            event.cycle = cycle;
            event.hemisphere = static_cast<std::uint16_t>(side);
            event.slice = static_cast<std::uint16_t>(transfer.mem_slice);
            event.bank = static_cast<std::uint16_t>(transfer.bank);
            event.port = static_cast<std::uint16_t>(
                transfer.kind
                    == MemArrayModel::MemTransfer::Kind::StoreStreamToSram);
            event.tile = static_cast<std::int16_t>(transfer.tile);
            event.stage = "sram";
            event.action = transfer.kind
                    == MemArrayModel::MemTransfer::Kind::StoreStreamToSram
                ? "write_commit" : "read_to_sr";
            event.opcode = transfer.kind
                    == MemArrayModel::MemTransfer::Kind::StoreStreamToSram
                ? "Write" : "Read";
            event.address = static_cast<std::int64_t>(transfer.address);
            event.stream_direction =
                transfer.stream.direction() == StreamDirection::East
                    ? "E" : "W";
            event.stream_index = static_cast<std::int32_t>(
                transfer.stream.index());
            event.sr_column = static_cast<std::int64_t>(
                transfer.sr_column);
            event.vector_tag = transfer.vector_tag;
            event.has_vector_tag = true;
            event.data_hex = transfer_bytes_hex(transfer);
            event.source = "mem_fu";
            record(std::move(event));
        }
    }
}

void MemExecutionTrace::write_csv(
    const std::filesystem::path& path) const
{
    if (stream_output_.is_open()) {
        if (std::filesystem::absolute(path).lexically_normal()
            != std::filesystem::absolute(stream_path_).lexically_normal())
            throw std::logic_error(
                "streaming MEM execution trace was opened at a different path");
        stream_output_.flush();
        if (!stream_output_)
            throw std::runtime_error(
                "failed to flush MEM execution trace: " + path.string());
        return;
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output)
        throw std::runtime_error(
            "cannot open MEM execution trace: " + path.string());
    write_header(output);
    auto events = events_;
    std::ranges::sort(events, [](const Event& left, const Event& right) {
        return std::tie(left.cycle, left.hemisphere, left.slice, left.bank,
                   left.port, left.tile, left.stage)
            < std::tie(right.cycle, right.hemisphere, right.slice, right.bank,
                   right.port, right.tile, right.stage);
    });
    for (const auto& event : events) {
        write_event(output, event);
    }
}

} // namespace ftlpu::software::runtime
