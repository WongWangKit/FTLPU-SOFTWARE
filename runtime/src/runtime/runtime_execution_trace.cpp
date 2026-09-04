#include "ftlpu/software/runtime/runtime_execution_trace.hpp"

#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

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
    case IcuQueueAction::MxmStreamNdIssue:
    case IcuQueueAction::VxmStreamNdIssue:
    case IcuQueueAction::SxmTileProgramIssue:
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

std::pair<std::string, std::string> describe_mem(
    std::size_t queue, const MemInstruction& instruction)
{
    const bool east =
        queue < InstructionControlUnit::kMemQueuesPerHemisphere;
    const std::size_t local =
        queue % InstructionControlUnit::kMemQueuesPerHemisphere;
    const std::size_t slice = local / hw::kMemBanksPerSlice;
    const std::size_t bank = local % hw::kMemBanksPerSlice;
    std::ostringstream detail;
    detail << "slice=" << slice << " bank=" << bank
           << " stream=" << stream_name(instruction.stream_id());
    return {std::string("MEM.") + (east ? "E." : "W.")
            + mem_opcode_name(instruction), detail.str()};
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

    if (programIssueEnabled) {
        for (const QueueRef& ref : queues_) {
            switch (ref.kind) {
            case QueueKind::Mem: {
                const auto& queue = icu.mem_iq(ref.index);
                const auto* instruction = last_issued(queue);
                if (instruction == nullptr) break;
                auto [resource, detail] = describe_mem(ref.index, *instruction);
                record(std::move(resource),
                    with_issue_pc(std::move(detail), queue));
                break;
            }
            case QueueKind::MxmLoad:
            case QueueKind::MxmCompute: {
                const auto& queue = ref.kind == QueueKind::MxmLoad
                    ? icu.mxm_load_iq(ref.index)
                    : icu.mxm_compute_iq(ref.index);
                const auto* instruction = last_issued(queue);
                if (instruction == nullptr) break;
                auto [resource, detail] = describe_mxm(ref.index,
                    mxms_per_hemisphere_, ref.kind, *instruction);
                record(std::move(resource),
                    with_issue_pc(std::move(detail), queue));
                break;
            }
            case QueueKind::MxmDequant: {
                const auto& queue = icu.mxm_dequant_iq(ref.index);
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

} // namespace ftlpu::software::runtime
