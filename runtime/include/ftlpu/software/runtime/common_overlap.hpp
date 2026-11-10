#pragma once

#include "ftlpu/software/runtime/issue_inspector.hpp"
#include "ftlpu/core/instruction_codec.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

namespace ftlpu::software::runtime {

// Offline proof over the final computation image. Expansion here is static
// analysis only: the executable retains its real finite Macro frontend, and
// DMA/RX/MEM synchronization is never removed from the executable image.
// A lease reserves a whole MEM issue port and a whole ordinary SR route in
// [E,C), including all tiles and passive bridges. At C computation is drained;
// a late transport keeps its lease in HOLD until its last SRAM commit and
// transport tail have drained. No assumed DDR latency enters this proof.
class CommonOverlapCalendar {
public:
    struct Selection {
        std::uint32_t launch{};
        std::vector<std::uint16_t> streams{};
    };
    explicit CommonOverlapCalendar(const BinaryProgram& program,
                                   std::size_t max_issues = 8'000'000)
        : streams_(program.hardware.streams_per_direction),
          lanes_(program.hardware.c2c_streams_per_direction),
          tile_rows_(program.hardware.tile_rows),
          stream_issues_(streams_), stream_spans_(streams_)
    {
        const auto& h = program.hardware;
        guard_ = static_cast<std::uint64_t>(h.system_register_columns)
            + h.mem_boundary_register_columns + h.mxm_pipeline_rows + h.tile_rows
            + std::max({h.mem_to_sxm_latency, h.mem_to_mxm_latency,
                h.mxm0_accumulator_latency, h.mxm1_accumulator_latency,
                h.accumulator_to_vxm_latency, h.accumulator_read_to_vxm_latency,
                h.swiglu_write_latency, h.qk_iw_to_compute_latency,
                h.vxm_weight_to_iw_latency});
        if (!streams_ || streams_ != hw::kWestStreams || !lanes_ || lanes_ > streams_)
            throw std::invalid_argument("unsupported overlap SR geometry");
        auto computation = program;
        std::set<std::size_t> mem_queues;
        std::erase_if(computation.queues, [&](const auto& q) {
            if (q.kind == QueueKind::C2cDma || q.kind == QueueKind::C2cRx
                || q.kind == QueueKind::C2cTx) return true;
            if (q.kind != QueueKind::Mem) return false;
            if (!mem_queues.insert(q.index).second)
                throw std::runtime_error("binary contains a duplicate MEM ICU queue");
            const auto runs = inspect_mem_macro_runs(program, q, max_issues);
            if (!runs) return false;
            for (const auto& run : *runs) {
                all_issues_.add(run.first, run.last);
                mem_issues_[q.index].add(run.first, run.last);
                if (run.stream >= 2 * streams_)
                    throw std::invalid_argument("overlap proof packed SR is invalid");
                if (run.stream >= streams_)
                    stream_issues_[run.stream - streams_].add(run.first, run.last);
            }
            return true;
        });
        const auto timeline = inspect_logical_issues(computation,max_issues);
        for (const auto& q : timeline.queues) for (const auto& issue : q.issues) {
            if (issue.cycle > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("overlap proof exceeds finite cycle bound");
            all_issues_.push_back(issue.cycle);
            const auto& c = issue.instruction;
            const auto bits = static_cast<std::uint64_t>(c.words[0])
                | (static_cast<std::uint64_t>(c.words[1]) << 32);
            const auto mark = [&](std::size_t base, std::size_t width) {
                if (base >= streams_ || width > streams_ - base)
                    throw std::invalid_argument("overlap proof SR range is invalid");
                for (std::size_t i=0; i<width; ++i)
                    stream_issues_[base+i].push_back(issue.cycle);
            };
            const auto packed = [&](std::size_t stream) {
                if (stream >= 2 * streams_)
                    throw std::invalid_argument("overlap proof packed SR is invalid");
                if (stream >= streams_) mark(stream-streams_,1);
            };
            switch (c.instruction_kind) {
            case InstructionKind::Mem: {
                const auto m = isa::decode_mem_instruction(bits);
                if (m.opcode != MemOpcode::Read && m.opcode != MemOpcode::Write)
                    throw std::invalid_argument("overlap proof requires direct MEM addresses");
                mem_issues_[q.index].push_back(issue.cycle);
                packed(m.stream);
                break;
            }
            case InstructionKind::Mxm: {
                const auto m = isa::decode_mxm_instruction(bits);
                if (m.opcode == MxmControlOpcode::AccumulatorRead
                    || (m.opcode == MxmControlOpcode::Compute
                        && m.accumulator_destination == MxmAccumulatorDestination::Stream)
                    || (m.opcode == MxmControlOpcode::IS
                        && m.decode_operation == MxmIsOperation::StreamCompute
                        && m.accumulator_destination == MxmAccumulatorDestination::Stream))
                    mark(m.stream_base % streams_, h.mxm_result_streams);
                break;
            }
            case InstructionKind::MxmDequant: break; // local weight registers
            case InstructionKind::Vxm: {
                const auto decoded = isa::decode_vxm_instruction(q.index,
                    {bits, c.words[2]});
                const auto& instruction = decoded.instruction;
                // A configuration can consume a vector train after its ICU
                // issue. Reserve through its final vector, plus the common
                // drain guard. Internal Original/Aux/Feedback operands are
                // local registers, not new SR reads.
                const auto end = issue.cycle + instruction.repeat_count - 1;
                if (end > std::numeric_limits<std::uint32_t>::max())
                    throw std::invalid_argument("overlap VXM train exceeds finite cycle bound");
                all_issues_.push_back(end);
                if (end>issue.cycle) all_spans_.push_back({issue.cycle,end});
                const auto reserve = [&](std::size_t base, std::size_t width) {
                    mark(base,width);
                    for (std::size_t i=0;i<width;++i) {
                        stream_issues_[base+i].push_back(end);
                        if(end>issue.cycle) stream_spans_[base+i].push_back({issue.cycle,end});
                    }
                };
                // One compact C0..C7 queue drives both physical halves. The
                // mirrored C8..C15 operands use different fixed SR bundles;
                // they remain occupied even when their producer is passive
                // forwarding rather than another MEM issue at this cycle.
                constexpr auto mirroredOffset = VxmLane::kAluCount / 2;
                if (q.index >= mirroredOffset)
                    throw std::invalid_argument("overlap proof VXM control stage is invalid");
                for (const auto stage : {q.index, q.index + mirroredOffset}) {
                    for (const bool rhs : {false,true}) {
                        const auto kind = (rhs ? instruction.rhs : instruction.lhs).kind;
                        if (kind == VxmLaneOperandKind::StreamFloat16
                            || kind == VxmLaneOperandKind::StreamBFloat16
                            || kind == VxmLaneOperandKind::StreamFloat32)
                            reserve(VxmLane::fixed_input_base_for_stage(stage,rhs,kind),
                                VxmLane::stream_operand_bytes(kind));
                    }
                    if (instruction.output_stream)
                        reserve(stage == q.index ? *instruction.output_stream :
                            VxmLane::fixed_output_stream_for_block(VxmLane::block_for_stage(stage)),
                            instruction.output_type == VxmCastTarget::Float32 ? 4 :
                            instruction.output_type == VxmCastTarget::Int8 ? 1 : 2);
                }
                break;
            }
            case InstructionKind::Sxm: {
                if (c.extension_words.size() < 2)
                    throw std::invalid_argument("overlap proof SXM stream list missing");
                const auto n=static_cast<std::uint64_t>(c.extension_words[0])+c.extension_words[1];
                if (n > c.extension_words.size()-2)
                    throw std::invalid_argument("overlap proof SXM stream list truncated");
                for (std::size_t i=0;i<n;++i) packed(c.extension_words[2+i]);
                break;
            }
            default: throw std::invalid_argument("unproved computation resource in overlap image");
            }
        }
        all_issues_.normalize();
        for (auto& v:stream_issues_) v.normalize();
        for (auto& [q,v]:mem_issues_) v.normalize();
    }

    Selection select(std::uint32_t stop,std::uint64_t earliest,
                     std::span<const std::size_t> queues) const
    {
        for (const auto q:queues) if (const auto it=mem_issues_.find(q);it!=mem_issues_.end())
            earliest=std::max(earliest,release(it->second,stop,tile_rows_));
        std::vector<std::uint64_t> available(streams_);
        for (std::size_t s=0;s<streams_;++s)
            available[s]=std::max({earliest,release(stream_issues_[s],stop,guard_),
                crossing_release(stream_spans_[s],stop,guard_)});
        const auto e=*std::min_element(available.begin(),available.end());
        if (e>stop) throw std::invalid_argument("no proved MEM/SR overlap lease before safe stop");
        Selection selected{static_cast<std::uint32_t>(e),{}};
        // Earliest E, then most lanes, then fixed high-to-low stream order.
        for (std::size_t s=streams_;s-->0 && selected.streams.size()<lanes_;)
            if (available[s]<=e) selected.streams.push_back(static_cast<std::uint16_t>(s));
        return selected;
    }
    void validate(std::uint32_t launch,std::uint32_t stop,
                  std::span<const std::size_t> queues,
                  std::span<const std::uint16_t> streams) const
    {
        if (std::max(release(all_issues_,stop,guard_),crossing_release(all_spans_,stop,guard_))>stop)
            throw std::invalid_argument("unsafe overlap stop: computation has no full drain window");
        if (launch>stop || streams.empty() || streams.size()>lanes_)
            throw std::invalid_argument("invalid overlap lease domain");
        for (const auto q:queues) if (const auto it=mem_issues_.find(q);it!=mem_issues_.end())
            if (release(it->second,stop,tile_rows_)>launch)
                throw std::invalid_argument("overlap lease conflicts with computation MEM port");
        for (const auto s:streams)
            if (s>=streams_ || std::max(release(stream_issues_[s],stop,guard_),
                    crossing_release(stream_spans_[s],stop,guard_))>launch)
                throw std::invalid_argument("overlap lease conflicts with computation SR propagation");
    }
private:
    using Span=std::pair<std::uint64_t,std::uint64_t>;
    static std::uint64_t crossing_release(const std::vector<Span>& spans,
                                         std::uint32_t stop,std::uint64_t guard) {
        std::uint64_t result=0;
        for(const auto& [begin,end]:spans)
            if(begin<stop && end>=stop) result=std::max(result,end+guard);
        return result;
    }
    // A span covers consecutive issue cycles only. Repeated VXM configuration
    // tails keep their separate crossing-stop check.
    struct IssueCycles {
        std::vector<Span> runs;
        void add(std::uint64_t first, std::uint64_t last) {
            if (!runs.empty() && first >= runs.back().first && first <= runs.back().second + 1)
                runs.back().second = std::max(runs.back().second, last);
            else
                runs.emplace_back(first, last);
        }
        void push_back(std::uint64_t cycle) { add(cycle, cycle); }
        void normalize() {
            std::ranges::sort(runs);
            std::size_t size = 0;
            for (const auto run : runs) {
                if (size && run.first <= runs[size - 1].second + 1)
                    runs[size - 1].second = std::max(runs[size - 1].second, run.second);
                else
                    runs[size++] = run;
            }
            runs.resize(size);
        }
    };
    static std::uint64_t release(const IssueCycles& issues,
                                 std::uint32_t stop,std::uint64_t guard) {
        const auto next = std::ranges::lower_bound(issues.runs, stop, {}, &Span::first);
        if (next == issues.runs.begin()) return 0;
        return std::min<std::uint64_t>((next - 1)->second, stop - 1) + guard;
    }
    std::size_t streams_,lanes_;
    std::uint64_t tile_rows_,guard_{};
    IssueCycles all_issues_;
    std::vector<IssueCycles> stream_issues_;
    std::vector<Span> all_spans_;
    std::vector<std::vector<Span>> stream_spans_;
    std::map<std::size_t,IssueCycles> mem_issues_;
};
} // namespace ftlpu::software::runtime
