#pragma once

#include "ftlpu/software/runtime/issue_inspector.hpp"
#include "ftlpu/core/instruction_codec.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
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
    explicit CommonOverlapCalendar(const BinaryProgram& program)
        : streams_(program.hardware.streams_per_direction),
          lanes_(program.hardware.c2c_streams_per_direction),
          tile_rows_(program.hardware.tile_rows),
          stream_issues_(streams_)
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
        std::erase_if(computation.queues, [](const auto& q) {
            return q.kind == QueueKind::C2cDma || q.kind == QueueKind::C2cRx
                || q.kind == QueueKind::C2cTx;
        });
        const auto timeline = inspect_logical_issues(computation,8'000'000);
        std::size_t count = 0;
        for (const auto& q : timeline.queues) for (const auto& issue : q.issues) {
            if (++count > 8'000'000 || issue.cycle > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("overlap proof exceeds finite issue/cycle bound");
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
                    || (m.opcode == MxmControlOpcode::Decode
                        && m.decode_operation == MxmDecodeOperation::StreamCompute
                        && m.accumulator_destination == MxmAccumulatorDestination::Stream))
                    mark(m.stream_base % streams_, h.mxm_result_streams);
                break;
            }
            case InstructionKind::MxmDequant: break; // local weight registers
            case InstructionKind::Vxm:
                // The chain VXM target replaced explicit SR operands with
                // captured input groups and internal feedback. The legacy
                // v38 lease proof cannot infer their lifetime from issue time.
                // QCTL uses the Native phase plan and does not use this proof.
                throw std::invalid_argument(
                    "legacy transport overlap proof does not support chain VXM; use queue control");
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
        normalize(all_issues_);
        for (auto& v:stream_issues_) normalize(v);
        for (auto& [q,v]:mem_issues_) normalize(v);
    }

    Selection select(std::uint32_t stop,std::uint64_t earliest,
                     std::span<const std::size_t> queues) const
    {
        for (const auto q:queues) if (const auto it=mem_issues_.find(q);it!=mem_issues_.end())
            earliest=std::max(earliest,release(it->second,stop,tile_rows_));
        std::vector<std::uint64_t> available(streams_);
        for (std::size_t s=0;s<streams_;++s)
            available[s]=std::max(earliest,release(stream_issues_[s],stop,guard_));
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
        if (release(all_issues_,stop,guard_)>stop)
            throw std::invalid_argument("unsafe overlap stop: computation has no full drain window");
        if (launch>stop || streams.empty() || streams.size()>lanes_)
            throw std::invalid_argument("invalid overlap lease domain");
        for (const auto q:queues) if (const auto it=mem_issues_.find(q);it!=mem_issues_.end())
            if (release(it->second,stop,tile_rows_)>launch)
                throw std::invalid_argument("overlap lease conflicts with computation MEM port");
        for (const auto s:streams)
            if (s>=streams_ || release(stream_issues_[s],stop,guard_)>launch)
                throw std::invalid_argument("overlap lease conflicts with computation SR propagation");
    }
private:
    static void normalize(std::vector<std::uint64_t>& v) {
        std::ranges::sort(v); v.erase(std::ranges::unique(v).begin(),v.end());
    }
    static std::uint64_t release(const std::vector<std::uint64_t>& v,
                                 std::uint32_t stop,std::uint64_t guard) {
        const auto next=std::lower_bound(v.begin(),v.end(),stop);
        return next==v.begin()?0:*(next-1)+guard;
    }
    std::size_t streams_,lanes_;
    std::uint64_t tile_rows_,guard_{};
    std::vector<std::uint64_t> all_issues_;
    std::vector<std::vector<std::uint64_t>> stream_issues_;
    std::map<std::size_t,std::vector<std::uint64_t>> mem_issues_;
};
} // namespace ftlpu::software::runtime
