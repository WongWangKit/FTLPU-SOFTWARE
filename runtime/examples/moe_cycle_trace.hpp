#pragma once
#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

// Diagnostic observer only. It never acknowledges work or supplies operands.
// Every exported cycle is the enclosing system's physical clock, even when a
// device sub-controller is gated or expert queues use their local task clock.
class MoeCycleTrace {
  using Payload = std::map<std::string, std::string>;

public:
  MoeCycleTrace(const std::filesystem::path &path,
                const ftlpu::software::runtime::BinaryProgram &program)
      : file_(path), program_(program) {
    if (!file_)
      throw std::runtime_error("cannot open MoE instruction-cycle trace");
    file_ << "cycle,unit,resource,stage,opcode,pc,detail,payload\n";
  }
  void enable(ftlpu::C2cDmaSystem &system) {
    system.chip().routing().enable_cycle_trace();
    system.program_loader().dma_trace_source().enable_cycle_trace();
    for (unsigned side = 0; side < 2; ++side)
      system.dma(static_cast<ftlpu::Hemisphere>(side)).enable_cycle_trace();
    for (unsigned i = 0; i < ftlpu::hw::kMxmCount; ++i)
      system.chip().mxm_unit(i).control().enable_cycle_trace();
  }
  void sample(ftlpu::C2cDmaSystem &system) {
    using namespace ftlpu;
    namespace rt = ftlpu::software::runtime;
    cycle_ = system.cycle() - 1;
    auto &chip = system.chip();
    auto &icu = chip.icu();
    context_.clear();
    if (const auto &pending = chip.routing_descriptor_pending(); pending)
      write("RCU", "RCU.Descriptor", "fetch",
            routing_opcode_name(pending->opcode), "",
            "modeled descriptor-word fetch latency",
            {{"descriptor", s(pending->descriptor)},
             {"ready_cycle", s(chip.routing_descriptor_ready_cycle())}});
    if (!chip.expert_tasks().idle())
      write("RCU", "RCU.Dispatch", "wait", "ROUTE_DISPATCH", "",
            "device task/metadata/input/weight/expert handshake active", {});
    if (!chip.expert_macros().trace().empty()) {
      const auto &last = chip.expert_macros().trace().back();
      if (cycle_ >= last.start_cycle &&
          (!last.finish_cycle || cycle_ <= last.finish_cycle))
        context_ = {{"task", s(last.task.identity.task_id)},
                    {"expert", s(last.task.expert_id)},
                    {"valid_rows", s(last.task.valid_rows)}};
    }
    const auto &rcu = chip.routing().last_cycle_trace();
    if (rcu.valid && rcu.sequence != rcu_sequence_) {
      rcu_sequence_ = rcu.sequence;
      Payload p{{"token", s(rcu.token)},
                {"expert", s(rcu.expert)},
                {"slot", s(rcu.slot)},
                {"position", s(rcu.position)},
                {"assignment", s(rcu.assignment)},
                {"value", s(rcu.value)},
                {"score_bits", s(rcu.choice_bits)},
                {"raw_bits", s(rcu.raw_bits)},
                {"incoming_id", s(rcu.incoming_id)},
                {"candidate_valid", s(rcu.candidate_valid)},
                {"candidate_id", s(rcu.candidate_id)},
                {"candidate_bits", s(rcu.candidate_bits)},
                {"exchanged", s(rcu.exchanged)},
                {"winner_id", s(rcu.winner_id)},
                {"descriptor", s(rcu.descriptor)},
                {"completion", s(rcu.completion)}};
      const std::string opcode = std::string(rcu.step) == "MetadataRead"
                                     ? "META_READ"
                                     : routing_opcode_name(rcu.opcode);
      write("RCU", "RCU.Core", "internal", opcode, "", rcu.step, std::move(p));
    }
    const auto &combine = chip.routing_combine().last_cycle_trace();
    if (combine.valid)
      write("RCU", "RCU.Combine", "internal", "ROUTE_COMBINE", "", combine.step,
            {{"token", s(combine.token)},
             {"slot", s(combine.slot)},
             {"feature", s(combine.feature)},
             {"assignment", s(combine.assignment)},
             {"weight_bits", s(combine.weight_bits)}});
    // Slice control rows advance after execution. Row 1 is the actual compact
    // command accepted at tile 0 in this cycle, including commands from the
    // RCU command-port mux that never pass through a native ICU iMEM queue.
    if (combine.valid && std::string(combine.step) == "Launch")
      for (unsigned alu = 0; alu < 2; ++alu) {
        if (const auto &packet = chip.vxm_unit().instruction_at(alu, 1);
            packet) {
          const auto d = VxmCompactInstructionCodec::decode(alu, *packet);
          write("VXM", "VXM.C" + s(alu), "controller_issue",
                VxmLane::operation_name(d.instruction.operation), "",
                "RCU command-port configuration",
                {{"origin", "ROUTE_COMBINE"},
                 {"control", hex(packet->control)},
                 {"immediate_bits", hex(packet->immediate_bits)},
                 {"token", s(combine.token)},
                 {"slot", s(combine.slot)},
                 {"repeat", s(d.instruction.repeat_count)}});
        }
      }
    if (icu.program_issue_enabled() && !icu.program_loading())
      for (const auto &q : program_.queues) {
        const auto index =
            (q.kind == rt::QueueKind::MxmLoad ||
             q.kind == rt::QueueKind::MxmCompute ||
             q.kind == rt::QueueKind::MxmDequant)
                ? (q.index / program_.hardware.mxms_per_hemisphere) *
                          hw::kMxmsPerHemisphere +
                      q.index % program_.hardware.mxms_per_hemisphere
                : q.index;
        switch (q.kind) {
        case rt::QueueKind::Mem:
          issue_mem(icu.mem_iq(index), index);
          break;
        case rt::QueueKind::MxmLoad:
          issue_mxm(icu.mxm_load_iq(index), index, "Load");
          break;
        case rt::QueueKind::MxmCompute:
          issue_mxm(icu.mxm_compute_iq(index), index, "Compute");
          break;
        case rt::QueueKind::Vxm: {
          const auto &queue = icu.vxm_iq(index);
          if (issued(queue)) {
            const auto &packet = *queue.last_dispatched();
            const auto d = VxmCompactInstructionCodec::decode(index, packet);
            write("VXM", "VXM.C" + s(index), "issue",
                  VxmLane::operation_name(d.instruction.operation), pc(queue),
                  "ICU compact configuration",
                  {{"control", hex(packet.control)},
                   {"immediate_bits", hex(packet.immediate_bits)},
                   {"repeat", s(d.instruction.repeat_count)},
                   {"chain_depth", s(static_cast<unsigned>(d.chain_depth))}});
          }
          break;
        }
        case rt::QueueKind::Routing: {
          const auto &queue = icu.routing_iq();
          if (issued(queue)) {
            const auto i = *queue.last_dispatched();
            write("RCU", "RCU.Command", "issue", routing_opcode_name(i.opcode),
                  pc(queue), "ISA command",
                  {{"descriptor", s(i.descriptor)},
                   {"completion", s(i.completion)}});
          }
          break;
        }
        default:
          break;
        }
      }
    for (unsigned side = 0; side < 2; ++side) {
      auto &mem = chip.mem_array(static_cast<Hemisphere>(side));
      const auto prefix = std::string(side ? "W" : "E");
      for (const auto &tr : mem.executed_instructions())
        write(
            "MEM", "MEM." + prefix + ".S" + s(tr.mem_slice) + ".B" + s(tr.bank),
            "pipeline", mem_name(tr.instruction), "",
            "MEM functional-unit instruction",
            {{"tile", s(tr.tile)},
             {tr.instruction.opcode == MemOpcode::Gather ? "row_field_indirect"
                                                         : "row",
              s(tr.instruction.address)},
             {"stream", stream(tr.instruction.stream_id())},
             {"owner", hex(tr.instruction.routing_owner)},
             {"transfer_tag", hex(tr.instruction.routing_transfer)},
             {"native_word",
              hex(isa::encode_mem_instruction(tr.instruction))}});
      for (const auto &tr : mem.executed_transfers()) {
        const bool write_sram =
            tr.kind == MemArrayModel::MemTransfer::Kind::StoreStreamToSram;
        write("MEM",
              "MEM." + prefix + ".S" + s(tr.mem_slice) + ".B" + s(tr.bank),
              write_sram ? "write_commit" : "read_commit",
              write_sram ? "WRITE" : "READ", "", "actual SRAM transfer",
              {{"tile", s(tr.tile)},
               {"row", s(tr.address)},
               {"stream", stream(tr.stream)},
               {"sr_column", s(tr.sr_column)},
               {"vector_tag", hex(tr.vector_tag)},
               {"bytes", bytes(tr.bytes)}});
      }
      if (const auto &tr = mem.last_control_transfer(); tr)
        write("MEM", "MEM." + prefix + ".S" + s(tr->slice) + ".B" + s(tr->bank),
              tr->write ? "control_write" : "control_read",
              tr->write ? "CONTROL_WRITE" : "CONTROL_READ", "",
              "shared SRAM control port; no SR data transfer",
              {{"tile", s(tr->tile)},
               {"row", s(tr->row)},
               {"bytes", bytes(tr->bytes)}});
      dma(system.dma(static_cast<Hemisphere>(side)), "DMA." + prefix, side);
    }
    dma(system.program_loader().dma_trace_source(), "DMA.ProgramLoader", 2);
    for (unsigned index = 0; index < hw::kMxmCount; ++index) {
      const auto &mxm_unit = chip.mxm_unit(index);
      const auto &control = mxm_unit.control();
      const auto mxm_resource =
          "MXM." + std::string(index < hw::kMxmsPerHemisphere ? "E" : "W") +
          s(index % hw::kMxmsPerHemisphere);
      for (unsigned tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
        if (const auto &i = control.last_load_execution(tile); i)
          mxm(*i, index, "Load.T" + s(tile), "execute", "",
              {{"tile", s(tile)}});
        if (const auto &i = control.last_compute_execution(tile); i)
          mxm(*i, index, "Compute.T" + s(tile), "execute", "",
              {{"tile", s(tile)}});
        for (unsigned column = 0; column < hw::kMxmSupercellsPerPlane; ++column)
          if (mxm_unit.computing_cell(tile, column))
            write("MXM", mxm_resource + ".MAC", "execute", "MAC", "",
                  "actual computing supercell",
                  {{"tile", s(tile)}, {"column_block", s(column)}});
      }
      for (const auto &result : mxm_unit.last_outputs()) {
        std::string values;
        for (auto v : result.values) {
          if (!values.empty())
            values += ", ";
          values += f(v);
        }
        write("MXM", mxm_resource + ".Result", "result", "ACCUMULATOR_RESULT",
              "", "actual completed column output",
              {{"row", s(result.row)},
               {"column_block", s(result.column_block)},
               {"values", values}});
      }
    }
    for (unsigned tile = 0; tile < hw::kTileRows; ++tile) {
      const auto &lane = chip.vxm_unit().superlane(tile).lane(0);
      for (unsigned stage = 0; stage < 16; ++stage) {
        const auto &tr = lane.last_trace()[stage];
        const std::string resource = "VXM.T" + s(tile) + ".C" + s(stage);
        Payload p{{"tile", s(tile)},
                  {"lane", "0"},
                  {"scope", "representative lane; instruction is broadcast"}};
        if (tr.request_operation) {
          p["lhs"] = f(tr.lhs.value_or(0));
          p["rhs"] = f(tr.rhs.value_or(0));
          write("VXM", resource, "execute",
                VxmLane::operation_name(*tr.request_operation), "",
                "actual operand acceptance", p);
        }
        if (tr.result_operation) {
          p["result"] = f(tr.result.value_or(0));
          write("VXM", resource, "result",
                VxmLane::operation_name(*tr.result_operation), "",
                "actual ALU result", p);
        }
        if (tr.state == VxmLaneAluTraceState::Executed &&
            !tr.request_operation && !tr.result_operation)
          write("VXM", resource, "pipeline", "IN_FLIGHT", "",
                "ALU/LUT pipeline active", p);
        if (tr.state == VxmLaneAluTraceState::Stalled)
          write("VXM", resource, "wait", "INPUT_WAIT", "",
                "configuration waiting for token", p);
      }
    }
    const auto &stats = system.program_loader().statistics();
    if (stats.written_words != imem_words_) {
      write("ICU", "IMEM.Writer", "write_commit", "PROGRAM_WORDS", "",
            "actual instruction/descriptor memory commit",
            {{"words", s(stats.written_words - imem_words_)},
             {"total_words", s(stats.written_words)}});
      imem_words_ = stats.written_words;
    }
  }

private:
  template <class T> static std::string s(T value) {
    return std::to_string(value);
  }
  static std::string f(float v) {
    std::ostringstream out;
    out << std::setprecision(9) << v;
    return out.str();
  }
  static std::string hex(std::uint64_t v) {
    std::ostringstream out;
    out << "0x" << std::hex << v;
    return out.str();
  }
  template <class Range> static std::string bytes(const Range &v) {
    std::ostringstream out;
    for (auto x : v)
      out << std::hex << std::setw(2) << std::setfill('0') << unsigned(x);
    return out.str();
  }
  static std::string csv(const std::string &v) {
    std::string r = "\"";
    for (auto c : v) {
      if (c == '"')
        r += '"';
      r += c;
    }
    return r + '"';
  }
  static std::string json(const std::string &v) {
    std::string r = "\"";
    for (auto c : v) {
      if (c == '"' || c == '\\')
        r += '\\';
      if (c == '\n') {
        r += "\\n";
        continue;
      }
      r += c;
    }
    return r + '"';
  }
  static std::string stream(ftlpu::StreamId id) {
    return std::string(id.direction() == ftlpu::StreamDirection::East ? "E"
                                                                      : "W") +
           s(id.index());
  }
  static const char *mem_name(const ftlpu::MemInstruction &i) {
    using ftlpu::MemOpcode;
    switch (i.opcode) {
    case MemOpcode::Read:
      return "READ";
    case MemOpcode::Write:
      return "WRITE";
    case MemOpcode::Gather:
      return "GATHER";
    case MemOpcode::Scatter:
      return "SCATTER";
    }
    return "UNKNOWN";
  }
  template <class Queue> static bool issued(const Queue &q) {
    using A = ftlpu::IcuQueueAction;
    const auto a = q.last_trace().action;
    return q.last_dispatched().has_value() &&
           (a == A::FunctionalIssue || a == A::MacroIssue ||
            a == A::VxmRun2DIssue || a == A::RepeatIssue ||
            a == A::Repeat2DIssue);
  }
  template <class Queue> static std::string pc(const Queue &q) {
    return q.last_trace().issue_pc ? s(*q.last_trace().issue_pc) : "";
  }
  template <class Queue> void issue_mem(const Queue &q, unsigned index) {
    if (!issued(q))
      return;
    using namespace ftlpu;
    const auto local = index % InstructionControlUnit::kMemQueuesPerHemisphere;
    const auto &i = *q.last_dispatched();
    write(
        "MEM",
        "MEM." +
            std::string(index < InstructionControlUnit::kMemQueuesPerHemisphere
                            ? "E"
                            : "W") +
            ".S" + s(local / hw::kMemBanksPerSlice) + ".B" +
            s(local % hw::kMemBanksPerSlice),
        "issue", mem_name(i), pc(q), "ICU instruction",
        {{"row", s(i.address)},
         {"stream", stream(i.stream_id())},
         {"native_word", hex(isa::encode_mem_instruction(i))}});
  }
  template <class Queue>
  void issue_mxm(const Queue &q, unsigned index, const char *type) {
    if (issued(q))
      mxm(*q.last_dispatched(), index, type, "issue", pc(q), {});
  }
  void mxm(const ftlpu::MxmControlInstruction &i, unsigned index,
           const std::string &type, const char *stage, const std::string &pc,
           Payload p) {
    using namespace ftlpu;
    const char *opcode = i.opcode == MxmControlOpcode::IW        ? "IW"
                         : i.opcode == MxmControlOpcode::Compute ? "COMPUTE"
                         : i.opcode == MxmControlOpcode::AccumulatorRead
                             ? "ACC_READ"
                             : "DECODE";
    p["buffer"] = s(i.weight_buffer);
    p["accumulator"] = s(i.accumulator_address);
    p["weight_column"] = s(i.weight_column);
    p["stream"] = s(i.stream_base);
    p["activation_stream"] = s(i.activation_stream_base);
    p["native_word"] = hex(isa::encode_mxm_instruction(i));
    write("MXM",
          "MXM." + std::string(index < hw::kMxmsPerHemisphere ? "E" : "W") +
              s(index % hw::kMxmsPerHemisphere) + "." + type,
          stage, opcode, pc, "actual decoded MXM instruction", std::move(p));
  }
  void dma(const ftlpu::C2cDmaEngine &engine, const std::string &resource,
           unsigned index) {
    if (dma_sequence_[index] == engine.cycle())
      return;
    dma_sequence_[index] = engine.cycle();
    for (const auto &tr : engine.last_cycle_trace())
      write("DMA", resource, tr.stage,
            tr.instruction.direction == ftlpu::C2cDmaDirection::Ddr4ToC2c
                ? "DDR_TO_C2C"
                : "C2C_TO_DDR",
            "", "actual DMA instruction / request / completion",
            {{"ddr_address", hex(tr.address)},
             {"base", hex(tr.instruction.ddr4_address)},
             {"vector_count", s(tr.instruction.vector_count)},
             {"vector_index", s(tr.vector_index)},
             {"next_vector", s(tr.next_vector)},
             {"completed", s(tr.completed)},
             {"outstanding", s(tr.outstanding)},
             {"fifo", s(tr.fifo)},
             {"stream", s(tr.instruction.stream_index)},
             {"tag", hex(tr.tag)},
             {"sync_tag", s(tr.instruction.sync_tag)}});
    for (unsigned stream = 0; stream < ftlpu::hw::kC2cStreamsPerDirection;
         ++stream) {
      const auto inbound = engine.receive_queue_size(stream),
                 outbound = engine.outbound_queue_size(stream);
      if (inbound || outbound)
        write("DMA", resource, "buffered", "FIFO", "",
              "actual DMA FIFO occupancy; not an ALU operation",
              {{"stream", s(stream)},
               {"inbound", s(inbound)},
               {"outbound", s(outbound)}});
    }
  }
  void write(const std::string &unit, const std::string &resource,
             const std::string &stage, const std::string &opcode,
             const std::string &pc, const std::string &detail, Payload p) {
    for (const auto &[key, value] : context_)
      if (!p.contains(key))
        p[key] = value;
    std::string payload = "{";
    for (const auto &[key, value] : p) {
      if (payload.size() > 1)
        payload += ',';
      payload += json(key) + ':' + json(value);
    }
    payload += '}';
    file_ << cycle_ << ',' << unit << ',' << csv(resource) << ',' << stage
          << ',' << opcode << ',' << pc << ',' << csv(detail) << ','
          << csv(payload) << '\n';
  }
  std::ofstream file_;
  const ftlpu::software::runtime::BinaryProgram &program_;
  Payload context_;
  std::uint64_t cycle_{}, rcu_sequence_{};
  std::array<std::uint64_t, 3> dma_sequence_{};
  std::size_t imem_words_{};
};
