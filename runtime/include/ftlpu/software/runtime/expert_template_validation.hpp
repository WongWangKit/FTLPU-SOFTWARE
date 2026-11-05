#pragma once
#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/system/icu.hpp"
#include <memory>

namespace ftlpu::software::runtime {
// Compile/load-time validation only. No route scores or run-time expert IDs
// are inspected here. Exercise the actual finite frontend at the row bounds.
inline void validate_expert_template_program(const BinaryProgram &p) {
  for (const auto &d : p.routing_descriptors)
    if (d.opcode == RoutingOpcode::ExpertTemplate) {
      const auto &t = d.expert_template;
      const RouteAddressDescriptor *address = nullptr;
      for (const auto &x : p.routing_descriptors)
        if (x.opcode == RoutingOpcode::Dispatch)
          address = &p.routing_descriptors
                         .at(x.dispatch.address_template_descriptor_id)
                         .address;
      if (!address)
        throw std::invalid_argument("expert template has no dispatch address");
      for (const auto &target : t.queues) {
        const QueueProgram *program = nullptr;
        for (const auto &q : p.queues) {
          auto index = std::size_t(q.index);
          if (q.kind == QueueKind::MxmLoad || q.kind == QueueKind::MxmCompute ||
              q.kind == QueueKind::MxmDequant)
            index = (index / p.hardware.mxms_per_hemisphere) *
                        hw::kMxmsPerHemisphere +
                    index % p.hardware.mxms_per_hemisphere;
          if (static_cast<unsigned>(q.kind) == target.kind &&
              index == target.index)
            program = &q;
        }
        if (!program)
          throw std::invalid_argument("expert template queue image missing");
        for (const auto &partition : p.queue_sync_plan.partitions)
          if (partition.kind == program->kind &&
              partition.index == program->index && !partition.entries.empty())
            throw std::invalid_argument(
                "expert-owned queues cannot carry global QCTL controls");
        for (const auto &q : p.transport_queues)
          if (target.kind == 0 && q.index == target.index)
            throw std::invalid_argument(
                "expert template cannot share a MEM transport queue");
        const auto rows_max = std::min<std::uint32_t>(32, address->max_rows);
        for (const auto rows : {1u, rows_max}) {
          MacroTaskParameters params;
          params.generation = 1;
          params.task_id = 1;
          params.values[0] = rows;
          params.values[2] = target.weight_base;
          params.values[4] = t.workspace_base;
          params.values[5] = t.scale_bf16;
          const auto assignment =
              t.output.rows >= rows ? t.output.rows - rows : 0;
          params.values[6] = assignment;
          if (target.kind == 0) {
            const auto local =
                target.index % InstructionControlUnit::kMemQueuesPerHemisphere;
            const auto h = target.index /
                           InstructionControlUnit::kMemQueuesPerHemisphere,
                       slice = local / hw::kMemBanksPerSlice,
                       bank = local % hw::kMemBanksPerSlice;
            for (std::size_t i = 0; i < address->destination.bytes(); ++i) {
              const auto plane = address->destination.planes[i];
              if (plane.hemisphere == h && plane.slice == slice &&
                  plane.bank == bank)
                params.values[1] = plane.base_row;
            }
            for (std::size_t i = 0; i < t.output.bytes(); ++i) {
              const auto plane = t.output.planes[i];
              if (plane.hemisphere == h && plane.slice == slice &&
                  plane.bank == bank)
                params.values[3] =
                    plane.base_row + assignment * t.output.token_stride;
            }
          }
          auto icu = std::make_unique<InstructionControlUnit>();
          load_queue_programs_into_icu(
              {*program}, *icu, p.hardware.mxms_per_hemisphere, nullptr, false);
          icu->visit_program_queue(
              static_cast<IcuMacroQueueKind>(target.kind), target.index, false,
              [&](auto &q) {
              const auto records=q.raw_macro_mode()?((q.read_raw_imem(0).lanes[0]>>4)&0x00ffffffu):q.imem_occupancy();
              for(const auto &b:target.bindings)if(b.record>=records)throw std::invalid_argument("expert binding record out of range");
              q.reserve_task_queue();
                q.begin_task(params, target.bindings);
                for (std::size_t n = 0; !q.task_ready(); ++n) {
                  if (n > 20000)
                    throw StaticScheduleError(
                        "expert frontend priming bound exceeded");
                  q.prepare_task_cycle();
                }
                q.launch_task();
                for (std::size_t n = 0; !q.done(); ++n) {
                  if (n > 1000000)
                    throw StaticScheduleError(
                        "expert template issue bound exceeded");
                  (void)q.dispatch_next();
                }
                q.finish_task();
              });
        }
      }
    }
}
} // namespace ftlpu::software::runtime
