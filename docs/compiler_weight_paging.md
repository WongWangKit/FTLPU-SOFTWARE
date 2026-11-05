# Compiler-owned weight paging

Connected logical graphs use the same compiler policy for weight capacity,
residency and overlap. A test supplies tensor equations and immutable weight
bytes; its name, model family and selected bring-up endpoint do not choose the
page schedule. A standalone operator or a dense composite still supplies its
native weight layout through an adapter. Sharing the planner does not mean
rewriting every operator into one physical layout.

## Compiler contract

`compiler/Target/graph_weight_paging.hpp` resolves the connected-graph policy
before native lowering. `compiler/Target/weight_page_plan.hpp` provides shared
capacity, integral-item splitting, storage overlap and lifetime checks.

| Policy | Meaning |
| --- | --- |
| Omitted or `ftlpu.weight_paging_policy = "auto"` | Reserve rotating weight slots when the target supports them; prove transfer overlap later. |
| `ftlpu.weight_paging_policy = "serialized"` | Keep dynamic DDR pages and completion synchronization, without graph weight-slot overlap. |
| `ftlpu.weight_page_rows` | Optional per-slice slot capacity, no larger than half of the configured SRAM bank. |
| `ftlpu.weight_page_ping_pong` | Compatibility alias; conflicting old/new policies are rejected. |
| `ftlpu.common_transport_overlap = false` | Explicitly opt out of overlapping transport. |

Automatic placement requires at least two SRAM banks, a valid target-defined
weight slice group and the common transport-overlap capability. Its default
slot occupies the upper half of each weight bank. The lower half remains the
native scratch arena. An explicit smaller page capacity moves the slot to the
end of its SRAM bank. Page dimensions follow tensor shape and weight layout;
they are not a table of Qwen projection names.

The graph must also contain W8 projection weight demand. Graphs containing
only vector operations, normalization or recurrence do not reserve unused
weight slots or move their scratch merely because the requested policy is
automatic. Synthetic tensor-view nodes have no native function to traverse.

The graph linker reserves weight slots before retaining activations and
intermediate results. It records each weight page's consumer and earliest
storage reuse position, and prevents overlap with the consumer's declared
native workspace. Placement changes precede native FU scheduling, so physical
MEM-to-consumer distances are recalculated. Native operators request scratch
queues outside reserved page queues where concurrent use is needed; the
unit-lower solve's coefficient buffers use this resource policy.

Dense Attention and FFN adapters retain their native page traversal and
activation/workspace ABI. Their page geometry and lifetime calculations use
the shared planning primitives. Already-paged composite operands are not
partially relocated by the generic graph linker. Such relocation would also
have to rewrite every internal consumer and workspace reservation.

`graph_plan.json` exposes the resolved policy, weight-slot row range and slices,
and `weight_page_placements` with bank, rows, consumer and release information.
Recompile a test to obtain the new plan; an old binary or CSV is not updated by
changing the compiler.

`runtime_icu_programs/program.json` separately records the final linked
`transport_overlap_protocol` and `overlap_analysis_fallback`. A compiler plan
with rotating SRAM slots may execute serialized transfers when the bounded
runtime proof falls back; the executed image, not the requested policy, is
authoritative for this distinction.

## Runtime contract

The compiler supplies page bindings, physical addresses, consumer/release
positions and a drained ingress candidate before each consumer. The runtime
packs those bindings into real DDR/C2C descriptors and links synchronized
transport commands into the real ICU image. It does not perform model lowering
or choose a model-specific page order.

For an overlapping page the linker proves a lease over the final raw-macro
computation image:

1. The destination `(hemisphere, slice, bank)` MEM ICU queues must be free for
   the transfer. Disjoint row addresses alone do not make one ICU queue
   concurrent; a READ3D cannot be interrupted by WRITE_SYNC.
2. Ordinary SR routes must be free through their full propagation lifetime.
   VXM checks cover the two physical halves driven by each C0--C7 control
   queue, FP16/BF16/FP32 byte widths, outputs and repeated vector trains.
   Internal feedback/register operands do not create another SR read.
3. The consumer stop must drain computation. A late transfer holds that stop
   until actual RX synchronization, MEM writes and the transport tail complete.
   A predicted DDR duration is never used as proof of data readiness.
4. The finite DMA/RX queues retain consumer order. A later page with an earlier
   apparent launch opportunity must not block an earlier consumer's page.

If the finite static issue-analysis budget is exhausted, the linker keeps the
compiler's drained common stops and serializes page loads there. The timeline
records `c2c.serial_fallback.proof_work_bound`. This is a generic analysis
limitation, not a tokenwise/GDN exception. Invalid addresses, malformed raw
instructions, invalid resource geometry and unsafe drain windows remain
errors; they are not silently treated as permission to overlap or serialize.
The linked binary records the resolved serial protocol and zero early-launch
fields, so an independent loader does not retry the exhausted optional proof.
The pager's analysis budget is bounded (8,000,000 issues by default); a small
budget regression exercises serialization, reload and actual execution.

Within-program page overlap is distinct from invocation-level nonpaged weight
uploads, cross-invocation lookahead, persistent State transfers and ICU iMEM
reloads. Enabling this policy does not promise all of those forms of overlap.

## Validation

`compiler/tests/logical_weight_paging_test.py` compiles model-independent
projection graphs. It checks automatic and serialized policies, compatibility
flags, different SRAM depths and page sizes, alternating banks, placement
bounds and retained-value conflicts. Its 14 cases also cover no-weight vector
graphs and resident Conv+SiLU graphs containing synthetic layout views.

`runtime_common_barrier_test` checks real completion and late/backpressured
transports, phase-preserving common stops, fixed VXM stream widths, mirrored
physical ALU ports and active vector trains. The real-weight chunkwise and
tokenwise attention tests provide end-to-end numerical and raw-macro execution
coverage; their current `summary.json` and executed ICU exports are the source
of measured results, not historical documentation numbers.

The tokenwise full-attention regression uses:

```powershell
python compiler/tests/qwen3_5_prefill_tokenwise_test.py --tokens 32 --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_tokenwise_gdn
```

It supplies nonzero initial State, uses raw macro instructions and wide SXM,
and compares both final State and attention outputs. Python performs input
packing and independent reference calculations, not physical page scheduling.

The decode batch-1 smoke also covers three separate invocations: pageable
input projections, a resident Conv/recurrence mixer, and the output projection.
The mixer does not reserve graph weight slots. This is a test of the shared
policy across invocation shapes, not a claim of overlap across those three
invocations.
