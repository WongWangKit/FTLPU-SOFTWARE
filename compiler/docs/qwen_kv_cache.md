# Qwen KV Cache

## Scope

The implementation gives each decoder layer a logical BF16 KV cache backed by
off-chip session storage while keeping only the executable's current window in
LPU MEM. It preserves the sequence-32 prefill schedule and establishes the
persistent-state ABI needed by decode. Token-by-token append, attention over a
past prefix, moving page offsets, and KV quantization are not implemented yet.

The cache capacity is a target-independent compiler option:

```powershell
ftlpu-opt input.mlir --target-config ftlpu-lpu32.json `
  --kv-cache-capacity 256 -o tensor.mlir
```

Capacity is the logical maximum and must be no smaller than the compiled
sequence length. The physical resident window is the sequence length rounded
up to `mxm_rows`; `page_tokens` is derived from the target topology. A value of
zero keeps the previous transient attention buffers.

## ABI

Each decoder executable exposes two internal bindings with stable indices:

| Binding | Index | Role | Executable SRAM shape |
| --- | ---: | --- | --- |
| K cache | 65536 | `state.kv.key` | `bf16[resident_tokens, kv_heads, head_dim]` |
| V cache | 65537 | `state.kv.value` | `bf16[resident_tokens, kv_heads, head_dim]` |

Schedule MEM transfers carry both `address_binding` and
`address_binding_access = "internal"`. Schedule-to-Command lowering preserves
that pair, and binary relocations therefore resolve against `ModelState`
instead of model inputs. `ModelPackage` v6 records each state's logical
`[capacity, kv_heads, head_dim]` shape plus `page_tokens` and
`resident_tokens`; each `ModelInvocation` references its layer's state names.

## Physical Layout

K uses `fp16_head_planar`. The global KV-head stride is retained because K is
consumed by query-head groups in both hemispheres. The runtime therefore
replicates K into every enabled hemisphere. V uses `fp16_value_x16`; each KV
head is stored only in its owning hemisphere and is already laid out for PV.
Both persistent windows are placed at the high end of their selected slice
groups and remain reserved until executable exit. This prevents a later FFN
stage from overwriting KV data before runtime pages it back out.

For sequence length 32 and Qwen2.5-1.5B capacity 256, two KV heads, head
dimension 128, two hemispheres, and 32-byte SRAM vectors:

| Quantity | K | V |
| --- | ---: | ---: |
| Logical bytes per layer | 128 KiB | 128 KiB |
| Logical bytes in the resident window | 16 KiB | 16 KiB |
| Rows per selected slice | 128 | 32 |
| Selected slices | 4 | 16 |
| Data transferred through C2C | 32 KiB | 16 KiB |
| SRAM interval reserved | 32 KiB | 32 KiB |

The V reservation is larger than the written data because one binding keeps a
global-head address range in both hemispheres. Splitting it into segmented
per-hemisphere bindings is a future allocator optimization.

All 28 Qwen layers have distinct logical states, totaling 7 MiB at capacity
256. Sequential invocations reuse the compiler-declared K/V staging addresses.
The bank0/bank1 executable variants therefore need at most one K/V pair per
ping-pong bank, rather than 28 pairs. The planner reserves separate slots when
target ABI, layout, shape, slices, bank, or binding index differs.

## Runtime Lifecycle

`ModelSession::load` allocates and zeros each full logical state in off-chip
session backing. Before a decoder invocation, runtime transfers that layer's
resident K/V window through DDR and C2C into the shared SRAM slots. After the
invocation it transfers both windows back and updates the logical backing.
`read_state(name)` returns the logical `[token, head, dimension]` image, and
`reset_states()` clears all logical states. No host API directly writes MEM.

For a multi-layer model, state names are `layers.N.key_cache` and
`layers.N.value_cache`. They are distinct from ping-pong weight pages. Runtime
statistics and the pipeline CSV expose `state_page_in/out` and
`C2C.StatePageIn/Out` separately from weight traffic.

## Verified Path

The sequence-32 Qwen2.5-1.5B layer-0 test compiles with capacity 256, builds a
logical ModelPackage, serializes a v6 package, and runs the CModel. The test
compares all 8,192 produced K values and all
8,192 V values with the Hugging Face fixture, checks that the unused capacity
remains zero, validates the 49,152-value decoder output, and verifies state
reset.

Current measured errors are:

| Output | Tolerance violations | Maximum absolute error |
| --- | ---: | ---: |
| K cache | 0 | 0.00390625 |
| V cache | 0 | 0 |
| Decoder output | 0 | 0.015625 |

## Next Decode Work

The next compiler/runtime contract needs a dynamic token position (or cache
length), a resident-window page offset, append-only K/V write ranges, QK/PV
bounds over the valid prefix, and causal-mask handling for
`past_len + current_len`. The logical state and page-size ABI already leaves
room for that moving window; optional per-head/per-page KV quantization can be
added after decode is numerically closed.
