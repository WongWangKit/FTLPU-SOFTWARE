# Qwen KV Cache

## Scope

The first implementation keeps a BF16 KV cache in LPU MEM for a compiled
decoder-layer invocation. It establishes the persistent-state ABI needed by
decode while preserving the existing sequence-32 prefill schedule. It does not
yet implement token-by-token append, attention over past tokens, DDR spill, or
KV quantization.

The cache capacity is a target-independent compiler option:

```powershell
ftlpu-opt input.mlir --target-config ftlpu-lpu32.json `
  --kv-cache-capacity 256 -o tensor.mlir
```

Capacity must be a multiple of the MXM tile width and no smaller than the
compiled sequence length. A value of zero keeps the previous transient
attention buffers.

## ABI

Each decoder executable exposes two internal bindings with stable indices:

| Binding | Index | Role | Type and logical shape |
| --- | ---: | --- | --- |
| K cache | 65536 | `state.kv.key` | `bf16[capacity, kv_heads, head_dim]` |
| V cache | 65537 | `state.kv.value` | `bf16[capacity, kv_heads, head_dim]` |

Schedule MEM transfers carry both `address_binding` and
`address_binding_access = "internal"`. Schedule-to-Command lowering preserves
that pair, and binary relocations therefore resolve against `ModelState`
instead of model inputs. `ModelPackage` records one named K/V state pair per
layer and each `ModelInvocation` references those state names.

## Physical Layout

K uses `fp16_head_planar`. The global KV-head stride is retained because K is
consumed by query-head groups in both hemispheres. The runtime therefore
replicates K into every enabled hemisphere. V uses `fp16_value_x16`; each KV
head is stored only in its owning hemisphere and is already laid out for PV.

For Qwen2.5-1.5B with capacity 256, two KV heads, head dimension 128, two
hemispheres, and 32-byte SRAM vectors:

| Quantity | K | V |
| --- | ---: | ---: |
| Logical bytes per layer | 128 KiB | 128 KiB |
| Rows per selected slice | 1024 | 256 |
| Selected slices | 4 | 16 |
| Data physically written | 256 KiB | 128 KiB |
| Space conservatively reserved | 256 KiB | 256 KiB |

The V reservation is larger than the written data because one binding keeps a
global-head address range in both hemispheres. Splitting it into segmented
per-hemisphere bindings is a future allocator optimization.

The current fixed-slice placement is deliberately a single-layer milestone,
not a claim that the entire model cache fits. With 8,192 rows per SRAM bank,
the K placement consumes 1,024 rows per layer on the same four slices and can
hold at most eight layers at capacity 256. V consumes 256 rows per layer and
can hold 32 layers. A 28-layer Qwen deployment therefore needs either
layer-aware spreading of K across more activation slices or DDR-paged KV; the
present global lifetime planner correctly rejects an overcommitted layout.

## Runtime Lifecycle

`ModelSession::load` allocates each state for the session lifetime and zeros it
through C2C. A decoder invocation relocates its internal K/V MEM commands to
that allocation. `read_state(name)` downloads and converts the physical layout
back to logical `[token, head, dimension]` order through C2C.
`reset_states()` zeros all persistent states through the same path. No host API
directly writes MEM.

For a multi-layer model, state names are `layers.N.key_cache` and
`layers.N.value_cache`. They are distinct from ping-pong weight pages and are
not evicted at invocation boundaries when their physical placement fits.

## Verified Path

The sequence-32 Qwen2.5-1.5B layer-0 test compiles with capacity 256, builds a
logical ModelPackage, converts it to a v5 paged package, reloads it from disk,
and runs the CModel. The test compares all 8,192 produced K values and all
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
length), append-only K/V write ranges, QK/PV bounds over the valid prefix, and
causal-mask handling for `past_len + current_len`. After that is numerically
closed, an off-chip page table and optional per-head/per-page KV quantization
can be added without changing the logical state ABI.
