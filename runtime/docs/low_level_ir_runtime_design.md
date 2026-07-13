# FTLPU Low-Level IR, Binary, and Runtime Design

This document defines the first software-side contract for lowering neural
network programs to the current FTLPU CModel.

The key CModel boundary is the ICU. `InstructionControlUnit` already exposes
static per-queue dispatch for MEM, MXM, and VXM, and
`instruction_codec.hpp` already defines compact instruction encodings. The
software stack should therefore target an explicit ICU timeline first, then
grow toward a fuller MLIR backend and hardware runtime.

## Target Stack

Recommended lowering path:

```text
framework graph
  -> ftlpu.tensor
  -> ftlpu.stream
  -> ftlpu.icu
  -> .ftlpu binary
  -> runtime
  -> CModel ICU
```

`ftlpu.tensor` keeps model-level semantics such as matmul, linear, SwiGLU, and
quantization parameters.

`ftlpu.stream` assigns tensors to MEM columns, SRAM addresses, byte streams,
MXM units, MXM weight buffers, and VXM ALU stages.

`ftlpu.icu` is the low-level IR. It contains explicit issue cycles, explicit
queues, and explicit MEM/MXM/VXM instructions. This is the main target for the
first compiler backend.

## ICU Queue Model

The low-level IR should use the same queue shape as the CModel ICU:

| Queue kind | Count | CModel API |
| --- | ---: | --- |
| `mem` | 44 | `enqueue_mem`, `enqueue_mem_nop`, `enqueue_mem_repeat` |
| `mxm_load` | 2 | `enqueue_mxm_load_nop` plus `IW` |
| `mxm_compute` | 2 | `enqueue_mxm_compute_nop` plus `Compute` |
| `mxm_output` | 2 | `enqueue_mxm_output_nop` plus `Output` |
| `vxm` | 16 | `enqueue_vxm`, `enqueue_vxm_nop`, `enqueue_vxm_repeat` |

The textual shape can look like this:

```mlir
ftlpu.icu.program @ffn attributes {
  arch = "ftlpu.cmodel.v1",
  tile_rows = 20,
  mem_columns = 44,
  lanes = 16,
  streams = 64
} {
  ftlpu.icu.mem_read     @mem32 cycle 27  addr 0  stream 0
  ftlpu.icu.mxm_iw       @mxm0  cycle 38  col 0   buffer 0
  ftlpu.icu.mxm_compute  @mxm0  cycle 78  buffer 0
  ftlpu.icu.mxm_output   @mxm0  cycle 116 stream_base 32
  ftlpu.icu.vxm_alu      @alu0  cycle 125 cast stream_i32 32 -> fp32
  ftlpu.icu.vxm_alu      @alu15 cycle 140 cast alu14 -> i8 out_stream 40
}
```

The compiler lowers this IR by bucketing operations per queue, sorting by
cycle, inserting `NOP N` for idle gaps, and optionally emitting `Repeat n,d`
for regular sequences. This is the reusable form of the CModel integration
test's current `OfflineIcuProgram`.

## Instruction Set

The first version should reuse the CModel codec directly:

- MEM: one 32-bit word for `Read`, `Write`, `Gather`, and `Scatter`.
- MXM: one 32-bit word for `IW`, `Compute`, and `Output`.
- VXM: three 32-bit words for ALU opcode, operands, immediates, cast target,
  and optional output stream.
- ICU command: one 32-bit word for `Instruction`, `NOP`, and `Repeat`.

VXM quantization metadata should not be hidden in the binary. Quant/dequant
must be lowered to explicit ALU instructions:

```text
cast_i32_to_f32 -> multiply(scale) -> multiply(1 / output_scale)
-> add(zero_point) -> cast_i8
```

## Binary Container

The `.ftlpu` binary should be a small little-endian container:

```text
FtlpuBinaryHeader
FtlpuSectionHeader[]
.arch
.queues
.data
.symbols
.debug
```

The `.queues` section is required in milestone 1. It stores queue headers and
queue commands:

```c
enum FtlpuQueueKind : uint16_t {
  FTLPU_Q_MEM = 0,
  FTLPU_Q_MXM_LOAD = 1,
  FTLPU_Q_MXM_COMPUTE = 2,
  FTLPU_Q_MXM_OUTPUT = 3,
  FTLPU_Q_VXM = 4,
};

struct FtlpuQueueCommand {
  uint32_t icu_command;
  uint16_t instruction_kind;
  uint16_t word_count;
  uint32_t words[3];
};
```

For instruction commands, `icu_command` uses opcode `Instruction`, and
`words` stores the encoded MEM/MXM/VXM instruction. For `NOP` and `Repeat`,
`word_count` is zero.

The `.data` section should later describe initial SRAM payloads and output
objects. The first runtime can support preloading constants and inputs, then
dumping marked output ranges after execution.

## Runtime Contract

The first runtime path is intentionally narrow:

```text
.ftlpu queues -> IcuProgram -> InstructionControlUnit -> TspSliceSystem::tick()
```

The software repository now contains a first `IcuProgram` implementation under
`runtime/include/ftlpu/software/runtime/icu_program.hpp`. It can:

- record scheduled MEM/MXM/VXM events;
- validate per-queue cycle conflicts;
- lower scheduled events into NOP plus instruction commands;
- load those commands into the CModel ICU.

The current prototype also has the first full software path:

```text
runtime/examples/simple_dispatch.lpuir
  -> parse_lpu_ir()
  -> IcuProgram::encode_queues()
  -> write_binary_program(.ftlpu)
  -> CModelRuntime::load_file()
  -> InstructionControlUnit queues
  -> dispatch_icu_cycles()
```

This is intentionally narrow: it proves that handwritten LPU IR can be compiled
to a binary queue file and loaded by runtime before the CModel clock starts.

The runtime should not bypass the ICU to directly call MEM/MXM/VXM internals.
That keeps the software contract aligned with the future hardware control path.

## Current CModel Boundary

The current CModel still has one important integration boundary:
`MxmGemmEngine` is not fully owned by `TspSliceSystem::tick()`. The ICU already
dispatches MXM load, compute, and output control pulses, but the numeric GEMM
engine is still bridged by the integration test.

Recommended order:

1. Keep this software runtime focused on ICU queue loading first.
2. Move or wrap the MXM GEMM bridge so runtime can execute the full FFN path.
3. Add `.ftlpu` reader/writer.
4. Add an MLIR backend that emits `ftlpu.icu`.

## First Milestones

Milestone A: shared ICU program container.

- Done as the initial `IcuProgram` skeleton in this repository.
- Next step is to replace the test-local `OfflineIcuProgram` with this shared
  implementation.

Milestone B: queue binary reader/writer.

- Serialize `IcuProgram::QueueCommand` into `.queues`.
- Round-trip through `instruction_codec.hpp`.

Milestone C: `ftlpu-run`.

- Load `.ftlpu`.
- Preload SRAM data.
- Load ICU queues.
- Tick `TspSliceSystem`.
- Dump logs and outputs.

Milestone D: MLIR backend.

- Emit textual `ftlpu.icu` first.
- Add scheduling and hazard diagnostics.
- Emit `.ftlpu` binary once textual schedules are stable.
