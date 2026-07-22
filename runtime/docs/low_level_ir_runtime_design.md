# FTLPU Command Binary and Runtime

[English](low_level_ir_runtime_design.md) | [简体中文](low_level_ir_runtime_design.zh-CN.md)

This document defines the current compiler/runtime contract at the CModel ICU
boundary.

## Lowering Boundary

```text
StableHLO
  -> ftlpu.kernel
  -> ftlpu.tensor
  -> ftlpu.stream
  -> ftlpu.schedule
  -> ftlpu.command
  -> .ftlpu binary
  -> CModelRuntime
  -> CModel ICU and TspSliceSystem::tick()
```

Schedule IR owns resource reservation and exact absolute cycles. Command IR is
the serializable boundary: it contains runtime bindings and flat MEM/MXM/VXM queue
commands. `ftlpu-translate` groups commands by queue, sorts them by cycle, and
encodes idle gaps and regular sequences as ICU NOP and Repeat commands.

## Runtime Bindings

Each `ftlpu.command.binding` and `BinaryBinding` records:

- input or output index;
- element type and logical shape;
- byte size;
- physical layout and MEM slice list;
- base SRAM row, instruction count, and signed address stride.

The implemented layouts are:

| Layout | Use | Placement |
| --- | --- | --- |
| `Vector` | int8 activation | one MEM slice, row-major 320-byte vectors |
| `MxmWeightStriped` | int8 MXM weight | columns striped over 16 MEM slices |
| `Int32BytePlanar` | int32 result | four MEM slices, one byte plane per slice |

`CModelRuntime::upload_input()` validates the binding and stages bytes into the
declared SRAM placement. `download_output()` reconstructs the logical tensor
from its physical byte planes.

## Binary Version 2

All scalar fields are little-endian. The file starts with:

```text
magic[8] = "FTLPUB01"
u32 version = 2
u64 max_cycle
u32 queue_count
u32 binding_count
BinaryBinding[binding_count]
QueueProgram[queue_count]
```

A binding contains fixed metadata followed by `rank` 64-bit dimensions and
`slice_count` 16-bit MEM slice identifiers. A queue record contains queue kind,
queue index, command count, and encoded `QueueCommand` records. Each command
stores the ICU command word, instruction kind, word count, and three 32-bit
instruction words. The reader retains version 1 compatibility; version 1 has
no binding count or binding records.

The instruction words reuse the CModel codec directly. MEM and MXM use one
word. VXM uses the full three-word container to encode its ALU opcode, two
typed operands, cast target, output stream, scale, and zero point. ICU queue
NOP and Repeat commands retain the absolute schedule produced by the compiler.

## Execution Contract

The runtime performs these steps:

1. Read and validate `.ftlpu` metadata and queues.
2. Upload all required input bindings into CModel SRAM.
3. Fill the corresponding ICU queues, including NOP and Repeat commands.
4. Advance the authoritative `TspSliceSystem::tick()` loop through the program
   and drain cycles.
5. Download output bindings from SRAM.

The runtime does not call MEM or MXM datapaths directly. This preserves the
same queue and timing contract that a future hardware runtime will use.

## 320x320 GEMM Regression

The end-to-end regression compiles StableHLO matmul to Command IR and binary,
uploads two 320x320 int8 matrices, executes the CModel, downloads a 320x320
int32 result, and compares all 102,400 elements with a CPU reference.

The current command timeline is:

```text
cycle 5..8   MEM weight reads start on E0..E15
cycle 18     MXM IW starts, repeated 20 times
cycle 33     MEM activation read starts on E16, repeated 320 times
cycle 38     MXM Compute starts, repeated 320 times
cycle 59     four MEM result writes start, repeated 320 times
```

Transport latency is defined from producer issue to consumer visibility and
includes the CModel tick phase. For example, slice 32 to the MXM boundary is
five cycles; MXM output to result slices 40..43 is two cycles.
