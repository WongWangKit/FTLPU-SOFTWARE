# Current ICU Macro ISA

Updated 2026-11-13. See [the detailed Chinese specification](icu_macro_isa.zh-CN.md).

## Execution formats

MEM and all three independent MXM queues use packed Macro streams. SXM Transpose
and Permute now use the same bounded packed Macro frontend on **separate queues**.
MEM/SXM iMEM words remain 96 bits; MXM words remain 128 bits. LPU32 active context
limits are 256 bits for MEM, MXM and SXM Transpose, and 384 bits for SXM Permute.
VXM remains a single-word 96-bit sequential CONFIG ISA. Synchronization and C2C
transport controls retain their semantics.

Repeat2D, FU-specific 2-D/3-D packets (including short forms), WRITE_READ_2D,
STREAM_ND, MEM slice programs, SXM tile programs and SXM/VXM RUN_2D/RUN_3D are
removed. Recompile old artifacts. There is no legacy-packet conversion executor.
Scalar/Repeat1D helpers used by low-level tests are not a compiler deployment mode.

## Direct lowering

Compiler affine domains are mathematical schedule representations, not hardware
instructions. MacroCommandMaterializer directly lowers them to command.macro,
flattening compatible axes and splitting remaining dimensions chronologically.
Blocked addresses, alternating buffers and terminal accumulator behavior are
resolved before packing; FU streams are not expanded and reverse-compressed.

Macro has its own inner/outer timing and operand induction fields, compact and
extended descriptors, delta dictionaries and runs. Those compression fields are
not separate hardware RUN_2D/READ_3D instructions. Address relocation precedes
physical packing. Count semantic templates, physical iMEM words and emitted FU
issues separately. The default is raw-macro; macro is an alias. none, control and
the old mem-slice-program option are rejected.

## SXM and scheduling limits

LPU32 SXM templates contain 16 native bits for transpose and 179 for permute,
including the complete lane map. Wide/local/broadcast FU behavior is preserved.
Changing a permute map requires a different template; there is no implicit map
rotation. Transpose/Permute retain 2048-word iMEMs and four bounded context slots.
The shared frontend uses the existing 3-word reservoir, 64-bit decode window,
8-entry DDB and target fetch timing. No capacities or bandwidths were increased.

Only one Macro executes per ICU queue. A later Macro's first issue must follow
the previous Macro's final issue. Buffered lookahead contexts cannot interleave
execution. Compilation, packing and CModel validate this rule. Common barriers
coordinate packed compute queues with MEM synchronized DDR/C2C transfers.

## Verification

Tests cover packed replay, exact FU issue timing/payloads, SXM geometry, forbidden
interleaving, relocation, synchronization, capacity and numerical transpose/RoPE.
direct_raw_macro_cli_test exercises default/explicit CLI paths and a numerical FFN.
Export actual runtime pipeline and relocated per-ICU programs for model checks.
The DeepSeek seq32 SWA regression covers the attention frontend and SWA core only,
not mHC, output projection, FFN or a complete decoder/model.
