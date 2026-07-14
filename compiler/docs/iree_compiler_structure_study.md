# IREE Compiler Structure Study For FTLPU

This note records the compiler structure we should learn from IREE and how it
should reshape the FTLPU compiler. The main point is not to copy IREE's code
volume, but to copy the engineering boundaries: dialects, passes, pipelines,
target backends, and serialization are separate concepts.

## What IREE Gets Right

### Tool Entrypoint Is Thin

`third_party/iree/tools/iree-compile-main.cc` only calls
`ireeCompilerRunMain(argc, argv)`. The actual compiler is behind a stable
driver API.

For FTLPU, `ftlpu_opt` should stay thin:

```text
main()
  -> CompilerSession
  -> CompilerInvocation
  -> parse input
  -> build selected pipeline
  -> run passes
  -> emit IR or binary
```

The tool should not know how StableHLO becomes schedule IR.

### Session And Invocation Are Separate

IREE's compiler driver has a session that owns global state such as dialect
registration, target registry, plugin activation, and options. Each invocation
owns one compilation unit and one pipeline execution.

For FTLPU:

- `CompilerSession`: target registry, dialect registry, global options.
- `CompilerInvocation`: input file, output file, selected phase range, current
  module, diagnostics.

This split matters once we add multiple compile commands, compile-from/to
phases, or use the compiler as a library from tests/runtime tools.

### Pipelines Are Named Phase Builders

IREE has explicit phases:

```text
Input
ABI
Preprocessing
GlobalOptimization
DispatchCreation
Flow
Stream
ExecutableSources
ExecutableConfigurations
ExecutableTargets
HAL
VM
End
```

Each phase is a pass pipeline builder with early-exit and late-entry support.
The important lesson is the shape:

```cpp
void buildFtlpuTransformPipeline(
    TargetRegistry& targets,
    PipelineOptions options,
    OpPassManager& pm,
    FtlpuPipelinePhase compileFrom,
    FtlpuPipelinePhase compileTo);
```

FTLPU should expose phases like:

```text
Start
StableHLOInput
Kernel
Tensor
Stream
Schedule
ExecutableConfig
ExecutableBinary
End
```

Then tests can intentionally stop at `--compile-to=stream` or resume from
`--compile-from=tensor`, instead of relying on ad-hoc comma strings.

### Dialects Own IR, Transforms Own Passes

IREE's Stream dialect is not a file named `passes.cpp`. It is split into:

```text
Dialect/Stream/IR
Dialect/Stream/Analysis
Dialect/Stream/Conversion
Dialect/Stream/Transforms
```

The Stream pipeline itself is also layered:

```text
stream.tensor.* -> stream.async.* -> stream.cmd.*
```

For FTLPU, we should mirror the boundary:

```text
compiler/lib/Dialect/Kernel/IR
compiler/lib/Dialect/Kernel/Transforms
compiler/lib/Dialect/Tensor/IR
compiler/lib/Dialect/Tensor/Analysis
compiler/lib/Dialect/Tensor/Transforms
compiler/lib/Dialect/Stream/IR
compiler/lib/Dialect/Stream/Analysis
compiler/lib/Dialect/Stream/Transforms
compiler/lib/Dialect/Schedule/IR
compiler/lib/Dialect/Schedule/Analysis
compiler/lib/Dialect/Schedule/Transforms
```

The current `compiler/src/passes.cpp` should become glue around real passes,
not the place where all IR construction lives.

### Passes Are Registered, Named, And Testable

IREE defines passes in `Passes.td`, implements them as one file per pass, and
registers both individual passes and named pipelines.

For FTLPU, even before we fully adopt TableGen, we should follow the same
contract:

- one pass class/function per transformation;
- stable pass name;
- pass options struct;
- dependent dialect declaration;
- verifier after each major lowering boundary;
- lit/FileCheck-style test per pass.

Current issue: `stablehlo-to-kernel,kernel-to-tensor,tensor-to-stream` works,
but those are string branches in one function. That is okay for a prototype,
not okay for a backend.

### Target Backend Is An Interface

IREE target plugins implement `TargetBackend` with these responsibilities:

- report supported types;
- register dependent dialects;
- build configuration pass pipeline;
- build translation pass pipeline;
- build linking pass pipeline;
- serialize executable binary.

FTLPU needs the same shape:

```cpp
class TargetBackend {
public:
  virtual std::string name() const = 0;
  virtual SupportedTypes supported_types() const = 0;
  virtual void build_configuration_pipeline(PassManager&) = 0;
  virtual void build_translation_pipeline(PassManager&) = 0;
  virtual void build_linking_pipeline(PassManager&) = 0;
  virtual LogicalResult serialize_executable(Module&, BinaryWriter&) = 0;
};
```

For the first backend:

```text
Target/FtlpuCModel
  -> configures MEM/MXM/VXM/SXM/ICU constraints
  -> lowers schedule IR to queue IR
  -> serializes .ftlpu
  -> runtime loads ICU queues and starts clocks
```

Later hardware backend can share most high-level passes but replace
serialization and target constraints.

## How This Applies To FTLPU IR Layers

The four layers are still reasonable, but they need more rigorous ownership.

### Kernel IR

Purpose: map model ops to LPU functional units.

Examples:

```text
ftlpu.kernel.mxm_matmul
ftlpu.kernel.vxm_swiglu
ftlpu.kernel.ffn
```

This layer should not assign addresses or cycles.

### Tensor IR

Purpose: own MEM objects and placement.

This is where the rank-5 address model belongs:

```text
[device, hemisphere, slice, bank, word, byte]
shape [N, 2, 44, 2, 4096]
```

Allocator and placement analysis should live here, not inside a print pass.

### Stream IR

Purpose: describe FU/MEM data movement and stream lifetimes.

Each stream needs:

```text
stream_id
direction
producer endpoint
consumer endpoint
produce_cycle
consume_cycle
transport_latency
byte range
stream register mapping
```

This layer is the right place for stream endpoint modeling, but final issue
cycles should still be owned by schedule IR.

### Schedule IR

Purpose: decide instruction issue cycle and queue ordering.

It must be resource aware:

- MEM read/write ports;
- MXM load and compute;
- VXM pipeline;
- SXM pipeline;
- transport latency;
- ICU queue availability;
- stream ID lifetime conflicts.

Schedule IR should be close to binary queue sections but still readable and
verifiable.

## Proposed FTLPU Compiler Directory Shape

Short-term target:

```text
compiler/
  include/ftlpu/compiler/
    API/
      compiler_session.hpp
      compiler_invocation.hpp
    Dialect/
      Kernel/
      Tensor/
      Stream/
      Schedule/
    Target/
      target_backend.hpp
      target_registry.hpp
      ftlpu_cmodel_target.hpp
    Pipelines/
      phases.hpp
      pipelines.hpp
    Support/
      diagnostics.hpp
      source_manager.hpp

  src/
    API/
    Dialect/
      Kernel/IR
      Kernel/Transforms
      Tensor/IR
      Tensor/Analysis
      Tensor/Transforms
      Stream/IR
      Stream/Analysis
      Stream/Transforms
      Schedule/IR
      Schedule/Analysis
      Schedule/Transforms
    Target/
      FtlpuCModel/
    Pipelines/
    Tools/
```

The existing `target_model.hpp/cpp` should be split:

- MEM address model -> `Dialect/Tensor/Analysis/MemoryLayout`.
- stream lifetime allocation -> `Dialect/Stream/Analysis/StreamAllocator`.
- resource/cycle scheduler -> `Dialect/Schedule/Analysis/ResourceScheduler`.
- CModel queue constraints -> `Target/FtlpuCModel/`.

## Migration Plan

### Step 1: Make The Prototype Honest

Keep the existing textual IR, but split source files:

```text
src/Dialect/Tensor/Analysis/MemoryAllocator.cpp
src/Dialect/Stream/Analysis/StreamManager.cpp
src/Dialect/Schedule/Analysis/ResourceScheduler.cpp
src/Pipelines/Pipelines.cpp
```

`passes.cpp` becomes a small adapter, not the implementation home.

### Step 2: Introduce Real Pipeline Phases

Add:

```cpp
enum class PipelinePhase {
  Start,
  StableHLOInput,
  Kernel,
  Tensor,
  Stream,
  Schedule,
  ExecutableBinary,
  End,
};
```

Update `ftlpu_opt` to support:

```text
--compile-from=
--compile-to=
--pipeline=ftlpu-lpu-pipeline
```

### Step 3: Add Verifiers

After every boundary:

- Kernel verifier: supported ops/types/shapes.
- Tensor verifier: address ranges do not overlap and fit MEM.
- Stream verifier: endpoints exist, direction is valid, stream lifetime does
  not conflict.
- Schedule verifier: no resource overlap, all consume cycles satisfy producer
  cycle plus latency.

This is where compiler quality starts becoming visible.

### Step 4: Add Target Backend Interface

Move CModel-specific constraints into:

```text
Target/FtlpuCModel/
```

The generic schedule layer should not hard-code "CModel". It should ask the
selected target for:

- FU inventory;
- MEM topology;
- transport latency table;
- queue mapping;
- binary serialization rules.

### Step 5: Replace String IR With MLIR Dialects

Once the above boundaries settle, define real MLIR dialects with ODS/TableGen.
Do this after the architecture is right; otherwise we will just encode bad
structure in TableGen.

## Immediate Critique Of Current Code

The current compiler is useful as a spike, but structurally weak:

- `passes.cpp` mixes parsing-level lowering, allocation, scheduling, and
  printing.
- IR is text-only, so passes cannot analyze/transform it robustly.
- no verifier per layer;
- no pass registration or pipeline builder;
- target model is not target-pluggable;
- tests check fragments rather than structural invariants;
- runtime binary emission is not yet a real target backend serialization phase.

The next serious milestone should be architectural, not adding more ops into
the current file.

## Recommended Next Commit

Refactor without changing behavior:

1. Move allocator/stream/scheduler classes into layer-specific analysis dirs.
2. Add `Pipelines/phases.hpp` and `Pipelines/pipelines.cpp`.
3. Add verifier functions for tensor/stream/schedule textual modules.
4. Keep all existing tests green.
5. Only then add more FFN/lower-to-binary behavior.
