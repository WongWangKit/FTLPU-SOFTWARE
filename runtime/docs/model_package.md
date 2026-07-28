# ModelPackage and ModelSession

`ModelPackage` is the model-level container above an FTLPU command binary. It
keeps command binaries reusable instead of statically expanding every decoder
layer into one very large ICU program.

## Package contents

The version-1 `.ftlpum` format contains:

- model and architecture identity;
- named constant tensors;
- raw or symmetric INT8 quantization metadata;
- named external and intermediate values;
- one or more embedded `.ftlpu` executables;
- an ordered invocation list that maps named values to executable bindings.

The file magic is `FTLPUM01`. Quantized tensor metadata includes encoding,
axis, block size, and scales. This metadata is independent of physical SRAM
layout; the embedded executable's `BinaryBinding` remains the source of truth
for physical placement.

## Session execution

`ModelSession` loads a package, accepts named external inputs, and executes its
invocations in order. For each invocation it:

1. resets and reloads the CModel ICU queues;
2. resolves constants or preceding results by name;
3. applies the planned host upload, device alias, or device layout copy;
4. runs the command program and drain cycles;
5. retains intermediate outputs in MEM and downloads external outputs.

`SessionMemoryPlanner` computes producer/consumer lifetimes before execution.
Compatible physical bindings alias directly. Incompatible FP16 layouts use a
CModel MEM-to-MEM layout transfer without materializing the logical tensor in
a host buffer. The transfer is an explicit backend operation so it can later
be replaced by an ICU MEM/SXM adapter executable on hardware.

At each executable boundary, the CModel clears ICU, stream, MXM, VXM, and SXM
execution state while preserving MEM SRAM. This gives each command binary a
cycle-zero execution context without losing device-resident values.

## Internal executable bindings

An executable binding with `access = internal` is physical storage owned by
the executable rather than a slot supplied by `ModelSession`. Binary format
version 7 gives such bindings a typed initializer and initializer parameters.
The runtime currently supports zero-fill, tiled causal-mask generation, and
FP16 RoPE cosine/sine table generation. The compiler emits the table shape,
physical slices, `theta`, and head dimension; `CModelRuntime::load` materializes
the data before ICU clocks start.

This keeps algorithmic constants in the compiler/runtime ABI while avoiding
large repeated payloads and test-only SRAM initialization code.

## Real SmolLM2 layer golden

`import_hf_decoder_layer.py` reads a standard Llama-compatible
`config.json` and safetensors checkpoint without PyTorch. It extracts any
decoder layer, transposes Hugging Face Linear weights to StableHLO layout,
applies symmetric per-tensor INT8 quantization, and emits a NumPy reference.

The SmolLM2-135M layer-0, sequence-length-128 test packages:

- the compiled reusable decoder-layer executable;
- real layer-0 Q/K/V/O and gate/up/down weights;
- the two real RMSNorm weights;
- an embedding-derived input and quantized golden output.

The `ModelSession -> ICU -> CModel` result matched all 73,728 FP16 values with
maximum absolute error `0.03125` and mean absolute error `0.000815428`.

The test now obtains its RoPE table exclusively from the typed internal binary
binding.

## Real two-layer golden

The sequence-length-128 layer-0/layer-1 test uses two independently compiled
executables because each layer has different W8A16 scales. `hidden.1` remains
resident in CModel MEM between invocations. A decoder compiled with the
`vxm_feedback` RMSNorm strategy uses `fp16_mxm_distributed_16` as its persistent
external-activation ABI. The final residual add writes directly to the same
slices and base row expected by the next decoder invocation, so the planner
selects a device alias instead of a layout copy.

The session performs 19 host uploads for the model input and layer constants,
one final host download, one device alias, zero device copies, and no host
transfer for `hidden.1`. All 73,728 final FP16 values pass against the NumPy
two-layer golden with maximum absolute error `0.0625` and mean absolute error
`0.00147592`.

## Reusable quantized executables

Binary format version 8 carries typed VXM immediate relocations. A relocation
identifies an input binding, quantization scale index, queue, command, and
operand. `ModelSession` resolves the invocation's bound `ModelTensor`, copies
the executable template, and patches only the declared immediates before
loading ICU queues.

Q/K/V/O and gate/up/down weight dequantization therefore share one decoder
executable across layers with different per-tensor scales. The real two-layer
package shrinks from 50,593,779 bytes with two specialized executables to
29,445,522 bytes with one reusable executable. The shared-executable CModel
golden passes with maximum absolute error `0.0625`, mean absolute error
`0.00138637`, one device alias, and zero device copies.

`build_hf_decoder_stack.py` extends the same flow to any contiguous HF decoder
layer range. It chains each layer's NumPy golden into the next layer input,
imports all real layer tensors, and packages the stack with one reusable
executable. By default it reads `num_hidden_layers` from `config.json`.
