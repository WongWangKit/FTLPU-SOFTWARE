# GLM-4-9B seq32 prefill

This entry adapts the original `zai-org/glm-4-9b` checkpoint (`chatglm`),
not GLM-Edge or GLM-4-9B-0414. It exercises decoder layer 0, including
both RMSNorms, Q/K/V projection and bias, partial RoPE, causal grouped-query
attention, output projection, residuals and SwiGLU FFN. It does not run
the complete 40-layer model, tokenizer, LM head or decode/KV export.

The full dimensions are hidden=4096, FFN=13696, Q heads=32, KV heads=2,
head width=128, rotary width=64, sequence length=32, batch=1.
The norm epsilon is 1.5625e-7. Source QKV and gate/up tensors are fused;
the importer splits them and preserves the nonzero QKV bias. Original
adjacent-pair RoPE is canonicalized only within the first 64 features.
The last 64 head features remain unrotated. Physical partial-RoPE packing
applies the same feature order to Q/K weights and biases.

## Reproduce

From the SOFTWARE repository root, with the configured CModel and compiler built:

```powershell
python tools/run_model_tests.py run --model glm4_9b --suite prefill --download
```

The downloader saves only the official first-layer tensors and embedding
rows 0–31, with revision and SHA-256 provenance, under
`models/official_dense/glm4_9b`. These 32 embedding rows are the test input;
they are not a natural-language prompt. The first downloaded revision was
`8cd2b585357ba9e702647ac4e6fa4fafe5cc7bee`.

Outputs are updated in `results/glm4_9b_decoder_layer_prefill`.
Consult `runner.json` and `result.json` for the actual run status. A compiled
binary or generated reference is not evidence that numerical execution passed.

## Execution and precision

The mathematical graph goes through the existing dense whole-layer graph
recognition, attention/FFN lowering and memory planner. Python handles model
import, reference calculation and orchestration; it does not emit FU schedules.
There are no GLM operators added to CModel.

Execution uses W8A16: official BF16 matrix weights are quantized using the
existing symmetric per-tensor INT8 importer, with BF16 activation checkpoints.
This is not native-BF16 matrix multiplication. The report separately measures
hardware vs the quantized reference and hardware vs the original BF16-weight
reference. The latter includes quantization error and should not be confused
with the hardware acceptance comparison.

The program uses raw macro ICU encoding and one-vector projection weight
loading. Common-barrier synchronization is selected with the compiler's
`--common-barrier` option because the whole layer exceeds the configured
shared queue-control record capacity. The configured hardware capacity is
not increased. DDR/C2C transfers and synchronization still execute in CModel.

Regression checks:

```powershell
python compiler/tests/glm4_decoder_importer_test.py
python compiler/tests/glm4_decoder_layout_test.py --tool build-ftlpu-vs2026-direct/compiler/ftlpu_opt.exe --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir tmp/glm4_layout_test
build-ftlpu-vs2026-direct/runtime/weight_page_builder_test.exe --partial-rope-bias
```

## Validated run (2026-11-04)

The layer-0 seq32 CModel run passed, with raw macro instructions, the
unmodified `ftlpu-lpu32.json` capacity and real DDR/C2C paging. The scheduled
execution limit was 3,306,382 cycles; the runtime trace spans 0–5,522,514,
including input/output and transport waits. This is a correctness baseline,
not a fully optimized prefetch schedule or a full-model TTFT measurement.

| Comparison | Max absolute | Mean absolute | Relative L2 | Cosine |
| --- | ---: | ---: | ---: | ---: |
| CModel vs W8A16 reference | 0.0078125 | 0.0000524616 | 0.0119750 | 0.99992944 |
| CModel vs native BF16-weight reference | 0.017578125 | 0.000482237 | 0.0810038 | 0.99675837 |
| W8A16 reference vs native BF16-weight reference | 0.021484375 | 0.000482159 | 0.0811197 | 0.99673646 |

The native reference is independent NumPy checkpoint math, not an official
Transformers execution. Its ChatGLM RMSNorm keeps the normalized values in
FP32 through gamma multiplication; attention scores apply the scale before
BF16 rounding. This comparison includes quantization and arithmetic
differences. The hardware acceptance comparison is the first row.

`result.json` records the metrics, package/runtime hashes and pipeline hash;
`runner.json` records the completed invocation and ICU export. The actual
trace is `runtime.pipeline.csv` (MEM, MXM, VXM, SXM, C2C and waits), with a
standalone `pipeline.html`. `icu_programs/index.csv` indexes the per-queue
programs exported from the runtime-linked `linked.ftlpu`; use the runtime
pipeline for actual issue cycles and dynamic transport behavior.

The model importer, full-size tensor layout, partial-RoPE bias packing,
existing six-family importer regression and 15 model-runner unit tests
passed. The saved numerical output was also rechecked with `--verify-only`.
