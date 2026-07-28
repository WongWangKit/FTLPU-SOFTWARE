# SmolLM2-135M Decoder Layer Pipeline

![Complete decoder-layer runtime pipeline](smollm2_decoder_layer_pipeline.png)

The figure is generated from `decoder_layer.runtime.csv`, which is decoded
from the compiler-produced `.ftlpu` binary before the CModel clock starts.
The rows aggregate commands by physical functional unit while preserving
MEM read/write, MXM load/compute, accumulator destination, VXM, and SXM
colors. The SVG retains exact-cycle tooltips.

The current `seq_len=128`, `hidden=576`, `intermediate=1536` schedule is:

| Stage | Cycle range |
| --- | ---: |
| RMSNorm 1 | 0..7566 |
| Q/K/V projection | 7566..30429 |
| QK | 30429..33810 |
| Softmax | 33810..38786 |
| PV | 38786..51257 |
| O projection | 51257..73685 |
| Residual 1 | 73695..76125 |
| RMSNorm 2 | 76143..83655 |
| Gate/Up projection | 83655..146032 |
| SwiGLU | 146032..158525 |
| Down projection | 158525..188810 |
| Residual 2 | 188810..191312 |
| Pipeline drain | 191312..192047 |

This is a correctness baseline, not a globally pipelined schedule. The
stage bands make the current cross-operator serialization visible and provide
a stable comparison point for later RMSNorm-to-projection, Attention block,
SwiGLU-to-down, and residual streaming optimizations.

