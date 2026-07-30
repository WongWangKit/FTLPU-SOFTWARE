# SmolLM2-135M GPU Benchmark

Benchmark date: 2026-07-30

## Common Environment

| Item | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Parameters | 134,515,008 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.3.0.dev0 |
| PyTorch CUDA build | 12.8 |
| Dtype | FP16 |
| Batch size | 1 |
| Sequence length | 128 |

The model was loaded from `models/SmolLM2-135M`.

## Tested GPUs

| GPU | Driver | Memory | Initial utilization |
| --- | --- | ---: | ---: |
| NVIDIA GeForce RTX 2080 Ti | 595.71.05 | 11,264 MiB | 0% |
| Tesla V100-PCIE-32GB | 580.95.05 | 32,768 MiB | 0% |

GPU 0 was used for both runs. GPU clocks were not locked.

## RTX 2080 Ti Results

| Operation | Mean | Median | P95 | Min | Max | Runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward, `use_cache=False`, 128 tokens | 35.102 ms | 34.936 ms | 35.977 ms | 34.839 ms | 36.673 ms | 100 |
| Prefill with KV Cache, 128 tokens | 38.661 ms | 38.510 ms | 39.532 ms | 38.381 ms | 40.142 ms | 100 |
| Decode one token, KV context length 128 | 33.544 ms | 33.472 ms | 33.877 ms | 33.316 ms | 35.458 ms | 100 |
| Generate 64 tokens from a 128-token context | 2,199.442 ms | 2,198.940 ms | 2,204.193 ms | 2,195.603 ms | 2,204.193 ms | 10 |

Additional results:

- Single-token decode throughput: **29.812 tokens/s**
- End-to-end 64-token generation throughput: **29.098 tokens/s**
- Peak PyTorch-allocated GPU memory: **282.5 MiB**
- Model loading and transfer to the GPU: **1.559 s**

## Tesla V100 Results

| Operation | Mean | Median | P95 | Min | Max | Runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward, `use_cache=False`, 128 tokens | 38.117 ms | 37.971 ms | 38.964 ms | 37.488 ms | 42.522 ms | 100 |
| Prefill with KV Cache, 128 tokens | 39.531 ms | 39.378 ms | 40.889 ms | 38.830 ms | 42.309 ms | 100 |
| Decode one token, KV context length 128 | 36.345 ms | 34.751 ms | 45.867 ms | 34.383 ms | 47.668 ms | 100 |
| Generate 64 tokens from a 128-token context | 2,309.030 ms | 2,310.461 ms | 2,315.461 ms | 2,302.033 ms | 2,315.461 ms | 10 |

Additional results:

- Single-token decode throughput: **27.514 tokens/s**
- End-to-end 64-token generation throughput: **27.717 tokens/s**
- Peak PyTorch-allocated GPU memory: **282.5 MiB**
- Model loading and transfer to the GPU: **1.541 s**

## Mean Latency Comparison

| Operation | RTX 2080 Ti | Tesla V100 | V100 latency difference |
| --- | ---: | ---: | ---: |
| Forward, `use_cache=False` | 35.102 ms | 38.117 ms | +8.59% |
| Prefill with KV Cache | 38.661 ms | 39.531 ms | +2.25% |
| Decode one token at context length 128 | 33.544 ms | 36.345 ms | +8.35% |
| Generate 64 tokens | 2,199.442 ms | 2,309.030 ms | +4.98% |

Positive percentages mean that the V100 took longer in this benchmark. These
results are for batch size 1 and do not characterize larger-batch throughput.

## Tesla V100 Batch Scaling

This sweep keeps the sequence and KV context lengths at 128 and increases the
batch size from 1 to 64.

| Batch | Prefill mean | Prefill throughput | Decode mean | Decode throughput | Generate 64 mean | Generation throughput | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.186 ms | 3,107.9 tokens/s | 36.407 ms | 27.5 tokens/s | 2,386.186 ms | 26.8 tokens/s | 282.5 MiB |
| 2 | 41.394 ms | 6,184.5 tokens/s | 36.148 ms | 55.3 tokens/s | 2,391.398 ms | 53.5 tokens/s | 300.3 MiB |
| 4 | 40.866 ms | 12,528.9 tokens/s | 36.380 ms | 110.0 tokens/s | 2,421.656 ms | 105.7 tokens/s | 335.9 MiB |
| 8 | 41.338 ms | 24,771.3 tokens/s | 36.996 ms | 216.2 tokens/s | 2,459.833 ms | 208.1 tokens/s | 407.4 MiB |
| 16 | 42.713 ms | 47,948.0 tokens/s | 37.493 ms | 426.7 tokens/s | 2,461.000 ms | 416.1 tokens/s | 549.4 MiB |
| 32 | 57.861 ms | 70,789.9 tokens/s | 37.424 ms | 855.1 tokens/s | 2,416.416 ms | 847.5 tokens/s | 839.9 MiB |
| 64 | 103.227 ms | 79,359.1 tokens/s | 37.384 ms | 1,711.9 tokens/s | 2,472.466 ms | 1,656.6 tokens/s | 1,403.7 MiB |

The throughput values are aggregate across the whole batch. At batch 64,
single-step decode latency is only 2.7% higher than at batch 1, while aggregate
decode throughput is 62.3 times higher. Prefill throughput begins to show
diminishing scaling between batch 32 and batch 64.

For each batch size, prefill and decode use 5 warm-up runs followed by 20
measured runs. Generation uses 1 warm-up run followed by 3 measured runs. The
decode input is one token per sequence with an existing 128-token KV Cache.
Generation produces 64 new tokens per sequence from 128 input tokens.

The forward result without a KV Cache is useful as a full-model compute
baseline. The prefill result with `use_cache=True` is the representative
prefill measurement for autoregressive inference.

## Measurement Method

All timed inputs were created on the GPU before timing. Each operation was
timed with `time.perf_counter()`, followed by `torch.cuda.synchronize()` so the
measurement includes completion of queued CUDA work.

The prefill benchmark:

1. Passes a random `[1, 128]` token tensor to the model.
2. Uses `use_cache=True`.
3. Returns logits and creates the 128-token KV Cache.
4. Uses 10 warm-up runs followed by 100 measured runs.

The decode benchmark:

1. Creates a 128-token KV Cache outside the timed region.
2. Passes one new `[1, 1]` token with that cache.
3. Times the model forward, logits production, and KV Cache update.
4. Recreates the same-length input cache before every measured run so every
   sample represents decode at context length 128.
5. Uses 10 warm-up runs followed by 100 measured runs.

The generation benchmark uses greedy decoding with `use_cache=True`. It
generates exactly 64 new tokens from 128 input tokens. It includes the initial
prefill, cached decode steps, Hugging Face generation-loop overhead, and CUDA
synchronization. It uses 2 warm-up runs followed by 10 measured runs.

## Timing Boundary

The measurements include:

- Python/PyTorch model-call and CUDA kernel-launch overhead
- GPU computation
- GPU memory traffic for weights, activations, logits, and KV Cache
- CUDA synchronization overhead

The measurements exclude:

- Model download
- Model loading, except for the separately reported load time
- Tokenization
- CPU-to-GPU input transfer
- GPU-to-CPU output transfer
- Text decoding

## Reproduce the Forward Benchmark

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark/benchmark_smollm2_gpu.py \
  --model models/SmolLM2-135M \
  --local-files-only \
  --sequence-length 128 \
  --warmup-runs 10 \
  --runs 100
```
