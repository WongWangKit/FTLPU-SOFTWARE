"""Independent CPU reference for the first pure-SWA layer's Q and shared KV, batch one."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from import_hf_decoder_layer import bf16


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "results/deepseek_v41_flash_layer0_qkv_prefill"
CONFIG = ROOT / "results/deepseek_v41_flash_assessment/official_config.json"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
CONFIG_SHA = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
MODEL_SHA = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
KERNEL_SHA = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
BLOCK = 32


def fp8_table():
    bits = np.arange(256, dtype=np.int32)
    exp = (bits >> 3) & 15
    mantissa = bits & 7
    magnitude = np.where(exp == 0, mantissa / 512, np.ldexp(1. + mantissa / 8, exp - 7))
    table = np.where(bits >= 128, -magnitude, magnitude).astype(np.float32)
    table[[127, 255]] = np.nan
    return table


E4M3FN = fp8_table()
# A midpoint selects the neighbour with an even low bit. 0x7e encodes +448.
MIDPOINTS = (E4M3FN[1:127] + E4M3FN[:126]) * np.float32(.5)


def encode_fp8(x):
    if not np.isfinite(x).all():
        raise ValueError("FP8 input must be finite")
    magnitude = np.minimum(np.abs(x), 448)
    lower = np.searchsorted(MIDPOINTS, magnitude)
    midpoint = MIDPOINTS[np.minimum(lower, len(MIDPOINTS) - 1)]
    code = lower + ((magnitude == midpoint) & (lower % 2 == 1))
    return code.astype(np.uint8) | (np.signbit(x).astype(np.uint8) << 7)


def scales(raw):
    if (raw == 255).any():
        raise ValueError("UE8M0 NaN scale")
    return np.ldexp(np.ones(raw.shape, dtype=np.float32), raw.astype(np.int16) - 127)


def act_quant(x):
    groups = x.reshape(len(x), -1, BLOCK)
    maxima = np.maximum(np.max(np.abs(groups), axis=-1), np.float32(1e-4))
    ratio = maxima * np.float32(1 / 448)
    fraction, exponent = np.frexp(ratio)
    # Frexp's exponent is one too high only for exact powers of two.
    biased = exponent - (fraction == .5) + 127
    if (biased < 0).any() or (biased >= 255).any():
        raise ValueError("Activation scale outside UE8M0")
    encoded = biased.astype(np.uint8)
    values = encode_fp8(groups / scales(encoded)[..., None]).reshape(x.shape)
    return values, encoded


def dequant_act(encoded, scale):
    return (E4M3FN[encoded].reshape(*scale.shape, BLOCK) * scales(scale)[..., None]).reshape(encoded.shape)


def check_quant(x, encoded, raw_scale):
    """Check scales and every quantized value via FP64 bin spacing and ties-to-even."""
    groups = x.astype(np.float64).reshape(*raw_scale.shape, BLOCK)
    maxima = np.maximum(np.abs(groups).max(axis=-1), 1.e-4)
    exp = np.ceil(np.log2(maxima / 448)).astype(np.int32)
    np.testing.assert_array_equal(raw_scale, exp + 127)
    scale = np.ldexp(np.ones_like(maxima), exp)[..., None]
    scaled = groups / scale
    # Normal E4M3 has three fraction bits, subnormal spacing is 2**-9.
    exponent = np.floor(np.log2(np.maximum(np.abs(scaled), 2. ** -6))).astype(np.int32)
    step = np.ldexp(np.ones_like(scaled), exponent - 3)
    reference = np.clip(np.rint(scaled / step) * step, -448, 448) * scale
    np.testing.assert_array_equal(dequant_act(encoded, raw_scale), reference.reshape(x.shape))
    np.testing.assert_array_equal(encoded >> 7, np.signbit(x))
    return dict(value_count=x.size, scale_count=raw_scale.size, comparison="exact")


def project(encoded, act_scale, weight, weight_scale):
    """Dot over 32 FP8 values, then apply both scales and accumulate groups in FP32."""
    rows, width = encoded.shape
    outputs = weight.shape[0]
    groups = width // BLOCK
    acts = E4M3FN[encoded].reshape(rows, groups, BLOCK).transpose(1, 0, 2).copy()
    a_scale = scales(act_scale)
    w_scale = scales(weight_scale)
    result = np.empty((rows, outputs), dtype=np.float32)
    # Limit dequantized working memory independently of matrix height.
    for start in range(0, outputs, 256):
        stop = min(start + 256, outputs)
        stripe = E4M3FN[weight[start:stop]]
        blocks = stripe.reshape(stop - start, groups, BLOCK).transpose(1, 2, 0).copy()
        products = acts @ blocks
        stripe_scales = w_scale[np.arange(start, stop) // BLOCK]
        total = np.zeros((rows, stop - start), dtype=np.float32)
        for group in range(groups):
            total += products[group] * a_scale[:, group, None] * stripe_scales[:, group]
        result[:, start:stop] = bf16(total)
    return result


def project_bf16_activation(x, weight, weight_scale):
    """Unquantized BF16 activations; BF16 dequant weights, ordered FP32 K32 sums.

    This deployment reference deliberately does not reproduce official FP8
    activation QDQ. Serial products match the specified MXM reduction order;
    all data here is used only for the independent CPU reference.
    """
    x = bf16(np.asarray(x, dtype=np.float32))
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("Expected X[M,K] and W[N,K]")
    if x.shape[1] % BLOCK or weight.shape[0] % BLOCK:
        raise ValueError("Expected complete K32/N32 weight blocks")
    if weight_scale.shape != (weight.shape[0] // BLOCK, weight.shape[1] // BLOCK):
        raise ValueError("FP8 weight scale shape mismatch")
    decoded = E4M3FN[weight]
    block_scales = scales(weight_scale)
    result = np.empty((len(x), len(weight)), dtype=np.float32)
    for start in range(0, len(weight), 256):
        stop = min(start + 256, len(weight))
        total = np.zeros((len(x), stop - start), dtype=np.float32)
        for group in range(x.shape[1] // BLOCK):
            partial = np.zeros_like(total)
            scale = block_scales[np.arange(start, stop) // BLOCK, group]
            for column in range(group * BLOCK, (group + 1) * BLOCK):
                rhs = bf16(decoded[start:stop, column] * scale)
                partial += x[:, column, None] * rhs[None, :]
            total += partial
        result[:, start:stop] = bf16(total)
    return result


def norm(x, gamma, eps):
    inv_rms = np.float32(1) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps))
    return bf16((x * inv_rms) * gamma)


def rope(x, dim, theta):
    tail = x[..., -dim:]
    freqs = 1 / np.power(np.float32(theta), np.arange(0, dim, 2, dtype=np.float32) / dim)
    angles = np.outer(np.arange(len(x), dtype=np.float32), freqs)
    shape = (len(x),) + (1,) * (x.ndim - 2) + (dim // 2,)
    c, s = np.cos(angles).reshape(shape), np.sin(angles).reshape(shape)
    result = x.copy()
    result[..., -dim::2] = bf16(tail[..., ::2] * c - tail[..., 1::2] * s)
    result[..., -dim + 1::2] = bf16(tail[..., ::2] * s + tail[..., 1::2] * c)
    return result


def tensor(weights, metadata, role, dtype, shape):
    item = metadata["tensors"][role]
    if item["dtype"] != dtype or tuple(item["shape"]) != tuple(shape):
        raise ValueError(f"{role}: expected {dtype} {shape}, received {item}")
    raw_dtype = "<u2" if dtype == "BF16" else "u1"
    path = weights / item["file"]
    if path.stat().st_size != np.prod(shape) * np.dtype(raw_dtype).itemsize:
        raise ValueError(f"Incorrect tensor byte count for {role}")
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != item["sha256"]:
        raise ValueError(f"SHA256 mismatch for {role}")
    raw = np.memmap(path, dtype=raw_dtype, shape=shape, mode="r")
    return (raw.astype(np.uint32) << 16).view(np.float32) if dtype == "BF16" else raw


def check_projection(encoded, act_scale, weight, weight_scale, actual):
    """Compare sampled rows/columns to independently dequantized FP64 scalar reductions."""
    rel_limit = 1.e-2  # accounts for the final BF16 rounding, incl. midpoint ties
    abs_limit = 2.e-6
    worst = 0.
    count = 0
    for m in (0, len(actual) // 2, len(actual) - 1):
        x = E4M3FN[encoded[m]].astype(np.float64) * np.repeat(scales(act_scale[m]), BLOCK)
        for n in (0, 31, 32, actual.shape[1] // 2 + 1, actual.shape[1] - 1):
            w = E4M3FN[weight[n]].astype(np.float64) * np.repeat(scales(weight_scale[n // BLOCK]), BLOCK)
            expected = np.sum(x * w)
            delta = abs(actual[m, n] - expected)
            if delta > abs_limit + rel_limit * abs(expected):
                raise ValueError(f"FP64 scalar projection check failed at [{m},{n}]")
            worst = max(worst, delta)
            count += 1
    return dict(count=count, max_abs_error=worst, atol=abs_limit, rtol=rel_limit)


def check_rope(y, rotated, dim, theta):
    """Use complex128 rotation as an oracle and bound error relative to each input pair."""
    np.testing.assert_array_equal(rotated[..., :-dim], y[..., :-dim])
    np.testing.assert_array_equal(rotated[0], y[0])
    z = y[..., -dim::2].astype(np.complex128) + 1j * y[..., -dim + 1::2].astype(np.float64)
    phase = np.outer(np.arange(len(y)), theta ** (-np.arange(0, dim, 2) / dim))
    expected = z * np.exp(1j * phase).reshape((len(y),) + (1,) * (y.ndim - 2) + (dim // 2,))
    errors = np.maximum(np.abs(rotated[..., -dim::2] - expected.real),
                        np.abs(rotated[..., -dim + 1::2] - expected.imag))
    worst = np.max(errors / np.maximum(np.abs(z), 1.e-30))
    if worst > .004:
        raise ValueError(f"Complex128 partial RoPE comparison failed: {worst}")
    return dict(count=z.size, max_component_error_relative_to_pair_length=float(worst), limit=.004)


def run(args):
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != CONFIG_SHA:
        raise ValueError("Pinned official config hash mismatch")
    cfg = json.loads(CONFIG.read_text())["text_config"]
    metadata = json.loads((args.weights / "provenance.json").read_text(encoding="utf-8"))
    if metadata["revision"] != REVISION:
        raise ValueError("Weights revision does not match the pinned model")
    hidden, rank = cfg["hidden_size"], cfg["q_lora_rank"]
    heads, dim = cfg["num_attention_heads"], cfg["head_dim"]
    seq, win = 32, cfg["sliding_window"]
    if cfg["compress_ratios"][0] != 0 or cfg["num_key_value_heads"] != 1:
        raise ValueError("Expected layer 0 pure SWA with single shared KV head")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    inventory, checks = {}, {}

    def save(name, value, logical_dtype):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains a non-finite value")
        path = out / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        inventory[name] = dict(file=path.name, shape=list(value.shape), logical_dtype=logical_dtype,
                               container=str(value.dtype), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    rng = np.random.Generator(np.random.PCG64(args.seed))
    x = bf16(rng.standard_normal((seq, hidden), dtype=np.float32))
    save("x", x, "bf16")
    x_codes, x_scales = act_quant(x)
    checks["x_act_quant"] = check_quant(x, x_codes, x_scales)
    save("x_fp8", x_codes, "e4m3fn bits")
    save("x_scale", x_scales, "ue8m0 bits")

    def linear(role, codes, scale, features):
        shape = (features, codes.shape[1])
        weight = tensor(args.weights, metadata, role, "F8_E4M3", shape)
        wscale = tensor(args.weights, metadata, role + "_scale", "F8_E8M0",
                        (features // BLOCK, codes.shape[1] // BLOCK))
        y = project(codes, scale, weight, wscale)
        checks[role] = check_projection(codes, scale, weight, wscale, y)
        return y

    def rms(role, y):
        gamma = tensor(args.weights, metadata, role, "BF16", (y.shape[-1],))
        z = norm(y, gamma, cfg["rms_norm_eps"])
        y64 = y.astype(np.float64)
        mathematical = y64 * gamma / np.sqrt(np.mean(y64 * y64, axis=-1, keepdims=True) + cfg["rms_norm_eps"])
        np.testing.assert_allclose(z, mathematical, rtol=.004, atol=1.e-9)
        checks[role] = dict(count=z.size, rtol=.004, atol=1.e-9,
                            max_abs_error=float(np.max(np.abs(z - mathematical))))
        save(role, z, "bf16")
        return z

    q_a = linear("wq_a", x_codes, x_scales, rank)
    save("q_a", q_a, "bf16")
    qr = rms("q_norm", q_a)
    kv_raw = linear("wkv", x_codes, x_scales, dim)
    save("kv_raw", kv_raw, "bf16")
    kv = rms("kv_norm", kv_raw)
    qr_codes, qr_scale = act_quant(qr)
    checks["q_norm_act_quant"] = check_quant(qr, qr_codes, qr_scale)
    save("q_norm_fp8", qr_codes, "e4m3fn bits")
    save("q_norm_scale", qr_scale, "ue8m0 bits")
    q = linear("wq_b", qr_codes, qr_scale, heads * dim).reshape(seq, heads, dim)
    save("q_pre_rope", q, "bf16")
    rd, theta = cfg["qk_rope_head_dim"], cfg["rope_theta"]
    for name, y in (("q_rope", q), ("kv_rope", kv)):
        rotated = rope(y, rd, theta)
        checks[name] = check_rope(y, rotated, rd, theta)
        save(name, rotated, "bf16")
        if name == "kv_rope":
            cache_codes, cache_scale = act_quant(rotated)
            checks["kv_act_quant"] = check_quant(rotated, cache_codes, cache_scale)
            save("kv_fp8", cache_codes, "e4m3fn bits")
            save("kv_scale", cache_scale, "ue8m0 bits")
            cache = np.zeros((win, dim), dtype=np.float32)
            cache[:seq] = bf16(dequant_act(cache_codes, cache_scale))
            save("window_kv_cache", cache, "bf16 cache holding dequantized fp8")

    report = dict(status="independent_cpu_reference", model=metadata["model"], revision=REVISION,
                  scope="layer 0 pure-SWA Q and shared KV, before sparse attention",
                  seed=args.seed, generator="numpy PCG64 standard_normal(dtype=float32), BF16 RNE",
                  batch=1, sequence=seq, start_pos=0, valid_cache_rows=seq,
                  sources={"config.json": CONFIG_SHA, "inference/model.py": MODEL_SHA,
                           "inference/kernel.py": KERNEL_SHA},
                  arithmetic=[
                      "Dynamic E4M3FN RNE per 32 activations, scale = power-of-two ceiling of max(absmax,1e-4)*(1/448) in FP32.",
                      "FP8 dot over each 32 K values, multiply activation and 32x32 weight UE8M0 scales, accumulate FP32 in K order, round BF16.",
                      "RMSNorm uses FP32 variance, eps=1e-20, multiply learned BF16 gamma before BF16 rounding.",
                      "Layer 0 uses base 10000 without YaRN, FP32 trig on adjacent pairs in last 64 dimensions, BF16 result.",
                      "All 512 post-RoPE KV features undergo dynamic per-32 FP8 quantization and BF16 dequantization before the cache write."
                  ],
                  limits="CPU mathematical emulation. FP32 dot/reduction/trig order can differ from CUDA TileLang. No CUDA kernel or device instruction executed in this reference.",
                  checks=checks, tensors=inventory)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Generated {len(inventory)} reference tensors at {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=CASE / "weights")
    parser.add_argument("--output", type=Path, default=CASE / "reference")
    parser.add_argument("--seed", type=int, default=1017)
    run(parser.parse_args())
