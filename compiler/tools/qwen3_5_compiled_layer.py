#!/usr/bin/env python3
"""Serial, compiled Qwen3.5 dense DeltaNet layer bring-up.

Every arithmetic stage is compiled from Kernel IR and runs on the CModel ICU/FU
path. Python only reshapes, broadcasts, pads and transfers stage bindings.
This is a correctness harness, not a single resident/paged layer program.
"""
from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from qwen3_5_reference import Qwen35DeltaConfig, Qwen35DeltaWeights


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


class CompiledStages:
    def __init__(self, compiler: Path, runner: Path, directory: Path):
        self.compiler, self.runner, self.directory = compiler, runner, directory
        directory.mkdir(parents=True, exist_ok=True)
        self.stages = []
        self.program_cache = directory / "program_cache"
        self.program_cache.mkdir(exist_ok=True)
        (directory / "summary.json").unlink(missing_ok=True)

    def run(self, name, source, inputs, outputs):
        prefix = self.directory / f"{len(self.stages):03d}_{name}"
        mlir, binary = prefix.with_suffix(".kernel.mlir"), prefix.with_suffix(".ftlpu")
        mlir.write_text(source, encoding="utf-8")
        # Identical tile programs have different input data, not different
        # hardware instructions. Compile each source once, keeping real ICU
        # execution for every tile. Include the compiler identity in the key.
        stat = self.compiler.stat()
        key = hashlib.sha256((source + str(self.compiler) + str(stat.st_mtime_ns)
                              + str(stat.st_size)).encode()).hexdigest()
        cached = self.program_cache / (key + ".ftlpu")
        if not cached.exists():
            result = subprocess.run([str(self.compiler), "--input", str(mlir), "--output", str(binary)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"compile {name} (exit {result.returncode}): {result.stderr}")
            shutil.copyfile(binary, cached)
        else:
            shutil.copyfile(cached, binary)
        paths = []
        for i, (value, dtype) in enumerate(inputs):
            path = Path(f"{prefix}.input{i}.bin")
            value = np.asarray(value)
            if dtype == "bf16":
                (bf16(value).view(np.uint32) >> 16).astype("<u2").tofile(path)
            else:
                value.astype("<f4" if dtype == "f32" else "i1").tofile(path)
            paths.append(str(path.resolve()))
        listing = Path(f"{prefix}.inputs.txt")
        listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
        fingerprint = hashlib.sha256(binary.read_bytes())
        runner_stat = self.runner.stat()
        fingerprint.update((str(self.runner)+str(runner_stat.st_mtime_ns)+str(runner_stat.st_size)).encode())
        for path in paths:
            fingerprint.update(hashlib.sha256(Path(path).read_bytes()).digest())
        identity = fingerprint.hexdigest()
        execution_file = Path(f"{prefix}.execution.json")
        previous = json.loads(execution_file.read_text(encoding='utf-8')) if execution_file.exists() else {}
        valid = previous.get('identity') == identity and len(previous.get('outputs', [])) == len(outputs)
        if valid:
            for i, digest in enumerate(previous['outputs']):
                path = Path(f"{prefix}.{i}.bin")
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    valid = False
                    break
        if valid:
            result = subprocess.CompletedProcess([], 0, previous['execution'], '')
        else:
            result = subprocess.run([str(self.runner), str(binary), str(listing), str(prefix)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"execute {name}: {result.stderr}")
        self.stages.append(dict(name=name, binary_bytes=binary.stat().st_size,
                                binary_path=str(binary.resolve()),execution=result.stdout.strip()))
        decoded = []
        for i, (shape, dtype) in enumerate(outputs):
            path = Path(f"{prefix}.{i}.bin")
            if dtype == "bf16":
                value = (np.fromfile(path, "<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                value = np.fromfile(path, "<f4")
            value = value.reshape(shape)
            if not np.isfinite(value).all():
                raise AssertionError(f"non-finite output in {name}")
            decoded.append(value)
        execution_file.write_text(json.dumps(dict(identity=identity,execution=result.stdout,
            outputs=[hashlib.sha256(Path(f"{prefix}.{i}.bin").read_bytes()).hexdigest()
                     for i in range(len(outputs))]),indent=2)+'\n',encoding='utf-8')
        (self.directory/'stages.partial.json').write_text(json.dumps(self.stages,indent=2)+'\n',encoding='utf-8')
        print(f"{len(self.stages):03d} {name}: {result.stdout.strip()}", flush=True)
        return decoded

    @staticmethod
    def type(shape, dtype):
        return "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"

    def alu(self, name, lhs, rhs=None, kind="multiply", immediate=None,
            dtype="f32", lhs_dtype="f32", rhs_dtype="f32", reduce=False):
        lhs = np.asarray(lhs, dtype=np.float32)
        if lhs.ndim != 2:
            raise ValueError("ALU inputs must be matrices")
        rhs = lhs if rhs is None else np.asarray(rhs, dtype=np.float32)
        if lhs.shape != rhs.shape:
            raise ValueError("explicitly broadcast inputs before vector lowering")
        # Keep scalar-broadcast SRAM and per-stage instruction queues bounded.
        # These are data-only slices; each tile still executes compiled VXM math.
        if lhs.size > 4096:
            if reduce:
                if lhs.shape[1] > 4096:
                    raise ValueError("row reduction currently supports at most 4096 features")
                tile_rows = max(1, 4096 // lhs.shape[1])
                return np.concatenate([self.alu(name + f"_tile{start}", lhs[start:start + tile_rows],
                    rhs[start:start + tile_rows], kind, immediate, dtype, lhs_dtype, rhs_dtype, True)
                    for start in range(0, lhs.shape[0], tile_rows)], axis=0)
            a_flat, b_flat = lhs.reshape(-1), rhs.reshape(-1)
            return np.concatenate([self.alu(name + f"_tile{start}",
                a_flat[start:start + 4096].reshape(1, -1), b_flat[start:start + 4096].reshape(1, -1),
                kind, immediate, dtype, lhs_dtype, rhs_dtype, False).reshape(-1)
                for start in range(0, lhs.size, 4096)]).reshape(lhs.shape)
        outshape = (lhs.shape[0], 1) if reduce else lhs.shape
        a, b, o = self.type(lhs.shape, lhs_dtype), self.type(rhs.shape, rhs_dtype), self.type(outshape, dtype)
        source = f'''module {{
  func.func @stage(%a: {a}, %b: {b}) -> {o} {{
    %o = ftlpu.kernel.vector_alu %a, %b {{kind = "{kind}",
      use_immediate = {str(immediate is not None).lower()}, immediate = {float(immediate or 0):.9e} : f32,
      reduce = {str(reduce).lower()}}} : ({a}, {b}) -> {o}
    return %o : {o}
  }}
}}'''
        return self.run(name, source, [(lhs, lhs_dtype), (rhs, rhs_dtype)], [(outshape, dtype)])[0]

    def sigmoid(self, name, x):
        if x.size <= 4096:
            graph = VectorGraph(self, name, [(x, "f32")])
            return graph.finish(graph.sigmoid(0))
        n = self.alu(name + "_neg", x, kind="negate", immediate=0)
        e = self.alu(name + "_exp", n, kind="exp", immediate=0)
        p = self.alu(name + "_one_plus", e, kind="add", immediate=1)
        return self.alu(name + "_recip", p, kind="reciprocal", immediate=0)

    def silu(self, name, x, dtype="bf16"):
        if x.size <= 4096:
            graph = VectorGraph(self, name, [(x, "f32")])
            return graph.finish(graph.add(0, graph.sigmoid(0), dtype=dtype))
        return self.alu(name, x, self.sigmoid(name, x), dtype=dtype)

    def norm(self, name, x, gamma, epsilon, mean=True, scale=1):
        if x.size <= 4096:
            graph = VectorGraph(self, name, [(x, "bf16"), (np.asarray(gamma).reshape(1, -1), "f32")])
            return graph.finish(graph.norm(0, 1, epsilon, mean, scale))
        if x.shape[1] <= 4096:
            rows = max(1, 4096 // x.shape[1])
            return np.concatenate([self.norm(name + f"_tile{start}", x[start:start + rows],
                gamma, epsilon, mean, scale) for start in range(0, x.shape[0], rows)], axis=0)
        # Scalar-broadcast rows let the terminal accumulator perform a real
        # hardware reduction over the feature axis without a CPU reduction.
        s = self.alu(name + "_sum_squares", x, x, reduce=True, lhs_dtype="bf16", rhs_dtype="bf16")
        if mean:
            s = self.alu(name + "_mean", s, immediate=1 / x.shape[1])
        s = self.alu(name + "_epsilon", s, kind="add", immediate=epsilon)
        inv = self.alu(name + "_rsqrt", s, kind="rsqrt", immediate=0)
        if scale != 1:
            inv = self.alu(name + "_scale", inv, immediate=scale)
        normalized = self.alu(name + "_normalized", x, np.broadcast_to(inv, x.shape), lhs_dtype="bf16")
        return self.alu(name, normalized, np.broadcast_to(gamma, x.shape), dtype="bf16")

    def softplus(self, name, x):
        if x.size <= 4096:
            graph = VectorGraph(self, name, [(x, "f32")])
            return graph.finish(graph.softplus(0))
        # log(1+exp(-abs(x))) = 2 * atanh(t), t=exp(-abs(x))/(2+exp(-abs(x))).
        # |t| <= 1/3: six odd terms leave <1.1e-7 absolute truncation error.
        # All coefficients are compile-time constants; every arithmetic step
        # executes ordinary VXM instructions, including the max(x,0) branch.
        neg = self.alu(name + "_neg", x, kind="negate", immediate=0)
        absolute = self.alu(name + "_abs", x, neg, kind="max")
        negative_abs = self.alu(name + "_negative_abs", absolute, kind="negate", immediate=0)
        e = self.alu(name + "_exp", negative_abs, kind="exp", immediate=0)
        denominator = self.alu(name + "_denominator", e, kind="add", immediate=2)
        inverse = self.alu(name + "_inverse", denominator, kind="reciprocal", immediate=0)
        t = self.alu(name + "_t", e, inverse)
        square = self.alu(name + "_square", t, t)
        polynomial = self.alu(name + "_coef11", square, immediate=1 / 11)
        polynomial = self.alu(name + "_coef9", polynomial, kind="add", immediate=1 / 9)
        for degree in (7, 5, 3, 1):
            polynomial = self.alu(name + f"_horner{degree}", polynomial, square)
            polynomial = self.alu(name + f"_coef{degree}", polynomial, kind="add", immediate=1 / degree)
        correction = self.alu(name + "_correction", polynomial, t)
        correction = self.alu(name + "_twice", correction, immediate=2)
        positive = self.alu(name + "_positive", x, kind="max", immediate=0)
        return self.alu(name, positive, correction, kind="add")

    def projection(self, name, x, weight, scale):
        m, k = x.shape
        n = weight.shape[0]
        if k > 4096:
            partials = [self.projection(name + f"_k{start}", x[:, start:start + 4096],
                        weight[:, start:start + 4096], scale) for start in range(0, k, 4096)]
            accumulated = partials[0]
            for i, partial in enumerate(partials[1:]):
                accumulated = self.alu(name + f"_sum{i}", accumulated, partial, kind="add",
                                       lhs_dtype="bf16" if i == 0 else "f32", rhs_dtype="bf16")
            return self.alu(name + "_cast", accumulated, kind="cast", immediate=0, dtype="bf16")
        max_columns = 64 * max(1, 64 // (k // 32))
        if n > max_columns:
            return np.concatenate([self.projection(name + f"_tile{start}", x,
                weight[start:start + max_columns], scale)
                for start in range(0, n, max_columns)], axis=1)
        padded_n = (n + 63) // 64 * 64
        packed = np.pad(weight.T, ((0, 0), (0, padded_n - n)))
        a, w, o = self.type((m, k), "bf16"), self.type(packed.shape, "i8"), self.type((m, padded_n), "bf16")
        source = f'''module {{ func.func @stage(%a: {a}, %w: {w}) -> {o} {{
          %o = "ftlpu.kernel.matmul"(%a, %w) {{m = {m} : i64, n = {padded_n} : i64,
             k = {k} : i64, unit = "MXM", rhs_scale = {float(scale):.9e} : f32}} : ({a}, {w}) -> {o}
          return %o : {o}
        }} }}'''
        return self.run(name, source, [(x, "bf16"), (packed, "i8")], [((m, padded_n), "bf16")])[0][:, :n].copy()

    def conv(self, x, weight):
        # The allocator reserves the input and causal padding together. At
        # seq32, 2048 channels exceed that activation row arena; 1024 fits
        # without enlarging physical SRAM or changing convolution semantics.
        if x.shape[1] > 1024:
            return np.concatenate([self.conv(x[:, start:start + 1024], weight[start:start + 1024])
                for start in range(0, x.shape[1], 1024)], axis=1)
        a, w = self.type(x.shape, "bf16"), self.type(weight.shape, "bf16")
        source = f'''module {{ func.func @stage(%a: {a}, %w: {w}) -> {a} {{
          %o = ftlpu.kernel.depthwise_conv1d %a, %w {{kernel_size = 4 : i64,
            causal = true, activation = "silu"}} : ({a}, {w}) -> {a}
          return %o : {a}
        }} }}'''
        return self.run("conv1d_silu", source, [(x, "bf16"), (weight, "bf16")], [(x.shape, "bf16")])[0]

    def delta(self, q, k, v, decay, beta, state, config):
        work = q.shape[0] * config.num_value_heads * (config.key_head_dim // 32) * (config.value_head_dim // 32)
        if work > 64:
            # The recurrent dependency remains in the transferred FP32 state.
            # Tile time/head domains before lowering to respect finite iMEM;
            # never compute a state update in the host orchestrator.
            h, dk, dv = config.num_value_heads, config.key_head_dim, config.value_head_dim
            qh, kh, vh = q.reshape(-1, h, dk), k.reshape(-1, h, dk), v.reshape(-1, h, dv)
            current = state.reshape(h, dk, dv).copy()
            result = np.empty_like(vh)
            single = replace(config, num_key_heads=1, num_value_heads=1)
            time_tile = max(1, 64 // ((dk // 32) * (dv // 32)))
            def execute_head(head):
                worker = CompiledStages(self.compiler, self.runner,
                    self.directory / f"delta_head{head:02d}")
                head_result = np.empty_like(vh[:,head])
                head_state = current[head].copy()
                for token in range(0, q.shape[0], time_tile):
                    stop = min(token + time_tile, q.shape[0])
                    out, updated = worker.delta(qh[token:stop, head], kh[token:stop, head],
                        vh[token:stop, head], decay[token:stop, head:head + 1],
                        beta[token:stop, head:head + 1], head_state, single)
                    head_state = updated
                    head_result[token:stop] = out
                return head,head_result,head_state,worker.stages
            # Independent heads are simulated in separate CModel instances;
            # recurrence within each head stays serial. This reduces test
            # wall time and is not a hardware parallelism/performance claim.
            with ThreadPoolExecutor(max_workers=min(4,h)) as executor:
                for head,values,updated,programs in executor.map(execute_head,range(h)):
                    result[:,head] = values
                    current[head] = updated
                    self.stages.extend(dict(s,head=head) for s in programs)
            return [result.reshape(v.shape), current.reshape(state.shape)]
        qtype, vtype = self.type(q.shape, "bf16"), self.type(v.shape, "bf16")
        stype, dtype = self.type(state.shape, "f32"), self.type(decay.shape, "f32")
        source = f'''module {{ func.func @stage(%q: {qtype}, %k: {qtype}, %v: {vtype},
            %d: {dtype}, %b: {dtype}, %s: {stype}) -> ({vtype}, {stype}) {{
          %o, %s1 = ftlpu.kernel.gated_delta_rule %q, %k, %v, %d, %b, %s {{
             heads = {config.num_value_heads} : i64, key_dim = {config.key_head_dim} : i64,
             value_dim = {config.value_head_dim} : i64}} :
             ({qtype}, {qtype}, {vtype}, {dtype}, {dtype}, {stype}) -> ({vtype}, {stype})
          return %o, %s1 : {vtype}, {stype}
        }} }}'''
        return self.run("delta_rule", source,
                        [(q, "bf16"), (k, "bf16"), (v, "bf16"), (decay, "f32"), (beta, "f32"), (state, "f32")],
                        [(v.shape, "bf16"), (state.shape, "f32")])

    def layer(self, x, weights: Qwen35DeltaWeights, packed, scales, c: Qwen35DeltaConfig):
        c.validate()
        weights.validate(c)
        if x.shape[0] % 32 or c.hidden_size % 32 or c.intermediate_size % 32:
            raise ValueError("compiled prefill requires tile-aligned tokens/hidden/intermediate")
        stages = {}
        project = lambda name, value: self.projection(name, value, packed[name], scales[name])
        n = self.norm("input_norm", x, bf16(1 + weights.input_norm), c.rms_norm_eps)
        qkv = project("in_proj_qkv", n)
        mixed = self.conv(qkv, weights.conv1d)
        q, k, v = np.split(mixed, [c.key_width, 2 * c.key_width], axis=1)
        repeat = c.num_value_heads // c.num_key_heads
        q = np.repeat(q.reshape(-1, c.num_key_heads, c.key_head_dim), repeat, axis=1).reshape(-1, c.key_head_dim)
        k = np.repeat(k.reshape(-1, c.num_key_heads, c.key_head_dim), repeat, axis=1).reshape(-1, c.key_head_dim)
        q = self.norm("query_l2", q, np.ones(c.key_head_dim, np.float32), 1e-6,
                      mean=False, scale=1 / np.sqrt(c.key_head_dim)).reshape(x.shape[0], -1)
        k = self.norm("key_l2", k, np.ones(c.key_head_dim, np.float32), 1e-6, mean=False).reshape(x.shape[0], -1)
        z = project("in_proj_z", n)
        b, a = project("in_proj_b", n), project("in_proj_a", n)
        if a.size <= 4096:
            graph = VectorGraph(self, "delta_controls", [(b, "bf16"), (a, "bf16"),
                (weights.dt_bias.reshape(1, -1), "f32"), (weights.a_log.reshape(1, -1), "f32")])
            beta_value = graph.sigmoid(0)
            biased = graph.add(1, 2, kind="add")
            softplus = graph.softplus(biased)
            exponent = graph.add(3, kind="exp", immediate=0)
            negexp = graph.add(exponent, kind="negate", immediate=0)
            log_value = graph.add(softplus, negexp)
            decay_value = graph.add(log_value, kind="exp", immediate=0)
            beta, logdecay, decay = graph.finish((beta_value, log_value, decay_value))
        else:
            beta = self.sigmoid("beta", b)
            biased = self.alu("decay_bias", a, np.broadcast_to(weights.dt_bias, a.shape), kind="add", lhs_dtype="bf16")
            softplus = self.softplus("softplus", biased)
            exponent = self.alu("a_log_exp", weights.a_log.reshape(1, -1), kind="exp", immediate=0)
            negexp = self.alu("a_log_neg", exponent, kind="negate", immediate=0)
            logdecay = self.alu("decay_log", softplus, np.broadcast_to(negexp, softplus.shape))
            decay = self.alu("decay", logdecay, kind="exp", immediate=0)
        state = np.zeros((c.num_value_heads * c.key_head_dim, c.value_head_dim), np.float32)
        core, final_state = self.delta(q, k, v, decay, beta, state, c)
        if core.size <= 4096:
            graph = VectorGraph(self, "delta_norm_gate", [(core.reshape(-1, c.value_head_dim), "bf16"),
                (weights.delta_norm.reshape(1, -1), "f32"), (z.reshape(-1, c.value_head_dim), "bf16")])
            normalized = graph.norm(0, 1, c.rms_norm_eps)
            gate_value = graph.add(2, graph.sigmoid(2), dtype="bf16")
            gated = graph.finish(graph.add(normalized, gate_value, dtype="bf16")).reshape(core.shape)
        else:
            normalized = self.norm("delta_norm", core.reshape(-1, c.value_head_dim), weights.delta_norm, c.rms_norm_eps)
            gated = self.alu("delta_gated", normalized.reshape(core.shape), self.silu("z_silu", z),
                             lhs_dtype="bf16", rhs_dtype="bf16", dtype="bf16")
        attention = project("out_proj", gated)
        residual = self.alu("residual0", x, attention, kind="add", lhs_dtype="bf16", rhs_dtype="bf16", dtype="bf16")
        ffn_in = self.norm("post_attention_norm", residual, bf16(1 + weights.post_attention_norm), c.rms_norm_eps)
        gate, up = project("gate_proj", ffn_in), project("up_proj", ffn_in)
        if gate.size <= 4096:
            graph = VectorGraph(self, "swiglu", [(gate, "bf16"), (up, "bf16")])
            activated = graph.add(0, graph.sigmoid(0), dtype="bf16")
            hidden = graph.finish(graph.add(activated, 1, dtype="bf16"))
        else:
            hidden = self.alu("swiglu", self.silu("gate_silu", gate), up,
                              lhs_dtype="bf16", rhs_dtype="bf16", dtype="bf16")
        down = project("down_proj", hidden)
        output = self.alu("output", residual, down, kind="add", lhs_dtype="bf16", rhs_dtype="bf16", dtype="bf16")
        stages.update(input_norm=n, qkv_projected=qkv, conv_qkv=mixed, beta=beta,
                      decay_log=logdecay, delta_core=core.reshape(x.shape[0], c.num_value_heads, c.value_head_dim),
                      delta_gated=gated, delta_output=attention, residual0=residual,
                      post_attention_norm=ffn_in, swiglu=hidden, down=down, output=output,
                      conv_state=qkv[-c.conv_kernel_size:].T.copy(),
                      recurrent_state=final_state.reshape(c.num_value_heads, c.key_head_dim, c.value_head_dim))
        (self.directory / "stages.json").write_text(json.dumps(self.stages, indent=2) + "\n")
        return output, stages


class VectorGraph:
    """Builds Kernel SSA; intermediate values are never evaluated by Python."""
    def __init__(self, stages, name, inputs):
        self.stages, self.name, self.inputs = stages, name, inputs
        self.values = [(tuple(value.shape), dtype) for value, dtype in inputs]
        self.lines = []

    def add(self, lhs, rhs=None, kind="multiply", immediate=None, dtype="f32", reduce=False):
        rhs = lhs if rhs is None else rhs
        shape, _ = self.values[lhs]
        outshape = (shape[0], 1) if reduce else shape
        result = len(self.values)
        types = self.stages.type
        a = types(*self.values[lhs])
        b = types(*self.values[rhs])
        o = types(outshape, dtype)
        self.lines.append(f'%v{result} = ftlpu.kernel.vector_alu %v{lhs}, %v{rhs} {{kind = "{kind}", '
            f'use_immediate = {str(immediate is not None).lower()}, immediate = {float(immediate or 0):.9e} : f32, '
            f'reduce = {str(reduce).lower()}}} : ({a}, {b}) -> {o}')
        self.values.append((outshape, dtype))
        return result

    def sigmoid(self, value):
        neg = self.add(value, kind="negate", immediate=0)
        exp = self.add(neg, kind="exp", immediate=0)
        one_plus = self.add(exp, kind="add", immediate=1)
        return self.add(one_plus, kind="reciprocal", immediate=0)

    def norm(self, value, gamma, epsilon, mean=True, scale=1):
        squared = self.add(value, value, reduce=True)
        if mean:
            squared = self.add(squared, immediate=1 / self.values[value][0][1])
        eps = self.add(squared, kind="add", immediate=epsilon)
        inverse = self.add(eps, kind="rsqrt", immediate=0)
        if scale != 1:
            inverse = self.add(inverse, immediate=scale)
        normalized = self.add(value, inverse)
        return self.add(normalized, gamma, dtype="bf16")

    def softplus(self, value):
        neg = self.add(value, kind="negate", immediate=0)
        absolute = self.add(value, neg, kind="max")
        negative = self.add(absolute, kind="negate", immediate=0)
        exp = self.add(negative, kind="exp", immediate=0)
        denominator = self.add(exp, kind="add", immediate=2)
        inverse = self.add(denominator, kind="reciprocal", immediate=0)
        t = self.add(exp, inverse)
        square = self.add(t, t)
        polynomial = self.add(square, immediate=1 / 11)
        polynomial = self.add(polynomial, kind="add", immediate=1 / 9)
        for degree in (7, 5, 3, 1):
            polynomial = self.add(polynomial, square)
            polynomial = self.add(polynomial, kind="add", immediate=1 / degree)
        correction = self.add(polynomial, t)
        correction = self.add(correction, immediate=2)
        positive = self.add(value, kind="max", immediate=0)
        return self.add(positive, correction, kind="add")

    def finish(self, result):
        outputs = result if isinstance(result, tuple) else (result,)
        types = self.stages.type
        arguments = ", ".join(f"%v{i}: {types(*item)}" for i, item in enumerate(self.values[:len(self.inputs)]))
        result_types = ", ".join(types(*self.values[i]) for i in outputs)
        signature = result_types if len(outputs) == 1 else f"({result_types})"
        body = "\n    ".join(self.lines)
        returned = ", ".join(f"%v{i}" for i in outputs)
        source = f"module {{ func.func @stage({arguments}) -> {signature} {{\n    {body}\n    return {returned} : {result_types}\n}} }}"
        values = self.stages.run(self.name, source, self.inputs, [self.values[i] for i in outputs])
        return values if isinstance(result, tuple) else values[0]


def load_imported_layer(directory: Path):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("layer_type") != "linear_attention":
        raise ValueError("compiled DeltaNet path requires a linear_attention layer")
    c = Qwen35DeltaConfig.from_hf(metadata)
    shapes = dict(in_proj_qkv=(c.conv_width, c.hidden_size),
        in_proj_z=(c.value_width, c.hidden_size), in_proj_b=(c.num_value_heads, c.hidden_size),
        in_proj_a=(c.num_value_heads, c.hidden_size), out_proj=(c.hidden_size, c.value_width),
        gate_proj=(c.intermediate_size, c.hidden_size), up_proj=(c.intermediate_size, c.hidden_size),
        down_proj=(c.hidden_size, c.intermediate_size))
    # Import files use native Matmul [in, out] order; this harness accepts
    # Linear [out, in]. A reshape without the transpose silently scrambles
    # rectangular real checkpoint projections.
    packed = {name: np.fromfile(directory / f"{name}.i8.bin", "i1").reshape(shape[1], shape[0]).T.copy()
              for name, shape in shapes.items()}
    scales = metadata["scales"]
    values = {name: packed[name].astype(np.float32) * np.float32(scales[name]) for name in shapes}
    def read_bf16(name, shape):
        return (np.fromfile(directory / f"{name}.bf16.bin", "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
    values.update(conv1d=read_bf16("conv1d", (c.conv_width, c.conv_kernel_size)),
                  delta_norm=read_bf16("delta_norm", (c.value_head_dim,)),
                  input_norm=read_bf16("input_norm", (c.hidden_size,)),
                  post_attention_norm=read_bf16("post_attention_norm", (c.hidden_size,)),
                  dt_bias=np.fromfile(directory / "dt_bias.f32.bin", "<f4"),
                  a_log=np.fromfile(directory / "a_log.f32.bin", "<f4"))
    x = read_bf16("input", (metadata["seq_len"], c.hidden_size))
    return x, Qwen35DeltaWeights(**values), packed, scales, c


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stages = CompiledStages(args.compile.resolve(), args.runner.resolve(), args.output_dir.resolve())
    output, tensors = stages.layer(*load_imported_layer(args.package))
    output.astype("<f4").tofile(args.output_dir / "output.f32.bin")
    tensors["recurrent_state"].astype("<f4").tofile(args.output_dir / "state.f32.bin")
    tensors["conv_state"].astype("<f4").tofile(args.output_dir / "conv_state.f32.bin")
    summary = dict(status="executed", numerical_reference_checked=False,
                   execution="serial_compiled_stages_with_host_layout_transfers", stage_count=len(stages.stages))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
