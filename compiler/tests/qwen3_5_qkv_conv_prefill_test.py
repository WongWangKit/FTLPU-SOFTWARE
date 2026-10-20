#!/usr/bin/env python3
"""Run real-size Qwen3.5 QKV/Conv, optionally with Q/K L2 norms and beta/g."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import bf16, load_imported_layer
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_reference import causal_depthwise_conv1d


def compare(actual, expected):
    a = actual.astype(np.float64).ravel()
    e = expected.astype(np.float64).ravel()
    if a.shape != e.shape or not np.isfinite(a).all():
        raise AssertionError('invalid QKV/Conv output')
    return dict(nrmse=float(np.linalg.norm(a-e) / max(np.linalg.norm(e), 1e-30)),
                cosine=float(np.dot(a, e) / max(np.linalg.norm(a)*np.linalg.norm(e), 1e-30)),
                max_abs=float(np.max(np.abs(a-e))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'runner', 'package', 'output-dir', 'target-config'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--graph-compile', type=Path)
    p.add_argument('--with-qk-norm', action='store_true')
    p.add_argument('--with-gates', action='store_true',
                   help='include Q/K L2Norm and the real beta/log-decay branches')
    p.add_argument('--compile-only', action='store_true')
    a = p.parse_args()
    a.with_qk_norm |= a.with_gates
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').unlink(missing_ok=True)

    x, weights, packed, scales, config = load_imported_layer(a.package)
    metadata = json.loads((a.package / 'metadata.json').read_text(encoding='utf-8'))
    provenance = metadata.get('checkpoint_provenance', {})
    if (provenance.get('repo') != 'Qwen/Qwen3.5-0.8B' or
            x.shape != (32, 1024) or config.conv_width != 6144):
        raise ValueError('requires the real seq32 Qwen3.5-0.8B layer-0 package')

    # The test begins at the QKV projection's input. This is a host-prepared
    # fixture, not an RMSNorm operator in the graph or ICU program.
    gamma = bf16(1 + weights.input_norm)
    normalized = bf16(x / np.sqrt(np.mean(x*x, axis=1, keepdims=True) +
                                  np.float32(config.rms_norm_eps)) * gamma)
    graph = LogicalGraph()
    qkv = graph.project('in_proj_qkv', normalized, packed['in_proj_qkv'],
                        scales['in_proj_qkv'])
    conv = graph.conv(qkv, weights.conv1d)
    if a.with_qk_norm:
        key_width = config.key_width
        heads = config.num_value_heads
        repeats = heads // config.num_key_heads
        query = conv[:, :key_width].reshape(-1, config.num_key_heads,
                                             config.key_head_dim).repeat(repeats, 1).reshape(
                                                 -1, config.key_head_dim)
        key = conv[:, key_width:2*key_width].reshape(-1, config.num_key_heads,
                                                     config.key_head_dim).repeat(repeats, 1).reshape(
                                                         -1, config.key_head_dim)
        query = graph.norm('query_l2', query, np.ones(config.key_head_dim, np.float32),
                           1e-6, False, 1/np.sqrt(config.key_head_dim)).reshape(32, -1)
        key = graph.norm('key_l2', key, np.ones(config.key_head_dim, np.float32),
                         1e-6, False).reshape(32, -1)
        finals = [query, key]
    else:
        finals = [conv]
    if a.with_gates:
        beta, log_decay = graph.gates(normalized, packed, scales, weights.dt_bias, weights.a_log)
        finals += [beta, log_decay]
    logical = graph.write(out / 'logical_graph', finals)
    if a.with_qk_norm:
        norms = [n for n in graph.nodes if n['name'] in ('query_l2', 'key_l2')]
        if len(norms) != 2 or any('ftlpu.kernel.l2_norm' not in n['source']
                                 or 'rms_norm' in n['source'] or len(n['inputs']) != 1
                                 for n in norms):
            raise AssertionError('Q/K must use native L2Norm without an RMSNorm/gamma rewrite')

    kernel = out / 'qkv_conv.kernel.mlir'
    binary = out / 'qkv_conv.ftlpu'
    prefix = out / 'qkv_conv'
    graph_compiler = a.graph_compile or a.compile.resolve().with_name(
        'ftlpu-graph-compile.exe' if sys.platform == 'win32' else 'ftlpu-graph-compile')
    subprocess.run([str(graph_compiler.resolve()), '--input', str(logical.resolve()),
                    '--output', str(kernel), '--emit-kernel', '--target-config',
                    str(a.target_config.resolve())], check=True)
    subprocess.run([str(a.compile.resolve()), '--input', str(kernel),
                    '--input-stage', 'kernel', '--output', str(binary),
                    '--mxm-execution', 'vector', '--target-config',
                    str(a.target_config.resolve())], check=True)
    if a.with_qk_norm:
        compiled_kernel = kernel.read_text(encoding='utf-8')
        schedule = (out / 'decoder_layer.schedule.mlir').read_text(encoding='utf-8')
        if (compiled_kernel.count('ftlpu.kernel.l2_norm') != 2
                or 'ftlpu.kernel.rms_norm' in compiled_kernel
                or 'l2norm.feedback' not in schedule or 'rmsnorm.feedback' in schedule):
            raise AssertionError('compiled Q/K normalization must remain native L2Norm')
        if 'l2norm.transpose_native_group' not in schedule:
            raise AssertionError('L2Norm must transpose directly from native Conv groups')
        outputs = [line for line in schedule.splitlines()
                   if 'ftlpu.schedule.binding' in line and 'access = "output"' in line]
        if len(outputs) != len(finals) or any('kind = "fp16_mxm_distributed_16"' not in line
                                    or 'tensor<32x2048xbf16>' not in line for line in outputs[:2]):
            raise AssertionError('read back native distributed outputs without a planar copy')
        if a.with_gates and (compiled_kernel.count('kind = "sigmoid"') != 1 or
                            compiled_kernel.count('kind = "softplus"') != 1):
            raise AssertionError('gate import must retain semantic sigmoid/softplus')

    plan = json.loads((out / 'graph_plan.json').read_text(encoding='utf-8'))
    if a.with_gates:
        gate_nodes = [node for node in plan['nodes'] if node['name'].startswith('delta_gates.tokens')]
        producers = [node for node in plan['nodes']
                     if node['name'].startswith(('in_proj_a.', 'in_proj_b.'))]
        if len(producers) != 2 or any(
                layout['kind'] != 'token_partition_planar' or layout['hemisphere'] != 'both'
                for node in producers for layout in node['direct_output_layouts']):
            raise AssertionError('a/b projection must directly write the two token partitions')
        if any('kind = "token_partition_planar"' not in line for line in outputs[2:]):
            raise AssertionError('beta/g readback must use the native token partitions without merging')
        if (len(gate_nodes) != 2 or gate_nodes[0]['nominal_start'] != gate_nodes[1]['nominal_start']
                or {layout['hemisphere'] for node in gate_nodes
                    for layout in node['direct_output_layouts']} != {'east', 'west'}):
            raise AssertionError('gate token halves must run concurrently on opposite hemispheres')
        gate_start = min(node['nominal_start'] for node in gate_nodes)
        if any(layout['kind'] != 'fp32_quad_planar' or layout['rows'] != 16
               for node in gate_nodes for layout in node['direct_output_layouts']):
            raise AssertionError('beta/g must keep packed words with 16 valid head lanes')
        norm_end = max(node['nominal_end'] for node in plan['nodes']
                       if node['name'] in ('query_l2', 'key_l2'))
        if any(int(re.search(r'cycle = (\d+)', line).group(1)) >= norm_end
               for line in schedule.splitlines() if 'ftlpu.schedule.sxm ' in line):
            raise AssertionError('gate branches must not introduce SXM layout conversion')
        gate_chains = {}
        for line in schedule.splitlines():
            if 'ftlpu.schedule.vxm ' not in line:
                continue
            cycle = int(re.search(r'cycle = (\d+)', line).group(1))
            if cycle < gate_start:
                continue
            queue = int(re.search(r'queue = (\d+)', line).group(1))
            opcode = re.search(r'opcode = "([^"]+)"', line).group(1)
            gate_chains.setdefault(cycle, {})[queue] = opcode
        beta_chain = gate_chains[min(gate_chains)]
        if any([beta_chain.get(column) for column in columns] != [
                'negate', 'exp', 'add', 'reciprocal', 'cast']
               for columns in (range(5), range(15, 10, -1))):
            raise AssertionError('beta must lower to a continuous four-operation ALU chain')
        if not any('multiply' in chain.values() and 'add' in chain.values()
                   for chain in gate_chains.values()):
            raise AssertionError('softplus polynomial multiply/add must share an ALU chain')
    names = [node['name'] for node in plan['nodes']]
    allowed = ('in_proj_qkv.', 'conv1d_silu.', 'conv_layout_group')
    if a.with_qk_norm:
        allowed += ('query_l2.', 'key_l2.')
    if a.with_gates:
        allowed += ('in_proj_b.', 'in_proj_a.', 'delta_gates')
    has_node = lambda prefix: any(name == prefix or name.startswith(prefix + '.') for name in names)
    if (not any(name.startswith('in_proj_qkv.') for name in names) or
            not any(name.startswith('conv1d_silu.') for name in names) or
            (a.with_qk_norm and
             (not has_node('query_l2') or not has_node('key_l2'))) or
            any(not (name in ('query_l2', 'key_l2') or name.startswith(allowed))
                for name in names)):
        raise AssertionError(f'graph contains unexpected operators: {names}')
    if a.with_qk_norm:
        conv_blocks = [node for node in plan['nodes']
                       if node['name'].startswith('conv1d_silu.block')]
        if (any(node['name'].startswith('conv_layout_group') for node in plan['nodes']) or
                not conv_blocks or
                any(layout['kind'] != 'fp16_mxm_distributed_16'
                    for node in conv_blocks
                    for layout in node['direct_output_layouts'])):
            raise AssertionError('Conv must write distributed16 directly before Q/K norm')
    if a.compile_only:
        print(json.dumps(dict(status='compiled', graph_nodes=names,
                              binary_bytes=binary.stat().st_size), indent=2), flush=True)
        return

    listing = out / 'inputs.txt'
    if not listing.exists():
        raise RuntimeError('compiler did not emit the graph input manifest')
    env = os.environ.copy()
    env['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
    env['FTLPU_SESSION_PROGRESS'] = '1'
    for key in ('FTLPU_SESSION_LINK_ONLY', 'FTLPU_SESSION_STOP_CYCLE'):
        env.pop(key, None)
    pipeline = out / 'runtime.pipeline.csv'
    for stale in ([out / f'qkv_conv.{i}.bin' for i in range(len(finals))] +
                  [out / 'qkv_conv.runtime.ftlpu', pipeline]):
        stale.unlink(missing_ok=True)
    with (out / 'execution.log').open('w', encoding='utf-8') as log:
        run = subprocess.run([str(a.runner.resolve()), str(binary), str(listing),
                              str(prefix), str(pipeline)], env=env, stdout=log,
                             stderr=subprocess.STDOUT)
    execution_log = (out / 'execution.log').read_text(encoding='utf-8')
    if run.returncode:
        raise RuntimeError(execution_log[-8000:])

    expected_qkv = normalized @ weights.in_proj_qkv.T
    expected_conv, _ = causal_depthwise_conv1d(
        expected_qkv, weights.conv1d, np.zeros_like(weights.conv1d))
    if a.with_qk_norm:
        q = expected_conv[:, :config.key_width].reshape(-1, config.num_key_heads,
                                                        config.key_head_dim)
        k = expected_conv[:, config.key_width:2*config.key_width].reshape(
            -1, config.num_key_heads, config.key_head_dim)
        q = np.repeat(q, config.num_value_heads // config.num_key_heads, axis=1).reshape(
            -1, config.key_head_dim)
        k = np.repeat(k, config.num_value_heads // config.num_key_heads, axis=1).reshape(
            -1, config.key_head_dim)
        expected = {
            'query_l2': (q / np.sqrt(np.sum(q*q, axis=1, keepdims=True) + 1e-6)
                         / np.sqrt(config.key_head_dim)).reshape(32, -1),
            'key_l2': (k / np.sqrt(np.sum(k*k, axis=1, keepdims=True) + 1e-6)).reshape(32, -1),
        }
    else:
        expected = {'conv1d_silu': expected_conv}
    if a.with_gates:
        b_ref = normalized.astype(np.float64) @ weights.in_proj_b.astype(np.float64).T
        a_ref = normalized.astype(np.float64) @ weights.in_proj_a.astype(np.float64).T
        expected['beta'] = 1 / (1 + np.exp(-b_ref))
        expected['g'] = -np.exp(weights.a_log.astype(np.float64)) * np.logaddexp(
            0, a_ref + weights.dt_bias.astype(np.float64))
    errors = {}
    for index, (name, reference) in enumerate(expected.items()):
        if name in ('beta', 'g'):
            actual = np.fromfile(out / f'qkv_conv.{index}.bin', '<f4').reshape(reference.shape)
        else:
            actual = (np.fromfile(out / f'qkv_conv.{index}.bin', '<u2').astype(np.uint32) << 16)
            actual = actual.view(np.float32).reshape(reference.shape)
        errors[name] = compare(actual, reference)
        if name == 'beta' and (np.any(actual < 0) or np.any(actual > 1)):
            raise AssertionError('beta must be in [0,1]')
        if name == 'g' and np.any(actual > 0):
            raise AssertionError('log-decay g must be nonpositive')
    passed = all(value['nrmse'] <= .03 and value['cosine'] >= .999
                 for value in errors.values())
    stats = [line for line in execution_log.splitlines() if line.startswith('cycles=')]
    execution_stats = {key: int(value) for key, value in
                       re.findall(r'(\w+)=(\d+)', stats[-1])} if stats else {}
    if execution_stats.get('outputs') != len(finals) or execution_stats.get('host_operations') != 0:
        raise AssertionError(f'unexpected runtime output or host operation count: {execution_stats}')
    runtime_binary = out / 'qkv_conv.runtime.ftlpu'
    if not runtime_binary.exists() or not pipeline.exists():
        raise RuntimeError('missing executed ICU program or runtime pipeline')
    exporter = a.runner.resolve().with_name(
        'ftlpu_icu_program_export.exe' if sys.platform == 'win32' else 'ftlpu_icu_program_export')
    export_dir = out / 'runtime_icu_programs'
    if exporter.exists():
        export_command = [str(exporter), str(runtime_binary), str(export_dir)]
        pre_execution = out / 'qkv_conv.pre_execution'
        if pre_execution.exists():
            export_command += ['--pre-execution-dir', str(pre_execution)]
        subprocess.run(export_command, check=True)
    summary = dict(status='passed' if passed else 'failed',
                   model=provenance['repo'], revision=provenance['revision'],
                   seq_len=32, hidden_size=1024, qkv_width=6144,
                   operators=['qkv_projection', 'causal_depthwise_conv1d_silu'] +
                             (['query_l2', 'key_l2'] if a.with_qk_norm else []) +
                             (['b_projection', 'a_projection', 'beta_sigmoid', 'g_softplus_scale']
                              if a.with_gates else []),
                   gate_outputs=(dict(beta='sigmoid(x @ W_b.T)',
                                      g='-exp(A_log) * softplus(x @ W_a.T + dt_bias)',
                                      shape=[32, config.num_value_heads], dtype='f32',
                                      negative_exp_a='constant folded checkpoint data')
                                 if a.with_gates else None),
                   rmsnorm='fixture only; not compiled or executed',
                   weights='real checkpoint layer-0, W8A16 projections and BF16 Conv',
                   graph_nodes=names, program_count=1,
                   host_intermediate_transfers=0,
                   arithmetic_backend='CModel ICU/FU',
                   execution_stats=execution_stats,
                   output_errors=errors,
                   pass_limits=dict(nrmse=.03, cosine=.999),
                   compiler_binary_bytes=binary.stat().st_size,
                   runtime_binary_sha256=hashlib.sha256(runtime_binary.read_bytes()).hexdigest(),
                   runtime_pipeline=str(pipeline),
                   runtime_icu_program_directory=str(export_dir) if exporter.exists() else None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: summary[key] for key in
                      ('status', 'operators', 'execution_stats', 'output_errors')},
                     indent=2), flush=True)
    if not passed:
        raise AssertionError('QKV/Conv numerical validation failed')


if __name__ == '__main__':
    main()
