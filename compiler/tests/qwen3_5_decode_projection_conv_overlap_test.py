"""Real-weight batch32 decode projection/Conv frontier (one token per request).

The projection keeps M=32. Every consumed row has its own three previous
token vectors; batch rows are never treated as causal neighbours. No GDN
state recurrence, output projection or FFN is included in this fixture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

from activation_batch_matmul_test import LogicalGraph, compile_graph, run
from projection_solve_overlap_test import trace_metrics
from qwen3_5_compiled_layer import bf16, load_imported_layer
from qwen3_5_decode_gdn_attention_test import initial_state
from qwen3_5_qkv_conv_prefill_test import compare
from qwen3_5_reference import causal_depthwise_conv1d
from tensor_geometry import Value
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from measure_compute_overlap import measure


def make_graph(args):
    x, weights, packed, scales, c = load_imported_layer(args.package)
    channels = args.channels
    if channels <= 0 or channels > c.conv_width or channels % c.key_head_dim:
        raise ValueError('channels must be a positive whole-head slice of checkpoint QKV')
    x = x[np.arange(32) % len(x)]
    graph = LogicalGraph()
    rhs = packed['in_proj_qkv'][:channels]
    projected = graph.project('in_proj_qkv', x, rhs, scales['in_proj_qkv'])
    refs = [x @ rhs.astype('f4').T * np.float32(scales['in_proj_qkv'])]
    outputs = [projected]
    taps = weights.conv1d[:channels]
    for request in range(args.consumers):
        cache = bf16(initial_state(request, c).conv[:channels])
        history = cache.T.copy()
        ty = graph.type(history.shape, 'bf16')
        source = f'''module {{ func.func @stage(%x: {ty}) -> {ty} {{
          %y = ftlpu.kernel.reshape %x : ({ty}) -> {ty}
          return %y : {ty}
        }} }}'''
        cached = graph.run('conv_cache_view', source, [(history, 'bf16')], [(history.shape, 'bf16')])[0]
        row = projected[request:request+1]
        window = Value.concatenate([cached[-3:], row])
        outputs.append(graph.conv(window, taps)[-1:])
        mixed, _ = causal_depthwise_conv1d(refs[0][request:request+1], taps, cache)
        refs.append(mixed)
    return graph, outputs, refs


def execute(args, mode):
    out = args.output_dir.resolve() / mode
    graph, outputs, refs = make_graph(args)
    suffix = '.exe' if sys.platform == 'win32' else ''
    def tool_hashes():
        return {name: hashlib.sha256((args.build_dir/area/(name + suffix)).read_bytes()).hexdigest()
                for area, name in [('compiler','ftlpu-graph-compile'), ('compiler','ftlpu-compile'),
                                   ('runtime','compiled_stage_file_runner')]}
    tools_before = tool_hashes()
    attrs = ['ftlpu.l2_head_partition = true',
             f'ftlpu.projection_conv_overlap = {str(mode == "on").lower()}']
    if args.page_rows:
        attrs.append(f'ftlpu.weight_page_rows = {args.page_rows} : i64')
    exe, env, binary = compile_graph(graph, outputs, out, args.build_dir.resolve(),
                                     args.target_config.resolve(), fp32_spills=True, module_attrs=attrs)
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out/'inputs.txt',
         out/'actual', out/'runtime.pipeline.csv'], out/'execution.log', env)
    log = (out/'execution.log').read_text().splitlines()[-1]
    counters = {key: int(val) for key, val in re.findall(r'(\w+)=(\d+)', log)}
    assert counters['host_operations'] == 0, log
    errors, digests = [], []
    for i, ref in enumerate(refs):
        path = out/f'actual.{i}.bin'
        actual = (np.fromfile(path, '<u2').astype('u4') << 16).view('f4').reshape(ref.shape)
        error = compare(actual, ref)
        assert error['nrmse'] < .03 and error['cosine'] > .999, (i, error)
        errors.append(error)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    run([exe('runtime', 'ftlpu_icu_program_export'), out/'actual.runtime.ftlpu',
         out/'runtime_icu_programs', '--pre-execution-dir', out/'actual.pre_execution'], out/'export.log', env)
    trace = out/'runtime.pipeline.csv'
    metrics = trace_metrics(trace)
    metrics['fu_activity'] = measure(trace)['total']
    assert tool_hashes() == tools_before, 'Build artifacts changed during this run'
    record = dict(numerics='passed', mode=mode, batch=32, consumers=args.consumers,
                  channels=args.channels, page_rows=args.page_rows, errors=errors,
                  output_hashes=digests, runtime=metrics, counters=counters, tool_sha256=tools_before)
    (out/'result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({k:record[k] for k in ('mode','consumers','channels','runtime','counters')}), flush=True)
    return record


def compare_runs(off, on):
    assert off['tool_sha256'] == on['tool_sha256'], 'A/B used different builds'
    assert on['output_hashes'] == off['output_hashes'], 'Overlap changed numerical results'
    assert on['runtime']['fu_activity']['simultaneous'] > off['runtime']['fu_activity']['simultaneous'], 'No effective additional compute overlap'
    assert on['runtime']['fu_activity']['cycles'] < off['runtime']['fu_activity']['cycles'], 'No full runtime gain including readback'
    assert on['runtime']['cold_cycles_through_compute'] < off['runtime']['cold_cycles_through_compute'], 'No cold latency gain'
    assert on['counters']['weight_page_prefetch_bytes'] <= off['counters']['weight_page_prefetch_bytes'], 'Repeated projection weights'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('package','build-dir','target-config','output-dir'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--channels', type=int, default=512)
    p.add_argument('--page-rows', type=int, default=512,
                   help='Page-row budget for the shape sweep; pass 0 for compiler default')
    p.add_argument('--consumers', type=int, default=32, choices=range(1,33))
    p.add_argument('--mode', choices=('off','on','both'), default='both')
    p.add_argument('--on-first', action='store_true',
                   help='Verify an overlap candidate before running its serial control')
    args = p.parse_args()
    if args.page_rows < 0:
        p.error('--page-rows must be nonnegative')
    if args.mode != 'both':
        execute(args, args.mode)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'summary.json'
    path.write_text(json.dumps(dict(status='running')))
    result = dict(scope='INT8 batch32 QKV channel slice + independent BF16 decode Conv/SiLU; no state recurrence or FFN',
                  channels=args.channels, consumers=args.consumers, page_rows=args.page_rows,
                  target=json.loads(args.target_config.read_text()))
    try:
        for mode in (('on','off') if args.on_first else ('off','on')):
            result[mode] = execute(args, mode)
        compare_runs(result['off'], result['on'])
    except Exception as error:
        path.write_text(json.dumps(dict(result, status='failed', reason=str(error)), indent=2))
        raise
    path.write_text(json.dumps(dict(result, status='passed'),indent=2))


if __name__ == '__main__':
    main()
