"""Real-weight full GDN decode: one token per request and DDR state writeback.

Batching is a serialized ModelSession baseline, not a batched projection GEMM.
Python only supplies logical tensors, serializes state, and evaluates references.
The same C++-compiled raw Macro executable performs every attention operation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import bf16, load_imported_layer
from qwen3_5_reference import gated_deltanet, zero_centered_rms_norm, Qwen35DeltaState
from qwen3_5_qkv_conv_prefill_test import compare


def run(command, log):
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{log}:\n{log.read_text(encoding="utf-8", errors="replace")[-3000:]}')


def initial_state(request, c):
    rng = np.random.default_rng(35261027 + request)
    return Qwen35DeltaState(rng.normal(0, .02, (c.conv_width, c.conv_kernel_size)).astype('f4'),
        rng.normal(0, .02, (c.num_value_heads, c.key_head_dim, c.value_head_dim)).astype('f4'))


def read_outputs(folder, c):
    def bf(path):
        return (np.fromfile(path, '<u2').astype('u4') << 16).view('f4')
    return (bf(folder / 'actual.0.bin').reshape(1, c.hidden_size),
        Qwen35DeltaState(bf(folder / 'actual.2.bin').reshape(c.conv_kernel_size, c.conv_width).T,
            np.fromfile(folder / 'actual.1.bin', '<f4').reshape(c.num_value_heads, c.key_head_dim, c.value_head_dim)))


def errors(actual, expected):
    return {name: compare(x, y) for name, x, y in zip(('attention', 'state', 'conv_state'),
        (actual[0], actual[1].recurrent, actual[1].conv),
        (expected[0], expected[1].recurrent, expected[1].conv))}


def trace_stats(path, state_index=None, conv_index=None):
    # Union of occupied intervals, not peak FLOPS or physical lane utilization.
    intervals = {k: [] for k in ('MXM', 'VXM', 'C2C', 'compute',
        'weight_transfer', 'state_ingress', 'conv_ingress', 'output_egress', 'page_wait')}
    end = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            start, stop = int(row['start']), int(row['end'])
            ni, no = int(row['inner_count']), int(row['outer_count'])
            di, do = int(row['inner_interval']), int(row['outer_interval'])
            end = max(end, stop + (ni - 1)*di + (no - 1)*do)
            kind = row['resource'].split('.')[0]
            labels = [kind] if kind in ('MXM','VXM','C2C') else []
            if kind in ('MXM','VXM'): labels.append('compute')
            if row['resource'].endswith('.Prefetch'): labels.append('weight_transfer')
            if row['resource'] == 'C2C.HostInput':
                match = re.search(r'\bbinding=(\d+)',row['detail'])
                index = int(match[1]) if match else None
                if index == state_index: labels.append('state_ingress')
                if index == conv_index: labels.append('conv_ingress')
            if row['resource'] == 'C2C.HostOutput': labels.append('output_egress')
            if row['resource'] == 'ICU.PageReadyWait': labels.append('page_wait')
            if not labels: continue
            for outer in range(no):
                for inner in range(ni):
                    if int(row.get('skip_first', 0)) and outer == 0 and inner == 0: continue
                    offset = outer*do + inner*di
                    for label in labels: intervals[label].append((start+offset, stop+offset))
    def merged(items):
        result = []
        for lo, hi in sorted(items):
            if result and lo <= result[-1][1]: result[-1] = (result[-1][0],max(hi,result[-1][1]))
            else: result.append((lo,hi))
        return result
    intervals = {k:merged(v) for k,v in intervals.items()}
    def overlap(a,b):
        i=j=total=0
        while i<len(a) and j<len(b):
            total += max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
            if a[i][1] <= b[j][1]: i+=1
            else: j+=1
        return total
    return dict(runtime_trace_cycles=end,
        activity_union_cycles={k:sum(hi-lo for lo,hi in v) for k,v in intervals.items()},
        transfer_compute_overlap_cycles={k:overlap(intervals[k],intervals['compute']) for k in
            ('weight_transfer','state_ingress','conv_ingress','output_egress')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'build-dir', 'target-config', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[1,32,64])
    p.add_argument('--steps', type=int, default=1,
        help='Decode steps per request; default is one step with final state/cache written to DDR')
    p.add_argument('--resume', action='store_true', help='Keep successfully checked steps of this run')
    a = p.parse_args()
    if a.steps < 1 or any(b < 1 for b in a.batches): p.error('positive batches/steps required')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    build = a.build_dir.resolve(); ext = '.exe' if sys.platform == 'win32' else ''
    executable = lambda area, name: build / area / (name+ext)
    x,w,packed,scales,c = load_imported_layer(a.package)
    target = json.loads(a.target_config.read_text())
    signature = dict(target=target, steps=a.steps,
        checkpoint=hashlib.sha256((a.package/'metadata.json').read_bytes()).hexdigest(),
        encoding='raw-macro', fixture_seed=35261027)
    manifest = out/'run_config.json'
    if a.resume and manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('resume configuration changed; use a fresh output directory')
    manifest.write_text(json.dumps(signature,indent=2))
    template = out / 'program'
    if not (a.resume and (template/'attention.ftlpu').exists()):
        run([sys.executable, Path(__file__).with_name('qwen3_5_attention_runtime_test.py'),
            '--compile', executable('compiler','ftlpu-compile'),
            '--graph-compile', executable('compiler','ftlpu-graph-compile'),
            '--runner', executable('runtime','compiled_stage_file_runner'),
            '--package', a.package.resolve(), '--target-config', a.target_config.resolve(),
            '--output-dir', template, '--decode-token', 0, '--compile-only',
            '--icu-compression', 'raw-macro'], out/'compile.log')
    # Resolve external logical bindings by compiler names; no SRAM packing or
    # instruction generation is performed here. ABI changes fail explicitly.
    schedule = (template/'decoder_layer.schedule.mlir').read_text()
    bindings = {name: int(index) for index,name in re.findall(
        r'access = "input", bytes = \d+ : i64, index = (\d+) : i64, name = "([^"]+)"', schedule)}
    state_index, x_index, conv_index = [bindings[name] for name in
        ('initial_recurrent_state', 'input_norm.input0', 'external_indexed_view')]
    paths = (template/'inputs.txt').read_text().splitlines()
    assert np.array_equal(np.fromfile(paths[conv_index], '<u2'),
        (bf16(initial_state(0,c).conv.T).view('u4') >> 16).astype('<u2').ravel()), 'Conv input ABI changed'
    records = []
    for batch in a.batches:
        ideal = [initial_state(i,c) for i in range(batch)]
        for step in range(a.steps):
            directory = out / f'batch{batch}' / f'step{step}'
            directory.mkdir(parents=True, exist_ok=True)
            record = dict(batch=batch, step=step, status='running', icu_compression='raw-macro',
                scope='input RMSNorm + complete GDN attention; no FFN/residual',
                batching='serialized independent requests in one ModelSession, shared checkpoint weights',
                state_policy='DDR/C2C recurrent state and Conv history in/out on every step',
                hardware=target, real_weights=True)
            checked = directory/'summary.json'
            reuse = a.resume and checked.exists() and json.loads(checked.read_text()).get('status') == 'passed'
            if not reuse: checked.write_text(json.dumps(record, indent=2))
            try:
                expected_local, expected_rollout = [], []
                for request in range(batch):
                    folder = directory/f'request{request}'; folder.mkdir(exist_ok=True)
                    previous = (initial_state(request,c) if step == 0 else
                        read_outputs(out/f'batch{batch}'/f'step{step-1}'/f'request{request}',c)[1])
                    token = x[(request*a.steps+step)%len(x):][:1]
                    normalized = zero_centered_rms_norm(token,w.input_norm,c.rms_norm_eps)
                    expected_local.append(gated_deltanet(normalized,w,c,previous))
                    rollout = gated_deltanet(normalized,w,c,ideal[request])
                    expected_rollout.append(rollout); ideal[request] = rollout[1]
                    if not reuse:
                        listing = list(paths)
                        for index,value,dtype in ((state_index,previous.recurrent,'f32'),
                            (x_index,token,'bf16'),(conv_index,previous.conv.T,'bf16')):
                            file = folder/f'input{index}.bin'
                            data = (bf16(value).view('u4')>>16).astype('<u2') if dtype=='bf16' else value.astype('<f4')
                            data.tofile(file); listing[index] = str(file)
                        (folder/'inputs.txt').write_text('\n'.join(listing)+'\n')
                print(f'full GDN raw Macro batch={batch} step={step}',flush=True)
                if not reuse:
                    run([executable('runtime','compiled_batch_file_runner'), template/'attention.ftlpu',
                        directory,batch],directory/'execution.log')
                all_errors = []
                for request in range(batch):
                    actual = read_outputs(directory/f'request{request}',c)
                    item = dict(request=request, local=errors(actual,expected_local[request]),
                        rollout=errors(actual,expected_rollout[request]))
                    all_errors.append(item)
                    for mode in ('local','rollout'):
                        assert all(e['nrmse'] < .03 and e['cosine'] > .999 for e in item[mode].values()), item
                stats = json.loads((directory/'execution.json').read_text())
                assert stats['host_operations'] == 0 and stats['requests'] == batch, stats
                assert stats['c2c_ingress_bytes'] >= batch*c.num_value_heads*c.key_head_dim*c.value_head_dim*4
                assert stats['c2c_egress_bytes'] >= batch*(c.num_value_heads*c.key_head_dim*c.value_head_dim*4+c.conv_width*c.conv_kernel_size*2)
                performance = trace_stats(directory/'runtime.pipeline.csv',state_index,conv_index)
                cycles = performance['runtime_trace_cycles']; clock = target['external_memory']['lpu_clock_mhz']
                if not reuse:
                    run([executable('runtime','ftlpu_icu_program_export'),directory/'actual.runtime.ftlpu',
                        directory/'runtime_icu_programs','--pre-execution-dir',directory/'actual.pre_execution'],directory/'export.log')
                record.update(status='passed', errors=all_errors, execution_stats=stats, **performance,
                    latency_ms=cycles/(clock*1000),requests_per_second=batch*clock*1e6/cycles,
                    limits=dict(nrmse=.03,cosine=.999),
                    utilization_definition='union of trace resource active intervals / end-to-end cycles; not peak FLOPS')
            except Exception as error:
                record.update(status='failed',error=str(error)); raise
            finally:
                checked.write_text(json.dumps(record,indent=2)+'\n')
                records.append(record)
                (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    # Same request prefixes must remain independent of the number of neighbors.
    for small,large in zip(sorted(a.batches),sorted(a.batches)[1:]):
        for step in range(a.steps):
            for request in range(small):
                for output in range(3):
                    tail=f'step{step}/request{request}/actual.{output}.bin'
                    assert (out/f'batch{small}'/tail).read_bytes() == (out/f'batch{large}'/tail).read_bytes(), 'batch isolation mismatch'
    (out/'batch_consistency.json').write_text(json.dumps(dict(status='passed',batches=a.batches,steps=a.steps)))


if __name__ == '__main__': main()
