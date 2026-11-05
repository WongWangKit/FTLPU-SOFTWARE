"""Exercise BF16 spill legalization for sigmoid and softplus gate branches."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import VectorGraph, bf16


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'graph-compile', 'runner', 'target-config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--log-softplus', action='store_true')
    p.add_argument('--token-halves', action='store_true')
    p.add_argument('--with-projection', action='store_true')
    p.add_argument('--paired-projections', action='store_true', help='Two narrow matrices with independent INT8 scales')
    p.add_argument('--fp32-ingress', action='store_true')
    p.add_argument('--decay-factor', action='store_true', help='Fuse a separate exp consumer of log-decay')
    p.add_argument('--with-recurrence', action='store_true', help='Check direct gate homes over two FP32 State steps')
    p.add_argument('--with-prefix', action='store_true', help='Check full-token chunkwise scan after partitioned gates')
    a = p.parse_args()
    if a.with_prefix and (a.decay_factor or a.with_recurrence):
        p.error('--with-prefix cannot be combined with decay-factor or recurrence')
    if a.with_recurrence and not (a.decay_factor and a.fp32_ingress):
        p.error('--with-recurrence requires --decay-factor --fp32-ingress')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='pending')))
    rng = np.random.default_rng(951)
    width=2 if a.with_recurrence else 16
    x = bf16(rng.normal(0, 1, (32, width)))
    if a.log_softplus:
        x[:8] = bf16(np.broadcast_to(np.linspace(-8, -3, 8, dtype='f4')[:, None], (8,width)).copy())
    bias = bf16(rng.normal(0, .2, (1, width)))
    if a.log_softplus:
        # Isolate Log tail accuracy from rounding the separate bias add.
        # Real checkpoint tests cover nonzero bias and its BF16 boundary.
        bias.fill(0)
    scale = bf16(-np.exp(rng.uniform(-1, 1, (1, width))).astype('f4'))
    graph = LogicalGraph()
    operand=x
    if a.with_projection or a.paired_projections:
        raw=bf16(rng.normal(0,.3,(32,64)))
        weight=rng.integers(-16,17,(width,64),dtype=np.int8)
        operand=graph.project('gate_projection',raw,weight,.01)
        x=bf16(raw @ bf16(weight.astype('f4')*.01).T)
    a_operand=operand; a_reference=x
    if a.paired_projections:
        weight_a=rng.integers(-27,28,(width,64),dtype=np.int8)
        a_operand=graph.project('other_projection',raw,weight_a,.023)
        a_reference=bf16(raw @ bf16(weight_a.astype('f4')*.023).T)
    gate_inputs=([(operand,'bf16'),(a_operand,'bf16'),(bias,'f32'),(scale,'f32')]
                 if a.paired_projections else [(operand,'bf16'),(bias,'f32'),(scale,'f32')])
    g = VectorGraph(graph, 'gates', gate_inputs)
    beta = g.add(0, kind='sigmoid', immediate=0)
    biased = g.add(1 if a.paired_projections else 0, 2 if a.paired_projections else 1, kind='add')
    softplus = g.add(biased, kind='softplus', immediate=0)
    decay = g.add(softplus, 3 if a.paired_projections else 2)
    finals = g.finish((beta, decay))
    if a.with_prefix:
        prefix = graph.vector_tiles('gate_prefix', [(finals[1], 'f32')],
            lambda v: v.add(0, kind='cumsum_rows', immediate=0))
        finals += [prefix]
    if a.decay_factor:
        factor=graph.vector_tiles('decay_factor',[(finals[1],'f32')],
            lambda v:v.add(0,kind='exp',immediate=0))
        finals=[finals[0],factor]
    if a.with_recurrence:
        c=SimpleNamespace(num_value_heads=2,num_key_heads=2,key_head_dim=128,value_head_dim=128)
        q,k,v=[bf16(rng.normal(0,.08,(2,256)).astype('f4')) for _ in range(3)]
        state=rng.normal(0,.02,(2,128,128)).astype('f4')
        rec=graph.delta(q,k,v,finals[1][:2],finals[0][:2],state.reshape(256,128),c,state_dtype='f32')
        native=rec[1].reshape(2,128,128).transpose(0,2,1).reshape(256,128)
        finals += [rec[0],native]
    logical = graph.write(out / 'graph', finals)
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    with (out / 'import.log').open('w') as log:
        subprocess.run([str(a.graph_compile.resolve()), '--input', str(logical), '--output', str(kernel),
                        '--emit-kernel', '--target-config', str(a.target_config.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
    kernel.write_text(kernel.read_text().replace('module {', 'module attributes {ftlpu.gdn_prefill_strategy = "tokenwise", ftlpu.sxm_wide = true, ftlpu.vector_fp32_ingress = '+str(a.fp32_ingress).lower()+', ftlpu.vector_bf16_spill = true, ftlpu.vector_log_softplus = ' + str(a.log_softplus).lower() + ', ftlpu.vector_bf16_token_halves = ' + str(a.token_halves).lower() + '} {', 1))
    with (out / 'compile.log').open('w') as log:
        subprocess.run([str(a.compile.resolve()), '--input', str(kernel), '--input-stage', 'kernel',
                        '--output', str(binary), '--icu-compression', 'raw-macro', '--target-config', str(a.target_config.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / 'execution.log').open('w') as log:
        subprocess.run([str(a.runner.resolve()), str(binary), str(out / 'inputs.txt'),
                        str(out / 'actual'), str(out / 'runtime.pipeline.csv')], stdout=log, stderr=subprocess.STDOUT, check=True)
    expected = [1/(1+np.exp(-x)), np.logaddexp(0, a_reference+bias)*scale]
    if a.decay_factor: expected[1]=np.exp(expected[1])
    errors = []
    for i, reference in enumerate(expected):
        actual = np.fromfile(out / f'actual.{i}.bin', '<f4').reshape(x.shape)
        error = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
        assert np.isfinite(actual).all() and error < .03, (i, error)
        if a.log_softplus and not a.decay_factor and i == 1:
            tail = (a_reference+bias < -3) & (np.abs(reference) > 1e-4)
            tail_relative = float(np.max(np.abs((actual-reference)[tail] / reference[tail]), initial=0))
            assert tail_relative < .04, ('softplus negative tail', tail_relative)
            assert np.all(actual[tail] < 0), 'negative scale times positive softplus must stay negative'
        errors.append(error)
    report = dict(status='passed', shape=list(x.shape), nrmse=errors,
                  precision=('Full FP32 intermediate ingress; FP32 chain arithmetic/output; BF16 projection input'
                             if a.fp32_ingress else 'BF16 upper-plane ingress at spill boundaries, FP32 chain arithmetic and output ABI'))
    if a.log_softplus:
        if not a.decay_factor: report['tail_max_relative_error'] = tail_relative
        assert 'log_lut' in (out / 'runtime.pipeline.csv').read_text()
    if a.decay_factor:
        plan=json.loads((out/'graph_plan.json').read_text())
        assert len([n for n in plan['nodes'] if n['name'] in ('gates','decay_factor')])==1
        report['fused_decay_factor']=True
    if a.paired_projections:
        plan=json.loads((out/'graph_plan.json').read_text())
        paired=[n for n in plan['nodes'] if n['name'].endswith('.paired')]
        assert len(paired)==1 and not any(n['name'].endswith('.page0') for n in plan['nodes'])
        assert not [t for t in plan['linker_mem_transports'] if t['phase'] in ('gates','decay_factor')]
        report['paired_projections']=True
    if a.with_prefix:
        gate_actual = np.fromfile(out/'actual.1.bin', '<f4').reshape(32,width)
        prefix_actual = np.fromfile(out/'actual.2.bin', '<f4').reshape(32,width)
        np.testing.assert_allclose(prefix_actual, np.cumsum(gate_actual, axis=0), rtol=1e-6, atol=1e-6)
        plan = json.loads((out/'graph_plan.json').read_text())
        assert {'gates', 'gate_prefix'} <= {n['name'] for n in plan['nodes']}
        report['full_token_prefix_checked'] = True
    if a.with_recurrence:
        from qwen3_5_prefill_tokenwise_test import reference
        beta_actual=np.fromfile(out/'actual.0.bin','<f4').reshape(32,width)[:2]
        eg_actual=np.fromfile(out/'actual.1.bin','<f4').reshape(32,width)[:2]
        y,s=reference(q.reshape(2,2,128),k.reshape(2,2,128),v.reshape(2,2,128),eg_actual,beta_actual,state,True,'fp32')
        ay=(np.fromfile(out/'actual.2.bin','<u2').astype('u4')<<16).view('f4').reshape(y.shape)
        ast=np.fromfile(out/'actual.3.bin','<f4').reshape(2,128,128).transpose(0,2,1)
        rec_errors=[float(np.linalg.norm(actual-ref)/np.linalg.norm(ref)) for actual,ref in ((ay,y),(ast,s))]
        assert max(rec_errors)<.002,rec_errors
        plan=json.loads((out/'graph_plan.json').read_text())
        gate_node=next(n['node'] for n in plan['nodes'] if n['name']=='decay_factor')
        gate_homes=plan['nodes'][gate_node]['direct_output_layouts']
        copies=[t for t in plan['linker_mem_transports'] if t['phase'].startswith('delta_rule.token')
                and t['vectors']==1 and any(t['source_slice']==h['slices'][0] and
                    h['base_row']<=t['source_row']<h['base_row']+32 for h in gate_homes)]
        assert not copies,copies
        report.update(recurrence_nrmse=rec_errors,gate_input_copies=len(copies))
    if a.token_halves and not a.with_recurrence and not a.with_prefix:
        schedule=(out/'decoder_layer.schedule.mlir').read_text()
        repeats=[int(re.search(r'repeat_count = (\d+)', line).group(1))
                 for line in schedule.splitlines() if 'ftlpu.schedule.vxm ' in line]
        assert repeats and set(repeats)=={16}, repeats
        outputs=[line for line in schedule.splitlines() if 'access = "output"' in line and 'placement =' in line]
        assert len(outputs)==2 and all('token_partition' in line for line in outputs)
        report['tokens_per_hemisphere']=16
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
