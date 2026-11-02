"""GDN prefill recurrence: C++ selects tokenwise lowering and owns the scan.

Python provides a single semantic recurrence, immutable inputs, and independent
references. No host recurrence output is supplied to the executable.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import bf16, load_imported_layer
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_reference import causal_depthwise_conv1d
from qwen3_5_qkv_conv_prefill_test import compare
from qwen3_5_decode_gdn_attention_test import run
from scaled_matvec_streaming_test import audit_streaming


def reference(q, k, v, eg, beta, state, reload=False, rounding='truncate'):
    """State ABI [head,key,value]; serial equations independent of lowering."""
    trunc = lambda x: (np.asarray(x, 'f4').view('u4') & np.uint32(0xffff0000)).view('f4')
    if rounding=='rne': trunc=bf16
    cast_state = trunc if reload in (True,'state') else lambda x:x
    cast_gate = trunc if reload in (True,'gates') else lambda x:x
    cast = trunc if reload in (True,'intermediates') else lambda x:x
    state = state.copy()
    result = []
    for t in range(len(q)):
        decayed = cast_state(state) * cast_gate(eg[t])[:, None, None]
        prediction = np.einsum('hkv,hk->hv', cast_state(decayed), k[t])
        delta = (v[t] - cast(prediction)) * cast_gate(beta[t])[:, None]
        state = cast_state(decayed) + k[t, :, :, None] * cast(delta)[:, None, :]
        if rounding=='rne' and reload in (True,'state'): state=cast_state(state)
        y = np.einsum('hkv,hk->hv', cast_state(state), q[t])
        result.append(trunc(y) if reload is True else y)
    return np.asarray(result), state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--package', type=Path)
    p.add_argument('--tokens', type=int, default=32, choices=(2, 32))
    p.add_argument('--core-only', action='store_true', help='Small nonzero-State handoff regression')
    p.add_argument('--rounding', choices=('rne','truncate'), default='rne')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--compile-only', action='store_true')
    a = p.parse_args()
    rne=a.rounding=='rne'
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    build = a.build_dir.resolve(); suffix = '.exe' if sys.platform == 'win32' else ''
    exe = lambda area, name: build / area / (name+suffix)
    g = LogicalGraph(); rng = np.random.default_rng(35261030)
    if a.core_only:
        c = SimpleNamespace(num_value_heads=2, num_key_heads=2, key_head_dim=128, value_head_dim=128)
        shape = (a.tokens, 2, 128)
        q = bf16(rng.normal(0, .02, shape).astype('f4'))
        k = bf16(rng.normal(0, .08, shape).astype('f4'))
        v = bf16(rng.normal(0, .1, shape).astype('f4'))
        eg = bf16(rng.uniform(.7, .99, (a.tokens, 2)).astype('f4'))
        beta = bf16(rng.uniform(.1, .8, (a.tokens, 2)).astype('f4'))
        operands = [q.reshape(a.tokens, -1), k.reshape(a.tokens, -1), v.reshape(a.tokens, -1), eg, beta]
    else:
        if not a.package: p.error('--package is required for the real-weight frontend')
        x, w, packed, scales, c = load_imported_layer(a.package); x = bf16(x)
        assert x.shape == (32, 1024)
        projected = g.project('in_proj_qkv', x, packed['in_proj_qkv'], scales['in_proj_qkv'])
        conv = g.conv(projected, w.conv1d)
        width, dim = c.key_width, c.key_head_dim
        query = g.norm('query_l2', conv[:, :width].reshape(-1, dim), np.ones(dim),
                       1e-6, False, 1/np.sqrt(dim)).reshape(32, width)
        key = g.norm('key_l2', conv[:, width:2*width].reshape(-1, dim), np.ones(dim),
                     1e-6, False).reshape(32, width)
        b, decay = g.gates(x, packed, scales, w.dt_bias, w.a_log, bf16_outputs=rne)
        factor = g.vector_tiles('state_decay_factor', [(decay, 'bf16' if rne else 'f32')],
                               lambda v: v.add(0, kind='exp', immediate=0, dtype='bf16' if rne else 'f32'))
        operands = [query[:a.tokens], key[:a.tokens], conv[:a.tokens, 2*width:], factor[:a.tokens], b[:a.tokens]]
        raw = x.astype('f4') @ w.in_proj_qkv.T
        mixed, _ = causal_depthwise_conv1d(raw, w.conv1d, np.zeros_like(w.conv1d))
        def norm(t, scale):
            t = t.reshape(32, -1, dim)
            return t / np.sqrt(np.sum(t*t, axis=-1, keepdims=True)+1e-6) * scale
        q = norm(mixed[:, :width], 1/np.sqrt(dim))[:a.tokens]
        k = norm(mixed[:, width:2*width], 1)[:a.tokens]
        v = mixed[:a.tokens, 2*width:].reshape(a.tokens, c.num_value_heads, c.value_head_dim)
        beta = (1/(1+np.exp(-(x @ w.in_proj_b.T))))[:a.tokens]
        log_decay = -np.exp(w.a_log)*np.logaddexp(0, x @ w.in_proj_a.T+w.dt_bias)
        eg = np.exp(log_decay[:a.tokens])
    state = rng.normal(0, .02, (c.num_value_heads, c.key_head_dim, c.value_head_dim)).astype('f4')
    outputs = g.delta(*operands, state.reshape(-1, c.value_head_dim), c,state_dtype='bf16' if rne else 'f32')
    # Export the native State orientation as a logical view. Do not introduce
    # a matrix transpose solely to serialize the test's final checkpoint.
    outputs[1] = outputs[1].reshape(c.num_value_heads,c.key_head_dim,c.value_head_dim).transpose(
        0,2,1).reshape(c.num_value_heads*c.value_head_dim,c.key_head_dim)
    if not a.core_only:
        native_v = conv[:, 2*width:].reshape(32,c.num_value_heads,c.value_head_dim)
        outputs += [query,key,native_v[:,::2,:].reshape(32,-1),native_v[:,1::2,:].reshape(32,-1),b,factor]
    binary, kernel = out/'program.ftlpu', out/'kernel.mlir'
    if not a.validate_only:
        (out/'summary.json').write_text(json.dumps(dict(status='pending')))
        logical = g.write(out/'logical', outputs)
        run([exe('compiler','ftlpu-graph-compile'), '--input', logical, '--output', kernel,
             '--emit-kernel', '--target-config', a.target_config.resolve()], out/'import.log')
        kernel.write_text(kernel.read_text().replace('module {',
            'module attributes {'+('ftlpu.vector_bf16_rne_spill = true, ' if rne else '')+
            'ftlpu.gdn_prefill_strategy = "tokenwise", ftlpu.vector_bf16_spill = true, '
            'ftlpu.common_barrier_protocol = true, ftlpu.sxm_wide = true, ftlpu.l2_head_partition = true, '
            'ftlpu.vector_log_softplus = true, ftlpu.vector_bf16_token_halves = true} {', 1))
        run([exe('compiler','ftlpu-compile'), '--input', kernel, '--input-stage', 'kernel',
             '--output', binary, '--target-config', a.target_config.resolve(), '--icu-compression', 'raw-macro',
             '--mxm-execution', 'ws'], out/'compile.log')
        if a.compile_only: return
        os.environ.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',
                          FTLPU_SESSION_PROGRESS='1')
        run([exe('runtime','compiled_stage_file_runner'), binary, out/'inputs.txt', out/'actual',
             out/'runtime.pipeline.csv'], out/'execution.log')
    actual_y = (np.fromfile(out/'actual.0.bin','<u2').astype('u4') << 16).view('f4').reshape(v.shape)
    actual_s = ((np.fromfile(out/'actual.1.bin','<u2').astype('u4')<<16).view('f4') if rne else
                np.fromfile(out/'actual.1.bin','<f4')).reshape(
        c.num_value_heads,c.value_head_dim,c.key_head_dim).transpose(0,2,1)
    expected_y, expected_s = reference(q,k,v,eg,beta,state)
    errors = dict(output=compare(actual_y,expected_y), state=compare(actual_s,expected_s))
    if a.core_only:
        ry, rs = reference(q,k,v,eg,beta,state,True,a.rounding)
    else:
        def bf(index,shape):
            return (np.fromfile(out/f'actual.{index}.bin','<u2').astype('u4')<<16).view('f4').reshape(shape)
        aq=bf(2,(32,c.num_key_heads,c.key_head_dim))[:a.tokens]
        ak=bf(3,(32,c.num_key_heads,c.key_head_dim))[:a.tokens]
        av=np.empty((32,c.num_value_heads,c.value_head_dim),dtype='f4')
        for idx,parity in ((4,0),(5,1)):
            av[:,parity::2,:]=bf(idx,(32,c.num_value_heads//2,c.value_head_dim))
        ab=(bf(6,(32,c.num_value_heads)) if rne else np.fromfile(out/'actual.6.bin','<f4').reshape(32,c.num_value_heads))[:a.tokens]
        ae=(bf(7,(32,c.num_value_heads)) if rne else np.fromfile(out/'actual.7.bin','<f4').reshape(32,c.num_value_heads))[:a.tokens]
        frontend_errors=dict(q=compare(aq,q),k=compare(ak,k),v=compare(av[:a.tokens],v),
                             beta=compare(ab,beta),eg=compare(ae,eg))
        assert all(e['nrmse']<.03 and e['cosine']>.999 for e in frontend_errors.values()),frontend_errors
        ry,rs=reference(aq,ak,av[:a.tokens],ae,ab,state,True,a.rounding)
    errors.update(reload_output=compare(actual_y,ry), reload_state=compare(actual_s,rs))
    actual_inputs=(q,k,v,eg,beta,state) if a.core_only else (aq,ak,av[:a.tokens],ae,ab,state)
    fy,fs=reference(*actual_inputs)
    precision=dict(actual_frontend_fp32=dict(output=compare(actual_y,fy),state=compare(actual_s,fs)))
    for mode in ('state','gates','intermediates'):
        py,ps=reference(*actual_inputs,reload=mode,rounding=a.rounding)
        precision[mode+'_'+a.rounding+'_only']=dict(output=compare(py,fy),state=compare(ps,fs))
    head_errors=[dict(output=compare(actual_y[:,h],ry[:,h]),state=compare(actual_s[h],rs[h]))
                 for h in range(c.num_value_heads)]
    (out/'numerics.json').write_text(json.dumps(dict(errors=errors,reload_by_head=head_errors),indent=2))
    plan=json.loads((out/'graph_plan.json').read_text())
    steps=[n for n in plan['nodes'] if n['name'].startswith('delta_rule.token')]
    assert len(steps)==a.tokens and all('chunkwise' not in n['name'] for n in plan['nodes'])
    # The exact updated-State home is consumed by the next token; materializing
    # a full State between steps is a regression even when numerically correct.
    transports=plan.get('linker_mem_transports',[])
    state_copies=[]
    for previous, current in zip(steps,steps[1:]):
        home=previous['direct_output_layouts'][3]
        assert current['direct_output_layouts'][3]==home, 'State should remain in its native recurrent home'
        for t in transports:
            if (t['phase']==current['name'] and t['source_bank']==home['bank']
                and t['source_slice'] in home['slices']
                and home['base_row']<=t['source_row']<home['base_row']+home['rows']):
                state_copies.append(t)
    assert not state_copies, state_copies
    stats={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',(out/'execution.log').read_text().splitlines()[-1])}
    assert stats.get('host_operations')==0
    with (out/'runtime.pipeline.csv').open() as f:
        cycles=max(int(r['end'])+(int(r['inner_count'])-1)*int(r['inner_interval'])+
                   (int(r['outer_count'])-1)*int(r['outer_interval']) for r in csv.DictReader(f))
    run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs',
         '--pre-execution-dir',out/'actual.pre_execution'],out/'export.log')
    passed=all(errors[n]['nrmse']<.002 and errors[n]['cosine']>.99999 for n in ('reload_output','reload_state'))
    passed &= all(e['nrmse']<.002 and e['cosine']>.99999 for h in head_errors for e in h.values())
    accuracy=all(errors[n]['nrmse']<.03 and errors[n]['cosine']>.999 for n in ('output','state'))
    result=dict(status=('passed' if accuracy else 'correctness_passed_accuracy_failed') if passed else 'failed',
        correctness_passed=passed,accuracy_passed=accuracy,strategy='tokenwise',tokens=a.tokens,
        real_weights=not a.core_only,nonzero_initial_state=True,scope='GDN recurrence through Q readout; no output gate/projection or FFN',
        rounding=a.rounding,
        precision=('BF16 RNE State/gate/intermediate storage; FP32 arithmetic and accumulation' if rne else
                   'BF16-truncated MEM/SR ingress; FP32 State ABI and accumulation; BF16 output'),
        errors=errors,reload_by_head=head_errors,runtime=stats,runtime_trace_cycles=cycles,host_state_handoffs=0,
        fp32_reference_within_3_percent=accuracy,precision_diagnostics=precision,
        frontend_errors={} if a.core_only else frontend_errors,
        output_state_layout='[head,value,key] native checkpoint view',
        native_state_homes=[n['direct_output_layouts'][3] for n in steps],
        streaming=audit_streaming(out))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert passed,errors


if __name__=='__main__': main()
