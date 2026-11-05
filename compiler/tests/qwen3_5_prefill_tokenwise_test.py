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
        if rounding=='fp32' and reload is True:
            # Only MXM's matrix input is narrowed. Never truncate the State
            # saved for the next token, gates, prediction or delta.
            decayed=state*eg[t,:,None,None]
            prediction=np.einsum('hkv,hk->hv',trunc(decayed),k[t])
            delta=(v[t]-prediction)*beta[t,:,None]
            state=decayed+k[t,:,:,None]*delta[:,None,:]
            result.append(bf16(np.einsum('hkv,hk->hv',trunc(state),q[t])))
            continue
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
    p.add_argument('--rounding', choices=('fp32','rne','truncate'), default='fp32',
                   help='FP32 State/ingress (default), BF16 RNE, or legacy high-plane truncation')
    p.add_argument('--recurrence-only', action='store_true', help='Stop before Z/gated RMSNorm/output projection')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--compile-only', action='store_true')
    a = p.parse_args()
    rne=a.rounding=='rne'
    full=a.rounding=='fp32'
    with_tail=not a.core_only and not a.recurrence_only
    if with_tail and a.tokens!=32: p.error('full attention currently requires --tokens 32')
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
        eg = rng.uniform(.7, .99, (a.tokens, 2)).astype('f4')
        beta = rng.uniform(.1, .8, (a.tokens, 2)).astype('f4')
        if not full: eg,beta=bf16(eg),bf16(beta)
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
    if with_tail:
        # Tensor equations only. Shared C++ planning owns the native layouts,
        # retained head-normalization tiles, VXM gate chain and MXM paging.
        z=g.project('in_proj_z',x,packed['in_proj_z'],scales['in_proj_z'])
        normalized=g.norm('delta_output_norm',outputs[0].reshape(-1,c.value_head_dim),
                          w.delta_norm,c.rms_norm_eps)
        def output_gate(graph):
            return graph.add(0,graph.add(1,graph.sigmoid(1)),dtype='bf16')
        gated=g.vector_tiles('delta_output_gate',
            [(normalized,'bf16'),(z.reshape(normalized.shape),'bf16')],output_gate)
        attention=g.project('out_proj',gated.reshape(32,c.value_width),packed['out_proj'],scales['out_proj'])
        outputs += [z,normalized,gated,attention]
    binary, kernel = out/'program.ftlpu', out/'kernel.mlir'
    if not a.validate_only:
        (out/'summary.json').write_text(json.dumps(dict(status='pending')))
        logical = g.write(out/'logical', outputs)
        run([exe('compiler','ftlpu-graph-compile'), '--input', logical, '--output', kernel,
             '--emit-kernel', '--target-config', a.target_config.resolve()], out/'import.log')
        kernel.write_text(kernel.read_text().replace('module {',
            'module attributes {'+('ftlpu.vector_bf16_rne_spill = true, ' if rne else '')+
            ('ftlpu.vector_fp32_ingress = true, ' if full else '')+
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
    if full:
        precision['mxm_bf16_high_planes_and_output_rne']=dict(output=compare(ry,fy),state=compare(rs,fs))
    for mode in (() if full else ('state','gates','intermediates')):
        py,ps=reference(*actual_inputs,reload=mode,rounding=a.rounding)
        precision[mode+'_'+a.rounding+'_only']=dict(output=compare(py,fy),state=compare(ps,fs))
    head_errors=[dict(output=compare(actual_y[:,h],ry[:,h]),state=compare(actual_s[h],rs[h]))
                 for h in range(c.num_value_heads)]
    (out/'numerics.json').write_text(json.dumps(dict(errors=errors,reload_by_head=head_errors),indent=2))
    plan=json.loads((out/'graph_plan.json').read_text())
    gate_mapping={}
    if full and not a.core_only:
        paired=[n for n in plan['nodes'] if n['name'].startswith(('in_proj_a.','in_proj_b.'))]
        assert len(paired)==1 and paired[0]['name'].endswith('.paired'),paired
        factor=next(n for n in plan['nodes'] if n['name']=='state_decay_factor')
        assert not any(n['name']=='delta_gates' for n in plan['nodes'])
        homes=factor['direct_output_layouts']
        assert len(homes)==2 and all(h['kind']=='fp32_quad_planar' for h in homes)
        copies=[t for t in plan['linker_mem_transports']
                if t['phase'].startswith('delta_rule.token') and t['vectors']==1
                and any(t['source_bank']==h['bank'] and t['source_slice']==h['slices'][0]
                    and h['base_row']<=t['source_row']<h['base_row']+h['rows'] for h in homes)]
        assert not copies,copies
        gate_mapping=dict(paired_projection=paired[0]['name'],fused_eg=True,
            recurrence_gate_copy_domains=0,projection_cycles=paired[0]['nominal_end']-paired[0]['nominal_start'],
            elementwise_cycles=factor['nominal_end']-factor['nominal_start'],
            span_cycles=factor['nominal_end']-paired[0]['nominal_start'])
    tail_errors={}
    if with_tail:
        az=bf(8,(32,c.num_value_heads,c.value_head_dim))
        an=bf(9,az.shape)
        ag=bf(10,az.shape)
        gate_node=next(n for n in plan['nodes'] if n['name']=='delta_output_gate')
        gate_layout=gate_node['direct_output_layouts'][0]['kind']
        if gate_layout.startswith('token_partition_'):
            ag=ag.reshape(2,32,c.num_value_heads//2,c.value_head_dim).transpose(1,2,0,3).reshape(az.shape)
        aa=bf(11,(32,c.hidden_size))
        normalize=lambda y:y/np.sqrt(np.mean(y*y,axis=-1,keepdims=True)+c.rms_norm_eps)*bf16(w.delta_norm)
        silu=lambda z:z/(1+np.exp(-z))
        z_ref=(x.astype('f4')@w.in_proj_z.T).reshape(az.shape)
        model_gated=normalize(expected_y)*silu(z_ref)
        tail_errors=dict(z_projection=compare(az,z_ref),
            rmsnorm_given_actual_input=compare(an,normalize(actual_y)),
            gate_given_actual_inputs=compare(ag,an*silu(az)),
            gated_norm_end_to_end=compare(ag,model_gated),
            output_projection_given_actual_input=compare(aa,ag.reshape(32,-1)@w.out_proj.T),
            attention_end_to_end=compare(aa,model_gated.reshape(32,-1)@w.out_proj.T))
        (out/'tail_numerics.json').write_text(json.dumps(tail_errors,indent=2))
        tail_heads=[dict(norm=compare(an[:,h],normalize(actual_y)[:,h]),
            gate=compare(ag[:,h],(an*silu(az))[:,h])) for h in range(c.num_value_heads)]
        tail_local=all(e['nrmse']<.03 and e['cosine']>.999 for h in tail_heads for e in h.values())
        assert tail_local,tail_heads
        assert any(n['name'].startswith('out_proj.page') for n in plan['nodes'])
        norm_nodes=[n for n in plan['nodes'] if n['name'].startswith('delta_output_norm.rows')]
        norm_transports=[t for t in plan.get('linker_mem_transports',[]) if t['phase'].startswith('delta_output_norm')]
        assert len(norm_nodes)==16 and not norm_nodes[0]['norm_weight_reused']
        assert all(n['norm_weight_reused'] and n['norm_tight_start'] for n in norm_nodes[1:])
        assert len(norm_transports)==8 and all(t['phase']==norm_nodes[0]['name'] for t in norm_transports)
        assert not any(t['phase'].startswith('delta_output_gate') for t in plan.get('linker_mem_transports',[]))
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
    cycles=0; epoch_offsets=[]
    with (out/'runtime.pipeline.csv').open() as f:
        for row in csv.DictReader(f):
            cycles=max(cycles,int(row['end'])+(int(row['inner_count'])-1)*int(row['inner_interval'])+
                       (int(row['outer_count'])-1)*int(row['outer_interval']))
            if row['resource']=='Paging.Consumer':
                logical_cycle=int(re.search(r'logical_cycle=(\d+)',row['detail'])[1])
                epoch_offsets.append((logical_cycle,int(row['start'])-logical_cycle))
    epoch_offsets.sort()
    def stage_range(node):
        start,end=node['nominal_start'],node['nominal_end']
        offset=next((delta for logical,delta in reversed(epoch_offsets) if logical<=start),0)
        return dict(name=node['name'],nominal_start=start,nominal_end=end,
                    runtime_start=start+offset,runtime_end=end+offset)
    run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs',
         '--pre-execution-dir',out/'actual.pre_execution'],out/'export.log')
    paging_image=json.loads((out/'runtime_icu_programs'/'program.json').read_text())
    passed=all(errors[n]['nrmse']<.002 and errors[n]['cosine']>.99999 for n in ('reload_output','reload_state'))
    passed &= all(e['nrmse']<.002 and e['cosine']>.99999 for h in head_errors for e in h.values())
    accuracy=all(errors[n]['nrmse']<.03 and errors[n]['cosine']>.999 for n in ('output','state'))
    if with_tail:
        passed &= all(tail_errors[n]['nrmse']<.03 and tail_errors[n]['cosine']>.999 for n in
            ('z_projection','rmsnorm_given_actual_input','gate_given_actual_inputs','output_projection_given_actual_input'))
        accuracy &= all(e['nrmse']<.03 and e['cosine']>.999 for e in tail_errors.values())
    result=dict(status=('passed' if accuracy else 'correctness_passed_accuracy_failed') if passed else 'failed',
        correctness_passed=passed,accuracy_passed=accuracy,strategy='tokenwise',tokens=a.tokens,
        real_weights=not a.core_only,nonzero_initial_state=True,scope='Full GDN attention: QKV/Conv/QK norm/gates/tokenwise recurrence/Z/gated RMSNorm/output projection; no input RMSNorm, residual or FFN' if with_tail else 'GDN recurrence through Q readout; no output gate/projection or FFN',
        rounding=a.rounding,transport_overlap=paging_image['transport_overlap_protocol'],
        weight_paging_policy=plan.get('weight_paging_policy'),paging_image=paging_image,paging_protocol='common_sync',
        precision=('FP32 State/gate/intermediate storage and full VXM ingress; BF16 high-plane MXM input; BF16 RNE output' if full else
                   'BF16 RNE State/gate/intermediate storage; FP32 arithmetic and accumulation' if rne else
                   'BF16-truncated MEM/SR ingress; FP32 State ABI and accumulation; BF16 output'),
        errors=errors,reload_by_head=head_errors,runtime=stats,runtime_trace_cycles=cycles,host_state_handoffs=0,
        fp32_reference_within_3_percent=accuracy,precision_diagnostics=precision,
        frontend_errors={} if a.core_only else frontend_errors,
        gate_mapping=gate_mapping,
        output_state_layout='[head,value,key] native checkpoint view',
        native_state_homes=[n['direct_output_layouts'][3] for n in steps],
        streaming=audit_streaming(out))
    if with_tail:
        result['attention_tail']=dict(status='passed' if passed and accuracy else 'failed',errors=tail_errors,
            norm_weight='delta_norm, no 1+weight offset',gated_output_layout=gate_layout,
            norm_weight_materializations=1,norm_weight_reuses=15,core_to_norm_layout_copies=0,norm_to_gate_layout_copies=0,
            stages=[stage_range(n) for n in plan['nodes']
                    if n['name'].startswith(('in_proj_z','delta_output_','out_proj'))],
            output_bindings=dict(z=8,normalized=9,gated=10,attention=11),
            body_mem_transports=[t for t in transports if t['phase'].startswith(('in_proj_z','delta_output_','out_proj'))])
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert passed,errors


if __name__=='__main__': main()
