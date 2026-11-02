"""Batch-one QKV projection, optionally followed by decode Conv1D and SiLU.

Python supplies tensors and an independent reference; C++ performs all lowering.
Real imported weights and wide SXM hardware are used. Q/K L2Norm and beta/g
gates, retained state decay/KS, correction, rank-one update and streamed
query readout, gated output RMSNorm and output projection are optional.
Input RMSNorm is excluded.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import load_imported_layer, bf16
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_decode_gdn_attention_test import run
from qwen3_5_decode_gdn_batched_test import read
from qwen3_5_qkv_conv_prefill_test import compare
from qwen3_5_reference import causal_depthwise_conv1d
from tensor_geometry import Value
from scaled_matvec_streaming_test import audit_streaming


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--with-conv', action='store_true', help='Append decode Conv1D+SiLU and export updated cache')
    parser.add_argument('--with-qk-norm', action='store_true', help='Append per-head Q/K L2Norm after Conv')
    parser.add_argument('--with-gates', action='store_true', help='Append beta and log-decay gate projections and ALU branches')
    parser.add_argument('--with-state-decay', action='store_true', help='Load nonzero DDR state and multiply it by exp(g), without recurrence')
    parser.add_argument('--with-ks', action='store_true', help='Retain scaled state and stream the same result into MXM for K*S')
    parser.add_argument('--with-correction', action='store_true', help='Fuse beta*(V-K*S) into the streamed matvec epilogue')
    parser.add_argument('--with-update', action='store_true', help='Append wide-SXM broadcast and VXM rank-one state update')
    parser.add_argument('--with-readout', action='store_true', help='Stream updated state into MXM for Q readout')
    parser.add_argument('--with-output-norm', action='store_true', help='Append real Z projection and per-head gated RMSNorm')
    parser.add_argument('--with-output-projection', action='store_true', help='Append real out_proj after gated RMSNorm using shared projection lowering')
    parser.add_argument('--log-softplus', action='store_true', help='Use native Log LUT plus a small-value polynomial for softplus')
    parser.add_argument('--validate-only', action='store_true', help='Validate existing runtime outputs and schedule without rerunning')
    args = parser.parse_args()
    if args.with_output_projection:
        args.with_output_norm = True
    if args.with_output_norm:
        args.with_readout = True
    if args.with_readout:
        args.with_update = True
    if args.with_update:
        args.with_correction = True
    if args.with_correction:
        args.with_ks = True
    if args.with_ks:
        args.with_state_decay = True
    if args.with_state_decay:
        args.with_gates = True
    if args.with_gates:
        args.with_qk_norm = True
    if args.with_qk_norm:
        args.with_conv = True
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(status='pending')))
    build = args.build_dir.resolve()
    suffix = '.exe' if sys.platform == 'win32' else ''
    exe = lambda area, name: build / area / (name + suffix)
    x, weights, packed, scales, config = load_imported_layer(args.package)
    # A standalone operator input, not a host-computed RMSNorm intermediate.
    x = bf16(x[:1])
    graph = LogicalGraph()
    qkv = graph.project('in_proj_qkv', x, packed['in_proj_qkv'], scales['in_proj_qkv'])
    finals = [qkv]
    if args.with_conv:
        history = bf16(np.random.default_rng(35261029).normal(
            0, .02, (config.conv_kernel_size, config.conv_width)).astype('f4'))
        t = graph.type(history.shape, 'bf16')
        source = f'''module {{ func.func @stage(%h: {t}) -> {t} {{
            %o = ftlpu.kernel.reshape %h : ({t}) -> {t}
            return %o : {t}
        }} }}'''
        cached = graph.run('conv_cache_view', source, [(history, 'bf16')],
                           [(history.shape, 'bf16')])[0]
        window = Value.concatenate([cached[-(config.conv_kernel_size-1):], qkv])
        mixed = graph.conv(window, weights.conv1d)[-1:]
        finals += [mixed, window]
        if args.with_qk_norm:
            q = graph.norm('query_l2', mixed[:, :config.key_width].reshape(-1,config.key_head_dim),
                np.ones(config.key_head_dim), 1e-6, False, 1/np.sqrt(config.key_head_dim))
            k = graph.norm('key_l2', mixed[:, config.key_width:2*config.key_width].reshape(-1,config.key_head_dim),
                np.ones(config.key_head_dim), 1e-6, False)
            finals += [q, k, mixed[:,2*config.key_width:]]
    if args.with_gates:
        beta, decay = graph.gates(x, packed, scales, weights.dt_bias, weights.a_log)
        finals += [beta, decay]
        if args.with_output_norm:
            z = graph.project('in_proj_z', x, packed['in_proj_z'], scales['in_proj_z'])
        if args.with_state_decay:
            # External state uses [head*key, value]: a logical tensor layout,
            # not a host-computed activation or instruction schedule.
            state = np.random.default_rng(35261030).normal(0, .02,
                (config.num_value_heads*config.key_head_dim, config.value_head_dim)).astype('f4')
            factor = graph.vector_tiles('state_decay_factor', [(decay,'f32')],
                lambda v: v.add(0, kind='exp', immediate=0))
            if args.with_ks:
                # External state ABI is [head,value,key]; this is immutable
                # input packing, not host evaluation of a runtime activation.
                state = state.reshape(config.num_value_heads,config.key_head_dim,config.value_head_dim).transpose(0,2,1).copy().reshape(-1,config.key_head_dim)
                keys = k.reshape(config.num_key_heads,config.key_head_dim).repeat(
                    config.num_value_heads//config.num_key_heads,axis=0)
                mt,dt,kt,pt = (graph.type(shape,dtype) for shape,dtype in (
                    (state.shape,'f32'),(factor.shape,'f32'),(keys.shape,'bf16'),
                    ((config.num_value_heads,config.value_head_dim),'f32')))
                source = f'''module {{ func.func @stage(%s: {mt}, %d: {dt}, %k: {kt}) -> ({mt}, {pt}) {{
                  %scaled, %p = ftlpu.kernel.scaled_matvec %s, %d, %k : ({mt}, {dt}, {kt}) -> ({mt}, {pt})
                  return %scaled, %p : {mt}, {pt}
                }} }}'''
                inputs=[(state,'f32'),(factor,'f32'),(keys,'bf16')]
                outputs=[(state.shape,'f32'),((config.num_value_heads,config.value_head_dim),'f32')]
                if args.with_correction:
                    values=mixed[:,2*config.key_width:].reshape(config.num_value_heads,config.value_head_dim)
                    vt=graph.type(values.shape,'bf16')
                    source=f'''module {{ func.func @stage(%s: {mt}, %d: {dt}, %k: {kt}, %v: {vt}, %b: {dt}) -> ({mt}, {pt}, {pt}) {{
                      %scaled, %p, %delta = ftlpu.kernel.scaled_matvec %s, %d, %k epilogue(%v, %b) : ({mt}, {dt}, {kt}, {vt}, {dt}) -> ({mt}, {pt}, {pt})
                      return %scaled, %p, %delta : {mt}, {pt}, {pt}
                    }} }}'''
                    inputs += [(values,'bf16'),(beta,'f32')]
                    outputs += [((config.num_value_heads,config.value_head_dim),'f32')]
                    if args.with_update:
                        source=source.replace('-> ('+mt+', '+pt+', '+pt+')','-> ('+mt+', '+pt+', '+pt+', '+mt+')').replace('%scaled, %p, %delta =','%scaled, %p, %delta, %updated =').replace('return %scaled, %p, %delta : '+mt+', '+pt+', '+pt,'return %scaled, %p, %delta, %updated : '+mt+', '+pt+', '+pt+', '+mt)
                        outputs += [(state.shape,'f32')]
                        if args.with_readout:
                            queries=q.reshape(config.num_key_heads,config.key_head_dim).repeat(config.num_value_heads//config.num_key_heads,axis=0)
                            source=source.replace('%b: '+dt+')','%b: '+dt+', %q: '+kt+')').replace('epilogue(%v, %b)','epilogue(%v, %b, %q)').replace('('+mt+', '+dt+', '+kt+', '+vt+', '+dt+')','('+mt+', '+dt+', '+kt+', '+vt+', '+dt+', '+kt+')').replace('-> ('+mt+', '+pt+', '+pt+', '+mt+')','-> ('+mt+', '+pt+', '+pt+', '+mt+', '+pt+')').replace('%delta, %updated =','%delta, %updated, %y =').replace('return %scaled, %p, %delta, %updated : '+mt+', '+pt+', '+pt+', '+mt,'return %scaled, %p, %delta, %updated, %y : '+mt+', '+pt+', '+pt+', '+mt+', '+pt)
                            inputs += [(queries,'bf16')]; outputs += [((config.num_value_heads,config.value_head_dim),'f32')]
                results=graph.run('state_decay_ks',source,inputs,outputs)
                decayed,prediction=results[:2]
            else:
                decayed = graph.vector_tiles('state_decay', [(state,'f32'),(factor.reshape(config.num_value_heads,1).repeat(config.key_head_dim,axis=0).repeat(config.value_head_dim,axis=1),'f32')],
                    lambda v: v.add(0,1))
            state_quarter = state.shape[0]//2
            if args.with_ks:
                # Export the native complete results. Splitting a full-depth
                # packed tensor into diagnostic halves would materialize
                # strided views and obscure the continuous compute pipeline.
                finals += [factor,decayed,prediction]
                if args.with_correction:
                    finals += [results[2]]
                if args.with_update:
                    finals += [results[3]]
                if args.with_readout:
                    finals += [results[4]]
            else:
                finals += [factor] + [decayed[i*state_quarter:(i+1)*state_quarter] for i in range(2)]
    if args.with_output_norm:
        readout_bf16 = graph.truncate_bf16('readout_bf16_view', results[4])
        normalized = graph.norm('delta_output_norm', readout_bf16,
            weights.delta_norm, config.rms_norm_eps)
        def output_gate(g):
            silu = g.add(1, g.sigmoid(1))
            return g.add(0, silu, dtype='bf16')
        gated = graph.vector_tiles('delta_output_gate',
            [(normalized, 'bf16'), (z.reshape(normalized.shape), 'bf16')], output_gate)
        finals += [z, normalized, gated]
        if args.with_output_projection:
            attention = graph.project('out_proj', gated.reshape(1, config.value_width),
                packed['out_proj'], scales['out_proj'])
            finals += [attention]
    if not args.validate_only:
        logical = graph.write(out / 'logical', finals)
        kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
        run([exe('compiler', 'ftlpu-graph-compile'), '--input', logical, '--output', kernel,
             '--emit-kernel', '--target-config', args.target_config.resolve()], out / 'import.log')
        kernel.write_text(kernel.read_text().replace('module {',
            'module attributes {ftlpu.vector_bf16_spill = true, ftlpu.common_barrier_protocol = true, ftlpu.sxm_wide = true, ftlpu.l2_head_partition = true, ftlpu.vector_log_softplus = '+str(args.log_softplus).lower()+'} {', 1))
        run([exe('compiler', 'ftlpu-compile'), '--input', kernel, '--input-stage', 'kernel',
             '--output', binary, '--target-config', args.target_config.resolve(),
             '--icu-compression', 'raw-macro', '--mxm-execution', 'ws'], out / 'compile.log')
    if args.with_conv:
        plan = json.loads((out / 'graph_plan.json').read_text())
        assert not any(n['name'].startswith('conv_layout_group') for n in plan['nodes'])
        conv_nodes = [n for n in plan['nodes'] if n['name'].startswith('conv1d_silu.block')]
        assert len(conv_nodes) == 2, 'Decode Conv should keep all heads in one bilateral domain'
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
    os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE'] = 'wide'
    os.environ['FTLPU_SESSION_PROGRESS'] = '1'
    if not args.validate_only:
        run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
             out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log')
    actual = read(out / 'actual.0.bin', (1, config.conv_width))
    expected = x.astype(np.float32) @ weights.in_proj_qkv.T
    metrics = compare(actual, expected)
    parts = {name: compare(actual[:, begin:end], expected[:, begin:end])
             for name, begin, end in (('q', 0, config.key_width),
                 ('k', config.key_width, 2*config.key_width),
                 ('v', 2*config.key_width, config.conv_width))}
    if args.with_conv:
        conv = read(out / 'actual.1.bin', (1, config.conv_width))
        cache = read(out / 'actual.2.bin', (config.conv_kernel_size, config.conv_width))
        reference_conv, reference_cache = causal_depthwise_conv1d(expected, weights.conv1d, history.T)
        parts['conv_silu'] = compare(conv, reference_conv)
        parts['conv_cache'] = compare(cache, reference_cache.T)
        # Also isolate Conv error from preceding projection quantization.
        isolated_conv, isolated_cache = causal_depthwise_conv1d(actual, weights.conv1d, history.T)
        parts['conv_given_actual_qkv'] = compare(conv, isolated_conv)
        assert np.array_equal(cache, isolated_cache.T), 'Conv cache update must preserve BF16 input bits'
        if args.with_qk_norm:
            for idx,name,begin,scale in [(3,'query_l2',0,1/np.sqrt(config.key_head_dim)),
                                       (4,'key_l2',config.key_width,1.)]:
                value = read(out / f'actual.{idx}.bin', (config.num_key_heads,config.key_head_dim))
                ref = conv[:,begin:begin+config.key_width].reshape(value.shape)
                ref = ref/np.sqrt(np.sum(ref*ref,axis=-1,keepdims=True)+1e-6)*scale
                parts[name] = compare(value,ref)
                for head in range(config.num_key_heads):
                    error = compare(value[head],ref[head])
                    assert error['nrmse'] < .03 and error['cosine'] > .999, (name,head,error)
                end_to_end = reference_conv[:,begin:begin+config.key_width].reshape(value.shape)
                end_to_end = end_to_end/np.sqrt(np.sum(end_to_end*end_to_end,axis=-1,keepdims=True)+1e-6)*scale
                parts[name+'_end_to_end'] = compare(value,end_to_end)
            v = read(out/'actual.5.bin',(1,config.value_width))
            assert np.array_equal(v,conv[:,2*config.key_width:]), 'V changed across Q/K normalization'
    if args.with_gates:
        b_ref = x.astype('f4') @ weights.in_proj_b.T
        a_ref = x.astype('f4') @ weights.in_proj_a.T + weights.dt_bias
        refs = [1/(1+np.exp(-b_ref)), -np.exp(weights.a_log)*np.logaddexp(0,a_ref)]
        for idx,name,ref in zip((6,7),('beta','g'),refs):
            val = np.fromfile(out/f'actual.{idx}.bin','<f4').reshape(ref.shape)
            parts[name] = compare(val,ref)
            relative = np.abs(val-ref)/np.maximum(np.abs(ref),1e-8)
            # EXP LUT uses FP16 FTZ; a decay of order 1e-9 may round to zero.
            # Keep tight absolute tolerance there and per-head relative checks
            # for all numerically significant gates.
            assert np.isfinite(val).all() and np.all(np.abs(val-ref) <= .05*np.abs(ref)+1e-7), (name,'per-head relative',relative.tolist())
    if args.with_state_decay:
        plan=json.loads((out/'graph_plan.json').read_text())
        transports=plan['linker_mem_transports']
        assert not any(t['phase'].startswith(('state_decay.rows','in_proj_','conv1d_silu'))
            for t in transports), 'Redundant operand/window materialization returned'
        factor_actual = np.fromfile(out/'actual.8.bin','<f4').reshape(1,config.num_value_heads)
        state_indices=[9] if args.with_ks else [9,10]
        state_actual = np.concatenate([np.fromfile(out/f'actual.{i}.bin','<f4')
            for i in state_indices]).reshape(state.shape)
        g_actual = np.fromfile(out/'actual.7.bin','<f4').reshape(1,config.num_value_heads)
        parts['state_decay_factor'] = compare(factor_actual, np.exp(g_actual))
        parts['state_decay_given_actual_g'] = compare(state_actual, state*np.repeat(np.exp(g_actual).reshape(-1,1),config.key_head_dim,axis=0))
        parts['state_decay_end_to_end'] = compare(state_actual, state*np.repeat(np.exp(refs[1]).reshape(-1,1),config.key_head_dim,axis=0))
        for h in range(config.num_value_heads):
            error = compare(state_actual[h*config.key_head_dim:(h+1)*config.key_head_dim],state[h*config.key_head_dim:(h+1)*config.key_head_dim]*np.exp(g_actual[0,h]))
            assert error['nrmse'] < .03 and error['cosine'] > .999, ('state head',h,error)
        if args.with_ks:
            key_actual=read(out/'actual.4.bin',(config.num_key_heads,config.key_head_dim))
            key_actual=np.repeat(key_actual,config.num_value_heads//config.num_key_heads,axis=0)
            prediction_actual=np.fromfile(out/'actual.10.bin','<f4').reshape(config.num_value_heads,config.value_head_dim)
            truncated=(state_actual.view('u4') & np.uint32(0xffff0000)).view('f4')
            ref=np.einsum('hvk,hk->hv',truncated.reshape(config.num_value_heads,config.value_head_dim,config.key_head_dim),key_actual)
            parts['ks_given_actual_scaled_state']=compare(prediction_actual,ref)
            for h in range(config.num_value_heads):
                error=compare(prediction_actual[h],ref[h])
                assert error['nrmse']<1e-5 and error['cosine']>.999999,('KS head',h,error)
            ref=np.einsum('hvk,hk->hv',state.reshape(config.num_value_heads,config.value_head_dim,config.key_head_dim)*np.exp(g_actual).reshape(-1,1,1),key_actual)
            parts['ks_end_to_end_given_actual_g_k']=compare(prediction_actual,ref)
            if args.with_correction:
                delta=np.fromfile(out/'actual.11.bin','<f4').reshape(prediction_actual.shape)
                actual_v=read(out/'actual.5.bin',prediction_actual.shape)
                actual_beta=np.fromfile(out/'actual.6.bin','<f4').reshape(-1,1)
                truncate=lambda x:(np.asarray(x,dtype='f4').view('u4') & np.uint32(0xffff0000)).view('f4')
                expected=(actual_v-truncate(prediction_actual))*truncate(actual_beta)
                parts['correction_given_actual_inputs']=compare(delta,expected)
                parts['correction_untruncated_inputs']=compare(delta,(actual_v-prediction_actual)*actual_beta)
                for h in range(config.num_value_heads):
                    error=compare(delta[h],expected[h])
                    assert error['nrmse']<1e-5 and error['cosine']>.999999,('correction head',h,error)
                if args.with_update:
                    shape=(config.num_value_heads,config.value_head_dim,config.key_head_dim)
                    updated=np.fromfile(out/'actual.12.bin','<f4').reshape(shape)
                    expected=truncate(state_actual).reshape(shape)+truncate(delta)[:,:,None]*key_actual[:,None,:]
                    parts['state_update_given_actual_inputs']=compare(updated,expected)
                    parts['state_update_untruncated_inputs']=compare(updated,state_actual.reshape(shape)+delta[:,:,None]*key_actual[:,None,:])
                    for h in range(config.num_value_heads):
                        error=compare(updated[h],expected[h])
                        assert error['nrmse']<1e-5 and error['cosine']>.999999,('updated state head',h,error)
                    if args.with_readout:
                        queries=read(out/'actual.3.bin',(config.num_key_heads,config.key_head_dim))
                        queries=np.repeat(queries,config.num_value_heads//config.num_key_heads,axis=0)
                        y=np.fromfile(out/'actual.13.bin','<f4').reshape(prediction_actual.shape)
                        reference=np.einsum('hvk,hk->hv',truncate(updated),queries)
                        parts['readout_given_actual_inputs']=compare(y,reference)
                        parts['readout_untruncated_state']=compare(y,np.einsum('hvk,hk->hv',updated,queries))
                        for h in range(config.num_value_heads):
                            error=compare(y[h],reference[h])
                            assert error['nrmse']<1e-5 and error['cosine']>.999999,('readout head',h,error)
    if args.with_output_norm:
        z_actual = read(out/'actual.14.bin', (1,config.value_width))
        norm_actual = read(out/'actual.15.bin', (config.num_value_heads,config.value_head_dim))
        gated_actual = read(out/'actual.16.bin', norm_actual.shape)
        parts['z_projection'] = compare(z_actual, x.astype('f4') @ weights.in_proj_z.T)
        yin = truncate(y)
        norm_ref = yin / np.sqrt(np.mean(yin*yin,axis=-1,keepdims=True)+config.rms_norm_eps) * bf16(weights.delta_norm)
        parts['output_rmsnorm'] = compare(norm_actual,norm_ref)
        zin=z_actual.reshape(norm_actual.shape)
        gate_ref=norm_actual*zin/(1+np.exp(-zin))
        parts['output_gate_given_actual_inputs']=compare(gated_actual,gate_ref)
        parts['gated_rmsnorm']=compare(gated_actual,norm_ref*zin/(1+np.exp(-zin)))
        for h in range(config.num_value_heads):
            for label, actual_head, ref_head in [('rmsnorm',norm_actual[h],norm_ref[h]),('gate',gated_actual[h],gate_ref[h])]:
                error=compare(actual_head,ref_head)
                assert error['nrmse']<.03 and error['cosine']>.999,(label,h,error)
        plan=json.loads((out/'graph_plan.json').read_text())
        schedule=(out/'decoder_layer.schedule.mlir').read_text()
        assert 'delta_output_gate.silu_multiply.native_chain' in schedule
        assert 'delta_output_norm.rmsnorm.direct_planar_ingress' in schedule
        assert 'delta_output_norm.rmsnorm.vxm_sxm_direct' in schedule
        assert not any(t['phase'].startswith('delta_output_gate') for t in plan['linker_mem_transports']), 'Gated RMSNorm must consume producer layouts directly'
        norm_binding=next(l for l in schedule.splitlines() if 'name = "rmsnorm.result.' in l)
        norm_sides=[int(v) for v in re.search(r'partition_row_sides = \[([^\]]+)\]',norm_binding)[1].split(',')]
        assert norm_sides[:16]==[0]*8+[1]*8, norm_sides
        if args.with_output_projection:
            attention_actual = read(out/'actual.17.bin', (1, config.hidden_size))
            parts['output_projection_given_actual_input'] = compare(attention_actual,
                gated_actual.reshape(1, config.value_width) @ weights.out_proj.T)
            parts['output_projection_after_norm_reference'] = compare(attention_actual,
                (norm_ref*zin/(1+np.exp(-zin))).reshape(1, config.value_width) @ weights.out_proj.T)
            assert any(n['name'].startswith('out_proj.page') for n in plan['nodes']), 'Output projection must use shared C++ projection lowering'
    rows = list(csv.DictReader((out / 'runtime.pipeline.csv').open()))
    state_ingress = []
    if args.with_state_decay:
        # Match payloads, not hard-coded input binding numbers.
        bindings = set()
        if args.with_ks:
            state_nodes=[n for n in plan['nodes'] if n['name'].startswith('state_decay_ks.heads')]
            payload_sizes=[n['direct_output_layouts'][0]['rows']*32*4 for n in state_nodes]
        else:
            payload_sizes=[state.nbytes//2]*2
        assert sum(payload_sizes)==state.nbytes
        offset=0
        for i,size in enumerate(payload_sizes):
            payload = state.tobytes()[offset:offset+size]
            offset+=size
            matches = [p for p in out.glob('input*.bin') if p.read_bytes() == payload]
            assert len(matches) == 1, ('state DDR payload',i,matches)
            bindings.add(int(matches[0].stem[5:]))
        state_ingress = [r for r in rows if r['resource']=='C2C.HostInput'
            and int(re.search(r'\bbinding=(\d+)',r['detail'])[1]) in bindings]
        assert len(state_ingress)==len(payload_sizes), 'Missing runtime state ingress'
        for transfer in state_ingress:
            assert any('WriteSync' in r['resource'] and
                int(transfer['start']) <= int(r['start']) < int(transfer['end'])
                for r in rows), 'State ingress must execute MEM_WRITE_SYNC'
    sxm_rows = [row for row in rows if any('SXM' in value.upper() for value in row.values())]
    # These stages require no SXM; reject any introduced legacy layout operations.
    if args.with_qk_norm:
        schedule = (out/'decoder_layer.schedule.mlir').read_text()
        sxm_lines = [line for line in schedule.splitlines() if 'ftlpu.schedule.sxm ' in line]
        assert sxm_lines and all('wide_matrix' in line and 'opcode = "transpose"' in line for line in sxm_lines)
        assert 'l2norm.direct_planar_ingress' in schedule
        assert 'l2norm.direct_planar_output' in schedule
        # Check the producer/consumer route, not only the timeline label:
        # output transpose must capture VXM streams, with no intermediate
        # writes from VXM or MEM reads feeding SXM in that output interval.
        direct_windows = [line for line in schedule.splitlines()
                          if 'ftlpu.schedule.timeline' in line and '.vxm_sxm_direct"' in line]
        assert len(direct_windows) == 2 + int(args.with_output_norm)
        def integer(line, field):
            return int(re.search(r'\b'+field+r' = (-?\d+) : i64', line)[1])
        for window in direct_windows:
            begin, end = integer(window,'start'), integer(window,'end')
            captures = [line for line in sxm_lines if begin <= integer(line,'cycle') < end]
            assert len(captures) == 2, window
            for line in captures:
                side = integer(line,'hemisphere')
                stream = (1-side)*8+2
                assert f'source_streams = [{stream}, {stream+1}]' in line, line
                assert integer(line,'cycle') < begin+config.key_head_dim, 'Transpose must overlap normalization'
            for line in schedule.splitlines():
                if 'ftlpu.schedule.mem_transfer ' not in line or not begin <= integer(line,'cycle') < end:
                    continue
                stream = integer(line,'packed_stream')
                assert not ('opcode = "write"' in line and stream in (2,3,10,11)), line
                assert not ('opcode = "read"' in line and stream in (16,17)), line
        assert 'l2norm.transpose_input' not in schedule and 'l2norm.restore_layout' not in schedule
        transpose_rows = [row for row in sxm_rows if row['resource'].endswith('.Transpose')]
        assert all(int(row['end']) - int(row['start']) == 70
                   and 'timing=fixed_fu_span' in row['detail']
                   and 'input=[+0,+35) output=[+35,+70)' in row['detail']
                   for row in transpose_rows), 'Wide SXM trace must include input and output skew tails'
        expected_transposes=8+((4 if args.with_correction else 2)*len(state_nodes) if args.with_ks else 4 if args.with_state_decay else 0)
        if args.with_update:expected_transposes+=2*config.num_value_heads*(config.value_head_dim//32)
        if args.with_output_norm:expected_transposes+=4
        assert len(transpose_rows) == expected_transposes, 'Q/K ingress/egress and one bilateral coefficient broadcast per state tile'
        if args.with_state_decay and not args.with_ks:
            assert 'wide_scalar_broadcast' in schedule
            broadcast_rows=[r for r in transpose_rows if int(r['inner_count'])==1]
            assert len(broadcast_rows)==4, 'FP32 broadcast needs two byte-pair transposes on both sides'
            transpose_rows=[r for r in transpose_rows if int(r['inner_count'])!=1]
        elif args.with_ks:
            assert 'scaled_matvec.vxm_to_mxm_and_mem' in schedule
            transpose_rows=[r for r in transpose_rows if int(r['inner_count'])!=1]
        assert all(int(row['inner_count']) == config.key_head_dim//32
                   and int(row['inner_interval']) == 32 for row in transpose_rows), transpose_rows
        assert schedule.count('name = "query_l2.l2norm.sxm_vxm_streaming"') == 1
        assert schedule.count('name = "key_l2.l2norm.sxm_vxm_streaming"') == 1
        # Runtime evidence: each bilateral ingress must begin a continuous
        # square pass before the final transpose tile has even been issued.
        ingress = sorted([r for r in transpose_rows if r['resource'] == 'SXM.E.Transpose'],key=lambda r:int(r['start']))[::2]
        if args.with_output_norm:ingress=ingress[:2]
        assert len(ingress) == 2
        assert 'l2norm.prepare_sxm_pair' not in schedule, 'Partitioned heads must not be mirrored before SXM'
        bindings = [line for line in schedule.splitlines() if 'name = "l2norm.result.' in line]
        assert len(bindings) == 2
        for binding in bindings:
            sides = [int(x) for x in re.search(r'partition_row_sides = \[([^\]]+)\]',binding)[1].split(',')]
            local = [int(x) for x in re.search(r'partition_row_indices = \[([^\]]+)\]',binding)[1].split(',')]
            # VXM writes on the opposite fabric; preserve that native result
            # home, including the mapping of unused rows to their padded head.
            assert sides[:16] == [1]*8+[0]*8, sides
            assert local[:16] == list(range(8))*2, local
        for tile_run in ingress:
            launch = int(tile_run['start'])
            square = [r for r in rows if r['resource'] == 'VXM.C1'
                      and r['detail'].startswith('mul ')
                      and launch < int(r['start']) < launch + 32*int(tile_run['inner_count'])]
            assert square, 'SXM ingress did not overlap VXM square reduction'
            first = min(int(r['start']) for r in square)
            assert first < launch + (int(tile_run['inner_count'])-1)*32
            active = set()
            for r in square:
                for outer in range(int(r['outer_count'])):
                    for inner in range(int(r['inner_count'])):
                        offset = outer*int(r['outer_interval'])+inner*int(r['inner_interval'])
                        active.update(range(int(r['start'])+offset,int(r['end'])+offset))
            assert set(range(first,first+config.key_head_dim)) <= active, 'Gaps in streamed square pass'
    else:
        assert not sxm_rows, sxm_rows[:3]
    log = (out / 'execution.log').read_text()
    assert 'sxm_architecture=wide' in log
    stats = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', log.splitlines()[-1])}
    assert stats.get('host_operations') == 0, stats
    if args.with_gates and args.log_softplus:
        assert any('log_lut' in r['detail'] for r in rows), 'Log must execute on VXM'
    cycles = max(int(r['end']) + (int(r['inner_count'])-1)*int(r['inner_interval'])
                 + (int(r['outer_count'])-1)*int(r['outer_interval']) for r in rows)
    hardware = json.loads(args.target_config.read_text())
    passed = all(e['nrmse'] < .03 and e['cosine'] > .999 for e in [metrics, *parts.values()])
    result = dict(status='passed' if passed else 'failed', batch=1,
        gate_softplus=('native_log_small_value_polynomial' if args.log_softplus else 'polynomial') if args.with_gates else None,
        gate_mapping='shared mirrored BF16-spill vector chains' if args.with_gates else None,
        state_decay=(dict(shape=list(state.shape), layout='head_value_by_key' if args.with_ks else 'head_key_by_value',
            nonzero_initial_state=True, recurrence=args.with_update, logical_bytes=state.nbytes,
            physical_order='key_block32, local_head, value_row; lane=key%32' if args.with_ks else 'value_block32, local_head, key_row; lane=value%32',
            hemisphere_policy='mirrored; head partition between hemispheres is not enabled',
            arithmetic='BF16-truncated chain inputs, FP32 multiply and output',
            ingress='synchronized runtime initialization, before compute',
            ingress_intervals=[[int(r['start']),int(r['end'])] for r in state_ingress],
            validation_outputs=state_indices, coefficient_broadcast='wide_sxm_repeat32_transpose_bf16_high_planes_per_tile' if args.with_ks else 'wide_sxm_repeat32_transpose') if args.with_state_decay else None),
        scope=('QKV projection + decode Conv1D/SiLU + Q/K L2Norm + beta/g + cache out' if args.with_gates else 'QKV projection + decode Conv1D/SiLU + Q/K L2Norm + cache out' if args.with_qk_norm else 'QKV projection + decode Conv1D/SiLU + cache out' if args.with_conv
               else 'QKV projection only') + (' + DDR state ingress + exp(g)*state' if args.with_state_decay else '') + (' + streamed KS' if args.with_ks else '') + (' + beta*(V-KS)' if args.with_correction else '') + (' + Sbar + delta outer K' if args.with_update else '; no state update') + (' + streamed Q readout' if args.with_readout else '; no query readout') + (' + Z projection + gated RMSNorm' if args.with_output_norm else '') + (' + output projection' if args.with_output_projection else '') + '; raw imported input, no input RMSNorm',
        ks_fusion=dict(enabled=True,route='VXM result -> MEM_WRITE_TAP -> MXM IW',
            scaled_state_mem_rereads=0,streaming=audit_streaming(out)) if args.with_ks else None,
        correction_fusion=dict(route='MXM result -> MEM_WRITE_TAP -> VXM C4-C7',
            state_chain='depth4 C0-C3',correction_chain='C4 subtract, C5 multiply local beta, C6 negate, C7 output',
            arithmetic='BF16-truncated KS/beta inputs; FP32 chained arithmetic and output',
            output_index=11,state_updated=args.with_update) if args.with_correction else None,
        rank_one_update=dict(output_index=12,route='wide SXM delta broadcast -> VXM multiply local K + auxiliary Sbar',
            state_layout='head,value,key; lane=key%32',outer_product_spills=0,
            broadcast_schedule='next head overlapped with current head update',
            broadcast_buffers=dict(slices=[50,51],banks=[1,0],rows_per_bank=config.value_head_dim,
                dtype='bf16',hemisphere_policy='mirrored',physical_bytes=2*2*config.value_head_dim*32*2),
            arithmetic='BF16-truncated delta/Sbar inputs, FP32 multiply then add and output') if args.with_update else None,
        query_readout=dict(output_index=13,route='VXM -> MEM_WRITE_TAP -> MXM IW; normalized scaled Q -> MXM activation',
            updated_state_mem_reads=0,arithmetic='BF16 high planes of updated FP32 state, BF16 Q, FP32 accumulation/output') if args.with_readout else None,
        gated_output_norm=dict(z_output=14,normalized_output=15,gated_output=16,
            shape=[config.num_value_heads,config.value_head_dim],epsilon=config.rms_norm_eps,
            gamma='delta_norm checkpoint weight, no +1 offset',output_projection=args.with_output_projection,
            gate_chain='C0 -z (capture normalized as auxiliary), C1 exp, C2 +1, C3 reciprocal, C4 *original z, C5 *auxiliary normalized, C6/C7 drain',
            gate_vectors_per_hemisphere=32,gate_activation_layout_copies=0,
            readout_cast='FP32 high-plane alias; BF16 truncation without copy',
            norm_output_ownership={'east':list(range(8)),'west':list(range(8,16))},
            gate_output_ownership={'west':list(range(8)),'east':list(range(8,16))}) if args.with_output_norm else None,
        output_projection=dict(output_index=17, input_shape=[1, config.value_width],
            weight_shape=list(packed['out_proj'].shape), output_shape=[1, config.hidden_size],
            lowering='shared C++ projection', mxm_execution='ws',
            checkpoint_weight='out_proj', input='gated RMSNorm result',
            activation_transport_domains=sum(t['phase'].startswith('out_proj') for t in plan['linker_mem_transports']),
            activation_transport_bytes=sum(t['vectors']*t['byte_planes'] for t in plan['linker_mem_transports']
                if t['phase'].startswith('out_proj')) * hardware['topology']['tiles_per_slice'] * hardware['topology']['lanes_per_tile']) if args.with_output_projection else None,
        input_shape=list(x.shape), output_shape=list(actual.shape),
        weight_shape=list(packed['in_proj_qkv'].shape), real_weights=True,
        weight_format='imported INT8 with tensor scale', icu_compression='raw-macro',
        sxm_architecture='wide', sxm_trace_rows=len(sxm_rows),
        norm_output_transport='VXM -> SR -> wide SXM; no normalized-output MEM spill' if args.with_qk_norm else None,
        sxm_validation='Only wide matrix transposes' if args.with_qk_norm else 'Hardware selected; this graph does not exercise transpose',
        head_partition=(dict(projection_conv_l2_input={'east':list(range(8)), 'west':list(range(8,16))},
            normalized_output={'west':list(range(8)), 'east':list(range(8,16))},
            missing_head_mirror_copies=0, scalar_return='passive SR after square reduction',
            layout_planning='C++ weight-column/view planning; no host activation reordering')
            if args.with_qk_norm else None),
        conv_cache_policy='nonzero BF16 history input and updated cache output' if args.with_conv else None,
        hardware=hardware, cycles=cycles,
        latency_ms=cycles/(hardware['external_memory']['lpu_clock_mhz']*1000),
        errors=metrics, component_errors=parts, runtime=stats)
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert passed, metrics


if __name__ == '__main__':
    main()
