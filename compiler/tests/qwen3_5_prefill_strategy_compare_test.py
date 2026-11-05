"""Compare GDN prefill cores with one immutable real-frontend fixture.

Only equations, test input serialization and independent references live here.
C++ owns all layouts, transfers, ICU generation and scheduling. This first
comparison intentionally supports only a single seq32 chunk with zero State.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16
from qwen3_5_decode_gdn_attention_test import run
from causal_exp_difference_runtime_test import causal_exp_source
from weighted_gram_runtime_test import scaled_weighted_gram_source, weighted_cross_gram_source
from unit_lower_solve_runtime_test import source as solve_source
from unit_lower_mxm_runtime_test import source as matrix_source
from head_weighted_product_runtime_test import source as product_source


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, expected):
    a, b = np.asarray(actual, 'f8').ravel(), np.asarray(expected, 'f8').ravel()
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    return dict(nrmse=float(np.linalg.norm(a-b) / max(bn, 1e-30)),
                max_abs=float(np.max(np.abs(a-b))),
                cosine=float(a @ b / (an*bn)) if an*bn else float(np.array_equal(a, b)))


def recurrent(fixture):
    """FP64 reference, no hardware rounding, semantic State [head,key,value]."""
    q, k, v, beta, g, state = [fixture[n].astype('f8') for n in ('q', 'k', 'v', 'beta', 'g', 'state')]
    ys = []
    for t in range(len(q)):
        state *= np.exp(g[t])[:, None, None]
        prediction = np.einsum('hkv,hk->hv', state, k[t])
        delta = beta[t, :, None] * (v[t]-prediction)
        state += k[t, :, :, None] * delta[:, None, :]
        ys.append(np.einsum('hkv,hk->hv', state, q[t]))
    return np.asarray(ys), state


def chunk_reference(fixture):
    """Independent exact chunk equations; guards masks, head and State axes."""
    q, k, v = [fixture[n].astype('f8').transpose(1, 0, 2) for n in ('q', 'k', 'v')]
    beta = fixture['beta'].astype('f8').T
    gt = fixture['g'].astype('f8').T.cumsum(axis=1)
    mask = np.tri(32, dtype=bool)
    r = np.exp(np.where(mask, gt[:, :, None]-gt[:, None, :], -np.inf))
    lower = np.tril(beta[:, :, None]*r*(k @ k.transpose(0, 2, 1)), -1)
    d = np.linalg.solve(np.eye(32)[None]+lower, beta[:, :, None]*v)
    y = (r*(q @ k.transpose(0, 2, 1))) @ d
    state = k.transpose(0, 2, 1) @ (r[:, -1, :, None]*d)
    return y.transpose(1, 0, 2), state


def freeze_fixture(source, folder):
    """Freeze known QKV/gates exports; never reconstruct g from exp(g)."""
    folder.mkdir(parents=True, exist_ok=True)
    status = json.loads((source/'summary.json').read_text())
    if status.get('status') != 'passed' or 'chunkwise_prefix' not in status:
        raise ValueError('Source must be a passed wide-SXM QKV/QK-norm/gates/chunkwise test')
    def read(index, shape, dtype):
        raw = np.fromfile(source/f'actual.{index}.bin', dtype)
        return ((raw.astype('u4') << 16).view('f4') if dtype == '<u2' else raw).reshape(shape)
    fixture = {n: read(i, (32, 16, 128), '<u2') for i, n in enumerate(('q', 'k', 'v'))}
    fixture.update(beta=read(3, (32, 16), '<f4'), g=read(4, (32, 16), '<f4'),
                   state=np.zeros((16, 128, 128), dtype='f4'))
    for n, a in fixture.items():
        assert np.isfinite(a).all(), n
        a.astype('<f4').tofile(folder/f'{n}.f32.bin')
    assert np.all(fixture['g'] <= 0) and np.all((fixture['beta'] >= 0) & (fixture['beta'] <= 1))
    metadata = dict(format='ftlpu.gdn.shared-core-fixture.v1', source=str(source.resolve()),
        source_files={f'actual.{i}.bin': digest(source/f'actual.{i}.bin') for i in range(5)},
        source_summary_sha256=digest(source/'summary.json'),
        initial_state='zero', conv_history='zero in source frontend', tokens=32, heads=16, key_dim=128, value_dim=128,
        inputs={n: dict(shape=list(a.shape), dtype='bf16 values in f32 container' if n in ('q','k','v') else 'f32',
                        file=f'{n}.f32.bin', sha256=digest(folder/f'{n}.f32.bin')) for n,a in fixture.items()},
        scope='GDN recurrence core only; frozen actual real-weight frontend outputs; no norm/gate/output-projection tail',
        precision=dict(qkv='identical BF16 bits', beta_g='identical FP32 bits', state='FP32, zero initially',
            vxm='full FP32 ingress/arithmetic', mxm='BF16 operands, FP32 accumulation',
            exp='same hardware Exp LUT, FP16 ingress/output and FTZ',
            output='BF16 RNE readout; FP32 final State',
            internal_difference='tokenwise MXM State high-plane truncation; chunkwise BF16 R/HD/weighted K and compensated FP32 triangular solve'))
    (folder/'manifest.json').write_text(json.dumps(metadata, indent=2))


def load_fixture(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    values = {}
    for name, item in manifest['inputs'].items():
        path = folder/item['file']
        assert digest(path) == item['sha256'], f'Fixture changed: {path}'
        values[name] = np.fromfile(path, '<f4').reshape(item['shape'])
    assert not np.any(values['state']), 'Nonzero State needs the three old-State terms in chunkwise'
    for name in ('q','k','v'):
        assert np.array_equal(values[name], bf16(values[name])), name
    return values, manifest


def build_core(graph, f, strategy):
    """Shared semantic boundary; native output views need no archive copy."""
    t,h,k = f['q'].shape
    d = f['v'].shape[-1]
    q,key,v = [f[n].reshape(t,-1) for n in ('q','k','v')]
    if strategy == 'tokenwise':
        factor = graph.vector_tiles('state_decay_factor', [(f['g'],'f32')],
                                    lambda a:a.add(0,kind='exp',immediate=0))
        c = SimpleNamespace(num_value_heads=h,key_head_dim=k,value_head_dim=d)
        y,s = graph.delta(q,key,v,factor,f['beta'],f['state'].reshape(-1,d),c)
        return [y, s.reshape(h,k,d).transpose(0,2,1).reshape(h*d,k)]
    gt = graph.vector_tiles('g_prefix_sum', [(f['g'],'f32')],
                            lambda a:a.add(0,kind='cumsum_rows',immediate=0))
    r = graph.run('chunk_causal_decay',causal_exp_source(t,h,True),[(gt,'f32')],[((t*h,t),'bf16')])[0]
    lower = graph.run('chunk_weighted_gram',scaled_weighted_gram_source(h,k),
        [(key,'bf16'),(r,'bf16'),(f['beta'],'f32'),(np.tri(t,k=-1,dtype='f4'),'bf16')],[((h*t,t),'f32')])[0]
    rhs = graph.run('chunk_rhs_beta_v',solve_source(h,d,'head_scale'),
        [(v,'bf16'),(f['beta'],'f32')],[((h*t,d),'f32')])[0]
    eye = np.broadcast_to(np.eye(t,dtype='f4'),(h,t,t)).copy().reshape(h*t,t)
    inverse = graph.run('chunk_unit_lower_inverse',matrix_source(h,t,'unit_lower_inverse'),
        [(lower,'f32'),(eye,'f32')],[((h*t,t),'f32')])[0]
    solved = graph.run('chunk_solve_matmul',matrix_source(h,d,'head_matmul'),
        [(inverse,'f32'),(rhs,'f32')],[((h*t,d),'f32')])[0]
    hh = graph.run('chunk_output_cross_gram',weighted_cross_gram_source(h,k),
        [(q,'bf16'),(key,'bf16'),(r,'bf16')],[((h*t,t),'f32')])[0]
    y,s = graph.run('chunk_output_state',product_source(h,k,d,True),
        [(key,'bf16'),(r,'bf16'),(solved,'f32'),(hh,'f32')],[((h*t,d),'f32'),((h*k,d),'f32')])
    # Cast in hardware, not in output postprocessing. Keep the native head ABI.
    y = graph.vector_tiles('readout_bf16_rne',[(y,'f32')],
                           lambda a:a.add(0,kind='cast',immediate=0,dtype='bf16'))
    return [y,s]


def union_size(spans):
    total, end = 0, -1
    for a,b in sorted(spans):
        total += max(0,b-max(a,end)); end=max(end,b)
    return total


def performance(out):
    stats = {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',(out/'execution.log').read_text().splitlines()[-1])}
    assert stats.get('host_operations') == 0 and stats.get('outputs') == 2, stats
    buckets = {'MXM':[], 'VXM':[]}
    per_unit = {}
    end = 0
    with (out/'runtime.pipeline.csv').open() as stream:
        for row in csv.DictReader(stream):
            a,b,ni,no,di,do = [int(row[n]) for n in ('start','end','inner_count','outer_count','inner_interval','outer_interval')]
            end=max(end,b+(ni-1)*di+(no-1)*do)
            resource=row['resource']
            kind='MXM' if resource.startswith('MXM.') and resource.endswith('.Compute') else (
                'VXM' if resource.startswith('VXM.') and 'bypass' not in row['detail'].lower() else None)
            if kind:
                spans=[(a+i*di+j*do,b+i*di+j*do) for j in range(no) for i in range(ni)
                       if j*ni+i >= int(row.get('skip_first',0))]
                buckets[kind].extend(spans); per_unit.setdefault(resource,[]).extend(spans)
    active={k:union_size(v) for k,v in buckets.items()}
    return dict(runtime=stats,trace_cycles=end,compute_timeline_cycles=stats['cycles'],
        denominator='logical execution cycles; excludes external blocked time and host input/output; includes on-chip transfers/conversions',
        active_union_cycles=active,active_union_percent={k:100*v/stats['cycles'] for k,v in active.items()},
        per_unit_active_cycles={k:union_size(v) for k,v in per_unit.items()},
        utilization_definition='time coverage of any MXM Compute / any non-Bypass VXM issue; not lane occupancy or peak FLOPS')


def validate(out, fixture, manifest, strategy):
    t,h,k = fixture['q'].shape; d=fixture['v'].shape[-1]
    y=(np.fromfile(out/'actual.0.bin','<u2').astype('u4')<<16).view('f4')
    s=np.fromfile(out/'actual.1.bin','<f4')
    if strategy=='tokenwise':
        y=y.reshape(t,h,d); s=s.reshape(h,d,k).transpose(0,2,1)
    else:
        y=y.reshape(2,h//2,t,d).transpose(2,1,0,3).reshape(t,h,d)
        s=s.reshape(2,h//2,k,d).transpose(1,0,2,3).reshape(h,k,d)
    expected=recurrent(fixture)
    algebra=chunk_reference(fixture)
    for a,b in zip(expected,algebra):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
    checks=dict(output=metrics(y,expected[0]),state=metrics(s,expected[1]))
    heads=[dict(output=metrics(y[:,i],expected[0][:,i]),state=metrics(s[i],expected[1][i])) for i in range(h)]
    passed=all(e['nrmse']<.03 and e['cosine']>.999 for row in [checks]+heads for e in row.values())
    plan=json.loads((out/'graph_plan.json').read_text())
    runtime_program=json.loads((out/'runtime_icu_programs/program.json').read_text())
    assert runtime_program['common_barrier_protocol'] and not runtime_program['transport_overlap_protocol']
    if strategy=='tokenwise':assert sum(n['name'].startswith('delta_rule.token') for n in plan['nodes'])==t
    else:assert any(n['name']=='chunk_unit_lower_inverse' for n in plan['nodes'])
    result=dict(status='passed' if passed else 'failed',strategy=strategy,case='seq32_zero_state',
        fixture=manifest,precision=manifest['precision'],scope=manifest['scope'],
        errors=checks,per_head=heads,thresholds=dict(nrmse_max=.03,cosine_min=.999),
        performance=performance(out),native_output_layout='token/head/value; State head/value/key' if strategy=='tokenwise' else 'even heads then odd heads',
        runtime_program=runtime_program,
        linker_mem_transport_domains=len(plan.get('linker_mem_transports',[])))
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    y.astype('<f4').tofile(out/'readout.canonical.f32.bin');s.astype('<f4').tofile(out/'state.canonical.f32.bin')
    print(json.dumps(dict(strategy=strategy,status=result['status'],errors=checks,performance=result['performance']),indent=2))
    assert passed, checks
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('build-dir','target-config','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-frontend',type=Path)
    p.add_argument('--strategy',choices=('tokenwise','chunkwise','both'),default='both')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--compile-only',action='store_true')
    modes.add_argument('--run-only',action='store_true')
    modes.add_argument('--validate-only',action='store_true')
    a=p.parse_args(); root=a.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    (root/'comparison.json').write_text(json.dumps(dict(status='pending',requested_strategy=a.strategy)))
    (root/'comparison.md').write_text('# GDN prefill comparison\n\nValidation is pending.\n')
    fixture_dir=root/'fixture'
    if not (fixture_dir/'manifest.json').exists():
        if not a.source_frontend:p.error('--source-frontend required to create the immutable fixture')
        freeze_fixture(a.source_frontend.resolve(),fixture_dir)
    fixture,manifest=load_fixture(fixture_dir)
    build=a.build_dir.resolve();exe=lambda area,n:build/area/(n+('.exe' if sys.platform=='win32' else ''))
    target=a.target_config.resolve()
    strategies=('tokenwise','chunkwise') if a.strategy=='both' else (a.strategy,)
    for strategy in strategies:
        out=root/strategy;out.mkdir(parents=True,exist_ok=True)
        binary,kernel=out/'program.ftlpu',out/'kernel.mlir'
        if not (a.validate_only or a.run_only):
            (out/'summary.json').write_text(json.dumps(dict(status='compiling',strategy=strategy)))
            graph=LogicalGraph();finals=build_core(graph,fixture,strategy)
            logical=graph.write(out/'logical',finals)
            run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel','--target-config',target],out/'import.log')
            attrs='ftlpu.vector_fp32_ingress = true, ftlpu.vector_bf16_spill = true, ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true, ftlpu.common_transport_overlap = false'
            if strategy=='tokenwise':attrs+=', ftlpu.gdn_prefill_strategy = "tokenwise"'
            kernel.write_text(kernel.read_text().replace('module {','module attributes {'+attrs+'} {',1))
            run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,'--target-config',target,'--icu-compression','raw-macro','--mxm-execution','ws'],out/'compile.log')
            (out/'run_contract.json').write_text(json.dumps(dict(fixture_sha256=digest(fixture_dir/'manifest.json'),
                target_sha256=digest(target),compiler_sha256=digest(exe('compiler','ftlpu-compile')),
                runner_sha256=digest(exe('runtime','compiled_stage_file_runner')),binary_sha256=digest(binary),
                strategy=strategy,icu_compression='raw-macro',sxm='wide',mxm_execution='ws',transport_overlap=False),indent=2))
        if a.compile_only:continue
        contract=json.loads((out/'run_contract.json').read_text())
        assert contract['fixture_sha256']==digest(fixture_dir/'manifest.json')
        assert contract['binary_sha256']==digest(binary)
        if not a.validate_only:
            (out/'summary.json').write_text(json.dumps(dict(status='running',strategy=strategy)))
            assert contract['runner_sha256']==digest(exe('runtime','compiled_stage_file_runner')), 'Runner changed; recompile comparison'
            os.environ.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',FTLPU_SESSION_PROGRESS='1')
            run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
            run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs','--pre-execution-dir',out/'actual.pre_execution'],out/'export.log')
        validate(out,fixture,manifest,strategy)
    if all((root/s/'summary.json').exists() and json.loads((root/s/'summary.json').read_text()).get('status')=='passed' for s in ('tokenwise','chunkwise')):
        summaries={s:json.loads((root/s/'summary.json').read_text()) for s in ('tokenwise','chunkwise')}
        contracts={s:json.loads((root/s/'run_contract.json').read_text()) for s in summaries}
        for key in ('fixture_sha256','target_sha256','compiler_sha256','runner_sha256'):
            assert contracts['tokenwise'][key]==contracts['chunkwise'][key],key
        report=dict(status='passed',case='seq32_zero_state',scope=manifest['scope'],precision=manifest['precision'],
            contracts=contracts,
            strategies={s:dict(errors=r['errors'],performance=r['performance']) for s,r in summaries.items()},
            tokenwise_to_chunkwise_cycle_ratio=summaries['tokenwise']['performance']['compute_timeline_cycles']/summaries['chunkwise']['performance']['compute_timeline_cycles'],
            cross_strategy={n:metrics(np.fromfile(root/'tokenwise'/f'{n}.canonical.f32.bin','<f4'),np.fromfile(root/'chunkwise'/f'{n}.canonical.f32.bin','<f4')) for n in ('readout','state')},
            pending=['nonzero initial State comparison','continuous multi-chunk comparison','complete attention comparison'])
        (root/'comparison.json').write_text(json.dumps(report,indent=2))
        lines=['# GDN prefill: shared seq32 zero-State core comparison','',
            'Same frozen real-weight frontend outputs: Q/K/V BF16; beta/g FP32; State FP32.',
            'Both return BF16 RNE readout and FP32 final State. No attention tail or FFN is included.','',
            '| Strategy | Logical cycles | External wait cycles | MXM time coverage | VXM time coverage | Readout NRMSE | State NRMSE |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for s,r in summaries.items():
            pf=r['performance']; e=r['errors']; u=pf['active_union_percent']
            lines.append(f"| {s} | {pf['compute_timeline_cycles']:,} | {pf['runtime'].get('weight_page_runtime_wait_cycles',0):,} | {u['MXM']:.2f}% | {u['VXM']:.2f}% | {100*e['output']['nrmse']:.4f}% | {100*e['state']['nrmse']:.4f}% |")
        lines += ['',f"Tokenwise / chunkwise logical-cycle ratio: {report['tokenwise_to_chunkwise_cycle_ratio']:.3f}x.",'',
            'Utilization is unit-activity time coverage, not lane occupancy; denominator excludes external blocked time.',
            'On-chip layout conversion and transfer costs are included. Cold input/output traffic is visible in each runtime pipeline.',
            'Different internal rounding boundaries remain: FP32 storage does not imply all MXM products are FP32.',
            'Both global and per-head checks passed against the same FP64 recurrence reference (NRMSE < 3%, cosine > 0.999).','',
            'Pending: nonzero State, continuous chunks, and end-to-end attention under the same contract.','']
        (root/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
