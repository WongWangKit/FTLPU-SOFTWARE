"""GLM-4-9B decode, layer 0, past32: input RMSNorm through O projection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sys
import tempfile

import numpy as np
import glm4_decode_test as glm
from activation_batch_matmul_test import compile_graph, run
from glm4_flash_packed_u_test import graph_digest, sha
from import_hf_decoder_layer import (
    bf16, biased_linear, linear, lpu_softmax, normalized_rope_linear_at_positions,
    rms_norm, rope_parameters, rope_theta_value,
)
from summarize_glm4_flash import phase_timing


def load_fixture(args):
    batch = args.batch
    reference = args.flash_results/f'batch{batch}'
    signature = json.loads((args.flash_results/'signature.json').read_text())
    original = json.loads((reference/'result.json').read_text())
    assert original['status'] == 'passed' and original['attention_strategy'] == 'flash'
    assert original['batch'] == batch and original['past_len'] == 32
    meta = json.loads((args.imported/'metadata.json').read_text())
    assert sha(args.imported/'metadata.json') == signature['metadata']
    assert sha(args.checkpoint/'config.json') == original['checkpoint_provenance']['config_sha256']
    assert (meta['model_type'],meta['layer'],meta['query_heads'],meta['kv_heads'],meta['head_dim']) == ('chatglm',0,32,2,128)
    config = glm.decoder_config(json.loads((args.checkpoint/'config.json').read_text()))
    config['head_dim'] = meta['head_dim']
    widths = dict(query=4096, key=256, value=256, output=4096)
    weights = {role: np.fromfile(args.imported/f'{role}.i8.bin','i1').reshape(4096,width)
               for role,width in widths.items()}
    biases = {role: glm.read(args.imported/f'{role}_bias.bf16.bin',(widths[role],))
              for role in ('query','key','value')}
    norm = glm.read(args.imported/'input_layernorm.bf16.bin',(4096,))
    token_ids = json.loads((reference/'token_ids.json').read_text())
    x = glm.read(args.imported/'input.bf16.bin',(32,4096))[token_ids['current']]
    assert x.shape == (batch,4096)
    # Check inputs against the actual data used in the accepted split execution.
    def constants(phase):
        root = reference/phase/'logical'
        graph = json.loads((root/'decoder_layer.graph.json').read_text())
        return {node['name']: [root/PureWindowsPath(graph['constants'][arg['constant']]['file']).name
                               if 'constant' in arg else None for arg in node['inputs']]
                for node in graph['nodes']}
    proj = constants('projections')
    assert np.array_equal(x.view('u4'),glm.read(proj['input_rmsnorm'][0],x.shape).view('u4'))
    assert sha(proj['input_rmsnorm'][1]) == sha(args.imported/'input_layernorm.bf16.bin')
    for role in biases:
        assert sha(proj[role+'_projection'][1]) == sha(args.imported/f'{role}.i8.bin')
        assert sha(proj[role+'_bias'][1]) == sha(args.imported/f'{role}_bias.bf16.bin')
    assert sha(constants('tail')['o_projection'][1]) == sha(args.imported/'output.i8.bin')
    with np.load(reference/'reference.npz',allow_pickle=False) as fixture:
        pk,pv,ek,ev = (fixture[key] for key in ('past_key','past_value','expected_key','expected_value'))
        fingerprint = str(fixture['fingerprint'])
    assert pk.shape == pv.shape == (batch,32,2,128) and ek.shape == ev.shape == (batch,33,2,128)
    return dict(x=x,past_key=pk,past_value=pv,expected_key=ek,expected_value=ev,weights=weights,
                biases=biases,norm=norm,config=config,scales=meta['scales'],
                provenance=dict(fixture_fingerprint=fingerprint,token_ids_sha256=sha(reference/'token_ids.json'),
                    hidden_sha256=sha(proj['input_rmsnorm'][0]),metadata_sha256=signature['metadata'],
                    config_sha256=sha(args.checkpoint/'config.json'),
                    past_kv_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in (pk,pv)],
                    weights_sha256={r:sha(args.imported/f'{r}.i8.bin') for r in widths},
                    original_target_sha256=signature['target'],original_runner_sha256=signature['runner']))


def reference_output(f):
    x,config,weights,biases,scales = (f[k] for k in ('x','config','weights','biases','scales'))
    normalized = rms_norm(x,f['norm'],config['rms_norm_eps'])
    inverse,amp,rotary = rope_parameters(config,33)
    positions = np.full(len(x),32,dtype='f4')
    def projection(role,heads):
        return normalized_rope_linear_at_positions(normalized,weights[role],scales[role],biases[role],
            heads,rope_theta_value(config),positions,epsilon=config['rms_norm_eps'],
            inverse_frequencies=inverse,attention_factor=amp,rotary_dim=rotary)
    q = projection('query',32)
    k = projection('key',2)
    v = biased_linear(normalized,weights['value'],scales['value'],biases['value']).reshape(-1,2,128)
    keys = np.concatenate((f['past_key'],k[:,None]),axis=1)
    values = np.concatenate((f['past_value'],v[:,None]),axis=1)
    score = bf16(q.reshape(-1,2,16,128) @ keys.transpose(0,2,3,1))
    probability = lpu_softmax(score,128,causal=False)
    context = bf16((probability @ values.transpose(0,2,1,3)).reshape(-1,4096))
    return linear(context,weights['output'],scales['output'],bf16_scale=True), context


def execute(args,name,graph_outputs,folder,record):
    graph,outputs = graph_outputs
    print(f'batch{args.batch} {name}: compile',flush=True)
    # Preserve the accepted projection/RoPE BF16 RNE boundary. The isolated
    # Flash graph uses its original packed FP32 m/l/U spill policy.
    full_spills = name=='attention'
    exe,env,binary = compile_graph(graph,outputs,folder,args.build_dir,args.target_config,
        fp32_spills=full_spills,module_attrs=[
            f'ftlpu.weight_paging_policy = "{args.weight_paging}"',
            f'ftlpu.passive_sr_multicast = {str(args.passive_sr_multicast == "on").lower()}'])
    print(f'batch{args.batch} {name}: execute raw macro',flush=True)
    run([exe('runtime','compiled_stage_file_runner'),binary,folder/'inputs.txt',folder/'actual',
         folder/'runtime.pipeline.csv'],folder/'execution.log',env)
    run([exe('runtime','ftlpu_icu_program_export'),folder/'actual.runtime.ftlpu',folder/'icu_programs',
         '--pre-execution-dir',folder/'actual.pre_execution'],folder/'export.log',env)
    log = (folder/'execution.log').read_text(errors='replace')
    assert re.search(r'^cycles=.*\bhost_operations=0\b',log,re.M), 'expected device-only execution'
    assert record['compiler_sha256'] == sha(exe('compiler','ftlpu-compile'))
    stamp = dict(binary_sha256=sha(binary),compiler_sha256=record['compiler_sha256'],
                 semantic_graph_sha256=graph_digest(folder))
    (folder/'completed.json').write_text(json.dumps(stamp))
    record['phases'][name] = dict(**stamp,vector_bf16_rne_spill=not full_spills,
        actual_program_sha256=sha(folder/'actual.runtime.ftlpu'),
        runtime_pipeline_sha256=sha(folder/'runtime.pipeline.csv'),timing=phase_timing(folder,record['clock_mhz']),
        output_sha256=[sha(folder/f'actual.{i}.bin') for i in range(len(outputs))])
    return [glm.read(folder/f'actual.{i}.bin',out.shape) for i,out in enumerate(outputs)]


def resume_projections(args,graph_outputs,folder,record,prior):
    """Reuse a completed split invocation only after checking its input graph and device outputs."""
    graph,outputs=graph_outputs
    if prior is None or any(prior.get(key)!=value for key,value in record.items()
                            if key not in ('status','phases')):
        return None
    phase=prior.get('phases',{}).get('projections')
    if not phase:
        return None
    try:
        stamp=json.loads((folder/'completed.json').read_text())
        assert phase['compiler_sha256']==stamp['compiler_sha256']==record['compiler_sha256']
        assert phase['binary_sha256']==stamp['binary_sha256']==sha(folder/'program.ftlpu')
        assert phase['actual_program_sha256']==sha(folder/'actual.runtime.ftlpu')
        assert phase['runtime_pipeline_sha256']==sha(folder/'runtime.pipeline.csv')
        assert phase['timing']==phase_timing(folder,record['clock_mhz'])
        assert phase['vector_bf16_rne_spill'] is True
        queue_dir=folder/'icu_programs'
        with (queue_dir/'index.csv').open(newline='') as data:
            queues=list(csv.DictReader(data))
        assert queues and all((queue_dir/queue['file']).stat().st_size for queue in queues)
        assert re.search(r'^cycles=.*\bhost_operations=0\b',(folder/'execution.log').read_text(),re.M)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            graph.write(Path(tmp)/'logical',outputs)
            assert graph_digest(Path(tmp))==graph_digest(folder)==stamp['semantic_graph_sha256']==phase['semantic_graph_sha256']
        archived=args.flash_results/f'batch{args.batch}/projections'
        assert not any(projection_snapshot_comparison(folder,archived)['bitwise_mismatches'].values())
        actual=[glm.read(folder/f'actual.{i}.bin',out.shape) for i,out in enumerate(outputs)]
        hashes=[sha(folder/f'actual.{i}.bin') for i in range(len(outputs))]
        assert phase.get('output_sha256',hashes)==hashes
    except (OSError,KeyError,ValueError,AssertionError) as exc:
        print(f'Cannot reuse completed projections ({type(exc).__name__}: {exc}); executing them again.',flush=True)
        return None
    record['phases']['projections']=dict(phase,output_sha256=hashes,reused_verified_execution=True)
    print(f'batch{args.batch} projections: reuse verified raw-macro execution',flush=True)
    return actual


def summarize_phases(folder,record):
    """Combine trace windows once and keep hardware paging counters alongside them."""
    fields=('execution_epochs','weight_page_prefetches','weight_page_prefetch_bytes',
            'weight_page_runtime_wait_cycles','synchronized_writes','host_operations')
    counters=dict.fromkeys(fields,0)
    windows=dict.fromkeys(('initial_ingress_window_cycles','device_execution_window_cycles',
        'output_readback_including_diagnostics_cycles','other_timeline_cycles','runtime_cycles'),0)
    static_cycles=0
    for name,phase in record['phases'].items():
        phase_dir=folder if record['mode']=='fused' else folder/name
        log=(phase_dir/'execution.log').read_text(errors='replace')
        final=re.search(r'^cycles=\d+ .*$',log,re.M)
        if not final:
            raise ValueError(f'missing runtime statistics in {phase_dir}')
        values={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',final[0])}
        timing=phase['timing']
        assert values['cycles']==timing['static_schedule_cycles']
        assert values['weight_page_runtime_wait_cycles']==timing['weight_page_wait_cycles']
        assert values['host_operations']==0
        phase['runtime_counters']={key:values[key] for key in fields}
        for key in fields:
            counters[key]+=values[key]
        for key in windows:
            windows[key]+=timing[key]
        static_cycles+=values['cycles']
    assert windows['runtime_cycles']==sum(value for key,value in windows.items() if key!='runtime_cycles')
    assert record['runtime_cycles']==windows['runtime_cycles']
    record['timing_summary']=dict(**windows,static_schedule_cycles=static_cycles,**counters)
    record['timing_notes']=[
        'Runtime cycles include ingress, device execution (including stalls), and output transfers.',
        'The runner prints static program max_cycle+128 as cycles. With common barriers its weight_page_runtime_wait_cycles counter is physical minus logical invocation cycles.',
        'The wait counter is already inside the device execution window. Output transfers include intermediate tensors at staged invocation boundaries and the final projected output and KV cache.'
    ]


def projection_snapshot_comparison(current, archived):
    stamp=json.loads((archived/'completed.json').read_text())
    assert stamp['binary_sha256']==sha(archived/'program.ftlpu')
    differences={}
    for i,role in enumerate(('query','key','value')):
        output=f'actual.{i}.bin'
        a=np.fromfile(current/output,'<u2')
        b=np.fromfile(archived/output,'<u2')
        assert a.shape==b.shape
        differences[role]=int(np.count_nonzero(a!=b))
    # Older accepted phase stamps recorded the binary without its compiler hash.
    return dict(archived_compiler_sha256=stamp.get('compiler_sha256'),archived_program_sha256=stamp['binary_sha256'],
                bitwise_mismatches=differences)


def run_mode(args,mode,f,oracle,root):
    folder=root/mode
    folder.mkdir(parents=True,exist_ok=True)
    ext='.exe' if sys.platform=='win32' else ''
    record=dict(status='running',mode=mode,model='GLM-4-9B W8A16',layer=0,batch=args.batch,past_len=32,
        decode_tokens_per_request=1,scope='raw hidden, RMSNorm, QKV, biases, partial RoPE, FlashAttention, O projection',
        outputs=['projected','present_key','present_value'],icu_encoding='raw-macro',sxm='wide',
        weight_paging_policy=args.weight_paging,passive_sr_multicast=args.passive_sr_multicast,phases={},
        compiler_sha256=sha(args.build_dir/'compiler'/('ftlpu-compile'+ext)),
        runner_sha256=sha(args.build_dir/'runtime'/('compiled_stage_file_runner'+ext)),
        target_sha256=sha(args.target_config),clock_mhz=json.loads(args.target_config.read_text())['external_memory']['lpu_clock_mhz'],
        **f['provenance'])
    result=folder/'result.json'
    prior=json.loads(result.read_text()) if args.resume_projections and mode=='staged' and result.exists() else None
    result.write_text(json.dumps(record,indent=2))
    try:
        # The archived runner identifies the input fixture. The two modes of
        # this benchmark must use the same current runner (checked in compare).
        # A change of runtime alone is measured by the frozen-program replay.
        assert record['target_sha256'] == record['original_target_sha256']
        x,w,s,b,pk,pv,c,n=(f[k] for k in ('x','weights','scales','biases','past_key','past_value','config','norm'))
        projected_ref,context_ref=oracle
        if mode=='fused':
            produced=glm.full_attention(x,w,s,b,pk,pv,c,norm0=n)
            y,key,value=execute(args,'full_attention',produced,folder,record)
            output_paths=[folder/f'actual.{i}.bin' for i in range(3)]
        else:
            projection_graph=glm.projections(x,w,s,b,n,c,32)
            saved=resume_projections(args,projection_graph,folder/'projections',record,prior)
            q,k,v=(saved if saved is not None else
                   execute(args,'projections',projection_graph,folder/'projections',record))
            record['projection_snapshot_comparison']=projection_snapshot_comparison(folder/'projections',
                args.flash_results/f'batch{args.batch}/projections')
            core=glm.attention(q,k,v,pk,pv,c,strategy='flash')
            context,key,value=execute(args,'attention',core,folder/'attention',record)
            record['context_metrics']=glm.errors(context,context_ref)
            assert record['context_metrics']['relative_l2']<.03,record['context_metrics']
            y,=execute(args,'output_projection',glm.output_projection(context,w,s),folder/'output_projection',record)
            output_paths=[folder/'output_projection/actual.0.bin',folder/'attention/actual.1.bin',folder/'attention/actual.2.bin']
            paths=[(phase,folder/phase/'runtime.pipeline.csv') for phase in record['phases']]
            combined_cycles=glm.concatenate_traces(paths,folder/'runtime.pipeline.csv')
            assert combined_cycles==sum(p['timing']['runtime_cycles'] for p in record['phases'].values())
        key=key.reshape(args.batch,33,2,128)
        value=value.reshape(key.shape)
        record['projected_metrics']=glm.errors(y,projected_ref)
        record['requests']=[glm.errors(out,ref) for out,ref in zip(y,projected_ref)]
        record['worst_request_relative_l2']=max(r['relative_l2'] for r in record['requests'])
        record['new_cache_metrics']={name:glm.errors(a[:,32],ref[:,32]) for name,a,ref in
            [('key',key,f['expected_key']),('value',value,f['expected_value'])]}
        record['historical_kv_bit_exact']=all(np.array_equal(a[:,:32].view('u4'),orig.view('u4'))
            for a,orig in ((key,pk),(value,pv)))
        record['output_files']=[str(p.relative_to(folder)) for p in output_paths]
        record['output_sha256']=[sha(p) for p in output_paths]
        record['runtime_cycles']=sum(p['timing']['runtime_cycles'] for p in record['phases'].values())
        record['runtime_ms']=record['runtime_cycles']/(record['clock_mhz']*1000)
        summarize_phases(folder,record)
        assert record['historical_kv_bit_exact']
        assert all(m['relative_l2']<.03 for m in record['new_cache_metrics'].values()),record['new_cache_metrics']
        assert np.isfinite(y).all() and record['projected_metrics']['relative_l2']<.03,record['projected_metrics']
        assert record['worst_request_relative_l2']<.08,record['worst_request_relative_l2']
        record['status']='passed'
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.write_text(json.dumps(record,indent=2))
    print(f'{result}: passed, {record["runtime_cycles"]} cycles',flush=True)


def compare(root,fixture_root):
    records=[json.loads((root/mode/'result.json').read_text()) for mode in ('staged','fused')]
    staged,fused=records
    assert all(r['status']=='passed' for r in records)
    staged['projection_snapshot_comparison']=projection_snapshot_comparison(root/'staged/projections',
        fixture_root/f'batch{staged["batch"]}/projections')
    for r in records:
        mode_dir=root/r['mode']
        summarize_phases(mode_dir,r)
        (mode_dir/'result.json').write_text(json.dumps(r,indent=2))
    for k in ('batch','compiler_sha256','runner_sha256','target_sha256','fixture_fingerprint','metadata_sha256',
              'weights_sha256','past_kv_sha256','hidden_sha256','config_sha256','clock_mhz','weight_paging_policy',
              'passive_sr_multicast'):
        assert staged[k]==fused[k],f'different {k}'
    differences={}
    for i,name in enumerate(staged['outputs']):
        arrays=[]
        for r in records:
            path=root/r['mode']/r['output_files'][i]
            assert sha(path)==r['output_sha256'][i]
            arrays.append(np.fromfile(path,'<u2'))
        assert arrays[0].shape==arrays[1].shape
        differences[name]=int(np.count_nonzero(arrays[0]!=arrays[1]))
    before,after=(r['runtime_cycles'] for r in records)
    result=dict(status='failed' if any(differences.values()) else 'passed',batch=fused['batch'],
        scope=fused['scope'],staged_runtime_cycles=before,fused_runtime_cycles=after,cycles_saved=before-after,
        speedup=before/after,bitwise_mismatches=differences,compiler_sha256=fused['compiler_sha256'],
        staged_timing=staged['timing_summary'],fused_timing=fused['timing_summary'],
        projection_snapshot_comparison=staged['projection_snapshot_comparison'])
    (root/'result.json').write_text(json.dumps(result,indent=2))
    assert result['status']=='passed',differences
    print(result,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,choices=(1,32),required=True)
    parser.add_argument('--mode',choices=('both','fused','staged','compare'),default='both')
    parser.add_argument('--resume-projections',action='store_true',
                        help='reuse split QKV when the same-build completed run and every archived word match')
    parser.add_argument('--weight-paging',choices=('auto','serialized'),default='auto',
                        help='apply this compiler weight paging policy in every invocation')
    parser.add_argument('--passive-sr-multicast',choices=('off','on'),default='on',
                        help='share inward SR reads between mirrored MEM destinations using WRITE_TAP')
    parser.add_argument('--build-dir',type=Path)
    parser.add_argument('--target-config',type=Path,default=Path('../FTLPU-CMODEL/config/ftlpu-lpu32.json'))
    parser.add_argument('--checkpoint',type=Path,default=Path('models/official_dense/glm4_9b'))
    parser.add_argument('--imported',type=Path,default=Path('tmp/model_tests/glm4_9b/golden'))
    parser.add_argument('--flash-results',type=Path,default=Path('results/glm4_9b_decoder_layer_decode_flash'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/glm4_9b_decoder_layer_decode_flash_projected/optimized'))
    args=parser.parse_args()
    root=args.output_dir.resolve()/f'batch{args.batch}'
    if args.mode=='compare':
        compare(root,args.flash_results)
        return
    if args.build_dir is None:
        parser.error('--build-dir required for execution')
    args.build_dir=args.build_dir.resolve()
    args.target_config=args.target_config.resolve()
    if root.is_relative_to(args.flash_results.resolve()):
        parser.error('select an output directory outside the fixture root')
    fixture=load_fixture(args)
    oracle=reference_output(fixture)
    for mode in (('staged','fused') if args.mode=='both' else (args.mode,)):
        run_mode(args,mode,fixture,oracle,root)
    if args.mode=='both':
        compare(root,args.flash_results)


if __name__=='__main__':
    main()
