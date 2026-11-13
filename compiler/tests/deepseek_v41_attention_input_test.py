"""Layer-0 hc_pre interface -> shared attention input/QKV/RoPE lowering.

Python supplies logical tensors and independent references only. The returned
normalized tensor is a diagnostic checkpoint, not a host input to projection.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from deepseek_v41_swa_test import Model
from deepseek_v41_qkv_norm_test import run, trace_times, bf16_input
from deepseek_v41_qkv_reference import REVISION, tensor, project_bf16_activation, rope
from import_hf_decoder_layer import bf16, bf16_bits, read_bf16
from deepseek_v41_native_rope_test import validate_rotary_pipeline
from deepseek_v41_swa_core_test import reference as swa_reference


def metrics(actual, expected):
    a=actual.astype('f8').ravel();b=expected.astype('f8').ravel()
    return dict(relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                cosine=float(np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),
                finite=bool(np.isfinite(a).all()),max_absolute_error=float(np.max(np.abs(a-b))))


def check_icu_export(directory):
    """Validate every loaded image, including the MEM transport store."""
    counts = {}
    images = {}
    with (directory / 'index.csv').open(newline='') as stream:
        index = list(csv.DictReader(stream))
    for entry in index:
        with (directory / entry['file']).open(newline='') as stream:
            rows = list(csv.DictReader(line for line in stream if not line.startswith('#')))
        word_bytes = 16 if entry['resource'].startswith('mxm_') else 12
        groups = {}
        native_images = set()
        for row in rows:
            if 'encoding=packed_macro' in row['details']:
                counts[row['opcode']] = counts.get(row['opcode'], 0) + 1
                assert row['pc_word'] == row['imem_words'] == ''
                assert row['raw_words'].startswith('file:')
                groups.setdefault(row['raw_words'][5:], []).append(row)
            else:
                match = re.search(r'(?:^| )imem_file=(\S+)', row['details'])
                assert match, ('missing physical image', entry, row)
                native_images.add(match.group(1))
        loaded_words = 0
        for name in set(groups) | native_images:
            image = (directory / name).read_bytes()
            assert len(image) % word_bytes == 0
            words = len(image) // word_bytes
            loaded_words += words
            if name in groups:
                control = int.from_bytes(image[:4], 'little')
                assert control & 15 == 7
                assert (control >> 4) & 0xffffff == len(groups[name])
            images[name] = dict(bytes=len(image), physical_words=words,
                                decoded_domains=len(groups.get(name, [])))
        assert loaded_words == int(entry['imem_words']), (entry, loaded_words)
    assert counts.get('MXM_LOAD_MACRO', 0) > 0 and counts.get('MXM_COMPUTE_MACRO', 0) > 0
    result = dict(status='passed', opcode_domains=counts, physical_images=images)
    (directory / 'macro_export_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','package','norm-package','model-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--validate-existing',action='store_true',help='Recheck an existing actual run only after all source/binary/target/executable hashes match')
    p.add_argument('--with-qkv-norm',action='store_true',help='Append the two real post-projection RMSNorm operators')
    p.add_argument('--with-qb-rope',action='store_true',help='Also append Q_B and Q/KV partial RoPE; implies --with-qkv-norm')
    p.add_argument('--rope-tile-overlap',action='store_true',help='Overlap the wide SXM tiles with the VXM RoPE computation')
    p.add_argument('--with-swa-core',action='store_true',help='Append shared planar causal SWA with sink; stops before inverse RoPE')
    p.add_argument('--swa-package',type=Path,help='Imported layer-0 sink provenance and weights')
    p.add_argument('--allow-layout-copies',action='store_true',help='Record the unoptimized layout baseline; numerical limits stay unchanged')
    a=p.parse_args();a.with_qb_rope|=a.with_swa_core;a.with_qkv_norm|=a.with_qb_rope
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((out/'summary.json').read_text()) if a.validate_existing else None
    pkg=a.package.resolve();norm_pkg=a.norm_package.resolve()
    meta=json.loads((pkg/'weights/provenance.json').read_text())
    norm_meta=json.loads((norm_pkg/'weights/provenance.json').read_text())
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert meta['revision']==norm_meta['revision']==REVISION
    assert meta['config_sha256']==norm_meta['config_sha256']==digest(a.model_config)
    config=json.loads(a.model_config.read_text())['text_config'];hidden=config['hidden_size']
    assert hidden==5120
    epsilon=config['rms_norm_eps'];x=np.load(pkg/'reference/x.npy',allow_pickle=False)
    bf16_input(x,(32,hidden))
    gamma=tensor(norm_pkg/'weights',norm_meta,'attn_norm','BF16',(hidden,))
    model=Model(out,quantize_activation=False);model.input('hc_pre_output',x);model.input('gamma',gamma)
    y=model.op('attention_rmsnorm','rms_norm',['hc_pre_output','gamma'],x.shape,
               f'axis=1:i64,epsilon={epsilon:.9e}:f32')
    xd=x.astype('f8')
    normalized=bf16((xd/np.sqrt(np.mean(xd*xd,axis=1,keepdims=True)+epsilon)*gamma).astype('f4'))
    outputs=[y];gold=[normalized];weights=[];norm_weights=[]
    for name,role,width in [('q_a','wq_a',1280),('kv_raw','wkv',512)]:
        w=tensor(pkg/'weights',meta,role,'F8_E4M3',(width,hidden))
        s=tensor(pkg/'weights',meta,role+'_scale','F8_E8M0',(width//32,hidden//32))
        model.input(role,w,'f8E4M3FN');model.input(role+'_scale',s,'f8E8M0FNU')
        outputs.append(model.projection(name,y,role,role+'_scale',32,width))
        gold.append(project_bf16_activation(normalized,w,s));weights.append((w,s))
    if a.with_qkv_norm:
        for i,(name,width) in enumerate([('q_norm',1280),('kv_norm',512)],1):
            g=tensor(pkg/'weights',meta,name,'BF16',(width,));norm_weights.append(g)
            model.input(name+'_gamma',g)
            outputs.append(model.op(name,'rms_norm',[outputs[i],name+'_gamma'],(32,width),
                                    f'axis=1:i64,epsilon={epsilon:.9e}:f32'))
            v=gold[i].astype('f8')
            gold.append(bf16((v/np.sqrt(np.mean(v*v,axis=1,keepdims=True)+epsilon)*g).astype('f4')))
    if a.with_qb_rope:
        assert config['q_lora_rank']==1280 and config['head_dim']==512
        assert config['num_attention_heads']==64 and config['qk_rope_head_dim']==64
        assert config['rope_theta']==10000 and config['compress_ratios'][0]==0
        w=tensor(pkg/'weights',meta,'wq_b','F8_E4M3',(32768,1280))
        s=tensor(pkg/'weights',meta,'wq_b_scale','F8_E8M0',(1024,40));qb_weights=(w,s)
        model.input('wq_b',w,'f8E4M3FN');model.input('wq_b_scale',s,'f8E8M0FNU')
        qb=model.projection('q_b',outputs[3],'wq_b','wq_b_scale',32,32768)
        qr=model.rotary('q_rope',model.reshape('q_heads',qb,(32,64,512)),64)
        kr=model.rotary('kv_rope',model.reshape('kv_head',outputs[4],(32,1,512)),1)
        outputs.extend([qb,model.reshape('q_rope_out',qr,(32,32768)),model.reshape('kv_rope_out',kr,(32,512))])
        qb_gold=project_bf16_activation(gold[3],w,s)
        gold.extend([qb_gold,rope(qb_gold.reshape(32,64,512),64,10000).reshape(32,32768),
                     rope(gold[4].reshape(32,1,512),64,10000).reshape(32,512)])
    if a.with_swa_core:
        assert a.swa_package and config['sliding_window']==128
        sink_meta=json.loads((a.swa_package/'weights/provenance.json').read_text())
        assert sink_meta['revision']==REVISION and sink_meta['config_sha256']==digest(a.model_config)
        sink_file=a.swa_package/'weights/attn_sink.bin'
        assert digest(sink_file)==sink_meta['tensors']['attn_sink']['sha256']
        sinks=np.fromfile(sink_file,'<f4');assert sinks.shape==(64,) and np.isfinite(sinks).all()
        q=model.reshape('swa_query',qr,(1,32,64,512));kv=model.reshape('swa_kv',kr,(1,32,1,512))
        context=model.op('swa','dense_attention',[q,kv,kv],(1,32,64,512),
            f'scale={512**-.5:.9e}:f32,causal=true,query_position=0:i64,sliding_window=128:i64,'+
            'attention_sinks=array<f32:'+','.join(f'{v:.9e}' for v in sinks)+'>')
        outputs.append(model.reshape('swa_context',context,(32,32768)))
        gold.append(swa_reference(gold[6].reshape(32,64,512),gold[7],sinks).reshape(32,32768))
    model.write(outputs,rope_tile_overlap=a.rope_tile_overlap)
    for name,value in zip(outputs,gold):np.save(out/(name+'.expected.npy'),value)
    limit=dict(relative_l2=.01,cosine=.9999,projection_local_bf16_bit_mismatches=0,
               rope_local_relative_l2=.001,rope_prefix_bit_mismatches=0,swa_local_relative_l2=.01)
    record=dict(status='preparing',scope='attention RMSNorm, Q_A/KV projections'
        +(' and Q_A/KV RMSNorm' if a.with_qkv_norm else '')
        +(', Q_B and Q/KV partial RoPE' if a.with_qb_rope else '')
        +(', causal SWA core with sink' if a.with_swa_core else '')+' only; hc_pre fixture',
        batch=1,sequence=32,input_shape=list(x.shape),outputs={n:list(v.shape) for n,v in zip(outputs,gold)},
        diagnostic_outputs=outputs[:8] if a.with_swa_core else outputs[:6] if a.with_qb_rope else outputs[:3] if a.with_qkv_norm else [y],
        persistent_state=None,revision=REVISION,raw_macro=True,rope_tile_overlap=a.rope_tile_overlap,
        allow_layout_copies=a.allow_layout_copies,
        precision='BF16 X/gamma/normalized/output; original E4M3FN weights + E8M0 block32 scales '
                  'dequantized to BF16 at MXM IW; FP32 RMSNorm and ordered K32 projection accumulation',
        epsilon=epsilon,limits=limit,input_sha256=digest(pkg/'reference/x.npy'),
        weight_sha256={role:item['sha256'] for role,item in meta['tensors'].items()
                       if role in ('wq_a','wq_a_scale','wkv','wkv_scale')
                           or (a.with_qkv_norm and role in ('q_norm','kv_norm'))
                           or (a.with_qb_rope and role in ('wq_b','wq_b_scale'))},
        gamma_sha256=norm_meta['tensors']['attn_norm']['sha256'],command=[sys.executable,*sys.argv])
    if a.with_qb_rope:
        record['rope']=dict(heads_q=64,heads_kv=1,head_dim=512,rotary_dim=64,
                           rotary_offset=448,theta=10000,positions=[0,31],yarn=False)
    if a.with_swa_core:
        record['swa']=dict(query_heads=64,kv_heads=1,head_dim=512,window=128,query_position=0,
            sink_sha256=digest(sink_file),cache_writeback=False,inverse_rope=False,output_projection=False)
    if previous:
        record['validation_command']=record['command'];record['command']=previous['command']
    ext='.exe' if os.name=='nt' else '';exe=lambda area,name:a.build_dir.resolve()/area/(name+ext)
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',
               FTLPU_GRAPH_PROGRESS='1',FTLPU_SESSION_PROGRESS='1')
    try:
        if not a.validate_existing:
            run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel',
                 '--target-config',a.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws',
                 '--output',out/'program.ftlpu'],out/'compile.log',env)
        plan=json.loads((out/'graph_plan.json').read_text());roles={}
        assert sum(n['name']=='attention_rmsnorm' for n in plan['nodes'])==1
        for name,width in [('q_a',1280),('kv_raw',512)]:
            nodes=[n for n in plan['nodes'] if n['name'].startswith(name+'.')]
            tasks=[t for n in nodes for t in n.get('native_projections',[])]
            assert tasks and sum(t['n'] for t in tasks)==width
            assert all(t['kind']=='linear' and t['m']==32 and t['k']==hidden
                       and t['rhs_scale_block_count']==t['n']//32*(hidden//32) for t in tasks)
            roles[name]=dict(pages=len(tasks),full_k=hidden,columns=width,
                            input_reused=[n.get('projection_input_reused') for n in nodes])
        assert not any('.qdq' in n['name'] for n in plan['nodes'])
        if a.with_qb_rope:
            tasks=[t for n in plan['nodes'] if n['name'].startswith('q_b.') for t in n.get('native_projections',[])]
            assert tasks and sum(t['n'] for t in tasks)==32768
            assert all(t['kind']=='linear' and t['m']==32 and t['k']==1280
                       and t['rhs_scale_block_count']==t['n']//32*40 for t in tasks)
            roles['q_b']=dict(pages=len(tasks),full_k=1280,columns=32768)
            rotary=[n for n in plan['nodes'] if n.get('native_planar_rotary')]
            if a.rope_tile_overlap:
                assert rotary and all(n.get('tile_pipeline') for n in rotary)
                record['pipeline_structure']=validate_rotary_pipeline(plan,True)
            if not a.allow_layout_copies:
                assert rotary and sum(n['transpose_input_vectors'] for n in rotary)==32*65*2
                assert all(n['intermediate_product_mem_writes']==0 for n in rotary)
                assert not any('pair_rotation' in n['name'] for n in plan['nodes'])
            record['rotary_schedule']=dict(groups=rotary,cycles=sum(n['end']-n['start'] for n in rotary),
                intermediate_product_mem_writes=sum(n['intermediate_product_mem_writes'] for n in rotary))
        schedule=(out/'decoder_layer.schedule.mlir').read_text()
        assert 'distributed_16' not in schedule,'distributed16 storage is not authorized'
        if a.with_swa_core:
            assert 'fp16_value_x16' not in schedule
            core_nodes=[n for n in plan['nodes'] if n.get('attention_core_schedule')=='flash_tile_pipeline']
            assert core_nodes and len(core_nodes)<=32
            record['swa_schedule']=dict(groups=core_nodes,
                compute_cycles=sum(n['nominal_end']-n['nominal_start'] for n in core_nodes))
        if not a.allow_layout_copies:
            copies=plan.get('linker_mem_transports',[])
            if a.with_qb_rope:
                # The new split-head context is packed only for the requested
                # host diagnostic. Earlier Q/KV outputs remain direct exports.
                exports=[c for c in copies if c['reason']=='graph_result_export']
                assert not exports or (a.with_swa_core and all(c['phase']=='validation_export.8' for c in exports))
                assert not any(c['phase'].startswith('q_b.') for c in copies)
            if a.with_qkv_norm:
                # One inward read per native hemisphere, local tap + far write.
                # This distributes column halves for full-width RMS reduction.
                norm_copies=[c for c in copies if c['phase'] in ('q_norm','kv_norm')]
                assert (a.with_qb_rope or len(copies)==4) and len(norm_copies)==4
                assert all(c['route']=='passive_sr_multicast' and c['local_tap'] for c in norm_copies)
                for name,width in [('q_norm',1280),('kv_norm',512)]:
                    rows=[c for c in copies if c['phase']==name]
                    assert len(rows)==2 and sum(c['vectors'] for c in rows)==width
                    assert sum(n['name']==name for n in plan['nodes'])==1
            else:
                assert not copies, 'native layouts must not be copied'
        assert plan.get('native_output_archive_copies')==0
        source=plan['nodes'][0]['direct_output_layouts'][0]
        operand_lines=[line for line in schedule.splitlines()
                       if re.search(r'name = "(?:q_a|kv_raw)\.page\d+\.operand"',line)]
        assert len(operand_lines)==6
        for line in operand_lines:
            for key in ('bank','base_row'):
                assert f'{key} = {source[key]} : i64' in line
            assert f'hemisphere = "{source["hemisphere"]}"' in line
            assert f'kind = "{source["kind"]}"' in line
            assert f'slices = {source["slices"]}' in line
        record.update(status='compiled_only',structure=dict(roles=roles,graph_nodes=len(plan['nodes']),
            linker_mem_transports=plan.get('linker_mem_transports',[]),
            native_output_archive_copies=plan.get('native_output_archive_copies'),activation_qdq_nodes=0,
            projection_inputs_alias_norm_output=True,projection_input_placement=source),
            signatures={k:digest(v) for k,v in dict(kernel=out/'kernel.mlir',program=out/'program.ftlpu',
                target=a.target_config,compiler=exe('compiler','ftlpu-compile'),
                runner=exe('runtime','compiled_stage_file_runner'),exporter=exe('runtime','ftlpu_icu_program_export')).items()})
        if a.prepare_only:return
        if previous:
            assert record['signatures']==previous['signatures'],'existing run inputs or toolchain changed'
            assert record['input_sha256']==previous['input_sha256'] and record['weight_sha256']==previous['weight_sha256']
            assert record['gamma_sha256']==previous['gamma_sha256']
        record['status']='executing';(out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        if not a.validate_existing:
            run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',
                 out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs',
             '--pre-execution-dir',out/'actual.pre_execution','--compute-imem-segments'],out/'export.log',env)
        record['icu_export_validation']=check_icu_export(out/'runtime_icu_programs')
        actual=[read_bf16(out/f'actual.{i}.bin',v.shape) for i,v in enumerate(gold)]
        measured={name:metrics(v,g) for name,v,g in zip(outputs,actual,gold)}
        if a.with_swa_core:
            local=swa_reference(actual[6].reshape(32,64,512),actual[7],sinks).reshape(32,32768)
            measured['swa_context']['swa_local']=metrics(actual[8],local)
            assert measured['swa_context']['swa_local']['relative_l2']<limit['swa_local_relative_l2']
        record['metrics']=measured
        for i,(w,s) in enumerate(weights,1):
            local=project_bf16_activation(actual[0],w,s)
            measured[outputs[i]]['projection_local_bf16_bit_mismatches']=int(np.count_nonzero(
                bf16_bits(local)!=bf16_bits(actual[i])))
            measured[outputs[i]]['projection_local_relative_l2']=metrics(actual[i],local)['relative_l2']
        for i,g in enumerate(norm_weights,1):
            v=actual[i].astype('f8')
            local=bf16((v/np.sqrt(np.mean(v*v,axis=1,keepdims=True)+epsilon)*g).astype('f4'))
            measured[outputs[i+2]]['norm_local']=metrics(actual[i+2],local)
        if a.with_qb_rope:
            local=project_bf16_activation(actual[3],*qb_weights)
            measured[outputs[5]]['projection_local_bf16_bit_mismatches']=int(np.count_nonzero(bf16_bits(local)!=bf16_bits(actual[5])))
            assert measured[outputs[5]]['projection_local_bf16_bit_mismatches']==0
            for source,dest,heads in [(5,6,64),(4,7,1)]:
                x=actual[source].reshape(32,heads,512);v=actual[dest].reshape(x.shape)
                local=rope(x,64,10000)
                check=dict(full=metrics(v,local),rotary_tail=metrics(v[...,448:],local[...,448:]),
                    prefix_bit_mismatches=int(np.count_nonzero(bf16_bits(x[...,:448])!=bf16_bits(v[...,:448]))),
                    token0_bit_mismatches=int(np.count_nonzero(bf16_bits(x[0])!=bf16_bits(v[0]))))
                measured[outputs[dest]]['rope_local']=check
                assert check['full']['relative_l2']<limit['rope_local_relative_l2'] and check['rotary_tail']['relative_l2']<limit['relative_l2']
                assert check['prefix_bit_mismatches']==check['token0_bit_mismatches']==0
        stats={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',(out/'execution.log').read_text().splitlines()[-1])}
        timing=trace_times(out/'runtime.pipeline.csv')
        timing['weight_page_wait_cycles']=stats['weight_page_runtime_wait_cycles']
        timing['invocation_excluding_weight_page_wait_cycles']=timing['invocation_cycles']-timing['weight_page_wait_cycles']
        record.update(metrics=measured,runtime=stats,timing=timing,
            runtime_pipeline=str(out/'runtime.pipeline.csv'),runtime_icu_index=str(out/'runtime_icu_programs/index.csv'))
        assert stats['host_operations']==0 and stats['outputs']==len(outputs)
        assert all(v['finite'] and v['relative_l2']<limit['relative_l2'] and v['cosine']>limit['cosine'] for v in measured.values()),measured
        assert all(measured[n]['projection_local_bf16_bit_mismatches']==0 for n in outputs[1:3]),measured
        assert all(v['norm_local']['relative_l2']<limit['relative_l2'] and v['norm_local']['cosine']>limit['cosine']
                   for v in measured.values() if 'norm_local' in v),measured
        record['status']='passed'
    except KeyboardInterrupt:
        record['status']='stopped_by_user';raise
    except Exception as exc:
        record.update(status='failed',error=str(exc));raise
    finally:
        (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
