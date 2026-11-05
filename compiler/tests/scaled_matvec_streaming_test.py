"""Numerical and dataflow check for retained matrix scaling + streamed matvec."""
import argparse
import bisect
import csv
import json
import os
import re
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16
from qwen3_5_decode_gdn_attention_test import run
from qwen3_5_qkv_conv_prefill_test import compare

def audit_streaming(out):
    """Check real generated domains, including both preserved-state outputs."""
    plan=json.loads((out/'graph_plan.json').read_text())
    all_lines=(out/'decoder_layer.schedule.mlir').read_text().splitlines()
    # Index once: a seq32 scan has many copies of the same primitive. Parsing
    # the entire schedule again for every token makes validation quadratic.
    timed=[]
    for line in all_lines:
        stamp=re.search(r'\b(?:cycle|start) = (\d+) : i64',line)
        if stamp:timed.append((int(stamp[1]),line))
    timed.sort(key=lambda item:item[0]);times=[item[0] for item in timed]
    node_lines={n['name']:[item[1] for item in timed[
        bisect.bisect_left(times,n['nominal_start']):bisect.bisect_left(times,n['nominal_end'])]]
        for n in plan['nodes'] if 'nominal_start' in n}
    result=[]
    for node in plan['nodes']:
        lines=node_lines.get(node['name'],[])
        full_fp32=any('lhs_kind = "stream_f32"' in line and 'queue = 0 : i64' in line for line in lines)
        marker='name = "'+node['name']+'.scaled_matvec.vxm_to_mxm_and_mem"'
        if not any(marker in line for line in lines):continue
        saved=node['direct_output_layouts'][0]
        epilogue=len(node['direct_output_layouts'])>=3
        update=len(node['direct_output_layouts'])>=4
        readout=len(node['direct_output_layouts'])==5
        update_marker='name = "'+node['name']+'.scaled_matvec.rank_one_update"'
        update_start=next((int(re.search(r'start = (\d+) : i64',l)[1]) for l in lines if update_marker in l),node['nominal_end'])
        product_home=node['direct_output_layouts'][1]
        product_reads=product_taps=0;correction_issues=[]
        reads=taps=update_reads=update_read_vectors=update_write_vectors=updated_reads=0;first_load=None;last_save=None
        update_home=node['direct_output_layouts'][3] if update else None
        load_domains={};multiply_domains=[]
        for line in lines:
            fields={k:int(v) for k,v in re.findall(r'(\w+) = (-?\d+) : i64',line)}
            cycle=fields.get('cycle',-1)
            if not node['nominal_start']<=cycle<node['nominal_end']:continue
            if 'ftlpu.schedule.mxm' in line and 'opcode = "iw"' in line:
                if cycle>=update_start:continue
                first_load=cycle if first_load is None else min(first_load,cycle)
                load_domains.setdefault(fields['unit_id'],[]).append((cycle,
                    cycle+fields['repeat_count']*fields['repeat_interval'],fields['weight_buffer']))
            if 'ftlpu.schedule.vxm' in line and 'opcode = "multiply"' in line and fields.get('queue')==0:
                multiply_domains.append((cycle,cycle+fields['repeat_count']*fields['repeat_interval']))
            if 'ftlpu.schedule.vxm' in line and 'opcode = "subtract"' in line and fields.get('queue')==4:
                correction_issues.append(cycle)
            if 'ftlpu.schedule.mem_transfer' not in line:continue
            if update and cycle>=update_start and fields.get('slice') in update_home['slices'] and fields.get('bank')==update_home['bank'] and update_home['base_row']<=fields.get('address',-1)<update_home['base_row']+update_home['rows']:
                if ('opcode = "write_tap"' if readout else 'opcode = "write"') in line:update_write_vectors+=fields.get('repeat_count',1)
                updated_reads+='opcode = "read"' in line
            if fields.get('slice') in product_home['slices'] and fields.get('bank')==product_home['bank'] and product_home['base_row']<=fields.get('address',-1)<product_home['base_row']+product_home['rows']:
                product_reads+='opcode = "read"' in line
                product_taps+='opcode = "write_tap"' in line
            if fields.get('slice') not in saved['slices'] or fields.get('bank')!=saved['bank']:continue
            if not saved['base_row']<=fields.get('address',-1)<saved['base_row']+saved['rows']:continue
            if cycle>=update_start:
                update_reads+='opcode = "read"' in line
                if 'opcode = "read"' in line:update_read_vectors+=fields.get('repeat_count',1)
            else:reads+='opcode = "read"' in line
            if 'opcode = "write_tap"' in line:
                taps+=1;last_save=max(last_save or 0,cycle+(fields['repeat_count']-1)*fields['repeat_interval'])
        assert reads==0 and taps>0 and first_load is not None and first_load<last_save,(node['name'],reads,taps,first_load,last_save)
        for domains in [multiply_domains,*load_domains.values()]:
            domains.sort()
            assert domains and all(a[1]==b[0] for a,b in zip(domains,domains[1:])),(node['name'],'steady-state bubble',domains)
        for domains in load_domains.values():
            assert all(a[2]!=b[2] for a,b in zip(domains,domains[1:])),(node['name'],'weight buffer reuse')
        if epilogue:
            assert product_reads==0 and product_taps>0 and correction_issues
            assert len(correction_issues)==product_home['rows']
            if product_home['rows']>1 and len(next(iter(load_domains.values())))>3:
                assert min(correction_issues)<last_save,(node['name'],'correction did not overlap state streaming')
        result.append(dict(node=node['name'],scaled_state_read_domains=reads,
            readout_enabled=readout,
            update_state_read_domains=update_reads,
            scaled_state_write_tap_domains=taps,first_mxm_load=first_load,last_state_save=last_save,
            blocks_per_hemisphere=len(next(iter(load_domains.values()))),
            steady_state_block_interval=32,steady_state_vxm_and_iw_idle_cycles=0))
        if epilogue:
            result[-1]['correction']=dict(product_mem_read_domains=product_reads,
                product_write_tap_domains=product_taps,vector_issues=len(correction_issues),
                first_issue=min(correction_issues),last_issue=max(correction_issues))
        if update:
            assert updated_reads==0
            # Schedule slice IDs are local; both hemispheres are counted.
            # Each selected State byte is read once, each output byte written once.
            sides=2 if saved['hemisphere']=='both' else 1
            assert update_read_vectors==saved['rows']*(4 if full_fp32 else 2)*sides,(node['name'],update_read_vectors,saved)
            assert update_write_vectors==update_home['rows']*len(update_home['slices'])*sides,(node['name'],update_write_vectors,update_home)
            result[-1]['rank_one_update']=dict(saved_state_byte_vector_reads=update_read_vectors,
                saved_input_bytes=4 if full_fp32 else 2,
                updated_byte_vector_writes=update_write_vectors,updated_state_mem_reads=updated_reads,
                updated_state_write_opcode='write_tap' if readout else 'write',outer_product_spills=0)
    assert result,'No fused dataflow found in native schedule'
    # Locate the uninterrupted state IW run, even if later projections also
    # issue MXM loads. Check runtime continuity, not nominal schedule times.
    with (out/'runtime.pipeline.csv').open() as f:
        runtime=list(csv.DictReader(f))
    # A multi-token scan can compress identical FU ranges across tokens into
    # repeat2d rows. Audit issued instances, not only the first CSV range.
    expanded=[]
    audited={'MXM.E0.Load','MXM.W0.Load','VXM.C1','VXM.C2','SXM.E.Transpose'}
    for row in runtime:
        if row['resource'] not in audited:
            expanded.append(row);continue
        for outer in range(int(row['outer_count'])):
            for inner in range(int(row['inner_count'])):
                if int(row['skip_first']) and outer==0 and inner==0:continue
                shift=outer*int(row['outer_interval'])+inner*int(row['inner_interval'])
                expanded.append({**row,'start':str(int(row['start'])+shift),
                    'end':str(int(row['end'])+shift),'pattern':'single','inner_count':'1','outer_count':'1'})
    runtime=sorted(expanded,key=lambda r:int(r['start']))
    count=sum(n['blocks_per_hemisphere']*(2 if n['readout_enabled'] else 1) for n in result)
    for side in ('E','W'):
        all_loads=[row for row in runtime if row['resource']==f'MXM.{side}0.Load']
        first_count=result[0]['blocks_per_hemisphere']
        candidates=[i for i in range(len(all_loads)-first_count+1)
            if all(int(r['end'])-int(r['start'])==32 and int(r['inner_count'])==1 and int(r['outer_count'])==1
                   for r in all_loads[i:i+first_count])
            and all(a['end']==b['start'] for a,b in zip(all_loads[i:i+first_count],all_loads[i+1:i+first_count]))]
        if result[0]['readout_enabled']:
            # Projections before the recurrence can also have a continuous IW
            # run. State decay is followed by rank-one updates whose readout
            # loads have inter-tile gaps; use both halves to identify the run.
            candidates=[i for i in candidates if i+count<=len(all_loads)
                and all(int(r['end'])-int(r['start'])==32
                    for r in all_loads[i+first_count:i+2*first_count])
                and all(int(b['start'])-int(a['end'])>=2
                    for a,b in zip(all_loads[i+first_count:i+2*first_count],
                                   all_loads[i+first_count+1:i+2*first_count]))]
        assert candidates, 'Missing uninterrupted runtime state IW run'
        loads=all_loads[candidates[0]:candidates[0]+count]
        assert len(loads)==count
        offset=0
        for node in result:
            lines=node_lines[node['node']]
            group=loads[offset:offset+node['blocks_per_hemisphere']]
            assert all(r['pattern']=='single' and int(r['end'])-int(r['start'])==32 for r in group)
            assert all(a['end']==b['start'] for a,b in zip(group,group[1:])),(node['node'],side,'runtime IW bubble')
            node.setdefault('runtime_mxm_load_ranges',{})[side]=[int(group[0]['start']),int(group[-1]['end'])]
            if side=='E' and 'rank_one_update' in node:
                native=next(n for n in plan['nodes'] if n['name']==node['node'])
                marker='name = "'+node['node']+'.scaled_matvec.rank_one_update"'
                meta=next({k:int(v) for k,v in re.findall(r'(\w+) = (-?\d+) : i64',l)} for l in lines if marker in l)
                start=meta['start']
                shift=int(group[0]['start'])-node['first_mxm_load']
                active=[row for row in runtime if row['resource']=='VXM.C1' and row['detail'].startswith('mul ')
                    and start+shift<=int(row['start'])<native['nominal_end']+shift]
                adds=[row for row in runtime if row['resource']=='VXM.C2' and row['detail'].startswith('add ')
                    and start+shift<=int(row['start'])<native['nominal_end']+shift]
                assert active and [(r['start'],r['end']) for r in active]==[(r['start'],r['end']) for r in adds], (
                    node['node'],start,shift,native['nominal_end'],len(active),len(adds),
                    [(r['start'],r['end']) for r in active[:3]],[(r['start'],r['end']) for r in adds[:3]],
                    [(r['start'],r['end']) for r in active[-3:]],[(r['start'],r['end']) for r in adds[-3:]])
                assert sum(int(r['end'])-int(r['start']) for r in active)==native['direct_output_layouts'][3]['rows']
                gaps=[int(b['start'])-int(a['end']) for a,b in zip(active,active[1:])]
                assert all(g>=2 for g in gaps)
                # A small K can make broadcasting slower than multiplication.
                # Such waits belong only at a head boundary, never inside it.
                heads,vd=meta['head_count'],meta['value_width']
                blocks=(meta['key_width']//32)*(vd//32 if node['readout_enabled'] else 1)
                segment=32 if node['readout_enabled'] else vd
                pair_count=node['rank_one_update']['saved_input_bytes']//2
                broadcast_cost=(vd//32)*80*pair_count+64
                queues={}
                for line in lines:
                    if 'ftlpu.schedule.mem_transfer' not in line:continue
                    fields={k:int(v) for k,v in re.findall(r'(\w+) = (-?\d+) : i64',line)}
                    if not start<=fields.get('cycle',-1)<native['nominal_end'] or fields.get('slice') not in ((48,49,50,51) if node['rank_one_update']['saved_input_bytes']==4 else (50,51)):continue
                    table_count=fields.get('repeat_count',1); address=fields['address']
                    assert 0<=address and address+(table_count-1)*fields.get('address_stride',1)<vd
                    queue=(fields['hemisphere'],fields['slice'],fields['bank'])
                    queues.setdefault(queue,[]).append((fields['cycle'],fields['cycle']+table_count, 'opcode = "write"' in line))
                assert {q[2] for q in queues}==({0,1} if heads>1 else {1})
                for domains in queues.values():
                    domains.sort()
                    assert all(a[1]<=b[0] for a,b in zip(domains,domains[1:])),domains
                    assert any(d[2] for d in domains) and any(not d[2] for d in domains)
                for i,gap in enumerate(gaps):
                    expected=2 if (i+1)%blocks else 2+max(0,broadcast_cost-blocks*(segment+2))
                    assert gap==expected,(i,gap,expected)
                transposes=[row for row in runtime if row['resource']=='SXM.E.Transpose'
                    and start+shift<=int(row['start'])<native['nominal_end']+shift]
                assert len(transposes)==heads*(vd//32)*pair_count
                if heads>1:
                    # SXM decode aligns with the current head's first multiply
                    # issue; K capture was issued one cycle earlier.
                    for h in range(1,heads):
                        assert int(transposes[h*(vd//32)*pair_count]['start'])==int(active[(h-1)*blocks]['start'])
                node['rank_one_update'].update(runtime_multiply_issue_range=[int(active[0]['start']),int(active[-1]['end'])],
                    runtime_vector_updates=sum(int(r['end'])-int(r['start']) for r in active),
                    runtime_segments=len(active),segment_boundary_cycles=2,
                    broadcast_banks=[1,0],broadcast_rows_per_bank=vd,
                    next_head_broadcast_overlapped=heads>1,head_boundary_wait_cycles=sum(g-2 for g in gaps))
            offset+=node['blocks_per_hemisphere']
            if node['readout_enabled']:
                readout_loads=loads[offset:offset+node['blocks_per_hemisphere']]
                assert len(readout_loads)==node['blocks_per_hemisphere']
                assert all(int(x['end'])-int(x['start'])==32 for x in readout_loads)
                assert all(int(b['start'])-int(a['end'])>=2 for a,b in zip(readout_loads,readout_loads[1:]))
                node.setdefault('runtime_readout_load_ranges',{})[side]=[int(readout_loads[0]['start']),int(readout_loads[-1]['end'])]
                offset+=node['blocks_per_hemisphere']
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ('build-dir','target-config','output-dir'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--key-dim',type=int,default=64,choices=(32,64,96))
    p.add_argument('--single-tile',action='store_true')
    p.add_argument('--with-correction',action='store_true')
    p.add_argument('--with-update',action='store_true')
    p.add_argument('--with-readout',action='store_true')
    p.add_argument('--retain-input',action='store_true',help='Keep an original matrix snapshot live after the update')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.with_readout:a.with_update=True
    if a.with_update:a.with_correction=True
    build=a.build_dir.resolve();suffix='.exe' if sys.platform=='win32' else ''
    exe=lambda area,name:build/area/(name+suffix)
    heads,kd,vd=(1,32,32) if a.single_tile else (2,a.key_dim,64)
    rng=np.random.default_rng(35261030)
    matrix=rng.normal(0,.2,(heads*vd,kd)).astype('f4')
    # A nonzero packed group offset catches accidental reuse of head 0's
    # coefficient when a large matrix is split into native head ranges.
    scale=np.array([[.125,.25,.375,.8125]],dtype='f4')
    key=bf16(rng.normal(0,.2,(heads,kd)).astype('f4'))
    query=bf16(rng.normal(0,.2,(heads,kd)).astype('f4'))
    value=bf16(rng.normal(0,.2,(heads,vd)).astype('f4'))
    coefficient=np.array([[.0625,.875,.1875,.6875]],dtype='f4')
    g=LogicalGraph()
    mt,dt,kt,pt=(g.type(s,t) for s,t in [(matrix.shape,'f32'),(scale.shape,'f32'),(key.shape,'bf16'),((heads,vd),'f32')])
    src=f'''module {{ func.func @stage(%s: {mt}, %d: {dt}, %k: {kt}) -> ({mt}, {pt}) {{
      %s1, %p = ftlpu.kernel.scaled_matvec %s, %d, %k {{group_offset = 2 : i64}} : ({mt}, {dt}, {kt}) -> ({mt}, {pt})
      return %s1, %p : {mt}, {pt}
    }} }}'''
    inputs=[(matrix,'f32'),(scale,'f32'),(key,'bf16')]
    outputs=[(matrix.shape,'f32'),((heads,vd),'f32')]
    if a.with_correction:
        vt=g.type(value.shape,'bf16')
        src=f'''module {{ func.func @stage(%s: {mt}, %d: {dt}, %k: {kt}, %v: {vt}, %b: {dt}) -> ({mt}, {pt}, {pt}) {{
          %s1, %p, %delta = ftlpu.kernel.scaled_matvec %s, %d, %k epilogue(%v, %b) {{group_offset = 2 : i64}} : ({mt}, {dt}, {kt}, {vt}, {dt}) -> ({mt}, {pt}, {pt})
          return %s1, %p, %delta : {mt}, {pt}, {pt}
        }} }}'''
        inputs += [(value,'bf16'),(coefficient,'f32')]
        outputs += [((heads,vd),'f32')]
        if a.with_update:
            src=src.replace('-> ('+mt+', '+pt+', '+pt+')','-> ('+mt+', '+pt+', '+pt+', '+mt+')').replace('%s1, %p, %delta =','%s1, %p, %delta, %updated =').replace('return %s1, %p, %delta : '+mt+', '+pt+', '+pt,'return %s1, %p, %delta, %updated : '+mt+', '+pt+', '+pt+', '+mt)
            outputs += [(matrix.shape,'f32')]
            if a.with_readout:
                src=src.replace('%b: '+dt+')','%b: '+dt+', %q: '+kt+')').replace('epilogue(%v, %b)','epilogue(%v, %b, %q)').replace('('+mt+', '+dt+', '+kt+', '+vt+', '+dt+')','('+mt+', '+dt+', '+kt+', '+vt+', '+dt+', '+kt+')').replace('-> ('+mt+', '+pt+', '+pt+', '+mt+')','-> ('+mt+', '+pt+', '+pt+', '+mt+', '+pt+')').replace('%delta, %updated =','%delta, %updated, %y =').replace('return %s1, %p, %delta, %updated : '+mt+', '+pt+', '+pt+', '+mt,'return %s1, %p, %delta, %updated, %y : '+mt+', '+pt+', '+pt+', '+mt+', '+pt)
                inputs += [(query,'bf16')];outputs += [((heads,vd),'f32')]
    values=g.run('scaled_matvec',src,inputs,outputs)
    if a.retain_input:
        snapshot=f'''module {{ func.func @stage(%m: {mt}, %d: {dt}, %k: {kt}) -> ({mt}, {pt}) {{
            %o, %p = ftlpu.kernel.scaled_matvec %m, %d, %k {{group_offset = 2 : i64}} : ({mt}, {dt}, {kt}) -> ({mt}, {pt})
            return %o, %p : {mt}, {pt}
        }} }}'''
        values += g.run('matrix_reuse',snapshot,inputs[:3],outputs[:2])
    graph=g.write(out/'logical',values);kernel=out/'kernel.mlir';binary=out/'program.ftlpu'
    run([exe('compiler','ftlpu-graph-compile'),'--input',graph,'--output',kernel,'--emit-kernel','--target-config',a.target_config.resolve()],out/'import.log')
    kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.sxm_wide = true, ftlpu.vector_bf16_spill = true, ftlpu.common_barrier_protocol = true} {',1))
    run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,'--target-config',a.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws'],out/'compile.log')
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1'
    os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
    run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
    saved=np.fromfile(out/'actual.0.bin','<f4').reshape(heads,vd,kd)
    actual=np.fromfile(out/'actual.1.bin','<f4').reshape(heads,vd)
    truncate=lambda x:(np.asarray(x,dtype='f4').view('u4') & np.uint32(0xffff0000)).view('f4')
    expected=truncate(matrix).reshape(heads,vd,kd)*truncate(scale[:,2:2+heads]).reshape(heads,1,1)
    product=np.einsum('hvk,hk->hv',truncate(expected),key)
    metrics=dict(scaled=compare(saved,expected),product=compare(actual,product))
    if a.with_correction:
        delta=np.fromfile(out/'actual.2.bin','<f4').reshape(heads,vd)
        ref=(value-truncate(actual))*truncate(coefficient[:,2:2+heads]).reshape(heads,1)
        metrics['correction']=compare(delta,ref)
        if a.with_update:
            updated=np.fromfile(out/'actual.3.bin','<f4').reshape(heads,vd,kd)
            ref=truncate(saved)+truncate(delta)[:,:,None]*key[:,None,:]
            metrics['updated']=compare(updated,ref)
            if a.with_readout:
                y=np.fromfile(out/'actual.4.bin','<f4').reshape(heads,vd)
                metrics['readout']=compare(y,np.einsum('hvk,hk->hv',truncate(updated),query))
    if a.retain_input:
        snapshot=np.fromfile(out/f'actual.{len(outputs)}.bin','<f4').reshape(matrix.shape)
        metrics['retained_input']=compare(snapshot,expected.reshape(matrix.shape))
    result=dict(status='passed' if all(m['nrmse']<.001 for m in metrics.values()) else 'failed',metrics=metrics,
                streaming=audit_streaming(out))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert result['status']=='passed'

if __name__=='__main__':main()
