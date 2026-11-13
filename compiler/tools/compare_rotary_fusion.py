"""Compare completed raw-macro runs; this tool never generates schedules."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def compare(before, after):
    old=json.loads((before/'summary.json').read_text())
    new=json.loads((after/'summary.json').read_text())
    assert old['status']==new['status']=='passed'
    for key in ('revision','input_sha256','weight_sha256','gamma_sha256','limits','precision','raw_macro','sequence','batch'):
        assert old[key]==new[key],f'comparison changed {key}'
    for key in ('target','runner','exporter'):
        assert old['signatures'][key]==new['signatures'][key],f'comparison changed {key}'
    plans=[json.loads((p/'graph_plan.json').read_text()) for p in (before,after)]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    unchanged={str(i):digest(before/f'actual.{i}.bin')==digest(after/f'actual.{i}.bin') for i in range(6)}
    assert all(unchanged.values()),'upstream checkpoints changed'
    all_outputs={name:digest(before/f'actual.{i}.bin')==digest(after/f'actual.{i}.bin')
                 for i,name in enumerate(new['outputs'])}
    timing={key:dict(before=old['timing'][key],after=new['timing'][key],
                    saved=old['timing'][key]-new['timing'][key]) for key in old['timing']}
    transports=[]
    for plan in plans:
        copies=plan['linker_mem_transports']
        rope=[c for c in copies if c['phase'].startswith(('q_rope.','kv_rope.'))]
        native=[n for n in plan['nodes'] if n.get('native_planar_rotary')]
        transports.append(dict(graph_nodes=len(plan['nodes']),all_linker_copy_domains=len(copies),
            rope_linker_copy_domains=len(rope),rope_linker_copy_vectors=sum(c['vectors'] for c in rope),
            native_rotary_groups=len(native),
            pipelined_rotary_groups=sum(bool(n.get('tile_pipeline')) for n in native),
            rotary_cycles=sum(n['end']-n['start'] for n in native),
            native_transpose_vectors=sum(n['transpose_input_vectors']+n['transpose_output_vectors'] for n in native),
            native_transpose_note='Each counted vector is one layout-transform read/write pair, not eliminated traffic',
            output_archive_copies=plan['native_output_archive_copies']))
    boundaries=[]
    for plan in plans:
        projection=max(n['nominal_end'] for n in plan['nodes'] if n['name'].startswith('q_b.'))
        end=max(n.get('end',n.get('nominal_end',0)) for n in plan['nodes'])
        boundaries.append(dict(projection_end=projection,rotary_end=end,tail_cycles=end-projection))
    segments=[]
    segment_file=after/'runtime_icu_programs/compute_imem_segments.csv'
    if segment_file.exists():
        with segment_file.open(newline='') as stream:segments=list(csv.DictReader(stream))
        assert all(int(s['capacity_checked'])==1 for s in segments)
    return dict(status='passed',comparison='same inputs, weights, precision, thresholds, target and runtime',
        before=str(before),after=str(after),upstream_outputs_bitwise_unchanged=unchanged,
        all_outputs_bitwise_unchanged=all_outputs,
        timing_cycles=timing,transport_before=transports[0],transport_after=transports[1],
        compiler_boundary_before=boundaries[0],compiler_boundary_after=boundaries[1],
        compute_imem_segments=segments,
        limitations=['Instruction-image DMA is not timed in either run',
            'Diagnostic HostOutput ICU batches are in the pipeline CSV but not the compute ICU index',
            'RoPE retains two wide-transpose scratch boundaries and a six-cycle accumulator configuration pitch',
            'No mHC, SWA attention core, cache writeback, output projection or FFN is executed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=compare(args.before,args.after)
    path=args.output or args.after/'fusion_comparison.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(path)
