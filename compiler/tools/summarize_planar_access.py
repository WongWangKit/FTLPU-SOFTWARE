"""Validate numerical identity and summarize graph-level address planning."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_bytes(plan):
    return sum(t['vectors']*t['byte_planes']*64 for t in plan['linker_mem_transports'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,default=Path('tmp/glm4_access_plan_1111'))
    p.add_argument('--results',type=Path,default=Path('results/glm4_9b_decoder_layer_decode_flash_streamed'))
    args=p.parse_args()
    saved=json.loads((args.build_dir/'before.json').read_text())
    report=[]
    for batch in (1,32):
        folder=args.results/f'batch{batch}'
        before=saved[str(batch)]['result']
        after=json.loads((folder/'result.json').read_text())
        assert before['status']==after['status']=='passed'
        for field in ('runner_sha256','target_sha256','fixture_fingerprint','precision',
                      'projection_output_sha256','past_kv_sha256'):
            assert before[field]==after[field],field
        assert before['phase']['output_sha256']==after['phase']['output_sha256']
        phase=folder/'attention'
        assert sha(phase/'actual.runtime.ftlpu')==after['phase']['actual_program_sha256']
        assert sha(phase/'runtime.pipeline.csv')==after['phase']['runtime_pipeline_sha256']
        for i,digest in enumerate(after['phase']['output_sha256']):
            assert sha(phase/f'actual.{i}.bin')==digest
        plan=json.loads((phase/'graph_plan.json').read_text())
        views=[a for n in plan['nodes'] for a in n.get('access_plans',[])]
        reasons=Counter(a['reason'] for a in views if 'reason' in a)
        compact=[n for n in plan['nodes'] if n.get('access_decision')=='compact_unique_rows_with_affine_read']
        assert compact and all(n['stored_rows']<n['logical_rows'] for n in compact)
        assert any(a['decision']=='address_view' for a in views)
        old_bytes,new_bytes=copy_bytes(saved[str(batch)]['plan']),copy_bytes(plan)
        assert new_bytes<old_bytes
        old_time,new_time=before['phase']['timing'],after['phase']['timing']
        assert new_time['runtime_cycles']<old_time['runtime_cycles']
        result=dict(batch=batch,bit_exact_context_and_cache=True,
                    before=old_time,after=new_time,
                    linker_read_write_bytes_before=old_bytes,
                    linker_read_write_bytes_after=new_bytes,
                    address_views=sum(a['decision']=='address_view' for a in views),
                    compact_views=len(compact),fallback_reasons=dict(reasons))
        report.append(result)
        (folder/'before_access_plan.json').write_text(json.dumps(saved[str(batch)],indent=2)+'\n')
        print(batch,old_time['runtime_cycles'],new_time['runtime_cycles'],old_bytes,new_bytes)
    (args.results/'planar_access.json').write_text(json.dumps(dict(
        scope='GLM-4-9B Flash decode core + KV append, batch1/32, past32, one decoded token',
        results=report,notes=[
            'Identical raw-macro runtime, precision, reference thresholds and fixtures.',
            'Copy bytes count explicit linker MEM reads plus writes; not total native MEM traffic.',
            'Runtime includes C2C and diagnostic cache readback; iMEM load/reload transfer is not timed.',
            'The initial common access contract covers mirrored planar views expressible as one READ3D; unsupported ownership and segmented sources retain explicit conversion.'
        ]),indent=2)+'\n')


if __name__=='__main__':main()
