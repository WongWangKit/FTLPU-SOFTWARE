"""Audit resident-only prefill and independent diagnostic executions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from glm4_prefill_attention_test import verify_resident_compute
from summarize_flash_probability_streaming import audit
from summarize_flash_prefill_transport import transport_bytes


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/glm4_9b_prefill_attention_comparison'))
    parser.add_argument('--build-dir',type=Path,required=True)
    args=parser.parse_args();root=args.results.resolve();build=args.build_dir.resolve()
    variants={}
    for prefix in ('','synthetic/'):
        for strategy in ('dense','flash'):
            folder=root/prefix/strategy;diagnostic=folder/'diagnostic'
            record=json.loads((folder/'result.json').read_text())
            assert record['status']=='passed' and record['output_policy']=='resident_context_separate_diagnostic'
            proof=verify_resident_compute(folder,diagnostic)
            phase=record['phase'];diag_phase=record['diagnostic_phase']
            assert sha(diagnostic/'actual.0.bin')==diag_phase['output_sha256'][0]
            assert sha(diagnostic/'actual.runtime.ftlpu')==diag_phase['actual_program_sha256']
            assert sha(diagnostic/'runtime.pipeline.csv')==diag_phase['runtime_pipeline_sha256']
            assert sha(diagnostic/'program.ftlpu')==diag_phase['binary_sha256']
            assert not (folder/'actual.0.bin').exists()
            assert phase['output_sha256']==[] and phase['timing']['output_readback_including_diagnostics_cycles']==0
            with (folder/'runtime.pipeline.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
            assert not any(r['resource']=='C2C.HostOutput' or r['resource'].endswith('.TX') for r in rows)
            plan=json.loads((folder/'graph_plan.json').read_text())
            assert plan['external_output_indices']==[] and len(plan['resident_output_views'])==1
            assert all(not t['phase'].startswith('validation_export.') for t in plan['linker_mem_transports'])
            assert all(not v['gathered'] for v in plan['resident_output_views'])
            main_audit=audit(folder,build,record)
            diagnostic_audit=audit(diagnostic,build)
            item=dict(status='passed',normal=phase['timing'],diagnostic=diag_phase['timing'],
                compiler_sha256=record['compiler_sha256'],runner_sha256=record['runner_sha256'],
                target_sha256=record['target_sha256'],scope=record['scope'],
                actual_program_sha256=phase['actual_program_sha256'],runtime_pipeline_sha256=phase['runtime_pipeline_sha256'],
                diagnostic_output_sha256=diag_phase['output_sha256'][0],
                context_metrics=record['context_metrics'],end_to_end_context_metrics=record['end_to_end_context_metrics'],
                external_result_bytes=0,normal_diagnostic_readback_cycles=0,
                normal_external_output_bindings=0,normal_context_gather_bytes=0,
                layout_copy_read_write_bytes=transport_bytes(folder),compute_schedule_proof=proof,
                compute_imem_segments=main_audit['compute_reload_segment_count'],
                diagnostic_compute_imem_segments=diagnostic_audit['compute_reload_segment_count'])
            if not prefix:
                before_folder=root/(strategy+'_before_external_output_policy')
                before=json.loads((before_folder/'result.json').read_text())
                assert sha(before_folder/'actual.0.bin')==diag_phase['output_sha256'][0]
                for key in ('runner_sha256','target_sha256','fixture_sha256','precision','softmax_precision'):
                    assert before[key]==record[key],key
                item.update(before=before['phase']['timing'],diagnostic_bitexact_to_previous=True,
                            fixture_sha256=record['fixture_sha256'])
            variants[prefix+strategy]=item
    selection=root/'regression/external_output_policy'
    policy=json.loads((selection/'summary.json').read_text());assert policy['status']=='passed'
    for case in ('all','selected','resident'):audit(selection/case,build)
    report=dict(status='passed',scope='GLM-4-9B B1 seq32 attention core only; no projection/O/FFN execution',
        output_policy='Explicit compiler external_output_indices; omitted roots remain in native SRAM through invocation end',
        kv_scope='Existing K/V enter this core from DDR. No new KV is produced or written by this core invocation.',
        persistent_state_warning='A whole-layer caller must explicitly select newly generated KV state outputs; do not use an empty export list for required persistent outputs.',
        boundary_limitation='Resident views preserve same-graph consumer layouts; this metadata alone is not an automatic cross-invocation alias ABI.',
        precision='Unchanged BF16 boundaries and FP32 arithmetic; aggregate error limits unchanged',
        variants=variants,selection_regression=policy,
        physical_constraints='Passed iMEM/transport audit; at most one active FU 3D context per ICU queue',
        timing_notes=['Normal times include actual input C2C and no diagnostic egress.',
                      'Diagnostic is a separate full execution, not a direct host snapshot of normal SRAM.',
                      'Native compute schedule operations are identical in normal and diagnostic builds.',
                      'iMEM image transfer latency remains unmodeled.'])
    (root/'resident_output_policy.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:dict(normal_cycles=v['normal']['runtime_cycles'],
        diagnostic_cycles=v['diagnostic']['runtime_cycles'],external_bytes=v['external_result_bytes'],
        layout_copies=v['layout_copy_read_write_bytes']) for k,v in variants.items()},indent=2))


if __name__=='__main__':main()
