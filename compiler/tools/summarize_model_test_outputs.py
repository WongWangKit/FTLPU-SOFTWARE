"""Audit executed output contracts and physical ICU capacity; no scheduling."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--imem-check',type=Path,required=True)
    a=p.parse_args()
    cases=[]
    for suite in ('prefill','decode'):
        root=a.results/f'glm4_9b_{suite}_attention_comparison'
        runner=json.loads((root/'runner.json').read_text())
        assert runner['status']=='passed' and runner['numeric_passed']
        for item in runner['cases']:
            directory=root/item['directory']
            record=json.loads((directory/'result.json').read_text())
            phase=record['phase']
            runtime=directory/('attention' if suite=='decode' else '')
            required=[] if suite=='prefill' else [1,2]
            assert phase['output_indices']==required
            audits=[]
            for folder,execution in ((runtime,phase),(runtime/'diagnostic',phase['diagnostic_phase'])):
                assert execution['actual_program_sha256']==sha(folder/'actual.runtime.ftlpu')
                assert execution['runtime_pipeline_sha256']==sha(folder/'runtime.pipeline.csv')
                for i,digest in zip(execution['output_indices'],execution['output_sha256']):
                    assert sha(folder/f'actual.{i}.bin')==digest
                with (folder/'runtime.pipeline.csv').open(newline='') as stream:
                    outputs=[r for r in csv.DictReader(stream) if r['resource']=='C2C.HostOutput']
                assert bool(outputs)==bool(execution['output_indices'])
                audit=json.loads(subprocess.check_output([str(a.imem_check.resolve()),
                    str((folder/'actual.runtime.ftlpu').resolve())],text=True))
                assert audit['aggregate']['fits'] and audit['transport']['fits']
                assert max(q['peak_macro_contexts'] for q in audit['aggregate']['queues'])<=1
                (folder/'imem_capacity.json').write_text(json.dumps(audit,indent=2)+'\n')
                audits.append(dict(directory=str(folder),fits=True,
                    physical_words=audit['aggregate']['physical_words_total'],
                    reload_segments=audit['compute_reload_segment_count']))
            cases.append(dict(suite=suite,strategy=record['attention_strategy'],
                batch=record['batch'],context_relative_l2=record['context_metrics']['relative_l2'],
                external_indices=required,primary_cycles=phase['timing']['runtime_cycles'],
                diagnostic_cycles=phase['diagnostic_phase']['timing']['runtime_cycles'],
                compute_schedule_proof=phase['compute_schedule_proof'],audits=audits))
    regression=a.results/'glm4_9b_prefill_attention_comparison/regression/external_output_policy/summary.json'
    assert json.loads(regression.read_text())['status']=='passed'
    report=dict(status='passed',cases=cases,policy_regression_sha256=sha(regression),
                scope='GLM attention cores; seq32 prefill and selected decode batches, not whole-model execution')
    output=a.results/'model_tests/output_contract_acceptance.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(output)
    for case in cases: print({k:v for k,v in case.items() if k not in ('compute_schedule_proof','audits')})


if __name__=='__main__': main()
