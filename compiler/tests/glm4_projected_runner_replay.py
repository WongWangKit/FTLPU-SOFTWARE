"""Compare two runtime linkers on an accepted GLM projected-attention program.

The candidate receives the frozen compiler output, ordered input bytes and
optional scales from the accepted execution. --rerun-reference also executes
the historical runner on that frozen bundle. Each replay exports its linked
ICU program and runtime trace.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np

from activation_batch_matmul_test import run
from glm4_flash_packed_u_test import graph_digest, sha
from summarize_glm4_flash import phase_timing


def freeze_inputs(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    frozen_program = destination/'program.ftlpu'
    shutil.copy2(source/'program.ftlpu', frozen_program)
    listing = []
    inputs = []
    for line in (source/'inputs.txt').read_text().splitlines():
        if not line:
            continue
        name, separator, scale = line.partition('\t')
        path = Path(name)
        frozen = destination/f'input{len(inputs)}.bin'
        shutil.copy2(path, frozen)
        digest = sha(frozen)
        assert digest == sha(path), f'input changed while copying {path}'
        inputs.append(dict(binding=len(inputs), name=frozen.name, sha256=digest, scale=scale or None))
        listing.append(str(frozen.resolve()) + separator + scale)
    assert inputs, 'empty input list'
    (destination/'inputs.txt').write_text('\n'.join(listing)+'\n')
    return frozen_program, inputs


def verify_frozen(program, binary_sha, inputs):
    assert sha(program) == binary_sha
    lines = (program.parent/'inputs.txt').read_text().splitlines()
    expected = []
    for item in inputs:
        path = program.parent/item['name']
        assert sha(path) == item['sha256']
        expected.append(str(path.resolve()) + ('\t'+item['scale'] if item['scale'] is not None else ''))
    assert lines == expected, 'input bindings or scales changed'


def replay(runner, output, program, record, input_manifest, expected, clock):
    output.mkdir(parents=True, exist_ok=True)
    env = {k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',
               FTLPU_SESSION_PROGRESS='1')
    verify_frozen(program, record['binary_sha256'], input_manifest)
    exporter = runner.parent/('ftlpu_icu_program_export' + runner.suffix)
    runner_hash = sha(runner)
    print(f'Executing {output.name} runner {runner_hash[:12]}', flush=True)
    run([runner, program, program.parent/'inputs.txt', output/'actual', output/'runtime.pipeline.csv'],
        output/'execution.log', env)
    run([exporter, output/'actual.runtime.ftlpu', output/'icu_programs', '--pre-execution-dir',
         output/'actual.pre_execution'], output/'export.log', env)
    assert sha(runner) == runner_hash
    verify_frozen(program, record['binary_sha256'], input_manifest)
    mismatches = {}
    outputs = []
    for index, (label, reference, reference_hash) in enumerate(expected):
        assert sha(reference) == reference_hash, 'accepted output changed'
        actual = output/f'actual.{index}.bin'
        original_words = np.fromfile(reference, '<u2')
        new_words = np.fromfile(actual, '<u2')
        assert original_words.shape == new_words.shape, label
        mismatches[label] = int(np.count_nonzero(original_words != new_words))
        outputs.append(sha(actual))
    stats = re.search(r'^cycles=\d+ .*$', (output/'execution.log').read_text(errors='replace'), re.M)
    assert stats, 'missing CModel completion statistics'
    counters = {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)', stats[0])}
    assert counters['host_operations'] == 0 and counters['outputs'] == len(expected)
    (output/'completed.json').write_text(json.dumps(record))
    return dict(runner_sha256=runner_hash, exporter_sha256=sha(exporter),
                linked_program_sha256=sha(output/'actual.runtime.ftlpu'),
                runtime_pipeline_sha256=sha(output/'runtime.pipeline.csv'),
                output_sha256=outputs, bitwise_mismatches=mismatches, runtime_counters=counters,
                timing=phase_timing(output,clock))


def inspect_accepted(source, accepted):
    mode = accepted['mode']
    final_files = [source/name for name in accepted['output_files']]
    assert len(final_files) == len(accepted['outputs']) == len(accepted['output_sha256']) == 3
    assert [sha(path) for path in final_files] == accepted['output_sha256']
    if mode == 'fused':
        phases = [('full_attention', accepted['outputs'])]
    elif mode == 'staged':
        phases = [('projections',['query','key','value']), ('attention',['context','present_key','present_value']),
                  ('output_projection',['projected'])]
    else:
        raise ValueError(f'unsupported mode {mode}')
    assert list(accepted['phases']) == [name for name,_ in phases]
    evidence = []
    for name, labels in phases:
        path = source if mode=='fused' else source/name
        phase = accepted['phases'][name]
        stamp = json.loads((path/'completed.json').read_text())
        assert phase['binary_sha256'] == stamp['binary_sha256'] == sha(path/'program.ftlpu')
        assert accepted['compiler_sha256'] == stamp['compiler_sha256'] == phase['compiler_sha256']
        assert phase['semantic_graph_sha256'] == stamp['semantic_graph_sha256'] == graph_digest(path)
        assert phase['actual_program_sha256'] == sha(path/'actual.runtime.ftlpu')
        assert phase['runtime_pipeline_sha256'] == sha(path/'runtime.pipeline.csv')
        assert phase['timing'] == phase_timing(path,accepted['clock_mhz'])
        outputs = [(label, path/f'actual.{i}.bin') for i,label in enumerate(labels)]
        expected = [(label,output,sha(output)) for label,output in outputs]
        evidence.append((name,path,phase,stamp,expected))
    assert sum(phase['timing']['runtime_cycles'] for _,_,phase,_,_ in evidence) == accepted['runtime_cycles']
    if mode == 'staged':
        # Anchor the intermediate outputs to the accepted downstream graph's constants.
        def constant(phase,node):
            logical=source/phase/'logical'
            graph=json.loads((logical/'decoder_layer.graph.json').read_text())
            arg=next(n for n in graph['nodes'] if n['name']==node)['inputs'][0]
            return logical/Path(graph['constants'][arg['constant']]['file']).name
        for i,node in enumerate(('query_input','new_key','new_value')):
            assert sha(source/'projections'/f'actual.{i}.bin') == sha(constant('attention',node))
        assert sha(source/'attention/actual.0.bin') == sha(constant('output_projection','o_projection'))
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-dir',type=Path,required=True,help='accepted batchN/fused or batchN/staged directory')
    parser.add_argument('--reference-runner',type=Path,required=True,help='historical runner named by the fixture signature')
    parser.add_argument('--candidate-runner',type=Path,help='updated runner to execute')
    parser.add_argument('--rerun-reference',action='store_true',help='also repeat the verified historical execution')
    parser.add_argument('--prepare-only',action='store_true',help='validate and freeze the inputs without simulator execution')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--target-config',type=Path,default=Path('../FTLPU-CMODEL/config/ftlpu-lpu32.json'))
    parser.add_argument('--flash-results',type=Path,default=Path('results/glm4_9b_decoder_layer_decode_flash'))
    args=parser.parse_args()
    if not args.prepare_only and args.candidate_runner is None:
        parser.error('--candidate-runner is required for execution')
    source=args.accepted_dir.resolve()
    archive=args.output_dir.resolve()
    if archive.is_relative_to(source) or source.is_relative_to(archive):
        parser.error('use a replay directory separate from the accepted execution')
    accepted=json.loads((source/'result.json').read_text())
    assert accepted['status']=='passed'
    signature=json.loads((args.flash_results/'signature.json').read_text())
    target=sha(args.target_config)
    reference_runner=args.reference_runner.resolve()
    assert sha(reference_runner)==accepted['runner_sha256']==signature['runner']
    if args.candidate_runner is not None:
        assert sha(args.candidate_runner)!=sha(reference_runner), 'candidate runner is identical to the reference'
    assert target==accepted['target_sha256']==signature['target']
    assert accepted['icu_encoding']=='raw-macro' and accepted['sxm']=='wide'
    assert accepted['outputs']==['projected','present_key','present_value']
    phases=inspect_accepted(source,accepted)
    archive.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',scope='same raw-macro programs on an updated runtime linker',
                mode=accepted['mode'],batch=accepted['batch'],fixture_fingerprint=accepted['fixture_fingerprint'],
                accepted_dir=str(source),reference_runner_sha256=accepted['runner_sha256'],
                compiler_sha256=accepted['compiler_sha256'],target_sha256=target,
                icu_encoding='raw-macro',sxm='wide',weight_paging_policy=accepted['weight_paging_policy'],
                clock_mhz=accepted['clock_mhz'],reference_source='replay' if args.rerun_reference else 'accepted_execution',
                phases={},reference_runtime_cycles=accepted['runtime_cycles'])
    report=archive/'result.json'
    if report.exists() and json.loads(report.read_text()).get('scope') != result['scope']:
        parser.error('output directory already contains a different benchmark')
    report.write_text(json.dumps(result,indent=2))
    staged=accepted['mode']=='staged'
    try:
        for name,path,phase,stamp,expected in phases:
            phase_inputs=archive/'inputs'/name if staged else archive/'inputs'
            program,inputs=freeze_inputs(path,phase_inputs)
            verify_frozen(program,stamp['binary_sha256'],inputs)
            proof=dict(program_sha256=stamp['binary_sha256'],semantic_graph_sha256=stamp['semantic_graph_sha256'],
                       inputs=inputs,expected_output_sha256=[digest for _,_,digest in expected],
                       reference=dict(linked_program_sha256=phase['actual_program_sha256'],
                           timing=phase['timing'],runtime_pipeline_sha256=phase['runtime_pipeline_sha256']))
            result['phases'][name]=proof
            if args.prepare_only:
                continue
            variants=[('candidate',args.candidate_runner.resolve())]
            if args.rerun_reference:
                variants.insert(0,('reference',reference_runner))
            for variant,runner in variants:
                output=archive/variant/name if staged else archive/variant
                measured=replay(runner,output,program,stamp,inputs,expected,accepted['clock_mhz'])
                proof[variant]=measured
                assert all(v==0 for v in measured['bitwise_mismatches'].values()),measured['bitwise_mismatches']
                if variant=='reference':
                    assert measured['linked_program_sha256']==phase['actual_program_sha256']
                    assert measured['timing']['runtime_cycles']==phase['timing']['runtime_cycles']
            report.write_text(json.dumps(result,indent=2))
        if args.prepare_only:
            result['status']='prepared'
        else:
            total=sum(phase['candidate']['timing']['runtime_cycles'] for phase in result['phases'].values())
            if staged:
                from qwen3_5_decode_gdn_batched_test import concatenate_traces
                variants=['candidate','reference'] if args.rerun_reference else ['candidate']
                for variant in variants:
                    base=archive/variant
                    merged=concatenate_traces([(name,base/name/'runtime.pipeline.csv') for name in result['phases']],
                                               base/'runtime.pipeline.csv')
                    assert merged==(total if variant=='candidate' else accepted['runtime_cycles'])
            result.update(status='passed',candidate_runner_sha256=sha(args.candidate_runner),
                          candidate_runtime_cycles=total,cycles_saved=accepted['runtime_cycles']-total,
                          speedup=accepted['runtime_cycles']/total)
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.write_text(json.dumps(result,indent=2))
    print(f"{report}: {result['status']}",flush=True)


if __name__=='__main__':
    main()
