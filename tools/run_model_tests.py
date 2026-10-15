#!/usr/bin/env python3
"""List, diagnose, and run the registered FTLPU model-level CModel tests.

Examples (from any working directory):
  python tools/run_model_tests.py list
  python tools/run_model_tests.py doctor --model qwen2_5_1_5b
  python tools/run_model_tests.py run --model qwen2_5_1_5b
  python tools/run_model_tests.py run --model glm_edge_1_5b --suite official --download

Each suite declares its actual coverage. These are decoder-layer tests, not
complete multi-layer text generation. Existing numerical harnesses remain the
authority for pass/fail; a compiled or skipped case is never a numerical pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'compiler/tools'))

DENSE = {
    'llama3_2_1b': 'Llama-3.2-1B', 'ministral3_3b': 'Ministral-3-3B',
    'phi4_mini': 'Phi-4-mini', 'smollm3_3b': 'SmolLM3-3B',
    'glm_edge_1_5b': 'GLM-Edge-1.5B', 'minicpm4_0_5b': 'MiniCPM4-0.5B',
}
CATALOG = {
    'qwen2_5_1_5b': dict(label='Qwen2.5-1.5B', suites=('prefill', 'decode', 'compile'),
                        scope='full-size synthetic layer; seq32 prefill / past32 decode1'),
    'qwen3_0_6b': dict(label='Qwen3-0.6B', suites=('prefill', 'compile'),
                      scope='full-size synthetic layer with Q/K norm; seq32 prefill'),
    'qwen3_5_0_8b': dict(label='Qwen3.5-0.8B', suites=('smoke', 'prefill', 'official', 'compile'),
                        scope='linear-attention layer only; mini synthetic smoke / official seq32'),
    **{name: dict(label=label, suites=('prefill', 'official', 'compile'),
                  scope='full-size nonzero-Q/K synthetic / official layer0 seq32')
       for name, label in DENSE.items()},
    'deepseek_native': dict(label='Native DeepSeek', suites=(), scope='not integrated into ICU/FU model tests'),
    'glm5_native': dict(label='Native GLM-5', suites=(), scope='not integrated; GLM-Edge coverage does not cover GLM-5'),
}
TOOL_NAMES = {'opt': ('compiler', 'ftlpu_opt'), 'compile': ('compiler', 'ftlpu-compile'),
              'export': ('runtime', 'ftlpu_icu_program_export'),
              'dense': ('runtime', 'hf_decoder_layer_model_session_test'),
              'qwen2': ('runtime', 'compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test'),
              'qwen3': ('runtime', 'compiled_qwen3_0_6b_decoder_layer_seq32_runtime_test'),
              'decode': ('runtime', 'compiled_qwen_real_decoder_layer_runtime_test'),
              'stage': ('runtime', 'compiled_stage_file_runner')}
BUILD_TARGETS = {key: name.replace('ftlpu-compile', 'ftlpu_compile')
                 for key, (_, name) in TOOL_NAMES.items()}


def read_cache(build):
    path = build / 'CMakeCache.txt'
    if not path.is_file():
        return {}
    return {line.split(':', 1)[0]: line.split('=', 1)[1]
            for line in path.read_text(encoding='utf-8', errors='replace').splitlines()
            if ':' in line and '=' in line and not line.startswith(('#', '//'))}


def default_build():
    for name in ('build', 'build-ftlpu-vs2026-direct', 'build-ftlpu-vs2026', 'build-vs2026'):
        path = ROOT / name
        cache = read_cache(path)
        if cache.get('CMAKE_HOME_DIRECTORY') and Path(cache['CMAKE_HOME_DIRECTORY']).resolve() == ROOT:
            return path
    return ROOT / 'build'


def executable(build, config, key):
    folder, name = TOOL_NAMES[key]
    suffix = '.exe' if os.name == 'nt' else ''
    candidates = [build / folder / config / (name + suffix), build / folder / (name + suffix)]
    return next((path for path in candidates if path.is_file()), candidates[1])


def tool_keys(model, suite, pipeline=True):
    keys = {'compile'}
    if model == 'qwen3_5_0_8b':
        keys.add('opt')
        if suite != 'compile': keys.add('stage')
    elif model.startswith('qwen'):
        if suite == 'decode': keys.update(('opt', 'decode', 'export'))
        elif suite != 'compile': keys.add('qwen2' if model.startswith('qwen2') else 'qwen3')
    elif suite == 'official': keys.add('dense')
    elif suite != 'compile': keys.add('dense')
    if pipeline and suite != 'compile' and model != 'qwen3_0_6b': keys.add('export')
    return keys


def result_directory(a, model):
    if a.output_root:
        return a.output_root / model / a.suite
    if model == 'qwen2_5_1_5b' and a.suite in ('prefill', 'decode'):
        return ROOT / 'results' / ('qwen2_5_decoder_layer_' + a.suite)
    if model == 'qwen3_5_0_8b' and a.suite in ('official', 'prefill'):
        return ROOT / 'results/qwen3_5_0_8b_decoder_layer_prefill/continuous'
    return ROOT / 'results/model_tests' / model / a.suite


def checkpoint_directory(a, model):
    return a.checkpoint or a.checkpoint_root / model


def selected_models(a):
    if a.all:
        models = [name for name, spec in CATALOG.items() if a.suite in spec['suites']]
    else:
        models = a.model or ['qwen2_5_1_5b']
    for model in models:
        if model not in CATALOG: raise ValueError(f'Unknown model: {model}; use list')
        if a.suite not in CATALOG[model]['suites']:
            raise ValueError(f'{model} does not support {a.suite}; supported: {CATALOG[model]["suites"] or "none"}')
    if a.checkpoint and len(models) != 1:
        raise ValueError('--checkpoint requires one model; use --checkpoint-root for a batch')
    return list(dict.fromkeys(models))


def needs_checkpoint(model, suite):
    return suite == 'official' or (model == 'qwen3_5_0_8b' and suite == 'prefill')


def check_environment(a, models, *, check_binaries=True):
    issues = []
    cache = read_cache(a.build_dir)
    if not a.target_config.is_file(): issues.append(f'Missing target config: {a.target_config}')
    cached_target = cache.get('FTLPU_HARDWARE_CONFIG')
    if cached_target and Path(cached_target).is_file() and a.target_config.is_file():
        if json.loads(Path(cached_target).read_text()) != json.loads(a.target_config.read_text()):
            issues.append('Target differs from the build configuration; configure a separate build with FTLPU_HARDWARE_CONFIG')
    for dependency in ('numpy',):
        if importlib.util.find_spec(dependency) is None:
            issues.append(f'Missing Python dependency: {dependency}; python -m pip install {dependency}')
    if a.download and importlib.util.find_spec('requests') is None:
        issues.append('Missing Python dependency: requests; python -m pip install requests')
    for model in models:
        if check_binaries:
            for key in sorted(tool_keys(model, a.suite, not a.no_pipeline)):
                path = executable(a.build_dir, a.config, key)
                if not path.is_file(): issues.append(f'Missing tool: {path}; build target {BUILD_TARGETS[key]}')
        if needs_checkpoint(model, a.suite):
            checkpoint = checkpoint_directory(a, model)
            if not a.download and not all((checkpoint / file).is_file() for file in ('config.json', 'model.safetensors', 'provenance.json')):
                issues.append(f'{model}: verified checkpoint missing at {checkpoint}; use --download or --checkpoint')
    return list(dict.fromkeys(issues))


def clean_environment(temp):
    environment = os.environ.copy()
    # Prevent diagnostic flags from turning a numerical run into link-only,
    # partial execution, direct uploads, or disabled validation.
    for key in list(environment):
        if key.startswith(('FTLPU_SESSION_', 'FTLPU_QWEN_', 'FTLPU_DENSE_', 'FTLPU_SKIP_QWEN_', 'FTLPU_DDR_')):
            environment.pop(key)
    environment.update(TEMP=str(temp), TMP=str(temp), PYTHONDONTWRITEBYTECODE='1', FTLPU_SESSION_PROGRESS='1')
    return environment


def commands(a, model, out, work):
    """Build explicit argument arrays; no shell interpolation or machine paths."""
    python = [sys.executable, '-B']
    compiler = str(executable(a.build_dir, a.config, 'compile'))
    opt = str(executable(a.build_dir, a.config, 'opt'))
    trace = out / 'runtime.pipeline.csv'
    phases = []
    if needs_checkpoint(model, a.suite) and a.download:
        checkpoint = checkpoint_directory(a, model)
        if not all((checkpoint / f).is_file() for f in ('config.json', 'model.safetensors', 'provenance.json')):
            phases.append(('download', python + [str(ROOT/'compiler/tools/download_official_dense_layer.py'),
                '--model', model, '--output-dir', str(checkpoint)]))
    if model == 'qwen3_5_0_8b':
        package = work / 'golden'
        if needs_checkpoint(model, a.suite):
            phases.append(('import', python + [str(ROOT/'compiler/tools/import_hf_qwen3_5_layer.py'),
                '--model-dir', str(checkpoint_directory(a, model)), '--output-dir', str(package), '--seq-len', '32']))
        command = python + [str(ROOT/'compiler/tests/qwen3_5_resident_decoder_layer_test.py'),
            '--opt', opt, '--compile', compiler, '--runner', str(executable(a.build_dir,a.config,'stage')),
            '--target-config', str(a.target_config), '--output-dir', str(out)]
        if needs_checkpoint(model, a.suite): command += ['--package', str(package)]
        if a.suite == 'compile': command += ['--compile-only']
        elif not a.no_pipeline: command += ['--pipeline-csv', str(trace)]
        phases.append(('resident', command))
    elif model.startswith('qwen'):
        source = ROOT/'compiler/examples'/('qwen2_5_1_5b_decoder_layer' if model.startswith('qwen2') else 'qwen3_0_6b_decoder_layer')/'decoder_layer_seq32.stablehlo.mlir'
        if a.suite == 'decode':
            phases.append(('decode', python + [str(ROOT/'compiler/tests/qwen2_5_1_5b_decode_cmodel_test.py'),
                '--opt',opt,'--compile',compiler,'--runtime-test',str(executable(a.build_dir,a.config,'decode')),
                '--icu-export',str(executable(a.build_dir,a.config,'export')), '--input',str(source),
                '--target-config',str(a.target_config),'--output-dir',str(out)]))
        else:
            phases.append(('compile', [compiler, '--input',str(source),
                '--output',str(out/'decoder_layer.ftlpu'),'--target-config',str(a.target_config),
                '--weight-bank','1','--kv-cache-capacity','256',
                '--mxm-execution','vector','--ffn-schedule','fused','--attention-schedule','tail',
                '--projection-rope-overlap','on']))
            if a.suite != 'compile':
                phases.append(('cmodel', [str(executable(a.build_dir,a.config,'qwen2' if model.startswith('qwen2') else 'qwen3')),
                                          str(out/'decoder_layer.ftlpu')]))
    elif a.suite == 'official':
        # Pass exact paths so checkpoint directory names need not match model IDs.
        checkpoint = checkpoint_directory(a,model)
        phases.append(('official', python + [str(ROOT/'compiler/tools/run_official_dense_acceptance.py'),
            '--models',model,'--model-dir',str(checkpoint),'--work',str(work.parent),
            '--result-dir',str(out),'--compiler',compiler,'--runtime',str(executable(a.build_dir,a.config,'dense')),
            '--target',str(a.target_config),'--jobs','1']))
        if a.no_pipeline: phases[-1][1].append('--no-pipeline')
    elif a.suite == 'compile':
        phases.append(('compile', python + [str(ROOT/'compiler/tools/compile_dense_decoder_case.py'),
            '--model',model,'--compiler',compiler,'--target-config',str(a.target_config), '--output',str(out/'decoder.ftlpu')]))
    else:
        phases.append(('synthetic', python + [str(ROOT/'compiler/tools/run_dense_decoder_smoke.py'),
            '--model',model,'--component','both','--qk-pattern','nonzero', '--compiler',compiler,
            '--runtime-test',str(executable(a.build_dir,a.config,'dense')),'--target-config',str(a.target_config),
            '--work-dir',str(work.parent),'--results-dir',str(out.parent),'--case-result-dir',str(out)]))
    return phases


def execute(command, log, environment):
    print('  ' + subprocess.list2cmdline(command), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                stream.write(line); stream.flush()
                print(line, end='', flush=True)
            return process.wait()


def digest(path):
    with path.open('rb') as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4*1024*1024), b''): result.update(chunk)
    return result.hexdigest()


def validate_checkpoint(a, model):
    from download_official_dense_layer import OFFICIAL_REPOS, EXTRA_OFFICIAL_REPOS
    directory = checkpoint_directory(a, model)
    provenance = json.loads((directory/'provenance.json').read_text(encoding='utf-8'))
    expected_repo = (OFFICIAL_REPOS | EXTRA_OFFICIAL_REPOS)[model]
    if (provenance.get('repo') != expected_repo or provenance.get('layer') != 0
            or provenance.get('embedding_token_ids', [])[:32] != list(range(32))
            or provenance.get('subset_sha256') != digest(directory/'model.safetensors')
            or provenance.get('config_sha256') != digest(directory/'config.json')):
        raise ValueError(f'{model}: official checkpoint provenance/config/subset mismatch at {directory}')
    return provenance


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4: raise
            time.sleep(0.1)


def run_case(a, model):
    out = result_directory(a, model).resolve()
    work = (ROOT/'tmp/model_tests'/model).resolve()
    plan = commands(a, model, out, work)
    if a.dry_run:
        print(f'[{model}/{a.suite}] {CATALOG[model]["scope"]}\n  results: {out}')
        for phase, command in plan: print(f'  [{phase}] '+subprocess.list2cmdline(command))
        return dict(model=model,suite=a.suite,status='planned',results=str(out))
    out.mkdir(parents=True,exist_ok=True); work.mkdir(parents=True,exist_ok=True)
    temp = ROOT/'tmp/model_tests/_temp'; temp.mkdir(parents=True,exist_ok=True)
    env = clean_environment(temp)
    official = needs_checkpoint(model,a.suite)
    report = dict(model=model,suite=a.suite,status='running',scope=CATALOG[model]['scope'],
        weights='official_w8a16' if official else ('synthetic_mini' if model=='qwen3_5_0_8b' else 'synthetic_full_shape'),
        runtime_attempted=False,runtime_executed=False,numeric_passed=False,started_at=datetime.now(timezone.utc).isoformat(),
        target_config=str(a.target_config),target_sha256=digest(a.target_config),phases=[],artifacts={},results=str(out))
    path = out/'runner.json'; write_json(path,report)
    for filename in ('runtime.pipeline.csv','pipeline.csv','pipeline.html','linked.ftlpu',
                     'decoder.linked.ftlpu','resident.runtime.ftlpu','summary.json','result.json','run.json',
                     'decoder_layer.stream.mlir','decoder_layer.schedule.mlir','decoder_layer.command.mlir'):
        (out/filename).unlink(missing_ok=True)
    # Snapshot counts can shrink after scheduling changes; never export leftover
    # images from an older run alongside the current runtime's instruction image.
    for name in ('pre_execution_programs','pre_execution'):
        for image in (out/name).glob('*.ftlpu'):
            image.unlink()
    for image in (out/'icu_programs').glob('*.icu.csv'):
        image.unlink()
    for name in ('index.csv','README.md'):
        (out/'icu_programs'/name).unlink(missing_ok=True)
    if not a.no_pipeline:
        trace = str(out/'runtime.pipeline.csv')
        env.update(FTLPU_DENSE_PIPELINE_CSV=trace, FTLPU_QWEN_PIPELINE_CSV=trace, FTLPU_QWEN_C2C_PIPELINE_CSV=trace,
                   FTLPU_QWEN_C2C_LINKED_BINARY=str(out/'linked.ftlpu'),
                   FTLPU_QWEN_C2C_PRE_EXECUTION_DIR=str(out/'pre_execution_programs'))
    try:
        for phase, command in plan:
            print(f'[{model}/{a.suite}: {phase}]',flush=True)
            if official and phase in ('import','official'):
                report['checkpoint_provenance'] = validate_checkpoint(a,model)
            if phase in ('cmodel','decode','synthetic','official','resident') and a.suite != 'compile':
                report['runtime_attempted'] = True
            start = time.monotonic(); code = execute(command,out/(phase+'.runner.log'),env)
            report['phases'].append(dict(phase=phase,command=command,returncode=code,wall_seconds=time.monotonic()-start))
            write_json(path,report)
            if code:
                message = (out/(phase+'.runner.log')).read_text(encoding='utf-8')
                if phase=='download' and a.allow_gated_skip and model=='llama3_2_1b' and 'GatedRepo' in message:
                    report.update(status='skipped_access',reason='Official repository access unavailable; explicitly allowed gated skip')
                    break
                backend_error = ''
                for name in ('result.json','summary.json'):
                    native = out/name
                    if native.is_file():
                        try:
                            detail = json.loads(native.read_text(encoding='utf-8')).get('error')
                            if detail: backend_error = ' '+str(detail)
                        except (ValueError,OSError):
                            pass
                hint = ' Rebuild the selected build directory with --build.' if 'unsupported FTLPU binary version' in message else ''
                raise RuntimeError(f'{phase} exited {code}; see {out/(phase+".runner.log")}.{backend_error}{hint}')
        else:
            report['runtime_executed'] = a.suite != 'compile'
            log = out/'cmodel.runner.log'
            if log.is_file():
                report['metrics'] = {key:float(value) for key,value in re.findall(
                    r'\b(\w*max_error|mean_error)\s*=\s*([-+\d.eE]+)',log.read_text(encoding='utf-8'))}
            for native_name in ('summary.json','result.json'):
                native = out/native_name
                if native.is_file():
                    value=json.loads(native.read_text(encoding='utf-8'))
                    report['native_report']=native_name
                    report['metrics']=value.get('hardware_vs_quantized_reference',value.get('max_abs_errors',{}))
                    if value.get('status')!='passed': raise RuntimeError(f'{native_name} did not report numerical success')
            report['numeric_passed'] = report['runtime_executed']
            report['status'] = 'passed' if report['numeric_passed'] else 'compiled'
            if not a.no_pipeline and a.suite!='compile':
                trace=out/('pipeline.csv' if a.suite=='decode' else 'runtime.pipeline.csv')
                if not trace.is_file(): raise RuntimeError('Runtime succeeded but requested pipeline CSV is missing')
                from run_dense_pipelines import embedded_viewer
                if not (out/'pipeline.html').is_file():
                    embedded_viewer(ROOT/'tools/pipeline_viewer',trace,out/'pipeline.html',CATALOG[model]['label'])
                report['artifacts'].update(pipeline_csv=trace.name,pipeline_html='pipeline.html')
                linked = next((p for p in (out/'linked.ftlpu',out/'decoder.linked.ftlpu',out/'resident.runtime.ftlpu') if p.is_file()),None)
                if linked:
                    export=[str(executable(a.build_dir,a.config,'export')),str(linked),str(out/'icu_programs')]
                    pre_execution = next((p for p in (out/'pre_execution_programs',out/'pre_execution') if p.is_dir()),None)
                    if pre_execution: export+=['--pre-execution-dir',str(pre_execution)]
                    start = time.monotonic()
                    code = execute(export,out/'icu_export.runner.log',env)
                    report['phases'].append(dict(phase='icu_export',command=export,returncode=code,
                        wall_seconds=time.monotonic()-start))
                    if code: raise RuntimeError(f'Actual ICU program export exited {code}; see icu_export.runner.log')
                    report['artifacts'].update(linked_program=linked.name,icu_programs='icu_programs')
    except Exception as error:
        report.update(status='failed',error=str(error))
    report['finished_at']=datetime.now(timezone.utc).isoformat(); write_json(path,report)
    print(f'[{model}/{a.suite}] {report["status"]}: {out}',flush=True)
    if report['status']=='failed': print('ERROR: '+report['error'],file=sys.stderr,flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action',choices=('list','doctor','run'))
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--model',nargs='+',help='Model IDs from list; default qwen2_5_1_5b')
    selection.add_argument('--all',action='store_true',help='Select all models supporting this suite')
    parser.add_argument('--suite',choices=('prefill','decode','official','smoke','compile'),default='prefill')
    parser.add_argument('--build-dir',type=Path,default=default_build())
    parser.add_argument('--config',default='Release')
    parser.add_argument('--target-config',type=Path)
    parser.add_argument('--output-root',type=Path,help='Override results as ROOT/model/suite')
    parser.add_argument('--checkpoint-root',type=Path,default=ROOT/'models/official_dense')
    parser.add_argument('--checkpoint',type=Path,help='Verified raw checkpoint directory for one model')
    parser.add_argument('--download',action='store_true',help='Download missing official layer subsets; reuse existing pinned checkpoint')
    parser.add_argument('--allow-gated-skip',action='store_true',help='Allow Llama download GatedRepo errors to be recorded as skipped')
    parser.add_argument('--no-pipeline',action='store_true',help='Disable optional trace/export; numerical validation stays enabled')
    parser.add_argument('--dry-run',action='store_true',help='Print commands without building, downloading or writing results')
    parser.add_argument('--build',action='store_true',help='Build required targets in an already configured build directory')
    a=parser.parse_args(argv)
    if a.action=='list':
        for name,spec in CATALOG.items(): print(f'{name:22} {",".join(spec["suites"]) or "UNSUPPORTED":32} {spec["scope"]}')
        print('prefill/official/decode are layer-level numerical tests; compile never executes CModel.\nsmoke is the synthetic mini Qwen3.5 linear-attention test. No whole-model generation claim.')
        return 0
    a.build_dir=a.build_dir.resolve()
    cache=read_cache(a.build_dir)
    a.target_config=(a.target_config or Path(cache.get('FTLPU_HARDWARE_CONFIG',str(ROOT.parent/'FTLPU-CMODEL/config/ftlpu-lpu32.json')))).resolve()
    for name in ('output_root','checkpoint_root','checkpoint'):
        if getattr(a,name): setattr(a,name,getattr(a,name).resolve())
    try:
        models=selected_models(a)
        if a.action=='run' and a.dry_run:
            for model in models: run_case(a,model)
            return 0
        if a.build and a.action=='run':
            if not cache: raise ValueError('--build needs CMakeCache.txt; configure CMake first (see README)')
            targets=sorted({BUILD_TARGETS[key] for model in models for key in tool_keys(model,a.suite,not a.no_pipeline)})
            subprocess.run(['cmake','--build',str(a.build_dir),'--config',a.config,'--target',*targets],cwd=ROOT,check=True)
        issues=check_environment(a,models)
        print(f'Python: {sys.executable}\nBuild: {a.build_dir}\nTarget: {a.target_config}')
        if issues:
            for issue in issues: print('ERROR: '+issue,file=sys.stderr)
            return 2
        if a.action=='doctor':
            print('Environment ready. Numerical model execution has not been performed.')
            return 0
        reports=[run_case(a,model) for model in models]
        summary=a.output_root or ROOT/'results/model_tests'
        summary.mkdir(parents=True,exist_ok=True)
        write_json(summary/'last_run.json',dict(cases=reports,passed=sum(r['status']=='passed' for r in reports),
            failed=sum(r['status']=='failed' for r in reports),skipped=sum(r['status']=='skipped_access' for r in reports),
            compiled=sum(r['status']=='compiled' for r in reports)))
        return 1 if any(r['status']=='failed' for r in reports) else 0
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        print('ERROR: '+str(error),file=sys.stderr); return 2


if __name__=='__main__':
    raise SystemExit(main())
