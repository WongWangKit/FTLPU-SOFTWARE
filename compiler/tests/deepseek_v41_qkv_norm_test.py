"""Isolated layer-0 Q/KV RMSNorm checks. Projection outputs are test inputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler/tools'))
from common_kernel_graph import rms_norm_source


def reference_tensor(directory, manifest, name, shape):
    entry = manifest['tensors'][name]
    path = directory / entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'Reference checksum mismatch for {name}')
    tensor = np.load(path, allow_pickle=False)
    if entry['logical_dtype'] != 'bf16' or tensor.shape != shape:
        raise ValueError(f'Expected a BF16 reference tensor of shape {shape} for {name}')
    return tensor


def run(args, log, env=None):
    with log.open('w', encoding='utf-8') as stream:
        p = subprocess.run(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT, env=env)
    if p.returncode:
        raise RuntimeError(f'{args[0]} failed; see {log}')


def trace_times(path):
    periods = {}
    diagnostic = 0
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            resource = row['resource']
            start = int(row['start'])
            end = (int(row['end']) + (int(row['inner_count'])-1)*int(row['inner_interval'])
                   + (int(row['outer_count'])-1)*int(row['outer_interval']))
            old = periods.get(resource, (start, end))
            periods[resource] = (min(start,old[0]), max(end,old[1]))
            if resource == 'C2C.HostOutput':
                diagnostic += (int(row['end']) - start) * (
                    int(row['inner_count']) * int(row['outer_count']) - int(row['skip_first']))
    start, end = periods['Session.Invocation']
    ingress = min(start, periods.get('C2C.HostInput', (start, start))[0])
    return dict(trace_end_exclusive=max(b for _,b in periods.values()),
                invocation_cycles=end-start, cold_cycles_through_compute=end-ingress,
                initial_ingress_cycles=start-ingress, diagnostic_readback_cycles=diagnostic)


def bf16_input(value, shape):
    value = np.ascontiguousarray(value, dtype='<f4')
    if value.shape != shape or not np.isfinite(value).all() or (value.view('<u4') & 65535).any():
        raise ValueError(f'expected exact BF16 values with shape {shape}')
    return (value.view('<u4') >> 16).astype('<u2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--target-config', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True, help='Directory with weights and reference subdirectories')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Compile the two independent kernels, without running them')
    args = parser.parse_args()
    pkg = args.package.resolve()
    cfg = json.loads(args.config.read_text())['text_config']
    limit = dict(nrmse_max=.01, cosine_min=.9999)
    exe = lambda group,name: args.build_dir.resolve()/group/(name+('.exe' if os.name=='nt' else ''))
    failed = []
    if not args.prepare_only:
        fixture = json.loads((pkg/'reference'/'summary.json').read_text())
        weights = pkg/'weights'
        provenance = json.loads((weights/'provenance.json').read_text())
        if (fixture['status'] != 'independent_cpu_reference' or fixture['batch'] != 1
                or fixture['sequence'] != 32 or fixture['revision'] != provenance['revision']
                or hashlib.sha256(args.config.read_bytes()).hexdigest() != provenance['config_sha256']):
            raise ValueError('Expected the pinned layer-0 batch-one seq32 prefill reference and config')
    for role, input_name, width in [('q_norm','q_a',cfg['q_lora_rank']),('kv_norm','kv_raw',cfg['head_dim'])]:
        dest = pkg/'isolated_norms'/role
        dest.mkdir(parents=True, exist_ok=True)
        summary = dest/'summary.json'
        summary.write_text(json.dumps(dict(status='pending',scope='isolated RMSNorm only',limits=limit)))
        shape = (32, width)
        kernel = dest/'kernel.mlir'
        kernel.write_text(rms_norm_source(shape,cfg['rms_norm_eps']).replace(
            'module {','module attributes {ftlpu.sxm_wide = true} {',1), encoding='utf-8')
        program = dest/'program.ftlpu'
        run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel',
             '--output',program,'--target-config',args.target_config.resolve(),'--icu-compression','raw-macro'],dest/'compile.log')
        if args.prepare_only:
            summary.write_text(json.dumps(dict(status='compiled_only',scope='isolated RMSNorm only',limits=limit)))
            continue
        meta = provenance['tensors'][role]
        payload = (weights/meta['file']).read_bytes()
        if meta['shape'] != [width] or hashlib.sha256(payload).hexdigest() != meta['sha256']:
            raise ValueError(f'unexpected weight shape/digest for {role}')
        if meta['dtype'] != 'BF16':
            raise ValueError(f'{role} requires the original BF16 gamma')
        gamma = np.frombuffer(payload, '<u2')
        x_ref = reference_tensor(pkg/'reference', fixture, input_name, shape)
        expected = reference_tensor(pkg/'reference', fixture, role, shape)
        bf16_input(x_ref,shape).tofile(dest/'activation.bin')
        gamma.tofile(dest/'gamma.bin')
        listing = dest/'inputs.txt'
        listing.write_text(f'{dest/"activation.bin"}\n{dest/"gamma.bin"}\n', encoding='utf-8')
        env = {k: v for k, v in os.environ.items() if not k.startswith('FTLPU_')}
        env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide')
        trace = dest/'runtime.pipeline.csv'
        run([exe('runtime','compiled_stage_file_runner'),program,listing,dest/'actual',trace],dest/'execution.log',env)
        run([exe('runtime','ftlpu_icu_program_export'),dest/'actual.runtime.ftlpu',dest/'runtime_icu_programs',
             '--pre-execution-dir',dest/'actual.pre_execution'],dest/'export.log')
        actual = (np.fromfile(dest/'actual.0.bin','<u2').astype('<u4') << 16).view('<f4').reshape(shape)
        x,y = actual.astype('f8').ravel(), expected.astype('f8').ravel()
        error = float(np.linalg.norm(x-y)/np.linalg.norm(y))
        cosine_denominator = np.linalg.norm(x)*np.linalg.norm(y)
        cosine = float(x@y/cosine_denominator) if cosine_denominator else None
        stats = {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',(dest/'execution.log').read_text().splitlines()[-1])}
        passed = (np.isfinite(actual).all() and error<=limit['nrmse_max'] and cosine is not None and cosine>=limit['cosine_min']
                  and stats.get('host_operations')==0 and stats.get('outputs')==1)
        report = dict(status='passed' if passed else 'failed',scope='isolated RMSNorm only; projection output supplied as input fixture',
                      shape=shape,role=role,epsilon=cfg['rms_norm_eps'],limits=limit,nrmse=error,cosine=cosine,
                      target=args.target_config.name,revision=fixture['revision'],gamma_sha256=meta['sha256'],
                      input_npy_sha256=fixture['tensors'][input_name]['sha256'],
                      reference_npy_sha256=fixture['tensors'][role]['sha256'],
                      runtime=stats,timing=trace_times(trace),raw_macro=True)
        summary.write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
        if not passed:
            failed.append(role)
    if failed:
        raise AssertionError(f'Failed numerical cases: {", ".join(failed)}')


if __name__ == '__main__':
    main()
