#!/usr/bin/env python3
"""Real checkpoint/dimensions through compiled ICU/FU arithmetic.

This is the serial stage correctness harness, not a resident or DDR-paged
single-program performance test. It reports that distinction explicitly.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import CompiledStages, load_imported_layer
from qwen3_5_reference import linear_decoder_layer


def metrics(actual, expected):
    a, e = actual.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
    if a.shape != e.shape or not np.isfinite(a).all():
        raise AssertionError('invalid output shape or non-finite values')
    error = a - e
    norm = np.linalg.norm(e)
    return dict(max_abs=float(np.max(np.abs(error))), rms=float(np.sqrt(np.mean(error*error))),
                normalized_rms=float(np.linalg.norm(error)/max(norm,1e-30)),
                cosine=float(np.dot(a,e)/max(np.linalg.norm(a)*norm,1e-30)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--compile', type=Path, required=True)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--check-existing', action='store_true', help='Validate completed CLI outputs without rerunning')
    a = p.parse_args()
    x, weights, packed, scales, c = load_imported_layer(a.package)
    metadata = json.loads((a.package/'metadata.json').read_text(encoding='utf-8'))
    provenance = metadata.get('checkpoint_provenance')
    if not provenance or provenance['repo'] != 'Qwen/Qwen3.5-0.8B':
        raise ValueError('requires the verified official Qwen3.5-0.8B checkpoint subset')
    if (c.hidden_size,c.intermediate_size,c.num_value_heads,c.key_head_dim,c.value_head_dim) != (1024,3584,16,128,128):
        raise ValueError('model dimensions do not match the official real-size configuration')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    summary_path = a.output_dir/'real_validation.json'
    summary_path.unlink(missing_ok=True)
    tensors = {}
    if a.check_existing:
        output = np.fromfile(a.output_dir/'output.f32.bin','<f4').reshape(x.shape)
        tensors['recurrent_state'] = np.fromfile(a.output_dir/'state.f32.bin','<f4').reshape(c.num_value_heads,c.key_head_dim,c.value_head_dim)
        tensors['conv_state'] = np.fromfile(a.output_dir/'conv_state.f32.bin','<f4').reshape(c.conv_width,c.conv_kernel_size)
        programs = json.loads((a.output_dir/'stages.json').read_text(encoding='utf-8'))
    else:
        stages = CompiledStages(a.compile.resolve(),a.runner.resolve(),a.output_dir.resolve())
        output,tensors = stages.layer(x,weights,packed,scales,c)
        programs = stages.stages
    output.astype('<f4').tofile(a.output_dir/'output.f32.bin')
    tensors['recurrent_state'].astype('<f4').tofile(a.output_dir/'state.f32.bin')
    tensors['conv_state'].astype('<f4').tofile(a.output_dir/'conv_state.f32.bin')
    expected_stages = {}
    expected,state = linear_decoder_layer(x,weights,c,stages=expected_stages)
    checks = {'output':metrics(output,expected),
              'recurrent_state':metrics(tensors['recurrent_state'],state.recurrent),
              'conv_state':metrics(tensors['conv_state'],state.conv)}
    for name,value in tensors.items():
        value.astype('<f4').tofile(a.output_dir/f'actual.{name}.f32.bin')
        if name in expected_stages:
            checks[name] = metrics(value,expected_stages[name])
    expected.astype('<f4').tofile(a.output_dir/'golden.output.f32.bin')
    state.recurrent.astype('<f4').tofile(a.output_dir/'golden.state.f32.bin')
    # Fixed before running: aggregate BF16/LUT error may amplify across a
    # trained layer, but a layout/algorithm failure must not be marked passed.
    limits = dict(normalized_rms=0.03,cosine=0.999)
    passed = all(checks[n]['normalized_rms'] <= limits['normalized_rms'] and checks[n]['cosine'] >= limits['cosine']
                 for n in ('output','recurrent_state','conv_state'))
    summary = dict(status='passed' if passed else 'failed', model=provenance['repo'],
        revision=provenance['revision'], layer=metadata['layer'], layer_type=metadata['layer_type'],
        seq_len=x.shape[0],hidden_size=c.hidden_size,intermediate_size=c.intermediate_size,
        key_heads=c.num_key_heads,value_heads=c.num_value_heads,head_dim=c.key_head_dim,
        synthetic_weights=False,projection_format=metadata['projection_format'],
        reference='FP32 NumPy equations with the same W8A16 quantized checkpoint weights',
        execution='serial_compiled_stages_with_host_layout_transfers',
        single_resident_program=False,ddr_c2c_paging=False,arithmetic_backend='CModel ICU/FU',
        program_count=len(programs),sum_binary_bytes=sum(s['binary_bytes'] for s in programs),
        pass_limits=limits,metrics=checks)
    summary_path.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if not passed:
        raise AssertionError('real-size checkpoint numerical validation failed; see real_validation.json')


if __name__ == '__main__':
    main()
