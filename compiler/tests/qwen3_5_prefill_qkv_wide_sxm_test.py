"""Incremental real-weight seq32 GDN frontend on selectable SXM hardware.

Only logical graph construction and independent references live here; C++ owns
layout, paging and raw-macro lowering. Optional token-0 decay/KS stops before
State update; no RMSNorm or FFN is added.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import load_imported_layer, bf16
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_reference import causal_depthwise_conv1d
from qwen3_5_qkv_conv_prefill_test import compare
from qwen3_5_decode_gdn_attention_test import run
from scaled_matvec_streaming_test import audit_streaming
from causal_exp_difference_runtime_test import causal_exp_source, check as check_causal_exp, reference as causal_exp_reference, read_transposed
from weighted_gram_runtime_test import weighted_gram_source, reference as weighted_gram_reference, read_head_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--sxm-architecture', choices=('wide', 'local'), default='wide')
    parser.add_argument('--with-qk-norm', action='store_true')
    parser.add_argument('--export-qkv', action='store_true', help='Diagnostic intermediate export; adds QKV output traffic')
    parser.add_argument('--validate-only', action='store_true', help='Validate existing CModel outputs and export runtime instructions')
    parser.add_argument('--with-gates', action='store_true', help='Append beta and log-decay using VXM Log Softplus')
    parser.add_argument('--with-state-decay', action='store_true',
                        help='Tokenwise first step: exp(g[0]) times a nonzero old State; stop before KS/update')
    parser.add_argument('--with-ks', action='store_true',
                        help='Stream token-0 State decay into MXM for KS; stop before State update')
    parser.add_argument('--with-gt', action='store_true',
                        help='Chunkwise first step: inclusive FP32 prefix sum of g along tokens; stop before gamma/R')
    parser.add_argument('--with-gamma', action='store_true',
                        help='Append gamma_t = exp(G_t), retaining G_t; stop before R/recurrence')
    parser.add_argument('--with-r', action='store_true',
                        help='Append causal R[i,j,h] = exp(G[i,h]-G[j,h]); stop before Gram/solve/State')
    parser.add_argument('--with-a', action='store_true',
                        help='Append A = (K K^T) * R through MXM/VXM; stop before beta/solve/State')
    parser.add_argument('--with-l',action='store_true',help='Fuse strict lower beta weighting into R, then form L through MXM; stop before solve')
    parser.add_argument('--fp32-ingress', action=argparse.BooleanOptionalAction, default=None,
                        help='Read complete FP32 streams; enabled by default for --with-l')
    args = parser.parse_args()
    if args.fp32_ingress is None:
        args.fp32_ingress = args.with_l
    args.with_a |= args.with_l
    args.with_state_decay |= args.with_ks
    args.with_r |= args.with_a
    args.with_gamma |= args.with_r
    args.with_gt |= args.with_gamma
    if args.with_gt and args.with_state_decay:
        parser.error('--with-gt is the chunkwise branch; do not combine it with tokenwise decay/KS')
    args.with_gates |= args.with_state_decay or args.with_gt
    args.with_qk_norm |= args.with_gates
    if args.with_qk_norm and args.export_qkv:
        parser.error('--export-qkv is not supported with --with-qk-norm')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(status='pending', scope='QKV projection + causal Conv1D/SiLU only',
                   sxm_architecture=args.sxm_architecture, icu_compression='raw-macro')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    build = args.build_dir.resolve()
    suffix = '.exe' if sys.platform == 'win32' else ''
    exe = lambda area, name: build / area / (name + suffix)
    x, weights, packed, scales, config = load_imported_layer(args.package)
    x = bf16(x)
    assert x.shape == (32, 1024) and config.conv_width == 6144
    # A real, nonzero external input exposes head broadcast and address errors.
    # Later tokens require the updated State and are intentionally not evaluated.
    state = np.random.default_rng(35261030).normal(0, .02,
        (config.num_value_heads * config.key_head_dim, config.value_head_dim)).astype('f4')
    if args.with_ks:
        # External input packing only. Put the contraction dimension in lanes
        # so VXM's scaled result can feed MXM IW without a runtime transpose.
        state = state.reshape(config.num_value_heads, config.key_head_dim,
            config.value_head_dim).transpose(0, 2, 1).copy().reshape(-1, config.key_head_dim)
    if not args.validate_only:
        graph = LogicalGraph()
        qkv = graph.project('in_proj_qkv', x, packed['in_proj_qkv'], scales['in_proj_qkv'])
        conv = graph.conv(qkv, weights.conv1d)
        finals = [qkv, conv] if args.export_qkv else [conv]
        if args.with_qk_norm:
            width, dim = config.key_width, config.key_head_dim
            query = graph.norm('query_l2', conv[:, :width].reshape(-1, dim),
                               np.ones(dim), 1e-6, False, 1/np.sqrt(dim)).reshape(32, width)
            key = graph.norm('key_l2', conv[:, width:2*width].reshape(-1, dim),
                             np.ones(dim), 1e-6, False).reshape(32, width)
            finals = [query, key, conv[:, 2*width:]]
            if args.with_ks:
                # Export alternating native head groups separately. State's
                # long-lived SRAM allocation can give these different bases;
                # validation must not force a full V archive to join them.
                native_v = conv[:, 2*width:].reshape(32, config.num_value_heads, config.value_head_dim)
                finals[2] = native_v[:, ::2, :].reshape(32, config.value_width // 2)
        if args.with_gates:
            beta, decay = graph.gates(x, packed, scales, weights.dt_bias, weights.a_log)
            finals += [beta, decay]
        if args.with_gt:
            gt = graph.vector_tiles('g_prefix_sum', [(decay, 'f32')],
                lambda v: v.add(0, kind='cumsum_rows', immediate=0))
            finals += [gt]
        if args.with_gamma:
            gamma = graph.vector_tiles('chunk_gamma', [(gt, 'f32')],
                lambda v: v.add(0, kind='exp', immediate=0))
            finals += [gamma]
        if args.with_r:
            r_shape,r_dtype=((32*config.num_value_heads,32),'bf16') if args.with_a else ((32*32,config.num_value_heads),'f32')
            r_matrix = graph.run('chunk_causal_decay', causal_exp_source(32, config.num_value_heads,args.with_a,args.with_l,args.with_l),
                [(gt, 'f32')]+([(beta,'f32')] if args.with_l else []), [(r_shape,r_dtype)])[0]
            finals += [r_matrix]
        if args.with_a:
            a_matrix = graph.run('chunk_weighted_gram', weighted_gram_source(config.num_value_heads, config.key_head_dim,True),
                [(key, 'bf16'), (r_matrix, 'bf16')], [((config.num_value_heads*32,32), 'f32')])[0]
            finals += [a_matrix]
        if args.with_state_decay:
            factor = graph.vector_tiles('state_decay_factor', [(decay[:1], 'f32')],
                lambda v: v.add(0, kind='exp', immediate=0))
            if args.with_ks:
                keys = key[:1].reshape(config.num_key_heads, config.key_head_dim).repeat(
                    config.num_value_heads // config.num_key_heads, axis=0)
                mt, dt, kt, pt = (graph.type(s, t) for s, t in (
                    (state.shape, 'f32'), (factor.shape, 'f32'), (keys.shape, 'bf16'),
                    ((config.num_value_heads, config.value_head_dim), 'f32')))
                source = f'''module {{ func.func @stage(%s: {mt}, %d: {dt}, %k: {kt}) -> ({mt}, {pt}) {{
                  %scaled, %p = ftlpu.kernel.scaled_matvec %s, %d, %k : ({mt}, {dt}, {kt}) -> ({mt}, {pt})
                  return %scaled, %p : {mt}, {pt}
                }} }}'''
                decayed, prediction = graph.run('state_decay_ks', source,
                    [(state, 'f32'), (factor, 'f32'), (keys, 'bf16')],
                    [(state.shape, 'f32'), ((config.num_value_heads, config.value_head_dim), 'f32')])
                finals += [factor, decayed, prediction]
                finals += [native_v[:, 1::2, :].reshape(32, config.value_width // 2)]
            else:
                coefficients = factor.reshape(config.num_value_heads, 1).repeat(
                    config.key_head_dim, axis=0).repeat(config.value_head_dim, axis=1)
                decayed = graph.vector_tiles('state_decay', [(state, 'f32'), (coefficients, 'f32')],
                    lambda v: v.add(0, 1))
                # Export the planner's two bounded head ranges without joining them.
                half = state.shape[0] // 2
                finals += [factor, decayed[:half], decayed[half:]]
        logical = graph.write(out / 'logical_graph', finals)
        kernel, binary = out / 'frontend.kernel.mlir', out / 'frontend.ftlpu'
        run([exe('compiler', 'ftlpu-graph-compile'), '--input', logical, '--output', kernel,
             '--emit-kernel', '--target-config', args.target_config.resolve()], out / 'import.log')
        kernel.write_text(kernel.read_text().replace('module {',
            'module attributes {ftlpu.vector_bf16_spill = true, ftlpu.common_barrier_protocol = true, '
            + f'ftlpu.vector_fp32_ingress = {str(args.fp32_ingress).lower()}, '
            + f'ftlpu.vector_bf16_token_halves = {str(args.with_gates).lower()}, ftlpu.vector_log_softplus = {str(args.with_gates).lower()}, ftlpu.l2_head_partition = {str(args.with_qk_norm and args.sxm_architecture == "wide").lower()}, ftlpu.sxm_wide = {str(args.sxm_architecture == "wide").lower()}' + '} {', 1))
        run([exe('compiler', 'ftlpu-compile'), '--input', kernel, '--input-stage', 'kernel',
             '--output', binary, '--target-config', args.target_config.resolve(),
             '--icu-compression', 'raw-macro', '--mxm-execution', 'ws'], out / 'compile.log')
        plan = json.loads((out / 'graph_plan.json').read_text())
        assert not any(n['name'].startswith('conv_layout_group') for n in plan['nodes']), \
            'Conv groups must retain native producer views instead of joining through MEM'
        if args.with_a:
            gram=next(n for n in plan['nodes'] if n['name']=='chunk_weighted_gram')
            assert gram['nominal_end']-gram['nominal_start']<1500, 'Continuous KKT schedule was not emitted; check compiler build'
        os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
        os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE'] = args.sxm_architecture
        os.environ['FTLPU_SESSION_PROGRESS'] = '1'
        run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
             out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log')
    def read(index, width=6144):
        return (np.fromfile(out / f'actual.{index}.bin', '<u2').astype('u4') << 16).view('f4').reshape(32, width)
    actual_conv = None if args.with_qk_norm else read(1 if args.export_qkv else 0)
    expected_qkv = x.astype('f4') @ weights.in_proj_qkv.T
    history = np.zeros_like(weights.conv1d)
    expected_conv, _ = causal_depthwise_conv1d(expected_qkv, weights.conv1d, history)
    if args.with_qk_norm:
        width, dim = config.key_width, config.key_head_dim
        def norm(x, scale):
            x = x.reshape(32, -1, dim)
            return (x / np.sqrt(np.sum(x*x, axis=-1, keepdims=True) + 1e-6) * scale).reshape(32, width)
        if args.with_ks:
            actual_v = np.empty((32, config.num_value_heads, config.value_head_dim), dtype='f4')
            for index, parity in ((2, 0), (8, 1)):
                actual_v[:, parity::2, :] = read(index, config.value_width // 2).reshape(
                    32, config.num_value_heads // 2, config.value_head_dim)
            actual_v = actual_v.reshape(32, config.value_width)
        else:
            actual_v = read(2, config.conv_width-2*width)
        errors = dict(query_l2=compare(read(0, width), norm(expected_conv[:, :width], 1/np.sqrt(dim))),
                      key_l2=compare(read(1, width), norm(expected_conv[:, width:2*width], 1)),
                      value=compare(actual_v, expected_conv[:, 2*width:]))
        summary['scope'] = 'QKV projection + Conv1D/SiLU + Q/K L2Norm; Q scaled; V preserved'
    else:
        errors = dict(conv_silu=compare(actual_conv, expected_conv))
    if args.with_gates:
        heads = config.num_value_heads
        b = x.astype('f4') @ weights.in_proj_b.T
        a = x.astype('f4') @ weights.in_proj_a.T + weights.dt_bias
        beta_reference = 1 / (1 + np.exp(-b))
        decay_reference = -np.exp(weights.a_log) * np.logaddexp(0, a)
        gate_actual = [np.fromfile(out / f'actual.{i}.bin', '<f4').reshape(32, heads) for i in (3,4)]
        errors.update(beta=compare(gate_actual[0], beta_reference), g=compare(gate_actual[1], decay_reference))
        # Do not let a large head hide errors in smaller decay values. The
        # absolute floor accounts for the existing Exp FP16 FTZ behavior.
        np.testing.assert_allclose(gate_actual[0], beta_reference, rtol=.01, atol=1e-4)
        np.testing.assert_allclose(gate_actual[1], decay_reference, rtol=.05, atol=1e-7)
        summary['gate_elementwise_tolerances'] = dict(beta=dict(rtol=.01, atol=1e-4), g=dict(rtol=.05, atol=1e-7))
        summary['scope'] += '; beta and g (Log Softplus)'
        summary['gate_output_shapes'] = [[32, heads], [32, heads]]
        summary['gate_precision'] = 'BF16 ingress at spill boundaries, FP32 chain arithmetic/output; FP16 LUT ingress'
    if args.with_gt:
        gt_actual = np.fromfile(out / 'actual.5.bin', '<f4').reshape(32, config.num_value_heads)
        ingress = gate_actual[1] if args.fp32_ingress else (gate_actual[1].view('u4') & np.uint32(0xffff0000)).view('f4')
        gt_reference = np.cumsum(ingress, axis=0, dtype=np.float32)
        np.testing.assert_allclose(gt_actual, gt_reference, rtol=1e-6, atol=1e-7)
        errors['gt_given_actual_bf16_g'] = compare(gt_actual, gt_reference)
        errors['gt_end_to_end'] = compare(gt_actual, np.cumsum(decay_reference, axis=0, dtype=np.float32))
        summary['scope'] += '; G_t inclusive prefix sum' + ('' if args.with_gamma else ' only; no gamma/R or recurrence')
        summary['chunkwise_prefix'] = dict(shape=[32, config.num_value_heads], output_index=5,
            axis=0, inclusive=True, accumulator='FP32, reset only on first token, emit every token',
            ingress_precision='FP32 full four-plane g reload' if args.fp32_ingress else 'BF16 upper-plane reload of FP32 g buffer', output_precision='FP32',
            lane_mapping='one head per lane, tokens arrive in time order',
            hemisphere_policy='mirrored full scan; no independent token halves')
    if args.with_gamma:
        gamma_actual = np.fromfile(out / 'actual.6.bin', '<f4').reshape(32, config.num_value_heads)
        gamma_ingress = gt_actual if args.fp32_ingress else (gt_actual.view('u4') & np.uint32(0xffff0000)).view('f4')
        # Hardware Exp has FP16 input/output boundaries and flushes FP16
        # subnormals to zero. Compare with independent exp, not the CModel LUT.
        lut_input = gamma_ingress.astype('f2').astype('f4')
        lut_input[np.abs(lut_input) < np.finfo(np.float16).tiny] = 0
        gamma_reference = np.exp(lut_input)
        np.testing.assert_allclose(gamma_actual, gamma_reference, rtol=.002,
                                   atol=float(np.finfo(np.float16).tiny))
        assert np.isfinite(gamma_actual).all() and np.all((gamma_actual >= 0) & (gamma_actual <= 1))
        assert np.all(np.diff(gamma_actual, axis=0) <= 0), 'Nonpositive g must give nonincreasing gamma'
        errors['gamma_given_actual_gt'] = compare(gamma_actual, np.exp(gt_actual))
        errors['gamma_given_quantized_gt'] = compare(gamma_actual, gamma_reference)
        gamma_model = np.exp(np.cumsum(decay_reference, axis=0, dtype=np.float32))
        errors['gamma_end_to_end'] = compare(gamma_actual, gamma_model)
        per_head = [compare(gamma_actual[:, h], gamma_model[:, h]) for h in range(config.num_value_heads)]
        local_per_head = [compare(gamma_actual[:, h], np.exp(gt_actual[:, h]))
                          for h in range(config.num_value_heads)]
        assert all(e['nrmse'] < .03 and e['cosine'] > .999 for e in local_per_head), local_per_head
        # Track upstream gate/scan error separately: exp amplifies its absolute
        # input error. A global average must not hide an inaccurate head.
        limited_heads = [h for h, e in enumerate(per_head) if e['nrmse'] >= .03 or e['cosine'] <= .999]
        summary['scope'] += '; gamma_t = exp(G_t)' + ('' if args.with_r else '; no R, triangular solve or State recurrence')
        summary['chunkwise_gamma'] = dict(shape=[32, config.num_value_heads], output_index=6,
            formula='exp(inclusive_cumsum(g, axis=0))', retained_gt_output=5,
            input_precision='BF16 upper-plane reload of FP32 G_t at native address',
            compute_precision='existing hardware Exp LUT with FP16 FTZ input/output', output_precision='FP32 buffer ABI',
            hemisphere_policy='distinct 16-token halves; one head per lane',
            elementwise_tolerance=dict(rtol=.002, atol=float(np.finfo(np.float16).tiny)),
            zero_outputs=int(np.count_nonzero(gamma_actual == 0)), per_head_end_to_end_errors=per_head,
            per_head_given_actual_gt_errors=local_per_head,
            end_to_end_head_tolerance=dict(nrmse_max=.03, cosine_min=.999),
            end_to_end_head_check_passed=not limited_heads, precision_limited_heads=limited_heads)
    if args.with_r:
        actual_r = read_transposed(out/'actual.7.bin',config.num_value_heads) if args.with_a else np.fromfile(out / 'actual.7.bin', '<f4').reshape(32, 32, config.num_value_heads)
        r_checks = check_causal_exp(actual_r, gt_actual,args.with_a,gate_actual[0] if args.with_l else None,args.with_l,args.fp32_ingress)
        causal = np.tri(32,k=-1 if args.with_l else 0, dtype=bool)[:, :, None]
        reference_r = lambda g: np.exp(np.where(causal, g[:, None, :]-g[None, :, :], -np.inf))
        expected_r = reference_r(np.cumsum(decay_reference, axis=0, dtype=np.float32))
        local_r=causal_exp_reference(gt_actual,args.fp32_ingress)
        actual_gt_r=reference_r(gt_actual)
        if args.with_l:
            beta_ingress=gate_actual[0] if args.fp32_ingress else (gate_actual[0].view('u4') & np.uint32(0xffff0000)).view('f4')
            local_r=local_r*beta_ingress[:,None,:]*causal
            actual_gt_r=actual_gt_r*gate_actual[0][:,None,:]
            expected_r=expected_r*beta_reference[:,None,:]
        errors['r_given_quantized_gt'] = compare(actual_r, local_r)
        errors['r_given_actual_gt'] = compare(actual_r, actual_gt_r)
        errors['r_end_to_end'] = compare(actual_r, expected_r)
        per_head = [compare(actual_r[:,:,h], expected_r[:,:,h]) for h in range(config.num_value_heads)]
        limited_heads = [h for h,e in enumerate(per_head) if e['nrmse'] >= .03 or e['cosine'] <= .999]
        summary['scope'] += ('; strict lower coefficient C=beta_i*R_ij' if args.with_l else '; causal R') + ('' if args.with_a else '; no KKt, triangular solve or State recurrence')
        summary['chunkwise_r'] = dict(shape=[32,32,config.num_value_heads], output_index=7,
            stored_shape=[32*config.num_value_heads,32] if args.with_a else [1024,config.num_value_heads],
            storage_layout='[i*H+head, j]; VXM directly to SXM, one j per lane' if args.with_a else '[i*T+j, head]; one head per lane, pair in successive MEM rows',
            formula='beta[i,h]*exp(G[i,h]-G[j,h]) for j<i; diagonal/upper=0' if args.with_l else 'exp(G[i,h]-G[j,h]) for j<i; diagonal=1; upper=0',
            row_scale=args.with_l,strict_lower=args.with_l,
            source='G_t output 5, not a quotient of gamma',
            precision=('FP32 G/beta ingress; ' if args.fp32_ingress else 'BF16 G/beta ingress; ')+'FP32 subtraction; FP16 FTZ Exp; '+('BF16 upper-plane output ABI' if args.with_a else 'FP32 output ABI'),
            direct_vxm_sxm=args.with_a, raw_r_mem_spill=not args.with_a,
            hemisphere_policy='mirrored head vectors, retaining existing G_t contract',
            difference_spill=False, expanded_g_broadcast=False, upper_triangle_exp=False,
            local_checks=r_checks, per_head_end_to_end_errors=per_head,
            end_to_end_head_check_passed=not limited_heads, precision_limited_heads=limited_heads)
    if args.with_a:
        actual_a = read_head_groups(out / 'actual.8.bin', config.num_value_heads)
        actual_keys = read(1,config.key_width)
        given_actual = weighted_gram_reference(actual_keys,actual_r)
        model_keys = norm(expected_conv[:,config.key_width:2*config.key_width],1)
        expected_a = weighted_gram_reference(model_keys,expected_r)
        errors['a_given_actual_k_r'] = compare(actual_a,given_actual)
        errors['a_end_to_end'] = compare(actual_a,expected_a)
        per_head = [compare(actual_a[h],expected_a[h]) for h in range(config.num_value_heads)]
        local_per_head = [compare(actual_a[h],given_actual[h]) for h in range(config.num_value_heads)]
        assert all(e['nrmse'] < .03 and e['cosine'] > .999 for e in local_per_head), local_per_head
        assert np.isfinite(actual_a).all()
        assert np.all(actual_a[:,np.triu_indices(32,0 if args.with_l else 1)[0],np.triu_indices(32,0 if args.with_l else 1)[1]]==0)
        limited_heads = [h for h,e in enumerate(per_head) if e['nrmse']>=.03 or e['cosine']<=.999]
        summary['scope'] += ('; L=strict_lower(beta_i*R_ij*(KK^T)_ij); no triangular solve or State recurrence' if args.with_l else '; A=(KK^T)*R; no beta weighting, triangular solve or State recurrence')
        summary['chunkwise_a'] = dict(shape=[config.num_value_heads,32,32],output_index=8,
            stored_head_order=list(range(0,config.num_value_heads,2))+list(range(1,config.num_value_heads,2)),
            formula='L[h,i,j] = beta[i,h]*sum_d K[i,h,d]*K[j,h,d]*R[i,j,h], j<i; otherwise 0' if args.with_l else 'A[h,i,j] = sum_d K[i,h,d]*K[j,h,d] * R[i,j,h]',
            precision='BF16 K, FP32 MXM accumulation, BF16 Gram stream/R ingress, FP32 VXM multiply/output',
            gram_spill=False,weight_load_mode='1vector',
            hemisphere_policy='distinct alternating heads; native partitioned output',
            per_head_given_actual_inputs_errors=local_per_head,
            per_head_end_to_end_errors=per_head,end_to_end_head_check_passed=not limited_heads,
            precision_limited_heads=limited_heads)
        if args.with_l:
            # Separate accumulated-g/BF16-difference error from the new fused
            # row weighting. Strict masking removes the large unit diagonal
            # that otherwise hides relative errors in strongly decaying heads.
            summary['chunkwise_a']['precision_diagnostics'] = dict(
                given_actual_gt_per_head=[compare(actual_a[h],weighted_gram_reference(actual_keys,actual_gt_r)[h]) for h in range(config.num_value_heads)],
                given_quantized_gt_per_head=[compare(actual_a[h],weighted_gram_reference(actual_keys,local_r)[h]) for h in range(config.num_value_heads)])
    if args.with_state_decay:
        factor_actual = np.fromfile(out / 'actual.5.bin', '<f4').reshape(1, config.num_value_heads)
        state_indices = (6,) if args.with_ks else (6, 7)
        state_actual = np.concatenate([np.fromfile(out / f'actual.{i}.bin', '<f4')
                                       for i in state_indices]).reshape(state.shape)
        def decayed_reference(g):
            return state * np.repeat(np.exp(g[:1]).reshape(-1, 1), config.key_head_dim, axis=0)
        errors.update(state_decay_factor=compare(factor_actual, np.exp(gate_actual[1][:1])),
                      state_decay_given_actual_g=compare(state_actual, decayed_reference(gate_actual[1])),
                      state_decay_end_to_end=compare(state_actual, decayed_reference(decay_reference)))
        per_head = []
        reference = decayed_reference(gate_actual[1])
        for h in range(config.num_value_heads):
            r = slice(h * config.key_head_dim, (h + 1) * config.key_head_dim)
            e = compare(state_actual[r], reference[r])
            assert e['nrmse'] < .03 and e['cosine'] > .999, ('state head', h, e)
            per_head.append(e)
        summary['scope'] += '; token 0 old State decay' + (' and KS prediction' if args.with_ks else ' only')
        summary['state_decay'] = dict(strategy='tokenwise', token_index=0,
            nonzero_initial_state=True, state_shape=[config.num_value_heads, config.key_head_dim, config.value_head_dim],
            logical_layout='[head*value, key]; lanes carry 32 keys' if args.with_ks else '[head*key, value]; lanes carry 32 values',
            recurrence_update=False, precision='BF16-truncated inputs, FP32 multiply/output; FP32 buffer ABI',
            factor_output=5, state_outputs=list(state_indices), per_head_errors=per_head)
        if args.with_ks:
            keys = read(1, config.key_width)[:1].reshape(config.num_key_heads, config.key_head_dim).repeat(
                config.num_value_heads // config.num_key_heads, axis=0)
            prediction = np.fromfile(out / 'actual.7.bin', '<f4').reshape(config.num_value_heads, config.value_head_dim)
            truncated = (state_actual.view('u4') & np.uint32(0xffff0000)).view('f4')
            contraction = lambda s, k: np.einsum('hvk,hk->hv', s.reshape(
                config.num_value_heads, config.value_head_dim, config.key_head_dim), k)
            ref = contraction(truncated, keys)
            errors['ks_given_actual_scaled_state'] = compare(prediction, ref)
            for h in range(config.num_value_heads):
                e = compare(prediction[h], ref[h])
                assert e['nrmse'] < 1e-5 and e['cosine'] > .999999, ('KS head', h, e)
            key_reference = norm(expected_conv[:, config.key_width:2*config.key_width], 1)[:1].reshape(
                config.num_key_heads, config.key_head_dim).repeat(config.num_value_heads // config.num_key_heads, axis=0)
            errors['ks_end_to_end'] = compare(prediction, contraction(decayed_reference(decay_reference), key_reference))
    if args.export_qkv:
        actual_qkv = read(0)
        isolated_conv, _ = causal_depthwise_conv1d(actual_qkv, weights.conv1d, history)
        errors.update(qkv=compare(actual_qkv, expected_qkv),
                      conv_given_actual_qkv=compare(actual_conv, isolated_conv))
    with (out / 'runtime.pipeline.csv').open() as f:
        rows = list(csv.DictReader(f))
    sxm = [r for r in rows if any('SXM' in v.upper() for v in r.values())]
    if not args.with_qk_norm:
        assert not sxm, 'Direct projection/Conv layout should not need SXM operations'
    else:
        assert sxm, 'L2 normalization must execute its transpose path'
        if args.sxm_architecture == 'wide':
            schedule = (out / 'decoder_layer.schedule.mlir').read_text()
            assert 'l2norm.prepare_sxm_pair' not in schedule
            assert 'l2norm.replicate_feedback_tile' not in schedule
            for name in ('query_l2', 'key_l2'):
                assert schedule.count(f'name = "{name}.l2norm.sxm_vxm_streaming"') == 8
                assert schedule.count(f'name = "{name}.l2norm.vxm_sxm_direct"') == 8
            assert 'l2norm.restore_layout' not in schedule
            direct_windows=[]
            memory=[]
            for line in schedule.splitlines():
                if 'l2norm.vxm_sxm_direct' not in line and 'ftlpu.schedule.mem_transfer' not in line:
                    continue
                fields={k:int(v) for k,v in re.findall(r'(\w+) = (-?\d+) : i64', line)}
                if 'l2norm.vxm_sxm_direct' in line:
                    direct_windows.append((fields['start'],fields['end']))
                else:
                    memory.append((line,fields))
            for begin,end in direct_windows:
                assert not any('opcode = "write"' in line and f['packed_stream'] in (2,3,10,11)
                    and begin<=f['cycle']<end for line,f in memory), 'Normalized VXM result was spilled before SXM'
            summary['l2_output_streaming'] = dict(route='VXM -> forward SR -> wide SXM -> final MEM',
                paired_head_domains=len(direct_windows), normalized_result_spill=False,
                raw_input_retention=True)
            summary['head_partition'] = 'complete alternating 128-feature heads per hemisphere'
            summary['sxm_tile_initiation_interval'] = 32
            if not args.with_ks:
                assert 'l2norm.output_binding_mirror' not in schedule
            assert 'mem_materialize.conv1d_silu' not in schedule
            summary['output_contract'] = ('Q native partition; V native head groups in outputs 2/8; K mirrored for KS consumer' if args.with_ks
                else 'native partitioned distributed16; no Q/K mirror or V archive')
            plan = json.loads((out / 'graph_plan.json').read_text())
            if not args.with_state_decay:
                assert not plan.get('linker_mem_transports', []), 'Frontend must use native input/output homes'
            summary['linker_mem_transport_domains'] = len(plan.get('linker_mem_transports', []))
    if args.with_gates:
        assert any('log_lut' in r['detail'] for r in rows), 'Runtime must execute the hardware Log opcode'
        plan=json.loads((out/'graph_plan.json').read_text())
        gates=next(n for n in plan['nodes'] if n['name']=='delta_gates')
        for name in ('in_proj_a.page0','in_proj_b.page0'):
            projection=next(n for n in plan['nodes'] if n['name']==name)
            assert projection['direct_output_layouts'][0]['kind']=='token_partition_planar'
        if not args.with_state_decay and not args.with_gt:
            assert all(p['kind'].startswith('token_partition') for p in gates['direct_output_layouts'])
        schedule=(out/'decoder_layer.schedule.mlir').read_text()
        gate_code=schedule.split('name = "delta_gates.operand"',1)[1]
        if args.with_state_decay:
            gate_code=gate_code.split('name = "state_decay_factor.operand"',1)[0]
        if args.with_gt:
            gate_code=gate_code.split('name = "g_prefix_sum.operand"',1)[0]
        repeats=[int(re.search(r'repeat_count = (\d+)',line).group(1))
                 for line in gate_code.splitlines() if 'ftlpu.schedule.vxm ' in line]
        assert repeats and set(repeats)=={16}, 'Every gate chain must process distinct 16-token halves'
        summary['gate_token_partition']=dict(tokens_per_hemisphere=16, projection_direct=True,
            native_outputs=all(p['kind'].startswith('token_partition') for p in gates['direct_output_layouts']),
            nominal_cycles=gates['nominal_end']-gates['nominal_start'])
    if args.with_gt:
        scan_nodes = [n for n in plan['nodes'] if n['name'].startswith('g_prefix_sum')]
        assert len(scan_nodes)==1 and scan_nodes[0]['name']=='g_prefix_sum', 'Prefix scan must not be token-tiled'
        assert 'cumsum_rows.fp32_accumulator' in schedule
        assert not plan.get('linker_mem_transports', []), 'g must feed the scan at its native address'
        summary['chunkwise_prefix']['nominal_cycles'] = scan_nodes[0]['nominal_end']-scan_nodes[0]['nominal_start']
        scan_events = [r for r in rows if r['resource']=='VXM.C1'
                       and r['detail'].startswith('add chain_depth=2')][-2:]
        assert len(scan_events)==2 and all(r['pattern']=='single' for r in scan_events)
        assert scan_events[0]['end']==scan_events[1]['start'], 'Prefix first/reset issue must join the remaining tokens'
        scan_start, scan_end = int(scan_events[0]['start']), int(scan_events[-1]['end'])
        assert scan_end-scan_start==32, 'All tokens must accumulate in consecutive cycles'
        summary['chunkwise_prefix']['runtime_accumulate_range'] = [scan_start, scan_end]
    if args.with_gamma:
        gamma_nodes = [n for n in plan['nodes'] if n['name'].startswith('chunk_gamma')]
        assert len(gamma_nodes) == 1 and gamma_nodes[0]['name'] == 'chunk_gamma'
        assert not plan.get('linker_mem_transports', []), 'Gamma must read G_t at its native address'
        gamma_code = schedule.split('name = "chunk_gamma.operand"', 1)[1]
        if args.with_r:
            gamma_code = gamma_code.split('name = "chunk_causal_decay.operand"', 1)[0]
        gamma_repeats = [int(re.search(r'repeat_count = (\d+)', line).group(1))
                         for line in gamma_code.splitlines() if 'ftlpu.schedule.vxm ' in line]
        assert gamma_repeats and set(gamma_repeats) == {16}, 'Gamma must split distinct token halves'
        gamma_events = [r for r in rows if r['resource'] == 'VXM.C1' and r['detail'].startswith('exp_lut')]
        event = gamma_events[-1]
        assert event['pattern'] == 'single' and int(event['end'])-int(event['start']) == 16
        summary['chunkwise_gamma'].update(nominal_cycles=gamma_nodes[0]['nominal_end']-gamma_nodes[0]['nominal_start'],
            runtime_exp_range=[int(event['start']), int(event['end'])], linker_mem_transport_domains=0)
    if args.with_r:
        r_nodes = [n for n in plan['nodes'] if n['name'].startswith('chunk_causal_decay')]
        assert len(r_nodes)==1 and r_nodes[0]['name']=='chunk_causal_decay'
        assert not plan.get('linker_mem_transports', [])
        assert (('causal_exp_difference.scaled_strict_vxm_sxm' if args.with_l else 'causal_exp_difference.vxm_sxm_direct') if args.with_a else 'causal_exp_difference.accumulator_exp') in schedule
        exp_events = [r for r in rows if r['resource']=='VXM.C5' and r['detail'].startswith('exp_lut')
                      and int(r['start']) >= summary['chunkwise_gamma']['runtime_exp_range'][1]]
        assert len(exp_events)==(31 if args.with_l else 32) and all(r['pattern']=='single' for r in exp_events)
        assert [int(r['end'])-int(r['start']) for r in exp_events] == list(range(1,32 if args.with_l else 33))
        summary['chunkwise_r'].update(nominal_cycles=r_nodes[0]['nominal_end']-r_nodes[0]['nominal_start'],
            runtime_exp_range=[int(exp_events[0]['start']),int(exp_events[-1]['end'])],
            exp_vectors_per_hemisphere=32*(31 if args.with_l else 33)//2, linker_mem_transport_domains=0)
    if args.with_a:
        a_node=next(n for n in plan['nodes'] if n['name']=='chunk_weighted_gram')
        assert not plan.get('linker_mem_transports',[]), 'K and R must retain their native producer homes'
        assert 'weighted_gram.r_head_transpose' not in schedule and 'weighted_gram.mxm_to_vxm' in schedule
        assert ('causal_exp_difference.scaled_strict_vxm_sxm' if args.with_l else 'causal_exp_difference.vxm_sxm_direct') in schedule
        r_code=schedule.split('name = "chunk_causal_decay.operand"',1)[1].split('name = "chunk_weighted_gram.operand"',1)[0]
        r_writes=[line for line in r_code.splitlines() if 'ftlpu.schedule.mem_transfer' in line and 'opcode = "write"' in line]
        assert r_writes and all(int(re.search(r'packed_stream = (\d+)',line).group(1)) in (36,37) for line in r_writes), 'R must only write SXM results, never raw VXM output'
        assert 'l2norm.output_binding_mirror' not in schedule, 'K must not be mirrored for Gram'
        a_code=schedule.split('name = "chunk_weighted_gram.operand"',1)[1]
        for line in a_code.splitlines():
            if 'ftlpu.schedule.mem_transfer' in line and 'opcode = "write"' in line:
                stream=int(re.search(r'packed_stream = (\d+)',line).group(1))
                assert stream not in (32,33,48,49), 'MXM Gram result must not be spilled before VXM'
        loads=[line for line in a_code.splitlines() if 'ftlpu.schedule.mxm_issue ' in line and 'opcode = "iw"' in line]
        assert loads and all('weight_load_mode = "vector"' in line for line in loads), 'Gram IW must use 1vector columns'
        a_start=summary['chunkwise_r']['runtime_exp_range'][1]
        compute=[r for r in rows if r['resource'].startswith('MXM.') and r['resource'].endswith('.Compute') and int(r['start'])>a_start]
        assert compute, 'Weighted Gram must execute on MXM'
        key_node=next(n for n in plan['nodes'] if n['name']=='key_l2')
        assert key_node['direct_output_layouts'][0]['bank_tile_rows']==4, 'K producer must stripe D32 tiles across banks'
        utilization={}
        for unit in sorted({r['resource'] for r in compute}):
            intervals=[]
            for event in compute:
                if event['resource']!=unit: continue
                for outer in range(int(event['outer_count'])):
                    for inner in range(int(event['inner_count'])):
                        delta=outer*int(event['outer_interval'])+inner*int(event['inner_interval'])
                        intervals.append((int(event['start'])+delta,int(event['end'])+delta))
            intervals.sort()
            assert len(intervals)==32 and all(b-a==32 for a,b in intervals), intervals
            assert all(intervals[i][1]==intervals[i+1][0] for i in range(len(intervals)-1)), 'KKT compute has inter-tile or inter-head bubbles'
            utilization[unit]=dict(start=intervals[0][0],end=intervals[-1][1],active_cycles=1024,idle_cycles=0)
        summary['chunkwise_a']['mxm_compute_utilization']=utilization

        summary['chunkwise_a'].update(nominal_cycles=a_node['nominal_end']-a_node['nominal_start'],
            runtime_mxm_range=[min(int(r['start']) for r in compute),max(int(r['end']) for r in compute)],
            linker_mem_transport_domains=0,gram_spill_audit_passed=True,r_ingress='native BF16 [i,head,j]',r_retranspose=False)
    if args.with_state_decay:
        schedule = (out / 'decoder_layer.schedule.mlir').read_text()
        assert 'gated_delta_rule' not in schedule, 'Test must stop before State update'
        transports = plan.get('linker_mem_transports', [])
        if args.with_ks:
            summary['ks_fusion'] = dict(prediction_output=7, streaming=audit_streaming(out),
                route='VXM -> MEM_WRITE_TAP -> MXM IW', scaled_state_mem_rereads=0)
            assert all(t['phase'].startswith('state_decay_ks') for t in transports), 'Unexpected frontend transport'
        else:
            assert 'wide_scalar_broadcast' in schedule, 'State coefficients must use the shared wide SXM broadcast'
            assert 'scaled_matvec' not in schedule, 'Test must stop at State decay'
            assert not transports, 'State and the g[0] view must use direct input homes'
        summary['state_decay']['linker_mem_transports'] = transports
        summary['state_decay']['nominal_stages'] = [
            dict(name=n['name'], start=n['nominal_start'], end=n['nominal_end'])
            for n in plan['nodes'] if n['name'].startswith('state_decay')]
    log = (out / 'execution.log').read_text()
    assert f'sxm_architecture={args.sxm_architecture}' in log
    stats = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', log.splitlines()[-1])}
    assert stats.get('host_operations') == 0
    cycles = max(int(r['end']) + (int(r['inner_count'])-1)*int(r['inner_interval'])
                 + (int(r['outer_count'])-1)*int(r['outer_interval']) for r in rows)
    exporter = exe('runtime', 'ftlpu_icu_program_export')
    run([exporter, out / 'actual.runtime.ftlpu', out / 'runtime_icu_programs',
         '--pre-execution-dir', out / 'actual.pre_execution'], out / 'export.log')
    passed = all(e['nrmse'] < .03 and e['cosine'] > .999 for e in errors.values())
    precision_limited = args.with_gamma and not summary['chunkwise_gamma']['end_to_end_head_check_passed']
    precision_limited |= args.with_r and not summary['chunkwise_r']['end_to_end_head_check_passed']
    precision_limited |= args.with_a and not summary['chunkwise_a']['end_to_end_head_check_passed']
    hardware = json.loads(args.target_config.read_text())
    summary.update(status=('passed_with_precision_limits' if precision_limited else 'passed') if passed else 'failed', seq_len=32,
        export_qkv=args.export_qkv,
        input_shape=list(x.shape), output_shape=[32, 6144], input='raw BF16 fixture; no host RMSNorm',
        real_weights=True, fp32_ingress=args.fp32_ingress, checkpoint_provenance=json.loads((args.package / 'metadata.json').read_text()).get('checkpoint_provenance', {}),
        sxm_trace_rows=len(sxm), with_qk_norm=args.with_qk_norm, with_gates=args.with_gates,
        with_state_decay=args.with_state_decay, with_ks=args.with_ks, with_gt=args.with_gt, with_gamma=args.with_gamma, with_r=args.with_r, with_a=args.with_a,with_l=args.with_l,
        errors=errors, runtime=stats, runtime_trace_cycles=cycles,
        latency_ms=cycles/(hardware['external_memory']['lpu_clock_mhz']*1000))
    if args.with_l:
        summary['chunkwise_l']=summary.pop('chunkwise_a')
        summary['chunkwise_l_coefficient']=summary.pop('chunkwise_r')
        summary['chunkwise_l']['coefficient_precision']=('FP32' if args.fp32_ingress else 'BF16')+' beta/G ingress; FP32 multiply after Exp; BF16 coefficient storage; FP32 final L'
        summary['chunkwise_l']['a_mem_spill']=False
        summary['errors']={('l_'+k[2:] if k.startswith('a_') else 'coefficient_'+k[2:] if k.startswith('r_') else k):v for k,v in errors.items()}
    if args.fp32_ingress:
        summary['errors']={k.replace('gt_given_actual_bf16_g','gt_given_actual_fp32_g').replace('given_quantized_gt','given_hardware_ingress'):v for k,v in summary['errors'].items()}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert passed, errors


if __name__ == '__main__':
    main()
