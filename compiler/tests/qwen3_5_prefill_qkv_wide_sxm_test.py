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
from unit_lower_solve_runtime_test import source as triangular_source
from unit_lower_mxm_runtime_test import source as matrix_source, polynomial
from weighted_gram_runtime_test import scaled_weighted_gram_source, weighted_cross_gram_source, weighted_gram_source, reference as weighted_gram_reference, read_head_groups


def paging_epoch_offsets(rows, log):
    """Map logical issue cycles to physical time using consumer release events."""
    consumers = [row for row in rows if row['resource'] == 'Paging.Consumer']
    if consumers:
        return sorted((int(re.search(r'logical_cycle=(\d+)', row['detail'])[1]),
            int(row['start']) - int(re.search(r'logical_cycle=(\d+)', row['detail'])[1]))
            for row in consumers)
    # Legacy traces did not expose consumer release. Retain their original
    # completion-event fallback; current traces always prefer actual consumers.
    return sorted((int(consumer), int(ready) - 1 - int(consumer)) for consumer, ready in
        re.findall(r'FTLPU executable page complete:.*?consumer=(\d+).*?actual_ready=(\d+)', log))


def physical_node_range(node, offsets, fallback=0):
    """Map both ends, including a barrier inside the node's logical interval."""
    def physical(cycle):
        delay = next((offset for consumer, offset in reversed(offsets) if consumer <= cycle), fallback)
        return cycle + delay
    begin, end = node['nominal_start'], node['nominal_end']
    # The half-open end belongs to the previous epoch if the next consumer
    # begins exactly there; that next page's wait is not part of this node.
    return [physical(begin), physical(end - 1) + 1 if end > begin else physical(end)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--sxm-architecture', choices=('wide', 'local'), default='wide')
    parser.add_argument('--projection-solve-overlap', choices=('on', 'off'), default='off',
                        help='Allow C++ to overlap independent projection and FP32 forward solve')
    parser.add_argument('--projection-conv-overlap', choices=('on', 'off'), default='off',
                        help='Allow C++ to pipeline projection pages with independent depthwise convolution blocks')
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
    parser.add_argument('--with-o',action='store_true',help='Append O=H D on MXM; zero initial State; stop before outgoing State')
    parser.add_argument('--with-outgoing-state',action='store_true',help='Append zero-State K^T diag(R_last) D; stop before output norm/projection')
    parser.add_argument('--with-output-norm',action='store_true',help='Append Z projection and per-head gated RMSNorm')
    parser.add_argument('--with-output-projection',action='store_true',help='Complete seq32 chunkwise attention through output projection')
    parser.add_argument('--fuse-output-state',action=argparse.BooleanOptionalAction,default=True,
                        help='Share each transposed D tile between BF16 O and outgoing State products')
    parser.add_argument('--o-mxm-precision',choices=('bf16','fp32-components'),default='bf16',
                        help='HD only: direct truncated BF16 product or compensated FP32 components')
    parser.add_argument('--with-h',action='store_true',help='Append H=tril((Q K^T)*R), including diagonal; stop before H D/State update')
    parser.add_argument('--with-solve',action='store_true',help='Append FP32 beta*V and unit-lower solve with zero initial State')
    parser.add_argument('--solve-strategy',choices=('mxm','forward','blocked'),default='mxm')
    parser.add_argument('--with-l',action='store_true',help='Fuse strict lower beta weighting into R, then form L through MXM; stop before solve')
    parser.add_argument('--fp32-ingress', action=argparse.BooleanOptionalAction, default=None,
                        help='Read complete FP32 streams; enabled by default for --with-l')
    args = parser.parse_args()
    args.with_output_norm |= args.with_output_projection
    args.with_outgoing_state |= args.with_output_norm
    fused_output_state=args.with_outgoing_state and args.fuse_output_state and args.o_mxm_precision=='bf16'
    args.with_o |= args.with_outgoing_state
    args.with_h |= args.with_o
    args.with_solve |= args.with_h
    args.with_l |= args.with_solve
    if args.fp32_ingress is None:
        args.fp32_ingress = args.with_l
    fused_coefficient=args.with_l and not args.with_h
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
                   sxm_architecture=args.sxm_architecture, icu_compression='raw-macro',
                   projection_solve_overlap=args.projection_solve_overlap,
                   projection_conv_overlap=args.projection_conv_overlap)
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
            r_matrix = graph.run('chunk_causal_decay', causal_exp_source(32, config.num_value_heads,args.with_a,fused_coefficient,fused_coefficient),
                [(gt, 'f32')]+([(beta,'f32')] if fused_coefficient else []), [(r_shape,r_dtype)])[0]
            finals += [r_matrix]
        if args.with_a:
            gram_source=scaled_weighted_gram_source(config.num_value_heads,config.key_head_dim) if args.with_h else weighted_gram_source(config.num_value_heads,config.key_head_dim,True)
            gram_inputs=[(key,'bf16'),(r_matrix,'bf16')]
            if args.with_h:
                gram_inputs += [(beta,'f32'),(np.tri(32,k=-1,dtype='f4'),'bf16')]
            a_matrix = graph.run('chunk_weighted_gram',gram_source,gram_inputs,[((config.num_value_heads*32,32),'f32')])[0]
            finals += [a_matrix]
        if args.with_solve:
            h,d=config.num_value_heads,config.value_head_dim
            rhs_matrix=graph.run('chunk_rhs_beta_v',triangular_source(h,d,'head_scale'),
                [(conv[:,2*width:],'bf16'),(beta,'f32')],[((h*32,d),'f32')])[0]
            if args.solve_strategy=='mxm':
                identity=np.broadcast_to(np.eye(32,dtype='f4'),(h,32,32)).copy().reshape(h*32,32)
                inverse=graph.run('chunk_unit_lower_inverse',matrix_source(h,32,'unit_lower_inverse'),
                    [(a_matrix,'f32'),(identity,'f32')],[((h*32,32),'f32')])[0]
                solved=graph.run('chunk_solve_matmul',matrix_source(h,d,'head_matmul'),
                    [(inverse,'f32'),(rhs_matrix,'f32')],[((h*32,d),'f32')])[0]
            else:
                solved=graph.run('chunk_unit_lower_solve',triangular_source(h,d,'unit_lower_solve_blocked' if args.solve_strategy=='blocked' else 'unit_lower_solve'),
                    [(a_matrix,'f32'),(rhs_matrix,'f32')],[((h*32,d),'f32')])[0]
            finals += [rhs_matrix,solved]
            if args.solve_strategy=='mxm':finals += [inverse]
        if args.with_h:
            h=config.num_value_heads
            output_h=graph.run('chunk_output_cross_gram',weighted_cross_gram_source(h,config.key_head_dim),
                [(query,'bf16'),(key,'bf16'),(r_matrix,'bf16')],[((h*32,32),'f32')])[0]
            finals += [output_h]
        if fused_output_state:
            from head_weighted_product_runtime_test import source as weighted_product_source
            output_o,state_out=graph.run('chunk_output_state',weighted_product_source(h,config.key_head_dim,config.value_head_dim,True),
                [(key,'bf16'),(r_matrix,'bf16'),(solved,'f32'),(output_h,'f32')],
                [((h*32,config.value_head_dim),'f32'),((h*config.key_head_dim,config.value_head_dim),'f32')])
            finals += [output_o,state_out]
        if args.with_o and not fused_output_state:
            o_source=matrix_source(h,config.value_head_dim,'head_matmul').replace('kind = "head_matmul",',
                f'kind = "head_matmul", mxm_precision = "{args.o_mxm_precision}",')
            output_o=graph.run('chunk_output_matmul',o_source,
                [(output_h,'f32'),(solved,'f32')],[((h*32,config.value_head_dim),'f32')])[0]
            finals += [output_o]
        if args.with_outgoing_state and not fused_output_state:
            from head_weighted_product_runtime_test import source as weighted_product_source
            state_out=graph.run('chunk_outgoing_state',weighted_product_source(h,config.key_head_dim,config.value_head_dim),
                [(key,'bf16'),(r_matrix,'bf16'),(solved,'f32')],[((h*config.key_head_dim,config.value_head_dim),'f32')])[0]
            finals += [state_out]
        if args.with_output_norm:
            z=graph.project('in_proj_z',x,packed['in_proj_z'],scales['in_proj_z'])
            # Logical view only: O exports even-head groups then odd-head
            # groups. Restore token/head axes without a host/device copy.
            core=output_o.reshape(2,h//2,32,config.value_head_dim).transpose(2,1,0,3).reshape(32*h,config.value_head_dim)
            core_bf16=graph.truncate_bf16('chunk_output_bf16_view',core)
            normalized=graph.norm('delta_output_norm',core_bf16,weights.delta_norm,config.rms_norm_eps)
            def output_gate(g):
                silu=g.add(1,g.sigmoid(1))
                return g.add(0,silu,dtype='bf16')
            gated=graph.vector_tiles('delta_output_gate',[(normalized,'bf16'),(z.reshape(normalized.shape),'bf16')],output_gate)
            finals += [z,normalized,gated]
            if args.with_output_projection:
                attention=graph.project('out_proj',gated.reshape(32,config.value_width),packed['out_proj'],scales['out_proj'])
                finals += [attention]
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
            + f'ftlpu.projection_solve_overlap = {str(args.projection_solve_overlap == "on").lower()}, '
            + f'ftlpu.projection_conv_overlap = {str(args.projection_conv_overlap == "on").lower()}, '
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
            assert gram['nominal_end']-gram['nominal_start']<(2600 if args.with_h else 1500), 'Continuous KKT schedule was not emitted; check compiler build'
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
        summary['gate_precision'] = ('FP32 full intermediate ingress' if args.fp32_ingress else 'BF16 ingress at spill boundaries') + ', FP32 chain arithmetic/output; FP16 LUT ingress'
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
        r_checks = check_causal_exp(actual_r, gt_actual,args.with_a,gate_actual[0] if fused_coefficient else None,fused_coefficient,args.fp32_ingress)
        causal = np.tri(32,k=-1 if fused_coefficient else 0, dtype=bool)[:, :, None]
        reference_r = lambda g: np.exp(np.where(causal, g[:, None, :]-g[None, :, :], -np.inf))
        expected_r = reference_r(np.cumsum(decay_reference, axis=0, dtype=np.float32))
        local_r=causal_exp_reference(gt_actual,args.fp32_ingress)
        actual_gt_r=reference_r(gt_actual)
        if fused_coefficient:
            beta_ingress=gate_actual[0] if args.fp32_ingress else (gate_actual[0].view('u4') & np.uint32(0xffff0000)).view('f4')
            local_r=local_r*beta_ingress[:,None,:]*causal
            actual_gt_r=actual_gt_r*gate_actual[0][:,None,:]
            expected_r=expected_r*beta_reference[:,None,:]
        errors['r_given_quantized_gt'] = compare(actual_r, local_r)
        errors['r_given_actual_gt'] = compare(actual_r, actual_gt_r)
        errors['r_end_to_end'] = compare(actual_r, expected_r)
        per_head = [compare(actual_r[:,:,h], expected_r[:,:,h]) for h in range(config.num_value_heads)]
        limited_heads = [h for h,e in enumerate(per_head) if e['nrmse'] >= .03 or e['cosine'] <= .999]
        summary['scope'] += ('; strict lower coefficient C=beta_i*R_ij' if fused_coefficient else '; causal R') + ('' if args.with_a else '; no KKt, triangular solve or State recurrence')
        summary['chunkwise_r'] = dict(shape=[32,32,config.num_value_heads], output_index=7,
            stored_shape=[32*config.num_value_heads,32] if args.with_a else [1024,config.num_value_heads],
            storage_layout='[i*H+head, j]; VXM directly to SXM, one j per lane' if args.with_a else '[i*T+j, head]; one head per lane, pair in successive MEM rows',
            formula='beta[i,h]*exp(G[i,h]-G[j,h]) for j<i; diagonal/upper=0' if fused_coefficient else 'exp(G[i,h]-G[j,h]) for j<i; diagonal=1; upper=0',
            row_scale=fused_coefficient,strict_lower=fused_coefficient,
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
        gram_actual_r=actual_r
        gram_expected_r=expected_r
        if args.with_h:
            strict=np.tri(32,k=-1,dtype=bool)[:,:,None]
            gram_actual_r=actual_r*gate_actual[0][:,None,:]*strict
            gram_expected_r=expected_r*beta_reference[:,None,:]*strict
        given_actual = weighted_gram_reference(actual_keys,gram_actual_r)
        model_keys = norm(expected_conv[:,config.key_width:2*config.key_width],1)
        expected_a = weighted_gram_reference(model_keys,gram_expected_r)
        errors['a_given_actual_k_r'] = compare(actual_a,given_actual)
        errors['a_end_to_end'] = compare(actual_a,expected_a)
        per_head = [compare(actual_a[h],expected_a[h]) for h in range(config.num_value_heads)]
        local_per_head = [compare(actual_a[h],given_actual[h]) for h in range(config.num_value_heads)]
        assert all(e['nrmse'] < .03 and e['cosine'] > .999 for e in local_per_head), local_per_head
        assert np.isfinite(actual_a).all()
        assert np.all(actual_a[:,np.triu_indices(32,0 if args.with_l else 1)[0],np.triu_indices(32,0 if args.with_l else 1)[1]]==0)
        limited_heads = [h for h,e in enumerate(per_head) if e['nrmse']>=.03 or e['cosine']<=.999]
        summary['scope'] += ('; L=strict_lower(beta_i*R_ij*(KK^T)_ij)' + ('' if args.with_solve else '; no triangular solve or State recurrence') if args.with_l else '; A=(KK^T)*R; no beta weighting, triangular solve or State recurrence')
        summary['chunkwise_a'] = dict(shape=[config.num_value_heads,32,32],output_index=8,
            stored_head_order=list(range(0,config.num_value_heads,2))+list(range(1,config.num_value_heads,2)),
            formula='L[h,i,j] = beta[i,h]*sum_d K[i,h,d]*K[j,h,d]*R[i,j,h], j<i; otherwise 0' if args.with_l else 'A[h,i,j] = sum_d K[i,h,d]*K[j,h,d] * R[i,j,h]',
            precision='BF16 K, FP32 MXM accumulation, BF16 Gram stream/R ingress, FP32 VXM multiply/output',
            gram_spill=False,weight_load_mode='1vector',row_scale_in_coefficient_preparation=args.with_h,
            hemisphere_policy='distinct alternating heads; native partitioned output',
            per_head_given_actual_inputs_errors=local_per_head,
            per_head_end_to_end_errors=per_head,end_to_end_head_check_passed=not limited_heads,
            precision_limited_heads=limited_heads)
        if fused_coefficient:
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
    plan=json.loads((out/'graph_plan.json').read_text())
    core_transports=[t for t in plan.get('linker_mem_transports',[])
        if not (args.with_output_norm and t['phase'].startswith(('in_proj_z','delta_output_','out_proj','validation_export.')))]
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
                assert not core_transports, 'Frontend must use native input/output homes'
            summary['linker_mem_transport_domains'] = len(plan.get('linker_mem_transports', []))
    if args.with_gates:
        assert any('log_lut' in r['detail'] for r in rows), 'Runtime must execute the hardware Log opcode'
        plan=json.loads((out/'graph_plan.json').read_text())
        gates=next(n for n in plan['nodes'] if n['name']=='delta_gates')
        paired = [n for n in plan['nodes'] if n['name'] in ('in_proj_a.paired','in_proj_b.paired')]
        if paired:
            assert len(paired)==1
            assert paired[0]['direct_output_layouts'][0]['kind']=='fp16_pair_planar'
        else:
            for name in ('in_proj_a.page0','in_proj_b.page0'):
                projection=next(n for n in plan['nodes'] if n['name']==name)
                assert projection['direct_output_layouts'][0]['kind']=='token_partition_planar'
        summary['gate_paired_projections'] = bool(paired)
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
        assert not core_transports, 'g must feed the scan at its native address'
        summary['chunkwise_prefix']['nominal_cycles'] = scan_nodes[0]['nominal_end']-scan_nodes[0]['nominal_start']
        scan_candidates = [r for r in rows if r['resource']=='VXM.C1'
                       and r['detail'].startswith('add chain_depth=2') and r['pattern']=='single']
        scan_events = next(([a,b] for a,b in zip(scan_candidates,scan_candidates[1:])
            if int(a['end'])-int(a['start'])==1 and a['end']==b['start'] and int(b['end'])-int(b['start'])==31),[])
        assert len(scan_events)==2 and all(r['pattern']=='single' for r in scan_events)
        assert scan_events[0]['end']==scan_events[1]['start'], 'Prefix first/reset issue must join the remaining tokens'
        scan_start, scan_end = int(scan_events[0]['start']), int(scan_events[-1]['end'])
        assert scan_end-scan_start==32, 'All tokens must accumulate in consecutive cycles'
        summary['chunkwise_prefix']['runtime_accumulate_range'] = [scan_start, scan_end]
    if args.with_gamma:
        gamma_nodes = [n for n in plan['nodes'] if n['name'].startswith('chunk_gamma')]
        assert len(gamma_nodes) == 1 and gamma_nodes[0]['name'] == 'chunk_gamma'
        assert not core_transports, 'Gamma must read G_t at its native address'
        gamma_code = schedule.split('name = "chunk_gamma.operand"', 1)[1]
        if args.with_r:
            gamma_code = gamma_code.split('name = "chunk_causal_decay.operand"', 1)[0]
        gamma_repeats = [int(re.search(r'repeat_count = (\d+)', line).group(1))
                         for line in gamma_code.splitlines() if 'ftlpu.schedule.vxm ' in line]
        assert gamma_repeats and set(gamma_repeats) == {16}, 'Gamma must split distinct token halves'
        gamma_events = [r for r in rows if r['resource'] == 'VXM.C1' and r['detail'].startswith('exp_lut')]
        event = next(r for r in gamma_events if int(r['start'])>=scan_end)
        assert event['pattern'] == 'single' and int(event['end'])-int(event['start']) == 16
        summary['chunkwise_gamma'].update(nominal_cycles=gamma_nodes[0]['nominal_end']-gamma_nodes[0]['nominal_start'],
            runtime_exp_range=[int(event['start']), int(event['end'])], linker_mem_transport_domains=0)
    if args.with_r:
        r_nodes = [n for n in plan['nodes'] if n['name'].startswith('chunk_causal_decay')]
        assert len(r_nodes)==1 and r_nodes[0]['name']=='chunk_causal_decay'
        assert not core_transports
        assert (('causal_exp_difference.scaled_strict_vxm_sxm' if fused_coefficient else 'causal_exp_difference.vxm_sxm_direct') if args.with_a else 'causal_exp_difference.accumulator_exp') in schedule
        exp_events = [r for r in rows if r['resource']=='VXM.C5' and r['detail'].startswith('exp_lut')
                      and int(r['start']) >= summary['chunkwise_gamma']['runtime_exp_range'][1]][:31 if fused_coefficient else 32]
        assert len(exp_events)==(31 if fused_coefficient else 32) and all(r['pattern']=='single' for r in exp_events)
        assert [int(r['end'])-int(r['start']) for r in exp_events] == list(range(1,32 if fused_coefficient else 33))
        summary['chunkwise_r'].update(nominal_cycles=r_nodes[0]['nominal_end']-r_nodes[0]['nominal_start'],
            runtime_exp_range=[int(exp_events[0]['start']),int(exp_events[-1]['end'])],
            exp_vectors_per_hemisphere=32*(31 if fused_coefficient else 33)//2, linker_mem_transport_domains=0)
    if args.with_a:
        a_node=next(n for n in plan['nodes'] if n['name']=='chunk_weighted_gram')
        assert not core_transports, 'K and R must retain their native producer homes'
        assert 'weighted_gram.r_head_transpose' not in schedule and 'weighted_gram.mxm_to_vxm' in schedule
        assert ('causal_exp_difference.scaled_strict_vxm_sxm' if fused_coefficient else 'causal_exp_difference.vxm_sxm_direct') in schedule
        r_code=schedule.split('name = "chunk_causal_decay.operand"',1)[1].split('name = "chunk_weighted_gram.operand"',1)[0]
        r_writes=[line for line in r_code.splitlines() if 'ftlpu.schedule.mem_transfer' in line and 'opcode = "write"' in line]
        assert r_writes and all(int(re.search(r'packed_stream = (\d+)',line).group(1)) in (36,37) for line in r_writes), 'R must only write SXM results, never raw VXM output'
        assert 'l2norm.output_binding_mirror' not in schedule, 'K must not be mirrored for Gram'
        a_code=schedule.split('name = "chunk_weighted_gram.operand"',1)[1].split('name = "chunk_rhs_beta_v.operand"',1)[0]
        for line in a_code.splitlines():
            if 'ftlpu.schedule.mem_transfer' in line and 'opcode = "write"' in line:
                stream=int(re.search(r'packed_stream = (\d+)',line).group(1))
                assert stream not in (32,33,48,49), 'MXM Gram result must not be spilled before VXM'
        loads=[line for line in a_code.splitlines() if 'ftlpu.schedule.mxm_issue ' in line and 'opcode = "iw"' in line]
        assert loads and all('weight_load_mode = "vector"' in line for line in loads), 'Gram IW must use 1vector columns'
        a_start=summary['chunkwise_r']['runtime_exp_range'][1]
        compute=[r for r in rows if r['resource'].startswith('MXM.') and r['resource'].endswith('.Compute') and int(r['start'])>a_start]
        if args.with_solve:
            # Bound the Gram audit to its first 32 physical compute tiles/side.
            compute=[r for unit in sorted({r['resource'] for r in compute}) for r in sorted([e for e in compute if e['resource']==unit],key=lambda e:int(e['start']))[:32]]
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
    epoch_offsets = paging_epoch_offsets(rows, log)
    assert f'sxm_architecture={args.sxm_architecture}' in log
    stats = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', log.splitlines()[-1])}
    assert stats.get('host_operations') == 0
    cycles = max(int(r['end']) + (int(r['inner_count'])-1)*int(r['inner_interval'])
                 + (int(r['outer_count'])-1)*int(r['outer_interval']) for r in rows)
    exporter = exe('runtime', 'ftlpu_icu_program_export')
    run([exporter, out / 'actual.runtime.ftlpu', out / 'runtime_icu_programs',
         '--pre-execution-dir', out / 'actual.pre_execution'], out / 'export.log')
    summary['paging_image']=json.loads((out/'runtime_icu_programs'/'program.json').read_text())
    passed = all(e['nrmse'] < .03 and e['cosine'] > .999 for e in errors.values())
    if args.with_solve:
        h,d=config.num_value_heads,config.value_head_dim
        # Both native exports retain grouped heads: even heads, then odd heads.
        unpack=lambda path:np.fromfile(path,'<f4').reshape(2,h//2,32,d).transpose(1,0,2,3).reshape(h,32,d)
        actual_b=unpack(out/'actual.9.bin');actual_d=unpack(out/'actual.10.bin')
        solve_v=actual_v.reshape(32,h,d).transpose(1,0,2)
        expected_b=gate_actual[0].T[:,:,None]*solve_v
        np.testing.assert_allclose(actual_b,expected_b,rtol=1e-6,atol=1e-6)
        system=np.eye(32,dtype='f8')[None]+actual_a.astype('f8')
        expected_d=np.linalg.solve(system,expected_b.astype('f8'))
        per_head=[compare(actual_d[j],expected_d[j]) for j in range(h)]
        residual=system@actual_d-actual_b
        relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(actual_b))
        local_limit=.01 if args.solve_strategy=='blocked' else .003 if args.solve_strategy=='mxm' else 2e-5
        assert all(e['nrmse']<local_limit for e in per_head),per_head
        assert relative_residual<local_limit,relative_residual
        model_b=beta_reference.T[:,:,None]*expected_conv[:,2*width:].reshape(32,h,d).transpose(1,0,2)
        model_d=np.linalg.solve(np.eye(32,dtype='f8')[None]+expected_a.astype('f8'),model_b.astype('f8'))
        model_errors=[compare(actual_d[j],model_d[j]) for j in range(h)]
        assert all(e['nrmse']<.03 for e in model_errors),model_errors
        summary['chunkwise_solve']=dict(status='passed',shape=[h,32,d],initial_state='zero',
            formula='B=beta*V; (I+L)D=B',precision='FP32 L/B/D storage and VXM accumulation; BF16 V input',
            per_head_given_actual_inputs=per_head,relative_residual=relative_residual,per_head_end_to_end=model_errors,end_to_end_nrmse_limit=.03,
            coefficient_workspace='one 32-vector row per hemisphere; two wide-SXM byte-plane passes',
            feedback='completed rows only, passive SR return to the fixed local VXM inputs',
            partial_sum_mem_spill=False,nonzero_state_supported=False,paging='compiler-planned weight slots; checked common-barrier C2C overlap')
        if args.solve_strategy=='mxm':
            actual_t=np.fromfile(out/'actual.11.bin','<f4').reshape(2,h//2,32,32).transpose(1,0,2,3).reshape(h,32,32)
            ref_t=polynomial(actual_a)
            inverse_errors=[compare(actual_t[j],ref_t[j]) for j in range(h)]
            assert all(e['nrmse']<.001 for e in inverse_errors),inverse_errors
            np.testing.assert_allclose(ref_t@expected_b,expected_d,rtol=1e-10,atol=1e-10)
            summary['chunkwise_solve'].update(strategy='nilpotent_mxm',
                formula='T=(I-L)(I+L^2)(I+L^4)(I+L^8)(I+L^16); D=T@(beta*V)',
                precision='FP32 L/P/T/B/D storage, VXM add and MXM accumulation; three BF16 components, six MXM partial products per matrix product',
                coefficient_workspace='FP32 P/T ping-pong plus one BF16 transposed weight tile per hemisphere',
                feedback='matrix ownership alternates hemispheres through passive SR or VXM; no completed-row feedback',
                inverse_per_head=inverse_errors,local_nrmse_and_residual_limit=local_limit,
                square_matrix_products_per_head=8,rectangular_products_per_head=1)
        elif args.solve_strategy=='blocked':
            summary['chunkwise_solve'].update(strategy='blocked_forward_16',block_size=16,
                precision='FP32 L/B/D and diagonal-block VXM solve; single truncated BF16 MXM product with FP32 accumulation for L10*D0',
                inverse_formed=False,iterative_refinement=False,local_nrmse_and_residual_limit=local_limit,
                rhs_update_fused_into_forward_row=True,coefficient_workspace='two 32-vector FP32 banks; one BF16 transposed weight tile; 16xD FP32 cross-block product',
                bf16_partial_products_per_tile=1)
        else:summary['chunkwise_solve']['strategy']='fp32_forward'
        summary['scope'] += '; FP32 B=beta*V and (I+L)D=B; zero initial State'
        invocation=next(r for r in rows if r['resource']=='Session.Invocation')
        scan_code=schedule.split('name = "g_prefix_sum.operand"',1)[1].split('name = "chunk_gamma.operand"',1)[0]
        scan_line=next(line for line in scan_code.splitlines() if 'accumulator_reset = true' in line and 'queue = 1 : i64' in line)
        scan_offset=scan_start-int(re.search(r'cycle = (\d+)',scan_line)[1])
        physical_range=lambda node:physical_node_range(node,epoch_offsets,scan_offset)
        summary['chunkwise_solve']['nominal_and_runtime_ranges']={}
        for name in (('chunk_rhs_beta_v','chunk_unit_lower_inverse','chunk_solve_matmul') if args.solve_strategy=='mxm' else ('chunk_rhs_beta_v','chunk_unit_lower_solve')):
            node=next(n for n in plan['nodes'] if n['name']==name)
            summary['chunkwise_solve']['nominal_and_runtime_ranges'][name]=dict(
                nominal=[node['nominal_start'],node['nominal_end']],
                runtime=physical_range(node),
                cycles=node['nominal_end']-node['nominal_start'])
        from head_scale_streaming_runtime_test import audit_runtime as audit_head_scale
        beta_range=summary['chunkwise_solve']['nominal_and_runtime_ranges']['chunk_rhs_beta_v']['runtime']
        summary['chunkwise_solve']['beta_v_streaming']=dict(
            beta_broadcast='split FP32 byte planes across SXM hemispheres; local tap and passive SR return',
            coefficient_storage='slice24-27, token bank ping-pong',
            value_layout_copy=False,
            paired_vxm_queues=audit_head_scale(out/'runtime.pipeline.csv',32*(h//2)*(d//32),
                                               h==16 and d==128,beta_range))
        if args.solve_strategy=='forward':
            from unit_lower_solve_runtime_test import audit_forward_solve
            solve_range=summary['chunkwise_solve']['nominal_and_runtime_ranges']['chunk_unit_lower_solve']['runtime']
            summary['chunkwise_solve']['scheduling']=audit_forward_solve(out,h,d,solve_range,'chunk_unit_lower_solve')
        if args.solve_strategy=='blocked':
            from unit_lower_blocked_runtime_test import audit_blocked_solve
            solve_range=summary['chunkwise_solve']['nominal_and_runtime_ranges']['chunk_unit_lower_solve']['runtime']
            summary['chunkwise_solve']['scheduling']=audit_blocked_solve(out,h,d,solve_range)
        if args.solve_strategy=='mxm':
            solve_start=summary['chunkwise_solve']['nominal_and_runtime_ranges']['chunk_unit_lower_inverse']['runtime'][0]
            solve_end=physical_range(next(n for n in plan['nodes'] if n['name']=='chunk_output_cross_gram'))[0] if args.with_h else None
            events=[r for r in rows if r['resource'].startswith('MXM.') and r['resource'].endswith('.Compute') and int(r['start'])>=solve_start and (solve_end is None or int(r['start'])<solve_end)]
            active={}
            for unit in sorted({r['resource'] for r in events}):
                cycles=sum((int(r['end'])-int(r['start']))*int(r['inner_count'])*int(r['outer_count']) for r in events if r['resource']==unit)
                assert cycles==(h//2)*(8+d//32)*6*32,(unit,cycles)
                active[unit]=cycles
            assert len(active)==2,active
            summary['chunkwise_solve'].update(mxm_active_cycles=active,bf16_partial_products_per_32x32_tile=6,
                weight_component_loads_per_32x32_tile=3,partial_products_stay_in_mxm_acc=True,
                local_nrmse_and_residual_limit=local_limit,inverse_nrmse_limit=.001)
        summary['chunkwise_solve']['device_session_end_cycle']=int(invocation['end'])
        summary['chunkwise_solve']['diagnostic_host_readback_cycles']=sum(int(r['end'])-int(r['start']) for r in rows if r['resource']=='C2C.HostOutput')

    if args.with_h:
        h=config.num_value_heads
        h_index=12 if args.solve_strategy=='mxm' else 11
        h_r=actual_r
        h_actual=read_head_groups(out/f'actual.{h_index}.bin',h)
        check_causal_exp(h_r,gt_actual,True,None,False,args.fp32_ingress)
        q_actual=read(0,width).reshape(32,h,dim).transpose(1,0,2)
        k_actual=read(1,width).reshape(32,h,dim).transpose(1,0,2)
        local=(q_actual@k_actual.transpose(0,2,1))*h_r.transpose(2,0,1)
        model_q=norm(expected_conv[:,:width],1/np.sqrt(dim)).reshape(32,h,dim).transpose(1,0,2)
        model_k=model_keys.reshape(32,h,dim).transpose(1,0,2)
        model_g=np.cumsum(decay_reference,axis=0,dtype=np.float32)
        mask=np.tri(32,dtype=bool)[:,:,None]
        model_r=np.exp(np.where(mask,model_g[:,None,:]-model_g[None,:,:],-np.inf))
        model=(model_q@model_k.transpose(0,2,1))*model_r.transpose(2,0,1)
        local_errors=[compare(h_actual[j],local[j]) for j in range(h)]
        model_errors=[compare(h_actual[j],model[j]) for j in range(h)]
        assert all(e['nrmse']<.01 and e['cosine']>.9999 for e in local_errors),local_errors
        assert all(e['nrmse']<.03 and e['cosine']>.999 for e in model_errors),model_errors
        assert np.isfinite(h_actual).all()
        assert np.all(h_actual[:,np.triu_indices(32,1)[0],np.triu_indices(32,1)[1]]==0)
        assert np.all(np.diagonal(h_r.transpose(2,0,1),axis1=1,axis2=2)==1)
        # Q and K differ: diagonal is dot(q_i,k_i), not the unit diagonal of KKt.
        np.testing.assert_allclose(np.diagonal(h_actual,axis1=1,axis2=2),
            np.diagonal(local,axis1=1,axis2=2),rtol=.009,atol=2e-5)
        node=next(n for n in plan['nodes'] if n['name']=='chunk_output_cross_gram')
        runtime_range=physical_range(node)
        util={}
        for unit in ('MXM.E0.Compute','MXM.W0.Compute'):
            spans=[]
            for event in rows:
                if event['resource']!=unit or not runtime_range[0]<=int(event['start'])<runtime_range[1]:continue
                for outer in range(int(event['outer_count'])):
                    for inner in range(int(event['inner_count'])):
                        delta=outer*int(event['outer_interval'])+inner*int(event['inner_interval'])
                        spans.append((int(event['start'])+delta,int(event['end'])+delta))
            spans.sort()
            assert len(spans)==32 and all(b-a==32 for a,b in spans),spans
            assert all(spans[i][1]==spans[i+1][0] for i in range(len(spans)-1)),spans
            util[unit]=dict(start=spans[0][0],end=spans[-1][1],active_cycles=1024,idle_cycles=0)
        assert not core_transports
        assert not any(n['name']=='chunk_output_decay' for n in plan['nodes'])
        assert len([n for n in plan['nodes'] if n['name']=='chunk_causal_decay'])==1
        assert 'cross_gram.mxm_to_vxm' in schedule
        summary['chunkwise_h']=dict(status='passed',shape=[h,32,32],output_index=h_index,
            formula='H[h,i,j]=dot(Q[i,h],K[j,h])*exp(G[i,h]-G[j,h]) for j<=i; otherwise zero; Q already scaled',
            diagonal_included=True,beta_weighting=False,precision='BF16 Q/K/R and MXM stream; FP32 MXM accumulation and VXM output',
            per_head_given_actual_inputs=local_errors,per_head_end_to_end=model_errors,
            nominal_cycles=node['nominal_end']-node['nominal_start'],runtime_range=runtime_range,
            mxm_compute_utilization=util,gram_mem_spill=False,linker_mem_transports=0,
            r_reused_from_output=7,output_decay_cycles=0,r_generation_count=1)
        summary['scope']+='; H=tril((Q K^T)*R)'
        if not args.with_o:summary['scope']+='; stop before H D and outgoing State'

    if args.with_o:
        d=config.value_head_dim
        actual_o=unpack(out/f'actual.{h_index+1}.bin')
        local_o=h_actual.astype('f8')@actual_d.astype('f8')
        model_o=model.astype('f8')@model_d
        local_errors=[compare(actual_o[j],local_o[j]) for j in range(h)]
        model_errors=[compare(actual_o[j],model_o[j]) for j in range(h)]
        direct_bf16=args.o_mxm_precision=='bf16'
        trunc_bf16=lambda x:(x.astype('f4').view('u4') & np.uint32(0xffff0000)).view('f4').astype('f8')
        hardware_ref=trunc_bf16(h_actual)@trunc_bf16(actual_d) if direct_bf16 else local_o
        hardware_errors=[compare(actual_o[j],hardware_ref[j]) for j in range(h)]
        assert all(e['nrmse']<1e-5 for e in hardware_errors),hardware_errors
        assert np.isfinite(actual_o).all()
        assert all(e['nrmse']<(.01 if direct_bf16 else .003) and e['cosine']>.9999 for e in local_errors),local_errors
        assert all(e['nrmse']<.03 and e['cosine']>.999 for e in model_errors),model_errors
        # Independent token recurrence checks the chunk algebra as well as the
        # new product. These references never participate in device execution.
        q=model_q;k=model_k
        values=expected_conv[:,2*width:].reshape(32,h,d).transpose(1,0,2)
        state_ref=np.zeros((h,dim,d),dtype='f8')
        recurrent=np.empty((h,32,d),dtype='f8')
        for token in range(32):
            state_ref*=np.exp(decay_reference[token]).astype('f8')[:,None,None]
            delta=beta_reference[token,:,None]*(values[:,token]-np.einsum('hk,hkv->hv',k[:,token],state_ref))
            state_ref+=np.einsum('hk,hv->hkv',k[:,token],delta)
            recurrent[:,token]=np.einsum('hk,hkv->hv',q[:,token],state_ref)
        algebra_error=float(np.linalg.norm(model_o-recurrent)/np.linalg.norm(recurrent))
        assert algebra_error<2e-5,algebra_error
        node=next(n for n in plan['nodes'] if n['name']==('chunk_output_state' if fused_output_state else 'chunk_output_matmul'))
        interval=physical_range(node)
        events=[r for r in rows if r['resource'].startswith('MXM.') and r['resource'].endswith('.Compute')
                and interval[0]<=int(r['start'])<interval[1]]
        active={unit:sum((int(r['end'])-int(r['start']))*int(r['inner_count'])*int(r['outer_count'])
            for r in events if r['resource']==unit) for unit in sorted({r['resource'] for r in events})}
        products=1 if direct_bf16 else 6
        assert len(active)==2 and all(c==(h//2)*(d//32)*(products*32+(dim if fused_output_state else 0)) for c in active.values()),active
        vxm_events=[r for r in rows if r['resource'].startswith('VXM.') and interval[0]<=int(r['start'])<interval[1]]
        if direct_bf16 and not fused_output_state:assert not vxm_events,vxm_events[:3]
        streaming_audit=None
        if fused_output_state:
            from head_weighted_product_runtime_test import audit as audit_shared_products
            streaming_audit=audit_shared_products(out/'runtime.pipeline.csv',h,dim,d,interval,with_output=True)
            summary['chunkwise_output_state_fusion']=dict(runtime_range=interval,
                nominal_cycles=node['nominal_end']-node['nominal_start'],**streaming_audit)
        elif direct_bf16:
            from head_matmul_streaming_runtime_test import audit as audit_head_matmul
            streaming_audit=audit_head_matmul(out/'runtime.pipeline.csv',h,d,interval)
        assert not core_transports
        h_home=next(n for n in plan['nodes'] if n['name']=='chunk_output_cross_gram')['direct_output_layouts'][0]
        d_home=next(n for n in plan['nodes'] if n['name']==('chunk_solve_matmul' if args.solve_strategy=='mxm' else 'chunk_unit_lower_solve'))['direct_output_layouts'][0]
        assert h_home['kind']==d_home['kind']=='token_partition_east_first_planar'
        assert node['direct_output_layouts'][0]['kind']=='token_partition_planar'
        summary['chunkwise_o']=dict(status='passed',shape=[h,32,d],output_index=h_index+1,formula='O=H@D',initial_state='zero',
            precision=('FP32 H/D storage, upper byte planes read as truncated BF16; one MXM product with FP32 accumulation/output' if direct_bf16 else 'FP32 H/D/O; three BF16 components and six MXM partial products with FP32 accumulation'),
            mxm_precision=args.o_mxm_precision,products_per_tile=products,per_head_hardware_reference=hardware_errors,
            per_head_given_actual_inputs=local_errors,per_head_end_to_end=model_errors,
            chunk_vs_recurrent_reference_nrmse=algebra_error,
            runtime_range=interval,nominal_cycles=node['nominal_end']-node['nominal_start'],mxm_active_cycles=active,
            runtime_scope='fused O and State' if fused_output_state else 'O only',shared_d_tile=fused_output_state,
            vxm_event_rows=len(vxm_events),
            source_layout_copies=0,component_scratch=not direct_bf16,weight_tile_transpose=True,
            transpose_mem_spill=not direct_bf16,sxm_to_mxm_direct=direct_bf16,streaming=streaming_audit,
            partial_products_stay_in_mxm_acc=True,
            outgoing_state_computed=fused_output_state,post_attention_gated_norm=False)
        summary['scope']+='; O=H D (zero initial State)'
        if not args.with_outgoing_state:summary['scope']+='; stop before outgoing State and gated output norm/projection'

    if args.with_outgoing_state:
        from head_weighted_product_runtime_test import reference as weighted_product_reference, audit as audit_weighted_product
        actual_state=np.fromfile(out/f'actual.{h_index+2}.bin','<f4').reshape(2,h//2,dim,d).transpose(1,0,2,3).reshape(h,dim,d)
        actual_keys=read(1,width).reshape(32,h,dim).transpose(1,0,2)
        hardware_ref=weighted_product_reference(actual_keys,actual_r.transpose(0,2,1),actual_d)
        exact_local=actual_keys.astype('f8').transpose(0,2,1)@(actual_r[-1].T[:,:,None]*actual_d.astype('f8'))
        model_state=model_k.astype('f8').transpose(0,2,1)@(model_r[-1].T[:,:,None]*model_d)
        algebra=float(np.linalg.norm(model_state-state_ref)/np.linalg.norm(state_ref))
        assert algebra<2e-5,algebra
        errors_hw=[compare(actual_state[j],hardware_ref[j]) for j in range(h)]
        errors_local=[compare(actual_state[j],exact_local[j]) for j in range(h)]
        errors_model=[compare(actual_state[j],state_ref[j]) for j in range(h)]
        assert np.isfinite(actual_state).all()
        assert all(e['nrmse']<1e-5 for e in errors_hw),errors_hw
        assert all(e['nrmse']<.01 for e in errors_local),errors_local
        assert all(e['nrmse']<.03 and e['cosine']>.999 for e in errors_model),errors_model
        node=next(n for n in plan['nodes'] if n['name']==('chunk_output_state' if fused_output_state else 'chunk_outgoing_state'))
        interval=physical_range(node)
        assert not core_transports
        summary['chunkwise_outgoing_state']=dict(status='passed',shape=[h,dim,d],output_index=h_index+2,
            formula='S_out=K^T diag(R[31,:]) D',initial_state='zero',storage='FP32',
            arithmetic='BF16 K/R, RNE BF16 weighted K, truncated BF16 D, FP32 MXM accumulation/output',
            per_head_hardware_reference=errors_hw,per_head_given_actual_inputs=errors_local,
            per_head_end_to_end=errors_model,chunk_vs_recurrent_reference_nrmse=algebra,
            runtime_range=interval,nominal_cycles=node['nominal_end']-node['nominal_start'],
            r_reused_from_output=7,r_generation_count=1,source_layout_copies=0,
            runtime=audit_weighted_product(out/'runtime.pipeline.csv',h,dim,d,interval,with_output=fused_output_state),
            runtime_scope='fused O and State' if fused_output_state else 'State only',shared_d_tile=fused_output_state,
            nonzero_incoming_state_supported=False,output_norm_projection_included=False)
        summary['scope']+='; zero-State outgoing State'
        if not args.with_output_norm:summary['scope']+='; stop before gated output norm/projection'

    if args.with_output_norm:
        tail_index=h_index+3
        z_actual=read(tail_index,config.value_width)
        norm_actual=read(tail_index+1,config.value_width).reshape(32,h,d)
        # The direct output ABI exports physical token partitions: even
        # heads for all tokens, then odd heads. Consumers use planner row
        # descriptors; unpack here only for the independent host comparison.
        gated_actual=read(tail_index+2,config.value_width).reshape(2,32,h//2,d).transpose(1,2,0,3).reshape(32,h,d)
        yin=trunc_bf16(actual_o).transpose(1,0,2)
        normalize=lambda y:y/np.sqrt(np.mean(y*y,axis=-1,keepdims=True)+config.rms_norm_eps)*bf16(weights.delta_norm)
        norm_ref=normalize(yin)
        zin=z_actual.reshape(32,h,d)
        silu=lambda z:z/(1+np.exp(-z))
        gate_ref=norm_actual*silu(zin)
        z_ref=x.astype('f4')@weights.in_proj_z.T
        model_gated=normalize(recurrent.transpose(1,0,2))*silu(z_ref.reshape(32,h,d))
        tail_errors=dict(z_projection=compare(z_actual,z_ref),rmsnorm=compare(norm_actual,norm_ref),
            gate_given_actual_inputs=compare(gated_actual,gate_ref),gated_norm_end_to_end=compare(gated_actual,model_gated))
        for head in range(h):
            for label,a,b in [('rmsnorm',norm_actual[:,head],norm_ref[:,head]),('gate',gated_actual[:,head],gate_ref[:,head])]:
                err=compare(a,b)
                assert err['nrmse']<.03 and err['cosine']>.999,(label,head,err)
        assert 'delta_output_gate.silu_multiply.native_chain' in schedule
        assert 'rmsnorm.direct_planar_ingress' in schedule
        assert not any(t['phase'].startswith('delta_output_gate') for t in plan.get('linker_mem_transports',[]))
        if args.with_output_projection:
            attention_actual=read(tail_index+3,config.hidden_size)
            tail_errors['output_projection_given_actual_input']=compare(attention_actual,gated_actual.reshape(32,-1)@weights.out_proj.T)
            tail_errors['attention_end_to_end']=compare(attention_actual,model_gated.reshape(32,-1)@weights.out_proj.T)
            assert any(n['name'].startswith('out_proj.page') for n in plan['nodes'])
        assert all(e['nrmse']<.03 and e['cosine']>.999 for e in tail_errors.values()),tail_errors
        if args.with_output_projection and plan.get('weight_page_ping_pong'):
            placements={p['binding']:p for p in plan['weight_page_placements']}
            transfers=[]
            pattern=(r'FTLPU executable page complete: binding=(\d+) page=\d+ bank=(\d+) '
                r'planned_start=(\d+) consumer=(\d+) actual_start=(\d+) '
                r'transport_ready=(\d+) actual_ready=(\d+)')
            consumer_times={c:c+offset for c,offset in epoch_offsets}
            for binding,bank,launch,consumer,begin,transport_ready,ready in re.findall(pattern,log):
                binding,bank,launch,consumer,begin,transport_ready,ready=map(int,
                    (binding,bank,launch,consumer,begin,transport_ready,ready))
                assert placements[binding]['bank']==bank
                event=next(r for r in rows if r['resource']=='C2C.E.Prefetch'
                    and re.search(rf'\bbindings={binding}\b',r['detail']))
                assert int(event['start'])==begin and int(event['end'])==transport_ready, \
                    'Prefetch trace must use the same physical clock as actual page completion'
                transfers.append(dict(binding=binding,bank=bank,logical_launch=launch,
                    logical_consumer=consumer,runtime_start=begin,runtime_ready=ready,
                    runtime_consumer=consumer_times[consumer],
                    ready_before_consumer=ready<=consumer_times[consumer]))
            transfers.sort(key=lambda p:p['logical_consumer'])
            expected_pages = 6 if summary.get('gate_paired_projections') else 7
            assert len(transfers)==expected_pages
            assert [p['bank'] for p in transfers]==[i%2 for i in range(expected_pages)]
            assert all(p['logical_launch']<p['logical_consumer'] for p in transfers)
            summary['weight_paging']=dict(strategy='compiler_reserved_bank_ping_pong',
                synchronization='real C2C MEM commit; consumer-ordered DMA/RX queues',
                slots=plan['weight_page_placements'],transfers=transfers)
        tail_stages=[]
        for n in plan['nodes']:
            if not n['name'].startswith(('in_proj_z','delta_output_','out_proj')):continue
            runtime_start,runtime_end=physical_range(n)
            tail_stages.append(dict(name=n['name'],nominal_start=n['nominal_start'],nominal_end=n['nominal_end'],
                runtime_start=runtime_start,runtime_end=runtime_end))
        gate_stage=next(n for n in tail_stages if n['name']=='delta_output_gate')
        gate_events=[r for r in rows if r['resource'].startswith('VXM.') and
            gate_stage['runtime_start']<=int(r['start'])<gate_stage['runtime_end']]
        assert {r['resource'] for r in gate_events}=={f'VXM.C{i}' for i in range(8)}
        assert len(gate_events)==8 and all(r['pattern']=='single' and int(r['end'])-int(r['start'])==1024 for r in gate_events)
        norm_nodes=[n for n in plan['nodes'] if n['name'].startswith('delta_output_norm.rows')]
        norm_transports=[t for t in plan.get('linker_mem_transports',[]) if t['phase'].startswith('delta_output_norm.rows')]
        assert len(norm_nodes)==16 and not norm_nodes[0]['norm_weight_reused']
        assert all(n['norm_weight_reused'] and n['norm_tight_start'] for n in norm_nodes[1:])
        assert len(norm_transports)==8 and all(t['phase']==norm_nodes[0]['name'] for t in norm_transports),norm_transports
        summary['attention_tail']=dict(status='passed',errors=tail_errors,
            formula='Z=X W_z; Y=RMSNorm_per_head(O)*SiLU(Z); attention=Y W_o',
            norm_weight='delta_norm, no 1+weight offset',precision='truncated BF16 O ingress; BF16 norm/gate storage; FP32 VXM arithmetic',
            gated_output_export_order='[head_parity, token, head_pair, value_feature]',
            initial_state='zero',input_rmsnorm=False,ffn=False,
            stages=tail_stages,gate_continuous_issue_cycles_per_queue=1024,gate_shared_icu_queues=8,
            norm_weight_materializations=1,norm_weight_reuses=15,norm_weight_transport_domains=len(norm_transports),
            norm_compute_span=norm_nodes[-1]['nominal_end']-norm_nodes[0]['nominal_start'],
            norm_span_including_weight_prepare=norm_nodes[-1]['nominal_end']-min(t['start'] for t in norm_transports),
            body_mem_transports=[t for t in plan.get('linker_mem_transports',[]) if not t['phase'].startswith('validation_export.')],
            diagnostic_export_domains=sum(t['phase'].startswith('validation_export.') for t in plan.get('linker_mem_transports',[])))
        summary['scope']+='; Z projection and gated RMSNorm'+('; output projection' if args.with_output_projection else '')
        summary['chunkwise_o']['post_attention_gated_norm']=True
        summary['chunkwise_outgoing_state']['output_norm_projection_included']=args.with_output_projection

    precision_limited = args.with_gamma and not summary['chunkwise_gamma']['end_to_end_head_check_passed']
    precision_limited |= args.with_r and not summary['chunkwise_r']['end_to_end_head_check_passed']
    precision_limited |= args.with_a and not summary['chunkwise_a']['end_to_end_head_check_passed']
    hardware = json.loads(args.target_config.read_text())
    summary.update(status=('passed_with_precision_limits' if precision_limited else 'passed') if passed else 'failed', seq_len=32,
        export_qkv=args.export_qkv, with_output_norm=args.with_output_norm,with_output_projection=args.with_output_projection,
        input_shape=list(x.shape), output_shape=[32, config.hidden_size if args.with_output_projection else config.value_width if args.with_output_norm else 6144], input='raw BF16 fixture; no host RMSNorm',
        real_weights=True, fp32_ingress=args.fp32_ingress, checkpoint_provenance=json.loads((args.package / 'metadata.json').read_text()).get('checkpoint_provenance', {}),
        sxm_trace_rows=len(sxm), with_qk_norm=args.with_qk_norm, with_gates=args.with_gates,
        with_state_decay=args.with_state_decay, with_ks=args.with_ks, with_gt=args.with_gt, with_gamma=args.with_gamma, with_r=args.with_r, with_a=args.with_a,with_l=args.with_l,with_solve=args.with_solve,with_h=args.with_h,with_o=args.with_o,with_outgoing_state=args.with_outgoing_state,
        errors=errors, runtime=stats, runtime_trace_cycles=cycles,
        latency_ms=cycles/(hardware['external_memory']['lpu_clock_mhz']*1000))
    if args.with_l:
        summary['chunkwise_l']=summary.pop('chunkwise_a')
        if not args.with_h:summary['chunkwise_l_coefficient']=summary.pop('chunkwise_r')
        summary['chunkwise_l']['shared_r']=args.with_h
        summary['chunkwise_l']['coefficient_precision']=('FP32 beta broadcast/masked coefficient storage; BF16 shared R; FP32 coefficient preparation and final VXM multiply' if args.with_h else ('FP32' if args.fp32_ingress else 'BF16')+' beta/G ingress; FP32 multiply after Exp; BF16 coefficient storage; FP32 final L')
        summary['chunkwise_l']['a_mem_spill']=False
        if args.with_h:
            summary['chunkwise_l'].update(coefficient_scratch=True,coefficient_spill_dtype='FP32',
                beta_mask_scratch=True,chain_depth=2,
                coefficient_reason='Fixed chain heads accept two streams; internal stages cannot fetch a third MEM operand',
                mask_constant_shape=[32,32],mask_constant_bytes=2048)

        summary['errors']={('l_'+k[2:] if k.startswith('a_') else 'coefficient_'+k[2:] if k.startswith('r_') and not args.with_h else k):v for k,v in errors.items()}
    if args.fp32_ingress:
        summary['errors']={k.replace('gt_given_actual_bf16_g','gt_given_actual_fp32_g').replace('given_quantized_gt','given_hardware_ingress'):v for k,v in summary['errors'].items()}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert passed, errors


if __name__ == '__main__':
    main()
