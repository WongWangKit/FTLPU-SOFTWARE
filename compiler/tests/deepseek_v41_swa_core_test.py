"""Native shared attention core: D512 latent KV and an observable per-head sink."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from deepseek_v41_qkv_norm_test import run, trace_times
from import_hf_decoder_layer import bf16, bf16_bits, read_bf16
from deepseek_v41_qkv_reference import act_quant, dequant_act, rope


def reference(q, kv, sinks, causal=True):
    scores = np.einsum('thd,sd->hts', q.astype('f8'), kv.astype('f8')) / np.sqrt(q.shape[-1])
    if causal:
        scores = np.where(np.arange(len(kv))[None, None, :] <= np.arange(len(q))[None, :, None], scores, -np.inf)
    maximum = np.maximum(scores.max(-1, keepdims=True), sinks[:, None, None])
    numerator = np.exp(scores - maximum)
    probabilities = numerator / (numerator.sum(-1, keepdims=True) + np.exp(sinks[:, None, None] - maximum))
    return np.einsum('hts,sd->thd', probabilities, kv.astype('f8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--reference-dir', type=Path)
    p.add_argument('--sink-file', type=Path)
    p.add_argument('--input-run',type=Path,help='Use Q/KV checkpoints from a passed continuous BF16 RoPE run')
    p.add_argument('--precision-policy',choices=('bf16-activation','official-fp8'),default='bf16-activation')
    p.add_argument('--device-rope',action='store_true',help='Exercise the native planar RoPE -> attention interface')
    p.add_argument('--device-kv-qdq', action='store_true', help='Run group32 FP8 QDQ on the device before the shared attention core')
    args = p.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    heads=args.heads
    rng=np.random.default_rng(41111)
    q=bf16(rng.normal(0,.35,(32,heads,512)).astype('f4'))
    kv=bf16(rng.normal(0,.35,(32,512)).astype('f4'))
    sinks=np.linspace(-2,3,heads,dtype='f4')
    if args.reference_dir:
        q=np.load(args.reference_dir/'q_rope.npy')[:,:heads].copy()
        kv=np.load(args.reference_dir/'kv_rope.npy').copy()
        if args.precision_policy=='official-fp8' and not args.device_kv_qdq:kv=dequant_act(*act_quant(kv))
    if args.input_run:
        previous=json.loads((args.input_run/'summary.json').read_text())
        assert previous['status']=='passed' and previous['sequence']==32
        q=read_bf16(args.input_run/'actual.6.bin',(32,64,512))[:,:heads].copy()
        kv=read_bf16(args.input_run/'actual.7.bin',(32,512))
    if args.sink_file:sinks=np.fromfile(args.sink_file,'<f4')[:heads]
    bf16_bits(q).tofile(out/'q.bin');bf16_bits(kv).tofile(out/'kv.bin')
    qt=f'tensor<1x32x{heads}x512xbf16>';kt='tensor<1x32x1x512xbf16>'
    kv_prefix=''
    kv_operand='%kv'
    query_operand='%q'
    if args.device_rope:
        assert not args.device_kv_qdq
        q3=f'tensor<32x{heads}x512xbf16>';k3='tensor<32x1x512xbf16>'
        kv_prefix=f'''%qr0 = ftlpu.kernel.reshape %q : ({qt}) -> {q3}
          %qr1 = ftlpu.kernel.rope %qr0 {{heads={heads}:i64,head_dim=512:i64,theta=1.000000e+04:f32,rotary_dim=64:i64,rotary_offset=448:i64,interleaved=true,inverse=false}} : ({q3}) -> {q3}
          %qr = ftlpu.kernel.reshape %qr1 : ({q3}) -> {qt}
          %kr0 = ftlpu.kernel.reshape %kv : ({kt}) -> {k3}
          %kr1 = ftlpu.kernel.rope %kr0 {{heads=1:i64,head_dim=512:i64,theta=1.000000e+04:f32,rotary_dim=64:i64,rotary_offset=448:i64,interleaved=true,inverse=false}} : ({k3}) -> {k3}
          %kr = ftlpu.kernel.reshape %kr1 : ({k3}) -> {kt}'''
        query_operand='%qr';kv_operand='%kr'
    if args.device_kv_qdq:
        kv_prefix=f'''%kv_flat = ftlpu.kernel.reshape %kv : ({kt}) -> tensor<32x512xbf16>
          %kv_quant = ftlpu.kernel.block_fp8_qdq %kv_flat : (tensor<32x512xbf16>) -> tensor<32x512xbf16>
          %window_kv = ftlpu.kernel.reshape %kv_quant : (tensor<32x512xbf16>) -> {kt}'''
        kv_operand='%window_kv'
    files=','.join(json.dumps((out/f).as_posix()) for f in ('q.bin','kv.bin'))
    (out/'kernel.mlir').write_text(f'''module attributes {{ftlpu.sxm_wide=true, ftlpu.vector_fp32_ingress=true, ftlpu.common_barrier_protocol=true}} {{
      func.func @swa(%q:{qt}, %kv:{kt}) -> tensor<32x{heads*512}xbf16>
      attributes {{ftlpu.logical_inputs=[{files}],ftlpu.artifact_directory="{out.as_posix()}"}} {{
        {kv_prefix}
        %o = ftlpu.kernel.dense_attention {query_operand}, {kv_operand}, {kv_operand} {{scale={512**-.5:.9e}:f32,
          causal=true,query_position=0:i64,sliding_window=128:i64,
          attention_sinks=array<f32:{','.join(f'{v:.9e}' for v in sinks)}>}}
          : ({qt},{kt},{kt}) -> {qt}
        %flat = ftlpu.kernel.reshape %o : ({qt}) -> tensor<32x{heads*512}xbf16>
        return %flat : tensor<32x{heads*512}xbf16>
      }}
    }}''')
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',FTLPU_GRAPH_PROGRESS='1')
    ext='.exe' if os.name=='nt' else ''
    exe=lambda area,name:args.build_dir.resolve()/area/(name+ext)
    result=dict(status='running',scope='SWA core only; Q/KV are fixtures, no projections',
                query_shape=list(q.shape),kv_shape=list(kv.shape),relative_l2_limit=.01,raw_macro=True,
                fixture_kv_fp8_qdq=bool(args.reference_dir) and args.precision_policy=='official-fp8' and not args.device_kv_qdq,
                device_kv_fp8_qdq=args.device_kv_qdq,command=[sys.executable,*sys.argv])
    try:
        run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel',
             '--target-config',args.target_config.resolve(),'--icu-compression','raw-macro',
             '--output',out/'program.ftlpu'],out/'compile.log',env)
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',
             out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs',
             '--pre-execution-dir',out/'actual.pre_execution','--compute-imem-segments'],out/'export.log',env)
        if not args.device_kv_qdq:
            schedule=(out/'decoder_layer.schedule.mlir').read_text()
            assert 'distributed_16' not in schedule and 'fp16_value_x16' not in schedule
            assert 'flash.tile_pipeline' in schedule
        actual=read_bf16(out/'actual.0.bin',q.shape)
        reference_kv=dequant_act(*act_quant(kv)) if args.device_kv_qdq else kv
        if args.device_rope:
            q=rope(q,64,10000);reference_kv=rope(kv,64,10000)
        expected=reference(q,reference_kv,sinks)
        error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        # Removing the sink must measurably change this fixture.
        no_sink=reference(q,reference_kv,np.full(heads,-1e20))
        effect=float(np.linalg.norm(expected-no_sink)/np.linalg.norm(expected))
        unmasked=reference(q,reference_kv,sinks,causal=False)
        causal_effect=float(np.linalg.norm(expected-unmasked)/np.linalg.norm(expected))
        stats=dict(re.findall(r'(\w+)=(\d+)',(out/'execution.log').read_text().splitlines()[-1]))
        result.update(relative_l2=error,sink_effect=effect,causal_effect=causal_effect,
                      runtime=stats,timing=trace_times(out/'runtime.pipeline.csv'))
        assert np.isfinite(actual).all() and error<result['relative_l2_limit'],result
        if not args.sink_file and not args.reference_dir and not args.input_run:
            assert effect>.1 and causal_effect>.1,(effect,causal_effect)
        assert stats['host_operations']=='0'
        result['status']='passed'
    except Exception as e:
        result.update(status='failed',error=str(e));raise
    finally:
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
