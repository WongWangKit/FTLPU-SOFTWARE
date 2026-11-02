"""Native pairwise causal exponential: layout, latch reset, FTZ and masking."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_decode_gdn_attention_test import run


def causal_exp_source(tokens, heads, transposed=False, row_scale=False, strict=False):
    shape,dtype=((tokens*heads,tokens),'bf16') if transposed else ((tokens*tokens,heads),'f32')
    a, o = LogicalGraph.type((tokens, heads), 'f32'), LogicalGraph.type(shape, dtype)
    return f'''module {{ func.func @stage(%x: {a}{", %b: " + a if row_scale else ""}) -> {o} {{
      %r = ftlpu.kernel.vector_alu %x, {"%b" if row_scale else "%x"} {{kind = "causal_exp_difference",
        transpose_result = {str(transposed).lower()}, row_scale = {str(row_scale).lower()}, strict_lower = {str(strict).lower()},
        use_immediate = {str(not row_scale).lower()}, immediate = 0.0 : f32, reduce = false}} : ({a}, {a}) -> {o}
      return %r : {o}
    }} }}'''


def reference(g, fp32=False):
    q = g.astype('f4') if fp32 else (g.astype('f4').view('u4') & np.uint32(0xffff0000)).view('f4')
    diff = q[:, None, :] - q[None, :, :]
    lower = np.tri(g.shape[0], dtype=bool)[:, :, None]
    # Don't evaluate masked positive differences: these can overflow Exp.
    diff = np.where(lower, diff, -np.inf).astype('f2').astype('f4')
    diff[np.abs(diff) < np.finfo(np.float16).tiny] = 0
    return np.exp(diff)


def read_transposed(path, heads):
    bits=np.fromfile(path,'<u2').astype('u4') << 16
    return bits.view('f4').reshape(32,heads,32).transpose(0,2,1)


def check(actual, g, transposed=False, beta=None, strict=False, fp32=False):
    t, h = g.shape
    actual = actual.reshape(t, t, h)
    expected = reference(g,fp32)
    if beta is not None:
        quantized=beta.astype('f4') if fp32 else (beta.astype('f4').view('u4') & np.uint32(0xffff0000)).view('f4')
        expected=expected*quantized[:,None,:]
    if strict: expected[np.arange(t),np.arange(t)]=0
    np.testing.assert_allclose(actual, expected, rtol=.01 if transposed else .002, atol=float(np.finfo(np.float16).tiny))
    np.testing.assert_allclose(actual[np.arange(t), np.arange(t)], expected[np.arange(t),np.arange(t)], rtol=.01,atol=0)
    assert np.all(actual[np.triu_indices(t, 0 if strict else 1)] == 0)
    assert np.isfinite(actual).all() and np.all((actual >= 0) & (actual <= 1))
    return dict(max_abs=float(np.max(abs(actual-expected))), zero_outputs=int(np.sum(actual == 0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fp32-ingress',action='store_true')
    a = p.parse_args()
    build = a.build_dir.resolve()
    suffix = '.exe' if sys.platform == 'win32' else ''
    exe = lambda area, name: build/area/(name+suffix)
    cases = []
    for t, h, direct, scaled in ((1,3,False,False), (8,7,False,False), (32,16,False,False), (32,16,True,False), (1,3,False,True), (8,7,False,True), (32,16,True,True)):
        out = a.output_dir.resolve()/f'seq{t}{"_direct" if direct else ""}{"_scaled_strict" if scaled else ""}'
        out.mkdir(parents=True, exist_ok=True)
        x = -np.cumsum(np.random.default_rng(t).uniform(.001, .3, (t,h)).astype('f4'), axis=0)
        x[:, 0] = 0  # Plateau, exact unit lower triangle.
        x[:, 1] = -np.arange(t, dtype='f4')*8  # FTZ and masked-overflow stress.
        graph = LogicalGraph()
        shape,dtype=((t*h,t),'bf16') if direct else ((t*t,h),'f32')
        beta=np.random.default_rng(t+11).uniform(0,1,(t,h)).astype('f4')
        beta[:,0]=1; beta[:,1]=0
        inputs=[(x,'f32')]+([(beta,'f32')] if scaled else [])
        result = graph.run('causal_difference', causal_exp_source(t,h,direct,scaled,scaled), inputs, [(shape,dtype)])[0]
        logical = graph.write(out/'logical', [result])
        kernel, binary = out/'kernel.mlir', out/'program.ftlpu'
        run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel',
             '--target-config',a.target_config.resolve()], out/'import.log')
        kernel.write_text(kernel.read_text().replace('module {', 'module attributes {'
            'ftlpu.vector_bf16_spill = true, ftlpu.vector_bf16_token_halves = true, '
            f'ftlpu.vector_fp32_ingress = {str(a.fp32_ingress).lower()}, ftlpu.common_barrier_protocol = true, ftlpu.sxm_wide = true}} {{',1))
        run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,
             '--target-config',a.target_config.resolve(),'--icu-compression','raw-macro'], out/'compile.log')
        os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1'
        os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
        run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
        actual=read_transposed(out/'actual.0.bin',h) if direct else np.fromfile(out/'actual.0.bin','<f4')
        error = check(actual, x, direct,beta if scaled else None,scaled,a.fp32_ingress)
        if direct and not scaled:
            baseline=np.fromfile(out.parent/'seq32'/'actual.0.bin','<u4') & np.uint32(0xffff0000)
            np.testing.assert_array_equal(actual,baseline.view('f4').reshape(t,t,h))
        plan = json.loads((out/'graph_plan.json').read_text())
        assert not plan.get('linker_mem_transports', [])
        assert len(plan['nodes']) == 1
        cases.append(dict(tokens=t, heads=h, direct_transpose=direct, scaled_strict=scaled, **error))
    (a.output_dir/'summary.json').write_text(json.dumps(dict(status='passed',cases=cases),indent=2))
    print(cases)


if __name__ == '__main__':
    main()
