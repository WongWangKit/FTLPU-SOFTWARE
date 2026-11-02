"""Row-prefix scan: lane independence, reset, and no token-half partition."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_decode_gdn_attention_test import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    build = a.build_dir.resolve()
    suffix = '.exe' if sys.platform=='win32' else ''
    exe = lambda area, name: build / area / (name+suffix)
    cases = []
    for tokens in (1, 32, 64):
        out = a.output_dir.resolve() / f'seq{tokens}'
        out.mkdir(parents=True, exist_ok=True)
        x = np.random.default_rng(35000+tokens).normal(0, .2, (tokens, 16)).astype('f4')
        x += np.arange(16, dtype='f4')[None, :] * .03125
        g = LogicalGraph()
        first = g.vector_tiles('prefix', [(x, 'f32')], lambda v: v.add(0, kind='cumsum_rows', immediate=0))
        second = g.vector_tiles('prefix_again', [(first, 'f32')], lambda v: v.add(0, kind='cumsum_rows', immediate=0))
        logical = g.write(out/'logical', [first, second])
        kernel, binary = out/'kernel.mlir', out/'program.ftlpu'
        run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel',
             '--target-config',a.target_config.resolve()],out/'import.log')
        kernel.write_text(kernel.read_text().replace('module {', 'module attributes {'
            'ftlpu.vector_bf16_spill = true, ftlpu.vector_bf16_token_halves = true, '
            'ftlpu.common_barrier_protocol = true, ftlpu.sxm_wide = true} {',1))
        run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,
             '--target-config',a.target_config.resolve(),'--icu-compression','raw-macro'],out/'compile.log')
        os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1'
        os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
        run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
        expected = x
        errors=[]
        for i in range(2):
            expected=np.cumsum((expected.view('u4') & np.uint32(0xffff0000)).view('f4'),axis=0,dtype=np.float32)
            actual=np.fromfile(out/f'actual.{i}.bin','<f4').reshape(x.shape)
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
            errors.append(float(np.max(np.abs(actual-expected))))
        plan=json.loads((out/'graph_plan.json').read_text())
        assert [n['name'] for n in plan['nodes']]==['prefix','prefix_again']
        cases.append(dict(tokens=tokens,max_abs_errors=errors))
    (a.output_dir/'summary.json').write_text(json.dumps(dict(status='passed',cases=cases),indent=2))
    print(cases)


if __name__=='__main__':
    main()
