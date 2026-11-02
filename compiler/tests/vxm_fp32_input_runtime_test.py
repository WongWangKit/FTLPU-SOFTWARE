"""Complete FP32 two-operand ingress through raw ICU packets and real MEM SR."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_decode_gdn_attention_test import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    build=a.build_dir.resolve();suffix='.exe' if sys.platform=='win32' else ''
    exe=lambda area,name:build/area/(name+suffix)
    x=-128-np.arange(1,129,dtype='f4').reshape(8,16)*2**-16
    y=np.full_like(x,-128)
    ty=LogicalGraph.type(x.shape,'f32')
    source=f'''module {{ func.func @stage(%x: {ty}, %y: {ty}) -> {ty} {{
      %r = ftlpu.kernel.vector_alu %x, %y {{kind = "subtract", use_immediate = false,
        immediate = 0.0 : f32, reduce = false}} : ({ty}, {ty}) -> {ty}
      return %r : {ty}
    }} }}'''
    graph=LogicalGraph(); result=graph.run('full_fp32_subtract',source,[(x,'f32'),(y,'f32')],[(x.shape,'f32')])[0]
    logical=graph.write(out/'logical',[result]);kernel=out/'kernel.mlir';binary=out/'program.ftlpu'
    run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel','--target-config',a.target_config.resolve()],out/'import.log')
    kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.vector_bf16_spill = true, ftlpu.vector_fp32_ingress = true} {',1))
    run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,'--target-config',a.target_config.resolve(),'--icu-compression','raw-macro'],out/'compile.log')
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1'
    run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
    actual=np.fromfile(out/'actual.0.bin','<f4').reshape(x.shape)
    np.testing.assert_array_equal(actual,x-y)
    assert np.all(actual!=0), 'BF16 truncation would lose every tested difference'
    (out/'summary.json').write_text(json.dumps(dict(status='passed',elements=actual.size,bit_exact=True,input='two full FP32 operands',icu='raw-macro'),indent=2))
    print('FP32 two-port MEM -> ICU -> VXM subtraction: bit-exact')

if __name__=='__main__': main()
