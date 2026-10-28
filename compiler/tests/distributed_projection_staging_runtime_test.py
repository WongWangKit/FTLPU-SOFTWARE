"""RMSNorm distributed output into projection via the passive MEM bridge."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compile','graph-compile','runner','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(status='pending')))
    rng=np.random.default_rng(842)
    x=bf16(rng.normal(0,.4,(32,128)));gamma=bf16(rng.normal(1,.1,128))
    weight=rng.integers(-32,33,(128,128),dtype=np.int8);scale=.01
    g=LogicalGraph();norm=g.norm('norm',x,gamma,1e-6)
    result=g.project('projection',norm,weight,scale)
    logical=g.write(out/'graph',[result]);kernel=out/'kernel.mlir';binary=out/'program.ftlpu'
    commands=([str(a.graph_compile.resolve()),'--input',str(logical),'--output',str(kernel),'--emit-kernel'],
        [str(a.compile.resolve()),'--input',str(kernel),'--input-stage','kernel','--output',str(binary),'--mxm-execution','vector'])
    for i,cmd in enumerate(commands):
        with (out/f'compile{i}.log').open('w') as log:
            subprocess.run(cmd+['--target-config',str(a.target_config.resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
        if i==0:kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.common_barrier_protocol = true} {',1))
    with (out/'execution.log').open('w') as log:
        subprocess.run([str(a.runner.resolve()),str(binary),str(out/'inputs.txt'),str(out/'actual'),str(out/'runtime.pipeline.csv')],stdout=log,stderr=subprocess.STDOUT,check=True)
    reference=(x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*gamma)@(weight.astype(np.float32)*scale).T
    actual=(np.fromfile(out/'actual.0.bin','<u2').astype(np.uint32)<<16).view(np.float32).reshape(reference.shape)
    nrmse=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
    assert np.isfinite(actual).all() and nrmse<.03,nrmse
    report=dict(status='passed',nrmse=nrmse,shape=list(actual.shape))
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
