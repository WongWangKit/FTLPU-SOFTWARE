"""Fixtures and independent final-output reference; all GDN execution is CModel."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from gated_delta_chunkwise_runtime_test import bf16, recurrent

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runner',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--tokens',type=int,choices=(32,64),default=64)
a=p.parse_args()
out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(2364)
q,k=[rng.normal(size=(a.tokens,128)).astype(np.float32) for _ in range(2)]
q=bf16(q/np.linalg.norm(q,axis=1,keepdims=True)/np.sqrt(128))
k=bf16(k/np.linalg.norm(k,axis=1,keepdims=True))
v=bf16(rng.normal(0,.2,(a.tokens,128)).astype(np.float32))
alpha=bf16(rng.uniform(.9,.999,(a.tokens,1)).astype(np.float32))
beta=bf16(rng.uniform(.05,.95,(a.tokens,1)).astype(np.float32))
state=bf16(rng.normal(0,.02,(128,128)).astype(np.float32))
for name,value in zip(('q','k','v','alpha','beta','state'),(q,k,v,alpha,beta,state)):
    value.astype('<f4').tofile(out/(name+'.bin'))
run=out/'native';run.mkdir(exist_ok=True)
with (out/'runner.log').open('w') as log:
    subprocess.run([str(a.runner.resolve()),str(out),str(run),str(a.tokens)],stdout=log,stderr=subprocess.STDOUT,check=True)
expected=recurrent(q,k,v,alpha,beta,state)
actual=[np.fromfile(run/'output.bin','<f4').reshape(a.tokens,128),np.fromfile(run/'state.bin','<f4').reshape(128,128)]
errors={name:float(np.linalg.norm(x-y)/np.linalg.norm(y)) for name,x,y in zip(('output_nrmse','state_nrmse'),actual,expected)}
summary=json.loads((run/'execution.json').read_text())
summary.update(status='passed' if all(np.isfinite(e) and e<.01 for e in errors.values()) else 'failed',
               errors=errors,threshold_nrmse=.01,tokens=a.tokens,chunk_size=32,heads=1,key_dim=128,value_dim=128,
               nonzero_initial_state=True,real_model_weights=False,
               reference='independent token recurrence; BF16 ingress, FP32 state',
               scope='complete GDN recurrence core; excludes projections/conv/gates and C2C')
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
assert summary['status']=='passed'
