"""Native chunkwise GDN through the production raw-macro compiler and CModel."""
import argparse,json,subprocess,os,csv
from pathlib import Path
import numpy as np
from gated_delta_chunkwise_runtime_test import bf16,recurrent
p=argparse.ArgumentParser()
for name in ('compile','runner','target-config','output-dir'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--tokens',type=int,default=32)
a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(2364);t=a.tokens;d=128
q,k=[rng.normal(size=(t,d)).astype('f4') for _ in range(2)]
q=bf16(q/np.linalg.norm(q,axis=1,keepdims=True)/np.sqrt(d));k=bf16(k/np.linalg.norm(k,axis=1,keepdims=True))
v=bf16(rng.normal(0,.2,(t,d)).astype('f4'));alpha=bf16(rng.uniform(.9,.999,(t,1)).astype('f4'));beta=bf16(rng.uniform(.05,.95,(t,1)).astype('f4'));s=bf16(rng.normal(0,.02,(d,d)).astype('f4'))
values=[q,k,v,alpha,beta,s];types=[f'tensor<{t}x128xbf16>']*3+[f'tensor<{t}x1xf32>']*2+['tensor<128x128xf32>'];paths=[]
for i,value in enumerate(values):
 f=out/f'in{i}.bin';paths.append(json.dumps(f.as_posix()));(value.view('u4')>>16).astype('<u2').tofile(f) if i<3 else value.astype('<f4').tofile(f)
args=', '.join(f'%a{i}: {ty}' for i,ty in enumerate(types));operands=', '.join(f'%a{i}' for i in range(6));results=f'({types[0]}, {types[-1]})'
kernel=out/'test.mlir';binary=out/'test.ftlpu'
kernel.write_text(f'''module attributes {{ftlpu.gdn_chunk_size = 32 : i64, ftlpu.common_barrier_protocol = true}} {{
func.func @decoder({args}) -> {results} attributes {{ftlpu.artifact_directory = {json.dumps(out.as_posix())}, ftlpu.logical_inputs = [{', '.join(paths)}]}} {{
%o, %s = ftlpu.kernel.gated_delta_rule {operands} {{heads = 1 : i64, key_dim = 128 : i64, value_dim = 128 : i64}} : ({', '.join(types)}) -> {results}
return %o, %s : {types[0]}, {types[-1]}
}} }}''')
for name,cmd in [('compile',[a.compile,'--input',kernel,'--input-stage','kernel','--output',binary,'--target-config',a.target_config,'--mxm-execution','vector','--icu-compression','raw-macro']),('execution',[a.runner,binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'])]:
 with (out/f'{name}.log').open('w') as log:subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,"FTLPU_FILE_RUNNER_MODEL_SESSION":"1","FTLPU_SESSION_PROGRESS":"1"})
actual=[(np.fromfile(out/'actual.0.bin','<u2').astype('u4')<<16).view('f4').reshape(t,d),np.fromfile(out/'actual.1.bin','<f4').reshape(d,d)]
schedule=(out/'decoder_layer.schedule.mlir').read_text()
for operation in ('KKt','KS','QS','QKt','attention_D','state_update'):
 assert f'chunk0.{operation}.mxm0' in schedule, operation
with (out/'runtime.pipeline.csv').open() as trace:
 resources={row['resource'] for row in csv.DictReader(trace)}
assert any(r.startswith('MXM.') for r in resources), resources
expected=recurrent(*values);errors=[float(np.linalg.norm(x-y)/np.linalg.norm(y)) for x,y in zip(actual,expected)]
summary=dict(status='passed' if max(errors)<.02 else 'failed',tokens=t,errors=errors,encoding='raw-macro',nonzero_initial_state=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary);assert summary['status']=='passed'
