import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
build=root.parent/'main-common-barrier-ninja/runtime'
seq=int(sys.argv[1]);mode=sys.argv[2]
exe=build/('compiled_smollm2_common_barrier_'+('seq128_' if seq==128 else '')+'decoder_test.exe')
for label,bw,extra in [('25600',25600,None),('51200-jitter',51200,17)]:
 env=os.environ.copy()
 env.pop('FTLPU_QUEUE_CONTROL_FINAL_ENCODING',None)
 env.pop('FTLPU_DDR_EXTRA_READ_LATENCY',None)
 if mode=='native':env['FTLPU_QUEUE_CONTROL_FINAL_ENCODING']='native'
 if extra is not None:env['FTLPU_DDR_EXTRA_READ_LATENCY']=str(extra)
 env['FTLPU_DDR_BANDWIDTH_MBPS']=str(bw)
 tag=f'final-{mode}-seq{seq}-{label}'
 env['FTLPU_QWEN_C2C_PIPELINE_CSV']=str(root/(tag+'.csv'))
 env['FTLPU_QWEN_C2C_LINKED_BINARY']=str(root/(tag+'-linked.ftlpu'))
 env['FTLPU_QUEUE_PROGRESS_CSV']=str(root/(tag+'-queues.csv'))
 args=[str(exe),str(root/f'compiled-seq{seq}-none.ftlpu')]
 if mode=='pseudo':args.append(str(root/f'compiled-seq{seq}-raw-macro.ftlpu'))
 with (root/(tag+'.log')).open('w') as log:
  code=subprocess.call(args,env=env,stdout=log,stderr=subprocess.STDOUT)
 print(tag,code,flush=True)
 if code:sys.exit(code)
