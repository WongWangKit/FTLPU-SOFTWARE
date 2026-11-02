import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
build=root.parent/'main-common-barrier-ninja/runtime'
for seq in (32,128):
 for mode in ('native','pseudo'):
  exe=build/('compiled_smollm2_common_barrier_'+('seq128_' if seq==128 else '')+'decoder_test.exe')
  env=os.environ.copy()
  for key in ('FTLPU_QUEUE_CONTROL_FINAL_ENCODING','FTLPU_DDR_EXTRA_READ_LATENCY'):
   env.pop(key,None)
  if mode=='native':env['FTLPU_QUEUE_CONTROL_FINAL_ENCODING']='native'
  env['FTLPU_DDR_BANDWIDTH_MBPS']='25600'
  saved=root/f'final-{mode}-seq{seq}-51200-linked.ftlpu'
  if not saved.exists():saved=root/f'final-{mode}-seq{seq}-51200.ftlpu.loaded.ftlpu'
  env['FTLPU_SESSION_EXECUTE_LINKED_IMAGE']=str(saved)
  tag=f'loaded-{mode}-seq{seq}-25600'
  if '--resume' in sys.argv and (root/(tag+'.log')).exists() and 'Complete SmolLM2 decoder layer passed:' in (root/(tag+'.log')).read_text():
   print(tag,'already passed',flush=True)
   continue
  env['FTLPU_QWEN_C2C_PIPELINE_CSV']=str(root/(tag+'.csv'))
  env['FTLPU_QUEUE_PROGRESS_CSV']=str(root/(tag+'-queues.csv'))
  args=[str(exe),str(root/f'compiled-seq{seq}-none.ftlpu')]
  if mode=='pseudo':args.append(str(root/f'compiled-seq{seq}-raw-macro.ftlpu'))
  with (root/(tag+'.log')).open('w') as log:
   code=subprocess.call(args,env=env,stdout=log,stderr=subprocess.STDOUT)
  print(tag,code,flush=True)
  if code:sys.exit(code)
