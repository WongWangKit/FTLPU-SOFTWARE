import csv,json,re,hashlib,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
build=root.parent/'main-common-barrier-ninja/runtime'
matrix=[]
for seq in (32,128):
 for label in ('51200','25600','51200-jitter'):
  case={'seq':seq,'ddr_case':label,'modes':{}}
  for mode in ('native','pseudo'):
   prefix=f'final-{mode}-seq{seq}-{label}'
   log=(root/(prefix+'.log')).read_text()
   assert 'Complete SmolLM2 decoder layer passed:' in log
   line=next(x for x in log.splitlines() if x.startswith('Qwen dynamic C2C pages passed:'))
   stats={k:int(v) for k,v in re.findall(r'(\w+)=([0-9]+)',line)}
   stats['completion_tail_cycles']=stats.pop('runtime_wait_cycles',stats.get('completion_tail_cycles',0))
   file=root/(prefix+'-linked.ftlpu')
   if not file.exists():file=root/(prefix+'.ftlpu.loaded.ftlpu')
   stats['linked_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
   stats['file']=file.name
   queuefile=root/(prefix+'-queues.csv')
   if queuefile.exists():
    rows=list(csv.DictReader(queuefile.open()))
    assert all(r['done']=='1' for r in rows)
    stats['queue_count']=len(rows)
    stats['queue_wait_range']=[min(int(r['wait_cycles']) for r in rows),max(int(r['wait_cycles']) for r in rows)]
    stats['waiting_queues']=sum(int(r['wait_cycles'])>0 for r in rows)
   case['modes'][mode]=stats
  case['trace_differences']=json.loads((root/f'final-seq{seq}-{label}-trace-differences.json').read_text())
  assert not case['trace_differences']
  assert case['modes']['native']['P']==case['modes']['pseudo']['P']
  matrix.append(case)
storage={}
for seq in (32,128):
 for mode in ('native','pseudo'):
  file=next(c for c in matrix if c['seq']==seq)['modes'][mode]['file']
  line=subprocess.check_output([str(build/'ftlpu_binary_inspect.exe'),str(root/file),'--storage-only'],text=True).strip()
  storage[f'{mode}-seq{seq}']={k:int(v) for k,v in re.findall(r'(\w+)=([0-9]+)',line)}
  (root/f'storage-{mode}-seq{seq}.log').write_text(line+'\n')
  print(mode,seq,line,flush=True)
result={'date':'2026-10-29','software_base':'7ba85e7','cmodel_base':'bc84188','matrix':matrix,'storage':storage}
(root/'validation-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for c in matrix:
 s=c['modes']['pseudo'];print(c['seq'],c['ddr_case'],'P',s['P'],'waits',s.get('queue_wait_range'),'prime',s['PRIME'],flush=True)
