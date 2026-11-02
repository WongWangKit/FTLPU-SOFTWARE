import csv,json,os,subprocess,sys,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
checks=[]
for seq in (32,128):
 for case in ('25600','51200-jitter'):
  rows={}
  for mode in ('native','pseudo'):
   rows[mode]={(r['kind'],r['index']):r for r in csv.DictReader((root/f'final-{mode}-seq{seq}-{case}-queues.csv').open())}
  assert rows['native'].keys()==rows['pseudo'].keys()
  for key in rows['native']:
   for field in ('wait_cycles','issued','physical_cycles','done'):
    assert rows['native'][key][field]==rows['pseudo'][key][field],(seq,case,key,field)
  checks.append({'seq':seq,'case':case,'queues':len(rows['native']),'fields':['wait_cycles','issued','physical_cycles','done'],'differences':0})
for seq in (32,128):
 for mode in ('native','pseudo'):
  log=root/f'loaded-{mode}-seq{seq}-25600.log'
  assert 'Complete SmolLM2 decoder layer passed:' in log.read_text(),log
 with (root/f'loaded-seq{seq}-trace-check.log').open('w') as log:
  subprocess.run([sys.executable,str(root/'compare-final.py'),str(seq),'25600','loaded'],stdout=log,stderr=subprocess.STDOUT,check=True)
(root/'queue-equivalence.json').write_text(json.dumps(checks,indent=2)+'\n')
manifest={}
for pattern in ('final-*.csv','loaded-*.csv','final-*-linked.ftlpu','compiled-*.ftlpu'):
 for p in root.glob(pattern):
  h=hashlib.sha256()
  with p.open('rb') as f:
   while chunk:=f.read(1024*1024):h.update(chunk)
  manifest[p.name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
(root/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('4 queue-stat comparisons, 4 saved-file executions and 2 saved-file physical traces passed; artifact manifest saved')
