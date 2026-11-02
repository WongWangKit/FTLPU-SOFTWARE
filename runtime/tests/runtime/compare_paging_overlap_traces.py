"""Compare actual compute traces in logical time, excluding physical HOLD.
Usage: python compare_paging_overlap_traces.py left.csv right.csv [report.json]
Pair this check with linked binary hashes: CSV does not contain full encodings.
"""
import bisect,csv,hashlib,json,re,sys
from collections import defaultdict
from pathlib import Path

def analyze(path):
 rows=list(csv.DictReader(Path(path).open(encoding='utf-8-sig',newline='')))
 invocation=[r for r in rows if r['resource']=='Session.Invocation']
 assert len(invocation)==1,'expected one actual main invocation'
 begin,end=int(invocation[0]['start']),int(invocation[0]['end'])
 waits=[]
 for r in rows:
  if r['resource']=='ICU.PageReadyWait' and r['detail'].startswith('wait=page_ready'):
   for j in range(int(r['outer_count'])):
    for i in range(int(r['inner_count'])):
     shift=i*int(r['inner_interval'])+j*int(r['outer_interval'])
     a,b=max(begin,int(r['start'])+shift),min(end,int(r['end'])+shift)
     if a<b:waits.append((a,b))
 merged=[]
 for a,b in sorted(waits):
  if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
  else:merged.append((a,b))
 starts=[a for a,b in merged];ends=[b for a,b in merged]
 prefix=[0]
 for a,b in merged:prefix.append(prefix[-1]+b-a)
 def logical(p):
  k=bisect.bisect_right(starts,p)-1
  if k>=0 and p<ends[k]:raise AssertionError(f'computation event in HOLD: P={p}')
  return p-begin-prefix[bisect.bisect_right(ends,p)]
 events=defaultdict(list)
 for r in rows:
  if not r['resource'].startswith(('MEM.','MXM.','VXM.','SXM.')):continue
  if r['resource'].endswith('.C2CWrite'):continue # page transport interval, not computation
  detail=re.sub(r' source=runtime issues=\d+$','',r['detail'])
  key=r['resource']+'|'+detail
  for j in range(int(r['outer_count'])):
   for i in range(int(r['inner_count'])):
    shift=i*int(r['inner_interval'])+j*int(r['outer_interval'])
    for p in range(max(begin,int(r['start'])+shift),min(end,int(r['end'])+shift)):
     try: events[key].append(logical(p))
     except AssertionError as ex: raise AssertionError(f'{ex}; {key}; row={r}') from ex
 digest=hashlib.sha256()
 for key,v in sorted(events.items()):
  v.sort();digest.update((key+'\n'+','.join(map(str,v))+'\n').encode())
 return {'file':str(path),'P':end-begin,'HOLD':prefix[-1],
   'L':end-begin-prefix[-1],'actual_compute_trace_cycles':sum(map(len,events.values())),
   'logical_trace_sha256':digest.hexdigest()},events

left,a=analyze(sys.argv[1]);right,b=analyze(sys.argv[2])
same=a==b
report={'left':left,'right':right,'actual_logical_trace_equal':same}
if not same:
 for key in sorted(a.keys()|b.keys()):
  if a.get(key)!=b.get(key):
   report['first_different_resource_detail']=key
   report['left_cycles']=a.get(key,[])[:15];report['right_cycles']=b.get(key,[])[:15];break
result=json.dumps(report,indent=2,ensure_ascii=False)
print(result)
if len(sys.argv)>3:Path(sys.argv[3]).write_text(result+'\n',encoding='utf-8')
sys.exit(0 if same else 1)
