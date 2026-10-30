import csv,re,collections,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
seq=sys.argv[1] if len(sys.argv)>1 else '32'
bw=sys.argv[2] if len(sys.argv)>2 else '51200'
prefix=sys.argv[3] if len(sys.argv)>3 else 'final'
def events(mode):
 result=collections.defaultdict(list)
 for r in csv.DictReader((root/f'{prefix}-{mode}-seq{seq}-{bw}.csv').open()):
  resource=r['resource']
  if not (resource.startswith(('MEM.','MXM.','VXM.','SXM.')) or resource in ('C2C.E.DMA','C2C.W.DMA','C2C.E.RX','C2C.W.RX','Paging.Activity')): continue
  resource=resource.replace('Read3D','Read').replace('Write3D','Write').replace('WriteTap3D','WriteTap')
  detail=re.sub(r' (pc|source|issues)=[^ ]+','',r['detail'])
  key=resource+' '+detail
  for j in range(int(r['outer_count'])):
   for i in range(int(r['inner_count'])):
    if int(r['skip_first']) and i==0 and j==0:continue
    start=int(r['start'])+i*int(r['inner_interval'])+j*int(r['outer_interval'])
    result[key].extend(range(start,start+int(r['end'])-int(r['start'])))
 for v in result.values():v.sort()
 return result
a,b=events('native'),events('pseudo');out=[]
for key in sorted(a.keys()|b.keys()):
 x,y=a[key],b[key]
 if x!=y:out.append({'key':key,'native_count':len(x),'pseudo_count':len(y),'first':next(((u,v) for u,v in zip(x,y) if u!=v),None),'deltas':dict(collections.Counter(v-u for u,v in zip(x,y)))})
(root/f'{prefix}-seq{seq}-{bw}-trace-differences.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'compared_resources':len(a.keys()|b.keys()),'native_cycles':sum(map(len,a.values())),'pseudo_cycles':sum(map(len,b.values())),'differences':len(out)}))
for v in out[:10]:print(v)
sys.exit(bool(out))
