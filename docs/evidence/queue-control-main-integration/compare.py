import csv, collections, hashlib, json, re, subprocess, sys
from pathlib import Path
sw=next(p for p in Path(__file__).resolve().parents if (p/'runtime/CMakeLists.txt').is_file())
root=sw/'out/main-qctl-integration'
seq=sys.argv[1]; bw='25600'
def events(mode):
    out=collections.defaultdict(list)
    for r in csv.DictReader((root/f'{mode}-seq{seq}-{bw}.csv').open()):
        resource=r['resource']
        if not(resource.startswith(('MEM.','MXM.','VXM.','SXM.')) or resource in ('C2C.E.DMA','C2C.W.DMA','C2C.E.RX','C2C.W.RX','Paging.Activity')): continue
        resource=resource.replace('Read3D','Read').replace('Write3D','Write').replace('WriteTap3D','WriteTap')
        key=resource+' '+re.sub(r' (pc|source|issues)=[^ ]+','',r['detail'])
        for j in range(int(r['outer_count'])):
            for i in range(int(r['inner_count'])):
                if int(r['skip_first']) and i==j==0:continue
                start=int(r['start'])+i*int(r['inner_interval'])+j*int(r['outer_interval'])
                out[key].extend(range(start,start+int(r['end'])-int(r['start'])))
    for v in out.values():v.sort()
    return out
a,b=events('native'),events('pseudo')
diff=[{'key':k,'native':len(a[k]),'macro':len(b[k]),'first':next(((x,y) for x,y in zip(a[k],b[k]) if x!=y),None)} for k in sorted(a.keys()|b.keys()) if a[k]!=b[k]]
result={'sequence':int(seq),'bandwidth':int(bw),'compared_resources':len(a.keys()|b.keys()),'native_activity_cycles':sum(map(len,a.values())),'macro_activity_cycles':sum(map(len,b.values())),'differences':diff}
(root/f'seq{seq}-trace-comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
assert not diff
exe=root/'build/runtime/binary_queue_control_test.exe'
native=root/f'native-seq{seq}-{bw}-linked.ftlpu';macro=root/f'pseudo-seq{seq}-{bw}-linked.ftlpu'
with (root/f'seq{seq}-image-comparison.log').open('w') as f:
    subprocess.run([str(exe),'--compare-final',str(native),str(macro)],stdout=f,stderr=subprocess.STDOUT,check=True)
    for image in (native,macro):subprocess.run([str(exe),'--validate-final',str(image)],stdout=f,stderr=subprocess.STDOUT,check=True)
print('execution image equivalence and independent final loads passed',flush=True)
