"""Import the remaining layer-0 SWA weights, without changing their FP8 bytes."""
import argparse
import json
import math
from pathlib import Path
import struct
import time

from import_deepseek_v41_layer0_qkv import (
    MODEL, REVISION, ROOT, INDEX_SHA, CONFIG_SHA, CHUNK, byte_range, sha, write_atomic)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cached-index',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    index=a.cached_index.read_bytes()
    if sha(index)!=INDEX_SHA:raise ValueError('pinned index checksum mismatch')
    mapping=json.loads(index)['weight_map']
    specs={'attn_sink':('attn_sink',[64],'F32'),
           'wo_a':('wo_a.weight',[8192,4096],'F8_E4M3'),
           'wo_a_scale':('wo_a.scale',[256,128],'F8_E8M0'),
           'wo_b':('wo_b.weight',[5120,8192],'F8_E4M3'),
           'wo_b_scale':('wo_b.scale',[160,256],'F8_E8M0')}
    manifest=dict(model=MODEL,revision=REVISION,index_sha256=INDEX_SHA,config_sha256=CONFIG_SHA,tensors={})
    old=json.loads((a.output/'provenance.json').read_text()) if (a.output/'provenance.json').exists() else {}
    headers={}
    for role,(suffix,shape,dtype) in specs.items():
        name='layers.0.attn.'+suffix;shard=mapping[name]
        if shard not in headers:
            size=struct.unpack('<Q',byte_range(shard,0,7))[0]
            if size>16*1024*1024:raise ValueError('unexpected header size')
            headers[shard]=(size+8,json.loads(byte_range(shard,8,size+7)))
        base,header=headers[shard];meta=header[name];start,end=meta['data_offsets']
        length=math.prod(shape)*(4 if dtype=='F32' else 1)
        if meta['shape']!=shape or meta['dtype']!=dtype or end-start!=length:raise ValueError(meta)
        path=a.output/(role+'.bin');previous=old.get('tensors',{}).get(role,{})
        if path.exists() and path.stat().st_size==length and sha(path.read_bytes())==previous.get('sha256'):
            payload=path.read_bytes()
        else:
            payload=bytearray()
            for first in range(base+start,base+end,CHUNK):
                last=min(base+end,first+CHUNK)-1
                for attempt in range(4):
                    try:
                        piece=byte_range(shard,first,last)
                        break
                    except ValueError:
                        if attempt==3:
                            # Some HTTP proxies truncate large range responses.
                            # Smaller, independently verified ranges retain the
                            # exact checkpoint bytes and original offsets.
                            piece=b''.join(byte_range(shard,j,min(last,j+256*1024-1))
                                for j in range(first,last+1,256*1024))
                        else:time.sleep(1)
                payload.extend(piece)
                print(role,len(payload),'/',length,flush=True)
            write_atomic(path,payload)
        manifest['tensors'][role]=dict(name=name,file=path.name,dtype=dtype,shape=shape,
            url=ROOT+shard,range_inclusive=[base+start,base+end-1],bytes=length,sha256=sha(payload))
        write_atomic(a.output/'provenance.json',(json.dumps(manifest,indent=2)+'\n').encode())


if __name__=='__main__':main()
