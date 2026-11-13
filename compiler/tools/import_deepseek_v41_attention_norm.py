"""Import the pinned layer-0 attention RMSNorm gamma without changing its bytes."""
import argparse
import json
from pathlib import Path
import struct

from import_deepseek_v41_layer0_qkv import MODEL, REVISION, ROOT, INDEX_SHA, CONFIG_SHA, byte_range, sha, write_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cached-index',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    index=args.cached_index.read_bytes()
    if sha(index)!=INDEX_SHA:raise ValueError('pinned checkpoint index mismatch')
    name='layers.0.attn_norm.weight';shard=json.loads(index)['weight_map'][name]
    manifest_path=args.output/'provenance.json';path=args.output/'attn_norm.bin'
    if manifest_path.exists() and path.exists():
        old=json.loads(manifest_path.read_text());entry=old.get('tensors',{}).get('attn_norm',{})
        if (old.get('revision')==REVISION and entry.get('name')==name
            and entry.get('shape')==[5120] and entry.get('dtype')=='BF16'
            and entry.get('sha256')==sha(path.read_bytes()) and path.stat().st_size==10240):
            print('Reusing verified attention RMSNorm gamma');return
    length=struct.unpack('<Q',byte_range(shard,0,7))[0]
    if length>16*1024*1024:raise ValueError('unexpected safetensors header size')
    header=json.loads(byte_range(shard,8,length+7));entry=header[name]
    start,end=entry['data_offsets'];base=length+8
    if entry['shape']!=[5120] or entry['dtype']!='BF16' or end-start!=10240:
        raise ValueError(f'Unexpected attention norm tensor: {entry}')
    payload=byte_range(shard,base+start,base+end-1)
    write_atomic(path,payload)
    record=dict(model=MODEL,revision=REVISION,index_sha256=INDEX_SHA,config_sha256=CONFIG_SHA,
        tensors={'attn_norm':dict(name=name,file=path.name,shape=[5120],dtype='BF16',bytes=len(payload),
            url=ROOT+shard,range_inclusive=[base+start,base+end-1],sha256=sha(payload))})
    write_atomic(manifest_path,(json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps(record))


if __name__=='__main__':main()
