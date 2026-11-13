"""Import pinned real mHC parameters; retain FP32 originals and explicit BF16 RNE copies."""
import argparse
import json
import math
from pathlib import Path
import struct
import numpy as np
from import_hf_decoder_layer import bf16_bits
from import_deepseek_v41_layer0_qkv import MODEL, REVISION, ROOT, INDEX_SHA, CONFIG_SHA, byte_range, sha, write_atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cached-index',type=Path,required=True)
    p.add_argument('--layer',type=int,default=0);p.add_argument('--branch',choices=('attn','ffn'),default='attn')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    raw=a.cached_index.read_bytes()
    if sha(raw)!=INDEX_SHA:raise ValueError('checkpoint index checksum mismatch')
    mapping=json.loads(raw)['weight_map'];headers={}
    manifest=dict(model=MODEL,revision=REVISION,layer=a.layer,branch=a.branch,index_sha256=INDEX_SHA,
                  config_sha256=CONFIG_SHA,conversion='FP32 checkpoint -> BF16 round-to-nearest-even',tensors={})
    for role,shape in [('fn',[24,20480]),('scale',[3]),('base',[24])]:
        name=f'layers.{a.layer}.hc_{a.branch}_{role}';shard=mapping[name]
        if shard not in headers:
            length=struct.unpack('<Q',byte_range(shard,0,7))[0]
            if length>16*1024*1024:raise ValueError('unexpected safetensors header')
            headers[shard]=(length+8,json.loads(byte_range(shard,8,length+7)))
        offset,header=headers[shard];entry=header[name];start,end=entry['data_offsets']
        if entry['dtype']!='F32' or entry['shape']!=shape or end-start!=4*math.prod(shape):raise ValueError(entry)
        data=b''.join(byte_range(shard,first,min(offset+end,first+256*1024)-1)
                      for first in range(offset+start,offset+end,256*1024))
        original=a.output/(role+'.fp32.bin');converted=a.output/(role+'.bf16.bin')
        values=np.frombuffer(data,dtype='<f4')
        if not np.isfinite(values).all():raise ValueError('non-finite checkpoint parameter')
        narrowed=bf16_bits(values).tobytes();write_atomic(original,data);write_atomic(converted,narrowed)
        manifest['tensors'][role]=dict(name=name,shape=shape,source_dtype='F32',dtype='BF16',
            original_file=original.name,original_sha256=sha(data),file=converted.name,sha256=sha(narrowed),
            bytes=len(narrowed),url=ROOT+shard,range_inclusive=[offset+start,offset+end-1])
        write_atomic(a.output/'provenance.json',(json.dumps(manifest,indent=2)+'\n').encode())
        print(name,shape,'BF16 bytes',len(narrowed),flush=True)


if __name__=='__main__':main()
