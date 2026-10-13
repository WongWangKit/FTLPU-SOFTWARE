#!/usr/bin/env python3
"""Download an exact checkpoint layer and selected leading embedding rows.

The subset remains safetensors-readable. Provenance records original shapes,
byte ranges, hashes and the immutable upstream revision; no weights are made up.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--revision', default='main')
    p.add_argument('--layer-prefix', required=True)
    p.add_argument('--embedding', required=True)
    p.add_argument('--embedding-rows', type=int, default=32)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    session = requests.Session()
    def get(url, **kwargs):
        r = session.get(url, timeout=180, **kwargs)
        r.raise_for_status()
        return r
    info = get(f'https://huggingface.co/api/models/{a.repo}/revision/{a.revision}').json()
    revision = info['sha']
    root = f'https://huggingface.co/{a.repo}/resolve/{revision}/'
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config = get(root + 'config.json').content
    (a.output_dir / 'config.json').write_bytes(config)
    index = get(root + 'model.safetensors.index.json').json()['weight_map']
    names = sorted(n for n in index if n.startswith(a.layer_prefix + '.') or n == a.embedding)
    if a.embedding not in names or len(names) < 2:
        raise ValueError('requested checkpoint tensors not found')
    headers = {}
    def read_range(filename, start, end):
        # Distinct URLs avoid a CDN reusing an earlier Range response.
        r = get(root + filename + f'?range={start}-{end}', headers={'Range': f'bytes={start}-{end}'})
        if r.status_code != 206 or r.headers.get('Content-Range', '').split('/')[0] != f'bytes {start}-{end}' or len(r.content) != end-start+1:
            raise ValueError(f'invalid byte-range response: {filename}, {start}:{end}')
        return r.content
    payload = bytearray()
    subset = {}
    provenance = dict(repo=a.repo, revision=revision, layer_prefix=a.layer_prefix,
                      embedding_token_ids=list(range(a.embedding_rows)), tensors={})
    for name in names:
        filename = index[name]
        if filename not in headers:
            size = struct.unpack('<Q', read_range(filename, 0, 7))[0]
            headers[filename] = (size+8, json.loads(read_range(filename, 8, size+7)))
        base, header = headers[filename]
        meta = header[name]
        begin, end = meta['data_offsets']
        shape = list(meta['shape'])
        if name == a.embedding:
            if not 0 < a.embedding_rows <= shape[0]:
                raise ValueError('invalid embedding row count')
            end = begin + (end-begin)//shape[0]*a.embedding_rows
            shape[0] = a.embedding_rows
        data = read_range(filename, base+begin, base+end-1)
        offset = len(payload)
        payload.extend(data)
        subset[name] = dict(dtype=meta['dtype'], shape=shape, data_offsets=[offset,len(payload)])
        provenance['tensors'][name] = dict(file=filename, original_shape=meta['shape'],
            downloaded_shape=shape, range=[base+begin,base+end-1], bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest())
        print(f'{name}: {shape}, {len(data)} bytes', flush=True)
    encoded = json.dumps(subset, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    path = a.output_dir / 'model.safetensors'
    with path.open('wb') as f:
        f.write(struct.pack('<Q', len(encoded))); f.write(encoded); f.write(payload)
    provenance['subset_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output_dir / 'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
