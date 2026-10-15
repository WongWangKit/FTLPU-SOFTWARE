#!/usr/bin/env python3
"""Download unchanged official layer tensors with pinned revision and hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

OFFICIAL_REPOS = {
    'llama3_2_1b': 'meta-llama/Llama-3.2-1B',
    'ministral3_3b': 'mistralai/Ministral-3-3B-Instruct-2512-BF16',
    'phi4_mini': 'microsoft/Phi-4-mini-instruct',
    'smollm3_3b': 'HuggingFaceTB/SmolLM3-3B',
    'glm_edge_1_5b': 'zai-org/glm-edge-1.5b-chat',
    'minicpm4_0_5b': 'openbmb/MiniCPM4-0.5B',
}
# Kept separate so the dense acceptance suite does not accidentally include
# the different Gated DeltaNet execution backend.
EXTRA_OFFICIAL_REPOS = {'qwen3_5_0_8b': 'Qwen/Qwen3.5-0.8B'}

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def download(model, output, revision='main', layer=0, embedding_rows=32, headers_only=False):
    import requests
    repo = (OFFICIAL_REPOS | EXTRA_OFFICIAL_REPOS)[model]
    session = requests.Session()
    token = os.environ.get('HF_TOKEN')
    token_file = Path.home() / '.cache/huggingface/token'
    if not token and token_file.is_file():
        token = token_file.read_text().strip()
    if token:
        session.headers['Authorization'] = 'Bearer ' + token
    session.headers['Accept-Encoding'] = 'identity'
    def get(url, **kwargs):
        for attempt in range(3):
            response = session.get(url, timeout=(20, 180), **kwargs)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1 + attempt)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f'{repo}: HTTP {response.status_code} '
                    + response.headers.get('X-Error-Code', ''))
            return response
    info = get(f'https://huggingface.co/api/models/{repo}/revision/{revision}').json()
    revision = info['sha']
    repo = info['id']
    root = f'https://huggingface.co/{repo}/resolve/{revision}/'
    config_bytes = get(root + 'config.json').content
    config = json.loads(config_bytes)
    prefix = f'language_model.model.layers.{layer}' if config.get('model_type') == 'mistral3' else f'model.layers.{layer}'
    embedding = 'language_model.model.embed_tokens.weight' if config.get('model_type') == 'mistral3' else 'model.embed_tokens.weight'
    if model == 'qwen3_5_0_8b':
        prefix = f'model.language_model.layers.{layer}'
        embedding = 'model.language_model.embed_tokens.weight'
    files = {f['rfilename'] for f in info['siblings']}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_bytes(config_bytes)
    layouts = {}
    def read_range(filename, begin, end):
        response = get(root + filename + f'?range={begin}-{end}',
            headers={'Range': f'bytes={begin}-{end}'})
        wanted = f'bytes {begin}-{end}'
        if response.status_code != 206 or response.headers.get('Content-Range', '').split('/')[0] != wanted or len(response.content) != end - begin + 1:
            raise RuntimeError(f'{repo}: invalid range response for {filename} [{begin},{end}]')
        return response.content
    def layout(filename):
        if filename not in layouts:
            size = struct.unpack('<Q', read_range(filename, 0, 7))[0]
            if size > 16 * 1024 * 1024:
                raise ValueError('unexpected safetensors header length')
            layouts[filename] = (size + 8, json.loads(read_range(filename, 8, size + 7)))
        return layouts[filename]
    if 'model.safetensors.index.json' in files:
        index_bytes = get(root + 'model.safetensors.index.json').content
        (output / 'upstream.index.json').write_bytes(index_bytes)
        weight_map = json.loads(index_bytes)['weight_map']
    elif 'model.safetensors' in files:
        _, header = layout('model.safetensors')
        weight_map = {name: 'model.safetensors' for name in header if name != '__metadata__'}
    else:
        raise ValueError(f'{repo}: no supported original HF safetensors file')
    names = sorted(n for n in weight_map if n.startswith(prefix + '.') or n == embedding)
    if embedding not in names or len(names) < 2:
        raise ValueError(f'{repo}: missing requested layer or embedding')
    provenance = dict(repo=repo, revision=revision, layer=layer, layer_prefix=prefix,
        source='official_publisher_huggingface_repository',
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        embedding_token_ids=list(range(embedding_rows)), tensors={})
    cache = output / '.tensor_cache'
    cache.mkdir(exist_ok=True)
    payloads = []
    subset = {}
    offset = 0
    for name in names:
        filename = weight_map[name]
        base, header = layout(filename)
        meta = header[name]
        begin, end = meta['data_offsets']
        shape = list(meta['shape'])
        if name == embedding:
            if not 0 < embedding_rows <= shape[0]:
                raise ValueError('invalid embedding rows')
            end = begin + (end - begin) // shape[0] * embedding_rows
            shape[0] = embedding_rows
        begin, end = base + begin, base + end
        record = dict(file=filename, dtype=meta['dtype'], original_shape=meta['shape'],
            downloaded_shape=shape, range=[begin, end - 1], bytes=end - begin)
        provenance['tensors'][name] = record
        print(name, meta['dtype'], shape, end - begin, flush=True)
        if headers_only:
            continue
        key = hashlib.sha256(f'{revision}/{filename}/{begin}/{end}'.encode()).hexdigest()
        path = cache / (key + '.bin')
        if not path.is_file() or path.stat().st_size != end - begin:
            partial = path.with_suffix('.part')
            with partial.open('wb') as stream:
                for cursor in range(begin, end, 4 * 1024 * 1024):
                    stream.write(read_range(filename, cursor, min(end, cursor + 4 * 1024 * 1024) - 1))
            partial.replace(path)
        record['sha256'] = sha256(path)
        payloads.append(path)
        subset[name] = dict(dtype=meta['dtype'], shape=shape, data_offsets=[offset, offset + end - begin])
        offset += end - begin
    if headers_only:
        (output / 'tensor_layout.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        return provenance
    encoded = json.dumps(subset, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    path = output / 'model.safetensors'
    with path.with_suffix('.part').open('wb') as stream:
        stream.write(struct.pack('<Q', len(encoded)))
        stream.write(encoded)
        for payload in payloads:
            with payload.open('rb') as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                    stream.write(block)
    path.with_suffix('.part').replace(path)
    provenance['subset_sha256'] = sha256(path)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return provenance

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=OFFICIAL_REPOS | EXTRA_OFFICIAL_REPOS, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--revision', default='main')
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--embedding-rows', type=int, default=32)
    p.add_argument('--headers-only', action='store_true')
    a = p.parse_args()
    download(a.model, a.output_dir, a.revision, a.layer, a.embedding_rows, a.headers_only)

if __name__ == '__main__':
    main()
