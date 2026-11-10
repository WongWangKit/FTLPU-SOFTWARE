"""Extract the byte-exact Q/KV projection and norm tensors from the pinned checkpoint."""
import argparse
import hashlib
import json
import math
import struct
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
ROOT = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
INDEX_SHA = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
CONFIG_SHA = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
MANIFEST_SHA = "613aef417df28c31806880e06d183749bd6574861c4382d878b648d0605b4c73"
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT / "results/deepseek_v41_flash_layer0_qkv_prefill/weights"
CHUNK = 4 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(filename, limit, span=None):
    url = ROOT + filename
    headers = {"Accept-Encoding": "identity"}
    if span is not None:
        start, end = span
        url += f"?range={start}-{end}"
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers=headers), timeout=75) as response:
                if span is not None:
                    content_range = response.headers.get("Content-Range", "").split("/")[0]
                    if response.status != 206 or content_range != f"bytes {start}-{end}":
                        raise ValueError(f"Server did not honor requested range for {filename}")
                data = response.read(limit + 1)
                if len(data) > limit or (span is not None and len(data) != limit):
                    raise ValueError(f"Unexpected response size for {filename}")
                return data
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def byte_range(filename, start, end):
    return fetch(filename, end - start + 1, (start, end))


def write_atomic(path, payload):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as out:
        temp = Path(out.name)
    try:
        temp.write_bytes(payload)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def pinned_config():
    config_path = PROJECT / "results/deepseek_v41_flash_assessment/official_config.json"
    config = config_path.read_bytes() if config_path.is_file() else fetch("config.json", 1024 * 1024)
    if sha(config) != CONFIG_SHA:
        raise ValueError("Publisher config does not match the pinned revision")
    if not config_path.is_file():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(config_path, config)
    return config


def checkpoint_layout(cached_index, config):
    index_name = "model.safetensors.index.json"
    index = cached_index.read_bytes() if cached_index else fetch(index_name, 16 * 1024 * 1024)
    if sha(index) != INDEX_SHA:
        raise ValueError("Index checksum does not match the pinned revision")
    weight_map = json.loads(index)["weight_map"]
    full_config = json.loads(config)
    text_config = full_config["text_config"]
    hidden, rank = text_config["hidden_size"], text_config["q_lora_rank"]
    width, heads = text_config["head_dim"], text_config["num_attention_heads"]
    blocks = full_config["quantization_config"]["weight_block_size"]
    specs = {
        "q_norm": ("q_norm.weight", [rank], "BF16"),
        "kv_norm": ("kv_norm.weight", [width], "BF16"),
    }
    for role, shape in (("wq_a", [rank, hidden]), ("wq_b", [heads * width, rank]),
                        ("wkv", [width, hidden])):
        scales = [(n + b - 1) // b for n, b in zip(shape, blocks)]
        specs[role + "_scale"] = (role + ".scale", scales, "F8_E8M0")
        specs[role] = (role + ".weight", shape, "F8_E4M3")
    manifest = {
        "model": MODEL, "revision": REVISION, "config_url": ROOT + "config.json",
        "config_sha256": CONFIG_SHA, "index_url": ROOT + index_name, "index_sha256": INDEX_SHA,
        "weight_block_size": blocks, "scale_fmt": full_config["quantization_config"]["scale_fmt"],
        "rms_norm_eps": text_config["rms_norm_eps"], "shards": {}, "tensors": {},
    }
    headers = {}
    for role, (suffix, shape, dtype) in specs.items():
        name = "layers.0.attn." + suffix
        shard = weight_map[name]
        if shard not in headers:
            size = struct.unpack("<Q", byte_range(shard, 0, 7))[0]
            if size > 16 * 1024 * 1024:
                raise ValueError("Unexpected safetensors header length")
            raw = byte_range(shard, 8, size + 7)
            headers[shard] = size + 8, json.loads(raw)
            manifest["shards"][shard] = {"url": ROOT + shard, "header_size": size,
                                          "header_sha256": sha(raw)}
        base, header = headers[shard]
        meta = header[name]
        begin, end = meta["data_offsets"]
        length = math.prod(shape) * (2 if dtype == "BF16" else 1)
        if meta["shape"] != shape or meta["dtype"] != dtype or end - begin != length:
            raise ValueError(f"Unexpected checkpoint geometry for {name}: {meta}")
        manifest["tensors"][role] = {
            "name": name, "file": role + ".bin", "dtype": dtype, "shape": shape,
            "shard": shard, "range_inclusive": [base + begin, base + end - 1],
            "bytes": length,
        }

    return manifest


def import_weights(output, cached_index=None, verify_only=False):
    config = None if verify_only else pinned_config()
    manifest_path = output / "provenance.json"
    manifest = None
    if manifest_path.is_file():
        raw = manifest_path.read_bytes().replace(b"\r\n", b"\n")
        if sha(raw) == MANIFEST_SHA:
            manifest = json.loads(raw)
    if manifest is None:
        if verify_only:
            raise ValueError("Missing or invalid pinned provenance.json")
        manifest = checkpoint_layout(cached_index, config)
    if not verify_only:
        output.mkdir(parents=True, exist_ok=True)

    def extract(record):
        destination = output / record["file"]
        expected = record.get("sha256")
        if expected and destination.is_file() and destination.stat().st_size == record["bytes"]:
            if sha(destination.read_bytes()) == expected:
                print(record["name"], "verified", flush=True)
                return
        if verify_only:
            raise ValueError(f"Failed offline verification of {destination}")
        start, last = record["range_inclusive"]
        payload = bytearray()
        for first in range(start, last + 1, CHUNK):
            payload.extend(byte_range(record["shard"], first, min(last, first + CHUNK - 1)))
        digest = sha(payload)
        if expected and digest != expected:
            raise ValueError(f"Range checksum mismatch for {record['name']}")
        write_atomic(destination, payload)
        record["sha256"] = digest
        print(record["name"], record["dtype"], record["shape"], digest, flush=True)

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(extract, manifest["tensors"].values()))
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    if sha(encoded) != MANIFEST_SHA:
        raise ValueError("Computed layout or payload checksums differ from the pinned subset")
    if not verify_only and (not manifest_path.is_file() or manifest_path.read_bytes() != encoded):
        write_atomic(manifest_path, encoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cached-index", type=Path, help="Reuse a SHA-256 verified local index")
    parser.add_argument("--verify-only", action="store_true", help="Check all local bytes offline")
    args = parser.parse_args()
    import_weights(args.output, args.cached_index, args.verify_only)
