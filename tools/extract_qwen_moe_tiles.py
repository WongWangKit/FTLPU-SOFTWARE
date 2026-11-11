"""Extract the BF16 operands of the MoE cycle experiment from a pinned Qwen shard."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen

REPO = "Qwen/Qwen3-30B-A3B"
REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
ROOT = f"https://huggingface.co/{REPO}/resolve/{REV}/"


def fetch(file, first=None, count=None):
    headers = {"Accept-Encoding": "identity"}
    url = ROOT + file
    if first is not None:
        last = first + count - 1
        url += f"?range={first}-{last}"
        headers["Range"] = f"bytes={first}-{last}"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=25) as response:
                if first is None:
                    return response.read()
                if (response.status != 206 or
                        response.headers.get("Content-Range", "").split("/")[0] != f"bytes {first}-{last}"):
                    raise ValueError(f"{file}: server did not honor requested range {first}-{last}")
                data = response.read(count + 1)
                if len(data) != count:
                    raise ValueError(f"{file}: short range")
                return data
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def get_header(file):
    length = struct.unpack("<Q", fetch(file, 0, 8))[0]
    if not 0 < length < 16 * 1024 * 1024:
        raise ValueError("unexpected Safetensors header size")
    return 8 + length, json.loads(fetch(file, 8, length))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = fetch("config.json")
    config = json.loads(config_bytes)
    index = json.loads(fetch("model.safetensors.index.json"))["weight_map"]
    prefix = "model.layers.0.mlp."
    items = [("model.embed_tokens.weight", 5, 32), (prefix + "gate.weight", 4, 32)]
    items += [(f"{prefix}experts.{expert}.{part}_proj.weight", 32, 32)
              for expert in range(4) for part in ("gate", "up", "down")]
    with ThreadPoolExecutor(max_workers=6) as pool:
        files = sorted({index[name] for name, _, _ in items})
        layouts = dict(zip(files, pool.map(get_header, files)))

        def extract(item):
            name, rows, columns = item
            file = index[name]
            base, layout = layouts[file]
            meta = layout[name]
            original_rows, original_columns = meta["shape"]
            if meta["dtype"] != "BF16" or rows > original_rows or columns > original_columns:
                raise ValueError(f"invalid operand shape or dtype for {name}")
            start = base + meta["data_offsets"][0]
            count = rows * original_columns * 2
            data = fetch(file, start, count)
            cropped = b"".join(data[r * original_columns * 2:r * original_columns * 2 + columns * 2]
                               for r in range(rows))
            print(name, meta["shape"], "->", (rows, columns), flush=True)
            return cropped, dict(name=name, file=file, dtype=meta["dtype"],
                                 original_shape=meta["shape"], slice_shape=[rows, columns],
                                 fetched_range=[start, start + count - 1],
                                 sha256=sha256(cropped).hexdigest())

        operands = list(pool.map(extract, items))
    binary = b"".join(data for data, _ in operands)
    (output / "qwen_tiles.bf16").write_bytes(binary)
    (output / "checkpoint_provenance.json").write_text(json.dumps(dict(
        repo=REPO, revision=REV, layer=0, embedding_token_ids=list(range(5)),
        model_shape={key: config[key] for key in
                     ("hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_tok")},
        config_sha256=sha256(config_bytes).hexdigest(), payload_sha256=sha256(binary).hexdigest(),
        tensors=[meta for _, meta in operands]), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
