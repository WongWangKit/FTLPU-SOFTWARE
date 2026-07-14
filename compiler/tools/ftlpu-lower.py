#!/usr/bin/env python3
"""Prototype FTLPU lowering passes for the first matmul path.

This is intentionally textual and narrow. It gives the project concrete IR
files for the planned layers before we wire these passes into MLIR proper.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


MATMUL_RE = re.compile(
    r"stablehlo\.dot_general.*?"
    r":\s*\(tensor<(?P<m>\d+)x(?P<k>\d+)x(?P<lhs_type>[a-z0-9]+)>,\s*"
    r"tensor<(?P<k2>\d+)x(?P<n>\d+)x(?P<rhs_type>[a-z0-9]+)>\)\s*"
    r"->\s*tensor<(?P<m2>\d+)x(?P<n2>\d+)x(?P<acc_type>[a-z0-9]+)>",
    re.DOTALL,
)


@dataclass(frozen=True)
class Matmul:
    m: int
    n: int
    k: int
    lhs_type: str
    rhs_type: str
    acc_type: str


def parse_stablehlo_matmul(text):
    match = MATMUL_RE.search(text)
    if not match:
        raise ValueError("expected one stablehlo.dot_general matmul")

    m = int(match.group("m"))
    k = int(match.group("k"))
    k2 = int(match.group("k2"))
    n = int(match.group("n"))
    m2 = int(match.group("m2"))
    n2 = int(match.group("n2"))
    if (m, k, n) != (m2, k2, n2):
        raise ValueError(
            f"inconsistent matmul shapes: lhs={m}x{k}, rhs={k2}x{n}, out={m2}x{n2}"
        )
    return Matmul(
        m=m,
        n=n,
        k=k,
        lhs_type=match.group("lhs_type"),
        rhs_type=match.group("rhs_type"),
        acc_type=match.group("acc_type"),
    )


def emit_tensor_ir(matmul):
    return f"""// FTLPU tensor IR lowered from StableHLO.
module {{
  ftlpu.tensor.func @main {{
    %c = ftlpu.tensor.matmul %a, %b {{
      m = {matmul.m},
      n = {matmul.n},
      k = {matmul.k},
      lhs_layout = "row_major",
      rhs_layout = "row_major",
      out_layout = "row_major",
      lhs_dtype = "{matmul.lhs_type}",
      rhs_dtype = "{matmul.rhs_type}",
      acc_dtype = "{matmul.acc_type}"
    }} : (tensor<{matmul.m}x{matmul.k}x{matmul.lhs_type}>, tensor<{matmul.k}x{matmul.n}x{matmul.rhs_type}>) -> tensor<{matmul.m}x{matmul.n}x{matmul.acc_type}>
  }}
}}
"""


def emit_stream_ir(matmul, tile):
    lines = [
        "// FTLPU stream IR lowered from ftlpu.tensor.",
        "// This layer binds each long logical stream to source, sink, and stream register ids.",
        "module {",
        "  ftlpu.stream.func @main {",
        (
            f"    ftlpu.stream.matmul_grid @matmul0 {{m = {matmul.m}, n = {matmul.n}, "
            f"k = {matmul.k}, vector_lanes = 16, south_to_north_tiles = {tile}, mxm_count = 2}}"
        ),
        (
            "    ftlpu.stream.channel @matmul0_lhs "
            f"{{stream_ids = [0..15], sregs = [0..11], source = \"MEM:A0\", "
            f"sink = \"MXM*:lhs\", start_addr = 0, bytes = {matmul.m * matmul.k}, "
            "vector = \"south_to_north\"}"
        ),
        (
            "    ftlpu.stream.channel @matmul0_rhs "
            f"{{stream_ids = [32..47], sregs = [0..11], source = \"MEM:B0\", "
            f"sink = \"MXM*:rhs\", start_addr = {matmul.m * matmul.k}, "
            f"bytes = {matmul.k * matmul.n}, vector = \"south_to_north\"}}"
        ),
        (
            "    ftlpu.stream.channel @matmul0_out "
            f"{{stream_ids = [48..63], sregs = [0..11], source = \"MXM*:output\", "
            f"sink = \"MEM:C0\", start_addr = {matmul.m * matmul.k + matmul.k * matmul.n}, "
            f"bytes = {matmul.m * matmul.n * 4}, vector = \"south_to_north\"}}"
        ),
    ]
    lines.extend(["  }", "}", ""])
    return "\n".join(lines)


def emit_schedule_ir(matmul, tile):
    lines = [
        "// FTLPU schedule IR lowered from ftlpu.stream.",
        "// This is a stage-level queue/timeline layer before .ftlpu emission.",
        "module {",
        "  ftlpu.schedule.program @main {",
        (
            "    ftlpu.schedule.mem_read_weight @matmul0_read_weight "
            f"{{cycle = 0, source = @B0, streams = [0..15], "
            f"sregs = [0..11], bytes = {matmul.k * matmul.n}, "
            f"vector = \"south_to_north\", south_to_north_tiles = {tile}}}"
        ),
        (
            "    ftlpu.schedule.mxm_load @matmul0_load "
            "{cycle = 8, mxms = [0, 1], weight_streams = [0..15]}"
        ),
        (
            "    ftlpu.schedule.mem_read_activation @matmul0_read_activation "
            f"{{cycle = 12, source = @A0, streams = [16..31], "
            f"sregs = [0..11], bytes = {matmul.m * matmul.k}, "
            f"vector = \"south_to_north\", south_to_north_tiles = {tile}}}"
        ),
        (
            "    ftlpu.schedule.mxm_compute @matmul0_compute "
            f"{{cycle = 16, mxms = [0, 1], m = {matmul.m}, n = {matmul.n}, k = {matmul.k}, "
            f"activation_streams = [16..31], output_streams = [48..63], vector_lanes = 16, "
            f"south_to_north_tiles = {tile}, accumulate = true}}"
        ),
        (
            "    ftlpu.schedule.mem_write @matmul0_write "
            f"{{cycle = 32, dest = @C0, streams = [48..63], sregs = [0..11], "
            f"bytes = {matmul.m * matmul.n * 4}, vector = \"south_to_north\", "
            f"south_to_north_tiles = {tile}}}"
        ),
    ]
    lines.extend(["  }", "}", ""])
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lower-to", required=True, choices=["tensor", "stream", "schedule"]
    )
    parser.add_argument("--tile", type=int, default=20)
    args = parser.parse_args(argv)

    text = args.input.read_text(encoding="utf-8")
    matmul = parse_stablehlo_matmul(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.lower_to == "tensor":
        output = emit_tensor_ir(matmul)
    elif args.lower_to == "stream":
        output = emit_stream_ir(matmul, args.tile)
    else:
        output = emit_schedule_ir(matmul, args.tile)
    args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
