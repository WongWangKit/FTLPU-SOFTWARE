#!/usr/bin/env python3
"""Prototype FTLPU lowering passes for the first matmul path.

This is intentionally textual and narrow. It gives the project concrete IR
files for the planned layers before we wire these passes into MLIR proper.
"""

import argparse
import math
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
    m_tiles = math.ceil(matmul.m / tile)
    n_tiles = math.ceil(matmul.n / tile)
    k_tiles = math.ceil(matmul.k / tile)
    lines = [
        "// FTLPU stream IR lowered from ftlpu.tensor.",
        "module {",
        "  ftlpu.stream.func @main {",
        "    %a = ftlpu.stream.memref @A {addr = 0, layout = \"row_major\"}",
        f"    %b = ftlpu.stream.memref @B {{addr = {matmul.m * matmul.k}, layout = \"row_major\"}}",
        f"    %c = ftlpu.stream.memref @C {{addr = {matmul.m * matmul.k + matmul.k * matmul.n}, layout = \"row_major\"}}",
        (
            f"    ftlpu.stream.matmul_grid %a, %b, %c {{m = {matmul.m}, n = {matmul.n}, "
            f"k = {matmul.k}, tile_m = {tile}, tile_n = {tile}, tile_k = {tile}, "
            f"m_tiles = {m_tiles}, n_tiles = {n_tiles}, k_tiles = {k_tiles}}}"
        ),
    ]
    for mt in range(m_tiles):
        for nt in range(n_tiles):
            for kt in range(k_tiles):
                lines.append(
                    (
                        "    ftlpu.stream.matmul_tile "
                        f"@m{mt}_n{nt}_k{kt} {{m_tile = {mt}, n_tile = {nt}, k_tile = {kt}, "
                        f"mxm = {((mt + nt) % 2)}, lhs_stream = {(mt * k_tiles + kt) % 64}, "
                        f"rhs_stream = {(nt * k_tiles + kt + 32) % 64}, out_stream = {(mt * n_tiles + nt) % 64}}}"
                    )
                )
    lines.extend(["  }", "}", ""])
    return "\n".join(lines)


def emit_schedule_ir(matmul, tile):
    m_tiles = math.ceil(matmul.m / tile)
    n_tiles = math.ceil(matmul.n / tile)
    k_tiles = math.ceil(matmul.k / tile)
    cycle = 0
    lines = [
        "// FTLPU schedule IR lowered from ftlpu.stream.",
        "// This is the low-level queue/timeline layer before .ftlpu emission.",
        "module {",
        "  ftlpu.schedule.program @main {",
    ]
    for mt in range(m_tiles):
        for nt in range(n_tiles):
            for kt in range(k_tiles):
                mxm = (mt + nt) % 2
                lhs_addr = (mt * tile * matmul.k) + (kt * tile)
                rhs_addr = (kt * tile * matmul.n) + (nt * tile)
                out_addr = (matmul.m * matmul.k + matmul.k * matmul.n) + (
                    mt * tile * matmul.n
                ) + (nt * tile)
                lines.extend(
                    [
                        f"    ftlpu.schedule.mem_read @mem0 cycle {cycle} addr {lhs_addr} bytes {tile * tile} stream 0",
                        f"    ftlpu.schedule.mem_read @mem1 cycle {cycle + 1} addr {rhs_addr} bytes {tile * tile} stream 32",
                        f"    ftlpu.schedule.mxm_load @mxm{mxm} cycle {cycle + 8} lhs_stream 0 rhs_stream 32",
                        f"    ftlpu.schedule.mxm_compute @mxm{mxm} cycle {cycle + 16} m {tile} n {tile} k {tile} accumulate {str(kt != 0).lower()}",
                    ]
                )
                if kt == k_tiles - 1:
                    lines.append(
                        f"    ftlpu.schedule.mxm_output @mxm{mxm} cycle {cycle + 28} stream 48"
                    )
                    lines.append(
                        f"    ftlpu.schedule.mem_write @mem2 cycle {cycle + 36} addr {out_addr} bytes {tile * tile * 4} stream 48"
                    )
                cycle += 48
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
