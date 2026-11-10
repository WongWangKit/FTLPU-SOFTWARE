"""Check that consecutive weight pages share only their own staged activation."""
import argparse
import json
from pathlib import Path

import numpy as np

from activation_batch_matmul_test import LogicalGraph, compile_graph, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("build-dir", "target-config", "output-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    graph = LogicalGraph()
    rng = np.random.default_rng(8402)
    sources = []
    activations = []
    for family, sign in enumerate((1, -1)):
        x = sign * rng.integers(1, 16, (2, 64)).astype(np.float32)
        source = graph.vector_tiles(
            f"source{family}", [(x, "bf16")],
            lambda v: v.add(0, immediate=1.0, dtype="bf16"))
        # Packed [2,64] is column-word major. A [1,128] projection requires
        # the middle two physical words in the opposite order.
        sources.append(x.reshape(1, 128))
        activations.append(source.reshape(1, 128))

    outputs, expected = [], []
    for family, (activation, x) in enumerate(zip(activations, sources)):
        # Each output is a sum of two small integers, exactly representable in BF16.
        weights = np.zeros((256, 128), dtype=np.int8)
        feature = rng.integers(128, size=256)
        weights[np.arange(256), feature] = 1
        weights[np.arange(256), (feature + 37) % 128] = 1
        outputs.append(graph.project(f"projection{family}", activation, weights, 1.0))
        expected.append(x @ weights.T)

    exe, env, binary = compile_graph(
        graph, outputs, out, args.build_dir.resolve(), args.target_config.resolve(),
        module_attrs=("ftlpu.weight_page_rows = 16 : i64",))
    plan = json.loads((out / "graph_plan.json").read_text())
    run([exe("runtime", "compiled_stage_file_runner"), binary, out / "inputs.txt",
         out / "actual", out / "runtime.pipeline.csv"], out / "execution.log", env)
    staging_homes = []
    for family, reference in enumerate(expected):
        words = np.fromfile(out / f"actual.{family}.bin", dtype="<u2")
        actual = (words.astype("<u4") << 16).view("<f4").reshape(reference.shape)
        np.testing.assert_array_equal(actual, reference)

        prefix = f"projection{family}.page"
        pages = [n for n in plan["nodes"] if n["name"].startswith(prefix)]
        assert len(pages) >= 2, f"{prefix} did not exercise paging"
        assert [n["projection_input_reused"] for n in pages] == [False] + [True] * (len(pages) - 1)
        transfers = [t for t in plan["linker_mem_transports"] if t["phase"].startswith(prefix)]
        assert {t["phase"] for t in transfers} == {pages[0]["name"]}
        staging_homes.append({(t["destination_side"], t["destination_slice"],
                               t["destination_bank"], t["destination_row"]) for t in transfers})
    assert staging_homes[0] == staging_homes[1], "The two families must exercise the same physical home"
    print("Both projection families matched exactly. Each staged its own activation once.")


if __name__ == "__main__":
    main()
