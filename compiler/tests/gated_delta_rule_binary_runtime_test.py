"""Exercise native recurrent ICU domains with nonzero carried FP32 state."""
import argparse
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def mixed_sources(a, tokens=2, heads=2):
    """Packed controls from both hemispheres feed one retained state home."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    from qwen3_5_logical_graph import LogicalGraph
    out = a.output_dir / f'mixed_sources_t{tokens}_h{heads}'
    out.mkdir(exist_ok=True)
    graph = LogicalGraph()
    beta = graph.vector_tiles('beta', [(np.full((tokens, heads), .2, np.float32), 'f32')],
                              lambda v: v.add(0, immediate=1))
    decay = graph.vector_tiles('decay', [(np.full((tokens, heads), .95, np.float32), 'f32')],
                               lambda v: v.add(0, immediate=1))
    q = np.full((tokens, heads * 32), .125, np.float32)
    value = np.full((tokens, heads * 32), .25, np.float32)
    config = SimpleNamespace(num_value_heads=heads, key_head_dim=32, value_head_dim=32)
    output, state = graph.delta(q, q, value, decay, beta,
                                np.zeros((heads * 32, 32), np.float32), config)
    source = graph.write(out, [output, state])
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    subprocess.run([str(a.graph_compile), '--input', str(source), '--output', str(kernel),
                    '--emit-kernel'], check=True)
    subprocess.run([str(a.compile), '--input', str(kernel), '--input-stage', 'kernel',
                    '--output', str(binary)], check=True)
    subprocess.run([str(a.runner), str(binary), str(out / 'inputs.txt'),
                    str(out / 'actual'), str(out / 'runtime.pipeline.csv')], check=True)
    actual = (np.fromfile(out / 'actual.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32)
    carried = np.float32(0)
    outputs = []
    for _ in range(tokens):
        carried *= np.float32(.95)
        carried += np.float32(.125) * np.float32(.2) * (np.float32(.25) - 4 * carried)
        outputs.append(4 * carried)
    expected = np.repeat(np.array(outputs, np.float32), heads * 32)
    bits = expected.view(np.uint32)
    expected = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)
    # Output is BF16, while carried state must retain FP32 precision.
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(np.fromfile(out / 'actual.1.bin', '<f4'),
                               carried, atol=1e-7, rtol=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'runtime-test', 'input', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--graph-compile', type=Path)
    p.add_argument('--runner', type=Path)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    template = a.input.read_text()
    for tokens, queue in ((2, 0), (2, 8), (32, 0)):
        source = template.replace('32x64', f'{tokens}x64').replace('32x2x', f'{tokens}x2x')
        signature = f'-> (tensor<{tokens}x64xbf16>, tensor<64x32xf32>) {{'
        assert source.count(signature) == 1
        source = source.replace(signature, signature[:-1] +
                                f'attributes {{ftlpu.vxm_queue_base = {queue} : i64}} {{')
        path = a.output_dir / f'delta_seq{tokens}_q{queue}.kernel.mlir'
        binary = path.with_suffix('.ftlpu')
        path.write_text(source)
        subprocess.run([str(a.compile), '--input', str(path), '--input-stage', 'kernel',
                        '--output', str(binary)], check=True)
        subprocess.run([str(a.runtime_test), str(binary), str(tokens)], check=True)
    if a.graph_compile and a.runner:
        mixed_sources(a)
        mixed_sources(a, tokens=4, heads=16)


if __name__ == '__main__':
    main()
