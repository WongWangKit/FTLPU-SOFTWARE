"""Chunkwise compiler bring-up; Python provides fixtures/references only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16


def recurrent(q, k, v, a, b, state):
    state = state.copy()
    result = []
    for i in range(len(q)):
        state *= a[i, 0]
        delta = b[i, 0] * (v[i] - k[i] @ state)
        state += np.outer(k[i], delta)
        result.append(q[i] @ state)
    return np.asarray(result), state


def chunkwise(q, k, v, a, b, state, chunk=32, low_precision=False):
    state = state.copy()
    def mm(lhs, rhs):
        return bf16(lhs) @ bf16(rhs) if low_precision else lhs @ rhs
    outputs = []
    for start in range(0, len(q), chunk):
        qs, ks, vs = q[start:start+chunk], k[start:start+chunk], v[start:start+chunk]
        alpha, beta = a[start:start+chunk, 0], b[start:start+chunk, 0]
        gamma = np.cumprod(alpha)
        decay = np.zeros((chunk, chunk), dtype=q.dtype)
        for i in range(chunk):
            decay[i, i] = 1
            if i: decay[i, :i] = decay[i-1, :i] * alpha[i]
        lower = np.tril(beta[:, None] * mm(ks, ks.T) * decay, -1)
        rhs = beta[:, None] * (vs - gamma[:, None] * mm(ks, state))
        d = np.linalg.solve(np.eye(chunk, dtype=q.dtype) + lower, rhs)
        outputs.append(gamma[:, None] * mm(qs, state) + mm(mm(qs, ks.T) * decay, d))
        state = gamma[-1] * state + mm(ks.T, decay[-1, :, None] * d)
    return np.concatenate(outputs), state


def lowered_chunkwise(q, k, v, a, b, state, chunk=32):
    """Mirror the C++ graph's decay rows and forward elimination, not solve()."""
    state = state.copy()
    outputs = []
    for start in range(0, len(q), chunk):
        qs, ks, vs = q[start:start+chunk], k[start:start+chunk], v[start:start+chunk]
        alpha, beta = a[start:start+chunk], b[start:start+chunk]
        row = np.zeros(64, dtype=q.dtype)
        row[0] = 1
        rows = []
        for i in range(chunk):
            row = row * alpha[i, 0]
            row[i+1] += 1
            rows.append(row.copy())
        rows = np.asarray(rows)
        gamma, decay = rows[:, :1], rows[:, 1:chunk+1]
        lower = (ks @ ks.T) * decay * beta
        rhs = (vs - (ks @ state) * gamma) * beta
        solved = []
        for i in range(chunk):
            di = rhs[:1].copy()
            solved.append(di)
            rhs = rhs[1:] - di * lower[i+1:, i:i+1]
        d = np.concatenate(solved)
        outputs.append(((qs @ ks.T) * decay) @ d + (qs @ state) * gamma)
        state = state * gamma[-1, 0] + ks.T @ (d * decay[-1, :, None])
    return np.concatenate(outputs), state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'graph-compile', 'runner', 'output-dir', 'target-config'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--tokens', type=int, choices=(32,64), default=64)
    p.add_argument('--head-dim', type=int, choices=(32,128), default=32)
    p.add_argument('--reference-only', action='store_true')
    p.add_argument('--compile-only', action='store_true')
    a = p.parse_args()
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2364)
    dim = a.head_dim
    q = rng.normal(size=(a.tokens,dim)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    q = bf16(q / np.linalg.norm(q,axis=-1,keepdims=True) / np.sqrt(dim))
    k = bf16(k / np.linalg.norm(k,axis=-1,keepdims=True))
    v = bf16(rng.normal(0,.2,size=q.shape).astype(np.float32))
    decay = rng.uniform(.6,.99,size=(a.tokens,1)).astype(np.float32)
    beta = rng.uniform(.05,.95,size=decay.shape).astype(np.float32)
    state = rng.normal(0,.02,size=(dim,dim)).astype(np.float32)
    reference = recurrent(q,k,v,decay,beta,state)
    precision = {}
    for mode in (False,True):
        estimate = chunkwise(q,k,v,decay,beta,state,low_precision=mode)
        precision['bf16_matmul_inputs' if mode else 'fp32'] = {
            name: float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
            for name,x,y in zip(('output_nrmse','state_nrmse'),estimate,reference)}
    arrays64 = [x.astype(np.float64) for x in (q,k,v,decay,beta,state)]
    for lhs,rhs in zip(recurrent(*arrays64),chunkwise(*arrays64)):
        np.testing.assert_allclose(lhs,rhs,atol=1e-12,rtol=1e-12)
    for lhs,rhs in zip(recurrent(*arrays64),lowered_chunkwise(*arrays64)):
        np.testing.assert_allclose(lhs,rhs,atol=1e-12,rtol=1e-12)
    lowering_errors = {
        name: float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
        for name,x,y in zip(('output_nrmse','state_nrmse'),
                           lowered_chunkwise(q,k,v,decay,beta,state),reference)}
    assert max(lowering_errors.values()) < 1e-5
    # Strong decay/complete reset must not require inverse decay or 0/0.
    arrays64[3][7]=0
    for lhs,rhs in zip(recurrent(*arrays64),chunkwise(*arrays64)):
        np.testing.assert_allclose(lhs,rhs,atol=1e-12,rtol=1e-12)
    for lhs,rhs in zip(recurrent(*arrays64),lowered_chunkwise(*arrays64)):
        np.testing.assert_allclose(lhs,rhs,atol=1e-12,rtol=1e-12)
    assert max(precision['fp32'].values()) < 1e-5
    (out/'reference_summary.json').write_text(json.dumps(dict(
        status='passed',backend='NumPy mathematical reference; not CModel',
        tokens=a.tokens,chunk_size=32,head_dim=dim,nonzero_initial_state=True,
        zero_decay_checked=True,precision_estimates=precision,
        forward_elimination_checked=True,lowering_fp32_errors=lowering_errors),indent=2))
    if a.reference_only:
        print('chunk32 reference passed: nonzero state, causal outputs, reset gate')
        return
    graph = LogicalGraph()
    config = SimpleNamespace(num_value_heads=1,key_head_dim=dim,value_head_dim=dim)
    outputs = graph.delta(q,k,v,decay,beta,state,config)
    node = graph.nodes[-1]
    node['source'] = node['source'].replace('heads = ', 'chunk_size = 32 : i64, heads = ', 1)
    source = graph.write(out / 'logical_graph', outputs)
    kernel, binary = out/'kernel.mlir', out/'program.ftlpu'
    common = ['--target-config', str(a.target_config.resolve())]
    subprocess.run([str(a.graph_compile.resolve()),'--input',str(source),'--output',str(kernel),
                    '--emit-kernel',*common],check=True)
    subprocess.run([str(a.compile.resolve()),'--input',str(kernel),'--input-stage','kernel',
                    '--output',str(binary),*common],check=True)
    if a.compile_only:
        print('connected kernel -> schedule -> command compilation passed')
        return
    with (out/'execution.log').open('w') as log:
        subprocess.run([str(a.runner.resolve()),str(binary),str(out/'inputs.txt'),
                        str(out/'actual'),str(out/'runtime.pipeline.csv')],stdout=log,stderr=subprocess.STDOUT,check=True)
    actual = [(np.fromfile(out/'actual.0.bin','<u2').astype(np.uint32)<<16).view(np.float32).reshape(q.shape),
              np.fromfile(out/'actual.1.bin','<f4').reshape(state.shape)]
    # The native contract is BF16 ingress/storage with FP32 accumulation.
    # Keep error against original FP32 gates/state visible independently;
    # do not attribute ingress quantization to the recurrence implementation.
    full_precision_errors = {
        name: float(np.linalg.norm(got-expected)/max(np.linalg.norm(expected),1e-30))
        for name,got,expected in zip(('output','state'),actual,reference)}
    arithmetic_reference = reference
    if dim == 128:
        truncate = lambda x: (x.view(np.uint32) & np.uint32(0xffff0000)).view(np.float32)
        arithmetic_reference = recurrent(q,k,v,truncate(decay),truncate(beta),truncate(state))
    errors = {}
    for name,got,expected in zip(('output','state'),actual,arithmetic_reference):
        error = float(np.linalg.norm(got-expected)/max(np.linalg.norm(expected),1e-30))
        errors[name] = {'nrmse':error,'max_abs':float(np.max(np.abs(got-expected)))}
    threshold = .01 if dim == 128 else .003
    summary = dict(status='passed' if all(e['nrmse']<threshold for e in errors.values()) else 'failed',
                   chunk_size=32,tokens=a.tokens,heads=1,key_dim=dim,value_dim=dim,
                   nonzero_initial_state=True,backend='native chunkwise BF16 storage / FP32 accumulation' if dim == 128 else 'FP32 VXM chunkwise baseline',
                   threshold_nrmse=threshold,
                   reference='independent FP32 token recurrence with the declared ingress precision',
                   ingress_conversion='FP32 upper16 truncation' if dim == 128 else 'none',
                   original_fp32_reference_nrmse=full_precision_errors,
                   scope='GDN recurrence through original whole-graph lowering; excludes surrounding decoder operators',
                   host_intermediate_transfers=0,errors=errors)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    assert summary['status']=='passed'


if __name__ == '__main__':
    main()
