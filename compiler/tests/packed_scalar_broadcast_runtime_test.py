"""FP32 packed scalars expanded across head channels through wide SXM."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from activation_batch_matmul_test import LogicalGraph, compile_graph, run


def grouped(value, heads, rows, cols):
    return value.reshape(heads//2,2,rows,cols).transpose(1,0,2,3).reshape(heads*rows,cols)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve()
    heads,dim=16,128
    # Independent, exactly representable low mantissa bits distinguish every
    # token/head, including head 0 and non-word-aligned heads 1..15.
    values=(np.arange(32*heads,dtype='f4').reshape(32,heads)+1)*np.float32(2**-16)+np.float32(.25)
    graph=LogicalGraph()
    scalars=graph.vector_tiles('scalars',[(values,'f32')],lambda v:v.add(0,kind='multiply',immediate=1))
    expanded=grouped(scalars.T.reshape(heads,32,1).repeat(dim,2),heads,32,dim)
    last=grouped(scalars[-1:].reshape(heads,1,1).repeat(128,1).repeat(dim,2),heads,128,dim)
    outputs=[graph.vector_tiles(name,[(value,'f32')],lambda v:v.add(0,kind='multiply',immediate=1))
             for name,value in [('per_token',expanded),('last_token',last)]]
    # A second row of the same producer needs a distinct retained table.
    # A one-row producer covers the pre-existing decode broadcast path.
    decode=graph.vector_tiles('decode_scalar',[(values[1:2],'f32')],lambda v:v.add(0,kind='multiply',immediate=1))
    for name,source in [('first_row',scalars[:1]),('decode',decode)]:
        view=source.reshape(heads,1,1).repeat(2,1).repeat(32,2).reshape(heads*2,32)
        outputs.append(graph.vector_tiles(name,[(view,'f32')],lambda v:v.add(0,kind='multiply',immediate=1)))
    # Exercise the other FP32 ingress port and retain the last-row table over
    # intervening consumers. Powers of two preserve an exact FP32 oracle.
    state=np.exp2((np.arange(heads*128*dim).reshape(heads*128,dim)%5)-2).astype('f4')
    outputs.append(graph.vector_tiles('state_decay',[(state,'f32'),(last,'f32')],
                                     lambda v:v.add(0,1,kind='multiply')))
    exe,env,binary=compile_graph(graph,outputs,out,a.build_dir.resolve(),a.target_config.resolve(),fp32_spills=True)
    run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
    expected=[grouped(values.T[:,:,None].repeat(dim,2),heads,32,dim),
              grouped(values[-1:].reshape(heads,1,1).repeat(128,1).repeat(dim,2),heads,128,dim)]
    expected.extend(row.reshape(heads,1,1).repeat(2,1).repeat(32,2).reshape(heads*2,32)
                    for row in (values[:1],values[1:2]))
    expected.append(state*expected[1])
    for i,reference in enumerate(expected):
        actual=np.fromfile(out/f'actual.{i}.bin','<f4').reshape(reference.shape)
        np.testing.assert_array_equal(actual,reference)
    schedule=(out/'decoder_layer.schedule.mlir').read_text()
    # Two State tiles must share the last-row table, while row 0 and the decode
    # producer each have their own. Also check the affine even/odd head reads.
    tables=schedule.count('name = "wide_scalar_broadcast"')
    assert tables==3,tables
    for base in (0,1):
        assert re.search(r'broadcast_head_base = '+str(base)+r' : i64, broadcast_head_stride = 2 : i64, broadcast_row_repeat = 128',schedule)
    (out/'summary.json').write_text(json.dumps(dict(status='passed',heads=heads,dim=dim,
        fp32_bit_exact=True,compact_tables=tables,icu_compression='raw-macro',
        checks=['per-token/head expansion','shared last-token State table',
                'distinct source-row tables','decode compact broadcast',
                'FP32 binary State decay with retained table']),indent=2))
    print('FP32 packed scalar broadcast passed (bit exact)')


if __name__=='__main__':main()
