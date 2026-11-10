"""Mathematical attention graph entry; all tiling/lowering belongs to C++."""
import math


def flash_attention(graph, name, query, key, value, *, causal=False, query_position=0):
    shape = (*query.shape[:3], value.shape[3])
    q, k, v, o = (graph.type(s, 'bf16') for s in (query.shape, key.shape, value.shape, shape))
    source = f'''module {{ func.func @attention(%q: {q}, %k: {k}, %v: {v}) -> {o} {{
      %o = ftlpu.kernel.flash_attention %q, %k, %v {{scale = {1/math.sqrt(query.shape[3]):.12e} : f32,
        causal = {str(causal).lower()}, query_position = {query_position} : i64}} : ({q}, {k}, {v}) -> {o}
      return %o : {o}
    }} }}'''
    return graph.run(name, source, [(query, 'bf16'), (key, 'bf16'), (value, 'bf16')], [(shape, 'bf16')])[0]
