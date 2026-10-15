"""Qwen3.5 equations and tensor views; hardware planning belongs to C++."""
import json
from pathlib import Path
import numpy as np
from qwen3_5_compiled_layer import VectorGraph, bf16
from tensor_geometry import Value
from common_kernel_graph import rms_norm_source, tensor_type


class LogicalGraph:
    type = staticmethod(tensor_type)
    def __init__(self):
        self.nodes = []

    def run(self, name, source, inputs, outputs, **kwargs):
        node = len(self.nodes)
        self.nodes.append(dict(name=name, source=source, inputs=inputs, outputs=outputs))
        return [Value(node, i, tuple(shape), dtype, np.arange(np.prod(shape), dtype=np.int64))
                for i, (shape, dtype) in enumerate(outputs)]

    def project(self, name, value, weight, scale):
        m, k = value.shape
        n = weight.shape[0]
        a, w, o = self.type((m,k),'bf16'), self.type((k,n),'i8'), self.type((m,n),'bf16')
        source = f'''module {{ func.func @stage(%a: {a}, %w: {w}) -> {o} {{
          %o = "ftlpu.kernel.matmul"(%a, %w) {{m = {m} : i64, n = {n} : i64,
          k = {k} : i64, unit = "MXM", rhs_scale = {float(scale):.9e} : f32}} : ({a}, {w}) -> {o}
          return %o : {o}
        }} }}'''
        return self.run(name, source, [(value,'bf16'),(weight.T.copy(),'i8')], [((m,n),'bf16')])[0]

    def norm(self, name, x, gamma, epsilon, mean=True, scale=1):
        if mean and scale == 1:
            return self.run(name, rms_norm_source(x.shape,epsilon),
                [(x,'bf16'),(bf16(np.asarray(gamma).reshape(-1)),'bf16')],[(x.shape,'bf16')])[0]
        graph = VectorGraph(self,name,[(x,'bf16'),(np.asarray(gamma).reshape(1,-1),'f32')])
        return graph.finish(graph.norm(0,1,epsilon,mean,scale))

    def conv(self, qkv, weight, **kwargs):
        a, w = self.type(qkv.shape,'bf16'), self.type(weight.shape,'bf16')
        source = f'''module {{ func.func @stage(%a: {a}, %w: {w}) -> {a} {{
          %o = ftlpu.kernel.depthwise_conv1d %a, %w {{kernel_size = 4 : i64,
          causal = true, activation = "silu"}} : ({a}, {w}) -> {a}
          return %o : {a}
        }} }}'''
        return self.run('conv1d_silu',source,[(qkv,'bf16'),(weight,'bf16')],[(qkv.shape,'bf16')])[0]

    def vector_tiles(self, name, inputs, build):
        graph = VectorGraph(self,name,inputs)
        return graph.finish(build(graph))

    def delta(self, q, k, v, decay, beta, state, c):
        qt, vt, dt, st = self.type(q.shape,'bf16'), self.type(v.shape,'bf16'), self.type(decay.shape,'f32'), self.type(state.shape,'f32')
        source = f'''module {{ func.func @stage(%q: {qt}, %k: {qt}, %v: {vt},
          %d: {dt}, %b: {dt}, %s: {st}) -> ({vt}, {st}) {{
          %o, %s1 = ftlpu.kernel.gated_delta_rule %q, %k, %v, %d, %b, %s {{heads = {c.num_value_heads} : i64,
          key_dim = {c.key_head_dim} : i64, value_dim = {c.value_head_dim} : i64}} : ({qt}, {qt}, {vt}, {dt}, {dt}, {st}) -> ({vt}, {st})
          return %o, %s1 : {vt}, {st}
        }} }}'''
        return self.run('delta_rule',source,[(q,'bf16'),(k,'bf16'),(v,'bf16'),
            (decay,'f32'),(beta,'f32'),(state,'f32')],[(v.shape,'bf16'),(state.shape,'f32')])

    def write(self, directory, finals):
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        constants, identities = [], {}
        def edge(value, dtype):
            if isinstance(value,Value):
                return dict(shape=list(value.shape),dtype=dtype,segments=[
                    dict(node=int(key[0]),output=int(key[1]),indices=indices.tolist(),positions=positions.tolist())
                    for key,indices,positions in value.segments()])
            identity = (id(value),dtype)
            if identity not in identities:
                index = len(constants)
                identities[identity] = index
                path = directory/f'constant{index}.bin'
                if dtype == 'bf16':
                    (bf16(value).view(np.uint32)>>16).astype('<u2').tofile(path)
                else:
                    np.asarray(value).astype('<f4' if dtype=='f32' else 'i1').tofile(path)
                constants.append(dict(shape=list(value.shape),dtype=dtype,file=str(path.resolve())))
            return dict(constant=identities[identity],shape=list(value.shape),dtype=dtype)
        nodes = [dict(name=node['name'],kernel=node['source'],
            inputs=[edge(v,d) for v,d in node['inputs']],
            outputs=[dict(shape=list(s),dtype=d) for s,d in node['outputs']]) for node in self.nodes]
        result = dict(format='ftlpu.logical-kernel-graph.v1',nodes=nodes,
                      outputs=[edge(v,v.dtype) for v in finals],constants=constants)
        path = directory/'decoder_layer.graph.json'
        path.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8')
        return path

    def layer(self,x,weights,packed,scales,c):
        project=lambda name,value:self.project(name,value,packed[name],scales[name])
        n=self.norm('input_norm',x,bf16(1+weights.input_norm),c.rms_norm_eps)
        qkv=project('in_proj_qkv',n)
        mixed=self.conv(qkv,weights.conv1d)
        q=mixed[:,:c.key_width].reshape(-1,c.num_key_heads,c.key_head_dim).repeat(c.num_value_heads//c.num_key_heads,1).reshape(-1,c.key_head_dim)
        k=mixed[:,c.key_width:2*c.key_width].reshape(-1,c.num_key_heads,c.key_head_dim).repeat(c.num_value_heads//c.num_key_heads,1).reshape(-1,c.key_head_dim)
        v=mixed[:,2*c.key_width:]
        q=self.norm('query_l2',q,np.ones(c.key_head_dim,np.float32),1e-6,False,1/np.sqrt(c.key_head_dim)).reshape(x.shape[0],-1)
        k=self.norm('key_l2',k,np.ones(c.key_head_dim,np.float32),1e-6,False).reshape(x.shape[0],-1)
        z=project('in_proj_z',n); b=project('in_proj_b',n); a=project('in_proj_a',n)
        graph=VectorGraph(self,'delta_controls',[(b,'bf16'),(a,'bf16'),
            (weights.dt_bias.reshape(1,-1),'f32'),(weights.a_log.reshape(1,-1),'f32')])
        beta=graph.sigmoid(0)
        biased=graph.add(1,2,kind='add')
        softplus=graph.softplus(biased)
        exponent=graph.add(3,kind='exp',immediate=0)
        negative=graph.add(exponent,kind='negate',immediate=0)
        logdecay=graph.add(softplus,negative)
        decay=graph.add(logdecay,kind='exp',immediate=0)
        beta,decay=graph.finish((beta,decay))
        state=np.zeros((c.num_value_heads*c.key_head_dim,c.value_head_dim),np.float32)
        core,state=self.delta(q,k,v,decay,beta,state,c)
        def norm_gate(graph):
            normalized=graph.norm(0,1,c.rms_norm_eps)
            gate=graph.add(2,graph.sigmoid(2),dtype='bf16')
            return graph.add(normalized,gate,dtype='bf16')
        gated=self.vector_tiles('delta_norm_gate',[(core.reshape(-1,c.value_head_dim),'bf16'),
            (weights.delta_norm.reshape(1,-1),'f32'),(z.reshape(-1,c.value_head_dim),'bf16')],norm_gate).reshape(core.shape)
        attention=project('out_proj',gated)
        residual=self.vector_tiles('residual0',[(x,'bf16'),(attention,'bf16')],lambda g:g.add(0,1,kind='add',dtype='bf16'))
        ffn=self.norm('post_attention_norm',residual,bf16(1+weights.post_attention_norm),c.rms_norm_eps)
        gate,up=project('gate_proj',ffn),project('up_proj',ffn)
        def swiglu(graph):
            activated=graph.add(0,graph.sigmoid(0),dtype='bf16')
            return graph.add(activated,1,dtype='bf16')
        hidden=self.vector_tiles('swiglu',[(gate,'bf16'),(up,'bf16')],swiglu)
        down=project('down_proj',hidden)
        output=self.vector_tiles('output',[(residual,'bf16'),(down,'bf16')],lambda g:g.add(0,1,kind='add',dtype='bf16'))
        return output,state,qkv[-c.conv_kernel_size:]
