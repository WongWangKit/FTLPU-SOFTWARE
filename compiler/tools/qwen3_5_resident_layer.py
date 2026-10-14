"""Construct a complete decoder graph; execution has no host intermediate math."""
from pathlib import Path
import re
import subprocess
import numpy as np
from qwen3_5_compiled_layer import VectorGraph, CompiledStages, bf16
from resident_schedule import Value


class ResidentGraph:
    type=staticmethod(CompiledStages.type)
    def __init__(self, opt, directory, target_config=None):
        self.opt,self.directory=Path(opt),Path(directory)
        self.target_config=Path(target_config).resolve() if target_config else None
        self.directory.mkdir(parents=True,exist_ok=True)
        self.nodes=[]

    def run(self,name,source,inputs,outputs):
        node=len(self.nodes)
        if ('ftlpu.kernel.vector_alu' in source or 'ftlpu.kernel.gated_delta_rule' in source) and node%2:
            source=re.sub(r'(@stage\([^{}]+?\)\s*->\s*(?:\([^{}]+\)|tensor<[^>]+>))\s*\{',
                r'\1 attributes {ftlpu.vxm_queue_base = 8 : i64} {',source,count=1,flags=re.S)
        prefix=self.directory/f'{node:03d}_{name}'
        kernel=prefix.with_suffix('.kernel.mlir'); schedule=prefix.with_suffix('.schedule.mlir')
        config_stamp=prefix.with_suffix('.target-config.txt')
        config_identity=self.target_config.read_text(encoding='utf-8') if self.target_config else 'compiler default'
        cached=config_stamp.exists() and config_stamp.read_text(encoding='utf-8')==config_identity and kernel.exists() and schedule.exists() and kernel.read_text(encoding='utf-8')==source and schedule.stat().st_mtime_ns>=self.opt.stat().st_mtime_ns
        if not cached:
            kernel.write_text(source,encoding='utf-8')
            command=[str(self.opt),'--input',str(kernel),'--output',str(schedule),
                '--pipeline','ftlpu-stablehlo-to-schedule']
            if self.target_config:command.extend(['--target-config',str(self.target_config)])
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode: raise RuntimeError(f'{name}: {result.stderr}')
            config_stamp.write_text(config_identity,encoding='utf-8')
        self.nodes.append(dict(name=name,inputs=inputs,outputs=outputs,schedule=schedule))
        return [Value(node,i,shape,dtype,np.arange(np.prod(shape),dtype=np.int64)) for i,(shape,dtype) in enumerate(outputs)]

    def project(self,name,value,weight,scale):
        m,k=value.shape; n=weight.shape[0]; padded=(n+63)//64*64
        # A page occupies at most half of the physical weight bank. The
        # split is derived from matrix geometry before any schedule exists.
        max_columns = 64 * max(1, 4096 // ((k // 32) * 4))
        if n > max_columns:
            return Value.concatenate([self.project(name + f'_page{start}', value,
                weight[start:start+max_columns], scale)
                for start in range(0,n,max_columns)], axis=1)
        packed=np.pad(weight.T,((0,0),(0,padded-n)))
        a,w,o=self.type((m,k),'bf16'),self.type(packed.shape,'i8'),self.type((m,padded),'bf16')
        source=f'''module {{ func.func @stage(%a: {a}, %w: {w}) -> {o} {{
          %o = "ftlpu.kernel.matmul"(%a, %w) {{m = {m} : i64, n = {padded} : i64,
          k = {k} : i64, unit = "MXM", rhs_scale = {float(scale):.9e} : f32}} : ({a}, {w}) -> {o}
          return %o : {o}
        }} }}'''
        return self.run(name,source,[(value,'bf16'),(packed,'i8')],[((m,padded),'bf16')])[0][:,:n]

    def norm(self,name,x,gamma,epsilon,mean=True,scale=1):
        if x.size > 4096:
            rows = max(1,4096 // x.shape[1])
            return Value.concatenate([self.norm(name+f'_tile{start}',x[start:start+rows],
                gamma,epsilon,mean,scale) for start in range(0,x.shape[0],rows)])
        graph=VectorGraph(self,name,[(x,'bf16'),(np.asarray(gamma).reshape(1,-1),'f32')])
        return graph.finish(graph.norm(0,1,epsilon,mean,scale))

    def join(self,name,value,hemisphere='west'):
        node=len(self.nodes)
        self.nodes.append(dict(name=name,inputs=[(value,value.dtype)],outputs=[(value.shape,value.dtype)],
            schedule=None,layout_join=True,hemisphere=hemisphere))
        return Value(node,0,value.shape,value.dtype,np.arange(value.size))

    def conv(self,qkv,weight):
        if qkv.shape[1] > 1024:
            groups=[]
            for start in range(0,qkv.shape[1],2048):
                parts=[self.conv(qkv[:,offset:offset+1024],weight[offset:offset+1024])
                    for offset in range(start,min(start+2048,qkv.shape[1]),1024)]
                groups.append(self.join(f'conv_layout_group{start}',Value.concatenate(parts,axis=1),
                    'east' if start==2048 else 'west'))
            return Value.concatenate(groups,axis=1)
        a,w=self.type(qkv.shape,'bf16'),self.type(weight.shape,'bf16')
        source=f'''module {{ func.func @stage(%a: {a}, %w: {w}) -> {a} {{
          %o = ftlpu.kernel.depthwise_conv1d %a, %w {{kernel_size = 4 : i64,
          causal = true, activation = "silu"}} : ({a}, {w}) -> {a}
          return %o : {a}
        }} }}'''
        return self.run('conv1d_silu',source,[(qkv,'bf16'),(weight,'bf16')],[(qkv.shape,'bf16')])[0]

    def vector_tiles(self,name,inputs,build):
        shape=inputs[0][0].shape;rows=max(1,4096//shape[1])
        if name in ('swiglu','residual0','output'):
            graph=VectorGraph(self,name,inputs)
            return graph.finish(build(graph))
        values=[]
        for start in range(0,shape[0],rows):
            graph=VectorGraph(self,name+f'_tile{start}',[(v[start:start+rows] if v.shape[0]>1 else v,d) for v,d in inputs])
            values.append(graph.finish(build(graph)))
        return Value.concatenate(values)

    def delta(self,q,k,v,decay,beta,state,c):
        outputs=[]
        # Only one token's scalar Q/K is materialized. FP32 recurrent state
        # is carried by device graph edges, with no intermediate host read.
        for token in range(q.shape[0]):
            qt,vt,dt,st=self.type((1,q.shape[1]),'bf16'),self.type((1,v.shape[1]),'bf16'),self.type((1,c.num_value_heads),'f32'),self.type(state.shape,'f32')
            source=f'''module {{ func.func @stage(%q: {qt}, %k: {qt}, %v: {vt},
              %d: {dt}, %b: {dt}, %s: {st}) -> ({vt}, {st}) {{
              %o, %s1 = ftlpu.kernel.gated_delta_rule %q, %k, %v, %d, %b, %s {{heads = {c.num_value_heads} : i64,
              key_dim = {c.key_head_dim} : i64, value_dim = {c.value_head_dim} : i64}} : ({qt}, {qt}, {vt}, {dt}, {dt}, {st}) -> ({vt}, {st})
              return %o, %s1 : {vt}, {st}
            }} }}'''
            out,state=self.run('delta_rule'+f'_token{token}',source,[(q[token:token+1],'bf16'),(k[token:token+1],'bf16'),
                (v[token:token+1],'bf16'),(decay[token:token+1],'f32'),(beta[token:token+1],'f32'),(state,'f32')],
                [((1,v.shape[1]),'bf16'),(state.shape,'f32')])
            outputs.append(out)
        return self.join('recurrent_output_layout',Value.concatenate(outputs)),state

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
