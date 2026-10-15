"""Official full-attention equations and checkpoint import; no hardware lowering."""
from pathlib import Path
import hashlib
import json
import numpy as np
from import_hf_decoder_layer import SafeTensorStore
from qwen3_5_compiled_layer import bf16
from qwen3_5_logical_graph import LogicalGraph


def load(directory, layer=3, seq=32):
    directory = Path(directory)
    provenance = json.loads((directory/'provenance.json').read_text())
    digest = hashlib.sha256((directory/'model.safetensors').read_bytes()).hexdigest()
    if provenance['repo'] != 'Qwen/Qwen3.5-0.8B' or provenance['subset_sha256'] != digest:
        raise ValueError('official checkpoint provenance mismatch')
    if provenance['config_sha256'] != hashlib.sha256((directory/'config.json').read_bytes()).hexdigest():
        raise ValueError('official checkpoint config hash mismatch')
    config = json.loads((directory/'config.json').read_text())['text_config']
    if config['layer_types'][layer] != 'full_attention':
        raise ValueError('selected checkpoint layer is not full attention')
    store = SafeTensorStore(directory)
    prefix = f'model.language_model.layers.{layer}'
    weights = {name: store.read(prefix+'.'+name+'.weight') for name in (
        'input_layernorm','post_attention_layernorm','self_attn.q_proj',
        'self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','self_attn.q_norm',
        'self_attn.k_norm','mlp.gate_proj','mlp.up_proj','mlp.down_proj')}
    x = store.read('model.language_model.embed_tokens.weight')[:seq]
    return bf16(x), weights, config, provenance


def quantize(weights, config):
    packed, scales, effective = {}, {}, dict(weights)
    for name, value in weights.items():
        if name.endswith('_proj'):
            scale = float(np.max(np.abs(value))/127) or 1.
            q = np.clip(np.rint(value/scale), -127, 127).astype(np.int8)
            packed[name], scales[name] = q, scale
            effective[name] = q.astype(np.float32)*np.float32(scale)
    # Q and its output gate are interleaved per head in the official checkpoint.
    # Split after quantizing the combined matrix, preserving its shared scale.
    h, d, hidden = config['num_attention_heads'], config['head_dim'], config['hidden_size']
    qg = packed.pop('self_attn.q_proj').reshape(h, 2, d, hidden)
    packed['query'] = qg[:,0].reshape(h*d, hidden).copy()
    packed['attention_gate'] = qg[:,1].reshape(h*d, hidden).copy()
    scales['query'] = scales['attention_gate'] = scales['self_attn.q_proj']
    for short, name in [('key','self_attn.k_proj'),('value','self_attn.v_proj'),
                        ('output','self_attn.o_proj'),('gate','mlp.gate_proj'),
                        ('up','mlp.up_proj'),('down','mlp.down_proj')]:
        if name in packed:
            packed[short], scales[short] = packed.pop(name), scales[name]
    return packed, scales, effective


def attention_source(seq, config, scales):
    """A logical dense GQA graph consumed by the shared native attention passes."""
    h, k, d, hidden = (config[n] for n in ('num_attention_heads','num_key_value_heads','head_dim','hidden_size'))
    width, kv = h*d, k*d
    ty = LogicalGraph.type
    t = lambda *shape: ty(shape, 'bf16')
    a, qw, kw, ow = t(seq,hidden), ty((hidden,width),'i8'), ty((hidden,kv),'i8'), ty((width,hidden),'i8')
    gate, gamma = t(seq,width), t(d)
    args = [('a',a),('qw',qw),('kw',kw),('vw',kw),('ow',ow),('qn',gamma),('kn',gamma),('gate',gate)]
    lines = ['module { func.func @stage('+', '.join('%'+n+': '+v for n,v in args)+') -> '+a+' {']
    def op(name, kind, operands, types, result, attrs=''):
        lines.append(f' %{name} = "ftlpu.kernel.{kind}"('+', '.join('%'+v for v in operands)+f') {{{attrs}}} : ('+', '.join(types)+f') -> {result}')
    def project(name, weight, n, result):
        op(name,'matmul',['a',weight],[a,qw if weight=='qw' else kw],result,
           f'm = {seq} : i64, n = {n} : i64, k = {hidden} : i64, unit = "MXM", rhs_scale = {scales[name]:.9e} : f32')
    project('query','qw',width,gate)
    project('key','kw',kv,t(seq,kv))
    project('value','vw',kv,t(seq,kv))
    eps = config['rms_norm_eps']
    rope = config['rope_parameters']
    for name, heads, g in [('query',h,'qn'),('key',k,'kn')]:
        src = gate if name=='query' else t(seq,kv)
        op(name+'_flat','reshape',[name],[src],t(seq*heads,d))
        op(name+'_norm','rms_norm',[name+'_flat',g],[t(seq*heads,d),gamma],t(seq*heads,d),f'axis = -1 : i64, epsilon = {eps:.9e} : f32')
        op(name+'_heads','reshape',[name+'_norm'],[t(seq*heads,d)],t(seq,heads,d))
        op(name+'_rope','rope',[name+'_heads'],[t(seq,heads,d)],t(seq,heads,d),
           f'heads = {heads} : i64, head_dim = {d} : i64, theta = {rope["rope_theta"]:.9e} : f32, rotary_dim = {int(d*rope["partial_rotary_factor"])} : i64')
    op('v_heads','reshape',['value'],[t(seq,kv)],t(seq,k,d))
    for name, src in [('k','key_rope'),('v','v_heads')]:
        op(name+'_repeat','gqa_broadcast',[src],[t(seq,k,d)],t(seq,h,d),f'query_heads = {h} : i64, kv_heads = {k} : i64')
    for name, src in [('q','query_rope'),('k','k_repeat'),('v','v_repeat')]:
        op(name+'_transpose','transpose',[src],[t(seq,h,d)],t(h,seq,d),'permutation = array<i64: 1, 0, 2>')
    op('scores','batch_matmul',['q_transpose','k_transpose'],[t(h,seq,d)]*2,t(h,seq,seq),'role = "qk", transpose_rhs = true')
    op('probability','softmax',['scores'],[t(h,seq,seq)],t(h,seq,seq),f'axis = -1 : i64, causal = true, scale = {d**-.5:.9e} : f32')
    op('pv','batch_matmul',['probability','v_transpose'],[t(h,seq,seq),t(h,seq,d)],t(h,seq,d),'role = "pv", transpose_rhs = false')
    op('context_t','transpose',['pv'],[t(h,seq,d)],t(seq,h,d),'permutation = array<i64: 1, 0, 2>')
    op('context','reshape',['context_t'],[t(seq,h,d)],gate)
    op('gated','sigmoid_multiply',['context','gate'],[gate,gate],gate)
    op('output','matmul',['gated','ow'],[gate,ow],a,
       f'm = {seq} : i64, n = {hidden} : i64, k = {width} : i64, unit = "MXM", rhs_scale = {scales["output"]:.9e} : f32')
    return '\n'.join(lines+[f' return %output : {a}', '} }'])


def graph_layer(x, weights, packed, scales, config):
    graph = LogicalGraph()
    norm = lambda name, value, role: graph.norm(name,value,bf16(1+weights[role]),config['rms_norm_eps'])
    n = norm('input_norm',x,'input_layernorm')
    gate = graph.project('attention_gate_projection',n,packed['attention_gate'],scales['attention_gate'])
    inputs = [(n,'bf16')]+[(packed[name].T.copy(),'i8') for name in ('query','key','value','output')]
    inputs += [(bf16(1+weights['self_attn.'+name+'_norm']),'bf16') for name in ('q','k')]+[(gate,'bf16')]
    attention = graph.run('full_attention',attention_source(x.shape[0],config,scales),inputs,[(x.shape,'bf16')])[0]
    residual = graph.vector_tiles('attention_residual',[(x,'bf16'),(attention,'bf16')],lambda g:g.add(0,1,kind='add',dtype='bf16'))
    ffn = norm('post_attention_norm',residual,'post_attention_layernorm')
    gate = graph.project('ffn_gate',ffn,packed['gate'],scales['gate'])
    up = graph.project('ffn_up',ffn,packed['up'],scales['up'])
    inner = graph.vector_tiles('swiglu',[(gate,'bf16'),(up,'bf16')],
        lambda g:g.add(g.add(0,g.sigmoid(0),dtype='bf16'),1,dtype='bf16'))
    down = graph.project('ffn_down',inner,packed['down'],scales['down'])
    output = graph.vector_tiles('output',[(residual,'bf16'),(down,'bf16')],lambda g:g.add(0,1,kind='add',dtype='bf16'))
    return graph, output


def reference_attention(n, w, c):
    """FP32 attention in original checkpoint head/rotary order."""
    rms = lambda x, gamma: x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+c['rms_norm_eps'])*(1+gamma)
    h,k,d = (c[name] for name in ('num_attention_heads','num_key_value_heads','head_dim'))
    x = n
    qg = (n@w['self_attn.q_proj'].T).reshape(len(x),h,2*d)
    q, gate = qg[:,:,:d],qg[:,:,d:].reshape(len(x),h*d)
    q = rms(q,w['self_attn.q_norm'])
    key = rms((n@w['self_attn.k_proj'].T).reshape(len(x),k,d),w['self_attn.k_norm'])
    v = (n@w['self_attn.v_proj'].T).reshape(len(x),k,d)
    rope = c['rope_parameters']; rotary = int(d*rope['partial_rotary_factor'])
    angle = np.arange(len(x),dtype=np.float32)[:,None]/np.float32(rope['rope_theta'])**(np.arange(0,rotary,2,dtype=np.float32)/rotary)
    cos,sin = np.cos(angle)[:,None,:],np.sin(angle)[:,None,:]
    def rotate(a):
        low,high = a[:,:,:rotary//2],a[:,:,rotary//2:rotary]
        return np.concatenate((low*cos-high*sin,high*cos+low*sin,a[:,:,rotary:]),axis=-1)
    q,key = rotate(q).transpose(1,0,2),np.repeat(rotate(key),h//k,axis=1).transpose(1,0,2)
    v = np.repeat(v,h//k,axis=1).transpose(1,0,2)
    scores = q@key.transpose(0,2,1)/np.float32(np.sqrt(d))
    scores = np.where(np.triu(np.ones((len(x),len(x)),bool),1),-np.inf,scores)
    exp = np.exp(scores-np.max(scores,axis=-1,keepdims=True));prob=exp/np.sum(exp,axis=-1,keepdims=True)
    context = (prob@v).transpose(1,0,2).reshape(len(x),h*d)
    return (context/(1+np.exp(-gate)))@w['self_attn.o_proj'].T


def reference(x, w, c):
    """Independent FP32 complete-layer equations."""
    rms = lambda x, gamma: x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+c['rms_norm_eps'])*(1+gamma)
    n = rms(x,w['input_layernorm'])
    attention = reference_attention(n,w,c)
    r = x+attention
    ffn = rms(r,w['post_attention_layernorm'])
    gate,up = ffn@w['mlp.gate_proj'].T,ffn@w['mlp.up_proj'].T
    output = r+((gate/(1+np.exp(-gate)))*up)@w['mlp.down_proj'].T
    return output
