"""Generic resident Schedule IR assembly and hardware layout bridges.

Only tensor geometry is inspected. All data movement/conversion is emitted as
MEM 3-D domains, SXM transpose/permute domains and ordinary VXM instructions.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def split_fields(text):
    fields, start, depth, quoted = [], 0, 0, False
    for i, char in enumerate(text):
        if char == '"': quoted = not quoted
        if quoted: continue
        if char in "{[": depth += 1
        if char in "}]": depth -= 1
        if char == ',' and depth == 0:
            fields.append(text[start:i].strip()); start = i + 1
    if text[start:].strip(): fields.append(text[start:].strip())
    return fields


def parse_attr(text):
    text = text.strip()
    if text.startswith('{'):
        return {k.strip(): parse_attr(v) for k, v in
                (field.split('=', 1) for field in split_fields(text[1:-1]))}
    if text.startswith('['): return [parse_attr(v) for v in split_fields(text[1:-1])]
    if text.startswith('"'): return json.loads(text)
    if text in ('true', 'false'): return text == 'true'
    number = text.split(':')[0].strip()
    return float(number) if any(c in number.lower() for c in ('.', 'e')) else int(number)


def attr(value, array=False):
    if isinstance(value, dict): return '{' + ', '.join(f'{k} = {attr(v)}' for k, v in value.items()) + '}'
    if isinstance(value, list): return '[' + ', '.join(attr(v, True) for v in value) + ']'
    if isinstance(value, str): return json.dumps(value)
    if isinstance(value, bool): return str(value).lower()
    if isinstance(value, float): return f'{value:.9e} : f32'
    return str(int(value)) + ('' if array else ' : i64')


def operation(line):
    match = re.search(r'ftlpu.schedule.(\w+)\s', line)
    if not match: return None
    first = line.index('{', match.end())
    depth = 0
    for i in range(first, len(line)):
        depth += (line[i] == '{') - (line[i] == '}')
        if depth == 0: return match[1], parse_attr(line[first:i+1]), line[:first], line[i+1:]
    raise ValueError('unterminated Schedule operation')


@dataclass
class Value:
    node: int
    output: int
    shape: tuple
    dtype: str
    indices: np.ndarray
    owners: np.ndarray | None = None

    @property
    def size(self): return int(np.prod(self.shape))
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple): shape = shape[0]
        geometry = self.indices.reshape(self.shape).reshape(*shape)
        owners = None if self.owners is None else self.owners.reshape(self.shape).reshape(*shape).reshape(-1)
        return Value(self.node, self.output, geometry.shape, self.dtype, geometry.reshape(-1), owners)
    def __getitem__(self, item):
        geometry = self.indices.reshape(self.shape)[item]
        owners = None if self.owners is None else self.owners.reshape(self.shape)[item].reshape(-1)
        return Value(self.node, self.output, geometry.shape, self.dtype, geometry.reshape(-1), owners)
    def repeat(self, count, axis):
        geometry = np.repeat(self.indices.reshape(self.shape), count, axis=axis)
        owners = None if self.owners is None else np.repeat(self.owners.reshape(self.shape), count, axis=axis).reshape(-1)
        return Value(self.node, self.output, geometry.shape, self.dtype, geometry.reshape(-1), owners)
    @property
    def T(self):
        geometry = self.indices.reshape(self.shape).T
        owners = None if self.owners is None else self.owners.reshape(self.shape).T.reshape(-1)
        return Value(self.node, self.output, geometry.shape, self.dtype, geometry.reshape(-1), owners)

    def segments(self):
        if self.owners is None:
            yield (self.node, self.output), self.indices, np.arange(self.size)
        else:
            for owner in np.unique(self.owners):
                positions = np.flatnonzero(self.owners == owner)
                yield (int(owner) // 256, int(owner) % 256), self.indices[positions], positions

    @staticmethod
    def concatenate(values, axis=0):
        indices = np.concatenate([v.indices.reshape(v.shape) for v in values], axis=axis)
        owners = np.concatenate([(np.full(v.shape, v.node * 256 + v.output, np.int64)
            if v.owners is None else v.owners.reshape(v.shape)) for v in values], axis=axis)
        return Value(-1, 0, indices.shape, values[0].dtype, indices.reshape(-1), owners.reshape(-1))


class Schedule:
    def __init__(self, target):
        self.target = target
        self.lines, self.counter, self.cursor = [], 0, 64
        self.slices = target['memory']['slices_per_hemisphere']
        self.group = target['streams']['mem_slices_per_register_group']
        self.period = 32 // target['throughput']['mem_read_bytes_per_cycle']
        self.bridge_domains=0
        self.raw_sr_copies=False

    def bridge_queue(self):
        base=(4,6,12,14)[self.bridge_domains%4]
        self.bridge_domains+=1
        return base

    @staticmethod
    def bridge_result(beat,source,dest,base):
        latency=((19 if source=='west' else 12) if dest=='east' else (24 if source=='west' else 17))
        if source==dest:latency+=(base-4)*2*(1 if source=='east' else -1)
        return beat+latency

    def emit(self, kind, values, operands='', type_='tensor<1x32xf32>'):
        suffix = ''
        if kind == 'vxm':
            name = f'%bridge{self.counter}'; self.counter += 1
            self.lines.append(f'{name} = ftlpu.schedule.vxm %dummy, %dummy {attr(values)} : ({type_}, {type_}) -> {type_}')
        else: self.lines.append(f'ftlpu.schedule.{kind} {attr(values)}{suffix}')

    def mem(self, cycle, placement, plane, opcode, address, stream, counts=(1,1,1), strides=(0,0,0), intervals=(1,1,1)):
        self.emit('mem_transfer', dict(cycle=int(cycle), hemisphere=int(placement['hemisphere']=='west'),
            slice=placement['slices'][plane], bank=placement.get('bank',0), opcode=opcode,
            address=int(address), packed_stream=stream, repeat_count=counts[0], repeat_interval=intervals[0],
            address_stride=strides[0], wave_count=counts[1], wave_interval=intervals[1], wave_address_stride=strides[1],
            group_count=counts[2], group_interval=intervals[2], group_address_stride=strides[2]))

    def vxm(self, cycle, queue=0, kind='cast', lhs='stream_f32', lhs_index=0,
            rhs='immediate', rhs_index=0, source='west', rhs_source='west', output='west',
            format_='fp32', out=-1, count=1, interval=None):
        values = dict(cycle=int(cycle), queue=queue, opcode=kind, chain_depth=8,
            lhs_kind=lhs, lhs_index=lhs_index, lhs_immediate=0.0,
            rhs_kind=rhs, rhs_index=rhs_index, rhs_immediate=0.0,
            cast_target=format_, output_stream=out, repeat_count=count, repeat_interval=1,
            input_hemisphere='east', output_hemisphere='east' if output=='west' else 'west')
        if interval is not None:
            values['repeat_count']=1
            values['wave_count']=count
            values['wave_interval']=interval
        if lhs.startswith('stream'): values['lhs_stream_source'] = source
        if rhs.startswith('stream'): values['rhs_stream_source'] = rhs_source
        self.emit('vxm', values)

    def sxm(self, cycle, hemisphere, opcode, sources, destinations, map_, count=1, interval=1, row=None, waves=1, wave_interval=1,map_stride=0):
        values=dict(cycle=int(cycle),hemisphere=int(hemisphere=='west'),opcode=opcode,
            source_streams=sources,destination_streams=destinations,permute_map=map_,weight_layout='resident_layout_bridge',
            repeat_count=count,repeat_interval=interval)
        values['wave_count']=waves;values['wave_interval']=wave_interval
        if row is not None: values['output_row'] = row
        if map_stride: values['permute_map_stride']=map_stride
        if opcode=='transpose' and len(sources)==2: values['input_row']=9
        self.emit('sxm',values)

    def scalar_copy(self, source, dest, indices, offsets=None):
        """Affine runs are formed from tensor-view geometry, never FU traces."""
        indices=np.asarray(indices).reshape(-1)
        offsets=np.arange(indices.size) if offsets is None else np.asarray(offsets).reshape(-1)
        if len(indices)!=len(offsets): raise ValueError('copy geometry mismatch')
        if source['hemisphere']==dest['hemisphere'] and source.get('bank',0)==dest.get('bank',0) and set(source['slices'])&set(dest['slices']):
            # A bank ICU cannot read and write concurrent domains. Route a
            # complete phase through a separately reserved temporary fabric.
            temp=dict(source,hemisphere='east' if source['hemisphere']=='west' else 'west',
                slices=list(range(48,48+len(source['slices']))),base_row=0,
                bank=0 if source['hemisphere']=='west' else 1,instruction_count=len(indices))
            self.scalar_copy(source,temp,indices)
            self.scalar_copy(temp,dest,np.arange(len(indices)),offsets)
            return
        domain=affine_geometry(indices,offsets)
        if domain is not None:
            counts,src_strides,dst_strides=domain
            self.scalar_domain(source,dest,int(indices[0]),int(offsets[0]),counts,src_strides,dst_strides)
            return
        pos=0
        while pos<len(indices):
            src_stride=int(indices[pos+1]-indices[pos]) if pos+1<len(indices) else 0
            dst_stride=int(offsets[pos+1]-offsets[pos]) if pos+1<len(indices) else 0
            end=pos+1
            while end<len(indices) and int(indices[end]-indices[end-1])==src_stride and int(offsets[end]-offsets[end-1])==dst_stride: end+=1
            count=end-pos
            beat=self.cursor+32
            queue=self.bridge_queue()
            n=len(source['slices'])
            for plane in range(n):
                slice_=source['slices'][plane]
                self.mem(beat-(slice_//self.group+1),source,plane,'read',source['base_row']+int(indices[pos]),32+plane,
                    (count,1,1),(src_stride,0,0),(self.period,1,1))
            self.vxm(beat-1,queue=queue,lhs='stream_f32' if n==4 else 'stream_bf16',source=source['hemisphere'],
                output=dest['hemisphere'], count=count)
            self.vxm(beat-1,queue=queue+1,lhs='previous',source=source['hemisphere'],output=dest['hemisphere'],
                format_='fp32' if len(dest['slices'])==4 else 'bf16',out=24,count=count)
            result=self.bridge_result(beat,source['hemisphere'],dest['hemisphere'],queue)
            for plane in range(len(dest['slices'])):
                self.mem(result+dest['slices'][plane]//self.group+1,dest,plane,'write',dest['base_row']+int(offsets[pos]),24+plane,
                    (count,1,1),(dst_stride,0,0),(self.period,1,1))
            self.cursor=result+count*self.period+96
            pos=end

    def scalar_domain(self,source,dest,src_offset,dst_offset,counts,src_strides,dst_strides):
        if self.raw_sr_copies and self.raw_mem_domain(source,dest,src_offset,dst_offset,counts,src_strides,dst_strides):
            return
        beat=self.cursor+32
        queue=self.bridge_queue()
        intervals=(self.period,counts[0]*self.period,counts[0]*counts[1]*self.period)
        count=int(np.prod(counts));n=len(source['slices'])
        for plane in range(n):
            slice_=source['slices'][plane]
            self.mem(beat-(slice_//self.group+1),source,plane,'read',source['base_row']+src_offset,32+plane,
                counts,src_strides,intervals)
        self.vxm(beat-1,queue=queue,lhs='stream_f32' if n==4 else 'stream_bf16',source=source['hemisphere'],output=dest['hemisphere'],count=count)
        self.vxm(beat-1,queue=queue+1,lhs='previous',output=dest['hemisphere'],format_='fp32' if len(dest['slices'])==4 else 'bf16',out=24,count=count)
        result=self.bridge_result(beat,source['hemisphere'],dest['hemisphere'],queue)
        for plane in range(len(dest['slices'])):
            self.mem(result+dest['slices'][plane]//self.group+1,dest,plane,'write',dest['base_row']+dst_offset,24+plane,
                counts,dst_strides,intervals)
        self.cursor=result+count*self.period+96

    def raw_mem_domain(self,source,dest,src_offset,dst_offset,counts,src_strides,dst_strides):
        """Copy bytes through directed SR links; no VALU configuration/cast.

        Same-fabric copies require a reachable downstream MEM boundary.
        Cross-fabric copies use the existing passive sixteen-column SR chain.
        Streams 20..23 avoid the active RMSNorm/Conv operand channels, including
        the mirrored RMSNorm channels 16..19. Paging sees the MEM SR usage.
        """
        if len(source['slices'])!=len(dest['slices']):return False
        paths=[]
        for plane,(src,dst) in enumerate(zip(source['slices'],dest['slices'])):
            sg,dg=src//self.group,dst//self.group
            if source['hemisphere']!=dest['hemisphere']:
                read_stream=52+plane;write_stream=20+plane
                latency=sg+self.target['throughput']['vxm_alus']+dg+3
            elif sg<dg:
                read_stream=write_stream=20+plane;latency=dg-sg
            elif sg>dg:
                read_stream=write_stream=52+plane;latency=sg-dg
            else:return False # no same-column U-turn link
            paths.append((read_stream,write_stream,latency))
        start=self.cursor+32
        intervals=(self.period,counts[0]*self.period,counts[0]*counts[1]*self.period)
        for plane,(read_stream,write_stream,latency) in enumerate(paths):
            self.mem(start,source,plane,'read',source['base_row']+src_offset,read_stream,counts,src_strides,intervals)
            self.mem(start+latency,dest,plane,'write',dest['base_row']+dst_offset,write_stream,counts,dst_strides,intervals)
        self.cursor=start+int(np.prod(counts))*self.period+max(p[2] for p in paths)+96
        return True

    def unpack(self, source, dest, shape, needed=None):
        """Packed [rows, columns] to row-major scalar broadcast."""
        rows, columns=shape
        byte_count=len(dest['slices'])
        paired=source['kind']=='fp16_hemisphere_pair_planar'
        step=64 if paired else 32
        needed_columns=set(int(index)%columns for index in needed) if needed is not None else set(range(columns))
        for column in range(min(columns,step)):
            selected=[c for c in range(column,columns,step) if c in needed_columns]
            if not selected: continue
            if selected!=list(range(selected[0],selected[-1]+1,step)):raise ValueError('non-affine packed columns')
            column=selected[0];blocks=len(selected)
            hemi=('west' if (column//32)%2 else 'east') if paired else source['hemisphere']
            if byte_count == 4 and hemi != dest['hemisphere']:
                raise ValueError('FP32 packed bridge requires a same-hemisphere archive')
            src=copy.deepcopy(source); src['hemisphere']=hemi
            address=(column//64 if paired else column//32)*rows
            tile, lane=(column%32)//8,column%8
            for pair in range(byte_count//2):
                transpose=self.cursor+32
                interval=128
                for plane in range(2):
                    slice_=src['slices'][2*pair+plane]
                    self.mem(transpose-(14-slice_//self.group),src,2*pair+plane,'read',src['base_row']+address,16+plane,
                        (rows,blocks,1),(1,rows,0),(interval,rows*interval,1))
                self.sxm(transpose,hemi,'transpose',[16,17],list(range(16,32)),list(range(32)),rows,interval,waves=blocks,wave_interval=rows*interval)
                permute=transpose+5
                map_=[tile*8+i%8 for i in range(32)]
                direct=hemi==dest['hemisphere']
                output_base=32 if direct else 48
                self.sxm(permute,hemi,'permute',list(range(16,32)),list(range(output_base,output_base+16)),map_,rows,interval,row=9+lane if lane<7 else 7,waves=blocks,wave_interval=rows*interval)
                # Selected transpose row has one scalar in all 32 lanes. Pass
                # it through the ordinary VXM to the destination fabric.
                beat=permute+15
                if direct:
                    result=permute+1
                else:
                    self.vxm(beat-1,queue=4,lhs='stream_bf16',lhs_index=16+2*lane,source=hemi,output=dest['hemisphere'],count=rows*blocks,interval=interval)
                    self.vxm(beat-1,queue=5,lhs='previous',source=hemi,output=dest['hemisphere'],format_='bf16',out=0,count=rows*blocks,interval=interval)
                    result=beat+(19 if dest['hemisphere']=='east' else (24 if hemi=='west' else 17))
                for plane in range(2):
                    self.mem((permute+14-dest['slices'][2*pair+plane]//self.group if direct else result+dest['slices'][2*pair+plane]//self.group+1),dest,2*pair+plane,'write',dest['base_row']+column,(32+2*lane+plane if direct else plane),
                        (rows,blocks,1),(columns,step,0),(interval,rows*interval,1))
                self.cursor=result+rows*blocks*interval+96

    def pack(self, source, dest, shape, indices, word_offsets=None):
        """Scalar broadcast to a packed vector, using four masked 8-lane groups.

        The eight source values are first staged on independent MEM pairs.
        Each transpose produces an eight-feature vector repeated on four
        tiles. Generic VXM masks/FP32 adds assemble the four feature groups.
        No host reads or numerical evaluation are involved.
        """
        rows, columns=shape
        if columns%32: raise ValueError('packed destination requires columns divisible by 32')
        words=rows*(columns//32)
        word_offsets=np.arange(words) if word_offsets is None else np.asarray(word_offsets)
        write_domain=affine_geometry(np.arange(words),word_offsets)
        if write_domain is None: raise ValueError('non-affine packed output word geometry')
        write_counts,_,write_strides=write_domain
        write_intervals=(32,write_counts[0]*32,write_counts[0]*write_counts[1]*32)
        indices=np.asarray(indices).reshape(rows,columns)
        floats=len(source['slices'])==4
        # The staging area is reserved independently of the archived values.
        staging=dict(kind='fp16_pair_planar',slices=[],base_row=0,bank=1,hemisphere='east')
        for byte_pair in range(2 if floats else 1):
            # Byte pairs can be packed independently without any arithmetic
            # on their representation. FP32 masks operate on the decoded
            # BF16 pair; this is restricted below to actual BF16 values.
            if floats: raise ValueError('FP32 scalar-to-packed bridge is not yet implemented')
            for slot in range(8):
                selected=[]
                for block in range(columns//32):
                    for row in range(rows):
                        for tile in range(4): selected.append(int(indices[row,block*32+tile*8+slot]))
                staging['slices']=[32+slot*2,33+slot*2]
                self.scalar_copy(source,staging,selected)
        ping=[dict(kind='fp32_quad_planar',slices=list(range(i,i+4)),base_row=0,bank=1,hemisphere='east') for i in (0,4)]
        masks=dict(kind='fp32_quad_planar',slices=[48,49,50,51],base_row=0,bank=1,hemisphere='east')
        for tile in range(4):
            transpose=self.cursor+32
            interval=32
            for slot in range(8):
                staging['slices']=[32+slot*2,33+slot*2]
                for plane in range(2):
                    slice_=staging['slices'][plane]
                    self.mem(transpose-(14-slice_//self.group),staging,plane,'read',tile,slot*2+plane,
                        (words,1,1),(4,0,0),(interval,1,1))
            self.sxm(transpose,'east','transpose',list(range(16)),list(range(16,32)),list(range(32)),words,interval)
            permute=transpose+5
            # Only row 0 is emitted. Reserve W24..31 for C2C instead of
            # assigning unused crossbar rows to those transport streams.
            self.sxm(permute,'east','permute',list(range(16,32)),list(range(32,48)),[i%8 for i in range(32)],words,interval,row=9)
            beat=permute+15
            for plane in range(4):
                self.mem(beat-(masks['slices'][plane]//self.group+1),masks,plane,'read',tile,40+plane,
                    (words,1,1),(0,0,0),(interval,1,1))
            self.vxm(beat-1,kind='multiply',lhs='stream_bf16',rhs='stream_f32',lhs_index=0,rhs_index=8,source='east',rhs_source='east',count=words,interval=interval)
            last=1
            if tile:
                previous=ping[(tile-1)%2]
                for plane in range(4):
                    self.mem(beat+1-(previous['slices'][plane]//self.group+1),previous,plane,'read',0,44+plane,
                        (words,1,1),(1,0,0),(interval,1,1))
                self.vxm(beat-1,queue=1,kind='add',lhs='previous',rhs='stream_f32',rhs_index=12,rhs_source='east',count=words,interval=interval)
                last=2
            output=ping[tile%2] if tile!=3 else dest
            hemis=['east','west'] if output['hemisphere']=='both' else [output['hemisphere']]
            for hemi in hemis:
                # Both fabrics cannot receive one VXM result simultaneously;
                # duplicate final packing for replicated projection inputs.
                if len(hemis)>1 and hemi=='west': continue
                self.vxm(beat-1,queue=last,lhs='previous',output=hemi,
                    format_='fp32' if tile!=3 else 'bf16',out=0,count=words,interval=interval)
                result=beat+(18 if hemi=="west" else 2*last+3)
                output_copy=dict(output,hemisphere=hemi)
                for plane in range(len(output['slices'])):
                    self.mem(result+output['slices'][plane]//self.group+1,output_copy,plane,'write',output['base_row']+(int(word_offsets[0]) if tile==3 else 0),plane,
                        write_counts if tile==3 else (words,1,1),write_strides if tile==3 else (1,0,0),write_intervals if tile==3 else (interval,1,1))
            self.cursor=result+words*interval+96
        if dest['hemisphere']=='both':
            self.scalar_copy(dict(dest,hemisphere='east'),dict(dest,hemisphere='west'),word_offsets,word_offsets)

    def unpack_view(self,source,dest,source_shape,indices,offsets):
        """Move an aligned tensor view from packed words into scalar rows."""
        rows,columns=source_shape
        indices=np.asarray(indices);offsets=np.asarray(offsets)
        paired=source['kind']=='fp16_hemisphere_pair_planar'
        step=64 if paired else 32
        _,word_counts=np.unique(np.unique(indices)//32,return_counts=True)
        full_words=bool(len(word_counts) and np.all(word_counts==32))
        selectors=[(lane,h) for h in range(2 if paired else 1) for lane in range(8)] if full_words else [(r,None) for r in range(min(step,columns))]
        for residue,half in selectors:
            selected=np.flatnonzero((indices%8==residue)&((indices%columns//32)%2==half)) if full_words and paired else np.flatnonzero(indices%8==residue) if full_words else np.flatnonzero(indices%columns%step==residue)
            if not len(selected):continue
            src=dict(source,hemisphere=('west' if (half if full_words else residue//32) else 'east') if paired else source['hemisphere'])
            words=(indices[selected]%columns//step)*rows+indices[selected]//columns
            domain=affine_geometry(words,offsets[selected])
            if domain is None: raise ValueError('non-affine unpacked tensor view')
            counts,src_strides,dst_strides=domain
            intervals=(32,counts[0]*32,counts[0]*counts[1]*32)
            n=int(np.prod(counts));tile,lane=(0 if full_words else (residue%32)//8),residue%8
            for pair in range(len(dest['slices'])//2):
                transpose=self.cursor+32
                for plane in range(2):
                    slice_=src['slices'][2*pair+plane]
                    self.mem(transpose-(14-slice_//self.group),src,2*pair+plane,'read',src['base_row']+int(words[0]),16+plane,counts,src_strides,intervals)
                self.sxm(transpose,src['hemisphere'],'transpose',[16,17],list(range(16,32)),list(range(32)),counts[0],32,waves=counts[1]*counts[2],wave_interval=counts[0]*32)
                permute=transpose+5;direct=src['hemisphere']==dest['hemisphere']
                if len(dest['slices'])==4 and not direct:raise ValueError('FP32 unpack requires the same hemisphere')
                base=32
                destinations=list(range(base,base+16))
                self.sxm(permute,src['hemisphere'],'permute',list(range(16,32)),destinations,[tile*8+i%8 for i in range(32)],counts[0],32,row=9+lane if lane<7 else 7,waves=counts[1]*counts[2],wave_interval=counts[0]*32,map_stride=8 if full_words else 0)
                if not direct:
                    beat=permute+15
                    self.vxm(beat-1,queue=4,lhs='stream_bf16',lhs_index=2*lane,source=src['hemisphere'],output=dest['hemisphere'],count=n,interval=32)
                    self.vxm(beat-1,queue=5,lhs='previous',output=dest['hemisphere'],format_='bf16',out=0,count=n,interval=32)
                    result=beat+((19 if src['hemisphere']=='west' else 12) if dest['hemisphere']=='east' else (24 if src['hemisphere']=='west' else 17))
                for plane in range(2):
                    cycle=permute+14-dest['slices'][2*pair+plane]//self.group if direct else result+dest['slices'][2*pair+plane]//self.group+1
                    self.mem(cycle,dest,2*pair+plane,'write',dest['base_row']+int(offsets[selected][0]),32+2*lane+plane if direct else plane,counts,dst_strides,intervals)
                self.cursor=permute+n*32+32

    def packed_copy(self, source, dest, words):
        beat=self.cursor+32
        n=len(source['slices'])
        for plane in range(n):
            self.mem(beat-(source['slices'][plane]//self.group+1),source,plane,'read',source['base_row'],32+plane,
                (words,1,1),(1,0,0),(self.period,1,1))
        self.vxm(beat-1,lhs='stream_f32' if n==4 else 'stream_bf16',source=source['hemisphere'],count=words)
        self.vxm(beat-1,queue=1,lhs='previous',output=dest['hemisphere'],format_='fp32' if n==4 else 'bf16',out=0,count=words)
        result=beat+(32 if source['hemisphere']=='west' else 17)
        for plane in range(n):
            self.mem(result+dest['slices'][plane]//self.group+1,dest,plane,'write',dest['base_row'],plane,
                (words,1,1),(1,0,0),(self.period,1,1))
        self.cursor=result+words*self.period+96

    def module(self, bindings, arguments=()):
        dummy='%dummy: tensor<1x32xf32>'
        args=', '.join([dummy]+list(arguments))
        return 'module attributes {ftlpu.target = '+attr(self.target)+'} {\nfunc.func @resident('+args+') {\n'+\
            '\n'.join(bindings+self.lines)+ '\nreturn\n}\n}\n'


class ResidentLinker:
    """Link operator-owned schedules before binary encoding.

    Intermediate tensors and static inputs occupy bank-1 ranges disjoint from
    all native mutable workspaces. Slice roles come from the target schedules.
    Lifetime allocation follows graph edges. This intentionally serial first
    implementation does not claim C2C overlap or optimized layout traffic.
    """
    def __init__(self, graph, finals, direct_initial_inputs=(), direct_edges=(), raw_sr_copies=False):
        self.graph,self.finals=graph,list(finals)
        first=graph.nodes[0]['schedule'].read_text().splitlines()[0]
        target=parse_attr(first[first.index('ftlpu.target = ')+15:-3])
        self.schedule=Schedule(target)
        self.schedule.raw_sr_copies=raw_sr_copies
        self.direct_outputs={}
        for producer,consumer in direct_edges:
            ni,idx=consumer
            value,dtype=graph.nodes[ni]['inputs'][idx]
            if not isinstance(value,Value) or ni<=producer[0] or dtype!=value.dtype or (value.node,value.output)!=tuple(producer) or value.owners is not None or not np.array_equal(value.indices,np.arange(value.size)):
                raise ValueError('direct edge requires a complete producer tensor')
            bindings=[p[1] for line in graph.nodes[ni]['schedule'].read_text(encoding='utf-8').splitlines()
                      if (p:=operation(line)) and p[0]=='binding' and p[1]['access']=='input' and p[1]['index']==idx]
            self.direct_outputs[tuple(producer)]=copy.deepcopy(bindings[0]['placement'])
        self.bindings,self.arguments,self.inputs=[],[],[]
        self.cache,self.live,self.static_ranges={},{},{}
        self.internal_index=1000
        self.static_cache={}
        self.static_values={}
        self.weight_inputs={}
        self.last_projection_input=None
        self.direct_initial_inputs=set()
        for ni,idx in direct_initial_inputs:
            if ni!=0:
                raise ValueError('direct initial input must belong to the first operator')
            value,dtype=graph.nodes[ni]['inputs'][idx]
            if isinstance(value,Value) or dtype=='i8':
                raise ValueError('direct initial input must be a resident external activation')
            if any(v is value for node in graph.nodes[1:] for v,_ in node['inputs']):
                raise ValueError('direct initial input cannot have later external consumers')
            bindings=[p[1] for line in graph.nodes[ni]['schedule'].read_text(encoding='utf-8').splitlines()
                      if (p:=operation(line)) and p[0]=='binding' and p[1]['access']=='input' and p[1]['index']==idx]
            placement=bindings[0]['placement']
            # All paged weights can be prefetched before this consumer.
            # Do not alias any of their physical storage with its input.
            for node in graph.nodes:
                if not node.get('schedule'):continue
                for line in node['schedule'].read_text(encoding='utf-8').splitlines():
                    p=operation(line)
                    if not p or p[0]!='binding' or p[1]['access']!='input':continue
                    w=p[1]['placement']
                    if w['kind']!='w8a16_mxm_weight_striped' or placement.get('bank',0)!=0:continue
                    same_side=placement['hemisphere']=='both' or w['hemisphere']=='both' or placement['hemisphere']==w['hemisphere']
                    if same_side and set(placement['slices'])&set(w['slices']) and placement['base_row']<w['instruction_count'] and placement['base_row']+placement['instruction_count']>0:
                        raise ValueError('direct initial input overlaps paged weight storage')
            self.direct_initial_inputs.add((id(value),dtype,attr(placement)))
        # Immutable inputs and archived outputs must survive every native
        # operator's mutable workspace, whose slice roles depend on target.
        # Derive touched row ranges from Schedule domains, not fixed slice IDs.
        self.native_ranges={}
        for node in graph.nodes:
            if not node.get('schedule'):continue
            for line in node['schedule'].read_text(encoding='utf-8').splitlines():
                parsed=operation(line)
                if not parsed or parsed[0]!='mem_transfer':continue
                v=parsed[1]
                if v.get('bank',0)!=1 or v['opcode'] not in ('write','write_tap'):continue
                low=high=v['address']
                for count,stride in (('repeat_count','address_stride'),('wave_count','wave_address_stride'),('group_count','group_address_stride')):
                    delta=(v.get(count,1)-1)*v.get(stride,0)
                    low+=min(0,delta);high+=max(0,delta)
                key=('west' if v['hemisphere'] else 'east',v['slice'])
                self.native_ranges.setdefault(key,[]).append((low,high+1))
        self.needed={};self.last={};self.consumer_copy_groups={}
        for i,node in enumerate(graph.nodes):
            for value,_ in node['inputs']:
                if isinstance(value,Value):
                    for key,indices,_ in value.segments():
                        self.last[key]=i
                        self.needed.setdefault(key,set()).update(map(int,indices))
            if raw_sr_copies and node.get('schedule'):
                for line in node['schedule'].read_text(encoding='utf-8').splitlines():
                    p=operation(line)
                    if not p or p[0]!='binding' or p[1]['access']!='input':continue
                    value,_=node['inputs'][p[1]['index']]
                    if isinstance(value,Value):
                        for key,_,_ in value.segments():
                            self.consumer_copy_groups.setdefault(key,set()).update(
                                self.copy_groups([p[1]['placement']]))
        for value in finals:
            for key,indices,_ in value.segments():
                self.last[key]=len(graph.nodes)
                self.needed.setdefault(key,set()).update(map(int,indices))
        for key,dest in self.direct_outputs.items():
            for node in graph.nodes[key[0]+1:min(self.last[key]+1,len(graph.nodes))]:
                if not node.get('schedule'):continue
                for line in node['schedule'].read_text(encoding='utf-8').splitlines():
                    p=operation(line)
                    if not p or p[0]!='mem_transfer':continue
                    v=p[1];side='west' if v['hemisphere'] else 'east'
                    if v['opcode'] not in ('write','write_tap') or v.get('bank',0)!=dest.get('bank',0) or v['slice'] not in dest['slices'] or dest['hemisphere'] not in (side,'both'):continue
                    low=high=v['address']
                    for count,stride in (('repeat_count','address_stride'),('wave_count','wave_address_stride'),('group_count','group_address_stride')):
                        delta=(v.get(count,1)-1)*v.get(stride,0);low+=min(0,delta);high+=max(0,delta)
                    if low<dest['base_row']+dest['instruction_count'] and dest['base_row']<high+1:
                        raise ValueError('direct consumer allocation is overwritten during the producer lifetime')

    def copy_groups(self,placements):
        """MEM groups with no direct same-fabric SR U-turn route."""
        return {(h,s//self.schedule.group) for p in placements
                for h in ('east','west') if p['hemisphere'] in (h,'both')
                for s in p['slices']}

    def value_copy_groups(self,value):
        return self.copy_groups([self.cache[key] for key,_,_ in value.segments()]) \
            if self.schedule.raw_sr_copies else set()

    def allocate(self,key,shape,dtype,hemisphere="west",packed=None,avoid_groups=()):
        if packed is None: packed=shape[-1]%32==0
        rows=int(np.prod(shape))//(32 if packed else 1); planes=4 if dtype=='f32' else 2
        depth=self.schedule.target['memory']['sram_depth_rows']
        start=depth//2
        pairs=(list(range(32,self.schedule.slices-planes+1,2))+list(range(8,32-planes+1,2))+list(range(0,8-planes+1,2))) if hemisphere=='west' else (list(range(32,self.schedule.slices-planes+1,2))+list(range(8,32-planes+1,2)))
        # Prefer a directed SR route without making available SRAM unusable.
        # If only a same-group placement fits, retain the existing bridge.
        pairs.sort(key=lambda first:any((hemisphere,s//self.schedule.group) in avoid_groups
                                       for s in range(first,first+planes)))
        for first in pairs:
            # Native projection results occupy bank 1 on slices 28/29.
            # Keep their workspace disjoint from archives on both sides.
            projection_workspace=first<30 and first+planes>28
            lower=(4096 if first>=48 else 2048 if first<8 or projection_workspace or (first<=25 and first+planes>24) else 0) if hemisphere=='west' else (start if first>=32 else 2048 if projection_workspace else 0)
            candidates={lower}
            for p in self.live.values(): candidates.add(p['base_row']+p['instruction_count'])
            native=[r for slice_ in range(first,first+planes) for r in self.native_ranges.get((hemisphere,slice_),[])]
            candidates.update(end for _,end in native)
            for row in sorted(candidates):
                if row<lower:continue
                if row+rows>depth: continue
                if any(row<end and begin<row+rows for begin,end in native):continue
                if any(p['hemisphere'] in (hemisphere,'both') and set(range(first,first+planes)) & set(p['slices']) and
                    row<p['base_row']+p['instruction_count'] and p['base_row']<row+rows for p in self.live.values()): continue
                p=dict(kind=('fp32_quad_planar' if planes==4 else 'fp16_pair_planar') if packed else ('fp32_scalar_broadcast_planar' if planes==4 else 'fp16_scalar_broadcast_planar'),
                    slices=list(range(first,first+planes)),base_row=row,bank=1,hemisphere=hemisphere,
                    instruction_count=rows,address_stride=1)
                self.cache[key]=p;self.live[key]=p
                p['_shape']=tuple(shape)
                return p
        (self.graph.directory/'sram_allocation_failure.json').write_text(json.dumps(
            {str(k):p for k,p in self.live.items()},indent=2),encoding='utf-8')
        raise ValueError(f'resident values exceed reserved SRAM: {key} {shape}')

    def binding(self,name,shape,dtype,placement,access,source=None,index=None):
        if index is None:
            index=self.internal_index;self.internal_index+=1
        tensor=CompiledType(shape,dtype)
        placement={k:v for k,v in placement.items() if not k.startswith('_')}
        values=dict(access=access,index=index,role=('weight' if placement.get('paged_weight') else 'activation') if access=='input' else 'result',
            name=name,bytes=int(np.prod(shape))*(4 if dtype=='f32' else 1 if dtype=='i8' else 2),placement=placement)
        argument=(source+' ') if source else ''
        operand_type=f'({tensor})' if source else '()'
        self.bindings.append(f'%resident_bind{len(self.bindings)} = ftlpu.schedule.binding {argument}{attr(values)} : {operand_type} -> {tensor}')

    def static_input(self,value,dtype,placement,name):
        p=copy.deepcopy(placement)
        identity=self.static_identity(value,dtype,p)
        direct=(id(value),dtype,attr(p)) in self.direct_initial_inputs
        if direct:identity=('direct_initial',id(value),identity)
        if identity in self.static_cache:
            result=self.static_cache[identity];self.static_values[(id(value),dtype)]=result
            return result
        if direct:
            # C2C writes straight into the first consumer's native allocation.
            # Its internal scratch retains the original banks and addresses.
            index=len(self.inputs)
            self.inputs.append((value,dtype));self.arguments.append(f'%input{index}: {CompiledType(value.shape,dtype)}')
            self.binding(name,value.shape,dtype,p,'input',f'%input{index}',index)
            result=(p,index,f'%resident_bind{len(self.bindings)-1}')
            self.static_cache[identity]=result;self.static_values[(id(value),dtype)]=result
            return result
        if dtype=='i8' and p['kind']=='w8a16_mxm_weight_striped':
            index=len(self.inputs)
            # Weight storage is separate from bank-1 activation archives.
            # Bank 0 is released by the previous operator's entire domain.
            p.update(bank=0,base_row=0,paged_weight=True,page_count=1,
                page_rows=p['instruction_count'],page_granularity=value.shape[1]//64,
                page_role_group_base=0,page_role_group_count=1,
                page_items_per_slice_group=value.shape[1]//64,page_bank_count=2,page_storage_slices=p['slices'])
            self.inputs.append((value,dtype));self.arguments.append(f'%input{index}: {CompiledType(value.shape,dtype)}')
            self.binding(name,value.shape,dtype,p,'input',f'%input{index}',index)
            result=(p,index,f'%resident_bind{len(self.bindings)-1}')
            self.static_cache[identity]=result;self.weight_inputs[index]=p
            return result
        if 'scalar_broadcast' in p['kind'] or p['kind'] in ('fp16_pair_planar','fp32_quad_planar'):
            p=self.allocate(('static',len(self.inputs)),value.shape,dtype,p['hemisphere'],packed='scalar_broadcast' not in p['kind'],
                            avoid_groups=self.copy_groups([p]) if self.schedule.raw_sr_copies else ())
            self.last[('static',len(self.inputs))]=len(self.graph.nodes)+1
            index=len(self.inputs);self.inputs.append((value,dtype))
            self.arguments.append(f'%input{index}: {CompiledType(value.shape,dtype)}')
            self.binding(name,value.shape,dtype,p,'input',f'%input{index}',index)
            result=(p,index,f'%resident_bind{len(self.bindings)-1}')
            self.static_cache[identity]=result
            self.static_values[(id(value),dtype)]=result
            return result
        keys=[(h,s) for h in ('east','west') if p['hemisphere'] in (h,'both') for s in p['slices']]
        row=max([512]+[self.static_ranges.get(key,512) for key in keys]
                +[end for key in keys for _,end in self.native_ranges.get(key,[])])
        count=p['instruction_count']
        if row+count>self.schedule.target['memory']['sram_depth_rows']:
            raise ValueError(f'static resident input exceeds reserved area: {name}')
        for key in keys:self.static_ranges[key]=row+count
        p['bank']=1;p['base_row']=row
        index=len(self.inputs)
        # Distributed immutable inputs are uploaded before execution too.
        # Reserve their ranges against bank-1 archives on either fabric.
        key=('static',index)
        self.live[key]=p
        self.last[key]=len(self.graph.nodes)+1
        self.inputs.append((value,dtype))
        self.arguments.append(f'%input{index}: {CompiledType(value.shape,dtype)}')
        self.binding(name,value.shape,dtype,p,'input',f'%input{index}',index)
        result=(p,index,f'%resident_bind{len(self.bindings)-1}')
        self.static_cache[identity]=result
        self.static_values[(id(value),dtype)]=result
        return result

    @staticmethod
    def static_identity(value,dtype,p):
        if 'scalar_broadcast' in p['kind']:
            # One stored constant can feed either fabric through a native
            # bridge; duplicating it at every consumer exhausts SRAM.
            return (hashlib.sha256(np.asarray(value).tobytes()).digest(),tuple(value.shape),dtype,p['kind'])
        return (hashlib.sha256(np.asarray(value).tobytes()).digest(),tuple(value.shape),dtype,p['kind'],tuple(p['slices']),p['hemisphere'])

    @staticmethod
    def packed_words(shape,indices):
        rows,columns=shape
        return (np.asarray(indices)%columns//32)*rows+np.asarray(indices)//columns

    def move(self,source,dest,source_shape,dest_shape,indices,positions):
        s=self.schedule
        same=lambda p:{k:v for k,v in p.items() if not k.startswith('_')}
        if same(source)==same(dest) and source_shape==dest_shape and np.array_equal(indices,positions):return
        distributed='fp16_mxm_distributed_16'
        if source['kind']==distributed or dest['kind']==distributed:
            if 'scalar_broadcast' in source['kind'] or 'scalar_broadcast' in dest['kind']:
                raise ValueError('distributed RMSNorm bridge requires a packed tensor view')
            indices=np.asarray(indices);positions=np.asarray(positions)
            chosen=np.flatnonzero(indices%32==0)
            if len(chosen)*32!=len(indices) or np.any(positions[chosen]%32):
                raise ValueError('distributed RMSNorm bridge requires complete aligned feature words')
            ix=indices[chosen];pos=positions[chosen]
            src_rows=ix//source_shape[1];dst_rows=pos//dest_shape[1]
            # Each token lane owns a MEM pair; each SRAM word contains all
            # 32 feature lanes. Copy those words using geometry-derived 3-D
            # domains, with no numerical evaluation or FU-trace compression.
            for src_lane in range(8 if source['kind']==distributed else 1):
                for dst_lane in range(8 if dest['kind']==distributed else 1):
                    selected=np.ones(len(ix),dtype=bool)
                    if source['kind']==distributed:selected &= src_rows%8==src_lane
                    if dest['kind']==distributed:selected &= dst_rows%8==dst_lane
                    if not np.any(selected):continue
                    a,b=ix[selected],pos[selected]
                    def word_geometry(p,shape,logical,lane):
                        if p['kind']==distributed:
                            row=logical//shape[1]
                            words=(row//32*(shape[1]//32)+logical%shape[1]//32)*4+row%32//8
                            return dict(p,slices=p['slices'][2*lane:2*lane+2]),words
                        if p['kind']!='fp16_pair_planar':
                            raise ValueError('unsupported packed layout for distributed RMSNorm bridge: '+p['kind'])
                        return p,self.packed_words(shape,logical)
                    src,words=word_geometry(source,source_shape,a,src_lane)
                    dst,offsets=word_geometry(dest,dest_shape,b,dst_lane)
                    for dst_hemi in ('east','west') if dst['hemisphere']=='both' else (dst['hemisphere'],):
                        src_hemi=dst_hemi if src['hemisphere']=='both' else src['hemisphere']
                        if s.raw_sr_copies and src['hemisphere']=='both' and any(a//s.group==b//s.group for a,b in zip(src['slices'],dst['slices'])):
                            src_hemi='west' if dst_hemi=='east' else 'east'
                        s.scalar_copy(dict(src,hemisphere=src_hemi),dict(dst,hemisphere=dst_hemi),words,offsets)
            return
        scalar_source='scalar_broadcast' in source['kind']
        scalar_dest='scalar_broadcast' in dest['kind']
        if scalar_source and scalar_dest:
            s.scalar_copy(source,dest,indices,positions)
        elif not scalar_source and scalar_dest:
            s.unpack_view(source,dest,source_shape,indices,positions)
        elif not scalar_source:
            columns=source_shape[1];rows=source_shape[0]
            for hemi in ('east','west') if source['hemisphere']=='both' else (source['hemisphere'],):
                chosen=np.flatnonzero((np.asarray(indices)%columns//32)%2==(hemi=='west')) if source['kind']=='fp16_hemisphere_pair_planar' else np.arange(len(indices))
                # Select each complete 32-lane word once, preserving lanes.
                chosen=chosen[np.asarray(indices)[chosen]%32==0]
                if not len(chosen):continue
                ix=np.asarray(indices)[chosen];pos=np.asarray(positions)[chosen]
                if np.any(pos%dest_shape[1]%32):raise ValueError('unaligned packed destination view')
                srcwords=(ix%columns//(64 if source['kind']=='fp16_hemisphere_pair_planar' else 32))*rows+ix//columns
                dstwords=self.packed_words(dest_shape,pos)
                for dsthemi in ('east','west') if dest['hemisphere']=='both' else (dest['hemisphere'],):
                    s.scalar_copy(dict(source,hemisphere=hemi),dict(dest,hemisphere=dsthemi),srcwords,dstwords)
        else:
            positions=np.asarray(positions);indices=np.asarray(indices)
            rr=np.unique(positions//dest_shape[1]);cc=np.unique(positions%dest_shape[1])
            if len(rr)*len(cc)!=len(positions) or len(cc)%32 or np.any(np.diff(cc)!=1):
                raise ValueError('scalar packing requires a rectangular aligned view')
            order=np.argsort(positions)
            first=positions[order].reshape(len(rr),len(cc))[:,::32]
            offsets=self.packed_words(dest_shape,first.T.reshape(-1))
            s.pack(source,dest,(len(rr),len(cc)),indices[order],offsets)

    def materialize(self,value,dest):
        for key,indices,positions in value.segments():
            self.move(self.cache[key],dest,self.graph.nodes[key[0]]['outputs'][key[1]][0],value.shape,indices,positions)

    def archive_node_outputs(self,ni,native_outputs):
        node=self.graph.nodes[ni]
        for oi,(shape,dtype) in enumerate(node['outputs']):
            key=(ni,oi)
            if key not in self.last:continue
            source=native_outputs[oi]
            if key in self.direct_outputs:
                dest=copy.deepcopy(self.direct_outputs[key]);dest['_shape']=tuple(shape)
                self.cache[key]=dest;self.live[key]=dest
                self.move(source,dest,shape,shape,np.arange(np.prod(shape)),np.arange(np.prod(shape)))
                continue
            archive_hemi='east' if dtype=='f32' or node['name'].startswith('key_l2') else 'west'
            avoid=self.copy_groups([source])|self.consumer_copy_groups.get(key,set()) if self.schedule.raw_sr_copies else set()
            dest=self.allocate(key,shape,dtype,archive_hemi,packed='scalar_broadcast' not in source['kind'],avoid_groups=avoid)
            self.move(source,dest,shape,shape,np.arange(np.prod(shape)),np.arange(np.prod(shape)))

    def build(self):
        s=self.schedule
        masks=np.zeros((4,32),np.float32)
        for tile in range(4): masks[tile,tile*8:(tile+1)*8]=1
        maskp=dict(kind='fp32_quad_planar',slices=[48,49,50,51],base_row=0,bank=1,hemisphere='east',instruction_count=4,address_stride=1)
        self.inputs.append((masks,'f32'));self.arguments.append('%input0: tensor<4x32xf32>')
        self.binding('layout_tile_masks',masks.shape,'f32',maskp,'input','%input0',0)
        zero=np.zeros((1,1),np.float32)
        zero_source=self.static_input(zero,'f32',dict(kind='fp32_scalar_broadcast_planar',
            slices=[48,49,50,51],hemisphere='east',bank=1,base_row=0,instruction_count=1,address_stride=1),'state_zero_word')[0]
        # Reserve every initial constant before allocating transient outputs:
        # all inputs are transferred before the continuous program starts.
        for node in self.graph.nodes:
            if node.get('layout_join'):continue
            for line in node['schedule'].read_text(encoding='utf-8').splitlines():
                parsed=operation(line)
                if not parsed or parsed[0]!='binding' or parsed[1]['access']!='input':continue
                v=parsed[1];value,dtype=node['inputs'][v['index']]
                if isinstance(value,Value) or (node['name'].startswith('delta_rule') and v['index']==5):continue
                self.static_input(value,dtype,v['placement'],f'{node["name"]}.input{v["index"]}')
        parallel_start=None;parallel_end=0;parallel_outputs=[];prepared=set()
        for ni,node in enumerate(self.graph.nodes):
            print(f'link {ni:02d} {node["name"]}',flush=True)
            group=node.get('parallel_group')
            if group and (ni==0 or self.graph.nodes[ni-1].get('parallel_group')!=group):
                # Materialize every independent operand before either chain
                # starts; a layout bridge must not borrow active VALU queues.
                for pi in range(ni,len(self.graph.nodes)):
                    member=self.graph.nodes[pi]
                    if member.get('parallel_group')!=group:break
                    for line in member['schedule'].read_text(encoding='utf-8').splitlines():
                        p=operation(line)
                        if not p or p[0]!='binding' or p[1]['access']!='input':continue
                        idx=p[1]['index'];value,_=member['inputs'][idx]
                        placement=p[1]['placement']
                        if isinstance(value,Value):
                            if any(key[0]>=ni for key,_,_ in value.segments()):
                                raise ValueError('parallel operators cannot depend on one another')
                            self.materialize(value,placement);prepared.add((pi,idx))
                        elif 'scalar_broadcast' in placement['kind'] or placement['kind'] in ('fp16_pair_planar','fp32_quad_planar','fp16_mxm_distributed_16'):
                            prior=self.static_values[(id(value),member['inputs'][idx][1])][0]
                            self.move(prior,placement,value.shape,value.shape,np.arange(value.size),np.arange(value.size))
                            prepared.add((pi,idx))
                parallel_start=s.cursor+64;parallel_end=parallel_start;parallel_outputs=[]
            projection=node['name'].startswith(('in_proj_','out_proj','gate_proj','up_proj','down_proj'))
            if not projection:self.last_projection_input=None
            if node.get('layout_join'):
                value=node['inputs'][0][0]
                dest=self.allocate((ni,0),value.shape,value.dtype,node['hemisphere'],packed=True,
                                   avoid_groups=self.value_copy_groups(value))
                self.materialize(value,dest)
                for key in list(self.live):
                    if self.last.get(key)==ni:self.live.pop(key)
                continue
            lines=node['schedule'].read_text(encoding='utf-8').splitlines()
            args={a:t for a,t in re.findall(r'(%arg\d+): (tensor<[^>]+>)',lines[1])}
            argument_map={};relocated={};native_outputs={};operations=[]
            for line in lines[2:]:
                parsed=operation(line)
                if not parsed: continue
                kind,v,pre,post=parsed
                if kind=='binding':
                    old_result=re.match(r'\s*(%\w+)\s*=',line).group(1)
                    oldarg=re.search(r'\bbinding\s+(%\w+)',pre)
                    if v['access']=='input':
                        idx=v['index'];value,dtype=node['inputs'][idx]
                        placement=v['placement']
                        prior_static=self.static_values.get((id(value),dtype)) if not isinstance(value,Value) else None
                        # RMSNorm intentionally reuses activation slice IDs
                        # for internal transpose scratch. Materialize its
                        # external input at the original allocation; blindly
                        # relocating all reads would also retarget scratch.
                        if isinstance(value,Value) or (prior_static and ('scalar_broadcast' in placement['kind'] or placement['kind'] in ('fp16_pair_planar','fp32_quad_planar','fp16_mxm_distributed_16'))):
                            if isinstance(value,Value):
                                identity=(value.shape,hashlib.sha256(value.indices.tobytes()+(value.owners.tobytes() if value.owners is not None else str((value.node,value.output)).encode())).digest(),attr(placement))
                                if (ni,idx) not in prepared and not (projection and idx==0 and self.last_projection_input==identity):self.materialize(value,placement)
                                if projection and idx==0:self.last_projection_input=identity
                            elif prior_static[0]!=placement and (ni,idx) not in prepared:
                                self.move(prior_static[0],placement,value.shape,value.shape,np.arange(value.size),np.arange(value.size))
                            v['access']='internal'
                            if oldarg:argument_map[oldarg.group(1)]=f'%s{ni}_{old_result[1:]}'
                            pre=pre.replace(oldarg.group(1),'') if oldarg else pre
                            post=re.sub(r': \([^)]*\)',': ()',post,count=1)
                            v['index']=self.internal_index;self.internal_index+=1
                        elif node['name'].startswith('delta_rule') and idx==5:
                            # Initialize at its actual lifetime, after earlier
                            # projections have released this bank. An initial
                            # host fill would be overwritten by their pages.
                            count=v['placement']['instruction_count']
                            s.scalar_copy(zero_source,v['placement'],np.zeros(count,np.int64),np.arange(count))
                            v['access']='internal'
                            if oldarg:argument_map[oldarg.group(1)]=f'%s{ni}_{old_result[1:]}'
                            pre=pre.replace(oldarg.group(1),'') if oldarg else pre
                            post=re.sub(r': \([^)]*\)',': ()',post,count=1)
                            v['index']=self.internal_index;self.internal_index+=1
                        else:
                            newp,index,binding_name=self.static_input(value,dtype,placement,f'{node["name"]}.input{idx}')
                            relocated[idx]=(placement,newp)
                            if oldarg:argument_map[oldarg.group(1)]=f'%input{index}'
                            # The globally declared binding already exists.
                            argument_map[old_result]=binding_name
                            continue
                    else:
                        if v['access']=='output': native_outputs[v['index']]=copy.deepcopy(v['placement'])
                        v['access']='internal';v['index']=self.internal_index;self.internal_index+=1
                operations.append((kind,v,pre,post))
            # One MEM ICU owns the whole synchronized page write. Leave a
            # geometry-derived admission window even for adjacent projection
            # pages whose activation buffer is reused without another pack.
            # Real DDR latency is absorbed by WRITE_SYNC/page-ready SYNC.
            page_rows=max((p['page_rows'] for _,p in relocated.values()
                if p.get('paged_weight')),default=0)
            s.cursor+=page_rows+32 if page_rows else 0
            start=parallel_start if group else s.cursor+64;end=start
            weight_ranges={}
            for kind,v,pre,post in operations:
                # This linked image owns its final SRAM addresses and embeds
                # the checkpoint's dequantization scale. Stage-local binding
                # indices must not survive as package relocations.
                v.pop('scale_binding',None)
                if kind=='mem_transfer':
                    v.pop('address_binding',None)
                    v.pop('address_binding_access',None)
                if kind=='mem_transfer' and v['opcode']=='read':
                    for idx,(old,new) in relocated.items():
                        hemi='west' if v['hemisphere'] else 'east'
                        if old['hemisphere'] not in (hemi,'both') or v['slice'] not in old['slices'] or v.get('bank',0)!=old.get('bank',0):continue
                        v['bank']=new['bank'];v['address']+=new['base_row']-old['base_row']
                        if new.get('paged_weight'):
                            index=self.static_cache[self.static_identity(node['inputs'][idx][0],node['inputs'][idx][1],old)][1]
                            v.update(address_binding=index,weight_page=0)
                            v['runtime_prefetch']=True
                            weight_ranges[index]=new
                if 'cycle' in v:
                    v['cycle']+=start
                    span=(v.get('repeat_count',1)-1)*v.get('repeat_interval',1)+(v.get('wave_count',1)-1)*v.get('wave_interval',1)+(v.get('group_count',1)-1)*v.get('group_interval',1)
                    end=max(end,v['cycle']+span+32)
                if kind=='timeline':
                    v['start']+=start;v['end']+=start;v['name']=node['name']+'.'+v['name']
                    end=max(end,v['end'])
                if 'ready_cycle' in v:v['ready_cycle']+=start
                v.pop('ftlpu.schedule_stage',None)
                line=pre+attr(v)+post
                def renamed(match):
                    name=match.group(0)
                    return argument_map.get(name,f'%s{ni}_{name[1:]}')
                line=re.sub(r'%\w+',renamed,line)
                # Mapped names are already global; the replacement does not
                # recursively apply to the replacement itself.
                s.lines.append(line)
            if group:
                parallel_end=max(parallel_end,end)
                parallel_outputs.append((ni,native_outputs))
                s.cursor=parallel_end+96
            else:s.cursor=end+96
            # ScheduleToCommand derives the page's complete consumption
            # interval from the tagged native MEM domains above.
            for key in list(self.live):
                if self.last[key]==ni: self.live.pop(key)
            if group:
                if ni+1==len(self.graph.nodes) or self.graph.nodes[ni+1].get('parallel_group')!=group:
                    for pi,outputs in parallel_outputs:self.archive_node_outputs(pi,outputs)
                    parallel_start=None
            else:self.archive_node_outputs(ni,native_outputs)
        for i,value in enumerate(self.finals):
            original=self.cache.get((value.node,value.output)) if value.owners is None else None
            if original and value.shape==original['_shape'] and np.array_equal(value.indices,np.arange(value.size)):
                p=original
            else:
                p=self.allocate(('final',i),value.shape,value.dtype,avoid_groups=self.value_copy_groups(value))
                self.materialize(value,p)
            self.binding(f'output{i}',value.shape,value.dtype,p,'output',index=i)
        return s.module(self.bindings,self.arguments)


def CompiledType(shape,dtype):
    return 'tensor<'+'x'.join(map(str,shape))+'x'+dtype+'>'


def affine_geometry(source,dest):
    """Recognize a rectangular tensor view's affine address domain.

    These arrays contain logical indices constructed from shapes/slices, not
    instructions or numerical tensor data. No FU schedule is generated first.
    """
    n=len(source)
    divisors=[i for i in range(1,n+1) if n%i==0]
    for inner in reversed(divisors):
        for wave in divisors:
            if n%(inner*wave):continue
            group=n//(inner*wave)
            counts=(inner,wave,group)
            strides=[];okay=True
            for values in (source,dest):
                si=int(values[1]-values[0]) if inner>1 else 0
                sw=int(values[inner]-values[0]) if wave>1 else 0
                sg=int(values[inner*wave]-values[0]) if group>1 else 0
                expected=int(values[0])+np.arange(inner)[None,None,:]*si+np.arange(wave)[None,:,None]*sw+np.arange(group)[:,None,None]*sg
                if not np.array_equal(expected.reshape(-1),values):okay=False;break
                strides.append((si,sw,sg))
            if okay:return counts,*strides
    return None
