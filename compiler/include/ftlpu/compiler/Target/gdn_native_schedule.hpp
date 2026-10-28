#pragma once
#include "ftlpu/compiler/Target/vector32_triangular_schedule.hpp"
#include <string>
#include <tuple>

namespace ftlpu::compiler {
// Experimental complete chunkwise lowering. Resident tensors use BF16 words
// of 32 lanes; the two hemispheres carry identical copies in this correctness
// baseline. Matrix products use FP32 VXM accumulation, not MXM. Epochs only
// bound iMEM occupancy: they never transfer intermediate tensors to the host.
struct GdnTensor {
    std::size_t pair, base, rows, cols;
    std::size_t words() const { return cols/32; }
    GdnTensor row(std::size_t r) const { return {pair,base+r*words(),1,cols}; }
};
struct GdnEpoch { std::string name; Vector32SolveSchedule program; };
class GdnNativeLowering {
    std::vector<GdnEpoch> epochs_;
    Vector32SolveSchedule p_;
    std::size_t t_=64;
    void begin() { p_={}; t_=64; }
    void finish(const std::string& name) {
        p_.end_cycle=t_+64; epochs_.push_back({name,std::move(p_)});
    }
    void mem(std::size_t pair, std::size_t time, std::size_t address,
             std::size_t count, std::int64_t stride, bool rhs, bool write=false) {
        for(std::size_t side=0;side<2;++side) for(std::size_t byte=0;byte<2;++byte) {
            const auto hop=(pair+byte)/hw::kMemSlicesPerGroup;
            IcuLoop3D loop{0,{count,1,1},{1,1,1}};
            auto a=MemIcuAddress3D::Affine(address,{stride,0,0});
            p_.mem.push_back({side,pair+byte,write?time+hop:time-hop,write
                ?MemIcuInstruction::Write3D(loop,a,StreamId::East(side*8+byte))
                :MemIcuInstruction::Read3D(loop,a,StreamId::West(side*16+(rhs?2:0)+byte))});
        }
    }
    void vxm(std::size_t queue,std::size_t time,VxmLaneAluInstruction i) {
        p_.vxm.push_back({queue,time,VxmCompactInstructionCodec::encode(queue,VxmChainDepth::Two,i)});
    }
    void binary_words(GdnTensor a,GdnTensor b,GdnTensor out,VxmAluOpcode op,
                      bool broadcast=false,bool immediate=false,float value=0) {
        const auto count=out.words();
        mem(a.pair,t_-1,a.base,count,1,false);
        if(!immediate) mem(b.pair,t_-1,b.base,count,broadcast?0:1,true);
        VxmLaneAluInstruction head{op,VxmLaneOperand::StreamBFloat16(),
            immediate?VxmLaneOperand::Imm(value):VxmLaneOperand::StreamBFloat16()};
        head.precision=VxmAluPrecision::Float32; head.repeat_count=count;
        VxmLaneAluInstruction tail{VxmAluOpcode::Bypass,VxmLaneOperand::Previous()};
        tail.precision=VxmAluPrecision::Float32; tail.repeat_count=count;
        tail.output_type=VxmCastTarget::BFloat16; tail.output_stream=0;
        vxm(0,t_-1,head); vxm(1,t_-1,tail);
        mem(out.pair,t_+(op==VxmAluOpcode::Multiply?3:2),out.base,count,1,false,true);
        t_+=count+48;
    }
    void mac_packets(std::size_t count) {
        VxmLaneAluInstruction head{VxmAluOpcode::Multiply,VxmLaneOperand::StreamBFloat16(),VxmLaneOperand::StreamBFloat16()};
        head.precision=VxmAluPrecision::Float32;head.repeat_count=count;vxm(0,t_-1,head);
        std::size_t time=t_-1;
        auto add=[&](std::size_t n,bool reset,bool emit){
            VxmLaneAluInstruction i{VxmAluOpcode::Add,VxmLaneOperand::Previous(),VxmLaneOperand::Acc()};
            i.precision=VxmAluPrecision::Float32;i.repeat_count=n;
            i.accumulator_write=true;i.accumulator_reset=reset;i.accumulator_emit=emit;
            if(emit){i.output_type=VxmCastTarget::BFloat16;i.output_stream=0;}
            vxm(1,time++,i);
        };
        if(count==1)add(1,true,true);
        else {add(1,true,false);if(count>2)add(count-2,false,false);add(1,false,true);}
    }
public:
    void initialize_constants() {
        begin();
        for(auto [pair,address,value]:std::vector<std::tuple<std::size_t,std::size_t,float>>{
                {6,0,1.f},{26,32,0.f},{28,0,0.f},{30,0,0.f},
                {32,0,0.f},{34,0,0.f},{36,0,0.f},{38,0,0.f},{40,0,0.f}}){
            VxmLaneAluInstruction head{VxmAluOpcode::Bypass,VxmLaneOperand::Imm(value),VxmLaneOperand::Imm(value)};
            head.precision=VxmAluPrecision::Float32;
            VxmLaneAluInstruction tail{VxmAluOpcode::Bypass,VxmLaneOperand::Previous()};
            tail.precision=VxmAluPrecision::Float32;
            tail.output_type=VxmCastTarget::BFloat16;tail.output_stream=0;
            vxm(0,t_-1,head);vxm(1,t_-1,tail);
            mem(pair,t_+2,address,1,0,false,true);t_+=64;
        }
        finish("constants.ones_zero");
        begin();
        const auto edge=hw::kMemGroups+hw::kC2cToSxmStreamRegisterColumns;
        for(std::size_t lane=0;lane<32;++lane){
            for(std::size_t side=0;side<2;++side){
                auto trans=SxmInstruction::Transpose({},{});
                for(std::size_t s=16;s<32;++s){trans.src_streams.push_back({s});trans.dst_streams.push_back({s});}
                // The physical permute encoding has no zero-fill selector.
                // Transpose one all-one row and seven all-zero rows, then
                // select their real lanes to construct a one-hot vector.
                SxmInstruction::PermuteMap map;map.fill(1);map[lane]=0;
                auto perm=SxmInstruction::Permute({}, {},map);
                for(std::size_t s=16;s<32;++s)perm.src_streams.push_back({s});
                for(std::size_t s=32;s<48;++s)perm.dst_streams.push_back({s});
                perm.output_row=SxmInstruction::kConsumeOutputRowBase;
                p_.sxm.push_back({side,t_,1,1,trans});p_.sxm.push_back({side,t_+5,1,1,perm});
                for(std::size_t byte=0;byte<2;++byte){
                    IcuLoop3D loop{0,{1,1,1},{1,1,1}};
                    const auto group=(6+byte)/hw::kMemSlicesPerGroup;
                    for(std::size_t row=0;row<8;++row){
                        const auto pair=row?26+2*row:6;
                        p_.mem.push_back({side,pair+byte,t_-edge+(pair+byte)/hw::kMemSlicesPerGroup,
                            MemIcuInstruction::Read3D(loop,MemIcuAddress3D::Affine(0,{0,0,0}),StreamId::East(16+2*row+byte))});
                    }
                    p_.mem.push_back({side,6+byte,t_+5+edge-group,
                        MemIcuInstruction::Write3D(loop,MemIcuAddress3D::Affine(1+lane,{0,0,0}),StreamId::West(byte))});
                }
            }
            t_+=64;
        }
        finish("constants.diagonal");
    }
    void binary(const std::string& name,GdnTensor a,GdnTensor b,GdnTensor out,
                VxmAluOpcode op,bool scale_rows=false,bool fixed_scale=false,
                bool immediate=false,float value=0) {
        begin();
        for(std::size_t r=0;r<out.rows;++r)
            binary_words(a.row(r),b.row(fixed_scale?0:r),out.row(r),op,scale_rows,immediate,value);
        finish(name);
    }
    void expand(const std::string& name,GdnTensor a,std::size_t dest) {
        // Split columns into epochs so transpose/permute iMEM stays bounded.
        const auto edge=hw::kMemGroups+hw::kC2cToSxmStreamRegisterColumns;
        for(std::size_t first=0;first<a.cols;first+=16) {
            begin();
            for(std::size_t c=first;c<std::min(a.cols,first+16);++c) {
                for(std::size_t side=0;side<2;++side) {
                    auto trans=SxmInstruction::Transpose({{16},{17}},{});
                    trans.input_row=SxmInstruction::kBroadcastInputRows;
                    for(std::size_t s=16;s<32;++s)trans.dst_streams.push_back({s});
                    SxmInstruction::PermuteMap map{};
                    for(std::size_t lane=0;lane<32;++lane)map[lane]=(c%32/8)*8+lane%8;
                    auto perm=SxmInstruction::Permute({}, {},map);
                    for(std::size_t s=16;s<32;++s)perm.src_streams.push_back({s});
                    for(std::size_t s=32;s<48;++s)perm.dst_streams.push_back({s});
                    perm.output_row=c%8<7?9+c%8:7;
                    p_.sxm.push_back({side,t_,a.rows,8,trans});p_.sxm.push_back({side,t_+5,a.rows,8,perm});
                    for(std::size_t byte=0;byte<2;++byte){
                        IcuLoop3D loop{0,{a.rows,1,1},{8,1,1}};
                        p_.mem.push_back({side,a.pair+byte,t_-edge+(a.pair+byte)/hw::kMemSlicesPerGroup,
                            MemIcuInstruction::Read3D(loop,MemIcuAddress3D::Affine(a.base+c/32,{static_cast<std::int64_t>(a.words()),0,0}),StreamId::East(16+byte))});
                        p_.mem.push_back({side,dest+byte,t_+5+edge-(dest+byte)/hw::kMemSlicesPerGroup,
                            MemIcuInstruction::Write3D(loop,MemIcuAddress3D::Affine(c,{static_cast<std::int64_t>(a.cols),0,0}),StreamId::West(c%8*2+byte))});
                    }
                }
                t_+=a.rows*8+64;
            }
            finish(name+".cols"+std::to_string(first));
        }
    }
    void product(const std::string& name,GdnTensor a,GdnTensor b,GdnTensor out) {
        if(a.cols!=b.rows||out.rows!=a.rows||out.cols!=b.cols)throw std::logic_error("GDN product dimensions");
        expand(name+".coefficients",a,50);
        for(std::size_t first=0;first<a.rows;first+=16){
            begin();
            for(std::size_t r=first;r<std::min(a.rows,first+16);++r)
                for(std::size_t block=0;block<out.words();++block){
                    mem(b.pair,t_-1,b.base+block,a.cols,b.words(),false);
                    mem(50,t_-1,r*a.cols,a.cols,1,true);
                    mac_packets(a.cols);
                    mem(out.pair,t_+a.cols+2,out.base+r*out.words()+block,1,0,false,true);
                    t_+=a.cols+48;
                }
            finish(name+".rows"+std::to_string(first));
        }
    }
    void solve(const std::string& name) {
        expand(name+".coefficients",{8,0,32,32},4);
        begin();
        for(std::size_t r=0;r<32;++r)for(std::size_t block=0;block<4;++block){
            if(r){mem(0,t_-1,block,r,4,false);mem(4,t_-1,r*32,r,1,true);}
            mem(2,t_+r-1,r*4+block,1,0,false);mem(6,t_+r-1,0,1,0,true);
            mac_packets(r+1);mem(0,t_+r+3,r*4+block,1,0,false,true);t_+=r+48;
        }
        finish(name);
    }
    void transpose_chunk_key(const std::string& name,GdnTensor k,GdnTensor kt) {
        // The producer supplies K once, in its normal packed row layout.
        // Expand its coefficients on chip, then gather each transposed row
        // with one-hot vectors. No second host-prepared K ingress is needed.
        if(k.rows!=32 || kt.cols!=32 || kt.rows!=k.cols)
            throw std::invalid_argument("GDN key transpose requires a full chunk32");
        expand(name+".coefficients",k,50);
        for(std::size_t first=0;first<k.cols;first+=16){
            begin();
            for(std::size_t c=first;c<std::min(k.cols,first+16);++c){
                mem(50,t_-1,c,32,k.cols,false);
                mem(6,t_-1,1,32,1,true);
                mac_packets(32);
                mem(kt.pair,t_+34,kt.base+c,1,0,false,true);
                t_+=80;
            }
            finish(name+".rows"+std::to_string(first));
        }
    }
    std::vector<GdnEpoch> lower(std::size_t tokens=64, bool broadcast_gates=false) {
        if(!tokens || tokens%32 || tokens>1024)
            throw std::invalid_argument("native GDN requires 1..32 complete chunk32 blocks");
        epochs_.clear();
        initialize_constants();
        for(std::size_t chunk=0;chunk<tokens/32;++chunk){
            const auto prefix="chunk"+std::to_string(chunk)+".";
            GdnTensor q{10,chunk*128,32,128},k{12,chunk*128,32,128},v{14,chunk*128,32,128},
                s{16,0,128,128},kt{18,chunk*128,128,32},gamma{24,0,32,32},rmat{26,0,32,32},
                temp{32,0,32,128},aux{34,0,32,128},beta{48,0,32,32};
            transpose_chunk_key(prefix+"key_transpose",k,kt);
            if(broadcast_gates){
                binary(prefix+"alpha",{20,chunk*32,32,32},{6,0,1,32},{46,0,32,32},VxmAluOpcode::Multiply,true,true);
                binary(prefix+"beta",{22,chunk*32,32,32},{6,0,1,32},{48,0,32,32},VxmAluOpcode::Multiply,true,true);
            }else{
                expand(prefix+"alpha",{20,chunk,1,32},46);expand(prefix+"beta",{22,chunk,1,32},48);
            }
            for(std::size_t row=0;row<32;++row){
                binary(prefix+"gamma"+std::to_string(row),row?gamma.row(row-1):GdnTensor{6,0,1,32},
                    {46,row,1,32},gamma.row(row),VxmAluOpcode::Multiply);
                binary(prefix+"decay_mul"+std::to_string(row),rmat.row(row?row-1:32),
                    {46,row,1,32},{32,0,1,32},VxmAluOpcode::Multiply);
                binary(prefix+"decay_diag"+std::to_string(row),{32,0,1,32},{6,1+row,1,32},rmat.row(row),VxmAluOpcode::Add);
            }
            product(prefix+"KKt",k,kt,{28,0,32,32});
            binary(prefix+"gram_decay",{28,0,32,32},rmat,{32,0,32,32},VxmAluOpcode::Multiply);
            binary(prefix+"negative_beta",beta,beta,{46,0,32,32},VxmAluOpcode::Multiply,false,false,true,-1);
            binary(prefix+"negative_L",{32,0,32,32},{46,0,32,32},{8,0,32,32},VxmAluOpcode::Multiply,true);
            product(prefix+"KS",k,s,{30,0,32,128});
            binary(prefix+"history_decay",{30,0,32,128},gamma,temp,VxmAluOpcode::Multiply,true);
            binary(prefix+"rhs_difference",v,temp,aux,VxmAluOpcode::Subtract);
            binary(prefix+"rhs_beta",aux,beta,{2,0,32,128},VxmAluOpcode::Multiply,true);
            solve(prefix+"solve");
            product(prefix+"QS",q,s,{38,0,32,128});
            binary(prefix+"output_history",{38,0,32,128},gamma,temp,VxmAluOpcode::Multiply,true);
            product(prefix+"QKt",q,kt,{36,0,32,32});
            binary(prefix+"attention_decay",{36,0,32,32},rmat,{28,0,32,32},VxmAluOpcode::Multiply);
            product(prefix+"attention_D",{28,0,32,32},{0,0,32,128},aux);
            binary(prefix+"output",temp,aux,{40,chunk*128,32,128},VxmAluOpcode::Add);
            expand(prefix+"last_decay",rmat.row(31),46);
            binary(prefix+"state_D",{0,0,32,128},{46,0,32,32},aux,VxmAluOpcode::Multiply,true);
            product(prefix+"state_update",kt,aux,{44,0,128,128});
            binary(prefix+"state_history",s,gamma.row(31),{42,0,128,128},VxmAluOpcode::Multiply,true,true);
            binary(prefix+"state_add",{42,0,128,128},{44,0,128,128},s,VxmAluOpcode::Add);
        }
        return std::move(epochs_);
    }
};
} // namespace ftlpu::compiler
