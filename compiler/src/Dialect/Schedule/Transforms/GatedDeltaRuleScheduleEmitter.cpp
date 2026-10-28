#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "DirectDomainEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/gdn_native_schedule.hpp"

#include <algorithm>
#include <functional>

namespace ftlpu::compiler::schedule {
namespace {

using ffn_detail::create_vxm;
using ffn_detail::create_vxm_with_terminal_controls;

mlir::DictionaryAttr placement(mlir::ArrayAttr values, int64_t index = 0)
{
    return llvm::cast<mlir::DictionaryAttr>(values[index])
        .getAs<mlir::DictionaryAttr>("placement");
}

int64_t integer(mlir::ArrayAttr values, int64_t index)
{
    return llvm::cast<mlir::IntegerAttr>(values[index]).getInt();
}

int64_t sliceAt(mlir::DictionaryAttr value, int64_t index)
{
    return llvm::cast<mlir::IntegerAttr>(
        value.getAs<mlir::ArrayAttr>("slices")[index]).getInt();
}

int64_t baseRow(mlir::DictionaryAttr value)
{
    return value.getAs<mlir::IntegerAttr>("base_row").getInt();
}

int64_t bank(mlir::DictionaryAttr value)
{
    if (const auto attr = value.getAs<mlir::IntegerAttr>("bank"))
        return attr.getInt();
    return 0;
}

bool isWest(mlir::DictionaryAttr value)
{
    const auto attr = value.getAs<mlir::StringAttr>("hemisphere");
    return attr && attr.getValue() == "west";
}

int64_t physicalQueue(mlir::DictionaryAttr value, int64_t slice,
    const target::LPUTargetModel& target)
{
    return slice + (isWest(value)
        ? target.memory().slices_per_hemisphere : 0);
}

llvm::StringRef inputHemisphere(mlir::DictionaryAttr value)
{
    return isWest(value) ? "west" : "east";
}

llvm::StringRef outputHemisphere(mlir::DictionaryAttr value)
{
    return isWest(value) ? "east" : "west";
}

void createInputBinding(mlir::IRRewriter& rewriter,
    stream::GatedDeltaRuleTaskOp op, mlir::Value value,
    llvm::StringRef role, mlir::DictionaryAttr valuePlacement)
{
    const auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
    if (!argument) return;
    const auto type = llvm::cast<mlir::RankedTensorType>(value.getType());
    const int64_t elementBytes = type.getElementType().isF32() ? 4 : 2;
    mlir::OperationState state(op.getLoc(), BindingOp::getOperationName());
    state.addOperands(value);
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr("index",
            rewriter.getI64IntegerAttr(argument.getArgNumber())),
        rewriter.getNamedAttr("access", rewriter.getStringAttr("input")),
        rewriter.getNamedAttr("role", rewriter.getStringAttr("activation")),
        rewriter.getNamedAttr("name", rewriter.getStringAttr(
            ("gated_delta_rule." + role).str())),
        rewriter.getNamedAttr("bytes", rewriter.getI64IntegerAttr(
            type.getNumElements() * elementBytes)),
        rewriter.getNamedAttr("placement", valuePlacement),
    });
    rewriter.create(state);
}

BindingOp createOutputBinding(mlir::IRRewriter& rewriter,
    stream::GatedDeltaRuleTaskOp op, mlir::Type type, int64_t index,
    llvm::StringRef role, mlir::DictionaryAttr valuePlacement)
{
    const auto tensor = llvm::cast<mlir::RankedTensorType>(type);
    const int64_t elementBytes = tensor.getElementType().isF32() ? 4 : 2;
    mlir::OperationState state(op.getLoc(), BindingOp::getOperationName());
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr("index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr("access", rewriter.getStringAttr("output")),
        rewriter.getNamedAttr("role", rewriter.getStringAttr("result")),
        rewriter.getNamedAttr("name", rewriter.getStringAttr(
            ("gated_delta_rule." + role).str())),
        rewriter.getNamedAttr("bytes", rewriter.getI64IntegerAttr(
            tensor.getNumElements() * elementBytes)),
        rewriter.getNamedAttr("placement", valuePlacement),
    });
    return llvm::cast<BindingOp>(rewriter.create(state));
}

class Emitter {
public:
    Emitter(mlir::IRRewriter& rewriter, stream::GatedDeltaRuleTaskOp op,
        const target::LPUTargetModel& target)
        : rewriter_(rewriter), op_(op), target_(target),
          query_(placement(op.getQueryAllocations())),
          key_(placement(op.getKeyAllocations())),
          value_(placement(op.getValueAllocations())),
          decay_(placement(op.getDecayAllocations())),
          beta_(placement(op.getBetaAllocations())),
          state_(placement(op.getStateAllocations())),
          decayed_(placement(op.getScratchAllocations(), 0)),
          product_(placement(op.getScratchAllocations(), 1)),
          remembered_(placement(op.getScratchAllocations(), 2)),
          delta_(placement(op.getScratchAllocations(), 3)),
          output_(placement(op.getOutputAllocations())),
          finalState_(placement(op.getFinalStateAllocations()))
    {
    }

    mlir::LogicalResult emit()
    {
        if(auto lowering=op_.getConfig().getAs<mlir::StringAttr>("lowering");
           lowering && lowering.getValue()=="native_chunkwise_bf16")
            return emitNative();
        if (const auto module = op_->getParentOfType<mlir::ModuleOp>())
            if (const auto spill = module->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_bf16_spill");
                spill && spill.getValue()
                && llvm::cast<mlir::RankedTensorType>(op_.getQuery().getType()).getDimSize(0) == 1)
                return emitFixedDecode();
        const auto function=op_->getParentOfType<mlir::func::FuncOp>();
        if (const auto policy=function->getAttrOfType<mlir::IntegerAttr>("ftlpu.vxm_queue_base"))
            queueBase_=policy.getInt();
        if (queueBase_ != 0 && queueBase_ != 8)
            return op_.emitError("recurrent VXM queue base must be 0 or 8");
        const auto qType = llvm::cast<mlir::RankedTensorType>(
            op_.getQuery().getType());
        const auto vType = llvm::cast<mlir::RankedTensorType>(
            op_.getValue().getType());
        if (op_.getKeyDim() % 32 || op_.getValueDim() % 32
            || qType.getDimSize(0) <= 0)
            return op_.emitError(
                "direct gated delta-rule schedule requires 32-aligned key/value dimensions");

        rewriter_.setInsertionPoint(op_);
        createInputBinding(rewriter_, op_, op_.getQuery(), "query", query_);
        createInputBinding(rewriter_, op_, op_.getKey(), "key", key_);
        createInputBinding(rewriter_, op_, op_.getValue(), "value", value_);
        createInputBinding(rewriter_, op_, op_.getDecay(), "decay", decay_);
        createInputBinding(rewriter_, op_, op_.getBeta(), "beta", beta_);
        createInputBinding(rewriter_, op_, op_.getInitialState(),
            "initial_state", state_);

        const int64_t tokens = qType.getDimSize(0);
        const int64_t heads = op_.getHeads();
        const int64_t keyDim = op_.getKeyDim();
        const int64_t valueDim = op_.getValueDim();
        const int64_t valueBlocks = valueDim / 32;
        const int64_t stateRows = heads * keyDim;
        const auto dataFormat = lpu_16bit_data_format(
            qType.getElementType());
        const auto streamKind = lpu_16bit_stream_kind(
            qType.getElementType());
        int64_t cursor = 96;

        heads_ = heads; blocks_ = valueBlocks; keyDim_ = keyDim;
        tokens_ = tokens; grid_ = true;
        // Each phase owns one complete head x value-block domain. No ICU
        // switches to another instruction before the domain finishes.
        for (int64_t token = 0; token < tokens; ++token) {
            const int64_t scalarAddress = token * heads;
            const int64_t keyAddress = token * heads * keyDim;
            cursor = binaryVectorPass(state_, 0, decay_, scalarAddress,
                decayed_, 0, keyDim, "multiply", "stream_f32", "stream_f32", cursor);
            cursor = reductionPass(decayed_, 0, key_, keyAddress,
                remembered_, 0, keyDim, streamKind, cursor, "fp32");
            cursor = binaryVectorPass(value_, token, remembered_, 0,
                delta_, 0, 1, "subtract", streamKind, "stream_f32", cursor);
            cursor = binaryVectorPass(delta_, 0, beta_, scalarAddress,
                remembered_, 0, 1, "multiply", "stream_f32", "stream_f32", cursor);
            cursor = binaryVectorPass(key_, keyAddress, remembered_, 0,
                product_, 0, keyDim, "multiply", streamKind, "stream_f32",
                cursor, "fp32", 1, 0, 1);
            cursor = binaryVectorPass(decayed_, 0, product_, 0,
                state_, 0, keyDim, "add", "stream_f32", "stream_f32", cursor);
            cursor = reductionPass(state_, 0, query_, keyAddress,
                output_, token, keyDim, streamKind, cursor, dataFormat);
        }
        grid_ = false;
        cursor = unaryCopy(state_, 0, finalState_, 0,
            stateRows * valueBlocks, cursor);

        mlir::OperationState timeline(
            op_.getLoc(), TimelineOp::getOperationName());
        timeline.addAttributes({
            rewriter_.getNamedAttr("name",
                rewriter_.getStringAttr("gated_delta_rule")),
            rewriter_.getNamedAttr("start", rewriter_.getI64IntegerAttr(0)),
            rewriter_.getNamedAttr("end", rewriter_.getI64IntegerAttr(cursor)),
        });
        rewriter_.create(timeline);
        auto outputBinding = createOutputBinding(rewriter_, op_,
            op_.getOutput().getType(), 0, "output", output_);
        auto stateBinding = createOutputBinding(rewriter_, op_,
            op_.getFinalState().getType(), 1, "final_state", finalState_);
        rewriter_.replaceOp(op_,
            {outputBinding.getValue(), stateBinding.getValue()});
        return mlir::success();
    }

private:
    mlir::LogicalResult emitFixedDecode() {
        // Fixed-port chain implementation of ONE recurrence step. Interface
        // state is FP32; every reload explicitly consumes its BF16 upper half.
        // Even head counts run independent head halves on the shared chains.
        // Results cross hemispheres on each pass; scratch ownership follows
        // that route rather than copying the state back between passes.
        if (target_.memory().bytes_per_word != 32
            || !llvm::cast<mlir::RankedTensorType>(op_.getQuery().getType()).getElementType().isBF16())
            return op_.emitError("fixed-chain decode requires LPU32 BF16 inputs");
        rewriter_.setInsertionPoint(op_);
        const mlir::DictionaryAttr inputs[] = {query_,key_,value_,decay_,beta_,state_};
        const char* names[] = {"query","key","value","decay","beta","initial_state"};
        for (int i=0;i<6;++i) createInputBinding(rewriter_,op_,op_->getOperand(i),names[i],inputs[i]);
        const int64_t heads=op_.getHeads(), dim=op_.getKeyDim(), blocks=op_.getValueDim()/32;
        if (dim%32 || op_.getValueDim()%32)
            return op_.emitError("fixed-chain decode requires 32-aligned dimensions");
        int64_t cursor=96;
        const auto partition = op_->getParentOfType<mlir::ModuleOp>()
            ->getAttrOfType<mlir::BoolAttr>("ftlpu.gdn_partition_heads");
        const bool partitioned = partition && partition.getValue() && heads % 2 == 0;
        const auto localHeads = partitioned ? heads / 2 : heads;
        bool flipped = false;
        auto mem=[&](mlir::DictionaryAttr p,int64_t address,int64_t count,int64_t stride,
                     int64_t cycle,bool write,bool fp32,int64_t port) {
            const int64_t first=(!write && fp32)?2:0, planes=write&&fp32?4:2;
            for(int64_t side=0;side<2;++side)for(int64_t byte=0;byte<planes;++byte){
                const auto slice=sliceAt(p,first+byte);
                int64_t offset = 0;
                if (partitioned && !write && side == (flipped ? 1 : 0)) {
                    if (p == query_ || p == key_) offset = localHeads * dim;
                    else if (p == value_) offset = localHeads * blocks;
                    else if (p == decay_ || p == beta_) offset = localHeads;
                }
                direct_domain_detail::emitMem3D(rewriter_,op_.getLoc(),target_,
                    cycle+(write?1:-1)*(slice/4),side*target_.memory().slices_per_hemisphere+slice,
                    write?"write":"read",baseRow(p)+address+offset,
                    write?(1-side)*8+byte:32+side*16+port+byte,
                    count,1,stride,1,1,0,1,1,0,-1,bank(p));
            }
        };
        auto alu=[&](int64_t cycle,int64_t queue,llvm::StringRef opcode,
                     llvm::StringRef lhs,llvm::StringRef rhs,int64_t count,
                     bool emit,bool reset,bool accumulate,bool bf16Output){
            auto v=create_vxm(rewriter_,op_.getLoc(),op_.getQuery(),op_.getKey(),op_.getOutput().getType(),
                cycle,queue,opcode,lhs,0,0.f,rhs,rhs=="stream_bf16"?2:0,0.f,
                bf16Output?"bf16":"fp32",emit?0:-1,count,1,"east","east",-1,
                reset,accumulate,!accumulate || emit,false,2);
            if(lhs=="stream_bf16")v->setAttr("lhs_stream_source",rewriter_.getStringAttr("local"));
            if(rhs=="stream_bf16")v->setAttr("rhs_stream_source",rewriter_.getStringAttr("local"));
        };
        auto pass=[&](mlir::DictionaryAttr lhs,int64_t la,bool lf,int64_t ls,
                      mlir::DictionaryAttr rhs,int64_t ra,bool rf,int64_t rs,
                      mlir::DictionaryAttr dst,int64_t da,int64_t ds,int64_t count,
                      llvm::StringRef opcode,bool reduce,bool bf16Output){
            const auto t=cursor;
            mem(lhs,la,count,ls,t-1,false,lf,0);
            mem(rhs,ra,count,rs,t-1,false,rf,2);
            alu(t-1,0,opcode,"stream_bf16","stream_bf16",count,false,false,false,false);
            if(reduce){
                alu(t-1,1,"add","previous","accumulator",1,count==1,true,true,bf16Output);
                if(count>2)alu(t,1,"add","previous","accumulator",count-2,false,false,true,bf16Output);
                if(count>1)alu(t+count-2,1,"add","previous","accumulator",1,true,false,true,bf16Output);
            }else alu(t-1,1,"bypass","previous","immediate",count,true,false,false,bf16Output);
            mem(dst,da,reduce?1:count,ds,t+(opcode=="multiply"?3:2)+(reduce?count-1:0),true,!bf16Output,0);
            cursor+=count+64;
        };
        for(int64_t head=0;head<localHeads;++head)for(int64_t block=0;block<blocks;++block){
            const auto s=block*localHeads*dim+head*dim, vector=head*blocks+block;
            flipped=false;
            pass(state_,s,true,1,decay_,head,true,0,decayed_,s,1,dim,"multiply",false,false);
            flipped=true;
            pass(decayed_,s,true,1,key_,head*dim,false,1,remembered_,vector,1,dim,"multiply",true,false);
            flipped=false;
            pass(value_,vector,false,1,remembered_,vector,true,1,delta_,vector,1,1,"subtract",false,false);
            flipped=true;
            pass(delta_,vector,true,1,beta_,head,true,0,remembered_,vector,1,1,"multiply",false,false);
            flipped=false;
            pass(key_,head*dim,false,1,remembered_,vector,true,0,product_,s,1,dim,"multiply",false,false);
            flipped=true;
            pass(decayed_,s,true,1,product_,s,true,1,finalState_,s,1,dim,"add",false,false);
            flipped=false;
            pass(finalState_,s,true,1,query_,head*dim,false,1,output_,vector,1,dim,"multiply",true,true);
        }
        mlir::OperationState timeline(op_.getLoc(),TimelineOp::getOperationName());
        timeline.addAttribute("name",rewriter_.getStringAttr("gated_delta_rule.fixed_decode_bf16_reload"));
        timeline.addAttribute("start",rewriter_.getI64IntegerAttr(0));
        timeline.addAttribute("end",rewriter_.getI64IntegerAttr(cursor));
        rewriter_.create(timeline);
        auto function = op_->getParentOfType<mlir::func::FuncOp>();
        const auto prefetch = op_->getParentOfType<mlir::ModuleOp>()
            ->getAttrOfType<mlir::BoolAttr>("ftlpu.gdn_prefetch_next_state");
        if (prefetch && prefetch.getValue()) {
            if (!partitioned || function.getNumArguments() != 7
                || function.getArgument(6).getType() != op_.getInitialState().getType())
                return op_.emitError("next-state prefetch requires a matching seventh input and partitioned heads");
            mlir::NamedAttrList next(state_);
            next.set("bank", rewriter_.getI64IntegerAttr(1-bank(state_)));
            next.set("paged_weight", rewriter_.getBoolAttr(true));
            next.set("runtime_prefetch", rewriter_.getBoolAttr(true));
            next.set("page_count", rewriter_.getI64IntegerAttr(1));
            next.set("page_bank_count", rewriter_.getI64IntegerAttr(2));
            next.set("page_rows", rewriter_.getI64IntegerAttr(localHeads*dim*blocks));
            next.set("prefetch_ready_cycle", rewriter_.getI64IntegerAttr(cursor));
            createInputBinding(rewriter_,op_,function.getArgument(6),"next_state",next.getDictionary(rewriter_.getContext()));
            op_->getPrevNode()->setAttr("role",rewriter_.getStringAttr("prefetch.input"));
        }
        auto output=createOutputBinding(rewriter_,op_,op_.getOutput().getType(),0,"output",output_);
        auto state=createOutputBinding(rewriter_,op_,op_.getFinalState().getType(),1,"final_state",finalState_);
        rewriter_.replaceOp(op_,{output.getValue(),state.getValue()});
        return mlir::success();
    }

    mlir::LogicalResult emitNative() {
        const auto tokens=llvm::cast<mlir::RankedTensorType>(op_.getQuery().getType()).getDimSize(0);
        if(op_.getHeads()!=1 || op_.getKeyDim()!=128 || op_.getValueDim()!=128 || tokens%32)
            return op_.emitError("native chunkwise GDN requires head-local chunk32 x 128");
        rewriter_.setInsertionPoint(op_);
        auto loc=op_.getLoc();
        int64_t cursor=64;
        const auto emitMem=[&](int64_t cycle,int64_t side,int64_t slice,int64_t memoryBank,
                bool write,int64_t address,int64_t count,int64_t stride,int64_t stream){
            direct_domain_detail::emitMem3D(rewriter_,loc,target_,cycle,
                side*target_.memory().slices_per_hemisphere+slice,write?"write":"read",
                address,stream,count,1,stride,1,1,0,1,1,0,-1,memoryBank);
        };
        // Raw byte-plane transfers preserve the graph's SRAM lifetime plan.
        // BF16 storage is explicit: FP32 interface values use their upper
        // planes on entry and zero-extended BF16 bits on exit.
        std::function<void(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t)> copy;
        copy=[&](int64_t sh,int64_t ss,int64_t sb,int64_t sa,int64_t dh,int64_t ds,int64_t db,int64_t da,int64_t count,int64_t stride,int64_t destinationStride){
            const auto sg=ss/int64_t(hw::kMemSlicesPerGroup),dg=ds/int64_t(hw::kMemSlicesPerGroup);
            if(sh==dh && sg==dg){
                const int64_t tmp=sg==50/int64_t(hw::kMemSlicesPerGroup)?46:50;
                copy(sh,ss,sb,sa,dh,tmp,0,0,count,stride,1);
                copy(dh,tmp,0,0,dh,ds,db,da,count,1,destinationStride);return;
            }
            int64_t read=20,write=20,latency;
            if(sh!=dh){read+=32;latency=sg+dg+2;}
            else if(sg<dg)latency=dg-sg;
            else {read+=32;write+=32;latency=sg-dg;}
            emitMem(cursor,sh,ss,sb,false,sa,count,stride,read);
            emitMem(cursor+latency,dh,ds,db,true,da,count,destinationStride,write);
            cursor+=count+latency+64;
        };
        const auto planeCopy=[&](mlir::DictionaryAttr p,int64_t plane,int64_t pair,int64_t count,bool output,int64_t sourceAddress=0,int64_t stride=1){
            const auto h=p.getAs<mlir::StringAttr>("hemisphere").getValue();
            for(int64_t side=0;side<2;++side){
                if(output && h!="both" && (side==1)!=(h=="west"))continue;
                const auto sourceSide=h=="both"?side:int64_t(h=="west");
                for(int64_t b=0;b<2;++b){
                    const bool matrix=pair==10 || pair==12 || pair==14 || pair==16 || pair==40;
                    if(matrix){
                        // Graph planar words are [column_block, row]; native
                        // chunkwise words are [row, column_block]. MEM address
                        // strides perform this permutation without SXM/ALU work.
                        const auto rows=count/4;
                        for(int64_t block=0;block<4;++block){
                            if(output)copy(side,pair+b,0,sourceAddress+block,side,sliceAt(p,plane+b),bank(p),baseRow(p)+block*rows,rows,4,1);
                            else copy(sourceSide,sliceAt(p,plane+b),bank(p),baseRow(p)+block*rows,side,pair+b,0,block,rows,1,4);
                        }
                    } else if(output)copy(side,pair+b,0,sourceAddress,side,sliceAt(p,plane+b),bank(p),baseRow(p),count,stride,1);
                    else copy(sourceSide,sliceAt(p,plane+b),bank(p),baseRow(p),side,pair+b,0,0,count,1,1);
                }
            }
        };
        const mlir::DictionaryAttr in[]={query_,key_,value_,decay_,beta_,state_};
        const char* names[]={"query","key","value","decay","beta","initial_state"};
        const int64_t pairs[]={10,12,14,20,22,16};
        const int64_t words[]={tokens*4,tokens*4,tokens*4,tokens,tokens,512};
        for(int i=0;i<6;++i){
            createInputBinding(rewriter_,op_,op_->getOperand(i),names[i],in[i]);
            planeCopy(in[i],i>=3?2:0,pairs[i],words[i],false);
        }
        for(const auto& epoch:GdnNativeLowering{}.lower(tokens,true)){
            const auto start=cursor;
            for(const auto& issue:epoch.program.mem){
                const auto& m=issue.instruction;
                if(m.address.outer_group_size!=1)throw std::runtime_error("native GDN non-affine domain");
                direct_domain_detail::emitMem3D(rewriter_,loc,target_,start+issue.cycle,
                    issue.side*target_.memory().slices_per_hemisphere+issue.slice,
                    m.opcode==MemIcuOpcode::Read3D?"read":"write",m.address.base_address,
                    // The public runtime uses the fixed cross-hemisphere VXM
                    // outputs. Both native halves compute identical values.
                    m.opcode==MemIcuOpcode::Write3D && m.stream<16 ? (m.stream^8) : m.stream,
                    m.loop.counts[0],m.loop.cycle_strides[0],m.address.inner_stride,
                    m.loop.counts[1],m.loop.cycle_strides[1],m.address.middle_stride,
                    m.loop.counts[2],m.loop.cycle_strides[2],m.address.outer_group_stride,-1,0);
            }
            for(const auto& issue:epoch.program.vxm){
                const auto decoded=VxmCompactInstructionCodec::decode(issue.queue,issue.instruction);
                const auto& v=decoded.instruction;
                const auto kind=[](VxmLaneOperandKind k)->llvm::StringRef{
                    switch(k){case VxmLaneOperandKind::Immediate:return "immediate";
                    case VxmLaneOperandKind::Previous:return "previous";
                    case VxmLaneOperandKind::Accumulator:return "accumulator";
                    case VxmLaneOperandKind::StreamBFloat16:return "stream_bf16";
                    default:throw std::runtime_error("unexpected native GDN operand");}
                };
                auto opcode=std::get<VxmAluOpcode>(v.operation);
                const auto opName=opcode==VxmAluOpcode::Add?"add":opcode==VxmAluOpcode::Subtract?"subtract":opcode==VxmAluOpcode::Multiply?"multiply":"bypass";
                mlir::OperationState state(loc,VxmOp::getOperationName());
                state.addOperands({op_.getQuery(),op_.getQuery()});state.addTypes(op_.getQuery().getType());
                const auto integer=[&](const char* name,int64_t value){state.addAttribute(name,rewriter_.getI64IntegerAttr(value));};
                const auto string=[&](const char* name,llvm::StringRef value){state.addAttribute(name,rewriter_.getStringAttr(value));};
                integer("cycle",start+issue.cycle);integer("queue",issue.queue);integer("chain_depth",2);
                string("opcode",opName);string("lhs_kind",kind(v.lhs.kind));string("rhs_kind",kind(v.rhs.kind));
                integer("lhs_index",0);integer("rhs_index",v.rhs.kind==VxmLaneOperandKind::StreamBFloat16?2:0);
                if(v.lhs.kind==VxmLaneOperandKind::StreamBFloat16)string("lhs_stream_source","local");
                if(v.rhs.kind==VxmLaneOperandKind::StreamBFloat16)string("rhs_stream_source","local");
                state.addAttribute("lhs_immediate",rewriter_.getF32FloatAttr(v.lhs.immediate));
                state.addAttribute("rhs_immediate",rewriter_.getF32FloatAttr(v.rhs.immediate));
                string("cast_target",v.output_type==VxmCastTarget::BFloat16?"bf16":"fp32");
                integer("output_stream",v.output_stream?int64_t(*v.output_stream):-1);
                integer("repeat_count",v.repeat_count);integer("repeat_interval",1);
                string("input_hemisphere","east");string("output_hemisphere","east");
                state.addAttribute("accumulator_reset",rewriter_.getBoolAttr(v.accumulator_reset));
                state.addAttribute("accumulator_write",rewriter_.getBoolAttr(v.accumulator_write));
                state.addAttribute("accumulator_emit",rewriter_.getBoolAttr(v.accumulator_emit));
                rewriter_.create(state);
            }
            for(const auto& issue:epoch.program.sxm){
                const auto& i=issue.instruction;mlir::OperationState state(loc,SxmOp::getOperationName());
                const auto integer=[&](const char* name,int64_t value){state.addAttribute(name,rewriter_.getI64IntegerAttr(value));};
                const auto array=[&](const char* name,const auto& values){llvm::SmallVector<mlir::Attribute> a;for(auto x:values)a.push_back(rewriter_.getI64IntegerAttr(x));state.addAttribute(name,rewriter_.getArrayAttr(a));};
                integer("cycle",start+issue.cycle);integer("hemisphere",issue.side);
                state.addAttribute("opcode",rewriter_.getStringAttr(i.opcode==SxmOpcode::Transpose?"transpose":"permute"));
                std::vector<int64_t> src,dst;for(auto s:i.src_streams)src.push_back(s.stream);for(auto s:i.dst_streams)dst.push_back(s.stream);
                array("source_streams",src);array("destination_streams",dst);array("permute_map",i.permute_map);
                if(i.input_row!=SxmInstruction::kAllInputRows)integer("input_row",i.input_row);
                if(i.output_row!=SxmInstruction::kAllOutputRows)integer("output_row",i.output_row);
                integer("repeat_count",issue.count);integer("repeat_interval",issue.interval);
                state.addAttribute("weight_layout",rewriter_.getStringAttr("native_gdn"));rewriter_.create(state);
            }
            cursor+=epoch.program.end_cycle;
            mlir::OperationState timeline(loc,TimelineOp::getOperationName());
            timeline.addAttribute("name",rewriter_.getStringAttr("imem.drain.gdn.native."+epoch.name));
            timeline.addAttribute("start",rewriter_.getI64IntegerAttr(start));timeline.addAttribute("end",rewriter_.getI64IntegerAttr(cursor));rewriter_.create(timeline);
        }
        planeCopy(output_,0,40,tokens*4,true);
        planeCopy(finalState_,0,26,512,true,32,0);
        planeCopy(finalState_,2,16,512,true);
        mlir::OperationState timeline(loc,TimelineOp::getOperationName());
        timeline.addAttribute("name",rewriter_.getStringAttr("gdn.native_chunkwise_bf16"));
        timeline.addAttribute("start",rewriter_.getI64IntegerAttr(0));timeline.addAttribute("end",rewriter_.getI64IntegerAttr(cursor));rewriter_.create(timeline);
        auto output=createOutputBinding(rewriter_,op_,op_.getOutput().getType(),0,"output",output_);
        auto state=createOutputBinding(rewriter_,op_,op_.getFinalState().getType(),1,"final_state_bf16_storage",finalState_);
        rewriter_.replaceOp(op_,{output.getValue(),state.getValue()});return mlir::success();
    }
    int64_t readLatency(mlir::DictionaryAttr p, int64_t slice) const
    {
        return target_.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, slice).value();
    }

    int64_t writeLatency(mlir::DictionaryAttr p, int64_t slice) const
    {
        static_cast<void>(p);
        return target_.transport_latency(target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice).value();
    }

    int64_t vxmResultCycle(mlir::DictionaryAttr destination,
        int64_t inputCycle, llvm::StringRef opcode) const
    {
        static_cast<void>(destination);
        // RUN_2D publishes the q1 result at the VXM-result source column.
        // MEM transport_latency below accounts for the remaining fabric
        // distance to each slice.  ADD/SUB+CAST needs five cycles from the
        // first external input; MUL+CAST needs one additional ALU cycle.
        return inputCycle + (opcode == "multiply" ? 6 : 5) + 2 * queueBase_;
    }

    std::pair<int64_t, int64_t> gridStrides(mlir::DictionaryAttr p) const
    {
        if (!grid_) return {0, 0};
        if (p == query_ || p == key_) return {keyDim_, 0};
        if (p == decay_ || p == beta_) return {1, 0};
        if (p == remembered_ || p == delta_) return {blocks_, 1};
        if (p == value_ || p == output_) return {blocks_ * tokens_, tokens_};
        return {keyDim_, heads_ * keyDim_};
    }

    void gridVxm(VxmOp op)
    {
        if (!grid_) return;
        op->setAttr("wave_count", rewriter_.getI64IntegerAttr(heads_ * blocks_));
        op->setAttr("wave_interval", rewriter_.getI64IntegerAttr(gridInterval_));
    }

    void readRows(mlir::DictionaryAttr p, int64_t address,
        int64_t stream, int64_t planes, int64_t rows,
        int64_t cycle, bool scalarBroadcast = false,
        int64_t addressStride = 1)
    {
        const int64_t tileCycles = 32
            / target_.throughput().mem_read_bytes_per_cycle;
        for (int64_t byte = 0; byte < planes; ++byte) {
            const int64_t slice = sliceAt(p, byte);
            direct_domain_detail::emitMem3D(rewriter_, op_.getLoc(), target_,
                cycle - readLatency(p, slice)
                    + (isWest(p) ? 2 * queueBase_ - (target_.throughput().vxm_alus - 1) : 0),
                physicalQueue(p, slice, target_), "read",
                baseRow(p) + address,
                target_.streams().streams_per_direction + stream + byte,
                rows, tileCycles,
                scalarBroadcast ? 0 : addressStride,
                grid_ ? heads_ : 1, grid_ ? gridInterval_ : 1, gridStrides(p).first,
                grid_ ? blocks_ : 1, grid_ ? heads_ * gridInterval_ : 1,
                gridStrides(p).second, -1, bank(p));
        }
    }

    void writeRows(mlir::DictionaryAttr p, int64_t address,
        int64_t stream, int64_t planes, int64_t rows, int64_t cycle,
        int64_t addressStride = 1)
    {
        const int64_t tileCycles = 32
            / target_.throughput().mem_read_bytes_per_cycle;
        for (int64_t byte = 0; byte < planes; ++byte) {
            const int64_t slice = sliceAt(p, byte);
            direct_domain_detail::emitMem3D(rewriter_, op_.getLoc(), target_,
                cycle + writeLatency(p, slice) - 1,
                physicalQueue(p, slice, target_), "write",
                baseRow(p) + address, stream + byte,
                rows, tileCycles, addressStride,
                grid_ ? heads_ : 1, grid_ ? gridInterval_ : 1, gridStrides(p).first,
                grid_ ? blocks_ : 1, grid_ ? heads_ * gridInterval_ : 1,
                gridStrides(p).second, -1, bank(p));
        }
    }

    int64_t binaryVectorPass(mlir::DictionaryAttr lhs, int64_t lhsAddress,
        mlir::DictionaryAttr rhs, int64_t rhsAddress,
        mlir::DictionaryAttr destination, int64_t destinationAddress,
        int64_t rows, llvm::StringRef opcode,
        llvm::StringRef lhsKind, llvm::StringRef rhsKind, int64_t cursor,
        llvm::StringRef castTarget = "fp32", int64_t lhsStride = 1,
        int64_t rhsStride = 1, int64_t destinationStride = 1)
    {
        const int64_t tileCycles = 32
            / target_.throughput().mem_read_bytes_per_cycle;
        const int64_t count = rows;
        const int64_t duration = rows * tileCycles;
        gridInterval_ = duration + 96;
        const int64_t input = cursor + 32;
        constexpr int64_t lhsStream = 0;
        constexpr int64_t rhsStream = 8;
        // Keep the return stream disjoint from the still-live operand
        // streams (0..3 and 8..11) on the shared west-going SR fabric.
        constexpr int64_t outputStream = 24;
        readRows(lhs, lhsAddress, lhsStream,
            lhsKind == "stream_f32" ? 4 : 2, rows, input,
            false, lhsStride);
        // Decay and beta are constant across every state row for one head.
        // Key/query use a scalar-broadcast physical layout too, but their row
        // address advances with the reduction dimension.
        const bool scalar = rhs == decay_ || rhs == beta_;
        readRows(rhs, rhsAddress, rhsStream,
            rhsKind == "stream_f32" ? 4 : 2, rows, input, scalar,
            rhsStride);
        auto head = create_vxm(rewriter_, op_.getLoc(), op_.getQuery(),
            op_.getKey(), op_.getOutput().getType(), input - 1, queueBase_, opcode,
            lhsKind, lhsStream, 0.0f, rhsKind, rhsStream, 0.0f,
            "fp32", -1, count, 1, inputHemisphere(lhs),
            outputHemisphere(destination), -1,
            false, false, true, false, 2);
        head->setAttr("rhs_stream_source", rewriter_.getStringAttr(inputHemisphere(rhs)));
        gridVxm(head);
        // RUN_2D schedules the entire chain at one launch cycle. Hardware
        // accounts for each ALU's latency; delaying the tail here applies
        // that latency twice and misaligns the MEM drain's tile phase.
        auto tail = create_vxm(rewriter_, op_.getLoc(), head.getResult(), op_.getKey(),
            op_.getOutput().getType(), input - 1, queueBase_ + 1, "cast",
            "previous", 0, 0.0f, "immediate", 0, 0.0f,
            castTarget, outputStream, count, 1, inputHemisphere(lhs),
            outputHemisphere(destination), -1,
            false, false, true, false, 2);
        gridVxm(tail);
        writeRows(destination, destinationAddress, outputStream,
            castTarget == "fp32" ? 4 : 2, rows,
            vxmResultCycle(destination, input, opcode), destinationStride);
        return input + (grid_ ? heads_ * blocks_ * gridInterval_ : duration) + 64;
    }

    int64_t reductionPass(mlir::DictionaryAttr matrix,
        int64_t matrixAddress, mlir::DictionaryAttr scalar,
        int64_t scalarAddress, mlir::DictionaryAttr destination,
        int64_t destinationAddress, int64_t reductionRows,
        llvm::StringRef scalarKind, int64_t cursor,
        llvm::StringRef castTarget)
    {
        const int64_t tileCycles = 32
            / target_.throughput().mem_read_bytes_per_cycle;
        const int64_t count = reductionRows;
        const int64_t duration = reductionRows * tileCycles;
        gridInterval_ = duration + 96;
        const int64_t input = cursor + 32;
        constexpr int64_t matrixStream = 0;
        constexpr int64_t scalarStream = 8;
        // Keep the return stream disjoint from the still-live operand
        // streams (0..3 and 8..11) on the shared west-going SR fabric.
        constexpr int64_t outputStream = 24;
        readRows(matrix, matrixAddress, matrixStream, 4,
            reductionRows, input);
        readRows(scalar, scalarAddress, scalarStream,
            scalarKind == "stream_f32" ? 4 : 2,
            reductionRows, input);
        auto head = create_vxm(rewriter_, op_.getLoc(), op_.getInitialState(),
            op_.getKey(), op_.getFinalState().getType(), input - 1, queueBase_,
            "multiply", "stream_f32", matrixStream, 0.0f,
            scalarKind, scalarStream, 0.0f, "fp32", -1,
            count, 1, inputHemisphere(matrix),
            outputHemisphere(destination), -1,
            false, false, true, false, 2);
        gridVxm(head);
        auto tail = create_vxm_with_terminal_controls(
            rewriter_, op_.getLoc(), op_.getInitialState(),
            op_.getKey(), op_.getFinalState().getType(), input - 1, queueBase_ + 1,
            "add", "previous", 0, 0.0f,
            "accumulator", 0, 0.0f, castTarget, outputStream, count, 1,
            inputHemisphere(matrix), outputHemisphere(destination), -1,
            true, true, true, false, 2, true, true);
        gridVxm(tail);
        const int64_t lastInput = input
            + (reductionRows - 1) * tileCycles;
        writeRows(destination, destinationAddress, outputStream,
            castTarget == "fp32" ? 4 : 2, 1,
            vxmResultCycle(destination, lastInput, "multiply"));
        return input + (grid_ ? heads_ * blocks_ * gridInterval_ : duration) + 72;
    }

    int64_t unaryCopy(mlir::DictionaryAttr source, int64_t sourceAddress,
        mlir::DictionaryAttr destination, int64_t destinationAddress,
        int64_t rows, int64_t cursor)
    {
        if (source == destination && sourceAddress == destinationAddress)
            return cursor;
        const int64_t tileCycles = 32
            / target_.throughput().mem_read_bytes_per_cycle;
        const int64_t count = rows;
        const int64_t duration = rows * tileCycles;
        const int64_t input = cursor + 32;
        constexpr int64_t stream = 0;
        readRows(source, sourceAddress, stream, 4, rows, input);
        auto head = create_vxm(rewriter_, op_.getLoc(),
            op_.getInitialState(), op_.getInitialState(),
            op_.getFinalState().getType(), input - 1, queueBase_, "cast",
            "stream_f32", stream, 0.0f,
            "immediate", 0, 0.0f, "fp32", -1, count, 1,
            inputHemisphere(source), outputHemisphere(destination), -1,
            false, false, true, false, 2);
        create_vxm(rewriter_, op_.getLoc(), head.getResult(),
            op_.getInitialState(), op_.getFinalState().getType(),
            input - 1, queueBase_ + 1, "cast", "previous", 0, 0.0f,
            "immediate", 0, 0.0f, "fp32", 24, count, 1,
            inputHemisphere(source), outputHemisphere(destination), -1,
            false, false, true, false, 2);
        writeRows(destination, destinationAddress, 24, 4, rows,
            vxmResultCycle(destination, input, "cast"));
        return input + duration + 64;
    }

    bool grid_{false};
    int64_t queueBase_{0};
    int64_t heads_{1}, blocks_{1}, keyDim_{32}, tokens_{1}, gridInterval_{1};
    mlir::IRRewriter& rewriter_;
    stream::GatedDeltaRuleTaskOp op_;
    const target::LPUTargetModel& target_;
    mlir::DictionaryAttr query_, key_, value_, decay_, beta_, state_;
    mlir::DictionaryAttr decayed_, product_, remembered_, delta_;
    mlir::DictionaryAttr output_, finalState_;
};

} // namespace

mlir::LogicalResult lowerGatedDeltaRuleSchedules(
    mlir::IRRewriter& rewriter, mlir::func::FuncOp function,
    const target::LPUTargetModel& target)
{
    llvm::SmallVector<stream::GatedDeltaRuleTaskOp> operations;
    function.walk([&](stream::GatedDeltaRuleTaskOp op) {
        operations.push_back(op);
    });
    for (auto op : operations)
        if (mlir::failed(Emitter(rewriter, op, target).emit()))
            return mlir::failure();
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
