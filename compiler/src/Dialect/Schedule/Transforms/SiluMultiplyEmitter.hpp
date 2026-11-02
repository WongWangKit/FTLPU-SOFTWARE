// Generic x * SiLU(z) DAG fusion. Placement descriptors are supplied by the
// graph planner; no model names or host-evaluated intermediates are involved.
struct SiluMultiplyMatch {
    mlir::Value gate, value;
    llvm::SmallVector<kernel::VectorAluOp> ops;
};
std::optional<SiluMultiplyMatch> matchSiluMultiply(mlir::func::FuncOp function) {
    llvm::SmallVector<kernel::VectorAluOp> ops;
    function.walk([&](kernel::VectorAluOp op){ops.push_back(op);});
    if(ops.size()!=6 || function.getNumArguments()!=2) return std::nullopt;
    const llvm::StringRef kinds[]={"negate","exp","add","reciprocal","multiply","multiply"};
    for(int i=0;i<6;++i) {
        if(ops[i].getKind()!=kinds[i] || ops[i].getReduce()) return std::nullopt;
        if(i<4 && !ops[i].getUseImmediate()) return std::nullopt;
        if(i<5 && !llvm::all_of(ops[i].getResult().getUsers(),
            [&](mlir::Operation* user){return user==ops[i+1].getOperation();})) return std::nullopt;
    }
    if(ops[2].getImmediate().convertToFloat()!=1.f) return std::nullopt;
    for(int i=1;i<4;++i) if(ops[i].getLhs()!=ops[i-1].getResult()) return std::nullopt;
    auto gate=ops[0].getLhs();
    const auto other=[](kernel::VectorAluOp op,mlir::Value v)->mlir::Value {
        if(op.getUseImmediate()) return {};
        if(op.getLhs()==v) return op.getRhs();
        if(op.getRhs()==v) return op.getLhs();
        return {};
    };
    if(other(ops[4],ops[3].getResult())!=gate) return std::nullopt;
    auto value=other(ops[5],ops[4].getResult());
    if(!value || !llvm::isa<mlir::BlockArgument>(gate) || !llvm::isa<mlir::BlockArgument>(value)) return std::nullopt;
    auto type=llvm::cast<mlir::RankedTensorType>(gate.getType());
    if(type!=value.getType() || !type.getElementType().isBF16() || type.getRank()!=2
        || type.getDimSize(0)%2 || type.getDimSize(1)%32) return std::nullopt;
    for(int i=0;i<5;++i) if(!ops[i].getResult().getType().getElementType().isF32()) return std::nullopt;
    if(ops.back().getResult().getType()!=type) return std::nullopt;
    return SiluMultiplyMatch{gate,value,ops};
}

mlir::LogicalResult emitNativeSiluMultiply(mlir::IRRewriter& r,
    mlir::func::FuncOp function,const target::LPUTargetModel& target,
    const SiluMultiplyMatch& match) {
    const auto loc=match.ops.front()->getLoc();
    const auto type=llvm::cast<mlir::RankedTensorType>(match.gate.getType());
    const auto rows=type.getDimSize(0),blocks=type.getDimSize(1)/32,half=rows/2;
    const auto wordsFor=[&](mlir::Value value) {
        return function->getAttrOfType<mlir::ArrayAttr>("native_pointwise_words_"+
            std::to_string(llvm::cast<mlir::BlockArgument>(value).getArgNumber()));
    };
    const auto gateWords=wordsFor(match.gate), valueWords=wordsFor(match.value);
    if(!gateWords || !valueWords) return function.emitError("native SiLU multiply requires planned words for both operands");
    auto out=match.ops.back()->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement");
    const auto field=[](mlir::DictionaryAttr d,llvm::StringRef name) {return d.getAs<mlir::IntegerAttr>(name).getInt();};
    const auto sideOf=[](mlir::DictionaryAttr p) {return p.getAs<mlir::StringAttr>("hemisphere").getValue()=="west"?1:0;};
    llvm::SmallVector<mlir::DictionaryAttr> sources[2][2];
    for(int64_t i=0;i<rows*blocks;++i) {
        auto g=llvm::cast<mlir::DictionaryAttr>(gateWords[i]);
        auto v=llvm::cast<mlir::DictionaryAttr>(valueWords[i]);
        auto side=sideOf(g.getAs<mlir::DictionaryAttr>("placement"));
        if(side!=sideOf(v.getAs<mlir::DictionaryAttr>("placement")))
            return function.emitError("native SiLU operands must share head ownership");
        sources[0][side].push_back(g);sources[1][side].push_back(v);
    }
    const int64_t count=half*blocks,start=64;
    r.setInsertionPoint(match.ops.front());
    for(int operand=0;operand<2;++operand) for(int side=0;side<2;++side) {
        const auto& words=sources[operand][side];
        if(words.size()!=count) return function.emitError("native SiLU requires balanced head partitions");
        const auto p=words.front().getAs<mlir::DictionaryAttr>("placement");
        const auto ids=p.getAs<mlir::ArrayAttr>("slices");
        const int64_t base=field(words[0],"address");
        const int64_t rowStride=half>1?field(words[1],"address")-base:0;
        const int64_t blockStride=blocks>1?field(words[half],"address")-base:0;
        for(int64_t b=0;b<blocks;++b) for(int64_t h=0;h<half;++h)
            if(words[b*half+h].getAs<mlir::DictionaryAttr>("placement")!=p
                || field(words[b*half+h],"address")!=base+h*rowStride+b*blockStride)
                return function.emitError("native SiLU word domain must be affine");
        for(int byte=0;byte<2;++byte) {
            const auto slice=llvm::cast<mlir::IntegerAttr>(ids[byte]).getInt();
            const auto latency=*target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,target::StreamDirection::West,slice);
            direct_domain_detail::emitMem3D(r,loc,target,start-latency,
                side*target.memory().slices_per_hemisphere+slice,"read",base,
                32+side*16+operand*2+byte,half,1,rowStride,blocks,half,blockStride,
                1,1,0,-1,field(p,"bank"));
        }
    }
    // C0 captures z as original and normalized x as auxiliary; both travel
    // with the chain. No sigmoid/SiLU intermediates are spilled to MEM.
    mlir::Value current=match.gate;
    for(int col=0;col<8;++col) {
        const llvm::StringRef opcodes[]={"negate","exp","add","reciprocal","multiply","multiply","bypass","bypass"};
        auto op=ffn_detail::create_vxm(r,loc,current,current,type,start-1,col,opcodes[col],
            col?"previous":"stream_bf16",0,0.f,
            col==0?"stream_bf16":col==4?"original":col==5?"auxiliary":"immediate",
            col==0?2:0,col==2?1.f:0.f,col==7?"bf16":"fp32",col==7?6:-1,
            count,1,"east","west",-1,false,false,true,false,8);
        current=op.getResult();
    }
    // Eight ALU stages + EXP/reciprocal extra latency + two multiplies.
    const int64_t latency=18;
    const auto ids=out.getAs<mlir::ArrayAttr>("slices");
    int64_t end=start+latency+count;
    for(int side=0;side<2;++side) for(int byte=0;byte<2;++byte) {
        const auto slice=llvm::cast<mlir::IntegerAttr>(ids[byte]).getInt();
        const auto route=*target.transport_latency(target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,target::StreamDirection::East,slice);
        direct_domain_detail::emitMem3D(r,loc,target,start+latency-1+route,
            side*target.memory().slices_per_hemisphere+slice,"write",field(out,"base_row"),
            (1-side)*8+6+byte,count,1,1,1,1,0,1,1,0,-1,field(out,"bank"));
        end=std::max(end,start+latency-1+route+count+3);
    }
    mlir::OperationState timeline(loc,TimelineOp::getOperationName());
    timeline.addAttributes({r.getNamedAttr("name",r.getStringAttr("silu_multiply.native_chain")),
        r.getNamedAttr("start",r.getI64IntegerAttr(start-1)),r.getNamedAttr("end",r.getI64IntegerAttr(end))});
    r.create(timeline);
    mlir::OperationState state(loc,BindingOp::getOperationName());
    state.addTypes(type);
    state.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(0)),
        r.getNamedAttr("access",r.getStringAttr("output")),r.getNamedAttr("role",r.getStringAttr("result")),
        r.getNamedAttr("bytes",r.getI64IntegerAttr(type.getNumElements()*2)),r.getNamedAttr("placement",out)});
    auto binding=llvm::cast<BindingOp>(r.create(state));
    r.replaceOp(match.ops.back(),binding.getValue());
    for(int i=4;i>=0;--i)r.eraseOp(match.ops[i]);
    return mlir::success();
}
