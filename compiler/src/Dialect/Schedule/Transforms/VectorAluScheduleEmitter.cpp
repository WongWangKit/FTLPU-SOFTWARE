#include "ftlpu/compiler/Analysis/planar_access_attrs.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"
#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/graph_weight_paging.hpp"
#include "DirectDomainEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"
#include "AttentionEmitterUtils.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/APFloat.h"
#include <algorithm>

namespace ftlpu::compiler::schedule {
namespace {
mlir::DictionaryAttr placement(mlir::IRRewriter& r,
    mlir::RankedTensorType type, int64_t slice, llvm::StringRef hemisphere,
    int64_t bank, bool packed, int64_t wordElements)
{
    llvm::SmallVector<mlir::Attribute> slices;
    const int64_t bytes = type.getElementType().isF32() ? 4 : 2;
    for (int64_t i = 0; i < bytes; ++i)
        slices.push_back(r.getI64IntegerAttr(slice + i));
    return r.getDictionaryAttr({
        r.getNamedAttr("kind", r.getStringAttr(packed ? (bytes == 4 ? "fp32_quad_planar" : "fp16_pair_planar") : (bytes == 4
            ? "fp32_scalar_broadcast_planar" : "fp16_scalar_broadcast_planar"))),
        r.getNamedAttr("slices", r.getArrayAttr(slices)),
        r.getNamedAttr("base_row", r.getI64IntegerAttr(0)),
        r.getNamedAttr("bank", r.getI64IntegerAttr(bank)),
        r.getNamedAttr("instruction_count", r.getI64IntegerAttr(packed
            ? type.getDimSize(0) * ((type.getDimSize(1) + wordElements - 1) / wordElements)
            : type.getNumElements())),
        r.getNamedAttr("address_stride", r.getI64IntegerAttr(1)),
        r.getNamedAttr("hemisphere", r.getStringAttr(hemisphere)),
    });
}

struct Storage {
    int64_t slice = 0;
    int64_t bank = 0;
    bool west = false;
    int64_t base = 0;
    bool both = false;
    bool tokenPartition = false;
    int64_t broadcastRowRepeat = 0;
    int64_t broadcastHeadBase = 0;
    int64_t broadcastHeadStride = 1;
    std::optional<analysis::PlanarReadPlan> access;
};

// A matrix word carries 32 independent value components. Only the row
// coefficient is replicated across its lanes, in one physical word per row.
bool packedUpdate(mlir::func::FuncOp function)
{
    auto policy=function->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_packed_update");
    return policy && policy.getValue();
}

bool reduceAlongRows(kernel::VectorAluOp op)
{
    auto axis=op->getAttrOfType<mlir::IntegerAttr>("reduction_axis");
    return op.getReduce() && axis && axis.getInt()==0;
}

Storage storageOf(mlir::DictionaryAttr p)
{
    return {llvm::cast<mlir::IntegerAttr>(p.getAs<mlir::ArrayAttr>("slices")[0]).getInt(),
        p.getAs<mlir::IntegerAttr>("bank").getInt(),
        p.getAs<mlir::StringAttr>("hemisphere").getValue() == "west",
        p.getAs<mlir::IntegerAttr>("base_row").getInt(),
        p.getAs<mlir::StringAttr>("hemisphere").getValue() == "both",
        p.getAs<mlir::StringAttr>("kind").getValue() == "token_partition_planar",
        p.getAs<mlir::IntegerAttr>("broadcast_row_repeat") ? p.getAs<mlir::IntegerAttr>("broadcast_row_repeat").getInt() : 0,
        p.getAs<mlir::IntegerAttr>("broadcast_head_base") ? p.getAs<mlir::IntegerAttr>("broadcast_head_base").getInt() : 0,
        p.getAs<mlir::IntegerAttr>("broadcast_head_stride") ? p.getAs<mlir::IntegerAttr>("broadcast_head_stride").getInt() : 1,
        analysis::decodePlanarRead(p)};
}

#include "PlanarAccessEmitter.hpp"
#include "VectorFusionEmitter.hpp"
#include "ScaledMatvecEmitter.hpp"
#include "SiluMultiplyEmitter.hpp"
#include "PackedRowScaleEmitter.hpp"
#include "WeightedGramEmitter.hpp"
#include "UnitLowerSolveEmitter.hpp"
#include "UnitLowerMxmEmitter.hpp"
#include "HeadWeightedProductEmitter.hpp"

// One streamed head followed by immediate operations. Keep multi-use values
// and quantizing edges as boundaries. This plan also determines SRAM lifetimes:
// a head operand remains live until the entire chain has drained.
bool isHeadMatmulAdd(kernel::VectorAluOp product, kernel::VectorAluOp sum)
{
    if(!product || !sum || product.getKind()!="head_matmul" || sum.getKind()!="add") return false;
    const auto precision=product->getAttrOfType<mlir::StringAttr>("mxm_precision");
    return precision && precision.getValue()=="bf16" && product.getResult().hasOneUse() &&
        !sum.getUseImmediate() && !sum.getReduce() && sum.getLhs()==product.getResult() &&
        sum.getRhs().getType()==product.getResult().getType() && sum.getResult().getType()==product.getResult().getType() &&
        sum.getResult().getType().getElementType().isF32();
}

bool isHeadMatmulCast(kernel::VectorAluOp sum, kernel::VectorAluOp cast)
{
    return cast && cast.getKind()=="multiply" && cast.getUseImmediate() && !cast.getReduce() &&
        cast.getImmediate().convertToFloat()==1.f && cast.getLhs()==sum.getResult() &&
        cast.getResult().getType().getElementType().isBF16() &&
        cast.getResult().getType().getShape()==sum.getResult().getType().getShape() &&
        llvm::all_of(sum.getResult().getUsers(),[&](mlir::Operation *user){return user==cast.getOperation();});
}

llvm::SmallVector<int64_t> chainColumns(llvm::ArrayRef<kernel::VectorAluOp> ops,
                                      bool packed)
{
    const auto columnFor = [](kernel::VectorAluOp op, int64_t first) {
        const auto kind = op.getKind();
        const int64_t residue = kind == "exp" ? 1
            : kind == "reciprocal" || kind == "rsqrt" || kind == "log" ? 3 : -1;
        while (residue >= 0 && first % 4 != residue) ++first;
        return first;
    };
    auto head = ops.front();
    if(packed && ops.size()>=2 && (matchScaleAdd(head,ops[1]) || matchAddScale(head,ops[1]) || matchSquareSum(head,ops[1])))
        return {0,1};
    if(packed)if(auto fold=matchPointwiseFold(ops);!fold.columns.empty())return fold.columns;
    if (packed && ops.size() >= 2 && reduceAlongRows(head)) {
        auto next = ops[1];
        auto reduction = head->getAttrOfType<mlir::StringAttr>("reduction_kind");
        if (reduction && reduction.getValue() == "add" && head.getKind() == "multiply"
            && head.getUseImmediate() && head.getImmediate().convertToFloat() == 1.f
            && next.getKind() == "reciprocal" && next.getUseImmediate() && !next.getReduce()
            && next.getLhs() == head.getResult()
            && head.getResult().getType().getElementType().isF32()
            && next.getResult().getType() == head.getResult().getType()
            && llvm::all_of(head.getResult().getUsers(), [&](auto *user) { return user == next.getOperation(); }))
            return {1, 3};
    }
    if(ops.size()>=2 && isHeadMatmulAdd(head,ops[1]))
        return ops.size()>=3 && isHeadMatmulCast(ops[1],ops[2])?
            llvm::SmallVector<int64_t>{0,1,2}:llvm::SmallVector<int64_t>{0,1};
    // Fuse a matrix contraction and its pointwise epilogue before emitting
    // hardware domains. The intermediate matrix never acquires a MEM stream.
    if(head.getKind()=="weighted_gram" && ops.size()>=3) {
        auto scale=ops[1],mask=ops[2];
        if(scale.getKind()=="gram_row_scale" && mask.getKind()=="head_mask" &&
           scale.getLhs()==head.getResult() && mask.getLhs()==scale.getResult() &&
           head.getResult().hasOneUse() && scale.getResult().hasOneUse()) return {0,1,2};
    }
    if(head.getKind()=="cross_gram") {
        auto next=ops.size()>1?ops[1]:kernel::VectorAluOp{};
        if(next && next.getKind()=="multiply" && !next.getUseImmediate() &&
           !next.getReduce() && next.getLhs()==head.getResult() && head.getResult().hasOneUse() &&
           next.getRhs().getType().getElementType().isBF16()) return {0,1};
        return {0};
    }
    if(head.getKind()=="head_last_row_scale_transpose") {
        auto next=ops.size()>1?ops[1]:kernel::VectorAluOp{};
        if(next && next.getKind()=="head_cross_product" && next.getLhs()==head.getResult() && head.getResult().hasOneUse()) {
            auto output=ops.size()>2?ops[2]:kernel::VectorAluOp{};
            if(output && output.getKind()=="head_matmul" && output.getRhs()==next.getRhs()) {
                auto precision=output->getAttrOfType<mlir::StringAttr>("mxm_precision");
                if(precision && precision.getValue()=="bf16") return {0,1,2};
            }
            return {0,1};
        }
        return {0};
    }
    const bool fixed = head->getParentOfType<mlir::func::FuncOp>()->hasAttr("ftlpu.vector_bf16_spill");
    const auto tokenPolicy=head->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_bf16_token_halves");
    const bool originalChain=fixed && tokenPolicy && tokenPolicy.getValue();
    llvm::SmallVector<int64_t> columns{columnFor(head,
        fixed && !head.getUseImmediate() && head.getLhs()==head.getRhs()?1:0)};
    if (head.getKind()=="cumsum_rows" || head.getKind()=="causal_exp_difference" || head.getKind()=="weighted_gram" || (head.getKind()=="unit_lower_solve" || head.getKind()=="unit_lower_solve_blocked") || head.getKind()=="head_scale" || head.getKind()=="unit_lower_inverse" || head.getKind()=="head_matmul") return columns;
    while (packed && !head.getReduce() && columns.size() < ops.size()) {
        auto prior = ops[columns.size() - 1];
        auto next = ops[columns.size()];
        const int64_t column = columnFor(next, columns.back() + 1);
        if (column >= (fixed ? 8 : 7) || next.getReduce() || next.getKind()=="cumsum_rows" || next.getKind()=="causal_exp_difference" || (next.getKind()=="weighted_gram" || next.getKind()=="cross_gram") || (next.getKind()=="unit_lower_solve" || next.getKind()=="unit_lower_solve_blocked") || next.getKind()=="head_scale" || next.getKind()=="unit_lower_inverse" || next.getKind()=="head_matmul"
            || (!next.getUseImmediate() && !(originalChain && next.getRhs()==head.getLhs()))
            || next.getLhs() != prior.getResult()
            || !prior.getResult().getType().getElementType().isF32()
            || !llvm::all_of(prior.getResult().getUsers(), [&](auto *user) {
                   return user == next.getOperation();
               })) break;
        columns.push_back(column);
    }
    return columns;
}

mlir::LogicalResult lowerOne(mlir::IRRewriter& r, kernel::VectorAluOp op,
    const target::LPUTargetModel& target, Storage lhsStorage,
    Storage rhsStorage, Storage outStorage, bool packed, int64_t queueBase,
    llvm::ArrayRef<kernel::VectorAluOp> chain, llvm::ArrayRef<int64_t> columns)
{
    if(chain.size()==2 && (matchScaleAdd(op,chain[1]) || matchAddScale(op,chain[1]) || matchSquareSum(op,chain[1])))
        return emitPlanarArithmeticFusion(r,chain,target,matchSquareSum(op,chain[1]));
    if(!matchPointwiseFold(chain).columns.empty())return emitPointwiseFold(r,chain,target);
    if(op.getKind()=="head_last_row_scale_transpose") {
        if(chain.size()!=2 && chain.size()!=3)return op.emitError("requires fused head_cross_product consumer");
        return emitHeadWeightedProduct(r,op,chain[1],target,chain.size()==3?chain[2]:kernel::VectorAluOp{});
    }
    if(op.getKind()=="head_cross_product")return op.emitError("requires fused scale-transpose producer");
    if(op.getKind()=="unit_lower_inverse" || op.getKind()=="head_matmul")
        return emitUnitLowerMxm(r,op,target,chain.size()>=2?chain[1]:kernel::VectorAluOp{},
            chain.size()==3?chain[2]:kernel::VectorAluOp{});
    if((op.getKind()=="unit_lower_solve" || op.getKind()=="unit_lower_solve_blocked") || op.getKind()=="head_scale") return emitUnitLowerSolve(r,op,target);
    if(op.getKind()=="cross_gram") {
        if(chain.size()!=2) return op.emitError("cross_gram currently requires a single native BF16 multiply epilogue");
        return emitWeightedGram(r,op,target,chain.back());
    }
    if(op.getKind()=="weighted_gram") return emitWeightedGram(r,op,target,{},chain.size()==3?chain[1]:kernel::VectorAluOp{},chain.size()==3?chain[2]:kernel::VectorAluOp{});
    if(op.getKind()=="gram_row_scale" || op.getKind()=="head_mask") return op.emitError("requires fused weighted Gram row-scale/mask chain");
    const auto lhsType = op.getLhs().getType();
    const auto rhsType = op.getRhs().getType();
    auto tail = chain.back();
    const auto outType = tail.getResult().getType();
    const int64_t wordElements = target.memory().bytes_per_word;
    const int64_t packedColumns = (lhsType.getDimSize(1) + wordElements - 1) / wordElements;
    const bool update=packedUpdate(op->getParentOfType<mlir::func::FuncOp>());
    const bool rowCoefficient=update && rhsType.getDimSize(1)==1;
    const int64_t storageWords = packed ? lhsType.getDimSize(0) * packedColumns : lhsType.getNumElements();
    if (storageWords > target.memory().sram_depth_rows)
        return op.emitError("vector scalar-broadcast storage exceeds SRAM; tile the vector before lowering");
    r.setInsertionPoint(op);
    const auto bind = [&](mlir::Value value, mlir::RankedTensorType type,
                          int64_t index, llvm::StringRef access,
                          Storage storage) {
        mlir::OperationState state(op.getLoc(), BindingOp::getOperationName());
        if (value) state.addOperands(value);
        state.addTypes(type);
        state.addAttributes({
            r.getNamedAttr("index", r.getI64IntegerAttr(index)),
            r.getNamedAttr("access", r.getStringAttr(access)),
            r.getNamedAttr("role", r.getStringAttr(access == "input" ? "activation" : "result")),
            r.getNamedAttr("name", r.getStringAttr("vector_alu")),
            r.getNamedAttr("bytes", r.getI64IntegerAttr(type.getNumElements()
                * (type.getElementType().isF32() ? 4 : 2))),
            r.getNamedAttr("placement", [&] {
                const bool wordPacked=packed && !(rowCoefficient && value==op.getRhs());
                mlir::NamedAttrList a(placement(r, type, storage.slice, storage.both ? "both" : storage.west ? "west" : "east", storage.bank, wordPacked, wordElements));
                a.set("base_row", r.getI64IntegerAttr(storage.base));
                if(storage.broadcastRowRepeat)
                    a.set("instruction_count",r.getI64IntegerAttr(storage.broadcastHeadBase+
                        (type.getDimSize(0)/storage.broadcastRowRepeat-1)*storage.broadcastHeadStride+1));
                if(storage.tokenPartition) a.set("kind",r.getStringAttr("token_partition_planar"));
                return a.getDictionary(r.getContext());
            }()),
        });
        return llvm::cast<BindingOp>(r.create(state));
    };
    const auto bindInput = [&](mlir::Value value, mlir::RankedTensorType type, Storage storage) {
        const auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
        if (!argument) return;
        for (auto binding : op->getBlock()->getOps<BindingOp>())
            if (binding.getAccess() == "input" && binding->getNumOperands() == 1
                && binding->getOperand(0) == value) return;
        bind(value, type, argument.getArgNumber(), "input", storage);
    };
    bindInput(op.getLhs(), lhsType, lhsStorage);
    if (op.getRhs() != op.getLhs())
        bindInput(op.getRhs(), rhsType, rhsStorage);
    const auto fp32Attr=op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_fp32_ingress");
    const bool fp32Ingress = (fp32Attr && fp32Attr.getValue()) ||
        llvm::any_of(chain, [](auto alu) { return alu->hasAttr("ftlpu.vector_require_fp32"); });
    // A one-key PV tile is a BF16 matrix scaled by one BF16 coefficient
    // per query row. Each coefficient word is already lane-broadcast and
    // is reused for every value block. Existing fixed ports 0/2 consume
    // the two BF16 byte pairs and the product is written with four planes.
    const bool bf16RowProduct = rowCoefficient && op.getKind()=="multiply" &&
        lhsType.getElementType().isBF16() && rhsType.getElementType().isBF16() &&
        outType.getElementType().isF32();
    const bool fp32Update = lhsType.getElementType().isF32() && rhsType.getElementType().isF32();
    if (update && (!fp32Ingress || !packed
        || op.getReduce() || op.getUseImmediate() || chain.size()!=1 ||
        (op.getKind()!="multiply" && op.getKind()!="add") ||
        !(fp32Update || bf16RowProduct) ||
        lhsType.getDimSize(1)%wordElements!=0 || rhsType.getDimSize(0)!=lhsType.getDimSize(0) ||
        (!rowCoefficient && rhsType.getDimSize(1)!=lhsType.getDimSize(1)) ||
        lhsType.getShape()!=outType.getShape() ||
        !(outType.getElementType().isF32() || outType.getElementType().isBF16())))
        return op.emitError("packed update requires fixed ingress and word-aligned FP32 operands, or a BF16 matrix times per-row BF16 coefficients yielding FP32");
    if (op.getKind()=="causal_exp_difference") {
        const auto transpose=op->getAttrOfType<mlir::BoolAttr>("transpose_result");
        const bool directTranspose=transpose && transpose.getValue();
        const auto scaleAttr=op->getAttrOfType<mlir::BoolAttr>("row_scale");
        const bool rowScale=scaleAttr && scaleAttr.getValue();
        const auto strictAttr=op->getAttrOfType<mlir::BoolAttr>("strict_lower");
        const bool strict=strictAttr && strictAttr.getValue();
        if(rowScale && (!rhsStorage.both || rhsStorage.tokenPartition))
            return op.emitError("scaled causal difference requires native mirrored row coefficients");
        const auto wide=op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
        if(directTranspose && (!wide || !wide.getValue()))
            return op.emitError("transposed causal difference requires wide SXM");
        if (!packed || packedColumns!=1 || !lhsStorage.both || !outStorage.both ||
            lhsStorage.tokenPartition || outStorage.tokenPartition ||
            !op->getParentOfType<mlir::func::FuncOp>()->hasAttr("ftlpu.vector_bf16_spill"))
            return op.emitError("causal_exp_difference requires mirrored packed BF16 ingress and full row domains");
        const bool fullInput=fp32Ingress && lhsType.getElementType().isF32();
        const int64_t tokens=lhsType.getDimSize(0), first=lhsType.getElementType().isF32() && !fullInput?2:0;
        const int64_t begin=96, pitch=tokens+24;
        int64_t transposeEnd=begin;
        const auto read=[&](int64_t issue,int64_t row,int64_t port,int64_t count) {
            for(int64_t side=0;side<2;++side) for(int64_t byte=0;byte<(fullInput?4:2);++byte) {
                const auto slice=lhsStorage.slice+first+byte;
                direct_domain_detail::emitMem3D(r,op.getLoc(),target,issue-slice/4,
                    side*target.memory().slices_per_hemisphere+slice,"read",lhsStorage.base+row,
                    32+side*16+port+byte,count,1,1,1,1,0,1,1,0,-1,lhsStorage.bank);
            }
        };
        const auto write=[&](int64_t issue,int64_t row,int64_t count) {
            for(int64_t side=0;side<2;++side) for(int64_t byte=0;byte<4;++byte) {
                const auto slice=outStorage.slice+byte;
                direct_domain_detail::emitMem3D(r,op.getLoc(),target,issue+slice/4,
                    side*target.memory().slices_per_hemisphere+slice,"write",outStorage.base+row,
                    (1-side)*8+6+byte,count,1,1,1,1,0,1,1,0,-1,outStorage.bank);
            }
        };
        for(int64_t i=0;i<tokens;++i) {
            const int64_t latch=begin+i*pitch, compute=latch+2, lower=i+(strict?0:1);
            if(rowScale) {
                // One head-vector per row enters C0 and is latched in C7.
                // It is reused for all j without an expanded broadcast table.
                const bool fullScale=fp32Ingress && rhsType.getElementType().isF32();
                const int64_t loadAt=latch-10, plane=rhsType.getElementType().isF32() && !fullScale?2:0;
                for(int64_t side=0;side<2;++side) for(int64_t byte=0;byte<(fullScale?4:2);++byte) {
                    const auto slice=rhsStorage.slice+plane+byte;
                    direct_domain_detail::emitMem3D(r,op.getLoc(),target,loadAt-slice/4,
                        side*target.memory().slices_per_hemisphere+slice,"read",rhsStorage.base+i,
                        32+side*16+byte,1,1,0,1,1,0,1,1,0,-1,rhsStorage.bank);
                }
                for(int64_t column=0;column<8;++column) {
                    auto loadScale=ffn_detail::create_vxm(r,op.getLoc(),op.getRhs(),op.getRhs(),outType,
                        loadAt,column,column==7?"add":"bypass",column?"previous":fullScale?"stream_f32":"stream_bf16",0,0.f,
                        column==7?"accumulator":"immediate",0,0.f,"fp32",-1,
                        1,1,"east","east",-1,column==7,column==7,column!=7,false,8);
                    if(column==0) loadScale->setAttr("lhs_stream_source",r.getStringAttr("local"));
                }
            }
            // The same MEM queue loads Gi once, then streams Gj. C1's FP32
            // accumulator holds Gi; no expanded broadcast or difference spill.
            read(latch,i,0,1);
            auto load=ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getLhs(),outType,
                latch,0,"bypass",fullInput?"stream_f32":"stream_bf16",0,0.f,"immediate",0,0.f,"fp32",-1,
                1,1,"east","east",-1,false,false,true,false,8);
            load->setAttr("lhs_stream_source",r.getStringAttr("local"));
            ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getLhs(),outType,
                latch,1,"add","previous",0,0.f,"accumulator",0,0.f,"fp32",-1,
                1,1,"east","east",-1,true,true,false,false,8);
            if(lower>0) {
            read(compute,0,0,lower);
            auto negative=ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getLhs(),outType,
                compute,0,"negate",fullInput?"stream_f32":"stream_bf16",0,0.f,"immediate",0,0.f,"fp32",-1,
                lower,1,"east","east",-1,false,false,true,false,8);
            negative->setAttr("lhs_stream_source",r.getStringAttr("local"));
            for(int64_t column=1;column<8;++column)
                ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getLhs(),outType,
                    compute,column,column==1?"add":column==5?"exp":column==7 && rowScale?"multiply":"bypass","previous",0,0.f,
                    (column==1 || (column==7 && rowScale))?"accumulator":"immediate",0,0.f,"fp32",column==7?6:-1,
                    lower,1,"east","east",-1,false,false,true,false,8);
            if(!directTranspose) write(compute+13+rowScale,i*tokens,lower);
            }
            if(lower<tokens) {
                // Drain the special-ALU pipeline before the fast zero path.
                // Avoid exp(positive upper-triangle differences) and inf*0.
                // Exp and the fast zero path must produce adjacent vectors
                // when the downstream wide SXM captures an entire row tile.
                const int64_t zero=compute+lower+(directTranspose?4:5)+rowScale;
                for(int64_t column=0;column<8;++column)
                    ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getLhs(),outType,
                        zero,column,"bypass",column?"previous":"immediate",0,0.f,"immediate",0,0.f,
                        "fp32",column==7?6:-1,tokens-lower,1,"east","east",-1,false,false,true,false,8);
                if(!directTranspose) write(zero+9,i*tokens+lower,tokens-lower);
            }
            if(directTranspose && i==0) {
                const int64_t heads=lhsType.getDimSize(1);
                const int64_t capture=compute+12+rowScale+*target.transport_latency(
                    target::StreamEndpoint::VxmResult,target::StreamEndpoint::SxmInput,
                    target::StreamDirection::East,0);
                llvm::SmallVector<int64_t> map;
                for(int64_t j=0;j<32;++j)map.push_back(j);
                for(int64_t side=0;side<2;++side) {
                    // Upper FP32 byte planes preserve the exact truncation
                    // formerly performed on the consumer's BF16 MEM read.
                    const int64_t stream=(1-side)*8+8;
                    // G continues to arrive on reverse port 0 while SXM
                    // drains. Reserve reverse ports 4/5 for transposed R.
                    attention_detail::emitSxm(r,op.getLoc(),capture,side,"transpose",
                        {stream,stream+1},{36,37},map,"wide_matrix",-1,-1,-1,tokens,pitch);
                    for(int64_t byte=0;byte<2;++byte) {
                        const auto slice=outStorage.slice+byte;
                        const int64_t at=capture+35+*target.transport_latency(
                            target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,
                            target::StreamDirection::West,slice);
                        direct_domain_detail::emitMem3D(r,op.getLoc(),target,at,
                            side*target.memory().slices_per_hemisphere+slice,"write",
                            outStorage.base,36+byte,heads,1,1,tokens,pitch,heads,1,1,0,-1,outStorage.bank);
                        transposeEnd=std::max(transposeEnd,at+(tokens-1)*pitch+32+4);
                    }
                }
            }
        }
        mlir::OperationState timeline(op.getLoc(),TimelineOp::getOperationName());
        timeline.addAttribute("name",r.getStringAttr(rowScale?"causal_exp_difference.scaled_strict_vxm_sxm":directTranspose?"causal_exp_difference.vxm_sxm_direct":"causal_exp_difference.accumulator_exp"));
        timeline.addAttribute("start",r.getI64IntegerAttr(begin));
        timeline.addAttribute("end",r.getI64IntegerAttr(std::max(transposeEnd,begin+tokens*pitch+(outStorage.slice+3)/4)));
        r.create(timeline);
        auto result=bind({},outType,0,"output",outStorage);
        r.replaceOp(op,result.getValue());
        return mlir::success();
    }
    if (reduceAlongRows(op) && chain.size() == 2 && tail.getKind() == "reciprocal") {
        // Keep the FP32 sum in C1, emit only its final token, and feed C3's
        // reciprocal directly. No sum spill/reload or BF16 boundary is added.
        if (!packed || !fp32Ingress || !lhsStorage.both || !outStorage.both)
            return op.emitError("sum-reciprocal chain requires mirrored packed FP32-capable ports");
        const int64_t count=lhsType.getDimSize(0), start=96, latency=8;
        const bool fullInput=lhsType.getElementType().isF32();
        for(int64_t word=0;word<packedColumns;++word) {
            const int64_t begin=start+word*count;
            for(int64_t side=0;side<2;++side) {
                for(int64_t byte=0;byte<(fullInput?4:2);++byte) {
                    const auto slice=lhsStorage.slice+byte;
                    direct_domain_detail::emitMem3D(r,op.getLoc(),target,begin-1-slice/4,
                        side*target.memory().slices_per_hemisphere+slice,"read",lhsStorage.base+word*count,
                        32+side*16+byte,count,1,1,1,1,0,1,1,0,-1,lhsStorage.bank);
                }
                for(int64_t byte=0;byte<4;++byte) {
                    const auto slice=outStorage.slice+byte;
                    direct_domain_detail::emitMem3D(r,op.getLoc(),target,begin+latency+count-1+slice/4,
                        side*target.memory().slices_per_hemisphere+slice,"write",outStorage.base+word,
                        (1-side)*8+2+byte,1,1,0,1,1,0,1,1,0,-1,outStorage.bank);
                }
            }
            auto input=ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getRhs(),outType,
                begin-1,0,"bypass",fullInput?"stream_f32":"stream_bf16",0,0.f,"immediate",0,0.f,"fp32",-1,
                count,1,"east","east",-1,false,false,true,false,4);
            input->setAttr("lhs_stream_source",r.getStringAttr("local"));
            ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getRhs(),outType,
                begin-1,1,"add","previous",0,0.f,"accumulator",0,0.f,"fp32",-1,
                1,1,"east","east",-1,true,true,count==1,false,4);
            if(count>2)ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getRhs(),outType,
                begin,1,"add","previous",0,0.f,"accumulator",0,0.f,"fp32",-1,
                count-2,1,"east","east",-1,false,true,false,false,4);
            if(count>1)ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getRhs(),outType,
                begin+count-2,1,"add","previous",0,0.f,"accumulator",0,0.f,"fp32",-1,
                1,1,"east","east",-1,false,true,true,false,4);
            ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getRhs(),outType,
                begin-1,2,"bypass","previous",0,0.f,"immediate",0,0.f,"fp32",-1,
                1,1,"east","east",-1,false,false,true,false,4);
            ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getRhs(),outType,
                begin-1,3,"reciprocal","previous",0,0.f,"immediate",0,0.f,"fp32",2,
                1,1,"east","east",-1,false,false,true,false,4);
        }
        mlir::OperationState timeline(op.getLoc(),TimelineOp::getOperationName());
        timeline.addAttribute("name",r.getStringAttr("row_sum_reciprocal.fp32_chain"));
        timeline.addAttribute("start",r.getI64IntegerAttr(start-1));
        timeline.addAttribute("end",r.getI64IntegerAttr(start+latency+(outStorage.slice+3)/4+packedColumns*count));
        r.create(timeline);
        auto result=bind({},outType,0,"output",outStorage);
        r.replaceOp(tail,result.getValue());r.eraseOp(op);
        return mlir::success();
    }
    if (op.getKind()=="cumsum_rows" || reduceAlongRows(op)) {
        // Each lane owns a column; streaming rows in order gives a separate
        // FP32 accumulator for every query without a cross-lane shuffle.
        // Scan exports every prefix, whereas reduction exports only the end.
        const bool scan=!reduceAlongRows(op);
        if (!packed || (scan && packedColumns!=1) || !lhsStorage.both || !outStorage.both ||
            (scan && !op->getParentOfType<mlir::func::FuncOp>()->hasAttr("ftlpu.vector_bf16_spill")))
            return op.emitError("axis-zero scan/reduction requires mirrored packed fixed ports");
        const auto kind=op->getAttrOfType<mlir::StringAttr>("reduction_kind");
        const bool maximum=!scan && kind && kind.getValue()=="max";
        if (!scan && (!fp32Ingress || !op.getUseImmediate() || !kind ||
            (kind.getValue()!="add" && !maximum) || op.getKind()!="multiply" ||
            op.getImmediate().convertToFloat()!=1.f || !outType.getElementType().isF32() ||
            outType.getDimSize(0)!=1 || outType.getDimSize(1)!=lhsType.getDimSize(1)))
            return op.emitError("axis-zero reduction requires unary FP32-accumulating sum/true maximum and [1,columns] output");
        if(maximum) {
            auto seed=op->getAttrOfType<mlir::BoolAttr>("reduction_seed_first");
            if(!seed || !seed.getValue())
                return op.emitError("axis-zero maximum requires a first-value seed");
        }
        const int64_t count=lhsType.getDimSize(0), start=96;
        const bool fullInput=fp32Ingress && lhsType.getElementType().isF32();
        const int64_t first=lhsType.getElementType().isF32() && !fullInput?2:0;
        for(int64_t word=0;word<packedColumns;++word) {
        const int64_t begin=start+word*count;
        for(int64_t side=0;side<2;++side) {
            for(int64_t byte=0;byte<(fullInput?4:2);++byte) {
                const auto slice=lhsStorage.slice+first+byte;
                direct_domain_detail::emitMem3D(r,op.getLoc(),target,begin-1-slice/4,
                    side*target.memory().slices_per_hemisphere+slice,"read",lhsStorage.base+word*count,
                    32+side*16+byte,count,1,1,1,1,0,1,1,0,-1,lhsStorage.bank);
            }
            for(int64_t byte=0;byte<4;++byte) {
                const auto slice=outStorage.slice+byte;
                direct_domain_detail::emitMem3D(r,op.getLoc(),target,begin+2+slice/4+(scan?0:count-1),
                    side*target.memory().slices_per_hemisphere+slice,"write",outStorage.base+word*(scan?count:1),
                    (1-side)*8+byte,scan?count:1,1,1,1,1,0,1,1,0,-1,outStorage.bank);
            }
        }
        auto input=ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getRhs(),outType,
            begin-1,0,"bypass",fullInput?"stream_f32":"stream_bf16",0,0.f,"immediate",0,0.f,"fp32",-1,
            count,1,"east","east",-1,false,false,true,false,2);
        input->setAttr("lhs_stream_source",r.getStringAttr("local"));
        // reset + add(x, accumulator=0) seeds even entirely negative blocks.
        ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getRhs(),outType,
            begin-1,1,"add","previous",0,0.f,"accumulator",0,0.f,"fp32",0,
            1,1,"east","east",-1,true,true,true,false,2);
        if(count>1) ffn_detail::create_vxm(r,op.getLoc(),op.getLhs(),op.getRhs(),outType,
            begin,1,maximum?"max":"add","previous",0,0.f,"accumulator",0,0.f,"fp32",0,
            count-1,1,"east","east",-1,false,true,true,false,2);
        }
        mlir::OperationState timeline(op.getLoc(),TimelineOp::getOperationName());
        timeline.addAttribute("name",r.getStringAttr(scan?"cumsum_rows.fp32_accumulator":
            maximum?"row_max.fp32_accumulator":"row_sum.fp32_accumulator"));
        timeline.addAttribute("start",r.getI64IntegerAttr(start-1));
        timeline.addAttribute("end",r.getI64IntegerAttr(start+2+(outStorage.slice+3)/4+packedColumns*count));
        r.create(timeline);
        auto result=bind({},outType,0,"output",outStorage);
        r.replaceOp(op,result.getValue());
        return mlir::success();
    }
    if (!packed && fp32Ingress) {
        // Scalar-planar rows replicate each scalar across the 32 lanes. Use
        // the same fixed local ports as packed chains; reduce across cycles,
        // with a fresh accumulator for every row. No legacy group mux exists.
        if (!lhsStorage.both || !outStorage.both || (!op.getUseImmediate() && !rhsStorage.both))
            return op.emitError("fixed scalar chains require mirrored placements");
        const int64_t start=96, cols=lhsType.getDimSize(1), rows=lhsType.getDimSize(0);
        const int64_t count=op.getReduce()?cols:rows*cols;
        const int64_t interval=cols+96;
        const bool fullLhs=lhsType.getElementType().isF32(), fullRhs=rhsType.getElementType().isF32();
        const int64_t rhsPort=fullLhs || fullRhs?4:2;
        const auto read=[&](mlir::RankedTensorType ty,Storage st,int64_t port,bool rhs) {
            const bool bc=rhs && ty.getDimSize(1)==1, br=rhs && ty.getDimSize(0)==1;
            for(int64_t side=0;side<2;++side)for(int64_t byte=0;byte<(ty.getElementType().isF32()?4:2);++byte) {
                const auto slice=st.slice+byte;
                direct_domain_detail::emitMem3D(r,op.getLoc(),target,start-1-slice/4,
                    side*target.memory().slices_per_hemisphere+slice,"read",st.base,
                    32+side*16+port+byte,cols,1,bc?0:1,
                    rows,op.getReduce()?interval:cols,br?0:ty.getDimSize(1),1,1,0,-1,st.bank);
            }
        };
        read(lhsType,lhsStorage,0,false);
        const bool same=op.getLhs()==op.getRhs();
        if(!op.getUseImmediate() && !same) read(rhsType,rhsStorage,rhsPort,true);
        int64_t latency=8;
        mlir::Value current=op.getLhs();
        for(int64_t col=0;col<8;++col) {
            auto at=llvm::find(columns,col);
            const bool arithmetic=at!=columns.end(), reduction=op.getReduce() && col==1;
            auto member=arithmetic?chain[at-columns.begin()]:op;
            auto reductionKind=op->getAttrOfType<mlir::StringAttr>("reduction_kind");
            auto opcode=reduction?(reductionKind?reductionKind.getValue():llvm::StringRef("add")):
                arithmetic?member.getKind():llvm::StringRef("bypass");
            auto lhs=col?llvm::StringRef("previous"):llvm::StringRef(fullLhs?"stream_f32":"stream_bf16");
            auto rhs=reduction?llvm::StringRef("accumulator"):
                arithmetic && !member.getUseImmediate()?(member!=op || same?llvm::StringRef("original"):
                llvm::StringRef(fullRhs?"stream_f32":"stream_bf16")):llvm::StringRef("immediate");
            latency+=opcode=="exp" || opcode=="reciprocal" || opcode=="rsqrt" || opcode=="log"?4:opcode=="multiply"?1:0;
            if(reduction) {
                const auto seed=op->getAttrOfType<mlir::BoolAttr>("reduction_seed_first");
                const bool seedFirst=seed && seed.getValue();
                for(int64_t row=0;row<rows;++row) {
                    auto first=ffn_detail::create_vxm(r,op.getLoc(),current,current,outType,
                        start-1+row*interval,col,seedFirst?llvm::StringRef("add"):opcode,
                        "previous",0,0.f,"accumulator",0,0.f,
                        "fp32",-1,1,1,"east","west",-1,true,true,true,false,8);
                    if(count>1)first=ffn_detail::create_vxm(r,op.getLoc(),current,current,outType,
                        start+row*interval,col,opcode,"previous",0,0.f,"accumulator",0,0.f,
                        "fp32",-1,count-1,1,"east","west",-1,false,true,true,false,8);
                    current=first.getResult();
                }
                continue;
            }
            auto v=ffn_detail::create_vxm(r,op.getLoc(),current,current,outType,
                start-1,col,opcode,lhs,0,0.f,rhs,(rhs=="stream_f32" || rhs=="stream_bf16")?rhsPort:0,
                arithmetic?member.getImmediate().convertToFloat():0.f,
                col==7?(outType.getElementType().isF32()?"fp32":"bf16"):"fp32",col==7?6:-1,
                count,1,"east","west",-1,false,false,true,false,8);
            if(lhs.starts_with("stream_"))v->setAttr("lhs_stream_source",r.getStringAttr("local"));
            if(rhs.starts_with("stream_"))v->setAttr("rhs_stream_source",r.getStringAttr("local"));
            if(op.getReduce()) {
                v->setAttr("wave_count",r.getI64IntegerAttr(rows));
                v->setAttr("wave_interval",r.getI64IntegerAttr(interval));
            }
            current=v.getResult();
        }
        for(int64_t side=0;side<2;++side)for(int64_t byte=0;byte<(outType.getElementType().isF32()?4:2);++byte) {
            const auto slice=outStorage.slice+byte;
            direct_domain_detail::emitMem3D(r,op.getLoc(),target,start+latency+slice/4+(op.getReduce()?cols-1:0),
                side*target.memory().slices_per_hemisphere+slice,"write",outStorage.base,(1-side)*8+6+byte,
                op.getReduce()?1:count,1,1,op.getReduce()?rows:1,op.getReduce()?interval:1,
                op.getReduce()?1:0,1,1,0,-1,outStorage.bank);
        }
        auto result=bind({},outType,0,"output",outStorage);
        r.replaceOp(tail,result.getValue());
        for(int64_t i=chain.size()-2;i>=0;--i)r.eraseOp(chain[i]);
        return mlir::success();
    }
    if (auto seed=op->getAttrOfType<mlir::BoolAttr>("reduction_seed_first"); seed && seed.getValue())
        return op.emitError("first-element reduction currently requires fixed scalar FP32 ingress");
    if (update || op->getParentOfType<mlir::func::FuncOp>()->hasAttr("ftlpu.vector_bf16_spill")) {
        if (!packed || op.getReduce() || !lhsStorage.both || !outStorage.both
            || (!op.getUseImmediate() && !rhsStorage.both))
            return op.emitError("fixed-chain BF16 spill requires packed mirrored storage and no reduction")
                << "; lhs=" << op->getAttr("ftlpu.vector_lhs_placement")
                << "; rhs=" << op->getAttr("ftlpu.vector_rhs_placement")
                << "; output=" << tail->getAttr("ftlpu.vector_result_placement");
        // Full FP32 ingress consumes all four byte planes at fixed ports 0/4.
        // Only legacy narrow ingress truncates a four-plane input to BF16.
        const auto partitionPolicy = op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_bf16_token_halves");
        const bool partition = !update && partitionPolicy && partitionPolicy.getValue()
            && lhsType.getDimSize(0) == 32 && packedColumns == 1;
        const int64_t half = lhsType.getDimSize(0) / 2;
        const auto phaseOf = [](mlir::Value value) -> int64_t {
            auto *owner = value.getDefiningOp();
            if (!owner || owner->hasAttr("token_mirrored")) return -1;
            auto phase = owner->getAttrOfType<mlir::IntegerAttr>("token_phase");
            return phase ? phase.getInt() : -1;
        };
        const int64_t lhsPhase = lhsStorage.tokenPartition ? 1 : phaseOf(op.getLhs());
        const int64_t phase = lhsPhase < 0 ? 0 : lhsPhase;
        int64_t start = 96;
        // Fill only an operand's missing token half when two DAG branches
        // arrive with opposite native VXM output orientations. This is passive
        // SR traffic, not repeated arithmetic, and is reused by later consumers.
        const auto mirrorValue = [&](Storage st, mlir::RankedTensorType type,
                                     int64_t valuePhase, int64_t begin) {
            int64_t cursor = begin;
            for (int64_t side=0; side<2; ++side) {
                int64_t drain=cursor;
                const int64_t row=st.base+(side==valuePhase?0:half);
                for(int64_t byte=0; byte<(type.getElementType().isF32()?4:2); ++byte) {
                    const int64_t slice=st.slice+byte, latency=2*(slice/4)+2;
                    direct_domain_detail::emitMem3D(r,op.getLoc(),target,cursor,
                        side*target.memory().slices_per_hemisphere+slice,"read",row,48+byte,
                        half,1,1,1,1,0,1,1,0,-1,st.bank);
                    direct_domain_detail::emitMem3D(r,op.getLoc(),target,cursor+latency,
                        (1-side)*target.memory().slices_per_hemisphere+slice,"write",row,16+byte,
                        half,1,1,1,1,0,1,1,0,-1,st.bank);
                    drain=std::max(drain,cursor+latency+half+1);
                }
                cursor=drain;
            }
            return cursor;
        };
        if (partition && !op.getUseImmediate() && op.getRhs()!=op.getLhs()
            && rhsType.getDimSize(0)>1) {
            const int64_t rhsPhase=rhsStorage.tokenPartition ? 1 : phaseOf(op.getRhs());
            if(rhsPhase>=0 && rhsPhase!=phase) {
                if(rhsStorage.tokenPartition) return op.emitError("compact partition operand requires matching token orientation");
                start=mirrorValue(rhsStorage,rhsType,rhsPhase,start)+32;
                op.getRhs().getDefiningOp()->setAttr("token_mirrored",r.getUnitAttr());
            }
        }
        const int64_t count=partition?half:storageWords;
        const bool fullLhs=fp32Ingress && lhsType.getElementType().isF32();
        const bool fullRhs=fp32Ingress && rhsType.getElementType().isF32();
        if(fullLhs && !op.getUseImmediate() && op.getLhs()!=op.getRhs() && !fullRhs)
            return op.emitError("mixed FP32 lhs and narrow rhs share fixed input bytes; legalize operands to the same input format");
        const int64_t rhsPort=fullRhs?4:2;
        const auto readFixed=[&](mlir::RankedTensorType type,Storage st,int64_t port,bool broadcast) {
            const bool full=fp32Ingress && type.getElementType().isF32();
            const int64_t offset=type.getElementType().isF32() && !full?2:0;
            for(int64_t side=0;side<2;++side)for(int64_t byte=0;byte<(full?4:2);++byte) {
                const auto slice=st.slice+offset+byte;
                if(st.access) {
                    if(partition)llvm::report_fatal_error("planar read ownership requires mirrored packed update");
                    auto access=*st.access;
                    if(broadcast) {
                        access.rows=lhsType.getDimSize(0);access.rowChunk=access.rows;
                        access.rowStride=0;access.groupStride=0;
                    }
                    if(rowCoefficient && port==rhsPort)access.columnStride=0;
                    emitPlanarReadDomain(r,op.getLoc(),target,start-1-slice/4,
                        side*target.memory().slices_per_hemisphere+slice,st.base,
                        32+side*16+port+byte,st.bank,access,packedColumns);
                    continue;
                }
                if(st.broadcastRowRepeat) {
                    const int64_t heads=lhsType.getDimSize(0)/st.broadcastRowRepeat;
                    direct_domain_detail::emitMem3D(r,op.getLoc(),target,start-1-slice/4,
                        side*target.memory().slices_per_hemisphere+slice,"read",st.base+st.broadcastHeadBase,
                        32+side*16+port+byte,st.broadcastRowRepeat,1,0,
                        heads,st.broadcastRowRepeat,st.broadcastHeadStride,packedColumns,lhsType.getDimSize(0),0,-1,st.bank);
                    continue;
                }
                direct_domain_detail::emitMem3D(r,op.getLoc(),target,start-1-slice/4,
                    side*target.memory().slices_per_hemisphere+slice,"read",st.base+(partition && !st.tokenPartition && !broadcast && side!=phase?half:0),
                    32+side*16+port+byte,partition?half:lhsType.getDimSize(0),1,broadcast?0:1,
                    packedColumns,lhsType.getDimSize(0),rowCoefficient && port==rhsPort?0:type.getDimSize(0),1,1,0,-1,st.bank);
            }
        };
        readFixed(lhsType,lhsStorage,0,false);
        const bool same=op.getLhs()==op.getRhs();
        if(!op.getUseImmediate() && !same)
            readFixed(rhsType,rhsStorage,rhsPort,rhsType.getDimSize(0)==1);
        int64_t latency=8;
        mlir::Value current=op.getLhs();
        for(int64_t col=0;col<8;++col) {
            auto at=llvm::find(columns,col);
            const bool arithmetic=at!=columns.end();
            auto member=arithmetic?chain[at-columns.begin()]:op;
            auto opcode=arithmetic?member.getKind():llvm::StringRef("bypass");
            auto lhs=col?llvm::StringRef("previous"):llvm::StringRef(fullLhs?"stream_f32":"stream_bf16");
            auto rhs=arithmetic && !member.getUseImmediate()
                ? (member!=op || same?llvm::StringRef("original"):llvm::StringRef(fullRhs?"stream_f32":"stream_bf16"))
                : llvm::StringRef("immediate");
            latency+=opcode=="exp" || opcode=="reciprocal" || opcode=="rsqrt" || opcode=="log"?4:opcode=="multiply"?1:0;
            auto v=ffn_detail::create_vxm(r,op.getLoc(),current,current,outType,
                start-1,col,opcode,lhs,0,0.f,rhs,(rhs=="stream_bf16" || rhs=="stream_f32")?rhsPort:0,
                arithmetic?member.getImmediate().convertToFloat():0.f,
                col==7?(outType.getElementType().isF32()?"fp32":"bf16"):"fp32",
                col==7?6:-1,count,1,"east","west",-1,false,false,true,false,8);
            if(fp32Ingress) {
                if(lhs=="stream_f32" || lhs=="stream_bf16") v->setAttr("lhs_stream_source",r.getStringAttr("local"));
                if(rhs=="stream_f32" || rhs=="stream_bf16") v->setAttr("rhs_stream_source",r.getStringAttr("local"));
            }
            current=v.getResult();
        }
        const bool returned = partition && llvm::all_of(tail.getResult().getUsers(),
            [](mlir::Operation *user) { return llvm::isa<mlir::func::ReturnOp>(user); });
        const bool exported = returned && op->getParentOfType<mlir::func::FuncOp>()->hasAttr("ftlpu.vector_native_partition_outputs");
        const bool directTranspose=tail->hasAttr("ftlpu.vector_result_tile_transpose");
        if(directTranspose) {
            const auto wide=op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
            const int64_t tile=target.throughput().mxm_rows;
            if(!wide || !wide.getValue() || partition || update || tile!=32 ||
               lhsType.getDimSize(0)!=tile || packedColumns!=2 || !outType.getElementType().isBF16())
                return op.emitError("streamed result transpose requires two full mirrored BF16 tiles and wide SXM");
            // The fixed chain's final BF16 output appears before its usual
            // MEM write by VXM->MEM transport latency. Capture that exact
            // stream, preserving the cast and the key reduction order.
            const int64_t capture=start+latency-1+*target.transport_latency(
                target::StreamEndpoint::VxmResult,target::StreamEndpoint::SxmInput,
                target::StreamDirection::East,0);
            llvm::SmallVector<int64_t> identity;
            for(int64_t lane=0;lane<tile;++lane)identity.push_back(lane);
            int64_t end=capture;
            for(int64_t side=0;side<2;++side) {
                const int64_t inputStream=(1-side)*8+6;
                // Fixed FP32 input occupies reverse ports 0..3. Keep SXM
                // output on 4/5 while the next tile still enters VXM.
                attention_detail::emitSxm(r,op.getLoc(),capture,side,"transpose",
                    {inputStream,inputStream+1},{36,37},identity,"wide_matrix",
                    -1,-1,-1,packedColumns,tile);
                for(int64_t byte=0;byte<2;++byte) {
                    const int64_t slice=outStorage.slice+byte;
                    const int64_t at=capture+tile+target.throughput().tile_rows-1+
                        *target.transport_latency(target::StreamEndpoint::SxmResult,
                            target::StreamEndpoint::Mem,target::StreamDirection::West,slice);
                    direct_domain_detail::emitMem3D(r,op.getLoc(),target,at,
                        side*target.memory().slices_per_hemisphere+slice,"write",outStorage.base,
                        36+byte,tile,1,1,packedColumns,tile,tile,1,1,0,-1,outStorage.bank);
                    end=std::max(end,at+packedColumns*tile);
                }
            }
            mlir::OperationState timeline(op.getLoc(),TimelineOp::getOperationName());
            timeline.addAttribute("name",r.getStringAttr("vector.vxm_sxm_direct_transpose"));
            timeline.addAttribute("start",r.getI64IntegerAttr(capture));
            timeline.addAttribute("end",r.getI64IntegerAttr(end));
            r.create(timeline);
        } else {
        for(int64_t side=0;side<2;++side)
            for(int64_t byte=0;byte<(outType.getElementType().isF32()?4:2);++byte) {
                const auto slice=outStorage.slice+byte;
                direct_domain_detail::emitMem3D(r,op.getLoc(),target,start+latency+slice/4,
                    side*target.memory().slices_per_hemisphere+slice,"write",outStorage.base+(partition && !exported && side==phase?half:0),
                    (1-side)*8+6+byte,count,1,1,1,1,0,1,1,0,-1,outStorage.bank);
            }
        if(returned && !exported)
            mirrorValue(outStorage,outType,1-phase,start+latency+(outStorage.slice+3)/4+count+2);
        }
        auto result=bind({},outType,0,"output",outStorage);
        if(returned && !exported) result->setAttr("token_mirrored",r.getUnitAttr());
        if(partition) {
            result->setAttr("token_phase",r.getI64IntegerAttr(1-phase));
            if(exported) {
                mlir::NamedAttrList layout(result.getPlacement());
                layout.set("kind",r.getStringAttr(phase==0?"token_partition_planar":"token_partition_east_first_planar"));
                result->setAttr("placement",layout.getDictionary(r.getContext()));
            }
        }
        r.replaceOp(tail,result.getValue());
        for(int64_t i=chain.size()-2;i>=0;--i)r.eraseOp(chain[i]);
        return mlir::success();
    }
    const auto planes = [](mlir::RankedTensorType t) { return t.getElementType().isF32() ? 4 : 2; };
    const auto kind = [](mlir::RankedTensorType t) -> llvm::StringRef {
        return t.getElementType().isF32() ? "stream_f32" : lpu_16bit_stream_kind(t.getElementType());
    };
    const auto format = outType.getElementType().isF32()
        ? llvm::StringRef("fp32") : lpu_16bit_data_format(outType.getElementType());
    const int64_t tileCycles = wordElements / target.throughput().mem_read_bytes_per_cycle;
    const int64_t lastColumn = op.getReduce() ? 1 : columns.back() + 1;
    int64_t extraLatency = 0;
    for (auto member : chain) {
        const auto kind = member.getKind();
        extraLatency += kind == "exp" || kind == "reciprocal" || kind == "rsqrt" || kind == "log"
            ? 4 : kind == "multiply" ? 1 : 0;
    }
    int64_t cursor = 96;
    const int64_t passes = 1;
    // Returning to the East fabric may share the West-going SR path with an
    // incoming West operand. Use disjoint streams instead of colliding W0.
    // Mirrored peers still have raw operands on SR0/SR8 while this chain
    // drains. Its final result must not replace the peer's incoming lanes.
    const bool dual = op->getParentOfType<mlir::func::FuncOp>()
        ->hasAttr("ftlpu.vector_token_partition");
    const int64_t outputStream = dual ? 16 : outStorage.west ? 0 : 24;
    const int64_t reductionRows = op.getReduce() ? lhsType.getDimSize(0) : 1;
    const int64_t reductionRowInterval = lhsType.getDimSize(1) * tileCycles + 96;
    for (int64_t pass = 0; pass < passes; ++pass) {
        const int64_t count = packed ? storageWords
            : op.getReduce() ? lhsType.getDimSize(1) : lhsType.getNumElements();
        const auto read = [&](mlir::RankedTensorType type, Storage storage,
                              int64_t stream, bool rhs) {
            const int64_t columns = lhsType.getDimSize(1);
            const bool broadcastColumns = rhs && type.getDimSize(1) == 1 && columns > 1;
            const bool broadcastRows = rhs && type.getDimSize(0) == 1 && lhsType.getDimSize(0) > 1;
            const int64_t firstAddress = op.getReduce()
                ? (broadcastRows ? 0 : pass * type.getDimSize(1)) : 0;
            // Packed planar addresses are [column-word][token]. Broadcast a
            // per-channel row by repeating its word over tokens, then advance
            // to the next column-word. Padding lanes never enter a reduction.
            const int64_t inner = packed ? lhsType.getDimSize(0) : op.getReduce() ? count : columns;
            const int64_t waves = packed ? packedColumns : lhsType.getDimSize(0);
            for (int64_t byte = 0; byte < planes(type); ++byte) {
                const int64_t slice = storage.slice + byte;
                const auto latency = target.transport_latency(target::StreamEndpoint::Mem,
                    target::StreamEndpoint::VxmInput, target::StreamDirection::West, slice).value();
                direct_domain_detail::emitMem3D(r, op.getLoc(), target,
                    cursor - latency, slice + (storage.west ? target.memory().slices_per_hemisphere : 0),
                    "read", storage.base + firstAddress,
                    target.streams().streams_per_direction + stream + byte,
                    inner, tileCycles, (packed ? broadcastRows : broadcastColumns) ? 0 : 1,
                    waves, op.getReduce() ? reductionRowInterval : inner * tileCycles,
                    packed ? type.getDimSize(0) : broadcastRows ? 0 : type.getDimSize(1),
                    1, 1, 0, -1, storage.bank);
            }
        };
        read(lhsType, lhsStorage, 0, false);
        if (!op.getUseImmediate() && op.getRhs() != op.getLhs()) read(rhsType, rhsStorage, 8, true);
        mlir::Value current = op.getLhs();
        for (int64_t col = 0; col <= lastColumn; ++col) {
            const auto at = llvm::find(columns, col);
            const bool arithmetic = at != columns.end();
            auto member = arithmetic ? chain[at - columns.begin()] : op;
            const bool reduction = op.getReduce() && col == lastColumn;
            const auto reductionKind = op->getAttrOfType<mlir::StringAttr>("reduction_kind");
            const auto opcode = reduction ? (reductionKind ? reductionKind.getValue() : llvm::StringRef("add"))
                : arithmetic ? member.getKind() : llvm::StringRef("cast");
            const auto lhsKind = col == 0 ? kind(lhsType) : llvm::StringRef("previous");
            const auto rhsKind = reduction ? llvm::StringRef("accumulator")
                : arithmetic && !member.getUseImmediate() ? kind(rhsType) : llvm::StringRef("immediate");
            auto vxm = ffn_detail::create_vxm_with_terminal_controls(r, op.getLoc(),
                current, arithmetic && member == op ? op.getRhs() : current, outType, cursor - 1, queueBase + col, opcode,
                lhsKind, 0, 0.0f, rhsKind, rhsKind.starts_with("stream_") && op.getRhs() != op.getLhs() ? 8 : 0, arithmetic ? member.getImmediate().convertToFloat() : 0.0f,
                col == lastColumn ? format : llvm::StringRef("fp32"),
                col == lastColumn ? outputStream : -1, count, 1, "east", outStorage.west ? "east" : "west", -1,
                reduction, reduction, true, false, 8, reduction, reduction);
            if (col == 0) vxm->setAttr("lhs_stream_source", r.getStringAttr(lhsStorage.west ? "west" : "east"));
            if (arithmetic && !member.getUseImmediate())
                vxm->setAttr("rhs_stream_source", r.getStringAttr(rhsStorage.west ? "west" : "east"));
            if (op.getReduce()) {
                vxm->setAttr("wave_count", r.getI64IntegerAttr(reductionRows));
                vxm->setAttr("wave_interval", r.getI64IntegerAttr(reductionRowInterval));
            }
            current = vxm.getResult();
        }
        // A result returning left traverses the final ALU columns twice.
        // West-boundary drains keep the existing passive-chain timing.
        const int64_t resultCycle = cursor + (outStorage.west
            ? (lhsStorage.west ? 32 - 2 * queueBase : 17)
            : (lhsStorage.west ? 19 : 4 + 2 * queueBase) + 2 * (lastColumn - 1)) + extraLatency
            + (op.getReduce() ? (count - 1) * tileCycles : 0);
        for (int64_t byte = 0; byte < planes(outType); ++byte) {
            const int64_t slice = outStorage.slice + byte;
            const auto latency = target.transport_latency(target::StreamEndpoint::VxmResult,
                target::StreamEndpoint::Mem, target::StreamDirection::East, slice).value();
            direct_domain_detail::emitMem3D(r, op.getLoc(), target,
                resultCycle + latency, slice + (outStorage.west ? target.memory().slices_per_hemisphere : 0),
                "write", outStorage.base + (op.getReduce() ? pass : 0), outputStream + byte,
                op.getReduce() ? 1 : count, tileCycles, 1,
                reductionRows, op.getReduce() ? reductionRowInterval : 1,
                op.getReduce() ? 1 : 0, 1, 1, 0, -1, outStorage.bank);
        }
        cursor += count * tileCycles + 96;
    }
    auto result = bind({}, outType, 0, "output", outStorage);
    r.replaceOp(tail, result.getValue());
    for (int64_t i = static_cast<int64_t>(chain.size()) - 2; i >= 0; --i)
        r.eraseOp(chain[i]);
    return mlir::success();
}
} // namespace

mlir::LogicalResult planVectorAluMemory(mlir::func::FuncOp function,
    const target::LPUTargetModel& target)
{
    if (mlir::failed(planScaledMatvecMemory(function,target))) return mlir::failure();
    // Expand mathematical unary operators before storage planning. Importers
    // supply sigmoid/softplus semantics, never a hardware approximation or
    // instruction sequence. Every generated value is executed by VXM.
    llvm::SmallVector<kernel::VectorAluOp> semantic;
    function.walk([&](kernel::VectorAluOp op) {
        if (op.getKind() == "sigmoid" || op.getKind() == "softplus")
            semantic.push_back(op);
    });
    mlir::IRRewriter expansion(function.getContext());
    for (auto op : semantic) {
        if (op->hasAttr("ftlpu.vector_result_placement"))
            return op.emitError("semantic vector operations must expand before memory planning");
        expansion.setInsertionPoint(op);
        const auto alu = [&](llvm::StringRef kind, mlir::Value lhs,
                             mlir::Value rhs = {}, float immediate = 0.0f) {
            mlir::OperationState state(op.getLoc(), kernel::VectorAluOp::getOperationName());
            state.addOperands({lhs, rhs ? rhs : lhs});
            state.addTypes(op.getResult().getType());
            state.addAttributes({
                expansion.getNamedAttr("kind", expansion.getStringAttr(kind)),
                expansion.getNamedAttr("use_immediate", expansion.getBoolAttr(!rhs)),
                expansion.getNamedAttr("immediate", expansion.getF32FloatAttr(immediate)),
                expansion.getNamedAttr("reduce", expansion.getBoolAttr(false)),
            });
            return expansion.create(state)->getResult(0);
        };
        auto x = op.getLhs();
        mlir::Value result;
        if (op.getKind() == "sigmoid") {
            auto neg = alu("negate", x);
            auto exponential = alu("exp", neg);
            result = alu("reciprocal", alu("add", exponential, {}, 1.0f));
        } else if (auto policy = function->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_log_softplus");
                   policy && policy.getValue()) {
            // Split t=exp(-abs(x)) at an exactly representable boundary.
            // Log sees >=1+1/16, avoiding FP16 cancellation around 1.
            // Its baseline is subtracted inside the chain before BF16 spill.
            // Below the boundary use u-u^2/2+u^3/3 (absolute error <4e-6).
            constexpr float boundary = 0.0625f;
            // Commutative max keeps negate -> max(previous, Original) in
            // one chain. No opposite-side spill/mirror is needed for abs(x).
            auto absolute = alu("max", alu("negate", x), x);
            auto exponential = alu("exp", alu("negate", absolute));
            auto upper = alu("max", exponential, {}, boundary);
            auto logarithm = alu("log", alu("add", upper, {}, 1.0f));
            // The standard runtime LUT stores its baseline in FP16. Match
            // that coefficient exactly so the clamped branch produces zero.
            llvm::APFloat baseline(std::log1p(boundary));
            bool losesInfo = false;
            baseline.convert(llvm::APFloat::IEEEhalf(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
            auto large = alu("subtract", logarithm, {}, baseline.convertToFloat());
            // p(t)=t-t^2/2+t^3/3 is monotone on [0,1], hence
            // p(min(t,a))=min(p(t),p(a)). Clamp after the polynomial:
            // every multiply can reuse the chain's Original=t, eliminating
            // the clamped-t spill and keeping both branch outputs on the
            // same token hemisphere. C7 performs the final negate/output.
            float limit=boundary*(1.0f/3.0f);
            limit=(limit-0.5f)*boundary;
            limit=(limit+1.0f)*boundary;
            auto polynomial = alu("add", alu("multiply", exponential, {}, 1.0f/3), {}, -0.5f);
            polynomial = alu("add", alu("multiply", polynomial, exponential), {}, 1.0f);
            polynomial = alu("multiply", polynomial, exponential);
            auto small = alu("negate", alu("max", alu("negate", polynomial), {}, -limit));
            auto correction = alu("add", large, small);
            result = alu("add", alu("max", x), correction);
        } else {
            // softplus(x) = max(x,0) + 2*atanh(t),
            // t = exp(-abs(x))/(2+exp(-abs(x))) in [0,1/3].
            // Six odd Taylor terms bound truncation error below 1.1e-7;
            // exp never sees a positive input, including for large logits.
            auto absolute = alu("max", x, alu("negate", x));
            auto exponential = alu("exp", alu("negate", absolute));
            auto inverse = alu("reciprocal", alu("add", exponential, {}, 2.0f));
            auto t = alu("multiply", exponential, inverse);
            auto square = alu("multiply", t, t);
            auto polynomial = alu("add", alu("multiply", square, {}, 1.0f/11), {}, 1.0f/9);
            for (int degree : {7, 5, 3, 1})
                polynomial = alu("add", alu("multiply", polynomial, square), {}, 1.0f/degree);
            auto correction = alu("multiply", alu("multiply", polynomial, t), {}, 2.0f);
            result = alu("add", alu("max", x), correction);
        }
        expansion.replaceOp(op, result);
    }
    llvm::SmallVector<kernel::VectorAluOp> ops;
    function.walk([&](kernel::VectorAluOp op) { ops.push_back(op); });
    if (ops.empty()) return mlir::success();
    // Capture phases finish before the binary stream starts. Their sources
    // may share MEM queues with ordinary data reads without simultaneous use.
    llvm::SmallVector<mlir::Attribute> captured;
    const auto captureArgument=[&](mlir::Value value) {
        if(auto arg=llvm::dyn_cast<mlir::BlockArgument>(value))
            captured.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(function.getContext(),64),arg.getArgNumber()));
    };
    if(ops.size()==2 && matchScaleAdd(ops[0],ops[1]) && ops[0].getRhs()!=ops[0].getLhs() && ops[0].getRhs()!=ops[1].getRhs())
        captureArgument(ops[0].getRhs());
    if(ops.size()==2 && matchAddScale(ops[0],ops[1]) && ops[1].getRhs()!=ops[0].getLhs() && ops[1].getRhs()!=ops[0].getRhs())
        captureArgument(ops[1].getRhs());
    if(auto fold=matchPointwiseFold(ops);fold.columns.size()==ops.size())
        for(size_t i=2;i<fold.inputs.size();++i)
            if(fold.inputs[i]!=fold.inputs[0] && fold.inputs[i]!=fold.inputs[1])captureArgument(fold.inputs[i]);
    if(!captured.empty())function->setAttr("ftlpu.vector_capture_arguments",mlir::ArrayAttr::get(function.getContext(),captured));
    if (function->hasAttr("ftlpu.vector_bf16_spill") && matchSiluMultiply(function))
        function->setAttr("ftlpu.native_silu_multiply",mlir::UnitAttr::get(function.getContext()));
    // A graph-level plan is authoritative; repeated lowering must not reset it.
    if (llvm::any_of(ops, [](auto op) { return op->hasAttr("ftlpu.vector_result_placement"); })) {
        for (auto op : ops) {
            const auto packed = op->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_packed");
            if (!packed) return op.emitError("incomplete vector memory plan: missing packing policy");
            for (const auto name : {"ftlpu.vector_lhs_placement", "ftlpu.vector_rhs_placement", "ftlpu.vector_result_placement"}) {
                const auto p = op->getAttrOfType<mlir::DictionaryAttr>(name);
                if (!p) return op.emitError("incomplete vector memory plan");
                const auto ids = p.getAs<mlir::ArrayAttr>("slices");
                const auto bank = p.getAs<mlir::IntegerAttr>("bank");
                const auto base = p.getAs<mlir::IntegerAttr>("base_row");
                const auto count = p.getAs<mlir::IntegerAttr>("instruction_count");
                const auto hemi = p.getAs<mlir::StringAttr>("hemisphere");
                if (!ids || ids.empty() || !bank || !base || !count || !hemi ||
                    (hemi.getValue() != "east" && hemi.getValue() != "west" && hemi.getValue() != "both") ||
                    bank.getInt() < 0 || bank.getInt() >= target.memory().banks_per_slice ||
                    base.getInt() < 0 || count.getInt() <= 0 ||
                    base.getInt() + count.getInt() > target.memory().sram_depth_rows)
                    return op.emitError("invalid vector memory plan bounds");
                int64_t prior = -1;
                for (auto id : ids) {
                    const auto slice = llvm::dyn_cast<mlir::IntegerAttr>(id);
                    if (!slice || slice.getInt() < 0 || slice.getInt() >= target.memory().slices_per_hemisphere ||
                        (prior >= 0 && slice.getInt() != prior + 1))
                        return op.emitError("vector memory plan requires consecutive byte-plane slices");
                    prior = slice.getInt();
                }
            }
        }
        return mlir::success();
    }
    mlir::IRRewriter r(function.getContext());
    const auto scalarPolicy=function->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_scalar_layout");
    const bool update=packedUpdate(function);
    const bool timeReduction=llvm::any_of(ops, [](auto op){return reduceAlongRows(op);});
    const bool packed = !(scalarPolicy && scalarPolicy.getValue()) && (update || llvm::all_of(ops, [&](auto op) {
        const auto lhs = op.getLhs().getType(), rhs = op.getRhs().getType();
        if(reduceAlongRows(op)) return true;
        if(op.getKind()=="head_last_row_scale_transpose" || op.getKind()=="head_cross_product" || op.getKind()=="gram_row_scale" || op.getKind()=="head_mask" || op.getKind()=="cross_gram" || op.getKind()=="weighted_gram" || (op.getKind()=="unit_lower_solve" || op.getKind()=="unit_lower_solve_blocked") || op.getKind()=="head_scale" || op.getKind()=="unit_lower_inverse" || op.getKind()=="head_matmul") return true;
        return !op.getReduce() && (op.getUseImmediate()
            || (lhs.getDimSize(1) == rhs.getDimSize(1)
                && (lhs.getDimSize(0) == rhs.getDimSize(0) || rhs.getDimSize(0) == 1)));
    }));
    // Quantize only existing spill boundaries, not values kept inside an ALU
    // chain. A fixed BF16 input port must consume an actual RNE output rather
    // than silently dropping the low planes of a stored FP32 intermediate.
    const auto rneSpills = function->getParentOfType<mlir::ModuleOp>()
        ->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_bf16_rne_spill");
    if (function->hasAttr("ftlpu.vector_bf16_spill") && rneSpills && rneSpills.getValue()) {
        for (std::size_t first=0; first<ops.size();) {
            const auto columns=chainColumns(llvm::ArrayRef(ops).drop_front(first),packed);
            auto value=ops[first+columns.size()-1].getResult();
            const bool returned=llvm::any_of(value.getUsers(),[](auto *user) {
                return llvm::isa<mlir::func::ReturnOp>(user);
            });
            auto type=llvm::cast<mlir::RankedTensorType>(value.getType());
            if (!returned && type.getElementType().isF32() &&
                !value.getDefiningOp()->hasAttr("ftlpu.vector_require_fp32"))
                value.setType(mlir::RankedTensorType::get(type.getShape(),r.getBF16Type()));
            first+=columns.size();
        }
    }
    llvm::DenseMap<mlir::Value, Storage> storage;
    llvm::DenseMap<mlir::Value, int64_t> lastUse;
    for (auto [index, op] : llvm::enumerate(ops)) {
        lastUse[op.getLhs()] = index;
        if (!op.getUseImmediate()) lastUse[op.getRhs()] = index;
    }
    for (std::size_t first = 0; first < ops.size();) {
        const auto columns = chainColumns(llvm::ArrayRef(ops).drop_front(first), packed);
        const int64_t end = first + columns.size() - 1;
        for (auto value : {ops[first].getLhs(), ops[first].getRhs()})
            lastUse[value] = std::max(lastUse.lookup(value), end);
        first = end + 1;
    }
    function.walk([&](mlir::func::ReturnOp ret) {
        for (auto value : ret.getOperands()) lastUse[value] = ops.size();
    });
    int64_t inputSlot = 0;
    for (auto op : ops)
        for (auto value : {op.getLhs(), op.getRhs()}) {
            if (!llvm::isa<mlir::BlockArgument>(value) || storage.contains(value)) continue;
            if (inputSlot >= 4) return op.emitError("vector graph currently supports at most four external inputs");
            storage[value] = ops.size() == 1
                ? Storage{inputSlot == 0 ? 0 : 8, 0, false}
                : Storage{4 * inputSlot, 0, true};
            ++inputSlot;
        }
    struct Live { int64_t slot; int64_t end; };
    llvm::SmallVector<Live> live;
    const int64_t slotsPerBank = (target.memory().slices_per_hemisphere - 16) / 4;
    const int64_t capacity = slotsPerBank * target.memory().banks_per_slice;
    if (slotsPerBank <= 0) return function.emitError("no vector graph scratch slice groups");
    const bool requireFp32 = llvm::any_of(ops, [](auto alu) {
        return alu->hasAttr("ftlpu.vector_require_fp32");
    });
    const auto plan = [&](mlir::Value value, Storage st) {
        const auto ingress=function->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_fp32_ingress");
        const bool fixedScalar = !packed && (requireFp32 || (ingress && ingress.getValue()));
        const auto type=llvm::cast<mlir::RankedTensorType>(value.getType());
        const bool wordPacked=packed && !(update && type.getDimSize(1)==1);
        return placement(r, type,
            st.slice, (fixedScalar || update || timeReduction || function->hasAttr("ftlpu.vector_bf16_spill")) ? "both" : st.west ? "west" : "east", st.bank, wordPacked, target.memory().bytes_per_word);
    };
    for (auto [index, op] : llvm::enumerate(ops)) {
        llvm::erase_if(live, [&](const Live& item) { return item.end < static_cast<int64_t>(index); });
        int64_t slot = 0;
        for (; slot < capacity; ++slot)
            if (llvm::none_of(live, [&](const Live& item) { return item.slot == slot; })) break;
        if (slot == capacity) return op.emitError("vector graph live values exceed MEM capacity");
        if (!storage.contains(op.getLhs()) || !storage.contains(op.getRhs()))
            return op.emitError("vector graph operand has no storage plan");
        const Storage output = ops.size() == 1 ? Storage{48, 0, true}
            : Storage{16 + (slot % slotsPerBank) * 4, slot / slotsPerBank, true};
        op->setAttr("ftlpu.vector_lhs_placement", plan(op.getLhs(), storage.lookup(op.getLhs())));
        op->setAttr("ftlpu.vector_rhs_placement", plan(op.getRhs(), storage.lookup(op.getRhs())));
        op->setAttr("ftlpu.vector_result_placement", plan(op.getResult(), output));
        if(op.getKind()=="head_last_row_scale_transpose" || op.getKind()=="head_cross_product") {
            for(auto name:{"ftlpu.vector_lhs_placement","ftlpu.vector_rhs_placement","ftlpu.vector_result_placement"}) {
                mlir::NamedAttrList p(op->getAttrOfType<mlir::DictionaryAttr>(name));
                p.set("hemisphere",r.getStringAttr("both"));
                if(op.getKind()=="head_cross_product")
                    p.set("kind",r.getStringAttr(llvm::StringRef(name)=="ftlpu.vector_rhs_placement"?"token_partition_east_first_planar":"token_partition_planar"));
                if(op.getKind()=="head_cross_product" && llvm::StringRef(name)=="ftlpu.vector_result_placement")
                    p.set("instruction_count",r.getI64IntegerAttr(op.getResult().getType().getNumElements()/64));
                op->setAttr(name,p.getDictionary(r.getContext()));
            }
        }
        if(op.getKind()=="gram_row_scale" || op.getKind()=="head_mask" || op.getKind()=="weighted_gram" || op.getKind()=="cross_gram" ||
           (op.getKind()=="multiply" && op.getLhs().getDefiningOp<kernel::VectorAluOp>() &&
            op.getLhs().getDefiningOp<kernel::VectorAluOp>().getKind()=="cross_gram")) {
            mlir::NamedAttrList result(op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement"));
            // Return native head groups [even heads, odd heads]. Logical
            // consumers can express the interleave as a tensor view rather
            // than requiring a physical output archive.
            result.set("kind",r.getStringAttr("token_partition_east_first_planar"));
            op->setAttr("ftlpu.vector_result_placement",result.getDictionary(r.getContext()));
            if(op.getKind()=="weighted_gram" && !op.getRhs().getType().getElementType().isBF16()) {
                auto table=mlir::RankedTensorType::get({1024,32},r.getBF16Type());
                op->setAttr("weighted_gram.r_table",placement(r,table,48,"both",0,true,32));
            }
        }
        if(op.getKind()=="gram_row_scale") {
            op->setAttr("gram_scale.heads",placement(r,mlir::RankedTensorType::get({op.getLhs().getType().getDimSize(0),32},r.getF32Type()),28,"both",0,true,32));
            op->setAttr("gram_scale.masked",placement(r,mlir::RankedTensorType::get({op.getLhs().getType().getDimSize(0),32},r.getF32Type()),24,"both",1,true,32));
            op->setAttr("gram_scale.broadcast",placement(r,mlir::RankedTensorType::get({op.getLhs().getType().getDimSize(0),32},r.getF32Type()),24,"both",0,true,32));
        }
        if((op.getKind()=="unit_lower_solve" || op.getKind()=="unit_lower_solve_blocked") || op.getKind()=="head_scale") {
            const bool solve=(op.getKind()=="unit_lower_solve" || op.getKind()=="unit_lower_solve_blocked");
            auto table=mlir::RankedTensorType::get({32,32},r.getF32Type());
            // Head scaling uses bank ping-pong near the SR midpoint, keeping
            // local and bridged byte planes close enough for a 32-cycle II.
            int64_t coefficientSlice=24;
            const auto module=function->getParentOfType<mlir::ModuleOp>();
            // The operator requests four independent coefficient byte planes.
            // Graph policy supplies reserved page queues; no model/test branch
            // is responsible for moving this scratch when overlap is enabled.
            if(solve && module->hasAttr("ftlpu.weight_paging_policy"))
                coefficientSlice=target::GraphWeightPaging::resolve(target,module->getAttrDictionary())
                    .scratchSlice(coefficientSlice,4,target.memory().slices_per_hemisphere);
            op->setAttr("solve.coefficients",placement(r,table,coefficientSlice,"both",0,true,32));
            op->setAttr("solve.coefficients_alternate",placement(r,table,coefficientSlice,"both",1,true,32));
            if(solve) {
                for(auto name:{"ftlpu.vector_lhs_placement","ftlpu.vector_rhs_placement"}) {
                    mlir::NamedAttrList input(op->getAttrOfType<mlir::DictionaryAttr>(name));
                    input.set("kind",r.getStringAttr(llvm::StringRef(name)=="ftlpu.vector_lhs_placement"?"token_partition_east_first_planar":"token_partition_planar"));
                    op->setAttr(name,input.getDictionary(r.getContext()));
                }
                auto feedback=mlir::RankedTensorType::get({32,op.getRhs().getType().getDimSize(1)},r.getF32Type());
                op->setAttr("solve.feedback",placement(r,feedback,36,"both",0,true,32));
                if(op.getKind()=="unit_lower_solve_blocked") {
                    auto tile=mlir::RankedTensorType::get({32,32},r.getBF16Type());
                    auto product=mlir::RankedTensorType::get({16,op.getRhs().getType().getDimSize(1)},r.getF32Type());
                    op->setAttr("solve_block.weight_tile",placement(r,tile,40,"both",0,true,32));
                    op->setAttr("solve_block.product",placement(r,product,44,"both",0,true,32));
                }
            }
            mlir::NamedAttrList out(op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement"));
            out.set("kind",r.getStringAttr(solve?"token_partition_east_first_planar":"token_partition_planar"));
            op->setAttr("ftlpu.vector_result_placement",out.getDictionary(r.getContext()));
        }
        if(op.getKind()=="unit_lower_inverse" || op.getKind()=="head_matmul") {
            const bool inverse=op.getKind()=="unit_lower_inverse";
            const bool sharedWeight=!inverse && llvm::any_of(op.getRhs().getUsers(),[](mlir::Operation *user) {
                auto other=llvm::dyn_cast<kernel::VectorAluOp>(user);
                return other && other.getKind()=="head_cross_product";
            });
            for(auto name:{"ftlpu.vector_lhs_placement","ftlpu.vector_rhs_placement","ftlpu.vector_result_placement"}) {
                mlir::NamedAttrList p(op->getAttrOfType<mlir::DictionaryAttr>(name));
                const bool result=llvm::StringRef(name)=="ftlpu.vector_result_placement";
                p.set("kind",r.getStringAttr((sharedWeight?!result:(inverse!=result))?"token_partition_east_first_planar":"token_partition_planar"));
                // Long native K contractions keep two ordinary mirrored
                // operand matrices; the result alone is partitioned. This
                // also gives C2C the normal finite planar allocation path.
                if(!inverse && !result && op.getLhs().getType().getDimSize(1)>32)
                    p.set("kind",r.getStringAttr("fp16_pair_planar"));
                p.set("hemisphere",r.getStringAttr("both"));
                op->setAttr(name,p.getDictionary(r.getContext()));
            }
            const auto precision=op->getAttrOfType<mlir::StringAttr>("mxm_precision");
            if(inverse || !precision || precision.getValue()!="bf16") {
            auto tile=mlir::RankedTensorType::get({32,32},r.getBF16Type());
            op->setAttr("solve_mxm.weight_tile",placement(r,tile,40,"both",0,true,32));
            auto residualType=mlir::RankedTensorType::get({32,32},r.getF32Type());
            op->setAttr("solve_mxm.a1",placement(r,residualType,24,"both",0,true,32));
            op->setAttr("solve_mxm.a2",placement(r,residualType,24,"both",1,true,32));
            op->setAttr("solve_mxm.b1",placement(r,residualType,28,"both",0,true,32));
            op->setAttr("solve_mxm.b2",placement(r,residualType,28,"both",1,true,32));
            op->setAttr("solve_mxm.residual_bridge",placement(r,residualType,36,"both",1,true,32));
            }
            if(inverse) {
                auto matrix=mlir::RankedTensorType::get({32,32},r.getF32Type());
                op->setAttr("solve_mxm.t0",placement(r,matrix,16,"both",0,true,32));
                op->setAttr("solve_mxm.t1",placement(r,matrix,16,"both",1,true,32));
                op->setAttr("solve_mxm.p0",placement(r,matrix,20,"both",0,true,32));
                op->setAttr("solve_mxm.p1",placement(r,matrix,20,"both",1,true,32));
            }
        }
        if(auto product=op.getLhs().getDefiningOp<kernel::VectorAluOp>(); isHeadMatmulAdd(product,op)) {
            // The previous FP32 partial sum remains a real input. The new
            // MXM product feeds C0 directly and therefore needs no MEM home.
            mlir::NamedAttrList prior(op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_rhs_placement"));
            prior.set("kind",r.getStringAttr("fp32_quad_planar"));
            prior.set("hemisphere",r.getStringAttr("both"));
            op->setAttr("ftlpu.vector_rhs_placement",prior.getDictionary(r.getContext()));
            mlir::NamedAttrList result(op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement"));
            result.set("kind",product->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement").get("kind"));
            result.set("hemisphere",r.getStringAttr("both"));
            op->setAttr("ftlpu.vector_result_placement",result.getDictionary(r.getContext()));
        }
        if(auto sum=op.getLhs().getDefiningOp<kernel::VectorAluOp>(); sum &&
            isHeadMatmulAdd(sum.getLhs().getDefiningOp<kernel::VectorAluOp>(),sum) && isHeadMatmulCast(sum,op)) {
            mlir::NamedAttrList result(op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement"));
            result.set("kind",sum->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement").get("kind"));
            result.set("hemisphere",r.getStringAttr("both"));
            op->setAttr("ftlpu.vector_result_placement",result.getDictionary(r.getContext()));
        }
        op->setAttr("ftlpu.vector_packed", r.getBoolAttr(packed));
        storage[op.getResult()] = output;
        live.push_back({slot, lastUse.lookup(op.getResult())});
    }
    return mlir::success();
}

mlir::LogicalResult lowerVectorAluSchedules(mlir::IRRewriter& r,
    mlir::func::FuncOp function, const target::LPUTargetModel& target)
{
    if (mlir::failed(planVectorAluMemory(function, target))) return mlir::failure();
    if (mlir::failed(lowerScaledMatvecSchedules(r,function,target))) return mlir::failure();
    if(function->hasAttr("native_row_scale_words"))
        return emitNativePackedRowScale(r,function,target);
    if(function->hasAttr("native_pointwise_words_0"))
        if(auto match=matchSiluMultiply(function))
            return emitNativeSiluMultiply(r,function,target,*match);
    llvm::SmallVector<kernel::VectorAluOp> ops;
    function.walk([&](kernel::VectorAluOp op) { ops.push_back(op); });
    const auto queuePolicy = function->getAttrOfType<mlir::IntegerAttr>("ftlpu.vxm_queue_base");
    const int64_t queueBase = queuePolicy ? queuePolicy.getInt() : 0;
    if (queueBase != 0 && queueBase != 8) return function.emitError("VXM queue base must select one physical eight-ALU group");
    for (std::size_t index = 0; index < ops.size();) {
        auto op = ops[index];
        const bool packed = op->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_packed").getValue();
        const auto columns = chainColumns(llvm::ArrayRef(ops).drop_front(index), packed);
        const auto chain = llvm::ArrayRef(ops).slice(index, columns.size());
        const auto count = chain.size();
        auto *before = op->getPrevNode();
        if (mlir::failed(lowerOne(r, op, target,
            storageOf(op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_lhs_placement")),
            storageOf(op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_rhs_placement")),
            storageOf(chain.back()->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement")),
            packed, queueBase, chain, columns))) return mlir::failure();
        const auto mirror = function->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_mirror");
        if (mirror && mirror.getValue()) {
            // Reflect the closed-form hardware domains, including the operand
            // references. No FU trace expansion or model-specific operation.
            const auto flip = [&](llvm::StringRef value) {
                return r.getStringAttr(value == "east" ? "west" : "east");
            };
            auto* generated = before ? before->getNextNode()
                                                 : &function.front().front();
            for (; generated; generated = generated->getNextNode()) {
                if (auto mem = llvm::dyn_cast<MemTransferOp>(generated)) {
                    mem->setAttr("hemisphere", r.getI64IntegerAttr(1 - mem.getHemisphere()));
                } else if (auto vxm = llvm::dyn_cast<VxmOp>(generated)) {
                    const int64_t queue = vxm.getQueue();
                    for (const auto prefix : {llvm::StringRef("lhs"), llvm::StringRef("rhs")}) {
                        const auto kindName = (prefix + "_kind").str();
                        const auto indexName = (prefix + "_index").str();
                        const auto kind = vxm->getAttrOfType<mlir::StringAttr>(kindName).getValue();
                        const auto sourceName = (prefix + "_stream_source").str();
                        if (const auto source = vxm->getAttrOfType<mlir::StringAttr>(sourceName))
                            vxm->setAttr(sourceName, flip(source.getValue()));
                        if (kind == "previous" || kind == "alu") {
                            const auto producer = kind == "previous" ? queue - 1
                                : vxm->getAttrOfType<mlir::IntegerAttr>(indexName).getInt();
                            const auto reflected = target.throughput().vxm_alus - 1 - producer;
                            vxm->setAttr(kindName, r.getStringAttr("stream_f32"));
                            vxm->setAttr(indexName, r.getI64IntegerAttr(
                                16 + 4 * (reflected % 4)));
                            vxm->setAttr(sourceName, r.getStringAttr("west"));
                        }
                    }
                    if (vxm.getOutputStreamAttr().getInt() < 0) {
                        vxm->setAttr("output_stream", r.getI64IntegerAttr(
                            16 + 4 * ((target.throughput().vxm_alus - 1 - queue) % 4)));
                    }
                    vxm->setAttr("queue", r.getI64IntegerAttr(target.throughput().vxm_alus - 1 - queue));
                    vxm->setAttr("input_hemisphere", flip(vxm.getInputHemisphere()));
                    vxm->setAttr("output_hemisphere", flip(vxm.getOutputHemisphere()));
                } else if (auto bind = llvm::dyn_cast<BindingOp>(generated)) {
                    mlir::NamedAttrList attributes(bind.getPlacement());
                    attributes.set("hemisphere", flip(bind.getPlacement()
                        .getAs<mlir::StringAttr>("hemisphere").getValue()));
                    bind->setAttr("placement", attributes.getDictionary(r.getContext()));
                    if (bind.getAccess() == "output") break;
                }
            }
        }
        auto *inserted = before ? before->getNextNode() : &function.front().front();
        for (; inserted; inserted = inserted->getNextNode()) {
            inserted->setAttr("ftlpu.schedule_stage", r.getI64IntegerAttr(index));
            if (auto binding = llvm::dyn_cast<BindingOp>(inserted))
                if (binding.getAccess() == "output") break;
        }
        index += count;
    }
    return mlir::success();
}
} // namespace ftlpu::compiler::schedule
