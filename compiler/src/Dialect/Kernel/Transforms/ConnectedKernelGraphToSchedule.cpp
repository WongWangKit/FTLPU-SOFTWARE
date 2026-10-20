#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"
#include "ftlpu/compiler/Transforms/logical_kernel_graph.hpp"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ftlpu::compiler {
namespace {

int64_t integer(mlir::Operation *op, llvm::StringRef name, int64_t fallback = 0) {
  auto a = op->getAttrOfType<mlir::IntegerAttr>(name);
  return a ? a.getInt() : fallback;
}
int64_t field(mlir::DictionaryAttr p, llvm::StringRef name, int64_t fallback = 0) {
  auto a = p.getAs<mlir::IntegerAttr>(name);
  return a ? a.getInt() : fallback;
}
llvm::SmallVector<int64_t> slices(mlir::DictionaryAttr p) {
  llvm::SmallVector<int64_t> result;
  for (auto a : p.getAs<mlir::ArrayAttr>("slices"))
    result.push_back(llvm::cast<mlir::IntegerAttr>(a).getInt());
  return result;
}
bool west(mlir::DictionaryAttr p) {
  return p.getAs<mlir::StringAttr>("hemisphere").getValue() == "west";
}
int64_t lastIssue(mlir::Operation *op) {
  int64_t end = integer(op, "cycle");
  for (auto axis : {std::pair{"repeat_count", "repeat_interval"},
                    std::pair{"wave_count", "wave_interval"},
                    std::pair{"group_count", "group_interval"}})
    end += (integer(op, axis.first, 1) - 1) * integer(op, axis.second, 1);
  return end + 1;
}

struct NativeStage {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  llvm::SmallVector<mlir::DictionaryAttr> inputs;
  mlir::DictionaryAttr output;
  int64_t end = 0;
};

// Reuse the ordinary operator-owned Tensor -> Stream -> closed-form Schedule
// lowerers. Only graph geometry is tiled here; no FU trace is expanded or
// reverse-compressed, and no external script participates in placement.
NativeStage lowerNative(mlir::Operation *source,
                        llvm::ArrayRef<mlir::Type> inputs,
                        mlir::RankedTensorType output,
                        mlir::DictionaryAttr overrides = {}, bool mirror = false) {
  auto *context = source->getContext();
  mlir::OpBuilder builder(context);
  NativeStage stage;
  stage.module = mlir::ModuleOp::create(source->getLoc());
  (*stage.module)->setAttrs(source->getParentOfType<mlir::ModuleOp>()->getAttrs());
  stage.function = mlir::func::FuncOp::create(source->getLoc(), "operator_domain",
      builder.getFunctionType(inputs, {output}));
  if (mirror) stage.function->setAttr("ftlpu.depthwise_conv_mirror", builder.getBoolAttr(true));
  stage.module->push_back(stage.function);
  auto *block = stage.function.addEntryBlock();
  builder.setInsertionPointToStart(block);
  mlir::OperationState state(source->getLoc(), source->getName());
  state.addOperands(block->getArguments());
  state.addTypes(output);
  mlir::NamedAttrList attrs(source->getAttrs());
  if (overrides) for (auto a : overrides) attrs.set(a.getName(), a.getValue());
  state.addAttributes(attrs);
  auto *op = builder.create(state);
  builder.create<mlir::func::ReturnOp>(source->getLoc(), op->getResults());
  mlir::PassManager pm(context);
  pm.enableVerifier(false);
  pm.addNestedPass<mlir::func::FuncOp>(create_lower_kernel_to_tensor_pass());
  pm.addNestedPass<mlir::func::FuncOp>(create_lower_tensor_to_stream_pass());
  pm.addNestedPass<mlir::func::FuncOp>(create_lower_stream_to_schedule_pass());
  if (mlir::failed(pm.run(*stage.module)))
    throw std::runtime_error("connected graph could not lower an operator domain");
  stage.function.walk([&](schedule::BindingOp binding) {
    if (binding.getAccess() == "input") {
      if (stage.inputs.size() <= binding.getIndex()) stage.inputs.resize(binding.getIndex()+1);
      stage.inputs[binding.getIndex()] = binding.getPlacement();
    } else if (binding.getAccess() == "output") stage.output = binding.getPlacement();
  });
  for (auto &item : stage.function.front()) {
    if (item.hasAttr("cycle")) stage.end = std::max(stage.end, lastIssue(&item)+32);
    if (item.hasAttr("end")) stage.end = std::max(stage.end, integer(&item,"end"));
  }
  return stage;
}

class GraphScheduler {
public:
  GraphScheduler(mlir::func::FuncOp f, const target::LPUTargetModel &t)
      : function(f), target(t), builder(f.getContext()), loc(f.getLoc()) {
    builder.setInsertionPoint(f.front().getTerminator());
  }
  mlir::func::FuncOp function;
  const target::LPUTargetModel &target;
  mlir::OpBuilder builder;
  mlir::Location loc;
  int64_t cursor = 64, nextInternal = 1000;

  void set(mlir::Operation *op, llvm::StringRef name, int64_t value) {
    op->setAttr(name, builder.getI64IntegerAttr(value));
  }
  mlir::DictionaryAttr placement(llvm::StringRef kind, llvm::ArrayRef<int64_t> ids,
                                llvm::StringRef side, int64_t bank,
                                int64_t base, int64_t count) {
    return builder.getDictionaryAttr({
      builder.getNamedAttr("kind",builder.getStringAttr(kind)),
      builder.getNamedAttr("slices",builder.getI64ArrayAttr(ids)),
      builder.getNamedAttr("hemisphere",builder.getStringAttr(side)),
      builder.getNamedAttr("bank",builder.getI64IntegerAttr(bank)),
      builder.getNamedAttr("base_row",builder.getI64IntegerAttr(base)),
      builder.getNamedAttr("instruction_count",builder.getI64IntegerAttr(count)),
      builder.getNamedAttr("address_stride",builder.getI64IntegerAttr(1))});
  }
  mlir::Value binding(mlir::RankedTensorType type, mlir::DictionaryAttr p,
                      llvm::StringRef access, int64_t index,
                      llvm::StringRef name, mlir::Value source = {}) {
    mlir::OperationState state(loc, schedule::BindingOp::getOperationName());
    if (source) state.addOperands(source);
    state.addTypes(type);
    state.addAttributes({
      builder.getNamedAttr("index",builder.getI64IntegerAttr(index)),
      builder.getNamedAttr("access",builder.getStringAttr(access)),
      builder.getNamedAttr("role",builder.getStringAttr(access == "input" && index != 0 ? "weight" : "activation")),
      builder.getNamedAttr("name",builder.getStringAttr(name)),
      builder.getNamedAttr("bytes",builder.getI64IntegerAttr(type.getNumElements()*type.getElementTypeBitWidth()/8)),
      builder.getNamedAttr("placement",p)});
    return builder.create(state)->getResult(0);
  }
  mlir::Value internal(mlir::RankedTensorType type, mlir::DictionaryAttr p,
                       llvm::StringRef name) {
    return binding(type,p,"internal",nextInternal++,name);
  }
  int64_t append(NativeStage &stage, int64_t start,
                  mlir::ValueRange operands, llvm::StringRef label,
                  const std::function<void(mlir::Operation*)> &adjust = {}) {
    mlir::IRMapping mapping;
    for (auto [from,to] : llvm::zip(stage.function.getArguments(),operands)) mapping.map(from,to);
    for (auto &original : stage.function.front()) {
      if (llvm::isa<mlir::func::ReturnOp>(original)) continue;
      if (auto input = llvm::dyn_cast<schedule::BindingOp>(original)) {
        if (input.getAccess() == "input") {
          mapping.map(input.getValue(),operands[input.getIndex()]);
          continue;
        }
      }
      auto *copy = builder.clone(original,mapping);
      copy->removeAttr("ftlpu.schedule_stage");
      if (auto binding = llvm::dyn_cast<schedule::BindingOp>(copy)) {
        binding->setAttr("access",builder.getStringAttr("internal"));
        set(copy,"index",nextInternal++);
      }
      if (adjust) adjust(copy);
      for (auto name : {"cycle","result_cycle","ready_cycle","start","end"})
        if (copy->hasAttr(name)) set(copy,name,integer(copy,name)+start);
      if (auto timeline = llvm::dyn_cast<schedule::TimelineOp>(copy))
        copy->setAttr("name",builder.getStringAttr((label+"."+timeline.getName()).str()));
    }
    return start+stage.end+96;
  }
  // A byte-plane domain. Counts and strides are derived directly from tensor
  // views. The coarse ICU domain remains indivisible on its physical queue.
  void mem(int64_t cycle, mlir::DictionaryAttr p, int64_t plane,
           llvm::StringRef opcode, int64_t row, int64_t stream,
           std::array<int64_t,3> counts, std::array<int64_t,3> strides,
           std::array<int64_t,3> intervals) {
    mlir::OperationState state(loc,schedule::MemTransferOp::getOperationName());
    const std::pair<llvm::StringRef,int64_t> values[] = {
      {"cycle",cycle},{"hemisphere",west(p) ? 1:0},{"slice",slices(p)[plane]},
      {"bank",field(p,"bank")},{"address",field(p,"base_row")+row},
      {"packed_stream",stream},{"repeat_count",counts[0]},
      {"repeat_interval",intervals[0]},{"address_stride",strides[0]},
      {"wave_count",counts[1]},{"wave_interval",intervals[1]},
      {"wave_address_stride",strides[1]},{"group_count",counts[2]},
      {"group_interval",intervals[2]},{"group_address_stride",strides[2]}};
    for (auto [name,value] : values) state.addAttribute(name,builder.getI64IntegerAttr(value));
    state.addAttribute("opcode",builder.getStringAttr(opcode));
    builder.create(state);
  }
  mlir::DictionaryAttr side(mlir::DictionaryAttr p, llvm::StringRef h) {
    mlir::NamedAttrList a(p);a.set("hemisphere",builder.getStringAttr(h));
    return a.getDictionary(builder.getContext());
  }
  void copy(mlir::DictionaryAttr source, mlir::DictionaryAttr dest,
            int64_t sourceRow, int64_t destRow,
            std::array<int64_t,3> counts,
            std::array<int64_t,3> sourceStrides,
            std::array<int64_t,3> destStrides) {
    int64_t period = target.memory().bytes_per_word/target.throughput().mem_read_bytes_per_cycle;
    const std::array<int64_t,3> intervals = {period,period*counts[0],period*counts[0]*counts[1]};
    int64_t start=cursor+32,maxLatency=0;
    auto from=slices(source),to=slices(dest);
    if (from.size()!=to.size()) throw std::runtime_error("byte-copy plane widths differ");
    for (size_t plane=0;plane<from.size();++plane) {
      int64_t sg=from[plane]/target.streams().mem_slices_per_register_group;
      int64_t dg=to[plane]/target.streams().mem_slices_per_register_group;
      int64_t readStream=20+plane,writeStream=20+plane,latency=0;
      if (west(source)!=west(dest)) {
        readStream+=target.streams().streams_per_direction;
        latency=sg+target.throughput().vxm_alus+dg+3;
      } else if (sg<dg) latency=dg-sg;
      else if (sg>dg) {
        readStream+=target.streams().streams_per_direction;
        writeStream+=target.streams().streams_per_direction;
        latency=sg-dg;
      } else throw std::runtime_error("graph plan selected an unreachable same-group SR copy");
      mem(start,source,plane,"read",sourceRow,readStream,counts,sourceStrides,intervals);
      mem(start+latency,dest,plane,"write",destRow,writeStream,counts,destStrides,intervals);
      maxLatency=std::max(maxLatency,latency);
    }
    cursor=start+counts[0]*counts[1]*counts[2]*period+maxLatency+96;
  }
};

class ConnectedKernelGraphPass final : public mlir::PassWrapper<
    ConnectedKernelGraphPass,mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConnectedKernelGraphPass)
  llvm::StringRef getArgument() const final {return "ftlpu-connected-kernel-graph-to-schedule";}
  llvm::StringRef getDescription() const final {return "Plan connected RMSNorm/projection/causal-convolution layouts and native ICU domains";}
  void runOnOperation() final {
    auto function=getOperation();
    if (function->hasAttr("ftlpu.logical_inputs")) {
      try {
        auto directory = function->getAttrOfType<mlir::StringAttr>("ftlpu.artifact_directory");
        if (!directory) throw std::runtime_error("connected decoder needs an artifact directory for packed inputs");
        auto parent = function->getParentOfType<mlir::ModuleOp>();
        auto lowered = lower_connected_decoder_graph(parent, directory.getValue().str());
        auto generated = *lowered->getOps<mlir::func::FuncOp>().begin();
        function.getBody().takeBody(generated.getBody());
        function.setType(generated.getFunctionType());
        function->removeAttr("ftlpu.logical_inputs");
        function->setAttr("ftlpu.connected_graph_schedule",mlir::UnitAttr::get(function.getContext()));
        for (auto attr : (*lowered)->getAttrs()) parent->setAttr(attr.getName(),attr.getValue());
        std::error_code ec;
        llvm::raw_fd_ostream out((std::filesystem::path(directory.getValue().str())/"decoder_layer.schedule.mlir").string(),ec);
        if (ec) throw std::runtime_error(ec.message());
        parent.print(out);
      } catch (const std::exception &e) {function.emitError(e.what());signalPassFailure();}
      return;
    }
    kernel::RmsNormOp norm;kernel::MatmulOp projection;kernel::DepthwiseConv1DOp conv;
    int64_t kernels=0;
    for (auto &op : function.front()) {
      if (auto n=llvm::dyn_cast<kernel::RmsNormOp>(op)) norm=n;
      else if (auto p=llvm::dyn_cast<kernel::MatmulOp>(op)) projection=p;
      else if (auto c=llvm::dyn_cast<kernel::DepthwiseConv1DOp>(op)) conv=c;
      else if (!llvm::isa<mlir::func::ReturnOp>(op)) return;
      if (!llvm::isa<mlir::func::ReturnOp>(op)) ++kernels;
    }
    if (kernels!=3 || !norm || !projection || !conv
        || projection.getLhs()!=norm.getResult() || conv.getInput()!=projection.getResult()) return;
    auto target=target::LPUTargetModel::from_operation(function);
    if (mlir::failed(target)) {signalPassFailure();return;}
    try {lower(function,*target,norm,projection,conv);}
    catch (const std::exception &e) {function.emitError(e.what());signalPassFailure();}
  }

  void lower(mlir::func::FuncOp function,const target::LPUTargetModel &target,
             kernel::RmsNormOp norm,kernel::MatmulOp projection,kernel::DepthwiseConv1DOp conv) {
    auto normalType=llvm::cast<mlir::RankedTensorType>(norm.getResult().getType());
    auto projectedType=llvm::cast<mlir::RankedTensorType>(projection.getResult().getType());
    auto resultType=llvm::cast<mlir::RankedTensorType>(conv.getResult().getType());
    int64_t rows=projection.getM(),width=projection.getN(),reduction=projection.getK();
    if (rows!=target.throughput().mxm_rows || width%64 || reduction%32
        || !normalType.getElementType().isBF16() || !projection.getRhs().getType().getElementType().isInteger(8)
        || conv.getKernelSize()!=4 || !conv.getCausal() || conv.getActivation()!="silu")
      throw std::runtime_error("connected graph requires one BF16 MXM token tile, W8 projection and causal kernel-4 SiLU");
    llvm::SmallVector<mlir::Value> parameters={norm.getInput(),norm.getWeight(),projection.getRhs(),conv.getWeight()};
    for (auto parameter:parameters) if (!llvm::isa<mlir::BlockArgument>(parameter))
      throw std::runtime_error("connected graph parameters must be function arguments");
    auto oldReturn=llvm::cast<mlir::func::ReturnOp>(function.front().getTerminator());
    for (auto value:oldReturn.getOperands()) if (value!=norm.getResult() && value!=projection.getResult() && value!=conv.getResult())
      throw std::runtime_error("unsupported connected graph return");
    GraphScheduler g(function,target);
    auto elem=normalType.getElementType();
    auto i8=g.builder.getIntegerType(8);
    const int64_t depth=target.memory().sram_depth_rows;
    const int64_t groupsPerPage=std::max<int64_t>(1,(depth/2)/((reduction/32)*4));
    const int64_t pageColumns=std::min(width,64*groupsPerPage);
    const int64_t pageCount=(width+pageColumns-1)/pageColumns;
    const int64_t convColumns=std::min<int64_t>(1024,width);
    auto rms=lowerNative(norm,{norm.getInput().getType(),norm.getWeight().getType()},normalType);
    auto projectedTileType=mlir::RankedTensorType::get({rows,pageColumns},elem);
    auto projectionStage=lowerNative(projection,{normalType,mlir::RankedTensorType::get({reduction,pageColumns},i8)},projectedTileType,
      g.builder.getDictionaryAttr({g.builder.getNamedAttr("n",g.builder.getI64IntegerAttr(pageColumns))}));
    auto convType=mlir::RankedTensorType::get({rows,convColumns},elem);
    auto convWeightType=mlir::RankedTensorType::get({convColumns,4},elem);
    auto primary=lowerNative(conv,{convType,convWeightType},convType);
    auto mirrored=lowerNative(conv,{convType,convWeightType},convType,{},true);

    // Reserve immutable inputs and every native bank-1 write before choosing
    // persistent output pairs. Allocation observes complete domain bounds.
    auto available=[&](int64_t first,int64_t count) {
      for (auto *stage:{&rms,&projectionStage,&primary,&mirrored}) {
        for (auto &op:stage->function.front()) if (auto transfer=llvm::dyn_cast<schedule::MemTransferOp>(op)) {
          if (field(transfer->getAttrDictionary(),"bank")!=1 || transfer.getOpcode()=="read") continue;
          int64_t slice=transfer.getSlice();
          if (slice<first || slice>first+1) continue;
          int64_t low=transfer.getAddress(),high=low;
          for (auto axis:{std::pair{"repeat_count","address_stride"},std::pair{"wave_count","wave_address_stride"},std::pair{"group_count","group_address_stride"}}) {
            auto delta=(integer(&op,axis.first,1)-1)*integer(&op,axis.second);
            low+=std::min<int64_t>(0,delta);high+=std::max<int64_t>(0,delta);
          }
          if (low<count && high>=0) return false;
        }
      }
      for (auto p:rms.inputs) if (field(p,"bank")==1 && field(p,"base_row")<count)
        for (auto s:slices(p)) if (s>=first && s<=first+1) return false;
      return true;
    };
    int64_t projectedPair=-1,resultPair=-1;
    const int64_t consumerGroup=slices(primary.inputs[0])[0]/target.streams().mem_slices_per_register_group;
    for (int64_t s=0;s+1<target.memory().slices_per_hemisphere;s+=2)
      if (s/target.streams().mem_slices_per_register_group!=consumerGroup && available(s,rows*width/64)) {projectedPair=s;break;}
    for (int64_t s=0;s+1<target.memory().slices_per_hemisphere;s+=2)
      if (s!=projectedPair && available(s,rows*width/32)) {resultPair=s;break;}
    if (projectedPair<0 || resultPair<0 || rows*width/32>depth)
      throw std::runtime_error("connected graph persistent outputs exceed SRAM");
    auto projectedP=g.placement("fp16_hemisphere_pair_planar",{projectedPair,projectedPair+1},"both",1,0,rows*width/64);
    auto resultP=g.placement("fp16_pair_planar",{resultPair,resultPair+1},"west",1,0,rows*width/32);
    int64_t convWeightRows=4*width/32,convWeightBase=depth-convWeightRows;
    if (convWeightBase<rows*reduction/256+1 || projectedPair<8 || resultPair<8)
      throw std::runtime_error("connected graph constant and output regions overlap");
    auto convWeightP=g.placement("fp16_depthwise_kernel4",slices(primary.inputs[1]),"both",1,convWeightBase,convWeightRows);
    auto inputIndex=[](mlir::Value value) {return llvm::cast<mlir::BlockArgument>(value).getArgNumber();};
    auto x=g.binding(normalType,rms.inputs[0],"input",inputIndex(norm.getInput()),"activation",norm.getInput());
    auto gamma=g.binding(llvm::cast<mlir::RankedTensorType>(norm.getWeight().getType()),rms.inputs[1],"input",inputIndex(norm.getWeight()),"normalization_weight",norm.getWeight());
    int64_t weightIndex=inputIndex(projection.getRhs());
    mlir::NamedAttrList weightAttrs(projectionStage.inputs[1]);
    int64_t pageRows=pageColumns/64*(reduction/32)*4;
    weightAttrs.set("bank",g.builder.getI64IntegerAttr(0));
    weightAttrs.set("paged_weight",g.builder.getBoolAttr(true));
    weightAttrs.set("runtime_prefetch",g.builder.getBoolAttr(true));
    weightAttrs.set("page_count",g.builder.getI64IntegerAttr(pageCount));
    weightAttrs.set("page_rows",g.builder.getI64IntegerAttr(pageRows));
    weightAttrs.set("page_granularity",g.builder.getI64IntegerAttr(pageColumns/64));
    weightAttrs.set("page_bank_count",g.builder.getI64IntegerAttr(2));
    weightAttrs.set("page_role_group_base",g.builder.getI64IntegerAttr(0));
    weightAttrs.set("page_role_group_count",g.builder.getI64IntegerAttr(1));
    weightAttrs.set("page_items_per_slice_group",g.builder.getI64IntegerAttr(pageColumns/64));
    weightAttrs.set("page_storage_slices",projectionStage.inputs[1].getAs<mlir::ArrayAttr>("slices"));
    llvm::SmallVector<int64_t> zeros(pageCount,0),ones(pageCount,1),pageRowCounts;
    for (int64_t page=0;page<pageCount;++page) pageRowCounts.push_back(std::min(pageColumns,width-page*pageColumns)/64*(reduction/32)*4);
    for (auto name:{"page_banks","page_slice_group_bases","page_base_rows"}) weightAttrs.set(name,g.builder.getI64ArrayAttr(zeros));
    weightAttrs.set("page_slice_group_counts",g.builder.getI64ArrayAttr(ones));
    weightAttrs.set("page_row_counts",g.builder.getI64ArrayAttr(pageRowCounts));
    auto weightP=weightAttrs.getDictionary(function.getContext());
    g.binding(llvm::cast<mlir::RankedTensorType>(projection.getRhs().getType()),weightP,"input",weightIndex,"projection_weight",projection.getRhs());
    g.binding(llvm::cast<mlir::RankedTensorType>(conv.getWeight().getType()),convWeightP,"input",inputIndex(conv.getWeight()),"convolution_weight",conv.getWeight());
    auto normalizedP=projectionStage.inputs[0];
    int64_t normalizedIndex=g.nextInternal;
    auto normalized=g.internal(normalType,normalizedP,"normalized_projection_input");
    g.cursor=g.append(rms,g.cursor,{x,gamma},"normalization");

    // Distributed token lanes -> projection pair-planar input. Each domain is
    // a rectangular tensor view; choose the opposite replica for a U-turn.
    auto rmsSlices=slices(rms.output);
    for (auto h:{llvm::StringRef("east"),llvm::StringRef("west")}) {
      for (int64_t lane=0;lane<8;++lane) {
        auto src=g.placement("fp16_pair_planar",{rmsSlices[2*lane],rmsSlices[2*lane+1]},h,field(rms.output,"bank"),field(rms.output,"base_row"),rows*reduction/256);
        if (rmsSlices[2*lane]/target.streams().mem_slices_per_register_group==slices(normalizedP)[0]/target.streams().mem_slices_per_register_group)
          src=g.side(src,h=="east" ? "west":"east");
        g.copy(src,g.side(normalizedP,h),0,lane,{4,reduction/32,1},{1,4,0},{8,rows,0});
      }
    }
    // A page changes weight residency and time, never the result allocation.
    // The producer writes directly into the retained full-tensor layout.
    for (int64_t first=0,page=0;first<width;first+=pageColumns,++page) {
      int64_t columns=std::min(pageColumns,width-first);
      auto localType=mlir::RankedTensorType::get({rows,columns},elem);
      auto localWeightType=mlir::RankedTensorType::get({reduction,columns},i8);
      NativeStage tail;
      auto *stage=&projectionStage;
      if (columns!=pageColumns) {
        tail=lowerNative(projection,{normalType,localWeightType},localType,
          g.builder.getDictionaryAttr({g.builder.getNamedAttr("n",g.builder.getI64IntegerAttr(columns))}));stage=&tail;
      }
      auto weightView=g.internal(localWeightType,weightP,"weight_page_view");
      auto oldSlices=slices(stage->output),newSlices=slices(projectedP);
      g.cursor=g.append(*stage,g.cursor,{normalized,weightView},"projection.page"+std::to_string(page),[&](mlir::Operation *op) {
        if (op->hasAttr("scale_binding")) g.set(op,"scale_binding",weightIndex);
        if (auto mem=llvm::dyn_cast<schedule::MemTransferOp>(op)) {
          if (auto binding=mem.getAddressBinding()) {
            if (*binding==1) {g.set(op,"address_binding",weightIndex);g.set(op,"weight_page",page);op->setAttr("runtime_prefetch",g.builder.getBoolAttr(true));}
            else {g.set(op,"address_binding",normalizedIndex);op->setAttr("address_binding_access",g.builder.getStringAttr("internal"));}
          }
          if (mem.getOpcode()=="write" && integer(op,"bank")==field(stage->output,"bank")) {
            auto it=llvm::find(oldSlices,mem.getSlice());
            if (it!=oldSlices.end()) {
              int64_t old=mem.getSlice(),dest=newSlices[it-oldSlices.begin()];
              g.set(op,"slice",dest);g.set(op,"bank",1);
              g.set(op,"address",mem.getAddress()+first/64*rows);
              g.set(op,"cycle",mem.getCycle()+dest/target.streams().mem_slices_per_register_group-old/target.streams().mem_slices_per_register_group);
            }
          }
        }
      });
    }
    for (int64_t first=0;first<width;first+=2*convColumns) {
      llvm::SmallVector<mlir::Value> prepared;
      int64_t members=std::min<int64_t>(2,(width-first+convColumns-1)/convColumns);
      std::vector<NativeStage> tails;
      std::array<NativeStage*,2> stages={&primary,&mirrored};
      tails.reserve(2);
      for (int64_t member=0;member<members;++member) {
        int64_t offset=first+member*convColumns,columns=std::min(convColumns,width-offset);
        if (columns!=convColumns) {
          auto t=mlir::RankedTensorType::get({rows,columns},elem),w=mlir::RankedTensorType::get({columns,4},elem);
          tails.push_back(lowerNative(conv,{t,w},t,{},member==1));stages[member]=&tails.back();
        }
        auto &stage=*stages[member];
        auto t=mlir::RankedTensorType::get({rows,columns},elem);
        prepared.push_back(g.internal(t,stage.inputs[0],"convolution_input_view"));
        for (int64_t half=0;half<2;++half) {
          auto src=g.side(projectedP,half==0 ? "east":"west");
          g.copy(src,stage.inputs[0],offset/64*rows,half*rows,
                 {rows,columns/64,1},{1,rows,0},{1,2*rows,0});
        }
      }
      int64_t start=g.cursor+64,end=start;
      for (int64_t member=0;member<members;++member) {
        auto &stage=*stages[member];
        int64_t offset=first+member*convColumns,columns=std::min(convColumns,width-offset);
        auto w=mlir::RankedTensorType::get({columns,4},elem);
        auto weightView=g.internal(w,convWeightP,"convolution_weight_view");
        auto oldOutput=slices(stage.output),outSlices=slices(resultP),weightSlices=slices(stage.inputs[1]);
        end=std::max(end,g.append(stage,start,{prepared[member],weightView},"convolution.block"+std::to_string(offset),[&](mlir::Operation *op) {
          if (auto mem=llvm::dyn_cast<schedule::MemTransferOp>(op)) {
            if (mem.getOpcode()=="read" && mem.getHemisphere()==(west(stage.inputs[1]) ? 1:0)
                && integer(op,"bank")==field(stage.inputs[1],"bank")) {
              auto it=llvm::find(weightSlices,mem.getSlice());
              if (it!=weightSlices.end()) {
                int64_t tap=(it-weightSlices.begin())/2;
                g.set(op,"bank",1);
                g.set(op,"address",mem.getAddress()+convWeightBase+offset/32+tap*(width-columns)/32);
              }
            }
            if (mem.getOpcode()=="write" && mem.getHemisphere()==(west(stage.output) ? 1:0)
                && integer(op,"bank")==field(stage.output,"bank")) {
              auto it=llvm::find(oldOutput,mem.getSlice());
              if (it!=oldOutput.end()) {
                int64_t old=mem.getSlice(),dest=outSlices[it-oldOutput.begin()];
                g.set(op,"slice",dest);g.set(op,"bank",1);g.set(op,"address",mem.getAddress()+offset/32*rows);
                g.set(op,"cycle",mem.getCycle()+(dest/target.streams().mem_slices_per_register_group-old/target.streams().mem_slices_per_register_group));
              }
            }
          }
        }));
      }
      g.cursor=end;
    }
    // One side is already in the final full-tensor allocation. Join only the
    // mirrored blocks, using passive cross-fabric SR rather than an ALU.
    for (int64_t first=convColumns;first<width;first+=2*convColumns) {
      auto count=std::min(convColumns,width-first)/32*rows;
      g.copy(g.side(resultP,"east"),resultP,first/32*rows,first/32*rows,
             {count,1,1},{1,0,0},{1,0,0});
    }
    llvm::SmallVector<mlir::Value> results;
    int64_t index=0;
    for (auto original:oldReturn.getOperands()) {
      auto p=original==norm.getResult() ? normalizedP : original==projection.getResult() ? projectedP : resultP;
      results.push_back(g.binding(llvm::cast<mlir::RankedTensorType>(original.getType()),p,"output",index++,"graph_result"));
    }
    g.builder.create<mlir::func::ReturnOp>(g.loc,results);
    oldReturn.erase();conv.erase();projection.erase();norm.erase();
    function->setAttr("ftlpu.schedule.closed_form",g.builder.getUnitAttr());
    function->setAttr("ftlpu.connected_graph_schedule",g.builder.getUnitAttr());
    function->setAttr("ftlpu.graph_plan",g.builder.getDictionaryAttr({
      g.builder.getNamedAttr("projection_pages",g.builder.getI64IntegerAttr(pageCount)),
      g.builder.getNamedAttr("projection_page_columns",g.builder.getI64IntegerAttr(pageColumns)),
      g.builder.getNamedAttr("convolution_channel_tile",g.builder.getI64IntegerAttr(convColumns)),
      g.builder.getNamedAttr("convolution_domains",g.builder.getI64IntegerAttr((width+convColumns-1)/convColumns)),
      g.builder.getNamedAttr("projection_output",projectedP),
      g.builder.getNamedAttr("output",resultP),
      g.builder.getNamedAttr("projection_archive_copies",g.builder.getI64IntegerAttr(0))}));
  }
};
} // namespace
std::unique_ptr<mlir::Pass> create_lower_connected_kernel_graph_pass() {
  return std::make_unique<ConnectedKernelGraphPass>();
}
} // namespace ftlpu::compiler
