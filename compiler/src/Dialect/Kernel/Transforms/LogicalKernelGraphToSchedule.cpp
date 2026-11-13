#include "ftlpu/compiler/Analysis/planar_access_attrs.hpp"
#include "ftlpu/compiler/Transforms/logical_kernel_graph.hpp"
#include "BlockFp8ActivationLowering.hpp"
// Logical tensor graphs are planned here, before any physical schedule exists.
// Python importers supply equations, tensor views and checkpoint bytes only.
#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Kernel/Analysis/decoder_graph.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Target/graph_weight_paging.hpp"
#include "ftlpu/compiler/Target/weight_page_plan.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
namespace fs = std::filesystem;
namespace fc = ftlpu::compiler;
using I = int64_t;
using V = std::vector<I>;
using Key = std::pair<I, I>;
using J = llvm::json::Object;
I product(const V &v) {
  return std::accumulate(v.begin(), v.end(), I(1), std::multiplies<I>());
}
V range(I n, I first = 0) {
  V v(n);
  std::iota(v.begin(), v.end(), first);
  return v;
}
std::string read(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot read " + p.string());
  return {std::istreambuf_iterator<char>(f), {}};
}
V ints(const llvm::json::Array &a) {
  V v;
  for (auto &x : a)
    v.push_back(*x.getAsInteger());
  return v;
}
I integer(mlir::Operation *op, llvm::StringRef name, I fallback = 0) {
  auto a = op->getAttrOfType<mlir::IntegerAttr>(name);
  return a ? a.getInt() : fallback;
}
I field(mlir::DictionaryAttr p, llvm::StringRef name, I fallback = 0) {
  auto a = p.getAs<mlir::IntegerAttr>(name);
  return a ? a.getInt() : fallback;
}
std::string text(mlir::DictionaryAttr p, llvm::StringRef name) {
  return p.getAs<mlir::StringAttr>(name).getValue().str();
}
V slices(mlir::DictionaryAttr p) {
  V v;
  for (auto a : p.getAs<mlir::ArrayAttr>("slices"))
    v.push_back(llvm::cast<mlir::IntegerAttr>(a).getInt());
  return v;
}
bool scalar(mlir::DictionaryAttr p) {
  return text(p, "kind").find("scalar_broadcast") != std::string::npos ||
         text(p, "kind") == "fp16_vxm_gamma_broadcast";
}
bool contains(const V &v, I x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}
bool overlap(mlir::DictionaryAttr a, mlir::DictionaryAttr b) {
  // A segmented matrix read aliases several producer queues. Reserve every
  // queue in ordinary allocation/conflict checks, not just its first home.
  if(auto segments=a.getAs<mlir::ArrayAttr>("mxm_rhs_segments"))
    return llvm::any_of(segments,[&](auto p){return overlap(llvm::cast<mlir::DictionaryAttr>(p),b);});
  if(auto segments=b.getAs<mlir::ArrayAttr>("mxm_rhs_segments"))
    return llvm::any_of(segments,[&](auto p){return overlap(a,llvm::cast<mlir::DictionaryAttr>(p));});
  if (text(a, "hemisphere") != text(b, "hemisphere") &&
      text(a, "hemisphere") != "both" && text(b, "hemisphere") != "both")
    return false;
  if (!field(a,"bank_tile_rows") && !field(b,"bank_tile_rows") && field(a, "bank") != field(b, "bank"))
    return false;
  for (I s : slices(a))
    if (contains(slices(b), s))
      return true;
  return false;
}
struct Ref {
  I node = -1, output = 0, index = 0;
};
class GraphSramAllocationFailure : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};
// Match the same precision and SSA boundaries as the streamed MXM epilogue
// emitter. Other head_matmul/add graphs retain ordinary memory planning.
bool isNativeMatmulAdd(fc::kernel::VectorAluOp product,fc::kernel::VectorAluOp sum) {
  if(!product || !sum || product.getKind()!="head_matmul" || sum.getKind()!="add")return false;
  const auto precision=product->getAttrOfType<mlir::StringAttr>("mxm_precision");
  return precision && precision.getValue()=="bf16" && product.getResult().hasOneUse() &&
      !sum.getUseImmediate() && !sum.getReduce() && sum.getLhs()==product.getResult() &&
      sum.getRhs().getType()==product.getResult().getType() && sum.getResult().getType()==product.getResult().getType() &&
      sum.getResult().getType().getElementType().isF32();
}
bool isNativeMatmulCast(fc::kernel::VectorAluOp sum,fc::kernel::VectorAluOp cast) {
  return sum && cast && cast.getKind()=="multiply" && cast.getUseImmediate() && !cast.getReduce() &&
      cast.getImmediate().convertToFloat()==1.f && cast.getLhs()==sum.getResult() &&
      cast.getResult().getType().getElementType().isBF16() &&
      cast.getResult().getType().getShape()==sum.getResult().getType().getShape() &&
      llvm::all_of(sum.getResult().getUsers(),[&](mlir::Operation *user){return user==cast.getOperation();});
}
struct Tensor {
  V shape;
  std::string dtype;
  std::vector<Ref> refs;
  std::vector<uint8_t> data;
  I constant = -1;
  I sourceOffset = 0;
  bool sourceContiguous = true;
  bool external() const { return constant >= 0; }
  I size() const { return product(shape); }
  bool byteWeight() const { return dtype == "i8" || dtype == "f8E4M3FN" || dtype == "f8E8M0FNU"; }
  I bytes() const { return dtype == "f32" ? 4 : byteWeight() ? 1 : 2; }
};
Tensor view(const Tensor &a, const V &indices, V shape) {
  Tensor b;
  b.shape = std::move(shape);
  b.dtype = a.dtype;
  b.constant = a.constant;
  b.sourceOffset = a.sourceOffset + (indices.empty() ? 0 : indices.front());
  b.sourceContiguous = a.sourceContiguous;
  for (I i=0; i<I(indices.size()); ++i)
    b.sourceContiguous &= indices[i] == indices.front() + i;
  if (a.external()) {
    b.data.resize(indices.size() * a.bytes());
    for (size_t i = 0; i < indices.size(); ++i)
      std::memcpy(b.data.data() + i * a.bytes(),
                  a.data.data() + indices[i] * a.bytes(), a.bytes());
  } else
    for (I i : indices)
      b.refs.push_back(a.refs.at(i));
  return b;
}
Tensor sliceRows(const Tensor &a, I first, I count) {
  if (a.shape[0] == 1)
    return a;
  I width = a.size() / a.shape[0];
  V shape = a.shape;
  shape[0] = count;
  return view(a, range(count * width, first * width), shape);
}
Tensor sliceColumns(const Tensor &a, I first, I count, bool pad = false) {
  Tensor b;
  b.shape = {a.shape[0], count};
  b.dtype = a.dtype;
  b.constant = a.constant;
  b.sourceOffset = a.sourceOffset + first;
  b.sourceContiguous = a.sourceContiguous && first == 0 && count == a.shape[1];
  if (a.external())
    b.data.resize(b.size() * b.bytes());
  for (I r = 0; r < a.shape[0]; ++r)
    for (I c = 0; c < count; ++c) {
      if (first + c >= a.shape[1]) {
        if (!pad)
          throw std::runtime_error("column view exceeds tensor");
        continue;
      }
      I src = r * a.shape[1] + first + c, dst = r * count + c;
      if (a.external())
        std::memcpy(b.data.data() + dst * b.bytes(),
                    a.data.data() + src * b.bytes(), b.bytes());
      else
        b.refs.push_back(a.refs[src]);
    }
  return b;
}
Tensor concat(const std::vector<Tensor> &parts, I axis = 0) {
  if (parts.empty())
    throw std::runtime_error("empty tensor concatenation");
  Tensor t;
  t.dtype = parts[0].dtype;
  t.shape = parts[0].shape;
  t.shape[axis] = 0;
  for (auto &p : parts)
    t.shape[axis] += p.shape[axis];
  if (axis == 0)
    for (auto &p : parts)
      t.refs.insert(t.refs.end(), p.refs.begin(), p.refs.end());
  else
    for (I row = 0; row < t.shape[0]; ++row)
      for (auto &p : parts)
        t.refs.insert(t.refs.end(), p.refs.begin() + row * p.shape[1],
                      p.refs.begin() + (row + 1) * p.shape[1]);
  return t;
}
mlir::RankedTensorType type(mlir::MLIRContext *c, const Tensor &t) {
  auto e = t.dtype == "f32"    ? mlir::Type(mlir::Float32Type::get(c))
           : t.dtype == "bf16" ? mlir::Type(mlir::BFloat16Type::get(c))
           : t.dtype == "f8E4M3FN" ? mlir::Type(mlir::Float8E4M3FNType::get(c))
           : t.dtype == "f8E8M0FNU" ? mlir::Type(mlir::Float8E8M0FNUType::get(c))
                               : mlir::Type(mlir::IntegerType::get(c, 8));
  return mlir::RankedTensorType::get(t.shape, e);
}
// A physical-layout slot belongs to Tensor IR (or the VXM graph's native
// storage plan), never to a generated MEM/FU instruction. The containing graph
// assigns these slots before Stream/Schedule lowering runs.
struct PlacementSlot {
  mlir::Operation *op = nullptr;
  std::string attr, member;
  bool mirrored = false;
  mlir::DictionaryAttr get() const {
    auto a = op->getAttr(attr);
    mlir::DictionaryAttr p;
    if (!member.empty())
      p = llvm::cast<mlir::DictionaryAttr>(a).getAs<mlir::DictionaryAttr>(
          member);
    else if (auto array = llvm::dyn_cast<mlir::ArrayAttr>(a))
      p = llvm::cast<mlir::DictionaryAttr>(array[0])
              .getAs<mlir::DictionaryAttr>("placement");
    else
      p = llvm::cast<mlir::DictionaryAttr>(a);
    return reflect(p);
  }
  mlir::DictionaryAttr reflect(mlir::DictionaryAttr p) const {
    if (!mirrored || text(p, "hemisphere") == "both")
      return p;
    mlir::NamedAttrList a(p);
    a.set("hemisphere", mlir::StringAttr::get(
                            op->getContext(),
                            text(p, "hemisphere") == "east" ? "west" : "east"));
    return a.getDictionary(op->getContext());
  }
  void set(mlir::DictionaryAttr physical) const {
    auto p = reflect(physical);
    auto a = op->getAttr(attr);
    if (!member.empty()) {
      mlir::NamedAttrList fields(llvm::cast<mlir::DictionaryAttr>(a));
      fields.set(member, p);
      op->setAttr(attr, fields.getDictionary(op->getContext()));
    } else if (auto array = llvm::dyn_cast<mlir::ArrayAttr>(a)) {
      auto entries =
          llvm::SmallVector<mlir::Attribute>(array.begin(), array.end());
      mlir::NamedAttrList fields(llvm::cast<mlir::DictionaryAttr>(entries[0]));
      fields.set("placement", p);
      if (auto address = fields.get("address")) {
        mlir::NamedAttrList a(llvm::cast<mlir::DictionaryAttr>(address));
        mlir::Builder b(op->getContext());
        a.set("hemisphere", p.get("hemisphere"));
        a.set("slice", b.getI64IntegerAttr(slices(p).front()));
        a.set("bank", p.get("bank"));
        a.set("word", p.get("base_row"));
        fields.set("address", a.getDictionary(op->getContext()));
      }
      entries[0] = fields.getDictionary(op->getContext());
      op->setAttr(attr, mlir::ArrayAttr::get(op->getContext(), entries));
    } else {
      auto vector = llvm::dyn_cast<fc::kernel::VectorAluOp>(op);
      if (vector && attr == "ftlpu.vector_result_placement")
        for (auto &use : vector.getResult().getUses())
          if (auto user =
                  llvm::dyn_cast<fc::kernel::VectorAluOp>(use.getOwner()))
            user->setAttr(use.getOperandNumber() == 0
                              ? "ftlpu.vector_lhs_placement"
                              : "ftlpu.vector_rhs_placement",
                          p);
      op->setAttr(attr, p);
    }
  }
};
struct Node {
  llvm::json::Array accessPlans;
  std::optional<Tensor> materializedView;
  std::string viewReason;
  bool addressOnlyView = false;
  // Adjacent-pair rotation on ordinary planar words. The native emitter
  // retains both products in the existing VXM accumulator.
  I planarRotaryDim = 0;

  llvm::json::Array nativeProjectionReport;
  std::string name;
  std::vector<Tensor> inputs, outputs;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  std::vector<mlir::DictionaryAttr> in, out;
  std::vector<std::vector<PlacementSlot>> inputSlots;
  std::vector<PlacementSlot> outputSlots;
  std::vector<bool> stagedInputs;
  std::vector<mlir::DictionaryAttr> workspaces;
  I parallel = -1;
  // Independent MXM projection / FP32 solve. This is not the hemisphere
  // partition group above: each operation retains its complete native ABI.
  I projectionSolvePeer = -1;
  I projectionPageFamily = -1;
  I projectionConvGroup = -1;
  bool projectionConvReady = false;
  bool projectionVectorMember = false;
  I pointwiseGroup = -1, pointwiseRow = 0;
  V pointwiseShape;
  bool recurrent = false;
  bool ffn = false;
  bool mirroredJoin = false;
  I headIngressArgument = -1, headIngressOffset = 0, headIngressWest = 0;
  bool matmulLhsJoin = false;
  bool streamedTransposeJoin = false;
};
class Planner {
public:
  mlir::MLIRContext &context;
  fc::target::LPUTargetModel target;
  fc::FfnScheduleStrategy ffnStrategy = fc::FfnScheduleStrategy::Tail;
  mlir::DictionaryAttr policies;
  bool persistentState = false;
  bool hasAttention = false;
  bool retainedAcrossFfn = false;
  bool distributedConvOutput = false;
  I decodeHeadWidth = 0, decodeConvWidth = 0;
  std::vector<Node> nodes;
  Planner(mlir::MLIRContext &c, fc::target::LPUTargetModel t)
      : context(c), target(t) {
    policies = fc::target::GraphWeightPaging::resolve(target, {})
        .attributes(&context, {});
  }
  Tensor output(I node, I index, V shape, std::string dtype) {
    Tensor t;
    t.shape = std::move(shape);
    t.dtype = std::move(dtype);
    for (I i = 0; i < t.size(); ++i)
      t.refs.push_back({node, index, i});
    return t;
  }
  std::vector<Tensor> add(std::string name, const std::string &source,
                          std::vector<Tensor> inputs,
                          std::vector<Tensor> outputs, I rows = -1,
                          I columns = -1, bool mirror = false,
                          I parallel = -1) {
    Node n;
    n.name = std::move(name);
    if (std::getenv("FTLPU_GRAPH_PROGRESS")) llvm::errs()<<"graph: planning "<<n.name<<"\n";
    n.inputs = std::move(inputs);
    n.parallel = parallel;
    n.module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    if (!n.module)
      throw std::runtime_error("invalid logical kernel");
    (*n.module)->setAttr("ftlpu.target", target.to_attribute(&context));
    if (policies) for (auto attr : policies) (*n.module)->setAttr(attr.getName(),attr.getValue());
    const bool attention = source.find("ftlpu.kernel.batch_matmul") != std::string::npos;
    if (attention)
      (*n.module)->setAttr("ftlpu.runtime_weight_pages", mlir::BoolAttr::get(&context, true));
    n.function = *n.module->getOps<mlir::func::FuncOp>().begin();
    mlir::IRRewriter fp8Rewriter(&context);
    for (auto op : llvm::make_early_inc_range(n.function.getOps<fc::kernel::BlockFp8QdqOp>()))
      fc::kernel::expandBlockFp8Arithmetic(op.getOperation(), op.getInput(), true, fp8Rewriter);
    // A scalar-broadcast/reduction region can explicitly retain the native
    // FP32 stream path rather than inheriting the packed BF16 chain policy.
    if(auto policy=n.function->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_bf16_spill");
       policy && !policy.getValue()) n.function->removeAttr("ftlpu.vector_bf16_spill");
    // Packed chains cannot represent a row reduction or its scalar broadcast.
    // Select the native FP32 scalar layout for the entire reduction region.
    bool hasRowReduction = false;
    n.function.walk([&](fc::kernel::VectorAluOp op) {
      const auto axis=op->getAttrOfType<mlir::IntegerAttr>("reduction_axis");
      hasRowReduction |= op.getReduce() && (!axis || axis.getInt()!=0);
    });
    if (hasRowReduction) n.function->removeAttr("ftlpu.vector_bf16_spill");
    if(source.find("ftlpu.kernel.rms_norm")!=std::string::npos && !n.inputs.empty()
        && !n.inputs[0].refs.empty()
        && llvm::all_of(n.inputs[0].refs,[&](const Ref& ref) {
            const auto &producer=nodes[ref.node];
            return producer.outputs[ref.output].dtype=="f32" ||
                (producer.outputs[ref.output].dtype=="bf16" && n.inputs[0].shape[1]>32 &&
                 text(producer.out[ref.output],"kind")=="fp16_pair_planar" &&
                 text(producer.out[ref.output],"hemisphere")=="both" &&
                 slices(producer.out[ref.output]).size()==2);
        }))
      n.function->setAttr("ftlpu.norm_retained_matrix_input",mlir::UnitAttr::get(&context));
    // RoPE tables are initialized eagerly in bank1, before the first RMSNorm
    // executes. Its gamma must be on the other bank even though attention's
    // ordinary scratch does not become live until later.
    if (persistentState || (hasAttention && nodes.empty()))
      n.function->setAttr("ftlpu.feedback_gamma_bank",mlir::IntegerAttr::get(mlir::IntegerType::get(&context,64),0));
    auto patterns = fc::kernel::match_decoder_graph(n.function);
    n.ffn = !patterns.ffns.empty();
    n.recurrent = !patterns.delta_nets.empty();
    // Native chunkwise keeps its state as an ordinary graph value; the
    // legacy scan's persistent FP32 scratch reservation does not apply.
    if (source.find("native_chunkwise_bf16") != std::string::npos)
      n.recurrent = false;
    if (n.ffn) {
      (*n.module)->setAttr("ftlpu.runtime_weight_pages", mlir::BoolAttr::get(&context, true));
      n.function->setAttr("ftlpu.prior_weight_pages", mlir::BoolAttr::get(&context,!nodes.empty()));
    }
    auto original =
        llvm::cast<mlir::RankedTensorType>(n.function.getArgument(0).getType());
    I originalRows = original.getShape()[0];
    for (size_t i = 0; i < n.inputs.size(); ++i)
      n.function.getArgument(i).setType(type(&context, n.inputs[i]));
    n.function.walk([&](mlir::Operation *op) {
      if (auto m = llvm::dyn_cast<fc::kernel::MatmulOp>(op); m && columns >= 0) {
        m->setAttr("n", mlir::IntegerAttr::get(
                            mlir::IntegerType::get(&context, 64), columns));
        m.getResult().setType(type(&context, outputs[0]));
      } else if (llvm::isa<fc::kernel::RmsNormOp,fc::kernel::L2NormOp>(op) && rows >= 0)
        op->getResult(0).setType(type(&context,outputs[0]));
      else if (auto conv = llvm::dyn_cast<fc::kernel::DepthwiseConv1DOp>(op))
        conv.getResult().setType(type(&context, outputs[0]));
      else if (auto delta = llvm::dyn_cast<fc::kernel::GatedDeltaRuleOp>(op)) {
        if (op->hasAttr("native_chunkwise_bf16"))
          op->setAttr("heads",mlir::IntegerAttr::get(mlir::IntegerType::get(&context,64),1));
        for (size_t i = 0; i < outputs.size(); ++i)
          delta.getResult(i).setType(type(&context, outputs[i]));
      }
      else if (llvm::isa<fc::kernel::ScaledMatvecOp>(op)) {
        for (size_t i=0;i<outputs.size();++i) op->getResult(i).setType(type(&context,outputs[i]));
      }
      else if (llvm::isa<fc::kernel::VectorAluOp>(op) && rows >= 0)
        for (auto result : op->getResults()) {
          auto t = llvm::cast<mlir::RankedTensorType>(result.getType());
          V shape(t.getShape().begin(), t.getShape().end());
          if (shape[0] == originalRows)
            shape[0] = rows;
          result.setType(
              mlir::RankedTensorType::get(shape, t.getElementType()));
        }
    });
    llvm::SmallVector<mlir::Type> args, results;
    for (auto &a : n.inputs)
      args.push_back(type(&context, a));
    for (auto &a : outputs)
      results.push_back(type(&context, a));
    n.function.setType(mlir::FunctionType::get(&context, args, results));
    if (mirror)
      n.function->setAttr(source.find("ftlpu.kernel.vector_alu") != std::string::npos
                              ? "ftlpu.vector_mirror" : "ftlpu.depthwise_conv_mirror",
                          mlir::BoolAttr::get(&context, true));
    if (auto policy = policies.getAs<mlir::BoolAttr>("ftlpu.l2_head_partition");
        policy && policy.getValue() && distributedConvOutput)
      n.function->setAttr("ftlpu.l2_head_partition", policy);
    if (auto policy = policies.getAs<mlir::BoolAttr>("ftlpu.vector_bf16_token_halves");
        policy && policy.getValue() && source.find("ftlpu.kernel.vector_alu")!=std::string::npos)
      n.function->setAttr("ftlpu.vector_bf16_token_halves",policy);
    if (distributedConvOutput &&
        source.find("ftlpu.kernel.depthwise_conv1d") != std::string::npos)
      n.function->setAttr("ftlpu.depthwise_conv_distributed_output",
                          mlir::BoolAttr::get(&context, true));
    if ((source.find("ftlpu.kernel.vector_alu") != std::string::npos ||
         source.find("ftlpu.kernel.gated_delta_rule") != std::string::npos) &&
        nodes.size() % 2)
      n.function->setAttr(
          "ftlpu.vxm_queue_base",
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 8));
    if (parallel >= 0 && source.find("ftlpu.kernel.vector_alu") != std::string::npos) {
      n.function->setAttr("ftlpu.vector_token_partition", mlir::BoolAttr::get(&context, true));
      n.function->setAttr("ftlpu.vxm_queue_base",
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 0));
    }
    mlir::PassManager pm(&context);
    pm.enableVerifier(false);
    pm.addNestedPass<mlir::func::FuncOp>(
        fc::create_lower_kernel_to_tensor_pass(attention || n.ffn ? 1 : -1));
    if (mlir::failed(pm.run(*n.module)))
      throw std::runtime_error("native tensor lowering failed: " + n.name);
    if (std::getenv("FTLPU_GRAPH_PROGRESS")) llvm::errs()<<"graph: tensor planned "<<n.name<<"\n";
    if (mlir::failed(fc::schedule::planVectorAluMemory(n.function, target)))
      throw std::runtime_error("native vector storage planning failed: " +
                               n.name);
    n.inputSlots.resize(n.inputs.size());
    n.outputSlots.resize(outputs.size());
    auto ret =
        llvm::cast<mlir::func::ReturnOp>(n.function.front().getTerminator());
    const auto slot = [&](mlir::Operation *op, std::string attr,
                          std::string member = {}) {
      return PlacementSlot{op, std::move(attr), std::move(member), mirror};
    };
    const auto inputSlot = [&](mlir::Value value, PlacementSlot p) {
      if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(value))
        n.inputSlots.at(arg.getArgNumber()).push_back(std::move(p));
    };
    n.function.walk([&](mlir::Operation *op) {
      if (auto p = llvm::dyn_cast<fc::tensor::ProjectionTaskOp>(op)) {
        if (p.getKind() == "linear") {
          inputSlot(p.getInput(), slot(op, "memory_plan", "input"));
          inputSlot(p.getWeight(), slot(op, "memory_plan", "weight"));
        } else {
          if (p.getKind() == "query")
            inputSlot(p.getInput(), slot(op, "memory_plan", "input"));
          inputSlot(p.getWeight(), slot(op, "memory_plan", p.getKind().str()+"_weight"));
          if (p.getBias())
            inputSlot(p.getBias(), slot(op, "memory_plan", p.getKind().str()+"_bias"));
        }
      } else if (auto p = llvm::dyn_cast<fc::tensor::RopeTaskOp>(op)) {
        if (p.getNormWeight())
          inputSlot(p.getNormWeight(), slot(op, "memory_plan", p.getKind().str()+"_norm_weight"));
      } else if (auto p = llvm::dyn_cast<fc::tensor::SigmoidMultiplyTaskOp>(op)) {
        inputSlot(p.getGate(), slot(op, "memory_plan", "output_gate"));
      } else if (auto p = llvm::dyn_cast<fc::tensor::L2NormTaskOp>(op)) {
        inputSlot(p.getInput(), slot(op, "input_allocations"));
      } else if (auto p = llvm::dyn_cast<fc::tensor::RmsNormTaskOp>(op)) {
        inputSlot(p.getInput(), slot(op, "input_allocations"));
        inputSlot(p.getWeight(), slot(op, "weight_allocations"));
      } else if (auto p = llvm::dyn_cast<fc::tensor::MatmulTaskOp>(op)) {
        inputSlot(p.getLhs(), slot(op, "lhs_allocations"));
        inputSlot(p.getRhs(), slot(op, "rhs_allocations"));
      } else if (auto p =
                     llvm::dyn_cast<fc::tensor::DepthwiseConv1DTaskOp>(op)) {
        inputSlot(p.getInput(), slot(op, "input_allocations"));
        inputSlot(p.getWeight(), slot(op, "weight_allocations"));
      } else if (auto p =
                     llvm::dyn_cast<fc::tensor::GatedDeltaRuleTaskOp>(op)) {
        const char *names[] = {"query_allocations", "key_allocations",
                               "value_allocations", "decay_allocations",
                               "beta_allocations",  "state_allocations"};
        for (I i = 0; i < 6; ++i)
          inputSlot(op->getOperand(i), slot(op, names[i]));
      } else if (auto p = llvm::dyn_cast<fc::kernel::ScaledMatvecOp>(op)) {
        inputSlot(p.getMatrix(),slot(op,"matrix_placement"));
        inputSlot(p.getScale(),slot(op,"scale_placement"));
        inputSlot(p.getVector(),slot(op,"vector_placement"));
        if(!p.getEpilogue().empty()) {
          inputSlot(p.getEpilogue()[0],slot(op,"value_placement"));
          inputSlot(p.getEpilogue()[1],slot(op,"coefficient_placement"));
          if(p.getEpilogue().size()==3)inputSlot(p.getEpilogue()[2],slot(op,"query_placement"));
        }
      } else if (auto p = llvm::dyn_cast<fc::kernel::AttentionCoreOp>(op)) {
        inputSlot(p.getQuery(), slot(op, "memory_plan", "query"));
        inputSlot(p.getKey(), slot(op, "memory_plan", "key"));
        inputSlot(p.getValue(), slot(op, "memory_plan", "value"));
        if(p.getSinkValues())inputSlot(p.getSinkValues(),slot(op,"memory_plan","sink"));
      } else if (auto p = llvm::dyn_cast<fc::kernel::VectorAluOp>(op)) {
        inputSlot(p.getLhs(), slot(op, "ftlpu.vector_lhs_placement"));
        inputSlot(p.getRhs(), slot(op, "ftlpu.vector_rhs_placement"));
      }
    });
    for (I i = 0; i < I(outputs.size()); ++i) {
      auto value = ret.getOperand(i);
      auto *op = value.getDefiningOp();
      if (llvm::isa<fc::tensor::ProjectionTaskOp>(op))
        n.outputSlots[i] = slot(op, "memory_plan", "result");
      else if (llvm::isa<fc::tensor::GatedDeltaRuleTaskOp>(op))
        n.outputSlots[i] =
            slot(op, i ? "final_state_allocations" : "output_allocations");
      else if (llvm::isa<fc::kernel::ScaledMatvecOp>(op))
        n.outputSlots[i] = slot(op,i==4 ? "readout_placement" : i==3 ? "updated_placement" : i==2 ? "correction_placement" : i ? "product_placement" : "scaled_placement");
      else if (llvm::isa<fc::kernel::AttentionCoreOp>(op))
        n.outputSlots[i] = slot(op, "memory_plan", "context");
      else if (llvm::isa<fc::kernel::VectorAluOp>(op))
        n.outputSlots[i] = slot(op, "ftlpu.vector_result_placement");
      else
        n.outputSlots[i] = slot(op, "result_allocations");
      n.out.push_back(n.outputSlots[i].get());
    }
    for (auto &slots : n.inputSlots) {
      if (slots.empty())
        throw std::runtime_error("unplanned native input: " + n.name);
      auto placement=slots.front().get();
      if(!placement)throw std::runtime_error("missing native input placement: "+n.name);
      n.in.push_back(placement);
    }
    // Only fixed native workspaces are reserved. Returned values will be
    // assigned graph lifetimes below; their standalone output areas disappear.
    const auto visit = [&](const auto &self, mlir::Attribute a) -> void {
      if (auto p = llvm::dyn_cast<mlir::DictionaryAttr>(a)) {
        if (p.get("slices") && p.get("kind") && p.get("hemisphere") &&
            p.get("instruction_count"))
          {
            if (p.get("dense_attention_role"))
              if (auto segments=p.getAs<mlir::ArrayAttr>("segments")) {
                for (auto segment : segments) {
                  mlir::NamedAttrList physical(p);
                  physical.erase("segments");
                  for (auto field : llvm::cast<mlir::DictionaryAttr>(segment))
                    physical.set(field.getName(),field.getValue());
                  n.workspaces.push_back(slot(n.function,"unused").reflect(physical.getDictionary(&context)));
                }
                return;
              }
            if (auto paged = p.getAs<mlir::BoolAttr>("paged_weight"); paged && paged.getValue()) {
              auto storage = p.getAs<mlir::ArrayAttr>("page_storage_slices");
              if (auto banks = p.getAs<mlir::ArrayAttr>("page_banks")) {
                auto bases = p.getAs<mlir::ArrayAttr>("page_slice_group_bases");
                auto counts = p.getAs<mlir::ArrayAttr>("page_slice_group_counts");
                auto rows = p.getAs<mlir::ArrayAttr>("page_base_rows");
                auto lengths = p.getAs<mlir::ArrayAttr>("page_row_counts");
                const auto pageField = [&](mlir::ArrayAttr array,size_t page,I fallback) {
                  if (!array) return fallback;
                  if (array.size()!=banks.size()) throw std::runtime_error("inconsistent weight-page geometry");
                  return llvm::cast<mlir::IntegerAttr>(array[page]).getInt();
                };
                for (size_t page=0; page<banks.size(); ++page) {
                  const I width = slices(p).size();
                  const I first = pageField(bases,page,field(p,"page_role_group_base"))*width;
                  const I count = pageField(counts,page,field(p,"page_role_group_count"))*width;
                  if (!storage || first<0 || count<=0 || first+count>I(storage.size()))
                    throw std::runtime_error("weight-page slices exceed physical storage");
                  mlir::NamedAttrList physical(p);
                  physical.set("slices",mlir::ArrayAttr::get(&context,storage.getValue().slice(first,count)));
                  mlir::Builder b(&context);
                  physical.set("bank",banks[page]);
                  physical.set("base_row",b.getI64IntegerAttr(pageField(rows,page,field(p,"base_row"))));
                  physical.set("instruction_count",b.getI64IntegerAttr(pageField(lengths,page,field(p,"instruction_count"))));
                  n.workspaces.push_back(slot(n.function,"unused").reflect(physical.getDictionary(&context)));
                }
                return;
              }
              I size = slices(p).size(), first = field(p,"page_role_group_base")*size;
              I count = field(p,"page_role_group_count")*size;
              mlir::NamedAttrList physical(p);
              physical.set("slices", mlir::ArrayAttr::get(&context,
                  storage.getValue().slice(first,count)));
              p = physical.getDictionary(&context);
            }
            n.workspaces.push_back(slot(n.function, "unused").reflect(p));
          }
        else
          for (auto f : p)
            self(self, f.getValue());
      } else if (auto array = llvm::dyn_cast<mlir::ArrayAttr>(a))
        for (auto item : array)
          self(self, item);
    };
    n.function.walk([&](mlir::Operation *op) {
      for (auto attr : op->getAttrs()) {
        bool returned = false;
        for (I idx = 0; idx < I(n.inputs.size()); ++idx) {
          // ScaledMatvec reads its remapped input homes directly. These
          // operands are reserved through n.in and graph lifetimes, not as
          // additional immutable workspaces at the standalone prototype.
          if (llvm::isa<fc::kernel::ScaledMatvecOp>(op))
            for (auto &input : n.inputSlots[idx])
              if (input.op == op && input.attr == attr.getName().strref() &&
                  input.member.empty())
                returned = true;
          if (!n.inputs[idx].external())
            continue;
          auto p = n.in[idx];
          // Paged inputs keep their native SRAM destinations. They are not
          // relocated immutable inputs: every page/bank must reserve space
          // against values retained across this composite operator.
          if (auto paged = p.getAs<mlir::BoolAttr>("paged_weight"); paged && paged.getValue())
            continue;
          bool movable = scalar(p) || text(p, "kind") == "fp16_pair_planar" ||
                         text(p, "kind") == "fp32_quad_planar" ||
                         text(p, "kind") == "fp16_mxm_distributed_16" ||
                         text(p, "kind") == "fp16_projection_bias_x4";
          if (movable)
            continue;
          for (auto &input : n.inputSlots[idx])
            if (input.op == op && input.attr == attr.getName().strref() &&
                input.member.empty())
              returned = true;
        }
        for (auto &out : n.outputSlots)
          if (out.op == op && out.attr == attr.getName().strref()) {
            if (out.member.empty())
              returned = true;
            else {
              auto dict = llvm::cast<mlir::DictionaryAttr>(attr.getValue());
              for (auto field : dict)
                if (field.getName().strref() != out.member)
                  visit(visit, field.getValue());
              returned = true;
            }
          }
        if (!returned)
          visit(visit, attr.getValue());
      }
    });
    for (size_t i = 0; i < outputs.size(); ++i)
      n.outputs.push_back(
          output(nodes.size(), i, outputs[i].shape, outputs[i].dtype));
    auto result = n.outputs;
    nodes.push_back(std::move(n));
    return result;
  }
  Tensor join(Tensor value, std::string name, bool mirrored=false) {
    Node n;
    n.name = std::move(name);
    n.mirroredJoin = mirrored;
    n.inputs = {std::move(value)};
    n.outputs = {output(nodes.size(), 0, n.inputs[0].shape, n.inputs[0].dtype)};
    auto out = n.outputs[0];
    nodes.push_back(std::move(n));
    return out;
  }
// Chunkwise mathematical decomposition before any physical
// layout or ICU schedule exists. This FP32 baseline uses generic vector
// multiply/reduce domains for matrix products; it is not an MXM fast path.
I deltaConstantId = 1000000;
Tensor deltaConstant(V shape, const std::vector<float> &values) {
  Tensor t;
  t.shape = std::move(shape); t.dtype = "f32"; t.constant = deltaConstantId++;
  t.data.resize(values.size() * sizeof(float));
  std::memcpy(t.data.data(), values.data(), t.data.size());
  return t;
}
Tensor deltaBroadcast(const Tensor &t, V shape) {
  if (t.shape == shape) return t;
  if ((t.shape[0] != 1 && t.shape[0] != shape[0]) ||
      (t.shape[1] != 1 && t.shape[1] != shape[1]))
    throw std::runtime_error("invalid chunkwise broadcast");
  V indices;
  for (I r=0; r<shape[0]; ++r)
    for (I c=0; c<shape[1]; ++c)
      indices.push_back((t.shape[0] == 1 ? 0 : r)*t.shape[1] +
                        (t.shape[1] == 1 ? 0 : c));
  return view(t, indices, shape);
}
Tensor deltaAlu(const std::string &name, Tensor a, Tensor b,
                const std::string &kind, bool reduce=false,
                const std::string &dtype="f32", bool fixed=false, bool requireFp32=false) {
  float scalar=0.f;
  if(b.external() && b.dtype=="f32" && b.size()==1 && b.data.size()==4)
    std::memcpy(&scalar,b.data.data(),4);
  const bool immediateOne=kind=="multiply" && scalar==1.f && b.external() && b.size()==1;
  b = deltaBroadcast(b, a.shape);
  Tensor output; output.shape = reduce ? V{a.shape[0],1} : a.shape;
  output.dtype = dtype;
  const auto strType = [&](const Tensor &t) {
    std::string s; llvm::raw_string_ostream os(s); type(&context,t).print(os); return s;
  };
  const auto at=strType(a), bt=strType(b), ot=strType(output);
  const std::string precision = requireFp32 ? ", ftlpu.vector_require_fp32" : "";
  if(immediateOne) {
    // Copies/casts do not need a full matrix of FP32 ones in SRAM. In large
    // batched GEMMs those otherwise become hundreds of isolated constants.
    const auto source="module { func.func @chunk(%a: "+at+") -> "+ot+
        (fixed?" attributes {ftlpu.vector_bf16_spill = true}":"")+
        " { %r = ftlpu.kernel.vector_alu %a, %a {kind = \"multiply\", use_immediate = true, "
        "immediate = 1.0 : f32, reduce = "+(reduce?"true":"false")+precision+"} : ("+at+", "+at+") -> "+ot+
        " return %r : "+ot+" } }";
    return add(name,source,{a},{output})[0];
  }
  auto source = "module { func.func @chunk(%a: " + at + ", %b: " + bt +
      ") -> " + ot + (fixed ? " attributes {ftlpu.vector_bf16_spill = true}" : "") +
      " { %r = ftlpu.kernel.vector_alu %a, %b {kind = \"" + kind +
      "\", use_immediate = false, immediate = 0.0 : f32, reduce = " +
      (reduce ? "true" : "false") + precision + "} : (" + at + ", " + bt + ") -> " + ot +
      " return %r : " + ot + " } }";
  return add(name, source, {a,b}, {output})[0];
}
Tensor deltaTranspose(const Tensor &a) {
  V indices;
  for (I c=0; c<a.shape[1]; ++c)
    for (I r=0; r<a.shape[0]; ++r) indices.push_back(r*a.shape[1]+c);
  return view(a,indices,{a.shape[1],a.shape[0]});
}
Tensor deltaProduct(const std::string &name, const Tensor &a, const Tensor &b) {
  const I m=a.shape[0], k=a.shape[1], n=b.shape[1];
  if (k != b.shape[0]) throw std::runtime_error("chunkwise matmul shape mismatch");
  std::vector<Tensor> parts;
  // Bound both the packed operands and the scalar reduction result homes.
  const I rows = std::max<I>(1,std::min<I>(8,
      target.memory().words_per_bank / (n*k)));
  for (I first=0; first<m; first+=rows) {
    const I count=std::min(rows,m-first);
    V lhs,rhs;
    for (I i=first; i<first+count; ++i)
      for (I j=0; j<n; ++j)
        for (I l=0; l<k; ++l) {
          lhs.push_back(i*k+l); rhs.push_back(l*n+j);
        }
    auto result = deltaAlu(name+".rows"+std::to_string(first),
        view(a,lhs,{count*n,k}),view(b,rhs,{count*n,k}),"multiply",true);
    result.shape={count,n}; parts.push_back(std::move(result));
  }
  return concat(parts);
}
std::vector<Tensor> lowerTokenwiseDelta(const std::string &name,
    const std::vector<Tensor> &input, const std::vector<Tensor> &outputs,
    I heads, I kd, I vd) {
  if(input[0].dtype!="bf16" || kd%32 || vd%32)
    throw std::runtime_error("tokenwise streamed GDN requires BF16 and 32-aligned heads");
  const auto strType=[&](const Tensor &t) {
    std::string text;llvm::raw_string_ostream os(text);type(&context,t).print(os);return text;
  };
  // The semantic State ABI is [head*key,value]. Internally retain
  // [head*value,key] throughout the scan, matching the shared decode kernel.
  V order;
  for(I h=0;h<heads;++h)for(I v=0;v<vd;++v)for(I k=0;k<kd;++k)
    order.push_back((h*kd+k)*vd+v);
  Tensor state=view(input[5],order,{heads*vd,kd});
  const auto fp32Policy=policies.getAs<mlir::BoolAttr>("ftlpu.vector_fp32_ingress");
  const bool full=fp32Policy && fp32Policy.getValue() && state.dtype=="f32";
  std::vector<Tensor> tokens;
  for(I t=0;t<input[0].shape[0];++t) {
    auto q=sliceRows(input[0],t,1),k=sliceRows(input[1],t,1),v=sliceRows(input[2],t,1);
    q.shape=k.shape={heads,kd};v.shape={heads,vd};
    // C4's FP32 LHS occupies bytes 8..11, aliasing its narrow RHS.
    // Widen only the small V vector so it can use the independent 12..15
    // port. State itself keeps its native producer home across all tokens.
    if(full) v=deltaAlu(name+".value_f32"+std::to_string(t),v,
        deltaConstant({1,1},{1.f}),"multiply",false,"f32",true);
    auto d=sliceRows(input[3],t,1),b=sliceRows(input[4],t,1);
    Tensor y; y.shape={heads,vd};y.dtype=state.dtype;
    const auto mt=strType(state),dt=strType(d),kt=strType(k),vt=strType(v),pt=strType(y);
    const auto results="("+mt+", "+pt+", "+pt+", "+mt+", "+pt+")";
    const auto source="module { func.func @step(%s: "+mt+", %d: "+dt+", %k: "+kt+
        ", %v: "+vt+", %b: "+dt+", %q: "+kt+") -> "+results+" { "
        "%scaled, %p, %delta, %updated, %y = ftlpu.kernel.scaled_matvec %s, %d, %k epilogue(%v, %b, %q) : ("+
        mt+", "+dt+", "+kt+", "+vt+", "+dt+", "+kt+") -> "+results+
        " return %scaled, %p, %delta, %updated, %y : "+mt+", "+pt+", "+pt+", "+mt+", "+pt+" } }";
    // Re-enter the generic grouped contraction planner to retain its capacity
    // tiling and the same emitter as batch-1 decode. No FU trace compression.
    auto r=lower(J{{"name",name+".token"+std::to_string(t)},{"kernel",source}},
        {state,d,k,v,b,q},{state,y,y,state,y});
    state=r[3];
    auto token=state.dtype=="bf16" ? r[4] :
        deltaAlu(name+".readout"+std::to_string(t),r[4],deltaConstant({1,1},{1.f}),
                 "multiply",false,"bf16",true);
    nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
    token.shape={1,heads*vd};tokens.push_back(std::move(token));
  }
  order.clear();
  for(I h=0;h<heads;++h)for(I k=0;k<kd;++k)for(I v=0;v<vd;++v)
    order.push_back((h*vd+v)*kd+k);
  return {concat(tokens),view(state,order,outputs[1].shape)};
}
std::vector<Tensor> lowerChunkwiseDelta(const std::string &name,
    const std::vector<Tensor> &originalInput, const std::vector<Tensor> &outputs,
    I heads, I chunk) {
  auto input = originalInput;
  if (chunk != 32 || heads <= 0)
    throw std::runtime_error("chunkwise baseline currently requires chunk32");
  const I tokens=input[0].shape[0], kd=input[0].shape[1]/heads,
          vd=input[2].shape[1]/heads;
  if (tokens%chunk || kd%32 || vd%32)
    throw std::runtime_error("chunkwise baseline requires complete chunks and 32-aligned heads");
  const auto one=deltaConstant({1,1},{1.0f});
  // Bind each external tensor once. Subsequent matrix views are SRAM
  // operands, not duplicated immutable uploads of expanded broadcasts.
  for (size_t i=0; i<input.size(); ++i)
    input[i]=deltaAlu(name+".ingress"+std::to_string(i),input[i],one,
                      "multiply",false,input[i].dtype);
  std::vector<Tensor> headOutputs,headStates;
  for (I h=0; h<heads; ++h) {
    Tensor state=sliceRows(input[5],h*kd,kd);
    std::vector<Tensor> chunks;
    for (I start=0; start<tokens; start+=chunk) {
      const auto prefix=name+".head"+std::to_string(h)+".chunk"+std::to_string(start);
      const auto q=sliceColumns(sliceRows(input[0],start,chunk),h*kd,kd);
      const auto k=sliceColumns(sliceRows(input[1],start,chunk),h*kd,kd);
      const auto v=sliceColumns(sliceRows(input[2],start,chunk),h*vd,vd);
      const auto decay=sliceColumns(sliceRows(input[3],start,chunk),h,1);
      const auto beta=sliceColumns(sliceRows(input[4],start,chunk),h,1);
      // R[i,j] = product(decay[j+1..i]); include incoming-state gamma
      // in column 0. Products avoid inverse decay and 0/0 underflow.
      const I width=64;
      std::vector<float> seed(width,0); seed[0]=1;
      Tensor row=deltaConstant({1,width},seed);
      std::vector<Tensor> decayRows;
      for (I i=0; i<chunk; ++i) {
        row=deltaAlu(prefix+".decay.mul"+std::to_string(i),row,sliceRows(decay,i,1),"multiply");
        std::vector<float> unit(width,0); unit[i+1]=1;
        row=deltaAlu(prefix+".decay.diag"+std::to_string(i),row,deltaConstant({1,width},unit),"add");
        decayRows.push_back(row);
      }
      const auto allDecay=concat(decayRows);
      const auto gamma=sliceColumns(allDecay,0,1);
      const auto r=sliceColumns(allDecay,1,chunk);
      auto gram=deltaProduct(prefix+".KKt",k,deltaTranspose(k));
      auto l=deltaAlu(prefix+".gram_decay",gram,r,"multiply");
      l=deltaAlu(prefix+".gram_beta",l,beta,"multiply");
      // Forward substitution reads only strictly lower entries of L.
      auto history=deltaProduct(prefix+".KS",k,state);
      history=deltaAlu(prefix+".history_decay",history,gamma,"multiply");
      auto rhs=deltaAlu(prefix+".rhs_subtract",v,history,"subtract");
      rhs=deltaAlu(prefix+".rhs_beta",rhs,beta,"multiply");
      std::vector<Tensor> solved;
      for (I i=0; i<chunk; ++i) {
        auto di=sliceRows(rhs,0,1); solved.push_back(di);
        const I remaining=chunk-i-1;
        if (!remaining) break;
        auto column=sliceColumns(sliceRows(l,i+1,remaining),i,1);
        auto correction=deltaAlu(prefix+".solve.mul"+std::to_string(i),
            deltaBroadcast(di,{remaining,vd}),column,"multiply");
        rhs=deltaAlu(prefix+".solve.sub"+std::to_string(i),
            sliceRows(rhs,1,remaining),correction,"subtract");
      }
      const auto d=concat(solved);
      auto attention=deltaProduct(prefix+".QKt",q,deltaTranspose(k));
      attention=deltaAlu(prefix+".causal_decay",attention,r,"multiply");
      auto output=deltaProduct(prefix+".local_output",attention,d);
      auto prior=deltaProduct(prefix+".QS",q,state);
      prior=deltaAlu(prefix+".output_decay",prior,gamma,"multiply");
      output=deltaAlu(prefix+".output_add",output,prior,"add");
      chunks.push_back(deltaAlu(prefix+".output_cast",output,one,"multiply",false,outputs[0].dtype));
      auto lastR=deltaTranspose(sliceRows(r,chunk-1,1));
      auto scaledD=deltaAlu(prefix+".state_delta_decay",d,lastR,"multiply");
      auto update=deltaProduct(prefix+".state_product",deltaTranspose(k),scaledD);
      state=deltaAlu(prefix+".state_decay",state,sliceRows(gamma,chunk-1,1),"multiply");
      state=deltaAlu(prefix+".state_add",state,update,"add");
    }
    headOutputs.push_back(concat(chunks)); headStates.push_back(state);
  }
  return {concat(headOutputs,1),concat(headStates)};
}


  // Standalone batched activation GEMM, distinct from the closed dense
  // attention matcher. Batch entries are independent matrices (decode
  // requests or KV groups). Tile selection and padding live here, not in
  // the model adapter. Reuse the raw-macro BF16 MXM / FP32 accumulation path.
  using ActivationLhsTiles=std::map<std::array<I,3>,Tensor>;
  // The prepared RHS is independent of the LHS row tile. Keep this cache
  // local to a fixed logical RHS (for example Q across Flash KV blocks),
  // rather than retaining an MXM context across unrelated ICU programs.
  // Normal tensor references keep every cached tile live in the SRAM plan.
  struct ActivationRhsTiles {
    bool initialized=false, transpose=false, nativeBf16=false;
    Tensor source;
    std::map<I,Tensor> joinedRows;
    std::map<std::array<I,2>,Tensor> prepared;
    std::map<I,ActivationRhsTiles> columns;
    bool matches(const Tensor &rhs, bool transposeRhs, bool native) const {
      if (!initialized || transpose != transposeRhs || nativeBf16 != native ||
          source.shape != rhs.shape || source.dtype != rhs.dtype ||
          source.constant != rhs.constant || source.sourceOffset != rhs.sourceOffset ||
          source.sourceContiguous != rhs.sourceContiguous || source.refs.size() != rhs.refs.size())
        return false;
      return std::equal(source.refs.begin(), source.refs.end(), rhs.refs.begin(),
          [](const Ref &left, const Ref &right) {
            return left.node == right.node && left.output == right.output && left.index == right.index;
          }) && (!rhs.external() || source.data == rhs.data);
    }
  };
  bool retainNativeRhs = false;
  I nativeRhsReuses = 0;
  ActivationRhsTiles recentNativeRhs;

  bool canRetainNativeRhs(const Tensor &lhs, const Tensor &rhs, bool transpose) const {
    if (!retainNativeRhs || transpose || lhs.shape.size() != 3 || lhs.shape[1] > 32 ||
        rhs.shape.size() != 3 || rhs.dtype != "bf16" || rhs.external() || rhs.refs.empty())
      return false;
    // Limit the live mirrored tiles to one BF16 slice-pair/bank worth of
    // rows per hemisphere. Include the duplicated last head of an odd batch.
    const I headPairs = (rhs.shape[0] + 1) / 2;
    const I elements = 2 * headPairs * rhs.shape[1] * rhs.shape[2];
    if (elements > target.memory().sram_depth_rows * target.memory().bytes_per_word)
      return false;
    const auto producer = rhs.refs.front();
    return nodes[producer.node].outputs[producer.output].dtype == "f32" &&
        llvm::all_of(rhs.refs, [&](const Ref &ref) {
          return ref.node == producer.node && ref.output == producer.output;
        });
  }
  Tensor lowerActivationBatchMatmul(const std::string &name,
      const Tensor &a, Tensor b, const Tensor &output, bool transpose,
      ActivationLhsTiles *sharedLhs=nullptr, ActivationRhsTiles *sharedRhs=nullptr,
      bool nativeBf16=false, bool drainReduction=false, bool retainKAccumulator=false) {
    const I batches=a.shape[0], m=a.shape[1], k=a.shape[2];
    const I n=transpose?b.shape[1]:b.shape[2];
    const auto wide=policies.getAs<mlir::BoolAttr>("ftlpu.sxm_wide");
    if(!wide || !wide.getValue() || k%32 || n%32 ||
       a.dtype!="bf16" || b.dtype!="bf16")
      throw std::runtime_error("activation batch matmul needs wide SXM, BF16 operands and 32-aligned K/N");
    if(sharedRhs) {
      if(!sharedRhs->initialized) {
        sharedRhs->initialized=true;sharedRhs->transpose=transpose;
        sharedRhs->nativeBf16=nativeBf16;sharedRhs->source=b;
      } else if (!sharedRhs->matches(b, transpose, nativeBf16))
        throw std::runtime_error("activation RHS cache requires an unchanged logical tensor and orientation");
    }
    // Producer homes can share a MEM ICU at disjoint rows. Bound each
    // uninterrupted read domain below the SXM-to-activation pipeline lead;
    // otherwise RHS prefetch and repeated LHS reads would interleave macros.
    if(n>64) {
      std::vector<Tensor> columns;
      for(I first=0;first<n;first+=64) {
        const I count=std::min<I>(64,n-first);V indices;
        for(I batch=0;batch<batches;++batch)
          if(transpose)for(I c=0;c<count;++c)for(I r=0;r<k;++r)
            indices.push_back((batch*n+first+c)*k+r);
          else for(I r=0;r<k;++r)for(I c=0;c<count;++c)
            indices.push_back((batch*k+r)*n+first+c);
        auto rhs=view(b,indices,transpose?V{batches,count,k}:V{batches,k,count});
        auto out=output;out.shape={batches,m,count};
        auto part=lowerActivationBatchMatmul(name+".col"+std::to_string(first),a,rhs,out,transpose,sharedLhs,
            sharedRhs?&sharedRhs->columns[first]:nullptr,nativeBf16,drainReduction,retainKAccumulator);
        part.shape={batches*m,count};columns.push_back(std::move(part));
      }
      auto result=concat(columns,1);result.shape=output.shape;return result;
    }
    const auto strType=[&](const Tensor &t) {
      std::string s;llvm::raw_string_ostream os(s);type(&context,t).print(os);return s;
    };
    const auto prepareBf16=[&](Tensor value,const std::string &label,bool matrixLhs=false) {
      // Reuse an exact producer matrix. More complex views are assembled
      // with MEM/SXM only, without expanding BF16 into four FP32 planes and
      // then immediately reading only its original upper two planes.
      bool identity=!value.external() && !value.refs.empty();
      if(identity) {
        const auto first=value.refs.front();
        identity=nodes[first.node].outputs[first.output].shape==value.shape;
        for(I i=0;identity && i<value.size();++i)
          identity=value.refs[i].node==first.node && value.refs[i].output==first.output && value.refs[i].index==i;
      }
      if(identity)return value;
      auto result=join(std::move(value),label,true);
      nodes.back().matmulLhsJoin=matrixLhs;
      return result;
    };
    std::vector<Tensor> batchResults;
    for(I first=0;first<batches;first+=2) {
      Tensor rhsSource=b;
      // Establish matrix rows before transposing an indexed producer view
      // (e.g. heads cut out of one flat projection row). The prepared RHS
      // cache retains this geometry across KV tiles as well as its BF16 tiles.
      const bool joinedRhs=transpose && !b.external();
      if(joinedRhs) {
        // Assemble complete KV rows for just this mirrored matrix pair.
        // Larger batches must not demand one monolithic SRAM allocation.
        if(sharedRhs && sharedRhs->joinedRows.count(first)) rhsSource=sharedRhs->joinedRows.at(first);
        else {
          V indices;
          for(I side=0;side<2;++side)for(I j=0;j<n*k;++j)
            indices.push_back(std::min(first+side,batches-1)*n*k+j);
          rhsSource=join(view(b,indices,{2*n,k}),name+".rhs_rows"+std::to_string(first));
          rhsSource.shape={2,n,k};
          if(sharedRhs)sharedRhs->joinedRows.emplace(first,rhsSource);
        }
      }
      std::vector<std::vector<Tensor>> rows(2);
      for(I row=0;row<m;row+=32) {
        const I live=std::min<I>(32,m-row);
        Tensor sum;
        // Bound a native K contraction by the two mirrored operand homes.
        // This is a logical tile choice before layout/schedule emission.
        const I reductionTile=nativeBf16 && retainKAccumulator && !transpose && n==32 && !sharedLhs && !sharedRhs?
            std::max<I>(32,target.memory().words_per_bank/4/32*32):32;
        for(I reduction=0;reduction<k;reduction+=reductionTile) {
          const I reductionCount=std::min(reductionTile,k-reduction);
          V ai,bi;
          for(I side=0;side<2;++side) {
            const I batch=std::min(first+side,batches-1);
            for(I r=0;r<32;++r)for(I c=0;c<reductionCount;++c)
              ai.push_back((batch*m+row+std::min(r,live-1))*k+reduction+c);
            for(I r=0;r<reductionCount;++r)for(I c=0;c<n;++c)
              bi.push_back(transpose?((joinedRhs?side:batch)*n+c)*k+reduction+r:
                  (batch*k+reduction+r)*n+c);
          }
          const auto prefix=name+".batch"+std::to_string(first)+".row"+
              std::to_string(row)+".k"+std::to_string(reduction);
          // Every output-column tile consumes exactly the same activation.
          // A caller may retain its prepared LHS across these GEMMs. The
          // cache belongs to one contraction (a and its orientation fixed).
          const std::array<I,3> lhsKey{first,row,reduction};
          Tensor lhs;
          if(sharedLhs && sharedLhs->count(lhsKey)) lhs=sharedLhs->at(lhsKey);
          else {
            lhs=nativeBf16?prepareBf16(view(a,ai,{64,reductionCount}),prefix+".lhs",true):
                deltaAlu(prefix+".lhs",view(a,ai,{64,32}),
                    deltaConstant({1,1},{1.f}),"multiply",false,"f32",true);
            if(sharedLhs)sharedLhs->emplace(lhsKey,lhs);
          }
          if(lhs.dtype!=(nativeBf16?"bf16":"f32"))
            throw std::runtime_error("activation LHS cache uses a different storage format");
          const std::array<I,2> rhsKey{first,reduction};
          Tensor rhs;
          if(sharedRhs && sharedRhs->prepared.count(rhsKey)) rhs=sharedRhs->prepared.at(rhsKey);
          else {
              const auto nativeRhs=view(rhsSource,bi,{2*reductionCount,n});
              // Let the memory planner read K32-aligned runs from their live
              // producer homes. It falls back to materialization if the
              // queue, precision or word-contiguity contract cannot be met.
              rhs=nativeBf16 && retainKAccumulator && reductionCount>32 && n==32 && !transpose && !rhsSource.external()?nativeRhs:
                  nativeBf16?prepareBf16(nativeRhs,prefix+".rhs"):
                deltaAlu(prefix+".rhs",view(rhsSource,bi,{64,n}),
                    deltaConstant({1,1},{1.f}),"multiply",false,"f32",true);
            if(sharedRhs)sharedRhs->prepared.emplace(rhsKey,rhs);
          }
          Tensor tile;tile.shape={64,n};tile.dtype="f32";
          const auto at=strType(lhs),bt=strType(rhs),ot=strType(tile);
          if(nativeBf16 && reduction) {
            // Keep the same 32-element FP32 partial sums and add order. The
            // new product goes directly from MXM to VXM, while the earlier
            // sum arrives on the other fixed FP32 input. The memory plan
            // may keep its native hemisphere ownership and select matching
            // logical halves from mirrored matrix operands.
            auto prior=sum;
            const bool cast=output.dtype=="bf16" && reduction+reductionCount==k;
            auto result=tile;if(cast)result.dtype="bf16";
            const auto resultType=strType(result);
            const std::string epilogue=cast?
                " %result = ftlpu.kernel.vector_alu %sum, %sum {kind = \"multiply\", "
                "use_immediate = true, immediate = 1.0 : f32, reduce = false} : ("+ot+", "+ot+") -> "+resultType:
                "";
            const auto source="module { func.func @stage(%a: "+at+", %b: "+bt+", %prior: "+ot+
                ") -> "+resultType+" attributes {ftlpu.vector_independent_ports = true} { "
                "%product = ftlpu.kernel.vector_alu %a, %b {kind = \"head_matmul\", "
                "mxm_precision = \"bf16\", use_immediate = false, immediate = 0.0 : f32, "
                "reduce = false} : ("+at+", "+bt+") -> "+ot+
                " %sum = ftlpu.kernel.vector_alu %product, %prior {kind = \"add\", "
                "use_immediate = false, immediate = 0.0 : f32, reduce = false} : ("+ot+", "+ot+") -> "+ot+
                epilogue+" return "+(cast?"%result":"%sum")+" : "+resultType+" } }";
            sum=add(prefix,source,{lhs,rhs,prior},{result})[0];
            // Long contractions can refill finite iMEM at materialized FP32
            // partial-sum boundaries. No active MXM accumulator spans a refill.
            if(drainReduction) {
              nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
              nodes.back().function->setAttr("ftlpu.tight_serial_schedule",mlir::UnitAttr::get(&context));
            }
            continue;
          }
          const auto source="module { func.func @stage(%a: "+at+", %b: "+bt+
              ") -> "+ot+(nativeBf16?" attributes {ftlpu.vector_independent_ports = true}":"")+
              " { %r = ftlpu.kernel.vector_alu %a, %b {kind = \"head_matmul\", "
              "mxm_precision = \"bf16\", use_immediate = false, immediate = 0.0 : f32, "
              "reduce = false} : ("+at+", "+bt+") -> "+ot+" return %r : "+ot+" } }";
          auto product=add(prefix,source,{lhs,rhs},{tile})[0];
          sum=reduction?deltaAlu(prefix+".accumulate",sum,product,"add",false,"f32",true):product;
          if(drainReduction) {
            nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
            nodes.back().function->setAttr("ftlpu.tight_serial_schedule",mlir::UnitAttr::get(&context));
          }
        }
        if(sum.dtype!=output.dtype)sum=deltaAlu(name+".cast",sum,
            deltaConstant({1,1},{1.f}),"multiply",false,output.dtype,true);
        // All tile results are in SRAM at this boundary. Large independent
        // batches can reload finite VXM/SXM iMEM without retaining an active
        // FU context across the reload.
        if(batches>8)
          nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
        for(I side=0;side<2 && first+side<batches;++side)
          rows[side].push_back(sliceRows(sum,side*32,live));
      }
      for(I side=0;side<2 && first+side<batches;++side)
        batchResults.push_back(concat(rows[side]));
    }
    auto result=concat(batchResults);result.shape=output.shape;return result;
  }

  // A word of BF16 probabilities contains one key's 32 independent query
  // rows. Wide SXM transposes one full tile per group for MXM activation;
  // all V column tiles then reuse the same converted LHS. Padding query
  // rows are simply discarded after PV; the reduction still visits key
  // positions in their original order with the original BF16 operands.
  Tensor lowerTimeMajorProbabilityValue(const std::string &name, Tensor pT,
      const Tensor &value, I live, const Tensor &out, ActivationRhsTiles *sharedValue=nullptr) {
    const I tile=target.throughput().mxm_rows;
    if(tile!=32 || live<=0 || live>tile || pT.dtype!="bf16" || value.dtype!="bf16" ||
       out.dtype!="f32" || (pT.shape!=V{2,tile,tile} && pT.shape!=V{2*tile,tile}) ||
       value.shape.size()!=3 || value.shape[0]!=2 || value.shape[1]!=tile ||
       value.shape[2]<=0 || value.shape[2]%tile || out.shape!=V{2,live,value.shape[2]})
      throw std::runtime_error("PV requires two key-major BF16 probability tiles and aligned V");
    const I width=value.shape[2];
    pT.shape={2*tile,tile};
    if(!pT.external() && I(pT.refs.size())!=pT.size())
      throw std::runtime_error("PV expects initialized probability words, including masked zero rows");
    // Full live tiles from [keys, 2*32] can transpose directly, including
    // when the two groups occupy different column words of one producer.
    // A partial KV tile may instead consist of short source runs mixed
    // with shared zero words. First assemble those whole words into a real
    // 32x32 tile so the wide-SXM transpose never reads uninitialized rows.
    bool direct=I(pT.refs.size())==pT.size();
    for(I side=0;direct && side<2;++side) {
      const auto ref=pT.refs[side*tile*tile];
      const auto &shape=nodes[ref.node].outputs[ref.output].shape;
      direct=shape.size()==2 && shape[1]>=tile && shape[1]%tile==0 && ref.index%shape[1]%tile==0 &&
          ref.index/shape[1]+tile<=shape[0];
      for(I k=0;direct && k<tile;++k)for(I q=0;direct && q<tile;++q) {
        const auto r=pT.refs[(side*tile+k)*tile+q];
        direct=r.node==ref.node && r.output==ref.output && r.index==ref.index+k*shape[1]+q;
      }
    }
    if(!direct)pT=join(pT,name+".padded_key_rows");
    V order;
    for(I side=0;side<2;++side)for(I q=0;q<tile;++q)for(I k=0;k<tile;++k)
      order.push_back((side*tile+k)*tile+q);
    auto transposed=view(pT,order,{2,tile,tile});
    auto padded=out;padded.shape={2,tile,width};
    ActivationLhsTiles sharedLhs;
    auto result=lowerActivationBatchMatmul(name,transposed,value,padded,false,&sharedLhs,sharedValue,true);
    order.clear();
    for(I side=0;side<2;++side)for(I q=0;q<live;++q)for(I d=0;d<width;++d)
      order.push_back((side*tile+q)*width+d);
    return view(result,order,out.shape);
  }
  #include "FlashTileLowering.inc"
  #include "FlashAttentionLowering.inc"
  #include "DenseAttentionLowering.inc"
  #include "ProjectionGraphLowering.inc"
  #include "RotaryGraphLowering.inc"
  #include "MhcGraphLowering.inc"

  std::vector<Tensor> lower(const J &j, std::vector<Tensor> input,
                            std::vector<Tensor> outputs) {
    auto source = j.getString("kernel")->str(),
         name = j.getString("name")->str();
    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    if (!parsed) throw std::runtime_error("invalid kernel subgraph: " + name);
    auto parsedFunction = *parsed->getOps<mlir::func::FuncOp>().begin();
    if(parsedFunction.front().getOperations().size()==2 && input.size()==1 && outputs.size()==1)
      if(auto op=llvm::dyn_cast<fc::kernel::TransposeOp>(parsedFunction.front().front())) {
        if(mlir::failed(op.verify()))throw std::runtime_error("invalid logical transpose");
        const auto permutation=op.getPermutation();const auto &shape=outputs[0].shape;
        V sourceStride(input[0].shape.size(),1),order;
        for(I i=I(sourceStride.size())-2;i>=0;--i)sourceStride[i]=sourceStride[i+1]*input[0].shape[i+1];
        for(I index=0;index<outputs[0].size();++index) {
          I remaining=index,sourceIndex=0;
          for(I dim=I(shape.size())-1;dim>=0;--dim) {
            sourceIndex+=(remaining%shape[dim])*sourceStride[permutation[dim]];remaining/=shape[dim];
          }
          order.push_back(sourceIndex);
        }
        return {view(input[0],order,shape)};
      }
    if(parsedFunction.front().getOperations().size()==2) {
      auto *operation=&parsedFunction.front().front();
      if(llvm::isa<fc::kernel::HcMixesOp,fc::kernel::HcPreOp,fc::kernel::HcPostOp>(operation)) {
        if(mlir::failed(mlir::verify(operation)))throw std::runtime_error("invalid hyper-connection signature");
        std::vector<Tensor> arguments;
        for(auto operand:operation->getOperands())
          arguments.push_back(input.at(llvm::cast<mlir::BlockArgument>(operand).getArgNumber()));
        if(auto op=llvm::dyn_cast<fc::kernel::HcMixesOp>(operation))
          return lowerHcMixes(name,arguments,op.getIterations(),op.getEpsilon().convertToFloat(),op.getNormEpsilon().convertToFloat());
        if(llvm::isa<fc::kernel::HcPreOp>(operation))return {lowerHcPre(name,arguments[0],arguments[1])};
        return {lowerHcPost(name,arguments)};
      }
    }
    if(parsedFunction.front().getOperations().size()==2 && input.size()==1 && outputs.size()==1)
      if(auto op=llvm::dyn_cast<fc::kernel::BlockFp8QdqOp>(parsedFunction.front().front())) {
        if(mlir::failed(op.verify()))throw std::runtime_error("invalid block FP8 QDQ signature");
        return {lowerBlockQuantizedActivation(name,input[0],true)};
      }
    if(parsedFunction.front().getOperations().size()==2 && input.size()==1 && outputs.size()==1)
      if(auto op=llvm::dyn_cast<fc::kernel::RopeOp>(parsedFunction.front().front())) {
        const auto interleaved=op->getAttrOfType<mlir::BoolAttr>("interleaved");
        if(interleaved && interleaved.getValue()) {
          if(mlir::failed(op.verify())) throw std::runtime_error("invalid standalone RoPE");
          const auto inverse=op->getAttrOfType<mlir::BoolAttr>("inverse");
          return {lowerInterleavedRotary(name,input[0],integer(op,"rotary_dim",op.getHeadDim()),
              integer(op,"rotary_offset",0),op.getTheta().convertToFloat(),inverse && inverse.getValue(),integer(op,"position_offset",0))};
        }
      }
    if (parsedFunction.front().getOperations().size() == 2 && input.size() == 3 && outputs.size() == 1)
      if (auto op = llvm::dyn_cast<fc::kernel::BlockFp8ProjectionOp>(parsedFunction.front().front())) {
        if (mlir::failed(op.verify())) throw std::runtime_error("invalid FP8 projection signature");
        const bool quantizeActivation=op.getQuantizeActivation().value_or(true);
        const auto arg = [](mlir::Value v) { return llvm::cast<mlir::BlockArgument>(v).getArgNumber(); };
        const auto &scales = input[arg(op.getWeightScales())];
        if (!scales.external()) throw std::runtime_error("FP8 weight scales must be constant");
        llvm::SmallVector<float> decoded;
        for (auto code : scales.data) {
          if (code == 255) throw std::runtime_error("UE8M0 NaN scale is invalid");
          decoded.push_back(std::ldexp(1.f, int(code) - 127));
        }
        mlir::Builder b(&context);
        const I groups=integer(op,"groups",1);
        if(groups>1) {
          const auto &x=input[arg(op.getInput())],&w=input[arg(op.getWeights())];
          const I inWidth=x.shape[1]/groups,outWidth=w.shape[0]/groups;
          std::vector<Tensor> results;
          for(I group=0;group<groups;++group) {
            auto output=outputs[0];output.shape[1]=outWidth;
            const auto groupScales=llvm::ArrayRef<float>(decoded).slice(group*(outWidth/32)*(inWidth/32),(outWidth/32)*(inWidth/32));
            const auto cfg=b.getDictionaryAttr({b.getNamedAttr("unit",b.getStringAttr("MXM")),
                b.getNamedAttr("rhs_scale",b.getF32FloatAttr(1.f)),
                b.getNamedAttr("rhs_block_scales",b.getDenseF32ArrayAttr(groupScales))});
            results.push_back(lowerProjection(name+".group"+std::to_string(group),sliceColumns(x,group*inWidth,inWidth),
                deltaTranspose(sliceRows(w,group*outWidth,outWidth)),output,cfg,quantizeActivation));
          }
          return {concat(results,1)};
        }
        const auto config = b.getDictionaryAttr({
            b.getNamedAttr("unit", b.getStringAttr("MXM")),
            b.getNamedAttr("rhs_scale", b.getF32FloatAttr(1.f)),
            b.getNamedAttr("rhs_block_scales", b.getDenseF32ArrayAttr(decoded))});
        return {lowerProjection(name, input[arg(op.getInput())],
            deltaTranspose(input[arg(op.getWeights())]), outputs[0], config, quantizeActivation)};
      }
    if(parsedFunction.front().getOperations().size()==2 && outputs.size()==1)
      if(auto op=llvm::dyn_cast<fc::kernel::DenseAttentionOp>(parsedFunction.front().front())) {
        if(mlir::failed(op.verify())) throw std::runtime_error("invalid dense attention signature");
        const auto arg=[](mlir::Value v){return llvm::cast<mlir::BlockArgument>(v).getArgNumber();};
        return {lowerDenseAttention(name,input[arg(op.getQuery())],input[arg(op.getKey())],
            input[arg(op.getValue())],outputs[0],op.getScale().convertToFloat(),op.getCausal(),op.getQueryPosition(),
            op->getAttrOfType<mlir::DenseF32ArrayAttr>("attention_sinks"),
            op->getAttrOfType<mlir::IntegerAttr>("sliding_window")?op->getAttrOfType<mlir::IntegerAttr>("sliding_window").getInt():0)};
      }
    if(parsedFunction.front().getOperations().size()==2 && input.size()==3 && outputs.size()==1)
      if(auto op=llvm::dyn_cast<fc::kernel::FlashAttentionOp>(parsedFunction.front().front())) {
        if(mlir::failed(op.verify())) throw std::runtime_error("invalid flash attention signature");
        const auto arg=[](mlir::Value v){return llvm::cast<mlir::BlockArgument>(v).getArgNumber();};
        return {lowerFlashAttention(name,input[arg(op.getQuery())],input[arg(op.getKey())],
            input[arg(op.getValue())],outputs[0],op.getScale().convertToFloat(),op.getCausal(),op.getQueryPosition())};
      }
    if(parsedFunction.front().getOperations().size()==2 && input.size()==2 && outputs.size()==1)
      if(auto op=llvm::dyn_cast<fc::kernel::BatchMatmulOp>(parsedFunction.front().front())) {
        const auto lhs=llvm::dyn_cast<mlir::BlockArgument>(op.getLhs());
        const auto rhs=llvm::dyn_cast<mlir::BlockArgument>(op.getRhs());
        if(lhs && rhs) {
          // Batched activation products already have BF16 operands, including
          // explicit high-plane views of FP32 State. Preserve those bytes and
          // reuse each prepared activation across output-column tiles. The
          // existing 32-K FP32 partial-product/add order stays unchanged.
          ActivationLhsTiles sharedLhs;
          const auto &a = input[lhs.getArgNumber()];
          const auto &b = input[rhs.getArgNumber()];
          const bool transpose = op.getTransposeRhs();
          const bool eligible = canRetainNativeRhs(a, b, transpose);
          const bool reuse = eligible && recentNativeRhs.matches(b, transpose, true);
          if (!reuse) recentNativeRhs = {};
          auto result = lowerActivationBatchMatmul(name, a, b, outputs[0], transpose,
              &sharedLhs, eligible ? &recentNativeRhs : nullptr, true);
          if (reuse) {
            ++nativeRhsReuses;
            recentNativeRhs = {};
          }
          return {std::move(result)};
        }
      }
    llvm::SmallVector<fc::kernel::MatmulOp> narrowPair;
    for(auto op:parsedFunction.getOps<fc::kernel::MatmulOp>()) narrowPair.push_back(op);
    if(narrowPair.size()==2 && outputs.size()==2 &&
        parsedFunction.front().getOperations().size()==3 &&
        llvm::all_of(narrowPair,[&](auto op) {
          return op.getM()==target.throughput().mxm_rows &&
              op.getN()>0 && op.getN()<=target.throughput().mxm_rows &&
              llvm::isa<mlir::BlockArgument>(op.getLhs()) &&
              llvm::isa<mlir::BlockArgument>(op.getRhs());
        })) {
      auto a=narrowPair[0],b=narrowPair[1];
      const I tile=target.throughput().mxm_rows;
      const auto argument=[](mlir::Value v) {return llvm::cast<mlir::BlockArgument>(v).getArgNumber();};
      const auto x=input[argument(a.getLhs())];
      const auto wa=input[argument(a.getRhs())],wb=input[argument(b.getRhs())];
      if(a.getLhs()!=b.getLhs() || a.getM()!=tile || b.getM()!=tile ||
          a.getK()!=b.getK() || a.getN()>tile || b.getN()>tile ||
          !wa.external() || !wb.external() || wa.dtype!="i8" || wb.dtype!="i8")
        throw std::runtime_error("invalid paired narrow projections");
      Tensor weight;weight.shape={x.shape[1],2*tile};weight.dtype="i8";
      weight.constant=deltaConstantId++;weight.data.resize(weight.size(),0);
      for(I row=0;row<x.shape[1];++row) {
        std::copy_n(wa.data.begin()+row*a.getN(),a.getN(),weight.data.begin()+row*2*tile);
        std::copy_n(wb.data.begin()+row*b.getN(),b.getN(),weight.data.begin()+row*2*tile+tile);
      }
      Tensor result=outputs[0];result.shape={tile,2*tile};
      mlir::OpBuilder build(&context);auto loc=a.getLoc();
      mlir::OwningOpRef<mlir::ModuleOp> part=mlir::ModuleOp::create(loc);
      auto fn=mlir::func::FuncOp::create(loc,"paired_projection",
          build.getFunctionType({type(&context,x),type(&context,weight)},{type(&context,result)}));
      part->push_back(fn);build.setInsertionPointToStart(fn.addEntryBlock());
      mlir::OperationState state(loc,fc::kernel::MatmulOp::getOperationName());
      state.addOperands(fn.getArguments());state.addTypes(type(&context,result));
      mlir::NamedAttrList attributes(a->getAttrs());
      attributes.set("n",build.getI64IntegerAttr(2*tile));
      attributes.set("rhs_scale_west",b.getRhsScaleAttr());
      attributes.set("ftlpu.replicated_narrow_pair",build.getUnitAttr());
      state.addAttributes(attributes);auto *joint=build.create(state);
      build.create<mlir::func::ReturnOp>(loc,joint->getResults());
      std::string pairedSource;llvm::raw_string_ostream pairedStream(pairedSource);part->print(pairedStream);
      auto value=add(name+".paired",pairedSource,{x,weight},{result})[0];
      return {sliceColumns(value,0,a.getN()),sliceColumns(value,tile,b.getN())};
    }
    if (!parsedFunction.getOps<fc::kernel::ScaledMatvecOp>().empty()) {
      const I groupOffset=(*parsedFunction.getOps<fc::kernel::ScaledMatvecOp>().begin()).getGroupOffset();
      const I heads=input[2].shape[0], columns=input[0].shape[0]/heads;
      // Input state and retained output occupy different slice groups.
      // They do not need to split one bank's depth between ping/pong halves.
      // The lifetime-aware allocator below still checks full-depth residency.
      // A staged mixer also retains external gate tables throughout ingress.
      // Leave half-depth headroom for those immutable bindings and the second
      // state image instead of requiring another completely empty slice group.
      const I stateRows = input[1].external() && input.size() >= 5 && input[4].external()
          ? target.memory().sram_depth_rows/2 : target.memory().sram_depth_rows;
      const I perTile=std::min<I>(32,std::max<I>(1,stateRows/
          (columns*((input[0].shape[1]+31)/32))));
      std::vector<Tensor> matrices,products,corrections,updates,readouts;
      for(I h=0;h<heads;h+=perTile) {
        const I count=std::min(perTile,heads-h);
        auto a=sliceRows(input[0],h*columns,count*columns);
        // Keep the packed scale vector intact: a sub-word head slice is not
        // a base-address adjustment. The emitter selects rows after SXM.
        auto d=input[1];
        auto k=sliceRows(input[2],h,count);
        auto outs=outputs;outs[0].shape=a.shape;outs[1].shape={count,columns};
        std::vector<Tensor> args={a,d,k};
        if(input.size()>=5) {
          args.push_back(sliceRows(input[3],h,count));args.push_back(input[4]);
          outs[2].shape={count,columns};
          if(outs.size()>=4)outs[3].shape=a.shape;
          if(input.size()==6) {
            args.push_back(sliceRows(input[5],h,count));outs[4].shape={count,columns};
          }
        }
        auto result=add(name+".heads"+std::to_string(h),source,args,outs);
        nodes.back().function.walk([&](fc::kernel::ScaledMatvecOp op) {
          mlir::Builder b(&context);
          op->setAttr("group_offset",b.getI64IntegerAttr(groupOffset+h));
        });
        matrices.push_back(result[0]);products.push_back(result[1]);
        if(result.size()>=3)corrections.push_back(result[2]);
        if(result.size()>=4)updates.push_back(result[3]);
        if(result.size()==5)readouts.push_back(result[4]);
      }
      if(!readouts.empty())return {concat(matrices),concat(products),concat(corrections),concat(updates),concat(readouts)};
      if(!updates.empty())return {concat(matrices),concat(products),concat(corrections),concat(updates)};
      if(!corrections.empty())return {concat(matrices),concat(products),concat(corrections)};
      return {concat(matrices),concat(products)};
    }
    if (input.size()==1 && outputs.size()==1 && llvm::all_of(parsedFunction.front(),[](mlir::Operation& op) {
          return llvm::isa<fc::kernel::ReshapeOp,mlir::func::ReturnOp>(op);
        })) return {view(input[0],range(input[0].size()),outputs[0].shape)};
    auto patterns = fc::kernel::match_decoder_graph(*parsed->getOps<mlir::func::FuncOp>().begin());
    if (!patterns.ffns.empty()) {
      auto &ffn = patterns.ffns.front();
      const I m = ffn.gate.getM(), hidden = ffn.gate.getN();
      const auto &hw = target.throughput();
      const I tile = hw.mxm_rows;
      // Keep room for Gate/Up temporaries next to the live activation.
      // Token tiling is a compiler decision; the caller still supplies one
      // independent batched FFN and receives its original row order.
      const I tokenCapacity=std::max<I>(tile,
          (target.memory().sram_depth_rows/2*target.memory().bytes_per_word/
           ffn.gate.getK()/tile)*tile);
      if(hw.vxm_alus==16 && m>tokenCapacity && patterns.ffns.size()==1 && outputs.size()==1) {
        const auto arg=llvm::dyn_cast<mlir::BlockArgument>(ffn.gate.getLhs());
        if(!arg)throw std::runtime_error("FFN token tiling requires a region input activation");
        std::vector<Tensor> pieces;
        for(I first=0;first<m;first+=tokenCapacity) {
          const I count=std::min(tokenCapacity,m-first);
          auto args=input,outs=outputs;
          args[arg.getArgNumber()]=sliceRows(args[arg.getArgNumber()],first,count);
          outs[0].shape[0]=count;
          auto part=mlir::parseSourceString<mlir::ModuleOp>(source,&context);
          auto fn=*part->getOps<mlir::func::FuncOp>().begin();
          mlir::Builder builder(&context);
          llvm::SmallVector<mlir::Type> argTypes;
          for(size_t i=0;i<args.size();++i) {
            auto t=type(&context,args[i]);fn.getArgument(i).setType(t);argTypes.push_back(t);
          }
          fn.walk([&](mlir::Operation *op) {
            if(auto matmul=llvm::dyn_cast<fc::kernel::MatmulOp>(op))
              matmul->setAttr("m",builder.getI64IntegerAttr(count));
            for(auto result:op->getResults())if(auto t=llvm::dyn_cast<mlir::RankedTensorType>(result.getType());
                t && t.getRank()==2 && t.getDimSize(0)==m)
              result.setType(mlir::RankedTensorType::get({count,t.getDimSize(1)},t.getElementType()));
          });
          fn.setType(builder.getFunctionType(argTypes,{type(&context,outs[0])}));
          std::string tiled;llvm::raw_string_ostream os(tiled);part->print(os);
          pieces.push_back(lower(J{{"name",name+".tokens"+std::to_string(first)},
              {"kernel",tiled}},std::move(args),std::move(outs))[0]);
        }
        return {concat(pieces)};
      }
      const I tokenRows = ((m + tile - 1) / tile) * tile;
      const I alignment = target.memory().hemispheres * tile;
      // The SR SwiGLU retains two Gate copies and Up over this output batch.
      // Derive the batch from physical residency, before emitting any ICU/FU
      // instructions. Decode retains its independent native strategy.
      const I workspaceRows = target.memory().sram_depth_rows /
          (retainedAcrossFfn ? 2 : 1);
      const I batch = hw.vxm_alus == 16 && m >= tile
          ? (workspaceRows / tokenRows) * alignment
          : hidden;
      if (batch <= 0)
        throw std::runtime_error("FFN token tile exceeds on-chip temporary capacity");
      if (hidden <= batch)
        return add(name, source, std::move(input), std::move(outputs));
      if (patterns.ffns.size() != 1 || outputs.size() != 1
          || hidden % alignment || input.size() != 4)
        throw std::runtime_error("on-chip FFN batching requires an aligned closed dense FFN");
      auto argument = [](mlir::Value value) -> size_t {
        auto arg = llvm::dyn_cast<mlir::BlockArgument>(value);
        if (!arg) throw std::runtime_error("FFN batch operand must be a region input");
        return arg.getArgNumber();
      };
      const size_t gateArg = argument(ffn.gate.getRhs());
      const size_t upArg = argument(ffn.up.getRhs());
      const size_t downArg = argument(ffn.output.getRhs());
      for (auto index : {gateArg, upArg, downArg})
        if (!input[index].external())
          throw std::runtime_error("FFN weight batching requires immutable weight tensors");
      Tensor sum;
      for (I first = 0; first < hidden; first += batch) {
        const I count = std::min(batch, hidden - first);
        auto args = input;
        args[gateArg] = sliceColumns(input[gateArg], first, count);
        args[upArg] = sliceColumns(input[upArg], first, count);
        args[downArg] = sliceRows(input[downArg], first, count);
        auto part = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
        auto fn = *part->getOps<mlir::func::FuncOp>().begin();
        auto subgraphs = fc::kernel::match_decoder_graph(fn);
        auto &sub = subgraphs.ffns.front();
        mlir::Builder b(&context);
        for (size_t i = 0; i < args.size(); ++i)
          fn.getArgument(i).setType(type(&context, args[i]));
        const auto hiddenType = mlir::RankedTensorType::get({m, count},
            llvm::cast<mlir::RankedTensorType>(sub.gate.getResult().getType()).getElementType());
        for (auto op : {sub.gate, sub.up}) {
          op->setAttr("n", b.getI64IntegerAttr(count));
          op.getResult().setType(hiddenType);
        }
        sub.swish.getResult().setType(hiddenType);
        sub.multiply.getResult().setType(hiddenType);
        sub.output->setAttr("k", b.getI64IntegerAttr(count));
        llvm::SmallVector<mlir::Type> argTypes;
        for (auto &arg : args) argTypes.push_back(type(&context, arg));
        fn.setType(b.getFunctionType(argTypes, fn.getResultTypes()));
        std::string batchSource;
        llvm::raw_string_ostream batchStream(batchSource);
        part->print(batchStream);
        auto partial = add(name + ".hidden" + std::to_string(first),
                           batchSource, std::move(args), outputs)[0];
        if (first == 0) {
          sum = std::move(partial);
          continue;
        }
        // The ordinary graph lifetime allocator retains only the running
        // result and normalized input across batches. All sums execute in
        // VXM with SRAM operands; no host operation or activation DDR spill.
        std::string ty;
        llvm::raw_string_ostream typeStream(ty);
        type(&context, outputs[0]).print(typeStream);
        const auto fixedSum=policies.getAs<mlir::BoolAttr>("ftlpu.vector_bf16_spill");
        std::string addSource = "module { func.func @sum(%a: " + ty
            + ", %b: " + ty + ") -> " + ty
            + (fixedSum && fixedSum.getValue()?" attributes {ftlpu.vector_bf16_spill = true}":"")
            + " { %r = ftlpu.kernel.vector_alu %a, %b {kind = \"add\","
              " use_immediate = false, immediate = 0.0 : f32, reduce = false} : ("
            + ty + ", " + ty + ") -> " + ty + " return %r : " + ty + " } }";
        const I rowsPerSum = std::max<I>(1,
            (target.memory().sram_depth_rows / 2)
                * target.memory().bytes_per_word / outputs[0].shape[1]);
        std::vector<Tensor> sums;
        for (I row = 0; row < m; row += rowsPerSum) {
          const I rows = std::min(rowsPerSum, m - row);
          auto result = outputs[0];
          result.shape[0] = rows;
          sums.push_back(add(name + ".sum" + std::to_string(first)
                + ".rows" + std::to_string(row), addSource,
              {sliceRows(sum, row, rows), sliceRows(partial, row, rows)},
              {result}, rows)[0]);
        }
        sum = concat(sums);
      }
      return {sum};
    }
    if (!patterns.attentions.empty()) {
      auto &attention = patterns.attentions.front();
      const I m = attention.query.getM(), width = attention.query.getK();
      const I heads = attention.query_rope.getHeads();
      const I kvHeads = attention.key_rope.getHeads();
      const I dim = attention.query_rope.getHeadDim();
      const auto &hw = target.throughput();
      const I tile = hw.mxm_rows, depth = target.memory().words_per_bank;
      if (m >= tile && m % tile == 0 && dim >= tile && dim % tile == 0
          && width % tile == 0 && kvHeads > 0 && heads % kvHeads == 0
          && hw.mxms_per_hemisphere == 1
          && target.uses_dedicated_slice_roles()) {
        const I blocks = (m + tile - 1) / tile, headBlocks = dim / tile;
        const I inputRows = blocks * (width / tile) * hw.tile_rows;
        const I tableRows = m * dim / (2 * tile);
        // One KV head accompanies each query batch. Bound both the explicit
        // RoPE products and the complete raw-Q staging before emission.
        const I productLimit = (depth - tableRows) / (headBlocks * m) - 1;
        const I stagingLimit = (depth - inputRows)
            / (headBlocks * m + ((headBlocks + 1) / 2) * blocks * hw.tile_rows);
        const I perKv = heads / kvHeads;
        I batch = std::min({heads, productLimit, stagingLimit,
                            depth / (headBlocks * m)});
        if (heads > batch) {
          batch = std::min(batch, perKv);
          while (batch > 0 && perKv % batch) --batch;
          if (batch <= 0)
            throw std::runtime_error("one attention head exceeds on-chip prefill capacity");
          if (patterns.attentions.size() != 1 || outputs.size() != 1
              || attention.query_norm || attention.key_norm || attention.output_gate)
            throw std::runtime_error("on-chip head batching requires a closed dense attention without head norm or output gate");
          auto argument = [](mlir::Value v) -> size_t {
            auto arg = llvm::dyn_cast<mlir::BlockArgument>(v);
            if (!arg) throw std::runtime_error("attention batch weights must be region inputs");
            return arg.getArgNumber();
          };
          const size_t qa = argument(attention.query.getRhs());
          const size_t ka = argument(attention.key.getRhs());
          const size_t va = argument(attention.value.getRhs());
          const size_t oa = argument(attention.output.getRhs());
          for (auto index : {qa, ka, va, oa})
            if (!input[index].external())
              throw std::runtime_error("attention head batching requires immutable weights");
          Tensor sum;
          for (I first = 0; first < heads; first += batch) {
            const I kv = first / perKv;
            auto args = input;
            args[qa] = sliceColumns(input[qa], first * dim, batch * dim);
            args[ka] = sliceColumns(input[ka], kv * dim, dim);
            args[va] = sliceColumns(input[va], kv * dim, dim);
            args[oa] = sliceRows(input[oa], first * dim, batch * dim);
            auto part = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
            auto fn = *part->getOps<mlir::func::FuncOp>().begin();
            auto matches = fc::kernel::match_decoder_graph(fn);
            auto &sub = matches.attentions.front();
            mlir::Builder b(&context);
            const auto element = llvm::cast<mlir::RankedTensorType>(sub.query.getResult().getType()).getElementType();
            const auto shaped = [&](llvm::ArrayRef<I> shape) { return mlir::RankedTensorType::get(shape, element); };
            const auto setResult = [&](mlir::Operation *op, llvm::ArrayRef<I> shape) { op->getResult(0).setType(shaped(shape)); };
            const auto bias = [&](fc::kernel::BiasAddOp op, I begin, I count) {
              if (!op) return;
              const auto index = argument(op->getOperand(1));
              if (!input[index].external()) throw std::runtime_error("attention bias must be immutable");
              args[index] = view(input[index], range(count,begin), {count});
              if (input[index].shape.size() == 2) args[index].shape = {1,count};
              setResult(op,{m,count});
            };
            bias(sub.query_bias, first*dim, batch*dim);
            bias(sub.key_bias, kv*dim, dim);
            bias(sub.value_bias, kv*dim, dim);
            for (auto projection : {sub.query, sub.key, sub.value}) {
              const I n = projection == sub.query ? batch*dim : dim;
              projection->setAttr("n", b.getI64IntegerAttr(n));
              setResult(projection,{m,n});
            }
            setResult(sub.query_reshape,{m,batch,dim});
            setResult(sub.key_reshape,{m,1,dim});
            setResult(sub.value_reshape,{m,1,dim});
            sub.query_rope->setAttr("heads",b.getI64IntegerAttr(batch));
            sub.key_rope->setAttr("heads",b.getI64IntegerAttr(1));
            setResult(sub.query_rope,{m,batch,dim});
            setResult(sub.key_rope,{m,1,dim});
            for (auto broadcast : {sub.key_broadcast, sub.value_broadcast}) {
              broadcast->setAttr("query_heads",b.getI64IntegerAttr(batch));
              broadcast->setAttr("kv_heads",b.getI64IntegerAttr(1));
              setResult(broadcast,{m,batch,dim});
            }
            for (auto transpose : {sub.query_transpose, sub.key_transpose, sub.value_transpose})
              setResult(transpose,{batch,m,dim});
            setResult(sub.qk,{batch,m,m});
            setResult(sub.softmax,{batch,m,m});
            setResult(sub.pv,{batch,m,dim});
            setResult(sub.context_transpose,{m,batch,dim});
            setResult(sub.context_reshape,{m,batch*dim});
            sub.output->setAttr("k",b.getI64IntegerAttr(batch*dim));
            llvm::SmallVector<mlir::Type> argTypes;
            for (size_t i=0;i<args.size();++i) {
              auto ty=type(&context,args[i]);fn.getArgument(i).setType(ty);argTypes.push_back(ty);
            }
            fn.setType(b.getFunctionType(argTypes,fn.getResultTypes()));
            std::string batchSource;llvm::raw_string_ostream stream(batchSource);part->print(stream);
            auto partial = add(name+".heads"+std::to_string(first),batchSource,std::move(args),outputs)[0];
            if (first == 0) { sum=std::move(partial);continue; }
            std::string ty;llvm::raw_string_ostream typeStream(ty);type(&context,outputs[0]).print(typeStream);
            const std::string addSource = "module { func.func @sum(%a: "+ty+", %b: "+ty+") -> "+ty
                +" { %r = ftlpu.kernel.vector_alu %a, %b {kind = \"add\", use_immediate = false, immediate = 0.0 : f32, reduce = false} : ("
                +ty+", "+ty+") -> "+ty+" return %r : "+ty+" } }";
            const I rowsPerSum=std::max<I>(1,(depth/2)*target.memory().bytes_per_word/width);
            std::vector<Tensor> sums;
            for (I row=0;row<m;row+=rowsPerSum) {
              const I rows=std::min(rowsPerSum,m-row);auto result=outputs[0];result.shape[0]=rows;
              sums.push_back(add(name+".head_sum"+std::to_string(first)+".rows"+std::to_string(row),addSource,
                  {sliceRows(sum,row,rows),sliceRows(partial,row,rows)},{result},rows)[0]);
            }
            sum=concat(sums);
          }
          return {sum};
        }
      }
    }
    // Retained FP32 matrices and BF16 planar readouts feed independent
    // RMSNorm rows directly, without joining producer tiles in MEM.
    // Tile the logical reduction domain here, not in model/test scripts.
    if(source.find("ftlpu.kernel.rms_norm")!=std::string::npos && input[0].shape[0]>32 &&
       !input[0].refs.empty() && llvm::all_of(input[0].refs,[&](const Ref &ref) {
         const auto &producer=nodes[ref.node];
            return producer.outputs[ref.output].dtype=="f32" ||
                (producer.outputs[ref.output].dtype=="bf16" && input[0].shape[1]>32 &&
                 text(producer.out[ref.output],"kind")=="fp16_pair_planar" &&
                 text(producer.out[ref.output],"hemisphere")=="both" &&
                 slices(producer.out[ref.output]).size()==2);
       })) {
      std::vector<Tensor> pieces;
      for(I first=0;first<input[0].shape[0];first+=32) {
        const I count=std::min<I>(32,input[0].shape[0]-first);
        auto args=input,outs=outputs;args[0]=sliceRows(input[0],first,count);outs[0].shape[0]=count;
        pieces.push_back(lower(J{{"name",name+".rows"+std::to_string(first)},{"kernel",source}},args,outs)[0]);
      }
      return {concat(pieces)};
    }
    if ((source.find("ftlpu.kernel.rms_norm") != std::string::npos
         || source.find("ftlpu.kernel.l2_norm") != std::string::npos)
        && input[0].shape[0] > 1 && input[0].shape[0] < 32) {
      // Independent normalization rows may occupy a partial hardware tile.
      // Duplicate a logical row in unused lanes; retain only live results.
      const I live = input[0].shape[0];
      if (input[0].external()) {
        // concat joins producer references, which external inputs do not
        // have. Preserve the actual ingress bytes when padding a partial
        // normalization tile instead of silently creating an empty view.
        const I width = input[0].size() / live;
        V indices = range(input[0].size());
        for (I row=live;row<32;++row) {
          const auto lastRow = range(width,(live-1)*width);
          indices.insert(indices.end(),lastRow.begin(),lastRow.end());
        }
        V shape = input[0].shape; shape[0] = 32;
        input[0]=view(input[0],indices,std::move(shape));
      } else {
        std::vector<Tensor> padded{input[0]};
        for (I row=live;row<32;++row) padded.push_back(sliceRows(input[0],live-1,1));
        input[0]=concat(padded);
      }
      auto out=outputs; out[0].shape[0]=32;
      auto values=add(name,source,input,out,32);
      return {sliceRows(values[0],0,live)};
    }
    if (source.find("ftlpu.kernel.batch_matmul") != std::string::npos)
      return add(name, source, std::move(input), std::move(outputs));
    if (source.find("ftlpu.kernel.matmul") != std::string::npos) {
      // Preserve composite decoder recognition and layout policies above.
      // A standalone ordinary dense projection enters the same planner as
      // the quantized projection's semantic adapter.
      if (parsedFunction.front().getOperations().size() == 2 && outputs.size() == 1)
        if (auto op = llvm::dyn_cast<fc::kernel::MatmulOp>(parsedFunction.front().front());
            op && llvm::isa<mlir::BlockArgument>(op.getLhs()) && llvm::isa<mlir::BlockArgument>(op.getRhs())) {
          if (mlir::failed(op.verify())) throw std::runtime_error("invalid projection signature");
          const auto arg = [](mlir::Value v) { return llvm::cast<mlir::BlockArgument>(v).getArgNumber(); };
          return {lowerProjection(name, input[arg(op.getLhs())], input[arg(op.getRhs())],
              outputs[0], op->getAttrDictionary())};
        }
      const auto paging=fc::target::GraphWeightPaging::resolve(target,policies);
      const I k=input[0].shape[1], width=input[1].shape[1];
      const I tile=target.throughput().mxm_rows;
      const I columnsPerItem=target.memory().hemispheres*tile;
      const fc::target::weight_pages::Geometry geometry{
          paging.rows,target.memory().banks_per_slice,1};
      const fc::target::weight_pages::ItemLayout layout{
          (k/tile)*target.throughput().tile_rows};
      const I items=fc::target::weight_pages::itemsPerGroup(geometry,layout);
      if(k<=0 || k%tile || items<=0)
        throw std::runtime_error("one projection reduction item exceeds weight page capacity");
      const I page=columnsPerItem*items;
      std::vector<Tensor> pieces;
      const I pageFamily=nodes.size();
      for (I first = 0; first < width; first += page) {
        I count = std::min(page, ((width-first+columnsPerItem-1)/columnsPerItem)*columnsPerItem);
        auto w = sliceColumns(input[1], first, count, true);
        const auto headPolicy = policies.getAs<mlir::BoolAttr>("ftlpu.l2_head_partition");
        const bool partitionHeads = distributedConvOutput && headPolicy && headPolicy.getValue()
            && width % 256 == 0 && count % 256 == 0;
        const bool decodeHeads = headPolicy && headPolicy.getValue() && decodeHeadWidth > 0
            && input[0].shape[0] == 1 && width % (2*decodeHeadWidth) == 0
            && count % (2*decodeHeadWidth) == 0;
        if (decodeHeads) {
          V order;
          for (I row=0;row<k;++row) for(I c=0;c<count;++c)
            order.push_back(row*count+(c/32%2)*(count/2)+(c/64)*32+c%32);
          w=view(w,order,w.shape);
        }
        if (partitionHeads) {
          V order;
          for (I row = 0; row < k; ++row)
            for (I c = 0; c < count; ++c) {
              const I word = c / 32, localWord = word / 2;
              const I logical = ((localWord / 4) * 2 + word % 2) * 128
                  + (localWord % 4) * 32 + c % 32;
              order.push_back(row * count + logical);
            }
          w = view(w, order, w.shape);
        }
        auto o = outputs[0];
        o.shape[1] = count;
        auto values = add(name + ".page" + std::to_string(first / page), source,
                          {input[0], w}, {o}, -1, count);
        nodes.back().projectionPageFamily=pageFamily;
        auto value = values[0];
        if (decodeHeads) {
          V order;
          for(I row=0;row<value.shape[0];++row) for(I c=0;c<count;++c)
            order.push_back(row*count+(c%(count/2)/32)*64+(c/(count/2))*32+c%32);
          value=view(value,order,value.shape);
        }
        if (partitionHeads) {
          V order;
          for (I row = 0; row < value.shape[0]; ++row)
            for (I c = 0; c < count; ++c) {
              const I head = c / 128, within = c % 128;
              order.push_back(row * count + (((head / 2) * 4 + within / 32) * 2
                  + head % 2) * 32 + within % 32);
            }
          value = view(value, order, value.shape);
        }
        pieces.push_back(sliceColumns(value, 0, std::min(count, width - first)));
      }
      return {concat(pieces, 1)};
    }
    if (source.find("ftlpu.kernel.depthwise_conv1d") != std::string::npos) {
      I width = input[0].shape[1], tile = std::min<I>(1024, width);
      // The native Conv lowering reserves both input and causal scratch in
      // the Activation arena before assigning their physical slice homes.
      // Bound the channel tile by that arena as the token count grows.
      const auto activationRows = [](I bytes) {
        return ((bytes + 319) / 320 - 1) * 16 + 1;
      };
      while (tile > 32 &&
          activationRows(input[0].shape[0] * tile * 2) +
          activationRows((input[0].shape[0] + input[1].shape[1] - 1) * tile * 2)
              > target.memory().words_per_bank)
        tile /= 2;
      // Short decode windows fit all channel heads in one bilateral domain.
      // The 1024-channel prefill split otherwise restages activation and
      // coefficients between heads even though SRAM capacity is available.
      if (input[0].shape[0] <= 4 && width % 64 == 0) {
        const I words = target.memory().sram_depth_rows / 2;
        const I capacity = 32 * std::min(words / (input[0].shape[0]+3), words / 4);
        if (width/2 <= capacity) tile=width/2;
      }
      const auto headPolicy = policies.getAs<mlir::BoolAttr>("ftlpu.l2_head_partition");
      // A decode window contains history from an ingress binding and one
      // fresh row from a batched projection. Partition by the actual MXM
      // page/hemisphere, not by contiguous halves of the entire output.
      // Keep identical tiles with overlap off for a fair scheduling A/B.
      if (input[0].shape[0] == 4 && !input[0].external() && width % 64 == 0) {
        std::map<I,std::array<V,2>> pages;
        bool eligible=true;
        for(I c=0;c<width;c+=32) {
          auto ref=input[0].refs[3*width+c];
          const auto& producer=nodes[ref.node];
          if(producer.projectionPageFamily<0 || ref.output!=0 ||
             producer.outputs[0].shape[0]<32 || ref.index%32) {eligible=false;break;}
          for(I lane=0;lane<32;++lane) {
            const auto r=input[0].refs[3*width+c+lane];
            if(r.node!=ref.node || r.output!=0 || r.index!=ref.index+lane)eligible=false;
            pages[ref.node][(ref.index%producer.outputs[0].shape[1]/32)%2].push_back(c+lane);
          }
        }
        for(const auto& page:pages)
          eligible &= !page.second[0].empty() && page.second[0].size()==page.second[1].size() &&
                      I(page.second[0].size())<=tile;
        if(eligible && pages.size()>=2) {
          std::vector<Tensor> pieces;V inverse(width);I offset=0;
          for(const auto& page:pages) {
            const I group=nodes.size();
            for(I part=0;part<2;++part) {
              const V& channels=page.second[1-part];
              const I n=channels.size();V ai,wi;
              for(I row=0;row<4;++row)for(I c:channels)ai.push_back(row*width+c);
              for(I c:channels)for(I t=0;t<input[1].shape[1];++t)wi.push_back(c*input[1].shape[1]+t);
              auto o=outputs[0];o.shape[1]=n;
              pieces.push_back(add(name+".block"+std::to_string(offset),source,
                  {view(input[0],ai,{4,n}),view(input[1],wi,{n,input[1].shape[1]})},
                  {o},-1,n,part!=0,group)[0]);
              nodes.back().function->setAttr("ftlpu.depthwise_conv_paired",mlir::BoolAttr::get(&context,true));
              for(I c:channels)inverse[c]=offset++;
            }
          }
          auto joined=concat(pieces,1);V order;
          for(I r=0;r<4;++r)for(I c:inverse)order.push_back(r*width+c);
          return {view(joined,order,joined.shape)};
        }
      }
      if (headPolicy && headPolicy.getValue() && decodeHeadWidth>0
          && width==decodeConvWidth && input[0].shape[0]<=4 && !input[0].external()) {
        // The current token's projection references determine channel ownership;
        // apply the identical permutation to Conv history and coefficients.
        V channels[2];
        bool eligible=true;
        for(I c=0;c<width;++c) {
          const auto ref=input[0].refs[(input[0].shape[0]-1)*width+c];
          // An external QKV input is a view/join node, not an MXM producer.
          // Such nodes have no native output slots to inspect.
          if (ref.output >= I(nodes[ref.node].outputSlots.size())) { eligible=false;break; }
          auto projection=llvm::dyn_cast_or_null<fc::tensor::ProjectionTaskOp>(nodes[ref.node].outputSlots[ref.output].op);
          if(!projection || projection.getKind()!="linear") { eligible=false;break; }
          channels[(ref.index/32)%2].push_back(c);
        }
        if(eligible && I(channels[0].size())==tile && I(channels[1].size())==tile) {
          const I group=nodes.size();std::vector<Tensor> halves;V inverse(width);
          for(I part=0;part<2;++part) {
            const I side=1-part;V ai,wi;
            for(I r=0;r<input[0].shape[0];++r) for(I c:channels[side]) ai.push_back(r*width+c);
            for(I c:channels[side]) for(I tap=0;tap<input[1].shape[1];++tap) wi.push_back(c*input[1].shape[1]+tap);
            auto o=outputs[0];o.shape[1]=tile;
            halves.push_back(add(name+".block"+std::to_string(part*tile),source,
                {view(input[0],ai,{input[0].shape[0],tile}),view(input[1],wi,{tile,input[1].shape[1]})},
                {o},-1,tile,part!=0,group)[0]);
            nodes.back().function->setAttr("ftlpu.depthwise_conv_paired",mlir::BoolAttr::get(&context,true));
            for(I c=0;c<tile;++c) inverse[channels[side][c]]=part*tile+c;
          }
          auto joined=concat(halves,1);V order;
          for(I r=0;r<joined.shape[0];++r) for(I c:inverse) order.push_back(r*width+c);
          return {view(joined,order,joined.shape)};
        }
      }
      const bool externalDecodeHeads = headPolicy && headPolicy.getValue()
          && decodeHeadWidth > 0 && width == decodeConvWidth
          && input[0].shape[0] <= 4 && width % (2*decodeHeadWidth) == 0;
      const I channelBlock = externalDecodeHeads ? decodeHeadWidth
          : distributedConvOutput && headPolicy && headPolicy.getValue()
          && width % 256 == 0 ? 128 : 32;
      std::vector<Tensor> groups;
      // Keep each hemisphere's native projection words together. These are
      // logical views only: neither splitting nor restoring channel order
      // emits a MEM copy.
      if ((externalDecodeHeads || (distributedConvOutput || input[0].shape[0] >= 32)
          && !input[0].external() && width % (2 * tile) == 0
          && llvm::all_of(input[0].refs, [&](const auto &ref) {
            if (ref.output >= I(nodes[ref.node].outputSlots.size())) return false;
            auto *owner = nodes[ref.node].outputSlots[ref.output].op;
            auto projection = llvm::dyn_cast_or_null<fc::tensor::ProjectionTaskOp>(owner);
            return projection && projection.getKind() == "linear";
          })) && width % (2 * tile) == 0) {
        for (I first = 0; first < width; first += 2 * tile) {
          std::vector<Tensor> halves;
          const I group = nodes.size();
          for (I side = 0; side < 2; ++side) {
            V ai, wi;
            const I parity = 1 - side; // primary chain reads West, mirror East
            for (I r = 0; r < input[0].shape[0]; ++r)
              for (I c = 0; c < tile; ++c)
                ai.push_back(r * width + first + (2 * (c / channelBlock) + parity) * channelBlock + c % channelBlock);
            for (I c = 0; c < tile; ++c)
              for (I tap = 0; tap < input[1].shape[1]; ++tap)
                wi.push_back((first + (2 * (c / channelBlock) + parity) * channelBlock + c % channelBlock) * input[1].shape[1] + tap);
            auto o = outputs[0]; o.shape[1] = tile;
            halves.push_back(add(name + ".block" + std::to_string(first + side * tile), source,
                {view(input[0], ai, {input[0].shape[0], tile}),
                 view(input[1], wi, {tile, input[1].shape[1]})}, {o}, -1, tile, side != 0, group)[0]);
            nodes.back().function->setAttr("ftlpu.depthwise_conv_paired", mlir::BoolAttr::get(&context,true));
          }
          auto joined = concat(halves, 1);
          V order;
          for (I r = 0; r < joined.shape[0]; ++r)
            for (I c = 0; c < 2 * tile; ++c)
              order.push_back(r * 2 * tile + (1 - (c / channelBlock) % 2) * tile + (c / (2 * channelBlock)) * channelBlock + c % channelBlock);
          groups.push_back(view(joined, order, joined.shape));
        }
        return {concat(groups, 1)};
      }
      for (I first = 0; first < width; first += tile * 2) {
        std::vector<Tensor> pieces;
        I group = nodes.size();
        for (I col = first; col < std::min(width, first + 2 * tile);
             col += tile) {
          I count = std::min(tile, width - col);
          auto a = sliceColumns(input[0], col, count),
               w = sliceRows(input[1], col, count), o = outputs[0];
          o.shape[1] = count;
          pieces.push_back(add(name + ".block" + std::to_string(col), source,
                               {a, w}, {o}, -1, count, col != first, group)[0]);
          nodes.back().function->setAttr("ftlpu.depthwise_conv_paired",
              mlir::BoolAttr::get(&context,width-first>=2*tile));
        }
        auto groupedResult = concat(pieces, 1);
        // Keep short decode outputs as producer views. Eagerly joining each
        // channel group copies a completed head before the next can execute.
        // Consumers and final export materialize only the words they need.
        groups.push_back((distributedConvOutput || input[0].shape[0] <= 4)
                             ? std::move(groupedResult)
                             : join(std::move(groupedResult),
                                    "conv_layout_group" + std::to_string(first)));
      }
      return {concat(groups, 1)};
    }
    if (!patterns.delta_nets.empty()) {
      auto function = *parsed->getOps<mlir::func::FuncOp>().begin();
      auto delta=patterns.delta_nets.front().recurrence;
      const I heads=delta.getHeads();
      auto strategy=function->getAttrOfType<mlir::StringAttr>("ftlpu.gdn_prefill_strategy");
      if(!strategy)strategy=policies.getAs<mlir::StringAttr>("ftlpu.gdn_prefill_strategy");
      if(strategy && strategy.getValue()!="tokenwise" && strategy.getValue()!="chunkwise")
        throw std::runtime_error("gdn_prefill_strategy must be tokenwise or chunkwise");
      if(strategy && strategy.getValue()=="tokenwise")
        return lowerTokenwiseDelta(name,input,outputs,heads,delta.getKeyDim(),delta.getValueDim());
      if(input[5].dtype=="bf16")
        throw std::runtime_error("BF16 recurrent State storage currently requires explicit tokenwise strategy");
      const I requestedChunk=integer(delta,"chunk_size",integer(function,"ftlpu.gdn_chunk_size"));
      if (input[0].dtype=="bf16" && delta.getKeyDim()==128 && delta.getValueDim()==128 &&
          (!requestedChunk || requestedChunk==32) &&
          input[0].shape[0]>=32 && input[0].shape[0]%32==0) {
        delta->setAttr("native_chunkwise_bf16",mlir::BoolAttr::get(&context,true));
        std::string nativeSource;llvm::raw_string_ostream nativeStream(nativeSource);parsed->print(nativeStream);
        std::vector<Tensor> headOutputs,headStates;
        for(I h=0;h<heads;++h){
          std::vector<Tensor> args={sliceColumns(input[0],h*128,128),sliceColumns(input[1],h*128,128),
              sliceColumns(input[2],h*128,128),sliceColumns(input[3],h,1),sliceColumns(input[4],h,1),sliceRows(input[5],h*128,128)};
          auto out=outputs;out[0].shape={input[0].shape[0],128};out[1].shape={128,128};
          auto values=add(name+".chunkwise.head"+std::to_string(h),nativeSource,args,out);
          headOutputs.push_back(values[0]);headStates.push_back(values[1]);
        }
        return {concat(headOutputs,1),concat(headStates)};
      }
      if (const I chunk = integer(patterns.delta_nets.front().recurrence,
              "chunk_size", integer(function, "ftlpu.gdn_chunk_size",strategy?32:0)))
        return lowerChunkwiseDelta(name, input, outputs,
            patterns.delta_nets.front().recurrence.getHeads(), chunk);
      std::vector<Tensor> pieces;
      Tensor state = input[5];
      // Convert a bounded token block once, then retain its native operands
      // throughout the serial state scan. Per-token graph nodes duplicate
      // packed-to-broadcast SXM programs even though their views are affine.
      // Q/K scalar homes have one row per feature and token.
      I blockRows = std::max<I>(1, std::min<I>(4,
          target.memory().words_per_bank / input[0].shape[1]));
      // Packed gate producers partition tokens between hemispheres. A
      // block must not span that boundary (FP32 unpack remains local).
      const I halfRows = (input[0].shape[0] + 1) / 2;
      while (blockRows > 1 && halfRows % blockRows) --blockRows;
      for (I row = 0; row < input[0].shape[0]; row += blockRows) {
        const I rows = std::min(blockRows, input[0].shape[0] - row);
        std::vector<Tensor> args;
        for (size_t i = 0; i < 5; ++i)
          args.push_back(sliceRows(input[i], row, rows));
        args.push_back(state);
        auto o = outputs[0];
        o.shape[0] = rows;
        auto values = add(name + ".token" + std::to_string(row), source, args,
                          {o, outputs[1]}, rows);
        pieces.push_back(values[0]);
        state = values[1];
      }
      return {join(concat(pieces), "recurrent_output_layout"), state};
    }
    // Independent packed token rows use mirrored VXM groups. Broadcast
    // constants remain whole; no model-specific operation is introduced.
    const I tokenRows = input.empty() || input[0].shape.size() != 2 ? 0 : input[0].shape[0];
    auto vectorFunction = *parsed->getOps<mlir::func::FuncOp>().begin();
    const bool pointwiseGraph = llvm::all_of(vectorFunction.front(), [](mlir::Operation &op) {
      if (auto vector = llvm::dyn_cast<fc::kernel::VectorAluOp>(op))
        return !vector.getReduce() && vector.getKind()!="cumsum_rows"
            && vector.getKind()!="causal_exp_difference" && vector.getKind()!="weighted_gram" && vector.getKind()!="cross_gram" && vector.getKind()!="unit_lower_solve" && vector.getKind()!="unit_lower_solve_blocked" && vector.getKind()!="head_scale" && vector.getKind()!="unit_lower_inverse" && vector.getKind()!="head_matmul" && vector.getKind()!="head_last_row_scale_transpose" && vector.getKind()!="head_cross_product";
      return llvm::isa<mlir::func::ReturnOp>(op);
    });
    if (pointwiseGraph && tokenRows > 1 &&
        llvm::all_of(input,[&](const Tensor &v) {return v.shape.size()==2 &&
            (v.shape[0]==1 || v.shape[0]==tokenRows);})) {
      const I words=(input[0].shape[1]+target.memory().bytes_per_word-1)/target.memory().bytes_per_word;
      const I capacity=std::max<I>(1,target.memory().sram_depth_rows/(2*words));
      if (tokenRows>capacity) {
        std::vector<std::vector<Tensor>> pieces(outputs.size());
        const I group=nodes.size();
        for (I first=0;first<tokenRows;first+=capacity) {
          const I count=std::min(capacity,tokenRows-first);
          auto args=input, outs=outputs;
          for (auto &a : args) if (a.shape[0] != 1) a=sliceRows(a,first,count);
          for (auto &o : outs) o.shape[0]=count;
          auto values=add(name+".rows"+std::to_string(first),source,args,outs,count);
          nodes.back().pointwiseGroup=group;
          nodes.back().pointwiseRow=first;
          nodes.back().pointwiseShape=outputs[0].shape;
          for (size_t i=0;i<values.size();++i) pieces[i].push_back(values[i]);
        }
        std::vector<Tensor> result;
        for (auto &p : pieces) result.push_back(concat(p));
        return result;
      }
    }
    if (pointwiseGraph && source.find("ftlpu.vector_bf16_spill") == std::string::npos
        && target.throughput().vxm_alus == 16
        && target.memory().hemispheres == 2
        && source.find("ftlpu.kernel.vector_alu") != std::string::npos
        && source.find("reduce = true") == std::string::npos && tokenRows >= 2
        && input[0].shape[1] <= target.memory().bytes_per_word
        && llvm::all_of(input, [&](const Tensor &v) {
             return v.shape.size() == 2 && (v.shape[0] == 1 || v.shape[0] == tokenRows);
           })
        && llvm::all_of(outputs, [&](const Tensor &v) {
             return v.shape.size() == 2 && v.shape[0] == tokenRows;
           })) {
      std::vector<std::vector<Tensor>> pieces(outputs.size());
      // Each mirrored pair must fit the physical bank, including the
      // retained outputs and conversion scratch reserved in its lower half.
      const I perSide = std::min<I>((tokenRows+1)/2,
                                   target.memory().sram_depth_rows/2);
      for (I base=0; base<tokenRows; base+=2*perSide) {
        const I pairRows=std::min<I>(2*perSide,tokenRows-base);
        const I half=(pairRows+1)/2, group=nodes.size();
        for (I side=0; side<2; ++side) {
          const I first=base+side*half;
          const I count=std::min<I>(half,base+pairRows-first);
          if (!count) continue;
          auto args = input, outs = outputs;
          for (auto &a : args) if (a.shape[0] != 1) a = sliceRows(a, first, count);
          for (auto &o : outs) o.shape[0] = count;
          auto values = add(name + ".tokens" + std::to_string(first), source,
                            args, outs, count, -1, side != 0, group);
          for (size_t i = 0; i < values.size(); ++i) pieces[i].push_back(values[i]);
        }
      }
      std::vector<Tensor> result;
      for (auto &p : pieces) result.push_back(concat(p));
      return result;
    }
    // Scalar broadcast is needed for row reductions. Decide tile geometry
    // from SRAM capacity here, rather than in the graph importer.
    if (source.find("reduce = true") != std::string::npos &&
        input[0].size() > target.memory().sram_depth_rows / 2) {
      I tile = std::max<I>(1, (target.memory().sram_depth_rows / 2) /
                                  input[0].shape[1]);
      std::vector<std::vector<Tensor>> pieces(outputs.size());
      for (I first = 0; first < input[0].shape[0]; first += tile) {
        I count = std::min(tile, input[0].shape[0] - first);
        std::vector<Tensor> args, outs = outputs;
        for (auto &a : input)
          args.push_back(sliceRows(a, first, count));
        for (auto &o : outs)
          if (o.shape[0] > 1)
            o.shape[0] = count;
        auto values = add(name + ".rows" + std::to_string(first), source, args,
                          outs, count);
        nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
        for (size_t i = 0; i < values.size(); ++i)
          pieces[i].push_back(values[i]);
      }
      std::vector<Tensor> result;
      for (auto &p : pieces)
        result.push_back(concat(p));
      return result;
    }
    return add(name, source, std::move(input), std::move(outputs));
  }
};
struct Domain {
  std::array<I, 3> count, src, dst;
};
bool affine(const V &src, const V &dst, Domain &d) {
  I n = src.size();
  if (!n || dst.size() != src.size())
    throw std::runtime_error("empty or inconsistent copy view");
  V divisors;
  for (I i = 1; i <= n; ++i)
    if (n % i == 0)
      divisors.push_back(i);
  for (auto inner = divisors.rbegin(); inner != divisors.rend(); ++inner)
    for (I wave : divisors) {
      if (n % (*inner * wave))
        continue;
      I group = n / (*inner * wave);
      d.count = {*inner, wave, group};
      bool valid = true;
      for (I side = 0; side < 2 && valid; ++side) {
        auto &v = side ? dst : src;
        auto &s = side ? d.dst : d.src;
        s = {*inner > 1 ? v[1] - v[0] : 0, wave > 1 ? v[*inner] - v[0] : 0,
             group > 1 ? v[*inner * wave] - v[0] : 0};
        for (I i = 0; i < n; ++i)
          if (v[i] != v[0] + (i % *inner) * s[0] + (i / *inner % wave) * s[1] +
                          (i / (*inner * wave)) * s[2]) {
            valid = false;
            break;
          }
      }
      if (valid)
        return true;
    }
  return false;
}
#include "ProjectionSolveOverlap.inc"
#include "ProjectionConvOverlap.inc"
#include "ProjectionVectorOverlap.inc"

class Scheduler {
public:
  mlir::MLIRContext &context;
  const fc::target::LPUTargetModel &target;
  mlir::OpBuilder b;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  mlir::Location loc;
  mlir::DictionaryAttr maskPlacement;
  I cursor = 64, internal = 1000, bridges = 0, period, group;
  llvm::json::Array transportDomains;
  std::string transportPhase = "initialization";
  std::string transportReason = "initialization_layout";
  bool consumerMulticast = false;
  Scheduler(mlir::MLIRContext &c, const fc::target::LPUTargetModel &t)
      : context(c), target(t), b(&c), loc(b.getUnknownLoc()) {
    period = target.memory().bytes_per_word /
             target.throughput().mem_read_bytes_per_cycle;
    group = target.streams().mem_slices_per_register_group;
    module = mlir::ModuleOp::create(loc);
    (*module)->setAttr("ftlpu.target", target.to_attribute(&context));
    function = mlir::func::FuncOp::create(
        loc, "resident",
        b.getFunctionType(
            {mlir::RankedTensorType::get({1, 32}, b.getF32Type())}, {}));
    module->push_back(function);
    b.setInsertionPointToEnd(function.addEntryBlock());
  }
  mlir::Attribute attr(const llvm::json::Value &v) {
    if (auto n = v.getAsInteger())
      return b.getI64IntegerAttr(*n);
    if (auto n = v.getAsBoolean())
      return b.getBoolAttr(*n);
    if (auto n = v.getAsNumber())
      return b.getF32FloatAttr(*n);
    if (auto s = v.getAsString())
      return b.getStringAttr(*s);
    if (auto a = v.getAsArray()) {
      llvm::SmallVector<mlir::Attribute> attrs;
      for (auto &x : *a)
        attrs.push_back(attr(x));
      return b.getArrayAttr(attrs);
    }
    throw std::runtime_error("unsupported schedule attribute");
  }
  mlir::DictionaryAttr dict(const J &j) {
    mlir::NamedAttrList a;
    for (auto &kv : j)
      a.set(kv.first,
            llvm::StringRef(kv.first).ends_with("immediate")
                ? mlir::Attribute(b.getF32FloatAttr(*kv.second.getAsNumber()))
                : attr(kv.second));
    return a.getDictionary(&context);
  }
  mlir::DictionaryAttr change(mlir::DictionaryAttr p, const J &j) {
    mlir::NamedAttrList a(p);
    for (auto &kv : j)
      a.set(kv.first, attr(kv.second));
    return a.getDictionary(&context);
  }
  llvm::json::Array array(const V &v) {
    llvm::json::Array a;
    for (I x : v)
      a.push_back(x);
    return a;
  }
  mlir::DictionaryAttr placement(std::string kind, V ids, std::string h, I bank,
                                 I row, I count = 0) {
    return dict(J{{"kind", kind},
                  {"slices", array(ids)},
                  {"hemisphere", h},
                  {"bank", bank},
                  {"base_row", row},
                  {"instruction_count", count},
                  {"address_stride", 1}});
  }
  mlir::Operation *emit(llvm::StringRef name, J attrs) {
    mlir::OperationState s(loc, ("ftlpu.schedule." + name).str());
    s.addAttributes(dict(attrs).getValue());
    if (name == "vxm") {
      s.addOperands({function.getArgument(0), function.getArgument(0)});
      s.addTypes(function.getArgument(0).getType());
    }
    return b.create(s);
  }
  mlir::Value binding(const Tensor &t, mlir::DictionaryAttr p,
                      std::string access, I index, std::string name,
                      mlir::Value input = {}) {
    mlir::OperationState s(loc, "ftlpu.schedule.binding");
    if (input)
      s.addOperands(input);
    s.addTypes(type(&context, t));
    s.addAttributes(
        dict(J{{"access", access},
               {"index", index},
               {"role",
                access == "input"
                    ? (p.getAs<mlir::BoolAttr>("paged_weight") ? "weight"
                                                               : "activation")
                    : "result"},
               {"name", name},
               {"bytes", t.size() * t.bytes()}})
            .getValue());
    s.addAttribute("placement", p);
    return b.create(s)->getResult(0);
  }
  void set(mlir::Operation *o, llvm::StringRef name, I n) {
    o->setAttr(name, b.getI64IntegerAttr(n));
  }
  void mem(I cycle, mlir::DictionaryAttr p, I plane, std::string opcode,
           I address, I stream, std::array<I, 3> count = {1, 1, 1},
           std::array<I, 3> stride = {0, 0, 0},
           std::array<I, 3> interval = {1, 1, 1}) {
    emit("mem_transfer", J{{"cycle", cycle},
                           {"hemisphere", I(text(p, "hemisphere") == "west")},
                           {"slice", slices(p).at(plane)},
                           {"bank", field(p, "bank")},
                           {"opcode", opcode},
                           {"address", address},
                           {"packed_stream", stream},
                           {"repeat_count", count[0]},
                           {"repeat_interval", interval[0]},
                           {"address_stride", stride[0]},
                           {"wave_count", count[1]},
                           {"wave_interval", interval[1]},
                           {"wave_address_stride", stride[1]},
                           {"group_count", count[2]},
                           {"group_interval", interval[2]},
                           {"group_address_stride", stride[2]}});
  }
  void vxm(I cycle, I queue = 0, std::string kind = "cast",
           std::string lhs = "stream_f32", I li = 0,
           std::string rhs = "immediate", I ri = 0, std::string source = "west",
           std::string rhsSource = "west", std::string output = "west",
           std::string format = "fp32", I out = -1, I count = 1,
           I interval = 0) {
    J a{{"cycle", cycle},
        {"queue", queue},
        {"opcode", kind},
        {"chain_depth", 8},
        {"lhs_kind", lhs},
        {"lhs_index", li},
        {"lhs_immediate", 0.0},
        {"rhs_kind", rhs},
        {"rhs_index", ri},
        {"rhs_immediate", 0.0},
        {"cast_target", format},
        {"output_stream", out},
        {"repeat_count", count},
        {"repeat_interval", 1},
        {"input_hemisphere", "east"},
        {"output_hemisphere", output == "west" ? "east" : "west"}};
    if (interval) {
      a["repeat_count"] = 1;
      a["wave_count"] = count;
      a["wave_interval"] = interval;
    }
    if (lhs.find("stream") == 0)
      a["lhs_stream_source"] = source;
    if (rhs.find("stream") == 0)
      a["rhs_stream_source"] = rhsSource;
    emit("vxm", std::move(a));
  }
  void sxm(I cycle, std::string h, std::string opcode, V sources,
           V destinations, V map, I count = 1, I interval = 1, I row = -1,
           I waves = 1, I waveInterval = 1, I mapStride = 0) {
    J a{{"cycle", cycle},
        {"hemisphere", I(h == "west")},
        {"opcode", opcode},
        {"source_streams", array(sources)},
        {"destination_streams", array(destinations)},
        {"permute_map", array(map)},
        {"weight_layout", "resident_layout_bridge"},
        {"repeat_count", count},
        {"repeat_interval", interval},
        {"wave_count", waves},
        {"wave_interval", waveInterval}};
    if (row >= 0)
      a["output_row"] = row;
    if (mapStride)
      a["permute_map_stride"] = mapStride;
    if (opcode == "transpose" && sources.size() == 2)
      a["input_row"] = 9;
    emit("sxm", std::move(a));
  }
  I bridgeQueue() {
    I result = std::array<I, 4>{4, 6, 12, 14}[bridges % 4];
    ++bridges;
    return result;
  }
  I bridgeResult(I beat, std::string source, std::string dest, I base) {
    I latency = dest == "east" ? (source == "west" ? 19 : 12)
                               : (source == "west" ? 24 : 17);
    if (source == dest)
      latency += (base - 4) * 2 * (source == "east" ? 1 : -1);
    return beat + latency;
  }
  bool sameCopyFormat(mlir::DictionaryAttr src, mlir::DictionaryAttr dst) {
    const auto bf16Pair = [](mlir::DictionaryAttr p) {
      const auto kind = text(p, "kind");
      return kind == "fp16_hemisphere_pair_planar" ||
             kind == "fp16_pair_planar" ||
             kind == "fp16_mxm_distributed_16" ||
            kind == "fp16_mxm_block4_partition_distributed_16" ||
             kind == "fp16_vxm_distributed_16" ||
             kind == "fp16_scalar_broadcast_planar" ||
             kind == "fp32_quad_planar" ||
             kind == "fp32_scalar_broadcast_planar";
    };
    const auto ss = slices(src), ds = slices(dst);
    return bf16Pair(src) && bf16Pair(dst) &&
        ((ss.size()==2 && ds.size()==2) || (ss.size()==4 && ds.size()==4 &&
         text(src,"kind").find("fp32_")==0 && text(dst,"kind").find("fp32_")==0));
  }
  bool directPassiveCopy(mlir::DictionaryAttr src, mlir::DictionaryAttr dst) {
    if (!sameCopyFormat(src,dst)) return false;
    const auto ss=slices(src), ds=slices(dst);
    return text(src,"hemisphere")!=text(dst,"hemisphere") ||
        (ss[0]/group!=ds[0]/group && ss[1]/group!=ds[1]/group);
  }
  void scalarDomain(mlir::DictionaryAttr src, mlir::DictionaryAttr dst, I a,
                    I z, const Domain &d, bool tapLocal = false) {
    I beat = cursor + 32, q = bridgeQueue(),
      n = product(V(d.count.begin(), d.count.end()));
    auto ss = slices(src), ds = slices(dst);
    std::array<I, 3> intervals = {period, d.count[0] * period,
                                  d.count[0] * d.count[1] * period};
    if (directPassiveCopy(src,dst)) {
      // MEM pipelines consecutive vectors with one-cycle issue spacing;
      // superlane skew is a drain latency, not a four-cycle initiation
      // interval. Native Conv/projection domains use the same contract.
      const std::array<I,3> copyIntervals={1,d.count[0],d.count[0]*d.count[1]};
      I maxLatency = 0;
      for (I p = 0; p < I(ss.size()); ++p) {
        I sourceGroup = ss[p] / group, destGroup = ds[p] / group;
        I readStream = 20 + p, writeStream = 20 + p, latency = 0;
        if (text(src, "hemisphere") != text(dst, "hemisphere")) {
          readStream += target.streams().streams_per_direction;
          // Passive SR crosses at the VXM input boundary, without traversing
          // the ALU chain. Include the two MEM/bridge boundary cycles only.
          latency = sourceGroup + destGroup + 2;
        } else if (sourceGroup < destGroup) {
          latency = destGroup - sourceGroup;
        } else if (sourceGroup > destGroup) {
          readStream += target.streams().streams_per_direction;
          writeStream += target.streams().streams_per_direction;
          latency = sourceGroup - destGroup;
        }
        mem(beat, src, p, "read", field(src, "base_row") + a,
            readStream, d.count, d.src, copyIntervals);
        mem(beat + latency, dst, p, "write",
            field(dst, "base_row") + z, writeStream, d.count, d.dst,
            copyIntervals);
        if (tapLocal) {
          auto near = change(dst,J{{"hemisphere",text(src,"hemisphere")}});
          mem(beat + sourceGroup - destGroup, near, p, "write_tap",
              field(near,"base_row") + z, readStream, d.count, d.dst, copyIntervals);
        }
        maxLatency = std::max(maxLatency, latency);
      }
      // A passive copy has no ALU pipeline to drain. Its final word is
      // visible after the route latency and the last superlane's skew.
      // The next domain already reserves its own 32-cycle issue lead-in.
      cursor = beat + (n-1) + maxLatency
          + target.throughput().tile_rows;
      transportDomains.push_back(J{{"phase",transportPhase},{"reason",transportReason},{"route",tapLocal?"passive_sr_multicast":"passive_sr"},
          {"local_tap",tapLocal},
          {"start",beat},{"end",cursor},{"vectors",n},{"byte_planes",I(ss.size())},
          {"source_side",text(src,"hemisphere")},{"source_slice",ss[0]},
          {"source_bank",field(src,"bank")},{"source_row",field(src,"base_row")+a},
          {"destination_side",text(dst,"hemisphere")},{"destination_slice",ds[0]},
          {"destination_bank",field(dst,"bank")},{"destination_row",field(dst,"base_row")+z}});
      return;
    }
    if(sameCopyFormat(src,dst)) {
      // A local same-column copy cannot tap its own newly emitted SR.
      // Route unchanged byte planes through the opposite hemisphere instead
      // of using legacy arbitrary-SR/FP32 VXM casts.
      const auto from=text(src,"hemisphere");
      auto temp=placement(ss.size()==4?"fp32_quad_planar":"fp16_pair_planar",
          range(ss.size(),48),from=="east"?"west":"east",0,0,n);
      Domain first=d;first.dst={1,d.count[0],d.count[0]*d.count[1]};
      scalarDomain(src,temp,a,0,first);
      Domain second=d;second.src=first.dst;
      scalarDomain(temp,dst,0,z,second);
      return;
    }
    for (I p = 0; p < I(ss.size()); ++p)
      mem(beat - (ss[p] / group + 1), src, p, "read",
          field(src, "base_row") + a, 32 + p, d.count, d.src, intervals);
    auto from = text(src, "hemisphere"), to = text(dst, "hemisphere");
    vxm(beat - 1, q, "cast", ss.size() == 4 ? "stream_f32" : "stream_bf16", 0,
        "immediate", 0, from, "west", to, "fp32", -1, n);
    vxm(beat - 1, q + 1, "cast", "previous", 0, "immediate", 0, from, "west",
        to, ds.size() == 4 ? "fp32" : "bf16", 24, n);
    I result = bridgeResult(beat, from, to, q);
    for (I p = 0; p < I(ds.size()); ++p)
      mem(result + ds[p] / group + 1, dst, p, "write",
          field(dst, "base_row") + z, 24 + p, d.count, d.dst, intervals);
    cursor = result + n * period + 96;
    transportDomains.push_back(J{{"phase",transportPhase},{"reason",transportReason},{"route","vxm_cast"},
        {"start",beat},{"end",cursor},{"vectors",n},{"byte_planes",I(ss.size())},
        {"source_side",text(src,"hemisphere")},{"source_slice",ss[0]},
        {"destination_side",text(dst,"hemisphere")},{"destination_slice",ds[0]}});
  }
  void copy(mlir::DictionaryAttr src, mlir::DictionaryAttr dst,
            const V &indices, const V &offsets) {
    if (text(dst,"hemisphere") == "both") {
      for (auto h : {"east","west"})
        copy(src, change(dst,J{{"hemisphere",h}}), indices, offsets);
      return;
    }
    if (text(src,"hemisphere") == "both")
      src = change(src,J{{"hemisphere",text(dst,"hemisphere")}});
    if (src==dst && indices==offsets) return;
    if (overlap(src, dst)) {
      auto temp = change(
          src,
          J{{"hemisphere", text(src, "hemisphere") == "west" ? "east" : "west"},
            {"slices", array(range(slices(src).size(), 48))},
            {"base_row", 0},
            {"bank", text(src, "hemisphere") == "west" ? 0 : 1},
            {"instruction_count", I(indices.size())}});
      copy(src, temp, indices, range(indices.size()));
      copy(temp, dst, range(indices.size()), offsets);
      return;
    }
    Domain d;
    if (affine(indices, offsets, d)) {
      scalarDomain(src, dst, indices[0], offsets[0], d);
      return;
    }
    const bool batchRows=directPassiveCopy(src,dst);
    struct CopyRegion { I source, destination; Domain domain; };
    const auto regions=[&](const V &reads,const V &writes) {
      std::vector<CopyRegion> result;
      I pos=0,n=reads.size();
      while(pos<n) {
        I a=pos+1<n?reads[pos+1]-reads[pos]:0;
        I z=pos+1<n?writes[pos+1]-writes[pos]:0,end=pos+1;
        while(end<n && reads[end]-reads[end-1]==a && writes[end]-writes[end-1]==z)++end;
        const I width=end-pos;
        Domain domain{{width,1,1},{a,0,0},{z,0,0}};
        // Preserve complete affine rows, including partially populated pages.
        constexpr I maxCount=I{1}<<16,maxInterval=(I{1}<<24)-1;
        const auto fits=[](I stride){return stride>=-(I{1}<<19) && stride<(I{1}<<19);};
        if(batchRows && width>1 && width<=maxCount && width<=n-end && fits(a) && fits(z)) {
          const I readRow=reads[end]-reads[pos],writeRow=writes[end]-writes[pos];
          const I limit=std::min({maxCount,maxInterval/width,(n-pos)/width});
          if(fits(readRow) && fits(writeRow)) {
            while(domain.count[1]<limit && reads[end]-reads[end-width]==readRow &&
                  writes[end]-writes[end-width]==writeRow) {
              I column=1;
              while(column<width && reads[end+column]-reads[end+column-1]==a &&
                    writes[end+column]-writes[end+column-1]==z)++column;
              if(column!=width)break;
              ++domain.count[1];end+=width;
            }
            if(domain.count[1]>1) {domain.src[1]=readRow;domain.dst[1]=writeRow;}
          }
        }
        result.push_back({reads[pos],writes[pos],domain});pos=end;
      }
      return result;
    };
    auto best=regions(indices,offsets);
    // These physical queues are disjoint (overlap was handled above), so
    // independent destination words need not follow logical tensor order.
    // Plan a source- or destination-major traversal before forming the 3D
    // domain. A token/head/feature view can otherwise fragment one affine
    // address map into thousands of tiny copies and pipeline drains.
    if(batchRows && std::set<I>(offsets.begin(),offsets.end()).size()==offsets.size()) {
      V order=range(indices.size()), reads, writes;
      reads.reserve(indices.size());writes.reserve(offsets.size());
      for(bool sourceMajor:{true,false}) {
        std::sort(order.begin(),order.end(),[&](I a,I b) {
          return sourceMajor?std::tie(indices[a],offsets[a])<std::tie(indices[b],offsets[b])
              :std::tie(offsets[a],indices[a])<std::tie(offsets[b],indices[b]);
        });
        reads.clear();writes.clear();
        for(I i:order) {reads.push_back(indices[i]);writes.push_back(offsets[i]);}
        if(affine(reads,writes,d)) {
          scalarDomain(src,dst,reads.front(),writes.front(),d);
          return;
        }
        auto candidate=regions(reads,writes);
        if(candidate.size()<best.size())best=std::move(candidate);
      }
    }
    for(const auto &part:best)scalarDomain(src,dst,part.source,part.destination,part.domain);
  }
  void unpack(mlir::DictionaryAttr source, mlir::DictionaryAttr dest,
              const V &shape, const V &indices, const V &offsets,
              I onlyResidue = -1, I sourceGroups = 1, bool issueSxm = true) {
    auto wide=function->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
    bool pairedWords=wide && wide.getValue() && onlyResidue<0 && sourceGroups==1 && issueSxm
        && text(source,"kind")=="fp16_hemisphere_pair_planar"
        && slices(source).size()==2 && slices(dest).size()==2
        && shape[1]%64==0 && indices.size()%32==0;
    for(I i=0;pairedWords && i<I(indices.size());i+=32)
      for(I lane=0;lane<32;++lane)pairedWords &= indices[i]%32==0 && indices[i+lane]==indices[i]+lane;
    if(pairedWords) {
      // Native MXM results alternate N32 words between hemispheres. Interpret
      // each side as its own packed matrix, then use the wide SXM unpack and
      // passive cross-side transport. The legacy VXM bridge cannot select an
      // arbitrary lane stream on a fixed ALU input port.
      const V compactShape{shape[0],shape[1]/2};
      for(I side=0;side<2;++side) {
        V selected,selectedOffsets;
        for(I i=0;i<I(indices.size());++i) {
          const I row=indices[i]/shape[1],column=indices[i]%shape[1];
          if(column/32%2!=side)continue;
          selected.push_back(row*compactShape[1]+column/64*32+column%32);
          selectedOffsets.push_back(offsets[i]);
        }
        if(selected.empty())continue;
        auto from=change(source,J{{"kind","fp16_pair_planar"},{"hemisphere",side?"west":"east"}});
        unpack(from,dest,compactShape,selected,selectedOffsets);
      }
      return;
    }
    bool wholeWords=wide && wide.getValue() && onlyResidue<0 && sourceGroups==1 && issueSxm
        && ((text(source,"kind")=="fp16_pair_planar" && slices(source).size()==2)
            || (text(source,"kind")=="fp32_quad_planar" && slices(source).size()==4))
        && slices(source).size()==slices(dest).size()
        && (text(source,"hemisphere")=="both" || text(dest,"hemisphere")=="both"
            || text(source,"hemisphere")==text(dest,"hemisphere"))
        && indices.size()%32==0 && shape[1]%32==0;
    for(I i=0;wholeWords && i<I(indices.size());i+=32)
      for(I lane=0;lane<32;++lane)wholeWords &= indices[i]%32==0 && indices[i+lane]==indices[i]+lane;
    if(wholeWords) {
      // Repeat a packed word as 32 SXM rows; transposition broadcasts each
      // element to one scalar-planar word. This is the inverse of pack().
      // Wide SXM permutes opaque 16-bit elements. FP32 uses two serial
      // byte-pair passes with identical indices; no arithmetic conversion.
      for(I plane=0;plane<I(slices(source).size());plane+=2)
      for(I i=0;i<I(indices.size());i+=32) {
        V selected(offsets.begin()+i,offsets.begin()+i+32);
        Domain write;
        if(!affine(range(32),selected,write))throw std::runtime_error("non-affine wide scalar output");
        const std::array<I,3> wi={1,write.count[0],write.count[0]*write.count[1]};
        const I address=indices[i]%shape[1]/32*shape[0]+indices[i]/shape[1];
        const I capture=cursor+32;
        I end=capture+80;
        for(I side=0;side<2;++side) {
          const std::string hemi=side?"west":"east";
          if(text(dest,"hemisphere")!="both" && text(dest,"hemisphere")!=hemi)continue;
          if(text(source,"hemisphere")!="both" && text(source,"hemisphere")!=hemi)continue;
          auto from=change(source,J{{"hemisphere",hemi}}),to=change(dest,J{{"hemisphere",hemi}});
          emit("sxm",J{{"cycle",capture},{"hemisphere",side},{"opcode","transpose"},
              {"source_streams",array({16+side*2,17+side*2})},{"destination_streams",array({36+side*2,37+side*2})},
              {"permute_map",array(range(32))},{"weight_layout","wide_matrix"},
              {"repeat_count",1},{"repeat_interval",32},{"wave_count",1},{"wave_interval",32}});
          for(I byte=0;byte<2;++byte) {
            const I rd=*target.transport_latency(fc::target::StreamEndpoint::Mem,
                fc::target::StreamEndpoint::SxmInput,fc::target::StreamDirection::East,slices(from)[plane+byte]);
            const I wr=*target.transport_latency(fc::target::StreamEndpoint::SxmResult,
                fc::target::StreamEndpoint::Mem,fc::target::StreamDirection::West,slices(to)[plane+byte]);
            mem(capture-rd,from,plane+byte,"read",field(from,"base_row")+address,16+side*2+byte,{32,1,1});
            mem(capture+35+wr,to,plane+byte,"write",field(to,"base_row")+selected[0],36+side*2+byte,write.count,write.dst,wi);
            end=std::max(end,capture+35+wr+36);
          }
        }
        cursor=end;
      }
      if(text(source,"hemisphere")!="both" && text(dest,"hemisphere")=="both") {
        auto local=change(dest,J{{"hemisphere",text(source,"hemisphere")}});
        auto peer=change(dest,J{{"hemisphere",text(source,"hemisphere")=="east"?"west":"east"}});
        copy(local,peer,offsets,offsets);
      }
      return;
    }
    // SXM operates on one hemisphere. "both" is a storage contract, not a
    // physical SXM endpoint: broadcast the completed scalar words explicitly.
    if (text(dest,"hemisphere") == "both") {
      auto local = change(dest,J{{"hemisphere","east"}});
      unpack(source,local,shape,indices,offsets,onlyResidue,sourceGroups,issueSxm);
      copy(local,change(dest,J{{"hemisphere","west"}}),offsets,offsets);
      return;
    }
    if (text(source,"hemisphere")=="both" && text(dest,"hemisphere")!="both")
      source=change(source,J{{"hemisphere",text(dest,"hemisphere")}});
    if (text(source,"kind")!="fp16_hemisphere_pair_planar" &&
        text(source,"hemisphere")!="both" &&
        text(source,"hemisphere")!=text(dest,"hemisphere")) {
      if (onlyResidue>=0 || offsets.empty() ||
          *std::max_element(offsets.begin(),offsets.end())>=target.memory().sram_depth_rows)
        throw std::runtime_error("cross-hemisphere unpack requires a bounded complete view");
      auto temporary=change(dest,J{{"hemisphere",text(source,"hemisphere")},
          {"slices",array(range(slices(dest).size(),48))},{"bank",0},{"base_row",0}});
      unpack(source,temporary,shape,indices,offsets,onlyResidue,sourceGroups,issueSxm);
      copy(temporary,dest,offsets,offsets);
      return;
    }
    I rows = shape[0], columns = shape[1];
    bool paired = text(source, "kind") == "fp16_hemisphere_pair_planar";
    I step = paired ? 64 : 32;
    std::map<I, std::set<I>> wordLanes;
    for (I ix : indices)
      wordLanes[ix / 32].insert(ix % 32);
    // Logical short rows occupy separate padded physical words. Combining
    // two 16-column rows does not make a complete 32-lane source word.
    bool full = columns % 32 == 0 && !wordLanes.empty();
    for (auto &p : wordLanes)
      full &= p.second.size() == 32;
    for (I half = 0; half < (paired && full ? 2 : 1); ++half)
      for (I residue = 0; residue < (full ? 8 : std::min(step, columns));
           ++residue) {
        if (onlyResidue >= 0 && residue != onlyResidue) continue;
        V words, offs;
        for (size_t i = 0; i < indices.size(); ++i) {
          I ix = indices[i];
          bool selected = full ? (ix % 8 == residue &&
                                  (!paired || ix % columns / 32 % 2 == half))
                               : (ix % columns % step == residue);
          if (selected) {
            words.push_back((ix % columns / step) * rows + ix / columns);
            offs.push_back(offsets[i]);
          }
        }
        if (words.empty())
          continue;
        auto src = change(
            source, J{{"hemisphere",
                       paired ? ((full ? half : residue / 32) ? "west" : "east")
                              : text(source, "hemisphere")}});
        Domain d;
        if (!affine(words, offs, d))
          throw std::runtime_error("non-affine packed-to-scalar tensor view");
        I count = product(V(d.count.begin(), d.count.end())),
          tile = full ? 0 : (residue % 32) / 8, lane = residue % 8;
        std::array<I, 3> interval = {32, d.count[0] * 32,
                                     d.count[0] * d.count[1] * 32};
        auto ss = slices(src), ds = slices(dest);
        for (I pair = 0; pair < I(ds.size()) / 2; ++pair) {
          I transpose = cursor + 32;
          for (I plane = 0; plane < 2; ++plane)
            mem(transpose - (14 - ss[2 * pair + plane] / group), src,
                2 * pair + plane, "read", field(src, "base_row") + words[0],
                16 + plane, d.count, d.src, interval);
          if (issueSxm) sxm(transpose, text(src, "hemisphere"), "transpose", {16, 17},
              range(16, 16), range(32), d.count[0], 32, -1,
              d.count[1] * d.count[2] * sourceGroups, d.count[0] * 32);
          I permute = transpose + 5;
          bool direct = text(src, "hemisphere") == text(dest, "hemisphere");
          if (ds.size() == 4 && !direct)
            throw std::runtime_error("FP32 unpack requires same hemisphere");
          V map;
          for (I i = 0; i < 32; ++i)
            map.push_back(tile * 8 + i % 8);
          if (issueSxm) sxm(permute, text(src, "hemisphere"), "permute", range(16, 16),
              range(16, 32), map, d.count[0], 32, lane < 7 ? 9 + lane : 7,
              d.count[1] * d.count[2] * sourceGroups, d.count[0] * 32, full ? 8 : 0);
          I result = 0;
          if (!direct) {
            I beat = permute + 15;
            auto from = text(src, "hemisphere"), to = text(dest, "hemisphere");
            vxm(beat - 1, 4, "cast", "stream_bf16", 2 * lane, "immediate", 0,
                from, "west", to, "fp32", -1, count, 32);
            vxm(beat - 1, 5, "cast", "previous", 0, "immediate", 0, "west",
                "west", to, "bf16", 0, count, 32);
            result = bridgeResult(beat, from, to, 4);
          }
          for (I p = 0; p < 2; ++p)
            mem(direct ? permute + 14 - ds[2 * pair + p] / group
                       : result + ds[2 * pair + p] / group + 1,
                dest, 2 * pair + p, "write", field(dest, "base_row") + offs[0],
                direct ? 32 + 2 * lane + p : p, d.count, d.dst, interval);
          cursor = onlyResidue >= 0 ? cursor + count * 32
                                   : permute + count * 32 + 32;
        }
      }
  }
  void pack(mlir::DictionaryAttr source, mlir::DictionaryAttr dest,
            const V &shape, const V &indices, const V &offsets) {
    I rows = shape[0], cols = shape[1], words = rows * (cols / 32);
    if (cols % 32 || slices(source).size() == 4)
      throw std::runtime_error("unsupported scalar packing geometry/type");
    Domain write;
    if (!affine(range(words), offsets, write))
      throw std::runtime_error("non-affine packed destination");
    std::array<I, 3> intervals = {32, write.count[0] * 32,
                                  write.count[0] * write.count[1] * 32};
    auto wide=function->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
    if (wide && wide.getValue() && text(source,"hemisphere")=="both") {
      // A scalar-planar word repeats its value on every lane. Transposing
      // 32 such words directly yields the desired packed vector in row 0.
      // Wide SXM needs neither the legacy 8-lane masks nor VXM accumulation.
      V reads;
      for(I block=0;block<cols/32;++block)for(I row=0;row<rows;++row)
        for(I lane=0;lane<32;++lane)reads.push_back(indices[row*cols+block*32+lane]);
      // Padded M tiles may repeat the last valid row. Keep that view exact
      // by issuing one 32-scalar transpose per packed destination word.
      for(I word=0;word<words;++word) {
        V selected(reads.begin()+word*32,reads.begin()+(word+1)*32);
        Domain read;
        if(!affine(selected,range(32),read))
          throw std::runtime_error("non-affine scalar-to-wide transpose word");
        const std::array<I,3> ri={1,read.count[0],read.count[0]*read.count[1]};
        const I capture=cursor+32;
        I end=capture+64;
        for(I side=0;side<2;++side) {
          const std::string hemi=side?"west":"east";
          if(text(dest,"hemisphere")!="both" && text(dest,"hemisphere")!=hemi)continue;
          auto from=change(source,J{{"hemisphere",hemi}}),to=change(dest,J{{"hemisphere",hemi}});
          emit("sxm",J{{"cycle",capture},{"hemisphere",side},{"opcode","transpose"},
              {"source_streams",array({16+side*2,17+side*2})},{"destination_streams",array({36+side*2,37+side*2})},
              {"permute_map",array(range(32))},{"weight_layout","wide_matrix"},
              {"repeat_count",1},{"repeat_interval",32},{"wave_count",1},{"wave_interval",32}});
          for(I byte=0;byte<2;++byte) {
            const I rd=*target.transport_latency(fc::target::StreamEndpoint::Mem,
                fc::target::StreamEndpoint::SxmInput,fc::target::StreamDirection::East,slices(from)[byte]);
            const I wr=*target.transport_latency(fc::target::StreamEndpoint::SxmResult,
                fc::target::StreamEndpoint::Mem,fc::target::StreamDirection::West,slices(to)[byte]);
            mem(capture-rd,from,byte,"read",field(from,"base_row")+selected[0],16+side*2+byte,read.count,read.src,ri);
            mem(capture+35+wr,to,byte,"write",field(to,"base_row")+offsets[word],36+side*2+byte);
            end=std::max(end,capture+35+wr+4);
          }
        }
        cursor=end;
      }
      return;
    }
    auto staging = placement("fp16_pair_planar", {}, "east", 1, 0);
    for (I slot = 0; slot < 8; ++slot) {
      V selected;
      for (I block = 0; block < cols / 32; ++block)
        for (I row = 0; row < rows; ++row)
          for (I tile = 0; tile < 4; ++tile)
            selected.push_back(
                indices[row * cols + block * 32 + tile * 8 + slot]);
      staging =
          change(staging, J{{"slices", array({32 + slot * 2, 33 + slot * 2})}});
      copy(source, staging, selected, range(selected.size()));
    }
    auto masks = maskPlacement;
    std::array<mlir::DictionaryAttr, 2> ping = {
        placement("fp32_quad_planar", {0, 1, 2, 3}, "east", 1, 0),
        placement("fp32_quad_planar", {4, 5, 6, 7}, "east", 1, 0)};
    for (I tile = 0; tile < 4; ++tile) {
      I transpose = cursor + 32;
      for (I slot = 0; slot < 8; ++slot) {
        staging = change(staging,
                         J{{"slices", array({32 + slot * 2, 33 + slot * 2})}});
        for (I p = 0; p < 2; ++p)
          mem(transpose - (14 - slices(staging)[p] / group), staging, p, "read",
              tile, slot * 2 + p, {words, 1, 1}, {4, 0, 0}, {32, 1, 1});
      }
      sxm(transpose, "east", "transpose", range(16), range(16, 16), range(32),
          words, 32);
      I permute = transpose + 5;
      V map;
      for (I i = 0; i < 32; ++i)
        map.push_back(i % 8);
      sxm(permute, "east", "permute", range(16, 16), range(16, 32), map, words,
          32, 9);
      I beat = permute + 15;
      for (I p = 0; p < 4; ++p)
        mem(beat - (slices(masks)[p] / group + 1), masks, p, "read", field(masks,"base_row")+tile,
            40 + p, {words, 1, 1}, {0, 0, 0}, {32, 1, 1});
      vxm(beat - 1, 0, "multiply", "stream_bf16", 0, "stream_f32", 8, "east",
          "east", "west", "fp32", -1, words, 32);
      I last = 1;
      if (tile) {
        auto previous = ping[(tile - 1) % 2];
        for (I p = 0; p < 4; ++p)
          mem(beat + 1 - (slices(previous)[p] / group + 1), previous, p, "read",
              0, 44 + p, {words, 1, 1}, {1, 0, 0}, {32, 1, 1});
        vxm(beat - 1, 1, "add", "previous", 0, "stream_f32", 12, "west", "east",
            "west", "fp32", -1, words, 32);
        last = 2;
      }
      auto output = tile == 3 ? dest : ping[tile % 2];
      auto hemi = text(output, "hemisphere");
      if (hemi == "both")
        hemi = "east";
      vxm(beat - 1, last, "cast", "previous", 0, "immediate", 0, "west", "west",
          hemi, tile == 3 ? "bf16" : "fp32", 0, words, 32);
      I result = beat + (hemi == "west" ? 18 : 2 * last + 3);
      auto p = change(output, J{{"hemisphere", hemi}});
      for (I plane = 0; plane < I(slices(p).size()); ++plane)
        mem(result + slices(p)[plane] / group + 1, p, plane, "write",
            field(p, "base_row") + (tile == 3 ? offsets[0] : 0), plane,
            tile == 3 ? write.count : std::array<I, 3>{words, 1, 1},
            tile == 3 ? write.dst : std::array<I, 3>{1, 0, 0},
            tile == 3 ? intervals : std::array<I, 3>{32, 1, 1});
      cursor = result + words * 32 + 96;
    }
    if (text(dest, "hemisphere") == "both")
      copy(change(dest, J{{"hemisphere", "east"}}),
           change(dest, J{{"hemisphere", "west"}}), offsets, offsets);
  }
};
class Linker {
public:
  Planner &planner;
  Scheduler s;
  std::vector<Tensor> finals;
  // Only selected results cross the invocation's external boundary. Other
  // returned roots remain live in native SRAM without a validation gather.
  std::set<I> externalOutputs;
  fs::path directory;
  std::map<Key, mlir::DictionaryAttr> cache, live;
  std::map<Key, I> last;
  struct ExclusiveReadQueue { Key owner; I begin, end; mlir::DictionaryAttr p; };
  std::vector<ExclusiveReadQueue> exclusiveReadQueues;
  std::map<std::pair<std::string, I>, std::vector<Key>> native;
  struct TimedWorkspace { I node, begin, end; };
  std::map<std::pair<std::string,I>,std::vector<TimedWorkspace>> nativeTimed, nativeBank0;
  std::map<std::pair<std::string, I>, I> staticEnd;
  struct Static {
    mlir::DictionaryAttr p;
    I index;
    mlir::Value binding;
  };
  std::map<std::string, Static> statics;
  std::vector<Tensor> inputs;
  llvm::json::Array nodeRanges;
  llvm::json::Array projectionSolveReport;
  llvm::json::Array projectionConvReport;
  llvm::json::Array projectionVectorReport;
  llvm::json::Array headIngressReport;
  std::map<I,mlir::DictionaryAttr> recurrentStates;
  std::map<I,mlir::Value> initialStateBindings;
  std::map<Key,mlir::DictionaryAttr> partitionOutputHomes;
  std::map<Key,mlir::DictionaryAttr> retainedOutputHomes;
  std::map<Key,mlir::DictionaryAttr> pointwiseHomes;
  std::map<Key,mlir::DictionaryAttr> externalVectorHomes;
  std::map<Key,Static> overlapExternalBindings;
  std::map<I,std::pair<I,I>> weightIngressWindows;
  // A packed producer row can serve multiple tiled consumers. Retain one
  // scalar-vector table per row, rather than expanding each consumer's view.
  using BroadcastKey = std::tuple<I,I,I>;
  std::map<BroadcastKey,mlir::DictionaryAttr> wideBroadcastHomes;
  std::map<Key,BroadcastKey> wideBroadcastOperands;
  std::set<BroadcastKey> emittedWideBroadcasts;
  bool weightPingPong = false;
  fc::target::GraphWeightPaging weightPolicy;
  std::set<I> weightSlotSlices;
  Linker(Planner &p, std::vector<Tensor> out, fs::path dir)
      : planner(p), s(p.context, p.target), finals(std::move(out)),
        directory(std::move(dir)) {
    if(auto policy=p.policies.get("ftlpu.external_output_indices")) {
      auto indices=llvm::dyn_cast<mlir::ArrayAttr>(policy);
      if(!indices) throw std::runtime_error("external_output_indices must be an integer array");
      for(auto item:indices) {
        auto index=llvm::dyn_cast<mlir::IntegerAttr>(item);
        if(!index || index.getInt()<0 || index.getInt()>=I(finals.size()) ||
           !externalOutputs.insert(index.getInt()).second)
          throw std::runtime_error("external_output_indices contains a duplicate or invalid result index");
      }
    } else for(I i=0;i<I(finals.size());++i)externalOutputs.insert(i);
    planProjectionSolveOverlap(p,finals);
    planProjectionConvOverlap(p,finals);
    planProjectionVectorOverlap(p,finals);
    weightPolicy=fc::target::GraphWeightPaging::resolve(p.target,p.policies);
    weightPingPong=weightPolicy.overlap;
    for(auto attr:p.policies) (*s.module)->setAttr(attr.getName(),attr.getValue());
    if (weightPingPong) {
      // Reserve both slots before allocating graph values. A bank's MEM ICU
      // cannot read an activation while a page WRITE_SYNC owns that queue,
      // even when their row ranges are disjoint. The executable's MEM/SR
      // lease proof therefore also checks queue lifetimes before a refill.
      // Use one contiguous weight group, leaving contiguous 16-slice
      // activation groups available to the wide transpose/norm kernels.
      // Change the tensor placements before native lowering so MEM issue
      // times are recomputed for the new physical SR distances.
      fc::target::weight_pages::SlotPlanner slots({s.target.memory().sram_depth_rows,
          s.target.memory().banks_per_slice,1});
      I nextSlot=0;
      for (I ni=0;ni<I(p.nodes.size());++ni) {
        auto& node=p.nodes[ni];
        for (I operand=0;operand<I(node.in.size());++operand) {
        auto placement=node.in[operand];
        // Dense/native composite adapters have already planned their complete
        // residency using the common planner. Do not rewrite one role without
        // also rewriting its internal page traversals and workspace ABI.
        if(auto native=placement.getAs<mlir::BoolAttr>("paged_weight");
            native && native.getValue()) continue;
        if (text(placement,"kind")=="w8a16_mxm_weight_striped") {
          const I rows=field(placement,"instruction_count");
          if (rows>weightPolicy.rows)
            throw std::runtime_error("projection page exceeds planned slot capacity");
          const I count=slices(placement).size();
          if(count>weightPolicy.slice_count)
            throw std::runtime_error("projection weight planes exceed reserved slice group");
          const auto group=range(count,weightPolicy.slice_base);
          std::optional<fc::target::weight_pages::Slot> selected;
          I release=0;
          for(I attempt=0;attempt<s.target.memory().banks_per_slice;++attempt) {
            fc::target::weight_pages::Slot candidate{(nextSlot+attempt)%s.target.memory().banks_per_slice,
                0,1,weightPolicy.base_row,rows};
            bool conflict=false;
            for(auto workspace:node.workspaces)
              if(field(workspace,"bank")==candidate.bank
                  && field(workspace,"base_row")<candidate.base_row+rows
                  && candidate.base_row<field(workspace,"base_row")+field(workspace,"instruction_count")
                  && llvm::any_of(slices(workspace),[&](I slice){return contains(group,slice);}))
                conflict=true;
            if(conflict) continue;
            release=slots.earliestReuse(candidate,0).value_or(ni+1);
            if(release>ni || !slots.reserve(candidate,{ni,ni+1})) continue;
            selected=candidate;break;
          }
          if(!selected) throw std::runtime_error("weight page conflicts with native consumer workspace");
          nextSlot=(selected->bank+1)%s.target.memory().banks_per_slice;
          placement=s.change(placement,J{{"slices",s.array(group)},
              {"bank",selected->bank},{"base_row",selected->base_row},
              {"page_consumer_node",ni},{"page_release_before_node",release}});
          node.in[operand]=placement;
          for(auto& slot:node.inputSlots[operand]) slot.set(placement);
          for(I slice:slices(placement)) weightSlotSlices.insert(slice);
        }
        }
      }
    }
    for (I i = 0; i < I(p.nodes.size()); ++i) {
      for (auto &a : p.nodes[i].inputs)
        for (auto ref : a.refs)
          last[{ref.node, ref.output}] = i;
      if (!p.nodes[i].module)
        continue;
      // Scaled/KS/delta results are physical scratch even if not exposed by
      // the graph. Give them real homes until the composite step retires.
      if(!p.nodes[i].function.getOps<fc::kernel::ScaledMatvecOp>().empty())
        for(I oi=0;oi<I(p.nodes[i].outputs.size());++oi)
          last.try_emplace({i,oi},i);
      for (auto workspace : p.nodes[i].workspaces) {
        if (field(workspace,"bank")==0) {
          for(auto h:{"east","west"})
            if(text(workspace,"hemisphere")==h || text(workspace,"hemisphere")=="both")
              for(I slice:slices(workspace)) nativeBank0[{h,slice}].push_back(
                {i,field(workspace,"base_row"),field(workspace,"base_row")+field(workspace,"instruction_count")});
          continue;
        }
        if (field(workspace, "bank") != 1) continue;
        for (auto h : {"east", "west"})
          if (text(workspace, "hemisphere") == h ||
              text(workspace, "hemisphere") == "both")
            for (I slice : slices(workspace))
              nativeTimed[{h, slice}].push_back(
                  {i, field(workspace, "base_row"),
                   field(workspace, "base_row") +
                       field(workspace, "instruction_count")});
      }
    }
    // Generic layout-conversion workspaces are graph resources too. They
    // cannot alias a long-lived producer result, including a paired result on
    // both hemispheres. Reserve them before selecting retained output slices.
    const I half = s.target.memory().sram_depth_rows / 2;
    for (I slice = 0; slice < 8; ++slice)
      native[{"east", slice}].push_back({0, half});
    for (I slice = 32; slice < s.target.memory().slices_per_hemisphere; ++slice)
      native[{"east", slice}].push_back({0, half});
    for (I slice = 48; slice < s.target.memory().slices_per_hemisphere; ++slice)
      native[{"west", slice}].push_back({0, half});
    for (auto &f : finals)
      for (auto ref : f.refs)
        last[{ref.node, ref.output}] = p.nodes.size();
  }
  mlir::DictionaryAttr
  allocate(Key key, const Tensor &t, std::string hemisphere = "west",
           I packed = -1, mlir::DictionaryAttr prototype = {},
           const std::vector<mlir::DictionaryAttr> &forbiddenQueues = {},
           const std::vector<mlir::DictionaryAttr> &exportSources = {}) {
    if (prototype)
      hemisphere = text(prototype, "hemisphere");
    if (packed < 0)
      packed = t.shape.back() % s.target.memory().bytes_per_word == 0;
    I rows = prototype ? field(prototype, "instruction_count")
                       : packed ? (t.size() / t.shape.back()) * ((t.shape.back() + s.target.memory().bytes_per_word - 1)
                                                  / s.target.memory().bytes_per_word) : t.size(),
      planes = prototype          ? I(slices(prototype).size())
               : t.dtype == "f32" ? 4
                                  : 2,
      depth = s.target.memory().sram_depth_rows;
    // Final view materialization starts after all native kernels. Earlier
    // kernel scratch reservations must not prevent reusing their SRAM here.
    // Nonzero external state is uploaded before the first node, even when
    // its recurrence consumer appears much later. Earlier native scratch
    // must not overwrite that ingress image while projections/norms run.
    const bool initialState = key.first == -4
        && planner.nodes[key.second].recurrent
        && planner.nodes[key.second].inputs.size()>5
        && planner.nodes[key.second].inputs[5].external()
        && llvm::any_of(planner.nodes[key.second].inputs[5].data,
                        [](uint8_t byte) { return byte != 0; });
    I born = key.first >= 0 ? key.first
                   : key.first == -4 ? (initialState ? 0 : key.second)
                   : key.first == -3 ? I(planner.nodes.size()) : 0;
    I dead = last.count(key) ? last.at(key) : planner.nodes.size();
    // A concurrent projection and its consumer groups can touch native
    // scratch throughout the same physical interval. Enclose that interval
    // when testing fixed (non-allocator) workspace row reservations.
    if(born>=0 && born<I(planner.nodes.size()) && planner.nodes[born].projectionConvGroup>=0)
      born=planner.nodes[born].projectionConvGroup;
    if(dead>=0 && dead<I(planner.nodes.size()) && planner.nodes[dead].projectionConvGroup>=0) {
      const I group=planner.nodes[dead].projectionConvGroup;
      while(dead+1<I(planner.nodes.size()) && planner.nodes[dead+1].projectionConvGroup==group)++dead;
    }
    V pairs;
    // Keep immutable constants in activation storage where possible. Filling
    // the small free tails of all weight slice groups fragments the 16-slice
    // layouts required by the following RMSNorm/projection consumers.
    if (key.first == -2)
      for (I first=0; first+planes<=s.target.memory().w8a16_weight_slice_base; first+=2)
        pairs.push_back(first);
    if (prototype)
      pairs.push_back(slices(prototype).front());
    for (I first = 32;
         first <= s.target.memory().slices_per_hemisphere - planes; first += 2)
      pairs.push_back(first);
    for (I first = 8; first <= 32 - planes; first += 2)
      pairs.push_back(first);
    if (hemisphere == "west" || prototype)
      for (I first = 0; first <= 8 - planes; first += 2)
        pairs.push_back(first);
    if (weightPingPong)
      for (I first=0;first+planes<=s.target.memory().slices_per_hemisphere;first+=2)
        pairs.push_back(first);
    if (weightPingPong && planes<=4)
      std::stable_sort(pairs.begin(),pairs.end(),[&](I a,I b) {
        const auto touches=[&](I first) { return llvm::any_of(range(planes,first),
            [&](I slice){return weightSlotSlices.count(slice)!=0;}); };
        return touches(a)<touches(b);
      });
    // Retained values normally use bank 1. A second live recurrent State may
    // exhaust its contiguous rows while bank 0 has space above native scratch.
    // Try bank 0 only after bank 1, checking every workspace over the value's
    // full lifetime and retaining the same MEM queue exclusion rules.
    const bool txExport=key.first==-3 && !exportSources.empty();
    const bool bankFallback = (key.first >= 0 || txExport) &&
        !(prototype && field(prototype,"bank_tile_rows"));
    std::vector<std::pair<I,I>> homes;
    for(I bank:{I(1),I(0)})if(bank==1 || bankFallback)
      for(I first:pairs)homes.emplace_back(bank,first);
    if(txExport) {
      const I group=s.target.streams().mem_slices_per_register_group;
      const I tx=s.target.streams().mem_boundary_register_columns-1;
      // TX taps the last MEM boundary; each plane starts at its group's
      // forward output boundary. Minimize the longest route, then their sum.
      const auto cost=[&](I first) {
        I worst=0,total=0;
        for(I plane=0;plane<planes;++plane) {
          const I hops=tx-((first+plane)/group+1);
          worst=std::max(worst,hops);total+=hops;
        }
        return std::make_pair(worst,total);
      };
      std::stable_sort(homes.begin(),homes.end(),[&](auto a,auto b) {
        return cost(a.second)<cost(b.second);
      });
    }
    for (auto [allocationBank,first] : homes) {
      if(txExport) {
        // Final copies can use S48+ as transient linker scratch. It is not
        // in the native kernel lifetime map during validation export.
        if(first+planes>48)continue;
        bool direct=true;
        for(auto source:exportSources) {
          if(text(source,"hemisphere")!=hemisphere && text(source,"hemisphere")!="both")continue;
          const auto ss=slices(source);
          const I group=s.target.streams().mem_slices_per_register_group;
          for(I p=0;p<planes;++p) {
            if(ss[p]/group==(first+p)/group ||
                (field(source,"bank")==allocationBank && contains(ss,first+p)))direct=false;
          }
        }
        if(!direct)continue;
      }
      bool projection = first < 30 && first + planes > 28;
      I lower = hemisphere == "west" ? (first >= 48 ? depth / 2
                                        : (first < 8 || projection ||
                                           (first <= 25 && first + planes > 24))
                                            ? 2048
                                            : 0)
                                     : (first >= 32  ? depth / 2
                                        : projection ? 2048
                                                     : 0);
      std::set<I> candidates = {lower};
      for (auto &kv : live)
        candidates.insert(field(kv.second, "base_row") +
                          field(kv.second, "instruction_count"));
      std::vector<Key> ranges;
      if (weightPingPong && llvm::any_of(range(planes,first),
            [&](I slice){return weightSlotSlices.count(slice)!=0;}))
        ranges.push_back({weightPolicy.base_row,weightPolicy.base_row+weightPolicy.rows});
      for (auto h : {"east", "west"})
        if (hemisphere == h || hemisphere == "both")
          for (I slice = first; slice < first + planes; ++slice)
          {
            if (allocationBank == 1) for (auto r : native[{h, slice}]) {
              ranges.push_back(r);
              candidates.insert(r.second);
            }
            if(allocationBank == 0 || (prototype && field(prototype,"bank_tile_rows")))
              for(auto r:nativeBank0[{h,slice}]) if(r.node>=born && r.node<=dead) {
                ranges.push_back({r.begin,r.end}); candidates.insert(r.end);
              }
            if (allocationBank == 1) for (auto r : nativeTimed[{h,slice}])
              if (r.node >= born && r.node <= dead) {
                ranges.push_back({r.begin,r.end}); candidates.insert(r.end);
              }
          }
      for (I row : candidates) {
        if (row < lower || row + rows > depth)
          continue;
        bool conflict = false;
        for (auto r : ranges)
          if (row < r.second && r.first < row + rows)
            conflict = true;
        auto p = s.placement(
            packed ? (planes == 4 ? "fp32_quad_planar" : "fp16_pair_planar")
                   : (planes == 4 ? "fp32_scalar_broadcast_planar"
                                  : "fp16_scalar_broadcast_planar"),
            range(planes, first), hemisphere, allocationBank, row, rows);
        if (prototype)
          p = s.change(p, J{{"kind", text(prototype, "kind")},
                            {"instruction_count", rows}});
        if (prototype && prototype.get("partition_row_sides")) {
          mlir::NamedAttrList mapped(p);
          for(auto name:{"partition_row_sides","partition_row_indices","partition_storage_rows"})
            mapped.set(name,prototype.get(name));
          p=mapped.getDictionary(&s.context);
        }
        if (prototype && prototype.get("dense_attention_role")) {
          // Relocating native context changes its physical home, not its
          // head/token stride or mirrored ownership. Transport and the dense
          // emitter must retain the same logical-to-physical mapping.
          mlir::NamedAttrList mapped(prototype);
          for (auto field : p) mapped.set(field.getName(),field.getValue());
          p=mapped.getDictionary(&s.context);
        }
        for (auto busy : forbiddenQueues)
          if (overlap(p, busy))
            conflict = true;
        for (const auto &busy : exclusiveReadQueues)
          if(key!=busy.owner && born<=busy.end && dead>=busy.begin && overlap(p,busy.p))
            conflict=true;
        // A striped producer finishes its reads before writing either bank.
        // Reserve the row range in both banks for its entire graph lifetime.
        if(prototype && field(prototype,"bank_tile_rows"))
          p=s.change(p,J{{"bank_tile_rows",field(prototype,"bank_tile_rows")}});
        for (auto &kv : live)
          if (overlap(p, kv.second) &&
              row < field(kv.second, "base_row") +
                        field(kv.second, "instruction_count") &&
              field(kv.second, "base_row") < row + rows)
            conflict = true;
        if (!conflict) {
          cache[key] = p;
          live[key] = p;
          return p;
        }
      }
    }
    if(txExport)throw GraphSramAllocationFailure("no direct C2C output home");
    for (auto busy : forbiddenQueues)
      llvm::errs() << "unavailable output queue: " << busy << "\n";
    for (const auto &entry : live)
      llvm::errs() << "live graph value " << entry.first.first << ":"
                   << entry.first.second << " through " << last[entry.first]
                   << ": " << entry.second << "\n";
    throw GraphSramAllocationFailure(
        "graph SRAM allocation failed for " + std::to_string(key.first) + ":" +
        std::to_string(key.second) + " rows=" + std::to_string(rows));
  }
  Static input(const Tensor &t, mlir::DictionaryAttr original,
               std::string name, bool directNativeIngress = false,
               I directLastUse = 0,
               const std::vector<mlir::DictionaryAttr> &forbiddenQueues = {},
               bool resolveOverlapNative = false, bool mirroredBinding = false,
               bool fixedSourceBinding = false, bool nativeBroadcast = false) {
    if(std::getenv("FTLPU_GRAPH_PROGRESS"))llvm::errs()<<"graph: binding "<<name<<" bytes="<<t.data.size()<<"\n";
    // A later independent projection may already have an immutable C2C home
    // from QKV. Reuse it when its physical queues satisfy the joint plan;
    // isolating execution queues must not imply uploading the tensor twice.
    if(resolveOverlapNative)
      for(const auto& entry:statics) {
        const auto& parent=inputs[entry.second.index];
        const auto home=entry.second.p;
        const auto kind=text(home,"kind"),requested=text(original,"kind");
        const bool compatible=kind=="fp16_pair_planar" && slices(home).size()==2
            && !home.get("storage_shape")
            && field(home,"instruction_count")==field(original,"instruction_count")
            && ((requested=="fp16_pair_planar" && slices(original).size()==2)
                || requested=="fp16_mxm_activation_planar");
        if(!compatible || parent.dtype!=t.dtype || parent.shape!=t.shape || parent.data!=t.data
            || text(home,"hemisphere")!=text(original,"hemisphere")
            || llvm::any_of(forbiddenQueues,[&](auto p){return overlap(home,p);}))continue;
        last[{-2,entry.second.index}]=std::max(last[{-2,entry.second.index}],directLastUse);
        return entry.second;
      }
    // Views of a retained activation use the original upload and carry
    // their logical offset into the on-chip layout conversion.
    if (t.external() && t.sourceContiguous && !fixedSourceBinding && !mirroredBinding && (!resolveOverlapNative || forbiddenQueues.empty()))
      for (const auto &entry : statics) {
        const auto &parent = inputs[entry.second.index];
        if (!parent.sourceContiguous ||
            parent.dtype != t.dtype || parent.size() < t.size() ||
            (parent.shape == t.shape && parent.sourceOffset == t.sourceOffset) ||
            text(entry.second.p,"kind") != "fp16_mxm_distributed_16") continue;
        const I offset = t.sourceOffset-parent.sourceOffset;
        if (offset<0 || offset+t.size()>parent.size() ||
            !std::equal(t.data.begin(),t.data.end(),parent.data.begin()+offset*t.bytes())) continue;
        auto alias = entry.second;
        alias.p = s.change(alias.p,J{{"storage_shape",s.array(parent.shape)},
                                    {"storage_offset",offset}});
        return alias;
      }
    // Equal immutable values can share a reservation across hemispheres;
    // move() routes them to the consumer's final placement. A side change
    // during memory planning must not allocate a new initially uploaded
    // binding in storage already used by an earlier producer.
    std::string identity(reinterpret_cast<const char *>(t.data.data()),
                         t.data.size());
    identity += t.dtype;
    for (I d : t.shape)
      identity += "," + std::to_string(d);
    if(!forbiddenQueues.empty()) identity += ":isolated:" + name;
    // A mirrored contraction home must not alias a same-valued single-side
    // upload: both physical fabrics will read their own matrix rows.
    if(mirroredBinding)identity += ":mirrored";
    // These are separate request argument subranges, even when an example
    // packet happens to contain identical bytes in two of them.
    if(fixedSourceBinding)identity += ":argument_range:" + name;
    // A reused external activation has one physical lifetime, even when its
    // consumers request different layouts. Keep a distributed on-chip home
    // and materialize each consumer layout from that shared binding.
    I activationLastUse = directLastUse;
    bool sharedActivation = false;
    mlir::DictionaryAttr activationPrototype;
    if (t.dtype == "bf16" && t.shape.size() == 2 && t.shape[0] > 1) {
      I consumers = 0;
      for (I ni=0; ni<I(planner.nodes.size()); ++ni)
        for (I ai=0; ai<I(planner.nodes[ni].inputs.size()); ++ai) {
          const auto &candidate = planner.nodes[ni].inputs[ai];
          if (candidate.external() && candidate.dtype == t.dtype &&
              candidate.sourceContiguous && t.sourceContiguous &&
              candidate.sourceOffset >= t.sourceOffset &&
              candidate.sourceOffset-t.sourceOffset+candidate.size() <= t.size() &&
              std::equal(candidate.data.begin(),candidate.data.end(),
                  t.data.begin()+(candidate.sourceOffset-t.sourceOffset)*t.bytes())) {
            ++consumers;
            activationLastUse = std::max(activationLastUse, ni);
            if (ai >= I(planner.nodes[ni].in.size())) continue;
            auto placement = planner.nodes[ni].in[ai];
            if (candidate.shape == t.shape &&
                text(placement,"kind") == "fp16_mxm_distributed_16")
              activationPrototype = placement;
          }
        }
      sharedActivation = consumers > 1 && bool(activationPrototype);
    }
    identity += sharedActivation ? "shared_activation" : text(original, "kind");
    // Equal weights in different paged kernels are distinct residency
    // intervals. A later head batch must reload its KV page after intervening
    // Q/O pages have reused that physical bank.
    if ((original.getAs<mlir::BoolAttr>("paged_weight") && original.getAs<mlir::BoolAttr>("paged_weight").getValue())
        || (t.byteWeight() && text(original,"kind")=="w8a16_mxm_weight_striped"))
      identity += ":" + name;
    if (!sharedActivation && !scalar(original)) {
      for (I x : slices(original))
        identity += "," + std::to_string(x);
    }
    if (auto it = statics.find(identity); it != statics.end()) {
      auto use = last.find({-2, it->second.index});
      if (use != last.end()) use->second = std::max(use->second, directLastUse);
      return it->second;
    }
    auto p = original;
    I index = inputs.size();
    if (sharedActivation) {
      last[{-2,index}] = activationLastUse;
      auto prototype = s.change(activationPrototype,J{{"hemisphere","east"},
          {"bank",1},{"base_row",0}});
      p = allocate({-2,index},t,"east",1,prototype);
    } else if (directNativeIngress) {
      // The first RMSNorm is the initial consumer of these external values.
      // C2C can pack its distributed activation and broadcast gamma directly
      // into the native placements; a generic two-plane staging binding would
      // immediately require a MEM -> VXM -> MEM layout conversion.
      if(weightPingPong && field(p,"base_row")<weightPolicy.base_row+weightPolicy.rows
          && weightPolicy.base_row<field(p,"base_row")+field(p,"instruction_count")
          && llvm::any_of(slices(p),[&](I slice){return weightSlotSlices.count(slice)!=0;}))
        throw std::runtime_error("direct native ingress overlaps reserved weight capacity: "+name);
      for (const auto &entry : live)
        if (overlap(p, entry.second))
          throw std::runtime_error("direct native input overlaps a live binding: " + name);
      live[{-2, index}] = p;
      last[{-2, index}] = directLastUse;
    } else if (auto paged = p.getAs<mlir::BoolAttr>("paged_weight"); paged && paged.getValue()) {
      // Native composite operators already own their physical page layout.
      // Preserve it and let compiler page SYNCs and runtime C2C feed the data.
    } else if (t.byteWeight() && text(p, "kind") == "w8a16_mxm_weight_striped") {
      p = s.change(p, J{{"bank", weightPingPong ? field(p,"bank") : 0},
                        {"base_row", weightPingPong ? field(p,"base_row") : 0},
                        {"paged_weight", true},
                        {"page_count", 1},
                        {"page_rows", field(p, "instruction_count")},
                        {"page_granularity", t.shape[1] / (s.target.memory().hemispheres*s.target.throughput().mxm_rows)},
                        {"page_role_group_base", 0},
                        {"page_role_group_count", 1},
                        {"page_items_per_slice_group", t.shape[1] / (s.target.memory().hemispheres*s.target.throughput().mxm_rows)},
                        {"page_bank_count", s.target.memory().banks_per_slice},
                        {"page_storage_slices", s.array(slices(p))}});
    } else if (text(p,"kind") == "fp16_depthwise_kernel4" && !forbiddenQueues.empty()) {
      // Keep the existing packed four-tap kernel, but assign its C2C upload
      // directly to eight queues outside the overlapping projection/page slot.
      p=allocate({-2,index},t,text(p,"hemisphere"),1,p,forbiddenQueues);
      last[{-2,index}]=directLastUse;
    } else if (text(p,"kind") == "fp16_vxm_gamma_broadcast") {
      last[{-2,index}] = directLastUse;
      if(nativeBroadcast) {
        // Planar RMSNorm reads one coefficient vector per cycle. Keep the
        // C2C broadcast in its final, lifetime-safe two-plane home.
        p=allocate({-2,index},t,"both",0,p,forbiddenQueues);
      } else {
      // Keep later normalization weights in more, shorter SRAM segments.
      // At use time copy the broadcast rows to the native two-segment home.
      auto prototype = s.change(p,J{{"slices",s.array(range(8))},
          {"hemisphere","west"},{"bank",1},{"base_row",0},
          {"instruction_count",(t.size()+3)/4}});
      p = allocate({-2,index},t,"west",0,prototype);
      }
    } else if (text(p,"kind") == "fp16_projection_bias_x4") {
      // All immutable biases are initialized up front, but retire after
      // their final attention batch. Later FFN temporaries may reuse them.
      I finalUse = directLastUse;
      for (I ni=0; ni<I(planner.nodes.size()); ++ni)
        for (const auto &candidate : planner.nodes[ni].inputs)
          if (candidate.external() && candidate.dtype == t.dtype
              && candidate.shape == t.shape && candidate.data == t.data)
            finalUse = std::max(finalUse,ni);
      last[{-2,index}] = finalUse;
      p = allocate({-2,index},t,text(p,"hemisphere"),1,p);
    } else if (p.get("dense_attention_role")) {
      // The dense kernels consume specialized native word layouts. External
      // arguments have an ordinary planar C2C home and are materialized by
      // the same compiler transport path as an on-chip producer.
      p = allocate({-2,index},t,"east",1,{},forbiddenQueues);
      last[{-2,index}] = directLastUse;
    } else if (scalar(p) || text(p, "kind") == "fp16_pair_planar" ||
               text(p, "kind") == "fp16_mxm_activation_planar" ||
               text(p, "kind") == "fp16_mxm_distributed_16" ||
               text(p, "kind") == "fp32_quad_planar" ||
               (resolveOverlapNative &&
                (text(p,"kind")=="token_partition_planar" ||
                 text(p,"kind")=="token_partition_east_first_planar"))) {
      p = allocate({-2, index}, t, text(p, "hemisphere"), !scalar(p),
                   forbiddenQueues.empty() && !(text(p,"kind")=="fp16_pair_planar" && slices(p).size()>2)
                       ?mlir::DictionaryAttr{}:p,forbiddenQueues);
      last[{-2, index}] = planner.nodes.size() + 1;
      if(t.dtype=="bf16" && t.size()/s.target.memory().bytes_per_word>s.target.memory().sram_depth_rows/2) {
        // A bank-sized ingress activation need not remain live after its
        // last native consumer. Synthetic aliases retain their own graph
        // lifetime; keep the conservative rule for those bindings.
        I finalUse=activationLastUse;bool syntheticAlias=false,matched=false;
        for(I ni=0;ni<I(planner.nodes.size());++ni)
          for(const auto &candidate:planner.nodes[ni].inputs)
            if(candidate.external() && candidate.dtype==t.dtype && candidate.shape==t.shape && candidate.data==t.data) {
              matched=true;finalUse=std::max(finalUse,ni);
              syntheticAlias |= !planner.nodes[ni].module;
            }
        for(const auto &value:finals)
          if(value.external() && value.dtype==t.dtype && value.shape==t.shape && value.data==t.data)
            syntheticAlias=true;
        if(matched && !syntheticAlias)last[{-2,index}]=finalUse;
      }
    } else {
      I row = 512;
      for (auto h : {"east", "west"})
        if (text(p, "hemisphere") == h || text(p, "hemisphere") == "both")
          for (I slice : slices(p)) {
            auto key = std::make_pair(std::string(h), slice);
            row = std::max(row, staticEnd[key]);
            for (auto r : native[key])
              row = std::max(row, r.second);
            // Immutable native-format operands are uploaded before execution.
            // Keep them clear of every bank-1 workspace, including raw Q/K
            // staging used before their own projection consumes the bias.
            for (auto r : nativeTimed[key])
              row = std::max(row, r.end);
          }
      // Generic planar inputs were allocated through live[], not staticEnd.
      // Native-format constants uploaded later must also avoid those inputs.
      for (const auto &entry : live) {
        const auto occupied = entry.second;
        if (field(occupied, "bank") != 1) continue;
        const auto side = text(occupied, "hemisphere");
        if (side != "both" && text(p, "hemisphere") != "both"
            && side != text(p, "hemisphere")) continue;
        const auto occupiedSlices = slices(occupied);
        if (llvm::any_of(slices(p), [&](I slice) {
              return llvm::is_contained(occupiedSlices, slice);
            }))
          row = std::max(row, field(occupied, "base_row")
                             + field(occupied, "instruction_count"));
      }
      if (row + field(p, "instruction_count") >
          s.target.memory().sram_depth_rows)
        throw std::runtime_error("immutable binding exceeds SRAM: " + name);
      if (weightPingPong && row+field(p,"instruction_count")>weightPolicy.base_row
          && llvm::any_of(slices(p),[&](I slice){return weightSlotSlices.count(slice)!=0;}))
        throw std::runtime_error("immutable binding overlaps weight ping-pong slot: " + name);
      p = s.change(p, J{{"bank", 1}, {"base_row", row}});
      for (auto h : {"east", "west"})
        if (text(p, "hemisphere") == h || text(p, "hemisphere") == "both")
          for (I slice : slices(p))
            staticEnd[{h, slice}] = row + field(p, "instruction_count");
      live[{-2, index}] = p;
      last[{-2, index}] = planner.nodes.size() + 1;
    }
    inputs.push_back(t);
    s.function.insertArgument(s.function.getNumArguments(), type(&s.context, t),
                              mlir::DictionaryAttr{}, s.loc);
    auto binding =
        s.binding(t, p, "input", index, name,
                  s.function.getArgument(s.function.getNumArguments() - 1));
    Static result{p, index, binding};
    statics.emplace(std::move(identity), result);
    return result;
  }
  V packedWords(const V &shape, const V &ix) {
    V v;
    for (I i : ix)
      v.push_back(i % shape[1] / s.target.memory().bytes_per_word * shape[0] + i / shape[1]);
    return v;
  }
  #include "DenseAttentionTransport.inc"
  bool moveWideTranspose(mlir::DictionaryAttr src, mlir::DictionaryAttr dst,
                         const V &sourceShape, const V &destShape,
                         const V &ix, const V &pos) {
    auto wide=s.function->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
    const bool distributed=text(src,"kind")=="fp16_mxm_distributed_16" ||
                           text(src,"kind")=="fp16_vxm_distributed_16";
    const auto sh=text(src,"hemisphere"),dh=text(dst,"hemisphere");
    if(!wide || !wide.getValue() ||
       (!distributed && text(src,"kind")!="fp16_pair_planar") ||
       text(dst,"kind")!="fp16_pair_planar" || slices(src).size()!=(distributed?16:2) ||
       slices(dst).size()!=2 || sourceShape.size()!=2 || destShape.size()!=2 ||
       sourceShape[1]%32 || destShape[1]%32 || ix.empty() || ix.size()%1024 ||
       (distributed && field(src,"bank_tile_rows")))return false;
    std::map<Key,std::map<Key,I>> tiles;
    for(I j=0;j<I(ix.size());++j) {
      const I row=pos[j]/destShape[1],col=pos[j]%destShape[1];
      tiles[{row/32,col/32}][{row%32,col%32}]=ix[j];
    }
    for(const auto &[tile,values]:tiles) {
      if(values.size()!=1024 || !values.count({0,0}))return false;
      const I origin=values.at({0,0}),sr=origin/sourceShape[1],sc=origin%sourceShape[1];
      if(sc%32 || sr+32>sourceShape[0] || sc+32>sourceShape[1] || (distributed && sr%32))return false;
      for(const auto &[rc,index]:values)
        if(index!=(sr+rc.second)*sourceShape[1]+sc+rc.first)return false;
    }
    for(const auto &[tile,values]:tiles) {
      const I origin=values.at({0,0}),sr=origin/sourceShape[1],sc=origin%sourceShape[1];
      const I sourceWord=distributed
          ? (sr/32*(sourceShape[1]/32)+sc/32)*s.target.throughput().tile_rows
          : sc/32*sourceShape[0]+sr;
      const I destWord=tile.second*destShape[0]+tile.first*32;
      const I capture=s.cursor+32;
      I end=capture+96;
      for(I side=0;side<2;++side) {
        const std::string hemi=side?"west":"east";
        if(sh!="both" && sh!=hemi)continue;
        if(sh=="both" && dh!="both" && dh!=hemi)continue;
        const bool cross=dh!="both" && dh!=hemi;
        auto from=s.change(src,J{{"hemisphere",hemi}}),to=s.change(dst,J{{"hemisphere",cross?dh:hemi}});
        s.emit("sxm",J{{"cycle",capture},{"hemisphere",side},{"opcode","transpose"},
            {"source_streams",s.array({16+side*2,17+side*2})},{"destination_streams",s.array({36+side*2,37+side*2})},
            {"permute_map",s.array(range(32))},{"weight_layout","wide_matrix"},
            {"repeat_count",1},{"repeat_interval",32},{"wave_count",1},{"wave_interval",32}});
        // distributed16 stripes token rows over eight byte-plane pairs.
        // The pairs feed the SAME SXM tile at successive capture cycles;
        // no intermediate planar MEM image or lane-zero copy is needed.
        for(I pair=0;pair<(distributed?8:1);++pair)for(I byte=0;byte<2;++byte) {
          const I plane=pair*2+byte;
          const I rd=*s.target.transport_latency(fc::target::StreamEndpoint::Mem,
              fc::target::StreamEndpoint::SxmInput,fc::target::StreamDirection::East,slices(from)[plane]);
          s.mem(capture+pair-rd,from,plane,"read",field(from,"base_row")+sourceWord,
              16+side*2+byte,{distributed?4:32,1,1},{1,0,0},{distributed?8:1,1,1});
        }
        for(I byte=0;byte<2;++byte) {
          // Westbound SXM words can cross the existing passive VXM-input
          // boundary into the opposite outward SR, without a MEM relay.
          const I wr=cross
              ? *s.target.transport_latency(fc::target::StreamEndpoint::SxmResult,
                    fc::target::StreamEndpoint::VxmInput,fc::target::StreamDirection::West,0)
                  +1+slices(to)[byte]/s.target.streams().mem_slices_per_register_group
              : *s.target.transport_latency(fc::target::StreamEndpoint::SxmResult,
                    fc::target::StreamEndpoint::Mem,fc::target::StreamDirection::West,slices(to)[byte]);
          s.mem(capture+35+wr,to,byte,"write",field(to,"base_row")+destWord,(cross?4:36)+side*2+byte,{32,1,1},{1,0,0});
          end=std::max(end,capture+35+wr+36);
        }
      }
      s.cursor=end;
      if(sh!="both" && dh=="both") {
        const auto words=range(32,destWord);
        s.copy(s.change(dst,J{{"hemisphere",sh}}),s.change(dst,J{{"hemisphere",sh=="west"?"east":"west"}}),words,words);
      }
    }
    return true;
  }
  void move(mlir::DictionaryAttr src, mlir::DictionaryAttr dst,
            const V &sourceShape, const V &destShape, const V &ix,
            const V &pos) {
    const I word = s.target.memory().bytes_per_word;
    if(auto access=fc::analysis::decodePlanarRead(src)) {
      const I lanes=scalar(src)?1:word;
      V physical;
      for(I index:ix)physical.push_back(access->address(index/sourceShape[1],
          index%sourceShape[1]/lanes)*lanes+index%lanes);
      mlir::NamedAttrList attrs(src);attrs.erase("planar_read");
      move(attrs.getDictionary(&s.context),dst,{field(src,"instruction_count"),lanes},
           destShape,physical,pos);
      return;
    }
    // A reshape changes the logical type, not the producer's physical matrix.
    if (auto shape = src.getAs<mlir::ArrayAttr>("storage_shape")) {
      V physical;
      for (auto d : shape) physical.push_back(llvm::cast<mlir::IntegerAttr>(d).getInt());
      mlir::NamedAttrList attrs(src);
      attrs.erase("storage_shape");
      const I offset = field(src,"storage_offset");
      attrs.erase("storage_offset");
      V physicalIndices = ix;
      for (auto &index : physicalIndices) index += offset;
      move(attrs.getDictionary(&s.context), dst, physical, destShape, physicalIndices, pos);
      return;
    }
    if (src == dst && sourceShape == destShape && ix == pos)
      return;
    const bool bf16NativePair=text(src,"kind")=="fp16_hemisphere_pair_planar"
        && sourceShape.size()==2 && sourceShape[1]%(2*word)==0
        && text(src,"hemisphere")=="both";
    if (text(src, "kind") == "column_partition_planar" ||
        text(src, "kind") == "fp32_hemisphere_pair_planar" || bf16NativePair) {
      // Treat each hemisphere's interleaved N32 blocks as a normal compact
      // planar matrix before recognizing a wide transpose. This also lets a
      // native BF16 projection feed token-lane QDQ without an archive copy.
      const bool interleaved = text(src, "kind") != "column_partition_planar";
      const I half = sourceShape[1] / 2;
      if (sourceShape[1] % (2 * word) || text(src, "hemisphere") != "both")
        throw std::runtime_error("invalid column-partitioned source");
      for (I side = 0; side < 2; ++side) {
        V local, positions;
        for (size_t i = 0; i < ix.size(); ++i)
          if ((interleaved ? (ix[i] % sourceShape[1]) / word % 2
                           : (ix[i] % sourceShape[1]) / half) == side) {
            const I column = ix[i] % sourceShape[1];
            local.push_back(ix[i] / sourceShape[1] * half +
                (interleaved ? (column / (2 * word)) * word + column % word : column % half));
            positions.push_back(pos[i]);
          }
        if (local.empty()) continue;
        auto plane = s.change(src, J{{"hemisphere", side ? "west" : "east"},
            {"kind", slices(src).size() == 4 ? "fp32_quad_planar" : "fp16_pair_planar"}});
        move(plane, dst, {sourceShape[0], half}, destShape, local, positions);
      }
      return;
    }
    if (auto shape = dst.getAs<mlir::ArrayAttr>("storage_shape")) {
      V physical;
      for (auto d : shape) physical.push_back(llvm::cast<mlir::IntegerAttr>(d).getInt());
      mlir::NamedAttrList attrs(dst);
      attrs.erase("storage_shape");
      const I offset=field(dst,"storage_offset");
      attrs.erase("storage_offset");
      V physicalPositions=pos;
      for (auto &index : physicalPositions) index+=offset;
      move(src,attrs.getDictionary(&s.context),sourceShape,physical,ix,physicalPositions);
      return;
    }
    if(moveDenseAttention(src,dst,sourceShape,destShape,ix,pos)) return;
    if (text(src,"kind") == "fp16_vxm_gamma_broadcast" &&
        text(dst,"kind") == "fp16_vxm_gamma_broadcast") {
      const I sourceRows=field(src,"instruction_count"), destRows=field(dst,"instruction_count");
      auto ss=slices(src), ds=slices(dst);
      for (I first=0;first<I(ix.size());) {
        const I sourceSegment=ix[first]/sourceRows, destSegment=pos[first]/destRows;
        I end=first+1;
        while (end<I(ix.size()) && ix[end]/sourceRows==sourceSegment &&
               pos[end]/destRows==destSegment) ++end;
        V reads,writes;
        for (I i=first;i<end;++i) {reads.push_back(ix[i]%sourceRows);writes.push_back(pos[i]%destRows);}
        auto a=s.change(src,J{{"kind","fp16_scalar_broadcast_planar"},
            {"slices",s.array({ss.at(2*sourceSegment),ss.at(2*sourceSegment+1)})}});
        auto z=s.change(dst,J{{"kind","fp16_scalar_broadcast_planar"},
            {"slices",s.array({ds.at(2*destSegment),ds.at(2*destSegment+1)})}});
        s.copy(a,z,reads,writes);
        first=end;
      }
      return;
    }
    if (text(dst,"kind") == "fp16_mxm_activation_planar") {
      // Both MXM activation ports consume the same K32 vector on every
      // reduction beat. Preserve the full address range in each byte pair;
      // this is replication, unlike O-result pairs which partition N.
      auto ids = slices(dst);
      for (I pair = 0; pair < I(ids.size())/2; ++pair) {
        auto physical = s.change(dst,J{{"kind","fp16_pair_planar"},
            {"slices",s.array({ids[2*pair],ids[2*pair+1]})}});
        move(src,physical,sourceShape,destShape,ix,pos);
      }
      return;
    }
    if (text(dst,"kind") == "fp16_pair_planar" && slices(dst).size() >= 4) {
      // Two packed BF16 pairs per hemisphere partition physical feature words.
      // Four planes here describe
      // two vectors, not the four bytes of an FP32 conversion.
      auto ids = slices(dst);
      const I hemispheres = text(dst,"hemisphere") == "both" ? 2 : 1;
      const I pairs=ids.size()/2;
      for (I h = 0; h < hemispheres; ++h)
      for (I pair = 0; pair < pairs; ++pair) {
        V selected, positions;
        for (size_t i = 0; i < pos.size(); ++i) {
          I col = pos[i] % destShape[1];
          if ((col/word)%pairs != pair
              || (hemispheres == 2 && (col/(pairs*word))%2 != h)) continue;
          selected.push_back(ix[i]);
          positions.push_back((pos[i]/destShape[1])*(destShape[1]/(pairs*hemispheres))+
                              (col/(pairs*word*hemispheres))*word+col%word);
        }
        if (selected.empty()) continue;
        auto physical = s.change(dst,J{{"slices",s.array({ids[2*pair],ids[2*pair+1]})},
            {"hemisphere",hemispheres == 2 ? (h ? "west" : "east") : text(dst,"hemisphere")}});
        move(src,physical,sourceShape,{destShape[0],destShape[1]/(pairs*hemispheres)},selected,positions);
      }
      return;
    }
    if (text(src,"kind") == "fp16_pair_planar" && slices(src).size() >= 4) {
      // Dense attention's canonical O result has two independent BF16
      // plane pairs per hemisphere. Split tensor geometry before transport;
      // four BF16 planes must never be interpreted as one FP32/scalar pair.
      const I hemispheres = text(src,"hemisphere") == "both" ? 2 : 1;
      const I pairs=slices(src).size()/2;
      const I columns = sourceShape[1], localColumns = columns/(pairs*hemispheres);
      auto ids = slices(src);
      for (I h = 0; h < hemispheres; ++h)
        for (I pair = 0; pair < pairs; ++pair) {
          V selected, positions;
          for (size_t i = 0; i < ix.size(); ++i) {
            I col = ix[i]%columns;
            if ((col/word)%pairs != pair || (hemispheres == 2 && (col/(pairs*word))%2 != h)) continue;
            selected.push_back((ix[i]/columns)*localColumns+
                (col/(pairs*word*hemispheres))*word+col%word);
            positions.push_back(pos[i]);
          }
          if (selected.empty()) continue;
          auto physical = s.change(src,J{{"slices",s.array({ids[2*pair],ids[2*pair+1]})},
              {"hemisphere",hemispheres == 2 ? (h ? "west" : "east") : text(src,"hemisphere")}});
          move(physical,dst,{sourceShape[0],localColumns},destShape,selected,positions);
        }
      return;
    }
    if (text(src, "kind") == "token_partition_planar" || text(src,"kind")=="token_partition_east_first_planar") {
      if(auto rowSides=src.getAs<mlir::ArrayAttr>("partition_row_sides")) {
        const auto rowIndices=src.getAs<mlir::ArrayAttr>("partition_row_indices");
        const I columns=sourceShape[1],rows=field(src,"partition_storage_rows");
        for(I side=0;side<2;++side) {
          V selected,positions;
          for(size_t i=0;i<ix.size();++i) {
            const I row=ix[i]/columns;
            if(llvm::cast<mlir::IntegerAttr>(rowSides[row]).getInt()!=side) continue;
            selected.push_back(llvm::cast<mlir::IntegerAttr>(rowIndices[row]).getInt()*columns+ix[i]%columns);
            positions.push_back(pos[i]);
          }
          if(selected.empty()) continue;
          auto physical=s.change(src,J{{"kind","fp16_pair_planar"},{"hemisphere",side?"west":"east"}});
          move(physical,dst,{rows,columns},destShape,selected,positions);
        }
        return;
      }
      const I half = (sourceShape[0] + 1) / 2, columns = sourceShape[1];
      for (I side = 0; side < 2; ++side) {
        V selected, positions;
        for (size_t i = 0; i < ix.size(); ++i)
          if ((ix[i] / columns >= half) == bool(side)) {
            selected.push_back((ix[i] / columns - side * half) * columns + ix[i] % columns);
            positions.push_back(pos[i]);
          }
        if (selected.empty()) continue;
        auto physical = s.change(src, J{{"kind", slices(src).size() == 4 ? "fp32_quad_planar" : "fp16_pair_planar"},
            {"hemisphere", (bool(side) != (text(src,"kind")=="token_partition_east_first_planar")) ? "east" : "west"}});
        move(physical, dst, {half, columns}, destShape, selected, positions);
      }
      return;
    }
    const auto distributed = [](mlir::DictionaryAttr p) {
      return text(p, "kind") == "fp16_mxm_distributed_16" ||
             text(p, "kind") == "fp16_vxm_distributed_16";
    };
    if(moveWideTranspose(src,dst,sourceShape,destShape,ix,pos))return;
    if (distributed(src) && scalar(dst)) {
      // A native distributed word contains 32 features. Recurrence consumes
      // one feature broadcast across value lanes, so unpack every feature,
      // rather than copying only the word-aligned element of each group.
      const auto ids = slices(src);
      struct BroadcastSource { mlir::DictionaryAttr placement; I columns; V indices, offsets; };
      std::vector<BroadcastSource> sources;
      for (I lane = 0; lane < 8; ++lane) {
        V selected, positions;
        for (size_t i = 0; i < ix.size(); ++i) {
          const I row = ix[i] / sourceShape[1], col = ix[i] % sourceShape[1];
          if (row % 8 != lane) continue;
          const I physicalWord = (row / word * (sourceShape[1] / word) + col / word)
              * s.target.throughput().tile_rows + row % word / 8;
          selected.push_back(physicalWord * word + col % word);
          positions.push_back(pos[i]);
        }
        if (selected.empty()) continue;
        auto physical = s.change(src, J{{"kind", "fp16_pair_planar"},
            {"slices", s.array({ids[2*lane], ids[2*lane+1]})},
            {"hemisphere", text(src,"hemisphere") == "both"
                ? text(dst,"hemisphere") : text(src,"hemisphere")}});
        const I columns = (*std::max_element(selected.begin(),selected.end()) / word + 1) * word;
        sources.push_back({physical, columns, std::move(selected), std::move(positions)});
      }
      // The source pairs feed the same SXM streams. Schedule them as one
      // source-pair domain per residue instead of restarting all eight SXM
      // programs for every distributed pair. MEM queues retain their own
      // affine addresses; no FU instruction stream is generated/compressed.
      bool grouped = sources.size() > 1 && slices(dst).size() == 2
          && text(dst,"hemisphere") != "both";
      for (I residue = 0; grouped && residue < 8; ++residue) {
        std::array<I,3> counts{};
        for (size_t n = 0; grouped && n < sources.size(); ++n) {
          const auto &source = sources[n];
          grouped &= text(source.placement,"hemisphere") == text(dst,"hemisphere");
          std::map<I,std::set<I>> lanes;
          V words, offsets;
          for (size_t j = 0; j < source.indices.size(); ++j) {
            I ix = source.indices[j];
            lanes[ix/32].insert(ix%32);
            if (ix%8 == residue) { words.push_back(ix/32); offsets.push_back(source.offsets[j]); }
          }
          for (auto &entry : lanes) grouped &= entry.second.size() == 32;
          Domain domain;
          grouped &= !words.empty() && affine(words, offsets, domain);
          if (!grouped) break;
          if (n == 0) counts = domain.count;
          else grouped &= counts == domain.count;
        }
      }
      if (grouped) {
        for (I residue = 0; residue < 8; ++residue) {
          for (size_t n = 0; n < sources.size(); ++n) {
            const auto &source = sources[n];
            s.unpack(source.placement, dst, {1,source.columns}, source.indices,
                     source.offsets, residue, sources.size(), n == 0);
          }
          s.cursor += 69;
        }
      } else {
        for (const auto &source : sources)
          s.unpack(source.placement, dst, {1,source.columns}, source.indices, source.offsets);
      }
      return;
    }
    if (distributed(src) || distributed(dst)) {
      std::map<Key,Key> words;
      for(size_t i=0;i<ix.size();++i) {
        const I sc=ix[i]%sourceShape[1],dc=pos[i]%destShape[1];
        const Key sourceWord{ix[i]/sourceShape[1],sc/word};
        const Key destWord{pos[i]/destShape[1],dc/word};
        auto [entry,inserted]=words.emplace(destWord,sourceWord);
        if(sc%word!=dc%word || (!inserted && entry->second!=sourceWord))
          throw std::runtime_error("distributed MEM copy requires lane-preserving words: "+s.transportPhase);
      }
      for (I sl = 0; sl < (distributed(src) ? 8 : 1); ++sl)
        for (I dl = 0; dl < (distributed(dst) ? 8 : 1); ++dl) {
          V a, z;
          for (size_t i = 0; i < ix.size(); ++i)
            if (ix[i] % sourceShape[1] % word == 0 &&
                (!distributed(src) ||
                 ix[i] / sourceShape[1] % 8 == sl) &&
                (!distributed(dst) ||
                 pos[i] / destShape[1] % 8 == dl)) {
              a.push_back(ix[i]);
              z.push_back(pos[i]);
            }
          if (a.empty())
            continue;
          auto geometry = [&](mlir::DictionaryAttr p, const V &shape,
                              const V &logical, I lane) {
            if (!distributed(p))
              return std::make_pair(p, packedWords(shape, logical));
            V words;
            for (I l : logical) {
              I row = l / shape[1];
              words.push_back((row / word * (shape[1] / word) + l % shape[1] / word) *
                                  s.target.throughput().tile_rows +
                              row % word / 8);
            }
            auto ids = slices(p);
            p = s.change(
                p, J{{"slices", s.array({ids[2 * lane], ids[2 * lane + 1]})}});
            return std::make_pair(p, words);
          };
          auto from = geometry(src, sourceShape, a, sl),
               to = geometry(dst, destShape, z, dl);
          for (auto h : {"east", "west"})
            if (text(to.first, "hemisphere") == h ||
                text(to.first, "hemisphere") == "both") {
              auto source = s.change(
                  from.first,
                  J{{"hemisphere", text(from.first, "hemisphere") == "both"
                                       ? h
                                       : text(from.first, "hemisphere")}});
              s.copy(source, s.change(to.first, J{{"hemisphere", h}}),
                     from.second, to.second);
            }
        }
      return;
    }
    if (scalar(src) && scalar(dst)) {
      s.copy(src, dst, ix, pos);
      return;
    }
    if (!scalar(src) && scalar(dst)) {
      s.unpack(src, dst, sourceShape, ix, pos);
      return;
    }
    if (!scalar(src)) {
      // A repeated scalar view is not a packed-word copy. Every lane of a
      // destination word names the same source element (e.g. gamma[t,h]
      // expanded over value channels). Extract/broadcast that element with
      // SXM; selecting only source indices divisible by 32 silently drops
      // the other heads and leaves destination words uninitialized.
      bool broadcast = !ix.empty() && ix.size()%word==0 && destShape[1]%word==0;
      V broadcastIndices, broadcastOffsets;
      for(I i=0;broadcast && i<I(ix.size());i+=word) {
        broadcast &= pos[i]%destShape[1]%word==0;
        for(I lane=0;broadcast && lane<word;++lane)
          broadcast &= ix[i+lane]==ix[i] && pos[i+lane]==pos[i]+lane;
        if(broadcast) {
          broadcastIndices.push_back(ix[i]);
          broadcastOffsets.push_back(pos[i]%destShape[1]/word*destShape[0]+pos[i]/destShape[1]);
        }
      }
      if(broadcast) {
        auto wide=s.function->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
        const auto sh=text(src,"hemisphere"),dh=text(dst,"hemisphere");
        const auto ss=slices(src),ds=slices(dst);
        if(wide && wide.getValue() && (ss.size()==2 || ss.size()==4) && ss.size()==ds.size()
            && (sh=="both" || sh==dh)) {
          // Repeat one packed word for a full wide-SXM tile. Its transposed
          // row l is the scalar from source lane l broadcast to all 32 lanes.
          // FP32 uses two byte-pair passes without narrowing. Each additional
          // destination word is another tile issue, never a legacy tile8
          // permute instruction on the wide SXM.
          std::map<I,std::map<I,V>> gathers;
          for(size_t j=0;j<broadcastIndices.size();++j) {
            const I index=broadcastIndices[j],col=index%sourceShape[1];
            const I address=col/word*sourceShape[0]+index/sourceShape[1];
            gathers[address][col%word].push_back(broadcastOffsets[j]);
          }
          for(const auto &[address,lanes]:gathers) {
            I copies=0;for(const auto &[lane,offsets]:lanes)copies=std::max<I>(copies,offsets.size());
            for(I copy=0;copy<copies;++copy) {
              V beats,offsets;
              for(const auto &[lane,places]:lanes)if(copy<I(places.size())) {
                beats.push_back(lane);offsets.push_back(places[copy]);
              }
              Domain write;
              if(!affine(beats,offsets,write))
                throw std::runtime_error("non-affine wide scalar broadcast destinations");
              auto intervals=write.src;
              for(auto &interval:intervals)interval=std::max<I>(1,interval);
              for(I plane=0;plane<I(ss.size());plane+=2) {
                const I capture=s.cursor+32;I end=capture+96;
                for(I side=0;side<2;++side) {
                  const std::string hemi=side?"west":"east";
                  if(dh!="both" && dh!=hemi)continue;
                  auto from=s.change(src,J{{"hemisphere",hemi}}),to=s.change(dst,J{{"hemisphere",hemi}});
                  s.emit("sxm",J{{"cycle",capture},{"hemisphere",side},{"opcode","transpose"},
                      {"source_streams",s.array({16+side*2,17+side*2})},
                      {"destination_streams",s.array({36+side*2,37+side*2})},
                      {"permute_map",s.array(range(32))},{"weight_layout","wide_matrix"},
                      {"repeat_count",1},{"repeat_interval",32},{"wave_count",1},{"wave_interval",32}});
                  for(I byte=0;byte<2;++byte) {
                    const I rd=*s.target.transport_latency(fc::target::StreamEndpoint::Mem,
                        fc::target::StreamEndpoint::SxmInput,fc::target::StreamDirection::East,ss[plane+byte]);
                    const I wr=*s.target.transport_latency(fc::target::StreamEndpoint::SxmResult,
                        fc::target::StreamEndpoint::Mem,fc::target::StreamDirection::West,ds[plane+byte]);
                    s.mem(capture-rd,from,plane+byte,"read",field(src,"base_row")+address,
                        16+side*2+byte,{32,1,1},{0,0,0});
                    s.mem(capture+35+wr+beats[0],to,plane+byte,"write",field(dst,"base_row")+offsets[0],
                        36+side*2+byte,write.count,write.dst,intervals);
                    end=std::max(end,capture+35+wr+beats.back()+4);
                  }
                }
                s.cursor=end;
              }
              // All SXM output words and MEM writes have drained. Large
              // State broadcasts must permit physical iMEM reloads instead
              // of exceeding the finite transpose/read/write queues.
              s.emit("timeline",J{{"name","imem.drain.scalar_broadcast"},
                  {"start",s.cursor-1},{"end",s.cursor}});
            }
          }
          return;
        }
        s.unpack(src,dst,sourceShape,broadcastIndices,broadcastOffsets);
        return;
      }
      I columns = sourceShape[1], rows = sourceShape[0];
      // Passive SR transport copies complete words; it cannot move one
      // source lane into a different destination lane. In particular a
      // flattened/reshaped transpose must not fall through here when the
      // full-tile SXM recognizer above cannot represent its source geometry.
      // Validate each touched destination word before selecting lane zero
      // to emit the copy. A short final word may contain undefined padding,
      // but every logical lane of that word must come from the same source.
      struct WordCopy { I source=-1; std::set<I> lanes; };
      std::map<I,WordCopy> wordCopies;
      for(size_t i=0;i<ix.size();++i) {
        const I sourceLane=ix[i]%columns%word;
        const I destinationColumn=pos[i]%destShape[1];
        const I destinationLane=destinationColumn%word;
        const I sourceWord=(ix[i]%columns/word)*rows+ix[i]/columns;
        const I destinationWord=(destinationColumn/word)*destShape[0]+pos[i]/destShape[1];
        auto &copy=wordCopies[destinationWord];
        if(sourceLane!=destinationLane || (copy.source>=0 && copy.source!=sourceWord))
          throw std::runtime_error("packed MEM copy requires lane-preserving words; materialize the logical matrix before transpose: "+s.transportPhase);
        copy.source=sourceWord;copy.lanes.insert(destinationLane);
      }
      for(const auto &[destination,copy]:wordCopies) {
        const I column=destination/destShape[0]*word;
        const I validLanes=std::min<I>(word,destShape[1]-column);
        if(I(copy.lanes.size())!=validLanes || !copy.lanes.count(0))
          throw std::runtime_error("packed MEM copy cannot write a partial logical word: "+s.transportPhase);
      }
      bool paired = text(src, "kind") == "fp16_hemisphere_pair_planar";
      // For a non-partitioned packed source the east and west iterations
      // write exactly the same destination words. West was the last writer.
      // Elide the overwritten east transfer only for disjoint physical
      // queues; in-place moves may need the original snapshot ordering.
      const auto elide=planner.policies.getAs<mlir::BoolAttr>("ftlpu.elide_overwritten_packed_copies");
      const bool eastOverwritten = (!elide || elide.getValue()) && !paired &&
          text(src,"hemisphere")=="both" && !overlap(src,dst);
      for (auto h : {"east", "west"})
        if (!(eastOverwritten && llvm::StringRef(h)=="east") &&
            (text(src, "hemisphere") == h || text(src, "hemisphere") == "both")) {
          V words, positions;
          for (size_t i = 0; i < ix.size(); ++i)
            if (ix[i] % columns % word == 0 &&
                (!paired ||
                 ix[i] % columns / word % 2 == I(std::string(h) == "west"))) {
              words.push_back((ix[i] % columns / (paired ? 2*word : word)) * rows +
                              ix[i] / columns);
              positions.push_back(pos[i]);
            }
          if (words.empty())
            continue;
          auto offsets = packedWords(destShape, positions);
          // An inward SR passes the nearer local MEM before crossing to the
          // other hemisphere. Both writes can tap one read of each byte plane.
          auto from = s.change(src, J{{"hemisphere", h}});
          const auto multicast=planner.policies.getAs<mlir::BoolAttr>("ftlpu.passive_sr_multicast");
          if (((multicast && multicast.getValue()) || s.consumerMulticast) && text(dst,"hemisphere")=="both" && !overlap(from,dst) &&
              s.sameCopyFormat(from,dst)) {
            const auto sourceSlices=slices(from), destSlices=slices(dst);
            bool inward=true;
            for (size_t p=0;p<sourceSlices.size();++p)
              inward &= sourceSlices[p]/s.group > destSlices[p]/s.group;
            Domain domain;
            if (inward && affine(words,offsets,domain)) {
              auto far=s.change(dst,J{{"hemisphere",llvm::StringRef(h)=="west"?"east":"west"}});
              s.scalarDomain(from,far,words.front(),offsets.front(),domain,true);
              continue;
            }
          }
          for (auto dh : {"east", "west"})
            if (text(dst, "hemisphere") == dh ||
                text(dst, "hemisphere") == "both")
              s.copy(s.change(src, J{{"hemisphere", h}}),
                     s.change(dst, J{{"hemisphere", dh}}), words, offsets);
        }
      return;
    }
    std::set<I> rr, cc;
    for (I p : pos) {
      rr.insert(p / destShape[1]);
      cc.insert(p % destShape[1]);
    }
    if (rr.size() * cc.size() != pos.size() || cc.size() % 32)
      throw std::runtime_error(
          "scalar packing requires rectangular tensor view");
    V order = range(pos.size());
    std::sort(order.begin(), order.end(),
              [&](I a, I b) { return pos[a] < pos[b]; });
    V sorted, first;
    for (I o : order)
      sorted.push_back(ix[o]);
    for (I c = 0; c < I(cc.size()); c += 32)
      for (I row = 0; row < I(rr.size()); ++row)
        first.push_back(pos[order[row * cc.size() + c]]);
    s.pack(src, dst, {I(rr.size()), I(cc.size())}, sorted,
           packedWords(destShape, first));
  }
  // A concat of native N pages may already be one physical matrix. Prove
  // every element's lane, hemisphere and address before replacing its view.
  mlir::DictionaryAttr contiguousPairView(const Tensor &value) {
    if(value.refs.empty() || value.shape.size()!=2 || value.shape[1]%64
        || (value.dtype!="bf16" && value.dtype!="f32"))return {};
    const auto first=value.refs.front();
    if(first.node<0 || first.index!=0 || !cache.count({first.node,first.output}))return {};
    auto native=cache.at({first.node,first.output});
    const auto kind=text(native,"kind");
    const I rows=value.shape[0],columns=value.shape[1];
    if(kind!=(value.dtype=="bf16"?"fp16_hemisphere_pair_planar":"fp32_hemisphere_pair_planar")
        || text(native,"hemisphere")!="both" || slices(native).size()!=size_t(value.bytes())
        || field(native,"address_stride")!=1 || field(native,"bank_tile_rows")
        || native.get("storage_shape"))return {};
    for(I j=0;j<value.size();++j) {
      const auto ref=value.refs[j];auto found=cache.find({ref.node,ref.output});
      if(ref.node<0 || found==cache.end())return {};
      const auto &producer=planner.nodes[ref.node].outputs[ref.output];
      const auto &shape=producer.shape;auto source=found->second;
      if(producer.dtype!=value.dtype || shape.size()!=2 || shape[0]!=rows || shape[1]%64
          || text(source,"kind")!=kind || text(source,"hemisphere")!="both"
          || slices(source)!=slices(native) || field(source,"bank")!=field(native,"bank")
          || field(source,"address_stride")!=1 || field(source,"bank_tile_rows")
          || source.get("storage_shape") || ref.index<0 || ref.index>=producer.size())return {};
      const I column=j%columns,sourceColumn=ref.index%shape[1];
      if(ref.index/shape[1]!=j/columns || sourceColumn%32!=column%32
          || sourceColumn/32%2!=column/32%2
          || field(source,"base_row")+sourceColumn/64*rows+ref.index/shape[1]
              !=field(native,"base_row")+column/64*rows+j/columns)return {};
    }
    if(field(native,"base_row")+value.size()/64>s.target.memory().words_per_bank)return {};
    return s.change(native,J{{"instruction_count",value.size()/64}});
  }
  void materialize(const Tensor &value, mlir::DictionaryAttr dest,
                   bool allowSerialDrains = false) {
    if(s.consumerMulticast) if(auto home=contiguousPairView(value)) {
      const I begin=s.cursor;
      move(home,dest,value.shape,value.shape,range(value.size()),range(value.size()));
      if(s.cursor>begin)s.emit("timeline",J{{"name","mem_materialize.native_column_view"},
          {"start",begin},{"end",s.cursor}});
      return;
    }
    std::map<Key, std::pair<V, V>> segments;
    for (size_t i = 0; i < value.refs.size(); ++i) {
      auto ref = value.refs[i];
      auto &p = segments[{ref.node, ref.output}];
      p.first.push_back(ref.index);
      p.second.push_back(i);
    }
    for (auto &kv : segments) {
      const I begin=s.cursor;
      auto source=cache.at(kv.first);
      const auto &producer=planner.nodes[kv.first.first].outputs[kv.first.second];
      // indexed_view truncate_bf16 is a byte-plane view, including when a
      // later GEMM materializes a non-contiguous tile of the State. Preserve
      // that contract instead of invoking an arithmetic FP32->BF16 bridge.
      if (value.dtype=="bf16" && producer.dtype=="f32") {
        const auto ids=slices(source);auto kind=text(source,"kind");
        if (ids.size()!=4 || (kind!="fp32_quad_planar" &&
            kind!="fp32_scalar_broadcast_planar" && kind!="token_partition_planar" &&
            kind!="token_partition_east_first_planar"))
          throw std::runtime_error("BF16 truncating view requires FP32 byte-plane storage");
        if(kind=="fp32_quad_planar")kind="fp16_pair_planar";
        if(kind=="fp32_scalar_broadcast_planar")kind="fp16_scalar_broadcast_planar";
        source=s.change(source,J{{"kind",kind},{"slices",s.array({ids[2],ids[3]})}});
      }
      move(source, dest, producer.shape,
           value.shape, kv.second.first, kv.second.second);
      if(s.cursor>begin) s.emit("timeline",J{{"name", "mem_materialize."+planner.nodes[kv.first.first].name},
          {"start",begin},{"end",s.cursor}});
      // A large concatenation may require many independent scalar-to-vector
      // packing programs. At a caller-proven serialized boundary, retain the
      // final destination and reload iMEM between complete producer domains.
      // This adds no tensor copy and never cuts an executing ICU instruction.
      if (allowSerialDrains && segments.size() > 1 && s.cursor > begin) {
        s.cursor += 96;
        s.emit("timeline", J{{"name", "imem.drain.materialize." +
            std::to_string(kv.first.first) + "." + std::to_string(s.cursor)},
            {"start", s.cursor - 1}, {"end", s.cursor}});
        s.cursor += 96;
      }
    }
  }
  // Whole-subgraph placement, analogous to the dense attention memory plan:
  // native operator tasks receive final storage before their ICU domains exist.
  void reserveRecurrentStates() {
    for (I ni = 0; ni < I(planner.nodes.size()); ++ni) {
      auto &node = planner.nodes[ni];
      if (!node.recurrent)
        continue;
      const auto &state = node.inputs[5];
      bool share = !state.refs.empty();
      I prior = share ? state.refs[0].node : -1;
      for (I j = 0; j < I(state.refs.size()); ++j)
        share &= state.refs[j].node == prior && state.refs[j].output == 1 &&
                 state.refs[j].index == j;
      share &= recurrentStates.count(prior) && last[{prior, 1}] == ni;
      if (share)
        recurrentStates[ni] = recurrentStates.at(prior);
      else {
        // Reserve carried FP32 state before scalar vectors fragment a bank.
        // Branches and externally retained snapshots need separate storage.
        recurrentStates[ni] =
            allocate({-4, ni}, node.outputs[1], text(node.in[5],"hemisphere"), 1, node.out[1]);
        last[{-4, ni}] = planner.nodes.size() + 1;
      }
    }
  }
  // A long-lived normalized tensor can occupy the consumers' native input
  // area after RMSNorm has finished. Its final transpose still needs a
  // separate bank from the normalization scratch, so retain the native drain
  // area for that phase and plan an explicit on-chip transfer afterwards.
  bool planRetainedRmsNorm(I ni, I oi) {
    auto &node = planner.nodes[ni];
    const Key key{ni, oi};
    if (!llvm::isa<fc::tensor::RmsNormTaskOp>(node.outputSlots[oi].op)
        || node.parallel >= 0 || last[key] <= ni) return false;
    if (auto config=node.outputSlots[oi].op->getAttrOfType<mlir::DictionaryAttr>("config"))
      if (auto planar=config.getAs<mlir::BoolAttr>("planar_streaming"); planar && planar.getValue())
        return false;
    const auto rowsOverlap = [](mlir::DictionaryAttr a, mlir::DictionaryAttr b) {
      return overlap(a,b)
          && field(a,"base_row") < field(b,"base_row") + field(b,"instruction_count")
          && field(b,"base_row") < field(a,"base_row") + field(a,"instruction_count");
    };
    const auto consumesWholeResult = [&](const Tensor &t) {
      if (t.shape != node.outputs[oi].shape || t.refs.size() != size_t(t.size())) return false;
      for (I j=0; j<t.size(); ++j)
        if (t.refs[j].node != ni || t.refs[j].output != oi || t.refs[j].index != j)
          return false;
      return true;
    };
    auto drain = node.out[oi];
    for (I ci=ni+1; ci<=last[key] && ci<I(planner.nodes.size()); ++ci) {
      auto &consumer = planner.nodes[ci];
      bool attentionInput = false;
      if (consumer.function)
        consumer.function.walk([&](fc::tensor::ProjectionTaskOp projection) {
          attentionInput |= projection.getKind() == "query";
        });
      if (!consumer.ffn && !attentionInput) continue;
      for (I ai=0; ai<I(consumer.in.size()); ++ai) {
        auto home = consumer.in[ai];
        if (!consumesWholeResult(consumer.inputs[ai])
            || text(home,"kind") != text(drain,"kind")
            || text(home,"hemisphere") != text(drain,"hemisphere")
            || slices(home).size() != slices(drain).size()
            || field(home,"instruction_count") != field(drain,"instruction_count")
            || overlap(home,drain)) continue;
        bool safe = true;
        for (const auto &entry : live) {
          const auto use = last.find(entry.first);
          if (use == last.end() || use->second > ni)
            safe &= !rowsOverlap(home,entry.second) && !rowsOverlap(drain,entry.second);
        }
        // The native drain must remain independent of the final transpose's
        // source. Earlier RMSNorm feedback storage may be reused at this point.
        auto rms = llvm::cast<fc::tensor::RmsNormTaskOp>(node.outputSlots[oi].op);
        auto normalized = llvm::cast<mlir::DictionaryAttr>(rms.getScratchAllocations()[1])
            .getAs<mlir::DictionaryAttr>("placement");
        safe &= !overlap(drain,normalized);
        for (I future=ni+1; safe && future<=last[key]
             && future<I(planner.nodes.size()); ++future) {
          const auto &next = planner.nodes[future];
          for (auto workspace : next.workspaces) {
            bool alias = false;
            for (I operand=0; operand<I(next.in.size()); ++operand) {
              const bool movableStaging = !next.inputs[operand].external()
                  && llvm::all_of(next.inputSlots[operand], [](const auto &slot) {
                       return llvm::isa<fc::kernel::VectorAluOp>(slot.op);
                     });
              alias |= workspace == next.in[operand]
                  && (consumesWholeResult(next.inputs[operand]) || movableStaging);
            }
            if (!alias && rowsOverlap(home,workspace)) { safe=false; break; }
          }
        }
        if (!safe) continue;
        cache[key] = live[key] = home;
        retainedOutputHomes[key] = home;
        return true;
      }
    }
    return false;
  }
  bool planRetainedAttention(I ni, I oi) {
    auto &node = planner.nodes[ni];
    const Key key{ni,oi};
    auto projection = llvm::dyn_cast<fc::tensor::ProjectionTaskOp>(node.outputSlots[oi].op);
    if (!projection || projection.getKind() != "output" || last[key] <= ni
        || node.outputs[oi].shape[0] < s.target.throughput().mxm_rows) return false;
    const auto drain = node.out[oi];
    bool overwritten = false;
    for (I future=ni+1; future<=last[key] && future<I(planner.nodes.size()); ++future)
      for (auto p : planner.nodes[future].workspaces)
        overwritten |= overlap(drain,p)
            && field(drain,"base_row") < field(p,"base_row") + field(p,"instruction_count")
            && field(p,"base_row") < field(drain,"base_row") + field(drain,"instruction_count");
    if (!overwritten) return false;
    for (const auto &entry : live)
      if (overlap(drain,entry.second)
          && field(drain,"base_row") < field(entry.second,"base_row") + field(entry.second,"instruction_count")
          && field(entry.second,"base_row") < field(drain,"base_row") + field(drain,"instruction_count"))
        throw std::runtime_error("attention native result overlaps a retained graph value");
    const I tile=s.target.throughput().mxm_rows;
    const auto &shape=node.outputs[oi].shape;
    const I rows=((shape[0]+tile-1)/tile)*(shape[1]/tile)*s.target.throughput().tile_rows;
    // Retain partial O results in the standard distributed layout. The copy
    // runs after O drains, before another head batch reuses native scratch.
    auto prototype=s.placement("fp16_mxm_distributed_16",range(16,s.target.memory().slices_per_hemisphere-16),"west",1,0,rows);
    mlir::DictionaryAttr home;
    try { home=allocate(key,node.outputs[oi],"west",1,prototype,{drain}); }
    catch (const std::runtime_error &) {
      prototype=s.change(prototype,J{{"hemisphere","east"}});
      home=allocate(key,node.outputs[oi],"east",1,prototype,{drain});
    }
    retainedOutputHomes[key]=home;
    return true;
  }
  bool planInPlacePointwise(I ni, I oi) {
    auto &node=planner.nodes[ni];
    if (node.parallel>=0 || !llvm::isa<fc::kernel::VectorAluOp>(node.outputSlots[oi].op)) return false;
    auto output=llvm::cast<fc::kernel::VectorAluOp>(node.outputSlots[oi].op);
    if (output.getReduce()) return false;
    const Key key{ni,oi};
    for (const auto &input : node.inputs) {
      if (input.external() || input.shape!=node.outputs[oi].shape || input.refs.empty()) continue;
      const auto first=input.refs.front();
      const Key parent{first.node,first.output};
      auto home=cache.at(parent);
      if (text(home,"kind")!="fp16_mxm_distributed_16" &&
          !(text(home,"kind")=="fp16_pair_planar" && slices(home).size()>=4)) continue;
      bool contiguous=true;
      for (I i=0;i<input.size();++i)
        contiguous &= input.refs[i].node==first.node && input.refs[i].output==first.output && input.refs[i].index==first.index+i;
      if (!contiguous) continue;
      // Only overwrite elements whose old version has no remaining reader.
      const auto usesOldElements=[&](const Tensor &v) {
        return llvm::any_of(v.refs,[&](const Ref &r) {return r.node==first.node && r.output==first.output &&
            r.index>=first.index && r.index<first.index+input.size();});
      };
      bool safe=true;
      for (I future=ni+1;safe && future<I(planner.nodes.size());++future)
        for (const auto &value : planner.nodes[future].inputs) safe &= !usesOldElements(value);
      for (const auto &value : finals) safe &= !usesOldElements(value);
      for (auto h : {"east","west"})
        if (text(home,"hemisphere")==h || text(home,"hemisphere")=="both")
          for (I slice : slices(home))
            for (auto r : nativeTimed[{h,slice}])
              if (r.node>ni && r.node<=last[key] && field(home,"base_row")<r.end &&
                  r.begin<field(home,"base_row")+field(home,"instruction_count")) safe=false;
      if (!safe) continue;
      V physical=planner.nodes[first.node].outputs[first.output].shape;
      if (auto shape=home.getAs<mlir::ArrayAttr>("storage_shape")) {
        physical.clear();
        for (auto d : shape) physical.push_back(llvm::cast<mlir::IntegerAttr>(d).getInt());
      }
      home=s.change(home,J{{"storage_shape",s.array(physical)},
          {"storage_offset",field(home,"storage_offset")+first.index}});
      // Drain to a short-lived planar scratch, then commit this row view.
      // The next chunk can reuse the scratch after this copy retires.
      const Key scratch{ni,-100-oi};
      last[scratch]=ni;
      auto busy=node.in;
      busy.insert(busy.end(),node.workspaces.begin(),node.workspaces.end());
      auto drain=allocate(scratch,node.outputs[oi],text(node.out[oi],"hemisphere"),1,node.out[oi],busy);
      node.out[oi]=drain;
      node.outputSlots[oi].set(drain);
      cache[key]=live[key]=home;
      retainedOutputHomes[key]=home;
      return true;
    }
    return false;
  }
  bool planGroupedPointwise(I ni, I oi) {
    auto &node=planner.nodes[ni];
    const I word=s.target.memory().bytes_per_word;
    if (node.pointwiseGroup<0 || node.outputs.size()!=1 || node.outputs[oi].dtype!="bf16" ||
        node.pointwiseShape.size()!=2 || node.pointwiseShape[1]%(8*word)) return false;
    // A following RMSNorm can consume each native row tile directly. A
    // multi-pair grouped archive would force a distributed16 conversion
    // and pin all activation queues during the reduction.
    for(auto &consumer:planner.nodes) {
      if(!consumer.module || consumer.inputs.empty() ||
         !llvm::any_of(consumer.inputs[0].refs,[&](const Ref &ref){return ref.node==ni && ref.output==oi;}))continue;
      bool norm=false;
      consumer.function.walk([&](fc::tensor::RmsNormTaskOp){norm=true;});
      if(norm)return false;
    }
    const Key key{ni,oi}, group{node.pointwiseGroup,oi};
    mlir::DictionaryAttr home;
    if (pointwiseHomes.count(group)) home=pointwiseHomes.at(group);
    else {
      auto tensor=node.outputs[oi]; tensor.shape=node.pointwiseShape;
      I finalUse=last[key];
      for (I i=ni;i<I(planner.nodes.size());++i)
        if (planner.nodes[i].pointwiseGroup==node.pointwiseGroup)
          finalUse=std::max(finalUse,last[{i,oi}]);
      const Key reservation{-4,ni};
      last[reservation]=finalUse;
      auto prototype=s.placement("fp16_pair_planar",range(8),"both",1,0,tensor.size()/(8*word));
      home=allocate(reservation,tensor,"both",1,prototype);
      pointwiseHomes[group]=home;
    }
    home=s.change(home,J{{"storage_shape",s.array(node.pointwiseShape)},
        {"storage_offset",node.pointwiseRow*node.pointwiseShape[1]}});
    const Key scratch{ni,-100-oi}; last[scratch]=ni;
    auto busy=node.in;
    busy.insert(busy.end(),node.workspaces.begin(),node.workspaces.end());
    auto drain=allocate(scratch,node.outputs[oi],text(node.out[oi],"hemisphere"),1,node.out[oi],busy);
    node.out[oi]=drain; node.outputSlots[oi].set(drain);
    cache[key]=live[key]=home; retainedOutputHomes[key]=home;
    return true;
  }
  std::vector<mlir::DictionaryAttr> projectionSolveQueues(I ni) {
    const auto& node=planner.nodes[ni];
    auto queues=node.in;
    queues.insert(queues.end(),node.out.begin(),node.out.end());
    queues.insert(queues.end(),node.workspaces.begin(),node.workspaces.end());
    for(I operand=0;operand<I(node.inputs.size());++operand) {
      if(auto external=externalVectorHomes.find({ni,operand});external!=externalVectorHomes.end())
        queues.push_back(external->second);
      for(const auto& ref:node.inputs[operand].refs)
        if(auto home=cache.find({ref.node,ref.output});home!=cache.end()
            && std::find(queues.begin(),queues.end(),home->second)==queues.end())
          queues.push_back(home->second);
    }
    return queues;
  }
  void reserveProjectionConvQueues(I ni, std::vector<mlir::DictionaryAttr>& busy) {
    const I page=planner.nodes[ni].projectionConvGroup;
    if(page<0)return;
    if(ni==page) {
      // A completed projection page feeds the concurrent depthwise wave.
      // Distinct SRAM rows on the same slice and bank still share an ICU.
      for(I j=page+1;j<I(planner.nodes.size()) && planner.nodes[j].projectionConvGroup==page;++j) {
        const auto& conv=planner.nodes[j];
        if(conv.projectionVectorMember) {
          auto used=projectionSolveQueues(j);
          busy.insert(busy.end(),used.begin(),used.end());
          // Layout copies may temporarily route through the linker scratch.
          for(I bank=0;bank<s.target.memory().banks_per_slice;++bank)
            busy.push_back(s.placement("reserved_linker_queue",range(4,48),"both",bank,0,1));
          continue;
        }
        if(auto tap=externalVectorHomes.find({j,1});tap!=externalVectorHomes.end())
          busy.push_back(tap->second);
        for(const auto& ref:conv.inputs.front().refs) {
          auto home=cache.find({ref.node,ref.output});
          if(home!=cache.end() && !llvm::is_contained(busy,home->second))busy.push_back(home->second);
        }
        for(auto p:conv.workspaces)busy.push_back(p);
      }
      return;
    }
    const auto& projection=planner.nodes[page];
    busy.insert(busy.end(),projection.out.begin(),projection.out.end());
    busy.insert(busy.end(),projection.in.begin(),projection.in.end());
    if(weightPingPong && !weightSlotSlices.empty()) {
      const V slot(weightSlotSlices.begin(),weightSlotSlices.end());
      for(I bank=0;bank<s.target.memory().banks_per_slice;++bank)
        busy.push_back(s.placement("reserved_weight_queue",slot,"both",bank,0,1));
    }
  }
  std::vector<mlir::DictionaryAttr> mirroredJoinQueues(I ni) {
    std::vector<mlir::DictionaryAttr> queues;
    for(const auto &consumer:planner.nodes) {
      if(!consumer.module || !consumer.function->hasAttr("ftlpu.vector_independent_ports"))continue;
      bool used=false;
      for(const auto &operand:consumer.inputs)for(const auto &ref:operand.refs)used|=ref.node==ni;
      if(!used)continue;
      for(const auto &operand:consumer.inputs) {
        if(llvm::any_of(operand.refs,[&](const Ref &ref){return ref.node==ni;}))continue;
        for(const auto &ref:operand.refs)
          if(auto found=cache.find({ref.node,ref.output});found!=cache.end() &&
             std::find(queues.begin(),queues.end(),found->second)==queues.end())queues.push_back(found->second);
      }
    }
    return queues;
  }
  // A mirrored distributed16 producer can supply both sides of the MXM
  // activation port in token order. Preserve its entire allocated footprint
  // for queue exclusion and attach the start row of each logical head.
  bool planDistributedMatmulLhs(I ni, const std::vector<mlir::DictionaryAttr> &otherQueues) {
    const auto &node=planner.nodes[ni];
    const auto &value=node.inputs[0];
    if(!node.matmulLhsJoin || value.shape!=V{64,32} || value.dtype!="bf16" ||
       value.refs.size()!=2048 || s.target.throughput().tile_rows!=4)return false;
    const auto origin=value.refs.front();
    const Key producer{origin.node,origin.output}, result{ni,0};
    const auto home=cache.at(producer);
    const auto &tensor=planner.nodes[origin.node].outputs[origin.output];
    if(text(home,"kind")!="fp16_mxm_distributed_16" || text(home,"hemisphere")!="both" ||
       slices(home).size()!=16 || field(home,"bank_tile_rows") ||
       tensor.dtype!="bf16" ||
       last.at(producer)<last.at(result))return false;
    if(llvm::any_of(otherQueues,[&](auto p){return overlap(p,home);}))return false;
    // A K-tile epilogue can select either logical head on each side. Both
    // RHS heads must also be available there. Prepared mirrored RHS joins
    // guarantee that contract even when their addresses are assigned later.
    for(I user=ni+1;user<=last.at(result) && user<I(planner.nodes.size());++user) {
      const auto &consumer=planner.nodes[user];
      if(consumer.inputs.empty() || consumer.inputs[0].refs.empty() ||
         consumer.inputs[0].refs.front().node!=ni)continue;
      if(consumer.inputs.size()<2 || consumer.inputs[1].refs.empty())return false;
      const auto rhs=consumer.inputs[1].refs.front();
      if(!planner.nodes[rhs.node].mirroredJoin)return false;
    }

    // Grouped norm keeps its logical [token*head, feature] result type while
    // writing [token, head*feature] physical rows. Use the storage view in
    // the same way as move() when interpreting its flattened references.
    V physical=tensor.shape;
    if(auto storage=home.getAs<mlir::ArrayAttr>("storage_shape")) {
      physical.clear();
      for(auto dimension:storage)physical.push_back(llvm::cast<mlir::IntegerAttr>(dimension).getInt());
    }
    if(physical.size()!=2 || physical[1]<=0 || physical[1]%32)return false;
    const I columns=physical[1], offset=field(home,"storage_offset");
    V baseRows;
    for(I head=0;head<2;++head) {
      const auto first=value.refs[head*1024];
      const I firstRow=(first.index+offset)/columns, column=(first.index+offset)%columns;
      if(firstRow<0 || firstRow%32 || firstRow+32>physical[0] || column<0 || column%32)return false;
      for(I token=0;token<32;++token)for(I lane=0;lane<32;++lane) {
        const auto ref=value.refs[head*1024+token*32+lane];
        if(ref.node!=origin.node || ref.output!=origin.output ||
           ref.index+offset!=(firstRow+token)*columns+column+lane)return false;
      }
      baseRows.push_back(field(home,"base_row")+(firstRow/32*(columns/32)+column/32)*4);
    }
    cache[result]=live[result]=s.change(home,J{{"mxm_lhs_head_rows",s.array(baseRows)}});
    return true;
  }
  // Fold a sole, full-tile transpose consumer into a packed BF16 producer.
  // This is a physical edge fusion before schedule lowering: the VXM cast
  // retains its rounding, wide SXM captures its SR output, and the consumer
  // owns the resulting matrix layout. No FU trace is reconstructed.
  bool planStreamingMatmulLhs(I ni, const std::vector<mlir::DictionaryAttr> &otherQueues) {
    auto &node=planner.nodes[ni];
    const auto &value=node.inputs[0];
    const I tile=s.target.throughput().mxm_rows;
    const auto wide=planner.policies.getAs<mlir::BoolAttr>("ftlpu.sxm_wide");
    if(!wide || !wide.getValue() || !node.matmulLhsJoin || tile!=32 ||
       value.dtype!="bf16" || value.shape!=V{2*tile,tile} ||
       value.refs.size()!=size_t(2*tile*tile))return false;
    const auto origin=value.refs.front();
    if(origin.node!=ni-1 || origin.output!=0 || origin.index!=0)return false;
    auto &producer=planner.nodes[origin.node];
    if(!producer.module || producer.outputs.size()!=1 || producer.parallel>=0 ||
       producer.outputs[0].shape!=V{tile,2*tile} ||
       !producer.function->hasAttr("ftlpu.vector_bf16_spill"))return false;
    auto ops=producer.function.getOps<fc::kernel::VectorAluOp>();
    if(std::distance(ops.begin(),ops.end())!=1)return false;
    auto cast=*ops.begin();
    if(cast.getKind()!="multiply" || !cast.getUseImmediate() || cast.getReduce() ||
       cast.getImmediate().convertToFloat()!=1.f ||
       !cast.getResult().getType().getElementType().isBF16())return false;
    for(I side=0;side<2;++side)for(I q=0;q<tile;++q)for(I k=0;k<tile;++k) {
      const auto ref=value.refs[(side*tile+q)*tile+k];
      if(ref.node!=origin.node || ref.output!=0 || ref.index!=k*2*tile+side*tile+q)return false;
    }
    for(I user=0;user<I(planner.nodes.size());++user)if(user!=ni)
      for(const auto &input:planner.nodes[user].inputs)
        if(llvm::any_of(input.refs,[&](const Ref &ref){return ref.node==origin.node;}))return false;
    for(const auto &result:finals)
      if(llvm::any_of(result.refs,[&](const Ref &ref){return ref.node==origin.node;}))return false;
    const Key source{origin.node,0}, result{ni,0};
    const auto home=cache.at(source);
    if(text(home,"kind")!="fp16_pair_planar" || text(home,"hemisphere")!="both" ||
       slices(home).size()!=2 || field(home,"instruction_count")!=2*tile ||
       llvm::any_of(otherQueues,[&](auto p){return overlap(p,home);}) ||
       llvm::any_of(producer.in,[&](auto p){return overlap(p,home);}))return false;
    // The producer's allocation originally ended at this join. Check the
    // longer consumer lifetime against fixed native workspaces before
    // retaining it; later dynamic allocations see the live alias below.
    const I begin=field(home,"base_row"), end=begin+field(home,"instruction_count");
    for(auto side:{"east","west"})for(I slice:slices(home)) {
      const auto &ranges=field(home,"bank")==0?nativeBank0[{side,slice}]:nativeTimed[{side,slice}];
      for(const auto &range:ranges)
        if(range.node>origin.node && range.node<=last.at(result) && begin<range.end && range.begin<end)
          return false;
    }
    // The two views have the same footprint. Extend this home through all
    // PV users rather than allocating a second transposition buffer.
    cache[result]=live[result]=home;
    node.streamedTransposeJoin=true;
    cast->setAttr("ftlpu.vector_result_tile_transpose",s.b.getUnitAttr());
    return true;
  }
  void reserveAttentionInitializers() {
    // Native masks are uploaded before the entire graph starts, not when the
    // attention node starts. A node-local scratch allocation would let a
    // preceding projection/vector kernel overwrite the initialized values.
    struct MaskHome { std::string role; I rows; mlir::DictionaryAttr home; };
    std::vector<MaskHome> masks;
    for(auto &node:planner.nodes) {
      if(!node.module)continue;
      node.function.walk([&](fc::kernel::AttentionCoreOp core) {
        if(auto causal=core.getConfig().getAs<mlir::BoolAttr>("causal"); !causal || !causal.getValue())return;
        auto original=core->getAttrOfType<mlir::DictionaryAttr>("memory_plan");
        mlir::NamedAttrList plan(original);
        auto busy=node.workspaces;
        for(const char *role:{"causal_mask","causal_mask_mxm1"}) {
          auto old=original.getAs<mlir::DictionaryAttr>(role);
          if(!old)continue;
          const I rows=field(old,"instruction_count");
          auto found=std::find_if(masks.begin(),masks.end(),[&](const MaskHome &entry) {
            return entry.role==role && entry.rows==rows &&
                llvm::none_of(busy,[&](auto p){return overlap(entry.home,p);});
          });
          mlir::DictionaryAttr home;
          if(found!=masks.end())home=found->home;
          else {
            Tensor mask;mask.shape={rows,s.target.memory().bytes_per_word};mask.dtype="bf16";
            const Key key{-7,I(masks.size())};last[key]=planner.nodes.size();
            home=allocate(key,mask,"both",1,old,busy);
            masks.push_back({role,rows,home});
          }
          mlir::NamedAttrList placement(old);
          for(auto attr:home)placement.set(attr.getName(),attr.getValue());
          home=placement.getDictionary(&s.context);
          plan.set(role,home);
          for(auto &workspace:node.workspaces)if(workspace==old)workspace=home;
          busy.push_back(home);
        }
        core->setAttr("memory_plan",plan.getDictionary(&s.context));
      });
    }
  }
  #include "PackedRowScalePlan.inc"

  #include "PlanarAccessPlanning.inc"
  #include "RotaryPlanarSchedule.inc"
  #include "RotaryTilePipeline.inc"
  void planMemory() {
    reserveAttentionInitializers();
    I joined = 0;
    for (I ni = 0; ni < I(planner.nodes.size()); ++ni) {
      auto &node = planner.nodes[ni];
      if(std::getenv("FTLPU_GRAPH_PROGRESS"))llvm::errs()<<"graph: memory "<<ni<<" "<<node.name<<"\n";
      I first = ni;
      if(node.projectionSolvePeer>=0)first=std::min(first,node.projectionSolvePeer);
      if(node.projectionConvGroup>=0)first=std::min(first,node.projectionConvGroup);
      if (node.parallel >= 0)
        while (first > 0 && planner.nodes[first - 1].parallel == node.parallel)
          --first;
      for (auto it = live.begin(); it != live.end();)
        if (last.count(it->first) && last.at(it->first) < first)
          it = live.erase(it);
        else
          ++it;
      if (!node.module) {
        if (node.planarRotaryDim) { planPlanarRotary(ni); continue; }
        if (node.inputs[0].external()) live[{ni,0}] = cache.at({ni,0});
        else if(node.mirroredJoin) {
          auto otherQueues=mirroredJoinQueues(ni);
          if(!planStreamingMatmulLhs(ni,otherQueues) && !planDirectPlanarJoin(ni,otherQueues) && !planCompactPlanarJoin(ni,otherQueues) && !planDistributedMatmulLhs(ni,otherQueues))
            allocate({ni,0},node.outputs[0],"both",1,{},otherQueues);
        }
        else allocate({ni, 0}, node.outputs[0], joined++ % 2 ? "east" : "west");
        continue;
      }
      node.stagedInputs.assign(node.inputs.size(), false);
      if(node.projectionVectorMember && node.projectionConvGroup==ni) {
        std::vector<mlir::DictionaryAttr> busy;
        reserveProjectionConvQueues(ni,busy);
        busy.push_back(node.in[1]);
        const Key inputKey{ni,-200};
        I end=ni;
        while(end+1<I(planner.nodes.size()) && planner.nodes[end+1].projectionConvGroup==ni)++end;
        last[inputKey]=end;
        auto home=allocate(inputKey,node.inputs[0],"both",1,node.in[0],busy);
        node.in[0]=home;
        for(auto &slot:node.inputSlots[0])slot.set(home);
      }
      for (I idx=0;idx<I(node.inputs.size());++idx) {
        // This activation has a queue-exclusive home for the whole concurrent
        // group. Ordinary producer aliases would bypass that reservation.
        if(idx==0 && node.projectionVectorMember && node.projectionConvGroup==ni)continue;
        auto home=externalVectorHomes.find({ni,idx});
        if(home==externalVectorHomes.end()) continue;
        node.in[idx]=home->second;
        for(auto &slot:node.inputSlots[idx]) slot.set(home->second);
        node.stagedInputs[idx]=true;
      }
      if (node.recurrent) {
        node.in[5] = recurrentStates.at(ni);
        for (auto &slot : node.inputSlots[5])
          slot.set(node.in[5]);
      }
      planNativePackedRowScale(node);
      // Exact compatible edges are true aliases. Layout changes and tensor
      // views remain explicit transports at the consumer, never an archive.
      for (I idx = 0; idx < I(node.inputs.size()); ++idx) {
        auto &value = node.inputs[idx];
        if(node.function->hasAttr("native_row_scale_words")) continue;
        if (node.recurrent && idx == 5)
          continue;
        if (value.refs.empty())
          continue;
        auto firstRef = value.refs[0];
        // A joint pair writes each output word directly into its own mirrored
        // byte pair. A narrow pointwise consumer aliases that pair; neither
        // splitting the matrix nor establishing token ownership needs a copy.
        if(value.shape.size()==2 && value.shape[1]<=32 && !node.inputSlots[idx].empty() &&
            llvm::all_of(node.inputSlots[idx],[](const auto &slot){return llvm::isa<fc::kernel::VectorAluOp>(slot.op);})) {
          auto source=cache.at({firstRef.node,firstRef.output});
          const auto &shape=planner.nodes[firstRef.node].outputs[firstRef.output].shape;
          const auto ids=slices(source);
          const I word=shape.size()==2 && shape[1]>0 ? firstRef.index%shape[1]/32 : -1;
          bool valid=shape.size()==2 && shape[0]==value.shape[0] && shape[1]==64 && ids.size()==4 &&
              text(source,"kind")=="fp16_pair_planar" && text(source,"hemisphere")=="both" &&
              firstRef.index==word*32 && !source.get("storage_shape") && value.dtype=="bf16";
          for(I j=0;valid && j<value.size();++j) {
            const auto ref=value.refs[j];valid=ref.node==firstRef.node && ref.output==firstRef.output &&
                ref.index==(j/value.shape[1])*64+word*32+j%value.shape[1];
          }
          if(valid) {
            auto home=s.change(source,J{{"slices",s.array({ids[2*word],ids[2*word+1]})},{"instruction_count",value.shape[0]}});
            node.in[idx]=home;for(auto &slot:node.inputSlots[idx])slot.set(home);
            node.stagedInputs[idx]=true;continue;
          }
        }
        // Shared-weight contractions may use the same native D in two ops.
        // Alias all operand slots together; a second use is not a layout copy.
        if(text(node.in[idx],"kind")=="token_partition_east_first_planar" &&
           !node.inputSlots[idx].empty() && llvm::all_of(node.inputSlots[idx],[](const auto &slot) {
             auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(slot.op);
             return op && (op.getKind()=="head_matmul" || op.getKind()=="head_cross_product");
           })) {
          auto home=cache.at({firstRef.node,firstRef.output});
          bool identity=text(home,"kind")=="token_partition_east_first_planar" &&
              planner.nodes[firstRef.node].outputs[firstRef.output].shape==value.shape;
          for(I j=0;identity && j<value.size();++j)
            identity=value.refs[j].node==firstRef.node && value.refs[j].output==firstRef.output && value.refs[j].index==j;
          if(identity) {
            node.in[idx]=home;for(auto &slot:node.inputSlots[idx])slot.set(home);
            node.stagedInputs[idx]=true;continue;
          }
        }
        if(node.inputSlots[idx].size()==1) {
          auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(node.inputSlots[idx][0].op);
          if(op && idx==0 && op.getKind()=="head_matmul") {
            auto home=cache.at({firstRef.node,firstRef.output});
            if(home.get("mxm_lhs_head_rows")) {
              node.in[idx]=home;node.inputSlots[idx][0].set(home);
              node.stagedInputs[idx]=true;continue;
            }
          }
          if(op && idx==2 && node.in.size()==3 && op.getRhs()==node.function.getArgument(idx)) {
            auto product=op.getLhs().getDefiningOp<fc::kernel::VectorAluOp>();
            const auto fullMirrored=[](mlir::DictionaryAttr p) {
              return text(p,"hemisphere")=="both" &&
                  (p.get("mxm_lhs_head_rows") || (text(p,"kind")=="fp16_pair_planar" && slices(p).size()==2) ||
                   (text(p,"kind")=="fp32_quad_planar" && slices(p).size()==4));
            };
            // With full matrix operands on both sides, choose their logical
            // half to match the prior sum already resident on each side.
            // Otherwise retain the normal full-quad staging fallback.
            if(isNativeMatmulAdd(product,op) && product.getLhs()==node.function.getArgument(0) &&
               product.getRhs()==node.function.getArgument(1) &&
               fullMirrored(node.in[0]) && fullMirrored(node.in[1])) {
              auto home=cache.at({firstRef.node,firstRef.output});
              const auto kind=text(home,"kind");
              bool identity=value.dtype=="f32" && slices(home).size()==4 &&
                  text(home,"hemisphere")=="both" && !home.get("storage_shape") &&
                  (kind=="token_partition_planar" || kind=="token_partition_east_first_planar") &&
                  planner.nodes[firstRef.node].outputs[firstRef.output].shape==value.shape;
              for(I j=0;identity && j<value.size();++j)
                identity=value.refs[j].node==firstRef.node && value.refs[j].output==firstRef.output && value.refs[j].index==j;
              if(identity) {
                node.in[idx]=home;node.inputSlots[idx][0].set(home);node.stagedInputs[idx]=true;continue;
              }
            }
          }
          if(op && ((op.getKind()=="unit_lower_solve" || op.getKind()=="unit_lower_solve_blocked") || op.getKind()=="head_scale" || op.getKind()=="unit_lower_inverse" || op.getKind()=="head_matmul" || op.getKind()=="head_cross_product")) {
            if(op.getKind()=="head_scale" && idx==0) {
              llvm::SmallVector<mlir::Attribute> words;
              for(I at=0;at<value.size();at+=32) {
                const auto ref=value.refs[at];auto home=cache.at({ref.node,ref.output});
                const auto shape=planner.nodes[ref.node].outputs[ref.output].shape;
                const auto ids=slices(home);const I token=ref.index/shape[1],col=ref.index%shape[1];
                for(I lane=0;lane<32;++lane) {
                  auto q=value.refs[at+lane];
                  if(q.node!=ref.node || q.output!=ref.output || q.index!=ref.index+lane)
                    throw std::runtime_error("head_scale requires contiguous feature vectors");
                }
                I address=0;
                if(text(home,"kind")=="fp16_mxm_distributed_16" && ids.size()==16) {
                  address=((token/32)*(shape[1]/32)+col/32)*4+(token%32)/8;
                  home=s.change(home,J{{"kind","fp16_pair_planar"},{"slices",s.array({ids[2*(token%8)],ids[2*(token%8)+1]})}});
                } else if(text(home,"kind")=="fp16_pair_planar" && ids.size()==2) address=(col/32)*shape[0]+token;
                else throw std::runtime_error("head_scale requires a native planar/distributed value producer");
                words.push_back(s.b.getDictionaryAttr({s.b.getNamedAttr("placement",home),s.b.getNamedAttr("address",s.b.getI64IntegerAttr(field(home,"base_row")+address))}));
              }
              op->setAttr("solve.value_words",s.b.getArrayAttr(words));node.stagedInputs[idx]=true;continue;
            }
            bool identity=planner.nodes[firstRef.node].outputs[firstRef.output].shape==value.shape;
            for(I j=0;identity && j<value.size();++j) identity=value.refs[j].node==firstRef.node && value.refs[j].output==firstRef.output && value.refs[j].index==j;
            if(identity) {
              auto home=cache.at({firstRef.node,firstRef.output});
              const auto kind=text(home,"kind");
              const bool mxmSolve=op.getKind()=="unit_lower_inverse" || op.getKind()=="head_matmul";
              const auto required=op.getKind()=="unit_lower_inverse"?"token_partition_east_first_planar":"token_partition_planar";
              bool matchingMatmulHomes=false;
              const auto mirroredBf16=[](mlir::DictionaryAttr p) {
                return text(p,"kind")=="fp16_pair_planar" && text(p,"hemisphere")=="both" && slices(p).size()==2;
              };
              if(op.getKind()=="head_matmul" && idx<2 && node.inputs.size()>=2 &&
                 (kind=="token_partition_east_first_planar" || kind=="fp32_quad_planar" || mirroredBf16(home)) && !node.inputs[1-idx].refs.empty()) {
                const auto other=node.inputs[1-idx].refs.front();
                const auto found=cache.find({other.node,other.output});
                matchingMatmulHomes=found!=cache.end() &&
                    (text(found->second,"kind")=="token_partition_east_first_planar" ||
                     text(found->second,"kind")=="fp32_quad_planar" || mirroredBf16(found->second) ||
                     found->second.get("mxm_lhs_head_rows"));
                const auto &otherValue=node.inputs[1-idx];
                matchingMatmulHomes &= planner.nodes[other.node].outputs[other.output].shape==otherValue.shape;
                for(I j=0;matchingMatmulHomes && j<otherValue.size();++j)
                  matchingMatmulHomes=otherValue.refs[j].node==other.node &&
                      otherValue.refs[j].output==other.output && otherValue.refs[j].index==j;
              }
              if((kind=="fp32_quad_planar" && op.getKind()!="head_matmul") || (op.getKind()=="head_cross_product" && kind=="token_partition_east_first_planar") || matchingMatmulHomes || (mxmSolve ? kind==required :
                  (kind=="token_partition_east_first_planar" || kind=="token_partition_planar"))) {
                node.in[idx]=home;node.inputSlots[idx][0].set(home);node.stagedInputs[idx]=true;continue;
              }
            }
          }
        }

        if(idx<=1 && !node.inputSlots[idx].empty() &&
            llvm::all_of(node.inputSlots[idx],[](const auto &slot) {
                auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(slot.op);
                return op && (op.getKind()=="weighted_gram" || op.getKind()=="cross_gram" || op.getKind()=="head_last_row_scale_transpose");
            })) {
          auto source=cache.at({firstRef.node,firstRef.output});
          const auto &shape=planner.nodes[firstRef.node].outputs[firstRef.output].shape;
          bool identity=value.size()==product(shape) && firstRef.index==0;
          for(I j=0;identity && j<value.size();++j)
            identity=value.refs[j].node==firstRef.node && value.refs[j].output==firstRef.output && value.refs[j].index==j;
          if(identity && text(source,"kind")=="fp16_mxm_distributed_16" && slices(source).size()==16 &&
              llvm::isa<fc::tensor::L2NormTaskOp>(planner.nodes[firstRef.node].outputSlots[firstRef.output].op)) {
            node.in[idx]=source;
            for(auto &slot:node.inputSlots[idx]) slot.set(source);
            node.stagedInputs[idx]=true;
            // This consumer reads alternating native L2 heads on their home
            // hemisphere. Do not mirror the producer merely for this edge.
            bool partitionConsumers=true;
            for(auto &consumer:planner.nodes) for(I operand=0;operand<I(consumer.inputs.size());++operand) {
              const bool uses=llvm::any_of(consumer.inputs[operand].refs,[&](const Ref &ref) {
                return ref.node==firstRef.node && ref.output==firstRef.output;
              });
              if(!uses)continue;
              partitionConsumers &= operand<=1 && I(consumer.inputSlots.size())>operand && !consumer.inputSlots[operand].empty() &&
                  llvm::all_of(consumer.inputSlots[operand],[](const auto &slot) {
                    auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(slot.op);
                    return op && (op.getKind()=="weighted_gram" || op.getKind()=="cross_gram" || op.getKind()=="head_last_row_scale_transpose");
                  });
            }
            if(partitionConsumers)
              planner.nodes[firstRef.node].function->setAttr("ftlpu.l2_native_partition_output",s.b.getUnitAttr());
            continue;
          }
        }
        // Native fusion requires word descriptors for both operands. An
        // external operand has no producer partition; leave both inputs to
        // the generic staging path rather than publishing a partial plan.
        if(node.function->hasAttr("ftlpu.native_silu_multiply") && value.dtype=="bf16"
            && llvm::none_of(node.inputs, [](const Tensor &t) { return t.external(); })
            && value.shape.size()==2 && value.shape[0]%2==0 && value.shape[1]%32==0) {
          llvm::SmallVector<mlir::Attribute> words;
          V rowSides,rowIndices;I localRows[2]={0,0};
          bool valid=true;const I rows=value.shape[0],blocks=value.shape[1]/32;
          for(I block=0;valid && block<blocks;++block) for(I row=0;valid && row<rows;++row) {
            const I at=row*value.shape[1]+block*32;
            const auto ref=value.refs[at];auto p=cache.at({ref.node,ref.output});
            const auto &shape=planner.nodes[ref.node].outputs[ref.output].shape;
            valid=shape.size()==2 && shape[1]%32==0 && ref.index%32==0 && slices(p).size()==2 && !p.get("storage_shape");
            for(I lane=0;valid && lane<32;++lane) {
              const auto r=value.refs[at+lane];
              valid=r.node==ref.node && r.output==ref.output && r.index==ref.index+lane;
            }
            if(!valid)break;
            I side=0,address=0;
            if(text(p,"kind")=="fp16_hemisphere_pair_planar") {
              const I col=ref.index%shape[1];side=col/32%2;
              address=field(p,"base_row")+(col/64)*shape[0]+ref.index/shape[1];
            } else if(text(p,"kind")=="token_partition_planar" && p.get("partition_row_sides")) {
              const I logicalRow=ref.index/shape[1];
              side=llvm::cast<mlir::IntegerAttr>(p.getAs<mlir::ArrayAttr>("partition_row_sides")[logicalRow]).getInt();
              const I local=llvm::cast<mlir::IntegerAttr>(p.getAs<mlir::ArrayAttr>("partition_row_indices")[logicalRow]).getInt();
              address=field(p,"base_row")+(ref.index%shape[1]/32)*field(p,"partition_storage_rows")+local;
            } else {valid=false;break;}
            p=s.change(p,J{{"kind","fp16_pair_planar"},{"hemisphere",side?"west":"east"}});
            words.push_back(s.b.getDictionaryAttr({s.b.getNamedAttr("placement",p),s.b.getNamedAttr("address",s.b.getI64IntegerAttr(address))}));
            if(block==0) {rowSides.push_back(1-side);rowIndices.push_back(localRows[side]++);}
          }
          if(valid && localRows[0]==rows/2 && localRows[1]==rows/2) {
            node.function->setAttr("native_pointwise_words_"+std::to_string(idx),s.b.getArrayAttr(words));
            node.out[0]=s.change(node.out[0],J{{"kind","token_partition_planar"},{"hemisphere","both"},
                {"partition_row_sides",s.array(rowSides)},{"partition_row_indices",s.array(rowIndices)},
                {"partition_storage_rows",rows/2}});
            node.stagedInputs[idx]=true;continue;
          }
        }
        // A row-wise scalar broadcast can retain a compact table of vectors:
        // repeat a packed coefficient word 32 times through the wide SXM.
        // The consumer repeats table addresses instead of expanding to state size.
        if(value.shape.size()==2 && value.shape[1]%32==0 && !node.inputSlots[idx].empty() &&
            node.function->hasAttr("ftlpu.vector_bf16_spill") &&
            llvm::all_of(node.inputSlots[idx],[](const auto &slot){return llvm::isa<fc::kernel::VectorAluOp>(slot.op);}) &&
            (*node.module)->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide")) {
          const Key sourceKey{firstRef.node,firstRef.output};
          const auto &sourceTensor=planner.nodes[firstRef.node].outputs[firstRef.output];
          auto source=cache.at(sourceKey);
          const I heads=sourceTensor.shape.back(), width=value.shape[1];
          const I sourceRow=firstRef.index/heads, headBase=firstRef.index%heads;
          I repeat=0;
          while(repeat<value.shape[0] && value.refs[repeat*width].index==firstRef.index) ++repeat;
          const I headStride=repeat<value.shape[0]
              ? value.refs[repeat*width].index-firstRef.index : 1;
          bool valid=heads<=32 && sourceTensor.shape.size()==2 &&
              text(source,"kind")=="fp32_quad_planar" && text(source,"hemisphere")=="both" &&
              !source.get("storage_shape") && !source.get("storage_offset") &&
              value.dtype=="f32" && repeat>0 && value.shape[0]%repeat==0 && headStride>0;
          for(I j=0;valid && j<value.size();++j) {
            const auto r=value.refs[j];
            valid=r.node==firstRef.node && r.output==firstRef.output &&
                r.index==firstRef.index+((j/width)/repeat)*headStride && r.index/heads==sourceRow;
          }
          if(valid) {
            const BroadcastKey tableKey{firstRef.node,firstRef.output,sourceRow};
            if(!wideBroadcastHomes.count(tableKey)) {
              Tensor table=sourceTensor;table.shape={heads,32};
              const Key reservation{-5,I(wideBroadcastHomes.size())};last[reservation]=last[sourceKey];
              wideBroadcastHomes[tableKey]=allocate(reservation,table,"both",1);
            }
            auto home=s.change(wideBroadcastHomes.at(tableKey),J{{"broadcast_row_repeat",repeat},
                {"broadcast_head_base",headBase},{"broadcast_head_stride",headStride}});
            node.in[idx]=home;
            for(auto &slot:node.inputSlots[idx]) slot.set(home);
            node.stagedInputs[idx]=true;
            wideBroadcastOperands[{ni,idx}]=tableKey;
            continue;
          }
        }
        // A single packed row of a mirrored pointwise producer is already a
        // complete vector operand. Preserve its address instead of copying a
        // token view into the consumer's provisional activation allocation.
        if (value.shape.size()==2 && value.shape[0]==1 && value.shape[1]<=32 &&
            !node.inputSlots[idx].empty() &&
            llvm::all_of(node.inputSlots[idx],[&](const auto &slot){
              return (node.function->hasAttr("ftlpu.vector_bf16_spill") &&
                      llvm::isa<fc::kernel::VectorAluOp>(slot.op)) ||
                  ((idx==1 || idx==4) && llvm::isa<fc::kernel::ScaledMatvecOp>(slot.op));
            })) {
          auto source=cache.at({firstRef.node,firstRef.output});
          const auto &shape=planner.nodes[firstRef.node].outputs[firstRef.output].shape;
          const auto kind=text(source,"kind");
          bool valid=shape.size()==2 && shape[1]==value.shape[1] &&
              firstRef.index%shape[1]==0 && text(source,"hemisphere")=="both" &&
              !source.get("storage_shape") && !source.get("storage_offset") &&
              ((value.dtype=="f32" && kind=="fp32_quad_planar" && slices(source).size()==4) ||
               (value.dtype=="bf16" && kind=="fp16_pair_planar" && slices(source).size()==2));
          for(I j=0;valid && j<value.size();++j) {
            const auto r=value.refs[j];
            valid=r.node==firstRef.node && r.output==firstRef.output && r.index==firstRef.index+j;
          }
          if(valid) {
            auto home=s.change(source,J{{"base_row",field(source,"base_row")+firstRef.index/shape[1]},
                {"instruction_count",1}});
            node.in[idx]=home;
            for(auto &slot:node.inputSlots[idx]) slot.set(home);
            node.stagedInputs[idx]=true;
            continue;
          }
        }
        // Decode causal windows are logical views of history and projection
        // words. Describe those homes directly instead of joining a window.
        if (idx==0 && node.function->hasAttr("ftlpu.depthwise_conv_paired") &&
            value.shape.size()==2 && value.shape[0]<=4 &&
            value.shape[1]%32==0 && node.inputSlots[idx].size()==1 &&
            llvm::isa<fc::tensor::DepthwiseConv1DTaskOp>(node.inputSlots[idx][0].op)) {
          llvm::SmallVector<mlir::Attribute> words;
          bool valid=true;
          for(I block=0;valid && block<value.shape[1]/32;++block)
            for(I row=0;valid && row<value.shape[0];++row) {
              I at=row*value.shape[1]+block*32;
              const auto ref=value.refs[at];
              auto p=cache.at({ref.node,ref.output});
              const auto shape=planner.nodes[ref.node].outputs[ref.output].shape;
              const bool paired=text(p,"kind")=="fp16_hemisphere_pair_planar";
              valid=shape.size()==2 && !p.get("storage_shape") &&
                  (paired || text(p,"kind")=="fp16_pair_planar") &&
                  ref.index%shape[1]%32==0;
              for(I lane=0;valid && lane<32;++lane) {
                const auto r=value.refs[at+lane];
                valid=r.node==ref.node && r.output==ref.output && r.index==ref.index+lane;
              }
              if(!valid) break;
              if(paired) p=s.change(p,J{{"hemisphere",ref.index%shape[1]/32%2?"west":"east"}});
              const I address=ref.index%shape[1]/(paired?64:32)*shape[0]+ref.index/shape[1];
              words.push_back(s.b.getDictionaryAttr({s.b.getNamedAttr("placement",p),
                  s.b.getNamedAttr("row",s.b.getI64IntegerAttr(address))}));
            }
          if(valid) {
            node.inputSlots[idx][0].op->setAttr("native_window_words",s.b.getArrayAttr(words));
            node.stagedInputs[idx]=true;
            continue;
          }
        }
        // Packed beta/decay rows live on their producer's token hemisphere.
        // Broadcast locally through SXM; VXM can select that hemisphere for
        // its RHS while the recurrent state remains in its retained home.
        if (node.recurrent && text(node.in[idx],"hemisphere") != "both" && (idx == 3 || idx == 4)) {
          auto source = cache.at({firstRef.node, firstRef.output});
          const auto &shape = planner.nodes[firstRef.node].outputs[firstRef.output].shape;
          std::string hemisphere = text(source,"hemisphere");
          if (text(source,"kind") == "token_partition_planar")
            hemisphere = firstRef.index / shape[1] < (shape[0]+1)/2 ? "west" : "east";
          if (hemisphere != "both") {
            node.in[idx] = s.change(node.in[idx], J{{"hemisphere",hemisphere}});
            for (auto &slot : node.inputSlots[idx]) slot.set(node.in[idx]);
          }
        }

        // A grouped last-axis reduction may view columns from several native
        // distributed producers. Describe those columns before scheduling;
        // the L2 emitter transposes each group directly from its producer.
        const auto planarNorm = [&]() {
          if(node.inputSlots[idx].empty())return false;
          auto config=node.inputSlots[idx][0].op->getAttrOfType<mlir::DictionaryAttr>("config");
          auto flag=config?config.getAs<mlir::BoolAttr>("planar_streaming"):mlir::BoolAttr{};
          return flag && flag.getValue();
        }();
        if (idx == 0 && node.inputSlots[idx].size() == 1 && !planarNorm &&
            llvm::isa<fc::tensor::L2NormTaskOp,fc::tensor::RmsNormTaskOp>(node.inputSlots[idx][0].op)) {
          // Wide SXM accepts row vectors directly from a planar producer.
          // Describe the logical view rather than materializing distributed16
          // (including duplicated padding heads) before the transpose.
          const auto widePolicy=(*node.module)->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
          bool planar=widePolicy && widePolicy.getValue() && value.shape[0]==32
              && value.shape[1]%32==0;
          llvm::SmallVector<mlir::Attribute> words;
          for(I i=0;planar && i<value.size();i+=32) {
            const auto ref=value.refs[i];auto p=cache.at({ref.node,ref.output});
            const auto &shape=planner.nodes[ref.node].outputs[ref.output].shape;
            I storageRows=shape[0],storageRow=ref.index/shape[1];
            // A semantic FP32 -> BF16 truncating view aliases the upper
            // two byte planes. Readout is already mirrored on both fabrics;
            // assign complete heads to each side without moving the data.
            if(value.dtype=="bf16" && planner.nodes[ref.node].outputs[ref.output].dtype=="f32"
                && (text(p,"kind")=="fp32_quad_planar" || text(p,"kind")=="token_partition_planar"
                    || text(p,"kind")=="token_partition_east_first_planar") && slices(p).size()==4) {
              const auto ids=slices(p);
              auto side=text(p,"hemisphere")=="both"
                  ? (ref.index/shape[1]<shape[0]/2?"west":"east") : text(p,"hemisphere");
              if(text(p,"kind")!="fp32_quad_planar") {
                storageRows=(shape[0]+1)/2;
                if(text(p,"kind")=="token_partition_east_first_planar")side=storageRow<storageRows?"east":"west";
                storageRow%=storageRows;
              }
              p=s.change(p,J{{"kind","fp16_pair_planar"},{"slices",s.array({ids[2],ids[3]})},
                  {"hemisphere",side}});
            }
            // Mirrored BF16 readouts already contain each complete row on
            // both fabrics. Alternate reduction rows between them; the norm
            // result naturally leaves on the opposite fabric. No copy is
            // needed to establish the paired-head ownership.
            if(value.dtype=="bf16" && text(p,"kind")=="fp16_pair_planar"
                && text(p,"hemisphere")=="both" && slices(p).size()==2)
              p=s.change(p,J{{"hemisphere",(i/value.shape[1])%2?"east":"west"}});
            planar=shape.size()==2 && shape[1]%32==0 && ref.index%32==0
                && text(p,"kind")=="fp16_pair_planar" && !p.get("storage_shape")
                && slices(p).size()==2;
            // Single-vector BF16 rows retain the generic reduction schedule:
            // its accumulator drain does not use this multi-vector feedback.
            if(value.shape[1]==32 && planner.nodes[ref.node].outputs[ref.output].dtype=="bf16")
              planar=false;
            for(I j=0;planar && j<32;++j) {
              auto r=value.refs[i+j];planar=r.node==ref.node && r.output==ref.output && r.index==ref.index+j;
            }
            if(planar) {
              mlir::NamedAttrList d;d.set("placement",p);
              d.set("address",s.b.getI64IntegerAttr(field(p,"base_row")+
                  (ref.index%shape[1]/32)*storageRows+storageRow));
              words.push_back(d.getDictionary(&s.context));
            }
          }
          if(planar) {
            node.inputSlots[idx][0].op->setAttr("native_planar_words",s.b.getArrayAttr(words));
            node.out[0]=s.change(node.out[0],J{{"kind","fp16_pair_planar"},
                {"slices",s.array({48,49})},{"hemisphere","west"},
                {"instruction_count",value.size()/32}});
            const auto partition=(*node.module)->getAttrOfType<mlir::BoolAttr>("ftlpu.l2_head_partition");
            if(partition && partition.getValue()) {
              const I blocks=value.shape[1]/32;
              V heads[2],rowSides,rowIndices;
              for(I row=0;row<32;++row) {
                const auto p=llvm::cast<mlir::DictionaryAttr>(words[row*blocks]).getAs<mlir::DictionaryAttr>("placement");
                const I side=text(p,"hemisphere")=="west"?1:0;
                for(I b=1;b<blocks;++b)
                  if(text(llvm::cast<mlir::DictionaryAttr>(words[row*blocks+b]).getAs<mlir::DictionaryAttr>("placement"),"hemisphere")!=text(p,"hemisphere"))
                    throw std::runtime_error("decode L2 head spans hemispheres after producer partition");
                I local=0;
                for(;local<I(heads[side].size());++local) {
                  bool same=true;
                  for(I b=0;b<blocks;++b) same &= words[row*blocks+b]==words[heads[side][local]*blocks+b];
                  if(same) break;
                }
                if(local==I(heads[side].size())) heads[side].push_back(row);
                // Fixed VXM outputs leave on the opposite fabric. Retain that
                // native home rather than copying normalized heads back.
                rowSides.push_back(1-side);rowIndices.push_back(local);
              }
              llvm::SmallVector<mlir::Attribute> paired;
              for(I side=0;side<2;++side) {
                if(heads[side].empty()) throw std::runtime_error("decode head partition needs heads on both fabrics");
                for(I row=0;row<32;++row) for(I b=0;b<blocks;++b)
                  paired.push_back(words[heads[side][std::min<I>(row,heads[side].size()-1)]*blocks+b]);
              }
              node.inputSlots[idx][0].op->setAttr("native_partitioned_planar_words",s.b.getArrayAttr(paired));
              node.out[0]=s.change(node.out[0],J{{"kind","token_partition_planar"},{"hemisphere","both"},
                  {"partition_row_sides",s.array(rowSides)},{"partition_row_indices",s.array(rowIndices)},
                  {"partition_storage_rows",32}});
            }
            node.stagedInputs[idx]=true;continue;
          }
          if(llvm::isa<fc::tensor::RmsNormTaskOp>(node.inputSlots[idx][0].op)) continue;
          const auto &firstShape = planner.nodes[firstRef.node].outputs[firstRef.output].shape;
          I tokens = firstShape[0];
          const I width = value.shape[1];
          // A consumer may normalize one aligned token chunk of a larger
          // Conv producer. Infer the logical head stride from its view;
          // producer row count is not the consumer's token count.
          for (I row=1; row<value.shape[0]; ++row) {
            const auto ref=value.refs[row*width];
            if(ref.node==firstRef.node && ref.output==firstRef.output &&
                ref.index==firstRef.index+firstShape[1] && value.shape[0]%row==0) {
              tokens=value.shape[0]/row; break;
            }
          }
          const I groups = tokens ? value.shape[0] / tokens : 0;
          bool compatible = tokens > 0 && tokens % 32 == 0 && width % 32 == 0 &&
              groups > 1 && value.shape[0] == tokens * groups;
          llvm::SmallVector<mlir::Attribute> sources;
          for (I g = 0; compatible && g < groups; ++g) {
            auto ref = value.refs[g * width];
            auto src = cache.at({ref.node, ref.output});
            const auto &shape = planner.nodes[ref.node].outputs[ref.output].shape;
            I column = ref.index % shape[1], firstToken = ref.index / shape[1];
            compatible = shape.size() == 2 && firstToken%32==0 && shape[0] >= firstToken+tokens &&
                column % 32 == 0 && column + width <= shape[1] &&
                text(src, "kind") == "fp16_mxm_distributed_16" &&
                !src.get("storage_shape") && slices(src).size() == 16;
            for (I t = 0; compatible && t < tokens; ++t)
              for (I f = 0; compatible && f < width; ++f) {
                auto r = value.refs[(t * groups + g) * width + f];
                compatible = r.node == ref.node && r.output == ref.output &&
                    r.index == (firstToken+t) * shape[1] + column + f;
              }
            if (compatible) {
              mlir::NamedAttrList descriptor;
              descriptor.set("placement", s.change(src,J{{"base_row",
                  field(src,"base_row")+(firstToken/32)*(shape[1]/8)}}));
              descriptor.set("column", s.b.getI64IntegerAttr(column));
              descriptor.set("columns", s.b.getI64IntegerAttr(shape[1]));
              sources.push_back(descriptor.getDictionary(&s.context));
            }
          }
          if (!compatible && tokens > 0 && tokens % 32 == 0 && width % 32 == 0
              && groups > 1 && value.shape[0] == tokens * groups) {
            // A head may span native words on both hemispheres. Describe each
            // word to the transpose itself instead of copying the full head.
            compatible = true; sources.clear();
            for (I g = 0; compatible && g < groups; ++g) {
              llvm::SmallVector<mlir::Attribute> chunks;
              for (I c = 0; compatible && c < width; c += 32) {
                const auto ref = value.refs[g * width + c];
                auto src = cache.at({ref.node, ref.output});
                const auto &shape = planner.nodes[ref.node].outputs[ref.output].shape;
                const I column = ref.index%shape[1], firstToken=ref.index/shape[1];
                compatible = shape.size() == 2 && firstToken%32==0 && shape[0]>=firstToken+tokens && column % 32 == 0
                    && column + 32 <= shape[1] && text(src,"kind") == "fp16_mxm_distributed_16"
                    && !src.get("storage_shape") && slices(src).size() == 16;
                for (I t = 0; compatible && t < tokens; ++t)
                  for (I f = 0; compatible && f < 32; ++f) {
                    const auto r = value.refs[(t * groups + g) * width + c + f];
                    compatible = r.node == ref.node && r.output == ref.output
                        && r.index == (firstToken+t) * shape[1] + column + f;
                  }
                mlir::NamedAttrList d;
                d.set("placement",s.change(src,J{{"base_row",
                    field(src,"base_row")+(firstToken/32)*(shape[1]/8)}})); d.set("column",s.b.getI64IntegerAttr(column));
                d.set("columns",s.b.getI64IntegerAttr(shape[1]));
                chunks.push_back(d.getDictionary(&s.context));
              }
              mlir::NamedAttrList d; d.set("chunks",s.b.getArrayAttr(chunks));
              sources.push_back(d.getDictionary(&s.context));
            }
          }
          if (compatible) {
            auto *owner = node.inputSlots[idx][0].op;
            owner->setAttr("native_input_groups", s.b.getArrayAttr(sources));
            owner->setAttr("native_input_rows", s.b.getI64IntegerAttr(tokens));
            node.stagedInputs[idx] = true;
            continue;
          }
        }
        // A packed pointwise consumer can read a column-word view of a
        // paired projection directly, including a partially occupied word.
        // Keep logical head count unchanged; padding lanes are independent.
        const auto firstPlacement = cache.at({firstRef.node, firstRef.output});
        const auto &producerShape = planner.nodes[firstRef.node].outputs[firstRef.output].shape;
        // A channel block of a planar causal window occupies consecutive
        // vector words. Rebind its base address instead of staging a second
        // window before every Conv group. Preserve the selected hemisphere.
        if (idx == 0 && node.inputSlots[idx].size() == 1
            && llvm::isa<fc::tensor::DepthwiseConv1DTaskOp>(node.inputSlots[idx][0].op)
            && text(firstPlacement,"kind") == "fp16_pair_planar"
            && text(firstPlacement,"hemisphere") == text(node.in[idx],"hemisphere")
            && !firstPlacement.get("storage_shape")
            && producerShape.size() == 2 && value.shape.size() == 2
            && value.shape[0] == producerShape[0]
            && firstRef.index % 32 == 0 && firstRef.index < producerShape[1]
            && value.shape[1] % 32 == 0) {
          bool contiguous = true;
          for (I j=0; contiguous && j<value.size(); ++j) {
            const auto ref=value.refs[j];
            contiguous = ref.node==firstRef.node && ref.output==firstRef.output
                && ref.index==firstRef.index+(j/value.shape[1])*producerShape[1]+j%value.shape[1];
          }
          if (contiguous) {
            auto p=s.change(firstPlacement,J{
                {"base_row",field(firstPlacement,"base_row")+(firstRef.index/32)*producerShape[0]},
                {"instruction_count",value.size()/32}});
            node.in[idx]=p;node.inputSlots[idx][0].set(p);
            node.function->setAttr("ftlpu.depthwise_conv_direct_input",s.b.getBoolAttr(true));
            node.stagedInputs[idx]=true;
            continue;
          }
        }
        if (idx == 0 && node.inputSlots[idx].size() == 1
            && llvm::isa<fc::tensor::DepthwiseConv1DTaskOp>(node.inputSlots[idx][0].op)
            && text(firstPlacement, "kind") == "fp16_hemisphere_pair_planar"
            && !firstPlacement.get("storage_shape") && slices(firstPlacement).size() == 2
            && producerShape.size() == 2 && value.shape.size() == 2
            && firstRef.index % 32 == 0 && value.shape[1] % 32 == 0) {
          const I col = firstRef.index % producerShape[1];
          bool affine = firstRef.index / producerShape[1] == 0 && value.shape[0] == producerShape[0];
          for (I j = 0; affine && j < value.size(); ++j) {
            const auto ref = value.refs[j]; const I c = j % value.shape[1];
            affine = ref.node == firstRef.node && ref.output == firstRef.output
                && ref.index == (j / value.shape[1]) * producerShape[1] + col + (c / 32) * 64 + c % 32;
          }
          if (affine) {
            auto p = s.change(firstPlacement, J{{"kind", "fp16_pair_planar"},
                {"hemisphere", (col / 32) % 2 ? "west" : "east"},
                {"base_row", field(firstPlacement,"base_row") + (col / 64) * producerShape[0]},
                {"instruction_count", value.size() / 32}});
            node.in[idx] = p;
            node.inputSlots[idx][0].set(p);
            node.function->setAttr("ftlpu.depthwise_conv_direct_input", s.b.getBoolAttr(true));
            node.stagedInputs[idx] = true;
            continue;
          }
        }
        if (text(firstPlacement,"kind")=="token_partition_planar"
            && node.function->hasAttr("ftlpu.vector_bf16_token_halves")
            && value.shape.size()==2 && value.shape[0]==32 && value.shape[1]<=32
            && producerShape.size()==2 && producerShape[0]==32 && firstRef.index==0
            && !firstPlacement.get("partition_row_sides")) {
          bool direct=true;
          for(I j=0;j<value.size();++j) {
            const auto ref=value.refs[j];
            direct &= ref.node==firstRef.node && ref.output==firstRef.output
                && ref.index==(j/value.shape[1])*producerShape[1]+j%value.shape[1];
          }
          if(direct) {
            node.in[idx]=firstPlacement;
            for(auto &slot:node.inputSlots[idx]) slot.set(firstPlacement);
            node.stagedInputs[idx]=true;
            continue;
          }
        }
        if (text(firstPlacement, "kind") == "token_partition_planar"
            && value.shape.size() == 2 && producerShape.size() == 2
            && value.shape[1] <= s.target.memory().bytes_per_word) {
          const I half = (producerShape[0] + 1) / 2;
          const I row = firstRef.index / producerShape[1], col = firstRef.index % producerShape[1];
          const I side = row >= half, local = row - side * half;
          bool view = col % s.target.memory().bytes_per_word == 0
              && local + value.shape[0] <= half;
          for (I j = 0; view && j < value.size(); ++j) {
            auto ref = value.refs[j];
            view = ref.node == firstRef.node && ref.output == firstRef.output
                && ref.index == firstRef.index + j / value.shape[1] * producerShape[1] + j % value.shape[1];
          }
          if (view && text(node.in[idx], "hemisphere") == (side ? "east" : "west")) {
            auto p = s.change(firstPlacement, J{{"kind", value.dtype == "f32" ? "fp32_quad_planar" : "fp16_pair_planar"},
                {"hemisphere", side ? "east" : "west"},
                {"base_row", field(firstPlacement,"base_row") + col / s.target.memory().bytes_per_word * half + local},
                {"instruction_count", value.shape[0]}});
            node.in[idx] = p;
            for (auto &slot : node.inputSlots[idx]) slot.set(p);
            node.stagedInputs[idx] = true;
            continue;
          }
        }
        const I wordElements = s.target.memory().bytes_per_word;
        if (!node.function->hasAttr("ftlpu.vector_bf16_spill")
            && text(firstPlacement, "kind") == "fp16_hemisphere_pair_planar"
            && text(node.in[idx], "kind") == "fp16_pair_planar"
            && slices(firstPlacement).size() == 2 && !firstPlacement.get("storage_shape")
            && producerShape.size() == 2 && value.shape.size() == 2
            && firstRef.index / producerShape[1] + value.shape[0] <= producerShape[0]
            && value.shape[1] <= wordElements
            && firstRef.index % producerShape[1] % wordElements == 0
            && firstRef.index % producerShape[1] + value.shape[1] <= producerShape[1]
            && (!node.function->hasAttr("ftlpu.vector_token_partition")
                || text(node.in[idx], "hemisphere") ==
                    (firstRef.index % producerShape[1] / wordElements % 2 ? "west" : "east"))
            && !node.inputSlots[idx].empty()
            && llvm::all_of(node.inputSlots[idx], [](const auto &slot) {
                 return llvm::isa<fc::kernel::VectorAluOp>(slot.op);
               })) {
          bool view = true;
          for (I j = 0; view && j < value.size(); ++j) {
            const auto ref = value.refs[j];
            view = ref.node == firstRef.node && ref.output == firstRef.output
                && ref.index == (j / value.shape[1]) * producerShape[1]
                                    + firstRef.index + j % value.shape[1];
          }
          if (view) {
            auto p = s.change(firstPlacement, J{{"kind", "fp16_pair_planar"},
                {"hemisphere", (firstRef.index % producerShape[1] / wordElements) % 2 ? "west" : "east"},
                {"base_row", field(firstPlacement, "base_row")
                    + (firstRef.index % producerShape[1]) / (2 * wordElements) * producerShape[0]
                    + firstRef.index / producerShape[1]},
                {"instruction_count", value.shape[0]}});
            node.in[idx] = p;
            for (auto &slot : node.inputSlots[idx]) slot.set(p);
            // Per-channel constants should enter from the same fabric as
            // the packed projection operand. In particular an opposite-side
            // scalar broadcast must not delay the binary head while raw
            // projection lanes continue through its previous-result SRs.
            for (I ci = 0; ci < I(node.inputs.size()); ++ci) {
              const auto &constant = node.inputs[ci];
              if (!constant.external() || constant.shape != V{1, value.shape[1]}
                  || node.inputSlots[ci].empty() || scalar(node.in[ci])) continue;
              const bool rowOperand = llvm::all_of(node.inputSlots[ci], [&](const auto &slot) {
                auto op = llvm::dyn_cast<fc::kernel::VectorAluOp>(slot.op);
                return op && !op.getUseImmediate()
                    && op.getLhs() == node.function.getArgument(idx)
                    && op.getRhs() == node.function.getArgument(ci);
              });
              if (!rowOperand) continue;
              auto constantPlacement = s.change(node.in[ci], J{{"hemisphere", text(p, "hemisphere")}});
              node.in[ci] = constantPlacement;
              for (auto &slot : node.inputSlots[ci]) slot.set(constantPlacement);
            }
            node.stagedInputs[idx] = true; // already resident in the consumer layout
            continue;
          }
        }
        if(planSegmentedMxmRhs(ni,idx) || planPlanarInput(ni,idx))continue;
        bool identity = true;
        for (I j = 0; j < value.size(); ++j)
          identity &= value.refs[j].node == firstRef.node &&
                      value.refs[j].output == firstRef.output &&
                      value.refs[j].index == j;
        auto key = Key{firstRef.node, firstRef.output};
        if (!identity ||
            planner.nodes[key.first].outputs[key.second].shape != value.shape)
          continue;
        auto src = cache.at(key), old = node.in[idx];
        if (auto shape = src.getAs<mlir::ArrayAttr>("storage_shape")) {
          V physical;
          for (auto d : shape) physical.push_back(llvm::cast<mlir::IntegerAttr>(d).getInt());
          if (physical != value.shape)
            continue; // The consumer must account for this physical view.
        }
        // A linear MXM consumes a two-slice activation pair, while the
        // preceding feedback RMSNorm produces sixteen distributed slices.
        // Stage that edge in the projection's own closed-form MEM/SR domain
        // instead of materializing it through the generic VXM copy bridge.
        if (text(src, "kind") == "fp16_mxm_distributed_16" &&
            text(old, "kind") == "fp16_pair_planar" &&
            slices(src).size() == 16 && slices(old).size() == 2 &&
            value.shape.size() == 2 && value.shape[0] % 32 == 0 &&
            value.shape[1] % 32 == 0 &&
            node.inputSlots[idx].size() == 1) {
          auto *owner = node.inputSlots[idx][0].op;
          if (auto projection = llvm::dyn_cast<fc::tensor::ProjectionTaskOp>(owner);
              projection && projection.getKind() == "linear") {
            mlir::NamedAttrList plan(projection.getMemoryPlan());
            if (!plan.get("source_input")) {
              plan.set("source_input", src);
              owner->setAttr("memory_plan", plan.getDictionary(owner->getContext()));
              node.stagedInputs[idx] = true;
            }
          }
        }
        if (node.stagedInputs[idx])
          continue;
        if (text(src, "kind") != text(old, "kind") ||
            text(src, "hemisphere") != text(old, "hemisphere") ||
            slices(src).size() != slices(old).size())
          continue;
        // Conv's mirrored pipeline and RMSNorm's distributed transpose have
        // fixed input/output sides; compatible aliases preserve those sides.
        node.in[idx] = src;
        for (auto &slot : node.inputSlots[idx])
          slot.set(src);
      }
      // Materialized VXM operands are writes too. Their standalone staging
      // locations may contain a retained activation needed by a later FFN
      // batch; reserving only node outputs does not protect that activation.
      for (I idx = 0; idx < I(node.inputs.size()); ++idx) {
        auto &value = node.inputs[idx];
        if (value.external() || node.stagedInputs[idx]
            || !llvm::all_of(node.inputSlots[idx], [](const auto &slot) {
                 return llvm::isa<fc::kernel::VectorAluOp>(slot.op);
               })) continue;
        auto p = node.in[idx];
        bool alias = !value.refs.empty();
        if (alias) {
          const auto ref = value.refs.front();
          alias = cache.at({ref.node,ref.output}) == p
              && planner.nodes[ref.node].outputs[ref.output].shape == value.shape;
          for (I j = 0; alias && j < value.size(); ++j)
            alias = value.refs[j].node == ref.node
                && value.refs[j].output == ref.output && value.refs[j].index == j;
        }
        // Two fixed VXM inputs are issued together. A row-scale operand
        // can be scalar-broadcast while the value operand is packed, but
        // their MEM ICUs must still be distinct, even for disjoint rows.
        const auto update=node.function->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_packed_update");
        bool conflict=false;
        if((update && update.getValue()) || node.function->hasAttr("ftlpu.vector_independent_ports"))
          for(I other=0;other<I(node.in.size());++other)
            if(other!=idx && overlap(p,node.in[other])) conflict=true;
        if (alias && !conflict) continue;
        for (const auto &entry : live)
          conflict |= overlap(p,entry.second)
              && field(p,"base_row") < field(entry.second,"base_row") + field(entry.second,"instruction_count")
              && field(entry.second,"base_row") < field(p,"base_row") + field(p,"instruction_count");
        if (!conflict) continue;
        const Key staging{ni, -idx-1};
        last[staging] = ni;
        std::vector<mlir::DictionaryAttr> busy = node.workspaces;
        for (I other = 0; other < I(node.in.size()); ++other)
          if (other != idx) busy.push_back(node.in[other]);
        p = allocate(staging,value,text(p,"hemisphere"),!scalar(p),p,busy);
        node.in[idx] = p;
        for (auto &slot : node.inputSlots[idx]) slot.set(p);
      }
      for (I oi = 0; oi < I(node.outputs.size()); ++oi) {
        Key key{ni, oi};
        if (!last.count(key))
          continue;
        auto matmul=llvm::dyn_cast<fc::kernel::VectorAluOp>(node.outputSlots[oi].op);
        bool matmulAdd=false;
        if(matmul) {
          auto prior=matmul.getLhs().getDefiningOp<fc::kernel::VectorAluOp>();
          if(isNativeMatmulCast(prior,matmul) &&
             isNativeMatmulAdd(prior.getLhs().getDefiningOp<fc::kernel::VectorAluOp>(),prior))matmul=prior;
        }
        if(matmul && node.in.size()==3) {
          auto product=matmul.getLhs().getDefiningOp<fc::kernel::VectorAluOp>();
          if(isNativeMatmulAdd(product,matmul) && product.getLhs()==node.function.getArgument(0) &&
             product.getRhs()==node.function.getArgument(1) && matmul.getRhs()==node.function.getArgument(2)) {
            matmul=product;matmulAdd=true;
          }
        }
        if(matmul && matmul.getKind()=="head_matmul" && node.in.size()>=2 &&
           (node.in[0].get("mxm_lhs_head_rows") || text(node.in[0],"kind")=="token_partition_east_first_planar" || text(node.in[0],"kind")=="fp32_quad_planar" ||
            (text(node.in[0],"kind")=="fp16_pair_planar" && text(node.in[0],"hemisphere")=="both" && slices(node.in[0]).size()==2)) &&
           (text(node.in[1],"kind")=="token_partition_east_first_planar" || text(node.in[1],"kind")=="fp32_quad_planar" ||
            (text(node.in[1],"kind")=="fp16_pair_planar" && text(node.in[1],"hemisphere")=="both" && slices(node.in[1]).size()==2))) {
          // Read matching native head groups where their producers left them.
          // MXM drains across the passive bridge, so only the output ownership
          // changes; no H/D interleave or hemisphere copy is required.
          // Replicated quad-planar matrices also read logical heads 0..H/2
          // on East, unlike the default West-first partitioned inputs.
          node.out[oi]=s.change(node.out[oi],J{{"kind","token_partition_planar"}});
        }
        if(matmulAdd && node.in.size()==3 &&
           text(node.in[0],"hemisphere")=="both" && text(node.in[1],"hemisphere")=="both" &&
           (node.in[0].get("mxm_lhs_head_rows") || text(node.in[0],"kind")=="fp16_pair_planar" || text(node.in[0],"kind")=="fp32_quad_planar") &&
           (text(node.in[1],"kind")=="fp16_pair_planar" || text(node.in[1],"kind")=="fp32_quad_planar") &&
           (text(node.in[2],"kind")=="token_partition_planar" || text(node.in[2],"kind")=="token_partition_east_first_planar")) {
          // VXM follows the prior sum's owner; the streamed result crosses
          // once to the opposite fabric. Successive K tiles alternate owners
          // instead of copying the partial sum back into mirrored storage.
          node.out[oi]=s.change(node.out[oi],J{{"kind",text(node.in[2],"kind")=="token_partition_planar"?
              "token_partition_east_first_planar":"token_partition_planar"}});
        }
        if (planRetainedRmsNorm(ni,oi) || planRetainedAttention(ni,oi) || planInPlacePointwise(ni,oi) || planGroupedPointwise(ni,oi))
          continue;
        if (oi==3 && llvm::isa<fc::kernel::ScaledMatvecOp>(node.outputSlots[oi].op)) {
          // The rank-one phase starts only after scaling/contraction drains.
          // Reuse a dead external matrix home, preserving the separately
          // retained scaled matrix. Never overwrite a shared/live snapshot.
          const auto &matrix=node.inputs[0];
          auto home=node.in[0];
          Key matrixKey{-1,-1};
          bool reusable=false;
          if(matrix.external()) reusable=externalVectorHomes.count({ni,0});
          else if(!matrix.refs.empty()) {
            const auto ref=matrix.refs[0];matrixKey={ref.node,ref.output};
            reusable=last.at(matrixKey)==ni && cache.at(matrixKey)==home &&
                planner.nodes[ref.node].outputs[ref.output].shape==matrix.shape;
            for(I j=0;reusable && j<matrix.size();++j)
              reusable=matrix.refs[j].node==ref.node && matrix.refs[j].output==ref.output && matrix.refs[j].index==j;
          }
          // Non-overlapping rows still share one MEM ICU queue. The update
          // writes must not interleave with K or broadcast-table reads.
          for(I operand=1;operand<I(node.in.size());++operand)
            if(overlap(home,node.in[operand]))reusable=false;
          for(auto workspace:node.workspaces)
            if(overlap(home,workspace))reusable=false;
          const auto sameSource=[&](const Tensor &t) {
            return t.external() && (t.constant==matrix.constant ||
                (t.dtype==matrix.dtype && t.data==matrix.data));
          };
          for(I future=ni+1;future<I(planner.nodes.size());++future)
            for(const auto &t:planner.nodes[future].inputs)
              if(sameSource(t))reusable=false;
          for(const auto &t:finals)if(sameSource(t))reusable=false;
          const auto intersects=[&](mlir::DictionaryAttr other) {
            return overlap(home,other)
                && field(home,"base_row")<field(other,"base_row")+field(other,"instruction_count")
                && field(other,"base_row")<field(home,"base_row")+field(home,"instruction_count");
          };
          for(const auto &entry:externalVectorHomes)
            if(entry.first.first>=ni && entry.first!=Key{ni,0} && intersects(entry.second))reusable=false;
          for(const auto &entry:live)
            if(intersects(entry.second) && entry.first!=matrixKey &&
                !(entry.first.first==-2 && entry.second==home))reusable=false;
          if(reusable) {
            node.outputSlots[oi].set(home);node.out[oi]=home;
            cache[key]=live[key]=home;
            continue;
          }
        }
        // Choose the consumer layout at the L2 producer. Four physical rows
        // hold one D32 tile; alternate their bank without duplicating K.
        if(auto norm=llvm::dyn_cast<fc::tensor::L2NormTaskOp>(node.outputSlots[oi].op);
            norm && norm.getInput().getType().getDimSize(1)==128 && norm->hasAttr("native_input_groups")) {
          bool used=false, onlyGram=true;
          for(auto &consumer:planner.nodes) for(I operand=0;operand<I(consumer.inputs.size());++operand) {
            if(!llvm::any_of(consumer.inputs[operand].refs,[&](const Ref &ref){return ref.node==ni && ref.output==oi;})) continue;
            used=true;
            if(!consumer.module) { onlyGram=false; continue; }
            const auto &view=consumer.inputs[operand];
            onlyGram &= view.size()==node.outputs[oi].size();
            for(I j=0;onlyGram && j<view.size();++j)
              onlyGram &= view.refs[j].node==ni && view.refs[j].output==oi && view.refs[j].index==j;
            onlyGram &= operand<=1 && !consumer.inputSlots[operand].empty() && llvm::all_of(consumer.inputSlots[operand],[](const auto &slot){
              auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(slot.op); return op && (op.getKind()=="weighted_gram" || op.getKind()=="cross_gram" || op.getKind()=="head_last_row_scale_transpose");
            });
          }
          auto groups=norm->getAttrOfType<mlir::ArrayAttr>("native_input_groups");
          bool paired=groups && !groups.empty() && groups.size()%2==0 && integer(norm,"native_input_rows")==32;
          for(I g=0;paired && g<I(groups.size());++g) {
            auto descriptor=llvm::cast<mlir::DictionaryAttr>(groups[g]);
            auto source=descriptor.getAs<mlir::DictionaryAttr>("placement");
            paired=source && !descriptor.getAs<mlir::ArrayAttr>("chunks") &&
                (text(source,"hemisphere")=="east")== (g%2==0);
          }
          auto wide=(*node.module)->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
          auto split=(*node.module)->getAttrOfType<mlir::BoolAttr>("ftlpu.l2_head_partition");
          if(used && onlyGram && paired && wide && wide.getValue() && split && split.getValue()
              && s.target.throughput().mxm_weight_activation_overlap_enabled)
            node.out[oi]=s.change(node.out[oi],J{{"bank_tile_rows",4}});
        }
        auto busy = node.in;
        reserveProjectionConvQueues(ni,busy);
        if(node.projectionSolvePeer>=0) {
          auto peerQueues=projectionSolveQueues(node.projectionSolvePeer);
          busy.insert(busy.end(),peerQueues.begin(),peerQueues.end());
        }
        // Native head scaling reads V from per-word producer descriptors,
        // not from its standalone input prototype. Its pipelined writes must
        // not share those actual MEM ICU queues, even at disjoint addresses.
        if (auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(node.outputSlots[oi].op))
          if (auto words=op->getAttrOfType<mlir::ArrayAttr>("solve.value_words"))
            for (auto word:words) {
              auto p=llvm::cast<mlir::DictionaryAttr>(word).getAs<mlir::DictionaryAttr>("placement");
              if (std::find(busy.begin(),busy.end(),p)==busy.end()) busy.push_back(p);
            }
        // Returned intermediates can still feed a later ALU in this region.
        // Moving their homes must preserve separate read/write ICU queues,
        // not merely disjoint SRAM rows (e.g. softmax denominator and result).
        if (llvm::isa<fc::kernel::VectorAluOp>(node.outputSlots[oi].op))
          for (I other=0; other<I(node.out.size()); ++other)
            if (other!=oi) busy.push_back(node.out[other]);
        if(node.function->hasAttr("ftlpu.native_silu_multiply"))
          for(const auto &operand:node.inputs)for(const auto &ref:operand.refs) {
            auto home=cache.find({ref.node,ref.output});
            if(home!=cache.end() && std::find(busy.begin(),busy.end(),home->second)==busy.end())busy.push_back(home->second);
          }
        if (oi==4 && llvm::isa<fc::kernel::ScaledMatvecOp>(node.outputSlots[oi].op)) {
          // Rank-one/readout begins after all decay, V and beta reads finish.
          // K and Q remain live; other input queues may hold readout at
          // disjoint addresses without intersecting those earlier domains.
          busy={node.in[2],node.in[5]};
        }
        busy.insert(busy.end(), node.workspaces.begin(), node.workspaces.end());
        // Direct planar RMSNorm reads later feature words while the SXM taps
        // earlier words into its replay buffer. Reserve distinct MEM ICU
        // queues at the producer, even when the SRAM rows do not overlap.
        for(auto &consumer:planner.nodes) {
          if(!consumer.module || consumer.inputs.empty() ||
             !consumer.function->hasAttr("ftlpu.norm_retained_matrix_input") ||
             !llvm::any_of(consumer.inputs[0].refs,[&](const Ref &ref){return ref.node==ni && ref.output==oi;}))continue;
          consumer.function.walk([&](fc::tensor::RmsNormTaskOp norm) {
            if(norm.getInput().getType().getDimSize(0)!=32)return;
            auto scratch=norm.getScratchAllocations();
            if(!scratch.empty())busy.push_back(llvm::cast<mlir::DictionaryAttr>(scratch[0]).getAs<mlir::DictionaryAttr>("placement"));
          });
        }
        // Shared-D products and native SiLU chains read multiple producer
        // homes concurrently. Reserve independent operand queues at their
        // producers, without staging them.
        for(auto &consumer:planner.nodes) {
          // Synthetic view/materialization nodes have no native function.
          if(!consumer.module) continue;
          bool hasScale=false,hasMatmul=false,usesOutput=false;
          consumer.function.walk([&](fc::kernel::VectorAluOp op) {
            hasScale |= op.getKind()=="head_last_row_scale_transpose";
            hasMatmul |= op.getKind()=="head_matmul";
          });
          const auto packedUpdate=consumer.function->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_packed_update");
          const bool distinctOperands=consumer.function->hasAttr("ftlpu.vector_scalar_layout") ||
              consumer.function->hasAttr("ftlpu.vector_independent_ports") ||
              (packedUpdate && packedUpdate.getValue());
          if(!(hasScale && hasMatmul) && !consumer.function->hasAttr("ftlpu.native_silu_multiply") && !distinctOperands)continue;
          for(auto &operand:consumer.inputs)for(const auto &ref:operand.refs)
            usesOutput |= ref.node==ni && ref.output==oi;
          if(!usesOutput)continue;
          for(auto &operand:consumer.inputs) {
            // Segments of one SiLU operand are read consecutively, so they
            // may share a queue at distinct addresses. Only the other operand
            // is simultaneous; spreading every segment over new queues can
            // also push the norm's return path onto its feedback SR fabric.
            if((consumer.function->hasAttr("ftlpu.native_silu_multiply") || distinctOperands) &&
                llvm::any_of(operand.refs,[&](const Ref &ref){return ref.node==ni && ref.output==oi;}))continue;
            for(const auto &ref:operand.refs) {
              auto home=cache.find({ref.node,ref.output});
              if(home!=cache.end() && std::find(busy.begin(),busy.end(),home->second)==busy.end())busy.push_back(home->second);
            }
          }
        }
        // ScaledMatvec overlaps successive heads: a previous head's delta
        // broadcast can still read while KS/readout writes the next result.
        // Separate result queues, even when their SRAM row ranges differ.
        if (llvm::isa<fc::kernel::ScaledMatvecOp>(node.outputSlots[oi].op))
          for (I prior=0; prior<oi; ++prior)
            if (oi<3 || prior!=1)
              busy.push_back(node.out[prior]);
        // A streamed Gram epilogue reads its coefficient while the final K
        // activation domain is still live. Plan the producer's final home
        // on independent ICU queues; different SRAM rows are insufficient.
        // This keeps the fused SXM output native instead of copying it at A.
        // Pipelined head_scale likewise broadcasts beta while native V reads
        // are active; choose beta's producer home off the V ICU queues.
        for (auto &consumer : planner.nodes) {
          if (consumer.inputs.size()<2 || consumer.inputSlots.size()<2 ||
              consumer.inputSlots[1].empty()) continue;
          if (!llvm::all_of(consumer.inputSlots[1],[](const auto &slot) {
                auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(slot.op);
                return op && (op.getKind()=="head_last_row_scale_transpose" || op.getKind()=="head_scale" || (op.getKind()=="unit_lower_solve" || op.getKind()=="unit_lower_solve_blocked") || (op.getKind()=="weighted_gram" && op.getRhs().getType().getElementType().isBF16()));
              })) continue;
          if (!llvm::any_of(consumer.inputs[1].refs,[&](const Ref &ref) {
                return ref.node==ni && ref.output==oi;
              })) continue;
          for (const auto &ref : consumer.inputs[0].refs) {
            auto home=cache.find({ref.node,ref.output});
            if (home!=cache.end() && std::find(busy.begin(),busy.end(),home->second)==busy.end())
              busy.push_back(home->second);
          }
        }
        for(auto &consumer:planner.nodes) {
          if(consumer.inputs.size()!=3 || consumer.inputSlots.size()!=3 || consumer.inputSlots[0].empty()) continue;
          auto contraction=llvm::dyn_cast<fc::kernel::VectorAluOp>(consumer.inputSlots[0][0].op);
          if(!contraction || contraction.getKind()!="cross_gram") continue;
          if(!llvm::any_of(consumer.inputs[2].refs,[&](const Ref &ref){return ref.node==ni && ref.output==oi;})) continue;
          for(I operand=0;operand<2;++operand) for(const auto &ref:consumer.inputs[operand].refs) {
            auto home=cache.find({ref.node,ref.output});
            if(home!=cache.end() && std::find(busy.begin(),busy.end(),home->second)==busy.end()) busy.push_back(home->second);
          }
        }
        if(oi==4 && llvm::isa<fc::kernel::ScaledMatvecOp>(node.outputSlots[oi].op))
          // Readout returns while the next updated-state tile still drains.
          // Different rows cannot make two commands share this MEM ICU.
          busy.push_back(node.out[3]);
        if (node.ffn) {
          // The FFN result is written by Down after Gate/Up have retired.
          // Protect Down's live operand queues, rather than every queue that
          // was used earlier anywhere inside the composite operator.
          busy.clear();
          auto down=llvm::cast<fc::tensor::MatmulTaskOp>(node.outputSlots[oi].op);
          for (auto array : {down.getLhsAllocations(),down.getRhsAllocations()})
            for (auto a : array) {
              auto p=llvm::cast<mlir::DictionaryAttr>(a).getAs<mlir::DictionaryAttr>("placement");
              busy.push_back(p);
              // Logical weight stream slices can differ from their physical
              // C2C page homes. Down reads partial sums while reading weights,
              // so exclude every physical page queue, not only logical IDs.
              if(auto storage=p.getAs<mlir::ArrayAttr>("page_storage_slices"))
                if(auto banks=p.getAs<mlir::ArrayAttr>("page_banks")) {
                  std::set<I> used;
                  for(auto b:banks)used.insert(llvm::cast<mlir::IntegerAttr>(b).getInt());
                  V ids;for(auto id:storage)ids.push_back(llvm::cast<mlir::IntegerAttr>(id).getInt());
                  for(I bank:used)busy.push_back(s.change(p,J{{"slices",s.array(ids)},{"bank",bank}}));
                }
            }
        }
        if (auto output = llvm::dyn_cast<fc::tensor::ProjectionTaskOp>(node.outputSlots[oi].op);
            output && output.getKind() == "output") {
          // A composite attention result is produced only after Q/K/RoPE,
          // softmax and PV have retired. Respect the native late-stage plan
          // instead of treating every scratch queue as live at O projection.
          busy.clear();
          for (auto entry : output.getMemoryPlan())
            if (entry.getName().strref() != "result")
              if (auto p = llvm::dyn_cast<mlir::DictionaryAttr>(entry.getValue())) busy.push_back(p);
          node.function.walk([&](fc::tensor::SigmoidMultiplyTaskOp gate) {
            busy.push_back(gate.getMemoryPlan().getAs<mlir::DictionaryAttr>("gated_context"));
          });
        }
        // A narrow projection used exclusively by token-partitioned vector
        // consumers can scatter its first output word directly at the MXM
        // stream egress. Padded feature words must have no live consumers.
        if (auto projection = llvm::dyn_cast<fc::tensor::ProjectionTaskOp>(node.outputSlots[oi].op);
            projection && projection.getKind() == "linear" && node.projectionSolvePeer<0
            && s.target.memory().hemispheres == 2
            && s.target.throughput().mxm_rows == s.target.memory().bytes_per_word
            && node.outputs[oi].shape == V{s.target.throughput().mxm_rows,
                                          2 * s.target.memory().bytes_per_word}) {
          const I word = s.target.memory().bytes_per_word;
          auto policy = fc::target::mxm_execution_policy_from_operation(projection.getOperation());
          if (mlir::failed(policy)) throw std::runtime_error("invalid projection execution policy");
          const auto inputType = llvm::cast<mlir::RankedTensorType>(projection.getInput().getType());
          const auto weightType = llvm::cast<mlir::RankedTensorType>(projection.getWeight().getType());
          auto strategy = fc::target::plan_mxm_execution_strategy(
              {node.outputs[oi].shape[0], node.outputs[oi].shape[1], inputType.getShape()[1],
               inputType.getElementType().isBF16(), weightType.getElementType().isInteger(8),
               true, true}, s.target, *policy);
          bool eligible = mlir::succeeded(strategy) && strategy->uses_local_dequant(), used = false;
          for (auto &consumer : planner.nodes)
            for (auto &input : consumer.inputs)
              for (auto ref : input.refs) if (ref.node == ni && ref.output == oi) {
                used = true;
                eligible &= consumer.function && (consumer.function->hasAttr("ftlpu.vector_token_partition")
                    || consumer.function->hasAttr("ftlpu.vector_bf16_token_halves"))
                    && ref.index % (2 * word) < word;
              }
          for (auto &output : finals)
            for (auto ref : output.refs)
              if (ref.node == ni && ref.output == oi) eligible = false;
          if (used && eligible)
            node.out[oi] = s.change(node.out[oi], J{{"kind", "token_partition_planar"},
                {"instruction_count", s.target.throughput().mxm_rows}});
        }
        const bool partition = node.function->hasAttr("ftlpu.vector_token_partition");
        // Mirrored RMSNorm needs both column halves. Place the producer
        // farther from the VXM boundary than that consumer's input so one
        // inward read can tap the local home and continue to the other side.
        // allocate() still proves lifetime/capacity and excludes active queues.
        if(text(node.out[oi],"kind")=="fp16_hemisphere_pair_planar") {
          I inputGroup=-1;
          for(const auto &consumer:planner.nodes) {
            if(consumer.inputs.empty() || consumer.inputSlots.empty() || consumer.inputSlots[0].size()!=1)continue;
            auto norm=llvm::dyn_cast<fc::tensor::RmsNormTaskOp>(consumer.inputSlots[0][0].op);
            if(!norm)continue;
            auto planar=norm.getConfig().getAs<mlir::BoolAttr>("planar_streaming");
            if(!planar || !planar.getValue())continue;
            if(llvm::none_of(consumer.inputs[0].refs,[&](const Ref &ref){return ref.node==ni && ref.output==oi;}))continue;
            for(I slice:slices(consumer.in[0]))inputGroup=std::max(inputGroup,slice/s.group);
          }
          if(inputGroup>=0) {
            const auto storage=s.target.activation_storage_slices();
            for(size_t pair=0;pair+1<storage.size();pair+=2)
              if(storage[pair]/s.group>inputGroup && storage[pair+1]==storage[pair]+1) {
                node.out[oi]=s.change(node.out[oi],J{{"slices",s.array({storage[pair],storage[pair+1]})}});
                break;
              }
          }
        }
        const Key partitionKey{node.parallel, oi};
        mlir::DictionaryAttr dest;
        if (partition && partitionOutputHomes.count(partitionKey)) {
          dest = s.change(partitionOutputHomes.at(partitionKey), J{{"hemisphere", "east"},
              {"instruction_count", field(node.out[oi], "instruction_count")}});
          cache[key] = live[key] = dest;
        } else if (partition) {
          auto prototype = s.change(node.out[oi], J{{"hemisphere", "both"}});
          if (ni + 1 < I(planner.nodes.size()) && planner.nodes[ni+1].parallel == node.parallel) {
            auto &peer = planner.nodes[ni+1];
            busy.insert(busy.end(), peer.in.begin(), peer.in.end());
            busy.insert(busy.end(), peer.workspaces.begin(), peer.workspaces.end());
            last[key] = std::max(last[key], last[{ni+1,oi}]);
          }
          auto home = allocate(key, node.outputs[oi], "both", 1, prototype, busy);
          partitionOutputHomes[partitionKey] = home;
          dest = s.change(home, J{{"hemisphere", "west"}});
          cache[key] = dest; // live retains both halves until their final consumer
        } else if (auto projection = llvm::dyn_cast<fc::tensor::ProjectionTaskOp>(node.outputSlots[oi].op);
                   projection && projection.getKind() == "output"
                   && [&] {
                     const auto p = node.out[oi];
                     const I begin = field(p, "base_row"), end = begin + field(p, "instruction_count");
                     // The native attention allocator has already checked O
                     // against its live context. Earlier Q/K/RoPE scratch is
                     // retired here; only retained values and later kernels
                     // can invalidate this late result placement.
                     for (const auto &entry : live)
                       if (overlap(p, entry.second)
                           && begin < field(entry.second,"base_row") + field(entry.second,"instruction_count")
                           && field(entry.second,"base_row") < end) return false;
                     for (auto h : {"east", "west"})
                       if (text(p,"hemisphere") == h || text(p,"hemisphere") == "both")
                         for (I slice : slices(p))
                           for (auto r : nativeTimed[{h,slice}])
                             if (r.node > ni && r.node <= last[key]
                                 && begin < r.end && r.begin < end) return false;
                     return true;
                   }()) {
          dest = node.out[oi];
          cache[key] = live[key] = dest;
        } else
        if (node.recurrent && oi == 1) {
          dest = recurrentStates.at(ni);
          cache[key] = dest;
        } else
          try {
            dest = allocate(key, node.outputs[oi], node.recurrent ? text(node.out[oi],"hemisphere") : "west",
                            !scalar(node.out[oi]), node.out[oi], busy);
          } catch (const std::runtime_error &) {
            if (!llvm::isa<fc::kernel::VectorAluOp>(node.outputSlots[oi].op))
              throw;
            // A VXM result can drain to either hemisphere. Spread long-lived
            // reduction outputs instead of introducing an archive transfer.
            auto alternate = s.change(
                node.out[oi],
                J{{"hemisphere", text(node.out[oi], "hemisphere") == "west"
                                     ? "east"
                                     : "west"}});
            dest = allocate(key, node.outputs[oi], "west", !scalar(alternate),
                            alternate, busy);
          }
        if (auto *owner = node.outputSlots[oi].op;
            owner->hasAttr("native_input_groups")) {
          const I tokens = integer(owner, "native_input_rows");
          dest = s.change(dest, J{{"storage_shape", s.array(
              {tokens, node.outputs[oi].size() / tokens})}});
          cache[key] = live[key] = dest;
        }
        node.outputSlots[oi].set(dest);
        node.out[oi] = dest;
      }
    }
    // All assignments are now fixed. Only now may native emitters generate
    // MEM/MXM/VXM/SXM domains, deriving route timing from the chosen slices.
    for (auto &node : planner.nodes) {
      if (!node.module) continue;
      const I nodeIndex=&node-planner.nodes.data();
      bool terminal=true;
      for(const auto &consumer:planner.nodes)
        for(const auto &input:consumer.inputs)
          for(const auto &ref:input.refs) terminal &= ref.node!=nodeIndex;
      if(terminal) {
        node.function->setAttr("ftlpu.vector_native_partition_outputs",s.b.getUnitAttr());
        node.function->setAttr("ftlpu.l2_native_partition_output",s.b.getUnitAttr());
      }
      node.function.walk([&](fc::tensor::ProjectionTaskOp op) {
        const auto config = op.getConfig();
        const auto scales = config.getAs<mlir::DenseF32ArrayAttr>("rhs_block_scales");
        node.nativeProjectionReport.push_back(J{{"kind", op.getKind().str()},
            {"m", field(config, "m")}, {"n", field(config, "n")}, {"k", field(config, "k")},
            {"weight_dtype", llvm::isa<mlir::Float8E4M3FNType>(op.getWeight().getType().getElementType()) ? "f8E4M3FN" : "i8"},
            {"rhs_scale_block_count", scales ? I(scales.size()) : 0}});
      });
      mlir::PassManager pm(&s.context);
      pm.enableVerifier(false);
      pm.addNestedPass<mlir::func::FuncOp>(
          fc::create_lower_tensor_to_stream_pass());
      pm.addNestedPass<mlir::func::FuncOp>(
          fc::create_lower_stream_to_schedule_pass(planner.ffnStrategy));
      if (mlir::failed(pm.run(*node.module)))
        throw std::runtime_error("planned graph schedule lowering failed: " +
                                 node.name);
      auto ret = llvm::cast<mlir::func::ReturnOp>(
          node.function.front().getTerminator());
      for (I oi = 0; oi < I(node.out.size()); ++oi) {
        auto binding =
            ret.getOperand(oi).getDefiningOp<fc::schedule::BindingOp>();
        if (binding && text(binding.getPlacement(),"kind")=="fp16_mxm_block4_partition_distributed_16") {
          auto layout=binding.getPlacement();
          if(s.change(layout,J{{"kind",text(node.out[oi],"kind")}})!=node.out[oi])
            throw std::runtime_error("native L2 output changed allocated storage");
          node.out[oi]=layout; cache[{nodeIndex,oi}]=layout;
        }
        if (binding && binding->hasAttr("token_phase")
            && (text(binding.getPlacement(),"kind")=="token_partition_planar"
                || text(binding.getPlacement(),"kind")=="token_partition_east_first_planar")) {
          auto layout=binding.getPlacement();
          if(s.change(layout,J{{"kind",text(node.out[oi],"kind")}})!=node.out[oi])
            throw std::runtime_error("token partition changed allocated storage");
          node.out[oi]=layout;
          const I index=&node-planner.nodes.data();
          cache[{index,oi}]=layout;
        }
        if (!binding || binding.getPlacement() != node.out[oi]) {
          llvm::errs() << "planned output: " << node.out[oi] << "\n";
          if (binding) llvm::errs() << "emitted output: " << binding.getPlacement() << "\n";
          throw std::runtime_error(
              "native emitter did not honor graph output placement: " +
              node.name);
        }
      }
    }
  }
  void checkProjectionConvQueues() {
    using Queue = std::tuple<I,I,I>;
    struct Usage {
      std::set<Queue> mem;
      std::set<I> streams;
      bool mxm=false, vxm=false, sxm=false;
    };
    const auto usage=[](Node& node) {
      Usage result;
      node.function.walk([&](mlir::Operation* op) {
        auto name=op->getName().getStringRef();
        result.mxm |= name.starts_with("ftlpu.schedule.mxm");
        result.vxm |= name.starts_with("ftlpu.schedule.vxm");
        result.sxm |= name.starts_with("ftlpu.schedule.sxm");
        if(auto transfer=llvm::dyn_cast<fc::schedule::MemTransferOp>(op)) {
          result.mem.emplace(transfer.getHemisphere(), transfer.getSlice(), integer(op,"bank"));
          result.streams.insert(transfer.getPackedStream());
        }
      });
      return result;
    };
    for(I i=0;i<I(planner.nodes.size());++i) {
      auto& page=planner.nodes[i];
      if(page.projectionConvGroup!=i)continue;
      const auto p=usage(page);
      std::string reason="enabled";
      if(!p.mxm || p.vxm || p.sxm || p.mem.empty() ||
         llvm::any_of(p.streams,[](I sr){return sr!=10 && sr!=11 && sr!=31 && sr!=42 && sr!=43;}))
        reason="serialized_projection_resources";
      I end=i+1;
      llvm::json::Array names;
      for(;end<I(planner.nodes.size()) && planner.nodes[end].projectionConvGroup==i;++end) {
        auto& conv=planner.nodes[end];
        const auto v=usage(conv);
        names.push_back(conv.name);
        // The mirrored member shares its partner's VXM columns and emits
        // only its physical MEM transfers. Each pair needs exactly one
        // native VXM program, emitted by its non-mirrored member.
        if(v.mxm || v.sxm || v.mem.empty() ||
           (conv.projectionVectorMember ? !v.vxm :
               v.vxm==conv.function->hasAttr("ftlpu.depthwise_conv_mirror")))
          reason="serialized_conv_resources";
        for(I sr:v.streams) {
          if(p.streams.count(sr))reason="serialized_shared_sr";
          // Runtime C2C sinks use the highest westbound stream IDs.
          if(sr >= 2*s.target.streams().streams_per_direction -
                   s.target.streams().c2c_streams_per_direction)reason="serialized_c2c_sr";
        }
        for(auto queue:v.mem) {
          if(p.mem.count(queue))reason="serialized_shared_mem_queue";
          // Both banks of these queues can service the current/next weight
          // page. Never move an ordinary Conv instruction across WRITE_SYNC.
          if(weightPingPong && weightSlotSlices.count(std::get<1>(queue)))
            reason="serialized_weight_prefetch_queue";
        }
      }
      for(I j=i;j<end;++j) {
        auto& member=planner.nodes[j];
        if(member.function->hasAttr("ftlpu.imem_drain_after"))reason="serialized_drain";
        for(I output=0;output<I(member.outputs.size());++output)
          if(retainedOutputHomes.count({j,output}))reason="serialized_output_copy";
      }
      for(I j=i;j<end;++j)planner.nodes[j].projectionConvReady=reason=="enabled";
      if(page.projectionVectorMember)
        projectionVectorReport.push_back(J{{"projection",page.name},{"vectors",std::move(names)},
                                          {"status",reason}});
      else
        projectionConvReport.push_back(J{{"projection",page.name},{"convolutions",std::move(names)},
                                         {"status",reason}});
      i=end-1;
    }
  }
  void checkProjectionSolveQueues() {
    using Queue=std::tuple<I,I,I>;
    for(I ni=0;ni<I(planner.nodes.size());++ni) {
      auto& projection=planner.nodes[ni];
      const I peer=projection.projectionSolvePeer;
      if(peer<=ni)continue;
      auto& solve=planner.nodes[peer];
      const auto memQueues=[](Node& node) {
        std::set<Queue> queues;
        node.function.walk([&](fc::schedule::MemTransferOp op) {
          queues.emplace(op.getHemisphere(),op.getSlice(),integer(op,"bank"));
        });
        return queues;
      };
      auto p=memQueues(projection),q=memQueues(solve);
      bool conflict=llvm::any_of(p,[&](const Queue& queue){return q.count(queue)!=0;});
      bool projectionUsesVector=false,projectionStagesInput=false;
      projection.function.walk([&](fc::schedule::MemTransferOp op) {
        const I stream=op.getPackedStream();
        // A distributed activation staging prologue is passive MEM/SR,
        // not VXM. It must not share the solve's fixed-port streams.
        projectionStagesInput |= stream!=8 && stream!=9 && stream!=31
            && stream!=40 && stream!=41;
      });
      projection.function.walk([&](mlir::Operation* op) {
        const auto name=op->getName().getStringRef();
        projectionUsesVector |= name.starts_with("ftlpu.schedule.vxm") || name.starts_with("ftlpu.schedule.sxm");
      });
      const bool enabled=!conflict && !projectionUsesVector && !projectionStagesInput;
      projectionSolveReport.push_back(J{{"projection",projection.name},{"solve",solve.name},
          {"status",enabled?"enabled":conflict?"serialized_shared_mem_queue":"serialized_projection_staging"},
          {"projection_mem_queues",I(p.size())},{"solve_mem_queues",I(q.size())},
          {"compute_imem_reload","VXM/SXM only; MEM/MXM retain continuous programs"}});
      if(!enabled) {
        projection.projectionSolvePeer=-1;solve.projectionSolvePeer=-1;
      }
    }
  }
  void verifyProjectionVectorEmission(I ni, mlir::Operation *watermark,
                                      mlir::Operation *preparationEnd) {
    const auto &node=planner.nodes[ni];
    if(!node.projectionVectorMember || !node.projectionConvReady ||
       node.projectionConvGroup==ni)return;
    using Queue=std::tuple<I,I,I>;
    std::set<Queue> queues;
    std::set<I> streams;
    planner.nodes[node.projectionConvGroup].function.walk([&](fc::schedule::MemTransferOp op) {
      queues.emplace(op.getHemisphere(),op.getSlice(),integer(op,"bank"));
      streams.insert(op.getPackedStream());
    });
    // Preparation is scheduled inside the projection interval as well. Check
    // every emitted transport, including broadcasts and passive scratch hops.
    // A failed proof discards this schedule and retries from the original SSA.
    bool preparing=watermark!=preparationEnd;
    for(auto *op=watermark->getNextNode();op;op=op->getNextNode()) {
      if(auto mem=llvm::dyn_cast<fc::schedule::MemTransferOp>(op)) {
        if(queues.count({mem.getHemisphere(),mem.getSlice(),integer(op,"bank")}) ||
           streams.count(mem.getPackedStream()) ||
           (weightPingPong && weightSlotSlices.count(mem.getSlice())))
          throw GraphSramAllocationFailure("projection/vector preparation shares a MEM queue or SR");
      } else if(op->getName().getStringRef().starts_with("ftlpu.schedule.sxm")) {
        for(auto name:{"source_streams","destination_streams"})
          if(auto ids=op->getAttrOfType<mlir::ArrayAttr>(name))
            for(auto id:ids)if(streams.count(llvm::cast<mlir::IntegerAttr>(id).getInt()))
              throw GraphSramAllocationFailure("projection/vector transpose shares an SR");
      } else if(op->getName().getStringRef().starts_with("ftlpu.schedule.mxm")) {
        throw GraphSramAllocationFailure("projection/vector preparation requires MXM");
      } else if(preparing && op->getName().getStringRef().starts_with("ftlpu.schedule.vxm")) {
        throw GraphSramAllocationFailure("projection/vector preparation requires VXM");
      }
      if(op==preparationEnd)preparing=false;
    }
  }
  void prepare(I ni, std::set<Key> &prepared) {
    auto &node = planner.nodes[ni];
    for (I idx = 0; idx < I(node.inputs.size()); ++idx) {
      auto &a = node.inputs[idx];
      auto p = node.in[idx];
        if (!a.external()) {
          if (!node.stagedInputs[idx])
            materialize(a, p);
        prepared.insert({ni, idx});
      } else if (node.stagedInputs[idx]) {
        prepared.insert({ni,idx});
      } else if (scalar(p) || text(p, "kind") == "fp16_pair_planar" ||
                 text(p, "kind") == "fp32_quad_planar" ||
                 text(p, "kind") == "fp16_mxm_distributed_16") {
        auto prior = nodeInput(ni,idx);
        if (prior.p != p)
          move(prior.p, p, a.shape, a.shape, range(a.size()), range(a.size()));
        prepared.insert({ni, idx});
      }
    }
  }
  Static nodeInput(I ni,I idx) {
    if(auto it=overlapExternalBindings.find({ni,idx});it!=overlapExternalBindings.end())
      return it->second;
    const auto& node=planner.nodes[ni];
    return input(node.inputs[idx],node.in[idx],node.name+".input"+std::to_string(idx));
  }
  void build() {
    if(std::getenv("FTLPU_GRAPH_PROGRESS"))llvm::errs()<<"graph: building input bindings\n";
    fs::create_directories(directory);
    reserveRecurrentStates();
    // The reserved state lifetime is already part of the whole-graph plan.
    // Upload a nonzero initial image directly there, without an archive copy.
    for (I ni=0;ni<I(planner.nodes.size());++ni) {
      const auto& node=planner.nodes[ni];
      if (!node.recurrent || !node.inputs[5].external()) continue;
      const auto& state=node.inputs[5];
      if (llvm::all_of(state.data,[](uint8_t x){return x==0;})) continue;
      const I index=inputs.size(); inputs.push_back(state);
      s.function.insertArgument(s.function.getNumArguments(),type(&s.context,state),mlir::DictionaryAttr{},s.loc);
      initialStateBindings[ni]=s.binding(state,recurrentStates.at(ni),"input",index,
          "initial_recurrent_state",s.function.getArgument(s.function.getNumArguments()-1));
    }
    Tensor mask;
    mask.shape = {4, 32};
    mask.dtype = "f32";
    mask.constant = 100000;
    mask.data.resize(4 * 32 * 4);
    for (I tile = 0; tile < 4; ++tile)
      for (I lane = 0; lane < 32; ++lane) {
        float x = lane / 8 == tile ? 1.f : 0.f;
        std::memcpy(mask.data.data() + (tile * 32 + lane) * 4, &x, 4);
      }
    const auto maskHome=s.placement("fp32_quad_planar", {48,49,50,51}, "east",1,0,4);
    try {
      s.maskPlacement=input(mask,maskHome,"layout_tile_masks",false,I(planner.nodes.size()),
          {s.placement("fp32_quad_planar",range(48),"east",1,0,1)}).p;
    } catch(const std::runtime_error &error) {
      if(std::string(error.what()).find("graph SRAM allocation failed") == std::string::npos)throw;
      // A large FFN can occupy all upper slices. Only the packer's ping-pong
      // queues (0..7) and staging queues (32..47) must be excluded; ordinary
      // activation queues remain legal when lifetime/row checks permit.
      V excluded=range(8);const auto staging=range(16,32);
      excluded.insert(excluded.end(),staging.begin(),staging.end());
      s.maskPlacement=input(mask,maskHome,"layout_tile_masks",false,I(planner.nodes.size()),
          {s.placement("fp32_quad_planar",excluded,"east",1,0,1)}).p;
    }
    Tensor zero;
    zero.shape = {1, 1};
    zero.dtype = "f32";
    zero.constant = 100001;
    zero.data.resize(4);
    auto zeroP = input(zero,
                       s.placement("fp32_scalar_broadcast_planar",
                                   {48, 49, 50, 51}, "east", 1, 0, 1),
                       "state_zero_word")
                     .p;
    for (I ni = 0; ni < I(planner.nodes.size()); ++ni) {
      auto &node = planner.nodes[ni];
      if (!node.module) {
        if (node.inputs[0].external()) {
          const auto& value=node.inputs[0];
          auto prototype=s.placement(value.dtype=="f32"?"fp32_quad_planar":"fp16_pair_planar",
              value.dtype=="f32"?V{44,45,46,47}:V{44,45},
              node.mirroredJoin?"both":node.headIngressArgument>=0 && !node.headIngressWest?"east":"west",1,0,
              value.shape[0]*((value.shape[1]+31)/32));
          if(!node.mirroredJoin && value.dtype=="bf16" && value.shape[1]%32==0) {
            const I words=value.size()/32, columns=value.shape[1]/32;
            // Use the existing two-pair C2C ABI. For a larger ingress image,
            // stripe feature words across both hemispheres as well, rather
            // than inventing an unsupported 8/16-slice planar binding.
            I pairs=words>s.target.memory().sram_depth_rows/2 && columns%2==0?2:1;
            I halves=pairs==2 && words/pairs>s.target.memory().sram_depth_rows/2 && columns%4==0?2:1;
            if(pairs>1)prototype=s.change(prototype,J{{"slices",s.array(range(2*pairs,32))},
                {"hemisphere",halves==2?"both":"west"},{"instruction_count",words/(pairs*halves)}});
          }
          auto busy=node.mirroredJoin?mirroredJoinQueues(ni):std::vector<mlir::DictionaryAttr>{};
          I rotaryFirst=planner.nodes.size(),rotaryLast=-1;
          if(rotaryTileOverlap())for(I ci=0;ci<I(planner.nodes.size());++ci) {
            const auto &consumer=planner.nodes[ci];
            if(!consumer.planarRotaryDim || !llvm::any_of(consumer.inputs[1].refs,
                [ni](const Ref &r){return r.node==ni;}))continue;
            rotaryFirst=std::min(rotaryFirst,ci);rotaryLast=std::max(rotaryLast,ci);
            // Earlier external producers already have C2C homes; later
            // retained producers will honor the queue reservation below.
            std::set<Key> producers;
            for(const auto &r:consumer.inputs[0].refs)producers.insert({r.node,r.output});
            for(const auto &key:producers)if(auto it=cache.find(key);it!=cache.end())
              if(!llvm::is_contained(busy,it->second))busy.push_back(it->second);
          }
          // Keep retained graph inputs off the native core's destination
          // queues. Row-disjoint storage on the same ICU would still require
          // a two-hop copy through scratch to serialize read and write.
          // Multi-row tensors retain C2C striping: forcing their native
          // queue exclusions can cost more DDR time than the copy saves.
          if(value.shape[0]==1) for (const auto &consumer : planner.nodes)
            for (I ai=0;ai<I(consumer.inputs.size()) && ai<I(consumer.in.size());++ai)
              if (consumer.in[ai].get("dense_attention_role") &&
                  llvm::any_of(consumer.inputs[ai].refs,[&](const Ref &ref){return ref.node==ni;}) &&
                  !llvm::is_contained(busy,consumer.in[ai])) {
                busy.push_back(consumer.in[ai]);
                if(auto segments=consumer.in[ai].getAs<mlir::ArrayAttr>("segments"))
                  for(auto attr:segments) {
                    auto segment=llvm::cast<mlir::DictionaryAttr>(attr);
                    busy.push_back(s.change(consumer.in[ai],J{{"slices",s.array(slices(segment))},
                        {"bank",field(segment,"bank")}}));
                  }
              }
          if(node.headIngressArgument>=0)for(auto& consumer:planner.nodes) {
            if(!consumer.module || consumer.inputs.empty() ||
                !llvm::any_of(consumer.inputs[0].refs,[&](Ref ref){return ref.node==ni;}))continue;
            consumer.function.walk([&](fc::tensor::L2NormTaskOp norm) {
              // Wide SXM simultaneously reads subsequent words and taps
              // earlier transposed data into feedback scratch on both sides.
              const auto scratch=norm.getScratchAllocations();
              if(!scratch.empty()) {
                auto replay=llvm::cast<mlir::DictionaryAttr>(scratch[0]).getAs<mlir::DictionaryAttr>("placement");
                if(replay)busy.push_back(s.change(replay,J{{"hemisphere","both"}}));
              }
            });
          }
          // A projected current row and its external history can be consumed
          // while the next projection reads X. SRAM rows are not independent
          // issue slots on the same MEM ICU. Prevent the history binding
          // allocator from reusing X's queues just because row ranges fit.
          for(I j=ni+1;j<I(planner.nodes.size());++j) {
            const auto& peer=planner.nodes[j];
            if(peer.projectionConvGroup<0 || peer.projectionConvGroup==j ||
               peer.inputs.empty() || !llvm::any_of(peer.inputs[0].refs,
                   [ni](const Ref& ref){return ref.node==ni;}))continue;
            const auto used=projectionSolveQueues(peer.projectionConvGroup);
            for(auto q:used)if(q && !llvm::is_contained(busy,q))busy.push_back(q);
          }
          const auto home=input(value,prototype,node.name,false,
              busy.empty() && !node.mirroredJoin && node.headIngressArgument<0 ? 0:ni,
              busy,false,node.mirroredJoin,node.headIngressArgument>=0);
          cache[{ni,0}]=home.p;
          if(rotaryLast>=0)
            exclusiveReadQueues.push_back({{-2,home.index},rotaryFirst,rotaryLast,home.p});
          if(node.headIngressArgument>=0)
            headIngressReport.push_back(J{{"source_argument",node.headIngressArgument},
               {"binding",home.index},{"offset_bytes",node.headIngressOffset},
               {"size_bytes",value.size()*value.bytes()},{"hemisphere",text(home.p,"hemisphere")}});
        }
        continue;
      }
      bool initialRmsNorm = false;
      if (ni == 0)
        node.function.walk([&](fc::tensor::RmsNormTaskOp) {
          initialRmsNorm = true;
        });
      for (I i = 0; i < I(node.inputs.size()); ++i)
        if (node.inputs[i].external() && !(node.recurrent && i == 5)) {
          if(planner.nodes.size()==1 && node.in[i].get("dense_attention_role")) {
            bindDenseIngress(node.inputs[i],node.in[i],node.name+".input"+std::to_string(i));
            overlapExternalBindings[{ni,i}]={node.in[i],-1,{}};
            continue;
          }
          std::vector<mlir::DictionaryAttr> concurrentInputs;
          if(node.projectionSolvePeer>=0)
            concurrentInputs=projectionSolveQueues(node.projectionSolvePeer);
          const bool convTaps=node.projectionConvGroup>=0 && ni!=node.projectionConvGroup &&
              text(node.in[i],"kind")=="fp16_depthwise_kernel4";
          if(convTaps) {
            reserveProjectionConvQueues(ni,concurrentInputs);
            concurrentInputs.insert(concurrentInputs.end(),node.workspaces.begin(),node.workspaces.end());
            concurrentInputs.push_back(node.in[0]);
            // The native window reads history/projected words at their
            // producer homes rather than node.in[0]. Coefficients arrive
            // on distinct SRs on those very cycles, so they also require
            // independent MEM issue queues.
            // One coefficient upload is shared by the independent requests
            // in this wave. Reserve against *all* their history homes before
            // selecting that upload's queues; a later request may use a
            // different slice pair even though its coefficient bytes match.
            const I leader=node.projectionConvGroup;
            for(I peer=leader+1;peer<I(planner.nodes.size()) &&
                planner.nodes[peer].projectionConvGroup==leader;++peer) {
              const auto& member=planner.nodes[peer];
              if(member.in.empty() || member.function->hasAttr("ftlpu.depthwise_conv_mirror")!=
                  node.function->hasAttr("ftlpu.depthwise_conv_mirror"))continue;
              Key prev{-1,-1};
              for(const auto& ref:member.inputs[0].refs) {
                Key key{ref.node,ref.output};
                if(key==prev)continue;
                prev=key;
                auto home=cache.find(key);
                if(home!=cache.end() && !llvm::is_contained(concurrentInputs,home->second))
                  concurrentInputs.push_back(home->second);
              }
            }
          }
          const auto fp32InputPolicy=(*node.module)->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_fp32_ingress");
          bool binaryFp32=false;
          if(fp32InputPolicy && fp32InputPolicy.getValue())
            node.function.walk([&](fc::kernel::VectorAluOp op) {
              binaryFp32 |= !op.getUseImmediate() && op.getLhs()!=op.getRhs()
                  && (op.getKind()=="add" || op.getKind()=="subtract"
                      || op.getKind()=="multiply" || op.getKind()=="max" || op.getKind()=="head_last_row_scale_transpose" || op.getKind()=="head_scale" || (op.getKind()=="unit_lower_solve" || op.getKind()=="unit_lower_solve_blocked"));
            });
          // Concurrent fixed-port operands need independent MEM ICU queues,
          // even when the C2C allocator could place them in disjoint rows.
          if(binaryFp32 || !node.function.getOps<fc::kernel::ScaledMatvecOp>().empty())
            for(I earlier=0;earlier<i;++earlier)
              concurrentInputs.push_back(externalVectorHomes.count({ni,earlier})
                  ? externalVectorHomes.at({ni,earlier}) : node.in[earlier]);
          bool planarGamma=false;
          if(ni>0 && i==1 && node.inputSlots[i].size()==1)
            if(auto norm=llvm::dyn_cast<fc::tensor::RmsNormTaskOp>(node.inputSlots[i][0].op))
              if(auto flag=norm.getConfig().getAs<mlir::BoolAttr>("planar_streaming"))
                planarGamma=flag.getValue() && slices(node.in[i]).size()==2;
          if(planarGamma) {
            concurrentInputs.push_back(node.in[0]);
            concurrentInputs.insert(concurrentInputs.end(),node.workspaces.begin(),node.workspaces.end());
            concurrentInputs.insert(concurrentInputs.end(),node.out.begin(),node.out.end());
          }
          auto prior=input(node.inputs[i], node.in[i],
                node.name + ".input" + std::to_string(i),
                (initialRmsNorm &&
                    (text(node.in[i], "kind") == "fp16_mxm_distributed_16" ||
                     text(node.in[i], "kind") == "fp16_pair_planar" ||
                     text(node.in[i], "kind") == "fp16_vxm_gamma_broadcast"))
                    || (ni == 0
                        && (text(node.in[i], "kind") == "fp16_mxm_distributed_16"
                            || (node.ffn && i == 0 &&
                                text(node.in[i], "kind") == "fp16_mxm_activation_planar"))),
                node.ffn ? I(planner.nodes.size()) : ni,
                concurrentInputs,node.projectionSolvePeer>=0,false,false,planarGamma);
          if(planarGamma) {
            externalVectorHomes[{ni,i}]=prior.p;
            overlapExternalBindings[{ni,i}]=prior;
          }
          if(convTaps) {
            externalVectorHomes[{ni,i}]=prior.p;
            overlapExternalBindings[{ni,i}]=prior;
          }
          if(node.projectionSolvePeer>=0) {
            overlapExternalBindings[{ni,i}]=prior;
            // Resolve configurable native inputs before emitting their read
            // domains. Late bank relocation would invalidate queue isolation.
            const auto kind=text(prior.p,"kind");
            if(kind==text(node.in[i],"kind") &&
                text(prior.p,"hemisphere")==text(node.in[i],"hemisphere") &&
                (kind=="token_partition_planar" || kind=="token_partition_east_first_planar"))
              externalVectorHomes[{ni,i}]=prior.p;
          }
          // Packed VXM inputs have configurable slice/base addresses. Read
          // the lifetime-safe C2C home directly instead of copying the whole
          // tensor back into the emitter's provisional input slots.
          if (!node.inputSlots[i].empty() &&
              llvm::all_of(node.inputSlots[i],[&](const auto &slot) {
                if(llvm::isa<fc::kernel::VectorAluOp,fc::kernel::ScaledMatvecOp>(slot.op)) return true;
                auto projection=llvm::dyn_cast<fc::tensor::ProjectionTaskOp>(slot.op);
                return i==0 && projection && projection.getKind()=="linear";
              }) && text(prior.p,"kind")==text(node.in[i],"kind") &&
              text(prior.p,"hemisphere")==text(node.in[i],"hemisphere") &&
              (text(prior.p,"kind")=="fp32_quad_planar" ||
               text(prior.p,"kind")=="fp16_pair_planar"))
            externalVectorHomes[{ni,i}]=prior.p;
          // Linear projection's activation read domains accept arbitrary
          // slices/base/bank. All pages can use the immutable C2C home; a
          // provisional bank-0 address is not an execution requirement.
          if(i==0 && !node.inputSlots[i].empty() &&
              llvm::all_of(node.inputSlots[i],[](const auto &slot) {
                auto projection=llvm::dyn_cast<fc::tensor::ProjectionTaskOp>(slot.op);
                return projection && projection.getKind()=="linear";
              }) && text(prior.p,"kind")=="fp16_pair_planar" && slices(prior.p).size()==2
              && text(prior.p,"hemisphere")=="both" && !prior.p.get("storage_shape"))
            externalVectorHomes[{ni,i}]=prior.p;
        }
    }
    if(std::getenv("FTLPU_GRAPH_PROGRESS"))llvm::errs()<<"graph: planning retained memory\n";
    planMemory();
    checkProjectionSolveQueues();
    checkProjectionConvQueues();
    std::set<Key> prepared;
    // Retain immutable broadcast weights across consecutive native norm
    // tiles. Verify the emitted writes before reuse; equal constants alone
    // do not prove that their temporary execution home is still resident.
    struct ResidentNormWeight {
      I binding;
      mlir::DictionaryAttr home;
      mlir::Operation *watermark;
    };
    std::optional<ResidentNormWeight> residentNormWeight;
    struct StagedProjectionInput {
      I page;
      mlir::DictionaryAttr home;
      mlir::Operation *watermark;
    };
    std::optional<StagedProjectionInput> stagedProjectionInput;
    // The page DMA writes are linked later. Exclude their queues as well as
    // checking writes already present in the compute schedule below.
    std::set<I> pagedSlices = weightSlotSlices;
    for (const auto &entry : statics) {
      auto home = entry.second.p;
      if (!field(home, "page_rows"))
        continue;
      for (I slice : slices(home))
        pagedSlices.insert(slice);
      if (auto storage = home.getAs<mlir::ArrayAttr>("page_storage_slices"))
        for (auto slice : storage)
          pagedSlices.insert(llvm::cast<mlir::IntegerAttr>(slice).getInt());
    }
    I normSafeStart=-1, normNextCursor=-1;
    I parallelStart = 0, parallelEnd = 0;
    I projectionSolveStart=0,projectionSolveEnd=0;
    I projectionConvEnd=0;
    std::vector<I> parallelNodes;
    for (I ni = 0; ni < I(planner.nodes.size()); ++ni) {
      auto &node = planner.nodes[ni];
      bool nativeRmsNorm=false, reusedNormWeight=false;
      auto *vectorWatermark=&s.function.front().back();
      bool reusedProjectionInput = false;
      if(node.module && node.parallel<0)
        node.function.walk([&](fc::schedule::TimelineOp timeline) {
          nativeRmsNorm |= timeline.getName()=="rmsnorm.direct_planar_ingress";
        });
      if(!nativeRmsNorm)residentNormWeight.reset();
      llvm::outs() << "plan " << ni << " " << node.name << "\n";
      s.transportPhase=node.name;
      s.transportReason="consumer_layout_conversion";
      s.consumerMulticast=nativeRmsNorm;
      if(node.projectionSolvePeer>ni) {
        prepare(ni,prepared);
        prepare(node.projectionSolvePeer,prepared);
      }
      if(node.projectionConvReady && !node.projectionVectorMember && node.projectionConvGroup==ni) {
        for(I peer=ni;peer<I(planner.nodes.size()) &&
            planner.nodes[peer].projectionConvGroup==ni;++peer)
          prepare(peer,prepared);
      }
      for(I operand=0;operand<I(node.inputs.size());++operand) {
        auto found=wideBroadcastOperands.find({ni,operand});
        if(found==wideBroadcastOperands.end() || emittedWideBroadcasts.count(found->second)) continue;
        const auto key=found->second;
        const Key sourceKey{std::get<0>(key),std::get<1>(key)};
        const I sourceRow=std::get<2>(key);
        const auto src=cache.at(sourceKey),dst=wideBroadcastHomes.at(key);
        const I heads=planner.nodes[sourceKey.first].outputs[sourceKey.second].shape.back();
        const auto ss=slices(src),ds=slices(dst);
        const I begin=s.cursor;
        for(I pair=0;pair<2;++pair) {
          const I capture=s.cursor+32;
          I end=capture;
          for(I side=0;side<2;++side) {
            auto from=s.change(src,J{{"hemisphere",side?"west":"east"}});
            auto to=s.change(dst,J{{"hemisphere",side?"west":"east"}});
            s.emit("sxm",J{{"cycle",capture},{"hemisphere",side},{"opcode","transpose"},
                {"source_streams",s.array({16,17})},{"destination_streams",s.array({32+side*16,33+side*16})},
                {"permute_map",s.array(range(32))},{"weight_layout","wide_matrix"},
                {"repeat_count",1},{"repeat_interval",32},{"wave_count",1},{"wave_interval",32}});
            for(I byte=0;byte<2;++byte) {
              I plane=pair*2+byte;
              const I readLatency=*s.target.transport_latency(fc::target::StreamEndpoint::Mem,
                  fc::target::StreamEndpoint::SxmInput,fc::target::StreamDirection::East,ss[plane]);
              const I writeLatency=*s.target.transport_latency(fc::target::StreamEndpoint::SxmResult,
                  fc::target::StreamEndpoint::Mem,fc::target::StreamDirection::West,ds[plane]);
              s.mem(capture-readLatency,from,plane,"read",field(src,"base_row")+sourceRow,16+byte,{32,1,1},{0,0,0},{1,32,32});
              s.mem(capture+35+writeLatency,to,plane,"write",field(dst,"base_row"),32+side*16+byte,
                  {heads,1,1},{1,0,0},{1,heads,heads});
              end=std::max(end,capture+35+writeLatency+32);
            }
          }
          s.cursor=end+32;
        }
        s.emit("timeline",J{{"name","wide_scalar_broadcast"},{"start",begin},{"end",s.cursor}});
        emittedWideBroadcasts.insert(key);
      }
      llvm::outs().flush();
      if (!node.module) {
        // Balance packed materializations between the two SXM fabrics. A
        // fixed West home concentrates every later unpack on one iMEM.
        if (node.planarRotaryDim) {
          emitPlanarRotary(ni);
          for (auto it=live.begin();it!=live.end();)
            if(last[it->first]==ni)it=live.erase(it);else ++it;
          continue;
        }
        auto p = cache.at({ni, 0});
        J joinInfo{{"node",ni},{"name",node.name},{"layout_join",true}};
        joinInfo["access_decision"]=node.viewReason;
        s.transportReason=node.viewReason.empty()?"consumer_layout_conversion":node.viewReason;
        if(node.addressOnlyView) {
          joinInfo["materialized"]=false;
        } else if(node.materializedView) {
          auto attrs=mlir::NamedAttrList(p);attrs.erase("planar_read");
          materialize(*node.materializedView,attrs.getDictionary(&s.context));
          joinInfo["logical_rows"]=node.outputs[0].shape[0];
          joinInfo["stored_rows"]=node.materializedView->shape[0];
        } else
        if(node.streamedTransposeJoin) {
          joinInfo["vxm_sxm_direct_transpose"]=true;
          joinInfo["materialized_before_transpose"]=false;
        } else if(auto heads=p.getAs<mlir::ArrayAttr>("mxm_lhs_head_rows")) {
          joinInfo["native_mxm_lhs_view"]=true;
          joinInfo["source_slices"]=s.array(slices(p));
          joinInfo["source_bank"]=field(p,"bank");
          V rows;for(auto head:heads)rows.push_back(llvm::cast<mlir::IntegerAttr>(head).getInt());
          joinInfo["head_base_rows"]=s.array(rows);
        } else if (!node.inputs[0].external()) materialize(node.inputs[0], p);
        nodeRanges.push_back(std::move(joinInfo));
        for (auto it = live.begin(); it != live.end();)
          if (last[it->first] == ni)
            it = live.erase(it);
          else
            ++it;
        continue;
      }
      if (node.parallel >= 0 &&
          (ni == 0 || planner.nodes[ni - 1].parallel != node.parallel)) {
        for (I pi = ni; pi < I(planner.nodes.size()) &&
                        planner.nodes[pi].parallel == node.parallel;
             ++pi) {
          if(!node.projectionConvReady)prepare(pi, prepared);
        }
        parallelStart = s.cursor + 64;
        parallelEnd = parallelStart;
        parallelNodes.clear();
      }
      mlir::IRMapping mapping;
      std::map<I, std::pair<mlir::DictionaryAttr, Static>> relocated;
      for (I idx = 0; idx < I(node.inputs.size()); ++idx) {
        s.transportReason="consumer_layout_conversion";
        for(const auto &item:node.accessPlans)if(auto entry=item.getAsObject())
          if(entry->getInteger("input")==idx)
            if(auto reason=entry->getString("reason"))s.transportReason=reason->str();
        auto &a = node.inputs[idx];
        auto p = node.in[idx];
        bool movable = node.stagedInputs[idx] || p.get("dense_attention_role") || scalar(p) || text(p, "kind") == "fp16_pair_planar" ||
                       text(p, "kind") == "fp16_mxm_activation_planar" ||
                       text(p, "kind") == "fp32_quad_planar" ||
                       text(p, "kind") == "fp16_mxm_distributed_16";
        if (a.external() && node.recurrent && idx == 5) {
          if (initialStateBindings.count(ni)) mapping.map(node.function.getArgument(idx),initialStateBindings.at(ni));
          else {
            s.copy(zeroP, p, V(field(p, "instruction_count"), 0),range(field(p, "instruction_count")));
            mapping.map(node.function.getArgument(idx),s.binding(a, p, "internal", s.internal++,"initial_recurrent_state"));
          }
        } else if (a.external() && text(p,"kind") == "fp16_projection_bias_x4") {
          // Preserve the native bias read bank/queues. Relocating those reads
          // after scheduling can collide with the raw-Q FIFO on the other
          // bank; copy the immutable packed planes into their native home
          // before this attention batch starts instead.
          const auto prior=nodeInput(ni,idx);
          if (prior.p != p) {
            const auto sourceSlices=slices(prior.p), destSlices=slices(p);
            for (I pair=0;pair<I(destSlices.size())/2;++pair) {
              auto src=s.change(prior.p,J{{"kind","fp16_pair_planar"},
                  {"slices",s.array({sourceSlices[2*pair],sourceSlices[2*pair+1]})}});
              auto dst=s.change(p,J{{"kind","fp16_pair_planar"},
                  {"slices",s.array({destSlices[2*pair],destSlices[2*pair+1]})}});
              s.copy(src,dst,range(field(p,"instruction_count")),range(field(p,"instruction_count")));
            }
          }
          mapping.map(node.function.getArgument(idx),s.binding(a,p,"internal",s.internal++,node.name+".bias"));
        } else if (!a.external() || movable) {
          if (!prepared.count({ni, idx})) {
            if (!a.external()) {
              if (!node.stagedInputs[idx]) {
                const bool projectionInput = idx == 0 && node.projectionPageFamily >= 0 &&
                    node.parallel < 0 && node.projectionSolvePeer < 0 && !node.projectionConvReady &&
                    text(p, "kind") == "fp16_pair_planar" && !field(p, "bank_tile_rows") &&
                    llvm::none_of(slices(p), [&](I slice) { return pagedSlices.count(slice); });
                if (projectionInput && stagedProjectionInput && stagedProjectionInput->page + 1 == ni &&
                    planner.nodes[ni - 1].projectionPageFamily == node.projectionPageFamily &&
                    stagedProjectionInput->home == p) {
                  // Sibling pages are constructed from the same immutable SSA
                  // input view. Keep its first layout conversion while the
                  // staging queues remain untouched by any writer.
                  const auto side = text(p, "hemisphere");
                  const auto planes = slices(p);
                  reusedProjectionInput = true;
                  for (auto *op = stagedProjectionInput->watermark->getNextNode(); op; op = op->getNextNode()) {
                    if (auto mem = llvm::dyn_cast<fc::schedule::MemTransferOp>(op)) {
                      if (mem.getOpcode() == "read" || integer(op, "bank") != field(p, "bank") ||
                          !contains(planes, mem.getSlice()) ||
                          (side != "both" && side != (mem.getHemisphere() ? "west" : "east")))
                        continue;
                    } else if (!llvm::isa<fc::schedule::MemWriteRead2DOp, fc::schedule::MemWriteOp>(op)) {
                      continue;
                    }
                    reusedProjectionInput = false;
                    break;
                  }
                }
                if (!projectionInput || !reusedProjectionInput)
                  materialize(a, p, projectionInput && node.parallel < 0 &&
                      node.projectionSolvePeer < 0 && !node.projectionConvReady &&
                      node.function->hasAttr("ftlpu.imem_drain_after"));
                if (projectionInput)
                  stagedProjectionInput = StagedProjectionInput{ni, p, &s.function.front().back()};
              }
            }
            else if (!node.stagedInputs[idx]) {
              auto prior = nodeInput(ni,idx);
              bool reuse=false;
              const bool normWeight=nativeRmsNorm && text(p,"kind")=="fp16_vxm_gamma_broadcast";
              if(normWeight && residentNormWeight && residentNormWeight->binding==prior.index
                  && residentNormWeight->home==p) {
                reuse=true;
                for(auto *op=residentNormWeight->watermark->getNextNode();op;op=op->getNextNode())
                  if(auto mem=llvm::dyn_cast<fc::schedule::MemTransferOp>(op)) {
                    const auto side=mem.getHemisphere()?"west":"east";
                    if(mem.getOpcode().starts_with("write") && integer(op,"bank")==field(p,"bank")
                        && contains(slices(p),mem.getSlice())
                        && (text(p,"hemisphere")=="both" || text(p,"hemisphere")==side))reuse=false;
                  }
              }
              if (prior.p != p && !reuse)
                move(prior.p, p, a.shape, a.shape, range(a.size()),
                     range(a.size()));
              if(normWeight) {
                residentNormWeight=ResidentNormWeight{prior.index,p,&s.function.front().back()};
                reusedNormWeight=reuse;
              }
            }
          }
          mapping.map(node.function.getArgument(idx),
                      s.binding(a, p, "internal", s.internal++,
                                node.name + ".operand"));
        } else {
          auto prior = nodeInput(ni,idx);
          relocated[idx] = {p, prior};
          mapping.map(node.function.getArgument(idx), prior.binding);
        }
      }
      bool residentAttention = false;
      node.function.walk([&](fc::schedule::TimelineOp timeline) {
        residentAttention |= timeline.getName() == "qkv";
      });
      for (auto &kv : relocated)
        if (auto pingpong = kv.second.first.getAs<mlir::BoolAttr>("page_bank_ping_pong");
            pingpong && pingpong.getValue()) residentAttention = false;
      I pageRows = 0, physicalIngress = 0;
      for (auto &kv : relocated) {
        pageRows = std::max(pageRows, field(kv.second.second.p,"page_rows"));
        if (residentAttention && field(kv.second.second.p,"page_rows")) {
          const I sideBytes = (node.inputs[kv.first].size()*node.inputs[kv.first].bytes()+1)/2;
          const I bandwidth = s.target.streams().c2c_streams_per_direction *
              s.target.streams().c2c_bytes_per_stream_per_cycle;
          physicalIngress += (sideBytes+bandwidth-1)/bandwidth+64;
        }
      }
      pageRows = std::max(pageRows,physicalIngress);
      I ingressBegin = s.cursor;
      // FFN page readiness is a hardware event. Its ingress window describes
      // resource availability, not a transfer-duration delay on every ICU.
      // The consumer waits on real page commits. Ping-pong must not also
      // insert an estimated page-sized NOP window after the previous kernel.
      if (pageRows && !node.ffn && !weightPingPong) s.cursor += pageRows+32;
      if (pageRows && weightPingPong) {
        const auto& h=s.target.throughput();
        const I drain=s.target.streams().system_register_columns
            +s.target.streams().mem_boundary_register_columns+h.mxm_pipeline_rows+h.tile_rows
            +std::max({h.mem_to_sxm_latency,h.mem_to_mxm_latency,h.mxm0_accumulator_latency,
                h.mxm1_accumulator_latency,h.accumulator_to_vxm_latency,
                h.accumulator_read_to_vxm_latency,h.swiglu_write_latency,
                h.qk_iw_to_compute_latency,h.vxm_weight_to_iw_latency});
        s.cursor+=std::max<I>(0,drain-64);
      }
      // The native emitter's end already includes SR drain. Consecutive
      // norm tiles with no new ingress can start there, without the generic
      // linker gap (96) and launch allowance (64).
      const bool tightNorm=nativeRmsNorm && reusedNormWeight && normSafeStart>=0
          && s.cursor==normNextCursor && !pageRows;
      // These serial kernels publish their terminal MEM writes and native
      // drain timeline. Anchor the earliest native event at the preceding
      // drained cursor, rather than adding both generic launch allowances.
      const bool tightSerial=node.function->hasAttr("ftlpu.tight_serial_schedule") &&
          node.parallel<0 && node.projectionSolvePeer<0 && !pageRows;
      I nativeOrigin=std::numeric_limits<I>::max();
      if(tightSerial)for(auto &op:node.function.front()) {
        if(op.hasAttr("cycle"))nativeOrigin=std::min(nativeOrigin,integer(&op,"cycle"));
        if(op.hasAttr("start") && op.hasAttr("end"))nativeOrigin=std::min(nativeOrigin,integer(&op,"start"));
      }
      if(nativeOrigin==std::numeric_limits<I>::max())nativeOrigin=0;
      I start = node.parallel >= 0 ? parallelStart : tightNorm?normSafeStart:
          tightSerial?s.cursor-nativeOrigin:s.cursor+64, end = start;
      if(node.projectionSolvePeer>=0) {
        if(node.projectionSolvePeer>ni) {
          projectionSolveStart=start;projectionSolveEnd=start;
        } else start=end=projectionSolveStart;
      }
      I nativeFirst = std::numeric_limits<I>::max(), nativeLast = 0;
      // This conservative graph schedule finishes the preceding operator
      // (including its SR drain) before admitting the next page. The page
      // rendezvous blocks every subsequent FU queue if DDR is late. Advertise
      // this exclusive ingress window, rather than borrowing an SR until its
      // final use in the entire layer, which can be millions of cycles later.
      weightIngressWindows[ni] = {ingressBegin, start};
      for (auto &kv : relocated)
        if (field(kv.second.second.p, "page_rows")) {
          I windowBegin = ingressBegin, windowEnd = start;
          const I head = integer(node.function, "ftlpu.weight_rendezvous_head", ni);
          const auto placement = kv.second.second.p;
          if (weightPingPong && head >= 0 && head < ni && weightIngressWindows.count(head)
              && field(placement, "page_release_before_node", ni + 1) <= head) {
            const auto& first = planner.nodes.at(head);
            const bool separateBank = llvm::any_of(first.in, [&](auto other) {
              return text(other, "kind") == "w8a16_mxm_weight_striped"
                  && field(other, "bank") != field(placement, "bank");
            });
            if (separateBank) {
              windowBegin = weightIngressWindows.at(head).first;
              windowEnd = weightIngressWindows.at(head).second;
            }
          }
          s.emit("timeline",
                 J{{"name", "c2c.window.binding:" +
                                std::to_string(kv.second.second.index) +
                                ".page:0"},
                   {"start", windowBegin},
                   {"end", windowEnd}});
        }
      auto *vectorPreparationEnd=&s.function.front().back();
      for (auto &op : node.function.front()) {
        if (llvm::isa<mlir::func::ReturnOp>(op))
          continue;
        if (auto bind = llvm::dyn_cast<fc::schedule::BindingOp>(op)) {
          if (bind.getAccess() == "input") {
            auto value = mapping.lookup(node.function.getArgument(bind.getIndex()));
            if (auto destination = value.getDefiningOp<fc::schedule::BindingOp>())
              for (auto name : {"role", "initializer_config"})
                if (auto attr = bind->getAttr(name)) destination->setAttr(name, attr);
            // Scalar constants may have been staged through an internal binding.
            // Packing metadata belongs on the external byte upload as well.
            const auto &operand = node.inputs.at(bind.getIndex());
            if (operand.external())
              for (auto &entry : statics) {
                auto &constant = inputs.at(entry.second.index);
                if (constant.constant != operand.constant || constant.shape != operand.shape ||
                    constant.data != operand.data) continue;
                auto destination = entry.second.binding.getDefiningOp<fc::schedule::BindingOp>();
                for (auto name : {"role", "initializer_config"})
                  if (auto attr = bind->getAttr(name)) destination->setAttr(name, attr);
              }
            mapping.map(
                bind.getResult(),
                mapping.lookup(node.function.getArgument(bind.getIndex())));
            continue;
          }
        }
        auto *copy = s.b.clone(op, mapping);
        copy->removeAttr("scale_binding");
        copy->removeAttr("address_binding");
        copy->removeAttr("address_binding_access");
        copy->removeAttr("ftlpu.schedule_stage");
        bool nativePage = false;
        if (auto index = op.getAttrOfType<mlir::IntegerAttr>("address_binding")) {
          auto it = relocated.find(index.getInt());
          if (it != relocated.end() && it->second.first.getAs<mlir::BoolAttr>("paged_weight")) {
            s.set(copy,"address_binding",it->second.second.index);
            s.set(copy,"weight_page",integer(&op,"weight_page",0));
            copy->setAttr("runtime_prefetch",s.b.getBoolAttr(true));
            if (residentAttention)
              s.set(copy,"weight_ready_cycle",start);
            nativePage = true;
          }
        }
        if (auto bind = llvm::dyn_cast<fc::schedule::BindingOp>(copy)) {
          copy->setAttr("access", s.b.getStringAttr("internal"));
          s.set(copy, "index", s.internal++);
        }
        if (auto mem = llvm::dyn_cast<fc::schedule::MemTransferOp>(copy))
          if (mem.getOpcode() == "read" && !nativePage)
            for (auto &kv : relocated) {
              auto old = kv.second.first;
              auto now = kv.second.second;
              if (old.getAs<mlir::BoolAttr>("paged_weight")) continue;
              const I begin = field(old,"base_row"), finish = begin+field(old,"instruction_count");
              if (mem.getAddress() < begin || mem.getAddress() >= finish) continue;
              auto h = mem.getHemisphere() ? "west" : "east";
              if ((text(old, "hemisphere") != h &&
                   text(old, "hemisphere") != "both") ||
                  !contains(slices(old), mem.getSlice()) ||
                  integer(copy, "bank") != field(old, "bank"))
                continue;
              s.set(copy, "bank", field(now.p, "bank"));
              s.set(copy, "address",
                    mem.getAddress() + field(now.p, "base_row") -
                        field(old, "base_row"));
              if (now.p.getAs<mlir::BoolAttr>("paged_weight")) {
                s.set(copy, "address_binding", now.index);
                s.set(copy, "weight_page", 0);
                copy->setAttr("runtime_prefetch", s.b.getBoolAttr(true));
              }
            }
        if (copy->hasAttr("cycle")) {
          s.set(copy, "cycle", integer(copy, "cycle") + start);
          I span = 0;
          for (auto axis : {std::pair{"repeat_count", "repeat_interval"},
                            std::pair{"wave_count", "wave_interval"},
                            std::pair{"group_count", "group_interval"}})
            span += (integer(copy, axis.first, 1) - 1) *
                    integer(copy, axis.second, 1);
          end = std::max(end, integer(copy, "cycle") + span + 32);
          nativeFirst = std::min(nativeFirst, integer(copy, "cycle"));
          nativeLast = std::max(nativeLast, integer(copy, "cycle") + span + 1);
        }
        if (copy->hasAttr("start") && copy->hasAttr("end")) {
          s.set(copy, "start", integer(copy, "start") + start);
          s.set(copy, "end", integer(copy, "end") + start);
          end = std::max(end, integer(copy, "end"));
          if (auto n = copy->getAttrOfType<mlir::StringAttr>("name")) {
            auto name = n.getValue();
            constexpr llvm::StringLiteral prefix("c2c.window.binding:");
            if (name.starts_with(prefix)) {
              auto suffix = name.drop_front(prefix.size());
              auto parts = suffix.split(".page:");
              int64_t localBinding = -1;
              if (parts.first.getAsInteger(10, localBinding)
                  || parts.second.empty() || !relocated.count(localBinding))
                throw std::runtime_error("cannot relocate native C2C window binding");
              copy->setAttr("name", s.b.getStringAttr(prefix.str()
                  + std::to_string(relocated.at(localBinding).second.index)
                  + ".page:" + parts.second.str()));
            } else if (name == "common_barrier.drain_cost") {
              // Protocol timelines retain their reserved names after linking.
            } else {
              copy->setAttr("name", s.b.getStringAttr(node.name + "." + name));
            }
          }
        }
        if (copy->hasAttr("ready_cycle"))
          s.set(copy, "ready_cycle", integer(copy, "ready_cycle") + start);
      }
      llvm::json::Array layouts;
      for (auto p : node.out)
        layouts.push_back(J{{"kind", text(p, "kind")},{"bank_tile_rows",field(p,"bank_tile_rows")},
                            {"hemisphere", text(p, "hemisphere")},
                            {"slices", s.array(slices(p))},
                            {"bank", field(p, "bank")},
                            {"base_row", field(p, "base_row")},
                            {"rows", field(p, "instruction_count")}});
      llvm::json::Array pageWorkspaces;
      llvm::json::Array temporaryWorkspaces;
      llvm::json::Array retainedLayouts;
      for (I oi=0; oi<I(node.outputs.size()); ++oi)
        if (auto retained = retainedOutputHomes.find({ni,oi}); retained != retainedOutputHomes.end()) {
          const auto p = retained->second;
          retainedLayouts.push_back(J{{"output",oi},{"kind",text(p,"kind")},
              {"hemisphere",text(p,"hemisphere")},{"slices",s.array(slices(p))},
              {"bank",field(p,"bank")},{"base_row",field(p,"base_row")},
              {"rows",field(p,"instruction_count")}});
        }
      for (auto p : node.workspaces)
        if (auto paged = p.getAs<mlir::BoolAttr>("paged_weight"); paged && paged.getValue())
          pageWorkspaces.push_back(J{{"hemisphere",text(p,"hemisphere")},
              {"slices",s.array(slices(p))},{"bank",field(p,"bank")},
              {"base_row",field(p,"base_row")},{"rows",field(p,"instruction_count")}});
        else if (text(p,"kind") == "bf16_swiglu_temp_byte")
          temporaryWorkspaces.push_back(J{{"hemisphere",text(p,"hemisphere")},
              {"slices",s.array(slices(p))},{"bank",field(p,"bank")},
              {"base_row",field(p,"base_row")},{"rows",field(p,"instruction_count")}});
      nodeRanges.push_back(J{{"node", ni},
                             {"access_plans", std::move(node.accessPlans)},
                             {"native_projections", std::move(node.nativeProjectionReport)},
                             {"name", node.name},
                             {"attention_core_schedule",node.function->hasAttr("ftlpu.attention_core_schedule")
                                 ? node.function->getAttrOfType<mlir::StringAttr>("ftlpu.attention_core_schedule").getValue().str() : ""},
                             {"projection_solve_peer",node.projectionSolvePeer},
                             {"projection_conv_group",node.projectionConvGroup},
                             {"projection_conv_enabled",node.projectionConvReady},
                             {"projection_vector_group",node.projectionVectorMember?node.projectionConvGroup:-1},
                             {"projection_vector_enabled",node.projectionVectorMember && node.projectionConvReady},
                             {"norm_weight_reused",reusedNormWeight},
                             {"projection_input_reused",reusedProjectionInput},
                             {"norm_tight_start",tightNorm},
                             {"tight_serial_schedule",tightSerial},
                             {"native_ffn",node.ffn},
                             {"stateful_mixer",node.recurrent},
                             {"weight_page_workspaces",std::move(pageWorkspaces)},
                             {"temporary_workspaces",std::move(temporaryWorkspaces)},
                             {"retained_output_layouts",std::move(retainedLayouts)},
                             {"nominal_start", nativeFirst},
                             {"nominal_end", nativeLast},
                             {"direct_output_layouts", std::move(layouts)}});
      verifyProjectionVectorEmission(ni,vectorWatermark,vectorPreparationEnd);
      if(node.projectionSolvePeer>=0) {
        projectionSolveEnd=std::max(projectionSolveEnd,end);
        s.cursor=projectionSolveEnd+96;
      } else if (node.parallel >= 0) {
        parallelEnd = std::max(parallelEnd, end);
        parallelNodes.push_back(ni);
        s.cursor = parallelEnd + 96;
      } else
        s.cursor = end + (tightSerial?0:96);
      if(node.projectionConvReady) {
        if(node.projectionConvGroup==ni) {
          // The concurrent conv groups use their existing bilateral scheduler.
          // Hand it the same launch cycle, then advance between conv groups
          // on their own cursor while the continuous projection issues.
          projectionConvEnd=end;
          // Vector operands may need new MEM/SXM transfers. They can start
          // only after the projection's first native issue, which is the
          // compiler-generated paging consumer rendezvous. Launching their
          // preparation at the kernel's earlier base would occupy its drain.
          s.cursor=node.projectionVectorMember?nativeFirst:start-64;
        } else if(ni+1==I(planner.nodes.size()) ||
                  planner.nodes[ni+1].projectionConvGroup!=node.projectionConvGroup) {
          s.cursor=std::max(s.cursor,projectionConvEnd+96);
        }
      }
      for (I oi=0; oi<I(node.outputs.size()); ++oi)
        if (auto home = retainedOutputHomes.find({ni,oi}); home != retainedOutputHomes.end()) {
          const I begin = s.cursor;
          const auto &shape = node.outputs[oi].shape;
          move(node.out[oi], home->second, shape, shape,
               range(node.outputs[oi].size()), range(node.outputs[oi].size()));
          s.emit("timeline", J{{"name", node.name + ".retain_on_chip"},
                               {"start", begin}, {"end", s.cursor}});
        }
      if(node.function->hasAttr("ftlpu.imem_drain_after")) {
        if(node.parallel>=0)throw std::runtime_error("iMEM drain cannot split parallel nodes");
        s.emit("timeline",J{{"name","imem.drain."+node.name},
            {"start",s.cursor-1},{"end",s.cursor}});
        // A tight serial emitter already publishes a fully drained end.
        // Keep the reload boundary without adding another launch allowance.
        if(!tightSerial)s.cursor+=96;
      }
      normSafeStart=nativeRmsNorm && s.cursor==end+96?end:-1;
      normNextCursor=s.cursor;
      for (auto it = live.begin(); it != live.end();)
        if (last[it->first] == ni)
          it = live.erase(it);
        else
          ++it;
    }
    s.consumerMulticast=false;
    struct OutputHome {
      mlir::DictionaryAttr placement;
      V shape;
      bool gather;
    };
    bool usedTxHome=false;
    auto selectOutputHome=[&](I i,bool preferTx)->OutputHome {
      auto &value = finals[i];
      bool identity = !value.refs.empty();
      auto first = value.refs[0];
      for (I j = 0; j < I(value.refs.size()); ++j)
        identity &= value.refs[j].node == first.node &&
                    value.refs[j].output == first.output &&
                    value.refs[j].index == j;
      // Consecutive N pages already form one native hemisphere-pair matrix:
      // every N64 advances the physical row by M on each hemisphere. Prove
      // that address/lane/hemisphere map before exporting the logical concat;
      // do not gather projection pages just to construct a diagnostic output.
      if(auto native=contiguousPairView(value))return {native,value.shape,false};
      // Interleaved complete four-word groups already occupy matching native
      // distributed allocations. Export their physical halves, not a gathered
      // planar archive. Validate every logical element before selecting this ABI.
      if(!identity && value.dtype=="bf16" && value.shape.size()==2 && value.shape[1]%256==0) {
        const I columns=value.shape[1], local=columns/2;
        auto even=value.refs[0], odd=value.refs[128];
        auto ep=cache.at({even.node,even.output}), wp=cache.at({odd.node,odd.output});
        bool direct=text(ep,"kind")=="fp16_mxm_distributed_16" && text(wp,"kind")==text(ep,"kind")
            && text(ep,"hemisphere")=="east" && text(wp,"hemisphere")=="west"
            && s.change(ep,J{{"hemisphere","west"}})==wp
            && planner.nodes[even.node].outputs[even.output].shape==V({value.shape[0],local})
            && planner.nodes[odd.node].outputs[odd.output].shape==V({value.shape[0],local});
        for(I j=0;direct && j<value.size();++j) {
          const I c=j%columns; const auto owner=(c/128)%2 ? odd : even;
          const auto ref=value.refs[j];
          direct=ref.node==owner.node && ref.output==owner.output
              && ref.index==(j/columns)*local+(c/256)*128+c%128;
        }
        if(direct) {
          auto p=s.change(ep,J{{"kind","fp16_mxm_compact_block4_partition_distributed_16"},{"hemisphere","both"}});
          return {p,value.shape,false};
        }
      }
      // A prefix of a padded native tile is not the complete storage tensor.
      // Export only the requested live rows, not its unused physical lanes.
      if (identity)
        identity = value.size() == planner.nodes[first.node].outputs[first.output].size();
      // A column concat or a partial RoPE result is a view of live planar
      // producers. C2C can read those addresses directly. Describe complete
      // logical row runs with the existing strided output-piece ABI; do not
      // construct an on-chip archive just for host validation. This is only
      // an output binding, never a replacement for a consumer layout.
      if (!identity && (value.dtype=="bf16" || value.dtype=="f32")) {
        llvm::SmallVector<mlir::Attribute> pieces;
        mlir::DictionaryAttr firstPlacement;
        std::set<std::tuple<I,I,I,I>> physicalRows;
        bool direct=!value.refs.empty();
        I at=0;
        while(direct && at<value.size()) {
          const auto ref=value.refs[at];
          auto found=cache.find({ref.node,ref.output});
          if(ref.node<0 || found==cache.end()) {direct=false;break;}
          auto p=found->second;
          const auto &part=planner.nodes[ref.node].outputs[ref.output];
          const auto kind=text(p,"kind");
          const bool paired=kind=="fp16_hemisphere_pair_planar" || kind=="fp32_hemisphere_pair_planar";
          direct=(!p.get("dense_attention_role") || !llvm::StringRef(text(p,"dense_attention_role")).starts_with("flash_")) && part.dtype==value.dtype && part.shape.size()==2 &&
              slices(p).size()==size_t(value.bytes()) &&
              (paired || kind=="fp16_pair_planar" || kind=="fp32_quad_planar") &&
              !p.get("storage_shape") && !field(p,"bank_tile_rows") &&
              field(p,"address_stride")==1 && ref.index>=0 && ref.index<part.size();
          if(!direct)break;
          const I rows=part.shape[0],columns=part.shape[1],group=paired?64:32;
          const I column=ref.index%columns,row=ref.index/columns;
          direct=columns%group==0 && column%group==0 && (!paired || text(p,"hemisphere")=="both");
          I count=1;
          while(direct && at+count<value.size() && column+count<columns) {
            const auto next=value.refs[at+count];
            if(next.node!=ref.node || next.output!=ref.output || next.index!=ref.index+count)break;
            ++count;
          }
          if(!direct || count%group) {direct=false;break;}
          const I base=field(p,"base_row")+column/group*rows+row;
          if(base+(count/group-1)*rows>=s.target.memory().words_per_bank) {direct=false;break;}
          // Output pieces must not alias within one logical result. A
          // broadcast/repeated view still needs its normal materialization.
          for(I h=0;h<2;++h) if(paired || h==I(text(p,"hemisphere")!="east"))
            for(I slice:slices(p))for(I word=0;word<count/group;++word)
              direct &= physicalRows.emplace(h,slice,field(p,"bank"),base+word*rows).second;
          if(!direct)break;
          p=s.change(p,J{{"base_row",base},{"instruction_count",count/group},
              {"address_stride",rows},{"storage_shape",s.array({1,count})}});
          if(!paired && text(p,"hemisphere")=="both")p=s.change(p,J{{"hemisphere","west"}});
          if(!firstPlacement)firstPlacement=p;
          // Adjacent logical words from one producer form one strided C2C
          // piece. Do not emit a separate diagnostic binding for every row.
          bool merged=false;
          if(!pieces.empty()) {
            auto prior=llvm::cast<mlir::DictionaryAttr>(pieces.back());
            const I words=field(prior,"instruction_count"),more=count/group;
            const I stride=words==1?base-field(prior,"base_row"):field(prior,"address_stride");
            if(stride>0 && text(prior,"kind")==text(p,"kind") &&
               text(prior,"hemisphere")==text(p,"hemisphere") && slices(prior)==slices(p) &&
               field(prior,"bank")==field(p,"bank") &&
               base==field(prior,"base_row")+words*stride && (more==1 || rows==stride)) {
              pieces.back()=s.change(prior,J{{"instruction_count",words+more},{"address_stride",stride},
                  {"storage_shape",s.array({1,(words+more)*group})}});
              merged=true;
            }
          }
          if(!merged)pieces.push_back(p);
          at+=count;
        }
        if(direct && at==value.size() && !pieces.empty() && pieces.size()<=65536) {
          auto attrs=mlir::NamedAttrList(firstPlacement.getValue());
          attrs.erase("storage_shape");
          attrs.set("output_pieces",s.b.getArrayAttr(pieces));
          return {attrs.getDictionary(&s.context),value.shape,false};
        }
      }
      // A concatenation of complete native tensors can be exported directly
      // from its producer addresses. C2C packs each piece into the logical
      // output in order, after all consumers have completed.
      auto nativePieces = planner.policies.getAs<mlir::BoolAttr>("ftlpu.native_output_pieces");
      const bool planarDensePieces = !value.refs.empty() && llvm::all_of(value.refs,[&](const Ref &ref) {
        auto found=cache.find({ref.node,ref.output});
        return found!=cache.end() && found->second.get("dense_attention_role") && text(found->second,"dense_attention_role")=="context"
            && text(found->second,"kind")=="fp16_pair_planar";
      });
      if (!identity && ((nativePieces && nativePieces.getValue()) || planarDensePieces) &&
          (value.dtype == "f32" || value.dtype == "bf16")) {
        llvm::SmallVector<mlir::Attribute> pieces;
        mlir::DictionaryAttr firstPlacement;
        bool direct = true;
        I offset = 0;
        while (direct && offset < value.size()) {
          const auto ref = value.refs[offset];
          auto found = cache.find({ref.node,ref.output});
          if (ref.node < 0 || found == cache.end()) { direct = false; break; }
          auto p = found->second;
          const auto &part = planner.nodes[ref.node].outputs[ref.output];
          const auto kind = text(p,"kind");
          direct = (!p.get("dense_attention_role") || text(p,"dense_attention_role")=="context") && (kind == "fp32_quad_planar" || kind == "fp16_pair_planar") &&
              slices(p).size() == size_t(value.bytes()) && part.shape.size() == 2 &&
              !p.get("storage_shape") && !field(p,"bank_tile_rows") &&
              part.size() > 0 && part.size() <= value.size()-offset;
          for (I j = 0; direct && j < part.size(); ++j) {
            const auto item = value.refs[offset+j];
            direct = item.node == ref.node && item.output == ref.output && item.index == j;
          }
          if (!direct) break;
          if (text(p,"hemisphere") == "both") p = s.change(p,J{{"hemisphere","west"}});
          for (auto attr : pieces) {
            auto prior = llvm::cast<mlir::DictionaryAttr>(attr);
            if (overlap(prior,p) && field(prior,"base_row") < field(p,"base_row")+field(p,"instruction_count") &&
                field(p,"base_row") < field(prior,"base_row")+field(prior,"instruction_count"))
              direct = false;
          }
          if (!firstPlacement) firstPlacement = p;
          pieces.push_back(s.change(p,J{{"storage_shape",s.array(part.shape)}}));
          offset += part.size();
        }
        if (direct && pieces.size() > 1) {
          auto attrs = mlir::NamedAttrList(firstPlacement.getValue());
          attrs.set("output_pieces",s.b.getArrayAttr(pieces));
          return {attrs.getDictionary(&s.context),value.shape,false};
        }
      }
      // A concatenation of matching physical token partitions is only a
      // logical view. Export the native row-partition layout without FU copies.
      if (!identity && value.shape.size() == 2 && value.shape[0] >= 2
          && first.node >= 0 && cache.count({first.node, first.output})) {
        const I half = (value.shape[0] + 1) / 2, width = value.shape[1];
        const auto right = value.refs[half * width];
        auto leftP = cache.at({first.node, first.output});
        auto rightIt = cache.find({right.node, right.output});
        auto rightP = rightIt == cache.end() ? mlir::DictionaryAttr{} : rightIt->second;
        bool partitioned = rightP && right.node >= 0 && text(leftP,"kind") == "fp32_quad_planar"
            && text(rightP,"kind") == "fp32_quad_planar"
            && text(leftP,"hemisphere") == "west" && text(rightP,"hemisphere") == "east"
            && slices(leftP) == slices(rightP) && field(leftP,"base_row") == field(rightP,"base_row")
            && field(leftP,"bank") == field(rightP,"bank")
            && planner.nodes[first.node].parallel >= 0
            && planner.nodes[first.node].parallel == planner.nodes[right.node].parallel;
        for (I j = 0; partitioned && j < value.size(); ++j) {
          const auto ref = value.refs[j], owner = j < half * width ? first : right;
          partitioned = ref.node == owner.node && ref.output == owner.output
              && ref.index == j % (half * width);
        }
        if (partitioned) {
          auto p = s.change(leftP, J{{"kind", "token_partition_planar"}, {"hemisphere", "both"}});
          return {p,value.shape,false};
        }
      }
      // Export a linear tensor view from its actual storage geometry. Runtime
      // already unpacks native distributed matrices, so no FU copy is needed
      // merely to present a flat tensor to the caller.
      if (identity) {
        auto native = cache.at({first.node, first.output});
        const auto kind = text(native, "kind");
        const bool externalPlanar = scalar(native) || kind == "token_partition_planar"
            || kind == "token_partition_east_first_planar" ||
            (kind == "fp16_pair_planar" && slices(native).size() == 2) ||
            kind == "fp32_quad_planar" ||
            kind == "fp16_mxm_distributed_16" ||
            kind == "fp16_mxm_block4_partition_distributed_16" ||
            kind == "fp16_vxm_distributed_16";
        if (!externalPlanar || (native.get("dense_attention_role") && llvm::StringRef(text(native,"dense_attention_role")).starts_with("flash_")))
          identity = false;
      }
      mlir::DictionaryAttr finalPrototype;
      if (!identity && value.dtype == "bf16"
          && value.size()/s.target.memory().bytes_per_word > s.target.memory().sram_depth_rows
          && value.shape.back() % (4*s.target.memory().bytes_per_word) == 0)
        finalPrototype = s.placement("fp16_pair_planar", {16,17,18,19}, "both", 1, 0,
            value.size()/(4*s.target.memory().bytes_per_word));
      const bool packedPieces = !identity && llvm::all_of(value.refs, [&](const auto &ref) {
        const auto kind = text(cache.at({ref.node, ref.output}), "kind");
        return kind == "fp16_pair_planar" || kind == "fp32_quad_planar";
      });
      mlir::DictionaryAttr p;
      if(identity)p=cache.at({first.node,first.output});
      else {
        // Only re-place an output that already needs gathering. A direct
        // producer binding costs no archive copy and keeps its exact layout.
        if(preferTx && packedPieces && !finalPrototype) {
          std::set<Key> producers;
          for(const auto& ref:value.refs)producers.insert({ref.node,ref.output});
          std::vector<mlir::DictionaryAttr> sources;
          const size_t planes=value.dtype=="f32"?4:2;
          for(auto producer:producers) {
            auto home=cache.at(producer);
            if(slices(home).size()!=planes || field(home,"bank_tile_rows")) {
              sources.clear();break;
            }
            sources.push_back(home);
          }
          if(!sources.empty()) {
            try {
              p=allocate({-3,i},value,"west",1,{}, {},sources);
              usedTxHome=true;
            }
            catch(const GraphSramAllocationFailure&) {
              // No direct-copy destination fits. Preserve the original plan.
            }
          }
        }
        if(!p)p=allocate({-3,i},value,"west",packedPieces?1:-1,finalPrototype);
      }
      V exportedShape=value.shape;
      if (identity) {
        exportedShape=planner.nodes[first.node].outputs[first.output].shape;
        if (auto shape = p.getAs<mlir::ArrayAttr>("storage_shape")) {
          exportedShape.clear();
          for (auto d : shape) exportedShape.push_back(llvm::cast<mlir::IntegerAttr>(d).getInt());
        }
      }
      return {p,std::move(exportedShape),!identity};
    };
    // All exported buffers coexist through HostOutput. An earlier small
    // TX-optimized home can fragment the only legal run for a later large
    // tensor. Choose the complete set before emitting the first copy.
    std::vector<OutputHome> outputHomes(finals.size());
    try {
      for(I i:externalOutputs)outputHomes[i]=selectOutputHome(i,true);
    } catch(const GraphSramAllocationFailure &) {
      if(!usedTxHome)throw;
      for(I i=0;i<I(finals.size());++i) {
        cache.erase({-3,i});
        live.erase({-3,i});
      }
      for(I i:externalOutputs)outputHomes[i]=selectOutputHome(i,false);
    }
    for(I i:externalOutputs) {
      const auto &home=outputHomes[i];
      s.transportPhase="validation_export."+std::to_string(i);
      s.transportReason="graph_result_export";
      if(home.gather)materialize(finals[i],home.placement);
      Tensor exported=finals[i];
      exported.shape=home.shape;
      s.binding(exported,home.placement,"output",i,"output"+std::to_string(i));
    }
    llvm::json::Array residentOutputs;
    for(I i=0;i<I(finals.size());++i)if(!externalOutputs.count(i)) {
      const auto &value=finals[i];llvm::json::Array segments;
      for(I at=0;at<I(value.refs.size());) {
        const auto ref=value.refs[at];I count=1;
        while(at+count<I(value.refs.size()) && value.refs[at+count].node==ref.node &&
              value.refs[at+count].output==ref.output && value.refs[at+count].index==ref.index+count)++count;
        segments.push_back(J{{"logical_offset",at},{"elements",count},{"producer_node",ref.node},
            {"producer_output",ref.output},{"source_offset",ref.index}});
        at+=count;
      }
      residentOutputs.push_back(J{{"index",i},{"shape",s.array(value.shape)},{"dtype",value.dtype},
          {"storage","native_sram"},{"gathered",false},{"segments",std::move(segments)}});
    }
    s.b.create<mlir::func::ReturnOp>(s.loc);
    (*s.module)->setAttr("ftlpu.graph_lowering",
                         s.b.getStringAttr("native_cpp"));
    std::string plan;
    llvm::raw_string_ostream planStream(plan);
    llvm::json::Array weightPlacements;
    for (const auto& entry:statics) {
      const auto& value=entry.second;
      if (!field(value.p,"page_rows")) continue;
      weightPlacements.push_back(J{{"binding",value.index},
          {"bank",field(value.p,"bank")},{"base_row",field(value.p,"base_row")},
          {"consumer_node",field(value.p,"page_consumer_node",-1)},
          {"release_before_node",field(value.p,"page_release_before_node",-1)},
          {"rows",field(value.p,"page_rows")},{"slices",s.array(slices(value.p))}});
    }
    planStream << llvm::json::Value(
        J{{"planning", "tensor_subgraph_before_schedule"},
          {"external_output_indices",s.array(V(externalOutputs.begin(),externalOutputs.end()))},
          {"resident_output_views",std::move(residentOutputs)},
          {"weight_paging_policy",weightPingPong?"auto":"serialized"},
          {"weight_page_ping_pong",weightPingPong},
          {"weight_slot_base_row",weightPolicy.base_row},
          {"weight_slot_rows",weightPolicy.rows},
          {"weight_slot_slices",s.array(V(weightSlotSlices.begin(),weightSlotSlices.end()))},
          {"weight_page_placements",std::move(weightPlacements)},
          {"linker_mem_transports",std::move(s.transportDomains)},
          {"native_output_archive_copies", I(retainedOutputHomes.size())},
          {"projection_solve_overlap",std::move(projectionSolveReport)},
          {"projection_conv_overlap",std::move(projectionConvReport)},
          {"projection_vector_overlap",std::move(projectionVectorReport)},
          {"head_ingress_bindings",std::move(headIngressReport)},
          {"nodes", std::move(nodeRanges)}});
    std::ofstream(directory / "graph_plan.json") << plan << '\n';
    std::ofstream listing(directory / "inputs.txt");
    for (I i = 0; i < I(inputs.size()); ++i) {
      auto p = fs::absolute(directory / ("input" + std::to_string(i) + ".bin"));
      std::ofstream f(p, std::ios::binary);
      f.write(reinterpret_cast<const char *>(inputs[i].data.data()),
              inputs[i].data.size());
      listing << p.string() << "\n";
    }
  }
};
Tensor descriptor(const J &j, const std::vector<Tensor> &constants,
                  const std::vector<std::vector<Tensor>> &outputs) {
  Tensor t;
  t.shape = ints(*j.getArray("shape"));
  t.dtype = j.getString("dtype")->str();
  if (auto c = j.getInteger("constant"))
    return constants.at(*c);
  t.refs.resize(t.size());
  for (auto &item : *j.getArray("segments")) {
    auto &p = *item.getAsObject();
    auto &source =
        outputs.at(*p.getInteger("node")).at(*p.getInteger("output"));
    auto ix = ints(*p.getArray("indices")),
         pos = ints(*p.getArray("positions"));
    for (size_t i = 0; i < ix.size(); ++i)
      t.refs.at(pos[i]) = source.refs.at(ix[i]);
  }
  return t;
}
} // namespace

void ftlpu::compiler::import_logical_kernel_graph(const fs::path &input,
                                                 const fs::path &output) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, fc::kernel::KernelDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto parsed = llvm::json::parse(read(input));
  if (!parsed || !parsed->getAsObject()) throw std::runtime_error("invalid graph JSON");
  auto &graph = *parsed->getAsObject();
  if (graph.getString("format") != "ftlpu.logical-kernel-graph.v1")
    throw std::runtime_error("unsupported logical graph format");
  mlir::OpBuilder b(&context);
  auto loc = b.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Attribute> files;
  auto tensorType = [&](const J &j) {
    Tensor t;
    t.shape = ints(*j.getArray("shape"));
    t.dtype = j.getString("dtype")->str();
    return type(&context, t);
  };
  for (auto &item : *graph.getArray("constants")) {
    auto &j = *item.getAsObject();
    argTypes.push_back(tensorType(j));
    files.push_back(b.getStringAttr(j.getString("file")->str()));
  }
  auto function = mlir::func::FuncOp::create(loc, "decoder_layer", b.getFunctionType(argTypes, {}));
  module->push_back(function);
  function->setAttr("ftlpu.logical_inputs", b.getArrayAttr(files));
  function->setAttr("ftlpu.artifact_directory", b.getStringAttr(fs::absolute(output.parent_path()).string()));
  auto *block = function.addEntryBlock();
  b.setInsertionPointToStart(block);
  std::vector<llvm::SmallVector<mlir::Value>> nodeOutputs;
  auto edge = [&](const J &j) -> mlir::Value {
    auto resultType = tensorType(j);
    if (auto c = j.getInteger("constant")) {
      auto value = function.getArgument(*c);
      if (value.getType() != resultType) throw std::runtime_error("constant type mismatch");
      return value;
    }
    llvm::SmallVector<mlir::Value> inputs;
    V sources(resultType.getNumElements(), -1), indices(sources.size(), -1);
    for (auto &item : *j.getArray("segments")) {
      auto &segment = *item.getAsObject();
      auto value = nodeOutputs.at(*segment.getInteger("node"))[*segment.getInteger("output")];
      auto positions = ints(*segment.getArray("positions"));
      auto elements = ints(*segment.getArray("indices"));
      if (positions.size() != elements.size()) throw std::runtime_error("invalid view segment");
      for (size_t i = 0; i < positions.size(); ++i) {
        if (sources.at(positions[i]) != -1) throw std::runtime_error("duplicate view position");
        sources[positions[i]] = inputs.size(); indices[positions[i]] = elements[i];
      }
      inputs.push_back(value);
    }
    if (inputs.size() == 1 && inputs.front().getType() == resultType && indices == range(indices.size()))
      return inputs.front();
    mlir::OperationState state(loc, fc::kernel::IndexedViewOp::getOperationName());
    state.addOperands(inputs); state.addTypes(resultType);
    state.addAttribute("sources", b.getDenseI64ArrayAttr(sources));
    state.addAttribute("indices", b.getDenseI64ArrayAttr(indices));
    return b.create(state)->getResult(0);
  };
  I ordinal = 0;
  for (auto &item : *graph.getArray("nodes")) {
    auto &j = *item.getAsObject();
    auto region = mlir::parseSourceString<mlir::ModuleOp>(j.getString("kernel")->str(), &context);
    if (!region) throw std::runtime_error("invalid logical region");
    auto source = *region->getOps<mlir::func::FuncOp>().begin();
    mlir::IRMapping mapping;
    I ai = 0;
    for (auto &input : *j.getArray("inputs")) mapping.map(source.getArgument(ai++), edge(*input.getAsObject()));
    for (auto &op : source.front()) {
      if (llvm::isa<mlir::func::ReturnOp>(op)) continue;
      auto *copy = b.clone(op, mapping);
      copy->setAttr("ftlpu.logical_region", b.getI64IntegerAttr(ordinal));
      copy->setAttr("ftlpu.logical_name", b.getStringAttr(j.getString("name")->str()));
    }
    llvm::SmallVector<mlir::Value> results;
    for (auto value : source.front().getTerminator()->getOperands()) results.push_back(mapping.lookup(value));
    nodeOutputs.push_back(std::move(results)); ++ordinal;
  }
  llvm::SmallVector<mlir::Value> outputs;
  llvm::SmallVector<mlir::Type> outputTypes;
  for (auto &item : *graph.getArray("outputs")) {
    auto value = edge(*item.getAsObject()); outputs.push_back(value); outputTypes.push_back(value.getType());
  }
  b.create<mlir::func::ReturnOp>(loc, outputs);
  function.setType(b.getFunctionType(argTypes, outputTypes));
  if (mlir::failed(mlir::verify(*module))) throw std::runtime_error("invalid imported whole Kernel function");
  std::error_code ec; llvm::raw_fd_ostream stream(output.string(), ec);
  if (ec) throw std::runtime_error(ec.message());
  module->print(stream); stream << '\n';
}

static mlir::OwningOpRef<mlir::ModuleOp> lowerConnectedDecoderGraph(
    mlir::ModuleOp module, const fs::path &directory, bool retainNativeRhs,
    bool allowConvOverlap, bool allowVectorOverlap,
    I &nativeRhsReuses, bool &plannedConvOverlap, bool &plannedVectorOverlap,
    const std::string &vectorFallbackReason) {
  auto &context = *module.getContext();
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  if (!function.getBody().hasOneBlock() ||
      std::distance(module.getOps<mlir::func::FuncOp>().begin(),module.getOps<mlir::func::FuncOp>().end()) != 1)
    throw std::runtime_error("connected decoder requires one single-block function");
  auto target = fc::target::LPUTargetModel::from_operation(function);
  if (mlir::failed(target)) throw std::runtime_error("invalid target");
  auto graph = fc::kernel::match_decoder_graph(function);
  // Pattern recognition is shared with KernelToTensor. The stateful mixer
  // keeps its own pattern, while FFN is one native dense subgraph.
  Planner planner(context, *target);
  planner.retainNativeRhs = retainNativeRhs;
  planner.persistentState = !graph.delta_nets.empty();
  planner.hasAttention = !graph.attentions.empty();
  for (auto &ffn : graph.ffns)
    for (auto *user : ffn.output.getResult().getUsers())
      if (auto pointwise=llvm::dyn_cast<fc::kernel::VectorAluOp>(user))
        planner.retainedAcrossFfn |= !pointwise.getUseImmediate() && !pointwise.getReduce();
  // A head normalization that views the Conv result can consume distributed16
  // directly. Other Conv graphs keep their established planar result layout.
  const auto fromConvolution = [&](const auto &self, mlir::Value value) -> bool {
    auto *producer = value.getDefiningOp();
    if (!producer) return false;
    if (auto conv=llvm::dyn_cast<fc::kernel::DepthwiseConv1DOp>(producer))
      return conv.getInput().getType().getDimSize(0)%32==0;
    if (!llvm::isa<fc::kernel::IndexedViewOp>(producer)) return false;
    return llvm::any_of(producer->getOperands(), [&](mlir::Value operand) {
      return self(self, operand);
    });
  };
  for (auto &op : function.front()) {
    if (auto norm = llvm::dyn_cast<fc::kernel::L2NormOp>(op))
      planner.distributedConvOutput |= fromConvolution(fromConvolution, norm.getInput());
    if (auto norm = llvm::dyn_cast<fc::kernel::RmsNormOp>(op))
      planner.distributedConvOutput |= fromConvolution(fromConvolution,
                                                        norm.getInput());
  }
  // Decode head geometry comes from the semantic reduction and Conv graph,
  // not from checkpoint names or test-side physical permutations.
  for(auto &op:function.front()) {
    if(auto norm=llvm::dyn_cast<fc::kernel::L2NormOp>(op)) {
      const auto ty=norm.getInput().getType();
      if(ty.getRank()==2 && ty.getDimSize(0)>1 && ty.getDimSize(0)<32)
        planner.decodeHeadWidth=ty.getDimSize(1);
    }
    if(auto conv=llvm::dyn_cast<fc::kernel::DepthwiseConv1DOp>(op)) {
      const auto ty=conv.getInput().getType();
      if(ty.getDimSize(0)<=4) planner.decodeConvWidth=ty.getDimSize(1);
    }
  }
  mlir::NamedAttrList policies;
  for (auto name : {"ftlpu.weight_page_ping_pong", "ftlpu.common_transport_overlap", "ftlpu.vector_fp32_ingress", "ftlpu.mxm_execution_policy", "ftlpu.projection_rope_overlap", "ftlpu.rope_tile_overlap", "ftlpu.vector_bf16_spill", "ftlpu.vector_bf16_rne_spill", "ftlpu.gdn_decode_retrieval_mxm", "ftlpu.gdn_prefill_strategy", "ftlpu.common_barrier_protocol", "ftlpu.icu_compression", "ftlpu.ffn_schedule", "ftlpu.sxm_wide", "ftlpu.norm_distributed16", "ftlpu.l2_head_partition", "ftlpu.vector_log_softplus", "ftlpu.vector_bf16_token_halves"})
    if (auto attr = module->getAttr(name)) policies.set(name,attr);
  for(auto name:{"ftlpu.weight_paging_policy","ftlpu.weight_page_rows","ftlpu.projection_solve_overlap","ftlpu.projection_conv_overlap","ftlpu.elide_overwritten_packed_copies","ftlpu.native_output_pieces","ftlpu.passive_sr_multicast","ftlpu.projection_vector_overlap","ftlpu.external_output_indices"})
    if(auto attr=module->getAttr(name)) policies.set(name,attr);
  if(!allowConvOverlap)policies.set("ftlpu.projection_conv_overlap",mlir::BoolAttr::get(&context,false));
  if(!allowVectorOverlap)policies.set("ftlpu.projection_vector_overlap",mlir::BoolAttr::get(&context,false));
  auto weightPolicy=fc::target::GraphWeightPaging::resolve(*target,policies.getDictionary(&context));
  bool hasWeightDemand=false;
  function.walk([&](fc::kernel::BlockFp8ProjectionOp) { hasWeightDemand = true; });
  function.walk([&](fc::kernel::MatmulOp op) {
    const auto weightType = op.getRhs().getType().getElementType();
    hasWeightDemand |= weightType.isInteger(8) || llvm::isa<mlir::Float8E4M3FNType>(weightType);
  });
  // Resolve the effective reservation before native scratch/layout planning.
  // An automatic policy must not move scratch or change allocation choices
  // in a recurrence/normalization-only graph that has no weight page demand.
  weightPolicy.overlap &= hasWeightDemand;
  planner.policies = weightPolicy.attributes(&context,policies.getDictionary(&context));
  if (auto strategy = module->getAttrOfType<mlir::StringAttr>("ftlpu.ffn_schedule"))
    planner.ffnStrategy = strategy.getValue() == "fused" ? fc::FfnScheduleStrategy::Fused : fc::FfnScheduleStrategy::Tail;
  llvm::DenseMap<mlir::Value, Tensor> values;
  auto describe = [&](mlir::Value value) {
    auto ty = llvm::cast<mlir::RankedTensorType>(value.getType()); Tensor t;
    t.shape.assign(ty.getShape().begin(), ty.getShape().end());
    t.dtype = ty.getElementType().isBF16() ? "bf16" : ty.getElementType().isF32() ? "f32"
        : llvm::isa<mlir::Float8E4M3FNType>(ty.getElementType()) ? "f8E4M3FN"
        : llvm::isa<mlir::Float8E8M0FNUType>(ty.getElementType()) ? "f8E8M0FNU" : "i8";
    return t;
  };
  auto files = function->getAttrOfType<mlir::ArrayAttr>("ftlpu.logical_inputs");
  if (!files || files.size() != function.getNumArguments()) throw std::runtime_error("missing logical input files");
  for (auto [i,arg] : llvm::enumerate(function.getArguments())) {
    auto t = describe(arg); t.constant = i;
    auto data = read(llvm::cast<mlir::StringAttr>(files[i]).getValue().str());
    t.data.assign(data.begin(), data.end());
    if (I(t.data.size()) != t.size()*t.bytes()) throw std::runtime_error("logical input byte count mismatch");
    values[arg] = std::move(t);
  }
  std::map<I, llvm::SmallVector<mlir::Operation*>> groups;
  llvm::DenseMap<mlir::Operation*, I> groupOf;
  I nextGroup = 0;
  for (auto &op : function.front())
    if (!llvm::isa<fc::kernel::IndexedViewOp, mlir::func::ReturnOp>(op)) {
      I group = integer(&op,"ftlpu.logical_region",nextGroup);
      nextGroup = std::max(nextGroup,group+1); groupOf[&op] = group; groups[group].push_back(&op);
    }
  for (auto &ffn : graph.ffns) {
    I group = nextGroup++;
    for (auto *op : ffn.operations) {
      auto &old = groups[groupOf.find(op)->second]; llvm::erase(old,op);
      groupOf[op] = group;
    }
    groups[group].assign(ffn.operations.begin(),ffn.operations.end());
  }
  // Two independent narrow projections share X and fit one MXM output word
  // per hemisphere. Preserve each INT8 scale rather than requantizing either
  // checkpoint matrix to a common scale.
  llvm::SmallVector<fc::kernel::MatmulOp> narrow;
  for(auto op:function.getOps<fc::kernel::MatmulOp>()) {
    if(op.getM()!=target->throughput().mxm_rows || op.getN()>target->throughput().mxm_rows ||
       op.getN()==0 || !op.getRhs().getType().getElementType().isInteger(8) ||
       !op.getLhs().getType().getElementType().isBF16() ||
       groups[groupOf.lookup(op.getOperation())].size()!=1 ||
       !llvm::all_of(op.getResult().getUsers(),[](mlir::Operation *user){return llvm::isa<fc::kernel::VectorAluOp>(user);})) continue;
    auto policy=fc::target::mxm_execution_policy_from_operation(op.getOperation());
    if(mlir::failed(policy)) continue;
    auto strategy=fc::target::plan_mxm_execution_strategy(
        {I(op.getM()),2*I(target->throughput().mxm_rows),I(op.getK()),true,true,true,true},*target,*policy);
    if(mlir::failed(strategy) || !strategy->uses_local_dequant()) continue;
    auto found=llvm::find_if(narrow,[&](auto prior){return prior.getLhs()==op.getLhs() && prior.getK()==op.getK();});
    if(found==narrow.end()) {narrow.push_back(op);continue;}
    I from=groupOf.lookup(found->getOperation()),to=groupOf.lookup(op.getOperation());
    groups[from].push_back(op.getOperation());groups[to].clear();groupOf[op.getOperation()]=from;narrow.erase(found);
  }
  // Logical importer regions are not hardware barriers. A unary pointwise
  // consumer can join its pointwise producer before memory planning. Keep
  // externally used producer results as region returns; a private result
  // (e.g. multiply -> exp) then needs neither a spill nor a linker copy.
  // Row scans and specialized vector kernels are not pointwise, even when
  // reduce=false: their full-token layout cannot inherit token-half placement.
  auto isPointwise = [](fc::kernel::VectorAluOp op) {
    if (!op || op.getReduce()) return false;
    auto kind = op.getKind();
    return kind == "add" || kind == "subtract" || kind == "multiply" ||
           kind == "max" || kind == "negate" || kind == "cast" ||
           kind == "exp" || kind == "reciprocal" || kind == "rsqrt" ||
           kind == "log" || kind == "sigmoid" || kind == "softplus";
  };
  for (auto &operation : function.front()) {
    auto unary=llvm::dyn_cast<fc::kernel::VectorAluOp>(operation);
    if(!isPointwise(unary) || !unary.getUseImmediate()) continue;
    auto producer=unary.getLhs().getDefiningOp<fc::kernel::VectorAluOp>();
    if(!isPointwise(producer)) continue;
    I from=groupOf.lookup(producer.getOperation()),to=groupOf.lookup(&operation);
    if(from==to || groups[to].size()!=1) continue;
    if(!llvm::all_of(groups[from],[&](mlir::Operation *op) {
        return isPointwise(llvm::dyn_cast<fc::kernel::VectorAluOp>(op));
      })) continue;
    groups[from].push_back(&operation);groups[to].clear();groupOf[&operation]=from;
  }
  std::set<I> done;
  std::function<void(mlir::Value)> materialize;
  std::function<void(I)> lowerGroup;
  materialize = [&](mlir::Value value) {
    if (values.count(value)) return;
    auto viewOp = value.getDefiningOp<fc::kernel::IndexedViewOp>();
    if (!viewOp) { lowerGroup(groupOf.find(value.getDefiningOp())->second); return; }
    auto t = describe(value); t.refs.resize(t.size());
    for (auto input : viewOp.getInputs()) materialize(input);
    for (I i=0; i<t.size(); ++i) {
      auto &src = values.find(viewOp.getInputs()[viewOp.getSources()[i]])->second;
      if (src.external()) {
        auto incoming=viewOp.getInputs()[viewOp.getSources()[i]];
        auto origin=incoming;
        while(auto reshape=origin.getDefiningOp<fc::kernel::ReshapeOp>())
          origin=reshape->getOperand(0);
        const auto argument=llvm::dyn_cast<mlir::BlockArgument>(origin);
        const auto split=planner.policies.getAs<mlir::BoolAttr>("ftlpu.l2_head_partition");
        const auto wide=planner.policies.getAs<mlir::BoolAttr>("ftlpu.sxm_wide");
        bool eligible=argument && split && split.getValue() && wide && wide.getValue() &&
            src.dtype=="bf16" && src.shape.size()==2 && src.shape[0]==1 &&
            src.size()%32==0 && src.sourceContiguous;
        V owners(eligible?src.size()/32:0,-1);
        // Infer which fabric owns each complete reduction row from uses of
        // this external view. Never use tensor names or example bytes to
        // choose layout; unclaimed words remain a single ordinary ingress.
        if(eligible)for(auto* user:incoming.getUsers()) {
          auto v=llvm::dyn_cast<fc::kernel::IndexedViewOp>(user);
          if(!v)continue;
          for(auto* use:v.getResult().getUsers()) {
            auto norm=llvm::dyn_cast<fc::kernel::L2NormOp>(use);
            if(!norm)continue;
            const auto ty=norm.getInput().getType();
            if(ty.getRank()!=2 || ty.getDimSize(0)<2 || ty.getDimSize(0)>=32 ||
               ty.getDimSize(0)%2 || ty.getDimSize(1)%32)continue;
            for(I j=0;j<ty.getNumElements();++j) {
              if(v.getInputs()[v.getSources()[j]]!=incoming)continue;
              const I index=v.getIndices()[j];
              if(index<0 || index>=src.size() || index%32!=j%32) {eligible=false;break;}
              const I west=j/ty.getDimSize(1)<ty.getDimSize(0)/2;
              if(owners[index/32]>=0 && owners[index/32]!=west) {eligible=false;break;}
              owners[index/32]=west;
            }
          }
        }
        if(eligible && llvm::is_contained(owners,I(0)) && llvm::is_contained(owners,I(1))) {
          for(auto& owner:owners)if(owner<0)owner=1;
          std::vector<Tensor> pieces;
          for(I begin=0;begin<I(owners.size());) {
            I end=begin+1;
            while(end<I(owners.size()) && owners[end]==owners[begin])++end;
            const I first=begin*32,n=(end-begin)*32;
            auto part=view(src,range(n,first),{1,n});
            auto piece=planner.join(std::move(part),"head_ingress."+
                std::to_string(argument.getArgNumber())+"."+std::to_string(src.sourceOffset+first));
            auto& node=planner.nodes.back();
            node.headIngressArgument=argument.getArgNumber();
            node.headIngressOffset=(src.sourceOffset+first)*src.bytes();
            node.headIngressWest=owners[begin];
            pieces.push_back(std::move(piece));
            begin=end;
          }
          src=concat(pieces,1);
        } else src=planner.join(src,"external_indexed_view");
      }
      t.refs[i] = src.refs.at(viewOp.getIndices()[i]);
    }
    values[value] = std::move(t);
  };
  lowerGroup = [&](I group) {
    if (done.count(group)) return;
    auto &ops = groups.at(group);
    llvm::SmallVector<mlir::Value> inputs, outputs;
    for (auto *op : ops) {
      for (auto value : op->getOperands())
        if (!llvm::is_contained(ops,value.getDefiningOp()) && !llvm::is_contained(inputs,value)) inputs.push_back(value);
      for (auto value : op->getResults())
        if (llvm::isa<fc::kernel::GatedDeltaRuleOp, fc::kernel::ScaledMatvecOp>(op) ||
            llvm::any_of(value.getUsers(),[&](auto *user){return !llvm::is_contained(ops,user);})) outputs.push_back(value);
    }
    // The recurrent/state lowerers tile positional operand and result roles.
    // Retain even unused diagnostic results of scaled_matvec: its output
    // indices still denote scaled state, KS, delta, updated state and readout.
    // Preserve operand roles
    // even when SSA aliases (for example Q == K) use the same input value.
    const bool recurrentRoles = ops.size() == 1 &&
        llvm::isa<fc::kernel::GatedDeltaRuleOp, fc::kernel::ScaledMatvecOp>(ops.front());
    if (recurrentRoles)
      inputs.assign(ops.front()->operand_begin(), ops.front()->operand_end());
    for (auto input : inputs) materialize(input);
    mlir::OpBuilder b(&context); auto loc = function.getLoc();
    mlir::OwningOpRef<mlir::ModuleOp> part = mlir::ModuleOp::create(loc);
    llvm::SmallVector<mlir::Type> argTypes,resultTypes;
    std::vector<Tensor> args,results;
    for (auto value : inputs) {argTypes.push_back(value.getType()); args.push_back(values.find(value)->second);}
    for (auto value : outputs) {resultTypes.push_back(value.getType()); results.push_back(describe(value));}
    auto f = mlir::func::FuncOp::create(loc,"region",b.getFunctionType(argTypes,resultTypes)); part->push_back(f);
    if (auto policy = function->getAttr("ftlpu.vector_bf16_spill"))
      f->setAttr("ftlpu.vector_bf16_spill", policy);
    else if (auto policy = module->getAttr("ftlpu.vector_bf16_spill"))
      f->setAttr("ftlpu.vector_bf16_spill", policy);
    // Region extraction must preserve the decoder's GDN lowering policy.
    // Otherwise a whole-function chunkwise request silently becomes a scan
    // when Planner::lower matches the extracted recurrence subgraph.
    if (auto policy = function->getAttr("ftlpu.gdn_prefill_strategy"))
      f->setAttr("ftlpu.gdn_prefill_strategy", policy);
    else if (auto policy = module->getAttr("ftlpu.gdn_prefill_strategy"))
      f->setAttr("ftlpu.gdn_prefill_strategy", policy);
    if (auto policy = function->getAttr("ftlpu.gdn_chunk_size"))
      f->setAttr("ftlpu.gdn_chunk_size", policy);
    else if (auto policy = module->getAttr("ftlpu.gdn_chunk_size"))
      f->setAttr("ftlpu.gdn_chunk_size", policy);
    b.setInsertionPointToStart(f.addEntryBlock()); mlir::IRMapping mapping;
    for (auto [old,value] : llvm::zip(inputs,f.getArguments())) mapping.map(old,value);
    if (recurrentRoles) {
      auto *op = ops.front();
      mlir::OperationState state(op->getLoc(), op->getName());
      state.addOperands(f.getArguments());
      state.addTypes(op->getResultTypes());
      state.addAttributes(op->getAttrs());
      auto *copy = b.create(state);
      for (auto [old,value] : llvm::zip(op->getResults(),copy->getResults()))
        mapping.map(old,value);
    } else {
      for (auto *op : ops) b.clone(*op,mapping);
    }
    llvm::SmallVector<mlir::Value> returns;
    for (auto value : outputs) returns.push_back(mapping.lookup(value));
    b.create<mlir::func::ReturnOp>(loc,returns);
    std::string source; llvm::raw_string_ostream stream(source); part->print(stream);
    auto name = ops.back()->getAttrOfType<mlir::StringAttr>("ftlpu.logical_name");
    auto lowered = planner.lower(J{{"name",name ? name.getValue().str() : "region"+std::to_string(group)},
                                  {"kernel",source}},std::move(args),std::move(results));
    for (auto [value,t] : llvm::zip(outputs,lowered)) values[value] = std::move(t);
    done.insert(group);
  };
  std::vector<Tensor> finals;
  for (auto value : function.front().getTerminator()->getOperands()) {materialize(value); finals.push_back(values.find(value)->second);}
  fs::create_directories(directory);
  nativeRhsReuses = planner.nativeRhsReuses;
  Linker linker(planner,std::move(finals),directory);
  plannedConvOverlap=llvm::any_of(planner.nodes,[](const Node& node){
    return node.projectionConvGroup>=0 && !node.projectionVectorMember;
  });
  plannedVectorOverlap=llvm::any_of(planner.nodes,[](const Node& node){return node.projectionVectorMember;});
  linker.build();
  (*linker.s.module)->setAttr("ftlpu.decoder_graph_matcher",mlir::StringAttr::get(&context,"shared"));
  std::ofstream(directory/"lowering_path.json") << "{\"entry\":\"kernel\",\"graph_matcher\":\"match_decoder_graph\",\"ffn_subgraphs\":"
      << graph.ffns.size() << ",\"deltanet_subgraphs\":" << graph.delta_nets.size()
      << ",\"shared_native_rhs_reuses\":" << nativeRhsReuses
      << ",\"projection_conv_sram_fallback\":" << (allowConvOverlap?"false":"true")
      << ",\"projection_vector_fallback\":" << (allowVectorOverlap?"false":"true")
      << ",\"projection_vector_fallback_reason\":" << std::quoted(vectorFallbackReason)
      << ",\"memory_planning\":\"whole_graph_tensor_before_schedule\"}\n";
  return std::move(linker.s.module);
}

mlir::OwningOpRef<mlir::ModuleOp> ftlpu::compiler::lower_connected_decoder_graph(
    mlir::ModuleOp module, const fs::path &directory) {
  bool allowConvOverlap=true,allowVectorOverlap=true,retainNativeRhs=true;
  std::string vectorFallbackReason;
  for(;;) {
    bool plannedConvOverlap=false;
    bool plannedVectorOverlap=false;
    I nativeRhsReuses=0;
    try {
      return lowerConnectedDecoderGraph(module,directory,retainNativeRhs,
                                        allowConvOverlap,allowVectorOverlap,nativeRhsReuses,
                                        plannedConvOverlap,plannedVectorOverlap,vectorFallbackReason);
    } catch(const GraphSramAllocationFailure &error) {
      // Retry from SSA with shorter live ranges. Keep the existing State tile
      // reuse when serializing projection/conv suffices to fit the program.
      if(allowVectorOverlap && plannedVectorOverlap) {
        allowVectorOverlap=false;
        vectorFallbackReason=error.what();
      } else if(allowConvOverlap && plannedConvOverlap)allowConvOverlap=false;
      else if(retainNativeRhs && nativeRhsReuses)retainNativeRhs=false;
      else throw;
    }
  }
}

void ftlpu::compiler::compile_logical_kernel_graph(const fs::path &input,
                                                   const fs::path &output,
                                                   const fs::path &config,
                                                   const fs::path &directory) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, fc::kernel::KernelDialect,
                  fc::tensor::TensorDialect, fc::stream::StreamDialect,
                  fc::schedule::ScheduleDialect>();
  mlir::MLIRContext context(registry);
  context.disableMultithreading();
  context.loadAllAvailableDialects();
  fc::target::LPUTargetModel target;
  if (!config.empty()) {
    std::string error;
    auto parsed = fc::target::LPUTargetModel::from_json(read(config), error);
    if (mlir::failed(parsed))
      throw std::runtime_error(error);
    target = *parsed;
  }
  auto parsed = llvm::json::parse(read(input));
  if (!parsed)
    throw std::runtime_error("invalid graph JSON");
  auto &graph = *parsed->getAsObject();
  if (graph.getString("format") != "ftlpu.logical-kernel-graph.v1")
    throw std::runtime_error("unsupported logical graph format");
  std::vector<Tensor> constants;
  for (auto &item : *graph.getArray("constants")) {
    auto &j = *item.getAsObject();
    Tensor t;
    t.shape = ints(*j.getArray("shape"));
    t.dtype = j.getString("dtype")->str();
    t.constant = constants.size();
    auto data = read(j.getString("file")->str());
    t.data.assign(data.begin(), data.end());
    if (I(t.data.size()) != t.size() * t.bytes())
      throw std::runtime_error("constant byte count mismatch");
    constants.push_back(std::move(t));
  }
  Planner planner(context, target);
  const bool hasWeightDemand=llvm::any_of(*graph.getArray("nodes"),[](const auto& item) {
    const auto& node=*item.getAsObject();
    if(!node.getString("kernel").value_or("").contains("ftlpu.kernel.matmul")) return false;
    return llvm::any_of(*node.getArray("inputs"),[](const auto& input) {
      return input.getAsObject()->getString("dtype").value_or("")=="i8";
    });
  });
  if(!hasWeightDemand) {
    auto policy=fc::target::GraphWeightPaging::resolve(target,planner.policies);
    policy.overlap=false;
    planner.policies=policy.attributes(&context,{});
  }
  std::set<I> convolutionNodes;
  I logicalNode = 0;
  for (auto &item : *graph.getArray("nodes")) {
    auto &node = *item.getAsObject();
    const auto source = node.getString("kernel").value_or("");
    planner.hasAttention |= source.contains("ftlpu.kernel.batch_matmul");
    if (source.contains("ftlpu.kernel.depthwise_conv1d"))
      convolutionNodes.insert(logicalNode);
    if (source.contains("ftlpu.kernel.rms_norm") || source.contains("ftlpu.kernel.l2_norm"))
      for (auto &input : *node.getArray("inputs"))
        if (auto segments = input.getAsObject()->getArray("segments"))
          for (auto &segment : *segments)
            planner.distributedConvOutput |= convolutionNodes.count(
                *segment.getAsObject()->getInteger("node")) != 0;
    ++logicalNode;
  }
  std::vector<std::vector<Tensor>> outputs;
  for (auto &item : *graph.getArray("nodes")) {
    auto &j = *item.getAsObject();
    std::vector<Tensor> args, result;
    for (auto &a : *j.getArray("inputs"))
      args.push_back(descriptor(*a.getAsObject(), constants, outputs));
    for (auto &a : *j.getArray("outputs")) {
      auto &o = *a.getAsObject();
      Tensor t;
      t.shape = ints(*o.getArray("shape"));
      t.dtype = o.getString("dtype")->str();
      result.push_back(t);
    }
    outputs.push_back(planner.lower(j, args, result));
  }
  std::vector<Tensor> finals;
  for (auto &o : *graph.getArray("outputs"))
    finals.push_back(descriptor(*o.getAsObject(), constants, outputs));
  Linker linker(planner, finals, directory);
  linker.build();
  std::error_code ec;
  llvm::raw_fd_ostream stream(output.string(), ec);
  if (ec)
    throw std::runtime_error(ec.message());
  linker.s.module->print(stream);
  stream << "\n";
  llvm::outs() << "native graph: " << graph.getArray("nodes")->size()
               << " logical nodes, " << planner.nodes.size()
               << " physical domains, cycles=" << linker.s.cursor << "\n";
}
