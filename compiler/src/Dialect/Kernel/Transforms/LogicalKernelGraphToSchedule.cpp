#include "ftlpu/compiler/Transforms/logical_kernel_graph.hpp"
// Logical tensor graphs are planned here, before any physical schedule exists.
// Python importers supply equations, tensor views and checkpoint bytes only.
#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Kernel/Analysis/decoder_graph.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
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
  if (text(a, "hemisphere") != text(b, "hemisphere") &&
      text(a, "hemisphere") != "both" && text(b, "hemisphere") != "both")
    return false;
  if (field(a, "bank") != field(b, "bank"))
    return false;
  for (I s : slices(a))
    if (contains(slices(b), s))
      return true;
  return false;
}
struct Ref {
  I node = -1, output = 0, index = 0;
};
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
  I bytes() const { return dtype == "f32" ? 4 : dtype == "i8" ? 1 : 2; }
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
  bool recurrent = false;
  bool ffn = false;
};
class Planner {
public:
  mlir::MLIRContext &context;
  fc::target::LPUTargetModel target;
  fc::FfnScheduleStrategy ffnStrategy = fc::FfnScheduleStrategy::Tail;
  mlir::DictionaryAttr policies;
  bool persistentState = false;
  bool hasAttention = false;
  bool distributedConvOutput = false;
  std::vector<Node> nodes;
  Planner(mlir::MLIRContext &c, fc::target::LPUTargetModel t)
      : context(c), target(t) {}
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
    // RoPE tables are initialized eagerly in bank1, before the first RMSNorm
    // executes. Its gamma must be on the other bank even though attention's
    // ordinary scratch does not become live until later.
    if (persistentState || (hasAttention && nodes.empty()))
      n.function->setAttr("ftlpu.feedback_gamma_bank",mlir::IntegerAttr::get(mlir::IntegerType::get(&context,64),0));
    auto patterns = fc::kernel::match_decoder_graph(n.function);
    n.ffn = !patterns.ffns.empty();
    n.recurrent = !patterns.delta_nets.empty();
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
      } else if (auto conv = llvm::dyn_cast<fc::kernel::DepthwiseConv1DOp>(op))
        conv.getResult().setType(type(&context, outputs[0]));
      else if (auto delta = llvm::dyn_cast<fc::kernel::GatedDeltaRuleOp>(op))
        for (size_t i = 0; i < outputs.size(); ++i)
          delta.getResult(i).setType(type(&context, outputs[i]));
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
      else if (llvm::isa<fc::kernel::VectorAluOp>(op))
        n.outputSlots[i] = slot(op, "ftlpu.vector_result_placement");
      else
        n.outputSlots[i] = slot(op, "result_allocations");
      n.out.push_back(n.outputSlots[i].get());
    }
    for (auto &slots : n.inputSlots) {
      if (slots.empty())
        throw std::runtime_error("unplanned native input: " + n.name);
      n.in.push_back(slots.front().get());
    }
    // Only fixed native workspaces are reserved. Returned values will be
    // assigned graph lifetimes below; their standalone output areas disappear.
    const auto visit = [&](const auto &self, mlir::Attribute a) -> void {
      if (auto p = llvm::dyn_cast<mlir::DictionaryAttr>(a)) {
        if (p.get("slices") && p.get("kind") && p.get("hemisphere") &&
            p.get("instruction_count"))
          {
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
  Tensor join(Tensor value, std::string name) {
    Node n;
    n.name = std::move(name);
    n.inputs = {std::move(value)};
    n.outputs = {output(nodes.size(), 0, n.inputs[0].shape, n.inputs[0].dtype)};
    auto out = n.outputs[0];
    nodes.push_back(std::move(n));
    return out;
  }
  std::vector<Tensor> lower(const J &j, std::vector<Tensor> input,
                            std::vector<Tensor> outputs) {
    auto source = j.getString("kernel")->str(),
         name = j.getString("name")->str();
    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    if (!parsed) throw std::runtime_error("invalid kernel subgraph: " + name);
    auto patterns = fc::kernel::match_decoder_graph(*parsed->getOps<mlir::func::FuncOp>().begin());
    if (!patterns.ffns.empty()) {
      auto &ffn = patterns.ffns.front();
      const I m = ffn.gate.getM(), hidden = ffn.gate.getN();
      const auto &hw = target.throughput();
      const I tile = hw.mxm_rows;
      const I tokenRows = ((m + tile - 1) / tile) * tile;
      const I alignment = target.memory().hemispheres * tile;
      // The SR SwiGLU retains two Gate copies and Up over this output batch.
      // Derive the batch from physical residency, before emitting any ICU/FU
      // instructions. Decode retains its independent native strategy.
      const I batch = hw.vxm_alus == 16 && m >= tile
          ? (target.memory().sram_depth_rows / tokenRows) * alignment
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
        std::string addSource = "module { func.func @sum(%a: " + ty
            + ", %b: " + ty + ") -> " + ty
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
    if (source.find("ftlpu.kernel.batch_matmul") != std::string::npos)
      return add(name, source, std::move(input), std::move(outputs));
    if (source.find("ftlpu.kernel.matmul") != std::string::npos) {
      I k = input[0].shape[1], width = input[1].shape[1],
        page = 64 * std::max<I>(1, (target.memory().sram_depth_rows / 2) /
                                       ((k / 32) * 4));
      std::vector<Tensor> pieces;
      for (I first = 0; first < width; first += page) {
        I count = std::min(page, ((width - first + 63) / 64) * 64);
        auto w = sliceColumns(input[1], first, count, true);
        auto o = outputs[0];
        o.shape[1] = count;
        auto values = add(name + ".page" + std::to_string(first / page), source,
                          {input[0], w}, {o}, -1, count);
        pieces.push_back(
            sliceColumns(values[0], 0, std::min(count, width - first)));
      }
      return {concat(pieces, 1)};
    }
    if (source.find("ftlpu.kernel.depthwise_conv1d") != std::string::npos) {
      I width = input[0].shape[1], tile = std::min<I>(1024, width);
      std::vector<Tensor> groups;
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
        }
        auto groupedResult = concat(pieces, 1);
        groups.push_back(distributedConvOutput
                             ? std::move(groupedResult)
                             : join(std::move(groupedResult),
                                    "conv_layout_group" + std::to_string(first)));
      }
      return {concat(groups, 1)};
    }
    if (!patterns.delta_nets.empty()) {
      std::vector<Tensor> pieces;
      Tensor state = input[5];
      for (I row = 0; row < input[0].shape[0]; ++row) {
        std::vector<Tensor> args;
        for (size_t i = 0; i < 5; ++i)
          args.push_back(sliceRows(input[i], row, 1));
        args.push_back(state);
        auto o = outputs[0];
        o.shape[0] = 1;
        auto values = add(name + ".token" + std::to_string(row), source, args,
                          {o, outputs[1]}, 1);
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
      if (auto vector = llvm::dyn_cast<fc::kernel::VectorAluOp>(op)) return !vector.getReduce();
      return llvm::isa<mlir::func::ReturnOp>(op);
    });
    if (pointwiseGraph && target.throughput().vxm_alus == 16
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
      const I half = (tokenRows + 1) / 2, group = nodes.size();
      for (I first : {I(0), half}) {
        const I count = std::min(half, tokenRows - first);
        auto args = input, outs = outputs;
        for (auto &a : args) if (a.shape[0] != 1) a = sliceRows(a, first, count);
        for (auto &o : outs) o.shape[0] = count;
        auto values = add(name + ".tokens" + std::to_string(first), source,
                          args, outs, count, -1, first != 0, group);
        for (size_t i = 0; i < values.size(); ++i) pieces[i].push_back(values[i]);
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
  void scalarDomain(mlir::DictionaryAttr src, mlir::DictionaryAttr dst, I a,
                    I z, const Domain &d) {
    I beat = cursor + 32, q = bridgeQueue(),
      n = product(V(d.count.begin(), d.count.end()));
    auto ss = slices(src), ds = slices(dst);
    std::array<I, 3> intervals = {period, d.count[0] * period,
                                  d.count[0] * d.count[1] * period};
    // The MXM's two-hemisphere output and a Conv1D byte-plane input carry
    // identical BF16 words. Their physical MEM queues can exchange those
    // words on the passive SR path; an ALU cast chain would only re-encode
    // the same bits and consume VXM instruction bandwidth.
    const auto bf16Pair = [](mlir::DictionaryAttr p) {
      const auto kind = text(p, "kind");
      return kind == "fp16_hemisphere_pair_planar" ||
             kind == "fp16_pair_planar" ||
             kind == "fp16_mxm_distributed_16" ||
             kind == "fp16_vxm_distributed_16";
    };
    const bool passivePair =
        bf16Pair(src) && bf16Pair(dst) &&
        ss.size() == 2 && ds.size() == 2 &&
        (text(src, "hemisphere") != text(dst, "hemisphere") ||
         (ss[0] / group != ds[0] / group &&
          ss[1] / group != ds[1] / group));
    if (passivePair) {
      I maxLatency = 0;
      for (I p = 0; p < 2; ++p) {
        I sourceGroup = ss[p] / group, destGroup = ds[p] / group;
        I readStream = 20 + p, writeStream = 20 + p, latency = 0;
        if (text(src, "hemisphere") != text(dst, "hemisphere")) {
          readStream += target.streams().streams_per_direction;
          latency = sourceGroup + target.throughput().vxm_alus +
                    destGroup + 3;
        } else if (sourceGroup < destGroup) {
          latency = destGroup - sourceGroup;
        } else if (sourceGroup > destGroup) {
          readStream += target.streams().streams_per_direction;
          writeStream += target.streams().streams_per_direction;
          latency = sourceGroup - destGroup;
        }
        mem(beat, src, p, "read", field(src, "base_row") + a,
            readStream, d.count, d.src, intervals);
        mem(beat + latency, dst, p, "write",
            field(dst, "base_row") + z, writeStream, d.count, d.dst,
            intervals);
        maxLatency = std::max(maxLatency, latency);
      }
      cursor = beat + n * period + maxLatency + 96;
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
    I pos = 0, n = indices.size();
    while (pos < n) {
      I a = pos + 1 < n ? indices[pos + 1] - indices[pos] : 0,
        z = pos + 1 < n ? offsets[pos + 1] - offsets[pos] : 0, end = pos + 1;
      while (end < n && indices[end] - indices[end - 1] == a &&
             offsets[end] - offsets[end - 1] == z)
        ++end;
      scalarDomain(src, dst, indices[pos], offsets[pos],
                   Domain{{end - pos, 1, 1}, {a, 0, 0}, {z, 0, 0}});
      pos = end;
    }
  }
  void unpack(mlir::DictionaryAttr source, mlir::DictionaryAttr dest,
              const V &shape, const V &indices, const V &offsets) {
    I rows = shape[0], columns = shape[1];
    bool paired = text(source, "kind") == "fp16_hemisphere_pair_planar";
    I step = paired ? 64 : 32;
    std::map<I, std::set<I>> wordLanes;
    for (I ix : indices)
      wordLanes[ix / 32].insert(ix % 32);
    bool full = !wordLanes.empty();
    for (auto &p : wordLanes)
      full &= p.second.size() == 32;
    for (I half = 0; half < (paired && full ? 2 : 1); ++half)
      for (I residue = 0; residue < (full ? 8 : std::min(step, columns));
           ++residue) {
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
          sxm(transpose, text(src, "hemisphere"), "transpose", {16, 17},
              range(16, 16), range(32), d.count[0], 32, -1,
              d.count[1] * d.count[2], d.count[0] * 32);
          I permute = transpose + 5;
          bool direct = text(src, "hemisphere") == text(dest, "hemisphere");
          if (ds.size() == 4 && !direct)
            throw std::runtime_error("FP32 unpack requires same hemisphere");
          V map;
          for (I i = 0; i < 32; ++i)
            map.push_back(tile * 8 + i % 8);
          sxm(permute, text(src, "hemisphere"), "permute", range(16, 16),
              range(16, 32), map, d.count[0], 32, lane < 7 ? 9 + lane : 7,
              d.count[1] * d.count[2], d.count[0] * 32, full ? 8 : 0);
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
          cursor = permute + count * 32 + 32;
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
  fs::path directory;
  std::map<Key, mlir::DictionaryAttr> cache, live;
  std::map<Key, I> last;
  std::map<std::pair<std::string, I>, std::vector<Key>> native;
  struct TimedWorkspace { I node, begin, end; };
  std::map<std::pair<std::string,I>,std::vector<TimedWorkspace>> nativeTimed;
  std::map<std::pair<std::string, I>, I> staticEnd;
  struct Static {
    mlir::DictionaryAttr p;
    I index;
    mlir::Value binding;
  };
  std::map<std::string, Static> statics;
  std::vector<Tensor> inputs;
  llvm::json::Array nodeRanges;
  std::map<I,mlir::DictionaryAttr> recurrentStates;
  std::map<Key,mlir::DictionaryAttr> partitionOutputHomes;
  std::map<Key,mlir::DictionaryAttr> retainedOutputHomes;
  Linker(Planner &p, std::vector<Tensor> out, fs::path dir)
      : planner(p), s(p.context, p.target), finals(std::move(out)),
        directory(std::move(dir)) {
    for (I i = 0; i < I(p.nodes.size()); ++i) {
      for (auto &a : p.nodes[i].inputs)
        for (auto ref : a.refs)
          last[{ref.node, ref.output}] = i;
      if (!p.nodes[i].module)
        continue;
      for (auto workspace : p.nodes[i].workspaces) {
        if (field(workspace, "bank") != 1)
          continue;
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
           const std::vector<mlir::DictionaryAttr> &forbiddenQueues = {}) {
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
    const I born = key.first >= 0 ? key.first
                   : key.first == -4 ? key.second
                   : key.first == -3 ? I(planner.nodes.size()) : 0;
    const I dead = last.count(key) ? last.at(key) : planner.nodes.size();
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
    for (I first : pairs) {
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
      for (auto h : {"east", "west"})
        if (hemisphere == h || hemisphere == "both")
          for (I slice = first; slice < first + planes; ++slice)
          {
            for (auto r : native[{h, slice}]) {
              ranges.push_back(r);
              candidates.insert(r.second);
            }
            for (auto r : nativeTimed[{h,slice}])
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
            range(planes, first), hemisphere, 1, row, rows);
        if (prototype)
          p = s.change(p, J{{"kind", text(prototype, "kind")},
                            {"instruction_count", rows}});
        for (auto busy : forbiddenQueues)
          if (overlap(p, busy))
            conflict = true;
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
    for (auto busy : forbiddenQueues)
      llvm::errs() << "unavailable output queue: " << busy << "\n";
    for (const auto &entry : live)
      llvm::errs() << "live graph value " << entry.first.first << ":"
                   << entry.first.second << " through " << last[entry.first]
                   << ": " << entry.second << "\n";
    throw std::runtime_error(
        "graph SRAM allocation failed for " + std::to_string(key.first) + ":" +
        std::to_string(key.second) + " rows=" + std::to_string(rows));
  }
  Static input(const Tensor &t, mlir::DictionaryAttr original,
               std::string name, bool directNativeIngress = false,
               I directLastUse = 0) {
    // Views of a retained activation use the original upload and carry
    // their logical offset into the on-chip layout conversion.
    if (t.external() && t.sourceContiguous)
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
    if (auto paged = original.getAs<mlir::BoolAttr>("paged_weight"); paged && paged.getValue())
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
      for (const auto &entry : live)
        if (overlap(p, entry.second))
          throw std::runtime_error("direct native input overlaps a live binding: " + name);
      live[{-2, index}] = p;
      last[{-2, index}] = directLastUse;
    } else if (auto paged = p.getAs<mlir::BoolAttr>("paged_weight"); paged && paged.getValue()) {
      // Native composite operators already own their physical page layout.
      // Preserve it and let compiler page SYNCs and runtime C2C feed the data.
    } else if (t.dtype == "i8" && text(p, "kind") == "w8a16_mxm_weight_striped") {
      p = s.change(p, J{{"bank", 0},
                        {"base_row", 0},
                        {"paged_weight", true},
                        {"page_count", 1},
                        {"page_rows", field(p, "instruction_count")},
                        {"page_granularity", t.shape[1] / 64},
                        {"page_role_group_base", 0},
                        {"page_role_group_count", 1},
                        {"page_items_per_slice_group", t.shape[1] / 64},
                        {"page_bank_count", 2},
                        {"page_storage_slices", s.array(slices(p))}});
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
    } else if (scalar(p) || text(p, "kind") == "fp16_pair_planar" ||
               text(p, "kind") == "fp16_mxm_activation_planar" ||
               text(p, "kind") == "fp16_mxm_distributed_16" ||
               text(p, "kind") == "fp32_quad_planar") {
      p = allocate({-2, index}, t, text(p, "hemisphere"), !scalar(p));
      last[{-2, index}] = planner.nodes.size() + 1;
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
      if (row + field(p, "instruction_count") >
          s.target.memory().sram_depth_rows)
        throw std::runtime_error("immutable binding exceeds SRAM: " + name);
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
  void move(mlir::DictionaryAttr src, mlir::DictionaryAttr dst,
            const V &sourceShape, const V &destShape, const V &ix,
            const V &pos) {
    const I word = s.target.memory().bytes_per_word;
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
    if (text(dst,"kind") == "fp16_pair_planar" && slices(dst).size() == 4) {
      // Two packed BF16 pairs per hemisphere partition physical feature words.
      // Four planes here describe
      // two vectors, not the four bytes of an FP32 conversion.
      auto ids = slices(dst);
      const I hemispheres = text(dst,"hemisphere") == "both" ? 2 : 1;
      for (I h = 0; h < hemispheres; ++h)
      for (I pair = 0; pair < 2; ++pair) {
        V selected, positions;
        for (size_t i = 0; i < pos.size(); ++i) {
          I col = pos[i] % destShape[1];
          if ((col/word)%2 != pair
              || (hemispheres == 2 && (col/(2*word))%2 != h)) continue;
          selected.push_back(ix[i]);
          positions.push_back((pos[i]/destShape[1])*(destShape[1]/(2*hemispheres))+
                              (col/(2*word*hemispheres))*word+col%word);
        }
        if (selected.empty()) continue;
        auto physical = s.change(dst,J{{"slices",s.array({ids[2*pair],ids[2*pair+1]})},
            {"hemisphere",hemispheres == 2 ? (h ? "west" : "east") : text(dst,"hemisphere")}});
        move(src,physical,sourceShape,{destShape[0],destShape[1]/(2*hemispheres)},selected,positions);
      }
      return;
    }
    if (text(src,"kind") == "fp16_pair_planar" && slices(src).size() == 4) {
      // Dense attention's canonical O result has two independent BF16
      // plane pairs per hemisphere. Split tensor geometry before transport;
      // four BF16 planes must never be interpreted as one FP32/scalar pair.
      const I hemispheres = text(src,"hemisphere") == "both" ? 2 : 1;
      const I columns = sourceShape[1], localColumns = columns/(2*hemispheres);
      auto ids = slices(src);
      for (I h = 0; h < hemispheres; ++h)
        for (I pair = 0; pair < 2; ++pair) {
          V selected, positions;
          for (size_t i = 0; i < ix.size(); ++i) {
            I col = ix[i]%columns;
            if ((col/word)%2 != pair || (hemispheres == 2 && (col/(2*word))%2 != h)) continue;
            selected.push_back((ix[i]/columns)*localColumns+
                (col/(2*word*hemispheres))*word+col%word);
            positions.push_back(pos[i]);
          }
          if (selected.empty()) continue;
          auto physical = s.change(src,J{{"slices",s.array({ids[2*pair],ids[2*pair+1]})},
              {"hemisphere",hemispheres == 2 ? (h ? "west" : "east") : text(src,"hemisphere")}});
          move(physical,dst,{sourceShape[0],localColumns},destShape,selected,positions);
        }
      return;
    }
    if (text(src, "kind") == "token_partition_planar") {
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
            {"hemisphere", side ? "east" : "west"}});
        move(physical, dst, {half, columns}, destShape, selected, positions);
      }
      return;
    }
    const auto distributed = [](mlir::DictionaryAttr p) {
      return text(p, "kind") == "fp16_mxm_distributed_16" ||
             text(p, "kind") == "fp16_vxm_distributed_16";
    };
    if (distributed(src) || distributed(dst)) {
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
      I columns = sourceShape[1], rows = sourceShape[0];
      bool paired = text(src, "kind") == "fp16_hemisphere_pair_planar";
      for (auto h : {"east", "west"})
        if (text(src, "hemisphere") == h || text(src, "hemisphere") == "both") {
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
  void materialize(const Tensor &value, mlir::DictionaryAttr dest) {
    std::map<Key, std::pair<V, V>> segments;
    for (size_t i = 0; i < value.refs.size(); ++i) {
      auto ref = value.refs[i];
      auto &p = segments[{ref.node, ref.output}];
      p.first.push_back(ref.index);
      p.second.push_back(i);
    }
    for (auto &kv : segments)
      move(cache.at(kv.first), dest,
           planner.nodes[kv.first.first].outputs[kv.first.second].shape,
           value.shape, kv.second.first, kv.second.second);
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
            allocate({-4, ni}, node.outputs[1], "east", 1, node.out[1]);
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
  void planMemory() {
    I joined = 0;
    for (I ni = 0; ni < I(planner.nodes.size()); ++ni) {
      auto &node = planner.nodes[ni];
      I first = ni;
      if (node.parallel >= 0)
        while (first > 0 && planner.nodes[first - 1].parallel == node.parallel)
          --first;
      for (auto it = live.begin(); it != live.end();)
        if (last.count(it->first) && last.at(it->first) < first)
          it = live.erase(it);
        else
          ++it;
      if (!node.module) {
        allocate({ni, 0}, node.outputs[0], joined++ % 2 ? "east" : "west");
        continue;
      }
      node.stagedInputs.assign(node.inputs.size(), false);
      if (node.recurrent) {
        node.in[5] = recurrentStates.at(ni);
        for (auto &slot : node.inputSlots[5])
          slot.set(node.in[5]);
      }
      // Exact compatible edges are true aliases. Layout changes and tensor
      // views remain explicit transports at the consumer, never an archive.
      for (I idx = 0; idx < I(node.inputs.size()); ++idx) {
        auto &value = node.inputs[idx];
        if (node.recurrent && idx == 5)
          continue;
        if (value.refs.empty())
          continue;
        auto firstRef = value.refs[0];
        // A grouped last-axis reduction may view columns from several native
        // distributed producers. Describe those columns before scheduling;
        // the L2 emitter transposes each group directly from its producer.
        if (idx == 0 && node.inputSlots[idx].size() == 1 &&
            llvm::isa<fc::tensor::L2NormTaskOp>(node.inputSlots[idx][0].op)) {
          const auto &firstShape = planner.nodes[firstRef.node].outputs[firstRef.output].shape;
          const I tokens = firstShape[0], width = value.shape[1];
          const I groups = tokens ? value.shape[0] / tokens : 0;
          bool compatible = tokens > 0 && tokens % 32 == 0 && width % 32 == 0 &&
              groups > 1 && value.shape[0] == tokens * groups;
          llvm::SmallVector<mlir::Attribute> sources;
          for (I g = 0; compatible && g < groups; ++g) {
            auto ref = value.refs[g * width];
            auto src = cache.at({ref.node, ref.output});
            const auto &shape = planner.nodes[ref.node].outputs[ref.output].shape;
            I column = ref.index;
            compatible = shape.size() == 2 && shape[0] == tokens &&
                column % 32 == 0 && column + width <= shape[1] &&
                text(src, "kind") == "fp16_mxm_distributed_16" &&
                !src.get("storage_shape") && slices(src).size() == 16;
            for (I t = 0; compatible && t < tokens; ++t)
              for (I f = 0; compatible && f < width; ++f) {
                auto r = value.refs[(t * groups + g) * width + f];
                compatible = r.node == ref.node && r.output == ref.output &&
                    r.index == t * shape[1] + column + f;
              }
            if (compatible) {
              mlir::NamedAttrList descriptor;
              descriptor.set("placement", src);
              descriptor.set("column", s.b.getI64IntegerAttr(column));
              descriptor.set("columns", s.b.getI64IntegerAttr(shape[1]));
              sources.push_back(descriptor.getDictionary(&s.context));
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
        if (text(firstPlacement, "kind") == "fp16_hemisphere_pair_planar"
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
        if (alias) continue;
        bool conflict = false;
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
        if (planRetainedRmsNorm(ni,oi) || planRetainedAttention(ni,oi))
          continue;
        auto busy = node.in;
        busy.insert(busy.end(), node.workspaces.begin(), node.workspaces.end());
        if (node.ffn) {
          // The FFN result is written by Down after Gate/Up have retired.
          // Protect Down's live operand queues, rather than every queue that
          // was used earlier anywhere inside the composite operator.
          busy.clear();
          auto down=llvm::cast<fc::tensor::MatmulTaskOp>(node.outputSlots[oi].op);
          for (auto array : {down.getLhsAllocations(),down.getRhsAllocations()})
            for (auto a : array)
              busy.push_back(llvm::cast<mlir::DictionaryAttr>(a).getAs<mlir::DictionaryAttr>("placement"));
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
            projection && projection.getKind() == "linear"
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
                eligible &= consumer.function && consumer.function->hasAttr("ftlpu.vector_token_partition")
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
            dest = allocate(key, node.outputs[oi], "west",
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
      if (!node.module)
        continue;
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
        if (!binding || binding.getPlacement() != node.out[oi])
          throw std::runtime_error(
              "native emitter did not honor graph output placement: " +
              node.name);
      }
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
      } else if (scalar(p) || text(p, "kind") == "fp16_pair_planar" ||
                 text(p, "kind") == "fp32_quad_planar" ||
                 text(p, "kind") == "fp16_mxm_distributed_16") {
        auto prior = input(a, p, node.name + ".input" + std::to_string(idx));
        if (prior.p != p)
          move(prior.p, p, a.shape, a.shape, range(a.size()), range(a.size()));
        prepared.insert({ni, idx});
      }
    }
  }
  void build() {
    fs::create_directories(directory);
    reserveRecurrentStates();
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
    s.maskPlacement = input(mask,
        s.placement("fp32_quad_planar", {48,49,50,51}, "east",1,0,4),
        "layout_tile_masks").p;
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
      if (!node.module)
        continue;
      bool initialRmsNorm = false;
      if (ni == 0)
        node.function.walk([&](fc::tensor::RmsNormTaskOp) {
          initialRmsNorm = true;
        });
      for (I i = 0; i < I(node.inputs.size()); ++i)
        if (node.inputs[i].external() && !(node.recurrent && i == 5))
          input(node.inputs[i], node.in[i],
                node.name + ".input" + std::to_string(i),
                (initialRmsNorm &&
                    (text(node.in[i], "kind") == "fp16_mxm_distributed_16" ||
                     text(node.in[i], "kind") == "fp16_vxm_gamma_broadcast"))
                    || (ni == 0
                        && text(node.in[i], "kind") == "fp16_mxm_distributed_16"),
                node.ffn ? I(planner.nodes.size()) : ni);
    }
    planMemory();
    std::set<Key> prepared;
    I parallelStart = 0, parallelEnd = 0;
    std::vector<I> parallelNodes;
    for (I ni = 0; ni < I(planner.nodes.size()); ++ni) {
      auto &node = planner.nodes[ni];
      llvm::outs() << "plan " << ni << " " << node.name << "\n";
      llvm::outs().flush();
      if (!node.module) {
        nodeRanges.push_back(
            J{{"node", ni}, {"name", node.name}, {"layout_join", true}});
        // Balance packed materializations between the two SXM fabrics. A
        // fixed West home concentrates every later unpack on one iMEM.
        auto p = cache.at({ni, 0});
        materialize(node.inputs[0], p);
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
             ++pi)
          prepare(pi, prepared);
        parallelStart = s.cursor + 64;
        parallelEnd = parallelStart;
        parallelNodes.clear();
      }
      mlir::IRMapping mapping;
      std::map<I, std::pair<mlir::DictionaryAttr, Static>> relocated;
      for (I idx = 0; idx < I(node.inputs.size()); ++idx) {
        auto &a = node.inputs[idx];
        auto p = node.in[idx];
        bool movable = scalar(p) || text(p, "kind") == "fp16_pair_planar" ||
                       text(p, "kind") == "fp16_mxm_activation_planar" ||
                       text(p, "kind") == "fp32_quad_planar" ||
                       text(p, "kind") == "fp16_mxm_distributed_16";
        if (a.external() && node.recurrent && idx == 5) {
          if (std::any_of(a.data.begin(), a.data.end(),
                          [](uint8_t x) { return x != 0; }))
            throw std::runtime_error(
                "nonzero external recurrent state is not supported yet");
          s.copy(zeroP, p, V(field(p, "instruction_count"), 0),
                 range(field(p, "instruction_count")));
          mapping.map(node.function.getArgument(idx),
                      s.binding(a, p, "internal", s.internal++,
                                "initial_recurrent_state"));
        } else if (a.external() && text(p,"kind") == "fp16_projection_bias_x4") {
          // Preserve the native bias read bank/queues. Relocating those reads
          // after scheduling can collide with the raw-Q FIFO on the other
          // bank; copy the immutable packed planes into their native home
          // before this attention batch starts instead.
          const auto prior=input(a,p,node.name+".input"+std::to_string(idx));
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
              if (!node.stagedInputs[idx])
                materialize(a, p);
            }
            else {
              auto prior =
                  input(a, p, node.name + ".input" + std::to_string(idx));
              if (prior.p != p)
                move(prior.p, p, a.shape, a.shape, range(a.size()),
                     range(a.size()));
            }
          }
          mapping.map(node.function.getArgument(idx),
                      s.binding(a, p, "internal", s.internal++,
                                node.name + ".operand"));
        } else {
          auto prior = input(a, p, node.name + ".input" + std::to_string(idx));
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
      if (pageRows) s.cursor += pageRows+32;
      I start = node.parallel >= 0 ? parallelStart : s.cursor+64, end = start;
      I nativeFirst = std::numeric_limits<I>::max(), nativeLast = 0;
      // This conservative graph schedule finishes the preceding operator
      // (including its SR drain) before admitting the next page. The page
      // rendezvous blocks every subsequent FU queue if DDR is late. Advertise
      // this exclusive ingress window, rather than borrowing an SR until its
      // final use in the entire layer, which can be millions of cycles later.
      for (auto &kv : relocated)
        if (field(kv.second.second.p, "page_rows"))
          s.emit("timeline",
                 J{{"name", "c2c.window.binding:" +
                                std::to_string(kv.second.second.index) +
                                ".page:0"},
                   {"start", ingressBegin},
                   {"end", start}});
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
          if (auto n = copy->getAttrOfType<mlir::StringAttr>("name"))
            copy->setAttr("name",
                          s.b.getStringAttr(node.name + "." + n.getValue()));
        }
        if (copy->hasAttr("ready_cycle"))
          s.set(copy, "ready_cycle", integer(copy, "ready_cycle") + start);
      }
      llvm::json::Array layouts;
      for (auto p : node.out)
        layouts.push_back(J{{"kind", text(p, "kind")},
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
                             {"name", node.name},
                             {"native_ffn",node.ffn},
                             {"stateful_mixer",node.recurrent},
                             {"weight_page_workspaces",std::move(pageWorkspaces)},
                             {"temporary_workspaces",std::move(temporaryWorkspaces)},
                             {"retained_output_layouts",std::move(retainedLayouts)},
                             {"nominal_start", nativeFirst},
                             {"nominal_end", nativeLast},
                             {"direct_output_layouts", std::move(layouts)}});
      if (node.parallel >= 0) {
        parallelEnd = std::max(parallelEnd, end);
        parallelNodes.push_back(ni);
        s.cursor = parallelEnd + 96;
      } else
        s.cursor = end + 96;
      for (I oi=0; oi<I(node.outputs.size()); ++oi)
        if (auto home = retainedOutputHomes.find({ni,oi}); home != retainedOutputHomes.end()) {
          const I begin = s.cursor;
          const auto &shape = node.outputs[oi].shape;
          move(node.out[oi], home->second, shape, shape,
               range(node.outputs[oi].size()), range(node.outputs[oi].size()));
          s.emit("timeline", J{{"name", node.name + ".retain_on_chip"},
                               {"start", begin}, {"end", s.cursor}});
        }
      for (auto it = live.begin(); it != live.end();)
        if (last[it->first] == ni)
          it = live.erase(it);
        else
          ++it;
    }
    for (I i = 0; i < I(finals.size()); ++i) {
      auto &value = finals[i];
      bool identity = !value.refs.empty();
      auto first = value.refs[0];
      for (I j = 0; j < I(value.refs.size()); ++j)
        identity &= value.refs[j].node == first.node &&
                    value.refs[j].output == first.output &&
                    value.refs[j].index == j;
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
          s.binding(value, p, "output", i, "output" + std::to_string(i));
          continue;
        }
      }
      // Export a linear tensor view from its actual storage geometry. Runtime
      // already unpacks native distributed matrices, so no FU copy is needed
      // merely to present a flat tensor to the caller.
      if (identity) {
        auto native = cache.at({first.node, first.output});
        const auto kind = text(native, "kind");
        const bool externalPlanar = scalar(native) ||
            (kind == "fp16_pair_planar" && slices(native).size() == 2) ||
            kind == "fp32_quad_planar" ||
            kind == "fp16_mxm_distributed_16" ||
            kind == "fp16_vxm_distributed_16";
        if (!externalPlanar)
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
      auto p = identity ? cache.at({first.node, first.output})
                        : allocate({-3, i}, value, "west", packedPieces ? 1 : -1, finalPrototype);
      if (!identity)
        materialize(value, p);
      Tensor exported = value;
      if (identity) {
        exported.shape = planner.nodes[first.node].outputs[first.output].shape;
        if (auto shape = p.getAs<mlir::ArrayAttr>("storage_shape")) {
          exported.shape.clear();
          for (auto d : shape) exported.shape.push_back(llvm::cast<mlir::IntegerAttr>(d).getInt());
        }
      }
      s.binding(exported, p, "output", i, "output" + std::to_string(i));
    }
    s.b.create<mlir::func::ReturnOp>(s.loc);
    (*s.module)->setAttr("ftlpu.graph_lowering",
                         s.b.getStringAttr("native_cpp"));
    std::string plan;
    llvm::raw_string_ostream planStream(plan);
    planStream << llvm::json::Value(
        J{{"planning", "tensor_subgraph_before_schedule"},
          {"native_output_archive_copies", I(retainedOutputHomes.size())},
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

mlir::OwningOpRef<mlir::ModuleOp> ftlpu::compiler::lower_connected_decoder_graph(
    mlir::ModuleOp module, const fs::path &directory) {
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
  planner.persistentState = !graph.delta_nets.empty();
  planner.hasAttention = !graph.attentions.empty();
  // A head normalization that views the Conv result can consume distributed16
  // directly. Other Conv graphs keep their established planar result layout.
  const auto fromConvolution = [&](const auto &self, mlir::Value value) -> bool {
    auto *producer = value.getDefiningOp();
    if (!producer) return false;
    if (llvm::isa<fc::kernel::DepthwiseConv1DOp>(producer)) return true;
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
  mlir::NamedAttrList policies;
  for (auto name : {"ftlpu.mxm_execution_policy", "ftlpu.projection_rope_overlap"})
    if (auto attr = module->getAttr(name)) policies.set(name,attr);
  planner.policies = policies.getDictionary(&context);
  if (auto strategy = module->getAttrOfType<mlir::StringAttr>("ftlpu.ffn_schedule"))
    planner.ffnStrategy = strategy.getValue() == "fused" ? fc::FfnScheduleStrategy::Fused : fc::FfnScheduleStrategy::Tail;
  llvm::DenseMap<mlir::Value, Tensor> values;
  auto describe = [&](mlir::Value value) {
    auto ty = llvm::cast<mlir::RankedTensorType>(value.getType()); Tensor t;
    t.shape.assign(ty.getShape().begin(), ty.getShape().end());
    t.dtype = ty.getElementType().isBF16() ? "bf16" : ty.getElementType().isF32() ? "f32" : "i8";
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
      if (src.external()) throw std::runtime_error("constant indexed views must be folded by importer");
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
        if (llvm::isa<fc::kernel::GatedDeltaRuleOp>(op) ||
            llvm::any_of(value.getUsers(),[&](auto *user){return !llvm::is_contained(ops,user);})) outputs.push_back(value);
    }
    // The recurrent lowerer tiles six operand roles. Preserve those roles
    // even when SSA aliases (for example Q == K) use the same input value.
    const bool recurrentRoles = ops.size() == 1 &&
        llvm::isa<fc::kernel::GatedDeltaRuleOp>(ops.front());
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
  Linker linker(planner,std::move(finals),directory); linker.build();
  (*linker.s.module)->setAttr("ftlpu.decoder_graph_matcher",mlir::StringAttr::get(&context,"shared"));
  std::ofstream(directory/"lowering_path.json") << "{\"entry\":\"kernel\",\"graph_matcher\":\"match_decoder_graph\",\"ffn_subgraphs\":"
      << graph.ffns.size() << ",\"deltanet_subgraphs\":" << graph.delta_nets.size()
      << ",\"memory_planning\":\"whole_graph_tensor_before_schedule\"}\n";
  return std::move(linker.s.module);
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
