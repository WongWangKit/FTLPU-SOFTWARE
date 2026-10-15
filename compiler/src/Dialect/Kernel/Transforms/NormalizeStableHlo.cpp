#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Value.h"

#include <functional>

namespace ftlpu::compiler {
namespace {

bool named(mlir::Operation* op, llvm::StringRef name)
{
    return op && op->getName().getStringRef() == name;
}

mlir::Operation* unary(mlir::IRRewriter& rewriter, llvm::StringRef name,
    mlir::Location loc, mlir::Value input, mlir::Type result,
    llvm::ArrayRef<mlir::NamedAttribute> attributes = {})
{
    mlir::OperationState state(loc, name);
    state.addOperands(input);
    state.addTypes(result);
    state.addAttributes(attributes);
    return rewriter.create(state);
}

bool static_tensor(mlir::Type type)
{
    auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type);
    return ranked && ranked.hasStaticShape() && !ranked.getEncoding();
}

bool fold_constant_mask(mlir::Operation* op, mlir::IRRewriter& rewriter,
    bool& evaluation_failed)
{
    const auto name = op->getName().getStringRef();
    const bool layout_or_selection = name == "stablehlo.reshape"
        || name == "stablehlo.transpose" || name == "stablehlo.broadcast_in_dim"
        || name == "stablehlo.concatenate" || name == "stablehlo.slice"
        || name == "stablehlo.gather" || name == "stablehlo.select"
        || name == "stablehlo.convert";
    const bool integer_compute = name == "stablehlo.iota"
        || name == "stablehlo.compare" || name == "stablehlo.and"
        || name == "stablehlo.or" || name == "stablehlo.not"
        || name == "stablehlo.add" || name == "stablehlo.subtract"
        || name == "stablehlo.multiply" || name == "stablehlo.reduce_window"
        || name == "stablehlo.reduce";
    if ((!layout_or_selection && !integer_compute) || op->getNumResults() != 1)
        return false;
    auto type = llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!static_tensor(type) || type.getNumElements() <= 0 || type.getNumElements() > 65536)
        return false;
    if (integer_compute) {
        if (!llvm::isa<mlir::IntegerType>(type.getElementType())) return false;
        for (auto value : op->getOperands()) {
            auto operand_type = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
            if (!operand_type || !llvm::isa<mlir::IntegerType>(operand_type.getElementType()))
                return false;
        }
        bool integer_region = true;
        op->walk([&](mlir::Operation* nested) {
            if (nested == op) return;
            if (nested->getName().getStringRef() == "stablehlo.return") return;
            if (nested->getName().getStringRef() != "stablehlo.add") integer_region = false;
            for (auto result : nested->getResultTypes()) {
                auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(result);
                integer_region &= ranked && llvm::isa<mlir::IntegerType>(ranked.getElementType());
            }
        });
        if (!integer_region) return false;
    }
    llvm::SmallVector<mlir::DenseElementsAttr> inputs;
    for (auto operand : op->getOperands()) {
        auto constant = operand.getDefiningOp<mlir::stablehlo::ConstantOp>();
        if (!constant) return false;
        auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
        if (!value || value.getNumElements() > 65536) return false;
        inputs.push_back(value);
    }
    // Evaluate the actual op and attributes with the upstream reference
    // interpreter. No local implementation of gather/window/mask mathematics.
    auto module = mlir::ModuleOp::create(op->getLoc());
    mlir::OwningOpRef<mlir::ModuleOp> owner(module);
    mlir::OpBuilder builder(op->getContext());
    builder.setInsertionPointToStart(module.getBody());
    auto function = builder.create<mlir::func::FuncOp>(op->getLoc(), "main",
        builder.getFunctionType(op->getOperandTypes(), op->getResultTypes()));
    auto* block = function.addEntryBlock();
    builder.setInsertionPointToStart(block);
    mlir::IRMapping mapping;
    mapping.map(op->getOperands(), block->getArguments());
    auto* copied = builder.clone(*op, mapping);
    builder.create<mlir::func::ReturnOp>(op->getLoc(), copied->getResults());
    // This cloned region already contains only a validated StableHLO op;
    // use the upstream region interpreter without module dialect-conversion
    // passes (and their unrelated linalg dependency chain).
    llvm::SmallVector<mlir::stablehlo::InterpreterValue> arguments;
    for (auto input : inputs)
        arguments.emplace_back(mlir::stablehlo::makeTensor(input));
    auto evaluated = mlir::stablehlo::eval(function.getBody(), arguments);
    if (evaluated.size() != 1 || evaluated[0].getType() != type) {
        op->emitError("static mask evaluator changed the result signature");
        evaluation_failed = true;
        return false;
    }
    rewriter.setInsertionPoint(op);
    auto replacement = rewriter.create<mlir::stablehlo::ConstantOp>(op->getLoc(),
        mlir::stablehlo::makeDenseElementsAttr(evaluated[0].getTensor()));
    rewriter.replaceOp(op, replacement.getResult());
    return true;
}

bool normalize_pointwise_view(mlir::Operation* op, mlir::IRRewriter& rewriter)
{
    if (!named(op, "stablehlo.reshape")) return false;
    auto pointwise = op->getOperand(0).getDefiningOp();
    if (!pointwise || pointwise->getNumResults() != 1) return false;
    const auto name = pointwise->getName().getStringRef();
    if (name != "stablehlo.add" && name != "stablehlo.multiply"
        && name != "stablehlo.convert")
        return false;
    auto type = llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!static_tensor(type) || type.getRank() != 2) return false;
    llvm::SmallVector<mlir::Value> operands;
    for (auto operand : pointwise->getOperands()) {
        auto view = operand.getDefiningOp();
        if (!named(view, "stablehlo.reshape")) return false;
        auto original = view->getOperand(0);
        auto original_type = llvm::dyn_cast<mlir::RankedTensorType>(original.getType());
        if (!static_tensor(original_type) || original_type.getShape() != type.getShape())
            return false;
        operands.push_back(original);
    }
    rewriter.setInsertionPoint(op);
    mlir::OperationState state(op->getLoc(), name);
    state.addOperands(operands);
    state.addTypes(type);
    state.addAttributes(pointwise->getAttrs());
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return true;
}

// All rules below change indexing only. They preserve every arithmetic
// operation, dtype, dot precision/algorithm, and reduction iteration order.
bool normalize_layout(mlir::Operation* op, mlir::IRRewriter& rewriter)
{
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return false;
    const auto name = op->getName().getStringRef();
    if (name != "stablehlo.reshape" && name != "stablehlo.transpose"
        && name != "stablehlo.broadcast_in_dim")
        return false;
    if (!static_tensor(op->getOperand(0).getType())
        || !static_tensor(op->getResult(0).getType()))
        return false;
    auto input = op->getOperand(0);
    auto result = op->getResult(0);
    auto inner = input.getDefiningOp();
    if (name == "stablehlo.reshape") {
        if (input.getType() == result.getType()) {
            rewriter.replaceOp(op, input);
            return true;
        }
        if (!named(inner, name)
            || !static_tensor(inner->getOperand(0).getType()))
            return false;
        rewriter.setInsertionPoint(op);
        auto replacement = unary(rewriter, name, op->getLoc(),
            inner->getOperand(0), result.getType());
        rewriter.replaceOp(op, replacement->getResults());
        return true;
    }
    auto mapping = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
        name == "stablehlo.transpose" ? "permutation" : "broadcast_dimensions");
    if (!mapping) return false;
    bool identity = input.getType() == result.getType();
    for (auto [index, axis] : llvm::enumerate(mapping.asArrayRef()))
        identity &= axis == static_cast<int64_t>(index);
    if (identity) {
        rewriter.replaceOp(op, input);
        return true;
    }
    if (!named(inner, name)
        || !static_tensor(inner->getOperand(0).getType()))
        return false;
    auto inner_mapping = inner->getAttrOfType<mlir::DenseI64ArrayAttr>(
        name == "stablehlo.transpose" ? "permutation" : "broadcast_dimensions");
    if (!inner_mapping) return false;
    llvm::SmallVector<int64_t> composed;
    if (name == "stablehlo.transpose") {
        for (int64_t axis : mapping.asArrayRef())
            composed.push_back(inner_mapping[axis]);
    } else {
        for (int64_t axis : inner_mapping.asArrayRef())
            composed.push_back(mapping[axis]);
    }
    rewriter.setInsertionPoint(op);
    auto replacement = unary(rewriter, name, op->getLoc(),
        inner->getOperand(0), result.getType(),
        {rewriter.getNamedAttr(name == "stablehlo.transpose"
                ? "permutation" : "broadcast_dimensions",
            rewriter.getDenseI64ArrayAttr(composed))});
    rewriter.replaceOp(op, replacement->getResults());
    return true;
}

bool normalize_dot(mlir::Operation* op, mlir::IRRewriter& rewriter)
{
    auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
    if (!dot) return false;
    auto lhs = llvm::dyn_cast<mlir::RankedTensorType>(dot.getLhs().getType());
    auto rhs = llvm::dyn_cast<mlir::RankedTensorType>(dot.getRhs().getType());
    auto result = llvm::dyn_cast<mlir::RankedTensorType>(dot.getType());
    if (!static_tensor(lhs) || !static_tensor(rhs) || !static_tensor(result))
        return false;
    const auto dims = dot.getDotDimensionNumbers();
    auto lc = dims.getLhsContractingDimensions();
    auto rc = dims.getRhsContractingDimensions();
    auto lb = dims.getLhsBatchingDimensions();
    auto rb = dims.getRhsBatchingDimensions();
    if (lc.size() != 1 || rc.size() != 1) return false;
    const bool singleton = lhs.getRank() == 3 && rhs.getRank() == 3
        && result.getRank() == 3 && lb == llvm::ArrayRef<int64_t>({0})
        && rb == llvm::ArrayRef<int64_t>({0})
        && lhs.getDimSize(0) == 1 && rhs.getDimSize(0) == 1
        && result.getDimSize(0) == 1 && lc[0] != 0 && rc[0] != 0;
    const bool rank2 = lhs.getRank() == 2 && rhs.getRank() == 2
        && result.getRank() == 2 && lb.empty() && rb.empty();
    if (!singleton && !rank2) return false;
    int64_t lhs_contract = lc[0] - (singleton ? 1 : 0);
    int64_t rhs_contract = rc[0] - (singleton ? 1 : 0);
    if (lhs_contract < 0 || lhs_contract > 1
        || rhs_contract < 0 || rhs_contract > 1)
        return false;
    if (rank2 && lhs_contract == 1 && rhs_contract == 0) return false;

    rewriter.setInsertionPoint(op);
    mlir::Value left = dot.getLhs(), right = dot.getRhs();
    auto squeeze = [&](mlir::Value value, mlir::RankedTensorType type) {
        const auto squeezed = mlir::RankedTensorType::get(
            type.getShape().drop_front(), type.getElementType());
        return unary(rewriter, "stablehlo.reshape", op->getLoc(), value,
            squeezed)->getResult(0);
    };
    if (singleton) {
        left = squeeze(left, lhs);
        right = squeeze(right, rhs);
    }
    auto transpose = [&](mlir::Value value) {
        auto type = llvm::cast<mlir::RankedTensorType>(value.getType());
        const auto transposed = mlir::RankedTensorType::get(
            {type.getDimSize(1), type.getDimSize(0)}, type.getElementType());
        return unary(rewriter, "stablehlo.transpose", op->getLoc(), value,
            transposed, {rewriter.getNamedAttr("permutation",
                rewriter.getDenseI64ArrayAttr({1, 0}))})->getResult(0);
    };
    if (lhs_contract == 0) left = transpose(left);
    if (rhs_contract == 1) right = transpose(right);
    auto normalized_result = singleton
        ? mlir::RankedTensorType::get(result.getShape().drop_front(),
            result.getElementType())
        : result;
    mlir::OperationState state(op->getLoc(), "stablehlo.dot_general");
    state.addOperands({left, right});
    state.addTypes(normalized_result);
    state.addAttributes(op->getAttrs());
    state.attributes.set("dot_dimension_numbers",
        mlir::stablehlo::DotDimensionNumbersAttr::get(
            op->getContext(), {}, {}, {1}, {0}));
    auto replacement = rewriter.create(state)->getResult(0);
    if (singleton)
        replacement = unary(rewriter, "stablehlo.reshape", op->getLoc(),
            replacement, result)->getResult(0);
    rewriter.replaceOp(op, replacement);
    return true;
}

class NormalizeStableHloPass final
    : public mlir::PassWrapper<NormalizeStableHloPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeStableHloPass)
    NormalizeStableHloPass() = default;
    explicit NormalizeStableHloPass(bool fold_static_mask) : fold_static_mask_(fold_static_mask) {}
    llvm::StringRef getArgument() const final { return "ftlpu-normalize-stablehlo"; }
    llvm::StringRef getDescription() const final
    {
        return "Canonicalize layouts and optionally evaluate static masks without FP reassociation";
    }
    void runOnOperation() final
    {
        mlir::IRRewriter rewriter(&getContext());
        int64_t rewrites = 0;
        int64_t mask_folds = 0;
        bool changed;
        do {
            changed = false;
            llvm::SmallVector<mlir::Operation*> operations;
            for (auto& block : getOperation().getBody())
                for (auto& op : block) operations.push_back(&op);
            llvm::DenseSet<mlir::Operation*> mask_closure;
            std::function<void(mlir::Value)> remember = [&](mlir::Value value) {
                auto* defining = value.getDefiningOp();
                if (!defining || !mask_closure.insert(defining).second) return;
                for (auto operand : defining->getOperands()) remember(operand);
            };
            if (fold_static_mask_)
                for (auto* op : operations)
                    if (named(op, "stablehlo.select")) {
                        mask_closure.insert(op);
                        for (auto operand : op->getOperands()) remember(operand);
                    }
            for (auto* op : llvm::reverse(operations)) {
                // Tensor views can be conservatively retained by upstream CSE
                // across the pinned dialect/interface revisions. These known
                // functional ops have no state effects; remove only dead values.
                if (op->use_empty()
                    && (named(op, "stablehlo.reshape") || named(op, "stablehlo.transpose")
                        || named(op, "stablehlo.broadcast_in_dim") || named(op, "stablehlo.constant")
                        || named(op, "stablehlo.add") || named(op, "stablehlo.multiply")
                        || named(op, "stablehlo.convert"))) {
                    rewriter.eraseOp(op);
                    ++rewrites;
                    changed = true;
                    continue;
                }
                bool evaluation_failed = false;
                if (fold_static_mask_ && mask_closure.contains(op)
                    && fold_constant_mask(op, rewriter, evaluation_failed)) {
                    ++mask_folds;
                    changed = true;
                    continue;
                }
                if (evaluation_failed) {
                    signalPassFailure();
                    return;
                }
                if (normalize_pointwise_view(op, rewriter)
                    || normalize_layout(op, rewriter) || normalize_dot(op, rewriter)) {
                    ++rewrites;
                    changed = true;
                }
            }
        } while (changed);
        getOperation()->setAttr("ftlpu.stablehlo_normalization",
            rewriter.getDictionaryAttr({
                rewriter.getNamedAttr("layout_rewrites", rewriter.getI64IntegerAttr(rewrites)),
                rewriter.getNamedAttr("static_mask_folds", rewriter.getI64IntegerAttr(mask_folds)),
                rewriter.getNamedAttr("arithmetic_reassociation", rewriter.getBoolAttr(false)),
                rewriter.getNamedAttr("abi_changed", rewriter.getBoolAttr(false))}));
    }
private:
    bool fold_static_mask_ = false;
};

} // namespace

std::unique_ptr<mlir::Pass> create_normalize_stablehlo_pass(bool fold_static_mask)
{
    return std::make_unique<NormalizeStableHloPass>(fold_static_mask);
}

} // namespace ftlpu::compiler
