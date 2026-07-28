#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/STLExtras.h"

#include <cmath>
#include <cstdint>
#include <limits>

using namespace mlir;

#include "ftlpu/compiler/Dialect/Tensor/IR/TensorOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ftlpu/compiler/Dialect/Tensor/IR/TensorOps.cpp.inc"

namespace ftlpu::compiler::tensor {

ParseResult MatmulOp::parse(OpAsmParser& parser, OperationState& result)
{
    SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
    const SMLoc operand_location = parser.getCurrentLocation();
    Type type;
    if (parser.parseOperandList(operands, 2) || parser.parseOptionalAttrDict(result.attributes)
        || parser.parseColonType(type))
        return failure();

    const auto function_type = llvm::dyn_cast<FunctionType>(type);
    if (!function_type || function_type.getNumInputs() != 2 || function_type.getNumResults() != 1)
        return parser.emitError(operand_location, "expected (lhs, rhs) -> result function type");
    if (parser.resolveOperands(operands, function_type.getInputs(), operand_location, result.operands))
        return failure();
    result.addTypes(function_type.getResults());
    return success();
}

void MatmulOp::print(OpAsmPrinter& printer)
{
    printer << " " << getLhs() << ", " << getRhs() << " {";
    printer.increaseIndent();
    const auto print_attribute = [&](StringRef name, Attribute attribute, bool trailing_comma) {
        printer.printNewline();
        printer << name << " = ";
        printer.printAttribute(attribute);
        if (trailing_comma) printer << ',';
    };
    print_attribute("m", getMAttr(), true);
    print_attribute("n", getNAttr(), true);
    print_attribute("k", getKAttr(), true);
    print_attribute("unit", getUnitAttr(), true);
    print_attribute("rhs_scale", getRhsScaleAttr(), true);
    print_attribute("lhs_address", getLhsAddressAttr(), true);
    print_attribute("lhs_placement", getLhsPlacementAttr(), true);
    print_attribute("lhs_bytes", getLhsBytesAttr(), true);
    print_attribute("rhs_address", getRhsAddressAttr(), true);
    print_attribute("rhs_placement", getRhsPlacementAttr(), true);
    print_attribute("rhs_bytes", getRhsBytesAttr(), true);
    print_attribute("result_address", getResultAddressAttr(), true);
    print_attribute("result_placement", getResultPlacementAttr(), true);
    print_attribute("result_bytes", getResultBytesAttr(), false);
    printer.decreaseIndent();
    printer.printNewline();
    printer << "} : ";
    printer.printFunctionalType(getOperandTypes(), getOperation()->getResultTypes());
}

void TensorDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "ftlpu/compiler/Dialect/Tensor/IR/TensorOps.cpp.inc"
        >();
}

static LogicalResult verify_address(Operation* op, DictionaryAttr address, StringRef name)
{
    for (StringRef field : {"device", "slice", "bank", "word", "byte"}) {
        if (!address.getAs<IntegerAttr>(field))
            return op->emitOpError() << name << " is missing integer field '" << field << "'";
    }
    const auto hemisphere = address.getAs<StringAttr>("hemisphere");
    if (!hemisphere || (hemisphere.getValue() != "east" && hemisphere.getValue() != "west"))
        return op->emitOpError() << name << " requires hemisphere 'east' or 'west'";
    const auto in_range = [&](StringRef field, int64_t minimum, int64_t maximum) {
        const int64_t value = address.getAs<IntegerAttr>(field).getInt();
        return value >= minimum && value <= maximum;
    };
    auto targetModel = target::LPUTargetModel::from_operation(op);
    if (mlir::failed(targetModel)) return failure();
    const auto& memory = targetModel->memory();
    if (!in_range("device", 0, 0)
        || !in_range("slice", 0, memory.slices_per_hemisphere - 1)
        || !in_range("bank", 0, memory.banks_per_slice - 1)
        || !in_range("word", 0, memory.words_per_bank - 1)
        || !in_range("byte", 0, memory.bytes_per_word - 1))
        return op->emitOpError() << name << " is outside the LPU MEM address space";
    return success();
}

static FailureOr<int64_t> get_tensor_bytes(RankedTensorType type)
{
    if (!type || !type.hasStaticShape()) return failure();
    int64_t bits = 0;
    if (auto integer = llvm::dyn_cast<IntegerType>(type.getElementType()))
        bits = integer.getWidth();
    else if (auto floating = llvm::dyn_cast<FloatType>(type.getElementType()))
        bits = floating.getWidth();
    if (bits <= 0 || bits % 8 != 0) return failure();

    int64_t bytes = bits / 8;
    for (int64_t dimension : type.getShape()) {
        if (dimension <= 0 || bytes > std::numeric_limits<int64_t>::max() / dimension)
            return failure();
        bytes *= dimension;
    }
    return bytes;
}

static LogicalResult verify_placement(Operation* op, DictionaryAttr placement,
    StringRef name, StringRef expected_kind, int64_t expected_slices)
{
    const auto kind = placement.getAs<StringAttr>("kind");
    const auto slices = placement.getAs<ArrayAttr>("slices");
    const auto base_row = placement.getAs<IntegerAttr>("base_row");
    const auto instruction_count = placement.getAs<IntegerAttr>("instruction_count");
    const auto address_stride = placement.getAs<IntegerAttr>("address_stride");
    if (!kind || kind.getValue() != expected_kind || !slices
        || static_cast<int64_t>(slices.size()) != expected_slices
        || !base_row || !instruction_count || !address_stride)
        return op->emitOpError() << name << " is not a valid " << expected_kind << " placement";
    auto targetModel = target::LPUTargetModel::from_operation(op);
    if (mlir::failed(targetModel)) return failure();
    for (Attribute attribute : slices) {
        const auto slice = llvm::dyn_cast<IntegerAttr>(attribute);
        if (!slice || slice.getInt() < 0
            || slice.getInt()
                >= targetModel->memory().slices_per_hemisphere)
            return op->emitOpError() << name << " contains an invalid MEM slice";
    }
    if (base_row.getInt() < 0 || instruction_count.getInt() <= 0
        || address_stride.getInt() == 0)
        return op->emitOpError() << name << " contains invalid row geometry";
    return success();
}

LogicalResult MatmulOp::verify()
{
    const auto lhs_type = getLhs().getType();
    const auto rhs_type = getRhs().getType();
    const auto result_type = getResult().getType();
    if (lhs_type.getRank() != 2 || rhs_type.getRank() != 2 || result_type.getRank() != 2)
        return emitOpError("requires rank-2 tensors");
    if (lhs_type.getDimSize(0) != getM() || lhs_type.getDimSize(1) != getK()
        || rhs_type.getDimSize(0) != getK() || rhs_type.getDimSize(1) != getN()
        || result_type.getDimSize(0) != getM() || result_type.getDimSize(1) != getN())
        return emitOpError("tensor shapes do not match m, n, and k");
    const auto lhs_bytes = get_tensor_bytes(lhs_type);
    const auto rhs_bytes = get_tensor_bytes(rhs_type);
    const auto result_bytes = get_tensor_bytes(result_type);
    if (failed(lhs_bytes) || failed(rhs_bytes) || failed(result_bytes)
        || *lhs_bytes != static_cast<int64_t>(getLhsBytes())
        || *rhs_bytes != static_cast<int64_t>(getRhsBytes())
        || *result_bytes != static_cast<int64_t>(getResultBytes()))
        return emitOpError("allocated byte sizes do not match tensor types");
    if (failed(verify_address(getOperation(), getLhsAddress(), "lhs_address"))
        || failed(verify_address(getOperation(), getRhsAddress(), "rhs_address"))
        || failed(verify_address(getOperation(), getResultAddress(), "result_address")))
        return failure();
    if (failed(verify_placement(getOperation(), getLhsPlacement(), "lhs_placement", "vector", 1))
        || failed(verify_placement(getOperation(), getRhsPlacement(), "rhs_placement", "mxm_weight_striped", 16))
        || failed(verify_placement(getOperation(), getResultPlacement(), "result_placement", "int32_byte_planar", 4)))
        return failure();
    if (getUnit() != "MXM") return emitOpError("unit must be MXM");
    return success();
}

static LogicalResult verify_task_allocations(
    Operation* op, ArrayAttr allocations, StringRef name)
{
    for (auto [index, attribute] : llvm::enumerate(allocations)) {
        const auto allocation = llvm::dyn_cast<DictionaryAttr>(attribute);
        if (!allocation)
            return op->emitOpError() << name << "[" << index
                                     << "] must be a dictionary";
        const auto address = allocation.getAs<DictionaryAttr>("address");
        const auto placement = allocation.getAs<DictionaryAttr>("placement");
        const auto bytes = allocation.getAs<IntegerAttr>("bytes");
        if (!address || !placement || !bytes || bytes.getInt() <= 0)
            return op->emitOpError()
                << name << "[" << index
                << "] requires address, placement, and positive bytes";
        if (failed(verify_address(op, address, name)))
            return failure();
        for (StringRef field :
            {"kind", "slices", "base_row", "instruction_count", "address_stride"})
            if (!placement.get(field))
                return op->emitOpError()
                    << name << "[" << index
                    << "] placement is missing field '" << field << "'";
    }
    return success();
}

LogicalResult MatmulTaskOp::verify()
{
    const auto lhs = getLhs().getType();
    const auto rhs = getRhs().getType();
    const auto result = getResult().getType();
    if (lhs.getRank() != 2 || rhs.getRank() != 2 || result.getRank() != 2)
        return emitOpError("requires rank-2 tensors");
    if (lhs.getDimSize(0) != getM() || lhs.getDimSize(1) != getK()
        || rhs.getDimSize(0) != getK() || rhs.getDimSize(1) != getN()
        || result.getDimSize(0) != getM() || result.getDimSize(1) != getN())
        return emitOpError("tensor shapes do not match m, n, and k");
    if (failed(verify_task_allocations(
            getOperation(), getLhsAllocations(), "lhs_allocations"))
        || failed(verify_task_allocations(
            getOperation(), getRhsAllocations(), "rhs_allocations"))
        || failed(verify_task_allocations(
            getOperation(), getResultAllocations(), "result_allocations")))
        return failure();
    return success();
}

LogicalResult SwishTaskOp::verify()
{
    if (getInput().getType().getShape() != getResult().getType().getShape())
        return emitOpError("input and result shapes must match");
    return verify_task_allocations(
        getOperation(), getResultAllocations(), "result_allocations");
}

LogicalResult ElementwiseTaskOp::verify()
{
    if (getKind() != "multiply" && getKind() != "add"
        && getKind() != "add_quant")
        return emitOpError(
            "kind must be multiply, add, or add_quant");
    if (getLhs().getType().getShape() != getRhs().getType().getShape()
        || getLhs().getType().getShape() != getResult().getType().getShape())
        return emitOpError("operand and result shapes must match");
    if (!getLhsAllocations().empty()
        && failed(verify_task_allocations(
            getOperation(), getLhsAllocations(), "lhs_allocations")))
        return failure();
    if (!getRhsAllocations().empty()
        && failed(verify_task_allocations(
            getOperation(), getRhsAllocations(), "rhs_allocations")))
        return failure();
    return verify_task_allocations(
        getOperation(), getResultAllocations(), "result_allocations");
}

LogicalResult RmsNormTaskOp::verify()
{
    const auto input = getInput().getType();
    const auto weight = getWeight().getType();
    const auto result = getResult().getType();
    const int64_t rawAxis = getAxisAttr().getInt();
    const int64_t axis =
        rawAxis < 0 ? rawAxis + input.getRank() : rawAxis;
    if (!input.hasStaticShape() || !weight.hasStaticShape()
        || input.getRank() != 2 || weight.getRank() != 1
        || result != input || axis != input.getRank() - 1
        || weight.getDimSize(0) != input.getDimSize(axis)
        || weight.getElementType() != input.getElementType())
        return emitOpError("has incompatible RMSNorm tensor types");
    if (!std::isfinite(getEpsilon().convertToFloat())
        || getEpsilon().convertToFloat() <= 0.0f)
        return emitOpError("requires a finite positive epsilon");
    const auto strategy = getConfig().getAs<mlir::StringAttr>("strategy");
    if (!strategy
        || (strategy.getValue() != "vxm_square_mxm_reduce"
            && strategy.getValue() != "vxm_feedback"))
        return emitOpError("requires a supported RMSNorm strategy");
    const std::size_t expectedScratch =
        strategy.getValue() == "vxm_feedback" ? 3 : 2;
    if (getScratchAllocations().size() != expectedScratch)
        return emitOpError("scratch allocation count does not match strategy");
    if (failed(verify_task_allocations(
            getOperation(), getInputAllocations(), "input_allocations"))
        || failed(verify_task_allocations(
            getOperation(), getWeightAllocations(), "weight_allocations"))
        || failed(verify_task_allocations(
            getOperation(), getScratchAllocations(), "scratch_allocations"))
        || failed(verify_task_allocations(
            getOperation(), getResultAllocations(), "result_allocations")))
        return failure();
    return success();
}

LogicalResult SwigluOp::verify()
{
    const auto input = getInput().getType();
    const auto gate = getGateWeight().getType();
    const auto up = getUpWeight().getType();
    const auto result = getResult().getType();
    if (input.getRank() != 2 || gate.getRank() != 2 || up.getRank() != 2
        || result.getRank() != 2 || input.getDimSize(0) != getM()
        || input.getDimSize(1) != getK() || gate.getDimSize(0) != getK()
        || gate.getDimSize(1) != getN() || up.getShape() != gate.getShape()
        || result.getDimSize(0) != getM() || result.getDimSize(1) != getN())
        return emitOpError("tensor shapes do not match the SwiGLU dimensions");
    const auto input_bytes = get_tensor_bytes(input);
    const auto gate_bytes = get_tensor_bytes(gate);
    const auto up_bytes = get_tensor_bytes(up);
    const auto result_bytes = get_tensor_bytes(result);
    if (failed(input_bytes) || failed(gate_bytes) || failed(up_bytes) || failed(result_bytes)
        || *input_bytes != static_cast<int64_t>(getInputBytes())
        || *gate_bytes != static_cast<int64_t>(getGateWeightBytes())
        || *up_bytes != static_cast<int64_t>(getUpWeightBytes())
        || *result_bytes != static_cast<int64_t>(getResultBytes()))
        return emitOpError("allocated byte sizes do not match tensor types");
    if (failed(verify_address(getOperation(), getInputAddress(), "input_address"))
        || failed(verify_address(getOperation(), getGateWeightAddress(), "gate_weight_address"))
        || failed(verify_address(getOperation(), getUpWeightAddress(), "up_weight_address"))
        || failed(verify_address(getOperation(), getResultAddress(), "result_address"))
        || failed(verify_placement(getOperation(), getInputPlacement(), "input_placement", "vector", 1))
        || failed(verify_placement(getOperation(), getGateWeightPlacement(), "gate_weight_placement", "mxm_weight_striped", 16))
        || failed(verify_placement(getOperation(), getUpWeightPlacement(), "up_weight_placement", "mxm_weight_striped", 16))
        || failed(verify_placement(getOperation(), getResultPlacement(), "result_placement", "vector", 1)))
        return failure();
    return success();
}

static LogicalResult verify_attention_task_config(
    Operation* operation, DictionaryAttr config)
{
    const auto seqLen = config.getAs<IntegerAttr>("seq_len");
    const auto hidden = config.getAs<IntegerAttr>("hidden");
    const auto queryHeads = config.getAs<IntegerAttr>("query_heads");
    const auto kvHeads = config.getAs<IntegerAttr>("kv_heads");
    const auto headDim = config.getAs<IntegerAttr>("head_dim");
    const auto ropeTheta = config.getAs<FloatAttr>("rope_theta");
    const auto causal = config.getAs<BoolAttr>("causal");
    if (!seqLen || !hidden || !queryHeads || !kvHeads || !headDim
        || !ropeTheta || !causal || seqLen.getInt() <= 0
        || hidden.getInt() <= 0 || queryHeads.getInt() <= 0
        || kvHeads.getInt() <= 0 || headDim.getInt() <= 0
        || queryHeads.getInt() % kvHeads.getInt() != 0)
        return operation->emitOpError(
            "requires a complete grouped-query attention config");
    return success();
}

static LogicalResult verify_attention_memory_subplan(
    Operation* operation, DictionaryAttr memoryPlan)
{
    for (NamedAttribute entry : memoryPlan) {
        const auto placement = llvm::dyn_cast<DictionaryAttr>(entry.getValue());
        const auto kind =
            placement ? placement.getAs<StringAttr>("kind") : StringAttr {};
        const auto slices =
            placement ? placement.getAs<ArrayAttr>("slices") : ArrayAttr {};
        if (!kind || !slices
            || failed(verify_placement(operation, placement,
                entry.getName().strref(), kind.getValue(),
                static_cast<int64_t>(slices.size()))))
            return failure();
    }
    return success();
}

LogicalResult ProjectionTaskOp::verify()
{
    if (getKind() != "query" && getKind() != "key"
        && getKind() != "value" && getKind() != "output")
        return emitOpError("kind must be query, key, value, or output");
    if (getInput().getType().getRank() != 2
        || getWeight().getType().getRank() != 2
        || getResult().getType().getRank() != 2)
        return emitOpError("requires rank-2 input, weight, and result tensors");
    if (failed(verify_attention_task_config(getOperation(), getConfig())))
        return failure();
    return verify_attention_memory_subplan(
        getOperation(), getMemoryPlan());
}

LogicalResult RopeTaskOp::verify()
{
    if (getKind() != "query" && getKind() != "key")
        return emitOpError("kind must be query or key");
    if (getInput().getType() != getResult().getType())
        return emitOpError("must preserve its tensor type");
    if (failed(verify_attention_task_config(getOperation(), getConfig())))
        return failure();
    return verify_attention_memory_subplan(
        getOperation(), getMemoryPlan());
}

LogicalResult BatchMatmulTaskOp::verify()
{
    if (getKind() != "qk" && getKind() != "pv")
        return emitOpError("kind must be qk or pv");
    if (failed(verify_attention_task_config(getOperation(), getConfig())))
        return failure();
    return verify_attention_memory_subplan(
        getOperation(), getMemoryPlan());
}

LogicalResult SoftmaxTaskOp::verify()
{
    if (getInput().getType() != getResult().getType())
        return emitOpError("must preserve its tensor type");
    if (failed(verify_attention_task_config(getOperation(), getConfig())))
        return failure();
    return verify_attention_memory_subplan(
        getOperation(), getMemoryPlan());
}

LogicalResult TransposeTaskOp::verify()
{
    if (getKind() != "probability" && getKind() != "value")
        return emitOpError("kind must be probability or value");
    if (failed(verify_attention_task_config(getOperation(), getConfig())))
        return failure();
    return verify_attention_memory_subplan(
        getOperation(), getMemoryPlan());
}

} // namespace ftlpu::compiler::tensor
