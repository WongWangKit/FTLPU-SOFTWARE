#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"

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
    const target::LPUTargetModel target;
    const auto& memory = target.memory();
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
    const target::LPUTargetModel target;
    for (Attribute attribute : slices) {
        const auto slice = llvm::dyn_cast<IntegerAttr>(attribute);
        if (!slice || slice.getInt() < 0
            || slice.getInt() >= target.memory().slices_per_hemisphere)
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

LogicalResult FfnOp::verify()
{
    const auto input = getInput().getType();
    const auto gate = getGateWeight().getType();
    const auto up = getUpWeight().getType();
    const auto down = getDownWeight().getType();
    const auto result = getResult().getType();
    if (input.getDimSize(0) != static_cast<int64_t>(getM())
        || input.getDimSize(1) != static_cast<int64_t>(getK())
        || gate.getDimSize(0) != static_cast<int64_t>(getK())
        || gate.getDimSize(1) != static_cast<int64_t>(getHidden())
        || up.getShape() != gate.getShape()
        || down.getDimSize(0) != static_cast<int64_t>(getHidden())
        || down.getDimSize(1) != static_cast<int64_t>(getN())
        || result.getDimSize(0) != static_cast<int64_t>(getM())
        || result.getDimSize(1) != static_cast<int64_t>(getN()))
        return emitOpError("tensor shapes do not match the FFN dimensions");
    const auto input_bytes = get_tensor_bytes(input);
    const auto gate_bytes = get_tensor_bytes(gate);
    const auto up_bytes = get_tensor_bytes(up);
    const auto down_bytes = get_tensor_bytes(down);
    const auto result_bytes = get_tensor_bytes(result);
    if (failed(input_bytes) || failed(gate_bytes) || failed(up_bytes)
        || failed(down_bytes) || failed(result_bytes)
        || *input_bytes != static_cast<int64_t>(getInputBytes())
        || *gate_bytes != static_cast<int64_t>(getGateWeightBytes())
        || *up_bytes != static_cast<int64_t>(getUpWeightBytes())
        || *down_bytes != static_cast<int64_t>(getDownWeightBytes())
        || *result_bytes != static_cast<int64_t>(getResultBytes()))
        return emitOpError("allocated byte sizes do not match FFN tensors");
    for (auto [address, name] : {
             std::pair{getInputAddress(), StringRef("input_address")},
             std::pair{getGateWeightAddress(), StringRef("gate_weight_address")},
             std::pair{getUpWeightAddress(), StringRef("up_weight_address")},
             std::pair{getDownWeightAddress(), StringRef("down_weight_address")},
             std::pair{getHidden0Address(), StringRef("hidden0_address")},
             std::pair{getHidden1Address(), StringRef("hidden1_address")},
             std::pair{getResultAddress(), StringRef("result_address")}})
        if (failed(verify_address(getOperation(), address, name))) return failure();
    const target::LPUTargetModel target;
    const bool w8a16 = input.getElementType().isF16()
        && gate.getElementType().isInteger(8)
        && up.getElementType().isInteger(8)
        && down.getElementType().isInteger(8)
        && result.getElementType().isF16()
        && target.supports_w8a16_ffn_shape(getM(), getK(), getHidden(), getN());
    const int64_t expected_hidden_pass_bytes = w8a16
        ? getM() * (getHidden() / target.memory().hemispheres) * 2
        : getM() * 320;
    if (getHiddenPassBytes() != expected_hidden_pass_bytes)
        return emitOpError("hidden pass allocation does not match the FFN execution profile");
    if (w8a16) {
        if (failed(verify_placement(getOperation(), getInputPlacement(), "input_placement", "fp16_mxm_activation_planar", 4))
            || failed(verify_placement(getOperation(), getGateWeightPlacement(), "gate_weight_placement", "w8a16_mxm_weight_striped", 8))
            || failed(verify_placement(getOperation(), getUpWeightPlacement(), "up_weight_placement", "w8a16_mxm_weight_striped", 8))
            || failed(verify_placement(getOperation(), getDownWeightPlacement(), "down_weight_placement", "w8a16_mxm_weight_striped", 8))
            || failed(verify_placement(getOperation(), getHidden0Placement(), "hidden0_placement", "fp16_mxm_activation_planar", 4))
            || failed(verify_placement(getOperation(), getHidden1Placement(), "hidden1_placement", "fp16_mxm_activation_planar", 4))
            || failed(verify_placement(getOperation(), getResultPlacement(), "result_placement", "fp16_pair_planar", 4)))
            return failure();
        return success();
    }
    if (failed(verify_placement(getOperation(), getInputPlacement(), "input_placement", "vector", 1))
        || failed(verify_placement(getOperation(), getGateWeightPlacement(), "gate_weight_placement", "mxm_weight_striped", 16))
        || failed(verify_placement(getOperation(), getUpWeightPlacement(), "up_weight_placement", "mxm_weight_striped", 16))
        || failed(verify_placement(getOperation(), getDownWeightPlacement(), "down_weight_placement", "mxm_weight_striped", 16))
        || failed(verify_placement(getOperation(), getHidden0Placement(), "hidden0_placement", "vector", 1))
        || failed(verify_placement(getOperation(), getHidden1Placement(), "hidden1_placement", "vector", 1))
        || failed(verify_placement(getOperation(), getResultPlacement(), "result_placement", "vector", 1)))
        return failure();
    return success();
}

LogicalResult AttentionOp::verify()
{
    const auto input = getInput().getType();
    const auto query = getQueryWeight().getType();
    const auto key = getKeyWeight().getType();
    const auto value = getValueWeight().getType();
    const auto output = getOutputWeight().getType();
    const auto result = getResult().getType();
    const int64_t query_width = getQueryHeads() * getHeadDim();
    const int64_t kv_width = getKvHeads() * getHeadDim();
    if (input.getRank() != 2 || query.getRank() != 2 || key.getRank() != 2
        || value.getRank() != 2 || output.getRank() != 2 || result.getRank() != 2
        || input.getDimSize(0) != getSeqLen() || input.getDimSize(1) != getHidden()
        || query.getDimSize(0) != getHidden() || query.getDimSize(1) != query_width
        || key.getDimSize(0) != getHidden() || key.getDimSize(1) != kv_width
        || value.getShape() != key.getShape()
        || output.getDimSize(0) != query_width || output.getDimSize(1) != getHidden()
        || result.getShape() != input.getShape())
        return emitOpError("tensor shapes do not match the attention configuration");
    for (StringRef name : {"input", "query_weight", "key_weight", "value_weight",
             "output_weight", "query", "key", "value", "score", "probability",
             "probability_pack", "probability_diagonal", "exp", "rope", "context", "result"}) {
        const auto placement = getMemoryPlan().getAs<DictionaryAttr>(name);
        const auto kind = placement ? placement.getAs<StringAttr>("kind") : StringAttr {};
        const auto slices = placement ? placement.getAs<ArrayAttr>("slices") : ArrayAttr {};
        if (!kind || !slices || failed(verify_placement(getOperation(), placement, name,
                kind.getValue(), static_cast<int64_t>(slices.size()))))
            return failure();
    }
    return success();
}

} // namespace ftlpu::compiler::tensor
