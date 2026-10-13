#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "DirectDomainEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>

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
                cycle - readLatency(p, slice),
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
        constexpr int64_t outputStream = 0;
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
        const int64_t arithmeticDelay = opcode == "multiply" ? 1 : 0;
        auto head = create_vxm(rewriter_, op_.getLoc(), op_.getQuery(),
            op_.getKey(), op_.getOutput().getType(), input - 1, queueBase_, opcode,
            lhsKind, lhsStream, 0.0f, rhsKind, rhsStream, 0.0f,
            "fp32", -1, count, 1, inputHemisphere(lhs),
            outputHemisphere(destination), -1,
            false, false, true, false, 2);
        gridVxm(head);
        auto tail = create_vxm(rewriter_, op_.getLoc(), head.getResult(), op_.getKey(),
            op_.getOutput().getType(), input + arithmeticDelay, queueBase_ + 1, "cast",
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
        constexpr int64_t outputStream = 0;
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
            op_.getKey(), op_.getFinalState().getType(), input + 1, queueBase_ + 1,
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
            input, queueBase_ + 1, "cast", "previous", 0, 0.0f,
            "immediate", 0, 0.0f, "fp32", stream, count, 1,
            inputHemisphere(source), outputHemisphere(destination), -1,
            false, false, true, false, 2);
        writeRows(destination, destinationAddress, stream, 4, rows,
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
