#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "DirectDomainEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

namespace ftlpu::compiler::schedule {
namespace {

using ffn_detail::create_vxm;

mlir::DictionaryAttr placement(mlir::ArrayAttr allocations, int64_t index = 0)
{
    return llvm::cast<mlir::DictionaryAttr>(allocations[index])
        .getAs<mlir::DictionaryAttr>("placement");
}

int64_t firstSlice(mlir::DictionaryAttr value)
{
    return llvm::cast<mlir::IntegerAttr>(
        value.getAs<mlir::ArrayAttr>("slices")[0]).getInt();
}

int64_t sliceAt(mlir::DictionaryAttr value, int64_t index)
{
    const auto slices = value.getAs<mlir::ArrayAttr>("slices");
    return llvm::cast<mlir::IntegerAttr>(
        slices[static_cast<std::size_t>(index)]).getInt();
}

int64_t baseRow(mlir::DictionaryAttr value)
{
    return value.getAs<mlir::IntegerAttr>("base_row").getInt();
}

int64_t bank(mlir::DictionaryAttr value)
{
    const auto attr = value.getAs<mlir::IntegerAttr>("bank");
    return attr ? attr.getInt() : 0;
}

bool isWest(mlir::DictionaryAttr value)
{
    const auto hemisphere = value.getAs<mlir::StringAttr>("hemisphere");
    return hemisphere && hemisphere.getValue() == "west";
}

BindingOp outputBinding(mlir::IRRewriter& rewriter,
    stream::DepthwiseConv1DTaskOp op, int64_t index,
    mlir::DictionaryAttr resultPlacement)
{
    const auto type = llvm::cast<mlir::RankedTensorType>(
        op.getResult().getType());
    mlir::OperationState state(op.getLoc(), BindingOp::getOperationName());
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr("index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr("access", rewriter.getStringAttr("output")),
        rewriter.getNamedAttr("role", rewriter.getStringAttr("result")),
        rewriter.getNamedAttr("name",
            rewriter.getStringAttr("depthwise_conv1d.result")),
        rewriter.getNamedAttr("bytes", rewriter.getI64IntegerAttr(
            type.getNumElements() * 2)),
        rewriter.getNamedAttr("placement", resultPlacement),
    });
    return llvm::cast<BindingOp>(rewriter.create(state));
}

void inputBinding(mlir::IRRewriter& rewriter,
    stream::DepthwiseConv1DTaskOp op, mlir::Value value,
    llvm::StringRef role, mlir::DictionaryAttr valuePlacement)
{
    const auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
    if (!argument) return;
    const auto type = llvm::cast<mlir::RankedTensorType>(value.getType());
    mlir::OperationState state(op.getLoc(), BindingOp::getOperationName());
    state.addOperands(value);
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr("index", rewriter.getI64IntegerAttr(
            argument.getArgNumber())),
        rewriter.getNamedAttr("access", rewriter.getStringAttr("input")),
        rewriter.getNamedAttr("role", rewriter.getStringAttr(role)),
        rewriter.getNamedAttr("name", rewriter.getStringAttr(
            role == "weight" ? "depthwise_conv1d.weight"
                             : "depthwise_conv1d.input")),
        rewriter.getNamedAttr("bytes", rewriter.getI64IntegerAttr(
            type.getNumElements() * 2)),
        rewriter.getNamedAttr("placement", valuePlacement),
    });
    rewriter.create(state);
}

void emitTimeline(mlir::IRRewriter& rewriter,
    stream::DepthwiseConv1DTaskOp op, int64_t start, int64_t end)
{
    mlir::OperationState state(op.getLoc(), TimelineOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("name",
            rewriter.getStringAttr("depthwise_conv1d")),
        rewriter.getNamedAttr("start", rewriter.getI64IntegerAttr(start)),
        rewriter.getNamedAttr("end", rewriter.getI64IntegerAttr(end)),
    });
    rewriter.create(state);
}

mlir::LogicalResult lowerOne(mlir::IRRewriter& rewriter,
    stream::DepthwiseConv1DTaskOp op,
    const target::LPUTargetModel& target, int64_t outputIndex)
{
    const auto type = llvm::cast<mlir::RankedTensorType>(
        op.getResult().getType());
    if (op.getKernelSize() != 4 || !op.getCausal()
        || type.getRank() != 2
        || !is_lpu_16bit_float(type.getElementType()))
        return op.emitError(
            "direct depthwise Conv1D schedule requires causal kernel-4 "
            "16-bit [tokens, channels] tensors");

    const int64_t tokens = type.getDimSize(0);
    const int64_t channels = type.getDimSize(1);
    const int64_t lanes = target.throughput().lanes_per_tile;
    const int64_t channelsPerWord = target.memory().bytes_per_word;
    if (tokens <= 0 || channels <= 0 || channels % channelsPerWord != 0)
        return op.emitError(
            "channels must be aligned to one MEM byte plane (32 channels)");

    const auto input = placement(op.getInputAllocations());
    const auto weight = placement(op.getWeightAllocations());
    const auto paddedInput = placement(op.getScratchAllocations(), 0);
    const auto scratch0 = placement(op.getScratchAllocations(), 1);
    const auto scratch1 = placement(op.getScratchAllocations(), 2);
    const auto result = placement(op.getResultAllocations());
    const int64_t groups = channels / lanes;
    const int64_t featureBlocks = channels / channelsPerWord;
    const int64_t tileCycles = channelsPerWord
        / target.throughput().mem_read_bytes_per_cycle;
    const int64_t elements = tokens * groups;
    const int64_t elementsPerTile = tokens * featureBlocks;
    const int64_t inStream = op.getInputStreamBase();
    const int64_t weightStream = op.getWeightStreamBase();
    const int64_t outStream = op.getOutputStreamBase();
    const auto streamKind = lpu_16bit_stream_kind(type.getElementType());
    const auto dataFormat = lpu_16bit_data_format(type.getElementType());
    // Both MEM fabrics feed the VXM through their reverse-flow column-zero
    // boundary.  The VXM operand source selects which physical fabric is
    // sampled; it does not change the direction used inside that fabric.
    constexpr auto inputDirection = target::StreamDirection::West;
    const auto vxmInputHemisphere =
        [&](mlir::DictionaryAttr p) -> llvm::StringRef {
        return isWest(p) ? "west" : "east";
    };
    const auto vxmOutputHemisphere =
        [&](mlir::DictionaryAttr p) -> llvm::StringRef {
        return isWest(p) ? "east" : "west";
    };
    const auto physicalQueue = [&](mlir::DictionaryAttr p, int64_t slice) {
        return slice + (isWest(p) ? target.memory().slices_per_hemisphere : 0);
    };
    const auto readLatency = [&](mlir::DictionaryAttr p, int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,
            inputDirection, slice).value();
    };
    const auto writeLatency = [&](mlir::DictionaryAttr p, int64_t slice) {
        static_cast<void>(p);
        return target.transport_latency(target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice).value();
    };
    constexpr int64_t westIngressDelay = 0;
    const auto eastDrainDelay = [&](int64_t queue) {
        return target.throughput().vxm_alus - queue;
    };
    const auto outputDirectionDelay = [&](mlir::DictionaryAttr destination) {
        return isWest(destination) ? int64_t {0} : int64_t {2};
    };
    const auto plane = [&](mlir::DictionaryAttr p, int64_t pair,
                           int64_t cycle, int64_t stream,
                           llvm::StringRef opcode, int64_t addressOffset,
                           int64_t rowCount, int64_t rowsPerPlane) {
        for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = sliceAt(p, 2 * pair + byte);
            const int64_t issue = opcode == "read"
                ? cycle - readLatency(p, slice) - westIngressDelay
                : cycle + writeLatency(p, slice);
            const int64_t packedStream = opcode == "read"
                ? target.streams().streams_per_direction + stream + byte
                : stream + byte;
            direct_domain_detail::emitMem3D(rewriter, op.getLoc(), target,
                issue, physicalQueue(p, slice), opcode,
                baseRow(p) + addressOffset,
                packedStream, rowCount, tileCycles, 1,
                featureBlocks, rowCount * tileCycles, rowsPerPlane,
                1, 1, 0, -1, bank(p));
        }
    };
    const auto read = [&](mlir::DictionaryAttr p, int64_t cycle,
                          int64_t stream, int64_t addressOffset,
                          int64_t rowsPerPlane) {
        plane(p, 0, cycle, stream, "read", addressOffset,
            tokens, rowsPerPlane);
    };
    const auto write = [&](mlir::DictionaryAttr p, int64_t cycle,
                           int64_t stream) {
        const auto kind = p.getAs<mlir::StringAttr>("kind");
        if (kind && kind.getValue() == "fp16_mxm_distributed_16") {
            // Conv emits feature-block-major words, with token as the inner
            // axis. Route token lane r%8 straight to its distributed16 pair;
            // the four token waves occupy consecutive rows of that pair.
            for (int64_t pair = 0; pair < lanes; ++pair)
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = sliceAt(p, 2 * pair + byte);
                    direct_domain_detail::emitMem3D(rewriter, op.getLoc(), target,
                        cycle + pair * tileCycles + writeLatency(p, slice),
                        physicalQueue(p, slice), "write", baseRow(p),
                        stream + byte, target.throughput().tile_rows,
                        lanes * tileCycles, 1, featureBlocks,
                        tokens * tileCycles, target.throughput().tile_rows,
                        1, 1, 0, -1, bank(p));
                }
            return;
        }
        plane(p, 0, cycle, stream, "write", 0, tokens, tokens);
    };
    const auto readWeight = [&](int64_t cycle, int64_t stream,
                                int64_t tap) {
        for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = sliceAt(weight, 2 * tap + byte);
            direct_domain_detail::emitMem3D(rewriter, op.getLoc(), target,
                cycle - readLatency(weight, slice) - westIngressDelay,
                physicalQueue(weight, slice), "read",
                baseRow(weight) + tap * featureBlocks,
                target.streams().streams_per_direction + stream + byte,
                tokens, tileCycles, 0,
                featureBlocks, tokens * tileCycles, 1,
                1, 1, 0, -1, bank(weight));
        }
    };

    auto* precedingOperation = op->getPrevNode();
    rewriter.setInsertionPoint(op);
    inputBinding(rewriter, op, op.getInput(), "activation", input);
    inputBinding(rewriter, op, op.getWeight(), "weight", weight);
    const int64_t start = 16;
    const int64_t transportGuard =
        2 * target.streams().system_register_columns + 32;
    int64_t cycle = start;
    mlir::Value current = op.getInput();

    // Build the causal window once in SRAM.  Three leading rows are emitted
    // by a VXM immediate-zero domain; the real input follows at row +3.  The
    // four convolution taps can then use ordinary non-negative affine reads.
    auto zeroHead = create_vxm(rewriter, op.getLoc(), op.getInput(),
        op.getInput(), type, cycle - 1, 0, "add",
        "immediate", 0, 0.0f, "immediate", 0, 0.0f,
        "fp32", -1, 3 * groups, 1,
        vxmInputHemisphere(input), vxmOutputHemisphere(paddedInput), -1,
        false, false, true, false, 2);
    auto zero = create_vxm(rewriter, op.getLoc(), zeroHead.getResult(),
        op.getInput(), type, cycle, 1, "cast",
        "previous", 0, 0.0f, "immediate", 0, 0.0f,
        dataFormat, outStream, 3 * groups, 1,
        vxmInputHemisphere(input), vxmOutputHemisphere(paddedInput), -1,
        false, false, true, false, 2);
    for (int64_t tap = 0; tap < 4; ++tap)
        plane(paddedInput, tap, cycle + 1 + eastDrainDelay(1), outStream,
            tap == 3 ? "write" : "write_tap", 0, 3, tokens + 3);
    cycle += 3 * groups + transportGuard;
    // Preserve the input bytes while replicating them to the four causal
    // windows. The two fabrics are connected by the passive VXM SR chain;
    // no ALU decode/re-encode is needed for a same-format copy. Read and
    // write use a separate stream pair from the preceding zero-fill chain.
    constexpr int64_t copyStream = 20;
    const int64_t memGroup = target.streams().mem_slices_per_register_group;
    for (int64_t byte = 0; byte < 2; ++byte) {
        const int64_t sourceSlice = sliceAt(input, byte);
        const int64_t sourceGroup = sourceSlice / memGroup;
        const int64_t issue = cycle - readLatency(input, sourceSlice);
        direct_domain_detail::emitMem3D(rewriter, op.getLoc(), target,
            issue, physicalQueue(input, sourceSlice), "read", baseRow(input),
            target.streams().streams_per_direction + copyStream + byte,
            tokens, tileCycles, 1, featureBlocks, tokens * tileCycles,
            tokens, 1, 1, 0, -1, bank(input));
        for (int64_t tap = 0; tap < 4; ++tap) {
            const int64_t destinationSlice = sliceAt(paddedInput, 2 * tap + byte);
            const int64_t destinationGroup = destinationSlice / memGroup;
            const int64_t latency = sourceGroup + target.throughput().vxm_alus
                + destinationGroup + 3;
            direct_domain_detail::emitMem3D(rewriter, op.getLoc(), target,
                issue + latency, physicalQueue(paddedInput, destinationSlice),
                tap == 3 ? "write" : "write_tap", baseRow(paddedInput) + 3,
                copyStream + byte, tokens, tileCycles, 1,
                featureBlocks, tokens * tileCycles, tokens + 3,
                1, 1, 0, -1, bank(paddedInput));
        }
    }
    cycle += elements + transportGuard;

    // All four taps stream concurrently from independent padded slice pairs.
    // Increasing VALU columns form a spatial multiply/reduction tree; FP32
    // partials stay in ordinary SRs until the one final BF16 cast.
    constexpr int64_t tapColumns[] = {0, 1, 3, 4};
    const int64_t activationStreams[] = {inStream, inStream + 2, inStream + 4, inStream + 6};
    const int64_t weightStreams[] = {weightStream, weightStream + 2, weightStream + 4, weightStream + 6};
    for (int64_t tap = 0; tap < 4; ++tap) {
        // Raw West operands reach later columns earlier. Delay their ingress
        // by twice the column offset to align products in the East tree.
        const int64_t phase = cycle + 2 * tapColumns[tap];
        plane(paddedInput, tap, phase, activationStreams[tap], "read", tap,
            tokens, tokens + 3);
        readWeight(phase, weightStreams[tap], tap);
    }
    const auto column = [&](int64_t queue, llvm::StringRef opcode,
                            llvm::StringRef lhsKind, int64_t lhsIndex,
                            llvm::StringRef rhsKind, int64_t rhsIndex,
                            llvm::StringRef format = "fp32",
                            int64_t output = -1) {
        return create_vxm(rewriter, op.getLoc(), current, op.getWeight(),
            type, cycle - 1, queue, opcode, lhsKind, lhsIndex, 0.0f,
            rhsKind, rhsIndex, 0.0f, format, output,
            elementsPerTile, 1, "west", "east", -1,
            false, false, true, false, 8);
    };
    column(0, "multiply", streamKind, activationStreams[0], streamKind, weightStreams[0]);
    column(1, "multiply", streamKind, activationStreams[1], streamKind, weightStreams[1]);
    column(2, "add", "alu", 0, "alu", 1);
    column(3, "multiply", streamKind, activationStreams[2], streamKind, weightStreams[2]);
    column(4, "multiply", streamKind, activationStreams[3], streamKind, weightStreams[3]);
    column(5, "add", "alu", 3, "alu", 4);
    column(6, "add", "alu", 2, "alu", 5);
    // Opposite-chain raw operands cross this terminal column in SR0..15.
    // SR16's earlier FP32 carriers have been consumed before column 7, so
    // the terminal BF16 result can reuse it while draining to MEM.
    constexpr int64_t convolutionOutputStream = 16;
    auto convolution = column(7, "cast", "previous", 0,
        "immediate", 0, dataFormat, convolutionOutputStream);
    current = convolution.getResult();
    // West ingress to q0 (16), q0-to-q7 arithmetic (8), and the
    // q7-to-East-boundary drain (9) precede the destination MEM latency.
    const int64_t convolutionResultCycle = cycle
        + target.throughput().vxm_alus + 8 + eastDrainDelay(7);
    // Tap the same result into the primary buffer, then consume it into the
    // preserved-operand buffer. This eliminates a separate MEM/VXM copy pass.
    plane(op.getActivation() == "silu" ? scratch0 : result, 0,
            convolutionResultCycle, convolutionOutputStream,
        op.getActivation() == "silu" ? "write_tap" : "write", 0,
        tokens, tokens);
    if (op.getActivation() == "silu")
        write(scratch1, convolutionResultCycle, convolutionOutputStream);
    cycle += elements + transportGuard;

    if (op.getActivation() == "silu") {
        read(scratch0, cycle, inStream, 0, tokens);
        const int64_t preservedInputPhase = isWest(scratch0) ? 16 : 8;
        // SR28..31 carry the reciprocal FP32 result from q3. Keep the
        // preserved BF16 operand in the raw-input range so it cannot collide.
        constexpr int64_t preservedInputStream = 8;
        read(scratch1, cycle + preservedInputPhase,
            preservedInputStream, 0, tokens);
        const int64_t siluConfigCycle = cycle - 1;
        // From the input consumed one cycle after configuration: NEG(1),
        // EXP(5), ADD(1), RECIP(5), MUL(2), CAST(1), then ten passive
        // columns to the east VXM boundary. Removing the two ALU bypasses
        // replaces them with passive SR hops, retaining the boundary phase.
        const int64_t siluResultLatency = 27
            + (isWest(scratch0) ? target.throughput().vxm_alus - 1 : 0);
        auto neg = create_vxm(rewriter, op.getLoc(), current, current, type,
            siluConfigCycle, 0, "negate", streamKind, inStream, 0.0f,
            "immediate", 0, 0.0f, "fp32", -1, elementsPerTile, 1,
            vxmInputHemisphere(scratch0), vxmOutputHemisphere(result), -1,
            false, false, true, false, 8);
        auto exp = create_vxm(rewriter, op.getLoc(), neg.getResult(), current,
            type, siluConfigCycle, 1, "exp", "previous", 0, 0.0f,
            "immediate", 0, 0.0f, "fp32", -1, elementsPerTile, 1,
            vxmInputHemisphere(scratch0), vxmOutputHemisphere(result), -1,
            false, false, true, false, 8);
        auto onePlus = create_vxm(rewriter, op.getLoc(), exp.getResult(),
            current, type, siluConfigCycle, 2, "add", "previous", 0, 0.0f,
            "immediate", 0, 1.0f, "fp32", -1, elementsPerTile, 1,
            vxmInputHemisphere(scratch0), vxmOutputHemisphere(result), -1,
            false, false, true, false, 8);
        auto reciprocal = create_vxm(rewriter, op.getLoc(),
            onePlus.getResult(), current, type, siluConfigCycle, 3,
            "reciprocal", "previous", 0, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, elementsPerTile, 1,
            vxmInputHemisphere(scratch0), vxmOutputHemisphere(result),
            -1, false, false, true, false, 8);
        auto activated = create_vxm(rewriter, op.getLoc(),
            reciprocal.getResult(), current, type, siluConfigCycle, 4,
            "multiply", "previous", 0, 0.0f,
            streamKind, preservedInputStream, 0.0f,
            "fp32", -1, elementsPerTile, 1,
            vxmInputHemisphere(scratch0), vxmOutputHemisphere(result),
            -1, false, false, true, false, 8);
        constexpr int64_t siluOutputStream = 6;
        auto cast = create_vxm(rewriter, op.getLoc(), activated.getResult(),
            current, type, siluConfigCycle, 5, "cast",
            "previous", 0, 0.0f, "immediate", 0, 0.0f,
            dataFormat, siluOutputStream, elementsPerTile, 1,
            vxmInputHemisphere(scratch0), vxmOutputHemisphere(result),
            -1, false, false, true, false, 8);
        current = cast.getResult();
        write(result, siluConfigCycle + siluResultLatency,
            siluOutputStream);
    }
    // External operands may arrive from either fabric. The generated chains
    // always run through increasing VALU columns, so their internal FP32
    // carriers must flow east regardless of the external operand direction.
    auto* generated = precedingOperation ? precedingOperation->getNextNode()
                                         : &op->getBlock()->front();
    for (; generated != op.getOperation(); generated = generated->getNextNode()) {
        auto vxm = llvm::dyn_cast<VxmOp>(generated);
        if (!vxm) continue;
        const auto externalDirection = vxm.getInputHemisphere();
        if (vxm.getLhsKind().starts_with("stream_"))
            vxm->setAttr("lhs_stream_source",
                rewriter.getStringAttr(externalDirection));
        if (vxm.getRhsKind().starts_with("stream_"))
            vxm->setAttr("rhs_stream_source",
                rewriter.getStringAttr(externalDirection));
        vxm->setAttr("input_hemisphere", rewriter.getStringAttr("east"));
    }
    cycle += elements + 16;
    emitTimeline(rewriter, op, start, cycle);
    auto binding = outputBinding(rewriter, op, outputIndex, result);
    const auto function = op->getParentOfType<mlir::func::FuncOp>();
    const auto mirror = function->getAttrOfType<mlir::BoolAttr>(
        "ftlpu.depthwise_conv_mirror");
    if (mirror && mirror.getValue()) {
        // Reflect the closed-form hardware domains, including the operand
        // references. No FU trace expansion or model-specific operation.
        const auto flip = [&](llvm::StringRef value) {
            return rewriter.getStringAttr(value == "east" ? "west" : "east");
        };
        auto* generated = precedingOperation ? precedingOperation->getNextNode()
                                             : &op->getBlock()->front();
        for (; generated != op.getOperation(); generated = generated->getNextNode()) {
            if (auto mem = llvm::dyn_cast<MemTransferOp>(generated)) {
                mem->setAttr("hemisphere", rewriter.getI64IntegerAttr(1 - mem.getHemisphere()));
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
                        vxm->setAttr(kindName, rewriter.getStringAttr("stream_f32"));
                        vxm->setAttr(indexName, rewriter.getI64IntegerAttr(
                            16 + 4 * (reflected % 4)));
                        vxm->setAttr(sourceName, rewriter.getStringAttr("west"));
                    }
                }
                if (vxm.getOutputStreamAttr().getInt() < 0) {
                    vxm->setAttr("output_stream", rewriter.getI64IntegerAttr(
                        16 + 4 * ((target.throughput().vxm_alus - 1 - queue) % 4)));
                }
                vxm->setAttr("queue", rewriter.getI64IntegerAttr(target.throughput().vxm_alus - 1 - queue));
                vxm->setAttr("input_hemisphere", flip(vxm.getInputHemisphere()));
                vxm->setAttr("output_hemisphere", flip(vxm.getOutputHemisphere()));
            } else if (auto bind = llvm::dyn_cast<BindingOp>(generated)) {
                mlir::NamedAttrList attributes(bind.getPlacement());
                attributes.set("hemisphere", flip(bind.getPlacement()
                    .getAs<mlir::StringAttr>("hemisphere").getValue()));
                bind->setAttr("placement", attributes.getDictionary(rewriter.getContext()));
            }
        }
    }
    rewriter.replaceOp(op, binding.getValue());
    return mlir::success();
}

} // namespace

mlir::LogicalResult lowerDepthwiseConv1DSchedules(
    mlir::IRRewriter& rewriter, mlir::func::FuncOp function,
    const target::LPUTargetModel& target)
{
    llvm::SmallVector<stream::DepthwiseConv1DTaskOp> operations;
    function.walk([&](stream::DepthwiseConv1DTaskOp op) {
        operations.push_back(op);
    });
    int64_t outputIndex = 0;
    for (auto op : operations)
        if (mlir::failed(lowerOne(rewriter, op, target, outputIndex++)))
            return mlir::failure();
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
