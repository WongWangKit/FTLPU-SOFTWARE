#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;
namespace {

int64_t elementTypeBytes(mlir::Type type)
{
    if (type.isInteger(8)) return 1;
    if (is_lpu_16bit_float(type)) return 2;
    if (type.isF32()) return 4;
    return 0;
}

int64_t argumentIndex(mlir::Value value)
{
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
        return argument.getArgNumber();
    return -1;
}

llvm::SmallVector<int64_t> slices(mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    const auto values =
        placement.getAs<mlir::ArrayAttr>("slices");
    if (!values) return result;
    for (mlir::Attribute value : values)
        result.push_back(
            llvm::cast<mlir::IntegerAttr>(value).getInt());
    return result;
}

BindingOp createBinding(mlir::IRRewriter& rewriter,
    mlir::Location location, mlir::ValueRange source, int64_t index,
    llvm::StringRef access, llvm::StringRef role,
    mlir::RankedTensorType type, mlir::DictionaryAttr placement)
{
    mlir::OperationState state(
        location, BindingOp::getOperationName());
    state.addOperands(source);
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr(
            "index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr(
            "access", rewriter.getStringAttr(access)),
        rewriter.getNamedAttr(
            "role", rewriter.getStringAttr(role)),
        rewriter.getNamedAttr("bytes",
            rewriter.getI64IntegerAttr(type.getNumElements()
                * elementTypeBytes(type.getElementType()))),
        rewriter.getNamedAttr("placement", placement),
    });
    return llvm::cast<BindingOp>(rewriter.create(state));
}

mlir::FailureOr<mlir::Value> emitLinearProjection(
    mlir::IRRewriter& rewriter, stream::ProjectionTaskOp op,
    const target::LPUTargetModel& target, int64_t outputIndex)
{
    const auto memoryPlan = *op.getMemoryPlan();
    const auto inputPlacement =
        memoryPlan.getAs<mlir::DictionaryAttr>("input");
    const auto weightPlacement =
        memoryPlan.getAs<mlir::DictionaryAttr>("weight");
    const auto resultPlacement =
        memoryPlan.getAs<mlir::DictionaryAttr>("result");
    const auto mAttr = op.getConfig().getAs<mlir::IntegerAttr>("m");
    const auto nAttr = op.getConfig().getAs<mlir::IntegerAttr>("n");
    const auto kAttr = op.getConfig().getAs<mlir::IntegerAttr>("k");
    const auto scaleAttr =
        op.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
    if (!inputPlacement || !weightPlacement || !resultPlacement
        || !mAttr || !nAttr || !kAttr || !scaleAttr) {
        op.emitError("linear projection has an incomplete physical plan");
        return mlir::failure();
    }
    const auto inputSlices = slices(inputPlacement);
    const auto weightSlices = slices(weightPlacement);
    const auto resultSlices = slices(resultPlacement);
    const auto inputType =
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    const auto weightType =
        llvm::cast<mlir::RankedTensorType>(op.getWeight().getType());
    const auto resultType =
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
    const int64_t m = mAttr.getInt();
    const int64_t n = nAttr.getInt();
    const int64_t k = kAttr.getInt();
    auto executionPolicy =
        target::mxm_execution_policy_from_operation(op.getOperation());
    if (mlir::failed(executionPolicy)) {
        op.emitError("invalid MXM execution policy");
        return mlir::failure();
    }
    auto execution = target::plan_mxm_execution_strategy(
        {m, n, k, inputType.getElementType().isBF16(),
            weightType.getElementType().isInteger(8),
            is_lpu_16bit_float(resultType.getElementType()), true},
        target, *executionPolicy);
    if (mlir::failed(execution)) {
        op.emitError("cannot select an MXM execution strategy");
        return mlir::failure();
    }
    const bool localDequant = execution->uses_local_dequant();
    const std::size_t expectedActivationSlices = 2;
    const std::size_t expectedResultSlices = 4;
    if (inputSlices.size() != expectedActivationSlices
        || weightSlices.size() != 8
        || resultSlices.size() != expectedResultSlices) {
        op.emitError(
            "linear projection physical layout does not match the "
            "selected MXM execution strategy");
        return mlir::failure();
    }

    const int64_t tile = target.throughput().mxm_rows;
    if (m % tile || n % (2 * tile) || k % tile) {
        op.emitError("linear projection dimensions are not MXM aligned");
        return mlir::failure();
    }
    const int64_t inputBase =
        inputPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t weightBase =
        weightPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t resultBase =
        resultPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t inputBinding = argumentIndex(op.getInput());
    const int64_t weightBinding = argumentIndex(op.getWeight());
    if (inputBinding < 0 || weightBinding < 0) {
        op.emitError(
            "linear projection activation and weight must be function arguments");
        return mlir::failure();
    }

    rewriter.setInsertionPoint(op);
    createBinding(rewriter, op.getLoc(), op.getInput(), inputBinding,
        "input", "activation",
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType()),
        inputPlacement);
    createBinding(rewriter, op.getLoc(), op.getWeight(), weightBinding,
        "input", "weight",
        llvm::cast<mlir::RankedTensorType>(op.getWeight().getType()),
        weightPlacement);

    const llvm::StringRef dataFormat = lpu_16bit_data_format(
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType())
            .getElementType());
    const auto readLatency = [&](int64_t slice) {
        return slice
                / target.streams().mem_slices_per_register_group
            + 2;
    };
    const auto weightReadLatency = [&](int64_t slice) {
        if (!localDequant) return readLatency(slice);
        const auto latency = target.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, slice);
        return latency.value_or(-1);
    };
    int64_t maxWeightLatency = 0;
    for (int64_t slice : weightSlices) {
        if (weightReadLatency(slice) < 0)
            return mlir::failure();
        maxWeightLatency =
            std::max(maxWeightLatency, weightReadLatency(slice));
    }
    const int64_t weightToIw =
        target.throughput().vxm_weight_to_iw_latency;
    const int64_t vectorWeightLoadLead =
        (target.memory().hemispheres - 1)
            * target.throughput().lanes_per_tile
        + 3 + weightToIw + 1;
    const int64_t tokenBlocks = m / tile;
    const int64_t reductionBlocks = k / tile;
    const int64_t outputGroups = n / (2 * tile);
    const int64_t localMxm = 0;
    const int64_t accumulatorAddress =
        target.throughput().mxm_accumulator_blocks
            * target.throughput().mxm_rows
        - m;
    const int64_t accumulatorLatency =
        target.throughput().mxm0_accumulator_latency;
    const float scale =
        static_cast<float>(scaleAttr.getValueAsDouble());

    int64_t phaseStart = 0;
    for (int64_t outputGroup = 0;
         outputGroup < outputGroups; ++outputGroup) {
        for (int64_t reduction = 0;
             reduction < reductionBlocks; ++reduction) {
            const int64_t dequantStart =
                phaseStart + maxWeightLatency;
            const int64_t firstCompute = localDequant
                ? dequantStart
                    + target.throughput()
                          .mxm_local_load_to_compute_latency
                : dequantStart + vectorWeightLoadLead;
            const int64_t weightBuffer =
                (outputGroup * reductionBlocks + reduction)
                % target.throughput().mxm_weight_buffers;
            for (int64_t hemisphere = 0;
                 hemisphere < target.memory().hemispheres;
                 ++hemisphere) {
                const char* hemi =
                    hemisphere == 0 ? "east" : "west";
                for (int64_t pulse = 0; pulse < 4; ++pulse) {
                    const int64_t cycle = dequantStart
                        + (localDequant
                                ? 0
                                : hemisphere
                                    * target.throughput()
                                          .lanes_per_tile)
                        + pulse;
                    const int64_t address = weightBase
                        + (outputGroup * reductionBlocks + reduction)
                            * 4
                        + pulse;
                    for (int64_t stream = 0; stream < 8; ++stream) {
                        const int64_t slice = weightSlices[stream];
                        emitMem(rewriter, op.getLoc(),
                            cycle - weightReadLatency(slice),
                            hemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + slice,
                            "read", address,
                            localDequant
                                ? localMxm
                                        * target.throughput()
                                              .mxm_int8_load_streams_per_cycle
                                    + stream
                                : target.streams()
                                      .streams_per_direction
                                    + stream,
                            1, 1, 0, "sram", weightBinding);
                    }
                    const int64_t unit = hemisphere
                            * target.throughput()
                                  .mxms_per_hemisphere
                        + localMxm;
                    if (localDequant) {
                        emitMxmDequant(rewriter, op.getLoc(),
                            cycle, unit, scale, 1, 1, weightBinding);
                        emitMxm(rewriter, op.getLoc(), cycle, unit,
                            "iw", weightBuffer, 3 - pulse, 0, 0,
                            1, 1, 0, 1, "sram", true,
                            "supercell", 0, dataFormat,
                            execution->weight_input_mode());
                        continue;
                    }
                    for (int64_t lane = 0;
                         lane < target.throughput().lanes_per_tile;
                         ++lane) {
                        emitVxm(rewriter, op.getLoc(), op.getWeight(),
                            cycle, lane, "multiply", "stream_i8",
                            target.streams().streams_per_direction
                                + lane,
                            0.0f, "immediate", 0, scale,
                            "fp32", -1, hemi, hemi, weightBinding);
                        emitVxm(rewriter, op.getLoc(), op.getWeight(),
                            cycle + 1, 8 + lane, "cast", "alu",
                            lane, 0.0f, "immediate", 0, 0.0f,
                            dataFormat,
                            localMxm
                                    * target.throughput()
                                          .mxm_load_streams_per_cycle
                                + lane * 2,
                            hemi, hemi);
                    }
                    emitMxm(rewriter, op.getLoc(),
                        cycle + weightToIw,
                        unit,
                        "iw", weightBuffer, 3 - pulse, 0, 0,
                        1, 1, 0, 1, "stream", true,
                        "supercell", 0, dataFormat,
                        execution->weight_input_mode());
                }
            }

            const bool finalReduction =
                reduction + 1 == reductionBlocks;
            for (int64_t tokenBlock = 0;
                 tokenBlock < tokenBlocks; ++tokenBlock) {
                const int64_t computeCycle =
                    firstCompute + tokenBlock * tile;
                for (int64_t hemisphere = 0;
                     hemisphere < target.memory().hemispheres;
                     ++hemisphere) {
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = inputSlices[byte];
                        const auto latency = target.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::MxmActivation,
                            target::StreamDirection::East, slice);
                        if (!latency) return mlir::failure();
                        emitMem(rewriter, op.getLoc(),
                            computeCycle - *latency,
                            hemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + slice,
                            "read",
                            inputBase + reduction * m
                                + tokenBlock * tile,
                            hemisphere * 2 + byte, tile, 1, 1,
                            "sram", inputBinding);
                    }
                    emitMxm(rewriter, op.getLoc(), computeCycle,
                        hemisphere
                                * target.throughput()
                                      .mxms_per_hemisphere
                            + localMxm,
                        "compute", weightBuffer, 0,
                        hemisphere * 2, 0, tile, 1,
                        accumulatorAddress + tokenBlock * tile,
                        1, "sram", true, "supercell", 0,
                        dataFormat);
                }
            }
            phaseStart = firstCompute + tokenBlocks * tile;
            if (!finalReduction) continue;

            const int64_t castStart = phaseStart
                + accumulatorLatency
                + target.throughput().accumulator_read_to_vxm_latency;
            for (int64_t hemisphere = 0;
                 hemisphere < target.memory().hemispheres;
                 ++hemisphere) {
                const char* inputHemisphere =
                    hemisphere == 0 ? "east" : "west";
                const int64_t inputStream =
                    target.streams().streams_per_direction;
                const int64_t outputStream = hemisphere * 2;
                for (int64_t token = 0; token < m; ++token) {
                    const int64_t vxmCycle = castStart + token;
                    emitMxm(rewriter, op.getLoc(),
                        vxmCycle
                            - target.throughput()
                                  .accumulator_read_to_vxm_latency,
                        hemisphere
                                * target.throughput()
                                      .mxms_per_hemisphere
                            + localMxm,
                        "accumulator_read", 0, 0, 0, 0, 1, 1,
                        accumulatorAddress + token, 1, "sram",
                        true, "supercell", 0, dataFormat);
                    emitVxm(rewriter, op.getLoc(), op.getWeight(),
                        vxmCycle, hemisphere, "pass",
                        "stream_f32", inputStream, 0.0f,
                        "immediate", 0, 0.0f, dataFormat,
                        outputStream, inputHemisphere, "east");
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice =
                            resultSlices[hemisphere * 2 + byte];
                        const auto latency = target.transport_latency(
                            target::StreamEndpoint::VxmResult,
                            target::StreamEndpoint::Mem,
                            target::StreamDirection::East, slice);
                        if (!latency) return mlir::failure();
                        emitMem(rewriter, op.getLoc(),
                            vxmCycle + *latency, slice, "write",
                            resultBase + outputGroup * m + token,
                            outputStream + byte, 1, 1, 0);
                    }
                }
            }
            int64_t finalWriteLatency = 0;
            for (int64_t slice : resultSlices) {
                const auto latency = target.transport_latency(
                    target::StreamEndpoint::VxmResult,
                    target::StreamEndpoint::Mem,
                    target::StreamDirection::East, slice);
                if (!latency) return mlir::failure();
                finalWriteLatency =
                    std::max(finalWriteLatency, *latency);
            }
            phaseStart = castStart + m + finalWriteLatency;
        }
    }

    mlir::OperationState timeline(
        op.getLoc(), TimelineOp::getOperationName());
    timeline.addAttributes({
        rewriter.getNamedAttr(
            "name", rewriter.getStringAttr("linear_projection")),
        rewriter.getNamedAttr(
            "start", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr(
            "end", rewriter.getI64IntegerAttr(phaseStart)),
    });
    rewriter.create(timeline);
    auto output = createBinding(rewriter, op.getLoc(), {},
        outputIndex, "output", "result",
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType()),
        resultPlacement);
    return output.getValue();
}

} // namespace

mlir::LogicalResult lowerLinearProjectionSchedules(
    mlir::IRRewriter& rewriter, mlir::func::FuncOp function,
    const target::LPUTargetModel& target)
{
    llvm::SmallVector<stream::ProjectionTaskOp> projections;
    function.walk([&](stream::ProjectionTaskOp op) {
        if (op.getKind() == "linear") projections.push_back(op);
    });
    int64_t outputIndex = 0;
    for (stream::ProjectionTaskOp projection : projections) {
        auto result = emitLinearProjection(
            rewriter, projection, target, outputIndex++);
        if (mlir::failed(result)) return mlir::failure();
        rewriter.replaceOp(projection, *result);
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
