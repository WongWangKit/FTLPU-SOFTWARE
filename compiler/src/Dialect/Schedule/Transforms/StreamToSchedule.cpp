// Keep this translation unit rebuilt with target topology ABI changes.
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/paged_weight_residency.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/ffn_schedule_emitter.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>

namespace ftlpu::compiler {
namespace {

bool isScheduleOperation(mlir::Operation& operation)
{
    return operation.getName().getDialectNamespace() == "ftlpu.schedule";
}

int64_t integerAttribute(
    mlir::Operation& operation, llvm::StringRef name, int64_t fallback)
{
    if (auto value = operation.getAttrOfType<mlir::IntegerAttr>(name))
        return value.getInt();
    return fallback;
}

void shiftIntegerAttribute(mlir::Operation& operation,
    llvm::StringRef name, int64_t offset)
{
    auto value = operation.getAttrOfType<mlir::IntegerAttr>(name);
    if (!value) return;
    operation.setAttr(name, mlir::IntegerAttr::get(
        value.getType(), value.getInt() + offset));
}

void assignMxmDataFormats(mlir::func::FuncOp function)
{
    function.walk([&](schedule::MxmComputeOp op) {
        const auto activationType =
            llvm::cast<mlir::RankedTensorType>(
                op.getActivation().getType());
        op.setDataFormat(
            lpu_16bit_data_format(
                activationType.getElementType()));
    });
}

struct WeightBindingInfo {
    int64_t bytes = 0;
    int64_t rows = 0;
    int64_t columns = 0;
    mlir::DictionaryAttr placement;
};

struct StageWeightUse {
    int64_t binding = -1;
    int64_t page = -1;
    int64_t bank = 0;
    int64_t ready = std::numeric_limits<int64_t>::max();
    int64_t release = 0;
    int64_t bindingBytes = 0;
    int64_t rows = 0;
    int64_t columns = 0;
    mlir::DictionaryAttr placement;
};

int64_t elementTypeBytes(mlir::Type type)
{
    if (type.isInteger(8)) return 1;
    if (is_lpu_16bit_float(type)) return 2;
    if (type.isF32()) return 4;
    return 0;
}

std::optional<int64_t> sourceBindingIndex(mlir::Value value)
{
    for (int64_t depth = 0; depth < 32; ++depth) {
        if (const auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
            return argument.getArgNumber();
        if (auto binding = value.getDefiningOp<schedule::BindingOp>())
            return binding.getIndex();
        mlir::Operation* definition = value.getDefiningOp();
        if (!definition || definition->getNumOperands() != 1) return std::nullopt;
        value = definition->getOperand(0);
    }
    return std::nullopt;
}

llvm::DenseMap<int64_t, WeightBindingInfo>
collectWeightBindings(mlir::func::FuncOp function)
{
    llvm::DenseMap<int64_t, WeightBindingInfo> bindings;
    function.walk([&](schedule::BindingOp binding) {
        if (binding.getAccess() != "input" || binding.getRole() != "weight")
            return;
        const auto type = llvm::dyn_cast<mlir::RankedTensorType>(
            binding.getValue().getType());
        const int64_t rows = type && type.getRank() >= 1
            ? type.getDimSize(0) : 0;
        const int64_t columns = type && type.getRank() >= 2
            ? type.getDimSize(1) : 0;
        bindings[static_cast<int64_t>(binding.getIndex())] =
            {static_cast<int64_t>(binding.getBytes()),
                rows, columns,
                binding.getPlacement()};
    });
    return bindings;
}

void mergeWeightUse(
    llvm::SmallVectorImpl<StageWeightUse>& uses, StageWeightUse use)
{
    auto existing = std::find_if(uses.begin(), uses.end(),
        [&](const StageWeightUse& candidate) {
            return candidate.binding == use.binding
                && candidate.page == use.page
                && candidate.bank == use.bank;
        });
    if (existing == uses.end()) {
        uses.push_back(std::move(use));
        return;
    }
    existing->ready = std::min(existing->ready, use.ready);
    existing->release = std::max(existing->release, use.release);
}

llvm::SmallVector<StageWeightUse> collectStageWeightUses(
    llvm::ArrayRef<mlir::Operation*> stage,
    const llvm::DenseMap<int64_t, WeightBindingInfo>& bindings,
    int64_t cycleOffset)
{
    llvm::SmallVector<StageWeightUse> uses;
    for (mlir::Operation* operation : stage) {
        if (auto read = llvm::dyn_cast<schedule::MemReadOp>(operation)) {
            const mlir::DictionaryAttr readPlacement = read.getPlacement();
            const auto page = readPlacement
                .getAs<mlir::IntegerAttr>("weight_page");
            const auto bindingIndex = sourceBindingIndex(read.getInput());
            if (!page || !bindingIndex) continue;
            mlir::DictionaryAttr placement = read.getPlacement()
                .getAs<mlir::DictionaryAttr>("binding_placement");
            if (!placement) placement = read.getPlacement();
            int64_t bytes = 0;
            int64_t rows = 0;
            int64_t columns = 0;
            if (const auto binding = bindings.find(*bindingIndex);
                binding != bindings.end()) {
                bytes = binding->second.bytes;
                rows = binding->second.rows;
                columns = binding->second.columns;
                placement = binding->second.placement;
            } else if (const auto type = llvm::dyn_cast<mlir::RankedTensorType>(
                           read.getInput().getType())) {
                bytes = type.getNumElements()
                    * elementTypeBytes(type.getElementType());
                rows = type.getRank() >= 1 ? type.getDimSize(0) : 0;
                columns = type.getRank() >= 2 ? type.getDimSize(1) : 0;
            }
            const auto placementBank =
                readPlacement.getAs<mlir::IntegerAttr>("bank");
            const int64_t bank = integerAttribute(*operation, "bank",
                placementBank ? placementBank.getInt() : 0);
            const int64_t ready = read.getCycle() + cycleOffset;
            const int64_t release = ready
                + (read.getWaveCount().value_or(1) - 1)
                    * read.getWaveInterval().value_or(1)
                + read.getDuration();
            mergeWeightUse(uses,
                {*bindingIndex, page.getInt(), bank, ready, release,
                    bytes, rows, columns, placement});
            continue;
        }
        auto transfer = llvm::dyn_cast<schedule::MemTransferOp>(operation);
        if (!transfer || transfer.getOpcode() != "read"
            || !transfer.getWeightPage() || !transfer.getAddressBinding())
            continue;
        const int64_t bindingIndex = static_cast<int64_t>(
            transfer.getAddressBinding().value());
        const auto binding = bindings.find(bindingIndex);
        if (binding == bindings.end()) continue;
        const int64_t ready = transfer.getCycle() + cycleOffset;
        const int64_t release = ready
            + (transfer.getWaveCount().value_or(1) - 1)
                * transfer.getWaveInterval().value_or(1)
            + (transfer.getRepeatCount() - 1)
                * transfer.getRepeatInterval()
            + 1;
        const int64_t page = static_cast<int64_t>(
            transfer.getWeightPage().value());
        const int64_t bank = static_cast<int64_t>(
            transfer.getBank().value_or(0));
        mergeWeightUse(uses, StageWeightUse{bindingIndex, page, bank,
            ready, release, binding->second.bytes,
            binding->second.rows, binding->second.columns,
            binding->second.placement});
    }
    return uses;
}

int64_t divideCeil(int64_t numerator, int64_t denominator)
{
    return numerator / denominator + (numerator % denominator != 0);
}

int64_t pagedWeightBytes(const StageWeightUse& use)
{
    const auto pageCountAttr =
        use.placement.getAs<mlir::IntegerAttr>("page_count");
    const int64_t pageCount = std::max<int64_t>(
        1, pageCountAttr ? pageCountAttr.getInt() : 1);
    const auto granularityAttr =
        use.placement.getAs<mlir::IntegerAttr>("page_granularity");
    const int64_t granularity = granularityAttr
        ? granularityAttr.getInt() : 0;
    const auto kindAttr =
        use.placement.getAs<mlir::StringAttr>("kind");
    const llvm::StringRef kind = kindAttr ? kindAttr.getValue() : "";

    if (use.rows > 0 && use.columns > 0 && granularity > 0) {
        if (kind == "w8a16_mxm_weight_wave_striped") {
            if (use.rows < use.columns) {
                const int64_t totalPairs = use.columns / 64;
                const int64_t firstPair = use.page * granularity;
                const int64_t pairCount = std::clamp<int64_t>(
                    totalPairs - firstPair, 0, granularity);
                return pairCount * use.rows * 64;
            }
            const int64_t reductionBlocks = use.rows / 32;
            const int64_t pagesPerOutputWave = std::max<int64_t>(
                1, divideCeil(reductionBlocks, granularity));
            const int64_t reductionPage = use.page % pagesPerOutputWave;
            const int64_t firstReduction = reductionPage * granularity;
            const int64_t reductionCount = std::clamp<int64_t>(
                reductionBlocks - firstReduction, 0, granularity);
            return reductionCount * 32 * 128;
        }
        if (kind == "w8a16_attention_weight_striped"
            || kind == "w8a16_mxm_weight_striped") {
            const int64_t itemColumns =
                kind == "w8a16_attention_weight_striped" ? 128 : 64;
            const int64_t totalItems = use.columns / itemColumns;
            const int64_t firstItem = use.page * granularity;
            const int64_t itemCount = std::clamp<int64_t>(
                totalItems - firstItem, 0, granularity);
            return itemCount * use.rows * itemColumns;
        }
    }

    const int64_t quotient = use.bindingBytes / pageCount;
    const int64_t remainder = use.bindingBytes % pageCount;
    return quotient + (use.page < remainder ? 1 : 0);
}

int64_t pageTransferCycles(
    const StageWeightUse& use, const target::LPUTargetModel& target)
{
    const int64_t bytes = pagedWeightBytes(use);
    const auto hemisphere = use.placement.getAs<mlir::StringAttr>("hemisphere");
    const int64_t sideCount = hemisphere
            && hemisphere.getValue() != "both"
        ? 1 : target.memory().hemispheres;
    const int64_t sideBytes = divideCeil(bytes, sideCount);
    const int64_t c2cBytesPerCycle =
        target.streams().c2c_streams_per_direction
        * target.streams().c2c_bytes_per_stream_per_cycle;
    const int64_t c2cCycles = divideCeil(sideBytes, c2cBytesPerCycle);
    const auto& external = target.external_memory();
    const int64_t ddrCycles = divideCeil(
        bytes * external.lpu_clock_mhz * 100,
        external.ddr_peak_bandwidth_mbytes_per_second
            * external.ddr_scheduling_efficiency_percent);
    const int64_t queueDrain = divideCeil(
        external.ddr_request_queue_depth,
        target.streams().c2c_streams_per_direction);
    const int64_t transportGuard = queueDrain
        + hw::kMemEastBoundaryStreamRegisterColumn + hw::kTileRows
        + target.streams().c2c_streams_per_direction;
    return std::max(c2cCycles, ddrCycles)
        + external.ddr_read_latency_cycles
        + external.ddr_read_latency_jitter_cycles + transportGuard;
}

int64_t requiredWeightPrefetchDelay(
    llvm::ArrayRef<StageWeightUse> current,
    llvm::ArrayRef<StageWeightUse> previous,
    const target::LPUTargetModel& target)
{
    int64_t delay = 0;
    const int64_t dmaLead =
        target.external_memory().ddr_read_latency_cycles + 1;
    for (const StageWeightUse& use : current) {
        int64_t reusableCycle = 0;
        bool needsRefill = false;
        for (const StageWeightUse& prior : previous) {
            if (prior.binding == use.binding && prior.page == use.page
                && prior.bank == use.bank)
                continue;
            if (!schedule::pagedWeightResidencyOverlaps(
                    prior.placement, prior.bank, use.placement, use.bank))
                continue;
            needsRefill = true;
            reusableCycle = std::max(reusableCycle,
                std::max<int64_t>(0, prior.release - dmaLead));
        }
        if (!needsRefill) continue;
        delay = std::max(delay,
            reusableCycle + pageTransferCycles(use, target) - use.ready);
    }
    return std::max<int64_t>(0, delay);
}

void sequentializeScheduleStages(mlir::func::FuncOp function,
    const target::LPUTargetModel& target)
{
    llvm::SmallVector<llvm::SmallVector<mlir::Operation*>> stages;
    llvm::SmallVector<mlir::Operation*> current;
    std::optional<mlir::Location> currentLocation;
    for (mlir::Operation& operation : function.getBody().front()) {
        if (!isScheduleOperation(operation)) {
            if (!current.empty()) {
                stages.push_back(std::move(current));
                current.clear();
                currentLocation.reset();
            }
            continue;
        }
        if (!current.empty() && operation.getLoc() != *currentLocation) {
            stages.push_back(std::move(current));
            current.clear();
        }
        if (current.empty()) currentLocation = operation.getLoc();
        current.push_back(&operation);
    }
    if (!current.empty()) stages.push_back(std::move(current));

    int64_t cursor = 0;
    const auto weightBindings = collectWeightBindings(function);
    llvm::SmallVector<StageWeightUse> priorWeightUses;
    const int64_t streamDrainCycles =
        target.streams().system_register_columns;
    for (auto& stage : stages) {
        int64_t first = std::numeric_limits<int64_t>::max();
        int64_t end = 0;
        for (mlir::Operation* operation : stage) {
            if (auto cycle =
                    operation->getAttrOfType<mlir::IntegerAttr>("cycle")) {
                first = std::min(first, cycle.getInt());
                const int64_t duration = std::max<int64_t>(
                    1, integerAttribute(*operation, "duration",
                        integerAttribute(*operation, "repeat_count", 1)
                            * integerAttribute(
                                *operation, "repeat_interval", 1)));
                end = std::max(end, cycle.getInt() + duration);
            }
            if (auto resultCycle = operation->getAttrOfType<
                    mlir::IntegerAttr>("result_cycle")) {
                first = std::min(first, resultCycle.getInt());
                end = std::max(end, resultCycle.getInt()
                    + integerAttribute(
                        *operation, "result_duration", 1));
            }
            if (auto start =
                    operation->getAttrOfType<mlir::IntegerAttr>("start"))
                first = std::min(first, start.getInt());
            if (auto timelineEnd =
                    operation->getAttrOfType<mlir::IntegerAttr>("end"))
                end = std::max(end, timelineEnd.getInt());
        }
        if (first == std::numeric_limits<int64_t>::max()) continue;
        const int64_t baseOffset = cursor - first;
        auto stageWeightUses = collectStageWeightUses(
            stage, weightBindings, baseOffset);
        const int64_t prefetchDelay = requiredWeightPrefetchDelay(
            stageWeightUses, priorWeightUses, target);
        const int64_t offset = baseOffset + prefetchDelay;
        for (mlir::Operation* operation : stage) {
            shiftIntegerAttribute(*operation, "cycle", offset);
            shiftIntegerAttribute(*operation, "result_cycle", offset);
            shiftIntegerAttribute(*operation, "start", offset);
            shiftIntegerAttribute(*operation, "end", offset);
        }
        // A stage can finish issuing while its final vector beat is still
        // moving through passive stream-register links. Keep the next stage
        // from injecting a different producer onto those links until the
        // longest possible on-chip route has drained.
        for (StageWeightUse& use : stageWeightUses) {
            use.ready += prefetchDelay;
            use.release += prefetchDelay;
            priorWeightUses.push_back(std::move(use));
        }
        cursor += prefetchDelay
            + std::max<int64_t>(1, end - first) + streamDrainCycles;
    }

    llvm::SmallDenseSet<mlir::Value> returnedValues;
    function.walk([&](mlir::func::ReturnOp op) {
        for (mlir::Value value : op.getOperands())
            returnedValues.insert(value);
    });
    function.walk([&](schedule::BindingOp binding) {
        if (binding.getAccess() == "output"
            && !returnedValues.contains(binding.getValue()))
            binding->setAttr(
                "access", mlir::StringAttr::get(
                    function.getContext(), "internal"));
    });
    int64_t outputIndex = 0;
    function.walk([&](schedule::BindingOp binding) {
        if (binding.getAccess() == "output")
            binding->setAttr("index", mlir::IntegerAttr::get(
                binding.getIndexAttr().getType(), outputIndex++));
    });
}

class LowerStreamToSchedulePass final
    : public mlir::PassWrapper<LowerStreamToSchedulePass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStreamToSchedulePass)

    LowerStreamToSchedulePass() = default;
    explicit LowerStreamToSchedulePass(FfnScheduleStrategy ffnStrategy,
        AttentionScheduleStrategy attentionStrategy, bool stageTiming)
        : ffn_strategy_(ffnStrategy)
        , attention_strategy_(attentionStrategy)
        , stage_timing_(stageTiming)
    {
    }

    llvm::StringRef getArgument() const final { return "ftlpu-stream-to-schedule"; }
    llvm::StringRef getDescription() const final
    {
        return "Schedules LPU stream routes at exact CModel issue cycles";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        if (!function.getBody().hasOneBlock()) {
            function.emitError("cycle scheduling currently requires a single-block function");
            signalPassFailure();
            return;
        }

        mlir::IRRewriter rewriter(&getContext());
        auto target_model =
            target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target_model)) {
            signalPassFailure();
            return;
        }
        const target::LPUTargetModel& target = *target_model;
        const auto passStart = std::chrono::steady_clock::now();
        auto lastStage = passStart;
        const auto reportStage = [&](llvm::StringRef name) {
            if (!stage_timing_) return;
            const auto now = std::chrono::steady_clock::now();
            const double stageSeconds =
                std::chrono::duration<double>(now - lastStage).count();
            const double totalSeconds =
                std::chrono::duration<double>(now - passStart).count();
            llvm::errs() << "[ftlpu-stage-timing] " << name << ": "
                         << llvm::format("%.3f", stageSeconds)
                         << " s (total "
                         << llvm::format("%.3f", totalSeconds) << " s)\n";
            lastStage = now;
        };
        auto primitive_ffns =
            schedule::collectPrimitiveFfnSchedulePlans(function);
        if (mlir::failed(primitive_ffns)) {
            signalPassFailure();
            return;
        }
        reportStage("primitive-ffn-plan");
        int64_t ffnIndex = 0;
        for (schedule::PrimitiveFfnSchedulePlan& ffn : *primitive_ffns) {
            rewriter.setInsertionPoint(ffn.add);
            auto result = schedule::lowerFfnSchedule(
                rewriter, ffn, ffn_strategy_, target);
            if (mlir::failed(result)) {
                ffn.add.emitError(
                    "failed to schedule a primitive W8A16 FFN graph");
                signalPassFailure();
                return;
            }
            rewriter.replaceOp(ffn.add, *result);
            rewriter.eraseOp(ffn.down1);
            rewriter.eraseOp(ffn.down0);
            rewriter.eraseOp(ffn.hidden1_route);
            rewriter.eraseOp(ffn.hidden0_route);
            rewriter.eraseOp(ffn.multiply);
            rewriter.eraseOp(ffn.swish);
            rewriter.eraseOp(ffn.up);
            rewriter.eraseOp(ffn.gate);
            if (stage_timing_)
                reportStage("primitive-ffn-lower-"
                    + std::to_string(ffnIndex));
            ++ffnIndex;
        }
        if (mlir::failed(
                schedule::lowerAttentionSchedules(
                    rewriter, function, target, attention_strategy_))) {
            signalPassFailure();
            return;
        }
        reportStage("attention");
        if (mlir::failed(
                schedule::lowerRmsNormSchedules(rewriter, function, target))) {
            signalPassFailure();
            return;
        }
        reportStage("rmsnorm");
        if (mlir::failed(schedule::lowerLinearProjectionSchedules(
                rewriter, function, target))) {
            signalPassFailure();
            return;
        }
        reportStage("linear-projection");
        if (mlir::failed(schedule::lowerElementwiseSchedules(
                rewriter, function, target))) {
            signalPassFailure();
            return;
        }
        reportStage("elementwise");
        schedule::ResourceScheduler scheduler;
        schedule::StreamFabricScheduler streamScheduler(
            target.streams().system_register_columns,
            target.streams().streams_per_direction);
        if (mlir::failed(schedule::lowerSwigluSchedules(
                rewriter, function, target, scheduler, streamScheduler))
            || mlir::failed(schedule::lowerMatmulSchedules(
                rewriter, function, target, scheduler, streamScheduler))) {
            signalPassFailure();
            return;
        }
        reportStage("generic-kernels");

        assignMxmDataFormats(function);
        sequentializeScheduleStages(function, target);
        reportStage("finalize");
    }

private:
    FfnScheduleStrategy ffn_strategy_ = FfnScheduleStrategy::Tail;
    AttentionScheduleStrategy attention_strategy_ =
        AttentionScheduleStrategy::Tail;
    bool stage_timing_ = false;
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_stream_to_schedule_pass(
    FfnScheduleStrategy ffn_strategy,
    AttentionScheduleStrategy attention_strategy, bool stage_timing)
{
    return std::make_unique<LowerStreamToSchedulePass>(
        ffn_strategy, attention_strategy, stage_timing);
}

} // namespace ftlpu::compiler
