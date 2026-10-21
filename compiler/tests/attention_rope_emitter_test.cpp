#include "AttentionEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

// Exercise only the real activation-staging emitter. Full decoder lowering
// has unrelated MXM capacity limits for long sequences; this verifies the
// generated physical MEM operations without bypassing their verifier.
void testDistributedActivationStaging(int64_t seqLen)
{
    using namespace ftlpu::compiler;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<schedule::ScheduleDialect>();
    mlir::IRRewriter rewriter(&context);
    const auto location = rewriter.getUnknownLoc();
    auto module = mlir::ModuleOp::create(location);
    rewriter.setInsertionPointToStart(module.getBody());
    auto function = rewriter.create<mlir::func::FuncOp>(location, "staging",
        rewriter.getFunctionType({}, {}));
    auto* entry = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);
    const target::LPUTargetModel target;
    llvm::SmallVector<int64_t> slices;
    for (int64_t slice = 0; slice < 16; ++slice) slices.push_back(slice);
    constexpr int64_t base = 64, hiddenBlocks = 18, start = 96;
    schedule::attention_detail::emitDistributedActivationStagingReads(
        rewriter, location, target, slices, base, 1, seqLen, hiddenBlocks, start);
    rewriter.create<mlir::func::ReturnOp>(location);
    if (mlir::failed(mlir::verify(module)))
        throw std::logic_error("activation staging generated invalid MEM domains");

    using Queue = std::tuple<int64_t, int64_t, int64_t>;
    using Beat = std::pair<int64_t, int64_t>;
    std::map<Queue, std::set<Beat>> actual;
    function.walk([&](schedule::MemTransferOp op) {
        const Queue queue{op.getHemisphere(), op.getSlice(), *op.getBank()};
        const auto latency = target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput, target::StreamDirection::West, op.getSlice()).value();
        std::set<int64_t> cycles, addresses;
        for (int64_t g = 0; g < *op.getGroupCount(); ++g)
            for (int64_t w = 0; w < *op.getWaveCount(); ++w)
                for (int64_t r = 0; r < op.getRepeatCount(); ++r) {
                    const auto cycle = op.getCycle() + latency - start
                        + r * op.getRepeatInterval() + w * *op.getWaveInterval()
                        + g * *op.getGroupInterval();
                    const auto address = op.getAddress() + r * op.getAddressStride()
                        + w * *op.getWaveAddressStride() + g * *op.getGroupAddressStride();
                    if (!cycles.insert(cycle).second || !addresses.insert(address).second
                        || !actual[queue].emplace(cycle, address).second)
                        throw std::logic_error("activation staging duplicates a MEM cycle or SRAM row");
                }
    });
    std::map<Queue, std::set<Beat>> expected;
    // Logical traversal is independent of loop encoding: for each token and
    // 32-column feature block, select its lane pair and distributed SRAM row.
    for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere)
        for (int64_t feature = 0; feature < hiddenBlocks; ++feature)
            for (int64_t token = 0; token < seqLen; ++token)
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = 2 * (token % 8) + byte;
                    const int64_t row = base + ((token / 32) * hiddenBlocks + feature) * 4
                        + (token % 32) / 8;
                    expected[{hemisphere, slice, 1}].emplace(feature * seqLen + token, row);
                }
    if (actual != expected)
        throw std::logic_error("activation staging does not cover each logical token/feature exactly once");
}

int main()
{
    for (const auto seqLen : {1, 32, 64, 128})
        testDistributedActivationStaging(seqLen);
    using namespace ftlpu::compiler;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<schedule::ScheduleDialect>();

    mlir::IRRewriter rewriter(&context);
    auto module = mlir::ModuleOp::create(rewriter.getUnknownLoc());
    rewriter.setInsertionPointToStart(module.getBody());
    const auto tensor =
        mlir::RankedTensorType::get({32, 32}, rewriter.getF32Type());
    auto function = rewriter.create<mlir::func::FuncOp>(
        rewriter.getUnknownLoc(), "rope_emitter",
        rewriter.getFunctionType({tensor}, {tensor}));
    mlir::Block* entry = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    target::LPUTargetModel target;
    schedule::attention_detail::emitRopeOrCast(
        rewriter, rewriter.getUnknownLoc(), target, 10, 0, true,
        entry->getArgument(0), rewriter.getBF16Type());
    schedule::attention_detail::emitRopeOrCast(
        rewriter, rewriter.getUnknownLoc(), target, 20, 1, false,
        entry->getArgument(0), rewriter.getBF16Type());
    schedule::attention_detail::emitRopeOrCast(
        rewriter, rewriter.getUnknownLoc(), target, 30, 1, true,
        entry->getArgument(0), rewriter.getBF16Type());

    int ropeInstructions = 0;
    int castInstructions = 0;
    bool sawWestCast15 = false;
    bool sawWestCast14 = false;
    bool sawWestRopeHead = false;
    bool sawWestRopeLowTail = false;
    bool sawWestRopeHighTail = false;
    function.walk([&](schedule::VxmOp op) {
        if (op.getCycle() == 10 || op.getCycle() == 11)
            ++ropeInstructions;
        if (op.getCycle() == 20) ++castInstructions;
        if (op.getCycle() == 20 && op.getQueue() == 15)
            sawWestCast15 = true;
        if (op.getCycle() == 20 && op.getQueue() == 14)
            sawWestCast14 = true;
        if (op.getCycle() == 30 && op.getQueue() == 15)
            sawWestRopeHead = true;
        if (op.getCycle() == 31 && op.getQueue() == 13
            && op.getLhsKind() == "alu" && op.getLhsIndex() == 15
            && op.getRhsKind() == "alu" && op.getRhsIndex() == 14)
            sawWestRopeLowTail = true;
        if (op.getCycle() == 31 && op.getQueue() == 10
            && op.getLhsKind() == "alu" && op.getLhsIndex() == 12
            && op.getRhsKind() == "alu" && op.getRhsIndex() == 11)
            sawWestRopeHighTail = true;
    });
    if (ropeInstructions != 6 || castInstructions != 2
        || !sawWestCast15 || !sawWestCast14) {
        std::cerr << "unexpected VXM instruction counts: rope="
                  << ropeInstructions << ", cast=" << castInstructions << '\n';
        return 1;
    }
    if (!sawWestRopeHead || !sawWestRopeLowTail
        || !sawWestRopeHighTail) {
        std::cerr << "west RoPE does not follow C15-to-C10 propagation\n";
        return 1;
    }
    return 0;
}
