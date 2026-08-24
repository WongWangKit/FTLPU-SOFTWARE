#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main() try {
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
        rewriter.getUnknownLoc(), "swish_emitter",
        rewriter.getFunctionType({tensor, tensor}, {tensor}));
    mlir::Block* entry = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    target::LPUTargetModel target;
    auto [local, peer] = schedule::ffn_detail::emitFfnSwishAlu(
        rewriter, rewriter.getUnknownLoc(), tensor,
        entry->getArgument(0), entry->getArgument(1), target,
        FfnScheduleStrategy::Fused, 10, 0, 6);

    int instructions = 0;
    int expInstructions = 0;
    int reciprocalInstructions = 0;
    int castInstructions = 0;
    int64_t lastCycle = 0;
    function.walk([&](schedule::VxmOp op) {
        ++instructions;
        lastCycle = std::max(lastCycle, static_cast<int64_t>(op.getCycle()));
        if (op.getOpcode() == "exp") ++expInstructions;
        if (op.getOpcode() == "reciprocal") ++reciprocalInstructions;
        if (op.getOpcode() == "cast") ++castInstructions;
    });

    require(instructions == 8, "Swish emitter must issue one 8-stage VXM chain");
    require(expInstructions == 1, "Swish emitter must issue one exp");
    require(reciprocalInstructions == 1,
        "Swish emitter must issue one reciprocal");
    require(castInstructions == 0,
        "Swish BF16 conversion is a chain-tail cast target");
    require(lastCycle == 10, "Swish chain is issued in one cycle");
    require(local == peer,
        "one compact VXM packet represents both physical chain outputs");
    require(peer.getOutputHemisphere() == "west",
        "peer Swish output must cross to the other hemisphere");
    require(peer.getQueue() == 7 && peer.getOutputStream() == 6,
        "Swish output must use the fixed 8-stage chain tail");

    auto placement = schedule::ffn_detail::schedule_placement(rewriter,
        {31}, 8192 + 3 * 128 + 64, 1, 1, "east",
        "fp16_mxm_activation_planar");
    require(schedule::ffn_detail::get_base_row(placement) == 8640,
        "FFN schedule placement must preserve the physical base row");
    std::cout << "ffn_swish_emitter_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ffn_swish_emitter_test failed: " << error.what() << '\n';
    return 1;
}
