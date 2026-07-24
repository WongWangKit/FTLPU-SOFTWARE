#include "AttentionEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

ftlpu::compiler::stream::AttentionOp createAttention(
    mlir::OpBuilder& builder, mlir::Value input)
{
    using ftlpu::compiler::stream::AttentionOp;
    mlir::OperationState state(builder.getUnknownLoc(),
        AttentionOp::getOperationName());
    state.addOperands({input, input, input, input, input});
    state.addTypes(input.getType());
    state.addAttributes({
        builder.getNamedAttr("seq_len", builder.getI64IntegerAttr(32)),
        builder.getNamedAttr("hidden", builder.getI64IntegerAttr(32)),
        builder.getNamedAttr("query_heads", builder.getI64IntegerAttr(1)),
        builder.getNamedAttr("kv_heads", builder.getI64IntegerAttr(1)),
        builder.getNamedAttr("head_dim", builder.getI64IntegerAttr(32)),
        builder.getNamedAttr("rope_theta", builder.getF32FloatAttr(10000.0f)),
        builder.getNamedAttr("causal", builder.getBoolAttr(true)),
        builder.getNamedAttr("memory_plan", builder.getDictionaryAttr({})),
        builder.getNamedAttr("routes", builder.getArrayAttr({})),
    });
    return llvm::cast<AttentionOp>(builder.create(state));
}

} // namespace

int main()
{
    using namespace ftlpu::compiler;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<stream::StreamDialect>();
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
    auto attention = createAttention(rewriter, entry->getArgument(0));

    target::LPUTargetModel target;
    schedule::attention_detail::emitRopeOrCast(
        rewriter, attention, target, 10, 0, true, entry->getArgument(0));
    schedule::attention_detail::emitRopeOrCast(
        rewriter, attention, target, 20, 1, false, entry->getArgument(0));

    int ropeInstructions = 0;
    int castInstructions = 0;
    function.walk([&](schedule::VxmOp op) {
        if (op.getCycle() == 10 || op.getCycle() == 11)
            ++ropeInstructions;
        if (op.getCycle() == 20) ++castInstructions;
    });
    require(ropeInstructions == 6, "RoPE emitter must issue six VXM instructions");
    require(castInstructions == 2, "cast emitter must issue two VXM instructions");
}
