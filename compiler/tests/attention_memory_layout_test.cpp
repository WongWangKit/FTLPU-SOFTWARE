#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"

#include "mlir/IR/Builders.h"

#include <iostream>
#include <stdexcept>

int main() try
{
    using namespace ftlpu::compiler;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<stream::StreamDialect>();
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    const auto type = mlir::RankedTensorType::get({128, 576}, builder.getBF16Type());
    auto function = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
        "layout", builder.getFunctionType({type, type}, {type}));
    auto* block = function.addEntryBlock();
    builder.setInsertionPointToStart(block);
    const auto placement = [&](int64_t base, int64_t first, int64_t count) {
        llvm::SmallVector<mlir::Attribute> slices;
        for (int64_t i = 0; i < count; ++i)
            slices.push_back(builder.getI64IntegerAttr(first + i));
        return builder.getDictionaryAttr({
            builder.getNamedAttr("base_row", builder.getI64IntegerAttr(base)),
            builder.getNamedAttr("bank", builder.getI64IntegerAttr(0)),
            builder.getNamedAttr("slices", builder.getArrayAttr(slices))});
    };
    llvm::SmallVector<mlir::NamedAttribute> entries;
    for (const char* name : {"query_weight", "key_weight", "value_weight"})
        entries.push_back(builder.getNamedAttr(name, placement(0, 20, 8)));
    for (const char* name : {"score", "score_mxm1", "exp", "exp_mxm1"})
        entries.push_back(builder.getNamedAttr(name, placement(0, 0, 4)));
    for (const char* name : {"causal_mask", "causal_mask_mxm1"})
        entries.push_back(builder.getNamedAttr(name, placement(0, 0, 2)));
    for (const char* name : {"probability_pack", "probability_diagonal"})
        entries.push_back(builder.getNamedAttr(name, placement(0, 0, 16)));
    auto products = placement(100, 16, 4);
    llvm::SmallVector<mlir::NamedAttribute> productAttrs(products.begin(), products.end());
    productAttrs.push_back(builder.getNamedAttr("bank_interleaved", builder.getBoolAttr(true)));
    entries.push_back(builder.getNamedAttr("rope_product", builder.getDictionaryAttr(productAttrs)));
    entries.push_back(builder.getNamedAttr("rope_product_key", placement(500, 4, 4)));
    const auto config = builder.getDictionaryAttr({
        builder.getNamedAttr("seq_len", builder.getI64IntegerAttr(128)),
        builder.getNamedAttr("hidden", builder.getI64IntegerAttr(576)),
        builder.getNamedAttr("query_heads", builder.getI64IntegerAttr(9)),
        builder.getNamedAttr("kv_heads", builder.getI64IntegerAttr(3)),
        builder.getNamedAttr("head_dim", builder.getI64IntegerAttr(64))});
    schedule::AttentionTaskGraph graph;
    graph.output = builder.create<stream::ProjectionTaskOp>(builder.getUnknownLoc(),
        type, block->getArgument(0), block->getArgument(1), mlir::Value{},
        "output", config, builder.getDictionaryAttr(entries), builder.getArrayAttr({}));
    const target::LPUTargetModel target;
    const schedule::AttentionMemoryLayout layout(graph, target);
    using Kind = schedule::AttentionProjectionKind;
    const auto require = [](bool condition, const char* message) {
        if (!condition) throw std::logic_error(message);
    };
    // Key's low products own their slice group and base. Every token of
    // every KV head must remain inside that group's six 128-row vectors.
    for (int64_t head = 0; head < 3; ++head)
        for (int64_t product = 0; product < 2; ++product)
            for (int64_t token = 0; token < 128; ++token) {
                require(layout.ropeProductAddress(Kind::Key, head, 0, product, token)
                        == 500 + (head * 2 + product) * 128 + token,
                    "Key RoPE product escaped its dedicated placement");
                require(layout.ropeProductSlice(Kind::Key, product, token, 0)
                        == 4 + product * 2,
                    "Key RoPE product uses the wrong dedicated slice");
            }
    require(layout.ropeProductAddress(Kind::Query, 0, 0, 0, 0) == 100
            && layout.ropeProductAddress(Kind::Query, 0, 0, 2, 0) == 100,
        "compact Query products lost their separate-bank row reuse");
    require(layout.ropeProductAddress(Kind::Key, 0, 0, 2, 0) == 100 + 18 * 128,
        "Key high products changed their shared placement");
    mlir::NamedAttrList wideConfig(config);
    wideConfig.set("head_dim", builder.getI64IntegerAttr(128));
    graph.output.setConfigAttr(wideConfig.getDictionary(&context));
    const schedule::AttentionMemoryLayout wideHead(graph, target);
    require(wideHead.ropeProductAddress(Kind::Key, 1, 1, 1, 127)
            == 500 + 7 * 128 + 127,
        "wide Key RoPE product has the wrong pair-block stride");
    graph.output.setConfigAttr(config);
    productAttrs.back() = builder.getNamedAttr("bank_interleaved", builder.getBoolAttr(false));
    entries[entries.size() - 2] = builder.getNamedAttr("rope_product", builder.getDictionaryAttr(productAttrs));
    graph.output.setMemoryPlanAttr(builder.getDictionaryAttr(entries));
    const schedule::AttentionMemoryLayout legacy(graph, target);
    require(legacy.ropeProductAddress(Kind::Key, 0, 0, 0, 0) == 100 + 36 * 64,
        "legacy RoPE product addressing changed");
    module.erase();
    return 0;
} catch (const std::exception& error) {
    std::cerr << "attention_memory_layout_test failed: " << error.what() << '\n';
    return 1;
}
