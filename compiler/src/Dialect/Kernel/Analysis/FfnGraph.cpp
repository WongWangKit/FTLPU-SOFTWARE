#include "ftlpu/compiler/Dialect/Kernel/Analysis/ffn_graph.hpp"

namespace ftlpu::compiler::kernel {

std::optional<FfnGraph> match_ffn_graph(MatmulOp output)
{
    auto multiply = output.getLhs().getDefiningOp<ElementwiseOp>();
    if (!multiply || multiply.getKind() != "multiply") return std::nullopt;

    SwishOp swish = multiply.getLhs().getDefiningOp<SwishOp>();
    MatmulOp up = multiply.getRhs().getDefiningOp<MatmulOp>();
    if (!swish || !up) {
        swish = multiply.getRhs().getDefiningOp<SwishOp>();
        up = multiply.getLhs().getDefiningOp<MatmulOp>();
    }
    auto gate = swish
        ? swish.getInput().getDefiningOp<MatmulOp>()
        : MatmulOp {};
    if (!gate || !up || gate.getLhs() != up.getLhs()
        || gate.getM() != up.getM() || gate.getN() != up.getN()
        || gate.getK() != up.getK() || output.getM() != gate.getM()
        || output.getK() != gate.getN())
        return std::nullopt;

    return FfnGraph {output, gate, up, swish, multiply,
        {gate.getOperation(), up.getOperation(), swish.getOperation(),
            multiply.getOperation(), output.getOperation()}};
}

} // namespace ftlpu::compiler::kernel
