#include "ftlpu/compiler/Transforms/passes.hpp"

#include "ftlpu/compiler/Dialect/Kernel/Analysis/attention_graph.hpp"
#include "ftlpu/compiler/Dialect/Kernel/Analysis/ffn_graph.hpp"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

namespace ftlpu::compiler {
namespace {

std::string refusal(mlir::Operation* op,
    const llvm::DenseSet<mlir::Operation*>& structured)
{
    const auto name = op->getName().getStringRef();
    if (name.starts_with("stablehlo.")) {
        if (auto reason = op->getAttrOfType<mlir::StringAttr>("ftlpu.kernel_rejection_reason"))
            return reason.getValue().str();
        if (name == "stablehlo.select" || name == "stablehlo.compare")
            return "mask dataflow was not proven to match the supported causal form; "
                   "padding/additive mask semantics cannot be discarded";
        if (name == "stablehlo.sine" || name == "stablehlo.cosine")
            return "positional dataflow was not proven: actual position_ids, inv_freq, "
                   "offset, scaling and rotation convention must be supported";
        if (name == "stablehlo.reduce")
            return "reduction axes, body and normalization dataflow were not proven "
                   "to form supported RMSNorm or Softmax";
        if (name == "stablehlo.convert")
            return "required dtype conversion is not consumed by a supported "
                   "numeric profile or Kernel primitive";
        if (name == "stablehlo.logistic")
            return "sigmoid dataflow was not consumed by a supported SiLU/SwiGLU/FFN pattern";
        return "StableHLO computation has no supported Kernel conversion; "
               "its dataflow, dtype, mask, or positional constraints were not proven";
    }
    if (!name.starts_with("ftlpu.kernel.")) {
        if (name == "func.func" || name == "func.return"
            || name.starts_with("ftlpu.tensor.") || name.starts_with("ftlpu.stream.")
            || name.starts_with("ftlpu.schedule.") || name.starts_with("ftlpu.command."))
            return {};
        return "operation has no consumer in the supported Kernel/Tensor pipeline";
    }
    if (structured.contains(op)) return {};
    if (llvm::isa<kernel::MatmulOp, kernel::RmsNormOp,
            kernel::ElementwiseOp, kernel::SwigluOp>(op))
        return {};
    if (auto bmm = llvm::dyn_cast<kernel::BatchMatmulOp>(op))
        return bmm.getRole() == "generic"
            ? "generic batch_matmul has no standalone Tensor lowering"
            : "batch_matmul is not consumed by a complete AttentionGraph";
    return "auxiliary Kernel operation is not consumed by a complete "
           "AttentionGraph/FfnGraph and has no standalone Tensor lowering";
}

class VerifyKernelCoveragePass final
    : public mlir::PassWrapper<VerifyKernelCoveragePass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyKernelCoveragePass)
    VerifyKernelCoveragePass() = default;
    explicit VerifyKernelCoveragePass(bool partial) : partial_(partial) {}
    llvm::StringRef getArgument() const final { return "ftlpu-verify-kernel-coverage"; }
    llvm::StringRef getDescription() const final
    {
        return "Reject computation not consumed by the complete Kernel/Tensor path";
    }
    void runOnOperation() final
    {
        auto function = getOperation();
        llvm::DenseSet<mlir::Operation*> structured;
        function.walk([&](kernel::MatmulOp root) {
            if (auto attention = kernel::match_attention_graph(root))
                structured.insert(attention->operations.begin(), attention->operations.end());
            if (auto ffn = kernel::match_ffn_graph(root))
                structured.insert(ffn->operations.begin(), ffn->operations.end());
        });
        struct Rejected {
            std::string reason;
            llvm::SmallVector<mlir::Location> locations;
        };
        std::map<std::string, Rejected> rejected;
        function.walk([&](mlir::Operation* op) {
            const auto reason = refusal(op, structured);
            if (reason.empty()) return;
            auto& group = rejected[op->getName().getStringRef().str()];
            group.reason = reason;
            group.locations.push_back(op->getLoc());
        });
        mlir::Builder builder(&getContext());
        llvm::SmallVector<mlir::Attribute> groups;
        int64_t count = 0;
        for (const auto& [name, group] : rejected) {
            llvm::SmallVector<mlir::Attribute> locations;
            for (auto location : group.locations) {
                std::string text;
                llvm::raw_string_ostream stream(text);
                location.print(stream);
                locations.push_back(builder.getStringAttr(text));
            }
            count += static_cast<int64_t>(group.locations.size());
            groups.push_back(builder.getDictionaryAttr({
                builder.getNamedAttr("operation", builder.getStringAttr(name)),
                builder.getNamedAttr("count", builder.getI64IntegerAttr(group.locations.size())),
                builder.getNamedAttr("reason", builder.getStringAttr(group.reason)),
                builder.getNamedAttr("locations", builder.getArrayAttr(locations))}));
            llvm::errs() << "Kernel coverage: " << name << " count="
                         << group.locations.size() << ": " << group.reason << '\n';
            for (auto location : group.locations)
                llvm::errs() << "  " << location << '\n';
        }
        function->setAttr("ftlpu.kernel_complete", builder.getBoolAttr(rejected.empty()));
        function->setAttr("ftlpu.kernel_mode",
            builder.getStringAttr(partial_ ? "partial" : "complete"));
        function->setAttr("ftlpu.kernel_unconverted", builder.getArrayAttr(groups));
        if (rejected.empty()) return;
        if (partial_) {
            function.emitWarning() << "partial Kernel conversion: " << count
                << " computations have no complete downstream lowering; "
                   "output is for debugging and must not enter Tensor lowering";
        } else {
            function.emitError() << "incomplete Kernel conversion: " << count
                << " unsupported computations in " << rejected.size()
                << " operation kinds; refusing to enter downstream lowering";
            signalPassFailure();
        }
    }
private:
    bool partial_ = false;
};

} // namespace

std::unique_ptr<mlir::Pass> create_verify_kernel_coverage_pass(bool allow_partial)
{
    return std::make_unique<VerifyKernelCoveragePass>(allow_partial);
}

} // namespace ftlpu::compiler
