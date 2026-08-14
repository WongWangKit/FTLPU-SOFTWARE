#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace ftlpu::compiler {
namespace {

mlir::DictionaryAttr with_bank(mlir::MLIRContext* context,
    mlir::DictionaryAttr placement, std::int64_t bank)
{
    mlir::NamedAttrList attributes(placement);
    attributes.set("bank",
        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), bank));
    if (auto binding = placement.getAs<mlir::DictionaryAttr>(
            "binding_placement"))
        attributes.set("binding_placement",
            with_bank(context, binding, bank));
    return attributes.getDictionary(context);
}

class AssignWeightBankPass final
    : public mlir::PassWrapper<AssignWeightBankPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    explicit AssignWeightBankPass(std::int64_t bank) : bank_(bank) {}

    void runOnOperation() override
    {
        auto function = getOperation();
        auto target_model = target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target_model)) {
            signalPassFailure();
            return;
        }
        if (bank_ < 0
            || bank_ >= target_model->memory().banks_per_slice) {
            function.emitError("weight bank is outside the target memory");
            signalPassFailure();
            return;
        }

        llvm::SmallDenseSet<std::int64_t, 16> weight_bindings;
        function.walk([&](schedule::BindingOp binding) {
            if (binding.getAccess() != "input"
                || binding.getRole() != "weight")
                return;
            weight_bindings.insert(binding.getIndex());
            binding->setAttr("placement", with_bank(&getContext(),
                binding.getPlacement(), bank_));
        });
        function.walk([&](schedule::MemReadOp read) {
            if (read.getRole() != "weight") return;
            read->setAttr("placement", with_bank(&getContext(),
                read.getPlacement(), bank_));
        });
        function.walk([&](schedule::MemTransferOp transfer) {
            const auto binding = transfer.getAddressBinding();
            if (!binding || !weight_bindings.contains(*binding)) return;
            transfer->setAttr("bank",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(&getContext(), 64), bank_));
        });
    }

private:
    std::int64_t bank_;
};

} // namespace

std::unique_ptr<mlir::Pass> create_assign_weight_bank_pass(std::int64_t bank)
{
    return std::make_unique<AssignWeightBankPass>(bank);
}

} // namespace ftlpu::compiler
