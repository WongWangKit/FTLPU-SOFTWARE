#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"
#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "DirectDomainEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>

namespace ftlpu::compiler::schedule {
namespace {
mlir::DictionaryAttr placement(mlir::IRRewriter& r,
    mlir::RankedTensorType type, int64_t slice, llvm::StringRef hemisphere, int64_t bank = 0, bool packed = false)
{
    llvm::SmallVector<mlir::Attribute> slices;
    const int64_t bytes = type.getElementType().isF32() ? 4 : 2;
    for (int64_t i = 0; i < bytes; ++i)
        slices.push_back(r.getI64IntegerAttr(slice + i));
    return r.getDictionaryAttr({
        r.getNamedAttr("kind", r.getStringAttr(packed ? (bytes == 4 ? "fp32_quad_planar" : "fp16_pair_planar") : (bytes == 4
            ? "fp32_scalar_broadcast_planar" : "fp16_scalar_broadcast_planar"))),
        r.getNamedAttr("slices", r.getArrayAttr(slices)),
        r.getNamedAttr("base_row", r.getI64IntegerAttr(0)),
        r.getNamedAttr("bank", r.getI64IntegerAttr(bank)),
        r.getNamedAttr("instruction_count", r.getI64IntegerAttr(type.getNumElements() / (packed ? 32 : 1))),
        r.getNamedAttr("address_stride", r.getI64IntegerAttr(1)),
        r.getNamedAttr("hemisphere", r.getStringAttr(hemisphere)),
    });
}

struct Storage {
    int64_t slice = 0;
    int64_t bank = 0;
    bool west = false;
};

mlir::LogicalResult lowerOne(mlir::IRRewriter& r, kernel::VectorAluOp op,
    const target::LPUTargetModel& target, Storage lhsStorage,
    Storage rhsStorage, Storage outStorage, bool packed = false, int64_t queueBase = 0)
{
    const auto lhsType = op.getLhs().getType();
    const auto rhsType = op.getRhs().getType();
    const auto outType = op.getResult().getType();
    if (lhsType.getNumElements() / (packed ? 32 : 1) > target.memory().sram_depth_rows)
        return op.emitError("vector scalar-broadcast storage exceeds SRAM; tile the vector before lowering");
    r.setInsertionPoint(op);
    const auto bind = [&](mlir::Value value, mlir::RankedTensorType type,
                          int64_t index, llvm::StringRef access,
                          Storage storage) {
        mlir::OperationState state(op.getLoc(), BindingOp::getOperationName());
        if (value) state.addOperands(value);
        state.addTypes(type);
        state.addAttributes({
            r.getNamedAttr("index", r.getI64IntegerAttr(index)),
            r.getNamedAttr("access", r.getStringAttr(access)),
            r.getNamedAttr("role", r.getStringAttr(access == "input" ? "activation" : "result")),
            r.getNamedAttr("name", r.getStringAttr("vector_alu")),
            r.getNamedAttr("bytes", r.getI64IntegerAttr(type.getNumElements()
                * (type.getElementType().isF32() ? 4 : 2))),
            r.getNamedAttr("placement", placement(r, type, storage.slice, storage.west ? "west" : "east", storage.bank, packed)),
        });
        return llvm::cast<BindingOp>(r.create(state));
    };
    const auto bindInput = [&](mlir::Value value, mlir::RankedTensorType type, Storage storage) {
        const auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
        if (!argument) return;
        for (auto binding : op->getBlock()->getOps<BindingOp>())
            if (binding.getAccess() == "input" && binding->getNumOperands() == 1
                && binding->getOperand(0) == value) return;
        bind(value, type, argument.getArgNumber(), "input", storage);
    };
    bindInput(op.getLhs(), lhsType, lhsStorage);
    if (op.getRhs() != op.getLhs())
        bindInput(op.getRhs(), rhsType, rhsStorage);
    const auto planes = [](mlir::RankedTensorType t) { return t.getElementType().isF32() ? 4 : 2; };
    const auto kind = [](mlir::RankedTensorType t) -> llvm::StringRef {
        return t.getElementType().isF32() ? "stream_f32" : lpu_16bit_stream_kind(t.getElementType());
    };
    const auto format = outType.getElementType().isF32()
        ? llvm::StringRef("fp32") : lpu_16bit_data_format(outType.getElementType());
    const int64_t tileCycles = 32 / target.throughput().mem_read_bytes_per_cycle;
    const bool special = op.getKind() == "exp" || op.getKind() == "reciprocal" || op.getKind() == "rsqrt";
    const int64_t aluColumn = op.getKind() == "exp" ? 1 : special ? 3 : 0;
    const int64_t lastColumn = op.getReduce() ? 1 : aluColumn + 1;
    const int64_t extraLatency = special ? 4 : op.getKind() == "multiply" ? 1 : 0;
    int64_t cursor = 96;
    const int64_t passes = 1;
    const int64_t reductionRows = op.getReduce() ? lhsType.getDimSize(0) : 1;
    const int64_t reductionRowInterval = lhsType.getDimSize(1) * tileCycles + 96;
    for (int64_t pass = 0; pass < passes; ++pass) {
        const int64_t count = (op.getReduce() ? lhsType.getDimSize(1) : lhsType.getNumElements()) / (packed ? 32 : 1);
        const auto read = [&](mlir::RankedTensorType type, Storage storage,
                              int64_t stream, bool rhs) {
            const int64_t columns = lhsType.getDimSize(1);
            const bool broadcastColumns = rhs && type.getDimSize(1) == 1 && columns > 1;
            const bool broadcastRows = rhs && type.getDimSize(0) == 1 && lhsType.getDimSize(0) > 1;
            const int64_t firstAddress = op.getReduce()
                ? (broadcastRows ? 0 : pass * type.getDimSize(1)) : 0;
            const int64_t inner = packed ? count : op.getReduce() ? count : columns;
            const int64_t waves = packed ? 1 : lhsType.getDimSize(0);
            for (int64_t byte = 0; byte < planes(type); ++byte) {
                const int64_t slice = storage.slice + byte;
                const auto latency = target.transport_latency(target::StreamEndpoint::Mem,
                    target::StreamEndpoint::VxmInput, target::StreamDirection::West, slice).value();
                direct_domain_detail::emitMem3D(r, op.getLoc(), target,
                    cursor - latency, slice + (storage.west ? target.memory().slices_per_hemisphere : 0),
                    "read", firstAddress,
                    target.streams().streams_per_direction + stream + byte,
                    inner, tileCycles, broadcastColumns ? 0 : 1,
                    waves, op.getReduce() ? reductionRowInterval : inner * tileCycles,
                    broadcastRows ? 0 : type.getDimSize(1),
                    1, 1, 0, -1, storage.bank);
            }
        };
        read(lhsType, lhsStorage, 0, false);
        if (!op.getUseImmediate() && op.getRhs() != op.getLhs()) read(rhsType, rhsStorage, 8, true);
        mlir::Value current = op.getLhs();
        for (int64_t col = 0; col <= lastColumn; ++col) {
            const bool arithmetic = col == aluColumn;
            const bool reduction = op.getReduce() && col == lastColumn;
            const auto opcode = reduction ? llvm::StringRef("add")
                : arithmetic ? op.getKind() : llvm::StringRef("cast");
            const auto lhsKind = col == 0 ? kind(lhsType) : llvm::StringRef("previous");
            const auto rhsKind = reduction ? llvm::StringRef("accumulator")
                : arithmetic && !op.getUseImmediate() ? kind(rhsType) : llvm::StringRef("immediate");
            auto vxm = ffn_detail::create_vxm_with_terminal_controls(r, op.getLoc(),
                current, op.getRhs(), outType, cursor - 1, queueBase + col, opcode,
                lhsKind, 0, 0.0f, rhsKind, rhsKind.starts_with("stream_") && op.getRhs() != op.getLhs() ? 8 : 0, op.getImmediate().convertToFloat(),
                col == lastColumn ? format : llvm::StringRef("fp32"),
                col == lastColumn ? 0 : -1, count, 1, "east", outStorage.west ? "east" : "west", -1,
                reduction, reduction, true, false, 8, reduction, reduction);
            if (col == 0) vxm->setAttr("lhs_stream_source", r.getStringAttr(lhsStorage.west ? "west" : "east"));
            if (arithmetic && !op.getUseImmediate())
                vxm->setAttr("rhs_stream_source", r.getStringAttr(rhsStorage.west ? "west" : "east"));
            if (op.getReduce()) {
                vxm->setAttr("wave_count", r.getI64IntegerAttr(reductionRows));
                vxm->setAttr("wave_interval", r.getI64IntegerAttr(reductionRowInterval));
            }
            current = vxm.getResult();
        }
        const int64_t resultCycle = cursor + (lhsStorage.west ? 32 - 2 * queueBase : 17) + extraLatency
            + (op.getReduce() ? (count - 1) * tileCycles : 0);
        for (int64_t byte = 0; byte < planes(outType); ++byte) {
            const int64_t slice = outStorage.slice + byte;
            const auto latency = target.transport_latency(target::StreamEndpoint::VxmResult,
                target::StreamEndpoint::Mem, target::StreamDirection::East, slice).value();
            direct_domain_detail::emitMem3D(r, op.getLoc(), target,
                resultCycle + latency, slice + (outStorage.west ? target.memory().slices_per_hemisphere : 0),
                "write", op.getReduce() ? pass : 0, byte,
                op.getReduce() ? 1 : count, tileCycles, 1,
                reductionRows, op.getReduce() ? reductionRowInterval : 1,
                op.getReduce() ? 1 : 0, 1, 1, 0, -1, outStorage.bank);
        }
        cursor += count * tileCycles + 96;
    }
    auto result = bind({}, outType, 0, "output", outStorage);
    r.replaceOp(op, result.getValue());
    return mlir::success();
}
} // namespace

mlir::LogicalResult lowerVectorAluSchedules(mlir::IRRewriter& r,
    mlir::func::FuncOp function, const target::LPUTargetModel& target)
{
    llvm::SmallVector<kernel::VectorAluOp> ops;
    function.walk([&](kernel::VectorAluOp op) { ops.push_back(op); });
    if (ops.empty()) return mlir::success();
    const auto queuePolicy=function->getAttrOfType<mlir::IntegerAttr>("ftlpu.vxm_queue_base");
    const int64_t queueBase=queuePolicy ? queuePolicy.getInt() : 0;
    if (queueBase != 0 && queueBase != 8) return function.emitError("VXM queue base must select one physical eight-ALU group");
    // Elementwise graphs can keep all 32 lanes of every physical word.
    // Reductions and broadcasts retain their explicit scalar-row layout.
    const bool packed = llvm::all_of(ops, [](auto op) {
        return !op.getReduce() && op.getLhs().getType().getDimSize(1) % 32 == 0
            && (op.getUseImmediate() || op.getLhs().getType().getShape() == op.getRhs().getType().getShape());
    });
    const bool graph = ops.size() > 1;
    if (!graph) {
        auto op = ops.front();
        if (!llvm::isa<mlir::BlockArgument>(op.getLhs())
            || !llvm::isa<mlir::BlockArgument>(op.getRhs()))
            return op.emitError("vector input has no scalar-broadcast producer placement");
        return lowerOne(r, op, target, {0, 0, false},
            op.getRhs() == op.getLhs() ? Storage{0, 0, false} : Storage{8, 0, false}, {48, 0, true}, packed, queueBase);
    }
    llvm::DenseMap<mlir::Value, Storage> storage;
    llvm::DenseMap<mlir::Value, int64_t> lastUse;
    for (auto [index, op] : llvm::enumerate(ops)) {
        lastUse[op.getLhs()] = index;
        if (!op.getUseImmediate()) lastUse[op.getRhs()] = index;
    }
    function.walk([&](mlir::func::ReturnOp ret) {
        for (auto value : ret.getOperands()) lastUse[value] = ops.size();
    });
    int64_t inputSlot = 0;
    for (auto op : ops)
        for (auto value : {op.getLhs(), op.getRhs()}) {
            if (!llvm::isa<mlir::BlockArgument>(value) || storage.contains(value)) continue;
            if (inputSlot >= 4) return op.emitError("vector graph currently supports at most four external inputs");
            storage[value] = {4 * inputSlot++, 0, true};
        }
    struct Live { mlir::Value value; int64_t slot; int64_t end; };
    llvm::SmallVector<Live> live;
    const int64_t slotsPerBank = (target.memory().slices_per_hemisphere - 16) / 4;
    const int64_t capacity = slotsPerBank * target.memory().banks_per_slice;
    if (slotsPerBank <= 0) return function.emitError("no vector graph scratch slice groups");
    for (auto [index, op] : llvm::enumerate(ops)) {
        llvm::erase_if(live, [&](const Live& item) { return item.end < static_cast<int64_t>(index); });
        int64_t slot = 0;
        for (; slot < capacity; ++slot)
            if (llvm::none_of(live, [&](const Live& item) { return item.slot == slot; })) break;
        if (slot == capacity) return op.emitError("vector graph live values exceed MEM capacity");
        if (!storage.contains(op.getLhs()) || !storage.contains(op.getRhs()))
            return op.emitError("vector graph operand is not a scalar-broadcast input or producer");
        const Storage output{16 + (slot % slotsPerBank) * 4, slot / slotsPerBank, true};
        const auto oldResult = op.getResult();
        const int64_t end = lastUse.lookup(oldResult);
        const auto lhs = storage.lookup(op.getLhs());
        const auto rhs = storage.lookup(op.getRhs());
        auto* before = op->getPrevNode();
        if (mlir::failed(lowerOne(r, op, target, lhs, rhs, output, packed, queueBase))) return mlir::failure();
        // lowerOne replaces the Kernel result with a physical binding. The
        // subsequent users now refer to that binding, not the erased result.
        auto* inserted = before ? before->getNextNode() : &function.getBody().front().front();
        BindingOp result;
        for (; inserted; inserted = inserted->getNextNode()) {
            // Stage identity comes from SSA operation order, independently
            // of source/debug locations (which can all be unknown).
            inserted->setAttr("ftlpu.schedule_stage", r.getI64IntegerAttr(index));
            auto binding = llvm::dyn_cast<BindingOp>(inserted);
            if (binding && binding.getAccess() == "output") { result = binding; break; }
        }
        if (!result) return function.emitError("vector graph result placement was not emitted");
        storage[result.getValue()] = output;
        live.push_back({result.getValue(), slot, end});
    }
    return mlir::success();
}
} // namespace ftlpu::compiler::schedule
