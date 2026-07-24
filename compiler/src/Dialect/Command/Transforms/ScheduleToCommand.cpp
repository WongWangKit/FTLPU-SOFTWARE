#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace ftlpu::compiler {
namespace {

llvm::SmallVector<int64_t> placement_slices(mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    for (mlir::Attribute attribute : placement.getAs<mlir::ArrayAttr>("slices"))
        result.push_back(llvm::cast<mlir::IntegerAttr>(attribute).getInt());
    return result;
}

int64_t placement_integer(mlir::DictionaryAttr placement, llvm::StringRef name)
{
    return placement.getAs<mlir::IntegerAttr>(name).getInt();
}

mlir::StringAttr placement_hemisphere(mlir::DictionaryAttr placement,
    mlir::DictionaryAttr address)
{
    if (auto hemisphere = placement.getAs<mlir::StringAttr>("hemisphere"))
        return hemisphere;
    return address.getAs<mlir::StringAttr>("hemisphere");
}

llvm::StringRef element_type_name(mlir::Type type)
{
    auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
    if (integer && integer.getWidth() == 8) return "i8";
    if (integer && integer.getWidth() == 32) return "i32";
    if (type.isF16()) return "f16";
    if (type.isF32()) return "f32";
    return "unsupported";
}

int64_t element_type_bytes(mlir::Type type)
{
    if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
        return (integer.getWidth() + 7) / 8;
    if (type.isF16()) return 2;
    if (type.isF32()) return 4;
    return 0;
}

void create_binding(mlir::OpBuilder& builder, mlir::Location location,
    int64_t index, llvm::StringRef access, llvm::StringRef role,
    mlir::RankedTensorType type, int64_t bytes, mlir::DictionaryAttr placement)
{
    llvm::SmallVector<mlir::Attribute> shape;
    for (int64_t dimension : type.getShape())
        shape.push_back(builder.getI64IntegerAttr(dimension));
    mlir::OperationState state(location, command::BindingOp::getOperationName());
    state.addAttributes({
        builder.getNamedAttr("index", builder.getI64IntegerAttr(index)),
        builder.getNamedAttr("access", builder.getStringAttr(access)),
        builder.getNamedAttr("role", builder.getStringAttr(role)),
        builder.getNamedAttr("shape", builder.getArrayAttr(shape)),
        builder.getNamedAttr("element_type", builder.getStringAttr(element_type_name(type.getElementType()))),
        builder.getNamedAttr("bytes", builder.getI64IntegerAttr(bytes)),
        builder.getNamedAttr("placement", placement),
    });
    builder.create(state);
}

void create_mem_command(mlir::OpBuilder& builder, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t address,
    int64_t packed_stream, int64_t repeat_count, int64_t repeat_interval,
    int64_t address_stride, llvm::StringRef accumulator_destination = "sram")
{
    mlir::OperationState state(location, command::MemOp::getOperationName());
    state.addAttributes({
        builder.getNamedAttr("cycle", builder.getI64IntegerAttr(cycle)),
        builder.getNamedAttr("queue", builder.getI64IntegerAttr(queue)),
        builder.getNamedAttr("opcode", builder.getStringAttr(opcode)),
        builder.getNamedAttr("address", builder.getI64IntegerAttr(address)),
        builder.getNamedAttr("packed_stream", builder.getI64IntegerAttr(packed_stream)),
        builder.getNamedAttr("repeat_count", builder.getI64IntegerAttr(repeat_count)),
        builder.getNamedAttr("repeat_interval", builder.getI64IntegerAttr(repeat_interval)),
        builder.getNamedAttr("address_stride", builder.getI64IntegerAttr(address_stride)),
        builder.getNamedAttr("accumulator_destination", builder.getStringAttr(accumulator_destination)),
    });
    builder.create(state);
}

void create_mxm_command(mlir::OpBuilder& builder, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t weight_buffer,
    int64_t weight_column, int64_t activation_stream_base, int64_t output_stream_base,
    int64_t repeat_count, int64_t repeat_interval)
{
    mlir::OperationState state(location, command::MxmOp::getOperationName());
    state.addAttributes({
        builder.getNamedAttr("cycle", builder.getI64IntegerAttr(cycle)),
        builder.getNamedAttr("queue", builder.getI64IntegerAttr(queue)),
        builder.getNamedAttr("opcode", builder.getStringAttr(opcode)),
        builder.getNamedAttr("weight_buffer", builder.getI64IntegerAttr(weight_buffer)),
        builder.getNamedAttr("weight_column", builder.getI64IntegerAttr(weight_column)),
        builder.getNamedAttr("activation_stream_base", builder.getI64IntegerAttr(activation_stream_base)),
        builder.getNamedAttr("output_stream_base", builder.getI64IntegerAttr(output_stream_base)),
        builder.getNamedAttr("repeat_count", builder.getI64IntegerAttr(repeat_count)),
        builder.getNamedAttr("repeat_interval", builder.getI64IntegerAttr(repeat_interval)),
    });
    builder.create(state);
}

void create_vxm_command(mlir::OpBuilder& builder, schedule::VxmOp op)
{
    mlir::OperationState state(op.getLoc(), command::VxmOp::getOperationName());
    for (llvm::StringRef name : {"cycle", "queue", "opcode", "lhs_kind",
             "lhs_index", "lhs_immediate", "rhs_kind", "rhs_index",
             "rhs_immediate", "cast_target", "output_stream", "repeat_count",
             "repeat_interval", "input_hemisphere", "output_hemisphere"})
        state.addAttribute(name, op->getAttr(name));
    builder.create(state);
}

void create_sxm_command(mlir::OpBuilder& builder, schedule::SxmOp op)
{
    mlir::OperationState state(op.getLoc(), command::SxmOp::getOperationName());
    for (llvm::StringRef name : {"cycle", "hemisphere", "opcode", "source_streams",
             "destination_streams", "permute_map", "weight_layout"})
        state.addAttribute(name, op->getAttr(name));
    builder.create(state);
}

void create_mem_transfer_command(mlir::OpBuilder& builder,
    schedule::MemTransferOp op, const target::LPUTargetModel& target)
{
    mlir::OperationState state(op.getLoc(), command::MemOp::getOperationName());
    for (llvm::StringRef name : {"cycle", "opcode", "address", "packed_stream",
             "repeat_count", "repeat_interval", "address_stride", "accumulator_destination"})
        state.addAttribute(name, op->getAttr(name));
    state.addAttribute("queue", builder.getI64IntegerAttr(
        op.getHemisphere() * target.memory().slices_per_hemisphere
        + op.getSlice()));
    builder.create(state);
}

void create_mxm_issue_command(mlir::OpBuilder& builder, schedule::MxmIssueOp op)
{
    mlir::OperationState state(op.getLoc(), command::MxmOp::getOperationName());
    for (llvm::StringRef name : {"cycle", "opcode", "weight_buffer",
             "weight_column", "activation_stream_base", "output_stream_base", "repeat_count", "repeat_interval"})
        state.addAttribute(name, op->getAttr(name));
    state.addAttribute("queue", builder.getI64IntegerAttr(op.getUnitId()));
    builder.create(state);
}

class ScheduleToCommandPass final
    : public mlir::PassWrapper<ScheduleToCommandPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleToCommandPass)

    llvm::StringRef getArgument() const final { return "ftlpu-schedule-to-command"; }
    llvm::StringRef getDescription() const final
    {
        return "Lowers cycle-accurate Schedule IR to ICU queue Command IR";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        bool has_commands = false;
        function.walk([&](command::MemOp) { has_commands = true; });
        function.walk([&](command::MxmOp) { has_commands = true; });
        function.walk([&](command::VxmOp) { has_commands = true; });
        function.walk([&](command::SxmOp) { has_commands = true; });
        if (has_commands) {
            function.emitError("Command IR has already been generated");
            signalPassFailure();
            return;
        }

        llvm::SmallVector<schedule::MemReadOp> reads;
        llvm::SmallVector<schedule::MxmLoadOp> loads;
        llvm::SmallVector<schedule::MxmComputeOp> computes;
        llvm::SmallVector<schedule::VxmOp> vxms;
        llvm::SmallVector<schedule::SxmOp> sxms;
        llvm::SmallVector<schedule::MemTransferOp> mem_transfers;
        llvm::SmallVector<schedule::MxmIssueOp> mxm_issues;
        llvm::SmallVector<schedule::AttentionOp> attention_schedules;
        llvm::SmallVector<schedule::MemWriteOp> writes;
        llvm::SmallVector<schedule::MemAccumulateOp> accumulates;
        llvm::SmallVector<mlir::Operation*> schedule_operations;
        function.walk([&](schedule::MemReadOp op) {
            reads.push_back(op);
            schedule_operations.push_back(op);
        });
        function.walk([&](schedule::MxmLoadOp op) {
            loads.push_back(op);
            schedule_operations.push_back(op);
        });
        function.walk([&](schedule::MxmComputeOp op) {
            computes.push_back(op);
            schedule_operations.push_back(op);
        });
        function.walk([&](schedule::VxmOp op) {
            vxms.push_back(op);
            schedule_operations.push_back(op);
        });
        function.walk([&](schedule::SxmOp op) {
            sxms.push_back(op);
            schedule_operations.push_back(op);
        });
        function.walk([&](schedule::MemTransferOp op) {
            mem_transfers.push_back(op);
            schedule_operations.push_back(op);
        });
        function.walk([&](schedule::MxmIssueOp op) {
            mxm_issues.push_back(op);
            schedule_operations.push_back(op);
        });
        function.walk([&](schedule::AttentionOp op) { attention_schedules.push_back(op); });
        function.walk([&](schedule::MemWriteOp op) {
            writes.push_back(op);
            schedule_operations.push_back(op);
        });
        function.walk([&](schedule::MemAccumulateOp op) {
            accumulates.push_back(op);
            schedule_operations.push_back(op);
        });
        if (schedule_operations.empty()) {
            function.emitError("requires Schedule IR operations");
            signalPassFailure();
            return;
        }

        const target::LPUTargetModel target;
        mlir::OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(&function.getBody().front());
        llvm::SmallDenseSet<unsigned> bound_inputs;
        bool causal_mask_bound = false;
        for (schedule::AttentionOp attention : attention_schedules) {
            const mlir::Value inputs[] = {attention.getInput(), attention.getQueryWeight(),
                attention.getKeyWeight(), attention.getValueWeight(), attention.getOutputWeight()};
            const char* placements[] = {"input", "query_weight", "key_weight",
                "value_weight", "output_weight"};
            for (unsigned index = 0; index < std::size(inputs); ++index) {
                auto argument = llvm::dyn_cast<mlir::BlockArgument>(inputs[index]);
                if (!argument || !bound_inputs.insert(argument.getArgNumber()).second) continue;
                auto type = llvm::cast<mlir::RankedTensorType>(argument.getType());
                create_binding(builder, attention.getLoc(), argument.getArgNumber(), "input",
                    index == 0 ? "activation" : "weight", type,
                    type.getNumElements() * element_type_bytes(type.getElementType()),
                    attention.getMemoryPlan().getAs<mlir::DictionaryAttr>(placements[index]));
            }
            auto result_type = llvm::cast<mlir::RankedTensorType>(attention.getResult().getType());
            create_binding(builder, attention.getLoc(), 0, "output", "result", result_type,
                result_type.getNumElements() * element_type_bytes(result_type.getElementType()),
                attention.getMemoryPlan().getAs<mlir::DictionaryAttr>("result"));
            if (attention.getCausal() && !causal_mask_bound) {
                const int64_t tile = target.throughput().mxm_rows;
                auto mask_type = mlir::RankedTensorType::get(
                    {tile - 1, tile}, builder.getF32Type());
                for (const char* placement : {"causal_mask", "causal_mask_mxm1"})
                    create_binding(builder, attention.getLoc(), 0, "internal", "constant",
                        mask_type, mask_type.getNumElements() * 4,
                        attention.getMemoryPlan().getAs<mlir::DictionaryAttr>(placement));
                causal_mask_bound = true;
            }
        }
        for (schedule::MemReadOp read : reads) {
            auto argument = llvm::dyn_cast<mlir::BlockArgument>(read.getInput());
            if (!argument || !bound_inputs.insert(argument.getArgNumber()).second) continue;
            auto type = llvm::cast<mlir::RankedTensorType>(argument.getType());
            const int64_t element_bytes = element_type_bytes(type.getElementType());
            const int64_t binding_bytes = type.getNumElements() * element_bytes;
            mlir::DictionaryAttr binding_placement =
                read.getPlacement().getAs<mlir::DictionaryAttr>("binding_placement");
            if (!binding_placement) binding_placement = read.getPlacement();
            create_binding(builder, read.getLoc(), argument.getArgNumber(), "input",
                argument.getArgNumber() == 0 ? "activation" : "weight",
                type, binding_bytes, binding_placement);
        }
        llvm::SmallDenseSet<mlir::Value> returned_values;
        function.walk([&](mlir::func::ReturnOp op) {
            for (mlir::Value value : op.getOperands()) returned_values.insert(value);
        });
        int64_t output_index = 0;
        for (schedule::MemWriteOp write : writes) {
            if (!returned_values.contains(write.getOutput())) continue;
            auto type = llvm::cast<mlir::RankedTensorType>(write.getOutput().getType());
            mlir::DictionaryAttr explicit_binding =
                write.getPlacement().getAs<mlir::DictionaryAttr>("binding_placement");
            mlir::NamedAttrList placement(
                explicit_binding ? explicit_binding : write.getPlacement());
            if (auto slices = write.getPlacement().getAs<mlir::ArrayAttr>("binding_slices"))
                placement.set("slices", slices);
            if (auto count = write.getPlacement().getAs<mlir::IntegerAttr>("binding_instruction_count"))
                placement.set("instruction_count", count);
            placement.set("base_row", builder.getI64IntegerAttr(0));
            create_binding(builder, write.getLoc(), output_index++,
                "output", "result", type,
                type.getNumElements() * element_type_bytes(type.getElementType()),
                placement.getDictionary(&getContext()));
        }

        for (schedule::MemReadOp read : reads) {
            const auto slices = placement_slices(read.getPlacement());
            const int64_t base_row = placement_integer(read.getPlacement(), "base_row");
            const int64_t count = placement_integer(read.getPlacement(), "instruction_count");
            const int64_t stride = placement_integer(read.getPlacement(), "address_stride");
            auto hemisphere = placement_hemisphere(read.getPlacement(), read.getAddress());
            if (!hemisphere || slices.empty()
                || (slices.size() != 1
                    && static_cast<int64_t>(slices.size()) != read.getStreamCount())) {
                read.emitError("MEM read placement does not match its producer streams");
                signalPassFailure();
                return;
            }
            const bool west_hemisphere = hemisphere.getValue() == "west";
            const bool west_stream = read.getDirection() == "west";
            builder.setInsertionPointAfter(read);
            int64_t max_latency = 0;
            for (int64_t slice : slices)
                max_latency = std::max(max_latency, *target.transport_latency(
                    target::StreamEndpoint::Mem,
                    read.getRole() == "weight" ? target::StreamEndpoint::MxmWeight
                                                : target::StreamEndpoint::MxmActivation,
                    target::StreamDirection::East, slice));
            const int64_t command_base = stride < 0
                ? base_row - (count - 1) * stride : base_row;
            for (size_t index = 0; index < slices.size(); ++index) {
                const int64_t latency = *target.transport_latency(
                    target::StreamEndpoint::Mem,
                    read.getRole() == "weight" ? target::StreamEndpoint::MxmWeight
                                                : target::StreamEndpoint::MxmActivation,
                    target::StreamDirection::East, slices[index]);
                create_mem_command(builder, read.getLoc(),
                    read.getCycle() + max_latency - latency,
                    (west_hemisphere ? target.memory().slices_per_hemisphere : 0)
                        + slices[index],
                    "read", command_base,
                    (west_stream ? 32 : 0) + read.getStreamBase()
                        + static_cast<int64_t>(index),
                    count, 1, stride);
            }
        }

        for (schedule::MxmLoadOp load : loads) {
            builder.setInsertionPointAfter(load);
            int64_t repeat_count = load.getDuration();
            if (auto read = load.getInput().getDefiningOp<schedule::MemReadOp>())
                repeat_count = placement_integer(read.getPlacement(), "instruction_count");
            for (int64_t column = 0; column < repeat_count; ++column) {
                // The four west-to-east weight pulses reach the MXM column
                // controls in reverse physical order.
                const int64_t weight_column = target.throughput().tile_rows - 1
                    - column % target.throughput().tile_rows;
                create_mxm_command(builder, load.getLoc(), load.getCycle() + column, load.getUnitId(),
                    "iw", load.getWeightBuffer(), weight_column,
                    0, 0, 1, 1);
            }
        }
        for (schedule::MxmComputeOp compute : computes) {
            builder.setInsertionPointAfter(compute);
            create_mxm_command(builder, compute.getLoc(), compute.getCycle(), compute.getUnitId(),
                "compute", compute.getWeightBuffer(), 0, compute.getActivationStreamBase(),
                compute.getOutputStreamBase(), compute.getDuration(), 1);
        }
        for (schedule::VxmOp vxm : vxms) {
            builder.setInsertionPointAfter(vxm);
            create_vxm_command(builder, vxm);
        }
        for (schedule::SxmOp sxm : sxms) {
            builder.setInsertionPointAfter(sxm);
            create_sxm_command(builder, sxm);
        }
        for (schedule::MemTransferOp mem : mem_transfers) {
            builder.setInsertionPointAfter(mem);
            create_mem_transfer_command(builder, mem, target);
        }
        for (schedule::MxmIssueOp mxm : mxm_issues) {
            builder.setInsertionPointAfter(mxm);
            create_mxm_issue_command(builder, mxm);
        }
        for (schedule::MemAccumulateOp accumulate : accumulates) {
            const auto slices = placement_slices(accumulate.getPlacement());
            const int64_t base_row = placement_integer(accumulate.getPlacement(), "base_row");
            const bool west = accumulate.getHemisphere() == "west";
            if (slices.size() != 4 || slices[1] != slices[0] + 1
                || slices[2] != slices[0] + 2 || slices[3] != slices[0] + 3) {
                accumulate.emitError("MEM accumulator must occupy one contiguous four-slice group");
                signalPassFailure();
                return;
            }
            builder.setInsertionPointAfter(accumulate);
            create_mem_command(builder, accumulate.getLoc(), accumulate.getCycle(),
                (west ? target.memory().slices_per_hemisphere : 0) + slices.front(),
                "accumulate", base_row, 32 + accumulate.getStreamBase(),
                accumulate.getRepeatCount(), accumulate.getRepeatInterval(),
                accumulate.getAddressStride(), accumulate.getDestination());
        }
        for (schedule::MemWriteOp write : writes) {
            const auto slices = placement_slices(write.getPlacement());
            const int64_t base_row = placement_integer(write.getPlacement(), "base_row");
            const int64_t count = placement_integer(write.getPlacement(), "instruction_count");
            const int64_t stride = placement_integer(write.getPlacement(), "address_stride");
            auto hemisphere = placement_hemisphere(write.getPlacement(), write.getAddress());
            if (!hemisphere
                || static_cast<int64_t>(slices.size()) != write.getStreamCount()) {
                write.emitError("result write placement does not match its producer streams");
                signalPassFailure();
                return;
            }
            const bool west_hemisphere = hemisphere.getValue() == "west";
            const bool west_stream = write.getDirection() == "west";
            builder.setInsertionPointAfter(write);
            for (size_t index = 0; index < slices.size(); ++index) {
                create_mem_command(builder, write.getLoc(),
                    write.getCycle(),
                    (west_hemisphere ? target.memory().slices_per_hemisphere : 0) + slices[index],
                    "write", base_row,
                    (west_stream ? 32 : 0) + write.getStreamBase() + static_cast<int64_t>(index),
                    count, 1, stride);
            }
        }

        function.walk([](mlir::func::ReturnOp op) {
            op->setOperands(mlir::ValueRange {});
        });
        function.setType(mlir::FunctionType::get(
            &getContext(), function.getArgumentTypes(), mlir::TypeRange {}));
        llvm::SmallVector<mlir::Operation*> ordered_schedule_ops;
        function.walk([&](mlir::Operation* op) {
            if (llvm::isa<schedule::MemReadOp, schedule::MxmLoadOp,
                    schedule::MxmComputeOp, schedule::VxmOp,
                    schedule::SxmOp,
                    schedule::MemTransferOp, schedule::MxmIssueOp,
                    schedule::MemAccumulateOp, schedule::MemWriteOp>(op))
                ordered_schedule_ops.push_back(op);
        });
        for (auto it = ordered_schedule_ops.rbegin(); it != ordered_schedule_ops.rend(); ++it)
            (*it)->erase();
        for (auto it = attention_schedules.rbegin(); it != attention_schedules.rend(); ++it)
            (*it)->erase();
        llvm::SmallVector<mlir::Operation*> dead_stream_ops;
        function.walk([&](mlir::Operation* op) {
            if (llvm::isa<stream::RouteOp, stream::DequantizeOp>(op))
                dead_stream_ops.push_back(op);
        });
        for (auto it = dead_stream_ops.rbegin(); it != dead_stream_ops.rend(); ++it)
            if ((*it)->use_empty()) (*it)->erase();
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_schedule_to_command_pass()
{
    return std::make_unique<ScheduleToCommandPass>();
}

} // namespace ftlpu::compiler
