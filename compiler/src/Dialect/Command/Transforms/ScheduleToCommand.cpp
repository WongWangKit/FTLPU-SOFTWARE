#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>

namespace ftlpu::compiler {
namespace {

constexpr int64_t kMaxRepeatCount = 1023;
constexpr int64_t kMaxRepeatInterval = 255;
constexpr int64_t kMinRepeatStride = -2048;
constexpr int64_t kMaxRepeatStride = 2047;

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

int64_t placement_integer_or(mlir::DictionaryAttr placement,
    llvm::StringRef name, int64_t fallback)
{
    if (auto value = placement.getAs<mlir::IntegerAttr>(name))
        return value.getInt();
    return fallback;
}

int64_t mem_queue(const target::LPUTargetModel& target,
    int64_t hemisphere, int64_t slice, int64_t bank)
{
    return hemisphere * target.memory().slices_per_hemisphere
            * target.memory().banks_per_slice
        + slice * target.memory().banks_per_slice + bank;
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
    if (type.isBF16()) return "bf16";
    if (type.isF32()) return "f32";
    return "unsupported";
}

int64_t element_type_bytes(mlir::Type type)
{
    if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
        return (integer.getWidth() + 7) / 8;
    if (is_lpu_16bit_float(type)) return 2;
    if (type.isF32()) return 4;
    return 0;
}

std::optional<int64_t> source_binding_index(mlir::Value value)
{
    llvm::SmallVector<mlir::Value> pending {value};
    llvm::SmallDenseSet<mlir::Value, 16> visited;
    std::optional<int64_t> result;
    while (!pending.empty()) {
        mlir::Value current = pending.pop_back_val();
        if (!visited.insert(current).second) continue;
        if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(current)) {
            const int64_t index = argument.getArgNumber();
            if (result && *result != index) return std::nullopt;
            result = index;
            continue;
        }
        mlir::Operation* defining = current.getDefiningOp();
        if (!defining) continue;
        if (auto binding = llvm::dyn_cast<schedule::BindingOp>(defining)) {
            if (binding.getAccess() != "input") continue;
            const int64_t index = binding.getIndex();
            if (result && *result != index) return std::nullopt;
            result = index;
            continue;
        }
        for (mlir::Value operand : defining->getOperands())
            pending.push_back(operand);
    }
    return result;
}

void create_binding(mlir::OpBuilder& builder, mlir::Location location,
    int64_t index, llvm::StringRef access, llvm::StringRef role,
    llvm::StringRef name, int64_t readyCycle, mlir::RankedTensorType type,
    int64_t bytes, mlir::DictionaryAttr placement,
    llvm::StringRef initializer = "none",
    mlir::DictionaryAttr initializerConfig = {})
{
    llvm::SmallVector<mlir::Attribute> shape;
    for (int64_t dimension : type.getShape())
        shape.push_back(builder.getI64IntegerAttr(dimension));
    mlir::OperationState state(location, command::BindingOp::getOperationName());
    state.addAttributes({
        builder.getNamedAttr("index", builder.getI64IntegerAttr(index)),
        builder.getNamedAttr("access", builder.getStringAttr(access)),
        builder.getNamedAttr("role", builder.getStringAttr(role)),
        builder.getNamedAttr("name", builder.getStringAttr(name)),
        builder.getNamedAttr(
            "ready_cycle", builder.getI64IntegerAttr(readyCycle)),
        builder.getNamedAttr(
            "initializer", builder.getStringAttr(initializer)),
        builder.getNamedAttr("initializer_config",
            initializerConfig
                ? initializerConfig
                : builder.getDictionaryAttr({})),
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
    int64_t address_stride, int64_t wave_count = 1,
    int64_t wave_interval = 1, int64_t wave_address_stride = 0,
    int64_t address_binding = -1)
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
    });
    if (wave_count > 1) {
        state.addAttribute(
            "wave_count", builder.getI64IntegerAttr(wave_count));
        state.addAttribute(
            "wave_interval", builder.getI64IntegerAttr(wave_interval));
        state.addAttribute("wave_address_stride",
            builder.getI64IntegerAttr(wave_address_stride));
    }
    if (address_binding >= 0)
        state.addAttribute("address_binding",
            builder.getI64IntegerAttr(address_binding));
    builder.create(state);
}

command::MxmOp create_mxm_command(
    mlir::OpBuilder& builder, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t weight_buffer,
    int64_t weight_column, int64_t activation_stream_base, int64_t output_stream_base,
    int64_t repeat_count, int64_t repeat_interval, int64_t accumulator_address,
    int64_t accumulator_row_stride, llvm::StringRef accumulator_destination,
    bool accumulator_clear = true, llvm::StringRef data_format = "fp16",
    llvm::StringRef accumulator_output_format = "fp32")
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
        builder.getNamedAttr("accumulator_address", builder.getI64IntegerAttr(accumulator_address)),
        builder.getNamedAttr("accumulator_row_stride", builder.getI64IntegerAttr(accumulator_row_stride)),
        builder.getNamedAttr("accumulator_destination", builder.getStringAttr(accumulator_destination)),
        builder.getNamedAttr("accumulator_clear", builder.getBoolAttr(accumulator_clear)),
        builder.getNamedAttr("data_format", builder.getStringAttr(data_format)),
        builder.getNamedAttr("accumulator_output_format",
            builder.getStringAttr(accumulator_output_format)),
    });
    return llvm::cast<command::MxmOp>(builder.create(state));
}

void create_vxm_command(mlir::OpBuilder& builder, schedule::VxmOp op,
    int64_t repeatCount = -1, int64_t repeatInterval = -1)
{
    mlir::OperationState state(op.getLoc(), command::VxmOp::getOperationName());
      for (llvm::StringRef name : {"cycle", "queue", "opcode", "lhs_kind",
               "lhs_index", "lhs_immediate", "lhs_stream_source",
               "rhs_kind", "rhs_index", "rhs_stream_source",
               "rhs_immediate", "cast_target", "output_stream", "repeat_count",
               "repeat_interval", "input_hemisphere", "output_hemisphere",
               "accumulator_reset", "accumulator_write", "accumulator_emit",
               "local_scalar_write"})
          if (mlir::Attribute attribute = op->getAttr(name))
              state.addAttribute(name, attribute);
    state.addAttribute("chain_depth",
        op->getAttr("chain_depth")
            ? op->getAttr("chain_depth")
            : builder.getI64IntegerAttr(8));
    if (auto scaleBinding = op.getScaleBindingAttr())
        state.addAttribute("scale_binding", scaleBinding);
    if (repeatCount >= 0)
        state.attributes.set("repeat_count",
            builder.getI64IntegerAttr(repeatCount));
    if (repeatInterval >= 0)
        state.attributes.set("repeat_interval",
            builder.getI64IntegerAttr(repeatInterval));
    builder.create(state);
}

bool same_vxm_command(schedule::VxmOp lhs, schedule::VxmOp rhs)
{
    for (llvm::StringRef name : {"queue", "opcode", "chain_depth", "lhs_kind",
             "lhs_index", "lhs_immediate", "lhs_stream_source",
             "rhs_kind", "rhs_index", "rhs_stream_source",
             "rhs_immediate", "cast_target", "output_stream",
             "input_hemisphere", "output_hemisphere", "scale_binding",
             "accumulator_reset", "accumulator_write", "accumulator_emit",
             "local_scalar_write"}) {
        if (lhs->getAttr(name) != rhs->getAttr(name)) return false;
    }
    return lhs.getRepeatCount() == 1 && rhs.getRepeatCount() == 1;
}

void create_sxm_command(mlir::OpBuilder& builder, schedule::SxmOp op)
{
    mlir::OperationState state(op.getLoc(), command::SxmOp::getOperationName());
    for (llvm::StringRef name : {"cycle", "hemisphere", "opcode", "source_streams",
               "destination_streams", "permute_map", "weight_layout",
               "output_row", "input_row", "output_tile"})
        if (op->getAttr(name))
        state.addAttribute(name, op->getAttr(name));
    state.addAttribute("repeat_count", builder.getI64IntegerAttr(
        op.getRepeatCount().value_or(1)));
    state.addAttribute("repeat_interval", builder.getI64IntegerAttr(
        op.getRepeatInterval().value_or(1)));
    builder.create(state);
}

void create_mem_transfer_command(mlir::OpBuilder& builder,
    schedule::MemTransferOp op, const target::LPUTargetModel& target)
{
    mlir::OperationState state(op.getLoc(), command::MemOp::getOperationName());
    for (llvm::StringRef name : {"cycle", "opcode", "address", "packed_stream",
             "repeat_count", "repeat_interval", "address_stride"})
        state.addAttribute(name, op->getAttr(name));
    state.addAttribute("queue", builder.getI64IntegerAttr(
        mem_queue(target, op.getHemisphere(), op.getSlice(),
            op.getBank().value_or(0))));
    for (llvm::StringRef name :
        {"wave_count", "wave_interval", "wave_address_stride",
            "address_binding"})
        if (mlir::Attribute attribute = op->getAttr(name))
            state.addAttribute(name, attribute);
    builder.create(state);
}

void create_mxm_issue_command(mlir::OpBuilder& builder, schedule::MxmIssueOp op)
{
    mlir::OperationState state(op.getLoc(), command::MxmOp::getOperationName());
    for (llvm::StringRef name : {"cycle", "opcode", "weight_buffer",
             "weight_column", "activation_stream_base", "output_stream_base",
             "repeat_count", "repeat_interval", "accumulator_address",
             "accumulator_row_stride", "accumulator_destination",
             "accumulator_clear"})
        state.addAttribute(name, op->getAttr(name));
    state.addAttribute("data_format", op.getDataFormatAttr()
            ? op.getDataFormatAttr()
            : builder.getStringAttr("fp16"));
    state.addAttribute("accumulator_output_format",
        op.getAccumulatorOutputFormatAttr()
            ? op.getAccumulatorOutputFormatAttr()
            : builder.getStringAttr("fp32"));
    state.addAttribute("queue", builder.getI64IntegerAttr(op.getUnitId()));
    for (llvm::StringRef name :
        {"weight_load_mode", "weight_inner_column",
            "weight_input_mode", "weight_stream_base", "wave_count",
            "wave_interval", "wave_weight_column_stride",
            "wave_accumulator_address_stride",
            "group_count", "group_interval"})
        if (mlir::Attribute attribute = op->getAttr(name))
            state.addAttribute(name, attribute);
    builder.create(state);
}

void create_mxm_dequant_command(
    mlir::OpBuilder& builder, schedule::MxmDequantOp op)
{
    mlir::OperationState state(
        op.getLoc(), command::MxmDequantOp::getOperationName());
    state.addAttributes({
        builder.getNamedAttr("cycle", op.getCycleAttr()),
        builder.getNamedAttr(
            "queue", builder.getI64IntegerAttr(op.getUnitId())),
        builder.getNamedAttr("scale", op.getScaleAttr()),
        builder.getNamedAttr(
            "repeat_count", op.getRepeatCountAttr()),
        builder.getNamedAttr(
            "repeat_interval", op.getRepeatIntervalAttr()),
    });
    for (llvm::StringRef name : {"wave_count", "wave_interval"})
        if (mlir::Attribute attribute = op->getAttr(name))
            state.addAttribute(name, attribute);
    if (auto scaleBinding = op.getScaleBindingAttr())
        state.addAttribute("scale_binding", scaleBinding);
    builder.create(state);
}

void create_mem_bundle_command(mlir::OpBuilder& builder,
    mlir::Location location, llvm::ArrayRef<int64_t> cycles,
    llvm::ArrayRef<int64_t> queues, llvm::StringRef opcode,
    llvm::ArrayRef<int64_t> addresses,
    llvm::ArrayRef<int64_t> packedStreams, int64_t repeatCount,
    int64_t repeatInterval, int64_t addressStride,
    int64_t waveCount = 1, int64_t waveInterval = 1,
    int64_t waveAddressStride = 0, int64_t addressBinding = -1)
{
    const auto array = [&](llvm::ArrayRef<int64_t> values) {
        llvm::SmallVector<mlir::Attribute> attributes;
        attributes.reserve(values.size());
        for (int64_t value : values)
            attributes.push_back(builder.getI64IntegerAttr(value));
        return builder.getArrayAttr(attributes);
    };
    mlir::OperationState state(
        location, command::MemBundleOp::getOperationName());
    state.addAttributes({
        builder.getNamedAttr("cycles", array(cycles)),
        builder.getNamedAttr("queues", array(queues)),
        builder.getNamedAttr("opcode", builder.getStringAttr(opcode)),
        builder.getNamedAttr("addresses", array(addresses)),
        builder.getNamedAttr("packed_streams", array(packedStreams)),
        builder.getNamedAttr(
            "repeat_count", builder.getI64IntegerAttr(repeatCount)),
        builder.getNamedAttr(
            "repeat_interval", builder.getI64IntegerAttr(repeatInterval)),
        builder.getNamedAttr(
            "address_stride", builder.getI64IntegerAttr(addressStride)),
    });
    if (waveCount > 1) {
        state.addAttribute(
            "wave_count", builder.getI64IntegerAttr(waveCount));
        state.addAttribute(
            "wave_interval", builder.getI64IntegerAttr(waveInterval));
        state.addAttribute("wave_address_stride",
            builder.getI64IntegerAttr(waveAddressStride));
    }
    if (addressBinding >= 0)
        state.addAttribute("address_binding",
            builder.getI64IntegerAttr(addressBinding));
    builder.create(state);
}

mlir::DictionaryAttr command_pattern_attributes(mlir::Operation* operation,
    mlir::MLIRContext* context, llvm::ArrayRef<llvm::StringRef> excluded)
{
    llvm::SmallVector<mlir::NamedAttribute> attributes;
    for (mlir::NamedAttribute attribute : operation->getAttrs())
        if (!llvm::is_contained(excluded,
                attribute.getName().strref()))
            attributes.push_back(attribute);
    return mlir::DictionaryAttr::get(context, attributes);
}

void compress_mxm_command_waves(
    mlir::func::FuncOp function, mlir::Builder& builder)
{
    struct Group {
        mlir::DictionaryAttr key;
        llvm::SmallVector<command::MxmOp> operations;
    };
    std::map<const void*, Group> groups;
    function.walk([&](command::MxmOp op) {
        if (op.getOpcode() != "compute"
            || op.getWaveCount().value_or(1) != 1
            || op.getGroupCount().value_or(1) != 1)
            return;
        auto key = command_pattern_attributes(op, function.getContext(),
            {"cycle", "accumulator_address", "wave_count",
                "wave_interval", "wave_accumulator_address_stride"});
        auto& group = groups[key.getAsOpaquePointer()];
        group.key = key;
        group.operations.push_back(op);
    });

    llvm::SmallVector<command::MxmOp> toErase;
    for (auto& [key, group] : groups) {
        (void)key;
        auto& operations = group.operations;
        llvm::sort(operations,
            [](command::MxmOp lhs, command::MxmOp rhs) {
                return lhs.getCycle() < rhs.getCycle();
            });
        std::map<std::pair<int64_t, int64_t>, std::size_t> lookup;
        for (std::size_t index = 0; index < operations.size(); ++index)
            lookup.emplace(
                std::pair {static_cast<int64_t>(operations[index].getCycle()),
                    static_cast<int64_t>(
                        operations[index].getAccumulatorAddress())},
                index);
        llvm::SmallVector<bool> used(operations.size(), false);
        for (std::size_t seed = 0; seed < operations.size(); ++seed) {
            if (used[seed]) continue;
            llvm::SmallVector<std::size_t> best {seed};
            const int64_t seedCycle = operations[seed].getCycle();
            const int64_t seedAddress =
                operations[seed].getAccumulatorAddress();
            const std::size_t candidateEnd = std::min(
                operations.size(), seed + std::size_t {17});
            for (std::size_t candidate = seed + 1;
                 candidate < candidateEnd; ++candidate) {
                if (used[candidate]) continue;
                const int64_t interval =
                    operations[candidate].getCycle() - seedCycle;
                const int64_t stride =
                    operations[candidate].getAccumulatorAddress()
                    - seedAddress;
                if (interval <= 0 || interval > 65535
                    || stride < -32768 || stride > 32767)
                    continue;
                llvm::SmallVector<std::size_t> sequence {seed};
                for (int64_t wave = 1; wave < 1023; ++wave) {
                    auto found = lookup.find(
                        {seedCycle + wave * interval,
                            seedAddress + wave * stride});
                    if (found == lookup.end() || used[found->second])
                        break;
                    sequence.push_back(found->second);
                }
                if (sequence.size() > best.size())
                    best = std::move(sequence);
            }
            if (best.size() < 2) {
                used[seed] = true;
                continue;
            }
            command::MxmOp representative = operations[best.front()];
            representative->setAttr("wave_count",
                builder.getI64IntegerAttr(best.size()));
            representative->setAttr("wave_interval",
                builder.getI64IntegerAttr(
                    operations[best[1]].getCycle() - seedCycle));
            representative->setAttr("wave_accumulator_address_stride",
                builder.getI64IntegerAttr(
                    operations[best[1]].getAccumulatorAddress()
                    - seedAddress));
            for (std::size_t index : best) {
                used[index] = true;
                if (index != best.front())
                    toErase.push_back(operations[index]);
            }
        }
    }
    for (command::MxmOp operation : toErase) operation.erase();
}

void compress_mxm_command_groups(
    mlir::func::FuncOp function, mlir::Builder& builder)
{
    struct Group {
        mlir::DictionaryAttr key;
        llvm::SmallVector<command::MxmOp> operations;
    };
    std::map<const void*, Group> groups;
    function.walk([&](command::MxmOp op) {
        if (op.getRepeatCount() != 1
            || op.getGroupCount().value_or(1) != 1)
            return;
        auto key = command_pattern_attributes(op, function.getContext(),
            {"cycle", "group_count", "group_interval"});
        auto& group = groups[key.getAsOpaquePointer()];
        group.key = key;
        group.operations.push_back(op);
    });

    llvm::SmallVector<command::MxmOp> toErase;
    for (auto& [key, group] : groups) {
        (void)key;
        auto& operations = group.operations;
        llvm::sort(operations,
            [](command::MxmOp lhs, command::MxmOp rhs) {
                return lhs.getCycle() < rhs.getCycle();
            });
        llvm::SmallVector<bool> used(operations.size(), false);
        std::map<int64_t, std::size_t> lookup;
        for (std::size_t index = 0; index < operations.size(); ++index)
            lookup.emplace(operations[index].getCycle(), index);
        for (std::size_t seed = 0; seed < operations.size(); ++seed) {
            if (used[seed]) continue;
            llvm::SmallVector<std::size_t> best {seed};
            const int64_t seedCycle = operations[seed].getCycle();
            const std::size_t candidateEnd = std::min(
                operations.size(), seed + std::size_t {17});
            for (std::size_t candidate = seed + 1;
                 candidate < candidateEnd; ++candidate) {
                if (used[candidate]) continue;
                const int64_t interval =
                    operations[candidate].getCycle() - seedCycle;
                if (interval <= 0 || interval > 65535) continue;
                llvm::SmallVector<std::size_t> sequence {seed};
                for (int64_t groupIndex = 1;
                     groupIndex < 1023; ++groupIndex) {
                    auto found = lookup.find(
                        seedCycle + groupIndex * interval);
                    if (found == lookup.end() || used[found->second]) break;
                    sequence.push_back(found->second);
                }
                if (sequence.size() > best.size())
                    best = std::move(sequence);
            }
            if (best.size() < 2) {
                used[seed] = true;
                continue;
            }
            command::MxmOp representative = operations[best.front()];
            representative->setAttr("group_count",
                builder.getI64IntegerAttr(best.size()));
            representative->setAttr("group_interval",
                builder.getI64IntegerAttr(
                    operations[best[1]].getCycle() - seedCycle));
            for (std::size_t index : best) {
                used[index] = true;
                if (index != best.front())
                    toErase.push_back(operations[index]);
            }
        }
    }
    for (command::MxmOp operation : toErase) operation.erase();
}

void compress_mxm_dequant_command_waves(
    mlir::func::FuncOp function, mlir::Builder& builder)
{
    struct Group {
        mlir::DictionaryAttr key;
        llvm::SmallVector<command::MxmDequantOp> operations;
    };
    std::map<const void*, Group> groups;
    function.walk([&](command::MxmDequantOp op) {
        if (op.getWaveCount().value_or(1) != 1) return;
        auto key = command_pattern_attributes(op, function.getContext(),
            {"cycle", "wave_count", "wave_interval"});
        auto& group = groups[key.getAsOpaquePointer()];
        group.key = key;
        group.operations.push_back(op);
    });
    llvm::SmallVector<command::MxmDequantOp> toErase;
    for (auto& [key, group] : groups) {
        (void)key;
        auto& operations = group.operations;
        llvm::sort(operations,
            [](command::MxmDequantOp lhs,
                command::MxmDequantOp rhs) {
                return lhs.getCycle() < rhs.getCycle();
            });
        std::map<int64_t, std::size_t> lookup;
        for (std::size_t index = 0; index < operations.size(); ++index)
            lookup.emplace(operations[index].getCycle(), index);
        llvm::SmallVector<bool> used(operations.size(), false);
        for (std::size_t seed = 0; seed < operations.size(); ++seed) {
            if (used[seed]) continue;
            llvm::SmallVector<std::size_t> best {seed};
            const int64_t seedCycle = operations[seed].getCycle();
            const std::size_t candidateEnd = std::min(
                operations.size(), seed + std::size_t {17});
            for (std::size_t candidate = seed + 1;
                 candidate < candidateEnd; ++candidate) {
                if (used[candidate]) continue;
                const int64_t interval =
                    operations[candidate].getCycle() - seedCycle;
                if (interval <= 0 || interval > 65535) continue;
                llvm::SmallVector<std::size_t> sequence {seed};
                for (int64_t wave = 1; wave < 1023; ++wave) {
                    auto found = lookup.find(seedCycle + wave * interval);
                    if (found == lookup.end() || used[found->second]) break;
                    sequence.push_back(found->second);
                }
                if (sequence.size() > best.size())
                    best = std::move(sequence);
            }
            if (best.size() < 2) {
                used[seed] = true;
                continue;
            }
            command::MxmDequantOp representative =
                operations[best.front()];
            representative->setAttr("wave_count",
                builder.getI64IntegerAttr(best.size()));
            representative->setAttr("wave_interval",
                builder.getI64IntegerAttr(
                    operations[best[1]].getCycle() - seedCycle));
            for (std::size_t index : best) {
                used[index] = true;
                if (index != best.front())
                    toErase.push_back(operations[index]);
            }
        }
    }
    for (command::MxmDequantOp operation : toErase)
        operation.erase();
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
        function.walk([&](command::MemBundleOp) {
            has_commands = true;
        });
        function.walk([&](command::MxmOp) { has_commands = true; });
        function.walk([&](command::MxmDequantOp) {
            has_commands = true;
        });
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
        llvm::SmallVector<schedule::MxmAccumulatorReadOp> accumulator_reads;
        llvm::SmallVector<schedule::VxmOp> vxms;
        llvm::SmallVector<schedule::SxmOp> sxms;
        llvm::SmallVector<schedule::MemTransferOp> mem_transfers;
        llvm::SmallVector<schedule::MxmIssueOp> mxm_issues;
        llvm::SmallVector<schedule::MxmDequantOp> mxm_dequants;
        llvm::SmallVector<schedule::BindingOp> bindings;
        llvm::SmallVector<schedule::TimelineOp> timelines;
        llvm::SmallVector<schedule::MemWriteOp> writes;
        llvm::SmallVector<schedule::MxmAccumulateOp> accumulates;
        function.walk([&](schedule::MemReadOp op) {
            reads.push_back(op);
        });
        function.walk([&](schedule::MxmLoadOp op) {
            loads.push_back(op);
        });
        function.walk([&](schedule::MxmComputeOp op) {
            computes.push_back(op);
        });
        function.walk([&](schedule::MxmAccumulatorReadOp op) {
            accumulator_reads.push_back(op);
        });
        function.walk([&](schedule::VxmOp op) {
            vxms.push_back(op);
        });
        function.walk([&](schedule::SxmOp op) {
            sxms.push_back(op);
        });
        function.walk([&](schedule::MemTransferOp op) {
            mem_transfers.push_back(op);
        });
        function.walk([&](schedule::MxmIssueOp op) {
            mxm_issues.push_back(op);
        });
        function.walk([&](schedule::MxmDequantOp op) {
            mxm_dequants.push_back(op);
        });
        function.walk([&](schedule::BindingOp op) {
            bindings.push_back(op);
        });
        function.walk([&](schedule::TimelineOp op) {
            timelines.push_back(op);
        });
        function.walk([&](schedule::MemWriteOp op) {
            writes.push_back(op);
        });
        function.walk([&](schedule::MxmAccumulateOp op) {
            accumulates.push_back(op);
        });
        if (reads.empty() && loads.empty() && computes.empty()
            && accumulator_reads.empty() && vxms.empty() && sxms.empty()
            && mem_transfers.empty() && mxm_issues.empty()
            && mxm_dequants.empty()
            && bindings.empty() && timelines.empty() && writes.empty()
            && accumulates.empty()) {
            function.emitError("requires Schedule IR operations");
            signalPassFailure();
            return;
        }

        auto target_model =
            target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target_model)) {
            signalPassFailure();
            return;
        }
        const target::LPUTargetModel& target = *target_model;
        mlir::OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(&function.getBody().front());
        llvm::SmallDenseSet<unsigned> bound_inputs;
        int64_t outputReadyCycle = 0;
        for (schedule::MemWriteOp write : writes)
            outputReadyCycle = std::max(
                outputReadyCycle,
                static_cast<int64_t>(
                    write.getCycle()
                    + (write.getWaveCount().value_or(1) - 1)
                        * write.getWaveInterval().value_or(1)
                    + write.getDuration()));
        for (schedule::BindingOp binding : bindings) {
            if (binding.getAccess() == "input"
                && !bound_inputs.insert(binding.getIndex()).second)
                continue;
            auto type = llvm::cast<mlir::RankedTensorType>(
                binding.getValue().getType());
            create_binding(builder, binding.getLoc(), binding.getIndex(),
                binding.getAccess(), binding.getRole(),
                binding.getName().value_or(binding.getRole()),
                binding.getReadyCycle().value_or(
                    binding.getAccess() == "output"
                        ? outputReadyCycle : 0),
                type,
                binding.getBytes(), binding.getPlacement(),
                binding.getInitializer().value_or("none"),
                binding.getInitializerConfig().value_or(
                    builder.getDictionaryAttr({})));
        }
        for (schedule::TimelineOp timeline : timelines) {
            mlir::OperationState state(
                timeline.getLoc(), command::TimelineOp::getOperationName());
            state.addAttributes({
                builder.getNamedAttr("name", timeline.getNameAttr()),
                builder.getNamedAttr("start", timeline.getStartAttr()),
                builder.getNamedAttr("end", timeline.getEndAttr()),
            });
            builder.create(state);
        }
        struct WeightPageInterval {
            int64_t binding;
            int64_t page;
            int64_t bank;
            int64_t ready{std::numeric_limits<int64_t>::max()};
            int64_t release{0};
            mlir::Location location;
        };
        std::map<std::tuple<int64_t, int64_t, int64_t>,
            WeightPageInterval> weightPages;
        for (schedule::MemReadOp read : reads) {
            const auto page = read.getPlacement()
                .getAs<mlir::IntegerAttr>("weight_page");
            if (!page) continue;
            const auto binding = source_binding_index(read.getInput());
            if (!binding) {
                read.emitError(
                    "paged weight read must resolve to one input binding");
                signalPassFailure();
                return;
            }
            const int64_t bank = placement_integer_or(
                read.getPlacement(), "bank", 0);
            auto [position, inserted] = weightPages.try_emplace(
                std::tuple {*binding, page.getInt(), bank},
                WeightPageInterval {*binding, page.getInt(), bank,
                    std::numeric_limits<int64_t>::max(), 0,
                    read.getLoc()});
            auto& interval = position->second;
            interval.ready = std::min<int64_t>(
                interval.ready, read.getCycle());
            interval.release = std::max<int64_t>(interval.release,
                read.getCycle()
                    + (read.getWaveCount().value_or(1) - 1)
                        * read.getWaveInterval().value_or(1)
                    + read.getDuration());
        }
        for (schedule::MemTransferOp transfer : mem_transfers) {
            const auto page = transfer.getWeightPage();
            if (!page) continue;
            const auto binding = transfer.getAddressBinding();
            if (!binding) {
                transfer.emitError(
                    "paged weight transfer must resolve to one input binding");
                signalPassFailure();
                return;
            }
            const int64_t bindingIndex = static_cast<int64_t>(*binding);
            const int64_t pageIndex = static_cast<int64_t>(*page);
            const int64_t bank = transfer.getBank().value_or(0);
            auto [position, inserted] = weightPages.try_emplace(
                std::tuple {bindingIndex, pageIndex, bank},
                WeightPageInterval {bindingIndex, pageIndex, bank,
                    std::numeric_limits<int64_t>::max(), 0,
                    transfer.getLoc()});
            auto& interval = position->second;
            const int64_t endCycle = transfer.getCycle()
                + (transfer.getWaveCount().value_or(1) - 1)
                    * transfer.getWaveInterval().value_or(1)
                + (transfer.getRepeatCount() - 1)
                    * transfer.getRepeatInterval()
                + 1;
            interval.ready = std::min<int64_t>(
                interval.ready, transfer.getCycle());
            interval.release = std::max<int64_t>(
                interval.release, endCycle);
        }
        builder.setInsertionPointToStart(&function.getBody().front());
        for (const auto& [key, interval] : weightPages) {
            (void)key;
            mlir::OperationState state(interval.location,
                command::WeightPageOp::getOperationName());
            state.addAttributes({
                builder.getNamedAttr("binding_index",
                    builder.getI64IntegerAttr(interval.binding)),
                builder.getNamedAttr("page_index",
                    builder.getI64IntegerAttr(interval.page)),
                builder.getNamedAttr("bank",
                    builder.getI64IntegerAttr(interval.bank)),
                builder.getNamedAttr("ready_cycle",
                    builder.getI64IntegerAttr(interval.ready)),
                builder.getNamedAttr("release_cycle",
                    builder.getI64IntegerAttr(interval.release)),
            });
            builder.create(state);
        }
        for (schedule::MemReadOp read : reads) {
            auto argument =
                llvm::dyn_cast<mlir::BlockArgument>(read.getInput());
            if (!argument
                || !bound_inputs.insert(
                    argument.getArgNumber()).second)
                continue;
            auto type = llvm::cast<mlir::RankedTensorType>(
                argument.getType());
            const int64_t bindingBytes = type.getNumElements()
                * element_type_bytes(type.getElementType());
            mlir::DictionaryAttr bindingPlacement =
                read.getPlacement().getAs<mlir::DictionaryAttr>(
                    "binding_placement");
            if (!bindingPlacement)
                bindingPlacement = read.getPlacement();
            create_binding(builder, read.getLoc(),
                argument.getArgNumber(), "input",
                argument.getArgNumber() == 0
                    ? "activation"
                    : "weight",
                "arg" + std::to_string(argument.getArgNumber()), 0,
                type, bindingBytes, bindingPlacement);
        }
        llvm::SmallDenseSet<mlir::Value> returnedValues;
        function.walk([&](mlir::func::ReturnOp op) {
            for (mlir::Value value : op.getOperands())
                returnedValues.insert(value);
        });
        int64_t outputIndex = 0;
        for (schedule::MemWriteOp write : writes) {
            if (!returnedValues.contains(write.getOutput()))
                continue;
            auto type = llvm::cast<mlir::RankedTensorType>(
                write.getOutput().getType());
            mlir::DictionaryAttr explicitBinding =
                write.getPlacement().getAs<mlir::DictionaryAttr>(
                    "binding_placement");
            mlir::NamedAttrList placement(explicitBinding
                    ? explicitBinding
                    : write.getPlacement());
            if (auto slices =
                    write.getPlacement().getAs<mlir::ArrayAttr>(
                        "binding_slices"))
                placement.set("slices", slices);
            if (auto count =
                    write.getPlacement().getAs<mlir::IntegerAttr>(
                        "binding_instruction_count"))
                placement.set("instruction_count", count);
            placement.set(
                "base_row", builder.getI64IntegerAttr(0));
            create_binding(builder, write.getLoc(), outputIndex++,
                "output", "result", "result",
                write.getCycle()
                    + (write.getWaveCount().value_or(1) - 1)
                        * write.getWaveInterval().value_or(1)
                    + write.getDuration(), type,
                type.getNumElements()
                    * element_type_bytes(type.getElementType()),
                placement.getDictionary(&getContext()));
        }
        llvm::SmallVector<int64_t> loadRepeatCounts;
        loadRepeatCounts.reserve(loads.size());
        for (schedule::MxmLoadOp load : loads) {
            int64_t repeatCount = load.getDuration();
            if (auto read =
                    load.getInput()
                        .getDefiningOp<schedule::MemReadOp>())
                repeatCount = placement_integer(
                    read.getPlacement(), "instruction_count");
            loadRepeatCounts.push_back(repeatCount);
        }
        llvm::sort(mem_transfers,
            [&](schedule::MemTransferOp lhs,
                schedule::MemTransferOp rhs) {
                const int64_t lhsQueue = lhs.getHemisphere()
                        * target.memory().slices_per_hemisphere
                        * target.memory().banks_per_slice
                    + lhs.getSlice() * target.memory().banks_per_slice
                    + lhs.getBank().value_or(0);
                const int64_t rhsQueue = rhs.getHemisphere()
                        * target.memory().slices_per_hemisphere
                        * target.memory().banks_per_slice
                    + rhs.getSlice() * target.memory().banks_per_slice
                    + rhs.getBank().value_or(0);
                return lhsQueue != rhsQueue
                    ? lhsQueue < rhsQueue
                    : lhs.getCycle() < rhs.getCycle();
            });
        for (std::size_t index = 0;
             index < mem_transfers.size();) {
            schedule::MemTransferOp first = mem_transfers[index];
            const int64_t queue = mem_queue(target, first.getHemisphere(),
                first.getSlice(),
                first.getBank().value_or(0));
            std::size_t end = index + 1;
            int64_t interval = 1;
            int64_t stride = first.getAddressStride();
            const int64_t firstWaveCount =
                first->getAttrOfType<mlir::IntegerAttr>("wave_count")
                ? first->getAttrOfType<mlir::IntegerAttr>("wave_count").getInt()
                : 1;
            if (first.getRepeatCount() == 1 && firstWaveCount == 1
                && end < mem_transfers.size()) {
                schedule::MemTransferOp second = mem_transfers[end];
                const int64_t secondQueue = second.getHemisphere()
                        * target.memory().slices_per_hemisphere
                        * target.memory().banks_per_slice
                    + second.getSlice() * target.memory().banks_per_slice
                    + second.getBank().value_or(0);
                const bool compatible =
                    secondQueue == queue
                    && second.getRepeatCount() == 1
                    && !second->getAttr("wave_count")
                    && second.getOpcode() == first.getOpcode()
                    && second.getPackedStream()
                        == first.getPackedStream()
                    && second.getAddressBinding()
                        == first.getAddressBinding();
                if (compatible) {
                    interval = second.getCycle() - first.getCycle();
                    stride = second.getAddress() - first.getAddress();
                    if (interval > 0
                        && interval <= kMaxRepeatInterval
                        && stride >= kMinRepeatStride
                        && stride <= kMaxRepeatStride) {
                        ++end;
                        while (end < mem_transfers.size()
                            && static_cast<int64_t>(end - index)
                                < kMaxRepeatCount) {
                            schedule::MemTransferOp next =
                                mem_transfers[end];
                            const int64_t nextQueue =
                                mem_queue(target, next.getHemisphere(),
                                    next.getSlice(),
                                    next.getBank().value_or(0));
                            const int64_t repeat =
                                static_cast<int64_t>(end - index);
                            if (nextQueue != queue
                                || next.getRepeatCount() != 1
                                || next->getAttr("wave_count")
                                || next.getOpcode()
                                    != first.getOpcode()
                                || next.getPackedStream()
                                    != first.getPackedStream()
                                || next.getAddressBinding()
                                    != first.getAddressBinding()
                                || next.getCycle()
                                    != first.getCycle()
                                        + repeat * interval
                                || next.getAddress()
                                    != first.getAddress()
                                        + repeat * stride)
                                break;
                            ++end;
                        }
                    } else {
                        end = index + 1;
                        interval = first.getRepeatInterval();
                        stride = first.getAddressStride();
                    }
                }
            }
            builder.setInsertionPointAfter(first);
            const int64_t runLength =
                static_cast<int64_t>(end - index);
            if (runLength > 1) {
                create_mem_command(builder, first.getLoc(),
                    first.getCycle(), queue, first.getOpcode(),
                    first.getAddress(), first.getPackedStream(),
                    runLength, interval, stride, 1, 1, 0,
                    first.getAddressBinding().value_or(-1));
            } else {
                create_mem_transfer_command(
                    builder, first, target);
            }
            for (std::size_t erase = index; erase < end; ++erase)
                mem_transfers[erase].erase();
            index = end;
        }
        for (schedule::MxmIssueOp mxm : mxm_issues) {
            builder.setInsertionPointAfter(mxm);
            create_mxm_issue_command(builder, mxm);
            mxm.erase();
        }
        for (schedule::MxmDequantOp dequant : mxm_dequants) {
            builder.setInsertionPointAfter(dequant);
            create_mxm_dequant_command(builder, dequant);
            dequant.erase();
        }
        llvm::sort(vxms, [](schedule::VxmOp lhs, schedule::VxmOp rhs) {
            return lhs.getQueue() != rhs.getQueue()
                ? lhs.getQueue() < rhs.getQueue()
                : lhs.getCycle() < rhs.getCycle();
        });
        for (std::size_t index = 0; index < vxms.size();) {
            schedule::VxmOp first = vxms[index];
            std::size_t end = index + 1;
            int64_t interval = first.getRepeatInterval();
            if (end < vxms.size()
                && same_vxm_command(first, vxms[end])) {
                  interval =
                      vxms[end].getCycle() - first.getCycle();
                  if (interval == 1) {
                    ++end;
                    while (end < vxms.size()
                        && static_cast<int64_t>(end - index)
                            < kMaxRepeatCount) {
                        const int64_t repeat =
                            static_cast<int64_t>(end - index);
                        if (!same_vxm_command(first, vxms[end])
                            || vxms[end].getCycle()
                                != first.getCycle()
                                    + repeat * interval)
                            break;
                        ++end;
                    }
                } else {
                    end = index + 1;
                    interval = first.getRepeatInterval();
                }
            }
            builder.setInsertionPointAfter(first);
            const int64_t runLength =
                static_cast<int64_t>(end - index);
            create_vxm_command(builder, first,
                runLength > 1 ? runLength : -1,
                runLength > 1 ? interval : -1);
            index = end;
        }
        for (schedule::SxmOp sxm : sxms) {
            builder.setInsertionPointAfter(sxm);
            create_sxm_command(builder, sxm);
        }
        for (schedule::MemReadOp read : reads) {
            const auto slices = placement_slices(read.getPlacement());
            const int64_t base_row = placement_integer(read.getPlacement(), "base_row");
            const int64_t count = placement_integer(read.getPlacement(), "instruction_count");
            const int64_t stride = placement_integer(read.getPlacement(), "address_stride");
            const int64_t bank = placement_integer_or(
                read.getPlacement(), "bank", 0);
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
            const target::StreamDirection direction = west_stream
                ? target::StreamDirection::West
                : target::StreamDirection::East;
            const bool weightRead = read.getRole().starts_with("weight");
            const target::StreamEndpoint destination =
                weightRead
                ? target::StreamEndpoint::MxmWeight
                : read.getRole() == "activation"
                ? target::StreamEndpoint::MxmActivation
                : target::StreamEndpoint::VxmInput;
            int64_t max_latency = 0;
            for (int64_t slice : slices) {
                auto latency = target.transport_latency(
                    target::StreamEndpoint::Mem, destination,
                    direction, slice);
                if (!latency) {
                    read.emitError(
                        "target does not support the scheduled MEM route");
                    signalPassFailure();
                    return;
                }
                max_latency = std::max(max_latency, *latency);
            }
            const int64_t command_base = stride < 0
                ? base_row - (count - 1) * stride : base_row;
            const int64_t addressBinding =
                source_binding_index(read.getInput()).value_or(-1);
            if (weightRead && addressBinding < 0) {
                read.emitError(
                    "weight MEM read must resolve to one input binding");
                signalPassFailure();
                return;
            }
            llvm::SmallVector<int64_t> cycles;
            llvm::SmallVector<int64_t> queues;
            llvm::SmallVector<int64_t> addresses;
            llvm::SmallVector<int64_t> packedStreams;
            cycles.reserve(slices.size());
            queues.reserve(slices.size());
            addresses.reserve(slices.size());
            packedStreams.reserve(slices.size());
            for (size_t index = 0; index < slices.size(); ++index) {
                const int64_t latency = *target.transport_latency(
                    target::StreamEndpoint::Mem, destination,
                    direction, slices[index]);
                cycles.push_back(static_cast<int64_t>(read.getCycle())
                    + max_latency - latency);
                queues.push_back(mem_queue(target,
                    west_hemisphere ? 1 : 0, slices[index], bank));
                addresses.push_back(command_base);
                packedStreams.push_back((west_stream ? 32 : 0)
                    + static_cast<int64_t>(read.getStreamBase())
                    + static_cast<int64_t>(index));
            }
            builder.setInsertionPointAfter(read);
            create_mem_bundle_command(builder, read.getLoc(), cycles,
                queues, "read", addresses, packedStreams, count, 1,
                stride, read.getWaveCount().value_or(1),
                read.getWaveInterval().value_or(1),
                read.getWaveAddressStride().value_or(0), addressBinding);
        }

        for (std::size_t loadIndex = 0;
             loadIndex < loads.size(); ++loadIndex) {
            schedule::MxmLoadOp load = loads[loadIndex];
            builder.setInsertionPointAfter(load);
            const int64_t repeat_count =
                loadRepeatCounts[loadIndex];
            const int64_t columnsPerWave =
                target.throughput().tile_rows;
            for (int64_t column = 0; column < repeat_count;
                 column += columnsPerWave) {
                // The four west-to-east weight pulses reach the MXM column
                // controls in reverse physical order.
                const int64_t waveCount = std::min(
                    columnsPerWave, repeat_count - column);
                const int64_t weight_column = columnsPerWave - 1;
                auto command = create_mxm_command(builder, load.getLoc(),
                    load.getCycle() + column, load.getUnitId(),
                    "iw", load.getWeightBuffer(), weight_column,
                    0, 0, 1, 1, 0, 1, "sram", true,
                    load.getDataFormat().value_or("fp16"));
                if (waveCount > 1) {
                    command->setAttr("wave_count",
                        builder.getI64IntegerAttr(waveCount));
                    command->setAttr("wave_interval",
                        builder.getI64IntegerAttr(1));
                    command->setAttr("wave_weight_column_stride",
                        builder.getI64IntegerAttr(-1));
                }
                if (auto mode = load.getWeightInputModeAttr())
                    command->setAttr("weight_input_mode", mode);
                command->setAttr("weight_stream_base",
                    builder.getI64IntegerAttr(load.getStreamBase()));
                if (load.getGroupCount().value_or(1) > 1) {
                    command->setAttr("group_count",
                        builder.getI64IntegerAttr(static_cast<int64_t>(
                            load.getGroupCount().value_or(1))));
                    command->setAttr("group_interval",
                        builder.getI64IntegerAttr(static_cast<int64_t>(
                            load.getGroupInterval().value_or(1))));
                }
            }
        }
        for (schedule::MxmComputeOp compute : computes) {
            schedule::MxmAccumulateOp accumulator;
            for (mlir::Operation* user : compute.getResult().getUsers()) {
                if (auto candidate = llvm::dyn_cast<schedule::MxmAccumulateOp>(user)) {
                    if (accumulator) {
                        compute.emitError("must have exactly one accumulator consumer");
                        signalPassFailure();
                        return;
                    }
                    accumulator = candidate;
                }
            }
            if (!accumulator) {
                for (mlir::Operation* operation = compute->getNextNode();
                     operation; operation = operation->getNextNode()) {
                    auto candidate =
                        llvm::dyn_cast<schedule::MxmAccumulateOp>(
                            operation);
                    if (!candidate
                        || candidate.getStreamBase()
                            != compute.getOutputStreamBase()
                        || candidate.getUnitId() != compute.getUnitId())
                        continue;
                    accumulator = candidate;
                    break;
                }
            }
            if (!accumulator) {
                compute.emitError("requires an accumulator consumer");
                signalPassFailure();
                return;
            }
            builder.setInsertionPointAfter(compute);
            auto command = create_mxm_command(builder, compute.getLoc(), compute.getCycle(), compute.getUnitId(),
                "compute", compute.getWeightBuffer(), 0, compute.getActivationStreamBase(),
                compute.getOutputStreamBase(), compute.getDuration(), 1,
                accumulator.getAccumulatorAddress(),
                accumulator.getAccumulatorStride(),
                accumulator.getDestination() == "local" ? "sram" : "stream",
                true, compute.getDataFormat().value_or("fp16"),
                accumulator.getAccumulatorOutputFormat().value_or("fp32"));
            if (compute.getWaveCount().value_or(1) > 1) {
                command->setAttr("wave_count",
                    builder.getI64IntegerAttr(static_cast<int64_t>(
                        compute.getWaveCount().value_or(1))));
                command->setAttr("wave_interval",
                    builder.getI64IntegerAttr(static_cast<int64_t>(
                        compute.getWaveInterval().value_or(1))));
                command->setAttr("wave_accumulator_address_stride",
                    builder.getI64IntegerAttr(static_cast<int64_t>(compute
                        .getWaveAccumulatorAddressStride().value_or(0))));
            }
        }
        for (schedule::MxmAccumulatorReadOp read : accumulator_reads) {
            builder.setInsertionPointAfter(read);
            auto command = create_mxm_command(builder, read.getLoc(), read.getCycle(),
                read.getUnitId(), "accumulator_read", 0, 0, 0,
                read.getOutputStreamBase(), 1, 1,
                read.getAccumulatorAddress(), 1, "sram",
                read.getClear(), read.getDataFormat().value_or("fp16"));
        }
        for (schedule::MemWriteOp write : writes) {
            const auto slices = placement_slices(write.getPlacement());
            const int64_t base_row = placement_integer(write.getPlacement(), "base_row");
            const int64_t count = placement_integer(write.getPlacement(), "instruction_count");
            const int64_t stride = placement_integer(write.getPlacement(), "address_stride");
            const int64_t bank = placement_integer_or(
                write.getPlacement(), "bank", 0);
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
            llvm::SmallVector<int64_t> cycles(
                slices.size(), write.getCycle());
            llvm::SmallVector<int64_t> queues;
            llvm::SmallVector<int64_t> addresses(
                slices.size(), base_row);
            llvm::SmallVector<int64_t> packedStreams;
            queues.reserve(slices.size());
            packedStreams.reserve(slices.size());
            for (size_t index = 0; index < slices.size(); ++index) {
                queues.push_back(mem_queue(target,
                    west_hemisphere ? 1 : 0, slices[index], bank));
                packedStreams.push_back((west_stream ? 32 : 0)
                    + write.getStreamBase()
                    + static_cast<int64_t>(index));
            }
            create_mem_bundle_command(builder, write.getLoc(), cycles,
                queues, "write", addresses, packedStreams, count, 1,
                stride, write.getWaveCount().value_or(1),
                write.getWaveInterval().value_or(1),
                write.getWaveAddressStride().value_or(0));
        }

        compress_mxm_command_waves(function, builder);
        compress_mxm_command_groups(function, builder);
        compress_mxm_dequant_command_waves(function, builder);

        function.walk([](mlir::func::ReturnOp op) {
            op->setOperands(mlir::ValueRange {});
        });
        function.setType(mlir::FunctionType::get(
            &getContext(), function.getArgumentTypes(), mlir::TypeRange {}));
        // Preserve block order while collecting the complete lowered SSA
        // graph. Reverse deletion then removes Schedule consumers before the
        // Stream route aliases they consume, and aliases before bindings.
        llvm::SmallVector<mlir::Operation*> lowered_ops;
        function.walk([&](mlir::Operation* op) {
            if (llvm::isa<schedule::MemReadOp, schedule::MxmLoadOp,
                    schedule::MxmComputeOp,
                    schedule::MxmAccumulatorReadOp, schedule::VxmOp,
                    schedule::SxmOp,
                    schedule::MemTransferOp, schedule::MxmIssueOp,
                    schedule::MxmDequantOp,
                    schedule::MxmAccumulateOp, schedule::MemWriteOp,
                    schedule::BindingOp, schedule::TimelineOp,
                    stream::RouteOp, stream::DequantizeOp>(op))
                lowered_ops.push_back(op);
        });
        for (auto it = lowered_ops.rbegin(); it != lowered_ops.rend(); ++it) {
            if (!(*it)->use_empty()) {
                (*it)->emitError(
                    "lowered operation still has live users after Command lowering");
                signalPassFailure();
                return;
            }
            (*it)->erase();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_schedule_to_command_pass()
{
    return std::make_unique<ScheduleToCommandPass>();
}

} // namespace ftlpu::compiler
