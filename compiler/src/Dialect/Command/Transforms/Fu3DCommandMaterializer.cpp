#include "ftlpu/compiler/Dialect/Command/Transforms/fu_3d_command_materializer.hpp"

#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/icu/fu_3d_codec.hpp"
#include "ftlpu/icu/vxm_run_2d.hpp"

#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ftlpu::compiler::command {
namespace {

template <typename Packet>
mlir::ArrayAttr packetWords(mlir::OpBuilder& builder,
    const Packet& packet)
{
    llvm::SmallVector<mlir::Attribute> words;
    words.reserve(Packet::kWordCount * Packet::kLanesPerWord);
    for (const auto& physicalWord : packet.words)
        for (const auto lane : physicalWord.lanes)
            words.push_back(builder.getI64IntegerAttr(lane));
    return builder.getArrayAttr(words);
}

template <typename Op, typename Packet>
mlir::Operation* createRawPacketOp(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t cycle, std::size_t queue,
    const Packet& packet)
{
    if (cycle > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error(
            "compiler schedule cycle does not fit Command IR i64");
    mlir::OperationState state(location, Op::getOperationName());
    state.addAttributes({
        builder.getNamedAttr("cycle", builder.getI64IntegerAttr(
            static_cast<std::int64_t>(cycle))),
        builder.getNamedAttr("queue",
            builder.getI64IntegerAttr(static_cast<std::int64_t>(queue))),
        builder.getNamedAttr("words", packetWords(builder, packet)),
    });
    return builder.create(state);
}

} // namespace

mlir::LogicalResult materializeMem3DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MemIcuInstruction& instruction, int64_t addressBinding,
    llvm::StringRef addressBindingAccess, std::string* error)
{
    if (error) error->clear();
    try {
        auto hardwareInstruction = instruction;
        const auto cycle = hardwareInstruction.loop.start_cycle;
        hardwareInstruction.loop.start_cycle = 0;
        mlir::Operation* operation = createRawPacketOp<Mem3DOp>(builder,
            location, cycle, queue,
            isa::encode_mem_icu_3d_instruction(hardwareInstruction));
        if (addressBinding >= 0) {
            operation->setAttr("address_binding",
                builder.getI64IntegerAttr(addressBinding));
            if (!addressBindingAccess.empty())
                operation->setAttr("address_binding_access",
                    builder.getStringAttr(addressBindingAccess));
        }
        return mlir::success();
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return mlir::failure();
    }
}

mlir::LogicalResult materializeMemWriteRead2DCommand(
    mlir::OpBuilder& builder, mlir::Location location,
    std::size_t cycle, std::size_t queue,
    const MemIcuWriteRead2DInstruction& instruction, std::string* error)
{
    if (error) error->clear();
    try {
        createRawPacketOp<MemWriteRead2DOp>(builder, location, cycle, queue,
            isa::encode_mem_icu_write_read_2d_instruction(instruction));
        return mlir::success();
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return mlir::failure();
    }
}

mlir::LogicalResult materializeMxmLoad3DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MxmLoadIcuInstruction& instruction, std::string* error)
{
    if (error) error->clear();
    try {
        auto hardwareInstruction = instruction;
        const auto cycle = hardwareInstruction.loop.start_cycle;
        hardwareInstruction.loop.start_cycle = 0;
        createRawPacketOp<MxmLoad3DOp>(builder, location, cycle, queue,
            isa::encode_mxm_load_icu_3d_instruction(hardwareInstruction));
        return mlir::success();
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return mlir::failure();
    }
}

mlir::LogicalResult materializeMxmDequant3DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MxmDequantIcuInstruction& instruction, int64_t scaleBinding,
    std::string* error)
{
    if (error) error->clear();
    try {
        auto hardwareInstruction = instruction;
        const auto cycle = hardwareInstruction.loop.start_cycle;
        hardwareInstruction.loop.start_cycle = 0;
        mlir::Operation* operation = createRawPacketOp<MxmDequant3DOp>(builder,
            location, cycle, queue,
            isa::encode_mxm_dequant_icu_3d_instruction(hardwareInstruction));
        if (scaleBinding >= 0)
            operation->setAttr("scale_binding",
                builder.getI64IntegerAttr(scaleBinding));
        return mlir::success();
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return mlir::failure();
    }
}

mlir::LogicalResult materializeMxmCompute3DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const MxmComputeIcuInstruction& instruction, std::string* error)
{
    if (error) error->clear();
    try {
        auto hardwareInstruction = instruction;
        const auto cycle = hardwareInstruction.loop.start_cycle;
        hardwareInstruction.loop.start_cycle = 0;
        createRawPacketOp<MxmCompute3DOp>(builder, location, cycle, queue,
            isa::encode_mxm_compute_icu_3d_instruction(hardwareInstruction));
        return mlir::success();
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return mlir::failure();
    }
}

mlir::LogicalResult materializeFfnUp3DCommands(mlir::OpBuilder& builder,
    mlir::Location location,
    const schedule::FfnUp3DLoweringResult& lowering,
    std::string* error)
{
    if (error) error->clear();
    try {
        for (const auto& command : lowering.mem_commands) {
            if (command.instruction.loop.start_cycle != 0)
                throw std::invalid_argument(
                    "FFN Up MEM hardware instruction has a nonzero origin");
            createRawPacketOp<Mem3DOp>(builder, location, command.cycle,
                command.queue,
                isa::encode_mem_icu_3d_instruction(command.instruction));
        }
        for (const auto& command : lowering.mxm_load_commands) {
            if (command.instruction.loop.start_cycle != 0)
                throw std::invalid_argument(
                    "FFN Up MXM load hardware instruction has a nonzero origin");
            createRawPacketOp<MxmLoad3DOp>(builder, location, command.cycle,
                command.queue, isa::encode_mxm_load_icu_3d_instruction(
                    command.instruction));
        }
        for (const auto& command : lowering.mxm_dequant_commands) {
            if (command.instruction.loop.start_cycle != 0)
                throw std::invalid_argument(
                    "FFN Up MXM dequant hardware instruction has a nonzero origin");
            createRawPacketOp<MxmDequant3DOp>(builder, location,
                command.cycle,
                command.queue, isa::encode_mxm_dequant_icu_3d_instruction(
                    command.instruction));
        }
        for (const auto& command : lowering.mxm_compute_commands) {
            if (command.instruction.loop.start_cycle != 0)
                throw std::invalid_argument(
                    "FFN Up MXM compute hardware instruction has a nonzero origin");
            createRawPacketOp<MxmCompute3DOp>(builder, location,
                command.cycle,
                command.queue, isa::encode_mxm_compute_icu_3d_instruction(
                    command.instruction));
        }
        return mlir::success();
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return mlir::failure();
    }
}

mlir::LogicalResult materializeVxmRun2DCommand(mlir::OpBuilder& builder,
    mlir::Location location, std::size_t queue,
    const VxmIcuRun2DInstruction& instruction, std::string* error,
    int64_t scaleBinding)
{
    if (error) error->clear();
    try {
        auto hardwareInstruction = instruction;
        const auto cycle = hardwareInstruction.loop.start_cycle;
        hardwareInstruction.loop.start_cycle = 0;
        mlir::Operation* operation = createRawPacketOp<VxmRun2DOp>(builder,
            location, cycle, queue,
            isa::encode_vxm_icu_run_2d_instruction(hardwareInstruction));
        if (scaleBinding >= 0)
            operation->setAttr("scale_binding",
                builder.getI64IntegerAttr(scaleBinding));
        return mlir::success();
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return mlir::failure();
    }
}

mlir::LogicalResult materializeSxmRun2DCommand(mlir::OpBuilder& builder,
    mlir::Location location, bool transpose, std::size_t queue,
    const SxmIcuRun2DInstruction& instruction, std::string* error)
{
    if (error) error->clear();
    try {
        const auto expected = transpose
            ? SxmOpcode::Transpose : SxmOpcode::Permute;
        if (instruction.instruction.opcode != expected)
            throw std::invalid_argument(
                "SXM RUN_2D queue kind does not match its tile opcode");
        auto hardwareInstruction = instruction;
        const auto cycle = hardwareInstruction.loop.start_cycle;
        if (cycle > static_cast<std::size_t>(
                        std::numeric_limits<std::int64_t>::max()))
            throw std::overflow_error(
                "compiler schedule cycle does not fit Command IR i64");
        hardwareInstruction.loop.start_cycle = 0;
        const auto packet =
            isa::encode_sxm_icu_run_2d_instruction(hardwareInstruction);
        mlir::OperationState state(location,
            SxmRun2DOp::getOperationName());
        state.addAttributes({
            builder.getNamedAttr("kind",
                builder.getStringAttr(transpose ? "transpose" : "permute")),
            builder.getNamedAttr("cycle", builder.getI64IntegerAttr(
                static_cast<std::int64_t>(cycle))),
            builder.getNamedAttr("queue", builder.getI64IntegerAttr(
                static_cast<std::int64_t>(queue))),
            builder.getNamedAttr("words", packetWords(builder, packet)),
        });
        builder.create(state);
        return mlir::success();
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return mlir::failure();
    }
}

} // namespace ftlpu::compiler::command
