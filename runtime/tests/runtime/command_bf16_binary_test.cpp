#include "ftlpu/software/runtime/binary.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    if (argc != 2)
        throw std::runtime_error(
            "usage: command_bf16_binary_test program.ftlpu");

    const auto program =
        read_binary_program(std::filesystem::path(argv[1]));
    require(program.bindings.size() == 1,
        "expected one BF16 binding");
    require(program.bindings[0].element_type
            == BindingElementType::BF16,
        "BF16 binding element type was lost");

    bool sawMxm = false;
    bool sawVxm = false;
    for (const auto& queue : program.queues) {
        for (const auto& command : queue.commands) {
            if (queue.kind == QueueKind::MxmCompute) {
                const auto encoded =
                    static_cast<isa::EncodedMxmInstruction>(
                        command.words[0])
                    | (static_cast<isa::EncodedMxmInstruction>(
                           command.words[1])
                        << 32);
                const auto instruction =
                    isa::decode_mxm_instruction(encoded);
                require(instruction.data_format
                        == MxmDataFormat::BFloat16,
                    "MXM BF16 format bit was lost");
                sawMxm = true;
            }
            if (queue.kind == QueueKind::Vxm) {
                require(command.word_count == 3,
                    "VXM compact packet is not 96 bits");
                const auto decoded = isa::decode_vxm_instruction(
                    queue.index, isa::EncodedVxmInstruction {
                        static_cast<std::uint64_t>(command.words[0])
                            | (static_cast<std::uint64_t>(
                                   command.words[1])
                                << 32),
                        command.words[2]});
                const auto& instruction = decoded.instruction;
                require(instruction.lhs.kind
                        == VxmLaneOperandKind::StreamBFloat16,
                    "VXM BF16 stream operand was lost");
                require(instruction.output_type
                        == VxmCastTarget::BFloat16,
                    "VXM BF16 cast target was lost");
                require(decoded.chain_depth == VxmChainDepth::Two,
                    "VXM chain depth was lost");
                sawVxm = true;
            }
        }
    }
    require(sawMxm && sawVxm,
        "expected BF16 MXM and VXM queues");

    std::cout << "command_bf16_binary_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "command_bf16_binary_test failed: "
              << ex.what() << '\n';
    return 1;
}
