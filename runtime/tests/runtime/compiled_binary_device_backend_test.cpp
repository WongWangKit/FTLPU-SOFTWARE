#include "ftlpu/software/runtime/binary_program_adapter.hpp"
#include "ftlpu/software/runtime/binding_transfer.hpp"
#include "ftlpu/software/runtime/cmodel_device_backend.hpp"

#include "ftlpu/core/instruction_codec.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace ftlpu;
using namespace ftlpu::software::runtime;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ExecutableTargetIdentity transformer_compiler_identity()
{
    const auto abi = executable_target_abi({
        2,     // hemispheres
        44,    // MEM slices per hemisphere
        2,     // SRAM banks per tile
        40960, // rows per bank
        8,     // bytes per SRAM word
        4,     // tile rows
        8,     // lanes per tile
        32,    // physical vector bytes
        32,    // streams per direction
        8,     // MEM read bytes/cycle
        8,     // MEM write bytes/cycle
        4,     // MXM count
        32,    // MXM rows
        32,    // MXM columns
        16,    // MXM load streams/cycle
        64,    // MXM load bytes/cycle
        8,     // VXM ALUs
    });
    return {"lpu_cmodel_transformer_eval_v1", abi};
}

QueueCommand control(isa::EncodedIcuCommand encoded)
{
    return QueueCommand {
        encoded,
        InstructionKind::None,
        0,
        {},
    };
}

QueueCommand mem(MemInstruction instruction)
{
    QueueCommand command {
        static_cast<isa::EncodedIcuCommand>(
            isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mem,
        1,
        {},
    };
    command.words[0] = isa::encode_mem_instruction(instruction);
    return command;
}

BinaryBinding vector_binding(
    BindingAccess access,
    std::uint32_t index,
    std::size_t byte_size,
    std::uint16_t slice,
    std::int64_t row)
{
    BinaryBinding binding;
    binding.index = index;
    binding.access = access;
    binding.element_type = BindingElementType::I8;
    binding.layout = BindingLayout::Vector;
    binding.byte_size = byte_size;
    binding.base_row = row;
    binding.address_stride = 1;
    binding.shape = {static_cast<std::uint64_t>(byte_size)};
    binding.slices = {slice};
    binding.hemisphere_mask = 2; // west
    return binding;
}

BinaryProgram make_mem_copy_binary(const TargetDescription& target)
{
    constexpr std::size_t kSourceSlice = 0;
    constexpr std::size_t kDestinationSlice = 4;
    constexpr std::size_t kInputRow = 100;
    constexpr std::size_t kOutputRow = 117;
    const auto west_queue_base = target.mem_slices_per_hemisphere;
    const auto stream = StreamId::East(0);

    BinaryProgram binary;
    const auto identity = transformer_compiler_identity();
    binary.target_name = identity.name;
    binary.target_abi = identity.abi;
    binary.max_cycle = 2;
    binary.queues = {
        {
            QueueKind::Mem,
            west_queue_base + kSourceSlice,
            {mem(MemInstruction::Read(kInputRow, stream))},
        },
        {
            QueueKind::Mem,
            west_queue_base + kDestinationSlice,
            {
                control(isa::encode_icu_nop(2)),
                mem(MemInstruction::Write(kOutputRow, stream)),
            },
        },
    };
    binary.bindings = {
        vector_binding(
            BindingAccess::Input, 0, target.vector_bytes,
            kSourceSlice, kInputRow),
        vector_binding(
            BindingAccess::Output, 0, target.vector_bytes,
            kDestinationSlice, kOutputRow),
    };
    return binary;
}

} // namespace

int main()
{
    try {
        CModelDeviceBackend backend;
        const auto& target = backend.target();
        auto binary = make_mem_copy_binary(target);
        const auto compiler_identity = transformer_compiler_identity();

        if (target.executable_target != compiler_identity) {
            bool rejected = false;
            try {
                (void)adapt_binary_program(
                    binary, target, target.tile_rows + 8);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected,
                "incompatible CModel configuration bypassed ABI validation");
            return 0;
        }

        const auto adapted = adapt_binary_program(
            binary, target, target.tile_rows + 8);
        std::vector<std::uint8_t> input(target.vector_bytes);
        for (std::size_t byte = 0; byte < input.size(); ++byte) {
            input[byte] =
                static_cast<std::uint8_t>((byte * 13 + 7) & 0xffu);
        }

        const auto input_transfers =
            pack_binding(adapted.bindings[0], input, target);
        for (const auto& transfer : input_transfers) {
            backend.upload(
                transfer.address, transfer.bytes, transfer.purpose);
        }

        backend.load(adapted.device_program);
        backend.launch();
        backend.wait();

        auto output_transfers = pack_binding(
            adapted.bindings[1],
            std::vector<std::uint8_t>(target.vector_bytes),
            target);
        for (auto& transfer : output_transfers) {
            transfer.bytes = backend.download(
                transfer.address,
                transfer.bytes.size(),
                transfer.purpose);
        }
        const auto output =
            unpack_binding(adapted.bindings[1], output_transfers, target);
        require(output == input,
            "BinaryProgram MEM source-to-destination DMA round trip failed");
        require(backend.state() == DeviceBackendState::Complete,
            "backend did not reach Complete state");
        const auto statistics = backend.statistics();
        require(statistics.dma_upload_bytes >= input.size(),
            "backend did not account for input/program DMA uploads");
        require(statistics.dma_download_bytes == output.size(),
            "backend did not account for output DMA download");
    } catch (const std::exception& error) {
        std::cerr << "compiled_binary_device_backend_test: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
