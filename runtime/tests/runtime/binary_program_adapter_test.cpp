#include "ftlpu/software/runtime/binary_program_adapter.hpp"
#include "ftlpu/software/runtime/binding_transfer.hpp"
#include "ftlpu/software/runtime/cmodel_device_backend.hpp"

#include "ftlpu/core/instruction_packet.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
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

template <typename Exception = std::exception>
std::string require_throws(const std::function<void()>& fn)
{
    try {
        fn();
    } catch (const Exception& error) {
        return error.what();
    }
    throw std::runtime_error("expected operation to throw");
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

BinaryProgram valid_program(const TargetDescription& target)
{
    BinaryProgram binary;
    binary.target_name = target.executable_target.name;
    binary.target_abi = target.executable_target.abi;
    binary.max_cycle = 3;
    binary.queues.push_back(QueueProgram {
        QueueKind::Mem,
        target.mem_slices_per_hemisphere,
        {
            control(isa::encode_icu_nop(3)),
            mem(MemInstruction::Read(7, StreamId::East(0))),
        },
    });
    return binary;
}

BinaryBinding vector_binding(const TargetDescription& target)
{
    BinaryBinding binding;
    binding.access = BindingAccess::Input;
    binding.element_type = BindingElementType::I8;
    binding.layout = BindingLayout::Vector;
    binding.byte_size = target.vector_bytes;
    binding.base_row = 11;
    binding.address_stride = 1;
    binding.shape = {
        static_cast<std::uint64_t>(target.vector_bytes),
    };
    binding.slices = {2};
    binding.hemisphere_mask =
        target.hemisphere_count >= 2 ? 3 : 1;
    return binding;
}

} // namespace

int main()
{
    try {
        const auto target = make_cmodel_target_description();
        const auto binary = valid_program(target);
        const auto adapted = adapt_binary_program(binary, target, 9);
        require(adapted.device_program.execution_cycles == 13,
            "execution_cycles did not include zero-based final cycle");
        require(adapted.device_program.image.sections().size() == 1,
            "adapter did not emit one non-empty section");
        const auto& section =
            adapted.device_program.image.sections().front();
        require(
            section.target
                == IcuLocation::Mem(Hemisphere::West, 0),
            "flattened MEM queue did not map to west slice zero");
        require(section.packets.size() == 2,
            "adapter emitted an unexpected packet count");
        require(
            isa::decode_icu_packet(section.packets[0]).opcode
                == IcuControlOpcode::Nop,
            "NOP command was not encoded as an ICU control packet");
        require(
            isa::decode_mem_packet(section.packets[1]).address
                == MemLocalWordAddress13(7),
            "MEM command did not survive decode/re-encode");

        auto mismatch = binary;
        mismatch.target_abi ^= 1;
        const auto mismatch_message =
            require_throws<std::invalid_argument>([&] {
                (void)adapt_binary_program(mismatch, target, 0);
            });
        require(mismatch_message.find("target mismatch")
                != std::string::npos,
            "ABI mismatch error is not explicit");

        auto queue_oob = binary;
        queue_oob.queues.front().index =
            target.hemisphere_count
            * target.mem_slices_per_hemisphere;
        (void)require_throws<std::out_of_range>([&] {
            (void)adapt_binary_program(queue_oob, target, 0);
        });

        auto too_long = binary;
        too_long.queues.front().commands.assign(
            target.ifetch_packets + 1,
            control(isa::encode_icu_nop(1)));
        const auto length_message =
            require_throws<std::length_error>([&] {
                (void)adapt_binary_program(too_long, target, 0);
            });
        require(length_message.find("IFetch capacity")
                != std::string::npos,
            "IFetch overflow error is not explicit");

        auto sxm_conflict = binary;
        sxm_conflict.queues = {
            {QueueKind::SxmTranspose, 0,
                {control(isa::encode_icu_nop(1))}},
            {QueueKind::SxmPermute, 0,
                {control(isa::encode_icu_nop(1))}},
        };
        const auto sxm_message =
            require_throws<std::invalid_argument>([&] {
                (void)adapt_binary_program(sxm_conflict, target, 0);
            });
        require(sxm_message.find("both SXM") != std::string::npos,
            "SXM queue collision was not diagnosed");

        const auto binding = vector_binding(target);
        std::vector<std::uint8_t> logical(target.vector_bytes);
        for (std::size_t index = 0; index < logical.size(); ++index) {
            logical[index] =
                static_cast<std::uint8_t>((index * 17 + 5) & 0xffu);
        }
        const auto transfers = pack_binding(binding, logical, target);
        require(transfers.size()
                == (target.hemisphere_count >= 2 ? 2 : 1),
            "hemisphere_mask did not replicate Vector transfer");
        require(unpack_binding(binding, transfers, target) == logical,
            "Vector binding pack/unpack did not round trip");

        auto unsupported = binding;
        unsupported.layout = BindingLayout::Fp16BytePlanar;
        const auto layout_message =
            require_throws<std::invalid_argument>([&] {
                (void)pack_binding(unsupported, logical, target);
            });
        require(layout_message.find("Fp16BytePlanar")
                != std::string::npos,
            "unsupported layout error omitted the layout name");
    } catch (const std::exception& error) {
        std::cerr << "binary_program_adapter_test: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
