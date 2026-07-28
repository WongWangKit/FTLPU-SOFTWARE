// Keep serialization rebuilt when BinaryProgram or BinaryBinding ABI evolves.
#include "ftlpu/software/runtime/binary.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ftlpu::software::runtime {

namespace {

constexpr std::array<char, 8> kMagic {'F', 'T', 'L', 'P', 'U', 'B', '0', '1'};

template <typename T>
void write_scalar(std::ostream& os, T value)
{
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!os) {
        throw std::runtime_error("failed to write FTLPU binary");
    }
}

template <typename T>
T read_scalar(std::istream& is)
{
    T value {};
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!is) {
        throw std::runtime_error("truncated FTLPU binary");
    }
    return value;
}

void write_binding(std::ostream& os, const BinaryBinding& binding)
{
    write_scalar<std::uint32_t>(os, binding.index);
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.access));
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.element_type));
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.layout));
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.shape.size()));
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.slices.size()));
    write_scalar<std::uint16_t>(os, binding.hemisphere_mask);
    write_scalar<std::uint64_t>(os, binding.byte_size);
    write_scalar<std::int64_t>(os, binding.base_row);
    write_scalar<std::int64_t>(os, binding.instruction_count);
    write_scalar<std::int64_t>(os, binding.address_stride);
    for (std::uint64_t dimension : binding.shape) write_scalar(os, dimension);
    for (std::uint16_t slice : binding.slices) write_scalar(os, slice);
    if (binding.role.size() > std::numeric_limits<std::uint16_t>::max()
        || binding.name.size() > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error("FTLPU binding metadata string is too long");
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(binding.role.size()));
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(binding.name.size()));
    write_scalar<std::uint64_t>(os, binding.ready_cycle);
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(binding.initializer));
    write_scalar<std::uint16_t>(os, 0);
    write_scalar<float>(os, binding.rope_theta);
    write_scalar<std::uint32_t>(os, binding.rope_head_dim);
    os.write(binding.role.data(),
        static_cast<std::streamsize>(binding.role.size()));
    os.write(binding.name.data(),
        static_cast<std::streamsize>(binding.name.size()));
    if (!os) throw std::runtime_error("failed to write binding metadata");
}

BinaryBinding read_binding(std::istream& is, std::uint32_t version)
{
    BinaryBinding binding;
    binding.index = read_scalar<std::uint32_t>(is);
    binding.access = static_cast<BindingAccess>(read_scalar<std::uint16_t>(is));
    binding.element_type = static_cast<BindingElementType>(read_scalar<std::uint16_t>(is));
    binding.layout = static_cast<BindingLayout>(read_scalar<std::uint16_t>(is));
    const auto rank = read_scalar<std::uint16_t>(is);
    const auto slice_count = read_scalar<std::uint16_t>(is);
    const auto hemisphere_mask = read_scalar<std::uint16_t>(is);
    binding.hemisphere_mask = version >= 3 ? hemisphere_mask : 1;
    binding.byte_size = read_scalar<std::uint64_t>(is);
    binding.base_row = read_scalar<std::int64_t>(is);
    binding.instruction_count = read_scalar<std::int64_t>(is);
    binding.address_stride = read_scalar<std::int64_t>(is);
    binding.shape.reserve(rank);
    binding.slices.reserve(slice_count);
    for (std::uint16_t i = 0; i < rank; ++i)
        binding.shape.push_back(read_scalar<std::uint64_t>(is));
    for (std::uint16_t i = 0; i < slice_count; ++i)
        binding.slices.push_back(read_scalar<std::uint16_t>(is));
    if (version >= 6) {
        const auto role_size = read_scalar<std::uint16_t>(is);
        const auto name_size = read_scalar<std::uint16_t>(is);
        binding.ready_cycle = read_scalar<std::uint64_t>(is);
        if (version >= 7) {
            binding.initializer = static_cast<BindingInitializer>(
                read_scalar<std::uint16_t>(is));
            (void)read_scalar<std::uint16_t>(is);
            binding.rope_theta = read_scalar<float>(is);
            binding.rope_head_dim = read_scalar<std::uint32_t>(is);
        } else if (binding.layout == BindingLayout::Fp32CausalMaskTile) {
            binding.initializer = BindingInitializer::CausalMask;
        }
        binding.role.resize(role_size);
        binding.name.resize(name_size);
        is.read(binding.role.data(),
            static_cast<std::streamsize>(binding.role.size()));
        is.read(binding.name.data(),
            static_cast<std::streamsize>(binding.name.size()));
        if (!is) throw std::runtime_error("truncated binding metadata");
    }
    return binding;
}

} // namespace

void write_binary_program(const BinaryProgram& program, std::ostream& os)
{
    os.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_scalar<std::uint32_t>(os, kBinaryFormatVersion);
    if (program.target_name.empty()
        || program.target_name.size() > std::numeric_limits<std::uint16_t>::max()
        || program.target_abi == 0)
        throw std::runtime_error("FTLPU binary requires a valid target identity");
    write_scalar<std::uint64_t>(os, program.target_abi);
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(program.target_name.size()));
    os.write(program.target_name.data(),
        static_cast<std::streamsize>(program.target_name.size()));
    if (!os) throw std::runtime_error("failed to write FTLPU target name");
    write_scalar<std::uint64_t>(os, static_cast<std::uint64_t>(program.max_cycle));
    write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(program.queues.size()));
    write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(program.bindings.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.scale_relocations.size()));

    for (const auto& binding : program.bindings) write_binding(os, binding);

    for (const auto& queue : program.queues) {
        write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(queue.kind));
        write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(queue.index));
        write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(queue.commands.size()));
        for (const auto& command : queue.commands) {
            write_scalar<std::uint32_t>(os, command.command);
            write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(command.instruction_kind));
            write_scalar<std::uint16_t>(os, command.word_count);
            for (const auto word : command.words) {
                write_scalar<std::uint32_t>(os, word);
            }
            write_scalar<std::uint16_t>(os,
                static_cast<std::uint16_t>(command.extension_words.size()));
            for (const auto word : command.extension_words) write_scalar<std::uint32_t>(os, word);
        }
    }
    for (const auto& relocation : program.scale_relocations) {
        write_scalar<std::uint32_t>(os, relocation.binding_index);
        write_scalar<std::uint32_t>(os, relocation.scale_index);
        write_scalar<std::uint16_t>(
            os, static_cast<std::uint16_t>(relocation.queue_kind));
        write_scalar<std::uint16_t>(os, relocation.queue_index);
        write_scalar<std::uint32_t>(os, relocation.command_index);
        write_scalar<std::uint16_t>(
            os, static_cast<std::uint16_t>(relocation.operand));
    }
}

BinaryProgram read_binary_program(std::istream& is)
{
    std::array<char, 8> magic {};
    is.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!is || magic != kMagic) {
        throw std::runtime_error("invalid FTLPU binary magic");
    }

    const auto version = read_scalar<std::uint32_t>(is);
    if (version < 1 || version > kBinaryFormatVersion) {
        throw std::runtime_error("unsupported FTLPU binary version");
    }

    auto program = BinaryProgram {};
    if (version >= 5) {
        program.target_abi = read_scalar<std::uint64_t>(is);
        const auto target_name_size = read_scalar<std::uint16_t>(is);
        program.target_name.resize(target_name_size);
        is.read(program.target_name.data(),
            static_cast<std::streamsize>(program.target_name.size()));
        if (!is) throw std::runtime_error("truncated FTLPU target name");
    } else {
        program.target_name = "legacy-unidentified";
        program.target_abi = 0;
    }
    program.max_cycle = static_cast<std::size_t>(read_scalar<std::uint64_t>(is));
    const auto queue_count = read_scalar<std::uint32_t>(is);
    const auto binding_count = version >= 2 ? read_scalar<std::uint32_t>(is) : 0;
    const auto relocation_count =
        version >= 8 ? read_scalar<std::uint32_t>(is) : 0;
    program.bindings.reserve(binding_count);
    for (std::uint32_t binding_id = 0; binding_id < binding_count; ++binding_id)
        program.bindings.push_back(read_binding(is, version));
    program.queues.reserve(queue_count);

    for (std::uint32_t queue_id = 0; queue_id < queue_count; ++queue_id) {
        auto queue = QueueProgram {};
        queue.kind = static_cast<QueueKind>(read_scalar<std::uint16_t>(is));
        queue.index = read_scalar<std::uint16_t>(is);
        const auto command_count = read_scalar<std::uint32_t>(is);
        queue.commands.reserve(command_count);

        for (std::uint32_t command_id = 0; command_id < command_count; ++command_id) {
            auto command = QueueCommand {};
            command.command = read_scalar<std::uint32_t>(is);
            command.instruction_kind = static_cast<InstructionKind>(read_scalar<std::uint16_t>(is));
            command.word_count = read_scalar<std::uint16_t>(is);
            const std::size_t serialized_words = version >= 3 ? command.words.size() : 3;
            for (std::size_t word = 0; word < serialized_words; ++word)
                command.words[word] = read_scalar<std::uint32_t>(is);
            if (version >= 4) {
                const auto extension_count = read_scalar<std::uint16_t>(is);
                command.extension_words.reserve(extension_count);
                for (std::uint16_t word = 0; word < extension_count; ++word)
                    command.extension_words.push_back(read_scalar<std::uint32_t>(is));
            }
            queue.commands.push_back(command);
        }
        program.queues.push_back(queue);
    }
    program.scale_relocations.reserve(relocation_count);
    for (std::uint32_t index = 0; index < relocation_count; ++index) {
        BinaryScaleRelocation relocation;
        relocation.binding_index = read_scalar<std::uint32_t>(is);
        relocation.scale_index = read_scalar<std::uint32_t>(is);
        relocation.queue_kind =
            static_cast<QueueKind>(read_scalar<std::uint16_t>(is));
        relocation.queue_index = read_scalar<std::uint16_t>(is);
        relocation.command_index = read_scalar<std::uint32_t>(is);
        relocation.operand = static_cast<VxmImmediateOperand>(
            read_scalar<std::uint16_t>(is));
        program.scale_relocations.push_back(relocation);
    }

    return program;
}

void write_binary_program(
    const BinaryProgram& program, const std::filesystem::path& path)
{
    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw std::runtime_error("failed to open FTLPU binary for writing");
    write_binary_program(program, os);
}

BinaryProgram read_binary_program(const std::filesystem::path& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw std::runtime_error("failed to open FTLPU binary for reading");
    return read_binary_program(is);
}

} // namespace ftlpu::software::runtime
