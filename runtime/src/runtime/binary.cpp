#include "ftlpu/software/runtime/binary.hpp"

#include <array>
#include <cstdint>
#include <fstream>
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

} // namespace

void write_binary_program(const BinaryProgram& program, const std::filesystem::path& path)
{
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("failed to open FTLPU binary for writing");
    }

    os.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_scalar<std::uint32_t>(os, 1);
    write_scalar<std::uint64_t>(os, static_cast<std::uint64_t>(program.max_cycle));
    write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(program.queues.size()));

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
        }
    }
}

BinaryProgram read_binary_program(const std::filesystem::path& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("failed to open FTLPU binary for reading");
    }

    std::array<char, 8> magic {};
    is.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!is || magic != kMagic) {
        throw std::runtime_error("invalid FTLPU binary magic");
    }

    const auto version = read_scalar<std::uint32_t>(is);
    if (version != 1) {
        throw std::runtime_error("unsupported FTLPU binary version");
    }

    auto program = BinaryProgram {};
    program.max_cycle = static_cast<std::size_t>(read_scalar<std::uint64_t>(is));
    const auto queue_count = read_scalar<std::uint32_t>(is);
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
            for (auto& word : command.words) {
                word = read_scalar<std::uint32_t>(is);
            }
            queue.commands.push_back(command);
        }
        program.queues.push_back(queue);
    }

    return program;
}

} // namespace ftlpu::software::runtime
