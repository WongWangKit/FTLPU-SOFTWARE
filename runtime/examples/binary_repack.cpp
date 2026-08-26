#include "ftlpu/software/runtime/binary.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv)
try {
    if (argc != 3)
        throw std::runtime_error(
            "usage: ftlpu_binary_repack input.ftlpu output.ftlpu");

    const auto input = std::filesystem::absolute(argv[1]).lexically_normal();
    const auto output = std::filesystem::absolute(argv[2]).lexically_normal();
    if (input == output
        || (std::filesystem::exists(output)
            && std::filesystem::equivalent(input, output)))
        throw std::runtime_error("input and output paths must be different");

    const auto inputBytes = std::filesystem::file_size(input);
    const auto program =
        ftlpu::software::runtime::read_binary_program(input);
    ftlpu::software::runtime::write_binary_program(program, output);
    const auto outputBytes = std::filesystem::file_size(output);

    std::size_t commands = 0;
    for (const auto& queue : program.queues)
        commands += queue.commands.size();
    std::cout << "repacked queues=" << program.queues.size()
              << " commands=" << commands
              << " input_bytes=" << inputBytes
              << " output_bytes=" << outputBytes
              << " saved_bytes="
              << (inputBytes > outputBytes ? inputBytes - outputBytes : 0)
              << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
