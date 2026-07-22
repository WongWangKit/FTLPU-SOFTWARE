#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc, char** argv)
try {
    if (argc < 2 || argc > 3)
        throw std::runtime_error(
            "usage: compiled_queue_program_runtime_test program.ftlpu [--describe|--probe-mem19|queue-index|first:last]");

    using namespace ftlpu::software::runtime;
    const auto program = read_binary_program(std::filesystem::path(argv[1]));
    if (program.queues.empty() || program.max_cycle == 0)
        throw std::logic_error("compiled queue program has no scheduled instructions");
    if (argc == 3 && std::string_view(argv[2]) == "--probe-mem19") {
        ftlpu::TspSliceSystem system;
        system.icu().enqueue_mem_nop(19, 8);
        std::cout << "direct MEM[19] NOP probe passed\n";
        return 0;
    }
    if (argc == 3 && std::string_view(argv[2]) == "--describe") {
        for (std::size_t index = 0; index < program.queues.size(); ++index) {
            const auto& queue = program.queues[index];
            std::cout << "queue[" << index << "] kind=" << static_cast<int>(queue.kind)
                      << " index=" << queue.index << " commands=" << queue.commands.size() << '\n';
        }
        return 0;
    }
    const std::vector<QueueProgram>* queues_to_load = &program.queues;
    std::vector<QueueProgram> selected_queues;
    if (argc == 3) {
        const std::string selector(argv[2]);
        const auto separator = selector.find(':');
        if (separator == std::string::npos) {
            const auto selected_index = static_cast<std::size_t>(std::stoull(selector));
            if (selected_index >= program.queues.size())
                throw std::out_of_range("queue index is outside the binary queue list");
            selected_queues = {program.queues[selected_index]};
        } else {
            const auto first = static_cast<std::size_t>(std::stoull(selector.substr(0, separator)));
            const auto last = static_cast<std::size_t>(std::stoull(selector.substr(separator + 1)));
            if (first >= last || last > program.queues.size())
                throw std::out_of_range("queue range is outside the binary queue list");
            selected_queues.assign(program.queues.begin() + first, program.queues.begin() + last);
        }
        queues_to_load = &selected_queues;
    }
    std::size_t command_count = 0;
    for (const auto& queue : *queues_to_load) command_count += queue.commands.size();
    std::cout << "max_cycle=" << program.max_cycle << " queues=" << queues_to_load->size()
              << " commands=" << command_count << std::endl;
    std::cout << "constructing CModel system" << std::endl;
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    std::cout << "CModel system constructed" << std::endl;
    std::cout << "loading ICU queues" << std::endl;
    if (queues_to_load == &program.queues) {
        CModelRuntime runtime(*system);
        runtime.load(program);
    } else {
        load_queue_programs_into_icu(*queues_to_load, system->icu());
    }
    std::cout << "ICU queues loaded" << std::endl;
    std::cout << "compiled_queue_program_runtime_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_queue_program_runtime_test failed: " << ex.what() << '\n';
    return 1;
}
