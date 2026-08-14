#include "ftlpu/software/runtime/model_package.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
try {
    std::filesystem::path input;
    std::filesystem::path output;
    std::uint16_t first_bank = 0;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&]() -> std::string {
            if (++index >= argc)
                throw std::runtime_error("missing value for " + argument);
            return argv[index];
        };
        if (argument == "--input") input = next();
        else if (argument == "--output") output = next();
        else if (argument == "--first-bank")
            first_bank = static_cast<std::uint16_t>(std::stoul(next()));
        else throw std::runtime_error("unknown argument: " + argument);
    }
    if (input.empty() || output.empty())
        throw std::runtime_error(
            "usage: ftlpu-pack-model-weights --input model.ftlpum "
            "--output paged.ftlpum [--first-bank 0|1]");

    using namespace ftlpu::software::runtime;
    ModelPackage package = read_model_package(input);
    build_weight_pages(package, WeightPageBuildOptions {first_bank, true});
    write_model_package(package, output);

    std::size_t segments = 0;
    std::size_t bytes = 0;
    for (const ModelWeightPage& page : package.weight_pages) {
        segments += page.segments.size();
        for (const std::string& name : page.tensors)
            for (const ModelTensor& tensor : package.tensors)
                if (tensor.name == name) bytes += tensor.data.size();
    }
    std::cout << "wrote " << output << " pages="
              << package.weight_pages.size() << " segments=" << segments
              << " packed_bytes=" << bytes << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ftlpu-pack-model-weights failed: "
              << error.what() << '\n';
    return 1;
}
