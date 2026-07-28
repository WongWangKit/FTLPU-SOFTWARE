#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {

enum class ModelTensorEncoding : std::uint16_t {
    Raw = 0,
    SymmetricPerTensorI8 = 1,
    SymmetricPerAxisI8 = 2,
    SymmetricPerBlockI8 = 3,
};

struct ModelTensor {
    std::string name{};
    BindingElementType element_type{BindingElementType::I8};
    std::vector<std::uint64_t> shape{};
    std::vector<std::uint8_t> data{};
    ModelTensorEncoding encoding{ModelTensorEncoding::Raw};
    std::int32_t quantized_axis{-1};
    std::uint32_t quantized_block_size{0};
    std::vector<float> scales{};
};

struct ModelValue {
    std::string name{};
    BindingElementType element_type{BindingElementType::F16};
    std::vector<std::uint64_t> shape{};
    bool external_input{false};
    bool external_output{false};
};

struct ModelExecutable {
    std::string name{};
    BinaryProgram program{};
};

struct ModelBindingRef {
    std::uint32_t binding_index{0};
    std::string value{};
};

struct ModelInvocation {
    std::string name{};
    std::uint32_t executable_index{0};
    std::vector<ModelBindingRef> inputs{};
    std::vector<ModelBindingRef> outputs{};
};

struct ModelPackage {
    std::string model_name{};
    std::string architecture{};
    std::vector<ModelTensor> tensors{};
    std::vector<ModelValue> values{};
    std::vector<ModelExecutable> executables{};
    std::vector<ModelInvocation> invocations{};
};

void validate_model_package(const ModelPackage& package);
void write_model_package(
    const ModelPackage& package, const std::filesystem::path& path);
ModelPackage read_model_package(const std::filesystem::path& path);

} // namespace ftlpu::software::runtime
