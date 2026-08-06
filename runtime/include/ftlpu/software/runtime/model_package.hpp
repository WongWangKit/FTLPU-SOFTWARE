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
    // Populated by lazy package loading. `program` then contains only target
    // and binding metadata until the executable is materialized for dispatch.
    std::vector<std::uint8_t> serialized_program{};
};

enum class ModelPackageLoadMode {
    Eager,
    LazyExecutables,
};

struct ModelEmbeddingLookup {
    std::string name{};
    std::string token_ids{};
    std::string table{};
    std::string output{};
};

struct ModelHostLmHead {
    std::string name{};
    std::string hidden{};
    std::string weight{};
    std::string output{};
    bool last_token_only{true};
};

enum class ModelStateKind : std::uint16_t {
    KvKey = 1,
    KvValue = 2,
};

struct ModelState {
    std::string name{};
    ModelStateKind kind{ModelStateKind::KvKey};
    BindingElementType element_type{BindingElementType::F16};
    std::vector<std::uint64_t> shape{};
    std::uint32_t layer{0};
    std::uint32_t max_tokens{0};
};

struct ModelBindingRef {
    std::uint32_t binding_index{0};
    std::string value{};
};

struct ModelStateBindingRef {
    std::uint32_t binding_index{0};
    std::string state{};
};

struct ModelInvocation {
    std::string name{};
    std::uint32_t executable_index{0};
    std::vector<ModelBindingRef> inputs{};
    std::vector<ModelBindingRef> outputs{};
    std::vector<ModelStateBindingRef> states{};
};

struct ModelPackage {
    std::string model_name{};
    std::string architecture{};
    std::vector<ModelTensor> tensors{};
    std::vector<ModelValue> values{};
    std::vector<ModelEmbeddingLookup> embedding_lookups{};
    std::vector<ModelHostLmHead> host_lm_heads{};
    std::vector<ModelState> states{};
    std::vector<ModelExecutable> executables{};
    std::vector<ModelInvocation> invocations{};
};

void validate_model_package(const ModelPackage& package);
void write_model_package(
    const ModelPackage& package, const std::filesystem::path& path);
ModelPackage read_model_package(const std::filesystem::path& path);
ModelPackage read_model_package(
    const std::filesystem::path& path, ModelPackageLoadMode mode);
BinaryProgram materialize_model_executable(
    const ModelExecutable& executable);

} // namespace ftlpu::software::runtime
