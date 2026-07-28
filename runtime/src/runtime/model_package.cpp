#include "ftlpu/software/runtime/model_package.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace ftlpu::software::runtime {
namespace {

constexpr std::array<char, 8> kMagic {'F', 'T', 'L', 'P', 'U', 'M', '0', '1'};
constexpr std::uint32_t kVersion = 1;

template <typename T>
void write_scalar(std::ostream& stream, T value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!stream) throw std::runtime_error("failed to write FTLPU model package");
}

template <typename T>
T read_scalar(std::istream& stream)
{
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) throw std::runtime_error("truncated FTLPU model package");
    return value;
}

void write_string(std::ostream& stream, std::string_view value)
{
    if (value.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("FTLPU model package string is too long");
    write_scalar(stream, static_cast<std::uint32_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!stream) throw std::runtime_error("failed to write FTLPU model package string");
}

std::string read_string(std::istream& stream)
{
    const auto size = read_scalar<std::uint32_t>(stream);
    std::string value(size, '\0');
    stream.read(value.data(), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("truncated FTLPU model package string");
    return value;
}

template <typename T>
void write_vector(std::ostream& stream, const std::vector<T>& values)
{
    if (values.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("FTLPU model package vector is too large");
    write_scalar(stream, static_cast<std::uint32_t>(values.size()));
    for (const T& value : values) write_scalar(stream, value);
}

template <typename T>
std::vector<T> read_vector(std::istream& stream)
{
    const auto size = read_scalar<std::uint32_t>(stream);
    std::vector<T> values;
    values.reserve(size);
    for (std::uint32_t index = 0; index < size; ++index)
        values.push_back(read_scalar<T>(stream));
    return values;
}

void write_bytes(std::ostream& stream, const std::vector<std::uint8_t>& data)
{
    write_scalar(stream, static_cast<std::uint64_t>(data.size()));
    stream.write(reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size()));
    if (!stream) throw std::runtime_error("failed to write FTLPU model tensor");
}

std::vector<std::uint8_t> read_bytes(std::istream& stream)
{
    const auto size = read_scalar<std::uint64_t>(stream);
    if (size > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("FTLPU model tensor is too large");
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(data.size()));
    if (!stream) throw std::runtime_error("truncated FTLPU model tensor");
    return data;
}

void write_binding_refs(
    std::ostream& stream, const std::vector<ModelBindingRef>& refs)
{
    write_scalar(stream, static_cast<std::uint32_t>(refs.size()));
    for (const auto& ref : refs) {
        write_scalar(stream, ref.binding_index);
        write_string(stream, ref.value);
    }
}

std::vector<ModelBindingRef> read_binding_refs(std::istream& stream)
{
    const auto count = read_scalar<std::uint32_t>(stream);
    std::vector<ModelBindingRef> refs;
    refs.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index)
        refs.push_back({read_scalar<std::uint32_t>(stream), read_string(stream)});
    return refs;
}

} // namespace

void validate_model_package(const ModelPackage& package)
{
    if (package.model_name.empty() || package.architecture.empty())
        throw std::invalid_argument(
            "FTLPU model package requires model and architecture names");
    if (package.executables.empty())
        throw std::invalid_argument(
            "FTLPU model package requires at least one executable");

    std::unordered_set<std::string> names;
    for (const auto& tensor : package.tensors) {
        if (tensor.name.empty() || !names.insert(tensor.name).second)
            throw std::invalid_argument(
                "FTLPU model package tensor names must be unique");
        if (tensor.encoding != ModelTensorEncoding::Raw
            && tensor.scales.empty())
            throw std::invalid_argument(
                "quantized FTLPU model tensor requires scales");
    }
    for (const auto& value : package.values) {
        if (value.name.empty() || !names.insert(value.name).second)
            throw std::invalid_argument(
                "FTLPU model package value names must be unique");
    }

    for (const auto& executable : package.executables)
        if (executable.name.empty())
            throw std::invalid_argument(
                "FTLPU model package executable requires a name");

    for (const auto& invocation : package.invocations) {
        if (invocation.name.empty()
            || invocation.executable_index >= package.executables.size())
            throw std::invalid_argument(
                "FTLPU model invocation has an invalid executable");
        std::unordered_set<std::uint32_t> input_indices;
        for (const auto& ref : invocation.inputs) {
            if (!names.contains(ref.value)
                || !input_indices.insert(ref.binding_index).second)
                throw std::invalid_argument(
                    "FTLPU model invocation has an invalid input binding");
        }
        std::unordered_set<std::uint32_t> output_indices;
        for (const auto& ref : invocation.outputs) {
            if (!names.contains(ref.value)
                || !output_indices.insert(ref.binding_index).second)
                throw std::invalid_argument(
                    "FTLPU model invocation has an invalid output binding");
        }
    }
}

void write_model_package(
    const ModelPackage& package, const std::filesystem::path& path)
{
    validate_model_package(package);
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("failed to open FTLPU model package for writing");
    stream.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_scalar(stream, kVersion);
    write_string(stream, package.model_name);
    write_string(stream, package.architecture);

    write_scalar(stream, static_cast<std::uint32_t>(package.tensors.size()));
    for (const auto& tensor : package.tensors) {
        write_string(stream, tensor.name);
        write_scalar(stream, static_cast<std::uint16_t>(tensor.element_type));
        write_scalar(stream, static_cast<std::uint16_t>(tensor.encoding));
        write_scalar(stream, tensor.quantized_axis);
        write_scalar(stream, tensor.quantized_block_size);
        write_vector(stream, tensor.shape);
        write_vector(stream, tensor.scales);
        write_bytes(stream, tensor.data);
    }

    write_scalar(stream, static_cast<std::uint32_t>(package.values.size()));
    for (const auto& value : package.values) {
        write_string(stream, value.name);
        write_scalar(stream, static_cast<std::uint16_t>(value.element_type));
        std::uint16_t flags = value.external_input ? 1 : 0;
        if (value.external_output) flags |= 2;
        write_scalar(stream, flags);
        write_vector(stream, value.shape);
    }

    write_scalar(stream, static_cast<std::uint32_t>(package.executables.size()));
    for (const auto& executable : package.executables) {
        write_string(stream, executable.name);
        std::ostringstream binary(std::ios::out | std::ios::binary);
        write_binary_program(executable.program, binary);
        const std::string data = binary.str();
        write_scalar(stream, static_cast<std::uint64_t>(data.size()));
        stream.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!stream)
            throw std::runtime_error(
                "failed to write embedded FTLPU executable");
    }

    write_scalar(stream, static_cast<std::uint32_t>(package.invocations.size()));
    for (const auto& invocation : package.invocations) {
        write_string(stream, invocation.name);
        write_scalar(stream, invocation.executable_index);
        write_binding_refs(stream, invocation.inputs);
        write_binding_refs(stream, invocation.outputs);
    }
}

ModelPackage read_model_package(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("failed to open FTLPU model package for reading");
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != kMagic)
        throw std::runtime_error("invalid FTLPU model package magic");
    if (read_scalar<std::uint32_t>(stream) != kVersion)
        throw std::runtime_error("unsupported FTLPU model package version");

    ModelPackage package;
    package.model_name = read_string(stream);
    package.architecture = read_string(stream);

    const auto tensor_count = read_scalar<std::uint32_t>(stream);
    package.tensors.reserve(tensor_count);
    for (std::uint32_t index = 0; index < tensor_count; ++index) {
        ModelTensor tensor;
        tensor.name = read_string(stream);
        tensor.element_type =
            static_cast<BindingElementType>(read_scalar<std::uint16_t>(stream));
        tensor.encoding =
            static_cast<ModelTensorEncoding>(read_scalar<std::uint16_t>(stream));
        tensor.quantized_axis = read_scalar<std::int32_t>(stream);
        tensor.quantized_block_size = read_scalar<std::uint32_t>(stream);
        tensor.shape = read_vector<std::uint64_t>(stream);
        tensor.scales = read_vector<float>(stream);
        tensor.data = read_bytes(stream);
        package.tensors.push_back(std::move(tensor));
    }

    const auto value_count = read_scalar<std::uint32_t>(stream);
    package.values.reserve(value_count);
    for (std::uint32_t index = 0; index < value_count; ++index) {
        ModelValue value;
        value.name = read_string(stream);
        value.element_type =
            static_cast<BindingElementType>(read_scalar<std::uint16_t>(stream));
        const auto flags = read_scalar<std::uint16_t>(stream);
        value.external_input = (flags & 1) != 0;
        value.external_output = (flags & 2) != 0;
        value.shape = read_vector<std::uint64_t>(stream);
        package.values.push_back(std::move(value));
    }

    const auto executable_count = read_scalar<std::uint32_t>(stream);
    package.executables.reserve(executable_count);
    for (std::uint32_t index = 0; index < executable_count; ++index) {
        ModelExecutable executable;
        executable.name = read_string(stream);
        const auto size = read_scalar<std::uint64_t>(stream);
        std::string data(static_cast<std::size_t>(size), '\0');
        stream.read(data.data(), static_cast<std::streamsize>(data.size()));
        if (!stream)
            throw std::runtime_error("truncated embedded FTLPU executable");
        std::istringstream binary(data, std::ios::in | std::ios::binary);
        executable.program = read_binary_program(binary);
        package.executables.push_back(std::move(executable));
    }

    const auto invocation_count = read_scalar<std::uint32_t>(stream);
    package.invocations.reserve(invocation_count);
    for (std::uint32_t index = 0; index < invocation_count; ++index) {
        ModelInvocation invocation;
        invocation.name = read_string(stream);
        invocation.executable_index = read_scalar<std::uint32_t>(stream);
        invocation.inputs = read_binding_refs(stream);
        invocation.outputs = read_binding_refs(stream);
        package.invocations.push_back(std::move(invocation));
    }
    validate_model_package(package);
    return package;
}

} // namespace ftlpu::software::runtime
