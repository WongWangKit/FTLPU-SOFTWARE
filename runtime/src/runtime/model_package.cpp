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
constexpr std::uint32_t kVersion = 5;

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

void write_state_binding_refs(
    std::ostream& stream, const std::vector<ModelStateBindingRef>& refs)
{
    write_scalar(stream, static_cast<std::uint32_t>(refs.size()));
    for (const auto& ref : refs) {
        write_scalar(stream, ref.binding_index);
        write_string(stream, ref.state);
    }
}

std::vector<ModelStateBindingRef> read_state_binding_refs(
    std::istream& stream)
{
    const auto count = read_scalar<std::uint32_t>(stream);
    std::vector<ModelStateBindingRef> refs;
    refs.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index)
        refs.push_back(
            {read_scalar<std::uint32_t>(stream), read_string(stream)});
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
        if ((tensor.encoding == ModelTensorEncoding::SymmetricPerTensorI8
                || tensor.encoding
                    == ModelTensorEncoding::SymmetricPerAxisI8
                || tensor.encoding
                    == ModelTensorEncoding::SymmetricPerBlockI8)
            && tensor.scales.empty())
            throw std::invalid_argument(
                "quantized FTLPU model tensor requires scales");
    }
    for (const auto& value : package.values) {
        if (value.name.empty() || !names.insert(value.name).second)
            throw std::invalid_argument(
                "FTLPU model package value names must be unique");
    }
    for (const auto& lookup : package.embedding_lookups) {
        if (lookup.name.empty() || lookup.token_ids.empty()
            || lookup.table.empty() || lookup.output.empty()
            || !names.contains(lookup.token_ids)
            || !names.contains(lookup.table)
            || !names.contains(lookup.output))
            throw std::invalid_argument(
                "FTLPU embedding lookup references an invalid model value");
    }
    for (const auto& lm_head : package.host_lm_heads) {
        if (lm_head.name.empty() || lm_head.hidden.empty()
            || lm_head.weight.empty() || lm_head.output.empty()
            || !names.contains(lm_head.hidden)
            || !names.contains(lm_head.weight)
            || !names.contains(lm_head.output))
            throw std::invalid_argument(
                "FTLPU host LM head references an invalid model value");
    }
    std::unordered_set<std::string> state_names;
    for (const auto& state : package.states) {
        if (state.name.empty() || !names.insert(state.name).second
            || !state_names.insert(state.name).second
            || state.shape.empty() || state.max_tokens == 0
            || state.shape.front() != state.max_tokens)
            throw std::invalid_argument(
                "FTLPU model state metadata is invalid");
    }

    for (const auto& executable : package.executables)
        if (executable.name.empty())
            throw std::invalid_argument(
                "FTLPU model package executable requires a name");

    std::unordered_set<std::uint32_t> weight_layers;
    for (std::size_t index = 0; index < package.weight_pages.size(); ++index) {
        const auto& page = package.weight_pages[index];
        if (page.bank >= hw::kMemBanksPerSlice || page.tensors.empty()
            || !weight_layers.insert(page.layer).second)
            throw std::invalid_argument(
                "FTLPU model weight page metadata is invalid");
        for (const std::string& tensor : page.tensors)
            if (!names.contains(tensor))
                throw std::invalid_argument(
                    "FTLPU weight page references an unknown tensor");
        for (const auto& segment : page.segments) {
            const auto tensor = std::find_if(package.tensors.begin(),
                package.tensors.end(), [&](const ModelTensor& candidate) {
                    return candidate.name == segment.tensor;
                });
            const std::uint64_t bytes = static_cast<std::uint64_t>(
                segment.vector_count) * hw::kPhysicalVectorBytes;
            if (tensor == package.tensors.end()
                || tensor->encoding
                    != ModelTensorEncoding::TargetPackedSramVectors
                || segment.hemisphere >= hw::kHemispheres
                || segment.slice >= hw::kMemSliceColumns
                || segment.stream >= hw::kStreamsPerDirection
                || segment.vector_count == 0
                || segment.byte_offset + bytes > tensor->data.size())
                throw std::invalid_argument(
                    "FTLPU weight page has an invalid packed segment");
        }
        if (index != 0
            && page.bank == package.weight_pages[index - 1].bank)
            throw std::invalid_argument(
                "adjacent FTLPU weight pages must use different banks");
    }

    std::optional<std::uint32_t> previous_invocation_page;
    for (const auto& invocation : package.invocations) {
        if (invocation.name.empty()
            || invocation.executable_index >= package.executables.size())
            throw std::invalid_argument(
                "FTLPU model invocation has an invalid executable");
        if (invocation.weight_page != 0xffffffffu
            && invocation.weight_page >= package.weight_pages.size())
            throw std::invalid_argument(
                "FTLPU model invocation has an invalid weight page");
        if (invocation.weight_page != 0xffffffffu) {
            if (previous_invocation_page
                && *previous_invocation_page != invocation.weight_page
                && package.weight_pages[*previous_invocation_page].bank
                    == package.weight_pages[invocation.weight_page].bank)
                throw std::invalid_argument(
                    "adjacent paged invocations must alternate MEM banks");
            previous_invocation_page = invocation.weight_page;
        } else {
            previous_invocation_page.reset();
        }
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
        std::unordered_set<std::uint32_t> state_indices;
        for (const auto& ref : invocation.states) {
            if (!state_names.contains(ref.state)
                || !state_indices.insert(ref.binding_index).second)
                throw std::invalid_argument(
                    "FTLPU model invocation has an invalid state binding");
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
        if (!executable.serialized_program.empty()) {
            write_scalar(stream, static_cast<std::uint64_t>(
                executable.serialized_program.size()));
            stream.write(reinterpret_cast<const char*>(
                    executable.serialized_program.data()),
                static_cast<std::streamsize>(
                    executable.serialized_program.size()));
            if (!stream)
                throw std::runtime_error(
                    "failed to write embedded FTLPU executable");
            continue;
        }
        std::ostringstream binary(std::ios::out | std::ios::binary);
        write_binary_program(executable.program, binary);
        const std::string data = binary.str();
        write_scalar(stream, static_cast<std::uint64_t>(data.size()));
        stream.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!stream)
            throw std::runtime_error(
                "failed to write embedded FTLPU executable");
    }

    write_scalar(stream,
        static_cast<std::uint32_t>(package.embedding_lookups.size()));
    for (const auto& lookup : package.embedding_lookups) {
        write_string(stream, lookup.name);
        write_string(stream, lookup.token_ids);
        write_string(stream, lookup.table);
        write_string(stream, lookup.output);
    }

    write_scalar(stream,
        static_cast<std::uint32_t>(package.host_lm_heads.size()));
    for (const auto& lm_head : package.host_lm_heads) {
        write_string(stream, lm_head.name);
        write_string(stream, lm_head.hidden);
        write_string(stream, lm_head.weight);
        write_string(stream, lm_head.output);
        write_scalar(stream,
            static_cast<std::uint8_t>(lm_head.last_token_only));
    }

    write_scalar(stream, static_cast<std::uint32_t>(package.states.size()));
    for (const auto& state : package.states) {
        write_string(stream, state.name);
        write_scalar(stream, static_cast<std::uint16_t>(state.kind));
        write_scalar(stream,
            static_cast<std::uint16_t>(state.element_type));
        write_scalar(stream, state.layer);
        write_scalar(stream, state.max_tokens);
        write_vector(stream, state.shape);
    }

    write_scalar(stream,
        static_cast<std::uint32_t>(package.weight_pages.size()));
    for (const auto& page : package.weight_pages) {
        write_scalar(stream, page.layer);
        write_scalar(stream, page.bank);
        write_scalar(stream,
            static_cast<std::uint32_t>(page.tensors.size()));
        for (const auto& tensor : page.tensors) write_string(stream, tensor);
        write_scalar(stream,
            static_cast<std::uint32_t>(page.segments.size()));
        for (const auto& segment : page.segments) {
            write_string(stream, segment.tensor);
            write_scalar(stream, segment.byte_offset);
            write_scalar(stream, segment.hemisphere);
            write_scalar(stream, segment.slice);
            write_scalar(stream, segment.base_row);
            write_scalar(stream, segment.vector_count);
            write_scalar(stream, segment.stream);
        }
    }

    write_scalar(stream, static_cast<std::uint32_t>(package.invocations.size()));
    for (const auto& invocation : package.invocations) {
        write_string(stream, invocation.name);
        write_scalar(stream, invocation.executable_index);
        write_binding_refs(stream, invocation.inputs);
        write_binding_refs(stream, invocation.outputs);
        write_state_binding_refs(stream, invocation.states);
        write_scalar(stream, invocation.weight_page);
    }
}

ModelPackage read_model_package(
    const std::filesystem::path& path, ModelPackageLoadMode mode)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("failed to open FTLPU model package for reading");
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != kMagic)
        throw std::runtime_error("invalid FTLPU model package magic");
    const std::uint32_t version = read_scalar<std::uint32_t>(stream);
    if (version < 1 || version > kVersion)
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
        if (size > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()))
            throw std::runtime_error(
                "embedded FTLPU executable is too large");
        if (mode == ModelPackageLoadMode::LazyExecutables) {
            executable.serialized_program.resize(
                static_cast<std::size_t>(size));
            stream.read(reinterpret_cast<char*>(
                    executable.serialized_program.data()),
                static_cast<std::streamsize>(
                    executable.serialized_program.size()));
            if (!stream)
                throw std::runtime_error(
                    "truncated embedded FTLPU executable");
            executable.program = read_binary_program_metadata(
                executable.serialized_program);
            package.executables.push_back(std::move(executable));
            continue;
        }
        const std::streampos begin = stream.tellg();
        if (begin == std::streampos(-1))
            throw std::runtime_error(
                "failed to locate embedded FTLPU executable");
        executable.program = read_binary_program(stream);
        const std::streampos end = stream.tellg();
        if (end == std::streampos(-1)
            || static_cast<std::uint64_t>(end - begin) != size)
            throw std::runtime_error(
                "embedded FTLPU executable size mismatch");
        package.executables.push_back(std::move(executable));
    }

    if (version >= 2) {
        const auto lookup_count = read_scalar<std::uint32_t>(stream);
        package.embedding_lookups.reserve(lookup_count);
        for (std::uint32_t index = 0; index < lookup_count; ++index) {
            package.embedding_lookups.push_back(ModelEmbeddingLookup {
                read_string(stream), read_string(stream),
                read_string(stream), read_string(stream)});
        }
    }
    if (version >= 3) {
        const auto lm_head_count = read_scalar<std::uint32_t>(stream);
        package.host_lm_heads.reserve(lm_head_count);
        for (std::uint32_t index = 0; index < lm_head_count; ++index) {
            package.host_lm_heads.push_back(ModelHostLmHead {
                read_string(stream), read_string(stream),
                read_string(stream), read_string(stream),
                read_scalar<std::uint8_t>(stream) != 0});
        }
    }
    if (version >= 4) {
        const auto state_count = read_scalar<std::uint32_t>(stream);
        package.states.reserve(state_count);
        for (std::uint32_t index = 0; index < state_count; ++index) {
            ModelState state;
            state.name = read_string(stream);
            state.kind =
                static_cast<ModelStateKind>(
                    read_scalar<std::uint16_t>(stream));
            state.element_type =
                static_cast<BindingElementType>(
                    read_scalar<std::uint16_t>(stream));
            state.layer = read_scalar<std::uint32_t>(stream);
            state.max_tokens = read_scalar<std::uint32_t>(stream);
            state.shape = read_vector<std::uint64_t>(stream);
            package.states.push_back(std::move(state));
        }
    }
    if (version >= 5) {
        const auto page_count = read_scalar<std::uint32_t>(stream);
        package.weight_pages.reserve(page_count);
        for (std::uint32_t index = 0; index < page_count; ++index) {
            ModelWeightPage page;
            page.layer = read_scalar<std::uint32_t>(stream);
            page.bank = read_scalar<std::uint16_t>(stream);
            const auto tensor_count = read_scalar<std::uint32_t>(stream);
            page.tensors.reserve(tensor_count);
            for (std::uint32_t tensor = 0; tensor < tensor_count; ++tensor)
                page.tensors.push_back(read_string(stream));
            const auto segment_count = read_scalar<std::uint32_t>(stream);
            page.segments.reserve(segment_count);
            for (std::uint32_t segment = 0; segment < segment_count;
                 ++segment) {
                ModelWeightPage::Segment value;
                value.tensor = read_string(stream);
                value.byte_offset = read_scalar<std::uint64_t>(stream);
                value.hemisphere = read_scalar<std::uint16_t>(stream);
                value.slice = read_scalar<std::uint16_t>(stream);
                value.base_row = read_scalar<std::uint32_t>(stream);
                value.vector_count = read_scalar<std::uint32_t>(stream);
                value.stream = read_scalar<std::uint16_t>(stream);
                page.segments.push_back(std::move(value));
            }
            package.weight_pages.push_back(std::move(page));
        }
    }

    const auto invocation_count = read_scalar<std::uint32_t>(stream);
    package.invocations.reserve(invocation_count);
    for (std::uint32_t index = 0; index < invocation_count; ++index) {
        ModelInvocation invocation;
        invocation.name = read_string(stream);
        invocation.executable_index = read_scalar<std::uint32_t>(stream);
        invocation.inputs = read_binding_refs(stream);
        invocation.outputs = read_binding_refs(stream);
        if (version >= 4)
            invocation.states = read_state_binding_refs(stream);
        if (version >= 5)
            invocation.weight_page = read_scalar<std::uint32_t>(stream);
        package.invocations.push_back(std::move(invocation));
    }
    validate_model_package(package);
    return package;
}

ModelPackage read_model_package(const std::filesystem::path& path)
{
    return read_model_package(path, ModelPackageLoadMode::Eager);
}

BinaryProgram materialize_model_executable(
    const ModelExecutable& executable)
{
    if (executable.serialized_program.empty())
        return executable.program;
    return read_binary_program(executable.serialized_program);
}

} // namespace ftlpu::software::runtime
