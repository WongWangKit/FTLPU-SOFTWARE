// Keep serialization rebuilt when BinaryProgram or BinaryBinding ABI evolves.
#include "ftlpu/software/runtime/binary.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <streambuf>

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

class SpanStreamBuffer final : public std::streambuf {
public:
    explicit SpanStreamBuffer(std::span<const std::uint8_t> data)
    {
        auto* begin = const_cast<char*>(
            reinterpret_cast<const char*>(data.data()));
        setg(begin, begin, begin + data.size());
    }

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
        std::ios_base::openmode mode) override
    {
        if ((mode & std::ios_base::in) == 0) return pos_type(off_type(-1));
        char* base = eback();
        char* target = nullptr;
        if (direction == std::ios_base::beg)
            target = base + offset;
        else if (direction == std::ios_base::cur)
            target = gptr() + offset;
        else if (direction == std::ios_base::end)
            target = egptr() + offset;
        if (target < base || target > egptr()) return pos_type(off_type(-1));
        setg(base, target, egptr());
        return pos_type(target - base);
    }

    pos_type seekpos(pos_type position,
        std::ios_base::openmode mode) override
    {
        return seekoff(static_cast<off_type>(position), std::ios_base::beg,
            mode);
    }
};

class ByteReader {
public:
    explicit ByteReader(std::span<const std::uint8_t> data)
        : data_(data)
    {
    }

    template <typename T>
    T read()
    {
        require(sizeof(T));
        T value;
        std::memcpy(&value, data_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    void read_bytes(void* destination, std::size_t size)
    {
        require(size);
        std::memcpy(destination, data_.data() + offset_, size);
        offset_ += size;
    }

    void skip(std::uint64_t size)
    {
        if (size > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("FTLPU binary section is too large");
        require(static_cast<std::size_t>(size));
        offset_ += static_cast<std::size_t>(size);
    }

    bool empty() const { return offset_ == data_.size(); }

private:
    void require(std::size_t size) const
    {
        if (size > data_.size() - offset_)
            throw std::runtime_error("truncated FTLPU binary");
    }

    std::span<const std::uint8_t> data_;
    std::size_t offset_{0};
};

BinaryBinding read_binding(ByteReader& reader, std::uint32_t version)
{
    BinaryBinding binding;
    binding.index = reader.read<std::uint32_t>();
    binding.access =
        static_cast<BindingAccess>(reader.read<std::uint16_t>());
    binding.element_type =
        static_cast<BindingElementType>(reader.read<std::uint16_t>());
    binding.layout =
        static_cast<BindingLayout>(reader.read<std::uint16_t>());
    const auto rank = reader.read<std::uint16_t>();
    const auto slice_count = reader.read<std::uint16_t>();
    const auto hemisphere_mask = reader.read<std::uint16_t>();
    binding.hemisphere_mask = version >= 3 ? hemisphere_mask : 1;
    binding.byte_size = reader.read<std::uint64_t>();
    binding.base_row = reader.read<std::int64_t>();
    binding.instruction_count = reader.read<std::int64_t>();
    binding.address_stride = reader.read<std::int64_t>();
    binding.shape.reserve(rank);
    binding.slices.reserve(slice_count);
    for (std::uint16_t i = 0; i < rank; ++i)
        binding.shape.push_back(reader.read<std::uint64_t>());
    for (std::uint16_t i = 0; i < slice_count; ++i)
        binding.slices.push_back(reader.read<std::uint16_t>());
    if (version >= 6) {
        const auto role_size = reader.read<std::uint16_t>();
        const auto name_size = reader.read<std::uint16_t>();
        binding.ready_cycle = reader.read<std::uint64_t>();
        if (version >= 7) {
            binding.initializer = static_cast<BindingInitializer>(
                reader.read<std::uint16_t>());
            (void)reader.read<std::uint16_t>();
            binding.rope_theta = reader.read<float>();
            binding.rope_head_dim = reader.read<std::uint32_t>();
        } else if (binding.layout == BindingLayout::Fp32CausalMaskTile) {
            binding.initializer = BindingInitializer::CausalMask;
        }
        binding.role.resize(role_size);
        binding.name.resize(name_size);
        reader.read_bytes(binding.role.data(), binding.role.size());
        reader.read_bytes(binding.name.data(), binding.name.size());
    }
    return binding;
}

void write_timeline(std::ostream& os, const BinaryTimeline& timeline)
{
    if (timeline.name.empty()
        || timeline.name.size() > std::numeric_limits<std::uint16_t>::max()
        || timeline.end_cycle < timeline.start_cycle)
        throw std::runtime_error("invalid FTLPU binary timeline");
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(timeline.name.size()));
    write_scalar<std::uint16_t>(os, 0);
    write_scalar<std::uint64_t>(os, timeline.start_cycle);
    write_scalar<std::uint64_t>(os, timeline.end_cycle);
    os.write(timeline.name.data(),
        static_cast<std::streamsize>(timeline.name.size()));
    if (!os) throw std::runtime_error("failed to write timeline metadata");
}

BinaryTimeline read_timeline(std::istream& is)
{
    BinaryTimeline timeline;
    const auto name_size = read_scalar<std::uint16_t>(is);
    (void)read_scalar<std::uint16_t>(is);
    timeline.start_cycle = read_scalar<std::uint64_t>(is);
    timeline.end_cycle = read_scalar<std::uint64_t>(is);
    timeline.name.resize(name_size);
    is.read(timeline.name.data(),
        static_cast<std::streamsize>(timeline.name.size()));
    if (!is || timeline.name.empty()
        || timeline.end_cycle < timeline.start_cycle)
        throw std::runtime_error("invalid FTLPU binary timeline");
    return timeline;
}

BinaryTimeline read_timeline(ByteReader& reader)
{
    BinaryTimeline timeline;
    const auto name_size = reader.read<std::uint16_t>();
    (void)reader.read<std::uint16_t>();
    timeline.start_cycle = reader.read<std::uint64_t>();
    timeline.end_cycle = reader.read<std::uint64_t>();
    timeline.name.resize(name_size);
    reader.read_bytes(timeline.name.data(), timeline.name.size());
    if (timeline.name.empty()
        || timeline.end_cycle < timeline.start_cycle)
        throw std::runtime_error("invalid FTLPU binary timeline");
    return timeline;
}

struct BinaryHeader {
    BinaryProgram program;
    std::uint32_t version;
    std::uint32_t queue_count;
    std::uint32_t relocation_count;
    std::uint32_t address_relocation_count;
};

BinaryHeader read_header(ByteReader& reader)
{
    std::array<char, 8> magic {};
    reader.read_bytes(magic.data(), magic.size());
    if (magic != kMagic)
        throw std::runtime_error("invalid FTLPU binary magic");
    const auto version = reader.read<std::uint32_t>();
    if (version < 1 || version > kBinaryFormatVersion)
        throw std::runtime_error("unsupported FTLPU binary version");

    BinaryProgram program;
    if (version >= 5) {
        program.target_abi = reader.read<std::uint64_t>();
        const auto target_name_size = reader.read<std::uint16_t>();
        program.target_name.resize(target_name_size);
        reader.read_bytes(
            program.target_name.data(), program.target_name.size());
        if (version >= 16) {
            program.hardware.visit([&](std::uint32_t& value) {
                value = reader.read<std::uint32_t>();
            });
        } else {
            if (version >= 9)
                program.hardware.sram_depth_rows =
                    reader.read<std::uint32_t>();
            if (version >= 15)
                program.hardware.mxms_per_hemisphere =
                    reader.read<std::uint32_t>();
        }
    } else {
        program.target_name = "legacy-unidentified";
        program.target_abi = 0;
    }
    program.max_cycle =
        static_cast<std::size_t>(reader.read<std::uint64_t>());
    const auto queue_count = reader.read<std::uint32_t>();
    const auto binding_count =
        version >= 2 ? reader.read<std::uint32_t>() : 0;
    const auto relocation_count =
        version >= 8 ? reader.read<std::uint32_t>() : 0;
    const auto address_relocation_count =
        version >= 9 ? reader.read<std::uint32_t>() : 0;
    const auto timeline_count =
        version >= 11 ? reader.read<std::uint32_t>() : 0;
    const auto memory_floor_count =
        version >= 12 ? reader.read<std::uint32_t>() : 0;
    program.bindings.reserve(binding_count);
    for (std::uint32_t index = 0; index < binding_count; ++index)
        program.bindings.push_back(read_binding(reader, version));
    program.timelines.reserve(timeline_count);
    for (std::uint32_t index = 0; index < timeline_count; ++index)
        program.timelines.push_back(read_timeline(reader));
    program.memory_floors.reserve(memory_floor_count);
    for (std::uint32_t index = 0;
         index < memory_floor_count; ++index) {
        program.memory_floors.push_back(BinaryMemoryFloor {
            reader.read<std::uint16_t>(),
            reader.read<std::uint16_t>(),
            reader.read<std::uint32_t>(),
        });
    }
    return {std::move(program), version, queue_count, relocation_count,
        address_relocation_count};
}

void skip_bytes(std::istream& is, std::uint64_t bytes)
{
    if (bytes > static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("FTLPU binary section is too large");
    is.seekg(static_cast<std::streamoff>(bytes), std::ios_base::cur);
    if (!is) throw std::runtime_error("truncated FTLPU binary");
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
    if (program.target_abi != executable_target_abi(program.hardware))
        throw std::runtime_error(
            "FTLPU target ABI does not match its hardware configuration");
    if (program.hardware.mxms_per_hemisphere == 0)
        throw std::runtime_error(
            "FTLPU binary requires a nonzero MXM topology");
    program.hardware.visit([&](std::uint32_t value) {
        write_scalar<std::uint32_t>(os, value);
    });
    write_scalar<std::uint64_t>(os, static_cast<std::uint64_t>(program.max_cycle));
    write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(program.queues.size()));
    write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(program.bindings.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.scale_relocations.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.address_relocations.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.timelines.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.memory_floors.size()));

    for (const auto& binding : program.bindings) write_binding(os, binding);
    for (const auto& timeline : program.timelines)
        write_timeline(os, timeline);
    for (const BinaryMemoryFloor& floor : program.memory_floors) {
        write_scalar<std::uint16_t>(os, floor.hemisphere);
        write_scalar<std::uint16_t>(os, floor.slice);
        write_scalar<std::uint32_t>(os, floor.first_free_row);
    }

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
    for (const auto& relocation : program.address_relocations) {
        write_scalar<std::uint32_t>(os, relocation.binding_index);
        write_scalar<std::uint16_t>(
            os, static_cast<std::uint16_t>(relocation.binding_access));
        write_scalar<std::uint16_t>(
            os, static_cast<std::uint16_t>(relocation.queue_kind));
        write_scalar<std::uint16_t>(os, relocation.queue_index);
        write_scalar<std::uint32_t>(os, relocation.command_index);
        write_scalar<std::uint16_t>(
            os, relocation.write_port ? 1 : 0);
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
        if (version >= 16) {
            program.hardware.visit([&](std::uint32_t& value) {
                value = read_scalar<std::uint32_t>(is);
            });
        } else {
            if (version >= 9)
                program.hardware.sram_depth_rows =
                    read_scalar<std::uint32_t>(is);
            if (version >= 15)
                program.hardware.mxms_per_hemisphere =
                    read_scalar<std::uint32_t>(is);
        }
    } else {
        program.target_name = "legacy-unidentified";
        program.target_abi = 0;
    }
    program.max_cycle = static_cast<std::size_t>(read_scalar<std::uint64_t>(is));
    const auto queue_count = read_scalar<std::uint32_t>(is);
    const auto binding_count = version >= 2 ? read_scalar<std::uint32_t>(is) : 0;
    const auto relocation_count =
        version >= 8 ? read_scalar<std::uint32_t>(is) : 0;
    const auto address_relocation_count =
        version >= 9 ? read_scalar<std::uint32_t>(is) : 0;
    const auto timeline_count =
        version >= 11 ? read_scalar<std::uint32_t>(is) : 0;
    const auto memory_floor_count =
        version >= 12 ? read_scalar<std::uint32_t>(is) : 0;
    program.bindings.reserve(binding_count);
    for (std::uint32_t binding_id = 0; binding_id < binding_count; ++binding_id)
        program.bindings.push_back(read_binding(is, version));
    program.timelines.reserve(timeline_count);
    for (std::uint32_t timeline_id = 0;
         timeline_id < timeline_count; ++timeline_id)
        program.timelines.push_back(read_timeline(is));
    program.memory_floors.reserve(memory_floor_count);
    for (std::uint32_t index = 0;
         index < memory_floor_count; ++index) {
        program.memory_floors.push_back(BinaryMemoryFloor {
            read_scalar<std::uint16_t>(is),
            read_scalar<std::uint16_t>(is),
            read_scalar<std::uint32_t>(is),
        });
    }
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
            queue.commands.push_back(std::move(command));
        }
        program.queues.push_back(std::move(queue));
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
    program.address_relocations.reserve(address_relocation_count);
    for (std::uint32_t index = 0;
         index < address_relocation_count; ++index) {
        BinaryAddressRelocation relocation;
        relocation.binding_index = read_scalar<std::uint32_t>(is);
        if (version >= 10)
            relocation.binding_access = static_cast<BindingAccess>(
                read_scalar<std::uint16_t>(is));
        relocation.queue_kind =
            static_cast<QueueKind>(read_scalar<std::uint16_t>(is));
        relocation.queue_index = read_scalar<std::uint16_t>(is);
        relocation.command_index = read_scalar<std::uint32_t>(is);
        if (version >= 14)
            relocation.write_port =
                read_scalar<std::uint16_t>(is) != 0;
        program.address_relocations.push_back(relocation);
    }

    return program;
}

BinaryProgram read_binary_program_metadata(std::istream& is)
{
    std::array<char, 8> magic {};
    is.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!is || magic != kMagic)
        throw std::runtime_error("invalid FTLPU binary magic");

    const auto version = read_scalar<std::uint32_t>(is);
    if (version < 1 || version > kBinaryFormatVersion)
        throw std::runtime_error("unsupported FTLPU binary version");

    BinaryProgram program;
    if (version >= 5) {
        program.target_abi = read_scalar<std::uint64_t>(is);
        const auto target_name_size = read_scalar<std::uint16_t>(is);
        program.target_name.resize(target_name_size);
        is.read(program.target_name.data(),
            static_cast<std::streamsize>(program.target_name.size()));
        if (!is) throw std::runtime_error("truncated FTLPU target name");
        if (version >= 16) {
            program.hardware.visit([&](std::uint32_t& value) {
                value = read_scalar<std::uint32_t>(is);
            });
        } else {
            if (version >= 9)
                program.hardware.sram_depth_rows =
                    read_scalar<std::uint32_t>(is);
            if (version >= 15)
                program.hardware.mxms_per_hemisphere =
                    read_scalar<std::uint32_t>(is);
        }
    } else {
        program.target_name = "legacy-unidentified";
        program.target_abi = 0;
    }
    program.max_cycle =
        static_cast<std::size_t>(read_scalar<std::uint64_t>(is));
    const auto queue_count = read_scalar<std::uint32_t>(is);
    const auto binding_count =
        version >= 2 ? read_scalar<std::uint32_t>(is) : 0;
    const auto relocation_count =
        version >= 8 ? read_scalar<std::uint32_t>(is) : 0;
    const auto address_relocation_count =
        version >= 9 ? read_scalar<std::uint32_t>(is) : 0;
    const auto timeline_count =
        version >= 11 ? read_scalar<std::uint32_t>(is) : 0;
    const auto memory_floor_count =
        version >= 12 ? read_scalar<std::uint32_t>(is) : 0;

    program.bindings.reserve(binding_count);
    for (std::uint32_t index = 0; index < binding_count; ++index)
        program.bindings.push_back(read_binding(is, version));
    program.timelines.reserve(timeline_count);
    for (std::uint32_t index = 0; index < timeline_count; ++index)
        program.timelines.push_back(read_timeline(is));
    program.memory_floors.reserve(memory_floor_count);
    for (std::uint32_t index = 0;
         index < memory_floor_count; ++index) {
        program.memory_floors.push_back(BinaryMemoryFloor {
            read_scalar<std::uint16_t>(is),
            read_scalar<std::uint16_t>(is),
            read_scalar<std::uint32_t>(is),
        });
    }

    for (std::uint32_t queue = 0; queue < queue_count; ++queue) {
        (void)read_scalar<std::uint16_t>(is);
        (void)read_scalar<std::uint16_t>(is);
        const auto command_count = read_scalar<std::uint32_t>(is);
        for (std::uint32_t command = 0;
             command < command_count; ++command) {
            (void)read_scalar<std::uint32_t>(is);
            (void)read_scalar<std::uint16_t>(is);
            (void)read_scalar<std::uint16_t>(is);
            skip_bytes(is, (version >= 3 ? 4u : 3u)
                    * sizeof(std::uint32_t));
            if (version >= 4) {
                const auto extension_count =
                    read_scalar<std::uint16_t>(is);
                skip_bytes(is, static_cast<std::uint64_t>(extension_count)
                        * sizeof(std::uint32_t));
            }
        }
    }

    skip_bytes(is, static_cast<std::uint64_t>(relocation_count)
            * (sizeof(std::uint32_t) * 3 + sizeof(std::uint16_t) * 3));
    const std::uint64_t address_relocation_size =
        sizeof(std::uint32_t) * 2 + sizeof(std::uint16_t) * 2
        + (version >= 10 ? sizeof(std::uint16_t) : 0);
    skip_bytes(is,
        static_cast<std::uint64_t>(address_relocation_count)
            * address_relocation_size);
    return program;
}

BinaryProgram read_binary_program(std::span<const std::uint8_t> data)
{
    ByteReader reader(data);
    BinaryHeader header = read_header(reader);
    BinaryProgram& program = header.program;
    program.queues.reserve(header.queue_count);
    for (std::uint32_t queue_id = 0;
         queue_id < header.queue_count; ++queue_id) {
        QueueProgram queue;
        queue.kind =
            static_cast<QueueKind>(reader.read<std::uint16_t>());
        queue.index = reader.read<std::uint16_t>();
        const auto command_count = reader.read<std::uint32_t>();
        queue.commands.reserve(command_count);
        for (std::uint32_t command_id = 0;
             command_id < command_count; ++command_id) {
            QueueCommand command;
            command.command = reader.read<std::uint32_t>();
            command.instruction_kind =
                static_cast<InstructionKind>(
                    reader.read<std::uint16_t>());
            command.word_count = reader.read<std::uint16_t>();
            const std::size_t serialized_words =
                header.version >= 3 ? command.words.size() : 3;
            for (std::size_t word = 0;
                 word < serialized_words; ++word)
                command.words[word] = reader.read<std::uint32_t>();
            if (header.version >= 4) {
                const auto extension_count =
                    reader.read<std::uint16_t>();
                command.extension_words.reserve(extension_count);
                for (std::uint16_t word = 0;
                     word < extension_count; ++word)
                    command.extension_words.push_back(
                        reader.read<std::uint32_t>());
            }
            queue.commands.push_back(std::move(command));
        }
        program.queues.push_back(std::move(queue));
    }
    program.scale_relocations.reserve(header.relocation_count);
    for (std::uint32_t index = 0;
         index < header.relocation_count; ++index) {
        BinaryScaleRelocation relocation;
        relocation.binding_index = reader.read<std::uint32_t>();
        relocation.scale_index = reader.read<std::uint32_t>();
        relocation.queue_kind =
            static_cast<QueueKind>(reader.read<std::uint16_t>());
        relocation.queue_index = reader.read<std::uint16_t>();
        relocation.command_index = reader.read<std::uint32_t>();
        relocation.operand = static_cast<VxmImmediateOperand>(
            reader.read<std::uint16_t>());
        program.scale_relocations.push_back(relocation);
    }
    program.address_relocations.reserve(
        header.address_relocation_count);
    for (std::uint32_t index = 0;
         index < header.address_relocation_count; ++index) {
        BinaryAddressRelocation relocation;
        relocation.binding_index = reader.read<std::uint32_t>();
        if (header.version >= 10)
            relocation.binding_access = static_cast<BindingAccess>(
                reader.read<std::uint16_t>());
        relocation.queue_kind =
            static_cast<QueueKind>(reader.read<std::uint16_t>());
        relocation.queue_index = reader.read<std::uint16_t>();
        relocation.command_index = reader.read<std::uint32_t>();
        if (header.version >= 14)
            relocation.write_port =
                reader.read<std::uint16_t>() != 0;
        program.address_relocations.push_back(relocation);
    }
    if (!reader.empty())
        throw std::runtime_error("FTLPU binary has trailing data");
    return std::move(program);
}

BinaryProgram read_binary_program_metadata(
    std::span<const std::uint8_t> data)
{
    ByteReader reader(data);
    BinaryHeader header = read_header(reader);
    for (std::uint32_t queue = 0;
         queue < header.queue_count; ++queue) {
        (void)reader.read<std::uint16_t>();
        (void)reader.read<std::uint16_t>();
        const auto command_count = reader.read<std::uint32_t>();
        for (std::uint32_t command = 0;
             command < command_count; ++command) {
            (void)reader.read<std::uint32_t>();
            (void)reader.read<std::uint16_t>();
            (void)reader.read<std::uint16_t>();
            reader.skip((header.version >= 3 ? 4u : 3u)
                * sizeof(std::uint32_t));
            if (header.version >= 4) {
                const auto extension_count =
                    reader.read<std::uint16_t>();
                reader.skip(
                    static_cast<std::uint64_t>(extension_count)
                    * sizeof(std::uint32_t));
            }
        }
    }
    reader.skip(static_cast<std::uint64_t>(header.relocation_count)
        * (sizeof(std::uint32_t) * 3 + sizeof(std::uint16_t) * 3));
    const std::uint64_t address_relocation_size =
        sizeof(std::uint32_t) * 2 + sizeof(std::uint16_t) * 2
        + (header.version >= 10 ? sizeof(std::uint16_t) : 0);
    reader.skip(
        static_cast<std::uint64_t>(header.address_relocation_count)
        * address_relocation_size);
    if (!reader.empty())
        throw std::runtime_error("FTLPU binary has trailing data");
    return std::move(header.program);
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
