#include "ftlpu/software/runtime/binary.hpp"

#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rt = ftlpu::software::runtime;
namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
template <typename Function>
void rejects(Function&& function, const char* message)
{
    try { function(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
std::span<const std::uint8_t> bytes(const std::string& data)
{
    return {reinterpret_cast<const std::uint8_t*>(data.data()), data.size()};
}
std::string serialize(const rt::BinaryProgram& program)
{
    std::ostringstream stream(std::ios::out | std::ios::binary);
    rt::write_binary_program(program, stream);
    return stream.str();
}
void put32(std::string& data, std::size_t offset, std::uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i) data[offset + i] = static_cast<char>(value >> (8 * i));
}
template <typename T>
void append(std::string& data, T value)
{
    for (unsigned i = 0; i < sizeof(T); ++i) data.push_back(static_cast<char>(value >> (8 * i)));
}
std::string legacy_v36(unsigned version = 36)
{
    rt::ExecutableHardwareConfig hardware;
    rt::TargetAbiHasher legacy;
    legacy.add(21);
    hardware.visit_command_abi([&](std::uint32_t value) { legacy.add(value); });
    if (version == 37) { legacy.add(hardware.common_barrier_encoding_version); legacy.add(hardware.common_barrier_capacity); }
    std::string data = "FTLPUB01";
    append<std::uint32_t>(data, version);
    append<std::uint64_t>(data, legacy.value());
    const std::string name(rt::kLpu32StreamTargetName);
    append<std::uint16_t>(data, static_cast<std::uint16_t>(name.size()));
    data += name;
    const auto field = [&](std::uint32_t value) { append<std::uint32_t>(data,value); };
    if (version == 37) hardware.visit_pre_v38(field); else hardware.visit_pre_v37(field);
    append<std::uint64_t>(data, 0);
    for (unsigned i = 0; i < (version == 37 ? 11u : 8u); ++i) append<std::uint32_t>(data, 0);
    if (version == 37) { data += "BARR"; append<std::uint16_t>(data,1); append<std::uint16_t>(data,8); append<std::uint32_t>(data,0); append<std::uint32_t>(data,0); }
    return data;
}
rt::BinaryProgram fixture()
{
    rt::BinaryProgram program;
    program.max_cycle = 30;
    program.common_barrier_protocol = true;
    program.transport_overlap_protocol = true;
    rt::BinaryBinding binding;
    binding.index = 7;
    binding.paged_weight = true;
    binding.page_count = 2;
    binding.page_bank_count = 2;
    binding.page_rows = 1;
    program.bindings.push_back(binding);
    program.weight_page_uses = {{7, 0, 0, 10, 15, true, 0x10001},
        {7, 1, 1, 10, 15, true, 0x10001}, {7, 0, 0, 20, 30, true, 0x10002}};
    for (auto& use : program.weight_page_uses) use.transport_launch_cycle = 3;
    program.barrier_events = {{41, {0, 1}}, {42, {2}}};
    program.barriers = {{10, 41}, {20, 42}};
    // A control record in an independent Native MEM queue must remain separate
    // from computation after each full reader. Empty Native calculation queue
    // also verifies that its count is not confused with the transport count.
    program.queues.push_back({rt::QueueKind::Mem, 0, {}});
    program.transport_queues.push_back({rt::QueueKind::Mem, 0, {rt::encode_icu_control_raw_word(ftlpu::IcuControlInstruction::Nop(1))}});
    program.address_relocations.push_back({7, rt::BindingAccess::Input,
        rt::QueueKind::Mem, 0, 0, false, true});
    return program;
}
void check_all_readers(const std::string& data, std::size_t expected,
                       bool queues = false)
{
    std::istringstream full(data, std::ios::in | std::ios::binary);
    std::istringstream metadata(data, std::ios::in | std::ios::binary);
    auto a = rt::read_binary_program(full);
    auto b = rt::read_binary_program(bytes(data));
    auto c = rt::read_binary_program_metadata(metadata);
    auto d = rt::read_binary_program_metadata(bytes(data));
    for (const auto* program : {&a, &b, &c, &d}) {
        require(program->barriers.size() == expected, "barriers lost by a reader");
        require(program->barrier_events.size() == expected, "events lost by a reader");
        if (expected != 0) {
            require(program->common_barrier_protocol && program->transport_overlap_protocol
                && program->weight_page_uses[0].transport_launch_cycle == 3, "v38 overlap fields lost");
            require(program->barrier_events[0].page_use_indices == std::vector<std::uint32_t>({0, 1}),
                    "all-of dependencies lost");
            require(program->weight_page_uses.size() == 3
                && program->weight_page_uses[0].transport_launch_event_id == 0x10001,
                "transport instances or launch association lost");
        }
    }
    if (queues) {
        for (const auto* program : {&a, &b}) {
            require(program->queues.size() == 1, "calculation queue count changed");
            require(program->transport_queues.size() == 1, "transport queue lost");
            require(program->transport_queues[0].commands.size() == 1, "transport control lost");
            require(program->address_relocations.size() == 1
                && program->address_relocations[0].transport_queue, "transport relocation selector lost");
        }
    }
}
void reject_all_readers(const std::string& data)
{
    rejects([&] { std::istringstream stream(data); (void)rt::read_binary_program(stream); }, "stream accepted corrupt barriers");
    rejects([&] { (void)rt::read_binary_program(bytes(data)); }, "span accepted corrupt barriers");
    rejects([&] { std::istringstream stream(data); (void)rt::read_binary_program_metadata(stream); }, "metadata stream accepted corrupt barriers");
    rejects([&] { (void)rt::read_binary_program_metadata(bytes(data)); }, "metadata span accepted corrupt barriers");
}
}
int main()
try {
    rt::BinaryProgram empty;
    const auto empty_data = serialize(empty);
    require(empty_data.find("BARR") != std::string::npos, "v37 empty table missing explicit block");
    check_all_readers(empty_data, 0);
    auto unsupported_target = empty;
    unsupported_target.hardware.common_barrier_encoding_version = 0;
    unsupported_target.hardware.common_barrier_capacity = 0;
    unsupported_target.target_abi = rt::executable_target_abi(unsupported_target.hardware);
    check_all_readers(serialize(unsupported_target), 0);
    unsupported_target.common_barrier_protocol = true;
    rejects([&] { (void)serialize(unsupported_target); }, "unsupported target accepted the protocol");
    const auto program = fixture();
    const auto data = serialize(program);
    const auto block = data.find("BARR");
    require(block != std::string::npos, "barrier block missing");
    require(data.substr(block, 16) == std::string("BARR\1\0\10\0\2\0\0\0\0\0\0\0", 16), "BarrierBlock is not the explicit LE header");
    require(static_cast<std::uint8_t>(data[block + 16]) == 10, "stop cycle LE encoding changed");
    check_all_readers(data, 2, true);
    std::string launchPattern;
    append<std::uint32_t>(launchPattern,0x10001); append<std::uint32_t>(launchPattern,3);
    const auto launchOffset=data.find(launchPattern);
    require(launchOffset!=std::string::npos,"test launch field not found");
    for (const auto cycle : {4u,11u,0xffffffffu}) {
        auto corrupt=data; put32(corrupt,launchOffset+4,cycle); reject_all_readers(corrupt);
    }
    check_all_readers(legacy_v36(), 0);
    check_all_readers(legacy_v36(37), 0);
    for (const auto [offset, value] : {std::pair<std::size_t, std::uint32_t>{0, 'X'},
            {4, 2}, {6, 7}, {12, 1}}) {
        auto bad = data;
        bad[block + offset] = static_cast<char>(value);
        reject_all_readers(bad);
    }
    for (const auto [offset, value] : {std::pair<std::size_t, std::uint32_t>{8, 1025},
            {16 + 4, 99}, {24, 10}}) {
        auto bad = data;
        put32(bad, block + offset, value);
        reject_all_readers(bad);
    }
    auto truncated = data.substr(0, block + 16 + 15);
    reject_all_readers(truncated);
    auto bad_dependency = data;
    // Second event is the final 12 metadata bytes before BARR.
    put32(bad_dependency, block - 4, 99);
    reject_all_readers(bad_dependency);
    for (unsigned mutation = 0; mutation < 17; ++mutation) {
        auto bad = fixture();
        if (mutation == 0) bad.barrier_events[1].event_id = 41;
        if (mutation == 1) bad.barrier_events[1].page_use_indices = {0};
        if (mutation == 2) bad.barriers[1].stop_cycle = 10;
        if (mutation == 3) bad.barrier_events[0].page_use_indices.clear();
        if (mutation == 4) bad.common_barrier_protocol = false;
        if (mutation == 5) bad.weight_page_uses[0].page_index = 2;
        if (mutation == 6) bad.hardware.common_barrier_capacity = 1;
        if (mutation == 7) bad.max_cycle = std::uint64_t{1} << 32;
        if (mutation == 8) bad.weight_page_uses[0].transport_launch_event_id = 0;
        if (mutation == 9) bad.weight_page_uses[2].transport_launch_event_id = 0x10001;
        if (mutation == 10) bad.barrier_events[0].page_use_indices.assign(65536, 0);
        if (mutation == 11) bad.bindings[0].access = rt::BindingAccess::Internal;
        if (mutation == 12) bad.weight_page_uses[0].bank = 1;
        if (mutation == 13) bad.weight_page_uses[0].transport_launch_cycle = 11;
        if (mutation == 14) bad.weight_page_uses[1].transport_launch_cycle = 4;
        if (mutation == 15) bad.hardware.common_transport_overlap_version = 0;
        if (mutation == 16) bad.weight_page_uses[0].transport_launch_event_id = 0;
        rejects([&] { (void)serialize(bad); }, "writer accepted invalid common barriers");
    }
    rejects([] { (void)rt::make_binary_barrier(std::uint64_t{1} << 32, 1); }, "linker truncated cycle overflow");
    rejects([] { (void)rt::make_binary_barrier(0, 0); }, "zero event accepted");
    require(rt::make_binary_barrier(std::numeric_limits<std::uint32_t>::max(), 1).stop_cycle
            == std::numeric_limits<std::uint32_t>::max(), "maximum cycle changed");
    std::cout << "binary_barrier_test passed: v38 four-reader roundtrip, v36/v37 compatibility, corruptions, instances, capacity and cycle bounds\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
