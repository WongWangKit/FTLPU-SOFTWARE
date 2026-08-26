#pragma once

#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/target_abi.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {

inline constexpr std::uint32_t kBinaryFormatVersion = 24;

enum class BindingAccess : std::uint16_t {
    Input = 0,
    Output = 1,
    Internal = 2,
};

enum class BindingElementType : std::uint16_t {
    I8 = 1,
    I32 = 2,
    F16 = 3,
    F32 = 4,
    BF16 = 5,
};

enum class BindingLayout : std::uint16_t {
    Vector = 1,
    MxmWeightStriped = 2,
    Int32BytePlanar = 3,
    Fp16BytePlanar = 4,
    Fp16MxmActivationPlanar = 5,
    W8A16MxmWeightStriped = 6,
    Fp16PairPlanar = 7,
    W8A16AttentionWeightStriped = 8,
    W8A16MxmWeightWaveStriped = 9,
    Fp32CausalMaskTile = 10,
    Fp16SxmDistributed16 = 11,
    Fp16VxmDistributed16 = 12,
    Fp16MxmDistributed16 = 13,
    Fp16RopeTable = 14,
    W8A16Block8WeightWaveStriped = 15,
    Fp16MxmBlock8Distributed16 = 16,
    Fp16VxmRowParallel8 = 17,
    W8A16MxmWeightReplicated = 18,
    Fp16CausalMaskTile = 19,
    Fp16ProbabilityX16 = 20,
    Fp16ProbabilityDiagonal = 21,
    Fp16VxmGammaBroadcast = 22,
};

enum class BindingInitializer : std::uint16_t {
    None = 0,
    Zero = 1,
    CausalMask = 2,
    RopeTable = 3,
};

struct BinaryBinding {
    std::uint32_t index{0};
    BindingAccess access{BindingAccess::Input};
    BindingElementType element_type{BindingElementType::I8};
    BindingLayout layout{BindingLayout::Vector};
    std::uint64_t byte_size{0};
    std::int64_t base_row{0};
    std::int64_t instruction_count{0};
    std::int64_t address_stride{0};
    std::vector<std::uint64_t> shape{};
    std::vector<std::uint16_t> slices{};
    // Stable semantic identity and the first cycle at which consumers may
    // observe the complete binding contents.
    std::string role{};
    std::string name{};
    std::uint64_t ready_cycle{0};
    // Bit 0 selects east and bit 1 selects west. Inputs may be replicated to both.
    std::uint16_t hemisphere_mask{1};
    // Physical SRAM bank shared by every (hemisphere, slice) placement in
    // this binding. Bank-local addresses remain in base_row.
    std::uint16_t bank{0};
    BindingInitializer initializer{BindingInitializer::None};
    float rope_theta{0.0f};
    std::uint32_t rope_head_dim{0};
    // Intra-executable ping-pong paging metadata. A paged binding contains
    // the logical tensor; only one page is resident in each physical bank.
    bool paged_weight{false};
    std::uint32_t page_count{0};
    std::uint32_t page_rows{0};
    std::uint32_t page_granularity{0};
    std::uint32_t page_role_group_base{0};
    std::uint32_t page_role_group_count{0};
    std::uint32_t page_items_per_slice_group{0};
    std::uint32_t page_bank_count{0};
    std::vector<std::uint16_t> page_storage_slices{};
};

struct BinaryWeightPageUse {
    std::uint32_t binding_index{0};
    std::uint32_t page_index{0};
    std::uint16_t bank{0};
    std::uint64_t ready_cycle{0};
    std::uint64_t release_cycle{0};
};

enum class VxmImmediateOperand : std::uint16_t {
    Lhs = 0,
    Rhs = 1,
};

struct BinaryScaleRelocation {
    std::uint32_t binding_index{0};
    std::uint32_t scale_index{0};
    QueueKind queue_kind{QueueKind::Vxm};
    std::uint16_t queue_index{0};
    std::uint32_t command_index{0};
    VxmImmediateOperand operand{VxmImmediateOperand::Rhs};
};

struct BinaryAddressRelocation {
    std::uint32_t binding_index{0};
    BindingAccess binding_access{BindingAccess::Input};
    QueueKind queue_kind{QueueKind::Mem};
    std::uint16_t queue_index{0};
    std::uint32_t command_index{0};
    // ReadWrite has a second independently relocatable SRAM address.
    bool write_port{false};
};

struct BinaryTimeline {
    std::string name{};
    std::uint64_t start_cycle{0};
    std::uint64_t end_cycle{0};
};

struct BinaryMemoryFloor {
    std::uint16_t hemisphere{0};
    std::uint16_t slice{0};
    std::uint32_t first_free_row{0};
    std::uint16_t bank{0};
};

struct BinaryProgram {
    std::string target_name{std::string(kLpu32StreamTargetName)};
    std::uint64_t target_abi{kLpu32StreamTargetAbi};
    ExecutableHardwareConfig hardware{};
    std::size_t max_cycle{0};
    std::vector<QueueProgram> queues{};
    std::vector<BinaryBinding> bindings{};
    std::vector<BinaryTimeline> timelines{};
    // Per-physical-slice upper bounds for statically addressed command
    // scratch. Resident allocations start at or above these rows.
    std::vector<BinaryMemoryFloor> memory_floors{};
    std::vector<BinaryScaleRelocation> scale_relocations{};
    std::vector<BinaryAddressRelocation> address_relocations{};
    std::vector<BinaryWeightPageUse> weight_page_uses{};
};

void write_binary_program(const BinaryProgram& program, const std::filesystem::path& path);
BinaryProgram read_binary_program(const std::filesystem::path& path);
void write_binary_program(const BinaryProgram& program, std::ostream& stream);
BinaryProgram read_binary_program(std::istream& stream);
BinaryProgram read_binary_program(std::span<const std::uint8_t> data);
BinaryProgram read_binary_program_metadata(std::istream& stream);
BinaryProgram read_binary_program_metadata(
    std::span<const std::uint8_t> data);

} // namespace ftlpu::software::runtime
