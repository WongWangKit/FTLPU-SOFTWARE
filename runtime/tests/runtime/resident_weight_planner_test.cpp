#include "ftlpu/software/runtime/session_memory_planner.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;

BinaryBinding binding(std::uint32_t index, std::int64_t base,
    std::int64_t rows, std::vector<std::uint16_t> slices,
    BindingElementType type, BindingLayout layout,
    std::uint64_t bytes, std::string role)
{
    BinaryBinding result;
    result.index = index;
    result.access = BindingAccess::Input;
    result.element_type = type;
    result.layout = layout;
    result.byte_size = bytes;
    result.base_row = base;
    result.instruction_count = rows;
    result.address_stride = 1;
    result.shape = type == BindingElementType::I8
        ? std::vector<std::uint64_t> {32, bytes / 32}
        : std::vector<std::uint64_t> {bytes / 2};
    result.slices = std::move(slices);
    result.role = std::move(role);
    result.name = "input." + std::to_string(index);
    result.hemisphere_mask = 3;
    return result;
}

BinaryProgram program(std::vector<std::uint16_t> weight_slices,
    std::vector<std::uint16_t> gamma_slices)
{
    BinaryProgram result;
    result.memory_rows_per_slice = 65536;
    result.bindings.push_back(binding(0, 0, 2304, gamma_slices,
        BindingElementType::F16, BindingLayout::Fp16MxmDistributed16,
        128 * 576 * 2, "activation"));
    result.bindings.push_back(binding(1, 5120, 72, gamma_slices,
        BindingElementType::F16, BindingLayout::Fp16VxmDistributed16,
        576 * 2, "weight"));
    result.bindings.push_back(binding(2, 0, 720, weight_slices,
        BindingElementType::I8, BindingLayout::W8A16AttentionWeightStriped,
        576 * 576, "weight"));
    result.bindings.push_back(binding(3, 720, 288, weight_slices,
        BindingElementType::I8, BindingLayout::W8A16AttentionWeightStriped,
        576 * 192, "weight"));
    result.bindings.push_back(binding(4, 1008, 288, weight_slices,
        BindingElementType::I8, BindingLayout::W8A16AttentionWeightStriped,
        576 * 192, "weight"));
    result.bindings.push_back(binding(5, 1296, 648, weight_slices,
        BindingElementType::I8, BindingLayout::W8A16AttentionWeightStriped,
        576 * 576, "weight"));
    result.bindings.push_back(binding(6, 5192, 72, gamma_slices,
        BindingElementType::F16, BindingLayout::Fp16VxmDistributed16,
        576 * 2, "weight"));
    result.bindings.push_back(binding(7, 10000, 1728, weight_slices,
        BindingElementType::I8, BindingLayout::W8A16MxmWeightStriped,
        576 * 1536, "weight"));
    result.bindings.push_back(binding(8, 11728, 1728, weight_slices,
        BindingElementType::I8, BindingLayout::W8A16MxmWeightStriped,
        576 * 1536, "weight"));
    result.bindings.push_back(binding(9, 13456, 3456, weight_slices,
        BindingElementType::I8, BindingLayout::W8A16MxmWeightStriped,
        1536 * 576, "weight"));
    BinaryBinding output = result.bindings.front();
    output.access = BindingAccess::Output;
    output.index = 0;
    output.base_row = 5632;
    output.role = "result";
    result.bindings.push_back(std::move(output));

    const auto encoded =
        ftlpu::isa::encode_mem_instruction(
            ftlpu::MemInstruction::Write(7000, 0));
    QueueProgram scratch;
    scratch.kind = QueueKind::Mem;
    scratch.index = 0;
    scratch.commands.push_back(QueueCommand {
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mem,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {
            static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32),
            0,
            0,
        },
    });
    scratch.commands.push_back(QueueCommand {
        ftlpu::isa::encode_icu_repeat(
            ftlpu::InstructionControlUnit::Repeat {2, 16, 64}),
    });
    result.queues.push_back(std::move(scratch));
    result.memory_floors.push_back({0, 0, 7129});
    return result;
}

} // namespace

int main()
try {
    ModelPackage package;
    const std::vector<std::vector<std::uint16_t>> weight_groups {
        {0, 4, 8, 12, 16, 20, 24, 28},
        {1, 5, 9, 13, 17, 21, 25, 29},
        {2, 6, 10, 14, 18, 22, 26, 30},
        {3, 7, 11, 15, 19, 23, 27, 31},
        {32, 33, 34, 35, 36, 37, 38, 39},
    };
    const std::vector<std::uint16_t> gamma_slices {
        40, 41, 42, 43, 24, 25, 26, 27,
        28, 29, 30, 31, 16, 17, 18, 19};
    for (const auto& group : weight_groups)
        package.executables.push_back(
            {"decoder.variant", program(group, gamma_slices), {}});

    package.values.push_back(
        {"hidden.0", BindingElementType::F16, {128, 576}, true, false});
    for (std::size_t layer = 0; layer < 30; ++layer) {
        const std::string prefix =
            "layers." + std::to_string(layer) + ".";
        std::vector<ModelBindingRef> inputs {{0,
            "hidden." + std::to_string(layer)}};
        for (std::uint32_t index = 1; index <= 9; ++index) {
            const std::string name =
                prefix + "constant." + std::to_string(index);
            package.tensors.push_back(ModelTensor {
                name, index == 1 || index == 6
                    ? BindingElementType::F16
                    : BindingElementType::I8});
            inputs.push_back({index, name});
        }
        const std::string output =
            "hidden." + std::to_string(layer + 1);
        package.values.push_back({output, BindingElementType::F16,
            {128, 576}, false, layer == 29});
        package.invocations.push_back({
            "layers." + std::to_string(layer),
            static_cast<std::uint32_t>(layer / 6),
            std::move(inputs), {{0, output}}});
    }

    const SessionMemoryPlan plan =
        SessionMemoryPlanner::plan(package);
    if (plan.invocations.size() != 30
        || plan.resident_tensors.size() != 270)
        throw std::logic_error(
            "30-layer plan did not make every constant resident");

    using Slice = std::tuple<std::uint16_t, std::uint16_t>;
    std::map<Slice, std::vector<std::pair<std::int64_t, std::int64_t>>>
        occupied;
    for (const auto& tensor : plan.resident_tensors) {
        const BinaryBinding& current = tensor.binding;
        const std::int64_t end =
            current.base_row + current.instruction_count;
        if (end > 65536)
            throw std::logic_error(
                "resident constant exceeds physical MEM");
        if ((current.hemisphere_mask & 1) != 0
            && std::find(current.slices.begin(),
                   current.slices.end(), 0)
                != current.slices.end()
            && current.base_row < 7129)
            throw std::logic_error(
                "resident constant overlaps anonymous command scratch");
        for (std::uint16_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            if ((current.hemisphere_mask & (1u << hemisphere)) == 0)
                continue;
            for (std::uint16_t slice : current.slices) {
                auto& intervals = occupied[{hemisphere, slice}];
                for (const auto& [begin, previous_end] : intervals)
                    if (current.base_row < previous_end
                        && begin < end)
                        throw std::logic_error(
                            "resident constants overlap");
                intervals.push_back({current.base_row, end});
            }
        }
    }

    std::cout << "resident_weight_planner_test passed layers=30"
              << " constants=" << plan.resident_tensors.size()
              << " executable_variants="
              << package.executables.size() << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "resident_weight_planner_test failed: "
              << exception.what() << '\n';
    return 1;
}
