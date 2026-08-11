#include "ftlpu/software/runtime/session_memory_planner.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace ftlpu::software::runtime {
namespace {

const BinaryBinding& find_binding(const BinaryProgram& program,
    BindingAccess access, std::uint32_t index)
{
    for (const BinaryBinding& binding : program.bindings)
        if (binding.access == access && binding.index == index)
            return binding;
    throw std::invalid_argument(
        "model invocation references a missing executable binding");
}

const ModelValue* find_value(
    const ModelPackage& package, const std::string& name)
{
    for (const ModelValue& value : package.values)
        if (value.name == name) return &value;
    return nullptr;
}

bool is_tensor(const ModelPackage& package, const std::string& name)
{
    for (const ModelTensor& tensor : package.tensors)
        if (tensor.name == name) return true;
    return false;
}

const ModelState& find_state(
    const ModelPackage& package, const std::string& name)
{
    for (const ModelState& state : package.states)
        if (state.name == name) return state;
    throw std::invalid_argument(
        "model invocation references an unknown persistent state");
}

bool is_state_binding(const ModelInvocation& invocation,
    const BinaryBinding& binding)
{
    if (binding.access != BindingAccess::Internal) return false;
    return std::any_of(invocation.states.begin(), invocation.states.end(),
        [&](const ModelStateBindingRef& ref) {
            return ref.binding_index == binding.index;
        });
}

using PhysicalSlice =
    std::tuple<std::uint64_t, std::uint16_t, std::uint16_t>;

template <typename Callback>
void for_each_physical_slice(const BinaryProgram& program,
    const BinaryBinding& binding, Callback&& callback)
{
    for (std::uint16_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
        if ((binding.hemisphere_mask & (1u << hemisphere)) == 0) continue;
        for (std::uint16_t slice : binding.slices)
            callback(PhysicalSlice {
                program.target_abi, hemisphere, slice});
    }
}

std::pair<std::int64_t, std::int64_t> binding_row_range(
    const BinaryBinding& binding)
{
    if (binding.instruction_count <= 0)
        throw std::invalid_argument(
            "model binding has no physical instruction rows");
    const std::int64_t final_offset =
        (binding.instruction_count - 1) * binding.address_stride;
    return {
        binding.base_row + std::min<std::int64_t>(0, final_offset),
        binding.base_row + std::max<std::int64_t>(0, final_offset) + 1,
    };
}

bool is_embedding_output(
    const ModelPackage& package, const std::string& name)
{
    for (const ModelEmbeddingLookup& lookup : package.embedding_lookups)
        if (lookup.output == name) return true;
    return false;
}

} // namespace

bool bindings_physically_alias(
    const BinaryBinding& source, const BinaryBinding& destination)
{
    return source.element_type == destination.element_type
        && source.layout == destination.layout
        && source.byte_size == destination.byte_size
        && source.base_row == destination.base_row
        && source.instruction_count == destination.instruction_count
        && source.address_stride == destination.address_stride
        && source.shape == destination.shape
        && source.slices == destination.slices
        && source.hemisphere_mask == destination.hemisphere_mask;
}

SessionMemoryPlan SessionMemoryPlanner::plan(const ModelPackage& package)
{
    // Host LM heads execute after this device-only invocation plan.
    SessionMemoryPlan result;
    result.invocations.resize(package.invocations.size());

    struct Producer {
        std::size_t invocation{0};
        std::uint32_t binding_index{0};
    };
    std::unordered_map<std::string, Producer> producers;
    std::unordered_map<std::string, std::vector<std::size_t>> consumers;
    std::map<PhysicalSlice, std::int64_t> reserved_floor;
    std::map<PhysicalSlice, std::int64_t> memory_capacity;
    for (std::size_t invocation_index = 0;
         invocation_index < package.invocations.size(); ++invocation_index) {
        const ModelInvocation& invocation =
            package.invocations[invocation_index];
        for (const ModelBindingRef& output : invocation.outputs) {
            if (!producers.emplace(output.value,
                    Producer {invocation_index, output.binding_index}).second)
                throw std::invalid_argument(
                    "model value has more than one invocation producer");
        }
        for (const ModelBindingRef& input : invocation.inputs)
            consumers[input.value].push_back(invocation_index);
    }

    for (const auto& [name, producer] : producers) {
        const auto consumer = consumers.find(name);
        if (consumer == consumers.end() || consumer->second.empty()) continue;
        const std::size_t last_consumer = consumer->second.back();
        if (last_consumer <= producer.invocation)
            throw std::invalid_argument(
                "model value must be consumed after it is produced");
        result.lifetimes.push_back(
            {name, producer.invocation, last_consumer});
    }

    // Binary bindings do not enumerate every anonymous command scratch
    // interval. Conservatively reserve everything below the highest declared
    // non-resident row, then pack constants downward from the top of MEM.
    for (std::size_t invocation_index = 0;
         invocation_index < package.invocations.size();
         ++invocation_index) {
        const ModelInvocation& invocation =
            package.invocations[invocation_index];
        const BinaryProgram& program =
            package.executables.at(invocation.executable_index).program;
        for (const BinaryMemoryFloor& floor : program.memory_floors) {
            if (floor.hemisphere >= 2
                || floor.first_free_row > program.hardware.sram_depth_rows)
                throw std::invalid_argument(
                    "binary MEM floor exceeds physical MEM");
            const PhysicalSlice slice {
                program.target_abi, floor.hemisphere, floor.slice};
            reserved_floor[slice] = std::max<std::int64_t>(
                reserved_floor[slice], floor.first_free_row);
        }
        for (const BinaryBinding& binding : program.bindings) {
            bool resident_input = false;
            if (binding.access == BindingAccess::Input) {
                for (const ModelBindingRef& input : invocation.inputs) {
                    if (input.binding_index == binding.index
                        && is_tensor(package, input.value)) {
                        resident_input = true;
                        break;
                    }
                }
            }
            if (resident_input || is_state_binding(invocation, binding)
                || binding.slices.empty())
                continue;
            const auto [begin, end] = binding_row_range(binding);
            if (begin < 0
                || end > static_cast<std::int64_t>(
                    program.hardware.sram_depth_rows))
                throw std::invalid_argument(
                    "compiler-assigned binding exceeds physical MEM");
            for_each_physical_slice(program, binding,
                [&](const PhysicalSlice& slice) {
                    reserved_floor[slice] =
                        std::max(reserved_floor[slice], end);
                });
        }
        for (const BinaryBinding& binding : program.bindings) {
            for_each_physical_slice(program, binding,
                [&](const PhysicalSlice& slice) {
                    auto [it, inserted] = memory_capacity.emplace(
                        slice, program.hardware.sram_depth_rows);
                    if (!inserted && it->second
                            != static_cast<std::int64_t>(
                                program.hardware.sram_depth_rows))
                        throw std::invalid_argument(
                            "executables with one target ABI disagree "
                            "on MEM capacity");
                });
        }
    }
    using FreeInterval = std::pair<std::int64_t, std::int64_t>;
    std::map<PhysicalSlice, std::vector<FreeInterval>> free_intervals;
    for (const auto& [slice, capacity] : memory_capacity)
        free_intervals[slice].push_back(
            {reserved_floor[slice], capacity});

    struct ResidentRequest {
        std::size_t invocation_index;
        std::uint32_t binding_index;
        std::string value;
        const BinaryProgram* program;
        BinaryBinding binding;
        std::int64_t extent;
        bool persistent_state{false};
    };
    std::vector<ResidentRequest> resident_requests;
    std::map<PhysicalSlice, std::int64_t> requested_rows;
    std::map<std::pair<std::size_t, std::uint32_t>, BinaryBinding>
        resolved_residents;
    for (std::size_t invocation_index = 0;
         invocation_index < package.invocations.size();
         ++invocation_index) {
        const ModelInvocation& invocation =
            package.invocations[invocation_index];
        const BinaryProgram& program =
            package.executables.at(invocation.executable_index).program;
        for (const ModelBindingRef& input : invocation.inputs) {
            if (!is_tensor(package, input.value)) continue;
            BinaryBinding resident = find_binding(
                program, BindingAccess::Input, input.binding_index);
            const auto [begin, end] = binding_row_range(resident);
            resident_requests.push_back({
                invocation_index, input.binding_index, input.value,
                &program, resident, end - begin, false});
            for_each_physical_slice(program, resident,
                [&](const PhysicalSlice& slice) {
                    requested_rows[slice] += end - begin;
                });
        }
    }
    std::unordered_set<std::string> requested_states;
    for (std::size_t invocation_index = 0;
         invocation_index < package.invocations.size();
         ++invocation_index) {
        const ModelInvocation& invocation =
            package.invocations[invocation_index];
        const BinaryProgram& program =
            package.executables.at(invocation.executable_index).program;
        for (const ModelStateBindingRef& ref : invocation.states) {
            const BinaryBinding& binding = find_binding(
                program, BindingAccess::Internal, ref.binding_index);
            const ModelState& state = find_state(package, ref.state);
            if (binding.element_type != state.element_type
                || binding.shape != state.shape)
                throw std::invalid_argument(
                    "persistent state does not match its executable binding");
            if (!requested_states.insert(ref.state).second) continue;
            const auto [begin, end] = binding_row_range(binding);
            resident_requests.push_back({
                invocation_index, ref.binding_index, ref.state,
                &program, binding, end - begin, true});
            for_each_physical_slice(program, binding,
                [&](const PhysicalSlice& slice) {
                    requested_rows[slice] += end - begin;
                });
        }
    }
    std::map<std::vector<std::uint16_t>, std::int64_t> group_extents;
    for (const ResidentRequest& request : resident_requests)
        group_extents[request.binding.slices] += request.extent;
    std::stable_sort(resident_requests.begin(),
        resident_requests.end(),
        [&](const ResidentRequest& lhs,
            const ResidentRequest& rhs) {
            if (lhs.binding.slices.size()
                != rhs.binding.slices.size())
                return lhs.binding.slices.size()
                    > rhs.binding.slices.size();
            if (group_extents.at(lhs.binding.slices)
                != group_extents.at(rhs.binding.slices))
                return group_extents.at(lhs.binding.slices)
                    > group_extents.at(rhs.binding.slices);
            if (lhs.binding.slices != rhs.binding.slices)
                return lhs.binding.slices < rhs.binding.slices;
            return lhs.extent > rhs.extent;
        });
    for (ResidentRequest& request : resident_requests) {
        const std::int64_t old_begin =
            binding_row_range(request.binding).first;
        std::vector<PhysicalSlice> physical_slices;
        for_each_physical_slice(*request.program, request.binding,
            [&](const PhysicalSlice& slice) {
                physical_slices.push_back(slice);
            });
        if (physical_slices.empty())
            throw std::invalid_argument(
                "resident model tensor has no physical MEM slices: value="
                + request.value);
        std::vector<std::int64_t> candidates;
        for (const PhysicalSlice& slice : physical_slices) {
            for (const auto& [begin, end] : free_intervals.at(slice))
                if (end - begin >= request.extent) {
                    candidates.push_back(begin);
                    candidates.push_back(end - request.extent);
                }
        }
        if (request.binding.slices.size() >= 16)
            std::sort(candidates.begin(), candidates.end());
        else
            std::sort(candidates.begin(), candidates.end(), std::greater<>());
        candidates.erase(
            std::unique(candidates.begin(), candidates.end()),
            candidates.end());
        std::optional<std::int64_t> allocated_begin;
        for (std::int64_t candidate : candidates) {
            const bool fits = std::all_of(
                physical_slices.begin(), physical_slices.end(),
                [&](const PhysicalSlice& slice) {
                    const auto& intervals = free_intervals.at(slice);
                    return std::any_of(intervals.begin(), intervals.end(),
                        [&](const FreeInterval& interval) {
                            return interval.first <= candidate
                                && candidate + request.extent
                                    <= interval.second;
                        });
                });
            if (fits) {
                allocated_begin = candidate;
                break;
            }
        }
        if (!allocated_begin) {
            const auto [target, hemisphere, physical_slice] =
                physical_slices.front();
            std::ostringstream interval_details;
            for (const PhysicalSlice& slice : physical_slices) {
                const auto [slice_target, slice_hemisphere,
                    slice_index] = slice;
                if (slice_hemisphere != hemisphere) continue;
                interval_details << " s" << slice_index << '=';
                for (const auto& [begin, end] : free_intervals.at(slice))
                    interval_details << '[' << begin << ',' << end << ')';
            }
            throw std::invalid_argument(
                "resident model tensor does not fit in a common MEM interval: "
                "value=" + request.value
                + " target_abi=" + std::to_string(target)
                + " hemisphere=" + std::to_string(hemisphere)
                + " slice=" + std::to_string(physical_slice)
                + " extent=" + std::to_string(request.extent)
                + " requested_total="
                + std::to_string(requested_rows[physical_slices.front()])
                + " usable_capacity="
                + std::to_string(
                    static_cast<std::int64_t>(
                        request.program->hardware.sram_depth_rows)
                    - reserved_floor[physical_slices.front()])
                + " free_intervals=" + interval_details.str());
        }
        for (const PhysicalSlice& slice : physical_slices) {
            auto& intervals = free_intervals.at(slice);
            const auto containing = std::find_if(
                intervals.begin(), intervals.end(),
                [&](const FreeInterval& interval) {
                    return interval.first <= *allocated_begin
                        && *allocated_begin + request.extent
                            <= interval.second;
                });
            const FreeInterval original = *containing;
            const auto position =
                static_cast<std::size_t>(containing - intervals.begin());
            intervals.erase(containing);
            if (original.first < *allocated_begin)
                intervals.insert(intervals.begin() + position,
                    {original.first, *allocated_begin});
            if (*allocated_begin + request.extent < original.second) {
                const auto right_position = position
                    + (original.first < *allocated_begin ? 1 : 0);
                intervals.insert(intervals.begin() + right_position,
                    {*allocated_begin + request.extent, original.second});
            }
        }
        request.binding.base_row += *allocated_begin - old_begin;
        if (request.persistent_state) {
            result.persistent_states.push_back(
                {request.value, request.binding});
        } else {
            resolved_residents[
                {request.invocation_index, request.binding_index}] =
                request.binding;
            result.resident_tensors.push_back(
                {request.value, request.binding});
        }
    }

    for (std::size_t invocation_index = 0;
         invocation_index < package.invocations.size(); ++invocation_index) {
        const ModelInvocation& invocation =
            package.invocations[invocation_index];
        const ModelExecutable& executable =
            package.executables.at(invocation.executable_index);
        SessionInvocationPlan& invocation_plan =
            result.invocations[invocation_index];

        for (const ModelBindingRef& input : invocation.inputs) {
            const BinaryBinding& destination = find_binding(
                executable.program, BindingAccess::Input,
                input.binding_index);
            SessionInputPlan input_plan;
            input_plan.binding_index = input.binding_index;
            input_plan.value = input.value;

            const auto producer = producers.find(input.value);
            if (producer == producers.end()) {
                const ModelValue* value = find_value(package, input.value);
                if (!is_tensor(package, input.value)
                    && !is_embedding_output(package, input.value)
                    && (!value || !value->external_input))
                    throw std::invalid_argument(
                        "model invocation input has no available producer");
                if (is_tensor(package, input.value)) {
                    input_plan.transfer = SessionTransferKind::Resident;
                    input_plan.resolved_binding =
                        resolved_residents.at(
                            {invocation_index, input.binding_index});
                } else {
                    input_plan.transfer = SessionTransferKind::HostUpload;
                }
            } else {
                if (producer->second.invocation >= invocation_index)
                    throw std::invalid_argument(
                        "model invocation consumes a future value");
                const ModelInvocation& producer_invocation =
                    package.invocations[producer->second.invocation];
                const ModelExecutable& producer_executable =
                    package.executables.at(
                        producer_invocation.executable_index);
                const BinaryBinding& source = find_binding(
                    producer_executable.program, BindingAccess::Output,
                    producer->second.binding_index);
                if (source.element_type != destination.element_type
                    || source.shape != destination.shape
                    || source.byte_size != destination.byte_size)
                    throw std::invalid_argument(
                        "device-resident value has incompatible bindings");
                if (producer_executable.program.target_abi
                    != executable.program.target_abi)
                    throw std::invalid_argument(
                        "device-resident value cannot cross target ABIs");
                input_plan.transfer =
                    bindings_physically_alias(source, destination)
                    ? SessionTransferKind::DeviceAlias
                    : SessionTransferKind::DeviceCopy;
                input_plan.producer = producer->second.invocation;
                input_plan.release_after_transfer =
                    invocation_index == consumers.at(input.value).back();
            }
            invocation_plan.inputs.push_back(std::move(input_plan));
        }

        for (const ModelBindingRef& output : invocation.outputs) {
            (void)find_binding(executable.program, BindingAccess::Output,
                output.binding_index);
            const ModelValue* value = find_value(package, output.value);
            SessionOutputPlan output_plan;
            output_plan.binding_index = output.binding_index;
            output_plan.value = output.value;
            output_plan.retain_on_device =
                consumers.contains(output.value)
                && !consumers.at(output.value).empty();
            output_plan.download_to_host =
                value && value->external_output;
            invocation_plan.outputs.push_back(std::move(output_plan));
        }
        for (const ModelStateBindingRef& ref : invocation.states) {
            const auto state = std::find_if(
                result.persistent_states.begin(),
                result.persistent_states.end(),
                [&](const SessionMemoryPlan::PersistentState& candidate) {
                    return candidate.state == ref.state;
                });
            if (state == result.persistent_states.end())
                throw std::logic_error(
                    "persistent state has no physical allocation");
            const BinaryBinding& executable_binding = find_binding(
                executable.program, BindingAccess::Internal,
                ref.binding_index);
            if (executable_binding.element_type
                    != state->binding.element_type
                || executable_binding.layout != state->binding.layout
                || executable_binding.shape != state->binding.shape
                || executable_binding.byte_size != state->binding.byte_size
                || executable_binding.slices != state->binding.slices
                || executable_binding.hemisphere_mask
                    != state->binding.hemisphere_mask)
                throw std::invalid_argument(
                    "persistent state bindings are physically incompatible");
            invocation_plan.states.push_back(
                {ref.binding_index, ref.state, state->binding});
        }
    }
    return result;
}

} // namespace ftlpu::software::runtime
