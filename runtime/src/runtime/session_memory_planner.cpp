#include "ftlpu/software/runtime/session_memory_planner.hpp"

#include <stdexcept>
#include <unordered_map>

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
    SessionMemoryPlan result;
    result.invocations.resize(package.invocations.size());

    struct Producer {
        std::size_t invocation{0};
        std::uint32_t binding_index{0};
    };
    std::unordered_map<std::string, Producer> producers;
    std::unordered_map<std::string, std::vector<std::size_t>> consumers;

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
        if (consumer->second.size() != 1)
            throw std::invalid_argument(
                "device-resident model values currently require one consumer");
        const std::size_t last_consumer = consumer->second.back();
        if (last_consumer <= producer.invocation)
            throw std::invalid_argument(
                "model value must be consumed after it is produced");
        result.lifetimes.push_back(
            {name, producer.invocation, last_consumer});
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
                    && (!value || !value->external_input))
                    throw std::invalid_argument(
                        "model invocation input has no available producer");
                input_plan.transfer = SessionTransferKind::HostUpload;
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
                input_plan.release_after_transfer = true;
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
    }
    return result;
}

} // namespace ftlpu::software::runtime
