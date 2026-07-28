#include "ftlpu/software/runtime/model_session.hpp"

#include <stdexcept>
#include <utility>

namespace ftlpu::software::runtime {
namespace {

const BinaryBinding& find_binding(const BinaryProgram& program,
    BindingAccess access, std::uint32_t index)
{
    for (const BinaryBinding& binding : program.bindings)
        if (binding.access == access && binding.index == index)
            return binding;
    throw std::logic_error(
        "session plan references a missing executable binding");
}

} // namespace

ModelSession::ModelSession(TspSliceSystem& system)
    : runtime_(system)
{
}

void ModelSession::load(ModelPackage package)
{
    validate_model_package(package);
    memory_plan_ = SessionMemoryPlanner::plan(package);
    package_ = std::move(package);
    values_.clear();
    device_values_.clear();
    stats_ = {};
    loaded_ = true;
}

void ModelSession::load_file(const std::filesystem::path& path)
{
    load(read_model_package(path));
}

const ModelValue* ModelSession::find_value_metadata(
    const std::string& name) const
{
    for (const auto& value : package_.values)
        if (value.name == name) return &value;
    return nullptr;
}

void ModelSession::set_input(
    std::string name, std::span<const std::uint8_t> data)
{
    if (!loaded_) throw std::logic_error("no FTLPU model package is loaded");
    const ModelValue* metadata = find_value_metadata(name);
    if (!metadata || !metadata->external_input)
        throw std::invalid_argument(
            "FTLPU model input is not declared as external");
    values_[std::move(name)] =
        std::vector<std::uint8_t>(data.begin(), data.end());
}

const std::vector<std::uint8_t>& ModelSession::resolve_value(
    const std::string& name) const
{
    if (const auto value = values_.find(name); value != values_.end())
        return value->second;
    for (const auto& tensor : package_.tensors)
        if (tensor.name == name) return tensor.data;
    throw std::out_of_range("FTLPU model value is not available: " + name);
}

void ModelSession::run_invocation(
    std::size_t index, std::size_t drain_cycles)
{
    if (!loaded_) throw std::logic_error("no FTLPU model package is loaded");
    if (index >= package_.invocations.size())
        throw std::out_of_range("FTLPU model invocation index is out of range");
    const auto& invocation = package_.invocations[index];
    const auto& executable = package_.executables.at(
        invocation.executable_index);
    const SessionInvocationPlan& invocation_plan =
        memory_plan_.invocations.at(index);

    runtime_.load(executable.program);
    for (const SessionInputPlan& input : invocation_plan.inputs) {
        if (input.transfer == SessionTransferKind::HostUpload) {
            runtime_.upload_input(
                input.binding_index, resolve_value(input.value));
            ++stats_.host_uploads;
            continue;
        }
        const auto source = device_values_.find(input.value);
        if (source == device_values_.end())
            throw std::logic_error(
                "device-resident model value is not available");
        if (source->second.target_abi != executable.program.target_abi)
            throw std::logic_error(
                "device-resident value cannot cross target ABIs");
        const BinaryBinding& destination = find_binding(executable.program,
            BindingAccess::Input, input.binding_index);
        if (input.transfer == SessionTransferKind::DeviceAlias) {
            if (!bindings_physically_alias(
                    source->second.binding, destination))
                throw std::logic_error(
                    "session device alias no longer matches executable ABI");
            ++stats_.device_aliases;
        } else {
            runtime_.copy_binding(source->second.binding, destination);
            ++stats_.device_copies;
            stats_.device_copy_bytes +=
                static_cast<std::size_t>(destination.byte_size);
        }
        if (input.release_after_transfer)
            device_values_.erase(source);
    }
    runtime_.run_cycles(executable.program.max_cycle + drain_cycles);
    for (const SessionOutputPlan& output : invocation_plan.outputs) {
        if (output.retain_on_device) {
            device_values_[output.value] = DeviceValue {
                find_binding(executable.program, BindingAccess::Output,
                    output.binding_index),
                executable.program.target_abi,
            };
        }
        if (output.download_to_host) {
            values_[output.value] =
                runtime_.download_output(output.binding_index);
            ++stats_.host_downloads;
        }
    }
}

void ModelSession::run(std::size_t drain_cycles)
{
    if (!loaded_) throw std::logic_error("no FTLPU model package is loaded");
    device_values_.clear();
    stats_ = {};
    for (std::size_t index = 0; index < package_.invocations.size(); ++index)
        run_invocation(index, drain_cycles);
}

const ModelPackage& ModelSession::package() const
{
    if (!loaded_) throw std::logic_error("no FTLPU model package is loaded");
    return package_;
}

const std::vector<std::uint8_t>& ModelSession::value(
    const std::string& name) const
{
    return resolve_value(name);
}

const SessionMemoryPlan& ModelSession::memory_plan() const
{
    if (!loaded_) throw std::logic_error("no FTLPU model package is loaded");
    return memory_plan_;
}

const ModelSessionStats& ModelSession::stats() const
{
    if (!loaded_) throw std::logic_error("no FTLPU model package is loaded");
    return stats_;
}

} // namespace ftlpu::software::runtime
