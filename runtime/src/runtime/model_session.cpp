#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ftlpu::software::runtime {
namespace {

bool is_16bit_float(BindingElementType type)
{
    return type == BindingElementType::F16
        || type == BindingElementType::BF16;
}

float decode_16bit_float(
    std::uint16_t bits, BindingElementType type)
{
    if (type == BindingElementType::BF16)
        return Bf16::from_bits(bits).to_float();
    if (type == BindingElementType::F16)
        return Fp16::from_bits(bits).to_float();
    throw std::logic_error("model value is not a 16-bit float");
}

std::uint16_t encode_16bit_float(
    float value, BindingElementType type)
{
    if (type == BindingElementType::BF16)
        return Bf16::from_float(value).bits();
    if (type == BindingElementType::F16)
        return Fp16::from_float(value).bits();
    throw std::logic_error("model value is not a 16-bit float");
}

const BinaryBinding& find_binding(const BinaryProgram& program,
    BindingAccess access, std::uint32_t index)
{
    for (const BinaryBinding& binding : program.bindings)
        if (binding.access == access && binding.index == index)
            return binding;
    throw std::logic_error(
        "session plan references a missing executable binding");
}

const ModelTensor& find_tensor(
    const ModelPackage& package, const std::string& name)
{
    for (const ModelTensor& tensor : package.tensors)
        if (tensor.name == name) return tensor;
    throw std::logic_error(
        "executable scale relocation requires a model tensor input");
}

const ModelBindingRef& find_input_ref(
    const ModelInvocation& invocation, std::uint32_t binding_index)
{
    for (const ModelBindingRef& input : invocation.inputs)
        if (input.binding_index == binding_index) return input;
    throw std::logic_error(
        "executable scale relocation references an unbound input");
}

std::size_t element_count(const std::vector<std::uint64_t>& shape)
{
    std::size_t result = 1;
    for (const std::uint64_t dimension : shape) {
        if (dimension > std::numeric_limits<std::size_t>::max() / result)
            throw std::overflow_error("model tensor shape is too large");
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

BinaryProgram parameterize_program(const ModelPackage& package,
    const ModelInvocation& invocation,
    const SessionInvocationPlan& invocation_plan,
    BinaryProgram program)
{
    const bool report_progress =
        std::getenv("FTLPU_SESSION_PROGRESS") != nullptr;
    for (const BinaryScaleRelocation& relocation :
         program.scale_relocations) {
        const ModelBindingRef& input =
            find_input_ref(invocation, relocation.binding_index);
        const ModelTensor& tensor = find_tensor(package, input.value);
        if (tensor.encoding == ModelTensorEncoding::Raw
            || relocation.scale_index >= tensor.scales.size())
            throw std::logic_error(
                "executable scale relocation requires quantized tensor metadata");
        auto queue = std::find_if(program.queues.begin(),
            program.queues.end(), [&](const QueueProgram& candidate) {
                return candidate.kind == relocation.queue_kind
                    && candidate.index == relocation.queue_index;
            });
        if (queue == program.queues.end()
            || relocation.command_index >= queue->commands.size())
            throw std::logic_error(
                "executable scale relocation references a missing command");
        QueueCommand& command =
            queue->commands[relocation.command_index];
        if (command.instruction_kind != InstructionKind::Vxm
            || command.word_count != 4)
            throw std::logic_error(
                "scale relocation target is not a VXM instruction");
        isa::EncodedVxmInstruction encoded {command.words};
        VxmLaneAluInstruction instruction =
            isa::decode_vxm_instruction(encoded);
        VxmLaneOperand& operand =
            relocation.operand == VxmImmediateOperand::Lhs
            ? instruction.lhs : instruction.rhs;
        if (operand.kind != VxmLaneOperandKind::Immediate)
            throw std::logic_error(
                "scale relocation target is not an immediate operand");
        operand = VxmLaneOperand::Imm(
            tensor.scales[relocation.scale_index]);
        command.words = isa::encode_vxm_instruction(instruction).words;
    }
    std::unordered_set<std::uint64_t> relocated_bindings;
    std::unordered_map<std::uint64_t, std::size_t> relocation_counts;
    std::unordered_map<std::uint64_t,
        std::pair<std::int64_t, std::int64_t>> relocation_address_ranges;
    const auto relocation_key =
        [](BindingAccess access, std::uint32_t index) {
            return (static_cast<std::uint64_t>(access) << 32) | index;
        };
    for (const BinaryAddressRelocation& relocation :
         program.address_relocations) {
        const auto input = std::find_if(invocation_plan.inputs.begin(),
            invocation_plan.inputs.end(),
            [&](const SessionInputPlan& candidate) {
                return relocation.binding_access == BindingAccess::Input
                    && candidate.binding_index
                        == relocation.binding_index;
            });
        const auto state = std::find_if(invocation_plan.states.begin(),
            invocation_plan.states.end(),
            [&](const SessionStatePlan& candidate) {
                return relocation.binding_access == BindingAccess::Internal
                    && candidate.binding_index
                        == relocation.binding_index;
            });
        const bool resident_input =
            input != invocation_plan.inputs.end()
            && input->transfer == SessionTransferKind::Resident;
        if (!resident_input && state == invocation_plan.states.end())
            continue;
        const BindingAccess access = relocation.binding_access;
        const std::uint64_t key =
            relocation_key(access, relocation.binding_index);
        relocated_bindings.insert(key);
        ++relocation_counts[key];
        const BinaryBinding& original_binding =
            find_binding(program, access, relocation.binding_index);
        const BinaryBinding& resolved_binding = resident_input
            ? input->resolved_binding : state->resolved_binding;
        const std::int64_t delta =
            resolved_binding.base_row
            - original_binding.base_row;
        auto queue = std::find_if(program.queues.begin(),
            program.queues.end(), [&](const QueueProgram& candidate) {
                return candidate.kind == relocation.queue_kind
                    && candidate.index == relocation.queue_index;
            });
        if (queue == program.queues.end()
            || relocation.command_index >= queue->commands.size())
            throw std::logic_error(
                "address relocation references a missing command");
        QueueCommand& command = queue->commands[relocation.command_index];
        if (command.instruction_kind != InstructionKind::Mem
            || command.word_count == 0 || command.word_count > 2)
            throw std::logic_error(
                "address relocation target is not a MEM instruction");
        const isa::EncodedMemInstruction encoded =
            static_cast<isa::EncodedMemInstruction>(command.words[0])
            | (static_cast<isa::EncodedMemInstruction>(command.words[1])
                << 32);
        MemInstruction instruction = isa::decode_mem_instruction(encoded);
        const std::size_t sourceAddress = relocation.write_port
            ? instruction.write_address : instruction.address;
        auto [range, inserted] =
            relocation_address_ranges.try_emplace(key,
                static_cast<std::int64_t>(sourceAddress),
                static_cast<std::int64_t>(sourceAddress));
        if (!inserted) {
            range->second.first = std::min(
                range->second.first,
                static_cast<std::int64_t>(sourceAddress));
            range->second.second = std::max(
                range->second.second,
                static_cast<std::int64_t>(sourceAddress));
        }
        const std::int64_t relocated =
            static_cast<std::int64_t>(sourceAddress) + delta;
        if (relocated < 0
            || relocated >= program.memory_rows_per_slice)
            throw std::logic_error(
                "resident address relocation exceeds physical MEM: binding="
                + std::to_string(relocation.binding_index)
                + " original_base="
                + std::to_string(original_binding.base_row)
                + " resolved_base="
                + std::to_string(resolved_binding.base_row)
                + " command_address="
                + std::to_string(sourceAddress)
                + " relocated=" + std::to_string(relocated));
        if (relocation.write_port)
            instruction.write_address =
                static_cast<std::size_t>(relocated);
        else
            instruction.address = static_cast<std::size_t>(relocated);
        const isa::EncodedMemInstruction patched =
            isa::encode_mem_instruction(instruction);
        command.words[0] = static_cast<std::uint32_t>(patched);
        command.words[1] = static_cast<std::uint32_t>(patched >> 32);
        command.word_count =
            static_cast<std::uint16_t>((patched >> 32) == 0 ? 1 : 2);
    }
    for (const SessionInputPlan& input : invocation_plan.inputs) {
        if (input.transfer != SessionTransferKind::Resident) continue;
        const BinaryBinding& original_binding = find_binding(
            program, BindingAccess::Input, input.binding_index);
        const std::uint64_t key =
            relocation_key(BindingAccess::Input, input.binding_index);
        if (report_progress) {
            const auto range = relocation_address_ranges.find(key);
            std::clog << "FTLPU resident relocation: binding="
                      << input.binding_index << " name="
                      << original_binding.name << " base="
                      << original_binding.base_row << "->"
                      << input.resolved_binding.base_row << " commands="
                      << relocation_counts[key];
            if (range != relocation_address_ranges.end())
                std::clog << " address_range=["
                          << range->second.first << ','
                          << range->second.second << ']';
            std::clog << std::endl;
        }
        if (input.resolved_binding.base_row != original_binding.base_row
            && !relocated_bindings.contains(key))
            throw std::logic_error(
                "resident binding moved without MEM address relocation: "
                "binding=" + std::to_string(input.binding_index)
                + " original_base="
                + std::to_string(original_binding.base_row)
                + " resolved_base="
                + std::to_string(input.resolved_binding.base_row));
        auto binding = std::find_if(program.bindings.begin(),
            program.bindings.end(), [&](const BinaryBinding& candidate) {
                return candidate.access == BindingAccess::Input
                    && candidate.index == input.binding_index;
            });
        if (binding == program.bindings.end())
            throw std::logic_error(
                "resident session input references a missing binding");
        *binding = input.resolved_binding;
    }
    for (const SessionStatePlan& state : invocation_plan.states) {
        const BinaryBinding& original_binding = find_binding(
            program, BindingAccess::Internal, state.binding_index);
        if (state.resolved_binding.base_row != original_binding.base_row
            && !relocated_bindings.contains(relocation_key(
                BindingAccess::Internal, state.binding_index)))
            throw std::logic_error(
                "persistent state moved without MEM address relocation: "
                "binding=" + std::to_string(state.binding_index));
        auto binding = std::find_if(program.bindings.begin(),
            program.bindings.end(), [&](const BinaryBinding& candidate) {
                return candidate.access == BindingAccess::Internal
                    && candidate.index == state.binding_index;
            });
        if (binding == program.bindings.end())
            throw std::logic_error(
                "persistent session state references a missing binding");
        *binding = state.resolved_binding;
    }
    return program;
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
    const bool report_progress =
        std::getenv("FTLPU_SESSION_PROGRESS") != nullptr;
    if (report_progress)
        std::clog << "FTLPU session resident uploads: "
                  << memory_plan_.resident_tensors.size() << std::endl;
    for (const auto& resident : memory_plan_.resident_tensors) {
        const std::size_t upload_index = stats_.resident_uploads + 1;
        if (report_progress
            && (memory_plan_.resident_tensors.size() <= 16
                || upload_index == 1
                || upload_index == memory_plan_.resident_tensors.size()
                || upload_index % 25 == 0))
            std::clog << "FTLPU resident upload "
                      << upload_index << '/'
                      << memory_plan_.resident_tensors.size() << ": "
                      << resident.value << " base_row="
                      << resident.binding.base_row << " bytes="
                      << resident.binding.byte_size << std::endl;
        runtime_.upload_binding(
            resident.binding, resolve_value(resident.value));
        ++stats_.resident_uploads;
        stats_.resident_upload_bytes +=
            static_cast<std::size_t>(resident.binding.byte_size);
    }
    for (const auto& state : memory_plan_.persistent_states) {
        const std::vector<std::uint8_t> zero(
            static_cast<std::size_t>(state.binding.byte_size), 0);
        runtime_.upload_binding(state.binding, zero);
        ++stats_.state_initializations;
        stats_.state_initialization_bytes += zero.size();
    }
    loaded_ = true;
}

void ModelSession::load_file(const std::filesystem::path& path)
{
    load(read_model_package(path, ModelPackageLoadMode::LazyExecutables));
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
    const BinaryProgram program = parameterize_program(
        package_, invocation, invocation_plan,
        materialize_model_executable(executable));

    runtime_.load(program);
    for (const SessionInputPlan& input : invocation_plan.inputs) {
        if (input.transfer == SessionTransferKind::Resident)
            continue;
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
        if (source->second.target_abi != program.target_abi)
            throw std::logic_error(
                "device-resident value cannot cross target ABIs");
        const BinaryBinding& destination = find_binding(program,
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
    runtime_.run_cycles(program.max_cycle + drain_cycles);
    const bool validate_fp16 =
        std::getenv("FTLPU_SESSION_VALIDATE_FP16") != nullptr;
    for (const SessionOutputPlan& output : invocation_plan.outputs) {
        if (validate_fp16) {
            const BinaryBinding& binding = find_binding(program,
                BindingAccess::Output, output.binding_index);
            if (is_16bit_float(binding.element_type)) {
                const std::vector<std::uint8_t> data =
                    runtime_.download_output(output.binding_index);
                for (std::size_t element = 0;
                     element < data.size() / sizeof(std::uint16_t);
                     ++element) {
                    std::uint16_t bits = 0;
                    std::memcpy(&bits,
                        data.data()
                            + element * sizeof(std::uint16_t),
                        sizeof(bits));
                    if (!std::isfinite(decode_16bit_float(
                            bits, binding.element_type)))
                        throw std::logic_error(
                            "model invocation produced non-finite "
                            "16-bit float: "
                            + invocation.name + " element="
                            + std::to_string(element));
                }
            }
        }
        if (output.retain_on_device) {
            device_values_[output.value] = DeviceValue {
                find_binding(program, BindingAccess::Output,
                    output.binding_index),
                program.target_abi,
            };
        }
        if (output.download_to_host) {
            values_[output.value] =
                runtime_.download_output(output.binding_index);
            ++stats_.host_downloads;
        }
    }
}

void ModelSession::run_embedding_lookups()
{
    for (const ModelEmbeddingLookup& lookup : package_.embedding_lookups) {
        const ModelValue* ids_metadata =
            find_value_metadata(lookup.token_ids);
        const ModelValue* output_metadata =
            find_value_metadata(lookup.output);
        const ModelTensor& table = find_tensor(package_, lookup.table);
        if (!ids_metadata || ids_metadata->element_type != BindingElementType::I32
            || ids_metadata->shape.size() != 1 || !output_metadata
            || !is_16bit_float(output_metadata->element_type)
            || output_metadata->shape.size() != 2
            || table.element_type != output_metadata->element_type
            || table.encoding != ModelTensorEncoding::Raw
            || table.shape.size() != 2
            || output_metadata->shape[0] != ids_metadata->shape[0]
            || output_metadata->shape[1] != table.shape[1])
            throw std::logic_error(
                "embedding lookup requires i32 ids and a matching raw "
                "rank-2 16-bit float table");
        const auto& ids = resolve_value(lookup.token_ids);
        if (ids.size() != element_count(ids_metadata->shape) * sizeof(std::int32_t)
            || table.data.size() != element_count(table.shape) * sizeof(std::uint16_t))
            throw std::logic_error("embedding lookup tensor byte size mismatch");

        const std::size_t row_bytes =
            static_cast<std::size_t>(table.shape[1]) * sizeof(std::uint16_t);
        std::vector<std::uint8_t> output(
            static_cast<std::size_t>(ids_metadata->shape[0]) * row_bytes);
        for (std::size_t row = 0; row < ids_metadata->shape[0]; ++row) {
            std::int32_t token = 0;
            std::memcpy(&token, ids.data() + row * sizeof(token), sizeof(token));
            if (token < 0 || static_cast<std::uint64_t>(token) >= table.shape[0])
                throw std::out_of_range("embedding token id is out of range");
            std::memcpy(output.data() + row * row_bytes,
                table.data.data() + static_cast<std::size_t>(token) * row_bytes,
                row_bytes);
        }
        values_[lookup.output] = std::move(output);
        ++stats_.host_operations;
    }
}

void ModelSession::run_host_lm_heads()
{
    for (const ModelHostLmHead& lm_head : package_.host_lm_heads) {
        const ModelValue* hidden_metadata =
            find_value_metadata(lm_head.hidden);
        const ModelValue* output_metadata =
            find_value_metadata(lm_head.output);
        const ModelTensor& weight =
            find_tensor(package_, lm_head.weight);
        if (!hidden_metadata
            || !is_16bit_float(hidden_metadata->element_type)
            || hidden_metadata->shape.size() != 2 || !output_metadata
            || output_metadata->shape.size() != 2
            || (output_metadata->element_type
                    != hidden_metadata->element_type
                && output_metadata->element_type != BindingElementType::F32)
            || weight.element_type != hidden_metadata->element_type
            || weight.encoding != ModelTensorEncoding::Raw
            || weight.shape.size() != 2
            || hidden_metadata->shape[1] != weight.shape[1]
            || output_metadata->shape[0]
                != (lm_head.last_token_only
                        ? 1 : hidden_metadata->shape[0])
            || output_metadata->shape[1] != weight.shape[0])
            throw std::logic_error(
                "host LM head requires matching [tokens, hidden] and "
                "[vocab, hidden] 16-bit float tensors");

        const auto& hidden = resolve_value(lm_head.hidden);
        if (hidden.size()
                != element_count(hidden_metadata->shape)
                    * sizeof(std::uint16_t)
            || weight.data.size()
                != element_count(weight.shape)
                    * sizeof(std::uint16_t))
            throw std::logic_error("host LM head tensor byte size mismatch");

        const std::size_t hidden_size =
            static_cast<std::size_t>(weight.shape[1]);
        const std::size_t vocabulary =
            static_cast<std::size_t>(weight.shape[0]);
        const std::size_t output_rows =
            static_cast<std::size_t>(output_metadata->shape[0]);
        const std::size_t first_hidden_row = lm_head.last_token_only
            ? static_cast<std::size_t>(hidden_metadata->shape[0] - 1)
            : 0;
        const std::size_t output_element_bytes =
            output_metadata->element_type == BindingElementType::F32
            ? sizeof(float) : sizeof(std::uint16_t);
        std::vector<std::uint8_t> output(
            output_rows * vocabulary * output_element_bytes);

        const auto read_16bit = [&](const std::uint8_t* source) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, source, sizeof(bits));
            return decode_16bit_float(
                bits, hidden_metadata->element_type);
        };
        for (std::size_t row = 0; row < output_rows; ++row) {
            const std::size_t hidden_row = first_hidden_row + row;
            for (std::size_t token = 0; token < vocabulary; ++token) {
                float accumulator = 0.0f;
                for (std::size_t column = 0;
                     column < hidden_size; ++column) {
                    const std::size_t hidden_index =
                        hidden_row * hidden_size + column;
                    const std::size_t weight_index =
                        token * hidden_size + column;
                    accumulator +=
                        read_16bit(hidden.data()
                            + hidden_index * sizeof(std::uint16_t))
                        * read_16bit(weight.data.data()
                            + weight_index * sizeof(std::uint16_t));
                }
                const std::size_t output_index =
                    row * vocabulary + token;
                if (output_metadata->element_type
                    == BindingElementType::F32) {
                    std::memcpy(output.data()
                            + output_index * sizeof(float),
                        &accumulator, sizeof(accumulator));
                } else {
                    const std::uint16_t bits = encode_16bit_float(
                        accumulator, output_metadata->element_type);
                    std::memcpy(output.data()
                            + output_index * sizeof(bits),
                        &bits, sizeof(bits));
                }
            }
        }
        values_[lm_head.output] = std::move(output);
        ++stats_.host_operations;
    }
}

void ModelSession::run(std::size_t drain_cycles)
{
    if (!loaded_) throw std::logic_error("no FTLPU model package is loaded");
    device_values_.clear();
    const std::size_t resident_uploads = stats_.resident_uploads;
    const std::size_t resident_upload_bytes =
        stats_.resident_upload_bytes;
    stats_ = {};
    stats_.resident_uploads = resident_uploads;
    stats_.resident_upload_bytes = resident_upload_bytes;
    run_embedding_lookups();
    const bool report_progress =
        std::getenv("FTLPU_SESSION_PROGRESS") != nullptr;
    for (std::size_t index = 0; index < package_.invocations.size(); ++index) {
        if (report_progress)
            std::clog << "FTLPU session invocation "
                      << (index + 1) << '/'
                      << package_.invocations.size() << ": "
                      << package_.invocations[index].name << std::endl;
        run_invocation(index, drain_cycles);
    }
    run_host_lm_heads();
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
