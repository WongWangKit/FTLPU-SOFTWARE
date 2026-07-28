#pragma once

#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/model_package.hpp"
#include "ftlpu/software/runtime/session_memory_planner.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ftlpu::software::runtime {

struct ModelSessionStats {
    std::size_t host_uploads{0};
    std::size_t host_downloads{0};
    std::size_t device_aliases{0};
    std::size_t device_copies{0};
    std::size_t device_copy_bytes{0};
};

class ModelSession {
public:
    explicit ModelSession(TspSliceSystem& system);

    void load(ModelPackage package);
    void load_file(const std::filesystem::path& path);
    void set_input(std::string name, std::span<const std::uint8_t> data);
    void run(std::size_t drain_cycles = 64);
    void run_invocation(std::size_t index, std::size_t drain_cycles = 64);

    const ModelPackage& package() const;
    const std::vector<std::uint8_t>& value(const std::string& name) const;
    const SessionMemoryPlan& memory_plan() const;
    const ModelSessionStats& stats() const;

private:
    struct DeviceValue {
        BinaryBinding binding{};
        std::uint64_t target_abi{0};
    };

    const std::vector<std::uint8_t>& resolve_value(const std::string& name) const;
    const ModelValue* find_value_metadata(const std::string& name) const;

    CModelRuntime runtime_;
    ModelPackage package_{};
    SessionMemoryPlan memory_plan_{};
    ModelSessionStats stats_{};
    bool loaded_{false};
    std::unordered_map<std::string, std::vector<std::uint8_t>> values_{};
    std::unordered_map<std::string, DeviceValue> device_values_{};
};

} // namespace ftlpu::software::runtime
