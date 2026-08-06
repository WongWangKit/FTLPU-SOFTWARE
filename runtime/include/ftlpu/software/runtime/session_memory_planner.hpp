#pragma once

#include "ftlpu/software/runtime/model_package.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {

enum class SessionTransferKind {
    HostUpload,
    Resident,
    DeviceAlias,
    DeviceCopy,
};

struct SessionValueLifetime {
    std::string value{};
    std::size_t producer{0};
    std::size_t last_consumer{0};
};

struct SessionInputPlan {
    std::uint32_t binding_index{0};
    std::string value{};
    SessionTransferKind transfer{SessionTransferKind::HostUpload};
    std::size_t producer{0};
    bool release_after_transfer{false};
    BinaryBinding resolved_binding{};
};

struct SessionOutputPlan {
    std::uint32_t binding_index{0};
    std::string value{};
    bool retain_on_device{false};
    bool download_to_host{false};
};

struct SessionStatePlan {
    std::uint32_t binding_index{0};
    std::string state{};
    BinaryBinding resolved_binding{};
};

struct SessionInvocationPlan {
    std::vector<SessionInputPlan> inputs{};
    std::vector<SessionOutputPlan> outputs{};
    std::vector<SessionStatePlan> states{};
};

struct SessionMemoryPlan {
    std::vector<SessionValueLifetime> lifetimes{};
    std::vector<SessionInvocationPlan> invocations{};
    struct ResidentTensor {
        std::string value{};
        BinaryBinding binding{};
    };
    std::vector<ResidentTensor> resident_tensors{};
    struct PersistentState {
        std::string state{};
        BinaryBinding binding{};
    };
    std::vector<PersistentState> persistent_states{};
};

class SessionMemoryPlanner {
public:
    static SessionMemoryPlan plan(const ModelPackage& package);
};

bool bindings_physically_alias(
    const BinaryBinding& source, const BinaryBinding& destination);

} // namespace ftlpu::software::runtime
