#pragma once

#include "ftlpu/software/runtime/model_package.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {

enum class SessionTransferKind {
    HostUpload,
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
};

struct SessionOutputPlan {
    std::uint32_t binding_index{0};
    std::string value{};
    bool retain_on_device{false};
    bool download_to_host{false};
};

struct SessionInvocationPlan {
    std::vector<SessionInputPlan> inputs{};
    std::vector<SessionOutputPlan> outputs{};
};

struct SessionMemoryPlan {
    std::vector<SessionValueLifetime> lifetimes{};
    std::vector<SessionInvocationPlan> invocations{};
};

class SessionMemoryPlanner {
public:
    static SessionMemoryPlan plan(const ModelPackage& package);
};

bool bindings_physically_alias(
    const BinaryBinding& source, const BinaryBinding& destination);

} // namespace ftlpu::software::runtime
