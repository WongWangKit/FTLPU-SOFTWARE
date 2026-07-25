#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ftlpu::compiler::stream {

struct RouteLifetime {
    std::string phase;
    std::string role;
    target::StreamEndpoint source;
    target::StreamEndpoint destination;
    target::StreamDirection direction;
    std::string buffer;
    int64_t producer_offset;
    int64_t consumer_offset;
};

class StreamRoutePlan {
public:
    void add(llvm::StringRef phase, llvm::StringRef role,
        target::StreamEndpoint source, target::StreamEndpoint destination,
        target::StreamDirection direction, llvm::StringRef buffer,
        int64_t producerOffset, int64_t consumerOffset);

    llvm::ArrayRef<RouteLifetime> routes() const { return routes_; }
    int64_t duration() const { return duration_; }
    bool valid() const;

private:
    std::vector<RouteLifetime> routes_;
    int64_t duration_ = 0;
};

StreamRoutePlan plan_attention_routes();

} // namespace ftlpu::compiler::stream
