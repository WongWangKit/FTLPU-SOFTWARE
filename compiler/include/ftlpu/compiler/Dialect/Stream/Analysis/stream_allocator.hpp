#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace ftlpu::compiler::stream {

struct StreamBinding {
    int64_t stream_base;
    int64_t stream_count;
    int64_t register_id;
    target::StreamDirection direction;
    int64_t live_start;
    int64_t live_end;
};

class StreamAllocator {
public:
    explicit StreamAllocator(const target::LPUTargetModel& target) : target_(target) {}

    mlir::FailureOr<StreamBinding> allocate(target::StreamEndpoint source,
        target::StreamEndpoint destination, target::StreamDirection direction,
        int64_t mem_slice, int64_t live_start, int64_t live_end);

private:
    bool conflicts(target::StreamDirection direction, int64_t stream_base,
        int64_t stream_count,
        int64_t live_start, int64_t live_end) const;

    const target::LPUTargetModel& target_;
    llvm::SmallVector<StreamBinding, 16> bindings_;
};

} // namespace ftlpu::compiler::stream
