#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_allocator.hpp"

namespace ftlpu::compiler::stream {

mlir::FailureOr<StreamBinding> StreamAllocator::allocate(
    target::StreamEndpoint source,
    target::StreamEndpoint destination,
    target::StreamDirection direction,
    int64_t mem_slice, int64_t live_start, int64_t live_end)
{
    return allocate(source, destination, direction, mem_slice,
        live_start, live_end, std::nullopt);
}

mlir::FailureOr<StreamBinding> StreamAllocator::allocate(target::StreamEndpoint source,
    target::StreamEndpoint destination, target::StreamDirection direction,
    int64_t mem_slice, int64_t live_start, int64_t live_end,
    std::optional<int64_t> stream_count_override)
{
    if (live_start < 0 || live_end <= live_start) return mlir::failure();
    const auto register_id = target_.stream_register_id(source, destination, direction, mem_slice);
    const auto route_stream_count =
        target_.route_stream_count(source, destination, direction);
    const std::optional<int64_t> stream_count =
        stream_count_override ? stream_count_override
                              : route_stream_count;
    if (!register_id || !stream_count || *stream_count <= 0
        || *stream_count > target_.streams().streams_per_direction)
        return mlir::failure();

    for (int64_t stream_base = 0;
         stream_base + *stream_count <= target_.streams().streams_per_direction;
         ++stream_base) {
        if (conflicts(direction, stream_base, *stream_count, live_start, live_end)) continue;
        StreamBinding binding{
            stream_base, *stream_count, *register_id, direction, live_start, live_end};
        bindings_.push_back(binding);
        return binding;
    }
    return mlir::failure();
}

mlir::LogicalResult StreamAllocator::reserve(
    target::StreamDirection direction, int64_t stream_base,
    int64_t stream_count, int64_t live_start, int64_t live_end)
{
    if (stream_base < 0 || stream_count <= 0
        || stream_base + stream_count
            > target_.streams().streams_per_direction
        || live_start < 0 || live_end <= live_start
        || conflicts(direction, stream_base, stream_count,
            live_start, live_end))
        return mlir::failure();
    bindings_.push_back(StreamBinding {stream_base, stream_count, -1,
        direction, live_start, live_end});
    return mlir::success();
}

bool StreamAllocator::conflicts(target::StreamDirection direction, int64_t stream_base,
    int64_t stream_count,
    int64_t live_start, int64_t live_end) const
{
    for (const StreamBinding& binding : bindings_) {
        if (binding.direction != direction) continue;
        const bool streams_overlap = stream_base < binding.stream_base + binding.stream_count
            && stream_base + stream_count > binding.stream_base;
        const bool lifetimes_overlap = live_start < binding.live_end
            && live_end > binding.live_start;
        if (streams_overlap && lifetimes_overlap)
            return true;
    }
    return false;
}

} // namespace ftlpu::compiler::stream
