#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/stream_fabric_scheduler.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace ftlpu::compiler::schedule::detail {

int64_t get_slice(mlir::DictionaryAttr address);

llvm::SmallVector<int64_t> get_slices(mlir::DictionaryAttr placement);

mlir::DictionaryAttr subrange_placement(mlir::OpBuilder& builder,
    mlir::DictionaryAttr placement, int64_t row_offset, int64_t row_count);

mlir::DictionaryAttr weight_pass_placement(mlir::OpBuilder& builder,
    mlir::DictionaryAttr placement, int64_t pass);

std::string mem_read_resource(int64_t slice, int64_t bank = 0);
std::string mem_write_resource(int64_t slice, int64_t bank = 0);

int64_t value_ready_cycle(mlir::Value value);

void add_stream_windows(llvm::SmallVectorImpl<ResourceWindow>& windows,
    llvm::StringRef direction, int64_t base, int64_t count,
    int64_t offset, int64_t duration);

struct TimedStreamRoute {
    int64_t offset{0};
    StreamRouteWindow route;

    TimedStreamRoute(int64_t routeOffset, StreamRouteWindow routeWindow)
        : offset(routeOffset), route(std::move(routeWindow))
    {
    }
};

int64_t reserve_resources_and_streams(ResourceScheduler& resources,
    StreamFabricScheduler& streams, int64_t earliest_cycle,
    llvm::ArrayRef<ResourceWindow> resource_windows,
    llvm::ArrayRef<TimedStreamRoute> stream_routes);

} // namespace ftlpu::compiler::schedule::detail
