#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_route_plan.hpp"
#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_allocator.hpp"

#include <algorithm>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main()
{
    using namespace ftlpu::compiler::stream;
    const StreamRoutePlan plan = plan_attention_routes();
    require(plan.valid(), "attention route plan is invalid");
    require(plan.routes().size() == 24,
        "attention route plan lost a physical transfer");
    require(plan.duration() == 54,
        "attention route lifetime duration changed");
    require(plan.routes().front().role == "query_weight",
        "attention route plan has the wrong first transfer");
    require(plan.routes().back().role == "result",
        "attention route plan has the wrong final transfer");
    const auto probability = std::find_if(plan.routes().begin(),
        plan.routes().end(), [](const RouteLifetime& route) {
            return route.role == "probability_activation";
        });
    require(probability != plan.routes().end()
            && probability->buffer == "probability_diagonal",
        "PV route does not consume the SXM-transposed probability layout");

    StreamRoutePlan invalid;
    invalid.add("phase", "role",
        ftlpu::compiler::target::StreamEndpoint::Mem,
        ftlpu::compiler::target::StreamEndpoint::MxmWeight,
        ftlpu::compiler::target::StreamDirection::East, "buffer", 2, 2);
    require(!invalid.valid(), "zero-length route lifetime was accepted");

    ftlpu::compiler::target::LPUTargetModel target;
    StreamAllocator allocator(target);
    require(mlir::succeeded(allocator.reserve(
        ftlpu::compiler::target::StreamDirection::East,
        0, 24, 0, 1)), "failed to reserve a planned stream range");
    auto output = allocator.allocate(
        ftlpu::compiler::target::StreamEndpoint::VxmResult,
        ftlpu::compiler::target::StreamEndpoint::Mem,
        ftlpu::compiler::target::StreamDirection::East,
        0, 0, 1, 2);
    require(mlir::succeeded(output),
        "failed to allocate around a planned stream range");
    require(output->stream_base == 24,
        "allocator ignored a planned stream reservation");
}
