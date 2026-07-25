#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_route_plan.hpp"

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

    StreamRoutePlan invalid;
    invalid.add("phase", "role",
        ftlpu::compiler::target::StreamEndpoint::Mem,
        ftlpu::compiler::target::StreamEndpoint::MxmWeight,
        ftlpu::compiler::target::StreamDirection::East, "buffer", 2, 2);
    require(!invalid.valid(), "zero-length route lifetime was accepted");
}
