#include "ftlpu/compiler/Dialect/Schedule/Analysis/stream_fabric_scheduler.hpp"

#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main()
{
    using namespace ftlpu::compiler;
    using namespace ftlpu::compiler::schedule;

    StreamFabricScheduler fabric(16, 32);
    StreamRouteWindow mxmToMem {
        target::StreamDirection::West, 15, 8,
        4, 2, 3, 1, 100, StreamConsumerMode::Consume};
    require(fabric.reserve_at(10, mxmToMem),
        "valid MXM-to-MEM route was rejected");
    require(fabric.is_reserved(
                10, 15, target::StreamDirection::West, 4)
            && fabric.is_reserved(
                17, 8, target::StreamDirection::West, 4),
        "route did not occupy each SR hop");
    require(StreamFabricScheduler::path_columns(15, 8).size() - 1 == 7,
        "path latency must count hops, not inclusive columns");

    auto collision = mxmToMem;
    collision.token_id = 200;
    require(!fabric.reserve_at(10, collision),
        "different token was allowed to overwrite an SR cell");
    require(fabric.reserve(10, collision) == 13,
        "route was not shifted to the first collision-free cycle");

    StreamFabricScheduler multicast(16, 32);
    auto firstConsumer = mxmToMem;
    firstConsumer.destination_column = 10;
    firstConsumer.beat_count = 1;
    firstConsumer.token_id = 300;
    require(multicast.reserve_at(0, firstConsumer),
        "first multicast consumer failed");
    auto secondConsumer = firstConsumer;
    require(multicast.reserve_at(0, secondConsumer),
        "same token could not be broadcast to a second consumer");
    auto downstream = firstConsumer;
    downstream.destination_column = 8;
    require(!multicast.reserve_at(0, downstream),
        "consumed token incorrectly continued downstream");

    StreamFabricScheduler tapped(16, 32);
    firstConsumer.consumer_mode = StreamConsumerMode::Tap;
    downstream.consumer_mode = StreamConsumerMode::Consume;
    require(tapped.reserve_at(0, firstConsumer), "tap route failed");
    require(tapped.reserve_at(0, downstream),
        "tap did not preserve downstream forwarding");

    StreamFabricScheduler tapThenConsume(16, 32);
    require(tapThenConsume.reserve_at(0, firstConsumer),
        "combined tap/consume setup rejected its tap");
    auto consumingPeer = firstConsumer;
    consumingPeer.consumer_mode = StreamConsumerMode::Consume;
    require(tapThenConsume.reserve_at(0, consumingPeer),
        "combined tap/consume setup rejected its consumer");
    require(!tapThenConsume.reserve_at(0, downstream),
        "a tap incorrectly overrode consumption at the same SR cell");

    StreamFabricScheduler oppositeDirections(16, 32);
    auto east = StreamRouteWindow {target::StreamDirection::East,
        2, 5, 7, 1, 1, 1, 400, StreamConsumerMode::Consume};
    auto west = StreamRouteWindow {target::StreamDirection::West,
        5, 2, 7, 1, 1, 1, 500, StreamConsumerMode::Consume};
    require(oppositeDirections.reserve_at(0, east)
            && oppositeDirections.reserve_at(0, west),
        "east and west register banks were incorrectly aliased");

    target::LPUTargetModel target;
    require(target.stream_source_column(target::StreamEndpoint::Mem,
                target::StreamDirection::East, 0) == 1
            && target.stream_destination_column(
                target::StreamEndpoint::MxmActivation,
                target::StreamDirection::East, 0) == 15,
        "MEM-to-MXM endpoint columns do not match CModel topology");
    require(target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight,
                target::StreamDirection::East, 20) == 10
            && target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight,
                target::StreamDirection::East, 24) == 9,
        "MEM-to-MXM latency must be physical hop distance plus consume cycle");
    require(target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, 0) == 1
            && target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, 8) == 3,
        "MEM-to-VXM latency must be physical hop distance plus consume cycle");
    require(target.stream_source_column(target::StreamEndpoint::MxmResult,
                target::StreamDirection::West, 0) == 15
            && target.stream_destination_column(
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, 0) == 0,
        "MXM-to-VXM endpoint columns do not match CModel topology");
    require(target.stream_destination_column(target::StreamEndpoint::Mem,
                target::StreamDirection::West, 51) == 13,
        "west MEM input boundary does not match its slice group");
}
