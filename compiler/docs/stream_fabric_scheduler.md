# Stream Fabric Scheduler

`StreamFabricScheduler` is the cycle-accurate ownership model for the LPU
stream-register fabric. It is used alongside `ResourceScheduler`: the latter
reserves functional units and MEM ports, while the former reserves transport
cells.

## Reservation model

A reservation is indexed by:

```
(cycle, SR column, east/west direction, stream id)
```

One vector beat advances by one SR column per cycle. A `StreamRouteWindow`
describes its source and destination columns, contiguous stream range, beat
count and interval, token identity, and destination behavior.

- Different tokens cannot occupy the same cell in the same cycle.
- East and west register banks are independent.
- The same token may share cells for hardware multicast.
- `Tap` preserves the token for a downstream consumer.
- `Consume` prevents the same token from continuing beyond that endpoint.

`reserve_resources_and_streams` searches for the first cycle at which both FU
resources and all routes are legal, then commits both sets atomically.

## Target timing boundary

SR traversal and endpoint pipelines are separate quantities. The fabric
scheduler models one-column-per-cycle traversal. MEM injection, MXM/VXM/SXM
capture, result launch, and MEM write latency come from
`LPUTargetModel::transport_latency`. Emitters must place route windows at the
corresponding endpoint cycle; counting inclusive route columns is not a valid
replacement for target latency.

The current schedulable unit is a complete vector beat. Tile/lane skew remains
inside the CModel and target latency tables until Schedule IR exposes that
detail explicitly.

## Current integration

- Generic Matmul and legacy SwiGLU reserve FU and stream resources together.
- Block8 FFN hidden replication reserves west and east routes and uses the
  current CModel passive VXM bridge latency.
- Stream source/destination columns and transport latency are target-model
  properties, so stream count, MEM grouping, and topology stay configurable.

The accompanying unit test covers collision shifting, multicast, tap versus
consume, independent directions, route latency, and CModel topology mapping.
