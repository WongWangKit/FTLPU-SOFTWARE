#pragma once

#include "ftlpu/system/icu.hpp"
#include <cstdint>

namespace ftlpu::software::runtime {

// Native and QCTL use the same rendezvous and physical release edge.
// Epoch supplies event_tag, releases, next_release, actual_ready_cycle,
// release_origin_cycle and released. One phase retires per physical edge.
template<class Epoch>
void release_queue_phase(Epoch& epoch, InstructionControlUnit& icu,
    std::int64_t now)
{
    if (epoch.released || !epoch.actual_ready_cycle) return;
    if (epoch.releases.empty()) {
        icu.broadcast_tagged_notification(epoch.event_tag);
        epoch.released = true;
        return;
    }
    if (!epoch.release_origin_cycle) {
        constexpr auto startupWindow = hw::kMxmBoundaryStreamRegisterColumn + 1;
        for (const auto& release : epoch.releases) {
            if (release.phase_offset > startupWindow && !release.activation_producer
                && release.location.kind != IcuLocationKind::MxmCompute)
                continue;
            if (!icu.blocked_on_tagged_event(release.location, epoch.event_tag)) return;
        }
    }
    const auto begin = epoch.next_release;
    const auto offset = epoch.releases.at(begin).phase_offset;
    auto end = begin;
    while (end < epoch.releases.size() && epoch.releases[end].phase_offset == offset) ++end;
    const auto expected = epoch.release_origin_cycle
        ? *epoch.release_origin_cycle + static_cast<std::int64_t>(offset)
        : *epoch.actual_ready_cycle;
    if (now < expected) return;
    for (auto i = begin; i < end; ++i)
        if (!icu.blocked_on_tagged_event(epoch.releases[i].location, epoch.event_tag)) return;
    if (!epoch.release_origin_cycle)
        epoch.release_origin_cycle = now - static_cast<std::int64_t>(offset);
    for (auto i = begin; i < end; ++i)
        icu.notify_tagged(epoch.releases[i].location, epoch.event_tag);
    epoch.next_release = end;
    epoch.released = end == epoch.releases.size();
}
} // namespace ftlpu::software::runtime
