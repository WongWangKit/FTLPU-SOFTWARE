#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ftlpu::compiler::target::weight_pages {

// Physical rows are per slice; one group contains the load slices required by
// the native weight ABI. Callers own logical tensor traversal and group->slice
// mapping. This planner owns capacity, integral item pages and SRAM residency.
struct Geometry {
    int64_t bank_rows{};
    int64_t banks{};
    int64_t slice_groups{};
};

struct ItemLayout {
    int64_t rows_per_item{};
    int64_t item_alignment{1};
    int64_t headroom_rows{};
    int64_t base_row{};
};

struct ItemPage {
    int64_t item_begin{};
    int64_t item_count{};
    int64_t group_count{};
    int64_t rows_per_group{};
};

inline bool valid(const Geometry& g) {
    return g.bank_rows > 0 && g.banks > 0 && g.slice_groups > 0;
}

inline int64_t itemsPerGroup(const Geometry& g, const ItemLayout& layout) {
    if (!valid(g) || layout.rows_per_item <= 0 || layout.item_alignment <= 0
        || layout.headroom_rows < 0 || layout.base_row < 0
        || layout.base_row >= g.bank_rows
        || layout.headroom_rows >= g.bank_rows - layout.base_row)
        return 0;
    const auto raw = (g.bank_rows - layout.base_row - layout.headroom_rows)
        / layout.rows_per_item;
    return raw / layout.item_alignment * layout.item_alignment;
}

inline std::vector<ItemPage> splitItems(int64_t count, int64_t groups,
                                      const Geometry& g, const ItemLayout& layout) {
    const auto perGroup = itemsPerGroup(g, layout);
    if (count <= 0 || groups <= 0 || groups > g.slice_groups || perGroup <= 0
        || groups > std::numeric_limits<int64_t>::max() / perGroup)
        return {};
    const auto perPage = groups * perGroup;
    std::vector<ItemPage> result;
    for (int64_t begin = 0; begin < count;) {
        const auto n = std::min(perPage, count - begin);
        const auto largest = std::min(n, perGroup);
        const auto padded = ((largest - 1) / layout.item_alignment + 1)
            * layout.item_alignment;
        result.push_back({begin, n, (n - 1) / perGroup + 1,
                          padded * layout.rows_per_item});
        begin += n;
    }
    return result;
}

struct Slot {
    int64_t bank{};
    int64_t slice_group_begin{};
    int64_t slice_group_count{1};
    int64_t base_row{};
    int64_t rows{};
    bool operator==(const Slot&) const = default;
};

// All lifetimes are half-open compiler positions (graph indices or cycles).
// Do not mix units in one planner. Pinned reservations occupy their slot for
// every lifetime, e.g. constants initialized before graph execution.
struct Lifetime {
    int64_t begin{};
    int64_t end{};
};

inline bool contains(const Geometry& g, const Slot& s) {
    return valid(g) && s.bank >= 0 && s.bank < g.banks
        && s.slice_group_begin >= 0 && s.slice_group_count > 0
        && s.slice_group_count <= g.slice_groups
        && s.slice_group_begin <= g.slice_groups - s.slice_group_count
        && s.base_row >= 0 && s.rows > 0 && s.rows <= g.bank_rows
        && s.base_row <= g.bank_rows - s.rows;
}

inline bool overlaps(const Slot& a, const Slot& b) {
    return a.bank == b.bank
        && a.slice_group_begin < b.slice_group_begin + b.slice_group_count
        && b.slice_group_begin < a.slice_group_begin + a.slice_group_count
        && a.base_row < b.base_row + b.rows
        && b.base_row < a.base_row + a.rows;
}

inline bool overlaps(Lifetime a, Lifetime b) {
    return a.begin < b.end && b.begin < a.end;
}

class SlotPlanner {
public:
    explicit SlotPlanner(Geometry geometry) : geometry_(geometry) {
        if (!valid(geometry)) throw std::invalid_argument("invalid weight page geometry");
    }

    bool reserve(Slot slot, Lifetime lifetime, bool pinned = false) {
        if (!contains(geometry_, slot) || lifetime.begin < 0
            || lifetime.end <= lifetime.begin) return false;
        for (const auto& r : reservations_)
            if (overlaps(slot, r.slot)
                && (pinned || r.pinned || overlaps(lifetime, r.lifetime)))
                return false;
        reservations_.push_back({slot, lifetime, pinned});
        return true;
    }

    // Prefer disjoint residency in the preferred bank before occupying another
    // bank. A consumer/page boundary alone is not a reason to rotate banks.
    // The caller can request the other bank when overlap warrants ping-pong.
    std::optional<Slot> allocate(int64_t groups, int64_t rows, Lifetime lifetime,
                               int64_t preferred_bank = 0, int64_t min_row = 0,
                               int64_t row_alignment = 1) {
        if (groups <= 0 || groups > geometry_.slice_groups || rows <= 0
            || rows > geometry_.bank_rows || preferred_bank < 0
            || preferred_bank >= geometry_.banks || min_row < 0
            || min_row > geometry_.bank_rows - rows || row_alignment <= 0
            || lifetime.begin < 0 || lifetime.end <= lifetime.begin) return {};
        for (int64_t bi = 0; bi < geometry_.banks; ++bi) {
            const auto bank = (preferred_bank + bi) % geometry_.banks;
            for (int64_t group = 0; group <= geometry_.slice_groups - groups; ++group) {
                std::vector<int64_t> candidates{min_row};
                for (const auto& r : reservations_)
                    if (r.slot.bank == bank
                        && group < r.slot.slice_group_begin + r.slot.slice_group_count
                        && r.slot.slice_group_begin < group + groups
                        && (r.pinned || overlaps(lifetime, r.lifetime)))
                        candidates.push_back(std::max(min_row, r.slot.base_row + r.slot.rows));
                std::sort(candidates.begin(), candidates.end());
                for (auto row : candidates) {
                    const auto remainder = row % row_alignment;
                    if (remainder) {
                        const auto padding = row_alignment - remainder;
                        if (padding > geometry_.bank_rows - row) continue;
                        row += padding;
                    }
                    Slot slot{bank, group, groups, row, rows};
                    if (reserve(slot, lifetime)) return slot;
                }
            }
        }
        return {};
    }

    // Conservative release for reusing an explicitly requested physical slot.
    // A pinned slot is unavailable; callers must select a different region.
    std::optional<int64_t> earliestReuse(Slot slot, int64_t earliest) const {
        if (!contains(geometry_, slot) || earliest < 0) return {};
        for (const auto& r : reservations_) if (overlaps(slot, r.slot)) {
            if (r.pinned) return {};
            earliest = std::max(earliest, r.lifetime.end);
        }
        return earliest;
    }

private:
    struct Reservation { Slot slot; Lifetime lifetime; bool pinned; };
    Geometry geometry_;
    std::vector<Reservation> reservations_;
};

// Numbered one-group tiles retain distinct row slots. Full-bank packing is
// default; alternating banks is an explicit scheduling request, not a side
// effect of making another logical transfer page.
inline std::optional<Slot> residentSlot(int64_t index, int64_t rows,
                                       const Geometry& g, int64_t initial_bank = 0,
                                       bool alternate_banks = false) {
    if (!valid(g) || index < 0 || rows <= 0 || rows > g.bank_rows
        || initial_bank < 0 || initial_bank >= g.banks) return {};
    const auto rowslots = g.bank_rows / rows;
    if (g.slice_groups > std::numeric_limits<int64_t>::max() / rowslots) return {};
    const auto perBank = rowslots * g.slice_groups;
    const auto bankSlot = alternate_banks ? index % g.banks : (index / perBank) % g.banks;
    const auto local = alternate_banks ? (index / g.banks) % perBank : index % perBank;
    return Slot{(initial_bank + bankSlot) % g.banks, local / rowslots, 1,
                (local % rowslots) * rows, rows};
}

} // namespace ftlpu::compiler::target::weight_pages
