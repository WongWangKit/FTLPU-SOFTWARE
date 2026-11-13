#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace ftlpu::compiler::analysis {

// A logical matrix view over an existing planar allocation. Offsets are
// physical MEM words, not elements or scheduled FU instructions. Byte planes
// and hemisphere ownership belong to the allocation, not to this view.
struct PlanarReadPlan {
  int64_t rows = 0;
  int64_t columns = 0; // 32-lane words, or one scalar-broadcast column
  int64_t offset = 0;
  int64_t rowChunk = 1;
  int64_t rowStride = 0;
  int64_t groupStride = 0;
  int64_t columnStride = 0;

  int64_t address(int64_t row, int64_t column) const {
    return offset + row / rowChunk * groupStride +
        row % rowChunk * rowStride + column * columnStride;
  }
};

// Prove that a logical view fits a single closed-form READ3D domain.
// The input is a layout map in column-major word order, before scheduling.
// Nonseparable, lane-changing or overly fragmented views remain explicit
// conversions; no hardware instruction trace is generated or compressed.
inline std::optional<PlanarReadPlan> planPlanarRead(
    int64_t rows, int64_t columns, const std::vector<int64_t>& offsets) {
  if (rows <= 0 || columns <= 0 ||
      int64_t(offsets.size()) != rows * columns) return std::nullopt;
  const int64_t columnStride = columns > 1 ? offsets[rows] - offsets[0] : 0;
  for (int64_t chunk = rows; chunk > 0; --chunk) {
    if (rows % chunk) continue;
    PlanarReadPlan p{rows, columns, offsets[0], chunk,
        chunk > 1 ? offsets[1] - offsets[0] : 0,
        chunk < rows ? offsets[chunk] - offsets[0] : 0, columnStride};
    bool exact = true;
    for (int64_t c = 0; exact && c < columns; ++c)
      for (int64_t r = 0; exact && r < rows; ++r)
        exact = offsets[c * rows + r] >= 0 &&
            p.address(r, c) == offsets[c * rows + r];
    if (exact) return p;
  }
  return std::nullopt;
}

inline std::optional<PlanarReadPlan> slicePlanarRows(
    const PlanarReadPlan& source, int64_t first, int64_t count) {
  if (first < 0 || count <= 0 || first + count > source.rows)
    return std::nullopt;
  std::vector<int64_t> offsets;
  for (int64_t c = 0; c < source.columns; ++c)
    for (int64_t r = first; r < first + count; ++r)
      offsets.push_back(source.address(r,c));
  return planPlanarRead(count, source.columns, offsets);
}
} // namespace ftlpu::compiler::analysis
