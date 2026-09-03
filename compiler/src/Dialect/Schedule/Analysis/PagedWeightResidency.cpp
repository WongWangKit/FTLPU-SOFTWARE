#include "ftlpu/compiler/Dialect/Schedule/Analysis/paged_weight_residency.hpp"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>

namespace ftlpu::compiler::schedule {
namespace {

struct WeightResidencyRegion {
    int64_t bank = 0;
    int64_t hemisphereMask = 3;
    int64_t rowBegin = 0;
    int64_t rowEnd = 1;
    llvm::SmallVector<int64_t, 32> slices;
};

int64_t integerAttrOr(
    mlir::DictionaryAttr placement, llvm::StringRef name, int64_t fallback)
{
    if (const auto value = placement.getAs<mlir::IntegerAttr>(name))
        return value.getInt();
    return fallback;
}

int64_t hemisphereMask(mlir::DictionaryAttr placement)
{
    const auto hemisphere = placement.getAs<mlir::StringAttr>("hemisphere");
    if (!hemisphere || hemisphere.getValue() == "both") return 3;
    if (hemisphere.getValue() == "east") return 1;
    if (hemisphere.getValue() == "west") return 2;
    return 3;
}

WeightResidencyRegion residencyRegion(
    mlir::DictionaryAttr placement, int64_t bank)
{
    WeightResidencyRegion region;
    region.bank = bank;
    region.hemisphereMask = hemisphereMask(placement);
    region.rowBegin = std::max<int64_t>(
        0, integerAttrOr(placement, "base_row", 0));
    const int64_t instructionCount = std::max<int64_t>(
        1, integerAttrOr(placement, "instruction_count", 1));
    const int64_t addressStride = std::max<int64_t>(
        1, integerAttrOr(placement, "address_stride", 1));
    const int64_t maxSpan = std::numeric_limits<int64_t>::max()
        - region.rowBegin;
    const int64_t rowSpan = instructionCount > maxSpan / addressStride
        ? maxSpan : instructionCount * addressStride;
    region.rowEnd = region.rowBegin + rowSpan;

    const auto logicalSlices = placement.getAs<mlir::ArrayAttr>("slices");
    const auto storageSlices =
        placement.getAs<mlir::ArrayAttr>("page_storage_slices");
    if (storageSlices) {
        const int64_t groupWidth = logicalSlices && !logicalSlices.empty()
            ? static_cast<int64_t>(logicalSlices.size()) : 8;
        const int64_t groupBase = std::max<int64_t>(
            0, integerAttrOr(placement, "page_role_group_base", 0));
        const int64_t groupCount = std::max<int64_t>(
            1, integerAttrOr(placement, "page_role_group_count", 1));
        const int64_t begin = std::min<int64_t>(
            storageSlices.size(), groupBase * groupWidth);
        const int64_t end = std::min<int64_t>(
            storageSlices.size(), (groupBase + groupCount) * groupWidth);
        for (int64_t index = begin; index < end; ++index)
            region.slices.push_back(
                llvm::cast<mlir::IntegerAttr>(storageSlices[index]).getInt());
    } else if (logicalSlices) {
        for (mlir::Attribute slice : logicalSlices)
            region.slices.push_back(
                llvm::cast<mlir::IntegerAttr>(slice).getInt());
    }
    std::sort(region.slices.begin(), region.slices.end());
    region.slices.erase(
        std::unique(region.slices.begin(), region.slices.end()),
        region.slices.end());
    return region;
}

int64_t placementBank(mlir::DictionaryAttr placement)
{
    return integerAttrOr(placement, "bank", 0);
}

} // namespace

bool pagedWeightResidencyOverlaps(
    mlir::DictionaryAttr lhs, mlir::DictionaryAttr rhs)
{
    if (!lhs || !rhs) return true;
    return pagedWeightResidencyOverlaps(
        lhs, placementBank(lhs), rhs, placementBank(rhs));
}

bool pagedWeightResidencyOverlaps(mlir::DictionaryAttr lhs, int64_t lhsBank,
    mlir::DictionaryAttr rhs, int64_t rhsBank)
{
    if (!lhs || !rhs) return true;
    const WeightResidencyRegion left = residencyRegion(lhs, lhsBank);
    const WeightResidencyRegion right = residencyRegion(rhs, rhsBank);
    if (left.bank != right.bank
        || (left.hemisphereMask & right.hemisphereMask) == 0
        || left.rowBegin >= right.rowEnd || right.rowBegin >= left.rowEnd)
        return false;
    if (left.slices.empty() || right.slices.empty()) return true;
    return std::ranges::any_of(left.slices, [&](int64_t slice) {
        return std::binary_search(
            right.slices.begin(), right.slices.end(), slice);
    });
}

} // namespace ftlpu::compiler::schedule
