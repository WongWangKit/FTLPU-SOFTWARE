#include "ftlpu/system/tsp_slice_system.hpp"

static_assert(ftlpu::TspSliceSystem::kMxmCount >= 2,
    "FTLPU CModel must expose at least one MXM per hemisphere");
