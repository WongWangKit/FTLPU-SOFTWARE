#pragma once
#include "ftlpu/compiler/Analysis/planar_access.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace ftlpu::compiler::analysis {
inline mlir::DictionaryAttr encodePlanarRead(mlir::Builder& b, const PlanarReadPlan& p) {
  return b.getDictionaryAttr({
    b.getNamedAttr("rows",b.getI64IntegerAttr(p.rows)),
    b.getNamedAttr("columns",b.getI64IntegerAttr(p.columns)),
    b.getNamedAttr("offset",b.getI64IntegerAttr(p.offset)),
    b.getNamedAttr("row_chunk",b.getI64IntegerAttr(p.rowChunk)),
    b.getNamedAttr("row_stride",b.getI64IntegerAttr(p.rowStride)),
    b.getNamedAttr("group_stride",b.getI64IntegerAttr(p.groupStride)),
    b.getNamedAttr("column_stride",b.getI64IntegerAttr(p.columnStride))});
}
inline std::optional<PlanarReadPlan> decodePlanarRead(mlir::DictionaryAttr home) {
  auto a=home.getAs<mlir::DictionaryAttr>("planar_read");
  if(!a)return std::nullopt;
  auto v=[&](const char* name){return a.getAs<mlir::IntegerAttr>(name).getInt();};
  PlanarReadPlan p{v("rows"),v("columns"),v("offset"),v("row_chunk"),
      v("row_stride"),v("group_stride"),v("column_stride")};
  if(p.rows<=0 || p.columns<=0 || p.rowChunk<=0 || p.rows%p.rowChunk)
    llvm::report_fatal_error("invalid planar access contract");
  return p;
}
} // namespace ftlpu::compiler::analysis
