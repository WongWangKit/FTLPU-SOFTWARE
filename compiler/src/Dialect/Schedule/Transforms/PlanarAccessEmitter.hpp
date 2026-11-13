// Emit an already-proven logical access domain. The domain comes from the
// graph/layout planner; this layer neither infers views nor creates copies.
inline void emitPlanarReadDomain(mlir::IRRewriter& r,mlir::Location loc,
    const target::LPUTargetModel& target,int64_t cycle,int64_t physicalSlice,
    int64_t base,int64_t stream,int64_t bank,
    const analysis::PlanarReadPlan& access,int64_t columns) {
  direct_domain_detail::emitMem3D(r,loc,target,cycle,physicalSlice,"read",
      base+access.offset,stream,access.rowChunk,1,access.rowStride,
      access.rows/access.rowChunk,access.rowChunk,access.groupStride,
      columns,access.rows,access.columnStride,-1,bank);
}
