#include "ScheduleEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"

#include "llvm/Support/FormatVariadic.h"

namespace ftlpu::compiler::schedule::detail {

int64_t get_slice(mlir::DictionaryAttr address)
{
    return address.getAs<mlir::IntegerAttr>("slice").getInt();
}

llvm::SmallVector<int64_t> get_slices(mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    for (mlir::Attribute attribute : placement.getAs<mlir::ArrayAttr>("slices"))
        result.push_back(llvm::cast<mlir::IntegerAttr>(attribute).getInt());
    return result;
}

mlir::DictionaryAttr subrange_placement(mlir::OpBuilder& builder,
    mlir::DictionaryAttr placement, int64_t row_offset, int64_t row_count)
{
    mlir::NamedAttrList attributes(placement);
    const int64_t base = placement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t stride =
        placement.getAs<mlir::IntegerAttr>("address_stride").getInt();
    attributes.set(
        "base_row", builder.getI64IntegerAttr(base + row_offset * stride));
    attributes.set(
        "instruction_count", builder.getI64IntegerAttr(row_count));
    return attributes.getDictionary(builder.getContext());
}

mlir::DictionaryAttr weight_pass_placement(mlir::OpBuilder& builder,
    mlir::DictionaryAttr placement, int64_t pass)
{
    mlir::NamedAttrList attributes(placement);
    const int64_t base = placement.getAs<mlir::IntegerAttr>("base_row").getInt();
    attributes.set("base_row", builder.getI64IntegerAttr(base + pass * 320));
    attributes.set("instruction_count", builder.getI64IntegerAttr(20));
    return attributes.getDictionary(builder.getContext());
}

std::string mem_resource(int64_t slice)
{
    return llvm::formatv("MEM.slice.{0}", slice).str();
}

int64_t value_ready_cycle(mlir::Value value)
{
    if (auto write = value.getDefiningOp<MemWriteOp>())
        return write.getCycle() + write.getDuration();
    return 0;
}

void add_stream_windows(llvm::SmallVectorImpl<ResourceWindow>& windows,
    llvm::StringRef direction, int64_t base, int64_t count,
    int64_t offset, int64_t duration)
{
    for (int64_t stream = base; stream < base + count; ++stream) {
        windows.push_back({llvm::formatv("stream.{0}.{1}", direction, stream).str(),
            offset, duration});
    }
}

} // namespace ftlpu::compiler::schedule::detail
