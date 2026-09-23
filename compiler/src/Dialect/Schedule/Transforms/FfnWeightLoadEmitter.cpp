#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>
#include <cassert>

namespace ftlpu::compiler::schedule::ffn_detail {
namespace {

int64_t functionArgumentIndex(mlir::Value value) {
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
    return argument.getArgNumber();
  return -1;
}

} // namespace

void setFfnLoopDomain3D(mlir::Operation* operation,
    mlir::OpBuilder& builder, FfnLoopDomain3D domain) {
  assert(domain.wave_count > 0 && domain.wave_interval > 0
      && domain.group_count > 0 && domain.group_interval > 0
      && "FFN loop domains require positive counts and intervals");
  operation->setAttr("wave_count",
                     builder.getI64IntegerAttr(domain.wave_count));
  operation->setAttr("wave_interval",
                     builder.getI64IntegerAttr(domain.wave_interval));
  operation->setAttr("wave_address_stride",
                     builder.getI64IntegerAttr(domain.wave_address_stride));
  operation->setAttr("group_count",
                     builder.getI64IntegerAttr(domain.group_count));
  operation->setAttr("group_interval",
                     builder.getI64IntegerAttr(domain.group_interval));
  operation->setAttr("group_address_stride",
                     builder.getI64IntegerAttr(domain.group_address_stride));
  if (domain.blocked_outer_address) {
    operation->setAttr("outer_group_size",
                       builder.getI64IntegerAttr(domain.outer_group_size));
    operation->setAttr("outer_inner_stride",
                       builder.getI64IntegerAttr(domain.outer_inner_stride));
    operation->setAttr("outer_group_stride",
                       builder.getI64IntegerAttr(domain.outer_group_stride));
  }
}

MemReadOp emitFfnMemRead3D(mlir::IRRewriter& rewriter,
    mlir::Location location, mlir::Value input, int64_t cycle,
    int64_t duration, int64_t streamBase, int64_t streamCount,
    int64_t registerId, mlir::StringAttr direction, mlir::StringAttr role,
    mlir::DictionaryAttr address, mlir::DictionaryAttr placement,
    int64_t bytes, FfnLoopDomain3D domain) {
  auto read = rewriter.create<MemReadOp>(location, input, cycle, duration,
      streamBase, streamCount, registerId, direction, role, address,
      placement, bytes);
  setFfnLoopDomain3D(read.getOperation(), rewriter, domain);
  return read;
}

MemTransferOp emitFfnMemTransfer3D(mlir::IRRewriter& rewriter,
    mlir::Location location, int64_t cycle, int64_t hemisphere,
    int64_t slice, llvm::StringRef opcode, int64_t address,
    int64_t packedStream, int64_t repeatCount, int64_t repeatInterval,
    int64_t addressStride, int64_t bank, FfnLoopDomain3D domain,
    int64_t addressBinding, llvm::StringRef addressBindingAccess,
    int64_t weightPage, int64_t logicalBaseRow) {
  mlir::OperationState state(location, MemTransferOp::getOperationName());
  state.addAttributes({
      rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
      rewriter.getNamedAttr("hemisphere",
                            rewriter.getI64IntegerAttr(hemisphere)),
      rewriter.getNamedAttr("slice", rewriter.getI64IntegerAttr(slice)),
      rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
      rewriter.getNamedAttr("address", rewriter.getI64IntegerAttr(address)),
      rewriter.getNamedAttr("packed_stream",
                            rewriter.getI64IntegerAttr(packedStream)),
      rewriter.getNamedAttr("repeat_count",
                            rewriter.getI64IntegerAttr(repeatCount)),
      rewriter.getNamedAttr("repeat_interval",
                            rewriter.getI64IntegerAttr(repeatInterval)),
      rewriter.getNamedAttr("address_stride",
                            rewriter.getI64IntegerAttr(addressStride)),
      rewriter.getNamedAttr("bank", rewriter.getI64IntegerAttr(bank)),
      rewriter.getNamedAttr("wave_count",
                            rewriter.getI64IntegerAttr(domain.wave_count)),
      rewriter.getNamedAttr("wave_interval",
                            rewriter.getI64IntegerAttr(domain.wave_interval)),
      rewriter.getNamedAttr("wave_address_stride",
          rewriter.getI64IntegerAttr(domain.wave_address_stride)),
      rewriter.getNamedAttr("group_count",
                            rewriter.getI64IntegerAttr(domain.group_count)),
      rewriter.getNamedAttr("group_interval",
                            rewriter.getI64IntegerAttr(domain.group_interval)),
      rewriter.getNamedAttr("group_address_stride",
          rewriter.getI64IntegerAttr(domain.group_address_stride)),
  });
  if (domain.blocked_outer_address) {
    state.addAttribute("outer_group_size",
                       rewriter.getI64IntegerAttr(domain.outer_group_size));
    state.addAttribute("outer_inner_stride",
                       rewriter.getI64IntegerAttr(domain.outer_inner_stride));
    state.addAttribute("outer_group_stride",
                       rewriter.getI64IntegerAttr(domain.outer_group_stride));
  }
  if (addressBinding >= 0) {
    state.addAttribute("address_binding",
                       rewriter.getI64IntegerAttr(addressBinding));
    state.addAttribute("address_binding_access",
                       rewriter.getStringAttr(addressBindingAccess));
  }
  if (weightPage >= 0)
    state.addAttribute("weight_page",
                       rewriter.getI64IntegerAttr(weightPage));
  if (logicalBaseRow >= 0)
    state.addAttribute("logical_base_row",
                       rewriter.getI64IntegerAttr(logicalBaseRow));
  return llvm::cast<MemTransferOp>(rewriter.create(state));
}

MxmIssueOp emitFfnMxmIssue3D(mlir::IRRewriter& rewriter,
    mlir::Location location, int64_t cycle, int64_t unit,
    llvm::StringRef opcode, int64_t weightBuffer, int64_t weightColumn,
    int64_t activationStream, int64_t outputStream,
    int64_t accumulatorAddress, int64_t accumulatorRowStride,
    llvm::StringRef accumulatorDestination, bool accumulatorClear,
    llvm::StringRef dataFormat, llvm::StringRef accumulatorOutputFormat,
    const FfnMxmDomain3D& domain) {
  mlir::OperationState state(location, MxmIssueOp::getOperationName());
  state.addAttributes({
      rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
      rewriter.getNamedAttr("unit_id", rewriter.getI64IntegerAttr(unit)),
      rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
      rewriter.getNamedAttr("weight_buffer",
                            rewriter.getI64IntegerAttr(weightBuffer)),
      rewriter.getNamedAttr("weight_column",
                            rewriter.getI64IntegerAttr(weightColumn)),
      rewriter.getNamedAttr("activation_stream_base",
                            rewriter.getI64IntegerAttr(activationStream)),
      rewriter.getNamedAttr("output_stream_base",
                            rewriter.getI64IntegerAttr(outputStream)),
      rewriter.getNamedAttr("repeat_count",
                            rewriter.getI64IntegerAttr(domain.repeat_count)),
      rewriter.getNamedAttr("repeat_interval",
                            rewriter.getI64IntegerAttr(domain.repeat_interval)),
      rewriter.getNamedAttr("repeat_weight_column_stride",
          rewriter.getI64IntegerAttr(domain.repeat_weight_column_stride)),
      rewriter.getNamedAttr("repeat_accumulator_address_stride",
          rewriter.getI64IntegerAttr(
              domain.repeat_accumulator_address_stride)),
      rewriter.getNamedAttr("wave_count",
                            rewriter.getI64IntegerAttr(domain.wave_count)),
      rewriter.getNamedAttr("wave_interval",
                            rewriter.getI64IntegerAttr(domain.wave_interval)),
      rewriter.getNamedAttr("wave_weight_column_stride",
          rewriter.getI64IntegerAttr(domain.wave_weight_column_stride)),
      rewriter.getNamedAttr("wave_accumulator_address_stride",
          rewriter.getI64IntegerAttr(
              domain.wave_accumulator_address_stride)),
      rewriter.getNamedAttr("group_count",
                            rewriter.getI64IntegerAttr(domain.group_count)),
      rewriter.getNamedAttr("group_interval",
                            rewriter.getI64IntegerAttr(domain.group_interval)),
      rewriter.getNamedAttr("group_weight_column_stride",
          rewriter.getI64IntegerAttr(domain.group_weight_column_stride)),
      rewriter.getNamedAttr("group_accumulator_address_stride",
          rewriter.getI64IntegerAttr(
              domain.group_accumulator_address_stride)),
      rewriter.getNamedAttr("accumulator_address",
                            rewriter.getI64IntegerAttr(accumulatorAddress)),
      rewriter.getNamedAttr("accumulator_row_stride",
                            rewriter.getI64IntegerAttr(accumulatorRowStride)),
      rewriter.getNamedAttr("accumulator_destination",
                            rewriter.getStringAttr(accumulatorDestination)),
      rewriter.getNamedAttr("accumulator_clear",
                            rewriter.getBoolAttr(accumulatorClear)),
      rewriter.getNamedAttr("data_format", rewriter.getStringAttr(dataFormat)),
      rewriter.getNamedAttr("weight_load_mode",
                            rewriter.getStringAttr("supercell")),
      rewriter.getNamedAttr("weight_inner_column",
                            rewriter.getI64IntegerAttr(0)),
      rewriter.getNamedAttr("weight_buffer_mode",
                            rewriter.getStringAttr(domain.weight_buffer_mode)),
  });
  if (!accumulatorOutputFormat.empty())
    state.addAttribute("accumulator_output_format",
                       rewriter.getStringAttr(accumulatorOutputFormat));
  if (domain.terminal_dimension >= 0)
    state.addAttribute("terminal_dimension",
                       rewriter.getI64IntegerAttr(domain.terminal_dimension));
  if (!domain.terminal_accumulator_destination.empty())
    state.addAttribute("terminal_accumulator_destination",
        rewriter.getStringAttr(domain.terminal_accumulator_destination));
  if (domain.terminal_accumulator_clear)
    state.addAttribute("terminal_accumulator_clear",
        rewriter.getBoolAttr(*domain.terminal_accumulator_clear));
  if (!domain.terminal_accumulator_output_format.empty())
    state.addAttribute("terminal_accumulator_output_format",
        rewriter.getStringAttr(
            domain.terminal_accumulator_output_format));
  return llvm::cast<MxmIssueOp>(rewriter.create(state));
}

MxmLoadOp emitFfnWeightTile(
    mlir::IRRewriter &rewriter, mlir::Location location,
    stream::RouteOp rawRoute, mlir::Type dequantizedType,
    llvm::ArrayRef<int64_t> weightSlices, const target::LPUTargetModel &target,
    float scale, int64_t startCycle, int64_t baseRow, int64_t hemisphere,
    int64_t localMxm, int64_t unit, int64_t weightBuffer, bool localDequant,
    int64_t bank, int64_t pageIndex, int64_t logicalBaseRow,
    mlir::DictionaryAttr bindingPlacement, FfnLoopDomain3D domain,
    llvm::StringRef weightBufferMode,
    std::optional<FfnLoopDomain3D> mxmControlDomain,
    bool emitMxmControl) {
  const auto &throughput = target.throughput();
  const int64_t duration = throughput.mxm_rows / throughput.lanes_per_tile;
  const int64_t encodedStreamBase = target.streams().streams_per_direction;
  const auto hemi = hemisphere_name(hemisphere);
  constexpr auto direction = target::StreamDirection::East;
  constexpr auto directionName = "east";
  const auto dataFormat = lpu_16bit_data_format(
      llvm::cast<mlir::RankedTensorType>(dequantizedType).getElementType());
  mlir::Value readValue;

  if (localDequant) {
    const int64_t packedLoadsPerDirectLoad =
        (throughput.mxm_load_streams_per_cycle +
         throughput.mxm_int8_load_streams_per_cycle - 1) /
        throughput.mxm_int8_load_streams_per_cycle;
    const int64_t streamPartitions =
        std::max(throughput.mxms_per_hemisphere, packedLoadsPerDirectLoad);
    const int64_t partitionWidth =
        target.streams().streams_per_direction / streamPartitions;
    const int64_t streamBase = localMxm * partitionWidth + partitionWidth -
                               throughput.mxm_int8_load_streams_per_cycle;
    assert(streamBase >= 0 &&
           streamBase + throughput.mxm_int8_load_streams_per_cycle <=
               target.streams().streams_per_direction &&
           "validated target must provide a disjoint INT8 weight window");
    for (int64_t stream = 0; stream < rawRoute.getStreamCount(); ++stream) {
      const int64_t slice = weightSlices[stream];
      const int64_t latency =
          target
              .transport_latency(target::StreamEndpoint::Mem,
                                 target::StreamEndpoint::MxmWeight, direction,
                                 slice)
              .value_or(slice / target.streams().mem_slices_per_register_group +
                        2);
      auto placement = schedule_placement(rewriter, {slice}, baseRow, duration,
                                          1, hemi, "schedule_slice", bank);
      mlir::NamedAttrList attributes(placement);
      attributes.set("binding_placement",
                     bindingPlacement ? bindingPlacement
                                      : rawRoute.getPlacement());
      if (pageIndex >= 0) {
        attributes.set("weight_page", rewriter.getI64IntegerAttr(pageIndex));
        attributes.set("logical_base_row",
                       rewriter.getI64IntegerAttr(logicalBaseRow));
      }
      auto read = emitFfnMemRead3D(
          rewriter, location, rawRoute.getInput(), startCycle - latency, duration,
          streamBase + stream, 1,
          slice / target.streams().mem_slices_per_register_group + 1,
          rewriter.getStringAttr(directionName),
          rewriter.getStringAttr("weight_i8"), rawRoute.getAddress(),
          attributes.getDictionary(rewriter.getContext()),
          duration * throughput.lanes_per_tile, domain);
      readValue = read.getOutput();
    }

    if (!emitMxmControl)
      return MxmLoadOp {};
    const FfnLoopDomain3D control = mxmControlDomain.value_or(domain);
    mlir::OperationState dequantState(location,
                                      MxmDequantOp::getOperationName());
    dequantState.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(startCycle)),
        rewriter.getNamedAttr("unit_id", rewriter.getI64IntegerAttr(unit)),
        rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)),
        rewriter.getNamedAttr("repeat_count",
                              rewriter.getI64IntegerAttr(duration)),
        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
    });
    const int64_t scaleBinding = functionArgumentIndex(rawRoute.getInput());
    if (scaleBinding >= 0)
      dequantState.addAttribute("scale_binding",
                                rewriter.getI64IntegerAttr(scaleBinding));
    dequantState.addAttribute("wave_count",
                              rewriter.getI64IntegerAttr(control.wave_count));
    dequantState.addAttribute("wave_interval",
                              rewriter.getI64IntegerAttr(control.wave_interval));
    dequantState.addAttribute("group_count",
                              rewriter.getI64IntegerAttr(control.group_count));
    dequantState.addAttribute("group_interval",
                              rewriter.getI64IntegerAttr(control.group_interval));
    rewriter.create(dequantState);

    const bool flatLoadDomain = control.wave_count == 1
        || control.group_count == 1
        || control.group_interval
            == control.wave_count * control.wave_interval;
    const int64_t loadDomainCount = flatLoadDomain
        ? 1 : control.group_count;
    MxmLoadOp lastLoad;
    for (int64_t group = 0; group < loadDomainCount; ++group) {
      const int64_t loopCount = flatLoadDomain
          ? control.wave_count * control.group_count
          : control.wave_count;
      const int64_t loopInterval = control.wave_count > 1
          ? control.wave_interval : control.group_interval;
      const int64_t groupBuffer = weightBuffer
          ^ ((group * control.wave_count) & 1);
      auto load = rewriter.create<MxmLoadOp>(
          location, readValue,
          startCycle + (flatLoadDomain ? 0 : group * control.group_interval),
          duration, streamBase,
          throughput.mxm_int8_load_streams_per_cycle, unit, groupBuffer);
      load->setAttr("data_format", rewriter.getStringAttr(dataFormat));
      load->setAttr("weight_load_mode", rewriter.getStringAttr("supercell"));
      load->setAttr("weight_input_mode",
                    rewriter.getStringAttr("int8_dequant_bf16"));
      load->setAttr("group_count",
                    rewriter.getI64IntegerAttr(loopCount));
      load->setAttr("group_interval",
                    rewriter.getI64IntegerAttr(loopInterval));
      load->setAttr("weight_buffer_mode",
                    rewriter.getStringAttr(loopCount > 1
                        ? weightBufferMode : "fixed"));
      lastLoad = load;
    }
    return lastLoad;
  }

  for (int64_t stream = 0; stream < rawRoute.getStreamCount(); ++stream) {
    const int64_t slice = weightSlices[stream];
    const int64_t latency =
        target
            .transport_latency(target::StreamEndpoint::Mem,
                               target::StreamEndpoint::MxmWeight, direction,
                               slice)
            .value_or(slice / target.streams().mem_slices_per_register_group +
                      2);
    auto placement = schedule_placement(rewriter, {slice}, baseRow, duration, 1,
                                        hemi, "schedule_slice", bank);
    mlir::NamedAttrList attributes(placement);
    attributes.set("binding_placement",
                   bindingPlacement ? bindingPlacement
                                    : rawRoute.getPlacement());
    if (pageIndex >= 0) {
      attributes.set("weight_page", rewriter.getI64IntegerAttr(pageIndex));
      attributes.set("logical_base_row",
                     rewriter.getI64IntegerAttr(logicalBaseRow));
    }
    auto read = emitFfnMemRead3D(
        rewriter, location, rawRoute.getInput(), startCycle - latency, duration, stream,
        1, slice / target.streams().mem_slices_per_register_group + 1,
        rewriter.getStringAttr(directionName),
        rewriter.getStringAttr("weight_i8"), rawRoute.getAddress(),
        attributes.getDictionary(rewriter.getContext()),
        duration * throughput.mxm_rows, domain);
    readValue = read.getOutput();
  }

  mlir::Value value = readValue;
  MxmLoadOp lastLoad;
  for (int64_t group = 0; group < domain.group_count; ++group) {
    const int64_t groupCycle = startCycle + group * domain.group_interval;
    for (int64_t stream = 0; stream < rawRoute.getStreamCount(); ++stream) {
      auto multiply = create_vxm(rewriter, location, readValue, readValue,
          dequantizedType, groupCycle, stream, "multiply", "stream_i8",
          encodedStreamBase + stream, 0.0f, "immediate", 0, scale,
          "fp32", -1, duration, 1, hemi, hemi,
          functionArgumentIndex(rawRoute.getInput()));
      multiply->setAttr("wave_count",
          rewriter.getI64IntegerAttr(domain.wave_count));
      multiply->setAttr("wave_interval",
          rewriter.getI64IntegerAttr(domain.wave_interval));
      value = multiply.getResult();
      auto cast = create_vxm(rewriter, location, value, readValue,
          dequantizedType, groupCycle + 1, 8 + stream, "cast", "alu",
          stream, 0.0f, "immediate", 0, 0.0f, dataFormat,
          localMxm * throughput.mxm_load_streams_per_cycle + stream * 2,
          duration, 1, hemi, hemi);
      cast->setAttr("wave_count",
          rewriter.getI64IntegerAttr(domain.wave_count));
      cast->setAttr("wave_interval",
          rewriter.getI64IntegerAttr(domain.wave_interval));
      value = cast.getResult();
    }
    const int64_t groupBuffer = weightBuffer
        ^ ((group * domain.wave_count) & 1);
    auto load = rewriter.create<MxmLoadOp>(location, value,
        groupCycle + throughput.vxm_weight_to_iw_latency, duration, 0,
        throughput.mxm_load_streams_per_cycle, unit, groupBuffer);
    load->setAttr("group_count",
                  rewriter.getI64IntegerAttr(domain.wave_count));
    load->setAttr("group_interval",
                  rewriter.getI64IntegerAttr(domain.wave_interval));
    load->setAttr("weight_buffer_mode",
                  rewriter.getStringAttr(domain.wave_count > 1
                      ? weightBufferMode : "fixed"));
    lastLoad = load;
  }
  return lastLoad;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
