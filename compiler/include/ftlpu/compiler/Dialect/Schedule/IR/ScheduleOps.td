include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ScheduleDialect : Dialect {
  let name = "ftlpu.schedule";
  let cppNamespace = "::ftlpu::compiler::schedule";
}

class FTLPU_Schedule_Op<string mnemonic, list<Trait> traits = []>
    : Op<ScheduleDialect, mnemonic, traits>;

def MemTransferOp : FTLPU_Schedule_Op<"mem_transfer", [MemoryEffects<[MemWrite]>]> {
  let summary = "Cycle-accurate transfer through a physical MEM slice";
  let arguments = (ins I64Attr:$cycle, I64Attr:$hemisphere, I64Attr:$slice,
      StrAttr:$opcode,
      I64Attr:$address, I64Attr:$packed_stream, I64Attr:$repeat_count,
      I64Attr:$repeat_interval, I64Attr:$address_stride,
      OptionalAttr<I64Attr>:$bank,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval,
      OptionalAttr<I64Attr>:$wave_address_stride,
      OptionalAttr<I64Attr>:$address_binding,
      OptionalAttr<I64Attr>:$weight_page,
      OptionalAttr<I64Attr>:$logical_base_row);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MxmIssueOp : FTLPU_Schedule_Op<"mxm_issue", [MemoryEffects<[MemWrite]>]> {
  let summary = "Cycle-accurate issue to a physical MXM unit";
  let arguments = (ins I64Attr:$cycle, I64Attr:$unit_id, StrAttr:$opcode,
      I64Attr:$weight_buffer, I64Attr:$weight_column, I64Attr:$activation_stream_base,
      I64Attr:$output_stream_base, I64Attr:$repeat_count,
      I64Attr:$repeat_interval, I64Attr:$accumulator_address,
      I64Attr:$accumulator_row_stride, StrAttr:$accumulator_destination,
      BoolAttr:$accumulator_clear,
      OptionalAttr<StrAttr>:$data_format,
      OptionalAttr<StrAttr>:$accumulator_output_format,
      OptionalAttr<StrAttr>:$weight_load_mode,
      OptionalAttr<I64Attr>:$weight_inner_column,
      OptionalAttr<StrAttr>:$weight_input_mode,
      OptionalAttr<I64Attr>:$weight_stream_base,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval,
      OptionalAttr<I64Attr>:$wave_weight_column_stride,
      OptionalAttr<I64Attr>:$wave_accumulator_address_stride,
      OptionalAttr<I64Attr>:$group_count,
      OptionalAttr<I64Attr>:$group_interval);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MxmDequantOp : FTLPU_Schedule_Op<"mxm_dequant",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Cycle-accurate issue to an MXM-local weight dequantizer";
  let arguments = (ins I64Attr:$cycle, I64Attr:$unit_id, F32Attr:$scale,
      I64Attr:$repeat_count, I64Attr:$repeat_interval,
      OptionalAttr<I64Attr>:$scale_binding,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MemReadOp : FTLPU_Schedule_Op<"mem_read"> {
  let summary = "Issue a cycle-accurate MEM read stream";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$cycle, I64Attr:$duration,
      I64Attr:$stream_base, I64Attr:$stream_count, I64Attr:$register_id,
      OptionalAttr<ArrayAttr>:$register_ids,
      StrAttr:$direction, StrAttr:$role, DictionaryAttr:$address,
      DictionaryAttr:$placement, I64Attr:$bytes,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval,
      OptionalAttr<I64Attr>:$wave_address_stride);
  let results = (outs AnyRankedTensor:$output);
  let builders = [OpBuilder<(ins "::mlir::Value":$input, "int64_t":$cycle,
      "int64_t":$duration, "int64_t":$stream_base, "int64_t":$stream_count,
      "int64_t":$register_id, "::mlir::StringAttr":$direction,
      "::mlir::StringAttr":$role, "::mlir::DictionaryAttr":$address,
      "::mlir::DictionaryAttr":$placement,
      "int64_t":$bytes), [{
    $_state.addOperands(input); $_state.addTypes(input.getType());
    $_state.addAttribute("cycle", $_builder.getI64IntegerAttr(cycle));
    $_state.addAttribute("duration", $_builder.getI64IntegerAttr(duration));
    $_state.addAttribute("stream_base", $_builder.getI64IntegerAttr(stream_base));
    $_state.addAttribute("stream_count", $_builder.getI64IntegerAttr(stream_count));
    $_state.addAttribute("register_id", $_builder.getI64IntegerAttr(register_id));
    $_state.addAttribute("direction", direction); $_state.addAttribute("role", role);
    $_state.addAttribute("address", address);
    $_state.addAttribute("placement", placement);
    $_state.addAttribute("bytes", $_builder.getI64IntegerAttr(bytes));
  }] >];
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def MxmLoadOp : FTLPU_Schedule_Op<"mxm_load"> {
  let summary = "Issue MXM IW instructions";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$cycle, I64Attr:$duration,
        I64Attr:$stream_base, I64Attr:$stream_count, I64Attr:$unit_id,
        I64Attr:$weight_buffer,
        OptionalAttr<StrAttr>:$data_format,
        OptionalAttr<StrAttr>:$weight_load_mode,
        OptionalAttr<I64Attr>:$weight_inner_column,
        OptionalAttr<StrAttr>:$weight_input_mode,
        OptionalAttr<I64Attr>:$group_count,
        OptionalAttr<I64Attr>:$group_interval);
  let results = (outs AnyRankedTensor:$output);
  let builders = [OpBuilder<(ins "::mlir::Value":$input, "int64_t":$cycle,
      "int64_t":$duration, "int64_t":$stream_base, "int64_t":$stream_count,
      "int64_t":$unit_id, "int64_t":$weight_buffer), [{
    $_state.addOperands(input); $_state.addTypes(input.getType());
    $_state.addAttribute("cycle", $_builder.getI64IntegerAttr(cycle));
    $_state.addAttribute("duration", $_builder.getI64IntegerAttr(duration));
    $_state.addAttribute("stream_base", $_builder.getI64IntegerAttr(stream_base));
    $_state.addAttribute("stream_count", $_builder.getI64IntegerAttr(stream_count));
    $_state.addAttribute("unit_id", $_builder.getI64IntegerAttr(unit_id));
    $_state.addAttribute("weight_buffer", $_builder.getI64IntegerAttr(weight_buffer));
    auto &properties = $_state.getOrAddProperties<Properties>();
    properties.data_format = $_builder.getStringAttr(
        ::llvm::cast<::mlir::ShapedType>(input.getType())
                .getElementType().isBF16()
            ? "bf16"
            : "fp16");
  }] >];
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def MxmComputeOp : FTLPU_Schedule_Op<"mxm_compute"> {
  let summary = "Issue cycle-accurate MXM Compute instructions";
  let arguments = (ins AnyRankedTensor:$activation, AnyRankedTensor:$weight,
      I64Attr:$cycle, I64Attr:$duration, I64Attr:$result_cycle,
      I64Attr:$result_duration, I64Attr:$activation_stream_base,
      I64Attr:$output_stream_base, I64Attr:$weight_buffer,
      I64Attr:$unit_id, OptionalAttr<StrAttr>:$data_format,
      I64Attr:$m, I64Attr:$n, I64Attr:$k,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval,
      OptionalAttr<I64Attr>:$wave_accumulator_address_stride);
  let results = (outs AnyRankedTensor:$result);
  let builders = [OpBuilder<(ins "::mlir::Value":$activation, "::mlir::Value":$weight,
      "::mlir::Type":$result_type, "int64_t":$cycle, "int64_t":$duration,
      "int64_t":$result_cycle, "int64_t":$result_duration,
      "int64_t":$activation_stream_base, "int64_t":$output_stream_base,
      "int64_t":$weight_buffer, "int64_t":$unit_id,
      "int64_t":$m, "int64_t":$n, "int64_t":$k), [{
    $_state.addOperands({activation, weight}); $_state.addTypes(result_type);
    $_state.addAttribute("cycle", $_builder.getI64IntegerAttr(cycle));
    $_state.addAttribute("duration", $_builder.getI64IntegerAttr(duration));
    $_state.addAttribute("result_cycle", $_builder.getI64IntegerAttr(result_cycle));
    $_state.addAttribute("result_duration", $_builder.getI64IntegerAttr(result_duration));
    $_state.addAttribute("activation_stream_base", $_builder.getI64IntegerAttr(activation_stream_base));
    $_state.addAttribute("output_stream_base", $_builder.getI64IntegerAttr(output_stream_base));
    $_state.addAttribute("weight_buffer", $_builder.getI64IntegerAttr(weight_buffer));
    $_state.addAttribute("unit_id", $_builder.getI64IntegerAttr(unit_id));
    auto &properties = $_state.getOrAddProperties<Properties>();
    properties.data_format = $_builder.getStringAttr(
        ::llvm::cast<::mlir::ShapedType>(activation.getType())
                .getElementType().isBF16()
            ? "bf16"
            : "fp16");
    $_state.addAttribute("m", $_builder.getI64IntegerAttr(m));
    $_state.addAttribute("n", $_builder.getI64IntegerAttr(n));
    $_state.addAttribute("k", $_builder.getI64IntegerAttr(k));
  }] >];
  let assemblyFormat = "$activation `,` $weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def MxmAccumulatorReadOp : FTLPU_Schedule_Op<"mxm_accumulator_read"> {
  let summary = "Read one row from an MXM-local accumulator to west streams";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$cycle,
      I64Attr:$unit_id, I64Attr:$accumulator_address,
      I64Attr:$output_stream_base, BoolAttr:$clear,
      OptionalAttr<StrAttr>:$data_format);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def VxmOp : FTLPU_Schedule_Op<"vxm", [MemoryEffects<[MemWrite]>]> {
    let summary = "Cycle-accurate instruction for one VXM ALU queue";
    let arguments = (ins AnyRankedTensor:$gate, AnyRankedTensor:$up,
        I64Attr:$cycle, I64Attr:$queue, StrAttr:$opcode,
        OptionalAttr<I64Attr>:$chain_depth,
        StrAttr:$lhs_kind, I64Attr:$lhs_index, F32Attr:$lhs_immediate,
      StrAttr:$rhs_kind, I64Attr:$rhs_index, F32Attr:$rhs_immediate,
      StrAttr:$cast_target, I64Attr:$output_stream,
      I64Attr:$repeat_count, I64Attr:$repeat_interval,
      StrAttr:$input_hemisphere, StrAttr:$output_hemisphere,
      OptionalAttr<I64Attr>:$scale_binding,
      OptionalAttr<BoolAttr>:$accumulator_reset,
      OptionalAttr<BoolAttr>:$accumulator_write,
      OptionalAttr<BoolAttr>:$accumulator_emit,
      OptionalAttr<BoolAttr>:$local_scalar_write);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$gate `,` $up attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def MxmAccumulateOp : FTLPU_Schedule_Op<"mxm_accumulate"> {
  let summary = "Commit an MXM result into its local accumulator or output stream";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$cycle,
      I64Attr:$unit_id, I64Attr:$stream_base, I64Attr:$stream_count,
      I64Attr:$accumulator_address, I64Attr:$accumulator_stride,
      StrAttr:$destination, I64Attr:$repeat_count, I64Attr:$repeat_interval,
      OptionalAttr<StrAttr>:$accumulator_output_format,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval,
      OptionalAttr<I64Attr>:$wave_accumulator_address_stride);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def MemWriteOp : FTLPU_Schedule_Op<"mem_write"> {
  let summary = "Issue a cycle-accurate MEM write stream";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$cycle, I64Attr:$duration,
      I64Attr:$stream_base, I64Attr:$stream_count, I64Attr:$register_id,
      StrAttr:$direction, DictionaryAttr:$address, DictionaryAttr:$placement,
      I64Attr:$bytes,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval,
      OptionalAttr<I64Attr>:$wave_address_stride);
  let results = (outs AnyRankedTensor:$output);
  let builders = [OpBuilder<(ins "::mlir::Value":$input, "int64_t":$cycle,
      "int64_t":$duration, "int64_t":$stream_base, "int64_t":$stream_count,
      "int64_t":$register_id, "::mlir::StringAttr":$direction,
      "::mlir::DictionaryAttr":$address, "::mlir::DictionaryAttr":$placement,
      "int64_t":$bytes), [{
    $_state.addOperands(input); $_state.addTypes(input.getType());
    $_state.addAttribute("cycle", $_builder.getI64IntegerAttr(cycle));
    $_state.addAttribute("duration", $_builder.getI64IntegerAttr(duration));
    $_state.addAttribute("stream_base", $_builder.getI64IntegerAttr(stream_base));
    $_state.addAttribute("stream_count", $_builder.getI64IntegerAttr(stream_count));
    $_state.addAttribute("register_id", $_builder.getI64IntegerAttr(register_id));
    $_state.addAttribute("direction", direction); $_state.addAttribute("address", address);
    $_state.addAttribute("placement", placement);
    $_state.addAttribute("bytes", $_builder.getI64IntegerAttr(bytes));
  }] >];
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def BindingOp : FTLPU_Schedule_Op<"binding"> {
  let summary = "Runtime-visible tensor or internal constant binding";
  let arguments = (ins Variadic<AnyRankedTensor>:$source,
      I64Attr:$index, StrAttr:$access, StrAttr:$role,
      OptionalAttr<StrAttr>:$name, OptionalAttr<I64Attr>:$ready_cycle,
      OptionalAttr<StrAttr>:$initializer,
      OptionalAttr<DictionaryAttr>:$initializer_config,
      I64Attr:$bytes, DictionaryAttr:$placement);
  let results = (outs AnyRankedTensor:$value);
  let assemblyFormat = "$source attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TimelineOp : FTLPU_Schedule_Op<"timeline"> {
  let summary = "Named cycle interval retained for schedule inspection";
  let arguments = (ins StrAttr:$name, I64Attr:$start, I64Attr:$end);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def SxmOp : FTLPU_Schedule_Op<"sxm"> {
  let summary = "Cycle-accurate SXM transpose or permute instruction";
  let arguments = (ins I64Attr:$cycle, I64Attr:$hemisphere, StrAttr:$opcode,
      ArrayAttr:$source_streams, ArrayAttr:$destination_streams,
      ArrayAttr:$permute_map, StrAttr:$weight_layout,
      OptionalAttr<I64Attr>:$output_row,
      OptionalAttr<I64Attr>:$input_row,
      OptionalAttr<I64Attr>:$output_tile,
      OptionalAttr<I64Attr>:$repeat_count,
      OptionalAttr<I64Attr>:$repeat_interval);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}
