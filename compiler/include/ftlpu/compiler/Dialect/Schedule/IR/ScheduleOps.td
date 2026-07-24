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
      StrAttr:$accumulator_destination);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MxmIssueOp : FTLPU_Schedule_Op<"mxm_issue", [MemoryEffects<[MemWrite]>]> {
  let summary = "Cycle-accurate issue to a physical MXM unit";
  let arguments = (ins I64Attr:$cycle, I64Attr:$unit_id, StrAttr:$opcode,
      I64Attr:$weight_buffer, I64Attr:$weight_column, I64Attr:$activation_stream_base,
      I64Attr:$output_stream_base, I64Attr:$repeat_count,
      I64Attr:$repeat_interval);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MemReadOp : FTLPU_Schedule_Op<"mem_read"> {
  let summary = "Issue a cycle-accurate MEM read stream";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$cycle, I64Attr:$duration,
      I64Attr:$stream_base, I64Attr:$stream_count, I64Attr:$register_id,
      StrAttr:$direction, StrAttr:$role, DictionaryAttr:$address,
      DictionaryAttr:$placement, I64Attr:$bytes);
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
      I64Attr:$weight_buffer);
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
      I64Attr:$unit_id,
      I64Attr:$m, I64Attr:$n, I64Attr:$k);
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
    $_state.addAttribute("m", $_builder.getI64IntegerAttr(m));
    $_state.addAttribute("n", $_builder.getI64IntegerAttr(n));
    $_state.addAttribute("k", $_builder.getI64IntegerAttr(k));
  }] >];
  let assemblyFormat = "$activation `,` $weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def VxmOp : FTLPU_Schedule_Op<"vxm", [MemoryEffects<[MemWrite]>]> {
  let summary = "Cycle-accurate instruction for one VXM ALU queue";
  let arguments = (ins AnyRankedTensor:$gate, AnyRankedTensor:$up,
      I64Attr:$cycle, I64Attr:$queue, StrAttr:$opcode,
      StrAttr:$lhs_kind, I64Attr:$lhs_index, F32Attr:$lhs_immediate,
      StrAttr:$rhs_kind, I64Attr:$rhs_index, F32Attr:$rhs_immediate,
      StrAttr:$cast_target, I64Attr:$output_stream,
      I64Attr:$repeat_count, I64Attr:$repeat_interval,
      StrAttr:$input_hemisphere, StrAttr:$output_hemisphere);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$gate `,` $up attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def MemAccumulateOp : FTLPU_Schedule_Op<"mem_accumulate"> {
  let summary = "Accumulate an MXM result stream in MEM or forward the final sum";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$cycle,
      I64Attr:$stream_base, I64Attr:$stream_count, DictionaryAttr:$address,
      DictionaryAttr:$placement, StrAttr:$hemisphere, StrAttr:$destination,
      I64Attr:$repeat_count, I64Attr:$repeat_interval, I64Attr:$address_stride);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def MemWriteOp : FTLPU_Schedule_Op<"mem_write"> {
  let summary = "Issue a cycle-accurate MEM write stream";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$cycle, I64Attr:$duration,
      I64Attr:$stream_base, I64Attr:$stream_count, I64Attr:$register_id,
      StrAttr:$direction, DictionaryAttr:$address, DictionaryAttr:$placement,
      I64Attr:$bytes);
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

def AttentionOp : FTLPU_Schedule_Op<"attention", [Pure]> {
  let summary = "Cycle-bounded grouped-query attention schedule";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$query_weight,
      AnyRankedTensor:$key_weight, AnyRankedTensor:$value_weight,
      AnyRankedTensor:$output_weight,
      I64Attr:$seq_len, I64Attr:$hidden, I64Attr:$query_heads,
      I64Attr:$kv_heads, I64Attr:$head_dim, F32Attr:$rope_theta,
      BoolAttr:$causal, DictionaryAttr:$memory_plan, ArrayAttr:$routes,
      ArrayAttr:$phases, ArrayAttr:$work_waves);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $query_weight `,` $key_weight `,` $value_weight `,` $output_weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SxmOp : FTLPU_Schedule_Op<"sxm"> {
  let summary = "Cycle-accurate SXM transpose or permute instruction";
  let arguments = (ins I64Attr:$cycle, I64Attr:$hemisphere, StrAttr:$opcode,
      ArrayAttr:$source_streams, ArrayAttr:$destination_streams,
      ArrayAttr:$permute_map, StrAttr:$weight_layout);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}
