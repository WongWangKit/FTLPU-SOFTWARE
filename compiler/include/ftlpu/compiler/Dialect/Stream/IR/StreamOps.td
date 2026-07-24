include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def StreamDialect : Dialect {
  let name = "ftlpu.stream";
  let cppNamespace = "::ftlpu::compiler::stream";
}

class FTLPU_Stream_Op<string mnemonic, list<Trait> traits = []>
    : Op<StreamDialect, mnemonic, traits>;

def DequantizeOp : FTLPU_Stream_Op<"dequantize", [Pure]> {
  let summary = "Convert eight W8 streams into sixteen FP16 MXM load streams";
  let arguments = (ins AnyRankedTensor:$input, F32Attr:$scale,
      I64Attr:$input_stream_base, I64Attr:$output_stream_base,
      StrAttr:$input_hemisphere, StrAttr:$output_hemisphere);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def RouteOp : FTLPU_Stream_Op<"route"> {
  let summary = "Route a logical tensor between LPU endpoints";
  let arguments = (ins AnyRankedTensor:$input,
                       I64Attr:$stream_base,
                       I64Attr:$stream_count,
                       I64Attr:$register_id,
                       StrAttr:$direction,
                       StrAttr:$source,
                       StrAttr:$destination,
                       I64Attr:$source_unit_id,
                       I64Attr:$destination_unit_id,
                       DictionaryAttr:$address,
                       DictionaryAttr:$placement,
                       I64Attr:$bytes,
                       I64Attr:$transport_latency);
  let results = (outs AnyRankedTensor:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input,
                   "int64_t":$stream_base, "int64_t":$stream_count,
                   "int64_t":$register_id,
                   "::mlir::StringAttr":$direction,
                   "::mlir::StringAttr":$source,
                   "::mlir::StringAttr":$destination,
                   "int64_t":$source_unit_id,
                   "int64_t":$destination_unit_id,
                   "::mlir::DictionaryAttr":$address,
                   "::mlir::DictionaryAttr":$placement,
                   "int64_t":$bytes, "int64_t":$transport_latency), [{
      $_state.addOperands(input);
      $_state.addTypes(input.getType());
      $_state.addAttribute("stream_base", $_builder.getI64IntegerAttr(stream_base));
      $_state.addAttribute("stream_count", $_builder.getI64IntegerAttr(stream_count));
      $_state.addAttribute("register_id", $_builder.getI64IntegerAttr(register_id));
      $_state.addAttribute("direction", direction);
      $_state.addAttribute("source", source);
      $_state.addAttribute("destination", destination);
      $_state.addAttribute("source_unit_id", $_builder.getI64IntegerAttr(source_unit_id));
      $_state.addAttribute("destination_unit_id", $_builder.getI64IntegerAttr(destination_unit_id));
      $_state.addAttribute("address", address);
      $_state.addAttribute("placement", placement);
      $_state.addAttribute("bytes", $_builder.getI64IntegerAttr(bytes));
      $_state.addAttribute("transport_latency", $_builder.getI64IntegerAttr(transport_latency));
    }]>
  ];
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def MatmulOp : FTLPU_Stream_Op<"matmul", [Pure]> {
  let summary = "MXM operation connected to routed tensor operands";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       I64Attr:$m, I64Attr:$n, I64Attr:$k,
                       I64Attr:$unit_id, I64Attr:$weight_buffer);
  let results = (outs AnyRankedTensor:$result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs,
                   "::mlir::Type":$result_type, "int64_t":$m,
                   "int64_t":$n, "int64_t":$k, "int64_t":$unit_id,
                   "int64_t":$weight_buffer), [{
      $_state.addOperands({lhs, rhs});
      $_state.addTypes(result_type);
      $_state.addAttribute("m", $_builder.getI64IntegerAttr(m));
      $_state.addAttribute("n", $_builder.getI64IntegerAttr(n));
      $_state.addAttribute("k", $_builder.getI64IntegerAttr(k));
      $_state.addAttribute("unit_id", $_builder.getI64IntegerAttr(unit_id));
      $_state.addAttribute("weight_buffer", $_builder.getI64IntegerAttr(weight_buffer));
    }]>
  ];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def MatmulTaskOp
    : FTLPU_Stream_Op<"matmul_task", [Pure, AttrSizedOperandSegments]> {
  let summary = "One or more MXM matmuls connected to allocated stream routes";
  let description = [{
    A logical matrix multiplication task may use one MXM or a set of parallel
    MXMs. Each lhs/rhs operand is an explicitly routed stream value. Unit,
    weight-buffer, and result-stream arrays have one entry per parallel MXM.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$lhs,
      Variadic<AnyRankedTensor>:$rhs, I64Attr:$m, I64Attr:$n, I64Attr:$k,
      ArrayAttr:$unit_ids, ArrayAttr:$weight_buffers,
      ArrayAttr:$result_stream_bases, ArrayAttr:$result_stream_counts,
      ArrayAttr:$result_allocations, DictionaryAttr:$config);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "`[` $lhs `]` `,` `[` $rhs `]` attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SwishTaskOp : FTLPU_Stream_Op<"swish_task", [Pure]> {
  let summary = "VXM swish over one MXM result stream";
  let arguments = (ins AnyRankedTensor:$input,
      I64Attr:$input_stream_base, I64Attr:$output_stream_base,
      DictionaryAttr:$config);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def ElementwiseTaskOp : FTLPU_Stream_Op<"elementwise_task", [Pure]> {
  let summary = "VXM binary elementwise task over functional-unit streams";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
      StrAttr:$kind, ArrayAttr:$input_stream_bases,
      I64Attr:$output_stream_base, ArrayAttr:$result_allocations,
      DictionaryAttr:$config);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SwigluOp : FTLPU_Stream_Op<"swiglu", [Pure]> {
  let summary = "Dual-MXM result streams consumed by the VXM SwiGLU pipeline";
  let arguments = (ins AnyRankedTensor:$activation, AnyRankedTensor:$gate_weight,
      AnyRankedTensor:$up_weight, I64Attr:$m, I64Attr:$n, I64Attr:$k,
      I64Attr:$gate_unit_id, I64Attr:$up_unit_id,
      I64Attr:$gate_weight_buffer, I64Attr:$up_weight_buffer,
      I64Attr:$gate_output_stream_base, I64Attr:$up_output_stream_base,
      I64Attr:$vxm_output_stream, I64Attr:$output_register_id,
      I64Attr:$output_transport_latency,
      F32Attr:$gate_scale, F32Attr:$up_scale, F32Attr:$output_scale,
      I64Attr:$output_zero_point, DictionaryAttr:$result_address,
      DictionaryAttr:$result_placement, I64Attr:$result_bytes);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$activation `,` $gate_weight `,` $up_weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def AttentionOp : FTLPU_Stream_Op<"attention", [Pure]> {
  let summary = "Grouped-query attention with allocated inter-unit stream routes";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$query_weight,
      AnyRankedTensor:$key_weight, AnyRankedTensor:$value_weight,
      AnyRankedTensor:$output_weight,
      I64Attr:$seq_len, I64Attr:$hidden, I64Attr:$query_heads,
      I64Attr:$kv_heads, I64Attr:$head_dim, F32Attr:$rope_theta,
      BoolAttr:$causal, DictionaryAttr:$memory_plan, ArrayAttr:$routes);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $query_weight `,` $key_weight `,` $value_weight `,` $output_weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
