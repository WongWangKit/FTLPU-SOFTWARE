include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TensorDialect : Dialect {
  let name = "ftlpu.tensor";
  let cppNamespace = "::ftlpu::compiler::tensor";
}

class FTLPU_Tensor_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorDialect, mnemonic, traits>;

def MatmulOp : FTLPU_Tensor_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication with assigned LPU MEM storage";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       I64Attr:$m, I64Attr:$n, I64Attr:$k,
                       StrAttr:$unit,
                       DictionaryAttr:$lhs_address,
                       DictionaryAttr:$lhs_placement,
                       DictionaryAttr:$rhs_address,
                       DictionaryAttr:$rhs_placement,
                       DictionaryAttr:$result_address,
                       DictionaryAttr:$result_placement,
                       I64Attr:$lhs_bytes,
                       I64Attr:$rhs_bytes,
                       I64Attr:$result_bytes);
  let results = (outs AnyRankedTensor:$result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs,
                   "::mlir::Type":$result_type, "int64_t":$m,
                   "int64_t":$n, "int64_t":$k,
                   "::mlir::StringAttr":$unit,
                   "::mlir::DictionaryAttr":$lhs_address,
                   "::mlir::DictionaryAttr":$lhs_placement,
                   "::mlir::DictionaryAttr":$rhs_address,
                   "::mlir::DictionaryAttr":$rhs_placement,
                   "::mlir::DictionaryAttr":$result_address,
                   "::mlir::DictionaryAttr":$result_placement,
                   "int64_t":$lhs_bytes, "int64_t":$rhs_bytes,
                   "int64_t":$result_bytes), [{
      $_state.addOperands({lhs, rhs});
      $_state.addTypes(result_type);
      $_state.addAttribute("m", $_builder.getI64IntegerAttr(m));
      $_state.addAttribute("n", $_builder.getI64IntegerAttr(n));
      $_state.addAttribute("k", $_builder.getI64IntegerAttr(k));
      $_state.addAttribute("unit", unit);
      $_state.addAttribute("lhs_address", lhs_address);
      $_state.addAttribute("lhs_placement", lhs_placement);
      $_state.addAttribute("rhs_address", rhs_address);
      $_state.addAttribute("rhs_placement", rhs_placement);
      $_state.addAttribute("result_address", result_address);
      $_state.addAttribute("result_placement", result_placement);
      $_state.addAttribute("lhs_bytes", $_builder.getI64IntegerAttr(lhs_bytes));
      $_state.addAttribute("rhs_bytes", $_builder.getI64IntegerAttr(rhs_bytes));
      $_state.addAttribute("result_bytes", $_builder.getI64IntegerAttr(result_bytes));
    }]>
  ];
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def MatmulTaskOp : FTLPU_Tensor_Op<"matmul_task", [Pure]> {
  let summary = "Matrix multiplication with explicit physical buffer plans";
  let description = [{
    Represents a hardware-mapped matrix multiplication after tensor memory
    planning. An empty result allocation list means that the result remains
    transient in a functional-unit accumulator or stream.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
      I64Attr:$m, I64Attr:$n, I64Attr:$k,
      ArrayAttr:$lhs_allocations, ArrayAttr:$rhs_allocations,
      ArrayAttr:$result_allocations, DictionaryAttr:$config);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SwishTaskOp : FTLPU_Tensor_Op<"swish_task", [Pure]> {
  let summary = "Swish with an explicit physical result plan";
  let arguments = (ins AnyRankedTensor:$input,
      ArrayAttr:$result_allocations, DictionaryAttr:$config);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def ElementwiseTaskOp : FTLPU_Tensor_Op<"elementwise_task", [Pure]> {
  let summary = "Binary elementwise computation with a physical result plan";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
      StrAttr:$kind, ArrayAttr:$result_allocations, DictionaryAttr:$config);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SwigluOp : FTLPU_Tensor_Op<"swiglu", [Pure]> {
  let summary = "Dual-MXM SwiGLU with assigned input, weight, and output storage";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$gate_weight,
      AnyRankedTensor:$up_weight, I64Attr:$m, I64Attr:$n, I64Attr:$k,
      F32Attr:$gate_scale, F32Attr:$up_scale, F32Attr:$output_scale,
      I64Attr:$output_zero_point,
      DictionaryAttr:$input_address, DictionaryAttr:$input_placement,
      DictionaryAttr:$gate_weight_address, DictionaryAttr:$gate_weight_placement,
      DictionaryAttr:$up_weight_address, DictionaryAttr:$up_weight_placement,
      DictionaryAttr:$result_address, DictionaryAttr:$result_placement,
      I64Attr:$input_bytes, I64Attr:$gate_weight_bytes,
      I64Attr:$up_weight_bytes, I64Attr:$result_bytes);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $gate_weight `,` $up_weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def FfnOp : FTLPU_Tensor_Op<"ffn", [Pure]> {
  let summary = "Complete FFN with physical storage for two hidden passes";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$gate_weight,
      AnyRankedTensor:$up_weight, AnyRankedTensor:$down_weight,
      I64Attr:$m, I64Attr:$k, I64Attr:$hidden, I64Attr:$n,
      F32Attr:$gate_scale, F32Attr:$up_scale, F32Attr:$hidden_scale,
      I64Attr:$hidden_zero_point, F32Attr:$down_lhs_scale,
      F32Attr:$down_rhs_scale, F32Attr:$output_scale,
      I64Attr:$output_zero_point,
      DictionaryAttr:$input_address, DictionaryAttr:$input_placement,
      DictionaryAttr:$gate_weight_address, DictionaryAttr:$gate_weight_placement,
      DictionaryAttr:$up_weight_address, DictionaryAttr:$up_weight_placement,
      DictionaryAttr:$down_weight_address, DictionaryAttr:$down_weight_placement,
      DictionaryAttr:$hidden0_address, DictionaryAttr:$hidden0_placement,
      DictionaryAttr:$hidden1_address, DictionaryAttr:$hidden1_placement,
      DictionaryAttr:$result_address, DictionaryAttr:$result_placement,
      I64Attr:$input_bytes, I64Attr:$gate_weight_bytes,
      I64Attr:$up_weight_bytes, I64Attr:$down_weight_bytes,
      I64Attr:$hidden_pass_bytes, I64Attr:$result_bytes);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $gate_weight `,` $up_weight `,` $down_weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def AttentionOp : FTLPU_Tensor_Op<"attention", [Pure]> {
  let summary = "Grouped-query causal attention with an assigned LPU MEM plan";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$query_weight,
      AnyRankedTensor:$key_weight, AnyRankedTensor:$value_weight,
      AnyRankedTensor:$output_weight,
      I64Attr:$seq_len, I64Attr:$hidden, I64Attr:$query_heads,
      I64Attr:$kv_heads, I64Attr:$head_dim, F32Attr:$rope_theta,
      BoolAttr:$causal, DictionaryAttr:$memory_plan);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $query_weight `,` $key_weight `,` $value_weight `,` $output_weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
