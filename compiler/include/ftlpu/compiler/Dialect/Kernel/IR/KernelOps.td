include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def KernelDialect : Dialect {
  let name = "ftlpu.kernel";
  let cppNamespace = "::ftlpu::compiler::kernel";
}

class FTLPU_Kernel_Op<string mnemonic, list<Trait> traits = []>
    : Op<KernelDialect, mnemonic, traits>;

def MatmulOp : FTLPU_Kernel_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication mapped to the LPU MXM unit";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       I64Attr:$m, I64Attr:$n, I64Attr:$k,
                       StrAttr:$unit);
  let results = (outs AnyRankedTensor:$result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs,
                   "::mlir::Type":$result_type, "int64_t":$m,
                   "int64_t":$n, "int64_t":$k), [{
      $_state.addOperands({lhs, rhs});
      $_state.addTypes(result_type);
      $_state.addAttribute("m", $_builder.getI64IntegerAttr(m));
      $_state.addAttribute("n", $_builder.getI64IntegerAttr(n));
      $_state.addAttribute("k", $_builder.getI64IntegerAttr(k));
      $_state.addAttribute("unit", $_builder.getStringAttr("MXM"));
    }]>
  ];
  let hasVerifier = 1;
}

def ReshapeOp : FTLPU_Kernel_Op<"reshape", [Pure]> {
  let summary = "Logical tensor reshape preserved in LPU Kernel IR";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TransposeOp : FTLPU_Kernel_Op<"transpose", [Pure]> {
  let summary = "Logical tensor permutation preserved for layout planning";
  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$permutation);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def RopeOp : FTLPU_Kernel_Op<"rope", [Pure]> {
  let summary = "Rotary positional embedding over one Q or K tensor";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$heads,
      I64Attr:$head_dim, F32Attr:$theta);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def GqaBroadcastOp : FTLPU_Kernel_Op<"gqa_broadcast", [Pure]> {
  let summary = "Broadcast KV heads to query heads for grouped-query attention";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$query_heads,
      I64Attr:$kv_heads);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def BatchMatmulOp : FTLPU_Kernel_Op<"batch_matmul", [Pure]> {
  let summary = "Rank-3 batched matrix multiplication";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
      BoolAttr:$transpose_rhs, StrAttr:$role);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SoftmaxOp : FTLPU_Kernel_Op<"softmax", [Pure]> {
  let summary = "Row softmax with optional causal masking";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$axis,
      F32Attr:$scale, BoolAttr:$causal);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SwishOp : FTLPU_Kernel_Op<"swish", [Pure]> {
  let summary = "Elementwise x times sigmoid(x) activation";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def ElementwiseOp : FTLPU_Kernel_Op<"elementwise", [Pure]> {
  let summary = "Binary elementwise operation";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
      StrAttr:$kind);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SwigluOp : FTLPU_Kernel_Op<"swiglu", [Pure]> {
  let summary = "Two MXM projections followed by a quantized VXM SwiGLU";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$gate_weight,
      AnyRankedTensor:$up_weight, I64Attr:$m, I64Attr:$n, I64Attr:$k,
      F32Attr:$gate_scale, F32Attr:$up_scale, F32Attr:$output_scale,
      I64Attr:$output_zero_point);
  let results = (outs AnyRankedTensor:$result);
  let builders = [OpBuilder<(ins "::mlir::Value":$input,
      "::mlir::Value":$gate_weight, "::mlir::Value":$up_weight,
      "::mlir::Type":$result_type, "int64_t":$m, "int64_t":$n,
      "int64_t":$k, "float":$gate_scale, "float":$up_scale,
      "float":$output_scale, "int64_t":$output_zero_point), [{
    $_state.addOperands({input, gate_weight, up_weight});
    $_state.addTypes(result_type);
    $_state.addAttribute("m", $_builder.getI64IntegerAttr(m));
    $_state.addAttribute("n", $_builder.getI64IntegerAttr(n));
    $_state.addAttribute("k", $_builder.getI64IntegerAttr(k));
    $_state.addAttribute("gate_scale", $_builder.getF32FloatAttr(gate_scale));
    $_state.addAttribute("up_scale", $_builder.getF32FloatAttr(up_scale));
    $_state.addAttribute("output_scale", $_builder.getF32FloatAttr(output_scale));
    $_state.addAttribute("output_zero_point", $_builder.getI64IntegerAttr(output_zero_point));
  }] >];
  let assemblyFormat = "$input `,` $gate_weight `,` $up_weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def FfnOp : FTLPU_Kernel_Op<"ffn", [Pure]> {
  let summary = "Complete gate/up SwiGLU and down-projection FFN";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$gate_weight,
      AnyRankedTensor:$up_weight, AnyRankedTensor:$down_weight,
      I64Attr:$m, I64Attr:$k, I64Attr:$hidden, I64Attr:$n,
      F32Attr:$gate_scale, F32Attr:$up_scale, F32Attr:$hidden_scale,
      I64Attr:$hidden_zero_point, F32Attr:$down_lhs_scale,
      F32Attr:$down_rhs_scale, F32Attr:$output_scale,
      I64Attr:$output_zero_point);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $gate_weight `,` $up_weight `,` $down_weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def AttentionOp : FTLPU_Kernel_Op<"attention", [Pure]> {
  let summary = "Grouped-query causal attention mapped to LPU MXM, VXM, and SXM units";
  let description = [{
    Canonical decoder attention.  The op deliberately carries architectural
    dimensions instead of a model name: query_heads may exceed kv_heads for
    grouped-query attention, and head_dim is the per-head RoPE domain.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$query_weight,
      AnyRankedTensor:$key_weight, AnyRankedTensor:$value_weight,
      AnyRankedTensor:$output_weight,
      I64Attr:$seq_len, I64Attr:$hidden, I64Attr:$query_heads,
      I64Attr:$kv_heads, I64Attr:$head_dim, F32Attr:$rope_theta,
      BoolAttr:$causal);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $query_weight `,` $key_weight `,` $value_weight `,` $output_weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
