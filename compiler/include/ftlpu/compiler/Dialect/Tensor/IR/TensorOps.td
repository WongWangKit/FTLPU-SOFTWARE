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
                       StrAttr:$unit, F32Attr:$rhs_scale,
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
                   "::mlir::StringAttr":$unit, "::mlir::FloatAttr":$rhs_scale,
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
      $_state.addAttribute("rhs_scale", rhs_scale);
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
      StrAttr:$kind, ArrayAttr:$lhs_allocations, ArrayAttr:$rhs_allocations,
      ArrayAttr:$result_allocations, DictionaryAttr:$config);
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

def ProjectionTaskOp : FTLPU_Tensor_Op<"projection_task", [Pure]> {
  let summary = "Matrix projection with an assigned physical buffer sub-plan";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight,
      StrAttr:$kind, DictionaryAttr:$config, DictionaryAttr:$memory_plan);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def RopeTaskOp : FTLPU_Tensor_Op<"rope_task", [Pure]> {
  let summary = "Rotary-position embedding with an assigned physical sub-plan";
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$kind,
      DictionaryAttr:$config, DictionaryAttr:$memory_plan);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def BatchMatmulTaskOp : FTLPU_Tensor_Op<"batch_matmul_task", [Pure]> {
  let summary = "Batched matrix multiplication with a physical buffer sub-plan";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
      StrAttr:$kind, DictionaryAttr:$config, DictionaryAttr:$memory_plan);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SoftmaxTaskOp : FTLPU_Tensor_Op<"softmax_task", [Pure]> {
  let summary = "Numerically stable row-wise softmax with a physical sub-plan";
  let arguments = (ins AnyRankedTensor:$input, DictionaryAttr:$config,
      DictionaryAttr:$memory_plan);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def RmsNormTaskOp : FTLPU_Tensor_Op<"rms_norm_task", [Pure]> {
  let summary = "RMS normalization with assigned input, scale, scratch, and result storage";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight,
      I64Attr:$axis, F32Attr:$epsilon, ArrayAttr:$input_allocations,
      ArrayAttr:$weight_allocations, ArrayAttr:$scratch_allocations,
      ArrayAttr:$result_allocations, DictionaryAttr:$config);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TransposeTaskOp : FTLPU_Tensor_Op<"transpose_task", [Pure]> {
  let summary = "SXM transpose or permutation with a physical sub-plan";
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$kind,
      DictionaryAttr:$config, DictionaryAttr:$memory_plan);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
