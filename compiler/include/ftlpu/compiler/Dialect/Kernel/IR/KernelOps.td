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
                       StrAttr:$unit, F32Attr:$rhs_scale);
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
      $_state.addAttribute("rhs_scale", $_builder.getF32FloatAttr(1.0f));
    }]>
  ];
  let hasVerifier = 1;
}

def BlockFp8QuantizeOp : FTLPU_Kernel_Op<"block_fp8_quantize", [Pure]> {
  let summary = "Dynamic BF16 to E4M3FN quantization per row and group of 32";
  let description = [{
    For each contiguous group of 32, take amax = max(max(abs(x)), 1e-4).
    The UE8M0 scale is the smallest power of two >= amax * float32(1/448).
    Divide by that scale, round to nearest even and saturate to finite E4M3FN.
    Both the quantized values and the runtime scales are observable.
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$values, AnyRankedTensor:$scales);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def BlockFp8MatmulOp : FTLPU_Kernel_Op<"block_fp8_matmul", [Pure]> {
  let summary = "E4M3FN projection with per-row K32 and per-weight-tile E8M0 scales";
  let description = [{
    Values are stored as X[M,K] and W[N,K]. Each output is a sum in increasing
    K32 group order. Accumulate a 32-term dot in FP32 and apply the two
    corresponding power-of-two scales. Accumulate the groups in FP32 and
    round once to BF16. Exact BF16 expansion can precede each dot when
    equivalent under the target's floating-point and reduction rules.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$lhs_scales,
                       AnyRankedTensor:$weights, AnyRankedTensor:$weight_scales);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $lhs_scales `,` $weights `,` $weight_scales attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def BlockFp8ProjectionOp : FTLPU_Kernel_Op<"block_fp8_projection", [Pure]> {
  let summary = "Block-scaled FP8 weight projection with optional activation QDQ";
  let description = [{
    BF16 X[M,K], checkpoint E4M3FN W[N,K], and constant UE8M0 S[N/32,K/32].
    By default quantize X dynamically per row and K32 group with the
    block_fp8_qdq semantics. With quantize_activation=false, consume the
    original BF16 X directly. Dequantize weights at IW, compute each K32 partial in FP32,
    sum partials in K order in FP32, and round the final result to BF16.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weights,
                       AnyRankedTensor:$weight_scales,
                       OptionalAttr<BoolAttr>:$quantize_activation);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $weights `,` $weight_scales attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def BlockFp8ScaleOp : FTLPU_Kernel_Op<"block_fp8_scale", [Pure]> {
  let summary = "Compute the decoded E8M0 scale for each group of 32 finite BF16 values";
  let description = [{
    For each row return 2^ceil(log2(max(1e-4, max(abs(x))) / 448)).
    The result is the scale represented in FP32, used for E4M3FN activation
    quantization. Input and result shapes are [groups,32] and [groups,1].
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def BlockFp8QdqOp : FTLPU_Kernel_Op<"block_fp8_qdq", [Pure]> {
  let summary = "Apply group-of-32 E4M3FN dynamic quantize/dequantize to BF16";
  let description = [{
    Scale each group of 32 finite BF16 inputs with the block_fp8_scale rule,
    quantize each normalized value to E4M3FN with round-to-nearest-even,
    then multiply by the group scale and round to BF16.
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def BiasAddOp : FTLPU_Kernel_Op<"bias_add", [Pure]> {
  let summary = "Broadcast a rank-1 projection bias over matrix rows";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$bias);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $bias attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def HcMixesOp : FTLPU_Kernel_Op<"hc_mixes", [Pure]> {
  let summary = "Hyper-connection coefficients from a BF16 residual stream";
  let description = [{
    X[T,H,D], W[H*(H+2),H*D], scale[3], base[H*(H+2)] are BF16.
    Normalize the FP32 projection by the flattened input RMS. Round this
    normalized projection to BF16, then compute sigmoid gates and the
    specified full Sinkhorn iteration in FP32. Return BF16 pre[T,H],
    post[T,H], comb[T,H,H] and normalized projection[T,H*(H+2)].
    pre belongs to the next sublayer; the current hc_pre receives its own
    incoming coefficient tensor. No model computation runs on the host.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight,
      AnyRankedTensor:$scale, AnyRankedTensor:$base,
      I64Attr:$iterations, F32Attr:$epsilon, F32Attr:$norm_epsilon);
  let results = (outs AnyRankedTensor:$pre, AnyRankedTensor:$post,
      AnyRankedTensor:$comb, AnyRankedTensor:$projection);
  let hasVerifier = 1;
}

def HcPreOp : FTLPU_Kernel_Op<"hc_pre", [Pure]> {
  let summary = "BF16 residual stream weighted collapse with FP32 accumulation";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$pre);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def HcPostOp : FTLPU_Kernel_Op<"hc_post", [Pure]> {
  let summary = "BF16 sublayer output expansion and residual mixing";
  let description = [{
    Y[T,D], residual[T,H,D], post[T,H], comb[T,H,H] -> output[T,H,D].
    output[t,j,d] = post[t,j]*Y[t,d] + sum_i comb[t,i,j]*residual[t,i,d].
    Products and sums are FP32; the final result is BF16 RNE.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$residual,
      AnyRankedTensor:$post, AnyRankedTensor:$comb);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def ReshapeOp : FTLPU_Kernel_Op<"reshape", [Pure]> {
  let summary = "Logical tensor reshape preserved in LPU Kernel IR";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def IndexedViewOp : FTLPU_Kernel_Op<"indexed_view", [Pure]> {
  let summary = "Logical gather/concatenation view, with no physical storage";
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
      DenseI64ArrayAttr:$sources, DenseI64ArrayAttr:$indices);
  let results = (outs AnyRankedTensor:$result);
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

def FlashAttentionOp : FTLPU_Kernel_Op<"flash_attention", [Pure]> {
  let summary = "Grouped scaled dot-product attention, tiled with online softmax";
  let arguments = (ins AnyRankedTensor:$query, AnyRankedTensor:$key,
      AnyRankedTensor:$value, F32Attr:$scale, BoolAttr:$causal,
      I64Attr:$query_position);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$query `,` $key `,` $value attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def DenseAttentionOp : FTLPU_Kernel_Op<"dense_attention", [Pure]> {
  let summary = "Grouped scaled dot-product attention through the native dense schedule";
  let arguments = (ins AnyRankedTensor:$query, AnyRankedTensor:$key,
      AnyRankedTensor:$value, F32Attr:$scale, BoolAttr:$causal,
      I64Attr:$query_position);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$query `,` $key `,` $value attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def AttentionCoreOp : FTLPU_Kernel_Op<"attention_core", [Pure]> {
  let summary = "Single-request native dense attention with flattened head dimensions";
  let arguments = (ins AnyRankedTensor:$query, AnyRankedTensor:$key,
      AnyRankedTensor:$value, Optional<AnyRankedTensor>:$sink_values, DictionaryAttr:$config,
      OptionalAttr<DictionaryAttr>:$memory_plan);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$query `,` $key `,` $value (`sink` $sink_values^)? attr-dict `:` functional-type(operands, results)";
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

def L2NormOp : FTLPU_Kernel_Op<"l2_norm", [Pure]> {
  let summary = "scale * input * rsqrt(sum(input squared) + epsilon)";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$axis,
      F32Attr:$epsilon, F32Attr:$scale);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def RmsNormOp : FTLPU_Kernel_Op<"rms_norm", [Pure]> {
  let summary = "Root-mean-square normalization over one tensor axis";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight,
      I64Attr:$axis, F32Attr:$epsilon);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SigmoidMultiplyOp : FTLPU_Kernel_Op<"sigmoid_multiply", [Pure]> {
  let summary = "Elementwise input times sigmoid(gate)";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$gate);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $gate attr-dict `:` functional-type(operands, results)";
}

def SwishOp : FTLPU_Kernel_Op<"swish", [Pure]> {
  let summary = "Elementwise x times sigmoid(x) activation";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def ScaledMatvecOp : FTLPU_Kernel_Op<"scaled_matvec", [Pure]> {
  let summary = "Grouped matrix scaling with retained matrix and vector contraction";
  let description = [{
    Matrix is [groups * columns, reduction], vector is [groups, reduction],
    and scale is [1, scale_groups]. Returns scale[group_offset + group] * matrix and its
    contraction with vector over reduction. Both results are observable.
    This storage orientation permits streaming scaled columns into MXM IW.
    Optional epilogue operands (value, coefficient) return
    coefficient[group_offset + group] * (value - product).
    A second optional epilogue result retains the rank-one update:
    updated[group, column, reduction] = scaled + correction * vector.
    With a third epilogue operand (readout vector), return a third epilogue
    result contracting the updated matrix with that vector over reduction.
  }];
  let arguments = (ins AnyRankedTensor:$matrix, AnyRankedTensor:$scale,
                       AnyRankedTensor:$vector, Variadic<AnyRankedTensor>:$epilogue,
                       DefaultValuedAttr<I64Attr, "0">:$group_offset);
  let results = (outs AnyRankedTensor:$scaled, AnyRankedTensor:$product,
                      Variadic<AnyRankedTensor>:$correction);
  let assemblyFormat = "$matrix `,` $scale `,` $vector (`epilogue` `(` $epilogue^ `)`)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def VectorAluOp : FTLPU_Kernel_Op<"vector_alu", [Pure]> {
  let summary = "Generic scalar-broadcast vector arithmetic and row reduction";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
      StrAttr:$kind, BoolAttr:$use_immediate, F32Attr:$immediate,
      BoolAttr:$reduce);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def DepthwiseConv1DOp : FTLPU_Kernel_Op<"depthwise_conv1d", [Pure]> {
  let summary = "Channel-wise causal one-dimensional convolution";
  let description = [{
    A model-independent depthwise convolution over a [tokens, channels]
    tensor.  Each channel owns one kernel and no cross-channel reduction is
    performed.  Causal mode supplies zeros before the first token.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight,
      I64Attr:$kernel_size, BoolAttr:$causal, StrAttr:$activation);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $weight attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def GatedDeltaRuleOp : FTLPU_Kernel_Op<"gated_delta_rule", [Pure]> {
  let summary = "Stateful recurrent gated delta-rule scan";
  let description = [{
    A model-independent recurrent linear-attention primitive. Query and key
    are normalized before this operation; decay contains the multiplicative
    state coefficient. The operation returns both token outputs and the final
    FP32 state so prefill and decode use the same state ABI.
  }];
  let arguments = (ins AnyRankedTensor:$query, AnyRankedTensor:$key,
      AnyRankedTensor:$value, AnyRankedTensor:$decay,
      AnyRankedTensor:$beta, AnyRankedTensor:$initial_state,
      I64Attr:$heads, I64Attr:$key_dim, I64Attr:$value_dim);
  let results = (outs AnyRankedTensor:$output,
      AnyRankedTensor:$final_state);
  let assemblyFormat = "$query `,` $key `,` $value `,` $decay `,` $beta `,` $initial_state attr-dict `:` functional-type(operands, results)";
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
