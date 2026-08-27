include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CommandDialect : Dialect {
  let name = "ftlpu.command";
  let cppNamespace = "::ftlpu::compiler::command";
}

class FTLPU_Command_Op<string mnemonic, list<Trait> traits = []>
    : Op<CommandDialect, mnemonic, traits>;

def BindingOp : FTLPU_Command_Op<"binding"> {
  let summary = "Runtime-visible tensor binding and physical MEM placement";
  let arguments = (ins I64Attr:$index, StrAttr:$access, StrAttr:$role,
      StrAttr:$name, I64Attr:$ready_cycle,
      StrAttr:$initializer, DictionaryAttr:$initializer_config,
      ArrayAttr:$shape, StrAttr:$element_type, I64Attr:$bytes,
      DictionaryAttr:$placement);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def TimelineOp : FTLPU_Command_Op<"timeline"> {
  let summary = "Named cycle interval retained in the executable";
  let arguments = (ins StrAttr:$name, I64Attr:$start, I64Attr:$end);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def WeightPageOp : FTLPU_Command_Op<"weight_page"> {
  let summary = "Runtime-visible residency interval for one logical weight page";
  let arguments = (ins I64Attr:$binding_index, I64Attr:$page_index,
      I64Attr:$bank, I64Attr:$ready_cycle, I64Attr:$release_cycle);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def LoopOp : FTLPU_Command_Op<"loop"> {
  let summary = "ICU control command replaying a prior instruction window";
  let arguments = (ins I64Attr:$cycle, StrAttr:$queue_kind,
      I64Attr:$queue, I64Attr:$window_size, I64Attr:$count,
      I64Attr:$interval, I64Attr:$address_stride);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MemOp : FTLPU_Command_Op<"mem", [MemoryEffects<[MemWrite]>]> {
  let summary = "Compressed command for one ICU MEM queue";
  let arguments = (ins I64Attr:$cycle, I64Attr:$queue, StrAttr:$opcode,
      I64Attr:$address, I64Attr:$packed_stream, I64Attr:$repeat_count,
      I64Attr:$repeat_interval, I64Attr:$address_stride,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval,
      OptionalAttr<I64Attr>:$wave_address_stride,
      OptionalAttr<I64Attr>:$address_binding);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MemBundleOp : FTLPU_Command_Op<"mem_bundle", [MemoryEffects<[MemWrite]>]> {
  let summary = "One logical MEM transfer issued to several physical queues";
  let arguments = (ins ArrayAttr:$cycles, ArrayAttr:$queues,
      StrAttr:$opcode, ArrayAttr:$addresses, ArrayAttr:$packed_streams,
      I64Attr:$repeat_count, I64Attr:$repeat_interval,
      I64Attr:$address_stride,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval,
      OptionalAttr<I64Attr>:$wave_address_stride,
      OptionalAttr<I64Attr>:$address_binding);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MxmOp : FTLPU_Command_Op<"mxm", [MemoryEffects<[MemWrite]>]> {
  let summary = "Compressed command for one ICU MXM queue";
  let arguments = (ins I64Attr:$cycle, I64Attr:$queue, StrAttr:$opcode,
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

def MxmDequantOp : FTLPU_Command_Op<"mxm_dequant",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Compressed scale command for one ICU MXM dequant queue";
  let arguments = (ins I64Attr:$cycle, I64Attr:$queue, F32Attr:$scale,
      I64Attr:$repeat_count, I64Attr:$repeat_interval,
      OptionalAttr<I64Attr>:$scale_binding,
      OptionalAttr<I64Attr>:$wave_count,
      OptionalAttr<I64Attr>:$wave_interval);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def VxmOp : FTLPU_Command_Op<"vxm", [MemoryEffects<[MemWrite]>]> {
    let summary = "Compressed command for one ICU VXM ALU queue";
    let arguments = (ins I64Attr:$cycle, I64Attr:$queue, StrAttr:$opcode,
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
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def SxmOp : FTLPU_Command_Op<"sxm", [MemoryEffects<[MemWrite]>]> {
  let summary = "Variable-payload command for one ICU SXM queue";
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
