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
      ArrayAttr:$shape, StrAttr:$element_type, I64Attr:$bytes,
      DictionaryAttr:$placement);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MemOp : FTLPU_Command_Op<"mem", [MemoryEffects<[MemWrite]>]> {
  let summary = "Compressed command for one ICU MEM queue";
  let arguments = (ins I64Attr:$cycle, I64Attr:$queue, StrAttr:$opcode,
      I64Attr:$address, I64Attr:$packed_stream, I64Attr:$repeat_count,
      I64Attr:$repeat_interval, I64Attr:$address_stride);
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
      BoolAttr:$accumulator_clear);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def VxmOp : FTLPU_Command_Op<"vxm", [MemoryEffects<[MemWrite]>]> {
  let summary = "Compressed command for one ICU VXM ALU queue";
  let arguments = (ins I64Attr:$cycle, I64Attr:$queue, StrAttr:$opcode,
      StrAttr:$lhs_kind, I64Attr:$lhs_index, F32Attr:$lhs_immediate,
      StrAttr:$rhs_kind, I64Attr:$rhs_index, F32Attr:$rhs_immediate,
      StrAttr:$cast_target, I64Attr:$output_stream,
      I64Attr:$repeat_count, I64Attr:$repeat_interval,
      StrAttr:$input_hemisphere, StrAttr:$output_hemisphere);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def SxmOp : FTLPU_Command_Op<"sxm", [MemoryEffects<[MemWrite]>]> {
  let summary = "Variable-payload command for one ICU SXM queue";
  let arguments = (ins I64Attr:$cycle, I64Attr:$hemisphere, StrAttr:$opcode,
      ArrayAttr:$source_streams, ArrayAttr:$destination_streams,
      ArrayAttr:$permute_map, StrAttr:$weight_layout);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}
