// Minimal frontend MLIR example for the FTLPU/IREE adapter.
// This is a common-IR staging file, not yet a scheduled LPU program.

module {
  func.func @main(%lhs: tensor<320x320xi8>, %rhs: tensor<320x320xi8>) -> tensor<320x320xi32> {
    %result = "stablehlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<320x320xi8>, tensor<320x320xi8>) -> tensor<320x320xi32>
    return %result : tensor<320x320xi32>
  }
}
