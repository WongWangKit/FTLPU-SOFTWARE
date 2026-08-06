module {
  func.func @block8_linear_projection(
      %activation: tensor<32x64xbf16>,
      %weight: tensor<64x64xi8>) -> tensor<32x64xbf16> {
    %weight_bf16 = stablehlo.convert %weight :
        (tensor<64x64xi8>) -> tensor<64x64xbf16>
    %scale = stablehlo.constant dense<6.250000e-02> : tensor<bf16>
    %scale_matrix = stablehlo.broadcast_in_dim %scale, dims = [] :
        (tensor<bf16>) -> tensor<64x64xbf16>
    %dequantized = stablehlo.multiply %weight_bf16, %scale_matrix :
        tensor<64x64xbf16>
    %result = stablehlo.dot_general %activation, %dequantized,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<32x64xbf16>, tensor<64x64xbf16>)
        -> tensor<32x64xbf16>
    return %result : tensor<32x64xbf16>
  }
}
