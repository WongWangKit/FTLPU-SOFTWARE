module {
  func.func @w8a16_ffn_128x64x128x64(
      %x: tensor<128x64xf16>,
      %gate_w: tensor<64x128xi8>,
      %up_w: tensor<64x128xi8>,
      %down_w: tensor<128x64xi8>) -> tensor<128x64xf16> {
    %x_f = stablehlo.convert %x : (tensor<128x64xf16>) -> tensor<128x64xf32>
    %gate_w_f = stablehlo.convert %gate_w : (tensor<64x128xi8>) -> tensor<64x128xf32>
    %up_w_f = stablehlo.convert %up_w : (tensor<64x128xi8>) -> tensor<64x128xf32>
    %down_w_f = stablehlo.convert %down_w : (tensor<128x64xi8>) -> tensor<128x64xf32>
    %gate = stablehlo.dot_general %x_f, %gate_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    %up = stablehlo.dot_general %x_f, %up_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    %sigmoid = stablehlo.logistic %gate : tensor<128x128xf32>
    %silu = stablehlo.multiply %gate, %sigmoid : tensor<128x128xf32>
    %hidden = stablehlo.multiply %silu, %up : tensor<128x128xf32>
    %down = stablehlo.dot_general %hidden, %down_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %result = stablehlo.convert %down : (tensor<128x64xf32>) -> tensor<128x64xf16>
    return %result : tensor<128x64xf16>
  }
}
