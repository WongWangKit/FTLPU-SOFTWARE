module {
  func.func @smollm2_135m_ffn(
      %x: tensor<32x576xf16>,
      %gate_w: tensor<576x1536xi8>,
      %up_w: tensor<576x1536xi8>,
      %down_w: tensor<1536x576xi8>) -> tensor<32x576xf16> {
    %x_f = stablehlo.convert %x : (tensor<32x576xf16>) -> tensor<32x576xf32>
    %gate_w_f = stablehlo.convert %gate_w : (tensor<576x1536xi8>) -> tensor<576x1536xf32>
    %up_w_f = stablehlo.convert %up_w : (tensor<576x1536xi8>) -> tensor<576x1536xf32>
    %down_w_f = stablehlo.convert %down_w : (tensor<1536x576xi8>) -> tensor<1536x576xf32>
    %gate = stablehlo.dot_general %x_f, %gate_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<32x576xf32>, tensor<576x1536xf32>) -> tensor<32x1536xf32>
    %up = stablehlo.dot_general %x_f, %up_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<32x576xf32>, tensor<576x1536xf32>) -> tensor<32x1536xf32>
    %sigmoid = stablehlo.logistic %gate : tensor<32x1536xf32>
    %silu = stablehlo.multiply %gate, %sigmoid : tensor<32x1536xf32>
    %hidden = stablehlo.multiply %silu, %up : tensor<32x1536xf32>
    %down = stablehlo.dot_general %hidden, %down_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<32x1536xf32>, tensor<1536x576xf32>) -> tensor<32x576xf32>
    %result = stablehlo.convert %down : (tensor<32x576xf32>) -> tensor<32x576xf16>
    return %result : tensor<32x576xf16>
  }
}
