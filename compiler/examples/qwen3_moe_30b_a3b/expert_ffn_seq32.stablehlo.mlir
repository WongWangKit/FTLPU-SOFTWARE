module {
  func.func @qwen3_moe_30b_a3b_expert_ffn_seq32(
      %x: tensor<32x2048xbf16>,
      %gate_w: tensor<2048x768xi8>,
      %up_w: tensor<2048x768xi8>,
      %down_w: tensor<768x2048xi8>) -> tensor<32x2048xbf16> {
    %x_f = stablehlo.convert %x : (tensor<32x2048xbf16>) -> tensor<32x2048xf32>
    %gate_w_f = stablehlo.convert %gate_w : (tensor<2048x768xi8>) -> tensor<2048x768xf32>
    %up_w_f = stablehlo.convert %up_w : (tensor<2048x768xi8>) -> tensor<2048x768xf32>
    %down_w_f = stablehlo.convert %down_w : (tensor<768x2048xi8>) -> tensor<768x2048xf32>
    %gate = stablehlo.dot_general %x_f, %gate_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<32x2048xf32>, tensor<2048x768xf32>) -> tensor<32x768xf32>
    %up = stablehlo.dot_general %x_f, %up_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<32x2048xf32>, tensor<2048x768xf32>) -> tensor<32x768xf32>
    %sigmoid = stablehlo.logistic %gate : tensor<32x768xf32>
    %silu = stablehlo.multiply %gate, %sigmoid : tensor<32x768xf32>
    %hidden = stablehlo.multiply %silu, %up : tensor<32x768xf32>
    %down = stablehlo.dot_general %hidden, %down_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<32x768xf32>, tensor<768x2048xf32>) -> tensor<32x2048xf32>
    %result = stablehlo.convert %down : (tensor<32x2048xf32>) -> tensor<32x2048xbf16>
    return %result : tensor<32x2048xbf16>
  }
}
