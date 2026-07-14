module {
  func.func @main(
      %x: tensor<160x320xi8>,
      %gate_w: tensor<320x320xi8>,
      %up_w: tensor<320x320xi8>) -> tensor<160x320xi8> {
    %gate = stablehlo.dot_general %x, %gate_w,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<160x320xi8>, tensor<320x320xi8>) -> tensor<160x320xi32>
    %up = stablehlo.dot_general %x, %up_w,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<160x320xi8>, tensor<320x320xi8>) -> tensor<160x320xi32>
    %gate_f = stablehlo.convert %gate : (tensor<160x320xi32>) -> tensor<160x320xf32>
    %up_f = stablehlo.convert %up : (tensor<160x320xi32>) -> tensor<160x320xf32>
    %sigmoid = stablehlo.logistic %gate_f : tensor<160x320xf32>
    %gated = stablehlo.multiply %gate_f, %sigmoid : tensor<160x320xf32>
    %swiglu_f = stablehlo.multiply %gated, %up_f : tensor<160x320xf32>
    %result = stablehlo.convert %swiglu_f : (tensor<160x320xf32>) -> tensor<160x320xi8>
    return %result : tensor<160x320xi8>
  }
}
