module {
  func.func @rmsnorm_128x576(
      %x: tensor<128x576xf16>,
      %weight: tensor<576xf16>) -> tensor<128x576xf16> {
    %x_f32 = stablehlo.convert %x :
        (tensor<128x576xf16>) -> tensor<128x576xf32>
    %weight_f32 = stablehlo.convert %weight :
        (tensor<576xf16>) -> tensor<576xf32>
    %square = stablehlo.multiply %x_f32, %x_f32 :
        tensor<128x576xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = "stablehlo.reduce"(%square, %zero) ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %value = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %value : tensor<f32>
    }) {dimensions = array<i64: 1>} :
        (tensor<128x576xf32>, tensor<f32>) -> tensor<128xf32>
    %hidden = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %hidden_broadcast = stablehlo.broadcast_in_dim %hidden, dims = [] :
        (tensor<f32>) -> tensor<128xf32>
    %mean = stablehlo.divide %sum, %hidden_broadcast : tensor<128xf32>
    %epsilon = stablehlo.constant dense<1.000000e-05> : tensor<f32>
    %epsilon_broadcast = stablehlo.broadcast_in_dim %epsilon, dims = [] :
        (tensor<f32>) -> tensor<128xf32>
    %variance = stablehlo.add %mean, %epsilon_broadcast : tensor<128xf32>
    %inverse_rms = stablehlo.rsqrt %variance : tensor<128xf32>
    %inverse_rms_broadcast = stablehlo.broadcast_in_dim %inverse_rms, dims = [0] :
        (tensor<128xf32>) -> tensor<128x576xf32>
    %normalized = stablehlo.multiply %x_f32, %inverse_rms_broadcast :
        tensor<128x576xf32>
    %weight_broadcast = stablehlo.broadcast_in_dim %weight_f32, dims = [1] :
        (tensor<576xf32>) -> tensor<128x576xf32>
    %scaled = stablehlo.multiply %normalized, %weight_broadcast :
        tensor<128x576xf32>
    %result = stablehlo.convert %scaled :
        (tensor<128x576xf32>) -> tensor<128x576xf16>
    return %result : tensor<128x576xf16>
  }
}
