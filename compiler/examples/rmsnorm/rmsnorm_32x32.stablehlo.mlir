module {
  func.func @rmsnorm_32x32(
      %x: tensor<32x32xbf16>,
      %weight: tensor<32xbf16>) -> tensor<32x32xbf16> {
    %x_f32 = stablehlo.convert %x :
        (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %weight_f32 = stablehlo.convert %weight :
        (tensor<32xbf16>) -> tensor<32xf32>
    %square = stablehlo.multiply %x_f32, %x_f32 : tensor<32x32xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = "stablehlo.reduce"(%square, %zero) ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %value = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %value : tensor<f32>
    }) {dimensions = array<i64: 1>} :
        (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %hidden = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %hidden_broadcast = stablehlo.broadcast_in_dim %hidden, dims = [] :
        (tensor<f32>) -> tensor<32xf32>
    %mean = stablehlo.divide %sum, %hidden_broadcast : tensor<32xf32>
    %epsilon = stablehlo.constant dense<1.000000e-05> : tensor<f32>
    %epsilon_broadcast = stablehlo.broadcast_in_dim %epsilon, dims = [] :
        (tensor<f32>) -> tensor<32xf32>
    %variance = stablehlo.add %mean, %epsilon_broadcast : tensor<32xf32>
    %inverse_rms = stablehlo.rsqrt %variance : tensor<32xf32>
    %inverse_rms_broadcast = stablehlo.broadcast_in_dim %inverse_rms,
        dims = [0] : (tensor<32xf32>) -> tensor<32x32xf32>
    %normalized = stablehlo.multiply %x_f32, %inverse_rms_broadcast :
        tensor<32x32xf32>
    %weight_broadcast = stablehlo.broadcast_in_dim %weight_f32, dims = [1] :
        (tensor<32xf32>) -> tensor<32x32xf32>
    %scaled = stablehlo.multiply %normalized, %weight_broadcast :
        tensor<32x32xf32>
    %result = stablehlo.convert %scaled :
        (tensor<32x32xf32>) -> tensor<32x32xbf16>
    return %result : tensor<32x32xbf16>
  }
}
