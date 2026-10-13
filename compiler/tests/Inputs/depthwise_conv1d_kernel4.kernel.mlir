module {
  func.func @depthwise_conv1d_kernel4(
      %input: tensor<32x32xbf16>,
      %weight: tensor<32x4xbf16>) -> tensor<32x32xbf16> {
    %0 = ftlpu.kernel.depthwise_conv1d %input, %weight {
      activation = "silu",
      causal = true,
      kernel_size = 4 : i64
    } : (tensor<32x32xbf16>, tensor<32x4xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}
