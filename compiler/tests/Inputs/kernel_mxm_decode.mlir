module attributes {ftlpu.icu_compression = "macro"} {
  func.func @main(%activation: tensor<1x32xbf16>,
                  %weight: tensor<32x64xi8>) -> tensor<1x64xbf16> {
    %result = "ftlpu.kernel.matmul"(%activation, %weight) {
      k = 32 : i64, m = 1 : i64, n = 64 : i64,
      unit = "MXM", rhs_scale = 0.125 : f32
    } : (tensor<1x32xbf16>, tensor<32x64xi8>) -> tensor<1x64xbf16>
    return %result : tensor<1x64xbf16>
  }
}
