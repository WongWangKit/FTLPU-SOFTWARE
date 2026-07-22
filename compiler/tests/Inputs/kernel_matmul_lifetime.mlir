module {
  func.func @main(
      %a: tensor<4x4xi8>, %b: tensor<4x4xi8>, %c: tensor<4x4xi8>,
      %d: tensor<4x4xi8>, %e: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %dead = "ftlpu.kernel.matmul"(%a, %b) {k = 4 : i64, m = 4 : i64, n = 4 : i64, unit = "MXM"} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %middle = "ftlpu.kernel.matmul"(%c, %d) {k = 4 : i64, m = 4 : i64, n = 4 : i64, unit = "MXM"} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %result = "ftlpu.kernel.matmul"(%a, %b) {k = 4 : i64, m = 4 : i64, n = 4 : i64, unit = "MXM"} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    return %result : tensor<4x4xi32>
  }
}
