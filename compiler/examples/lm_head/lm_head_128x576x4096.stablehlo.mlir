module {
  func.func @lm_head_shard(
      %hidden: tensor<128x576xf16>,
      %weight: tensor<576x4096xf16>) -> tensor<128x4096xf16> {
    %logits = stablehlo.dot_general %hidden, %weight,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<128x576xf16>, tensor<576x4096xf16>) -> tensor<128x4096xf16>
    return %logits : tensor<128x4096xf16>
  }
}
