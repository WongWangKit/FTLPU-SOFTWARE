module {
  func.func @smollm2_135m_lm_head_shard(
      %hidden: tensor<32x576xbf16>,
      %weight: tensor<576x4096xi8>) -> tensor<32x4096xbf16> {
    %weight_bf16 = stablehlo.convert %weight :
        (tensor<576x4096xi8>) -> tensor<576x4096xbf16>
    %scale = stablehlo.constant dense<1.000000e-02> : tensor<bf16>
    %scale_matrix = stablehlo.broadcast_in_dim %scale, dims = [] :
        (tensor<bf16>) -> tensor<576x4096xbf16>
    %scaled_weight = stablehlo.multiply %weight_bf16, %scale_matrix :
        tensor<576x4096xbf16>
    %logits = stablehlo.dot_general %hidden, %scaled_weight,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<32x576xbf16>, tensor<576x4096xbf16>)
        -> tensor<32x4096xbf16>
    return %logits : tensor<32x4096xbf16>
  }
}
