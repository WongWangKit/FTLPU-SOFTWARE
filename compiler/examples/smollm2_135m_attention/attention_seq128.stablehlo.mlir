module {
  func.func @smollm2_attention(
      %input: tensor<128x576xf16>,
      %query_weight: tensor<576x576xi8>,
      %key_weight: tensor<576x192xi8>,
      %value_weight: tensor<576x192xi8>,
      %output_weight: tensor<576x576xi8>) -> tensor<128x576xf16> {
    %result = "stablehlo.custom_call"(%input, %query_weight, %key_weight,
        %value_weight, %output_weight) {
      call_target_name = "ftlpu.attention",
      query_heads = 9 : i64,
      kv_heads = 3 : i64,
      head_dim = 64 : i64,
      rope_theta = 1.000000e+05 : f32,
      causal = true
    } : (tensor<128x576xf16>, tensor<576x576xi8>, tensor<576x192xi8>,
         tensor<576x192xi8>, tensor<576x576xi8>) -> tensor<128x576xf16>
    return %result : tensor<128x576xf16>
  }
}
