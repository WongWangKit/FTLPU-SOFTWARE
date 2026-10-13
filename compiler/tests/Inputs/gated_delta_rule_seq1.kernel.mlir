module {
  func.func @gated_delta_rule_seq1(
      %query: tensor<1x64xbf16>,
      %key: tensor<1x64xbf16>,
      %value: tensor<1x64xbf16>,
      %decay: tensor<1x2xf32>,
      %beta: tensor<1x2xf32>,
      %state: tensor<64x32xf32>)
      -> (tensor<1x64xbf16>, tensor<64x32xf32>) {
    %output, %final_state = ftlpu.kernel.gated_delta_rule
        %query, %key, %value, %decay, %beta, %state {
      heads = 2 : i64,
      key_dim = 32 : i64,
      value_dim = 32 : i64
    } : (tensor<1x64xbf16>, tensor<1x64xbf16>, tensor<1x64xbf16>,
         tensor<1x2xf32>, tensor<1x2xf32>, tensor<64x32xf32>)
        -> (tensor<1x64xbf16>, tensor<64x32xf32>)
    return %output, %final_state
        : tensor<1x64xbf16>, tensor<64x32xf32>
  }
}
