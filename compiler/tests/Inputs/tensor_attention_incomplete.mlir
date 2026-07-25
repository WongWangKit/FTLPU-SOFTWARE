module {
  func.func @incomplete_attention(
      %input: tensor<1x1xf16>,
      %weight: tensor<1x1xi8>) -> tensor<1x1xf16> {
    %0 = ftlpu.tensor.projection_task %input, %weight {
      config = {
        causal = true,
        head_dim = 64 : i64,
        hidden = 576 : i64,
        kv_heads = 3 : i64,
        query_heads = 9 : i64,
        rope_theta = 1.000000e+05 : f32,
        seq_len = 128 : i64
      },
      kind = "output",
      memory_plan = {}
    } : (tensor<1x1xf16>, tensor<1x1xi8>) -> tensor<1x1xf16>
    return %0 : tensor<1x1xf16>
  }
}
