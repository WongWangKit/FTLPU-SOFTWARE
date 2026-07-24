module {
  func.func @smollm2_attention(
      %input: tensor<128x576xf16>,
      %query_weight: tensor<576x576xi8>,
      %key_weight: tensor<576x192xi8>,
      %value_weight: tensor<576x192xi8>,
      %output_weight: tensor<576x576xi8>) -> tensor<128x576xf16> {
    %query_weight_f16 = stablehlo.convert %query_weight :
        (tensor<576x576xi8>) -> tensor<576x576xf16>
    %key_weight_f16 = stablehlo.convert %key_weight :
        (tensor<576x192xi8>) -> tensor<576x192xf16>
    %value_weight_f16 = stablehlo.convert %value_weight :
        (tensor<576x192xi8>) -> tensor<576x192xf16>
    %output_weight_f16 = stablehlo.convert %output_weight :
        (tensor<576x576xi8>) -> tensor<576x576xf16>

    %query_2d = stablehlo.dot_general %input, %query_weight_f16,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<128x576xf16>, tensor<576x576xf16>) -> tensor<128x576xf16>
    %key_2d = stablehlo.dot_general %input, %key_weight_f16,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<128x576xf16>, tensor<576x192xf16>) -> tensor<128x192xf16>
    %value_2d = stablehlo.dot_general %input, %value_weight_f16,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<128x576xf16>, tensor<576x192xf16>) -> tensor<128x192xf16>

    %query_heads = stablehlo.reshape %query_2d :
        (tensor<128x576xf16>) -> tensor<128x9x64xf16>
    %key_heads = stablehlo.reshape %key_2d :
        (tensor<128x192xf16>) -> tensor<128x3x64xf16>
    %value_heads = stablehlo.reshape %value_2d :
        (tensor<128x192xf16>) -> tensor<128x3x64xf16>

    %theta_f32 = stablehlo.constant dense<1.000000e+05> : tensor<f32>
    %frequency_index = stablehlo.iota dim = 0 : tensor<32xf32>
    %frequency_scale = stablehlo.constant dense<-3.125000e-02> : tensor<f32>
    %frequency_scale_32 = stablehlo.broadcast_in_dim %frequency_scale, dims = [] :
        (tensor<f32>) -> tensor<32xf32>
    %frequency_exponent = stablehlo.multiply %frequency_index, %frequency_scale_32 :
        tensor<32xf32>
    %theta_32 = stablehlo.broadcast_in_dim %theta_f32, dims = [] :
        (tensor<f32>) -> tensor<32xf32>
    %inverse_frequency = stablehlo.power %theta_32, %frequency_exponent :
        tensor<32xf32>
    %position = stablehlo.iota dim = 0 : tensor<128xf32>
    %position_column = stablehlo.reshape %position :
        (tensor<128xf32>) -> tensor<128x1xf32>
    %frequency_row = stablehlo.reshape %inverse_frequency :
        (tensor<32xf32>) -> tensor<1x32xf32>
    %position_matrix = stablehlo.broadcast_in_dim %position_column, dims = [0, 1] :
        (tensor<128x1xf32>) -> tensor<128x32xf32>
    %frequency_matrix = stablehlo.broadcast_in_dim %frequency_row, dims = [0, 1] :
        (tensor<1x32xf32>) -> tensor<128x32xf32>
    %angle = stablehlo.multiply %position_matrix, %frequency_matrix :
        tensor<128x32xf32>
    %cos = stablehlo.cosine %angle : tensor<128x32xf32>
    %sin = stablehlo.sine %angle : tensor<128x32xf32>
    %cos_base = stablehlo.reshape %cos :
        (tensor<128x32xf32>) -> tensor<128x1x32xf32>
    %sin_base = stablehlo.reshape %sin :
        (tensor<128x32xf32>) -> tensor<128x1x32xf32>
    %cos_q = stablehlo.broadcast_in_dim %cos_base, dims = [0, 1, 2] :
        (tensor<128x1x32xf32>) -> tensor<128x9x32xf32>
    %sin_q = stablehlo.broadcast_in_dim %sin_base, dims = [0, 1, 2] :
        (tensor<128x1x32xf32>) -> tensor<128x9x32xf32>
    %cos_k = stablehlo.broadcast_in_dim %cos_base, dims = [0, 1, 2] :
        (tensor<128x1x32xf32>) -> tensor<128x3x32xf32>
    %sin_k = stablehlo.broadcast_in_dim %sin_base, dims = [0, 1, 2] :
        (tensor<128x1x32xf32>) -> tensor<128x3x32xf32>

    %query_f32 = stablehlo.convert %query_heads :
        (tensor<128x9x64xf16>) -> tensor<128x9x64xf32>
    %key_f32 = stablehlo.convert %key_heads :
        (tensor<128x3x64xf16>) -> tensor<128x3x64xf32>
    %query_pairs = stablehlo.reshape %query_f32 :
        (tensor<128x9x64xf32>) -> tensor<128x9x32x2xf32>
    %key_pairs = stablehlo.reshape %key_f32 :
        (tensor<128x3x64xf32>) -> tensor<128x3x32x2xf32>
    %query_even_4d = "stablehlo.slice"(%query_pairs) {
      start_indices = array<i64: 0, 0, 0, 0>,
      limit_indices = array<i64: 128, 9, 32, 1>,
      strides = array<i64: 1, 1, 1, 1>
    } : (tensor<128x9x32x2xf32>) -> tensor<128x9x32x1xf32>
    %query_odd_4d = "stablehlo.slice"(%query_pairs) {
      start_indices = array<i64: 0, 0, 0, 1>,
      limit_indices = array<i64: 128, 9, 32, 2>,
      strides = array<i64: 1, 1, 1, 1>
    } : (tensor<128x9x32x2xf32>) -> tensor<128x9x32x1xf32>
    %key_even_4d = "stablehlo.slice"(%key_pairs) {
      start_indices = array<i64: 0, 0, 0, 0>,
      limit_indices = array<i64: 128, 3, 32, 1>,
      strides = array<i64: 1, 1, 1, 1>
    } : (tensor<128x3x32x2xf32>) -> tensor<128x3x32x1xf32>
    %key_odd_4d = "stablehlo.slice"(%key_pairs) {
      start_indices = array<i64: 0, 0, 0, 1>,
      limit_indices = array<i64: 128, 3, 32, 2>,
      strides = array<i64: 1, 1, 1, 1>
    } : (tensor<128x3x32x2xf32>) -> tensor<128x3x32x1xf32>
    %query_even = stablehlo.reshape %query_even_4d :
        (tensor<128x9x32x1xf32>) -> tensor<128x9x32xf32>
    %query_odd = stablehlo.reshape %query_odd_4d :
        (tensor<128x9x32x1xf32>) -> tensor<128x9x32xf32>
    %key_even = stablehlo.reshape %key_even_4d :
        (tensor<128x3x32x1xf32>) -> tensor<128x3x32xf32>
    %key_odd = stablehlo.reshape %key_odd_4d :
        (tensor<128x3x32x1xf32>) -> tensor<128x3x32xf32>

    %query_even_cos = stablehlo.multiply %query_even, %cos_q : tensor<128x9x32xf32>
    %query_odd_sin = stablehlo.multiply %query_odd, %sin_q : tensor<128x9x32xf32>
    %query_odd_cos = stablehlo.multiply %query_odd, %cos_q : tensor<128x9x32xf32>
    %query_even_sin = stablehlo.multiply %query_even, %sin_q : tensor<128x9x32xf32>
    %query_rotated_even = stablehlo.subtract %query_even_cos, %query_odd_sin :
        tensor<128x9x32xf32>
    %query_rotated_odd = stablehlo.add %query_odd_cos, %query_even_sin :
        tensor<128x9x32xf32>
    %query_rotated_even_4d = stablehlo.reshape %query_rotated_even :
        (tensor<128x9x32xf32>) -> tensor<128x9x32x1xf32>
    %query_rotated_odd_4d = stablehlo.reshape %query_rotated_odd :
        (tensor<128x9x32xf32>) -> tensor<128x9x32x1xf32>
    %query_rope_pairs = "stablehlo.concatenate"(
        %query_rotated_even_4d, %query_rotated_odd_4d) {dimension = 3 : i64} :
        (tensor<128x9x32x1xf32>, tensor<128x9x32x1xf32>) -> tensor<128x9x32x2xf32>
    %query_rope_f32 = stablehlo.reshape %query_rope_pairs :
        (tensor<128x9x32x2xf32>) -> tensor<128x9x64xf32>
    %query_rope = stablehlo.convert %query_rope_f32 :
        (tensor<128x9x64xf32>) -> tensor<128x9x64xf16>

    %key_even_cos = stablehlo.multiply %key_even, %cos_k : tensor<128x3x32xf32>
    %key_odd_sin = stablehlo.multiply %key_odd, %sin_k : tensor<128x3x32xf32>
    %key_odd_cos = stablehlo.multiply %key_odd, %cos_k : tensor<128x3x32xf32>
    %key_even_sin = stablehlo.multiply %key_even, %sin_k : tensor<128x3x32xf32>
    %key_rotated_even = stablehlo.subtract %key_even_cos, %key_odd_sin :
        tensor<128x3x32xf32>
    %key_rotated_odd = stablehlo.add %key_odd_cos, %key_even_sin :
        tensor<128x3x32xf32>
    %key_rotated_even_4d = stablehlo.reshape %key_rotated_even :
        (tensor<128x3x32xf32>) -> tensor<128x3x32x1xf32>
    %key_rotated_odd_4d = stablehlo.reshape %key_rotated_odd :
        (tensor<128x3x32xf32>) -> tensor<128x3x32x1xf32>
    %key_rope_pairs = "stablehlo.concatenate"(
        %key_rotated_even_4d, %key_rotated_odd_4d) {dimension = 3 : i64} :
        (tensor<128x3x32x1xf32>, tensor<128x3x32x1xf32>) -> tensor<128x3x32x2xf32>
    %key_rope_f32 = stablehlo.reshape %key_rope_pairs :
        (tensor<128x3x32x2xf32>) -> tensor<128x3x64xf32>
    %key_rope_kv = stablehlo.convert %key_rope_f32 :
        (tensor<128x3x64xf32>) -> tensor<128x3x64xf16>

    %key_grouped = stablehlo.broadcast_in_dim %key_rope_kv, dims = [0, 1, 3] :
        (tensor<128x3x64xf16>) -> tensor<128x3x3x64xf16>
    %value_grouped = stablehlo.broadcast_in_dim %value_heads, dims = [0, 1, 3] :
        (tensor<128x3x64xf16>) -> tensor<128x3x3x64xf16>
    %key_rope = stablehlo.reshape %key_grouped :
        (tensor<128x3x3x64xf16>) -> tensor<128x9x64xf16>
    %value_gqa = stablehlo.reshape %value_grouped :
        (tensor<128x3x3x64xf16>) -> tensor<128x9x64xf16>

    %query_bhsd = stablehlo.transpose %query_rope, dims = [1, 0, 2] :
        (tensor<128x9x64xf16>) -> tensor<9x128x64xf16>
    %key_bhsd = stablehlo.transpose %key_rope, dims = [1, 0, 2] :
        (tensor<128x9x64xf16>) -> tensor<9x128x64xf16>
    %value_bhsd = stablehlo.transpose %value_gqa, dims = [1, 0, 2] :
        (tensor<128x9x64xf16>) -> tensor<9x128x64xf16>

    %scores = stablehlo.dot_general %query_bhsd, %key_bhsd,
        batching_dims = [0] x [0], contracting_dims = [2] x [2],
        precision = [] :
        (tensor<9x128x64xf16>, tensor<9x128x64xf16>) -> tensor<9x128x128xf16>
    %scale = stablehlo.constant dense<1.250000e-01> : tensor<f16>
    %scale_broadcast = stablehlo.broadcast_in_dim %scale, dims = [] :
        (tensor<f16>) -> tensor<9x128x128xf16>
    %scaled_scores = stablehlo.multiply %scores, %scale_broadcast :
        tensor<9x128x128xf16>

    %query_index = stablehlo.iota dim = 1 : tensor<9x128x128xi32>
    %key_index = stablehlo.iota dim = 2 : tensor<9x128x128xi32>
    %causal = stablehlo.compare GE, %query_index, %key_index, SIGNED :
        (tensor<9x128x128xi32>, tensor<9x128x128xi32>) -> tensor<9x128x128xi1>
    %negative = stablehlo.constant dense<0xFC00> : tensor<f16>
    %negative_broadcast = stablehlo.broadcast_in_dim %negative, dims = [] :
        (tensor<f16>) -> tensor<9x128x128xf16>
    %masked_scores = stablehlo.select %causal, %scaled_scores, %negative_broadcast :
        tensor<9x128x128xi1>, tensor<9x128x128xf16>

    %negative_init = stablehlo.constant dense<0xFC00> : tensor<f16>
    %row_max = "stablehlo.reduce"(%masked_scores, %negative_init) ({
      ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
        %max = stablehlo.maximum %lhs, %rhs : tensor<f16>
        stablehlo.return %max : tensor<f16>
    }) {dimensions = array<i64: 2>} :
        (tensor<9x128x128xf16>, tensor<f16>) -> tensor<9x128xf16>
    %row_max_broadcast = stablehlo.broadcast_in_dim %row_max, dims = [0, 1] :
        (tensor<9x128xf16>) -> tensor<9x128x128xf16>
    %centered = stablehlo.subtract %masked_scores, %row_max_broadcast :
        tensor<9x128x128xf16>
    %exp = stablehlo.exponential %centered : tensor<9x128x128xf16>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %row_sum = "stablehlo.reduce"(%exp, %zero) ({
      ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
        %sum = stablehlo.add %lhs, %rhs : tensor<f16>
        stablehlo.return %sum : tensor<f16>
    }) {dimensions = array<i64: 2>} :
        (tensor<9x128x128xf16>, tensor<f16>) -> tensor<9x128xf16>
    %row_sum_broadcast = stablehlo.broadcast_in_dim %row_sum, dims = [0, 1] :
        (tensor<9x128xf16>) -> tensor<9x128x128xf16>
    %probability = stablehlo.divide %exp, %row_sum_broadcast :
        tensor<9x128x128xf16>

    %context_bhsd = stablehlo.dot_general %probability, %value_bhsd,
        batching_dims = [0] x [0], contracting_dims = [2] x [1],
        precision = [] :
        (tensor<9x128x128xf16>, tensor<9x128x64xf16>) -> tensor<9x128x64xf16>
    %context_shd = stablehlo.transpose %context_bhsd, dims = [1, 0, 2] :
        (tensor<9x128x64xf16>) -> tensor<128x9x64xf16>
    %context = stablehlo.reshape %context_shd :
        (tensor<128x9x64xf16>) -> tensor<128x576xf16>
    %result = stablehlo.dot_general %context, %output_weight_f16,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<128x576xf16>, tensor<576x576xf16>) -> tensor<128x576xf16>
    return %result : tensor<128x576xf16>
  }
}
