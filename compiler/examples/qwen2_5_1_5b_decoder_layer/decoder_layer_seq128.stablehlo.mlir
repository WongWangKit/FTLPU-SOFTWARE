module {
  func.func @qwen2_5_1_5b_decoder_layer_seq128(
      %x: tensor<128x1536xbf16>,
      %input_norm_weight: tensor<1536xbf16>,
      %query_weight: tensor<1536x1536xi8>,
      %key_weight: tensor<1536x256xi8>,
      %value_weight: tensor<1536x256xi8>,
      %output_weight: tensor<1536x1536xi8>,
      %post_attention_norm_weight: tensor<1536xbf16>,
      %gate_weight: tensor<1536x8960xi8>,
      %up_weight: tensor<1536x8960xi8>,
      %down_weight: tensor<8960x1536xi8>) -> tensor<128x1536xbf16> {
    %rms1_x_f32 = stablehlo.convert %x :
        (tensor<128x1536xbf16>) -> tensor<128x1536xf32>
    %rms1_weight_f32 = stablehlo.convert %input_norm_weight :
        (tensor<1536xbf16>) -> tensor<1536xf32>
    %rms1_square = stablehlo.multiply %rms1_x_f32, %rms1_x_f32 :
        tensor<128x1536xf32>
    %rms1_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %rms1_sum = "stablehlo.reduce"(%rms1_square, %rms1_zero) ({
      ^bb0(%rms1_lhs: tensor<f32>, %rms1_rhs: tensor<f32>):
        %rms1_value = stablehlo.add %rms1_lhs, %rms1_rhs : tensor<f32>
        stablehlo.return %rms1_value : tensor<f32>
    }) {dimensions = array<i64: 1>} :
        (tensor<128x1536xf32>, tensor<f32>) -> tensor<128xf32>
    %rms1_hidden = stablehlo.constant dense<1.536000e+03> : tensor<f32>
    %rms1_hidden_broadcast = stablehlo.broadcast_in_dim %rms1_hidden, dims = [] :
        (tensor<f32>) -> tensor<128xf32>
    %rms1_mean = stablehlo.divide %rms1_sum, %rms1_hidden_broadcast : tensor<128xf32>
    %rms1_epsilon = stablehlo.constant dense<1.000000e-06> : tensor<f32>
    %rms1_epsilon_broadcast = stablehlo.broadcast_in_dim %rms1_epsilon, dims = [] :
        (tensor<f32>) -> tensor<128xf32>
    %rms1_variance = stablehlo.add %rms1_mean, %rms1_epsilon_broadcast : tensor<128xf32>
    %rms1_inverse_rms = stablehlo.rsqrt %rms1_variance : tensor<128xf32>
    %rms1_inverse_rms_broadcast = stablehlo.broadcast_in_dim %rms1_inverse_rms, dims = [0] :
        (tensor<128xf32>) -> tensor<128x1536xf32>
    %rms1_normalized = stablehlo.multiply %rms1_x_f32, %rms1_inverse_rms_broadcast :
        tensor<128x1536xf32>
    %rms1_weight_broadcast = stablehlo.broadcast_in_dim %rms1_weight_f32, dims = [1] :
        (tensor<1536xf32>) -> tensor<128x1536xf32>
    %rms1_scaled = stablehlo.multiply %rms1_normalized, %rms1_weight_broadcast :
        tensor<128x1536xf32>
    %rms1_result = stablehlo.convert %rms1_scaled :
        (tensor<128x1536xf32>) -> tensor<128x1536xbf16>

    %attention_query_weight_bf16 = stablehlo.convert %query_weight :
        (tensor<1536x1536xi8>) -> tensor<1536x1536xbf16>
    %attention_key_weight_bf16 = stablehlo.convert %key_weight :
        (tensor<1536x256xi8>) -> tensor<1536x256xbf16>
    %attention_value_weight_bf16 = stablehlo.convert %value_weight :
        (tensor<1536x256xi8>) -> tensor<1536x256xbf16>
    %attention_output_weight_bf16 = stablehlo.convert %output_weight :
        (tensor<1536x1536xi8>) -> tensor<1536x1536xbf16>

    %attention_query_2d = stablehlo.dot_general %rms1_result, %attention_query_weight_bf16,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<128x1536xbf16>, tensor<1536x1536xbf16>) -> tensor<128x1536xbf16>
    %attention_key_2d = stablehlo.dot_general %rms1_result, %attention_key_weight_bf16,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<128x1536xbf16>, tensor<1536x256xbf16>) -> tensor<128x256xbf16>
    %attention_value_2d = stablehlo.dot_general %rms1_result, %attention_value_weight_bf16,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<128x1536xbf16>, tensor<1536x256xbf16>) -> tensor<128x256xbf16>

    %attention_query_heads = stablehlo.reshape %attention_query_2d :
        (tensor<128x1536xbf16>) -> tensor<128x12x128xbf16>
    %attention_key_heads = stablehlo.reshape %attention_key_2d :
        (tensor<128x256xbf16>) -> tensor<128x2x128xbf16>
    %attention_value_heads = stablehlo.reshape %attention_value_2d :
        (tensor<128x256xbf16>) -> tensor<128x2x128xbf16>

    %attention_theta_f32 = stablehlo.constant dense<1.000000e+06> : tensor<f32>
    %attention_frequency_index = stablehlo.iota dim = 0 : tensor<64xf32>
    %attention_frequency_scale = stablehlo.constant dense<-1.562500e-02> : tensor<f32>
    %attention_frequency_scale_32 = stablehlo.broadcast_in_dim %attention_frequency_scale, dims = [] :
        (tensor<f32>) -> tensor<64xf32>
    %attention_frequency_exponent = stablehlo.multiply %attention_frequency_index, %attention_frequency_scale_32 :
        tensor<64xf32>
    %attention_theta_32 = stablehlo.broadcast_in_dim %attention_theta_f32, dims = [] :
        (tensor<f32>) -> tensor<64xf32>
    %attention_inverse_frequency = stablehlo.power %attention_theta_32, %attention_frequency_exponent :
        tensor<64xf32>
    %attention_position = stablehlo.iota dim = 0 : tensor<128xf32>
    %attention_position_column = stablehlo.reshape %attention_position :
        (tensor<128xf32>) -> tensor<128x1xf32>
    %attention_frequency_row = stablehlo.reshape %attention_inverse_frequency :
        (tensor<64xf32>) -> tensor<1x64xf32>
    %attention_position_matrix = stablehlo.broadcast_in_dim %attention_position_column, dims = [0, 1] :
        (tensor<128x1xf32>) -> tensor<128x64xf32>
    %attention_frequency_matrix = stablehlo.broadcast_in_dim %attention_frequency_row, dims = [0, 1] :
        (tensor<1x64xf32>) -> tensor<128x64xf32>
    %attention_angle = stablehlo.multiply %attention_position_matrix, %attention_frequency_matrix :
        tensor<128x64xf32>
    %attention_cos = stablehlo.cosine %attention_angle : tensor<128x64xf32>
    %attention_sin = stablehlo.sine %attention_angle : tensor<128x64xf32>
    %attention_cos_base = stablehlo.reshape %attention_cos :
        (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %attention_sin_base = stablehlo.reshape %attention_sin :
        (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %attention_cos_q = stablehlo.broadcast_in_dim %attention_cos_base, dims = [0, 1, 2] :
        (tensor<128x1x64xf32>) -> tensor<128x12x64xf32>
    %attention_sin_q = stablehlo.broadcast_in_dim %attention_sin_base, dims = [0, 1, 2] :
        (tensor<128x1x64xf32>) -> tensor<128x12x64xf32>
    %attention_cos_k = stablehlo.broadcast_in_dim %attention_cos_base, dims = [0, 1, 2] :
        (tensor<128x1x64xf32>) -> tensor<128x2x64xf32>
    %attention_sin_k = stablehlo.broadcast_in_dim %attention_sin_base, dims = [0, 1, 2] :
        (tensor<128x1x64xf32>) -> tensor<128x2x64xf32>

    %attention_query_f32 = stablehlo.convert %attention_query_heads :
        (tensor<128x12x128xbf16>) -> tensor<128x12x128xf32>
    %attention_key_f32 = stablehlo.convert %attention_key_heads :
        (tensor<128x2x128xbf16>) -> tensor<128x2x128xf32>
    %attention_query_pairs = stablehlo.reshape %attention_query_f32 :
        (tensor<128x12x128xf32>) -> tensor<128x12x64x2xf32>
    %attention_key_pairs = stablehlo.reshape %attention_key_f32 :
        (tensor<128x2x128xf32>) -> tensor<128x2x64x2xf32>
    %attention_query_even_4d = "stablehlo.slice"(%attention_query_pairs) {
      start_indices = array<i64: 0, 0, 0, 0>,
      limit_indices = array<i64: 128, 12, 64, 1>,
      strides = array<i64: 1, 1, 1, 1>
    } : (tensor<128x12x64x2xf32>) -> tensor<128x12x64x1xf32>
    %attention_query_odd_4d = "stablehlo.slice"(%attention_query_pairs) {
      start_indices = array<i64: 0, 0, 0, 1>,
      limit_indices = array<i64: 128, 12, 64, 2>,
      strides = array<i64: 1, 1, 1, 1>
    } : (tensor<128x12x64x2xf32>) -> tensor<128x12x64x1xf32>
    %attention_key_even_4d = "stablehlo.slice"(%attention_key_pairs) {
      start_indices = array<i64: 0, 0, 0, 0>,
      limit_indices = array<i64: 128, 2, 64, 1>,
      strides = array<i64: 1, 1, 1, 1>
    } : (tensor<128x2x64x2xf32>) -> tensor<128x2x64x1xf32>
    %attention_key_odd_4d = "stablehlo.slice"(%attention_key_pairs) {
      start_indices = array<i64: 0, 0, 0, 1>,
      limit_indices = array<i64: 128, 2, 64, 2>,
      strides = array<i64: 1, 1, 1, 1>
    } : (tensor<128x2x64x2xf32>) -> tensor<128x2x64x1xf32>
    %attention_query_even = stablehlo.reshape %attention_query_even_4d :
        (tensor<128x12x64x1xf32>) -> tensor<128x12x64xf32>
    %attention_query_odd = stablehlo.reshape %attention_query_odd_4d :
        (tensor<128x12x64x1xf32>) -> tensor<128x12x64xf32>
    %attention_key_even = stablehlo.reshape %attention_key_even_4d :
        (tensor<128x2x64x1xf32>) -> tensor<128x2x64xf32>
    %attention_key_odd = stablehlo.reshape %attention_key_odd_4d :
        (tensor<128x2x64x1xf32>) -> tensor<128x2x64xf32>

    %attention_query_even_cos = stablehlo.multiply %attention_query_even, %attention_cos_q : tensor<128x12x64xf32>
    %attention_query_odd_sin = stablehlo.multiply %attention_query_odd, %attention_sin_q : tensor<128x12x64xf32>
    %attention_query_odd_cos = stablehlo.multiply %attention_query_odd, %attention_cos_q : tensor<128x12x64xf32>
    %attention_query_even_sin = stablehlo.multiply %attention_query_even, %attention_sin_q : tensor<128x12x64xf32>
    %attention_query_rotated_even = stablehlo.subtract %attention_query_even_cos, %attention_query_odd_sin :
        tensor<128x12x64xf32>
    %attention_query_rotated_odd = stablehlo.add %attention_query_odd_cos, %attention_query_even_sin :
        tensor<128x12x64xf32>
    %attention_query_rotated_even_4d = stablehlo.reshape %attention_query_rotated_even :
        (tensor<128x12x64xf32>) -> tensor<128x12x64x1xf32>
    %attention_query_rotated_odd_4d = stablehlo.reshape %attention_query_rotated_odd :
        (tensor<128x12x64xf32>) -> tensor<128x12x64x1xf32>
    %attention_query_rope_pairs = "stablehlo.concatenate"(
        %attention_query_rotated_even_4d, %attention_query_rotated_odd_4d) {dimension = 3 : i64} :
        (tensor<128x12x64x1xf32>, tensor<128x12x64x1xf32>) -> tensor<128x12x64x2xf32>
    %attention_query_rope_f32 = stablehlo.reshape %attention_query_rope_pairs :
        (tensor<128x12x64x2xf32>) -> tensor<128x12x128xf32>
    %attention_query_rope = stablehlo.convert %attention_query_rope_f32 :
        (tensor<128x12x128xf32>) -> tensor<128x12x128xbf16>

    %attention_key_even_cos = stablehlo.multiply %attention_key_even, %attention_cos_k : tensor<128x2x64xf32>
    %attention_key_odd_sin = stablehlo.multiply %attention_key_odd, %attention_sin_k : tensor<128x2x64xf32>
    %attention_key_odd_cos = stablehlo.multiply %attention_key_odd, %attention_cos_k : tensor<128x2x64xf32>
    %attention_key_even_sin = stablehlo.multiply %attention_key_even, %attention_sin_k : tensor<128x2x64xf32>
    %attention_key_rotated_even = stablehlo.subtract %attention_key_even_cos, %attention_key_odd_sin :
        tensor<128x2x64xf32>
    %attention_key_rotated_odd = stablehlo.add %attention_key_odd_cos, %attention_key_even_sin :
        tensor<128x2x64xf32>
    %attention_key_rotated_even_4d = stablehlo.reshape %attention_key_rotated_even :
        (tensor<128x2x64xf32>) -> tensor<128x2x64x1xf32>
    %attention_key_rotated_odd_4d = stablehlo.reshape %attention_key_rotated_odd :
        (tensor<128x2x64xf32>) -> tensor<128x2x64x1xf32>
    %attention_key_rope_pairs = "stablehlo.concatenate"(
        %attention_key_rotated_even_4d, %attention_key_rotated_odd_4d) {dimension = 3 : i64} :
        (tensor<128x2x64x1xf32>, tensor<128x2x64x1xf32>) -> tensor<128x2x64x2xf32>
    %attention_key_rope_f32 = stablehlo.reshape %attention_key_rope_pairs :
        (tensor<128x2x64x2xf32>) -> tensor<128x2x128xf32>
    %attention_key_rope_kv = stablehlo.convert %attention_key_rope_f32 :
        (tensor<128x2x128xf32>) -> tensor<128x2x128xbf16>

    %attention_key_grouped = stablehlo.broadcast_in_dim %attention_key_rope_kv, dims = [0, 1, 3] :
        (tensor<128x2x128xbf16>) -> tensor<128x2x6x128xbf16>
    %attention_value_grouped = stablehlo.broadcast_in_dim %attention_value_heads, dims = [0, 1, 3] :
        (tensor<128x2x128xbf16>) -> tensor<128x2x6x128xbf16>
    %attention_key_rope = stablehlo.reshape %attention_key_grouped :
        (tensor<128x2x6x128xbf16>) -> tensor<128x12x128xbf16>
    %attention_value_gqa = stablehlo.reshape %attention_value_grouped :
        (tensor<128x2x6x128xbf16>) -> tensor<128x12x128xbf16>

    %attention_query_bhsd = stablehlo.transpose %attention_query_rope, dims = [1, 0, 2] :
        (tensor<128x12x128xbf16>) -> tensor<12x128x128xbf16>
    %attention_key_bhsd = stablehlo.transpose %attention_key_rope, dims = [1, 0, 2] :
        (tensor<128x12x128xbf16>) -> tensor<12x128x128xbf16>
    %attention_value_bhsd = stablehlo.transpose %attention_value_gqa, dims = [1, 0, 2] :
        (tensor<128x12x128xbf16>) -> tensor<12x128x128xbf16>

    %attention_scores = stablehlo.dot_general %attention_query_bhsd, %attention_key_bhsd,
        batching_dims = [0] x [0], contracting_dims = [2] x [2],
        precision = [] :
        (tensor<12x128x128xbf16>, tensor<12x128x128xbf16>) -> tensor<12x128x128xbf16>
    %attention_scale = stablehlo.constant dense<8.838835e-02> : tensor<bf16>
    %attention_scale_broadcast = stablehlo.broadcast_in_dim %attention_scale, dims = [] :
        (tensor<bf16>) -> tensor<12x128x128xbf16>
    %attention_scaled_scores = stablehlo.multiply %attention_scores, %attention_scale_broadcast :
        tensor<12x128x128xbf16>

    %attention_query_index = stablehlo.iota dim = 1 : tensor<12x128x128xi32>
    %attention_key_index = stablehlo.iota dim = 2 : tensor<12x128x128xi32>
    %attention_causal = stablehlo.compare GE, %attention_query_index, %attention_key_index, SIGNED :
        (tensor<12x128x128xi32>, tensor<12x128x128xi32>) -> tensor<12x128x128xi1>
    %attention_negative = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %attention_negative_broadcast = stablehlo.broadcast_in_dim %attention_negative, dims = [] :
        (tensor<bf16>) -> tensor<12x128x128xbf16>
    %attention_masked_scores = stablehlo.select %attention_causal, %attention_scaled_scores, %attention_negative_broadcast :
        tensor<12x128x128xi1>, tensor<12x128x128xbf16>

    %attention_negative_init = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %attention_row_max = "stablehlo.reduce"(%attention_masked_scores, %attention_negative_init) ({
      ^bb0(%attention_lhs: tensor<bf16>, %attention_rhs: tensor<bf16>):
        %attention_max = stablehlo.maximum %attention_lhs, %attention_rhs : tensor<bf16>
        stablehlo.return %attention_max : tensor<bf16>
    }) {dimensions = array<i64: 2>} :
        (tensor<12x128x128xbf16>, tensor<bf16>) -> tensor<12x128xbf16>
    %attention_row_max_broadcast = stablehlo.broadcast_in_dim %attention_row_max, dims = [0, 1] :
        (tensor<12x128xbf16>) -> tensor<12x128x128xbf16>
    %attention_centered = stablehlo.subtract %attention_masked_scores, %attention_row_max_broadcast :
        tensor<12x128x128xbf16>
    %attention_exp = stablehlo.exponential %attention_centered : tensor<12x128x128xbf16>
    %attention_zero = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %attention_row_sum = "stablehlo.reduce"(%attention_exp, %attention_zero) ({
      ^bb0(%attention_lhs: tensor<bf16>, %attention_rhs: tensor<bf16>):
        %attention_sum = stablehlo.add %attention_lhs, %attention_rhs : tensor<bf16>
        stablehlo.return %attention_sum : tensor<bf16>
    }) {dimensions = array<i64: 2>} :
        (tensor<12x128x128xbf16>, tensor<bf16>) -> tensor<12x128xbf16>
    %attention_row_sum_broadcast = stablehlo.broadcast_in_dim %attention_row_sum, dims = [0, 1] :
        (tensor<12x128xbf16>) -> tensor<12x128x128xbf16>
    %attention_probability = stablehlo.divide %attention_exp, %attention_row_sum_broadcast :
        tensor<12x128x128xbf16>

    %attention_context_bhsd = stablehlo.dot_general %attention_probability, %attention_value_bhsd,
        batching_dims = [0] x [0], contracting_dims = [2] x [1],
        precision = [] :
        (tensor<12x128x128xbf16>, tensor<12x128x128xbf16>) -> tensor<12x128x128xbf16>
    %attention_context_shd = stablehlo.transpose %attention_context_bhsd, dims = [1, 0, 2] :
        (tensor<12x128x128xbf16>) -> tensor<128x12x128xbf16>
    %attention_context = stablehlo.reshape %attention_context_shd :
        (tensor<128x12x128xbf16>) -> tensor<128x1536xbf16>
    %attention_result = stablehlo.dot_general %attention_context, %attention_output_weight_bf16,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<128x1536xbf16>, tensor<1536x1536xbf16>) -> tensor<128x1536xbf16>
    %residual1 = stablehlo.add %x, %attention_result :
        tensor<128x1536xbf16>

    %rms2_x_f32 = stablehlo.convert %residual1 :
        (tensor<128x1536xbf16>) -> tensor<128x1536xf32>
    %rms2_weight_f32 = stablehlo.convert %post_attention_norm_weight :
        (tensor<1536xbf16>) -> tensor<1536xf32>
    %rms2_square = stablehlo.multiply %rms2_x_f32, %rms2_x_f32 :
        tensor<128x1536xf32>
    %rms2_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %rms2_sum = "stablehlo.reduce"(%rms2_square, %rms2_zero) ({
      ^bb0(%rms2_lhs: tensor<f32>, %rms2_rhs: tensor<f32>):
        %rms2_value = stablehlo.add %rms2_lhs, %rms2_rhs : tensor<f32>
        stablehlo.return %rms2_value : tensor<f32>
    }) {dimensions = array<i64: 1>} :
        (tensor<128x1536xf32>, tensor<f32>) -> tensor<128xf32>
    %rms2_hidden = stablehlo.constant dense<1.536000e+03> : tensor<f32>
    %rms2_hidden_broadcast = stablehlo.broadcast_in_dim %rms2_hidden, dims = [] :
        (tensor<f32>) -> tensor<128xf32>
    %rms2_mean = stablehlo.divide %rms2_sum, %rms2_hidden_broadcast : tensor<128xf32>
    %rms2_epsilon = stablehlo.constant dense<1.000000e-06> : tensor<f32>
    %rms2_epsilon_broadcast = stablehlo.broadcast_in_dim %rms2_epsilon, dims = [] :
        (tensor<f32>) -> tensor<128xf32>
    %rms2_variance = stablehlo.add %rms2_mean, %rms2_epsilon_broadcast : tensor<128xf32>
    %rms2_inverse_rms = stablehlo.rsqrt %rms2_variance : tensor<128xf32>
    %rms2_inverse_rms_broadcast = stablehlo.broadcast_in_dim %rms2_inverse_rms, dims = [0] :
        (tensor<128xf32>) -> tensor<128x1536xf32>
    %rms2_normalized = stablehlo.multiply %rms2_x_f32, %rms2_inverse_rms_broadcast :
        tensor<128x1536xf32>
    %rms2_weight_broadcast = stablehlo.broadcast_in_dim %rms2_weight_f32, dims = [1] :
        (tensor<1536xf32>) -> tensor<128x1536xf32>
    %rms2_scaled = stablehlo.multiply %rms2_normalized, %rms2_weight_broadcast :
        tensor<128x1536xf32>
    %rms2_result = stablehlo.convert %rms2_scaled :
        (tensor<128x1536xf32>) -> tensor<128x1536xbf16>

    %ffn_x_f = stablehlo.convert %rms2_result : (tensor<128x1536xbf16>) -> tensor<128x1536xf32>
    %ffn_gate_w_f = stablehlo.convert %gate_weight : (tensor<1536x8960xi8>) -> tensor<1536x8960xf32>
    %ffn_up_w_f = stablehlo.convert %up_weight : (tensor<1536x8960xi8>) -> tensor<1536x8960xf32>
    %ffn_down_w_f = stablehlo.convert %down_weight : (tensor<8960x1536xi8>) -> tensor<8960x1536xf32>
    %ffn_gate = stablehlo.dot_general %ffn_x_f, %ffn_gate_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<128x1536xf32>, tensor<1536x8960xf32>) -> tensor<128x8960xf32>
    %ffn_up = stablehlo.dot_general %ffn_x_f, %ffn_up_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<128x1536xf32>, tensor<1536x8960xf32>) -> tensor<128x8960xf32>
    %ffn_sigmoid = stablehlo.logistic %ffn_gate : tensor<128x8960xf32>
    %ffn_silu = stablehlo.multiply %ffn_gate, %ffn_sigmoid : tensor<128x8960xf32>
    %ffn_hidden = stablehlo.multiply %ffn_silu, %ffn_up : tensor<128x8960xf32>
    %ffn_down = stablehlo.dot_general %ffn_hidden, %ffn_down_w_f,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<128x8960xf32>, tensor<8960x1536xf32>) -> tensor<128x1536xf32>
    %ffn_result = stablehlo.convert %ffn_down : (tensor<128x1536xf32>) -> tensor<128x1536xbf16>
    %result = stablehlo.add %residual1, %ffn_result :
        tensor<128x1536xbf16>
    return %result : tensor<128x1536xbf16>
  }
}
