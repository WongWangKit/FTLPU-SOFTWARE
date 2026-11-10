// Q and shared KV before partial RoPE, batch 1, seq 32. The checkpoint keeps W[N,K].
!X = tensor<32x5120xbf16>
!X8 = tensor<32x5120xf8E4M3FN>
!Xs = tensor<32x160xf8E8M0FNU>
!Wa = tensor<1280x5120xf8E4M3FN>
!Sa = tensor<40x160xf8E8M0FNU>
!Wb = tensor<32768x1280xf8E4M3FN>
!Sb = tensor<1024x40xf8E8M0FNU>
!Wkv = tensor<512x5120xf8E4M3FN>
!Skv = tensor<16x160xf8E8M0FNU>
!Rank = tensor<32x1280xbf16>
!Rank8 = tensor<32x1280xf8E4M3FN>
!Ranks = tensor<32x40xf8E8M0FNU>
!Q = tensor<32x32768xbf16>
!KV = tensor<32x512xbf16>

module {
  func.func @layer0_projections(%x: !X, %wa: !Wa, %sa: !Sa, %wb: !Wb, %sb: !Sb,
      %wk: !Wkv, %sk: !Skv, %gq: tensor<1280xbf16>, %gkv: tensor<512xbf16>) -> (!Q, !KV) {
    %x8, %xs = ftlpu.kernel.block_fp8_quantize %x : (!X) -> (!X8, !Xs)
    %qa = ftlpu.kernel.block_fp8_matmul %x8, %xs, %wa, %sa : (!X8, !Xs, !Wa, !Sa) -> !Rank
    %kv0 = ftlpu.kernel.block_fp8_matmul %x8, %xs, %wk, %sk : (!X8, !Xs, !Wkv, !Skv) -> !KV
    %qr = ftlpu.kernel.rms_norm %qa, %gq {axis = -1 : i64, epsilon = 1.0e-20 : f32}
        : (!Rank, tensor<1280xbf16>) -> !Rank
    %kv = ftlpu.kernel.rms_norm %kv0, %gkv {axis = -1 : i64, epsilon = 1.0e-20 : f32}
        : (!KV, tensor<512xbf16>) -> !KV
    %r8, %rs = ftlpu.kernel.block_fp8_quantize %qr : (!Rank) -> (!Rank8, !Ranks)
    %q = ftlpu.kernel.block_fp8_matmul %r8, %rs, %wb, %sb : (!Rank8, !Ranks, !Wb, !Sb) -> !Q
    return %q, %kv : !Q, !KV
  }
}
