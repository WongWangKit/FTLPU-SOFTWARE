# W8A16 FFN M-Tile Weight 复用

![M=128 W8A16 FFN weight reuse schedule](w8a16_ffn_mtile_weight_reuse.svg)

该图来自当前 `M=128, K=64, intermediate=128, output=64` 的 Schedule IR
回归测试，并展开第一个 `K=32, N=32` projection block。四个物理 gate/up MXM
weight tile 只在 cycle `24/28/32/36` 各 load 一次到 `weight_buffer=0`；随后
四个逻辑 `M=32` activation tile 分别从 cycle `42`、`80`、`118`、`156` 开始 compute。
M tile 之间保留 target 要求的 38-cycle MXM issue interval，当前 M0..M3 仍复用驻留在
`weight_buffer=0` 的同一个 weight tile。

复用边界是逻辑 `(K tile, N tile)`。推进 K 或 N 时 weight tile 已改变，必须重新
load；推进 M 时只改变 activation tile，因此复用当前驻留的 weight buffer。
