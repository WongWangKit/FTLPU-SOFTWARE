# Qwen C2C 权重双缓冲部署

## 目标

Qwen decoder 层不再要求全部权重同时常驻片上 MEM。runtime 首先通过 C2C 将第 0 层已经量化、并按目标 SRAM 布局打包的权重写入 bank 0。计算第 `i` 层时，MXM 从当前 bank 读取权重；C2C 同时把第 `i+1` 层写入另一 bank。层结束后仅在下一页尚未 ready 时等待，然后交换 bank。

## 物理语义

- 每个 MEM slice 有 2 个独立单端口 SRAM bank。
- MEM ICU queue 唯一标识 `(hemisphere, slice, bank)`。
- SRAM 地址是 bank-local row，范围为 `0..32767`，每 row 为 32 bytes。
- bank 0 read 与 bank 1 write 可以同 cycle 发射；同一 bank 内仍服从单端口约束。
- C2C page ready 表示最后一个 MEM write 已经完成，不等同于 DMA 把最后一个 vector 放入 RX FIFO。

## 软件表示

Binary v17 在 `BinaryBinding` 和 `BinaryMemoryFloor` 中保存 bank。ModelPackage v5 用 `ModelWeightPage` 描述层号、目标 bank、target-packed tensor，以及每个物理 segment 的 DDR offset、半球、slice、row、vector 数和 stream。

只有 `TargetPackedSramVectors` tensor 可以作为 C2C page image。`ftlpu-pack-model-weights` 根据 executable binding 把已经量化的 row-major 权重离线重排成目标 SRAM row image；runtime 只负责搬运，不在部署时重新排列大权重。

分页编译时使用 `ftlpu-opt --weight-bank 0|1`。bank 从 Tensor IR 起参与物理规划，而不是在 Command IR 上后贴标签。Qwen2.5-1.5B 的 Attention 权重使用独立 slices 32..39，FFN gate/up/down 使用互不重叠的 slice plane，RMSNorm gamma 从 bank 顶部向下放置。

## 执行顺序

1. 预取 page 0 到 bank 0，并等待 SRAM commit。
2. 加载 layer 0 ICU 程序。
3. 启动 page 1 到 bank 1 的 C2C DMA。
4. 同一个 runtime cycle driver 同时推进 chip、DMA 和 DDR。
5. layer 0 完成后确认 page 1 ready；若未完成则只等待剩余搬运。
6. 加载 layer 1 ICU 程序，启动 page 2 到 bank 0，依次交替。

连续 SRAM row 写使用一条 MEM Write 加 ICU Repeat，指令数量按 segment 数增长，而不是按 32-byte vector 数增长。repeat interval 使用 `max(C2C RX tile replay cycles, DDR vector service cycles)`。

## 当前验证

- `c2c_weight_ping_pong_test`：同一 slice 的 bank 0 read 与 bank 1 C2C write 同 cycle 发射，数据逐字节正确。
- `weight_page_planner_test`：两层权重不进入 resident allocator，分别绑定 bank 0 和 bank 1。
- `model_session_c2c_weight_pages_test`：page 0 首装、layer 0 期间 page 1 预取、层间 bank 切换完整通过。
- `weight_page_builder_test`：bank1 host upload 与离线 page image 逐字节一致，并确认 bank0 未被写入。
- `qwen_weight_page_builder_test`：按 Qwen2.5-1.5B 的 Q/K/V/O、两组 norm、gate/up/down 真实 shape 生成一层 45 MiB page，共 192 个连续 C2C segment，且所有 row 均位于单个 32768-row bank 内。
- `qwen2_5_1_5b_paged_weight_layout_test`：从标准 StableHLO lower 到 Tensor/Stream IR，检查 bank、slice、row 范围和 head-dim 128 的 attention weight row 公式。

部署包转换命令：

```powershell
build-ftlpu-vs2026/runtime/ftlpu-pack-model-weights.exe `
  --input qwen.logical.ftlpum `
  --output qwen.paged.ftlpum `
  --first-bank 0
```

当前尚缺本机 Qwen2.5-1.5B checkpoint 驱动的两层 CModel 数值 golden。另一个已确认的问题是 seq_len=128 整层 Command IR 的完全展开和文本打印超过十分钟；部署路径下一步应让 binary emitter 直接消费压缩 Schedule/repeat 表示，避免物化巨型 Command MLIR 文本。

## 性能判断

prefill 的单层计算窗口较长，具备隐藏下一层权重搬运的机会。decode 的 `M=1` 计算窗口很短，若每层都搬入约几十 MiB 权重，通常会受 DDR/C2C 带宽限制。decode 后续需要更高链路带宽、按 projection/weight wave 更细粒度预取，或增加可常驻权重容量；双 bank 只能实现重叠，不能消除带宽下界。
