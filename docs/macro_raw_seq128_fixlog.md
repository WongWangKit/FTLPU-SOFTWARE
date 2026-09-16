# Macro Raw 路径 Seq128 修正日志

日期：2026-09-16；分支：`experiment/icu-macro-runtime-cmodel`

## 背景

在使用 SmolLM2 seq128 验证 compiler → `.ftlpu` → runtime → CModel raw iMEM
路径时，依次暴露了若干此前被 seq32 和最终输出检查掩盖的内存布局、流水时序及测试问题。
这些问题并非 Macro decoder 的数值语义错误；关闭 raw Macro 路径后，错误位置保持不变。

## 修正内容

- Attention 重叠判断改为仅在 bank 相同、slice 相交且 row 区间重叠时报告冲突。
- 修正 probability pack、Value、score/exp/mask 的 bank/slice 分配，避免真实 SRAM 覆盖。
- 将 RMS gamma 与 RoPE table 分离，避免初始化阶段常量互相覆盖。
- Key 低阶 RoPE product 使用独立 base row，避免覆盖 Query-IW。
- paired context 布局只写一份对应 hemisphere context，避免重复复制。
- 调整 FFN hidden/temp bank 与地址，保证输入、隐藏值和临时值物理隔离。
- Gate/Up 与 Down timeline 在复用权重 buffer 前等待 projection slot offset 和 MXM
  first-result latency，避免 seq128 尾部结果被下一次 dequant 覆盖。
- 修正 seq128 Value checkpoint 的 token-block 地址，并增加 Q/K RoPE、probability、
  context、FFN 和 raw Macro frontend 检查。

## 验证结果

- SmolLM2 seq128 完整 raw Macro 路径通过：73,728 个 BF16，最大误差 `0.03125`。
- 展开工作量 `2,218,123`，编码功能 entry `139,686`，逻辑工作量压缩率
  `93.7025%`。
- Macro v1 物理镜像：`24,040` slots / `292,012` bytes。
- 170 个 queue 均无 iMEM overflow 和 Macro context overflow。
- allocator、Attention planner/layout/RoPE、FFN timeline、Macro bitstream 专项测试通过。

逻辑工作量压缩率衡量 Macro/Repeat 用较少控制 entry 表达相同 issue 工作的能力，
不等同于 `.ftlpu` 文件压缩率、物理 iMEM 字节压缩率或执行性能提升。
