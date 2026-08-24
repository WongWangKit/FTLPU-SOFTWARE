# ICU 粗粒度调度 ISA v1

## 范围

v1 将确定性的 MEM/MXM 指令展开从 host runtime 下沉到各功能单元的本地
ICU。它不改变 Schedule IR 的 cycle，也不修改已有 MEM/MXM 原生指令。
VXM、SXM 暂时保留旧队列格式。

通过以下选项启用：

```text
ftlpu-opt ... --icu-macro-schedule
```

生成的 module 带有 `ftlpu.icu_macro_schedule = true`，binary 版本升级为
v22。

## 描述符

每条 macro 包含一条原生功能单元指令和如下调度字段：

| 字段 | 含义 |
| --- | --- |
| `start_cycle` | 第一次发射的绝对 cycle |
| `inner_count` | 单个内层 repeat 的发射次数 |
| `inner_interval` | 内层相邻发射的 cycle 间隔 |
| `inner_stride` | 内层每次发射的操作数字段增量 |
| `outer_count` | wave 重复次数 |
| `outer_interval` | 相邻 wave 起点间隔 |
| `outer_stride` | 每个 wave 的操作数字段增量 |
| `induction_target` | MEM 地址、MXM weight column、MXM accumulator 地址或无 |

binary v21 原型在原生指令 payload 后附加 `MACR` 标记和 8 个字段。当前
采用可变长 envelope 是为了先验证语义，并不代表最终 RTL 位宽已经冻结。

坐标 `(outer, inner)` 的实际发射为：

```text
cycle = start_cycle + outer * outer_interval + inner * inner_interval
operand_delta = outer * outer_stride + inner * inner_stride
```

## 编译器压缩

binary lowering 会识别同一队列中“原生指令形状相同、cycle 间隔固定、
地址或 column/accumulator 步长固定”的重复窗口，再折叠为可以交错执行的
二维 macro。该规则面向所有队列，不是 FFN 专用 helper。

压缩在 binary lowering 之前就有明确的 IR 表达：

- Schedule `mem_read`/`mem_write` 使用地址 wave；
- Schedule `mxm_load` 使用 outer group 表达重复 IW 窗口；
- Schedule `mxm_compute` 与 `mxm_accumulate` 共享 wave，归纳变量可推进
  accumulator address；
- Command `mem_bundle` 只保存一次物理 slice lane 集合，同时保留每条 lane
  自己的 queue、cycle、address 和 stream selector。

Schedule verifier 会展开所有逻辑发射点再检查资源占用，因此紧凑表示不会放宽
逐 cycle 精度。

SmolLM2-135M、seq_len=32、Vector FFN 的结果如下：

| 指标 | 旧队列 | Macro v1 | 降幅 |
| --- | ---: | ---: | ---: |
| 队列命令数 | 53,511 | 6,214 | 88.4% |
| Binary 字节数 | 1,620,663 | 412,565 | 74.5% |
| `max_cycle` | 46,421 | 46,421 | 不变 |

真实权重 Qwen2.5-1.5B 第 0 层 FFN、`seq_len=32`、每半球一个 Vector MXM、
每 superlane 128 KiB target 的结果如下：

| 产物 | 展开形式 | 层次化形式 | 降幅 |
| --- | ---: | ---: | ---: |
| Schedule IR | 510,631,417 B | 5,199,998 B | 99.0% |
| Command IR | 243,032,926 B | 2,248,327 B | 99.1% |
| Binary | 28,947,545 B | 885,357 B | 96.9% |
| `max_cycle` | 770,646 | 770,646 | 不变 |

压缩 binary 已通过 CModel 的 49,152 个 BF16 输出 golden 对比，mismatch 为
0；MAE 保持 `6.54569e-05`，最大误差保持 `0.00418091`。

## 本地 ICU 结构

每个 MEM/MXM ICU 包含 descriptor FIFO 和 next-issue calendar。多个描述符
可以同时 in-flight，因此不同 wave 可以穿插发射。CModel 使用按下一次发射
cycle 排序的最小堆；RTL 可采用小型有序 calendar、timing wheel 或有限路
比较树。

以下情况会立即报错：错过目标 cycle、同一队列同一 cycle 有两个描述符到期、
非法迭代空间、macro 活跃时混入旧式控制流。编译器在写 binary 前也会检查
同队列 cycle 冲突。

## RTL 定型建议

当前 extension-word 格式属于软件验证格式。硬件版建议使用固定描述符头、
FU 专用 payload、显式版本和长度，并将最大 in-flight 数纳入 target model；
超长 count/interval 可使用 escape word。硬件 capability bit 正式可用之前，
runtime 必须保留旧细粒度队列作为回退路径。
