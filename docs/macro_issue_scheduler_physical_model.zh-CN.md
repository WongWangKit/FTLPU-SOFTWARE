# Macro Issue Scheduler 物理模型

## 1. 冻结结论

Macro Issue Scheduler 是分布在每个物理 ICU queue 内的绝对周期事件调度器，
不是通用动态仲裁器。首版采用固定深度 Active Context RAM 和 flat parallel match：

```text
DDB / lazy admission (<= 1 context/cycle)
  -> fixed Active Context RAM
  -> valid && next_issue_cycle == current_cycle
  -> one-hot context select
  -> <= 1 native instruction/queue/cycle
```

物理 baseline：

| 项目 | 冻结值 |
|---|---:|
| MXM Active Context depth | 40 |
| MEM Active Context depth | 32 |
| Active Context width | 256 bit（8 × 32-bit word） |
| Issue width | 1 / queue / cycle |
| Admission width | 1 / queue / cycle |
| Match key | 32-bit absolute `next_issue_cycle` |
| Selection | flat valid-qualified equality match |
| Admission bypass | 无，新 context 下一周期可见 |

MXM load、compute 和 dequant 是独立物理 queue，各自独立 match 和 issue。当前
CModel 仍使用同一个 MXM depth 参数；未来编译器容量分析可以为三类 queue 分别
收敛更小的 target depth，但不改变本版逻辑协议。

## 2. Active Context 状态

每个 entry 使用以下固定物理编码：

```text
bit   0       valid
bit   49:1    current native instruction              49 bit
bit   51:50   induction_target                         2 bit
bit   67:52   inner_count_minus_1                     16 bit
bit   83:68   inner_remaining_minus_1                 16 bit
bit   99:84   outer_remaining_minus_1                 16 bit
bit  131:100  inner_interval_minus_1                  32 bit
bit  163:132  outer_residual_minus_1                  32 bit
bit  177:164  inner_operand_step, signed              14 bit
bit  191:178  row_transition_step, signed             14 bit
bit  223:192  next_issue_cycle                        32 bit
bit  255:224  final_issue_cycle                       32 bit
```

其中：

```text
inner_span = (inner_count - 1) * inner_interval
outer_residual = outer_interval - inner_span
row_transition_step = outer_stride
                    - (inner_count - 1) * inner_stride
```

entry 保存的是下一次将直接发射的 current native instruction，而不是需要在 issue
路径上重新做二维乘加的 base template。非末次 issue 后只更新 remaining counter、
current operand 和 `next_issue_cycle`。count 最大为 65,536；超范围 Macro 必须在
compiler/encoder 侧等价拆分。MEM address 与 MXM accumulator 都是 13-bit operand，
因此任意两个合法 operand 之间的差可由 signed 14 bit 保存。

`record_index/pc` 和 `admission_cycle` 只用于 CModel trace/统计，作为同 slot 的
sidecar 保存，不计入 256-bit 物理 RAM。

## 3. Match 与错误语义

对深度为 `N` 的 queue：

```text
match[i] = valid[i] && next_issue_cycle[i] == current_cycle
late[i]  = valid[i] && next_issue_cycle[i] <  current_cycle
```

- `match == 0`：本周期没有 Macro issue；
- `match` one-hot：发射对应 context；
- 多个 match：same-queue collision，程序非法；
- 任意 late：missed issue cycle，程序非法。

合法 binary 的 match 由编译器保证至多 one-hot，因此没有公平性、年龄优先级或
QoS 策略。multi-hit 和 late 检测是 fail-fast 验证，不会把 instruction 推迟到
后续周期。

## 4. 周期内状态转移

每个 cycle 的逻辑顺序固定为：

```text
1. compare all valid next_issue_cycle fields
2. issue the unique due context, if any
3. update its counter and next_issue_cycle, or complete/free it
4. accept at most one frontend admission
5. make the admitted context visible in the next cycle
```

因此同一周期可能同时需要一个 existing-context update 和一个 new-context write。
末次 issue 释放的 slot 可以直接作为本周期 admission 的目标，但 admission 不得
回到本周期的 match 路径。

## 5. 实现边界

首版明确不引入：

- heap/priority queue；
- calendar wheel；
- 多级 scheduler；
- collision 后的动态 arbitration；
- FU backpressure 导致的 replay。

综合器可以把 40 路 equality reduction 和 one-hot payload mux 实现为平衡逻辑树；
这属于门级实现，不改变 flat-match 架构。正常关键路径只包含 equality、one-hot
selection 和 native instruction 输出。context update 算术位于 issue 后的写回路径。

## 6. 编译器契约

硬件容量固定后，编译器最终必须对 packed bitstream 使用相同 frontend 参数进行
capacity-feasibility 检查。若某 queue 超过 40，应优先沿 outer/inner 时间维度拆分
长生命周期 Macro，在不改变任何 issue cycle 的前提下降低 Active Context 峰值。
该编译器变换不属于本次 scheduler RTL-like 模型实现范围。
