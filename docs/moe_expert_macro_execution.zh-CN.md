# 设备端专家 macro 模板执行

第三部分增加 ExpertMacroExecutor，作为 ExpertTaskController 的设备消费者。设备等待 INPUT_READY 和 WEIGHT_READY，生成参数快照、准备各队列、共同启动常驻程序，输出提交并排空后才返回 TASK_DONE。CPU 不逐任务重编译或修补程序。

## 配置与兼容性

Routing 描述符增加 ExpertTemplate 配置记录（类型 6），Routing 能力为 v4；不是可执行指令，SELECT/PLAN/ADDR/DISPATCH 编码及 v41 容器保持不变。配置记录包含输出视图、工作区基地址、BF16 scale 位模式和有限队列/绑定表。模板引用已通过原有装载通路写入的队列 iMEM，不从 DDR 动态换装程序。

一个程序一个模板，最多 64 个参与队列，每队列最多 32 条绑定规则，与其他 Routing 配置共享 4096 个 u32 存储上限。MEM/MXM 队列使用现有 packed Macro；VXM 使用原生配置队列，SXM 允许无动态字段的原生队列。参与队列独占其模板程序，不承载全局 QCTL 或 MEM transport 控制。

## 参数与绑定位置

每条绑定规则为六个 u32：逻辑 record/原生 PC、目标字段、参数槽、常数乘数、有符号加数、是否叠加原始值。目标字段支持 inner_count、outer_count、地址/accumulator 等原生操作数、dequant 的 BF16 scale，以及原生 VXM repeat_count。起始周期和发射间隔保持模板常量，不提供执行中改周期的接口。

参数槽为 valid_rows、输入基地址、固定权重基地址、输出基地址、工作区基地址、scale 位模式、assignment_base 和零。输出基地址由输出视图的 plane.base_row + assignment_base * token_stride 得到。每周期向一个参与队列提交快照，之后任务期间保持不变。

MEM/MXM 的绑定发生在解析后、写入 DDB 前。压缩解码器保留原始 start/operand delta 基准；绑定只修改单独的描述符副本，启用绑定时每个逻辑记录单独提交 DDB 项。每周期检查/应用一条绑定规则，再用一周期校验和提交；有限 DDB 满时反压。Active Context 仍通过原 pack_initial() 计算 remaining、next/final issue、outer residual 和 row transition，运行中不再修改。

VXM repeat_count 在原生取指路径绑定，并计入一个额外前端周期；它不经过当前 MEM/MXM packed Macro DDB。现有 VXM 配置采用三字 RUN_2D 包；绑定规则的 record 指向包首字 PC，实际替换续字内的 compact repeat 字段。物理位置通过现有 packet codec 确定，包括首字 Extended subtype 占用的四位，不使用简单的 88 位取模。只修改取指副本，原 iMEM 不变。repeat_count 表示实际配置执行次数，不是全局时钟数。

循环次数只允许依赖 valid_rows 或常量；同一 record 的同一字段不能重复绑定。原生 VXM 规则必须指向完整的 RUN_2D 包，SXM 暂不开放动态字段。

首版模板使用固定队列位置和固定权重布局，所有专家的 SRAM 落点、段长度及顺序必须一致，只有 DDR 来源可不同；WeightBase 使用已验证的静态模板配置，运行中不额外无周期遍历专家表。输入/输出基地址必须与对应 MEM 队列的 bank/plane 匹配。输出 bank 与原始激活、Gather 输入和驻留权重分离。

## 局部时间域与共同启动

```text
任务输入/权重就绪
 → 逐队列提交参数快照
 → PREPARE：取指、解析、绑定、DDB/初始 Active Context 准备
 → 所有参与队列准备完成
 → 下一物理周期共同进入 RUN，任务局部时钟为 0
 → 按原静态相对周期发射
 → 输出提交并排空
 → TASK_DONE
```

旧任务的 IQ、parser、DDB 和 Active Context 在重启时清除，原始 iMEM 保留；只重启模板所属队列，不复位全片、RCU 计划或 program generation。PREPARE 使用前端 priming 路径推进，不使用会冻结 parser 的普通 HOLD。绑定和准备延迟只移动共同起点 C0。

编译/装载前使用实际有限前端，在 1 行和块容量上界上验证模板准备、准入、context 容量和发射周期；运行中保留 missed-cycle、同队列发射重叠和字段范围检查。此验证针对静态时序和绑定，不代替算子数值验证。模板内部数据通路的 SR、FU 和 bank 资源仍需按静态计划保留，不支持参与队列在 RUN 中各自任意暂停。

## 完成条件

输出 MEM Write 在队列出口附带当前任务的 generation/task_id 侧带标识，不改变已有编码。MEM 只在真实 SRAM 写入时增加输出提交数；预期数为 valid_rows * (columns/32) * bytes_per_element * tile_count。超出、缺失或旧任务提交均报错，不能当作成功。

TASK_DONE 同时要求所有模板队列结束、预期输出段已提交，以及现有 MEM/MXM/VXM/SXM/SR 瞬态资源排空。首版使用现有全片瞬态排空检查，可能等待无关流量；没有隐式冻结无关队列。任务 trace 有限保留最近 256 项，记录身份、共同起点、结束周期和提交段数。

## 精简验证与范围

1. 原 raw_macro_decoder_test 内增加 32/5 行参数快照、原始 delta 保持、绑定周期和原生 VXM repeat 字段替换检查。
2. routing_runtime_test 先通过直接装载执行固定单行模板，再通过定时 DMA 装载执行真实 SELECT/PLAN/Gather/间接权重 DMA/动态模板链。验证模板为 32×32 BF16 投影、BF16 accumulator 输出和原生 VXM 乘 0.5 后输出 BF16；VXM 内部仍采用 FP32 运算。专家结果不要求以 FP32 存储。使用固定镜像输入组所需的零数据，不改动 VXM 硬件布局。
3. 动态链处理 [32,5,5] 三个任务，核对每个队列的共同周期、旧任务通知拒绝、输出提交和尾部哨兵。输出按 assignment 顺序保存，与参考逐项比较。

第四部分在上述用例后追加加权回并，另增加 K=2 用例。当前验证结果与完整链路周期记录见 [输出回并说明](moe_routing_combine.zh-CN.md)。旧 FP32 验收模板的周期不再作为当前 BF16 链路的结果。

第三部分完成的是参数化模板执行控制；输出加权回并由第四部分增加。验证模板不是完整 Gate/Up/SiLU/Down FFN；不包含完整 FFN lowering、任意量化布局或全模型适配，也不宣称 FP32 中间通路已全面支持。
