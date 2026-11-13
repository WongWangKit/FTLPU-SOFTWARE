# DeepSeek V4.1 Flash：独立 BF16 mHC

固定结果目录当前对应下文“VXM 融合结果”，此前三组结果保留为历史基线。

## 测试边界

本测试验证 layer 0 attention 分支的 mHC 系数生成、四流收缩与四流残差混合。
使用 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 的真实参数，固定 revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`。
这不是 SWA、FFN、完整 decoder 层或完整模型测试。

默认 T=32、H=4、D=5120，输入与输出如下：

| 张量 | 形状 | 含义 |
|---|---|---|
| X | [32,4,5120] | 四路残差流 |
| incoming_pre | [32,4] | 当前子层使用的输入混合系数 |
| Y | [32,5120] | 独立生成的子层输出测试输入，不在 host 上计算 attention |
| fn | [24,20480] | checkpoint 的 hc_attn_fn，显式转成 BF16 |
| scale / base | [3] / [24] | checkpoint 参数，显式转成 BF16 |
| pre / post / comb | [32,4] / [32,4] / [32,4,4] | mHC 产生的三组系数 |
| collapsed | [32,5120] | hc_pre 的输出 |
| expanded | [32,4,5120] | hc_post 的输出 |

没有持久状态。`pre` 是提供给下一子层的系数；当前 `hc_pre` 消费的是
`incoming_pre`，不能误接为当前 `hc_mixes` 的 `pre`。
默认使用四路不同的随机输入、非 one-hot 的 incoming_pre，避免重复输入掩盖
stream 索引或 comb 转置错误。它比第 0 层初始四流相同的退化情况覆盖更多路径。

## 算法与精度

1. 将 X 的 stream、hidden 维合并，计算其 inverse RMS。
2. 计算 24 维投影并乘 inverse RMS；在此边界转 BF16。
3. 前 4 维生成 `sigmoid + epsilon`；接下来 4 维生成 `2*sigmoid`。
4. 最后 16 维形成 4×4 logits。执行 row softmax + epsilon、一次 column
   normalization，再做 19 次 row/column normalization，保留官方完整 20 次迭代。
5. `collapsed[t,d] = sum_i incoming_pre[t,i] * X[t,i,d]`。
6. `expanded[t,j,d] = post[t,j] * Y[t,d] + sum_i comb[t,i,j] * X[t,i,d]`。
   comb 第一条 stream 轴是来源，第二条是目的。

输入、权重、scale/base、最终系数与输出按 BF16 RNE 存储。
投影积和、归一化及 Sinkhorn 中间值用 FP32；这是 BF16 存储、FP32 计算版本，
不是把所有中间量都截断成 BF16。原始 FP32 参数另外保存，CPU 对照单列
BF16 策略相对原精度公式的差异，避免混入设备实现误差。

设备相对同精度独立参考的验收要求：相对 L2 < 1%，cosine > 0.9999，
所有值有限；comb 非负，行列和与参考的最大绝对差 < 0.01。
20 次 Sinkhorn 不保证行和已在 1% 内接近 1；因此同时报告实际、参考的
行列和收敛偏差，而不把“已完全收敛”误当作官方有限次迭代的语义。

## LPU 映射与当前限制

- C++ `HcMixesOp/HcPreOp/HcPostOp` 分解成共用 MXM matmul、`VectorAluOp`、逻辑视图和归约。
  CModel 没有 mHC 专用计算，Python 不生成物理布局或 ICU 排程。
- 采用普通 planar token-column 布局：一个 32-lane 向量承载 32 个 token，
  hidden、stream、4×4 系数矩阵的索引映射到向量行。没有 distributed16。
- 系数投影复用 `lowerActivationBatchMatmul` 的 native BF16 MXM 核心，计算
  `W[24,20480] * X^T[20480,32]`，直接生成 [coefficient,token]，避免输出转置。
  每侧完成一半 K，即 [24,10240] × [10240,32]；24 个有效系数行使用 32 行
  MXM tile，忽略尾部行。K 不能对半按 32 对齐时采用共用核心的单 batch 路径。
- 每侧将多个 K32 编成连续域，两个权重缓冲交替加载，FP32 部分和保留在 MXM ACC。
  当前目标按容量分成 K2048 块，块内 64 次 K32 Compute 连续执行，只在块末
  通过已有 terminal 模式输出并清空 ACC。跨块的 FP32 add epilogue 仍由 VXM
  合并，两个半球的最终部分和再相加。只在乘 inverse RMS 之后转 BF16，
  未增加中间 BF16 舍入。块末的物化边界允许标准 iMEM 重装。
- K32 对齐的 RHS 片段直接保留原 producer 地址：每侧按所需的 stream/slice
  依次 READ3D 到 SXM，再直接到 MXM 权重端。跨 stream 边界只切换读取地址，
  不先拼接 mirrored 中间缓冲。规划器验证整个 32-lane word 连续、原值仍在
  生命周期内，以及所有源 MEM ICU 与其他同时读取的输入不冲突；不满足时保留
  通用 materialize 路径。此能力属于共用 native BF16 matmul lowering。
- sigmoid/exp、4×4 Sinkhorn、hc_pre/hc_post 继续使用 VXM。入口直接采用两侧
  mirrored planar 地址，在 mixes/pre/post 之间复用；系数保留一行并由 READ3D
  重复地址。兼容的即时运算串成 ALU 链，省去中间 MEM 往返，启动位置以实际
  排空边界为准。Python 不生成物理布局或分块排程。
- 当前仍有容量分块边界、数据依赖和通用 linker 的搬运，不宣称整个 mHC
  每个功能单元都满载；消除的是可以避免的间隔，不增加重复运算填图。
- 导出输出采用原生 producer pieces，避免为主机诊断强制归档大块结果。
  分块允许位于不同 bank，ModelSession 按各块地址分别通过 C2C 回读；没有
  修改 bank 容量或端口。原始诊断文件为 [flattened feature, token]，参考检查
  还原为上表逻辑顺序。

## 2026-11-11 VXM 初始基线（baseline_vxm）

真实尺寸独立 mHC：**passed**，raw-macro，`host_operations=0`。
最大相对 L2 误差约 0.193%，所有输出 cosine > 0.999998。

| 输出 | 相对 L2 误差 |
|---|---:|
| 归一化投影 | 0.07813% |
| pre | 0.07561% |
| post | 0.17760% |
| comb | 0.16498% |
| hc_pre 输出 | 0，逐值一致 |
| hc_post 输出（含系数生成误差） | 0.19313% |
| hc_post 局部检查（使用设备系数） | 0，逐值一致 |

20 次 Sinkhorn 后，设备的最大行和偏差为 0.0537385，独立参考为 0.0537416；
设备行列和与参考的最大差均约 0.004883。这说明有限次迭代尚未完全收敛，
不能声称输出 comb 的行和已严格接近 1；设备实现误差与该算法收敛误差分开记录。

500 MHz、51.2 GB/s 峰值 DDR、90% 调度效率下：

| 时段 | 周期 | 模拟时间 |
|---|---:|---:|
| 数据初始读入 | 189,148 | 0.378296 ms |
| 执行，含片上搬运/等待 | 5,222,751 | 10.445502 ms |
| 初始读入至执行完成 | 5,411,899 | 10.823798 ms |
| 额外诊断回读 | 266,987 | 0.533974 ms |
| 执行期间动态权重页等待 | 0 | 0 ms |

权重本轮驻留 MEM，初始读入开销仍计入表中。共导出 69 个 compute iMEM
分段；当前运行器不计这些指令镜像的 DMA 加载时延，上表不是完整产品 TTFT。
此表保留为 VXM 初始基线；MXM 版与其比较时必须使用相同输入、运行器和硬件配置。

附加验证：D=32 的完整 20 次 Sinkhorn 冒烟测试通过；T=64/D=32 的初始四流
one-hot hc_pre 测试逐值一致；C2C 回读与 weight_page_builder 回归通过。
Viewer 的现有 CSV worker 成功解析实际 trace：109,960 个 pattern、22 类资源，
结束周期 5,678,886。此项是解析检查，不是浏览器截图检查。

## 2026-11-11 MXM 投影结果

相同 T=32、H=4、D=5120 真实权重测试通过，完整执行系数生成、20 次 Sinkhorn、
hc_pre 与 hc_post。实际 trace 中东西两侧各有 320 个 K32 MXM Compute 块，
`host_operations=0`。输入字节、精度、目标配置和运行器 SHA256 均与保存的
VXM 基线一致，该历史比较记录在 `baseline_mxm/comparison.json`。

全部输出通过原阈值：归一化投影相对 L2 误差 0.07813%，最终四流输出 0.19313%。
pre、post、comb、hc_pre 与 hc_post 的输出字节与 VXM 基线一致；投影因 FP32
加法顺序改变存在差异，仍通过同一数值验收。D=32 的真实参数子集冒烟测试也通过。

| 时段 | VXM 基线周期 | MXM 版周期 | MXM 版时间 |
|---|---:|---:|---:|
| 数据初始读入 | 189,148 | 220,828 | 0.441656 ms |
| 执行，含片上搬运/等待 | 5,222,751 | 1,227,252 | 2.454504 ms |
| 初始读入至执行完成 | 5,411,899 | 1,448,080 | 2.896160 ms |
| 额外诊断回读 | 266,987 | 118,234 | 0.236468 ms |
| 执行期间动态权重页等待 | 0 | 0 | 0 ms |

执行时间缩短到基线的 23.50%（4.26 倍加速）；包含初始数据读入后为 3.74 倍。
数据入口时间有所增加，不能只用算子执行时间表示完整收益。以上使用同一
500 MHz、51.2 GB/s 峰值 DDR、90% 调度效率配置。

实际执行 ICU 导出包含 35,724 条 ICU 指令、32,802 个 iMEM words；
单队列最多 1,896 words。新程序不需要执行中的 compute iMEM 分段重装
（基线为 69 段），但初始指令镜像 DMA 仍不在模拟时延内。
编译计划含 1,335 个节点、0 个输出归档拷贝，无 distributed16；
通用 linker 的 MEM transport 记录有 1,868 条，K32 部分和仍会写回 MEM，
这次改善不能解释为已消除所有搬运。

实际流水 CSV 为 8,747,675 bytes；Viewer worker 解析通过，共 63,132 个 pattern、
27 类资源，结束周期 1,566,314。本地 viewer 服务已核验能读取同大小的最新文件。
以上是 CSV 解析与服务检查，不是浏览器截图验收。

## 2026-11-11 紧密排程基线（baseline_tight）

相同真实权重、T=32、H=4、D=5120、BF16 存储策略及运行器测试 **passed**。
输入字节、精度、目标配置和 runtime SHA256 与 `baseline_mxm` 一致，
`comparison.json` 自动验证这些条件。仍然执行完整 20 次 Sinkhorn、hc_pre、
hc_post；没有删算子、降低迭代次数、放宽误差阈值或修改 CModel 硬件。

| 时段 | 原 MXM 版周期 | 当前周期 | 当前模拟时间 |
|---|---:|---:|---:|
| 数据初始读入 | 220,828 | 142,961 | 0.285922 ms |
| 执行，含片上搬运/等待 | 1,227,252 | 304,892 | 0.609784 ms |
| 初始读入至执行完成 | 1,448,080 | 447,853 | 0.895706 ms |
| 额外诊断回读 | 118,234 | 123,486 | 0.246972 ms |
| 执行期间动态权重页等待 | 0 | 0 | 0 ms |

执行加速 **4.03 倍**，含初始数据读入加速 **3.23 倍**。权重本轮驻留，
动态页等待为零；指令镜像 DMA 尚不计时，不能将本表当作完整模型 TTFT。
诊断回读稍有增加，按实际值单列，不计入算子优化收益。

真实 runtime trace 的 MXM 调度如下（东西两侧相同，区间为左闭右开）：

| K 块 | Compute 周期区间 | 连续计算周期 |
|---|---|---:|
| 0 | [184767,186815) | 2,048 |
| 1 | [187011,189059) | 2,048 |
| 2 | [189249,191297) | 2,048 |
| 3 | [191490,193538) | 2,048 |
| 4 | [193728,195776) | 2,048 |

每侧仍执行 320 个 K32，不重复计算；原来的 320 段×32 周期改成 5 段×2048，
块内没有 Compute 空泡。K2048 块间仍有 190–196 周期，包含 terminal ACC
输出、跨块 FP32 合并、依赖排空和下一块启动。不能解释为整个程序全资源满载。
从第一次 Compute 到最后一次 Compute 的占用率由 4.09% 升至 93.01%；
这项是时间占用率，不是 24 个有效系数行在 32 行 tile 中的 lane 利用率。

VXM 采用兼容的 ALU 链、紧密启动和直接地址视图，仍有依赖驱动的串行归约。
按实际导出的 8 条 VXM ICU 队列平均，排除 bypass、以完整 invocation 为分母，
有效数学指令时间占比从 2.83% 升至 10.99%。这是队列发射占比，不能与物理
ALU/lane 利用率混用；它也表明后续仍有多列并行和跨算子融合的优化空间。

| 编译/程序指标 | 原 MXM 版 | 当前 |
|---|---:|---:|
| 图节点 | 1,335 | 326 |
| linker MEM transport 记录 | 1,868 | 148 |
| linker 搬运向量（按记录计，不含正常算子读写） | 637,680 | 480 |
| 原生输出归档拷贝 | 0 | 0 |
| ICU 指令 | 35,724 | 10,693 |
| iMEM words 合计 | 32,802 | 12,115 |
| 单队列最多 iMEM words | 1,896 | 792 |
| compute iMEM 分段重装 | 0 | 0 |

剩余搬运集中在跨半球合并和短系数布局；投影的 RHS 大块拼接已消除。
编译计划有 5 个 `segmented_native_mxm_read` 和 397 个 `address_view` 决策，
没有 distributed16。实际 ICU 导出包含 248 条 MEM_WRITE_SYNC、122 条 SYNC。

所有数值仍通过原阈值：归一化投影相对 L2 0.07813%，最终四流输出 0.19313%。
pre、post、comb、collapsed、expanded 五个输出与原 MXM 版逐字节一致；
projection 的 FP32 累加顺序变化没有增加可见误差。使用设备系数的 hc_post
局部检查逐值一致，`host_operations=0`。

验证还覆盖 D=32 的完整 mHC、T=64/D=32 的第 0 层 one-hot hc_pre、原 K32
head_matmul 及 native K2048 head_matmul。后者相对误差 1.49e-7，块内连续
2048 周期，只写最终结果。大分片输出的 binary 编码已修复一次逐渐膨胀的
容器复制；20,480 个输出片段的保存用例可正常编码，未更改指令 ABI。

固定结果目录包含 `summary.json`、`comparison.json`、`scheduling_audit.json`、
`structure.json`、`build_manifest.json` 以及实际执行 CSV/ICU program。
最新 pipeline 为 5,495,737 bytes，现有 Viewer worker 解析通过：39,756 个 pattern、
27 类资源、结束周期 571,339。本机 HTTP 返回 200，长度与本地文件一致；
此项是解析与文件服务验证，不是浏览器截图验收。

## 2026-11-11 VXM 融合结果（当前版本）

保持 T=32、H=4、D=5120、相同真实权重/输入、相同 runtime hash 和硬件配置。
本轮只修改 C++ 编译器融合和逻辑图分解，继续执行 raw-macro，未扩展 CModel 硬件。
FP32 乘法与加法仍分别舍入，不改成 FMA；所有既有 BF16 边界和 20 次 Sinkhorn
迭代保留。六个输出与 `baseline_tight` **逐字节一致**。

| 运算段 | 融合前 | 当前映射 |
|---|---|---|
| 每流平方及归约 | 平方结果落 MEM，再求和 | C0 读 X，C1 平方，C3 FP32 累加，仅写最终和；2→1 pass |
| 四流平方和→inverse RMS | 三次加法，再 mean/epsilon/rsqrt | C1/C3 预装两项，保持左结合求和，再接 mean/epsilon/rsqrt；4→1 pass |
| 投影两侧最终部分和→归一化 | add，再 multiply/BF16 | 预装 inverse RMS，C0 add→C1 multiply，链尾 BF16；2→1 pass |
| scale/bias→pre/post 门控 | scale、bias+sigmoid、尾部 epsilon/×2 | affine 乘加链，再 sigmoid+epsilon/×2 链；3→2 pass |
| logits scale/bias | multiply，再 add | 系数预装后 multiply→add；2→1 pass |
| 4 项 max / sum | 三次独立二元归约 | C0/C1/C3 逐项归约；3→1 pass |
| 每次 Sinkhorn 归一化 | 3 次 sum、epsilon+reciprocal、multiply | sum→epsilon→reciprocal 一条链，再 multiply；5→2 pass |
| hc_pre 四流加权求和 | 4 multiply+3 add+BF16 转换 | 首项 multiply，后 3 项 multiply→add，末项链尾 BF16；8→4 pass |
| hc_post 每路输出 | 4 residual multiply+3 add+post multiply+add | 首项 multiply，3 次 residual 乘加，最后 post 乘加/BF16；9→5 pass |

表内 pass 指整段数据扫描，不把局部系数预装算作一次完整扫描。
最后一次 Sinkhorn multiply 也直接输出 BF16，消除独立 `comb_bf16` 扫描。
实际 schedule 有 4 条 square-sum、42 条短归约、1 条 add-scale、43 条 scale-add
融合链。它们由共用 `VectorFusionEmitter.hpp` 识别 SSA 依赖后直接发射硬件域，
不从 FU trace 反向压缩。融合匹配限制为支持的普通 planar BF16/FP32 合约。

乘加链使用已有 C1 局部标量和 head 的 Auxiliary：先预装系数，C0 同时读取
当前 X 和旧 partial，C1 multiply，C2 add Auxiliary，末列写回。内部 ALU 没有
新增任意 SR 读端口。短归约最多预装两项到 C1/C3，然后从两个固定输入端口
送入剩余两项。只有明确分时捕获的输入可共用 MEM ICU；同时流式读取的输入
仍要求不冲突。硬件容量、队列串行规则和 FU 延迟没有放宽。

| 时间 | baseline_tight | 当前 | 当前 @500 MHz |
|---|---:|---:|---:|
| 初始数据读入 | 142,961 cycles | 143,079 cycles | 0.286158 ms |
| 执行，含片上搬运/等待 | 304,892 | 175,962 | 0.351924 ms |
| 数据入口至计算完成 | 447,853 | 319,041 | 0.638082 ms |
| 额外诊断回读 | 123,486 | 108,131 | 0.216262 ms |
| 执行中外存权重页等待 | 0 | 0 | 0 ms |

执行周期减少 **42.29%**，含入口数据读取减少 **28.76%**。权重驻留，动态页等待
为零；指令镜像 DMA 仍不计时。不能把该独立 mHC 时间当作模型 TTFT。

| 指标 | baseline_tight | 当前 |
|---|---:|---:|
| 图节点 | 326 | 159 |
| 普通 MEM.Read 发射周期累计 | 2,932,248 | 1,555,540 |
| 普通 MEM.Write 发射周期累计 | 1,878,176 | 890,696 |
| linker 搬运记录 / 向量 | 148 / 480 | 148 / 480 |
| 原生输出归档拷贝 | 0 | 0 |
| VXM 数学指令平均队列时间占比 | 10.99% | 18.67% |
| ICU 指令数 | 10,693 | 14,512 |
| iMEM words 合计 | 12,115 | 14,530 |
| 单队列最大 iMEM words | 792 | 1,947 |
| compute iMEM 分段重装 | 0 | 0 |

MEM 统计按实际 trace 的 slice 发射次数累计，包含两侧正常读写，不是 wall-clock
时间，也不是 linker 记录数。数学链内消除的 FP32 spill 不属于 linker transport，
因此 linker 记录数不变。VXM 指标以实际导出的 8 条 ICU 队列平均、完整 invocation
为分母、排除 bypass，不代表物理 ALU/lane 利用率。

融合需要短捕获指令及逐行更新 C1/C3，**ICU 指令数量增加**；本轮选择减少总时延
和大块 MEM 往返。最满 VXM 队列 1,947 words，未超过固定 2,048 容量。
实际程序有 256 条 MEM_WRITE_SYNC、126 条 SYNC，无执行中 iMEM 分段重装。
MXM 两侧仍各有 5 段连续 2,048-cycle Compute，块间间隔 190–196 cycles。

当前保留的边界是：每次 Sinkhorn 必须先得到全行/列分母再归一化；下一轮依赖
本轮矩阵；gate 的 affine 和 sigmoid 尾部受深度 8 与特殊 ALU 列约束；四路残差
加权受固定外部端口和局部寄存器约束。没有通过重复计算填满流水图。

数值验收：最终 expanded 相对 L2 **0.19313%**、cosine **0.99999815**；projection
相对 L2 **0.07813%**。原阈值不变，所有六个结果与基线完全一致，
`host_operations=0`。新增 13 组独立 raw-macro 回归覆盖 1/2/24/33/64 行、
32/64/96 列、BF16/FP32、广播、累加器复位/终值、跨 word 系数更新；融合与分开
执行逐值一致，基本乘加/平方和另外对照顺序 FP32 CPU 算术。还通过 D32 完整
mHC 和 T64/D32 one-hot 第 0 层入口测试。

`summary.json`、`comparison.json`、`structure.json`、`scheduling_audit.json`、
`build_manifest.json` 已更新。实际 pipeline 为 5,965,966 bytes，现有 Viewer
解析通过：43,584 个 pattern、27 类资源、结束周期 427,172；本机 HTTP 200 且
长度一致。这是解析与文件服务验证，不是浏览器截图验收。

## 复现

在 `FTLPU-SOFTWARE` 根目录运行；使用本次保存的 compiler/runtime 快照：

```powershell
D:/Apps/Anaconda3/python.exe compiler/tools/import_deepseek_v41_mhc.py --output results/deepseek_v41_flash_layer0_mhc/weights --cached-index tmp/deepseek41_official_index.json
D:/Apps/Anaconda3/python.exe compiler/tests/deepseek_v41_mhc_test.py --build-dir tmp/deepseek_mhc_fused_snapshot --target-config results/deepseek_v41_flash_layer0_mhc/baseline_tight/target_config.json --weights results/deepseek_v41_flash_layer0_mhc/weights --output-dir results/deepseek_v41_flash_layer0_mhc --baseline results/deepseek_v41_flash_layer0_mhc/baseline_tight
D:/Apps/Anaconda3/python.exe compiler/tests/vector_arithmetic_fusion_runtime_test.py --build-dir tmp/deepseek_mhc_fused_snapshot --target-config results/deepseek_v41_flash_layer0_mhc/baseline_tight/target_config.json --output-dir results/deepseek_v41_flash_layer0_mhc/fusion_regression
```

可选 `--hidden 32 --output-dir results/deepseek_v41_flash_layer0_mhc/smoke` 做真实
参数子集的冒烟测试；`--stage mixes|pre|post` 可独立定位。
`--fixture layer0-entry --stage pre --hidden 32 --tokens 64` 检查两个 token32
分块的第 0 层入口：重复 embedding、one-hot incoming_pre，要求输出逐值一致。
`--prepare-only` 只编译，`--reuse-prepared` 校验源文件、配置及 executable hash
后复用已编译程序。正式执行始终用 raw-macro。
加 `--baseline results/deepseek_v41_flash_layer0_mhc/baseline_tight` 可生成
`comparison.json`；比较会校验逻辑输入字节、精度、目标配置及运行器 hash 一致。

固定目录中 `summary.json` 是本次状态及数值依据；未标记 `passed` 不能视为通过。
`weights/provenance.json` 给出参数来源、范围与 SHA256；`prepared.json` 保存编译签名。
`runtime.pipeline.csv` 来自实际运行；`actual.runtime.ftlpu` 和 `runtime_icu_programs/`
是实际执行程序及每 ICU 的 CSV，包括入口的 MEM_WRITE_SYNC。

跨 bank 诊断输出的回归由 `model_session_c2c_io_test` 验证：同地址、不同 bank
的不同值必须被正确回读，而同 bank 的重叠分块必须拒绝。
`weight_page_builder_test` 同时覆盖原有 bank-local 打包/解包及输出分块序列化。
该测试中一次只为增加输出描述而重新加载未排空程序的旧写法已移除：输出描述
现在随首次加载一起提供，保持硬件排空契约不变。

时延包含模拟的数据入口与执行，单列权重页等待和诊断回读。
当前 ModelSession 不模拟指令镜像加载 DMA 的时间，因此不能直接当作包含全部
启动开销的产品 TTFT。硬件配置为 500 MHz、51.2 GB/s 峰值 DDR、90% 调度效率。

Viewer：
http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_mhc/runtime.pipeline.csv&v=mhc_fused_20261111
