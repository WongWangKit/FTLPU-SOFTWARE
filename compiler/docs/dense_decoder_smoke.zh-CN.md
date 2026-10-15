# Dense Transformer 单层数值测试

`compiler/tools/run_dense_decoder_smoke.py` 提供 seq32 单层 prefill 的完整链路：生成模型形状对应的 HF checkpoint、生成参考结果、编译 ICU 指令、打包、通过 ModelSession/CModel 执行并比较 BF16 输出。

支持的 fixture：`llama3_2_1b`、`ministral3_3b`、`phi4_mini`、`smollm3_3b`、`glm_edge_1_5b`、`minicpm4_0_5b`。Phi 和 GLM 的融合权重会在导入时拆分；MiniCPM 的两个 residual 分支均包含 depth scaling。

这些是完整矩阵维度的合成权重测试，不能视为官方预训练 checkpoint 验收。默认保留零 Q/K 基线；增加 `--qk-pattern nonzero` 使用稀疏非零 Q/K，覆盖 RoPE、QK 打分和非均匀 causal softmax，以及分页、GQA、residual 和 SwiGLU 的执行链路。Phi 的融合 QKV 使用同样的非零模式。真实权重分布和模型特有的长上下文行为仍需要另外验证。

非零测试建议使用 `--work-dir tmp/dense_nonzero_qk --results-dir results/dense_transformers_nonzero_qk`，保留零 Q/K 基线。参考计算额外保存 `golden.scores.bf16.bin`、`golden.probability.bf16.bin`；runner 校验 Q/K 非零、分数有限、最后一个 token 的注意力非均匀，并在 `result.json` 记录 `attention_coverage`。比较 CModel 输出的误差门限保持不变。

Phi-4-mini 的生成器明确表达 96/128 维 partial RoPE：前 48 维与后 48 维配对旋转，最后 32 维直接保留。LongRoPE 的 cos/sin attention factor 也在图中表达，lowering 从频率宽度和缩放算子提取参数。

当前硬件路径在打包 Q/K projection 权重时，对每个 head 的输出列做编译器选定的排列：`[0:48, 96:112, 48:96, 112:128]`。这使现有完整 128 维 VXM 配对布局的前 48 对执行 partial RoPE，剩余 16 对用 cos=1、sin=0 直通。Q 和 K 使用同一排列，QK 点积保持等价；V 和后续 context 不变。参考仍使用原始权重和原始维序，不使用 full RoPE 诊断参考替代验收。

该排列属于 C2C/SRAM 数据打包布局，不在 host 上计算 attention，不增加硬件 FU/ICU 操作。二进制版本 33 在 binding 元数据中增加 `rope_rotary_dim` 和 `rope_attention_factor`，旧版本仍按完整 head、factor=1 读取。当前 partial 路径支持无 Q/K bias、无 QK norm、无持久 KV 导出的 prefill；不支持的组合明确报错，避免缓存维序含义不一致。

Phi-4-mini 非零 Q/K seq32 整层复测通过：max_error=0.175781、mean_error=0.0012011，门限保持 max≤0.35、mean≤0.035。结合其余五个模型此前的通过结果，六个 fixture 均已通过非零 Q/K 数值测试；官方权重及长上下文仍未验收。

在 SOFTWARE 根目录运行（PowerShell）：

```powershell
python -B compiler/tools/run_dense_decoder_smoke.py `
  --model minicpm4_0_5b `
  --compiler build-ftlpu-vs2026-direct/compiler/ftlpu-compile.exe `
  --runtime-test build-ftlpu-vs2026-direct/runtime/hf_decoder_layer_model_session_test.exe `
  --work-dir tmp/dense_cmodel `
  --results-dir results/dense_transformers
```

默认读取相邻 CMODEL 仓库的 `config/ftlpu-lpu32.json`，使用 fused attention/FFN，关闭 projection/RoPE 流水重叠。可用 `--target-config` 指定目标，或用 `--executable` 测试已有编译产物。`--component attention` / `ffn` 将另一分支的输出权重置零，便于定位数值问题；默认 `both` 执行完整层。

Attention 的 Q/K/V/O 全部能驻留一个 bank 时仍使用分离 slice group。
若整组不能驻留、但各 projection 分别能驻留，则按执行顺序 Q → V → K → O
交替使用初始 bank → 另一 bank → 初始 bank → 另一 bank。只预装 Q；V/K/O
保留 `runtime_prefetch`，由运行时 C2C 在 SRAM 区域可复用后尽早启动。
这条路径不在 projection 时间轴里预留预测的 DDR 传输时长，实际缺页由各参与
ICU 的 page-ready 同步吸收。K 不会覆盖仍在读取的 Q，O 不会覆盖仍在读取的 V；
后续 Gate/Up 也通过同一物理驻留区间检查等待 Attention 权重释放。
当前每半球一个 MXM 的 FFN 固定执行完整 Gate → 完整 Up。Gate 需要补页
时由 page-ready 同步等待，不再根据 Attention 驻留重叠自动改成 Up 优先。
Down 页按 Gate、Up 的实际释放顺序复用权重区域。
编译器插入页 SYNC 时，在 host 元数据中记录粗域相位变化边界
（`icu.sync_origin.*`）。运行时重绑定页事件前按这些边界还原原始
`wait_cycle` 和地址重定位位置，避免重复插入同步后改变 MEM/MXM 的相对相位。
这部分不增加硬件 i-MEM 指令，也不展开 FU 指令；KV `MEM_READ_SYNC` 保留。
权重页预取严格按编译器 `ready_cycle` 的首次消费顺序执行，同周期保留
稳定顺序。单 MXM 的 FFN 因而是 Gate page0..N → Up page0..N →
Down page0..N。bank 或 SR 较早空闲不能让后消费的页插队；运行前预装
也只允许该消费序列的连续前缀。后页的 MEM_WRITE_SYNC 不会越过前页
的片内传输下界去占用更早的空窗，避免同步事件产生反向依赖。每次完整
执行一页，不交错执行两条 ICU 指令；实际外部完成仍由动态事件保护。
`dense_attention_ping_pong_test` 覆盖 bank 交替和 direct MEM_3D 到最终
`weight_page` 的预取标记传递。Llama3.2-1B 与 Phi-4-mini 已通过 seq32
整层合成非零 Q/K 数值复测，实际 CSV 确认 V/Q、K/V、O/K 的预取在
前序 projection 结束前启动，部分页已在计算开始前就完成。结果保存在 `results/dense_transformers_nonzero_qk` 对应模型目录的
`attention_ping_pong.json`、`pipeline.run.json` 和 `pipeline.html`。

每次更新对应结果目录，保存阶段日志、`result.json`、fixture manifest、参考元数据、编译产物和 `linked.ftlpu`。后者是本次真实执行的链接后 ICU 程序，包含 C2C 和运行时同步。设置 `FTLPU_DENSE_PIPELINE_CSV` 可导出完整运行时功能单元流水 CSV，设置 `FTLPU_SESSION_PROGRESS=1` 可打印仿真进度。

验证保持最大绝对误差 0.35、平均绝对误差 0.035 的上限；超过上限或产生非有限值即失败。失败时额外保存实际/参考 BF16 数据及已链接 ICU 程序用于定位。

本次适配涉及三个公共约束：MEM 页预取只占用实际可用的 ICU 空窗；跨半球 scalar residual 的临时写回与最终结果使用独立行，避免后续块覆盖前序结果；页同步按实际队列冲突边界组织。多个页有不同的首次使用时间，但若它们的预取占用了同一段 MEM 空窗，就使用同一个页就绪事件，并等待该事件的全部页提交。

启动窗口内的参与 ICU 以及该波 Native4 激活的 MEM 生产者先共同到达 SYNC，再按编译相位释放。下游阶段另外同步，避免两波激活重叠以及跨阶段死锁。Native4 生产者根据已编译的 MEM Read3D 与 MXM LoadActivation/直接 BF16 Compute3D 域、stream ID 和硬件 SR 延迟识别，不展开 FU 指令。启动之后保持该事件的时间原点，避免后续队列的到达延迟再次改变已经发出的计算与结果写回的相对时序。

在零等待的 ICU 指令边界插入 SYNC 时，该队列会多占一个 issue 周期；如果其他队列在 wait_cycle 中吸收该 SYNC，只使用链接后指令位置计算相位就会使计算与写回错开。释放相位因此扣除同步占用造成的累计漂移，保持编译时的 producer/consumer 间隔。回归测试同时覆盖有等待和零等待的跨域边界。

V projection 合并 WRITE_3D 的地址域按照每组四个 output block 推进，不能固定按一个 head 推进。head_dim=64 时，每个半球下一组相隔两个 KV head；head_dim=128 时相隔一个。Llama3.2-1B（8 个 KV head、head_dim=64）的完整数值测试覆盖该情况，MiniCPM 的两个 KV head 仅有一组，不能单独验证这个跨组步长。

所有 Attention projection 路径均不把 `page_transfer_cycles` 加入计算开始时间，
包括旧的非乒乓标记路径。该属性仅为传输分析元数据。FFN 的
`weightStartupDelay` 是 MXM 有限 weight buffer 的片内结果窗口约束，
不含 DDR 延迟；`FfnWeightTileTaskPlan` 的外部带宽估算只供分析测试，
不参与执行指令生成。
预取链接计划的 transport duration 也只计确定的片内 C2C 下界，不再
使用 DDR 预计周期占用 MEM 空窗或推迟下一页。运行时等待前一页真实
DDR/RX/MEM 写入完成才放行下一条页指令，消费者由各 ICU 的 page-ready
事件同步放行。DDR 频率、带宽、延迟和 jitter 的变化不改变编译的计算
周期及逻辑预取等待，实际物理停顿由 CModel 的完成事件决定。
回归测试覆盖旧非乒乓路径、串行多页 C2C 和不同 DDR 参数下逻辑计划一致。
