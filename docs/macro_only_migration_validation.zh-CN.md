# Macro-only ICU 迁移验证（2026-11-13）

## 本轮范围

MEM/MXM/SXM 计算指令统一为 packed Macro；移除旧硬件 2D/3D 循环包、STREAM_ND、MEM slice program、SXM tile program。VXM 保持 96-bit 顺序 CONFIG。编译器内部仿射调度域与 pipeline CSV 的 repeat2d 聚合格式不是硬件 ISA。

SXM Transpose/Permute 保持独立 ICU；沿用固定 reservoir/window/DDB、96-bit iMEM，LPU32 context 上限分别 256/384 bit。同一队列不允许不同 Macro 交错执行。SYNC 和 C2C 传输包保留。

## 模型范围与精度

DeepSeek V4.1 Flash 第 0 层，batch1、seq32：hc_pre fixture → RMSNorm → Q_A/KV → Q_A/KV RMSNorm → Q_B → partial RoPE → 带 sink 的 causal SWA core。输入 [32,5120]，最终 context [32,32768]，同时检查 8 个诊断中间输出。没有持久 State 输出，不包含 mHC、输出投影、FFN 或 KV cache 写回，不代表完整 decoder/模型部署。

官方 E4M3FN 权重及每 K32 的 E8M0 scale 保留，MXM IW 反量化至 BF16；激活/输出 BF16，RMSNorm 和投影累加 FP32。原阈值不变：整段 relative L2 ≤ 0.01、cosine ≥ 0.9999；局部投影 BF16 bit mismatch = 0，RoPE 非旋转前缀 bit mismatch = 0。

固定结果目录 `results/deepseek_v41_flash_layer0_swa_core_prefill`；迁移前对照在其 `baseline_before_macro_only` 子目录。

## 构建与回归

MSVC 构建使用 UTF-8 控制台，避免中文 /showIncludes 与 Ninja 前缀编码不一致。旧构建目录若没有头文件依赖记录，应执行 clean-first。本轮两个仓库均已做干净构建。

从 SOFTWARE 目录运行：

```powershell
.\tools\build_vs2026.cmd --build build-ftlpu-r-direct --clean-first --parallel 4
ctest --test-dir build-ftlpu-r-direct --output-on-failure -j 2 -I 1,58
ctest --test-dir build-ftlpu-r-direct --output-on-failure -R "direct_raw_macro_cli|macro_domain_lowering|standalone_ffn_up|mem_3d_blocked_outer|depthwise_conv1d_binary|sxm_binary|vxm_sequential|command_cmodel_compat"
```

CModel：`ctest --test-dir build-macro-only --output-on-failure`。

覆盖 packed image 逐位往返、实际 decoder 逐周期 replay、SXM 两种队列、Permute 模板供给不足反例、禁止同队列交错、重定位/队列控制、iMEM 容量、C2C SRAM 提交、wide transpose 和 RoPE 数值。

CLI 数值回归采用小 SwiGLU FFN（32×64→128→64，tail 排程），同时检查默认/显式 raw-macro 的等价性；Qwen2.5 seq32 检查分页整层编译。历史小 FFN fused 排程存在同队列读写区间重叠，本次没有放宽冲突检查。

## SWA 复现

从 SOFTWARE 目录运行：

```powershell
python compiler/tests/deepseek_v41_attention_input_test.py --build-dir build-ftlpu-r-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --package results/deepseek_v41_flash_layer0_qkv_prefill --norm-package results/deepseek_v41_flash_layer0_attention_rmsnorm --model-config results/deepseek_v41_flash_assessment/official_config.json --output-dir results/deepseek_v41_flash_layer0_swa_core_prefill --with-swa-core --swa-package results/deepseek_v41_flash_layer0_swa_prefill --rope-tile-overlap
```

产物：`summary.json`、`runtime.pipeline.csv`、`actual.runtime.ftlpu`、`runtime_icu_programs/index.csv`。每个 ICU CSV 指向实际装载的主映像、pre-execution 映像或 execution-segment 映像，物理指令大小以对应 packed 文件为准。

## 最终结果

两个仓库均构建通过。SOFTWARE 本轮选择的 71 个回归中 68 个通过；CModel 全部 66 个 CTest 中 59 个通过。迁移相关的 Macro 编码/实际解码、SXM、队列控制、容量、C2C 提交与预取测试通过；不能将此结果表述为全部历史模型回归通过。

本轮最终 SOFTWARE 合并回归命令（从 SOFTWARE 目录执行）：

```powershell
ctest --test-dir build-ftlpu-r-direct --output-on-failure -j 2 -I 1,58 -R "direct_raw_macro_cli_test|standalone_ffn_up_3d_compile_test|depthwise_conv1d_binary_runtime_test|macro_domain_lowering_test|depthwise_conv1d_direct_lowering_test|mem_3d_blocked_outer_test|qwen2_5_1_5b_decoder_layer_seq32_vector_weight_bank1_test|ftlpu_command_mxm_column_binary_test|ftlpu_command_cmodel_compat_binary_test|ftlpu_command_bf16_binary_test|qwen2_5_1_5b_decoder_layer_seq32_vector_ir_test|ftlpu_command_vxm_binary_runtime_test|gated_delta_rule_ir_lowering_test" -U
```

### SWA 数值及同条件时延

9 个实际输出与迁移前备份逐字节一致。最终 context 对参考的 relative L2 为 0.0036859346254018007（0.3686%），cosine 为 0.9999932312002863；局部 SWA relative L2 为 0.0024036897914377817。没有修改精度或放宽阈值。

500 MHz、51.2 GB/s、带宽效率 90%，迁移前后结果如下：

| 项目 | 迁移前周期 | 迁移后周期 |
| --- | ---: | ---: |
| 计算（扣除权重等待） | 961,303 | 961,303 |
| 权重页等待 | 24,950 | 24,950 |
| 初始数据读入 | 37,426 | 37,426 |
| 含初始读入及等待，至计算完成 | 1,023,679 | 1,023,679 |
| 诊断回读 | 650,794 | 650,794 |
| 含诊断回读的 trace 结束 | 1,674,473 | 1,674,473 |

计算完成时延为 **2.047358 ms**，不含诊断回读。指令 DMA 装载耗时仍未计入该时延。此轮是 ISA/执行链迁移，未改变算子计算排程。

独立 wide-SXM RoPE forward/inverse [32,8,512]、rotary64、offset448 也通过，relative L2=0，非旋转前缀 bit mismatch=0。

### 实际指令与分页

导出更新到真实运行包 `actual.runtime.ftlpu`，每 ICU 一个 CSV；包含主计算、pre-execution、execution-segment 与 execution-transport 的实际物理映像。现有导出校验覆盖 300 个映像的字宽、字数、Macro 描述符计数，共 75,290 个物理字、1,137,060 字节。该合计包含多阶段装载，不等于任一时刻的 iMEM 占用；MEM/SXM 为 96-bit，MXM 为 128-bit。

CSV 已包含全部 **480 条 MEM_WRITE_SYNC**，同时保留普通 SYNC 与 launch SYNC；transport 行注明其局部周期域，不能直接将其数字当作全局 FU issue cycle。

程序内动态分页沿用 common barrier；跨 invocation 预取保留，但只允许插入没有计算 Macro 的 transport 队列。与当前仍存活的 scratch 共用 bank 时，必须等待边界释放。相关测试包含非冲突预取可隐藏、冲突预取需等待及禁止向计算 Macro 队列插入传输的检查。已删除旧的逐域拆分/2D/3D 分页实现。

### 剩余历史用例问题

| 仓库/用例 | 当前失败原因及边界 |
| --- | --- |
| SOFTWARE: attention_target_layout_test | 仍断言 FFN 必须分配 distributed16 临时区；当前布局为 planar。布局实现与本轮起点备份相同。 |
| SOFTWARE: elementwise_vxm_sr_direction_test | 旧 explicit-SR 属性检查异常退出 0xc0000409；对应 emitter 本轮仅修改注释。 |
| SOFTWARE: gated_delta_rule_ir_lowering_test | 旧 kernel 传入 explicit-SR tap/terminal 控制，当前 chain VXM 不编码这些字段；该拒绝条件在本轮起点已存在。 |
| CModel: kernel.projection.w8a16、kernel.swiglu.w8a16 | 旧 fixture 访问 accumulator row1536，硬件仅128行。 |
| CModel: model.smollm2.prefill_attention | 旧 fixture 访问 accumulator row512，硬件仅128行。 |
| CModel: kernel.rmsnorm、model.smollm2.decode_attention | Bundle 活动期间重配 VXM input Buffer。 |
| CModel: model.decoder.decode_layer | MXM IW 先于对应 SR 权重到达。 |
| CModel: model.smollm2.decode_ffn | SRAM row 越界。 |

CModel 上述 7 项还分别运行了原有 `build-vs2026/Release` 可执行文件，均同样失败（个别旧可执行文件在更早的阶段失败）。没有通过扩大 SRAM/accumulator、放宽 Buffer 约束或恢复旧 ISA 来绕过这些问题。本轮没有将这些历史模型用例重写为新的模型部署验收。

### 可追溯产物

- [SWA 数值摘要](../results/deepseek_v41_flash_layer0_swa_core_prefill/summary.json)
- [逐输出位一致对比](../results/deepseek_v41_flash_layer0_swa_core_prefill/macro_only_comparison.json)
- [runtime pipeline](../results/deepseek_v41_flash_layer0_swa_core_prefill/runtime.pipeline.csv)
- [实际 ICU program 索引](../results/deepseek_v41_flash_layer0_swa_core_prefill/runtime_icu_programs/index.csv)
- [迁移验证摘要](../results/icu_macro_only_validation/summary.json)
- [SOFTWARE 最终回归日志](../results/icu_macro_only_validation/software_regression.log)
- [CModel 干净构建后回归日志](../results/icu_macro_only_validation/cmodel_regression.log)

旧 ISA 文档加了历史说明；保留的编译器多维几何域、数学算子名称与 viewer 的 `repeat2d` 聚合格式都不是硬件 2D/3D 执行入口。生产代码扫描已无 RUN_2D/RUN_3D、Fu3D、Repeat2D、STREAM_ND、SxmRun2D、MemSliceProgram、SxmTileProgram 的实现引用。
