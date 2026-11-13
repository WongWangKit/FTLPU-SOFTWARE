# 统一测试入口与按需 DDR 输出

统一入口为 `tools/run_model_tests.py`。它负责选择测试、检查依赖、记录命令和验收结果；布局、分页、融合、排程及 ICU 指令仍由 C++ 编译器生成，执行仍为 raw-macro CModel。它不改变数据类型或原有数值阈值。

## 输出契约

`compiler/tools/model_test_outputs.py` 定义四类输出，必须逐项声明：

| 用途 | 正常运行写 DDR | 示例 |
|---|---|---|
| `result` | 是 | 单层测试最终 hidden |
| `persistent_state` | 是 | KV cache、GDN recurrent state、conv state |
| `handoff` | 是 | 分成多次设备调用时，下一次调用需要的 Q/K/V、context |
| `diagnostic` | 否 | 只用于局部数值验收的中间 context |

不能仅因为一个张量是“模型中间值”就删除它的回写。跨调用输入尚未接入片上保留 ABI 时，仍须作为 handoff 输出；在同一编译图内消费的中间值由编译器规划，不声明为外部结果。

图测试通过 `ftlpu.external_output_indices` 把契约传给 C++ 编译器。正常运行只导出所选结果，诊断运行导出所有验收结果；比较两次运行的原生计算指令、图输入哈希，并逐字节检查共有的必需输出。契约未生效、漏写状态、残留旧输出或缺少实际 ICU/pipeline 均失败。

旧整层测试使用已有 ModelPackage 的 external_output 和 state binding：只暴露最终 hidden 与持久状态，不启用 `--checkpoint-outputs`。Qwen3.5 整层的三个结果是 hidden、recurrent state、conv state，均是必需输出。独立运行的旧调试脚本不自动继承统一入口的约定；新测试应注册到入口并显式声明契约。

## 运行方式

在仓库根目录运行，`<build>` 是编译器和 CModel runner 所在的构建目录：

```powershell
python tools/run_model_tests.py list
python tools/run_model_tests.py doctor --model glm4_9b --suite attention-prefill --build-dir <build>
python tools/run_model_tests.py run --model glm4_9b --suite attention-prefill --build-dir <build>
python tools/run_model_tests.py run --model glm4_9b --suite attention-decode --batches 1 32 --build-dir <build>
python tools/run_model_tests.py run --model glm4_9b --suite decode --batches 1 32 --attention-strategy flash --build-dir <build>
python tools/run_model_tests.py run --model qwen2_5_1_5b --suite prefill --build-dir <build>
```

可加 `--target-config` 指定对应硬件配置，`--dry-run` 查看命令及输出契约。GLM decode 固定 past_len=32，每请求生成一个 token；入口不接受 batch64。attention 比较固定运行 dense 和 Flash 两种实现。

attention-only 复用经过哈希校验的设备 QKV fixture：prefill 默认 `results/glm4_9b_prefill_attention_comparison/projections/fixture.npz`，decode 默认 `results/glm4_9b_decoder_layer_decode_flash`。可用 `--fixture` 覆盖。fixture 不存在时先准备原测试输入，入口不会用随机输入冒充真实权重结果。projection 的耗时不计入这两个 attention-only suite。

## 结果和计时

- `runner.json`：本次范围、命令、工具/输入信息、状态和各子测试路径。
- `output_contract.json`：必需输出、诊断输出及执行证据；不能把 `executed` 当成数值通过，最终数值结论看 `result.json`/`runner.json`。
- 正常目录的 `runtime.pipeline.csv`、`actual.runtime.ftlpu`、`icu_programs/`：实际正常执行。
- `diagnostic/`：独立诊断的数值文件、runtime pipeline 和实际 ICU program。
- `phase.timing`：正常执行，包括输入读入、设备执行、外存等待和必需输出回写；`phase.diagnostic_phase` 单独记录诊断运行，不加进正常时延。等待包含在设备窗口内，不能再相加。

默认更新固定目录：prefill attention 在 `results/glm4_9b_prefill_attention_comparison/{dense,flash}`，decode attention 在 `results/glm4_9b_decode_attention_comparison/{dense,flash}/batchN/attention`。这些 suite 总是保留 trace 和 ICU program；旧整层测试仍支持 `--no-pipeline`。

## 本次验证范围（2026-11-11）

入口/契约回归覆盖错误透传、旧结果失效、状态保留、handoff 保留、诊断计时隔离和实际产物检查。真实 GLM seq32 prefill 与 past32 batch1 decode 的 dense/Flash attention 核用同一已验证构建执行。旧整层入口已保留原后端和数值判定，本次未重跑所有模型整层。

本次构建目录为 `tmp/glm4_prefill_resident_outputs_1111`，硬件配置为 `../FTLPU-CMODEL/config/ftlpu-lpu32.json`（500 MHz / 51.2 GB/s）。使用上文命令将 `<build>` 替换为该目录即可复现；decode 本次传 `--batches 1`。

| attention 核 | 正常执行周期 | 包含全部验收输出的诊断执行周期 | context 相对 L2 |
|---|---:|---:|---:|
| prefill dense | 33,669 | 66,477 | 2.9106% |
| prefill Flash | 77,963 | 116,430 | 2.9079% |
| decode batch1 dense | 51,182 | 52,240 | 2.4424% |
| decode batch1 Flash | 18,675 | 19,225 | 2.4331% |

22 项入口/契约单元回归及 CModel 输出选择回归通过；后者还覆盖全导出、部分导出、无导出、非法索引和两 KV tile 的 Flash decode。8 份实际程序的物理 iMEM 检查通过，同一队列 FU 3D 活跃上下文不超过 1。汇总证据在 `results/model_tests/output_contract_acceptance.json`，可运行以下命令重新审计：

```powershell
python compiler/tools/summarize_model_test_outputs.py --imem-check tmp/glm4_prefill_resident_outputs_1111/check.exe
```

回归中发现旧 row-scale 快路径误用于 mirrored FP32 running numerator，导致 Flash decode 数值失败；该布局恢复到原有 planar 路径，partitioned producer 的 prefill 快路径保留。未改变精度和误差阈值，增加了包含状态输出的多 tile 回归。
