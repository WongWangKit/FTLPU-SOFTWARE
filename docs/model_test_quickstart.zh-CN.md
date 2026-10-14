# 模型测试统一入口

新同事使用 `tools/run_model_tests.py`。它复用现有 importer、compiler、runtime 和
CModel 数值验收，统一环境检查、命令参数、日志、退出码和结果目录。

**当前验收范围是 decoder 单层，不是完整多层模型的文本生成。**
测试报告会区分合成权重和官方权重，编译成功、权限跳过均不算数值通过。

## 第一次使用

1. 将 `FTLPU-SOFTWARE` 和 `FTLPU-CMODEL` 放在同一父目录。
2. 使用 Python 3.10+，安装依赖：`python -m pip install numpy requests`。
3. 按 [MLIR 编译器构建说明](../compiler/docs/mlir_build.zh-CN.md) 初始化固定版本
   submodule、构建 MLIR，再配置 FTLPU。已有完整构建环境可直接使用。
   Windows 请使用 MSVC 19.36+ 的 Developer Command Prompt，并保证 CMake/Ninja 在 PATH。
4. 在 `FTLPU-SOFTWARE` 中执行：

```powershell
python tools/run_model_tests.py list
python tools/run_model_tests.py doctor --model qwen2_5_1_5b
python tools/run_model_tests.py run --model minicpm4_0_5b --suite official --download --build
```

`--build` 只构建本次所需 CMake targets，不负责首次配置或安装工具链。
代码更新后建议带上它，使 compiler、runtime、CModel 和二进制 ABI 保持一致。
如果尚未构建，`doctor` 会列出缺少的工具和 target；它本身不运行数值测试。

脚本优先使用本仓库已配置的 `build`，其次尝试已存在的约定构建目录；显式指定更清楚：

```powershell
python tools/run_model_tests.py run --model minicpm4_0_5b --suite official --download --build-dir build-ftlpu-vs2026 --build
```

可以从其他工作目录通过脚本绝对路径运行；相对参数路径按调用者的工作目录解析。
Visual Studio 多配置生成器使用 `--config Release`，Ninja 单配置生成器使用已配置的构建类型。

## 选择模型和测试

| 模型 ID | `prefill` | `official` | 其他验收 |
|---|---|---|---|
| `qwen2_5_1_5b` | 完整尺寸合成权重，seq32 | 未接入此入口 | `decode`：past32 + decode1；`compile` |
| `qwen3_0_6b` | 完整尺寸合成权重，包含 Q/K norm，seq32 | 未接入此入口 | `compile` |
| `qwen3_5_0_8b` | 官方权重的 linear-attention 单层，seq32 | 同左 | `smoke`：缩小尺寸合成权重；`compile`：同一缩小尺寸图 |
| `llama3_2_1b` | 完整尺寸合成权重，非零 Q/K，seq32 | 官方 layer0 | `compile` |
| `ministral3_3b` | 同上 | 官方 layer0 | `compile` |
| `phi4_mini` | 同上 | 官方 layer0 | `compile` |
| `smollm3_3b` | 同上 | 官方 layer0 | `compile` |
| `glm_edge_1_5b` | 同上 | 官方 layer0 | `compile` |
| `minicpm4_0_5b` | 同上 | 官方 layer0 | `compile` |

`list` 是入口实际接入范围的清单，不表示所有组合本次都重新验收过。
原生 DeepSeek 和 GLM-5 暂未接入；GLM-Edge 的结果不代表 GLM-5。
Qwen3.5 此处不覆盖 full-attention 层、decode 或完整混合层模型。
`compile` 只生成 ICU image，不执行 CModel，不检查数值。

```powershell
# 默认 suite 是 prefill；可以一次选多个模型
python tools/run_model_tests.py run --model llama3_2_1b minicpm4_0_5b

# Qwen2.5 decode
python tools/run_model_tests.py run --model qwen2_5_1_5b --suite decode

# Qwen3.5 缩小尺寸数值测试，不需要下载权重
python tools/run_model_tests.py run --model qwen3_5_0_8b --suite smoke

# 官方权重数值验收：只在本地不存在时下载
python tools/run_model_tests.py run --model glm_edge_1_5b --suite official --download
python tools/run_model_tests.py run --model qwen3_5_0_8b --suite official --download

# 批量顺序验收本 suite 支持的模型，可能耗时较长
python tools/run_model_tests.py run --all --suite official --download --allow-gated-skip
```

官方下载保留原始 layer0 tensor 和前 32 个 token 的 embedding 行；记录固定 revision、
来源和 SHA256，导入阶段转换为硬件 W8A16。这不是完整 checkpoint 下载。
已有权重默认复用，不自动更新上游 revision。`--checkpoint` 接受单个经过来源记录的
原始权重子集目录，`--checkpoint-root` 接受批量模型 ID 子目录的父目录。
普通未记录 provenance 的任意 checkpoint 不会被此入口当作官方验收数据。

```powershell
python tools/run_model_tests.py run --model phi4_mini --suite official --checkpoint "D:/models/my-verified-phi-layer"
```

Llama 需要本机已获仓库访问权限的 `HF_TOKEN` 或 Hugging Face 缓存凭据。
默认下载失败使测试失败；只有显式 `--allow-gated-skip` 才允许将 Llama 的
`GatedRepo` 错误记为 `skipped_access`，不会跳过其他下载或数值错误。

## 结果和流水图

默认结果目录固定，每次同一用例更新同一位置：

- Qwen2.5 prefill：`results/qwen2_5_decoder_layer_prefill/`。
- Qwen2.5 decode：`results/qwen2_5_decoder_layer_decode/`。
- Qwen3.5 官方 prefill：`results/qwen3_5_0_8b_decoder_layer_prefill/continuous/`。
- 其他：`results/model_tests/<model>/<suite>/`。

自定义根目录用 `--output-root PATH`，输出到 `PATH/<model>/<suite>/`。
不创建带时间戳的额外目录。中间文件放在 `tmp/model_tests/`。

| 文件 | 用途 |
|---|---|
| `runner.json` | 统一状态、覆盖范围、权重类型、硬件配置 hash、命令、阶段耗时和数值结果 |
| `*.runner.log` | 各阶段的完整 stdout/stderr，失败先查看对应日志 |
| `summary.json` / `result.json` | 对应数值 backend 的报告（可用时） |
| `runtime.pipeline.csv` | 实际 CModel 执行流水；decode backend 使用 `pipeline.csv` |
| `pipeline.html` | 内嵌 CSV 的 Pipeline Viewer，直接用浏览器打开 |
| `linked.ftlpu` 或 backend 的 runtime image | 实际执行的 ICU image（backend 支持时） |
| `icu_programs/` | 每个 ICU 的实际指令 CSV，包含可用的最初传输 snapshots |

Qwen3-0.6B backend 当前可以导出运行流水，但未提供实际 linked ICU image，
入口不会拿静态编译 image 冒充实际执行指令。
批量汇总为 `results/model_tests/last_run.json`；自定义根目录时放到该根目录下。

退出码：`0` 表示所选用例通过、仅编译成功或显式允许的权限跳过，需结合报告状态判断；
`1` 表示执行、数值或所请求的产物失败；`2` 表示参数、环境或构建检查失败。
`numeric_passed` 只有执行并通过数值验收时才为 true；数值通过但流水或 ICU 导出失败时，
该字段保留 true，整体 `status` 仍为 `failed`，退出码为 `1`。

## 常用诊断

```powershell
# 只列执行计划，不创建目录、不下载、不构建
python tools/run_model_tests.py run --all --suite official --download --dry-run

# 数值检查仍启用，关闭可选流水记录/ICU 导出
python tools/run_model_tests.py run --model minicpm4_0_5b --no-pipeline

# 入口自身的离线回归测试
python tools/tests/test_run_model_tests.py
```

decode backend 的流水是其原生验收所需产物，`--no-pipeline` 只关闭入口额外导出。
完整尺寸 CModel 仿真可能很慢，运行中不要重复启动同一用例覆盖它的目录。
若日志提示 `unsupported FTLPU binary version`，在正确的 Developer Command Prompt 中
加 `--build` 重建所需 targets；不要禁用数值检查来绕过错误。

入口会移除继承的跳过数值、只链接、提前停止和 DDR 参数覆盖环境变量，
按 `--target-config` 指定的硬件配置执行。配置默认取 CMake cache 中的
`FTLPU_HARDWARE_CONFIG`；如与 runtime 构建配置不一致，`doctor` 会拒绝执行，
请为新配置单独配置并构建目录。

新增模型需要登记入口的 `CATALOG`，接入对应 importer/compiler/runtime 数值 backend，
声明真实覆盖范围，并补充可复现的 golden 对比。只有登记一个模型名称不能算支持。

## 本次入口验收记录（2026-10-14）

- MiniCPM4-0.5B 官方 layer0：数值通过，相对 L2 为 `0.00894323`、cosine 为
  `0.99996013`；导出实际 MEM/MXM/VXM/SXM/C2C 流水及 ICU 指令。
- Qwen2.5-1.5B 合成 seq32：当前 compiler/runtime 在 projection 数据写回时报
  `MEM Write reached an invalid stream segment`（cycle 8457，E slice16 bank0），
  入口如实返回失败。这一路径需要修复后再声明当前版本通过。
- 其余模型本次未通过该新入口重新执行，不能将 `list` 中的注册范围视为本次全通过。

建议新同事先用上述 MiniCPM 官方权重命令验证完整环境，再按 `list` 选择所需回归。
