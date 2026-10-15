# 一次配置 Linux 导出环境，复用验证过的 bundle

2026-10-14 已在本机完成配置和真实导出。Windows 入口统一为
`tools/run_xla.ps1`，无需每次安装依赖或手动激活环境。

## 集中位置

| 内容 | 位置 |
| --- | --- |
| 专用 WSL2 发行版 | `FTLPU-XLA`，Ubuntu 24.04，Python 3.12.3 |
| WSL 磁盘 | `.cache/wsl-xla/ext4.vhdx` |
| 固定 Python 环境 | 发行版内 `/opt/ftlpu-xla/venv` |
| Linux Python 字节码 | `/opt/ftlpu-xla/pycache`，避免向 Windows 源码缓存目录写入 |
| 安装日志、依赖锁 | `/opt/ftlpu-xla/setup.log`、`/opt/ftlpu-xla/requirements.lock.txt` |
| 共享官方 HF cache | 仓库 `.cache/hf-hub/` |
| 已验证产物 | 仓库 `out/hf-bundles/` |
| 产物索引 | `out/hf-bundles/verified_bundles.json` |

环境使用 torch `2.9.0+cpu`、torch-xla `2.9.0`、Transformers `4.57.1`、
Safetensors `0.8.0`、NumPy `2.3.5`、ml_dtypes `0.6.0`。
完整依赖锁已保存为 `tools/requirements.xla-linux.lock.txt`。
现有 Docker 发行版和 WSL 默认发行版保留。

环境已经持久保存于 WSL 磁盘。日常使用下述入口即可，不必重跑 setup。
只有需要重建环境时，才在 Linux 中运行
`bash /path/to/FTLPU-SOFTWARE/tools/setup_xla_linux.sh`；默认安装到 `/opt`，
应使用该专用发行版的 root 用户。

## 已验证 bundle

| Bundle | 规格 | 重载数值最大绝对误差 | 后端结果 |
| --- | --- | ---: | --- |
| `smollm2-135m-f32-b1-s16-xla2.9.0-20261014` | 完整 logits forward，float32，输入 `1×16`，输出 `1×16×49152` | `1.3828278e-4` | rank-3 修补后 Kernel pass 通过，所有 dot 已转换；仍含其他 StableHLO。Tensor 阶段无法分配函数输入 |
| `smollm2-layer0-mlp-f32-s16-xla2.9.0-20261014` | 真实 layer 0 MLP，float32，输入/输出 `16×576` | `1.9073486e-6` | Kernel pipeline 通过，3 次 dot 转为 kernel.matmul；仍保留 transpose/logistic。Tensor 阶段无法分配函数输入 |

两者均来自 `HuggingFaceTB/SmolLM2-135M`，固定 checkpoint commit 为
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`。使用无量化默认路径，没有重写
模型数学。MLP 入口只通过通用 `submodule_entry` 选择真实 HF 子模块。

完整 bundle 约 656 MB，含 275 个捕获状态 tensor、277 个实际函数参数位置、
1 个 exporter 额外常量，两个用户输入均被实际函数使用。它包含：

- XLA 原生 `functions/`、`data/`、`constants/`，以及 manifest、捕获签名和配置；
- 实际加载/运行 dtype、checkpoint 存储 dtype、实际 cast 和工具版本；
- `report.json` 中的成功数值检查及完整阶段耗时；
- `validation/reference_*.npy`：配置样例输入对应的真实框架参考输出；
- `bundle_checksums.json`：图、metadata、参数、常量、配置和参考输出的 SHA256。

只有成功完成重载数值比较的产物才会生成校验记录。参数即使仅改一个值，
shape/dtype 不变，原有校验记录也会失效。校验范围是这些产物的完整性和
配置样例的数值一致性，不代表所有可能输入都已通过测试。

## 日常复用

在仓库根目录的 PowerShell 中直接执行：

入口固定在仓库根目录运行 Python，也支持独立参数形式的 Windows 盘符绝对路径，
例如 `--bundle E:\...\bundle`，会通过 `wslpath` 转为 Linux 路径。

```powershell
& tools/run_xla.ps1 tools/run_saved_hf_bundle.py `
  --bundle out/hf-bundles/smollm2-135m-f32-b1-s16-xla2.9.0-20261014
```

这会校验完整性，通过原生 `StableHLOGraphModule.load()` 读取保存的图和参数，
生成配置中的样例输入并与保存的框架参考结果比较。**不会重新加载 HF 模型、
捕获图或执行前端 StableHLO 转换**。XLA CPU runtime 执行 bytecode 时仍可能
进行设备编译，这是开源 runtime 的执行行为。

两个真实 bundle 都已在独立进程中复用成功。完整 forward 的一次复用耗时
约 17.32 秒，其中包含文件完整性检查和 CPU runtime 初始化；MLP 约 0.73 秒。

Windows 上仅校验/消费图时也可使用当前 Python：

```powershell
python tools/export_hf_stablehlo.py `
  --validate-bundle out/hf-bundles/smollm2-135m-f32-b1-s16-xla2.9.0-20261014

build-codex-macro-vs-verify/compiler/Release/ftlpu_opt.exe `
  --input out/hf-bundles/smollm2-layer0-mlp-f32-s16-xla2.9.0-20261014/functions/forward.mlir `
  --output out/mlp.kernel.mlir --pipeline ftlpu-stablehlo-to-kernel --kernel-mode partial
```

该 Kernel 命令用于检查局部转换。当前完整模式会拒绝其中尚未支持的计算；
partial 输出不能进入 Tensor 及后续后端。表中 Tensor 内存错误为早期验证记录。

复制 bundle 时复制整个目录，保留相对结构。保存产物的执行和 compiler 消费
不要求原始 checkpoint cache；只有重新导出时需要它。bundle 是前端逻辑协议，
尚未打包为硬件 runtime 的 ModelPackage。

## 重新导出与模型下载

现有 bundle 不覆盖。修改形状、模型、入口或 dtype 后应指定新的输出目录：

```powershell
& tools/run_xla.ps1 tools/export_hf_stablehlo.py `
  --config tools/configs/smollm2_stablehlo.verified.json `
  --output out/hf-bundles/smollm2-new-shape --verify-numerics

& tools/run_xla.ps1 tools/export_hf_stablehlo.py `
  --config tools/configs/smollm2_mlp_stablehlo.verified.json `
  --output out/hf-bundles/smollm2-mlp-new-export --verify-numerics
```

本机 WSL 可访问安装源，但直接访问 Hugging Face 失败。已用 Windows 的官方
HF `snapshot_download` 写入共享 cache，Linux 的验证配置使用
`local_files_only=true`，可以离线导出。下次接入新模型，可先在安装了
`huggingface_hub` 的 Windows Python 中下载固定 commit：

```powershell
python tools/download_hf_export_checkpoint.py `
  --model HuggingFaceTB/SmolLM2-135M `
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
```

该脚本只使用官方 HF cache API，下载配置和 Safetensors checkpoint，不解析
或改写权重。原先的联网失败产物保留在 `*-network-failed` 目录中。

## 验证边界

Linux 前端回归 **38 项通过，没有 XLA 跳过项**，包含真实 HF/XLA save、重载
数值验证，以及禁止 HF loader/torch.export 被调用时的保存 bundle 复用测试。
Windows 上的完整 bundle 校验也通过。

完整 forward 的默认输入、变化 token、尾部 mask 均通过
`rtol=1e-4, atol=1e-5`。额外左 padding 测试没有达到这个全量严格标准：
3405/786432 个输出元素超阈值，主要位于被 mask 的 query；有效 token 的
638976 个输出元素中仍有 3 个超阈值，这 3 个元素的最大绝对误差约 `2.29e-5`。
全量最大绝对误差 `2.7275e-4`。没有放宽既有容差，也没有删掉 mask 或输出；
该情况不能称为通过。原始统计在
`out/hf-bundles/smollm2-135m-additional-cases-20261014.json`。

这次完成了真实前端导出与保存复用。完整模型的后端 lowering、FP32 子图的
物理内存规划以及左 padding 的严格数值一致性仍有明确限制。后端错误保存为
`*.backend*.stderr.txt` 和 `*.backend*.report.json`，没有为迁就后端而改写模型。

2026-10-14 的局部 rank-3 修补复用了同一 bundle，没有重新导出或改写参数。
完整图的 272 条 dot 转换为 212 条 `kernel.matmul` 和 60 条
`kernel.batch_matmul`，函数输入输出类型保持一致。Kernel pass 输出仍是混合 IR，
不能视作完整硬件 lowering 成功；Tensor 阶段报
`cannot allocate function inputs in the east MEM hemisphere`。
原始错误、Kernel IR 和结构化报告在
[`out/hf-lowering/rank3-fix-20261014`](../out/hf-lowering/rank3-fix-20261014/)，
修补范围与复现命令见 [rank-3 lowering 记录](../docs/hf_rank3_lowering.zh-CN.md)。

后续连接工作已在同一个环境中完成真实 SmolLM2 decoder layer-0 的独立 W8A16/BF16
bundle，以及完整 Kernel → Tensor → Stream → Schedule → Command → ModelPackage →
CModel 路径；保留上述 FP32 forward bundle。现在默认 Kernel 入口会拒绝混合计算，
旧 rank-3 局部实验须显式使用 `--kernel-mode partial`。运行、数据及限制见
[连接验证报告](../docs/hf_connection_validation.zh-CN.md)。环境没有重新安装。
