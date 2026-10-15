# Linux / WSL2 导出环境

HF → StableHLO 入口见 [前端指南](README.hf-stablehlo.zh-CN.md)。
PyTorch/XLA 导出在 Linux 上运行；Windows 使用 WSL2，并让编译器继续使用原有
Windows 工具链即可。导出只需要 CPU。

## 安装与复用

`requirements.xla-linux.txt` 列出直接依赖，`requirements.xla-linux.lock.txt`
记录实验环境的完整版本集合：Python 3.12.3、torch 2.9.0+cpu、torch-xla 2.9.0、
Transformers 4.57.1。BF16 原生保存兼容层针对 torch-xla 2.9.x；换版本需重新验证。

已有匹配环境可以直接复用。新建 Ubuntu 环境时，在 Linux 仓库根目录执行：

```bash
sudo bash tools/setup_xla_linux.sh
```

脚本在 `/opt/ftlpu-xla/venv` 安装锁定依赖并运行小型线性层 XLA 数值检查。
它可能安装系统 Python/venv 包，需要网络和 root 权限。
可通过 `FTLPU_XLA_HOME` 选择其他安装目录。
安装日志和实际依赖清单位于环境目录下的 `setup.log`、`requirements.lock.txt`。

手动创建环境也可使用：

```bash
python3 -m venv .venv-xla
.venv-xla/bin/python -m pip install -r tools/requirements.xla-linux.lock.txt
PJRT_DEVICE=CPU .venv-xla/bin/python tools/export_hf_stablehlo.py \
  --config tools/configs/smollm2_stablehlo.json --output artifacts/smollm2 --verify-numerics
```

## Windows 包装器

`tools/run_xla.ps1` 默认调用名为 `FTLPU-XLA` 的 WSL 发行版、root 用户和
`/opt/ftlpu-xla/venv/bin/python`。使用其他环境时可直接调用自己的 WSL Python。
在 Windows 仓库根目录执行：

```powershell
.\tools\run_xla.ps1 tools/export_hf_stablehlo.py `
  --config tools/configs/smollm2_stablehlo.json `
  --output artifacts/smollm2-stablehlo --verify-numerics
```

包装器保留脚本的退出码，把工作目录切到当前仓库，并转换命令行中已有的
Windows 绝对路径。JSON 内部的路径不会自动转换：例如 `E:\models\llama`
应写成 `/mnt/e/models/llama`；相对路径按仓库根目录解析。
默认 HF 缓存位于当前仓库 `.cache/hf-hub`。

模型可提前通过 `tools/download_hf_export_checkpoint.py --help` 查看下载选项，
或由 HF loader 首次导出时下载。离线配置要求对应模型及 revision 已缓存，
或 `model` 指向完整的本地目录。
