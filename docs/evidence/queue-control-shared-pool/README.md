# 共享控制池验证证据

2026-11-02，SOFTWARE `6f3473e`、CModel `c0988ab` 加当前未提交修改，均位于各工程的 `out/qctl-shared-pool` worktree。未合入 main 的并行开发工作。

- `native-*.txt` / `pseudo-*.txt`：25,600 MB/s 下完整 seq32/seq128 attention＋FFN，包含数值检查和实际流水重叠统计。
- `*-trace-comparison.json`：相同 RUN 起点下的物理周期对照，不对队列单独平移。Read3D 等名称及前端诊断字段归一化；具体算法见 `compare.py`。
- `*-image-comparison.txt`：最终计算和搬运镜像、操作数、relocation、事件及释放相位对照，以及最终镜像独立装载校验。
- `*-storage.txt`：最终文件、执行码流和控制表统计。
- `binary-tests.txt` / `legacy-binary-tests.txt`：新容器、四种 reader、容量/供给拒绝及旧 BARR 容器回归。
- `cmodel-tests.txt`：共享池早晚事件对照、有限端口/缓存及原有屏障测试。
- `artifacts.json`：原始 CSV、队列进度和二进制的字节数及 SHA-256。大文件留在本 SOFTWARE worktree 的 `out/validation`。

## 复现

`build-pool.cmd` 使用当前机器的 MSVC、LLVM/MLIR 依赖及两个隔离 worktree 的绝对路径，生成 SOFTWARE 主入口下独立的 `out/qctl-pool-build`，迁移机器时需调整这些路径。脚本和以下验证会生成构建产物及日志。

构建完成后，以 Python 依次运行本目录的 `validate.py test`、`validate.py compile 32`、`validate.py run 32`、`compare.py 32`，对 128 重复 compile/run/compare。脚本从自身位置向上查找本 SOFTWARE worktree；CModel 位置按同级工程的 `out/qctl-shared-pool` 推导。`run` 清理继承的 FTLPU 环境变量，并固定运行带宽为25,600 MB/s。

CModel 使用该 worktree 的 Release 构建，运行 `icu_queue_control_test` 和 `icu_barrier_test`。旧容器测试为 SOFTWARE 构建的 `binary_barrier_test`。存储统计使用 `ftlpu-binary-inspect` 的对应统计入口。

PRIME 不包含在主执行轨迹中，单独计费：共享池均为296拍。未重跑51,200 MB/s和完整延迟扰动矩阵；尚未集成 main 上并行开发的定时 DMA 完整程序装载协议，未进行 RTL 综合。
