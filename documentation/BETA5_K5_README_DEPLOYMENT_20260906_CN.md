# k5 beta5：README 更新与安装版本说明

日期：2026-09-06。这是机器部署补充，不是通用安装教程。
从零安装请阅读 [README_CN.md](../README_CN.md) 或 [README.md](../README.md)。

## 本次修改范围

更新两份 README，新增三个演示天线的输入文件。未修改物理算法、依赖配方、
构建脚本或已安装程序；未执行编译、shower 或 GPU 探针。当前用户明确要求
不使用 k5 显卡，应显式选择 `--backend openmp`；不要使用 `auto`、不带后端
限定的 `--check-backends` 或 CUDA `--dry-run`，它们可能初始化 GPU。

## 已核实的安装快照

从服务器 CMakeCache、安装清单和文件信息读取，未重新测试 GPU：

| 项目 | 已有内容 |
|---|---|
| 项目结构 | `corsika8_kokkos_beta5/`、`build/`、`install/` |
| 已安装后端 | `install/openmp`、`install/cuda` |
| CUDA 目标 | `AMPERE80` / 80，Kokkos Serial host，匹配此前确认的 A100 |
| OpenMP | CPU 执行后端，不初始化 GPU |
| 模型/类型 | 两套 Release、`WITH_FLUKA=ON`；应用默认 SIBYLL-2.3d |
| 外部模型复用 | `build/external/pythia8`、`build/external/tauola` |
| 物理源 | PROPOSAL 原生样条，Kokkos 4.7.03 |
| CUDA 文件时间 | 2026-09-05 18:44:05（服务器时间） |
| OpenMP 文件时间 | 2026-09-05 19:12:54（服务器时间） |

安装二进制 SHA-256：

```text
CUDA:   8f286f0d2dad81c721de913d3ce88608358e9ef86993d82b845a6e76b9756ad4
OpenMP: 95a06d805790dd9a823883f17d180355402cce55d1fda48357b7d7a5a211052a
```

编辑 README 不会升级这些程序。本地 9 月 6 日后续 RNG/物理修复及其验收，
不能仅凭项目名自动归属于以上 k5 二进制。需要另行同步算法、构建、安装
并核对新版本后才能使用对应的验收结论。

## 通用教程与 k5 历史部署的区别

新 README 从空环境构建，不假设已有外部模型或缓存。k5 的原部署因旧 Conda
中 Conan 依赖问题和网络限制，曾使用 `build/tools`、
`build/audit/k5_setup/conan` 包装器以及离线依赖归档；这些不是新用户的必需
目录，也不应不加说明地替代教程的全新环境流程。

历史 CUDA 安装使用 CUDA 13 工具链；通用教程的 CUDA 12.6.3 是另一套起步
配置，不是要求在 k5 降级或重复安装。k5 曾出现 NVML 与已加载驱动版本不
匹配，需管理员协调。显式 `cuda-ampere80` 只绕过构建阶段的架构探测，不能
修复驱动或证明 GPU 运行正常。旧部署脚本包含 GPU 构建/测试，当前不要运行。

旧说明 [BETA5_K5_SETUP_CN.md](BETA5_K5_SETUP_CN.md) 记录的是 9 月 5 日首次
CUDA 部署，其“只有 CUDA”描述不代表现状；本次核查已有 OpenMP 安装。
后续更新程序时应记录实际 compiler/profile、源码版本和二进制哈希，而不是
把新的 README 时间当作程序版本。

## 本次验证范围

核对源码中的 CLI、默认值、Conan 配方、构建脚本和天线坐标解析；检查文档
命令的 Bash 语法、内部链接及演示天线格式。同步前备份远程旧 README，
同步后核对文件 SHA-256 和安装二进制未改变。
这不是一次干净环境全编译或新的物理/性能验收。
