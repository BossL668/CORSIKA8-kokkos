# EGS4 电磁模块

EGS4 是可选的原生 C++/Kokkos 电磁后端，处理电子、正电子和光子；强子模型与 μ 子物理仍复用 C8 的现有模块。它不是对 C7 Fortran 电磁程序的调用包装，也不改变默认的 PROPOSAL 后端。

## 代码布局

| 位置 | 职责 |
| --- | --- |
| `corsika/modules/EGS4.hpp` | 模块公共入口 |
| `corsika/modules/egs4/` | 会话、表、显存预算以及 C8 粒子路由接口 |
| `corsika/detail/modules/egs4/` | 电磁反应、输运、散射、常驻队列、薄化、沉积及射电累加的内部实现 |
| `src/modules/egs4/` | 编译源文件、表内嵌模板和 `CORSIKA8NativeEgs4` 库目标 |
| `applications/detail/air_shower_kokkos/Egs4AirShowerRunner.hpp` | 主应用与模块的装配；公共参数沿用原应用 |
| `resources/c7_egs4/` | 项目自带的冻结 AIR-NTP 表及来源、许可说明 |
| `tests/modules/egs4/` | 无需 C7 安装的表与同种子输运回归测试 |

原 `extensions/c7_egs4` 中的独立应用、Fortran 参考驱动、Python 诊断脚本、临时构建覆盖层及历史验证快照已从生产源码树移出，保留在本地研究归档，不参与生产构建。主应用及多卡调度不需要 Python。

## 构建与调用

在原有 OpenMP、CUDA 或 CUDA/OpenMP 构建配置中增加：

```bash
-DCORSIKA_ENABLE_EGS4=ON
```

照常构建、安装 `c8_air_shower`；独立程序可链接模块库 `CORSIKA8NativeEgs4`，安装后为 `CORSIKA8::CORSIKA8NativeEgs4`。模块不再向应用泄露扩展目录的裸头文件路径。

原命令只需选择 `--em-backend egs4`，可选算法参数 `--egs4-stepfc` 默认仍为 `1`。例如：

```bash
c8_air_shower --em-backend egs4 \
  -A 56 -Z 26 -E 215400 -z 60 -N 10 -s 1931 \
  --antenna-file /path/to/antennas.txt -f /path/to/new-output
```

单卡用 `--device 0`，多卡用 `--device 0,1,2,3 --gpu-memory-fraction 0.9`，也支持空格分隔编号。方向、高度、磁场、大气、天线、cuts、薄化、穿界方式和射电后端均使用主应用的公共参数；不额外提供参考事例预设或 μ 子后端选项。双端构建支持选 CUDA 或 OpenMP，EGS4 暂不支持协同 `cuda-openmp` 执行模式。

默认表 `resources/c7_egs4/EGSDAT6_.4` 在构建时内嵌，运行时无需表路径或外部 C7 安装。输出 `native_egs4/config.yaml` 记录表 SHA256。`CORSIKA_EGS4_TABLE` 仅作为高级 CMake 覆盖选项保留；默认表 SHA256 为 `148c56f0f6397faf0f4e23d9a20b2d754f73bae02d644d1c3506153e2e8c990d`。

启用 `-DCORSIKA_EGS4_BUILD_TESTS=ON` 后：

```bash
cmake --build /path/to/build --target test_c8_egs4_embedded
ctest --test-dir /path/to/build -R c8_egs4_embedded_tables --output-on-failure
```

此次整理仅移动文件、修改 include 路径和构建装配；保留原命名空间、物理公式、随机数调用及默认配置，不代表新增一轮大样本物理验收。原效率诊断开关仍保持原默认值，不在此次整理中改变其行为。

2026-10-05 的隔离快照验证：主项目 CMake 配置及库导出生成成功；OpenMP 和 CUDA 的内嵌表、协调器契约测试均通过，各 10 项入口检查通过。EGS4 默认/短步长和连续事例，以及原 PROPOSAL/Kokkos 入口的 70 个 Parquet 输出与整理前逐值一致。这里的多卡检查使用协调器模拟 worker，不是新的四 L20 或 Fe500 统计验收。
