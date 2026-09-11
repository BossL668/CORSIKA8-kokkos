# beta5：ZHS 首末点数值伪影修复

## 1. 原因与修复范围

ZHS 先把矢势 A 累积到时间 bin，再由相邻 bin 的差分得到电场：

```text
E[i] = -(A[i+1] - A[i]) / dt
```

旧 `TimeDomainObserver::receive(VectorPotential)` 和 Kokkos `addSample()` 都先按贡献的时间标签裁剪，再把标签映射到最近 bin。一个略早于窗口起点或略晚于终点的标签，仍可能属于已经分配的首末矢势 bin；提前删除它会导致该 bin 不完整。即使真实矢势在窗口内恒定，也能产生假的首点或末点电场。问题同时存在于旧标量与加速路径，不是 GPU 独有的波形输出点数错误。

修复只改变**矢势**接受条件：先计算最近 bin，在有效 bin 范围内才累积。没有额外分配 guard 数组，也没有强制首末点为零、移动脉冲或做后处理滤波。以 400 ns、1 GHz 为例，仍是内部 401 个 A bin、输出 400 个 E 点。

| 文件 | 修改 |
|---|---|
| `corsika/detail/modules/radio/observers/TimeDomainObserver.inl` | 仅 VectorPotential overload 改为 bin 范围检查；直接 E 场 overload 不变 |
| `corsika/accelerator/radio/detail/RadioProjectionStep.hpp` | `addSample<Operations, true>` 供四处 ZHS 调用；CoREAS 保持原来的时间门禁 |
| `corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp` | 分块 GPU lambda 改为具名 `RadioTileProjection`，保持原来的分块与累积公式 |

所有新增 bin 检查均在转为无符号整数前进行，拒绝越界、NaN 和无穷；没有增加 waveforms、轨迹缓存或每事件存储。PROPOSAL 表、随机流、tracking、cut、thinning 和强子处理没有修改。

## 2. 构建中额外发现的内核入口问题

本机 NVCC 12.6.85 重新编译分块射电时，普通 kernel 正常，但 TeamPolicy 路径报 `cudaErrorInvalidDeviceFunction`。对 `__cudaRegisterFunction` 和 `cudaFuncGetAttributes` 的独立诊断显示：注册的 GPU lambda 入口和主机查询的 lambda 入口不同；冻结的旧测试程序在同一设备上正常。

将 GPU 分支的闭包改成具名 functor，避免 `if constexpr` 内两个 extended lambda 的入口编号问题。原有 scratch 大小、team/tile 数、循环次序及公式保留。不修改 Kokkos/Conan 缓存，不通过禁用分块测试掩盖问题。

## 3. 验收方法

独立标量 oracle 直接调用真实 CoREAS/ZHS 和 `TimeDomainObserver`，不是让 GPU 公式在主机上重复运行。

- 半 bin 边界：包含 `nextafter`、窗口外但仍属于有效 bin 的标签、非法输入、reset 后时间轴和点数。
- 固定轨迹：记录窗 `[0,10]` ns、平移窗 `[-2,8]` ns，以及完整窗 `[-10,30]` ns；两种亚 bin 相位分别激发旧首点/末点伪影。比较完整窗截取值，并保留窗外真实端点脉冲。
- 实际 scalar/Kokkos 射电：近 Cherenkov 区、普通/加权正电子、Fraunhofer 细分、真空、早晚轨迹、首末边界以及真实短脉冲；同时测试浮点/定点、普通/分块、reset/reuse。每个配置另加宽 2 ns、17 ns 检验共同区间的 A/E。
- 小型完整 shower：每后端 1 GeV 光子 `N=4`、10 GeV 质子 `N=2`，修复前/后以及后版本加宽窗口各跑一次，比较原始 Parquet/NPZ，而不是只看图。

完整 shower 使用固定 seed `26090729`、垂直、`emthin=1e-6`、IGRF14/2027、81 天线；默认 400 ns/10 ns 与 404 ns/12 ns 对照。OpenMP 4 线程；加速诊断采用 batch 16、resident 容量 4096、显存上限 15%，避免干扰正在运行的独立高能 campaign。该配置是局部正确性测试，不是高能吞吐量测试。每条路径分别与自己的旧版本比较，不要求独立 CPU/GPU shower 相同。

要求：非射电粒子数据、相互作用直方图完全一致；CoREAS 全部样点不变；ZHS 内部样点不变，允许修复首末点；修复后 ZHS 与加宽窗共同区间一致。时间原点减法及定点量化允许有声明的舍入容差。

验收脚本：

```text
tests/accelerator/testKokkosScalarRadioAlignment.cpp
tests/accelerator/ScalarRadioAlignmentDriver.cpp
tests/modules/testRadio.cpp
validation/accelerator/run_zhs_window_acceptance.py
validation/accelerator/analyze_zhs_window_fix.py
```

## 4. 已完成的验收

在 `corsika_venv` 的 Release 构建上：

| 项目 | 结果 |
|---|---|
| 原标量 Radio / observers / Propagators 测试 | 312 个断言通过 |
| 真实 scalar 与 Kokkos-OpenMP oracle | 29 个配置通过，含普通/分块及加宽窗口 |
| 真实 scalar 与 Kokkos-CUDA oracle | 同样 29 个配置通过；不再出现分块入口错误 |
| CUDA/OpenMP 组合程序中的 CUDA oracle | 同样通过 |
| 冻结轨迹前后诊断 | 两后端共 48 个配置通过；旧约 4.755 pV/m 的假首末脉冲消失，完整窗口中的真实端点脉冲保留 |
| 完整 shower 回归 | 三条路径、前后与加宽窗共 18 个进程 / 54 个 shower 正常完成 |

完整 shower 比较结果：

- 非射电 Parquet / NPZ 全部逐项一致，包括各组分 profile、能量沉积、地面粒子、相互作用及其直方图。
- CoREAS 全部样点与各自旧版逐项一致；ZHS 内部样点逐项一致。光子每组有 972 个分量样点发生预期边界修正，不是将整个波形重新拟合。
- 修复后 ZHS 与加宽窗截取值的最大绝对差异：标量 `4.54e-20 V/m`，OpenMP `4.34e-19 V/m`，CUDA `2.17e-19 V/m`。对应各组峰值归一化差异约 `7.54e-10`、`4.18e-9`、`2.46e-9`，符合窗口原点减法/定点量化容差。
- 表/辅助缓存哈希、物理计数、能量账本、backend 复用状态、YAML 字段与顺序保持一致。ZHS `zhs_contributions` 因补回边界贡献而增加；墙钟计时以及事后 `hadronic_worker_oracle` 按实测耗时预测的批次数允许变化，实际相互作用数与输运队列计数不放宽。
- 10 GeV 质子样本对非射电回归有意义，但部分/全部射电轨迹可能未进入当前窗口，不能用它单独证明射电修复。非零光子波形及固定轨迹 oracle 承担边界验证。
- `--help` 在使用相同 `argv[0]` 后字节一致。独立构建回归的峰值 RSS 为 609.6 MiB，系统可用内存至少 6522 MiB；测试进程没有 swap，也没有触发内存保护终止。这是小型测试的实测值，不代表高能事件的峰值上限。

这里验证的是**局部边界错误及修改的隔离性**，不是重新宣布完整能量账本覆盖或高能统计验收通过。没有进行独占设备性能 benchmark；测试时已有高能任务占用 GPU，不宣称新的速度提升。修复没有新增逐 shower 保留的数组。

## 5. 数据、兼容性与限制

结果位于 D 盘：

```text
CorsikaData/corsika_validation_results/
  acceptance_beta5_zhs_window_fix_20260907_v1/
    baseline/                  # 冻结的修复前程序
    fixed_track/               # 单轨迹 CPU / Kokkos CSV
    shower_final/              # 最终独立构建：前后、加宽窗、逐文件验收
    installed/                 # 安装后的组合程序：标量 / OpenMP / CUDA
    zhs_edge_fix_before_after.png
    fixed_track_acceptance.json
```

历史 500 例 Fe 等数据不覆盖。本轮能确定并修复的是窗口边界离散伪影，不能据此宣称以往全部射电统计差异已经消失。已有完整 shower 若未记录轨迹，不能仅靠删掉离群点恢复缺失的 A 贡献，需要重算射电或重新模拟。

已有高能生产任务继续使用冻结程序，不混入新语义。HIP/SYCL 共用修复源码，但本轮没有相应硬件验收，不标记通过。修复后也仍需选择足够长的窗口容纳真实脉冲。

## 6. 本地构建与安装状态

已更新 `build/cuda-openmp` 和 `install/cuda-openmp`。安装后又完成三条路径的 `N=4` 光子与 `N=2` 质子、普通/加宽窗口共 36 个 shower，全部正常结束；12 组 Parquet/NPZ 与已验收的独立构建输出逐项相同，包含完整射电数组。安装程序再与冻结旧版比较，全部六组边界修复门禁通过。安装后峰值 RSS 625.7 MiB，无测试进程 swap。

安装文件相对于 beta5 外层目录的位置：

```text
install/cuda-openmp/bin/c8_air_shower
SHA-256: af247c1ece6c346e97758ef3ff9913fd7e1baa9b60220b733a8aae32a9677b88
```

直接使用该程序，并通过 `--em-backend kokkos --radio-backend kokkos --kokkos-execution cuda|openmp` 选择加速模式。`--em-backend proposal --radio-backend cpu` 使用也已修复的 beta5 标量 observer。

**原独立 `install/cuda` / `install/openmp` 以及使用它们的统一启动器，本轮未更新**；需要按 README 重新构建后才能取得此修复。不要把上述组合程序已更新理解为所有历史安装/归档文件均已替换。源代码未推送远端，已有高能任务及 Fe 历史结果均保留原样。
