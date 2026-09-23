# 空气程序的纵向穿面计数与最简运行

2026-09-23。本次把独立比较验证版的可选计数口径接入正式空气应用；不把反应来源跟踪、完整谱系输出或人工扣除某些反应加入生产路径。

## 1. 默认值与物理边界

正式 `c8_air_shower` 和独立 `c8_corsika7_compare` 默认 `original-c8`。这里的“正向”是投影到 shower axis 后 `Xend > Xstart`，不是地理坐标中“向下”，也不是跨越岩石/空气边界的开关。

| 参数 | 方向 | 穿面区间（归一化深度 X/dX） |
|---|---|---|
| `--profile-crossings original-c8`（默认） | 仅正向 | `[Xstart, Xend]`，`ceil(start)…floor(end)` |
| `--profile-crossings forward` | 仅正向 | `(Xstart, Xend]` |
| `--profile-crossings both` | 双向均按正权重 | `(min(start,end), max(start,end)]` |

原 C8 闭区间在步进端点恰好落在计数面时，可能将共享端点计两次；保留它是兼容口径，不能把它与半开区间当成同一规则。零长度、非有限或完全超出数组范围的轨迹均不贡献穿面计数。显式传入未知字符串会报错，不静默改为默认值。

只控制光子、正负电子、正负 μ 子、强子/核子和全部带电粒子的**纵向粒子数 profile**。不改变真实粒子输运、向上运动、几何求交、随机流、物理反应、thinning、cut、观测面输出、能量沉积或 CoREAS/ZHS 轨迹。双向统计是正权重次数，不是带符号的净通量，也不是去重后的粒子身份数。

底层 `LongitudinalWriter` / accumulator 的构造默认值继续保留 `Both`，维持未指定策略的其他应用（包括山体）的原有行为；空气应用显式设置上述新默认。完整步进回传、紧凑投影、GPU 驻留 profile、OpenMP 主机分片及双端合并均传递同一配置。双端或分片合并时规则不匹配会报错。

输出 `profile/config.yaml` 记录 `profile-crossings`、`counting`、`crossing-interval` 和 `affects-transport: false`。比较 profile 前先查该配置，不能靠二进制文件名推断规则。

## 2. 从本地源码重新 build/install

按照主 README 配置 Conda、CUDA、Conan 与授权 FLUKA 后：

```bash
conda activate corsika_venv
cd corsika8_kokkos_beta5
C8_BUILD_JOBS=1 bash tools/build_kokkos_dual.sh -DWITH_FLUKA=ON
cd ..
```

组合安装为 `install/cuda-openmp`，同一个二进制可运行 CUDA 或 OpenMP。默认组合 Conan profile 面向 Ada89；其他显卡先选对应 profile，见主 README 第 7.2 节。也可用 `build_kokkos.sh openmp,cuda` 重建两个独立安装。只更新源码或 GitHub 不会更新已有 install 二进制。

## 3. 最简可复现测试

以下在项目根目录运行，不是源码子目录；先激活 `corsika_venv`。输出文件夹必须是新路径，避免覆盖旧样本。

```bash
export CORSIKA_DATA="$PWD/install/cuda-openmp/share/corsika/data"
install/cuda-openmp/bin/c8_air_shower \
  --em-backend kokkos --radio-backend kokkos --kokkos-execution cuda \
  -p 22 -E 10 -N 1 -s 12345 -f photon10GeV_cuda \
  --antenna-file corsika8_kokkos_beta5/examples/beta5/antennas_minimal_nwu.txt
```

这会生成 1 例 10 GeV 光子 shower，使用默认原 C8 profile 规则，CoREAS/ZHS 都启用；不是先前 Fe/C7 对比配置。GPU 显存预算保持自动容量、默认 70% 上限；小事例不保证占满预算。无需显式给波前粒子容量。

同一个程序改用 OpenMP（输出路径也换一个）：

```bash
install/cuda-openmp/bin/c8_air_shower \
  --em-backend kokkos --radio-backend kokkos --kokkos-execution openmp \
  --kokkos-num-threads 4 \
  -p 22 -E 10 -N 1 -s 12345 -f photon10GeV_openmp \
  --antenna-file corsika8_kokkos_beta5/examples/beta5/antennas_minimal_nwu.txt
```

若要旧双向口径，在命令末尾加 `--profile-crossings both`。若只要半开区间正向计数，加 `--profile-crossings forward`。想明确写出默认值则加 `--profile-crossings original-c8`。不要把启动器专用的 `--backend` 当成这个二进制的 `--kokkos-execution`。

无显卡服务器使用独立 OpenMP 安装；同一组合程序不能替代无 GPU 平台的独立构建。实际运行的最高能强子模型和低能 FLUKA/UrQMD 以 metadata 与构建日志为准；上面的 photon 简例不验证强子模型统计。

## 4. 验证范围与已知限制

- `testProfileCrossingModes` 对三个模式、五类加速 PID 做 CPU 逐面 oracle 对照，分别在 CUDA 与 OpenMP 执行；反向轨迹能量沉积必须保持不变。
- `testOutput` 中 `LongitudinalWriter` 用例检查所有计数组分、端点、切分及配置；已有分片回归检查默认低层行为没有变。
- 同种 seed、同一后端的小事例应保持非 profile 物理输出不变；不同后端浮点差异和调度不在此处承诺 bitwise 相同。
- 这次更新不是 C7/C8 射电等价修复，也不包含此前同轨迹诊断中尚待独立验收的 CoREAS 时间格点改动。计数规则一致不等于完整物理模型一致。

### 本机验收记录（2026-09-23）

Release / CUDA_OPENMP / Ada89 / FLUKA 组合构建；本机 RTX 4060。

- 七项 CTest：三个模式的 CUDA/OpenMP 逐面 oracle、主机输出、OpenMP 分片、掠射边界、CPU/CUDA 停下正电子，全部通过。
- 真实空气事件：1 GeV 光子，seed=12345，每次 `N=2`；CUDA 与 OpenMP 各测驻留累积和完整轨迹回传，另测标量 PROPOSAL。每一路径依次测试默认、`original-c8`、`forward`、`both`，共 20 次进程、40 例。
- 同一路径内，默认 profile 与显式 `original-c8` 完全相同；改变计数口径后，粒子、反应、产生 profile、能量沉积、CoREAS/ZHS 的实际 Ex/Ey/Ez 数组逐值不变。波形非零，第二例复用加速后端；不是空输出测试。
- 这只是计数策略及事件生命周期回归，不是大样本物理或性能验收。

可复跑脚本：`validation/accelerator/check_profile_crossing_modes.py`。本机明细位于项目根目录的 `build/profile-crossings-acceptance-20260923/`；每例记录命令、二进制 SHA256 和输出配置。脚本在可用内存低于 4 GiB 时会停止自己的测试进程。

代码入口：`applications/c8_air_shower.cpp`；共享规则 `corsika/framework/utility/LongitudinalCrossings.hpp`；主机 writer `corsika/detail/modules/writers/LongitudinalWriter.inl`；设备累积 `corsika/accelerator/em/detail/ProfileAccumulationStep.hpp`；配置传递 `applications/detail/air_shower_kokkos/KokkosAirShowerSetup.hpp`。
