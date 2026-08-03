# Phase 79：IGRF13、CUDA replay 与射电统计一致性

## 1. 本阶段回答的问题

目标不是把“同 seed”与“同 shower”混为一谈，而是建立三个可分别审计的层级：

```text
层级 A：原版 PROPOSAL vs 重构版 PROPOSAL
        同 seed、同顺序随机流，要求同一条 shower

层级 B：原版 PROPOSAL vs CUDA EM
        同 seed 作为配置标签，但 RNG/调度不同，要求 ensemble 统计一致

层级 C：同一组 CUDA e± 轨迹上的 CPU radio vs CUDA radio
        轨迹完全相同，要求 CoREAS/ZHS 波形逐 bin 数值一致
```

层级 C 已由 `run_radio_acceptance.py` 覆盖。本阶段新增层级 A/B 的三臂
`cuda-replay` 运行器和逐过程 trace。

## 2. 环境已统一

原版与重构版 `c8_air_shower` 现在都使用：

```text
模型文件       corsika/modules/data/GeoMag/IGRF13.COF
年份           2025.0
纬度           42.5527 deg N
经度           86.4153816422 deg E
海拔           2680.444195 m
观测面         2680.444195 m
注入高度       112750 m
```

两个源码树中的 `IGRF13.COF` SHA-256 相同：

```text
afa2c1d0d42956314691e0eaa198fc429cc2ddd2fdd16308d107af88ba53eac7
```

此前 CUDA `EnvironmentSnapshot` 仍写死 `{50e-6, 0, 0}` Tesla；现在改为上传
`GeomagneticModel::getField()` 返回的同一组三分量。否则即使所有随机数相同，
CPU/GPU 轨迹也不是同一个物理问题。

## 3. 21CMA 天线文件

射电模拟显式使用：

```text
/home/yuhanglu/21CMA/data/antennas.txt
```

文件有 81 个有效 NWU 坐标：

```text
1 个中心点
8 个方位
10 个非零半径：1, 5, 10, 50, 100, 200, 300, 400, 500, 600 m
```

原版和重构版现在都会为这些位置建立 CoREAS 与 ZHS observer。

这里修复了一个重要错误：CUDA 路径过去用 `ring_number != 0` 判断是否需要保存
射电轨迹。外部天线文件配合 `--ring 0` 时，虽然 observer 已经存在，CUDA
仍不会向 CoREAS/ZHS 投影轨迹。现在判断条件改为实际
`detectorCoREAS.size() != 0`。

## 4. CORSIKA 8 论文怎样判断一致性

参考：

- [arXiv:2409.15999](https://arxiv.org/abs/2409.15999)
- [论文 HTML](https://ar5iv.labs.arxiv.org/html/2409.15999)

原作者没有要求 CORSIKA 7、CORSIKA 8 和 ZHAireS 生成逐事件相同的 shower。
论文明确指出，不同代码不能复现完全相同的 shower，因此跨代码比较选择纵向
发展和 \(X_{\max}\) 相近的事件，并进一步比较 ensemble 均值。

验证分成以下几层：

1. **低层射电公式验证**：均匀磁场中相对论电子的圆周轨迹，CoREAS Endpoint
   和 ZHS 与解析解比较，约在 1% 内。
2. **相同 CORSIKA 8 shower 上比较 CoREAS/ZHS**：这时粒子轨迹相同，差异只
   来自射电形式和数值步长。
3. **跨模拟程序比较**：使用相同大气、磁场、折射率、cut、观察者和频段，
   比较纵向发展、横向粒子分布、总轨迹长度、能量 fluence footprint 和
   radiation energy。
4. **消除 shower 能量差别**：energy fluence 和 radiation energy 除以
   电磁 shower 能量的平方。
5. **频段分别检查**：论文重点报告 30–80 MHz 和 50–350 MHz。
6. **研究轨迹步长**：减小 `MaxRad` 后比较 C8/C7 与 CoREAS/ZHS 的收敛。

论文的结果可作为量级参考，而不是自动适用于任何配置的通用门限：

```text
C8 CoREAS vs ZHS（同 C8 shower）
  radiation energy：30–80 MHz 约 2%，50–350 MHz 约 1%

C8 vs C7（不同 shower ensemble）
  平均 radiation energy：30–80 MHz 约 10%，50–350 MHz 约 2%
```

## 5. 新增 `cuda-replay` 功能

### 5.1 应用程序 trace

两个 `c8_air_shower` 都新增：

```text
--cuda-replay-trace /path/to/process_trace.csv
```

原版/标量路径每次 PROPOSAL 离散相互作用记录：

```text
shower、ordinal、PDG、process_id、component_hash
相互作用前能量、v、顶点位置、方向、时间
```

CUDA 路径记录：

```text
history_id、step_id、PDG、process_id
步前/步后能量、连续沉积、weight、步起点和时间
```

CUDA lepton 的过程是在到达顶点后重新选择的。为避免 trace 中 brems/ionization
显示成 `process_id=0`，router 会按 history 和相互作用顺序，把最终选择的
过程标回对应的 `EmStepRecord`。

GPU summary 也补齐：

```text
photon_pair
bremsstrahlung
compton
photoelectric
ionization
annihilation
electron_pair
```

### 5.2 三臂运行器

```text
validation/gpu_em/run_cuda_replay.py
validation/gpu_em/compare_cuda_replay.py
```

运行器生成：

```text
legacy_proposal/
refactor_proposal/
cuda/
legacy_process_trace.csv
refactor_process_trace.csv
cuda_process_trace.csv
legacy_vs_refactor_proposal.json
legacy_vs_cuda.json
cuda_radio_statistics.csv
run_manifest.json
```

`run_manifest.json` 保存完整命令、seed、IGRF13/2025、天线文件和 wall time。

## 6. 比较指标

### 6.1 shower 发展

- 总能量沉积；
- \(X_{\max}\)；
- \(dE/dX\) 最大值；
- `profile`、`production_profile`、`energyloss` 的相对 L1/L2；
- 逐过程计数、相互作用能量、\(v\) 和高度分布；
- 同为标量 PROPOSAL 时额外检查
  `PDG × process × component × ordinal` 是否 lockstep 相同。

### 6.2 射电

对每个 shower、observer、CoREAS/ZHS：

1. FFT 后分别保留 30–80 MHz 和 50–350 MHz；
2. 计算

   \[
   f_i=\epsilon_0 c\Delta t\sum_t E_i^2(t),
   \qquad
   f_{\rm total}=f_x+f_y+f_z;
   \]

3. 用 \(E_{\rm EM}^2\) 归一化；energy-loss summary 现在单独记录
   `sum_dEdX_em`，分析器以它作为 \(E_{\rm EM}\) proxy。读取本功能加入前的
   历史输出时才回退到总 `sum_dEdX`；
4. 同半径八个方位取均值；
5. 计算 footprint 积分

   \[
   E_{\rm rad}=2\pi\int r\,\overline f(r)\,dr;
   \]

6. 比较 ensemble 均值、标准误、fluence map 差异。

默认参考门限沿用论文跨代码结果的量级：

```text
30–80 MHz     10%
50–350 MHz     2%
```

正式验收同时要求：

1. ensemble 均值相对差不超过频段门限；
2. 均值差不超过三倍 combined standard error。

两项之间是逻辑 **AND**，不是 OR。小 \(z\)-score 可能只是样本量不足，不能
用来掩盖超出物理精度门限的均值差。少于 20 场 shower 时，无论单事件多接近，
状态都只会是
`diagnostic_insufficient_statistics`。

## 7. 冒烟测试结果

输出：

```text
/tmp/c8_cuda_replay_smoke_seed990002_v2
```

配置为 1 TeV electron、seed 990002、1 shower、0.5 MeV cut、81 个天线。

原版与重构标量控制：

```text
PROPOSAL 选择记录数              48,821 / 48,821
过程计数与顺序                    相同
沉积能量相对差                    1.40e-6
Xmax 相对差                       1.32e-7
归一化 radiation-energy 差异      2.75e-6 至 2.83e-6
```

这证明结构重构没有把原版 shower 变成另一场统计 shower；剩余差异是
约 \(10^{-16}\) 级局部浮点变化累积到 writer 的结果，而不是随机过程重排。

原版与 CUDA 的同 seed 单事件诊断：

```text
沉积能量差                        1.4%
Xmax 差                           28.95 g/cm²（9.4%）
30–80 MHz radiation-energy 差
  CoREAS                           1.43%
  ZHS                              1.73%
50–350 MHz radiation-energy 差
  CoREAS                           9.24%
  ZHS                             11.05%
```

这些数字不能作为通过或失败结论，因为只有一场 shower。高频对轨迹细节和
shower-to-shower fluctuation 更敏感，必须跑 ensemble。

## 8. 20-shower 射电诊断

结果目录：

```text
/tmp/c8_cuda_replay_radio_1TeV_20_seed260729_v2
```

配置为 1 TeV electron、20 showers、seed 260729、IGRF13/2025、
`data/antennas.txt` 中 81 个有效天线、CoREAS 和 ZHS 同时输出。逐过程 trace
在这次 ensemble 中关闭，以免生成过大的诊断文件。

shower ensemble 均值：

| 观测量 | 原版 CPU | CUDA | 相对均值差 |
|---|---:|---:|---:|
| 沉积能量 (GeV) | 970.742 | 972.210 | 0.151% |
| \(X_{\max}\) (g/cm²) | 319.723 | 313.437 | 1.966% |
| \(dE/dX_{\max}\) (GeV/(g/cm²)) | 29.329 | 29.831 | 1.683% |

ensemble 平均纵向曲线的相对 L1 差异为：

```text
能量沉积             4.72%
charged profile      6.51%
photon profile       6.33%
electron profile     8.21%
positron profile     8.06%
```

用沉积 EM 能量平方归一化的 radiation-energy proxy：

| 算法 | 频段 | CUDA 相对原版均值差 | z-score | 精度门限 | 判定 |
|---|---|---:|---:|---:|---|
| CoREAS | 30–80 MHz | -17.76% | 1.78 | 10% | 未通过 |
| CoREAS | 50–350 MHz | -17.23% | 1.77 | 2% | 未通过 |
| ZHS | 30–80 MHz | -17.20% | 1.73 | 10% | 未通过 |
| ZHS | 50–350 MHz | -16.69% | 1.71 | 2% | 未通过 |

这里必须谨慎解释：

- 四组 CUDA 均值都低约 17%，所以当前结果明确 **没有达到预定精度目标**；
- 但每项只有约 \(1.7\sigma\)，20 场仍不能以 3σ 水平确认这是系统偏差；
- CoREAS 与 ZHS 同方向变化，优先怀疑二者共同消费的粒子轨迹、track
  segmentation、step filtering 或 shower ensemble，而不是单个射电算法；
- 下一轮应先增加到至少 100 场，并增加总 \(e^\pm\) track length、
  track-segment 数、按高度/能量分箱的 weighted track length 和 MaxRad
  step 分布，再决定修改哪一层物理实现。

修正后的报告为：

```text
/tmp/c8_cuda_replay_radio_1TeV_20_seed260729_v2/legacy_vs_cuda_corrected.json
/tmp/c8_cuda_replay_radio_1TeV_20_seed260729_v2/cuda_radio_statistics_corrected.csv
```

报告状态是 `failed`。这不是 CUDA 内核崩溃，而是物理精度验收未通过。

## 9. 使用命令

先做 20 场诊断：

```bash
conda run -n corsika_venv \
  python validation/gpu_em/run_cuda_replay.py \
  --legacy-executable \
    ../corsika-21cma/corsika-build/applications/c8_air_shower \
  --cuda-executable \
    ../corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --table \
    ../corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v9_1e-3_1EeV.c8emrt \
  --antenna-file /home/yuhanglu/21CMA/data/antennas.txt \
  --output-root /tmp/c8_cuda_replay_1TeV_20 \
  --energy-gev 1000 \
  --events 20 \
  --seed 260729 \
  --minimum-showers 20 \
  --disable-process-trace
```

若 20 场有边界差异，应扩展到至少 100 场；不能因为单事件或小样本失败就修改
物理实现或放宽门限。

逐过程 trace 很大：当前 1 TeV 单 shower 约为 CPU 17 MB、CUDA 36 MB。
射电 ensemble 应使用 `--disable-process-trace`；UHE 正式 ensemble 也应先关闭
完整 trace，使用已有 physics-ensemble runner
比较 shower observable；逐过程 trace 用于 TeV/PeV 定位，后续再实现按
history/process 抽样。

## 10. “完全复刻原版”的下一层

当前功能是 **capture-and-compare replay**，还不是让 GPU 消费原版决定的
**decision replay**。

若要求 CUDA 对一场原版 shower 产生逐过程相同结果，需要记录并注入：

```text
相互作用/衰变距离随机数
process、component、v
全部 final-state random vector
LPM 接受/拒绝
multiple-scattering 随机数
thinning 决定
稳定 history/secondary 编号
```

GPU 再按 `history_id × step_id` 消费这些决定。该模式很适合作为调试 oracle，
并可让最终射电波形逐 bin 比较；但它绕过 GPU 自己的 rate、inverse-CDF 和
Philox 采样，因此不能替代独立的 GPU 物理验证，也不代表生产模式的加速性能。

另一条路线是在 GPU 上完全复制原版顺序 RNG 和 LIFO 调度；这会强制大量串行
依赖，基本抵消 wavefront GPU 加速。科研生产应保留统计等价模式，同时把
decision replay 作为专门的兼容性测试后端。
