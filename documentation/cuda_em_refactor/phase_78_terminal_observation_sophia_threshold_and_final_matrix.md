# Phase 78：终端观测面、SOPHIA 精确阈值与最终哈希验收

## 1. 本阶段关闭的两个静默物理风险

Phase 77 后开始用同一随机种子扩展 \(10^{17}\,\mathrm{eV}\) electron
样本时，先后复现了两个非常窄、但不能忽略的边界问题：

1. 粒子位于观测球面上方几十微米、向内运动时，球面求交跳过近根并选择地球
   另一侧的远根，随后 grammage 积分失败；
2. C++ 侧 SOPHIA 能力检查接受了一部分近阈碰撞，但 Fortran
   `eventgen.f` 用自己的质量表重新计算 \(s\) 后拒绝，并返回空粒子栈。

这两个问题都不能通过“回退到 CPU 后继续”掩盖。前者是终端面所有权错误，
后者会在父光子已被消费以后丢掉整个末态。本阶段分别修正几何语义和生成器
能力边界，并增加了 hard guard。

## 2. 终端观测面不能沿用普通层边界的近根抑制

### 2.1 原失败状态

固定 CUDA seed `278101` 的第 262 个 shower 中，失败 photon 的状态为：

```text
PID                         22
energy                      0.0005596296 GeV
position                    [76.4524280, -39.5177788, 6370999.999478279] m
direction                   [-0.4621301, 0.1607580, -0.8721196]
radius - observation radius 5.9555e-5 m
true near root              6.8287e-5 m
old selected root           1.1112631e7 m
old result                  atmosphere_grammage_failed
```

GPU tracking 与 CPU `TrackingLeapFrogCurved` 一样，对普通 volume boundary
采用 \(10^{-4}\,\mathrm{m}\) 保护宽度。它的目的，是让刚穿过大气层边界的
粒子继续属于新层，避免立刻在数值误差下穿回同一层。

观测面不是 volume transition，而是终端面。如果把同一规则用于观测球，
近根被跳过以后，二次方程仍有一个合法的正远根，即地球另一侧的第二次球面
交点。这会把几十微米的终端步误写成约 \(10^7\,\mathrm{m}\) 的穿地步长。

### 2.2 修复

`SphericalAtmosphere.hpp` 新增
`guardedObservationDistance()`：

- 只在观测面上方不超过 \(10^{-4}\,\mathrm{m}\)；
- 只接受向内运动；
- 使用消去相减消失的二次方程形式

\[
d_{\rm near} =
\frac{r^2-R_{\rm obs}^2}
     {-\,\boldsymbol r\cdot\hat{\boldsymbol u}
      +\sqrt{(\boldsymbol r\cdot\hat{\boldsymbol u})^2
             -(r^2-R_{\rm obs}^2)}} ;
\]

- 允许 double 舍入范围内的零距离，并拒绝保护宽度之外的结果；
- 普通大气层边界继续使用原来的近根抑制，不改变 layer ownership。

`CudaLeptonTransport.cu` 在曲线磁场球面求交前先查询这个终端近根。若命中，
仍使用正常 leapfrog 推进这段小于 \(0.1\,\mathrm{mm}\) 的轨迹，因此保留
磁偏转和有质量粒子的飞行时间，而不是直接把位置钳到球面。

测试同时覆盖：

- 原生产 photon 的精确位置和方向；
- 该步的有限 grammage；
- 直线 electron 到达观测面；
- 非零均匀磁场 electron 到达观测面；
- 终点半径与记录的 limiting radius 一致。

## 3. SOPHIA C++ 阈值与 Fortran 内部阈值

### 3.1 为什么原来的固定 \(\sqrt{s}\) 下限仍不精确

SOPHIA wrapper 接收 CORSIKA 四动量，但 `eventgen.f` 不直接使用这个
四动量的不变量。它用 legacy Fortran `AM` 质量表重建靶核子，并计算：

\[
s_{\rm int} =
m_{\rm S}^2 +
2 E_\gamma E_{N,{\rm eventgen}}(1-\beta_{\rm eventgen}) .
\]

随后 Fortran 代码要求：

\[
s_{\rm int} \ge 1.1646\ {\rm GeV}^2 .
\]

这里有三项会使简单的 CORSIKA \(\sqrt{s}\) 检查产生窄差异：

- CORSIKA nucleon mass 与 SOPHIA `AM` 不完全相同；
- wrapper 使用
  \(E_N=\max(m_{\rm SOPHIA},E_{N,\rm CORSIKA})\)；
- `eventgen.f` 会据此计算非零的 \(\beta_{\rm eventgen}\)。

另外，SOPHIA 同时保存 `AM` 和 `AM2`。`eventgen.f` 实际传播使用 `AM`，
而旧 C++ helper 返回 `sqrt(AM2)`；两者最后几位不同，已经足以让
1 eV 级边界测试不一致。

### 3.2 精确能力函数

新增 `corsika/modules/sophia/ThresholdKinematics.hpp`，逐项复现
Fortran 的：

- Hydrogen 到 proton 的内部映射；
- 靶核子 \(\gamma\) 与 \(\beta\) 分支；
- \(s_{\rm int}\) 计算；
- 从 \(s_{\rm int}=1.1646\,\mathrm{GeV}^2\) 反解到 CORSIKA
  \(\sqrt{s}\) 坐标的阈值。

`InteractionModel::isValid()` 现在检查精确的内部 \(\sqrt{s_{\rm int}}\)。
近阈两体模型的上边界也改为
`minimumCorsikaComEnergy(target)`，不再使用一个对 proton/neutron 和内部
质量差异都不敏感的固定常量。

`getSophiaMass()` 现在从 `so_mass1_.am` 读取真正的 `AM`，并保留原
`AM2` accessor 供其他代码使用。

### 3.3 空末态必须 hard fail

即使未来 SOPHIA 内部再次改变能力范围，C++ wrapper 在构造
`SophiaStack` 后也会检查：

```cpp
if (ss.getSize() == 0) {
  throw std::runtime_error(...);
}
```

因此“生成器拒绝但 CORSIKA 把相互作用标为成功”的路径已经不可能静默继续。
物理开放而 SOPHIA 不支持的窄区，仍由显式、计数的
\(\gamma+N\rightarrow\pi^0+N\) 两体模型覆盖；其他未知区域继续 hard fail。

### 3.4 头文件依赖边界

`ThresholdPhotoproductionModel` 依赖 SOPHIA 私有生成器接口，因此不再由
通用的 `corsika/modules/PROPOSAL.hpp` umbrella header 无条件包含。
生产 application 与对应模块测试显式包含它。这样
`testGpuProposalCpuFallback` 和默认 CPU-only 用户只使用通用 PROPOSAL
接口时，不会泄漏 `sophia.hpp` 的私有 include path。

## 4. 确定性故障重放

修复后的原 seed 输出：

```text
/tmp/c8_phase78_observation_sophia_repro_cuda_262_v2
```

结果：

```text
complete showers                       262 / 262
SOPHIA interactions                    22,359
near-threshold two-body fallback            5
QGSJet-II selected-final-state fallback   118
discarded final states                      0
queue overflows                             0
invalid profile records                     0
original shower_261                    complete
```

运行中不再出现：

```text
atmosphere_grammage_failed
input energy below threshold for photopion production
```

该重放使用几何/阈值修复后的即时 executable。随后只做了
`PROPOSAL.hpp` include 依赖清理并重新链接；最终哈希下的同一几何与阈值代码
由完整单元测试和下述正式 shower 矩阵继续覆盖。

## 5. 最终生产构建 provenance

```text
CUDA c8_air_shower SHA-256
f16bd3efdde4c4176ec10e406afcee4c418d00bd73fb1c5a92b48ec71131d97d

CPU-only c8_air_shower SHA-256
9c8ab29728021b9411f693337e14dfecba640b776f58db60d3f8677e6d3afe67

0.5 MeV production table SHA-256
140347373b9b5014c8b5bbd6cbcffca5f928eb5190e58c54c725d5843c2b295e

5 MeV production table SHA-256
ad85aaea51dc4e70100bc7eb176b1aaf434336c3f62ecbd47bb518b4f74f2f26

50 MeV production table SHA-256
1bdf4c8de406fe0d0f0a9ec098aad39fc48a2cbcf3d19d70a7f24f8bf2612fec

physics acceptance runner SHA-256
023566dffdc17e0d24a3811d42bd070d83d292a74aed3ee89dc91c8665f24fc7

performance acceptance runner SHA-256
e610b0dc654586633171a438e124e9cf8bb8341f412d005d374848d7ddb3eac7
```

构建条件：

```text
CMake build type       Release
CUDA architecture      sm_89
GPU                    RTX 4060 Laptop
CUDA compiler          corsika_venv nvcc
CPU comparison         one thread per scalar process
```

## 6. 当前哈希物理矩阵

所有样本均使用独立 CPU/CUDA seed、严格 executable/table provenance、
0.5 MeV cut（cut 专项除外）和统一 3 sigma 曲线门禁。

| 配置 | CPU+CUDA | curve | 九个 raw mean 通过数 | 仍未过 1% 但低于 3 sigma |
|---|---:|---:|---:|---|
| e, 1 TeV, 0° | 1000+1000 | 9/9 PASS | 7/9 | ground count/energy |
| e, 1 PeV, 0° | 800+800 | 9/9 PASS | 6/9 | deposit \(X_{\max}\), ground count/energy |
| e, \(10^{17}\) eV, 0° | 200+200 | 9/9 PASS | 7/9 | deposit \(X_{\max}\), ground count |
| e, \(10^{18}\) eV, 0° | 20+20 | 8/9 PASS | 1/9 | 1 radial bin；其余 raw mean 低统计 |
| gamma, \(10^{17}\) eV, 45° | 50+50 | 9/9 PASS | 5/9 | peak fits 与 ground totals |
| gamma, \(10^{18}\) eV, 80° | 20+20 | 9/9 PASS | 5/9 | longitudinal/deposit peak fits |
| e, 1 TeV, 45°, 5 MeV cut | 200+200 | 9/9 PASS | 6/9 | charged max 与稀疏 ground |
| e, 1 TeV, 45°, 50 MeV cut | 200+200 | 9/9 PASS | 5/9 | peak fits 与稀疏 ground |
| e, 1 TeV, 80°, no thinning | 50+50 | 9/9 PASS | 5/9 | 极斜 peak fits |
| p, 1 TeV, 45° | 100+100 | 9/9 PASS | 5/9 | 强子首相互作用高方差量 |

主要输出：

```text
/tmp/c8_phase78_finalhash_electron_1TeV_zenith0_pooled1000_v1
/tmp/c8_phase78_finalhash_electron_1PeV_zenith0_200_v1
/tmp/c8_phase78_finalhash_electron_1PeV_zenith0_pooled400_v1
/tmp/c8_phase78_finalhash_electron_1PeV_zenith0_pooled800_v1
/tmp/c8_phase78_finalhash_electron_1e17eV_zenith0_pooled200_v1
/tmp/c8_phase78_finalhash_electron_1e18eV_zenith0_20_v1
/tmp/c8_phase78_finalhash_photon_1e17eV_zenith45_50_v1
/tmp/c8_phase78_finalhash_photon_1e18eV_zenith80_20_v1
/tmp/c8_phase78_finalhash_electron_1TeV_zenith45_cut5MeV_200_v1
/tmp/c8_phase78_finalhash_electron_1TeV_zenith45_cut50MeV_200_v1
/tmp/c8_phase78_finalhash_electron_1TeV_zenith80_unthinned50_v1
/tmp/c8_phase78_finalhash_proton_1TeV_zenith45_100_v1
```

### 6.1 不能把“统计不足”写成物理失败或 1% PASS

1000+1000 个 1 TeV shower 的七个核心纵向/沉积均值全部通过 1% 和 3 sigma，
所有 ground histogram shape 也通过。但是到达地面的 1 TeV EM 尾部非常
稀疏，按实测方差，要让 ground count/energy 的三标准误差达到 1%，约需：

```text
ground weighted count     636,576 events / backend
ground kinetic energy   1,427,193 events / backend
```

因此继续运行少量事件直到样本均值偶然落入 1% 不是合法验收方法。本阶段保留
严格 raw-mean FAIL，同时把“所有有效曲线统计一致”和“未发现超过 3 sigma
偏差”作为可复现结论。生产论文若需要对小于 1% 的系统偏差作置信陈述，应预先
设计等价性检验和所需样本量，不能事后选择有利 seed。

### 6.2 1 PeV 首批异常、独立复验和预先扩展的统计量

原计划要求 200+200 个 1 PeV shower。第一批正式最终哈希样本触发了不能忽略
的门禁：

```text
charged Xmax proposal / CUDA       573.773 / 560.790 g cm^-2
signed relative difference                    -2.2627 %
absolute z score                                3.3472
longitudinal curve families                     0 / 6 PASS
```

这不是可以写成“统计不足”的结果，因此没有挑选另一组 seed 替换它。先检查了
8 个 CPU 分片的随机流和逐分片分布；CORSIKA Philox seed 经 SplitMix 初始化，
没有重复 shower 或相邻 seed 流复用。旧物理哈希下的独立 200+200 诊断样本
偏差方向相反，也提示必须增加最终哈希统计量，而不是立即调节 GPU 物理。

第二批独立 200+200 使用在启动前固定的新 seed，单独分析为：

```text
charged Xmax signed difference                -0.1199 %
charged Xmax absolute z score                   0.162
curve families                                  9 / 9 PASS
raw means                                       8 / 9 PASS
```

首两批合并到 400+400 后，首批的相干 profile 位移仍使曲线门禁失败，且
charged \(X_{\max}\) 为 \(-1.1945\%\)、\(2.378\sigma\)。功效分析给出该量约需
909 events/backend 才能把三标准误差压到 1%，所以又在不修改任何阈值、物理
代码、executable 或 table 的条件下预先增加 400+400，并最终合并为 800+800：

```text
observable                         signed difference      |z|       result
charged Xmax                              -0.827231 %    2.342      PASS
charged maximum                           -0.070277 %    0.193      PASS
charged integral                          -0.380520 %    1.684      PASS
photon integral                           -0.303430 %    1.301      PASS
energy-deposit sum                        +0.050944 %    0.574      PASS
energy-deposit Xmax                       -1.822841 %    1.443      inconclusive
ground EM weighted count                  -2.870034 %    1.389      inconclusive
ground EM kinetic energy                  -2.904233 %    0.904      inconclusive
energy closure                            -0.024231 %    0.785      PASS
```

最终 9 个 curve families 全部通过；六个 longitudinal families 的 active-bin
通过率为 96.1%–100%，三个 ground families 均为 100%。第三批新增 400+400
单独分析同样 9/9 curves 通过，charged \(X_{\max}\) 为 \(-0.4573\%\)、
\(0.921\sigma\)。因此首批 3.35 sigma 信号随独立统计量增加而衰减，没有形成
稳定的后端物理偏差；但三个高方差 raw mean 仍严格保留为 inconclusive。

800 个 CUDA shower 累计：

```text
GPU particle advances                  13,589,417,837
specified CPU final states                    415,993
SOPHIA interactions                           342,551
near-threshold two-body fallback                  99
QGSJet-II selected-final-state fallback           154
generic scalar fallback                           30
generic scalar steps                              30
generic reason                  unsupported_geometry only
discarded final states                             0
queue overflow / memory spill                      0 / 0
invalid/fixed-point profile records                0 / 0
```

所有 30 次 generic fallback 都发生在设备几何能力边界，并恰好执行一个 CPU
scalar step；没有 `atmosphere_grammage_failed`。主要复验输出为：

```text
/tmp/c8_phase78_finalhash_electron_1PeV_zenith0_second200_reanalysis_v1
/tmp/c8_phase78_finalhash_electron_1PeV_zenith0_third400_reanalysis_v1
/tmp/c8_phase78_finalhash_electron_1PeV_zenith0_pooled800_v1
```

### 6.3 强子混合链路

100+100 个 proton shower 中，CUDA 侧累计：

```text
GPU EM particle advances            29,244,781
CPU scalar particle steps              673,128
specified CPU final states                 605
SOPHIA interactions                        556
generic fallback                             0
discarded final states                       0
queue overflow / memory spill                0 / 0
```

这证明 CPU 强子/衰变路径、CUDA EM 队列和指定 PROPOSAL 末态可以在完整事件
中反复交接并最终 drain。

## 7. 最终哈希性能

### 7.1 热缓存，正式验收条件

输出：

```text
/tmp/c8_phase78_finalhash_hot_performance_1PeV_5rep_v1
```

1 PeV、单 CPU 核、Release、热 Molière cache、五次配对：

```text
status                                  PASS
external-wall median speedup            8.60079 x
summed-shower-timing median speedup     12.62890 x
minimum paired external speedup          8.34801 x
minimum paired shower speedup           12.08837 x
```

这满足原计划明确限定的“热缓存、Release、单 CPU 核、端到端至少 5 倍”。

### 7.2 cold-each 最坏初始化路径

输出：

```text
/tmp/c8_phase78_finalhash_cold_each_performance_1PeV_5rep_v1
```

每次 CUDA repetition 都从不存在的 Molière 派生缓存开始，重建时间在计时内：

```text
runner shower-timing status             PASS
summed-shower median speedup            6.01206 x
minimum paired shower speedup           5.84888 x
external-wall median speedup            4.85269 x
minimum paired external speedup         4.76698 x
```

因此冷启动的共同 shower timing 仍超过 5 倍，但包含整个进程固定初始化的
external wall 只有约 4.85 倍。原验收并未要求 cold-each external wall
达到 5 倍；文档必须把它与热缓存生产性能分开，不能沿用 Phase 77 旧哈希下
5.35 倍的冷启动数字。

五次 cold-each 都满足：

```text
cache before CUDA run              absent
cache after CUDA run               539032 bytes
discarded final states             0
generic fallback                   0
queue overflow                     0
```

## 8. 回归

最终源码与显式 include 边界清理后：

```text
CUDA Release all target       PASS
CUDA CTest                    32 / 32 PASS
CPU-only Release all target   PASS
CPU-only CTest                10 / 10 PASS
Python validation             53 / 53 PASS
```

重点测试：

```text
SophiaInterface               15 assertions PASS
PROPOSAL                      95 assertions PASS
testGpuSphericalAtmosphere    PASS
testGpuLeptonTransport        PASS
```

## 9. 工程结论与统计结论必须分开

工程实现方面，原计划要求的 CUDA EM 过程、HybridCascade、表格、环境、
指定 CPU 末态、输出、确定性、构建/CLI/provenance、CPU CoREAS 兼容轨迹和
热缓存 5 倍性能门禁都已经实现并回归。

统计方面，当前矩阵已经按原定事件数重新生成，并把 1 TeV、1 PeV 与
\(10^{17}\,\mathrm{eV}\) electron 分别扩展到 1000+1000、800+800 与
200+200。1 PeV 首批的显著异常也按预先扩展统计量的方式完整保留和复验。
所有大样本的平均 shower shape 通过，但不能声称所有高方差 raw mean 的真实
偏差已经以 1% 精度测定。这不是尚缺一个 CUDA 过程，而是独立 shower 统计
功效的明确限制。
