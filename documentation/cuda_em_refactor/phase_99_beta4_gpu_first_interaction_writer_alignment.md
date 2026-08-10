# Phase 99：beta4 GPU 首相互作用输出与 CPU `InteractionWriter` 对齐

## 1. 问题

标量级联在每个实际相互作用或衰变之后调用完整的
`ProcessSequence::doSecondaries()`。`InteractionWriter` 位于 `EMThinning` 和
`ParticleCut` 之前，因此它记录第一个物理末态的全部未薄化次级粒子。

CUDA 原生 photon/lepton 末态不构造 CPU `StackView`，此前也没有向
`InteractionWriter` 回放记录。这造成两个错误：纯电磁初级的
`interactions.parquet` 可能是空的；如果稍后出现 CPU fallback，后面的事件还
可能被误标成首相互作用。强子初级通常首先在 CPU 上相互作用，因此原先的大型
proton/iron 系综没有直接暴露这个缺陷。

## 2. 修复设计

新增固定大小的 `GpuFirstInteractionSnapshot`。photon 和 charged-lepton 末态
kernel 在完整运动学已经构造、但 thinning mask 尚未压缩输运队列的位置检查
`parent.generation == 0`，并保存：

- 顶点处初级状态；
- GPU 过程编号；
- 1--3 个完整未薄化次级粒子的 PID、总能量和方向。

每个 shower 只有一个 generation-zero projectile。设备端候选计数不是
last-writer-wins：如果发现多于一个候选，`downloadFirstInteractionSnapshot()`
立即终止当前 shower。记录和计数共占 468 bytes；shower 结束时只下载这个
固定记录，不下载全部 final-state 数组，也不把粒子重新插回 CPU stack。

`InteractionWriter` 新增输出格式无关的 `FirstInteractionSnapshot` 入口。原 CPU
`doSecondaries()` 也先生成同一种 host snapshot，因此 CPU 与 CUDA 共用同一套：

- ShowerAxis slant-depth 投影；
- observation-plane 坐标和动量投影；
- Parquet 行和 YAML summary 写出；
- 每个 shower 只允许首条记录的仲裁。

CPU fallback、CPU 衰变和强子初级仍由原 `doSecondaries()` 路径记录。GPU
snapshot 不重新执行 thinning、cut 或 production-profile，也不参与物理调度。

## 3. 兼容性门禁

`ProcessSequenceCompatibility` 现在同时递归检查 `ContinuousProcess` 和
`SecondariesProcess`。当前应用显式登记：

- device-replaced：PROPOSAL continuous EM、EM thinning、ParticleCut；
- record-replayed：CoREAS、ZHS、longitudinal/observation output、production
  profile 和 InteractionWriter。

以后加入未登记、能够观察或修改 routed EM 状态的自定义
`SecondariesProcess` 时，CUDA 模式会在 shower 启动前失败，不再静默跳过。

## 4. 自动化验收

- CUDA/Release 全量 CTest：`34/34` 通过，包含 FLUKA；总计 221.41 s。
- Python 验证工具：`241/241` 通过。
- `testInteractionWriter` 同时覆盖 StackView 和直接 snapshot 入口，并验证单
  shower 幂等仲裁。
- GPU host compatibility 测试验证未知 `SecondariesProcess` 会被拒绝。
- photon pair、brems、annihilation、ionization、epair、完整 photon/lepton
  wavefront、radio projection 和 hybrid route 测试全部通过。

## 5. 真实运行验收

正式结果保存在数据根目录的下列文件夹：

| 事例 | 结果 |
|---|---|
| `beta4_first_interaction_fix_photon10GeV_theta80_v2` | parent PDG 22；Parquet 两行 `e-`,`e+`；候选 1，写出 1 |
| `beta4_first_interaction_fix_electron10GeV_theta80_v2` | parent PDG 11；Parquet 两行出射电子和 delta electron；候选 1，写出 1 |
| `beta4_first_interaction_fix_proton10GeV_vertical_v1` | CPU 首相互作用 9 个次级；GPU 候选 0、GPU 写出 0，无覆盖 |

photon trace 使用 `emthin=1`。首相互作用后只出现一个加权 GPU 子历史，而
`interactions.parquet` 仍保存两个物理次级，直接证明快照发生在 thinning 之前。

## 6. 标量路径回归

使用 phase 98 的相同固定种子 10 GeV、theta=80 deg muon 配置，重新运行
`--em-backend proposal`。修复前后的结果为：

- CUDA replay process trace：逐字节相同（均为 18,099 bytes）；
- `interactions.parquet`：schema、行数和值完全相同；
- longitudinal profile、`dEdX` 和 ground-particle Parquet：完全相同；
- interaction YAML summary：完全相同。

因此该修复补齐了 GPU 输出观察路径，没有改变原标量随机流、粒子树或物理
输出，也没有令强子初级的 CPU 首相互作用被 GPU 记录覆盖。
