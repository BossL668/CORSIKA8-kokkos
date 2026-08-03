# Phase 94：CUDA μ 子 parent production profile 记账修复

## 1. 问题结论

`final_vertical_proton_100TeV_emthin1e-6_cpu500_cuda500_fullaccel_v3` 中的
parent profile 差异不是统计涨落，也不是绘图程序问题。逐列检查表明：

- pion、kaon、heavy、hadron 和 photon parent 的 CPU/CUDA 均值一致；
- electron/positron parent 在两条路径中均为零；
- 原 CPU 的 muon-parent profile 积分均值为 `138368.754`；
- 修复前 CUDA 的对应均值只有 `406.484`。

缺失量集中在 GPU 常驻 μ 子输运，因此 `all` 列也随之缺失。

## 2. 原版 CORSIKA 8 的定义

`ProductionProfile::doSecondaries()` 遍历每个离散相互作用产生的 secondaries。
只要 secondary 是 μ 子，就在顶点位置写入一次，并按 projectile PID 分类 parent：

```cpp
if (is_muon(particle.getPID())) {
  this->write(particle.getPosition(), projectile.getPID(), particle.getWeight());
}
```

因此，PROPOSAL 的一次离散 μ 子 ionization 会产生继续传播的 μ 子，并在
`muon` parent 列计数一次。该 process 位于 EM thinning 之前，所以这里使用的
是 thinning 之前的 parent/secondary weight。writer 的顶点分箱是：

```cpp
bin = ceil(X / dX);
```

当前 CUDA 路径在设备上实现 μ 子 ionization；μ 子 bremsstrahlung、pair
production 及其他稀有末态仍显式返回 CPU。返回 CPU 的过程会正常经过
`ProductionProfile`，但设备上接受的 ionization 末态此前只进入纵向粒子数和
能损直方图，没有进入 production writer，这正是缺口。

## 3. 修复方式

### 3.1 设备端

resident profile 新增第八个固定点直方图：

```text
muon_parent_productions[bin]
```

它只在以下条件同时成立时累加：

1. 末态 record 已被 GPU 接受，不是 fallback、LPM suppression 或 transport
   candidate；
2. process 是离散 `IonizationProcessId`；
3. transport parent PID 是 `MuonMinus` 或 `MuonPlus`；
4. final-state record 的 `input_index` 与 `parent_history_id` 能找到唯一 transport。

顶点使用 transport end position 投影到 ShowerAxis，分箱严格使用
`ceil(X/dX)`，贡献严格使用 incoming parent weight。连续输运、磁场步、层边界、
观测面、cut 和 decay 均不会写入这个直方图。

### 3.2 CPU writer 合并

`ProductionWriter` 新增：

- `writeProjected(grammage, pid, weight)`；
- `addBin(bin, pid, weight)`；
- `getNBins()`。

原 `write(Point, ...)` 和 CUDA profile 合并共用同一分类函数。设备回传的 μ 子
parent bin 以 `Code::MuMinus` 作为类别标签写入；writer 会同时增加 `muon` 和
`all` 两列。μ+ 与 μ− 在该输出格式中本来就共用 `muon` parent 列。

### 3.3 可重复性与性能

修复不调用随机数、不改变 history ID、不改变队列、不改变末态，也不改变
thinning 决策。它仅在已有末态 record 上增加一次固定点 atomic add。每个
profile bin 新增 8 字节显存；本次 883 bins 的生产配置总增量为 7064 字节。

## 4. 验证结果

### 4.1 单元和设备测试

- `testModules 'Production*'`：2 个 test case、14 项断言通过；
- `testGpuPhotonWavefront`：52911 项检查通过；
- 合成设备测试同时输入：
  - `X=25 g/cm²`、weight=3 的 μ− ionization；
  - `X=35 g/cm²`、weight=7 的 electron ionization 负对照。
- 结果只有 muon-parent 的 bin `ceil(25/10)=3` 得到 3，总和为 3，电子负对照
  没有污染该列，`invalid_records=0`。

### 4.2 真实 100 TeV shower

使用独立验证可执行文件运行：质子、100 TeV、垂直入射、seed `10100051`、
`emthin=1e-6`、FLUKA 2025、CUDA EM。结果完整结束：

- device-resident muon-parent contribution：`150457.9999991064`；
- parquet 中最终 muon-parent profile 积分：`150729`；
- 差值：`271.0000008936`，来自 CPU fallback/scalar 路径；
- `invalid_records=0`；
- `fixed_point_overflows=0`。

这证明设备贡献与 CPU fallback 贡献恰好合并，未丢失也未重复。

将该事例放回原 CPU500 的事例分布：

- CPU500 积分均值/标准差：`138368.754 / 37193.791`；
- CPU500 2.5%、50%、97.5%分位：`65177.825 / 140562 / 205243.275`；
- 修复后 CUDA：`150729`，相对 CPU 均值为 `+0.332 sigma`，第60百分位；
- 归一化纵向形状距离处在 CPU 自身涨落的第32百分位；
- peak depth `530 g/cm²`，位于 CPU500 的 `430–740 g/cm²` 95%范围内。

诊断图和可复算数据位于：

```text
/mnt/d/CorsikaData/corsika_validation_results/
final_vertical_proton_100TeV_emthin1e-6_cpu500_cuda500_fullaccel_v3/
parent_profile_fix_diagnostics/
```

真实 smoke shower 位于：

```text
/mnt/d/CorsikaData/corsika_validation_results/
parent_profile_fix_proton_100TeV_seed10100051_smoke_v2/
```

## 5. 构建隔离与后续正式验收

IGRF14/2027 的 100 PeV 10-event campaign 在修复期间仍调用原
`c8_air_shower`。为防止一个 campaign 内混用两个二进制，本阶段新增了默认不
参与构建的独立目标：

```bash
cmake --build /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target c8_air_shower_parent_profile_validation -j 4
```

当前已修复的可执行文件是：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/
c8_air_shower_parent_profile_validation
```

待正在运行的 campaign 全部结束后，再重新链接常规 `c8_air_shower`。旧的
CUDA500 parquet 没有保存设备 μ 子 ionization 顶点，不能通过后处理无损补回；
正式的“500 vs 500 修复后图”必须重跑 CUDA ensemble。旧图应保留为修复前
证据，不能覆盖或改标签冒充修复后结果。
