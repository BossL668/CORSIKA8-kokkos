# beta5 射电路径：原版 CPU / Kokkos 对齐检查

日期：2026-09-06。本轮不改变 CPU CoREAS/ZHS 公式，不调整图形平滑，不消耗新的随机数。

## 检查方法

新增 `tests/accelerator/testKokkosScalarRadioAlignment.cpp`：同一条真实 CORSIKA
`Step<Particle>` 分别送入 **原版 `CoREAS::simulate()` / `ZHS::simulate()` 和
`TimeDomainObserver`**，以及 `KokkosRadioAccumulator`。这与以前“在 host 上运行
同一个 device helper”的检查不同，CPU 端是独立的原版实现。

覆盖电子/正电子、权重、普通轨迹、Fraunhofer 细分、近 Cherenkov 方向、真空、
传播表下边界、迟到轨迹和观测窗口末端；检查 fixed-point / double accumulation，
并比较 0.5 / 1 / 2 m 传播表步长。每个观察者比较 Ex/Ey/Ez，ZHS 同时比较原始
矢势和 `RadioProcess::endOfShower()` 所用的相邻样本差分，包含最后一个时间 bin。

这是**确定性局部轨迹验收**，不能代替修复后独立 shower 的统计验收，也不能证明
任意轨迹、任意硬件的逐位相等。

为避免用 NVCC 编译原 CORSIKA 的 CPU-only 头文件，测试拆为正常 C++ oracle 和
`ScalarRadioAlignmentDriver.cpp` 设备端驱动；没有把 CPU 算法改写成第二套测试公式。

### 本机 OpenMP 首轮实测

2026-09-06，独立 `build/physics_alignment/openmp`、4 线程测试通过。
日志：`build/audit/radio_openmp_alignment_20260906.log`。

- 19 组轨迹/模式/步长组合，包含 fixed-point 和 double 累积、direct/tiling 输入、
  reset 后第二次 shower 的同轨迹结果；
- double 累积最大相对峰值差：CoREAS `1.72e-11`，ZHS 矢势 `4.55e-13`；
- fixed-point 模式采用绝对量化门限与相对门限组合检查；极弱轨迹的相对误差可到
  `8.97e-5`，不能只给出 double 模式的小数值而声称整个生产定点路径也有相同精度；
- `window_end` 原 CPU 的倒数第二个 Ey bin 为 0，最后一个为
  `-4.13816e-12 V/m`；Kokkos 给出相同终端脉冲。这验证窗口效应在原 CPU 上也存在。

本轮独立 CUDA Release 也已实际通过上述 19 组对照，日志为验收目录的
`after/cuda_radio_oracle.log` 和 `after/cuda_final_ctest.log`。详细汇总见
[本轮修复报告](BETA5_CPU_PHYSICS_ALIGNMENT_FIXES_20260906_CN.md)。

## 本次发现并修正

### 1. 数值为零的 CoREAS Doppler 缺少 CPU 精度补救

原版 `CoREAS.inl` 在 `1 - n * beta.dot(emit) == 0` 时，用 `long double`
重算 pre/post/mid Doppler。Kokkos 原来直接继续除以零，可能令原本有限的场变成
NaN/Inf，并最终触发 fixed-point 溢出门禁。

`RadioProjectionStep.hpp::coReasDoppler()` 现在只在 double 结果恰好为零时重算：

- host 使用与原 CPU 相同的 long-double 表达式；
- GPU 使用 FMA 保留乘法残差，并补偿加法误差的双 double 算法；
- 普通非零计算不变，不设置经验 epsilon，不把数学上的真正零分母替换为人为常数。

设备上的双 double 与不同 CPU 平台的 `long double` 格式不保证逐位一致，但都恢复
原先被 double 消去的残差。测试包含一个精确的二进制消去例子（残差 `-2^-60`）
和真正零分母例子，后者必须保持零，不能被悄悄钳位。

### 2. 原 CPU 传播表下边界的返回值未完全复刻

`TabulatedFlatAtmospherePropagator.inl` 的 `sourceHeight == 0` 分支声明了局部同名
`ri_source` / `ri_destination`，导致最终 `SignalPath` 返回的两个折射率都仍为 1；
`sourceHeight < 0` 分支亦使返回的 destination 折射率为 1。原 Kokkos 实现返回了
查表的实际折射率，因而没有严格复刻原版算法。

本轮 Kokkos 采用与原 CPU 相同的返回约定，并保留原传播时间算法。这是**兼容性
修正，不是说原 CPU 的变量遮蔽是更好的物理公式**。若未来改进该边界，应同时修改
两个后端并重新验收。常规空气 shower 的表下边界位于观测面下约 1 km，通常不进入
这两个分支，因此它不能直接解释当前主要纵向 profile 的差异。

### 3. 射电轨迹诊断的质量约定

原 CPU 诊断读取 CORSIKA stack 的 kinetic energy。Kokkos 先前由总能量减去
PROPOSAL 的电子质量常量；现改用 CORSIKA 生成的 `TransportElectronMassGeV`。
该处只影响轨迹能量分箱和诊断积分，CoREAS/ZHS 投影本身使用位置、时间、
电荷和权重，不在此处使用质量。

### 4. 半个时间 bin 附近的舍入次序

独立调用实际 `TimeDomainObserver::receive()` 的 nextafter 边界测试确实失败：
CPU 使用 `floor(x + 0.5L)`，原 device 使用 `floor(x + 0.5)`。当 `x` 是略小于
0.5 的相邻 double 时，double 加法会先把结果舍入到 1，错误地将贡献移到后一 bin；
CPU 的长双精度加法则仍小于 1。

现在 `observerNearestBin()` 先拆开整数和小数部分，再判断小数是否 `>= 0.5`，
不需要 GPU 的 long-double 支持，也不引入经验 epsilon。仅替换原 CPU 使用
`0.5L` 的接收分箱及 CoREAS 同 bin 位移判定；ZHS 原公式中明确使用 double
`0.5` 的 start/end-bin 计算保持原样。本修正属于浮点边界附近的离散分箱一致性，
不能据此推断它是原 2000 例平均 profile 约 1% 差异的原因。

## ZHS 窗口末端的尖峰如何判断

两条路径都把矢势传给同一个 `TimeDomainObserver`，使用相同的接收窗口：
`time < start` 或 `time > start + duration` 的贡献被拒绝，窗口端点包含在内。
原 CPU 输出 writer 使用

```cpp
E[i] = -(A[i + 1] - A[i]) * sampleRate;
```

若跨越窗口末端的 ZHS 子段，其最后一个候选接收时间落在窗口之外，该贡献会先被
时间门禁拒绝，即使对应“最近时间 bin”在窗口末端。这样最后一个矢势样本可能
小于前一 bin，相邻差分便产生终端脉冲。不能仅凭图中末端有尖峰就断言是 Kokkos
bug，也不能把尖峰删掉来使曲线好看。

新增 `window_end` 用同一真实 CPU ZHS 轨迹验证这一点，输出 CPU / Kokkos
最后两 bin，要求两端一致。具体测得值以本轮独立构建的测试日志为准。

## 已检查但未改变的原版语义

| 项目 | 对齐情况 |
|---|---|
| 粒子范围 | CPU 和 Kokkos 均对电子/正电子投影；muon 不在当前 radio 范围 |
| 电荷/权重/单位 | 使用原 CPU 常量；field 为 V/m，ZHS 内部是矢势 |
| CoREAS 近 Cherenkov | 保留 `1e-3` 分支、同 bin 时间位移和有符号先后次序 |
| ZHS Fraunhofer | 保留原版 double divisor + ceil 次数的细分行为，不擅自换算法 |
| ZHS 反向到达 | 保留原版 subdivided 分支中间 bin 的符号约定 |
| 时间轴 | CoREAS 在原采样点，ZHS 在差分中点；不峰值对齐、不归一化 |
| 定点累积 | 有限量化精度；检查溢出并失败，不允许静默 wrap 后继续输出 |
| 后端复用 | reset 清零累积器；下载的 ZHS 矢势交给原 writer 差分 |

## 限制及后续

- 原 CPU CoREAS 近 Cherenkov 分支还存在 `paths1.clear()` / `paths2.clear()` 后读取
  `paths1[i]` / `paths2[i]` 的未定义行为。直线传播下通常残留相同内容，但不应把
  未定义内存读取当作长期正确性保证。本轮未更改原 CPU；应独立修复并做 sanitizer
  及曲折传播验证，而不是令 device 代码复刻非法访问。
- 本接口仍是 straight-line / flat-atmosphere propagator；不宣称等价于一般
  弯曲光线、多路径、firn 或 ice 射电传播。
- HIP / SYCL 尚未在本机硬件运行，本轮本机验收只覆盖 OpenMP 和 NVIDIA CUDA。
- CPU 长双精度与 GPU 双 double 不能用于宣称跨设备完整 shower 逐位一致。
