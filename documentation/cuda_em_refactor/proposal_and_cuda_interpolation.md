# PROPOSAL 原生插值与 CUDA 电磁表格插值

## 1. 结论

当前重构中存在两层不同的插值：

1. **原 CORSIKA 8 + PROPOSAL 路径本来就使用插值。**CORSIKA 创建
   PROPOSAL cross section、interaction 和 displacement 对象时显式传入
   `interpolate=true`。PROPOSAL 在 CPU 上用自己的缓存表计算反应率、连续
   能损、range 和随机能损的逆分布。
2. **CUDA 重构增加了第二层设备友好的插值。**GPU 不能直接持有包含虚函数、
   STL 容器、数值积分器和文件缓存的 PROPOSAL C++ 对象，因此
   `gpu_em_tablegen` 离线调用上述生产 CPU 路径，将结果压平成带版本和哈希的
   POD 表；CUDA kernel 再在这些表上插值。

所以，GPU 平坦表是为了重构新增的，但它拟合的参考值不是新定义的物理模型，
而是原生产路径的 `proposal_interpolated` 输出。

## 2. 原生产路径在哪里启用插值

`corsika/modules/proposal/ProposalProcessBase.hpp` 中的
`make_cross_sections()` 接收 `interpolate` 参数，并把它传给每一种
PROPOSAL cross section：

```cpp
inline PROPOSAL::crosssection_list_t make_cross_sections(
    Code code, PROPOSAL::Medium& medium, HEPEnergyType emCut,
    bool interpolate);
```

同一文件的生产映射对 photon、electron、positron、muon 和 tau 都显式调用：

```cpp
return make_cross_sections(Code::Electron, medium, cut, true);
```

这里最后的 `true` 不是 CUDA 重构加入的开关，而是原 CORSIKA 8
PROPOSAL 模块的生产设置。其作用包括：

- `CalculatedNdx(E)` 从 PROPOSAL 的截面插值对象取得每单位柱深反应率；
- `CalculateStochasticLoss(hash, E, u)` 使用 PROPOSAL 插值的逆累计分布，
  把均匀随机数 \(u\) 映射为相对能损 \(v\)；
- displacement/range 使用 PROPOSAL 自己的连续能损积分及插值缓存；
- PROPOSAL 首次遇到新粒子、介质或 cut 时生成缓存，后续运行复用。

因此日志中的

```text
Tables are not available and need to be created
```

通常表示 PROPOSAL 正在创建**第一层 CPU 插值缓存**，并不表示正在创建 CUDA
平坦表。

## 3. CUDA 为什么还需要另一份表

PROPOSAL 的运行时对象适合 CPU，但不适合作为 CUDA kernel 参数：

- 对象图包含继承、多态和动态分配；
- 求值器依赖主机端缓存与 STL 容器；
- 逆 CDF 可能调用 Newton、二分和数值积分；
- 数千个 GPU 线程同时走这些分支会严重发散，也无法直接调用主机代码。

`applications/gpu_em_tablegen.cpp` 因此构造与生产路径相同的参考对象：

```cpp
cross_sections_ = proposal::make_cross_sections(
    code_, medium_, energy_cut_MeV * 1_MeV, true);
interaction_ = PROPOSAL::make_interaction(
    cross_sections_, true, true);
```

然后离线采样并自适应细化：

- 分过程、分目标组分的 \(dN/dX(E)\)；
- 随机能损逆分布 \(v(E,u)\)；
- 连续 \(dE/dX(E)\)、range \(R(E)\) 和 \(R^{-1}\)；
- LPM 及 GPU 末态生成所需的参数。

文件中的 `reference_mode = "proposal_interpolated"` 明确记录了真值来源。
`proposal_direct` 只保留给 `interpolate=false` 的诊断实验，不是当前生产表的
验收基准。

## 4. 两层插值的计算关系

生产 CPU 路径可以写成：

\[
q_{\rm CPU}(E,u) =
{\cal I}_{\rm PROPOSAL}
\left[q_{\rm model}(E,u)\right],
\]

其中 \({\cal I}_{\rm PROPOSAL}\) 表示 PROPOSAL 自己的缓存插值、求根和积分
语义。

CUDA 生产路径近似的是这个 CPU 结果：

\[
q_{\rm GPU}(E,u) =
{\cal I}_{\rm flat}
\left[q_{\rm CPU}(E_i,u_j)\right].
\]

CUDA 误差门禁实际检查：

\[
\epsilon =
\frac{\left|q_{\rm GPU}-q_{\rm CPU}\right|}
     {\max\left(\left|q_{\rm CPU}\right|,\epsilon_{\rm floor}\right)}
\le 10^{-3}.
\]

这意味着当前的 \(10^{-3}\) 是**第二层平坦表相对现有生产 CPU 路径的误差**，
不是对基础截面理论本身的总系统误差。截面参数化、介质模型以及 PROPOSAL
第一层插值的不确定性需要单独研究。

当前平坦表采用的主要坐标是：

- rate：在正值区间使用 \(\log E-\log(dN/dX)\) 插值；
- 逆 CDF：行内使用分位坐标与 \(\log v\)，行间使用
  \(\log E-\log v\)；
- continuous range/dEdX：在正值量上使用对数坐标；
- 所有表都经过自适应 probe 和未参与细化的独立 validation。

## 5. 为什么不能盲目把每个点都平滑掉

在 5 MeV 和 50 MeV cut 表格验证中，原 PROPOSAL 7.6.2 的 Epair
插值逆 CDF 在约 \(145\) MeV、\(u\lesssim1.2\times10^{-4}\) 附近出现
目标组分相关的求根分支跳变。这个跳变通过直接调用
`CalculateStochasticLoss()` 重现，发生在 CUDA 表建立之前。

连续二维平坦表无法在跳变两侧同时满足 \(10^{-3}\)。生产实现没有：

- 把跳变错误归因于 GPU；
- 静默平滑该区间；
- 忽略这部分反应。

实现采用明确的能力边界：

```cpp
inline constexpr double EpairLossQuantileMinimum = 2.e-4;
```

被选中 Epair 顶点的最低 0.02% 分位返回指定 CPU PROPOSAL 末态生成器，并在
`cpu_fallbacks_by_process` 和 `cpu_fallbacks_by_reason` 中计数。这样保留了
原 CPU 数值语义，也使性能代价可审计。

类似地，若自适应网格触及点数上限，只有完整 probe 最大误差已经低于用户请求
的 \(10^{-3}\) 时才允许停止细化；之后仍必须通过独立 validation。误差超过
用户容差仍然是致命错误，不能生成生产表。

## 6. 应如何理解 CPU/GPU 一致性

验证分成三个层次：

1. **设备表检查**：同一批 \((E,u,\mathrm{process},\mathrm{component})\)
   查询在 CPU flat evaluator 和 CUDA evaluator 上逐项比较。
2. **参考误差检查**：flat evaluator 与原
   `proposal_interpolated` 调用比较，要求表内最大相对误差
   \(\le10^{-3}\)。
3. **shower 统计检查**：比较 \(X_{\max}\)、纵向曲线、能量沉积及地面
   分布；表格局部误差合格并不自动代表整个 shower 验收合格。

因此，回答“这个插值是不是为了 GPU 才有”的准确说法是：

> PROPOSAL 插值原本就有；CUDA 重构又增加了一个可在设备端高吞吐查询的
> 平坦插值层。第二层以第一层的生产输出为参考，并由严格误差、版本、哈希和
> CPU 回退机制约束。
