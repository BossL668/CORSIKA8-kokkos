# CUDA 电磁后端重构记录：阶段 11，photon-pair LPM suppression

## 1. 本阶段结论

本阶段把 CORSIKA 8 当前 CPU 路径中的 photon-pair LPM 拒绝采样，按相同
物理公式和相同处理顺序接入了 CUDA photon-pair 末态验证链路。

当前一个已经选定的 photon-pair 候选只能进入四个互斥队列之一：

```text
EmInteractionRecord
        │
        ├─ 非 GPU 过程或数据无效 ───────────────► CPU fallback
        │
        ├─ NoDiscreteInteraction ──────────────► continuation
        │
        └─ photon pair
              │
              ├─ 采样 x、方位角和两个 Sauter 极角
              ├─ 用相互作用顶点的局部密度计算 S_LPM
              ├─ Philox draw_id=5 采样 r
              │
              ├─ r <= S_LPM ─► e- + e+ secondaries
              └─ r >  S_LPM ─► 保留原光子，step_id + 1
```

四类输出都用 CUB exclusive scan 分配稳定位置。内核结束时强制检查：

\[
N_{\rm pair}+N_{\rm suppressed}+N_{\rm fallback}
+N_{\rm continuation}=N_{\rm input}.
\]

因此，本阶段不允许用全局原子追加，也不会静默丢掉被 LPM 抑制的光子。

需要特别说明：这仍是物理 validation bridge。`advanceWavefront()` 的 resident
双缓冲队列目前仍运行 toy branching；相互作用距离与真实 tracking/层边界的
竞争尚未接入。因此，这一阶段已经验证 LPM 公式、随机数、分支分类和父光子
保留语义，但还不是完整 shower 的生产 CUDA 路径。

## 2. CORSIKA 8 的 CPU 语义

### 2.1 LPM 在末态之后判断

CPU 实现在
`corsika/detail/modules/proposal/InteractionModel.inl`。执行顺序是：

1. 先为 PROPOSAL 末态生成器准备随机数；
2. 调用 `generateFinalState()` 得到两个次级粒子；
3. 从当前环境节点读取相互作用位置的质量密度；
4. 调用 `CheckForLPM()`；
5. 若被抑制，丢弃刚生成的 \(e^-e^+\)，把初始光子重新放回 CORSIKA 栈。

对应代码位于当前文件第 202–227 行。这里有两个容易实现错的点：

- LPM 不是简单乘入本阶段已有的 \(dN/dX\) rate 表；
- LPM 使用的 \(x\) 是已经生成的电子能量占比，而不是 photon-pair
  反应率选择阶段保存的 `v_loss`。

CPU 对 photon pair 使用

\[
x=\frac{E_{e^-}}{E_{e^-}+E_{e^+}},
\qquad
c_\rho=\frac{\rho_{\rm local}}{\rho_{\rm baseline}},
\]

然后调用

```cpp
photo_pair_lpm_->suppression_factor(E_MeV, x, target, c_rho);
```

最后生成均匀随机数 \(r\)。当

\[
r>S_{\rm LPM}
\]

时判定“相互作用被抑制”。这部分在同一文件第 292–346 行。

### 2.2 GPU 中的等价语义

设备端仍然先得到 \(x\) 和方向随机数，再计算 LPM。接受条件同样是

\[
r\leq S_{\rm LPM}.
\]

若被抑制，`PhotonPairLpmSuppressionRecord` 保存：

- 未改变的 photon PID、能量、位置、方向、时间和权重；
- 原 `history_id`、`parent_history_id` 和 generation；
- `step_id + 1`；
- component hash；
- \(S_{\rm LPM}\)、随机数 \(r\) 和 `draw_id=5`。

递增 `step_id` 是 counter-based RNG 所必需的：父光子下一次重新采样相互作用
时必须得到新随机数，不能在同一 key 上无限重复同一个 LPM 结果。

## 3. 设备端公式

实现位于 `corsika/gpu/em/PhotonPairLpm.hpp`，是 PROPOSAL 7.6.2
`PhotoPairLPM::suppression_factor()` 的逐公式移植。未启用 fast math，也没有
把结果强制 clamp 到 \([0,1]\)，从而保持 CORSIKA 的原始拒绝条件。

对目标核电荷 \(Z\)，先计算

\[
s_1=
\left[
\frac{1}{Z^{-1/3}B_Z}
\right]^2\sqrt{2},
\]

其中 \(B_Z\) 是 PROPOSAL component 的 `GetLogConstant()`。

局部密度修正为

\[
c_\rho=\frac{\rho_{\rm local}}{\rho_0},
\]

再计算

\[
s'=\frac{1}{8}
\sqrt{
\frac{E_{\rm LPM}}
{c_\rho E_\gamma x(1-x)}
}.
\]

PROPOSAL 的分段 \(\xi(s')\) 为

\[
\xi =
\begin{cases}
2, & s'<s_1,\\
1+h-\dfrac{0.08(1-h)\left[1-(1-h)^2\right]}{\ln s_1},
&s_1\leq s'<1,\\
1,&s'\geq1,
\end{cases}
\qquad
h=\frac{\ln s'}{\ln s_1}.
\]

介质极化修正为

\[
\Gamma=
1+4\pi\,\Sigma_Z\,r_e
\left(\frac{r_e}{\alpha x}\right)^2
n_{\rm mol}c_\rho ,
\qquad
s=\frac{s'}{\sqrt{\xi}}\Gamma.
\]

这里的乘法顺序在代码中与 PROPOSAL 保持一致；上式中的
\(\Sigma_Z\)、\(n_{\rm mol}\)、\(r_e\) 和 \(\alpha\) 都来自版本化物理缓存。

随后按 PROPOSAL 的分段近似计算 \(\phi(s)\) 和 \(G(s)\)：

\[
\phi(s)=
\begin{cases}
1-\exp\left[
-6s(1+(3-\pi)s)
+\dfrac{s^3}{0.623+0.796s+0.658s^2}
\right],&s<1.54954,\\
1-\dfrac{0.012}{s^4},&s\geq1.54954,
\end{cases}
\]

\[
\psi(s)=
1-\exp\left[
-4s-\frac{8s^2}
{1+3.936s+4.97s^2-0.05s^3+7.50s^4}
\right],
\]

\[
G(s)=
\begin{cases}
3\psi-2\phi,&s<0.710390,\\
\dfrac{36s^2}{36s^2+1},&0.710390\leq s<0.904912,\\
1-\dfrac{0.022}{s^4},&s\geq0.904912.
\end{cases}
\]

最终 photon-pair 生存因子是

\[
S_{\rm LPM}=
\frac{
\dfrac{\xi}{3}
\left[
\dfrac{G}{\Gamma^2}
+\dfrac{2\left(x^2+(1-x)^2\right)\phi}{\Gamma}
\right]
}{
1-\dfrac{4}{3}x(1-x)
}.
\]

非法能量、\(x\notin(0,1)\)、非正局部密度、未知 component 或损坏的参数快照
都返回显式状态，不产生 NaN 后继续运行。

## 4. v4 物理缓存

### 4.1 为什么必须从 v3 升级

\(E_{\rm LPM}\) 不是一个只由粒子质量决定的常数。PROPOSAL 在构造
`PhotoPairLPM` 时，会在 \(10^{14}\,\mathrm{MeV}\) 对当前介质和
`PhotoPairKochMotz` 截面做数值积分。设备端如果自行使用手写“空气常数”，
就可能与表格生成时的介质组分或 PROPOSAL 版本不一致。

因此，表格式升级到 v4，magic 从 `C8EMRT03` 变为 `C8EMRT04`。v4 payload
新增：

- 基准质量密度，单位 \(\mathrm{g\,cm^{-3}}\)；
- 分子数密度，单位 \(\mathrm{cm^{-3}}\)；
- 介质 `sum_charge`；
- 数值积分得到的 \(E_{\rm LPM}\)，单位 MeV；
- PROPOSAL 使用的经典电子半径和精细结构常数；
- 每个 component 的 hash、\(Z\) 和 radiation-log constant。

这些字段参与 payload SHA-256。reader 会检查所有数值有限且为正、LPM
component 集合与介质 component 集合完全一致。旧 v3 文件会因 magic/version
不符被明确拒绝，不能带着缺失的 LPM 元数据继续模拟。

### 4.2 tablegen 自检

`gpu_em_tablegen` 已升级为 `c8-gpu-em-tablegen-0.5`。它复现 PROPOSAL
构造器对 \(E_{\rm LPM}\) 的积分，并在写文件之前，用真正的
`PROPOSAL::crosssection::PhotoPairLPM` 扫描：

- 所有 dry-air component；
- \(10^3\)–\(10^{14}\,\mathrm{MeV}\)；
- \(x=10^{-3},10^{-2},0.1,0.5,0.9\)；
- \(10^{-7}\)–\(10\) 倍基准密度。

任何一点相对误差超过 \(2\times10^{-13}\) 都停止写表。本次正式 dry-air v4
构建的 host 公式最大相对误差为 0。

## 5. 随机数约定

photon-pair 末态现在使用固定 draw ID：

| draw ID | 含义 |
|---:|---|
| 1 | \(x\) 的 inverse-CDF 分位点 |
| 2 | pair 方位角 |
| 3 | electron Sauter 极角 |
| 4 | positron Sauter 极角 |
| 5 | LPM 接受/拒绝 |

每个随机数 key 仍为

```text
(seed, shower_id, history_id, step_id, process_id, draw_id)
```

因此，输入 wavefront 的排列、GPU 分桶顺序和 compact 后的位置不会改变某个
history 的 LPM 结果。测试同时比较原顺序、完全反序和相同 batch 重复运行。

## 6. 回退与失败策略

新增两个明确的 fallback reason：

- `LpmParametersUnavailable`：component 不存在、参数快照损坏或公式返回
  非有限结果；
- `InvalidMassDensity`：interaction vertex 的局部质量密度非有限或不为正。

当前验证接口对这两类输入返回 CPU fallback record，不会假设海平面密度，
也不会关闭 LPM。v4 缓存整体损坏、版本不匹配或 SHA-256 不符仍在 backend
初始化时直接抛出错误。

## 7. 主要代码改动

| 文件 | 作用 |
|---|---|
| `corsika/gpu/em/PhotonPairLpm.hpp` | pointer-free 参数快照和 host/device LPM 公式 |
| `corsika/gpu/em/Types.hpp` | 局部密度、LPM record、suppressed queue、统计和 fallback reason |
| `src/gpu/em/CudaPhotonPairFinalState.cu` | LPM 判断、第四类 scan/compaction、父光子保留 |
| `src/gpu/em/CudaEmBackend.cu` | 从 v4 cache 建立快照并累计 LPM trial/suppression |
| `corsika/gpu/em/tables/RateTable.hpp` | v4 LPM metadata schema |
| `src/gpu/em/tables/RateTable.cpp` | `C8EMRT04` 编解码、SHA 和严格验证 |
| `applications/gpu_em_tablegen.cpp` | \(E_{\rm LPM}\) 积分及 PROPOSAL 写前 oracle |
| `tests/gpu/testGpuPhotonPairLpm.cu` | 设备公式对 PROPOSAL 的逐点 oracle |
| `tests/gpu/testGpuPhotonPairFinalState.cpp` | 接受/抑制/回退/继续四队列端到端验证 |

## 8. 验证结果

### 8.1 CUDA 公式 oracle

`testGpuPhotonPairLpm` 对 450 个
component/energy/\(x\)/density 点比较 PROPOSAL、host 移植和 CUDA：

```text
1802 checks
maximum relative error = 8.37645e-16
PROPOSAL Air eLPM = 1.88062e12 MeV
```

测试还确认未知 component 和零密度不会被接受。

### 8.2 synthetic 末态 batch

fixture 使用较小的 synthetic \(E_{\rm LPM}\)，专门让一次 batch 同时经过接受
和抑制分支：

```text
4092 input interactions
1455 accepted GPU pairs
171 LPM-suppressed photons
2465 explicit fallbacks
1 continuation
115257 checks
```

检查内容包括：

- CPU reference 与 CUDA 的逐记录概率和随机数相等；
- 被抑制 photon 的全部物理状态不变；
- suppressed parent 的 `step_id` 恰好加一；
- 稳定输出顺序；
- wavefront 完全反序后每个 history 的物理结果不变；
- 非法局部密度走 `InvalidMassDensity`；
- 四类队列数量闭合；
- LPM trial 和 suppression backend statistics。

### 8.3 完整 dry-air v4 表

正式表：

```text
/tmp/c8_gpu_em_full_strict_v4_phase11.c8emrt
```

结果：

| 项目 | 数值 |
|---|---:|
| 文件大小 | 15,460,818 bytes |
| payload SHA-256 | `e237f8fde80676001a0307de56876c15807de59e5257197c489f906f893e8d4a` |
| 完整文件 SHA-256 | `caa435839bfa995f11c631dd896fc20a5f3bf81cc576ce905bfc2807d43bae42` |
| 最大 rate 相对误差 | \(9.982391\times10^{-4}\) |
| 最大 inverse-CDF 相对误差 | \(7.499968\times10^{-4}\) |
| tablegen wall time | 42.53 s |
| tablegen peak RSS | 99,580 KiB |
| CUDA table bytes | 15,426,740 |

相同热 cache 再生成
`/tmp/c8_gpu_em_full_strict_v4_phase11_repeat.c8emrt`，两个文件经 `cmp`
逐字节相同，完整文件 SHA-256 相同。

完整表测试：

```text
CUDA table query:
  13609 checks, 2720 queries

CUDA interaction selection:
  155207 checks, 8192 particles
  8116 selected, 76 fallback

CUDA photon-pair final state + LPM:
  257387 checks, 10173 interactions
  3142 accepted pairs
  6 LPM-suppressed photons
  7024 explicit fallbacks
  1 continuation
```

### 8.4 回归

| 回归项 | 结果 |
|---|---|
| CUDA `testGpu*` | 10/10 通过 |
| CPU GPU-schema/table tests | 3/3 通过 |
| `testGpuEmHost` | 163 checks |
| `testGpuEmRateTable` | 36 checks |
| `testGpuEmFlatRateTable` | 152 checks |
| `[ScalarCascadeStepper]` | 3 assertions |
| `[HybridCascade]` | 12 assertions |

尝试构建仓库的无目标 `all` 时，外部 Pythia 8.315 下载 URL 返回 HTTP 404。
这不影响上表所有定向 GPU/CPU 构建和测试，但在重新创建全新构建目录前需要
修复或镜像该外部依赖。

## 9. 复现命令

必须在 `conda run` 内部移除 `FLUPRO`，否则 Conda 激活脚本可能重新注入它：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv \
  env -u FLUPRO \
  cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target testGpuPhotonPairLpm testGpuPhotonPairFinalState \
  -j2
```

运行 CUDA 回归：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv \
  env -u FLUPRO \
  ctest --test-dir \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  -R '^testGpu' --output-on-failure
```

生成完整 v4 表：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv \
  env -u FLUPRO \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/\
gpu_em_tablegen /tmp/c8_gpu_em_full_strict_v4_phase11.c8emrt \
  --proposal-cache /tmp/c8_gpu_em_proposal_cache \
  --energy-min-MeV 0.6 \
  --energy-max-MeV 1e12 \
  --cut-MeV 0.5 \
  --tolerance 1e-3 \
  --loss-tolerance 1e-3 \
  --initial-intervals 16 \
  --max-points 20000 \
  --loss-initial-energy-intervals 8 \
  --loss-initial-quantile-intervals 8 \
  --loss-max-energy-points 4096 \
  --loss-max-quantile-points 2048 \
  --loss-validation-samples 64
```

## 10. 当前限制与下一阶段

本阶段没有声称完成以下工作：

1. `EmInteractionRecord::mass_density_g_per_cm3` 还没有由真实大气 tracking
   自动填入；验证测试显式提供局部密度。
2. interaction grammage 尚未与球形层边界、磁场步长和观测面竞争。
3. accepted secondaries 和 suppressed parents 尚未直接写回 resident GPU SoA。
4. validation bridge 仍为每个调用分配和释放临时设备内存。
5. bremsstrahlung LPM 与 electron/positron pair-production LPM 尚未移植。
6. CPU fallback 末态还没有批量接回 `ProposalFinalStateGenerator`。

所以下一阶段最合理的工作不是继续增加更多离散过程，而是先实现“真实
photon tracking 到一个最近限制”：

- 从球形五层大气计算当前 layer 和局部密度；
- 对 sampled interaction grammage 做密度积分与逆积分；
- 与 layer boundary、observation surface 和最大磁偏转步长竞争；
- 在真实 interaction vertex 填入 `mass_density_g_per_cm3`；
- 将本阶段接受、抑制和 fallback 三条物理路径接回 resident wavefront。

完成这一步后，LPM 对能量和高度的耦合才会真正进入一个完整 shower。
