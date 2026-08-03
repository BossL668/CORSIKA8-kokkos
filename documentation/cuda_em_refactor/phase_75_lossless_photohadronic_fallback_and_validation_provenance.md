# Phase 75：无损光核末态回退与统计样本构建来源门禁

## 1. 审计发现

在扩展 \(10^{17}\,\mathrm{eV}\) electron 的 CPU/CUDA 独立样本时，旧版本
反复输出：

```text
HE interaction model cannot handle ... Rho0, target argon ...
Skipping secondary production!
```

新增的 200 个 scalar PROPOSAL 和 200 个 CUDA shower 中分别出现 100 次和
84 次，共 184 次，即每个 shower 平均 0.46 次。这不是可以忽略的极稀有
边界。

问题位于 PROPOSAL photoproduction 已经选择过程、靶组分和能损以后：
SIBYLL 不接受氩靶的 \(\rho^0\)-nucleus 末态。旧代码返回
`ProcessReturn::Ok`，却没有生成任何次级。父光子随后被相互作用消费，因此
这相当于静默破坏粒子与能量账本。

这批旧样本只保留为缺陷诊断。即使 CPU/GPU 都走过同一个错误分支，也不能作为
“稀有过程不丢弃”的生产验收证据。

## 2. 物理模型回退

新增：

```text
corsika/modules/proposal/HadronicInteractionModelFallback.hpp
```

生产配置现在采用：

```text
preferred: SIBYLL
fallback:  QGSJet-II
```

适配器的契约是：

1. 重新按每核子四动量计算 \(\sqrt{s_{NN}}\)；
2. preferred model 支持时保持原 SIBYLL 末态；
3. preferred 拒绝而 QGSJet-II 支持时，生成同一个已选 photoproduction
   顶点的末态，并增加显式 fallback 计数；
4. 两个模型都拒绝时抛出异常，禁止以空末态继续。

`HadronicPhotonModel` 的高、低能拒绝分支也都由“警告并返回成功”改成
hard failure。这样任何未来未覆盖靶核或能区都不会再次被当成已完成相互作用。

每个 CUDA shower 新增：

```yaml
photo_hadronic_generator:
  preferred_interactions: ...
  fallback_interactions: ...
  discarded_final_states: 0
```

QGSJet-II 的初始化本身较重。生产对象因此由
`LazyHadronicInteractionModel<QGSJetII>` 包装：adapter 先查询 SIBYLL，
只有 SIBYLL 实际拒绝所选配置时才构造 QGSJet-II。没有 Argon 回退的短作业
不会为备用模型支付固定启动代价；summary 的
`fallback_model_initialized` 使该行为可审计。

## 3. 直接验证

模块级测试覆盖：

- preferred 拒绝 Argon、fallback 接管且只计数一次；
- preferred/fallback 都拒绝时 hard failure；
- 低能 nucleon final-state generator 拒绝 proton/neutron 时 hard failure；
- QGSJet-II 明确接受 `Rho0 + Argon`。

真实应用使用修复后二进制运行了 10 个
\(10^{17}\,\mathrm{eV}\) electron CUDA shower：

```text
events complete                       10 / 10
QGSJet-II Argon fallback              3
discarded_final_states                0
旧 "Skipping secondary production"    0
```

输出位置：

```text
/tmp/c8_phase75_photohadronic_fallback_cuda10_v1
```

## 4. 为什么需要二进制哈希

本阶段之前，增量统计工具会检查物理 CLI，但历史输出只保存可执行文件路径。
同一个 build 路径在重新编译后仍然相同，因此旧物理版本和新物理版本可能被误
合并。Phase 74 的诊断性扩展正好暴露了这个风险。

`run_physics_acceptance.py` 现在为每个新输出写入：

```text
validation_provenance.json
```

其中包含：

- `c8_air_shower` 的 SHA-256、大小、mtime 和路径；
- CUDA rate table 的 SHA-256、大小、mtime 和路径；
- runner 本身的来源哈希；
- 精确命令及命令哈希。

runner 在 subprocess 全部结束后重新哈希 executable 和 table。运行期间任何
一个文件发生变化都会使本次输出失去合并资格。

`compare_ensembles.py` 与增量 pooling 现在要求：

1. 同一 backend 的所有 shard 物理配置一致；
2. executable SHA-256 一致；
3. CUDA table SHA-256 一致；
4. proposal 与 CUDA 使用同一个 executable SHA-256。

旧输出默认被拒绝；`--allow-legacy-provenance` 只允许诊断，不构成生产证据。

端到端 smoke：

```text
/tmp/c8_phase75_provenance_runner_smoke_v1
```

它生成两个 scalar shard 和一个 CUDA 输出，三个 sidecar 的 executable
SHA-256 完全相同，CUDA sidecar 还记录了 rate-table SHA-256。尝试把没有
sidecar 的 Phase 61 输出加入新样本时，在任何 shower 启动前即被拒绝。

## 5. 对后续验收的影响

Phase 75 以前的统计结果仍可用于定位 GPU/CPU 差异和估计方差，但不再作为最终
生产物理结论。正式统计矩阵必须使用修复后的同一 executable hash 重新建立；
后续扩样只能合并具有相同 build/table provenance 的样本。
