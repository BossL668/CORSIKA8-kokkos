# Phase 47：强子混合链路与可读 CPU fallback provenance

## 1. 目标

纯 EM shower 不能覆盖 HybridCascade 的全部所有权边界。生产事件还必须保证：

- 初级强子、强子相互作用、μ、τ 和衰变由原 CORSIKA CPU 路径处理；
- CPU 产生的 \(\gamma/e^\pm\) 能进入常驻 CUDA EM 队列；
- CUDA 精确选中的稀有 EM 过程由 CPU 只生成指定末态；
- fallback 统计既可稳定地由数字 ID 读取，也能由人直接审计。

## 2. 强子—CUDA EM 冒烟

使用 1 TeV proton、45°、默认 0.3 GeV hadron/muon/tau cut、0.5 MeV EM
cut、\(10^{-4}\) EM thinning 和 CPU radio backend。输出：

```text
/tmp/c8_phase47_hadron_hybrid_smoke_v1
/tmp/c8_phase47_hadron_hybrid_smoke_v2
```

第一组事件结果：

```text
status / complete                 complete / true
shower wall time                  1226.73 ms
GPU EM particles                 297181
CPU scalar particle steps        4934
CPU specified final states       8
CPU generic fallback steps       0
cross-species host spill         0
queue overflow                   0
final pending photon/lepton      0 / 0
```

实际指定末态包含 photoproduction、photonuclear，以及一次表格 loss quantile
越界后的精确 CPU selected-loss completion。这证明 fallback 不是死代码，并且
强子与 CUDA EM 队列最终均完全 drain。

该事件不是纯 EM 输入，并且含小 batch 标量展开，所以 strict GPU-only energy
ledger 正确标记为 `complete_coverage: false`；不能把该局部账本误当作完整强子
事件能量守恒检验。

## 3. 可读名称

新增：

- `gpuEmProcessName(process_id)`；
- `proposalFallbackReasonName(reason)`。

`gpu_em/summary.yaml` 同时写出：

```yaml
cpu_fallbacks_by_process:
  1000000014: 8
cpu_fallbacks_by_process_name:
  photoproduction: 8
cpu_fallbacks_by_reason:
  12: 8
cpu_fallbacks_by_reason_name:
  cpu_only_process: 8
```

数字字段保持向后兼容；名称字段用于人工审核和论文 provenance。所有已知 GPU
过程、CPU-only 过程及 24 个 fallback reason 都有确定名称，未知值统一写为
`unknown`，不会用数组越界或未定义 enum 字符串化。

## 4. 验证

`testGpuEmHost` 增加已知过程、已知原因和未知过程映射检查。应用重新构建后，
第二个独立 seed 的 proton shower 成功结束，并实际写出上面的 named
photoproduction fallback。这个阶段证明的是单事件端到端链路；完整 21CMA
强子 ensemble 的统计验收仍是独立待办项。
