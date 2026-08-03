# Phase 59：Epair 包络回退的命名、授权与逐位等价

## 1. 问题

`ProposalCpuFallbackHandler::canHandle()` 原先只验证 fallback event 是否带有
完整的过程、目标组分和能损 \(v\)。这意味着一个数据字段完整的
`invalid_final_state` 也可能被 CPU PROPOSAL 接走，生产严格模式无法区分：

1. 预先定义的稀有过程回退；
2. 可审计的设备采样能力边界；
3. 真正的数值或末态错误。

Phase 58 的 1000 个 CUDA shower 中恰好出现三次 Epair rejection envelope
violation，使这个策略漏洞可以被真实生产样本重现。

## 2. 实现

新增：

```cpp
ProposalFallbackReason::EpairRejectionEnvelopeExceeded
```

输出名称为：

```text
epair_rejection_envelope_exceeded
```

`ProposalCpuFallbackHandler` 现在要求“事件数据完整”和“原因/过程组合被批准”
同时成立：

- `CpuOnlyProcess` 必须对应 process capability registry 中的 CPU-only
  final state；
- inverse-CDF 能区或分位边界必须保留可解析的 loss quantile；
- `EpairRejectionEnvelopeExceeded` 必须对应 electron/positron 的 Epair
  过程；
- `InvalidFinalState`、非法介质密度、LPM 参数缺失、transport/table 数值错误
  均不能因字段完整而被接走。

包络越界记录额外保存 Epair sampler status、\(\rho_{\max}\) 和 trial count，
便于定位真实的 envelope underestimate。

## 3. 测试

测试明确覆盖：

- 合法 CPU-only specified final state 被接受；
- 合法 Epair envelope fallback 被接受；
- 相同事件改成 Brems process 后被拒绝；
- 字段完整的 `InvalidFinalState` 被拒绝；
- inverse-CDF selected-loss completion 继续工作。

结果：

```text
GPU/CUDA CTest                         25 / 25 passed
Python validation unit tests          38 / 38 passed
c8_air_shower Release target          built
```

## 4. 1000-shower 精确重放

使用 Phase 58 相同 CUDA seed、配置和表格重新运行：

```text
/tmp/c8_phase59_electron_1TeV_cut50_classified_1000_v1
```

聚合统计：

| quantity | Phase 58 | Phase 59 |
|---|---:|---:|
| complete showers | 1000 | 1000 |
| GPU particles | 62,347,236 | 62,347,236 |
| CPU-only process | 12,041 | 12,041 |
| loss-quantile fallback | 598 | 598 |
| invalid final state | 3 | 0 |
| Epair envelope exceeded | 0 | 3 |
| overflow/spill/generic fallback | 0 | 0 |

`verify_reused_backend_equivalence.py` 对 1000 个配对 shower 检查：

```text
25 scalar columns        exact
6 curve observables      exact
3 ground histograms      exact
overall                  PASS
```

所有数值的 `different_values` 都是 0。该修改只改变失败语义、授权和诊断，
没有改变任何物理轨迹或输出 observable。

## 5. 结论

生产模式现在满足：

> 已注册的能力边界可以显式、计数地回到 CPU；真正的非法末态不能被一个宽泛
> 的“按过程可处理”判断掩盖。
