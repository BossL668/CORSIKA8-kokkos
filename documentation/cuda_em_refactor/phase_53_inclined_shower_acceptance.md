# Phase 53：80° 倾斜 shower 验收

## 1. 目的

把正式物理矩阵从垂直和 45° 扩展到 80°，重点覆盖：

- 球形五层大气中的超长几何路径；
- 大气层边界和近切向求交；
- 磁场偏转与多重散射的长程累计；
- thinning 对远离 shower 主体的稀疏 profile 尾部的影响。

配置为 1 TeV electron、0.5 MeV cut。第一组 CPU/GPU 各 200 个事件，使用
EM thinning \(10^{-4}\)、最大权重 100。

## 2. 200 + 200 thinning 结果

运行目录：

```text
/tmp/c8_phase53_electron_1TeV_zenith80_200_v1
```

主要结果：

- energy deposit、charged、EM、photon、positron 曲线通过；
- electron profile 在极稀疏尾部有 2 个 bin 未通过；
- charged integral 相对差 \(0.1061\%\)，\(z=0.102\)；
- deposit sum 相对差 \(0.00500\%\)，\(z=0.346\)；
- photon integral 相对差 \(1.099\%\)，但只有 \(0.404\sigma\)；
- ground EM count 和 energy 在这一配置下都为零。

严格总体状态保留为 FAIL，因为 electron 曲线仍有失败 bin，且 4 个关键均值
超过 1% 原始阈值但统计精度不足。这里不能把“统计不显著”写成物理通过。

## 3. 去 thinning 诊断

为了区分 GPU 输运差异和高权重尾部涨落，又运行 CPU/GPU 各 20 个不 thinning
事件：

```text
/tmp/c8_phase53_electron_1TeV_zenith80_unthinned20_v1
```

结果：

- 全部 longitudinal 和 ground curve gate 通过；
- \(X_{\max}\) 相对差 \(0.652\%\)；
- charged maximum 相对差 \(0.679\%\)；
- charged integral 相对差 \(0.870\%\)；
- deposit sum 相对差 \(0.00694\%\)；
- photon integral 和 deposit \(X_{\max}\) 因 20-event 样本仍为统计不充分。

这说明 200-event thinning 运行的两个 electron 尾部 bin 更符合 thinning
高权重涨落，而不是可重复的 tracking 或 CUDA 物理偏差。正式矩阵仍记录两个
运行的严格状态，不删除第一个 FAIL。

## 4. 结论

80° 的五层球形 tracking、边界、长路径连续损失和主要 shower shape 已得到
应用级覆盖。要把该矩阵项升级成无条件 PASS，仍需扩大 unthinned 样本，或
为 thinning 尾部采用预注册的统计区间，而不能事后放宽 bin 门禁。
