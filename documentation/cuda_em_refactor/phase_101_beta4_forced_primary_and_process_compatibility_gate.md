# Phase 101：beta4 强制初级顶点与全过程序列兼容性门禁

## 1. 问题确认

`Cascade::forceInteraction()` 和 `forceDecay()` 的语义是：让调度器下一次取出的
粒子立即在当前位置发生相互作用或衰变。它们控制的是 shower 内的物理顶点，
不是缩短运行时间的调试开关。

此前 `HybridCascade` 把这两个标志直接写入 `ScalarCascadeStepper`，但在真正调用
stepper 之前先询问 CUDA router。可直接路由的 photon、electron、positron 或
muon 初级会绕过标量 stepper，因此强制请求没有被消费。实际缺陷可由两个固定
种子事例重现：

| 配置 | 原版 CPU 顶点 | 修复前 CUDA、`gpu-min-batch=1` |
|---|---:|---:|
| 1 GeV photon，`theta=80 deg`，强制相互作用 | `time=0`、`X=0 g/cm²` | `time=2.806 us`、`X=76.773 g/cm²` |
| 10 GeV muon，`theta=80 deg`，强制衰变 | `time=0`、`X=0 g/cm²`、3 个次级 | 先在 `time=82.01 ns`、`X=2.227 g/cm²` 普通相互作用，2 个次级 |

较大的 `gpu-min-batch` 可能让 CPU wavefront expansion 偶然先处理初级，掩盖该
缺陷。因此不能把“某个批量参数下结果正确”作为接口语义成立的证据。

## 2. 强制初级顶点修复

`HybridCascade` 现在保存一个互斥的 one-shot 状态：`None`、`Interaction` 或
`Decay`。调度器取得下一粒子后按以下顺序处理：

1. 检查并消费待执行的强制动作；
2. 把动作设置到 `ScalarCascadeStepper`；
3. 禁止这一粒子进入 CUDA router，并执行一次标量强制顶点；
4. 正常回收末态；随后产生的 photon、electron、positron 和 muon 仍可进入 GPU。

这样只把用户指定的第一个顶点固定到 CPU，不把整个后续 shower 退回 CPU，
也不依赖 GPU 最小批量。连续设置同一种动作保持幂等；同时请求强制相互作用和
强制衰变会在运行前抛出错误，避免含糊地选择其中一个。

`gpu_em/summary.yaml` 新增：

```yaml
forced_primary:
  interaction_executed: 0
  decay_executed: 0
```

计数只在调度器真正消费动作时增加，可用于确认 production 命令的请求已经执行。

## 3. 兼容性门禁扩展

旧门禁只递归检查 `ContinuousProcess`，phase 99 扩展到
`SecondariesProcess`，仍可能让新加入的 interaction、decay、boundary 或 stack
过程被 CUDA 路由静默绕过。本阶段把启动前的闭集检查扩展为六类：

- `ContinuousProcess`；
- `SecondariesProcess`；
- `InteractionProcess`；
- `DecayProcess`；
- `BoundaryCrossingProcess`；
- `StackProcess`。

每个过程必须显式声明一种保持语义：设备端等价实现、由设备记录回放、延迟到
CPU、对 routed EM 不适用，或只作诊断。`c8_air_shower` 当前共登记 14 个过程
合同：4 个设备替代、6 个记录回放、1 个 CPU 延迟、2 个不适用于 routed EM、
1 个诊断过程。任一类别出现未登记类型，CUDA 模式会在第一个 shower 启动前
抛出异常，而不是继续产生缺少过程的输出。

输出 metadata 同时记录五种策略的数量以及六类未登记计数。正式输出必须满足：

```yaml
process_registry:
  registrations: 14
  unregistered_continuous_processes: 0
  unregistered_secondaries_processes: 0
  unregistered_interaction_processes: 0
  unregistered_decay_processes: 0
  unregistered_boundary_processes: 0
  unregistered_stack_processes: 0
  accepted: true
```

## 4. 自动化验收

新增 framework 测试使用一个会立即接收初级的伪 GPU router，验证强制动作在
`canRoute()` 之前执行，初级不被 stage，传播步数为零，强制末态只生成一次，
并验证相互作用/衰变互斥。新增 host 测试分别放入未登记的六类过程，确认每一类
都触发 fail-closed；补齐登记后门禁通过且策略计数正确。

验收结果：

- Release/CUDA 完整 CTest：34/34 通过，包含 FLUKA；
- Python GPU 验证：241/241 通过；
- 代码格式门禁：`git diff --check` 通过。

## 5. 固定种子 production 验收

对上述 photon 和 muon 事例重新运行 CUDA，均设置
`gpu-min-batch=1`，使初级在缺少强制仲裁时本可直接进入 GPU。修复后：

- photon 的 `interactions.parquet` 和 `interactions/summary.yaml` 与 CPU 基线
  完全相同，强制顶点为 `time=0`、`X=0 g/cm²`、2 个次级；
- muon 的相同两份 interaction 输出与 CPU 基线完全相同，强制顶点为
  `time=0`、`X=0 g/cm²`、3 个次级；
- 两者的 CPU wavefront expansion 均为零，证明正确结果不是小批量 CPU
  expansion 偶然造成；
- photon 在 `gpu-min-batch=1`、128 和 4096 下都得到与 CPU 相同的第一顶点，
  证明强制语义不依赖性能参数。

这里要求逐表相同的是用户强制指定的第一个标量顶点。之后 CPU 和 production
CUDA 使用不同的随机数寻址与调度，后续独立 shower 不承诺逐粒子相同。

另外，用未设置任何强制选项的 1 GeV、`theta=80 deg` electron 固定种子事例
对修复前后做回归。纵向 profile、能损、地面粒子、相互作用、production
profile、CoREAS 和 ZHS 共七个 Parquet 文件的 SHA-256 与字节数全部相同，证明
新的 one-shot 仲裁没有改变普通 CUDA 演化路径。

## 6. 结论

beta4 现在保持原 `Cascade` 的强制初级接口语义：指定顶点先执行，后续末态仍可
获得 CUDA 加速。全过程序列门禁将以后新增过程从“可能被静默跳过”变为“必须
显式说明 CUDA 下如何保持”，并把可审计状态写入每个 shower 的 GPU summary。
