# 阶段 32：transport、vertex 与 final-state 的设备端计数链

## 1. 本阶段目标

阶段 31 已经删除了 interaction selection 与 transport 之间的 host
同步，但每个非空 wavefront 后仍存在以下控制链：

```text
photon transport
    -> scan Interaction records
    -> D2H interaction count
    -> photon final-state
    -> endpoint summary D2H

lepton transport
    -> scan InteractionCandidate records
    -> D2H candidate count
    -> vertex reselection
    -> D2H interaction/continuation/fallback counts
    -> lepton final-state
    -> endpoint summary D2H
```

这些中间计数只决定后续 CUDA kernel 的有效输入前缀，CPU 在 endpoint
完成前不消费相应粒子。因此，本阶段把精确计数保留在 device，并让后续
kernel 使用：

```text
host launch upper bound + device exact prefix count
```

最终所有计数随 endpoint summary 一次返回。

## 2. 通用 transport-interaction batch

`DeviceTransportInteractionBatch` 新增：

```cpp
std::uint32_t* device_interaction_count;
bool count_deferred;
```

photon 与 lepton extraction 都执行稳定的：

1. classification flag；
2. CUB exclusive scan；
3. 单线程 device count reduction；
4. stable compaction。

独立验证 API 默认只下载一个 32-bit count。生产 pipeline 设置
`defer_count_download=true`，不发生 D2H。

该改变不使用 atomic append，因此 compact 后的 interaction 顺序仍与
transport record 顺序相同。

## 3. Photon：transport count 直接驱动 final-state

生产 photon pipeline 以 transport record count 作为 launch 上界，把
device interaction count 传给 final-state classification：

```text
transport upper bound = N_transport
exact final-state input = device N_interaction
```

每个 classification 线程先清零全部 scan flags，然后检查精确前缀：

```cpp
if (device_input_count != nullptr &&
    index >= *device_input_count) {
  return;
}
```

因此无效 suffix：

- 不读取未初始化的 compact interaction；
- 不产生 fallback；
- 不占用 secondary history ID；
- 在所有 scan 中贡献零。

`PhotonFinalStateSummaryLayout` 增加 `InputCount`。endpoint 下载 combined
summary 后，同时恢复：

- transport interaction count；
- final-state 的八类计数；
- endpoint、observation、cut 和 charged-secondary count。

相对于阶段 31，每个非空 photon transport wavefront：

```text
blocking D2H synchronization: -1
control D2H bytes:            -4
```

旧 extraction 下载 8 byte scan tail；新 combined summary只增加一个
4-byte input count。

## 4. Lepton：两级 device count 传递

### 4.1 Transport extraction 到 vertex

`LeptonVertexSummaryLayout` 保存：

```text
input candidate count
selected interaction count
continuation count
fallback count
error
```

vertex selection 使用 transport extraction 的 device count屏蔽无效
suffix，并把三路结果稳定 compact。它不再立即下载六个 scan tail。

### 4.2 Vertex 到 final-state

bremsstrahlung/annihilation/ionization/electron-pair final-state 以 vertex
input upper bound 启动，实际有效前缀直接读取：

```text
vertex_summary[InteractionCount]
```

十一组 final-state classifications 均先清零 flag，再检查 device exact
count。final-state summary 的守恒条件也比较实际 vertex interaction
count，而不是 launch 上界。

### 4.3 Vertex continuation 到 endpoint

endpoint 使用：

```text
vertex_summary[ContinuationCount]
```

直接散射 vertex continuation；无需 host count。最终 lepton endpoint
summary由三部分组成：

```text
4-word endpoint header
+ 5-word vertex summary
+ 12-word final-state summary
= 21 words / 84 bytes
```

一次 D2H 同时恢复 transport candidate、vertex 三路结果、final-state
计数和 endpoint 队列计数。

对存在 interaction candidate 的 lepton wavefront，相对于阶段 31：

```text
blocking D2H synchronizations: -2
control D2H bytes:             -12
```

其中 12 byte 来自旧 extraction 的 8 byte和旧 vertex scan tail 的
24 byte，减去新增 20-byte vertex summary。

若某个 transport wavefront 没有 interaction candidate，仍删除
extraction 同步；summary 保持统一固定布局，使生产路径没有 host 分支。

## 5. 错误与确定性约束

所有 deferred consumer 均验证：

- producer 的 deferred 状态有效；
- device count/summary 指针非空；
- producer output 指针与 consumer input 指针完全相同；
- launch upper bound 与 producer input count完全相同。

device summary继续执行：

- 输入分类守恒；
- secondary 数量守恒；
- history ID overflow；
- endpoint source-slot 唯一性；
- 非法方向与非法 final state 检查。

未开启 `--use_fast_math`，随机数 key、process draw ID、stable scan 和
history ID分配均未改变。

## 6. 测试结果

完整 CUDA 回归：

```text
24/24 passed
```

其中 photon 测试包含 selection mixed fallback，lepton 测试包含：

- electron/positron mixed batch；
- selection fallback；
- Molière scattering；
- vertex interaction/continuation/fallback；
- 四类 final state；
- endpoint、能量闭合与 history overflow。

## 7. 固定种子生产验证

### 7.1 10 GeV

| 指标 | 阶段 31 | 阶段 32 |
|---|---:|---:|
| GPU particles | 15,392 | 15,392 |
| photon / lepton wavefronts | 27 / 107 | 27 / 107 |
| eliminated synchronizations | 266 | 505 |
| eliminated control D2H | 5,376 B | 6,744 B |

### 7.2 1 TeV

| 指标 | 阶段 31 | 阶段 32 |
|---|---:|---:|
| GPU particles | 576,015 | 576,015 |
| photon / lepton wavefronts | 35 / 442 | 35 / 442 |
| eliminated synchronizations | 953 | 1,871 |
| eliminated control D2H | 20,384 B | 25,816 B |

### 7.3 1 PeV 与 GPU CoREAS/ZHS

| 指标 | 阶段 31 | 阶段 32 |
|---|---:|---:|
| GPU particles | 17,645,186 | 17,645,186 |
| photon / lepton wavefronts | 107 / 1,194 | 107 / 1,194 |
| photon transport→final-state fusions | 0 | 107 |
| lepton transport→vertex fusions | 0 | 1,194 |
| lepton vertex→final-state/endpoint fusions | 0 | 1,192 |
| eliminated synchronizations | 2,600 | 5,093 |
| eliminated control D2H | 55,444 B | 70,176 B |
| core runtime | 26.223 s | 25.768 s |
| router runtime | 11.499 s | 11.352 s |
| photon backend | 0.850 s | 0.860 s |
| lepton backend | 10.550 s | 10.393 s |
| outer wall time | 29.86 s | 29.37 s |

这是单次墙钟比较，只说明没有性能回退，不能单独作为显著加速结论。

## 8. 逐字节物理等价

10 GeV、1 TeV、1 PeV 的所有 Parquet/NPZ 输出均与阶段 31逐字节一致。
1 PeV 七个关键 SHA-256 为：

```text
profile:
3db76ad9089708601dacf98464a062285858c89553dc50f36a3840d3a323a792

dEdX:
37ea203f3faa372f9e6651dbe31e9cec3d3b639f21695547b46e9d7ab6be1986

ground particles:
7ab4b8c11d16ec819dd17cc0c381acc956180c1ee30cd758c09f55fcb663a204

interactions:
6b529acf3d26f1349f298da3f9c5b7e8d50e7aa452939698212bf969d435e06d

production profile:
14f0fc114c4671c25d80cf58607115b85f3b83803dd21078f78d1255de95a54a

CoREAS:
6b2ad3f1b5dbbb000e1e3fcb5743f244a70f841abe7479bcf30f122382b68cee

ZHS:
047973c559a6d642eb36ef19e734699e0f5a910045f066d6721bef46a7539692
```

这也验证了 radio accumulator 接收到的 \(e^\pm\) track segment 顺序与
数值没有变化。CoREAS/ZHS 的固定点波形结果完全一致。

## 9. 与射电计算同步方式的关系

CORSIKA 8 的 CPU CoREAS/ZHS process语义是“粒子每生成一段轨迹，就把该
segment交给射电模块累积”，并不是等整个 shower结束后重新遍历全部
粒子。

CUDA 后端保持同一物理语义，但按 wavefront 批处理：

```text
lepton transport产生 track segments
    -> GPU radio accumulator消费本批 segment
    -> 粒子进入下一 wavefront
    -> shower结束时 writer输出已累积波形
```

因此阶段 32删除的是 stage control count 的 host同步，不是取消射电计算
与输运之间的数据依赖。radio kernel仍在复用 transport workspace 前完成
对相应 track segment 的消费。

## 10. 下一步

下一步进入 \(10^{17}\)–\(10^{18}\,\mathrm{eV}\) 多 seed验收：

1. 检查显存上限、resident queue、spill 和 overflow；
2. 统计每种控制融合在 UHE shower 中的实际触发次数；
3. 对同 seed重复运行验证逐字节确定性；
4. 对多 seed CPU/GPU 物理 observable做统计比较；
5. 使用 Nsight Systems确认剩余同步来自必要的 endpoint消费还是仍可
   device-resident。
