# 阶段 28：融合 photon endpoint 与 charged-secondary 计数同步

## 1. 问题

阶段 27 已把 photon 末态产生的 \(e^-/e^+\) 保留在设备队列，但每个
photon wavefront 仍包含两个串行的主机同步点：

```text
photon endpoint compaction
    -> 下载 4 × uint32 endpoint summary
    -> CUB DeviceSelect::If(charged secondaries)
    -> 再下载 1 × size_t selected count
```

第二次下载只有 8 byte，却会阻塞 CPU，直到此前默认流中的全部工作完成。
在 1 PeV 固定样本中共有 107 个 photon wavefront，因此发生 107 次没有
物理数据价值的额外同步。

本阶段不增加逐粒子 workspace，也不改用不稳定的原子 append。它保留
CUB stable select，但把选择和计数纳入已有 endpoint 流水线，使一次
endpoint summary 同时返回 photon、observation、cut 和 charged-child
计数。

## 2. 新接口

`DeviceBatchStages.hpp` 新增非 owning sink：

```cpp
struct DeviceChargedSecondarySink {
  EmParticleState* output;
  std::size_t capacity;
  void* temporary_storage;
  std::size_t temporary_storage_bytes;
  std::size_t* selected_count;
};
```

其中：

- `output` 指向 `CudaEmBackend` 持有的 persistent lepton queue 尾部；
- `capacity` 是该连续尾区剩余容量；
- CUB 临时存储和 count 都在 shower 初始化时一次性分配；
- endpoint 模块只借用这些指针，不取得所有权。

`DevicePhotonEndpointBatch` 同时新增：

```cpp
std::size_t generated_lepton_count;
bool generated_leptons_compacted;
```

布尔量必须显式为真，后端才允许增长 pending queue。这样能区分：

- 成功选择，但 charged child 数量恰好为 0；
- 队列连续容量不足，根本没有执行选择。

## 3. 融合执行顺序

新的 `compactPhotonEndpointsOnDevice()` 顺序为：

```text
scatter transport/suppression/Compton photon endpoints
    -> scan next-photon flags
    -> scan observation flags
    -> CUB stable select charged final-state children
       directly into persistent lepton queue
    -> finalize one 6 × uint32 summary
    -> one D2H summary copy
    -> compact photon and observation endpoints
```

summary 布局为：

```text
[0] next photon count
[1] observation count
[2] endpoint error
[3] particle-cut count
[4] selected charged-child count
[5] charged selection succeeded
```

CUB 仍保持 final-state secondary 数组的相对顺序，因此 history ID、
generation、step ID 和同一顶点的 \(e^-/e^+\) 次序不会依赖 warp 调度。
没有使用全局原子计数来决定输出位置。

## 4. 容量和失败策略

后端在启动 photon pipeline 前构造当前 lepton queue 尾部的 sink。只有
当完整 `final_state_secondary_count` 能放入该连续尾区时，endpoint 模块
才执行 `DeviceSelect::If`。这里使用完整 secondary 数作为保守上界，避免
在不知道精确 charged 数量时写越界。

容量不足时：

1. `generated_leptons_compacted=false`；
2. pending queue 不增长；
3. `cross_species_host_spills` 增加；
4. 完整 secondary batch 下载到 host；
5. host 稳定过滤非 photon child，并交还现有 router。

因此容量不足只影响性能，不改变物理，也不会截断输出。

测试现在会在 1% 显存预算下反复填充 lepton queue，直到真实触发这个
路径，并要求：

- host 返回 charged secondaries 非空；
- pending count 不超过设备容量；
- 已经入队的粒子仍然保留；
- 不出现 CUDA queue overflow。

## 5. 显存和传输变化

设备常驻分配不变：

- 两个 cross-species 粒子队列不变；
- CUB select temporary 不变；
- device selected-count 不变。

endpoint summary 从 16 byte 增加到 24 byte，而原来的独立 selected-count
下载是 8 byte。因此传输的字节总量理论上相同，但同步次数从两次降到
一次。当前统计过去只单独计算第二次 8-byte 下载，所以 summary 中的
`device_to_host_bytes` 会严格减少：

\[
\Delta B_{\mathrm{D2H}}
  = 8\,N_{\mathrm{photon\ wavefront}}.
\]

## 6. 生产验证

编译目标：

```text
CORSIKA8GpuEm
testGpuPhotonWavefront
testGpuLeptonTransport
testGpuHybridRoute
c8_air_shower
```

均在 `corsika_venv`、CUDA Release 构建中成功。完整 CUDA 测试结果：

```text
24/24 passed
```

### 6.1 10 GeV

固定 seed 25001，resident profile，ring=0：

| 指标 | 阶段 27 | 阶段 28 |
|---|---:|---:|
| photon wavefronts | 27 | 27 |
| GPU particles | 15,392 | 15,392 |
| D2H | 193,016 B | 192,800 B |
| 减少量 |  | \(27\times8=216\) B |
| host spill | 0 | 0 |

两个独立阶段 28 进程逐字节一致，并且与阶段 27 相同：

```text
profile
e25f29f5026f0f6d38d137a4ae431d0b95e71db5a30598c9a9b747892f32000d

dEdX
f17236ef4668e40d570f27d593daf41008f6b5f54ab5d6d9b1be878528b280e0

ground particles
8e9dcf20e4b3637f97c0abd9595724b4b3b26f56dd6ab17fee23add45824ba3f
```

### 6.2 1 TeV

固定 seed 25002，`emthin=1e-4`，`max-weight=100`：

| 指标 | 阶段 27 | 阶段 28 |
|---|---:|---:|
| photon wavefronts | 35 | 35 |
| lepton wavefronts | 442 | 442 |
| GPU particles | 576,015 | 576,015 |
| D2H | 182,104 B | 181,824 B |
| 减少量 |  | \(35\times8=280\) B |
| host spill | 0 | 0 |

profile、dEdX、ground particles、interaction 和 production profile
五类文件全部与阶段 27 逐字节一致。

### 6.3 1 PeV 与 GPU CoREAS/ZHS

固定 seed 24027，`emthin=1e-4`，`max-weight=100`，8 个 CoREAS 和
8 个 ZHS observer：

| 指标 | 阶段 27 | 阶段 28 |
|---|---:|---:|
| photon wavefronts | 107 | 107 |
| lepton wavefronts | 1,194 | 1,194 |
| GPU particles | 17,645,186 | 17,645,186 |
| pending photon peak | 152,035 | 152,035 |
| pending lepton peak | 245,790 | 245,790 |
| D2H | 1,422,384 B | 1,421,528 B |
| 减少量 |  | \(107\times8=856\) B |
| host spill | 0 | 0 |
| core runtime | 26.326 s | 26.238 s |
| router runtime | 11.690 s | 11.669 s |

单次墙钟差异小于运行噪声，不能据此宣称统计显著的加速；可以严格确认的
改进是少了 107 个同步点。以下七类输出与阶段 27 全部逐字节一致：

```text
profile
3db76ad9089708601dacf98464a062285858c89553dc50f36a3840d3a323a792

dEdX
37ea203f3faa372f9e6651dbe31e9cec3d3b639f21695547b46e9d7ab6be1986

ground particles
7ab4b8c11d16ec819dd17cc0c381acc956180c1ee30cd758c09f55fcb663a204

interactions
6b529acf3d26f1349f298da3f9c5b7e8d50e7aa452939698212bf969d435e06d

production profile
14f0fc114c4671c25d80cf58607115b85f3b83803dd21078f78d1255de95a54a

CoREAS
6b2ad3f1b5dbbb000e1e3fcb5743f244a70f841abe7479bcf30f122382b68cee

ZHS
047973c559a6d642eb36ef19e734699e0f5a910045f066d6721bef46a7539692
```

radio/profile fixed-point overflow、profile invalid record、CUDA queue overflow
和最终 pending queue 均为 0。

## 7. 下一性能瓶颈

本阶段只清除了一个重复同步，没有声称已经完成整个 device-side
scheduler。当前更大的串行开销仍来自：

1. interaction selection、transport、vertex/final-state compaction 各自下载
   count，host 根据这些 count 决定下一 kernel；
2. 每个 resident wavefront 的 `measureCudaPipeline()` 最终等待默认流；
3. cross-species queue 仍是线性 head/tail，而不是可环回的 bounded ring；
4. 小前沿仍会回到 CPU expansion，以形成足够大的 GPU batch；
5. 需要多 seed、\(10^{17}\)–\(10^{18}\) eV ensemble 才能完成最终物理与
   端到端性能验收。

下一阶段应优先把 pipeline stage count 放进一个 device control block，
用固定上界 launch 或 CUDA Graph conditional/control flow 推进多层
wavefront，减少每层 host 决策；随后再用环形设备队列消除线性尾部碎片。
