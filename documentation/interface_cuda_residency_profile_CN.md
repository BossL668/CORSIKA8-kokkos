# CUDA 常驻粒子队列：PSR 实际 GPU 活动复测

日期：2026-09-11。结论：**粒子队列常驻显存，输运及次级整理实际在 GPU 执行**。
每批仍由 CPU 发射 kernel、接收诊断并同步；当前执行方式没有 GPU 自主循环到
整个级联结束的 persistent kernel。

此页保留多轮后端改造前的测量。当前接口已增加多 wavefront 常驻调用和设备诊断账本，
见 [模块接口说明](interface_resident_transport_CN.md)。大气后端也使用主机侧循环和
小型控制记录同步，不能将本页“没有 persistent kernel”理解为大气已有这种 kernel。

## 测试方法

使用 `psrpku2025_PKU` 上上一轮已通过验收的 CUDA 山体二进制，固定在
CPU 0–255（256 个物理核），GPU 为 NVIDIA T400 4GB。此次本地没有编译或运行测试。

新增独立诊断插件
[InterfaceResidencyTrace.cpp](../validation/terrain/InterfaceResidencyTrace.cpp)，
通过 Kokkos Tools 动态加载。CUPTI 12.6 Activity API 记录 GPU 实际 kernel、
实际内存复制及 CUDA runtime 调用；Kokkos Tools 回调记录数组名称、地址、空间及复制字节数。
插件明确设置 `requires_global_fencing=false`，不要求额外全局同步，也不改变物理随机数。
使用 `CONCURRENT_KERNEL` 活动类型，保留原来的 kernel 并发语义。

原山体二进制不重新链接，逐次校验其 SHA-256；此前交付的 26 个文件哈希均保持一致。
最终使用四次成功跟踪：1 GeV 光子与 10 TeV 强制 CC 的 νe 事例，各含 batched/resident。
每次均与此前未加载插件的同一事例比较：完整轨迹 CSV、材料表、所有诊断及加速器计数一致。
四份 CUPTI 跟踪的错误数和丢失记录数均为零。

## 实际数据

下表上传量仅指粒子输入，不含初始化阶段的材料表、几何和 Kokkos 分配元数据。
当前 `EmParticleState` 为 112 字节；上传字节数由命名复制记录与真实 CUDA 复制共同核对。

| 事例/队列 | 推进步数 | GPU 输运 kernel 次数 | 主机上传粒子数 | 粒子上传字节数 |
|---|---:|---:|---:|---:|
| 光子，batched | 1,489 | 59 | 1,489 | 166,768 |
| 光子，resident | 1,489 | 59 | 1 | 112 |
| νe，batched | 194,841 | 3,095 | 194,841 | 21,822,192 |
| νe，resident | 194,841 | 3,095 | 22,535 | 2,523,920 |

**光子常驻事例在首个输运 kernel 开始之后，CUPTI 记录的 H2D 复制为零。**
GPU 随后执行 59 个输运 kernel、59 个次级计数 kernel、118 个扫描 kernel、
59 个 scatter kernel，并通过设备到设备复制维持队列。
扫描每批包含两个实际 CUDA kernel，故其数量是 wavefront 数的两倍。

νe 常驻事例执行 3,095 个输运 kernel、3,095 个计数 kernel、6,190 个扫描 kernel、
3,095 个 scatter kernel。2,825 次输入上传对应 22,535 个 CPU 注入粒子，
指定 CPU 回退共 27 次，与未加跟踪的结果一致。存活粒子与电磁次级由设备队列继续推进。
材料及几何在开始输运后没有再次上传。

两次常驻事例中，名为 `interface_resident_particles` 的数组都在 `Cuda` 空间
分配一次，容量为 65,536 个粒子、7,340,032 字节（7 MiB）；运行中没有扩容或重新分配。
每批输入由此数组设备到设备复制到步进工作区。没有观察到该队列下载到主机。
退出时队列分配完整释放。这里的 7 MiB 只计算粒子 FIFO，不包括物理表、几何和其他工作区。

## 当前仍有的主机工作

常驻状态避免了后继粒子的重复上传，但诊断数据仍逐批返回主机：

| 常驻事例 | 完整步记录回传批次 | 完整步记录字节数 | 含扫描计数的总 D2H 字节数 |
|---|---:|---:|---:|
| 光子 | 59 | 1,357,968 | 1,358,440 |
| νe | 3,095 | 177,694,992 | 177,719,752 |

完整步记录包含输入/输出粒子及次级状态，用于现有诊断和指定回退。
因此，“队列不下载到主机”并不表示“粒子信息完全不回传”。扫描每批还返回 8 字节计数。
实际 CUDA runtime 跟踪也记录了 `cudaStreamSynchronize`；CPU 继续分配 history 编号、
发射后续批次、流式输出，并处理指定 CPU 过程。

此轮验证粒子驻留与 GPU 执行路径。它没有证明脱离 CPU 的长期常驻 kernel，
也没有证明加速比。带跟踪的 νe 总耗时为 batched 33.975 s、resident 34.420 s，
包括物理库初始化、CPU 过程、诊断和跟踪开销，不能作为吞吐性能验收。
成功运行的最大采样 RSS 为 826.61 MiB，最大采样显存增量为 245 MiB。

## 重现及记录

PSR 工作目录：

```text
/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-interface-resident-20260911/cuda-residency-profile/
  InterfaceResidencyTrace.cpp     已编译的插件源码副本
  libInterfaceResidencyTrace.so  独立诊断插件
  profile_interface_residency.py  带跟踪的逐事例对照
  analyze_interface_residency.py  核对实际 CUDA 活动、复制量、分配与队列计数
  runs-v1/                       光子对照及保留的第一次 νe 检查
  runs-v2/                       νe 最终通过的对照
  residency_analysis.json       机器可读结论
  profiling_provenance.json      源码/插件哈希、资源记录
```

插件运行环境沿用上一轮 PSR CUDA 环境，并在 `LD_LIBRARY_PATH` 中添加本目录的
`cupti/lib`。通过下面的脚本指定**新的**输出目录即可复测，输入命令与原二进制哈希
从上一轮验收清单读取：

```bash
taskset -c 0-255 python profile_interface_residency.py \
  --acceptance ../acceptance-cuda/acceptance.json \
  --plugin ./libInterfaceResidencyTrace.so \
  --root ./runs-new \
  --guard ../source/validation/terrain/run_guarded_diagnostic.py

python analyze_interface_residency.py \
  --profiles ./runs-new/profile_acceptance.json \
  --output ./analysis-new.json
```

首次 νe 对照被诊断比较器拒绝：YAML 中的 PDG 字典键是整数，而存档 JSON 中为字符串。
完整 CSV 和数值当时已经一致。比较器改为采用存档相同的 JSON 键表示后重跑通过；
未删除字段、放宽数值精度或改动物理。首次记录保留，未列入最终四次通过结果。

完整 GPU 跟踪、日志和小型结果归档取回本地项目的
`build/interface-resident-validation-20260911/cuda-residency-profile/`；完整物理 CSV 留在 PSR，
其 SHA-256 保存在对应 `profile_acceptance.json` 中。
