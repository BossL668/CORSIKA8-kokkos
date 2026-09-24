# 单 shower 多 GPU：实验实现（2026-09-24）

本功能仍处于验证阶段，**不能称为已通过 beta2 官方 C8 统计验收，也不承诺四倍加速**。
它不改变已安装的 `c8_air_shower`，不修改山体、CUDA/OpenMP kernel、稀有末态回退、
物理公式或生产默认参数。

## 已实现的边界

```text
原始 primary、全局物理参数
          ↓
标量前缀：展开强子级联及过大的 EM 分支，记录前缀的 profile / 射电
          ↓ 安全的下一步开始前，导出互不重叠的 EM / μ 根粒子
一次性按未加权能量代理分配（不改变粒子权重或物理 cut）
          ├─ worker 1：独立 GPU、CPU 栈、随机流、物理模型、回退
          ├─ worker 2：独立 GPU、CPU 栈、随机流、物理模型、回退
          ├─ worker 3：独立 GPU、CPU 栈、随机流、物理模型、回退
          └─ worker 4：独立 GPU、CPU 栈、随机流、物理模型、回退
          ↓ 全部正常退出、Parquet 正常关闭后
前缀 + 四个子簇：profile / 沉积 / 有符号 CoREAS、ZHS 场 / 地面粒子
```

- root 使用原标量步骤器，worker 直接调用现有 Kokkos 单端 runner；不另写输运。
- 导出发生在父过程的二次粒子、cut、thinning 已处理完毕之后，下一次物理步骤之前。
- 接收端重新确定几何节点，不序列化指针；保留位置、方向、时间、总能量、权重及祖先身份。
- 接收端不重新注入 primary，不重新强制相互作用，不重新薄化这些根粒子。
- 全局初级能量、薄化阈值/max-weight、轴、天线和时间网格不以子簇能量重算。
- 每个 worker 的新 history 使用独立的 48-bit 区间，根粒子的 history 不变。
- 导出调度器返回的待执行步骤号（从 0 开始），不导出它已自增的存储值，避免凭空多计一步。
- 稀有末态仍在产生它的 worker 内由已有 CPU 路径处理；不会阻塞别的 worker。
- 没有跨卡迁移、逐波前同步或“等最慢卡才能开始下一步”；只有最终完成汇总。
- 每个进程有独立工作目录，避免 Fortran 模型临时文件相互覆盖。

## 首版尚未完成的优化与限制

1. 这是一版**静态 EM 前沿**实现：强子骨架首先在前缀中展开，不是任意强子/核子全栈
   检查点。因此存在串行前缀，性能必须分开计时，不能隐去。
2. 当前每个 shower 启动 worker；还未实现跨 shower 常驻 worker。表缓存可复用。
3. 跨 worker 汇总当前为已关闭输出数组的补偿 `float64` 求和。设备内部仍保持原定点
   累积，但**尚未实现跨进程先合并原始定点整数再转换**。profile 的原始 Parquet 列
   是 float32，不能宣称此次合并是整数级无舍入误差。保留各分片便于复核。
4. 每个 worker 所报 dEdX/ground 是部分 shower 的账本；不能拿它与完整 primary
   能量相除后判断不守恒。验收必须加上前缀与全部 worker，仍需考虑原账本未覆盖的项。
5. 当前只支持固定能量、N=1、未强制 primary 的协调运行；多个事件由外层任务循环。
6. 调度随机流允许改变。不能把一个 seed 的不同 shower 树当成物理错误；同样也
   不能凭单例看起来相似宣布统计一致。

## 构建和运行

在已配置的 beta5 Kokkos 构建中启用独立目标，不覆盖原安装：

```bash
cmake -S corsika8_kokkos_beta5 -B build/cuda-openmp \
  -DCORSIKA_BUILD_MULTIGPU_APPLICATION=ON
cmake --build build/cuda-openmp --target c8_air_shower_multigpu -j2
python corsika8_kokkos_beta5/applications/detail/air_shower_multigpu/run_multigpu.py config.json
```

`config.json` 的 `command` 是独立二进制及必要的 loader 前缀；`physics_args` 是原空气
程序参数；协调器管理 `-N/-f/-s`、后端、设备号和显存比例，拒绝这些参数被重复传入。
`g01_smoke.json` 是服务器四卡功能检查示例，不是正式统计验收配置。
射电测试必须显式加 `--ring 1` 使用程序内置的 25 m 八天线环，不读 21CMA 天线表。
普通程序的 `--ring` 默认值为 0；缺少天线文件时会产生空射电文件，不能算射电验收通过。
协调器现在拒绝空 profile 或空 CoREAS/ZHS 输出。

`InteractionWriter` 只保存首次相互作用；最终结果只取前缀的全局首次反应，
不会拼接成四个虚假的“初级反应”。地面粒子记录则正常合并所有分片。

服务器工程目录：`/home/member/yuhanglu/workspace/corsika-beta5-multigpu-20260924/`。
服务器数据目录：`/data/yuhanglu/CorsikaData/beta5_multigpu_20260924/`。

## 安全与发布门禁

- `COMPLETE.json` 只表示进程及输出完整，另有 `physics_validated_against_beta2=false`。
- 任一 worker 出错不得合并为完整 shower；保留所有分片及失败原因，不静默重跑/换端。
- SHA-256 记录前沿与分片，根粒子只能被分配一次；网格/天线/单位不一致即拒绝合并。
- g01 测试暂停的只是明确识别的四个生产树；SIGSTOP 保留进程和输出。独立 watchdog
  到期恢复 SIGCONT。保留的生产显存不释放，测试上限为**剩余空闲显存的 50%**。
- beta2 参考固定为官方 `icrc2025-beta2` 源码提交
  `74ebbc28051de3600aab133c225e387801b035a5`。不能把 beta4 CPU 旧样本或当前 beta5
  标量控制实验冒充该参考。
- 官方 beta2 示例的空气磁场是固定 NWU `(50,0,0) μT`；当前 beta5 示例是 21CMA IGRF14。
  正式验收前需显式对齐这些条件，不能直接拿两个默认命令比较。
- 后续须完成 beta2 等条件独立样本的全组分 profile、标量分布、能量账本、Ex′/Ey′/Ez′
  CoREAS/ZHS 波形与耗时验收，以及 N>1 的资源回收和原单端回归。未拒绝差异不等于等价。

## 已执行检查与部署排错

- C++ 实际粒子栈导出/导入测试通过，检查 history、parent、weight 和零基 step；
  Python 协调器/合并/失败保护测试 8/8 通过。
- 去除实验宏区块后，原 `c8_air_shower.cpp` 的非注释 token 与 HEAD 一致；
  原安装二进制 SHA-256 仍为
  `4badd1be7058649a1c0dc6f2f25044a755d505d299707fb072ceb5d8a57e999f`。
- g01 单 worker 的无射电数据功能检查已正常结束（10 GeV gamma，173 roots）。
  其 CoREAS/ZHS 行数为 0，只能算输运/文件检查，不能算射电验收。
- 四 worker 初始测试在 Molière 表初始化阶段 SIGSEGV。调试映射与反汇编定位到
  `KokkosMoliereInterpolation<Cuda>::initialize` 的约 0.53 MiB 主机栈帧；
  `$rsp=0x7fffffdb8230` 已低于栈映射起点 `0x7fffffe34000`，下方又邻近 brk heap。
  这不是显存不足。该服务器通过兼容版 ld.so 启动新二进制；给实验子进程设置
  `MALLOC_MMAP_THRESHOLD_=65536` 后四卡正常完成，不修改生产内核或生产环境变量。
- 失败输出全部保留；每轮暂停的四个生产树均由 lease 恢复。调试器正常返回本身不够，
  还必须核对 worker 退出与 Parquet footer，损坏文件不能生成 COMPLETE。

### 四卡首个完整功能结果

数据：`/data/yuhanglu/CorsikaData/beta5_multigpu_20260924/photon10GeV_four_gpu_radio_v5/`（g01）。

| 项目 | 实测 |
|---|---:|
| 初级、能量、薄化 | gamma，10 GeV，关闭薄化 |
| 四卡根粒子数 | 43 / 43 / 43 / 44，共 173 |
| 前缀耗时 | 5.624 s |
| worker 各自进程耗时（含初始化和输出） | 7.257 / 7.250 / 7.849 / 7.226 s |
| 总墙钟（含前缀、worker、合并） | 14.127 s |
| 各卡射电轨迹数 | 3222 / 2000 / 2402 / 1436 |
| CoREAS、ZHS 输出行数 | 各 3200（内置八天线） |
| 最大绝对场 CoREAS / ZHS | 3.201e-11 / 3.158e-11 V/m |
| worker 正常退出 | 4/4 |
| 生产恢复 | lease_05.released；四卡原进程各 22891 MiB，采样 GPU 利用率 85–88% |

四个 worker 在协调时间 5.78–5.83 s 内全部启动，各自在 13.04–13.66 s 结束。
这证明进程并发与完整输出，不是 GPU 内核重叠比例或加速比的精密计时。
小样本以初始化为主，不能外推高能四卡性能。单 worker 的局部能量预算分母仍为全局
primary，日志中约 -77% 的“relative difference”不是整个 shower 的亏损；必须加前缀和全部分片。

### 含强子骨架的第二个完整用例

数据：`/data/yuhanglu/CorsikaData/beta5_multigpu_20260924/proton1TeV_four_gpu_radio_v7/`（g01）。

- 1 TeV proton，关闭薄化；331 个 EM/μ 根粒子按 71 / 86 / 88 / 86 分配。
- 总墙钟 14.436 s，四 worker 全部正常退出；地面输出 123 行，原始首次反应次级 40 行。
- 四卡合计 photon steps=73,220，lepton steps=516,993，radio tracks=484,194。
- 各 worker 的指定 CPU 末态为 13 / 14 / 10 / 11；CPU 强制衰变完成计数为
  8/8、13/13、11/11、7/7。这里的“强制”指后端已选定的衰变交给 CPU 执行，
  不是对全局 primary 强制相互作用。
- CoREAS/ZHS 各 3200 行，最大绝对场分别 2.497e-10 / 2.124e-10 V/m。
- `production_lease_07.released` 已生成，原四卡生产恢复；现场 GPU 利用率 85–98%。
- 该例验证了实际回退路径有执行，不等同于“人为延迟一个 worker 后其他卡仍持续运行”
  的隔离压力测试，后者仍待完成。

目前通过：独立目标构建、交接与合并单测、两个四卡功能用例、失败恢复保护。
仍待通过：官方 beta2 等条件统计、同轨迹分片射电精度、边界/异常压力、连续事件与性能门禁。
官方参考和本应用的默认磁场不同，必须先对齐，不得用这两个默认参数样例冒充正式对照。

以上不是正式 beta2 统计结论。beta2 大样本、同轨迹射电分片和稀有回退隔离门禁通过前，不替换生产程序。

## 100 PeV / 100 例新队列（2026-09-24）

按用户后续要求，独立协调器允许显式 `gpu_memory_fraction=0.9`，默认仍是 0.5。
它是后端按空闲显存估算的容量预算，不保证所有时刻实际占用或 GPU 利用率达到 90%。
旧的 SIGSTOP lease 不能释放显存，因此新队列使用批次边界让出设备；
未经确认不终止未闭合的旧批次。

- 每个 shower 同时分到四张 L20，而不是四卡各跑独立初级。
- 原 100 例 CPU 参考为 PSR 的 `beta2_fixed_proton_100PeV_...cpu100...`，
  它是采用 IGRF14/2027 的 beta2 标量应用，不是未改动的官方默认磁场示例。
- 固定种子 2026110001–2026110100；SIBYLL-2.3d＋FLUKA、theta47/phi180、
  emthin1e-6、自动 max-weight50、81天线、1GHz/400ns/10ns预触发不变。
- 使用当前修复后物理、原 C8 profile 端点规则；旧有缺陷的 Kokkos 数据不混入。
  当前修复与旧 CPU 的处理差异须在最终报告中列出，不预先宣布统计一致。
- `run_campaign.py` 冻结配置和文件哈希，逐例检查正常退出、Parquet有限值、
  队列及定点溢出，出错停队列并保留现场；正常完成可跳过已校验 seed。
- 每5秒记录四卡显存/利用率/功率/频率、主机可用内存、RSS、CPU ticks和阶段。
  保留至少64 GiB主机可用内存、100 GiB磁盘空间。
- CPU旧机器/四卡的时间比只作跨硬件描述。严格四卡扩展效率仍需同机单L20控制实验。

队列未完成前不属于100例物理验收通过。
# 2026-09-24：有界前沿输入优化

实验多卡 worker 默认使用 `BufferedFrontierRouter`，仍调用原单卡物理/射电后端。
未开始的根粒子保留在前沿文件中，仅在本 worker 的设备队列、标量栈和回退均排空后，
按 `--gpu-min-batch`（本轮4096，上限65536）读取下一批。
按块反向压栈，使实际 LIFO 消费保持文件的高能优先顺序。
这避免全量百万级主机栈在每次 GPU 回退后被 `setNodes()` 重复定位；
没有跳过任何必要的介质定位，没有更改 cut、薄化、随机过程或射电公式。

`FRONTIER_FEED.json` 记录消费数、批次数和最大导入批量，协调器必须核对消费数等于分配数。
未消费完外部前沿不得关闭 shower，跨批 history 重复会终止该例。
`--frontier-legacy-import` 仅用于独立诊断对照；正常空气程序不暴露这些实验选项。
在不定义实验宏时，空气入口和公共运行器的非注释代码 token 与原版相同。

此优化仍未消除串行前缀，也没有实现 GPU 工作期间提前填充下一批或跨卡迁移。
完整100例的统计验收及端到端加速结论须待数据完成后给出。
