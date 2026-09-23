# 单 OpenMP：profile／统计累积分片

日期：2026-09-19。22:25 状态：PSR、dirac 均通过完整门禁并接入当前生产。

## 修改的范围

单 OpenMP 轻子波前原来由全部线程对同一组整数计数器和 profile bin 执行
checked atomic。现将已有、已用于自适应双端的主机分片累积器接入单 OpenMP。
不是重写物理公式，也不是把一场 shower 拆成若干独立随机模拟。

```text
稳定粒子队列 → 原有输运／散射／末态 → 相同的逐步 profile 投影
                                       ↓
                           OpenMP worker → 有界整数分片
                                       ↓
                  shower 结束：canonical＋分片 checked 整数求和
                                       ↓
                         一次浮点转换 → 原 profile／YAML 输出
```

- 保留 `accumulateLeptonProfileStep` 的公式、定点尺度、原子溢出检查。
- 保留 reducer 的 `init/join/final`；Molière 迭代数与最大迭代数仍按波前
  归约一次，不能在分片重复累计。
- 光子和末态写入的 canonical ledger 仍计入最后合并，不丢弃、不重复。
- CUDA 路径不启用分片；双端保持原来的显式启用条件。
- 不修改 PROPOSAL 表、RNG、粒子顺序、薄化、cut、射电算法、CLI 或生产参数。
- 普通 `downloadProfile()` 现在可走 checked 整数快照合并。
  重复导出、溢出、非法记录必须报错；不得发布部分合并结果。

## 内存和生命周期

分片由线程数、profile bin 数、剩余后端内存预算共同决定。独立上限仍是
**16 MiB、256 片**；不是每线程复制一份粒子栈、物理表或射电缓存。
大 profile 可能无法做到每线程一片，此时少量线程共享分片，继续使用原有
checked atomic，仍然安全。预算不足 2 片时保留 canonical 路径。

`beginShower()` 清零整数 bin／计数器并更新定点尺度，复用分配。
主机分片字节计入工作区预算，不伪记为设备到主机传输量。

## 代码入口

- `src/accelerator/em/kokkos/KokkosBackendInstance.inl`：仅单 OpenMP 在初始化末尾启用。
- `corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp`：普通导出接入 checked 整数合并。
- `corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp`：复用已有主机专用 reducer。
- `corsika/accelerator/em/detail/HostProfileShards.hpp`：已有容量上限、清零及整数合并实现，本轮不改公式。
- `tests/accelerator/testKokkosHostProfileShards.cpp`：补齐普通导出、重复导出拒绝、32 次复用和溢出门禁。

## 验收方法

1. 两台服务器独立 Release 构建，不覆盖生产二进制／模型库。
2. 实际 Kokkos OpenMP 4／128／130 线程测试，比较原 canonical 与分片的
   每个整数 bin、所有 profile 计数器、归约统计；另比较普通浮点导出。
3. 强制 2／3 片碰撞、空波前、混合 PID、source holes、正负溢出、非法记录、
   重复提交、改变尺度后的 32 次清零复用。
4. 实际光子 `-N 32`，逐项比较所有 Parquet 数组和非计时物理计数。
   帮助文本仅规范化 `argv[0]` 的二进制路径，其他内容须一致。
5. 当前生产条件：215.4 TeV 质子、θ=60°、φ=225°、QGSJet-III＋FLUKA、
   `emthin=1e-6`、`max-weight=2.154`、110 天线、**仅 CoREAS**。
   各服务器同 CPU 集合、3 个固定种子，按 old→new／new→old 交替运行。
6. profile、production profile、地面粒子、相互作用、沉积、CoREAS 数组
   必须精确一致，native 表哈希和物理计数不得变化。只允许时间派生诊断、
   构建标识和本次新增的有界 profile 存储字节差异。

墙钟时间与 shower 时间分开记录；进程仍逐事例重启，没有偷偷改 `-N` 或薄化。
原有强子 shower 的不完整能量账本不是本项统计优化的验收目标；这里要求其
所有已有账本数值前后不变，不把原有 `complete_coverage=false` 宣称为新通过。

## 当前已完成的检查

- PSR 的 4／130 线程整数／导出／生命周期／溢出门禁通过。
- dirac 的 4／128 线程门禁通过。
- PSR 和 dirac 的实际 `N=32`：6 类 Parquet 数组和物理计数完全一致；6194 条 profile
  步进、30988 次 CoREAS contributions，非空测试。后续31场复用后端，
  profile 字节数保持89128，其中分片7424字节。
- 生产 checkpoint 的四项回归测试通过：索引完整且互斥、种子与参数不变、
  非连续索引正确快照、跨服务器合并拒绝重复／缺失／来源错误。

### PSR 130 核：三对完整生产参数结果

| seed | 原 shower / s | 分片 shower / s | 原进程总时间 / s | 分片进程总时间 / s |
|---|---:|---:|---:|---:|
| 13 | 85.743 | 65.229 | 157.804 | 137.553 |
| 25 | 80.353 | 63.070 | 152.713 | 134.738 |
| 37 | 78.357 | 59.118 | 150.646 | 130.843 |
| 中位 | **80.353** | **63.070** | **152.713** | **134.738** |

shower 时间缩短 **21.5%**（速度比1.274）；含逐事例启动的总时间缩短约
**11.8%**。三对的6类物理数组及所有非计时物理统计精确一致。
编译选项保持同一组 `-O3 -DNDEBUG -std=c++17 -fPIC -fopenmp`，未启用 fast-math。

seed13 的处理量前后同为172697704条 profile step、16697个轻子波前、
147543304条射电轨迹。轻子后端76.167s→55.699s，光子7.216s→7.208s。
本配置3820个bin，16MiB上限下实际为68片、16650752字节；共享冲突减轻，
不是承诺每线程独占一片。分片前后的传输统计都为244704字节，未把新增
主机存储量误计为传输流量。profile overflow/invalid均为0。

这些数字是本机本配置的三种子初步计时，不表示所有初级/能量都固定提速，
也不是C7/C8同物理精度的速度比较。

### dirac 128 核：三对完整生产参数结果

| seed | 原 shower / s | 分片 shower / s | 原进程总时间 / s | 分片进程总时间 / s |
|---|---:|---:|---:|---:|
| 7 | 225.814 | 194.242 | 391.983 | 360.599 |
| 19 | 170.299 | 147.308 | 336.561 | 316.119 |
| 31 | 191.564 | 155.511 | 358.385 | 321.750 |
| 中位 | **191.564** | **155.511** | **358.385** | **321.750** |

shower 时间缩短 **18.8%**（速度比1.232）；进程总时间缩短 **10.2%**。
同样三对的6类物理数组、表哈希和所有非计时物理统计精确一致。
峰值 RSS 在约2.28–2.35 GiB，未出现新增大规模内存占用。

这里分别比较各主机自身的 old/new，不能直接用 PSR／dirac 的秒数相除，
归因于分片优化。两台机器硬件和启动成本不同；完整计时样本量仍只有每机3对。

## 结果目录与生产切换

- PSR：`/data/yhlu/CorsikaData/corsika_validation_results/c8_openmp_profile_shards_acceptance_20260919/`
- dirac：`/mnt/data/yhlu/CorsikaData/c8_openmp_profile_shards_acceptance_20260919/`
- 独立二进制：各服务器 `c7-comparison-20260918[-psr]/profile-shards-20260919/build/applications/c8_corsika7_compare`。
- 驱动和对照脚本保存在工作区 `78comparison/*profile_shards*.py` / `*.sh`。

先等待当前事例完成，再暂停旧派发器进行隔离验收。dirac 旧派发器没有暂停
接口，采用明确标为 `NOT_A_SIMULATION` 的下一事件空目录守卫：旧程序在
启动下一事件前主动拒绝。该预期退出不是模拟失败，不终止正在计算的事件。

通过后使用独立版本继续同一组种子；旧二进制、旧 `campaign.json`、已完成
数据均保留。新事件 `COMPLETE.json` 写入实际新二进制 SHA-256 和版本标识。
PSR 仍选相对空闲130个物理核；dirac 保留原128核及 C7/C8 的顺序安排。

上面的正式整场计时已完成；不得把微基准倍数当作整场 shower 的加速比，
也不得将本项优化称为解决全部 C7/C8 性能差异。

### 后续数据使用

已核对 `run_tenth_openmp_split.snapshot()`、`collect_tenth_checkpoints.py` 和
`analyze_tenth_1000.py`：新版本不会因二进制哈希变化而被排除；索引、种子、
主机、线程数和输出哈希仍按原门禁检查，物理统计继续使用同一批次。
每例实际二进制哈希保存在 `COMPLETE.json`，新例还含
`implementation_revision=standalone-profile-shards-20260919`。

现有性能图只按主机／线程数分组，尚不按实现版本分组。因此混合批次的耗时
不能直接当作新版性能；本项优化的加速结论以这里的独立 old/new 配对计时为准。
后续全批次性能验收应从每例 `COMPLETE.json` 读取版本再分组，不覆盖旧耗时。

### PSR 已完成切换

- 21:57:16 激活 `c7c8-profile-shards-psr-20260919.service`。
- 旧生产完整完成至全局index75；新版本从index77、seed469继续，参数不变。
- 新冻结二进制 SHA-256：`4f8cf53027e1d27876c7ff8e67403a9a46e212b186450a14391fc66709dc1408`。
- 版本凭据在原批次 `provenance/openmp_profile_shards_20260919/`；
  旧二进制和旧campaign manifest不覆盖。新二进制仍依赖独立build下的模型库，
  campaign完成前不要删除该独立build；每次启动都会核对依赖哈希。
- 小型验收报告同步到本地D盘
  `D:\CorsikaData\corsika_validation_results\c8_openmp_profile_shards_acceptance_20260919_summary\psr\`，
  不下载原始shower数据。

### dirac 已完成切换

- 22:24:04 激活 `c7c8-profile-shards-dirac-20260919.service`，已启用持久用户服务。
- 原 C7 MPI 与 C8 OpenMP 均完整完成至各自 global index16；没有中断当前事例。
- 清除的是明确标为 `NOT_A_SIMULATION` 的空启动守卫，且以移动归档方式保留；
  没有删除任何模拟数据。C7 先从 index17、seed109 续跑，之后按原顺序调度。
- 新版 C8 首个待生产事例是 index18、seed115，128核，条件和目录不变。
- 新冻结二进制 SHA-256：`1e11be1b00c1f87171a60b97db39c908ba435c585a8838b3ea390087166a40f9`。
- 验收报告在D盘同一 summary 目录的 `dirac/`，原始模拟输出仍在服务器。
- 本地 CUDA 生产、单 CUDA／双端算法、C7 二进制及 MPI 划分均未修改。

回退时也应先完成当前事例，在事件边界选择归档旧版本；不要直接覆盖正在
使用的二进制、删除独立build模型库，或让新旧派发器同时启动同一批次。
