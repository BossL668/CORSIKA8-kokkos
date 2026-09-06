# Beta5：单核 CPU / Kokkos-CUDA / Kokkos-OpenMP 2000 例比较

## 当前状态（2026-09-05）

已完成参考样本配置和 2000 个独立种子的核验；CUDA 与 OpenMP 各 3 个
完整 shower 预检通过（本地 OpenMP 使用 16 线程）。按最新安排，正式
OpenMP 组转到 dirac，重新跑满 2000 例，每 shower 使用 128 线程。
本地只继续 CUDA 队列，**尚未完成大样本物理验收**。
不修改任何 EM、强子、射电算法或生产二进制，只新增任务编排及分析脚本。

13:31 核对：dirac 独立 Release 构建与安装完成，4 项 Beta5 启动/边界
测试全部通过，任务编排器 5 项测试通过。百万元素并行原语探针报告
`concurrency=128`、`gpu=false`，scan、稳定队列和往返检查通过。
已检查 FLUKA/SIBYLL 链接，OpenMP 二进制未链接 GPU runtime。
新 service 已进入 `waiting_for_previous_beta4_campaign`，**还未启动正式
OpenMP shower**；原语探针通过不等于 shower 物理验收。

## 1. 参考样本与固定配置

采用已有 1 TeV 倾斜质子样本，避免重跑 2000 个标量 CPU shower。
CPU 来源是历史 **beta4 P2.1 的标量 PROPOSAL + CPU CoREAS/ZHS**，不是
beta5 OpenMP，也不是把历史 CUDA 数据作为 CPU 数据。

| 项目 | 三组共同配置 |
|---|---|
| 初级粒子与能量 | proton / PDG 2212，1000 GeV |
| 方向 | θ=47°，φ=180° |
| 地磁场 | IGRF14，2027 年 |
| EM cut / thinning 输入 | 0.0005 GeV / 1e-6 |
| hadron、muon、tau cut | 各 0.3 GeV |
| max-weight | 不传参数，沿用 CPU 的自动默认值 |
| hadronic models | SIBYLL-2.3d + FLUKA |
| radio | CoREAS + ZHS，1 GHz，400 ns 窗，10 ns pretrigger |
| 天线 | 与 CPU 输入内容 SHA-256 一致，81 个 observer |
| shower core / ring | (0,0) / 0 |
| 种子 | 使用 CPU 原始命令中的 2000 个唯一 seed，范围 2026230001–2026232000 |

CPU 的每个输出都是 `-N 1`。因此两种 beta5 后端也逐次 `-N 1`，显式
重设同一个 seed；不能用一个 `-N 2000` 冒充 2000 次独立重设随机流。
索引使用历史分析中 `proposal.sources` 的顺序，**不按目录字符串或种子
重新猜测**。不同调度及浮点求值仍可能让同 seed 的树分叉，本轮不是
逐位相同 shower 的声明。

特别限制：这一能量下自动最大权重低于 1，原 CPU 配置实际上不会启用
thinning。保持配置不变是为了可比性；本轮不作为有效 thinning 分支验收。

## 2. 本地参考数据的真实可用性

D 盘保留全部 2000 个 CPU 的 seed、argv、primary、timing、来源哈希，
也保留逐 shower 标量、逐 shower/半径射电特征及 profile 均值/标准误。
**CPU 原始 Parquet 已迁往服务器，本次核对本地存在数为 0。**
2026-09-05 尝试 PSR SSH 返回 `No route to host`，未启动或修改远端任务。

因此可以先做配对标量均值检验、绘制 profile 均值/误差阴影及射电特征图；
但原 CPU 的完整 profile 协方差、全局分布 bootstrap、重新提取射电波形
仍需服务器原始数据。程序不会把缓存均值当作 2000 条原始 profile，也不会
把局部诊断报告标记成“全部验收通过”。

历史参考报告文件夹名：
`final_beta4p2p1cpu2000_cuda2000_proton_1TeV_theta47_phi180_emthin1e-6_igrf14_2027_v1`。

## 3. 运行策略和保护

- 两个 beta5 安装共享同一源码，使用各自已安装的二进制。
- GPU：Kokkos-CUDA，EM + radio，单线程 host，70% 空闲显存预算、batch 4096。
- CPU：dirac 的 Kokkos-OpenMP，**每个 shower 128 threads**，EM + radio；不初始化 GPU。
- dirac 的 CPU 0–127 对应 128 个不同物理核，用 `taskset` 和 OpenMP 亲和性
  约束；不是 128 个并发单核 shower。一个事件结束后立即执行下一个。
- 本地 CUDA 与远端 OpenMP 可以同时推进各自队列；每台机器一次一个 shower，
  不在同一 shower 混合多核 CPU/GPU 输运。
- 原本地 16 线程的 3 例只保留为预检，不混入 dirac 正式组。CUDA 的 3 例和
  dirac 新做的 3 例预检分别计入对应的 2000 例。每事件进程退出后释放内存。
- 本地旧 service 已停止，新 service 为 `c8-beta5-paired-cuda2000.service`，
  显式使用 `--backends cuda`，不会在 CUDA 结束后启动本地 OpenMP。
- dirac 使用 `c8-beta5-openmp128-build-queue.service`：独立 Release 构建和安装，
  保留 FLUKA、SIBYLL，检查不链接 GPU runtime；等已有 beta4 100 TeV 队列结束
  后，再预检并启动新队列。不会擅自停止旧任务，也不与旧队列争抢 128 核。
- 本地 cgroup `MemoryMax=5G`、`MemoryHigh=4G`、`MemorySwapMax=0`、`TasksMax=128`。
  dirac 构建/排队 service 使用 32 GiB 上限、24 GiB high、1024 tasks，单事件
  仍受下面的 4300 MiB RSS 和 20 min 诊断门禁约束。
- 启动前可用主机 RAM 至少 3 GiB，运行中低于 1.5 GiB 或进程 RSS 超过
  4300 MiB 停止该事件；D 盘剩余不足 15 GiB 不再启动新事件。
- 每事件 20 min 诊断时限；单个失败保留 seed 和日志，不用新 seed 替换；
  连续 3 个失败停止队列待诊断。预检任一失败不放行批量。
- 信号中断时终止当前子进程；续跑跳过已验证完成项，中断输出保留并改名，
  不覆盖。进程失败项须显式诊断后修复，不静默重跑成成功样本。
- 每事件核验 argv、primary、观测者、输出完成标志、表哈希及所有 Parquet
  可解码性/有限值，并记录进程 RSS、线程数、CPU 核时；GPU 另记录显存/利用率。
- 已安装二进制 SHA-256 改变时拒绝混合版本。生产运行期间不要重建覆盖安装。

当前 user service 可抵抗终端断开，**不能保留 WSL 关机后的进程状态**。
WSL 重启后用同一 `run_paired_acceptance.py run --output ...` 在新的受限
systemd service 中续跑；完成目录和种子清单无需重建。

## 4. 预检结果

CUDA/OpenMP 各 3 例均成功，表 SHA-256 和辅助表 SHA-256 跨两后端一致。
零 queue overflow、零 native inverse failure。预检服务主机内存峰值约
970.4 MiB，swap 峰值 0。OpenMP 记录到 16 线程，活跃阶段每秒约累计
15.8 CPU 秒，说明确实多核计算，不只是声明线程数。

| 计时口径（仅 3 例中位数，不代表大样本性能） | CUDA | OpenMP 16 |
|---|---:|---:|
| 应用 `simulation_timing` shower 区间 | 2.334 s | 3.699 s |
| 每次独立进程，含初始化（约 1 s 轮询精度） | 13.293 s | 14.114 s |

独立种子进程的初始化开销明显，不拿此端到端用时与历史多事件批次的纯
shower 时间直接相除。历史 CPU 是并发单核生产数据，和新数据的时间比值
只作描述性周转对比，不标注为隔离单核的受控硬件加速比。

## 5. 分析约定

`tools/analyze_paired_acceptance.py` 用于三组比较。现在分两台机器生产，单后端
runner 不自动把本地 3 例 OpenMP 预检当作正式组；需收齐 dirac 的正式结果后
再启动三组分析。分析规则：

1. 复用已有标量提取器，按 CPU 归档来源顺序映射 seed，核验新输出哈希。
2. 全 shower 配对 bootstrap（2500 次）计算关键均值差异的 95% 区间。
   配对均值检验对每后端 9 项作 Holm 校正，每组 alpha=0.025。
3. 同时报告“均值点估计在 1% 内”和“整个区间在 1% 内”，不把不显著等同于等价。
4. 绘制全部 EM、非 EM 与 muon parent profile，阴影为系综均值点态 95%
   正态近似区间；缓存 CPU profile 不用于假造配对协方差或全局置信带。
5. 射电使用 `pulse_analysis_modular` 的投影和脉冲分析，按完整后端样本
   进行原有宽度筛选，横轴为 `r_perp`，另存有效 shower 覆盖率。图为诊断，
   完整射电等价检验仍需后续验收，不在生成图后自动宣布通过。
6. 生产数量不满 2000、缺少 CPU 原始数据或未做完整分布验收时，都显式报告。

正式前以各 3 例完成一次完整分析流程测试（含射电），只验证分析可执行，
不使用 3 例做物理判断。

## 6. 文件位置与查看

本地 CUDA 和参考统计保存在 Windows D 盘：

```text
D:\CorsikaData\corsika_validation_results\
  beta5_scalar2000_openmp2000_cuda2000_proton_1TeV_theta47_phi180_emthin1e-6_igrf14_2027_v1\
    campaign.json             # 固定 2000 个 CPU seed 和输入来源
    physics_identity.json     # 两后端共有原生/辅助表身份
    active.json               # 最近启动事件，不是完成计数
    cuda/event_XXXX/           # result.json / log / telemetry / shower
    openmp/event_0000..0002/  # 仅本地 16 线程预检，不纳入远端正式组
    reference/                # 本地 CPU 小型统计缓存
    diagnostic_first_3/       # 仅分析流程预检
    local_comparison/         # 收齐两地正式样本后生成
    production_status.json    # 最终生成数量；不等于物理验收结果
```

```bash
systemctl --user status c8-beta5-paired-cuda2000
journalctl --user -u c8-beta5-paired-cuda2000 -n 20 --no-pager
```

dirac 正式输出位于：

```text
~/21CMA/corsika_validation_results/
  beta5_openmp128_2000_proton_1TeV_theta47_phi180_emthin1e-6_paired2026230001_dirac_v1/
    campaign.json          # 原 CPU seed 列表不变，仅改部署路径/线程数/二进制身份
    deployment_status.json # 等旧队列/预检/正式运行状态
    physics_identity.json  # 必须与本地 native/auxiliary 表哈希一致
    openmp/event_XXXX/     # 正式 128 线程样本及 CPU 核时/线程数监测
```

dirac 无法访问 Conan/GitHub 外网时，部署使用 Conan 官方 lockfile 机制锁定
已有 Kokkos 4.7.03 OpenMP 包，PROPOSAL 7.6.2 和 CubicInterpolation 0.1.5
也复用已存在的补丁包。所用 Kokkos recipe 修订与 beta5 最新修订仅相差
SYCL 链接参数，OpenMP 源码与选项一致；依赖图锁文件留在 `build/audit/`。
不手工修改 Conan 缓存、不替换已运行的 beta4 二进制、不更改 beta5 物理代码。

本轮不推送 GitHub、不删除历史数据、不修改 IDE 内存设置。

## 7. 2026-09-05 本地重启后的内存保护续跑

17:19 从已完成的 280 例续跑，第 281 例 seed 为 2026231164。再次核验
CUDA 可执行文件、FLUKA、天线哈希均未改变；不修改物理内核或随机流。
之前这轮生产本来也是每事件 `-N 1`，因此不能仅凭 WSL 重启就认定是
同一个 shower 进程跨事件泄漏。新的监控同时覆盖长寿命 Python 队列、
每事件子进程，以及结果校验和事件切换阶段。

- 新增独立 `tools/run_memory_guarded.py`，每 0.25 秒读取 `/proc`，无需
  等待 shower 主线程、GPU 查询或 Parquet 校验返回。
- WSL `MemAvailable` 低于 3072 MiB、任何子进程 RSS 超过 3072 MiB、
  队列 RSS 超过 1024 MiB、整棵进程树 RSS 保守求和超过 4096 MiB，任一
  条件触发即终止本地队列及其子进程。只操作自己启动的进程树，不杀 IDE
  或其他任务；用进程启动时间检查 PID 是否被复用。
- RSS 求和可能重复计入共享页，因此是保守门禁，不等于物理 RAM 净占用。
  文件缓存是否可回收由 `MemAvailable` 反映，不把 `buff/cache` 增长直接
  判作程序泄漏；GPU 显存也不等同于主机 RAM。
- systemd 保留 `MemoryHigh=4G`、`MemoryMax=5G`、`MemorySwapMax=0` 和
  `OOMPolicy=stop`，对快于轮询速度的增长提供内核级兜底。用户态停止有
  最多一个采样间隔及信号处理延迟，不是零延迟硬实时保证。
- 修正旧队列将内存异常视为普通事件失败、随后继续下一事件的行为：
  `MemorySafetyStop` 直接中止整条队列。内存事故后留下锁定标记，必须
  诊断并归档标记后才能人工续跑，不会自动重启或换 seed 掩盖问题。
- 监控器的真实子进程测试只分配 64 MiB，故意使用 32 MiB 门限，验证
  独立 session 子进程也被终止且再次启动被拒绝；另有长寿命父进程逐次
  保留内存、期间没有模拟子进程的测试，验证跨事件监控有效。监控测试 4/4、队列测试
  6/6 通过。这些是安全测试，不替代 shower 物理或长期内存验收。

service 已保存为用户配置中的持久 unit，但**没有启用开机自启动**：
重启后可明确执行 `systemctl --user start c8-beta5-paired-cuda2000`，
不会因终端断开退出，也不会在内存故障后自动重启。

本地 campaign 目录新增：

```text
guarded_production.log      # 持续落盘的队列输出，WSL 重启后保留
memory_guard.csv            # 主机可用内存、队列/子进程 RSS、子进程 PID
memory_guard_status.json    # 最近采样、峰值、最低可用内存、保护状态
MEMORY_GUARD_STOP.json      # 仅监控越限时出现，禁止未经诊断重新启动
runner_memory_stop.json    # 仅队列内部内存门禁触发时出现
```

门禁只能先停止并保留证据，不能自动推断泄漏根因或自动修改物理程序。
若后续触发，需区分模拟子进程、队列/分析器、系统缓存及 IDE 占用，再
进行有针对性的修复和同种子复测。此次未同步修改正在运行的 dirac 队列。

17:23 观察窗口：续跑后新增 14 个完整事件，总计 294/2000。模拟子进程
RSS 峰值约 764.6 MiB，队列进程约 130 MiB，最后约 30 秒在
129.6–130.5 MiB 间波动；整棵任务树随子进程启动/退出在约 130–900 MiB
间变化。systemd 服务峰值约 0.97 GiB，Swap 为 0，未触发内存门禁。
这是短期多事件观察，不是对历史重启原因或所有能区无泄漏的证明。
