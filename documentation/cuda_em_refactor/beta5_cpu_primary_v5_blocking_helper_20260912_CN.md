# CPU 优先 v5：限定辅助端的阻塞等待候选

实验范围：仅`--kokkos-execution openmp-cuda`。目标不是牺牲OpenMP吞吐换取
更高GPU占用率，而是主端工作不受破坏、辅助端贡献超过额外成本。本候选尚未
取得该性能结论，不作为生产推荐。

## 修改原因和边界

[v4记录](beta5_cpu_primary_v4_resident_helper_20260912_CN.md)中PSR首个有效
配对仍慢13.1%。辅助driver和130个OpenMP线程共享允许的130核，driver在CUDA
等待期间是否与主端争抢CPU，需要固定工作量实测，不能只看整体CPU百分比。

本轮不修改v4取数顺序、arena、GPU自主续跑次数、物理kernel、随机流、history
租约、PROPOSAL表或射电公式。只允许CPU优先辅助CUDA实例将原有控制/输出等待
替换为可复用的`cudaEventBlockingSync | cudaEventDisableTiming`事件：记录本端
stream，`cudaEventSynchronize`等待；不调用全局`cudaSetDeviceFlags`，不fence
另一个执行端。系统仍可在GPU工作时调度OpenMP线程。

单CUDA、单OpenMP、原GPU优先双端和旧adaptive模式保持原等待方式；无新增默认
计时开销。新事件第一次等待时在实际driver线程创建，跨shower复用；每shower
仅清零等待计数。配置变更要求空闲，progress回调与blocking模式互斥。异常只
清理该stream并上报失败；GPU硬件挂起仍依赖外部watchdog，事件本身无超时。

## 代码位置

- `corsika/accelerator/em/kokkos/CudaCompletionTicket.hpp`：可选阻塞event和消费。
- `corsika/accelerator/em/kokkos/ResidentExecutionWait.hpp`：实例级等待、异常
  drain、计数与复用；默认fence保留。
- `corsika/accelerator/em/detail/KokkosBackendInstance.hpp`和
  `src/accelerator/em/kokkos/KokkosBackendInstance.inl`：空闲配置入口和统计。
- `src/accelerator/em/kokkos/KokkosCooperativeBackend.cpp`：只有CPU优先实例启用。
- `applications/detail/air_shower_kokkos/KokkosShowerReport.hpp`：只有CPU优先
  报告新增`auxiliary_blocking_wait_enabled/calls/ms`。

`auxiliary_blocking_wait_ms`是主机等待墙钟时间，可能与OpenMP工作重叠，
**不是GPU kernel耗时，不可直接与两端总耗时相加**。

## 验收设计

1. 冻结v4源树/构建作为输入，v5独立构建只覆盖批准的11个文件；其他模块和
   生产install不改动。额外Types改动会触发source-boundary门禁。
2. CPU/单CUDA/单OpenMP固定种子N=2，比较全部物理数组、非计时字段和
   decision tape；真实EM+CoREAS+ZHS fixture与双端N=32检查复用和内存。
3. `CpuPrimaryBlockingWaitRuntime.cpp`验证真实非默认stream上的默认fence、
   阻塞事件、旧回调等待输出一致；检查事件复用、冲突拒绝和不等待无关stream。
4. `CpuPrimaryWaitContentionProbe.cpp`固定相同OpenMP输入、history、调用和
   GPU辅助工作，交替测helper-off/default-fence/blocking-event，每模式1次
   预热、5次测量。两端transport完成后才下载最终profile/波形并hash，避免
   把验证本身的输出整理误当成生产driver竞争。相同端点各模式输出hash必须
   一致，fallback按加速边界流出记账，不冒充完整强子shower能量闭合。
5. 再做相同条件、相同CPU绑定的完整Fe对照。PSR使用后130核，保留其他山体
   工作流，不在其CPU/GPU测试期间抢占资源或测量性能。

固定工作量probe是有界小测试，不是70%显存压力或完整shower加速测量。
端点墙钟窗口相交也不是kernel活动时间线；线程核时与GPU采样另行报告。

## 初版检查记录（历史）

- 本地独立CUDA+OpenMP Release构建通过，构建峰值进程树RSS约2.73GiB。
- 队列/调度/线程绑定3项CTest通过，验证工具38项单元测试通过。
- 首个v5构建的三种单端N=2各11个物理数组及非计时metadata一致，CUDA/OpenMP
  decision tape字节一致。双端N=32缓存和workspace恒定，RSS峰值约0.773GiB。
  CPU优先N=32只有`shower_4`用到GPU（2次端点调用、13次阻塞wait），不能把
  这组称为32次持续GPU压力测试。真实EM/CoREAS/ZHS fixture两例边界能量
  残差约`5.5e-9`，不是完整强子能量覆盖证明。
- 真实CUDA非默认stream测试通过：默认fence、blocking、旧callback各32次
  完全一致，原始blocking ticket另32次；计数重置复用、回调互斥和无关stream
  不被等待通过。该测试不宣称物理统计或性能通过。
- 固定工作probe首次暴露验证程序的重复导出：普通profile和fixed profile
  都会消费同一累积器，不能连续调用。仅修正probe为一次fixed导出再调用
  现有decode函数（射电同理），没有绕开生产的exactly-once门禁。旧失败保留。
- 随后的schema3 probe捕获了**新v5内部生命周期检查错误**：驻留输运更新
  `photon_transport_batches/lepton_transport_batches`，并不更新端点内的旧
  `particles_advanced`。原检查因此未禁止已输运端点切换等待模式。修复是在
  原条件上增加两种实际输运计数，保留触发测试，未改变计数累积或任何物理。
  第一个v5构建仍冻结；修正版在独立`blocking-helper-a2`目录重编译重测。
  不把只有1行warmup的失败probe写成有效三模式性能比较。
- PSR v4尚在等待其他山体任务释放资源；v5未替换该测量或生产二进制。

本地候选：`build/cpu-priority-v5-blocking-helper-20260912`。
门禁修正版：`build/cpu-priority-v5-blocking-helper-a2-20260912`。
结果：`/mnt/d/CorsikaData/corsika_validation_results/beta5_cpu_priority_v5_blocking_helper_local20_20260912`。

## a2 核心修正后的实测检查点

a2 Release构建、3项CTest、三种单端N=2逐数组/非计时metadata/decision tape
回归、真实双端EM/CoREAS/ZHS和N=32检查均重新通过；实际CUDA stream事件
测试也重新通过。原v5和验证程序的失败记录不删除、不算作完整性能样本。

固定工作probe使用20个OpenMP线程，**整个进程限制在逻辑CPU 0–19**，GPU
driver与OpenMP共享这20个CPU，而非使用额外空闲核；这是本地竞争诊断设置，
不是PSR的130线程结果。每端32,768个混合photon/electron/positron初始粒子，
初始能量10–10.9MeV；每模式1次预热加5次正式重复，固定全部输入和history。
CPU/GPU实际推进930,673/933,773步。相同端点在各模式下的调用序列、物理计数、
定点profile/波形、末态和fallback随机来源哈希完全一致；这不是跨端逐位一致。

| 对照 | 不启用辅助GPU的CPU时间 | 原fence的CPU时间 | blocking的CPU时间 | blocking相对无辅助 |
|---|---:|---:|---:|---:|
| 无射电，固定较大EM输入 | 0.371s | 0.713s | 0.388s | +4.6% |
| CoREAS+ZHS，干净重测 | 1.781s | 1.976s | 1.822s | +2.3% |

全射电同一GPU工作量中位时间0.503→0.468s（约减少6.95%），GPU driver核时
0.425→0.365 CPU秒；blocking模式每次480次等待。guard记录的进程树RSS约
0.57GiB，缓存/输出有限，能量加速边界残差最大约`9.7e-11`。固定snapshot和
解码数组来自同一次导出，不冒充独立下载路径对照；普通路径另由N=2回归覆盖。

这些数字说明该等待策略缓解了此固定工作量上的CPU竞争，**不证明完整shower
已加速，也不证明OpenMP完全无性能损失**。无射电+4.6%仍高于3%目标；全射电
+2.3%仅是这五次重复的描述性中位数，没有做1%等价或跨机器性能断言。
GPU/CPU同时承担的输入总量也不同于helper-off，不能用这里的joint时间直接
计算完整shower加速比。GPU常驻容量8,192、显存比例10%、4个observer均为
明确的小型诊断配置，不是70%显存的Fe生产测试。

第一轮全射电运行的最后约3秒与随后正确性runner的CPU前置检查相交。虽然
没有同时占GPU，仍将其计时标记为存在潜在CPU干扰，保留全部18行并重跑整个
矩阵。新的`KNOWN_INTERFERENCE.json`分析门禁会撤销计时有效性而保留正确性；
此处只引用后续`fixed-work-a4-large-radio-clean-report/`的完整干净重测。

本地a2回归目录：
`/mnt/d/CorsikaData/corsika_validation_results/beta5_cpu_priority_v5_blocking_helper_a2_local20_20260912`。
固定工作小报告位于原v5结果根目录的`fixed-work-a4-*-report/`；第一次全射电
的复核与干扰声明在`fixed-work-a4-large-radio-on-review1/`和对应guard目录，
不能继续使用该第一次报告作为性能通过依据。

PSR修正版11文件与验证工具已在独立只读r2输入包逐项校验。47项远端验证工具
测试和构建dry-run通过。最初的`c8-cpu-priority-v5-psr-build-queued-r2-20260912`
排队器命令行包含它正在等待的山体工作流路径，被旧v4的保守进程检查误认为
山体计算，形成相互等待。该排队器在编译尚未启动时停止；没有停止v4或山体
任务，也没有把被误识别的等待记为计算耗时。

修正后通过哈希锁定的JOB文件、同进程`runpy`入口启动，完整门禁参数不再
出现在OS进程命令行；门禁本身和实际资源检查保持不变。5项launcher测试
包含真实`/proc/self/cmdline`检查。新单元
`c8-cpu-priority-v5-psr-build-job-r1-20260912`已运行，核实v4的
`workflow_pids=[]`，且v4继续完成了第4个记录，没有互等。
它等待v4成功结束、其他山体工作流及GPU连续空闲120s，再用128个编译
jobs低优先级构建，不更换生产install。该门禁只保护启动，不是跨工作流的
资源锁；后续正式130线程、382–511绑定的完整Fe计时仍需实时资源检查。
排队构建不是已开始shower，也不是PSR性能通过。

## a3：补齐正常 checkpoint 的等待覆盖

完整调用链审计还发现：光子16波/轻子8波的辅助调用达到检查点、同种队列
仍非空时，会执行`queue.download()`。它的pack与D2H之后原来仍是普通fence，
不经过a2的blocking waiter；不是只有内存故障或最后flush才走这条路径。
现有v4总wave/call/packet计数不能反推“非空检查点次数”，不能估算其耗时占比。

a3只改三个共享头文件的等待接线：`KokkosWavefrontQueue.hpp`和两种
`KokkosResident*Cascade.hpp`。最终下载可接收现有waiter，但队列内部只接受
`blockingEnabled()`的等待方式；非blocking progress回调仍不在这里调用，
避免为原GPU优先路径引入新的重入点。pack、D2H、同步标志、host memcpy、
返回粒子顺序均不变；始终等待queue自身的owning stream，而非调用者的兼容
参数。pending压紧、迁移、实际扩容和其他旧接口本轮不改。

独立构建为`build/cpu-priority-v5-blocking-helper-a3-20260912`，Release已通过。
构建工具需要额外显式`--include-cpu-primary-checkpoint-wait`；三个共享头文件
只允许逐字匹配的等待补丁，任何额外物理改动都会拒绝。a2之前的11文件输入
包不修改；当前是14个允许覆盖文件中实际3个发生变化。运行回归与真实stream
新测试现已通过，具体结果见下节；前面的a2性能数字不能直接称为a3结果。

为让PSR测试使用这份完整候选，已核实旧a2 job-r1仍处于`waiting`、
`launches_started=false`且candidate目录尚不存在后，停止**仅该排队单元**。
它已证实没有互等；停止是为了更新候选，不是再次发生互等，也没有停止v4或
山体任务。新独立a3包已在本地回归后启动排队。额外的可选验证门禁会
在每个事例前同时检查已知CPU工作流和GPU；运行中CPU干扰只撤销计时有效性，
保留物理结果，不终止正常事件，也不将抽样监控描述为整台服务器独占。

## a3 完成的本地检查（2026-09-12）

- 三种单端固定种子 `N=2` 的物理数组、非计时 YAML 和表哈希一致；CUDA 和
  OpenMP decision tape 字节一致，`--help` 与失败门禁通过。
- 双端真实 EM＋CoREAS/ZHS fixture、两种优先级的 `N=32`、三项调度/队列/
  绑定 CTest 均通过。`N=32` 的表与 workspace 保持恒定，进程树 RSS 峰值约
  0.775 GiB，系统可用内存最低约 11.03 GiB；未出现观察范围内的逐事例增长。
  CPU 优先这组只有 `shower_4` 实际启用辅助 GPU：5 个 photon 和149个 lepton
  transport records、54次 blocking wait。因此不能称作32次持续双端压力测试。
  GPU 优先 `N=32` 的 NVML 有3次无法核验的旧 PID 观测，保留为正确性结果，
  **其耗时不能用于性能结论**。
- 原始 CUDA stream 测试通过：除默认/阻塞/回调路径外，新增32次不同队列
  长度的精确回传，包含全部粒子字段、signed zero、高位 history。空队列
  不新增等待，非阻塞回调不被 checkpoint 引入，故意传入不同 stream 时
  仍等待队列自己的 stream。该验证不改变任何物理计算。
- 逐项源码边界审计确认相对 a2 只变更上述三个等待接线头文件，无其他覆盖。

### 固定工作量耗时：保护主端，但尚不能宣称零损失

与 a2 相同的20逻辑 CPU绑定、每端32,768个输入、10–10.9 MeV及4天线小型
诊断；每模式1次预热、5次正式重复。两次运行串行进行，没有同时构建或执行
正确性测试；两份guard和全部18行检查均有效。所有端点输入、调用序列、
物理计数、profile、波形、末态、fallback随机来源、表和辅助缓存哈希一致。

| a3固定工作量 | 无辅助GPU的OpenMP中位时间 | 默认fence | 阻塞event | 阻塞相对无辅助 |
|---|---:|---:|---:|---:|
| 不计算射电 | 0.3638 s | 0.7035 s | 0.3929 s | +8.0% |
| CoREAS＋ZHS | 1.7908 s | 1.9750 s | 1.8115 s | +1.2% |

全射电同一辅助 GPU 工作量中位时间为0.5078→0.4756 s（减少约6.3%），
driver线程核时0.4248→0.3615 CPU秒（减少约14.9%），每次498个blocking
wait。RSS峰值约0.569 GiB。这说明等待修改在该负载上减少了主端竞争并未
以降低GPU效率为代价；**不代表130线程PSR或完整shower已经加速**。
无射电仍慢8.0%，不满足主端性能保护目标。两个模式的总输入量不同于
helper-off，也不能以这里的joint wall time直接计算shower加速比。

计时中的剩余driver核时不能都归因于提交kernel：逐次无射电记录中，
`driver wall - blocking wait wall`仅约0.048–0.058 s，而driver核时约
0.308–0.316 CPU秒。这意味着大量核时仍落在event记录/同步范围内部。
需进一步区分这两个API的主机开销；仅凭`cudaEventBlockingSync`标志不能
断言等待期间完全不占CPU。本轮不因此修改CUDA全局调度设置或牺牲OpenMP线程。

结果根目录：
`/mnt/d/CorsikaData/corsika_validation_results/beta5_cpu_priority_v5_checkpoint_helper_a3_local20_20260912`。
两份小报告分别位于`fixed-work-a5-noradio-report/REPORT_CN.md`和
`fixed-work-a5-radio-report/REPORT_CN.md`，包括全部重复、监控状态和SHA-256。

### 本地完整 Fe 对照：按用户要求降至10 TeV

用户要求在服务器另有任务期间先本地测量，因此另启动独立a3候选的Fe-56
100 TeV对照。20个OpenMP线程，整个进程限制在CPU0–19，比较`openmp`与
`openmp-cuda`。5个固定种子85000001–85000005交替运行，两者物理参数、
max-weight默认值、天线与射电时间窗一致；GPU辅助上限70%、batch4096。
守护进程树RSS上限6GiB，系统可用内存低于4GiB立即停止自有测试。

随后用户要求降低能量以缩短20线程测试时间。上述100 TeV队列已停止：
纯OpenMP seed85000001完整进程150.6044s，数据保留；同seed辅助GPU模式在
运行中停止，不参与配对计时。原目录有`STOP_REQUEST.md`，不再运行该队列。

本地服务：`c8-cpu-priority-v5-a3-local-fe20-r2-20260912`。
输出：`/mnt/d/CorsikaData/corsika_validation_results/beta5_cpu_priority_v5_checkpoint_helper_a3_Fe100TeV_local20_20260912_r2`。
该目录仅保留100 TeV的部分记录；排队/启动成功不等于完整shower性能通过。

首次启动时报告观察器抢先创建输出目录，触发runner的防覆盖检查；失败在
任何shower启动前发生，已保留原目录`STARTUP_FAILURE.md`，不计作性能样本。
重启使用新目录，核实runner的`STATUS.json`存在后才启动报告观察器。

新的10 TeV服务为`c8-cpu-priority-v5-a3-local-fe10tev20-20260912`，结果目录：
`/mnt/d/CorsikaData/corsika_validation_results/beta5_cpu_priority_v5_checkpoint_helper_a3_Fe10TeV_local20_20260912`。
仅将`-E`降至10000 GeV，20线程、CPU0–19绑定、两种模式、种子矩阵、薄化/
max-weight等命令参数保持不变。验证工具新增明确的`--fe-energy-gev 10000`，
正确标注`Fe10TeV`；不会用100 TeV标签掩盖能量变化。

同一a3二进制已通过的完整回归可通过`--reuse-correctness`显式引用：验证
binary/fixture/before哈希、source和线程数，核验三种单端回归、trace及两端
N32/fixture，保存18份证据SHA。只复用正确性，**不复用此前的性能数据**。
新验证包为`build/cpu-priority-v5-lowenergy-tools-20260912`，149项host-only
测试通过；原a3冻结构建与PSR包未改变。报告观察器抢建目录的竞态也已在
工作树与新验证包修复，有真实目录回归；不修改当前shower程序。

低能结果必须同时检查GPU实际输运量。如果辅助端完全空闲，不能凭两种模式
耗时接近就声称辅助GPU不影响CPU。完整shower的每步吞吐也受能谱、射电和
动态分配影响，需与前面的同输入固定工作量控制一起解释。

### 10 TeV 五对实测完成：CPU 吞吐接近，但整体没有净收益

上述10 TeV测试现已10/10完成，运行与报告systemd单元均正常退出。5个种子
均保留，物理配置匹配；全部监控有效、输出关闭、输运步数账目与回退清空
检查通过。`COMPLETED_OUTPUT_AUDIT.json`仅证明这些完整性条件，不是500例
物理统计或完整强子能量守恒证明。正确性复用的18份证据也重新通过哈希核验。

下表均取各模式5个事件的中位数，不混用平均值、配对比值和中位数之比。

| 指标 | 纯 OpenMP，20线程 | CPU优先 OpenMP＋CUDA，20线程 | 双端相对变化 |
|---|---:|---:|---:|
| CPU端输运调用墙时 | 11.7958 s | 11.9429 s | +1.25% |
| CPU端每事件输运步/s | 573,456 | 568,049 | −0.94% |
| 完整 shower 墙时 | 12.6294 s | 12.5970 s | −0.26% |
| 完整进程墙时 | 20.8209 s | 22.3613 s | +7.40% |
| 进程减 shower 时间 | 8.2641 s | 9.7932 s | 多1.5291 s |
| 已记录的一次 backend 初始化 | 0.6374 s | 1.1144 s | 多0.4771 s |

CPU端调用时间包含本端EM、射电、复制和同步，并非CPU核时或纯kernel时间。
两模式能谱、粒子树和波前形状不完全相同：逐种子CPU吞吐变化从−8.22%到
+11.07%，不能以中位数接近就宣布没有CPU竞争。作为口径交叉检查，配对
吞吐比的中位数为0.998086，而汇总steps/汇总CPU时间的比值为1.003331；
它们不是上表各组吞吐中位数之比。各项中位数也不满足可加性。

GPU确实工作，承担总输运步数的0.1945%–2.1644%，中位1.6707%。当前策略
保留每PID完整CPU arena，仅将至少4096个的余量交给辅助GPU；不为提高GPU
利用率抢走主端正在处理的队列。该工作份额与保护策略相容，但现有计数
不能定量区分每一种未获任务的原因，也不能从CUDA端的pending峰值推断CPU峰值。

CPU无工作等待中位仅2.96 ms、最大约6 ms。GPU结果领取延迟中位5.78 s是
完成至协调器收取结果的累计mailbox延迟，**不是CPU等待GPU的时间**。五对
均没有workspace-limit checkpoint、显存spill或overflow。双端进程树RSS
峰值1.308 GiB，整个测试系统可用内存最低10.563 GiB；辅助模式设备总显存
峰值5388 MiB。显存分配不是有效计算吞吐，不能用它证明GPU充分利用。

结论：低能完整shower耗时近乎持平，但进程总时间仍慢7.4%。其中额外外层
开销约1.53 s，只有约0.48 s能归入已记录的backend初始化，不能把其余全部
称为初始化。再结合固定输入无射电测试的CPU +8.0%，目前**没有通过“不降低
OpenMP效率并获得净加速”的性能目标**。不放松CPU保护、不改物理，也不将
该实验候选替换生产安装；下一步应针对辅助driver等待开销和有用工作量
进行窄范围验证，而非单纯提高GPU分配比例。

结果目录下的`performance/REPORT_CN.md`和`performance/Fe10TeV_timing.png`
包含全部配对时间；`diagnosis/REPORT_CN.md`给出两端真实步数、wavefront、
fallback及等待计数。`CONCLUSION_CN.md`为本轮简短结论。

### 独立CUDA event API诊断

在100 TeV队列完全停止、10 TeV计时尚未启动的间隙，单独进行无射电固定工作量
注入诊断，不改核心。`cuda_event_wait_preload.c`仅观察已有event API，原样
传递参数/返回/errno，不增加CUDA调用；6项fake-cudart主机测试通过。

18/18行输入、调用与物理哈希与未注入a5相同。阻塞事件flags=3，Record和
Synchronize各2988次（6轮包含1次预热）：Record墙时0.012831s、核时0.012612
CPU秒；Synchronize墙时5.594421s、核时3.377343 CPU秒。后者占这次阻塞模式
GPU driver总核时3.731046 CPU秒的90.52%，Record占0.34%。无未知event、
记录溢出或计时错误，正常退出后0个存活event。

这支持剩余主机开销主要位于同步API内部，而不是记录event；**不能再凭该
计时区分驱动自旋、系统调用、提交或唤醒，更不能将注入前后时间比称为提速**。
正式a5未注入报告保持不变。独立报告为前述a3小测试根目录下的
`event-api-injected-diagnostic-report/REPORT_CN.md`。

### PSR验证队列

新的a3构建单元`c8-cpu-priority-v5-psr-checkpoint-build-job-r3-20260912`
与后续单元`c8-cpu-priority-v5-psr-checkpoint-followon-r3-20260912`已启动
持久等待。新入口仅将哈希锁定JOB/SPEC文件路径放入命令行；再次核实它们
没有被旧v4误判成山体进程，不存在先前的互等。

本地回归通过后冻结的r3输入包仅包含14个允许覆盖文件；远端输入SHA、87项
验证工具测试和只读预检已通过。构建成功后依次运行真实stream检查、固定
工作量无/有射电检查、单端/N32回归，以及5个Fe-56 100 TeV种子的交替
`openmp`/`openmp-cuda`对照，**两种模式均130线程、CPU382–511**，显存上限
70%、batch4096、物理参数和max-weight不变。

后续流程有9小时总期限；每步和每事例都等待已知CPU工作流及GPU空闲，
错误或不完整结果不会算成通过。当前尚未获得该候选完整PSR Fe耗时，既有
v4结果仍保留。远端结果目录为：
`/data/yhlu/CorsikaData/corsika_validation_results/psr_cpu_priority_v5_checkpoint_followon_130_20260912`。

后续状态：上述r3队列因旧v4临时服务记录消失而保护退出，未开始构建。
用户确认服务器空闲后，r4独立构建已成功并进入新验证流程，样本改为与
本地一致的100 TeV质子。旧队列和失败记录保留；详见
[本地/PSR质子部署与验收记录](beta5_cpu_primary_proton100TeV_local_psr_20260912_CN.md)。
