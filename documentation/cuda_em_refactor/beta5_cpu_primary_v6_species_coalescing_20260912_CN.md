# CPU优先v6：有界的小物种波前合并

状态：候选已实现；本地和PSR的Release、单端真实输出与decision-tape回归、
双端fixture及N=32门禁通过。PSR五对100 TeV已完成；本地按用户要求在最后
一个事件中途停止，保留9个完整事件。后续仅在服务器测试。未运行install，
不宣称净加速。

## 问题与单变量修改

v5在PSR的5对100 TeV质子中，双端CPU步数吞吐中位低约6.4%，GPU步数占比
约3.8%。某事件CPU波前数由8877增至19620。相同seed不保证相同shower树，
这些数字只能指向批次形状的诊断，不能证明单一原因。

本候选只改变`openmp-cuda`的CPU种类选择，不改变GPU分配份额、输入容量、
波数上限、history租约大小、fallback合批、阻塞等待、kernel、随机公式、
PROPOSAL表、cut、thinning或射电公式。单端和原GPU优先路径不进入新分支。

规则：

1. 原本将处理的光子/轻子队列低于`min(min-batch,本种类容量)`，且另一种
   队列已经达到其有用批量时，先处理另一种。
2. 一个物种最多连续暂缓8次CPU调度选择，第9次必须获得服务；不使用墙钟
   自适应控制，不等待未来输入，不许丢弃低于阈值的粒子。
3. 两种队列都小、都足够大或只剩一种时，保留原顺序/立即排空。
4. 每个物种内部仍先消费resident，再补入waiting；粒子字段不改。

这不保证完整CPU批次与纯OpenMP完全相同，也不保证相同seed产生相同树。
它仅消除一种已确认存在于代码中的不必要小队列优先调度。

## 代码与诊断

- `CpuPrimarySubshowerPolicy.hpp`：纯host选择函数及8次饥饿保护。
- `IndependentSubshowerPump.hpp`：仅CPU优先、CPU端调用新选择函数。
- `KokkosShowerReport.hpp`：策略名`cpu-primary-v6-species-coalescing`，增加
  `host_species_coalesces`、`host_species_maximum_deferrals`和选择语义。
- `testIndependentSubshowerPump.cpp`：对称边界、单独尾部、两种小/大队列、
  小容量、空队列状态重置；模拟持续大队列证明第9次服务小队列、无粒子/步数
  丢失或重复；原GPU优先的种类选择不变。

复用已有`adaptive_species_coalesces[1]`存储计数，不启用adaptive控制器；
输出使用CPU专属名称，GPU优先metadata不新增这些字段。验证脚本识别v6，
保留v5所有门禁，并拒绝负数、非整数或超过CPU工作段数的合并计数。

## 构建与冻结

以每台机器的v5 a3冻结目录构建，源代码边界严格只允许上述4个文件变更，
没有复制当前工作树中无关的山体开发。旧二进制和结果保留。

本地构建：`build/cpu-priority-v6-species-20260912`；程序SHA-256：
`f45b3f693514d4067feb7ab5502075a1bcf726c1f6222527f5f0e24e2ac8fa5f`。
构建约221 s；3个调度/绑定CTest通过，26个验证脚本测试通过。

本地首个等待器在尚无simulation时发现`--before`使用相对路径，已停止该
等待器，旧STATUS保留；r2全部改用绝对路径。没有停止正在运行的shower。

PSR构建：`build/psr-cpu-priority-v6-species-r2-20260912`，与本地使用同一份
4文件只读补丁包，复用PSR现有依赖、FLUKA与SIBYLL。没有重新制表。

PSR首个stage在编译前遇到复制自冻结源树的只读文件权限，未开始shower。
构建工具现在仅将新candidate中的独立普通文件临时设为可写后覆盖；拒绝
符号链接、越界父目录和硬链接，不改变冻结源文件权限。2个新测试及8个原
构建测试通过。失败stage和状态保留；r2使用新的目录和持久服务。

## 验收与资源保护

两机依次执行CPU/单CUDA/单OpenMP的N=2物理数组和decision-tape回归、真实
双端EM/CoREAS/ZHS fixture、两种优先级N=32生命周期，再运行5对100 TeV质子。
各项未通过即停止后续shower，报告服务保留失败状态而不宣布通过。

参数保持v5：质子100000 GeV、vertical、phi=0、emthin=1e-6、max-weight=0、
79天线、IGRF14(2027)、完整CoREAS/ZHS、70% GPU预算、min-batch4096；种子
85000001–85000005，AB/BA交替。本地20线程CPU0–19，PSR130线程CPU382–511。
不增加物理cut、不通过改变薄化提速。

本地独立systemd测试树上限8 GiB，逐事件RSS上限6 GiB，可用内存下限4 GiB；
PSR40 GiB cgroup与逐事件24 GiB上限。监控不完整的事件保留，只撤销严格
计时有效性。所有样本保留，包含慢事件。

结果位置：

- 本地：`/mnt/d/CorsikaData/corsika_validation_results/beta5_cpu_priority_v6_species_proton100TeV_local20_r2_20260912`
- PSR：`/data/yhlu/CorsikaData/corsika_validation_results/psr_cpu_priority_v6_species_proton100TeV_130_r2_20260912`

独立报告服务在结束后生成`performance/`、`diagnosis/`和完整输出审计；PSR
原始数据留在服务器，只同步图片和小报告。N=32是固定工作集生命周期检查，
不是任意能量无内存泄漏证明；5对计时不替代500例物理统计验收。

## PSR五对初步结果

五对事件的监控均有效。纯OpenMP进程时间：均值46.738 s、中位48.630 s；
双端：均值48.094 s、中位48.255 s。中位缩短0.77%，但均值反而增加2.90%，
不能据此认定优化有效。必须同时查看工作量差异，不能只展示中位数。

PSR五个双端事件均为`host_species_coalesces=0`；该选择规则没有触发，
不能将这些事件的耗时变化归因于v6。PSR完整计数见原始各事件
`gpu_em/summary.yaml`。下一候选将单独检查CPU小尾部返回阈值，不改变粒子cut。
