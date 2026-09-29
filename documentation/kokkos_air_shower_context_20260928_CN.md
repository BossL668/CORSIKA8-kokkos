# 空气簇射 Kokkos 上下文接口整理（2026-09-28）

## 修改内容

主应用改为：

```cpp
air::runKokkosAirShower(accelerated_session, gpu_cli, cuda_event, context);
```

原来的 52 个位置参数收拢为 4 个顶层参数。新增
`applications/detail/air_shower_kokkos/KokkosAirShowerContext.hpp`，
按职责归组：

| 分组 | 内容 |
|---|---|
| Geometry | 环境、坐标、传播步长、观测高度、磁场 |
| Outputs | profile/能损/粒子输出及射电探测器 |
| Processes | 已构造的物理过程、薄化、cut 与过程注册所需类型 |
| Execution | sequence、tracking、stack、输出管理器、CPU worker pool、首反应回调 |
| Diagnostics | 原有强子模型计数器及事件开始时的统计快照 |

模型、writer、stack、探测器和回调均通过引用绑定，生命周期仍由主应用管理。
磁场三分量数组和标量配置按值保存，避免临时数组悬垂引用。
context 在当前事件调用处构造，不得存入异步任务或让其活得比绑定对象更久。
拷贝 context 只拷贝引用绑定和少量配置，不复制模型/粒子栈。

新四参数重载使用一处可单测的映射转交给原有计算入口。
旧长参数重载保持兼容，C7 对照应用仍可使用。
此次没有修改物理 kernel、随机数、过程顺序、CPU 回退、表查询、
统计快照捕获时机、输出注册或异常清理。
没有取消任何原有物理输出或诊断字段，也没有增加命令行参数。

这是一轮接口组织整理，不是把任意新应用自动接入 GPU 的通用插件；
主应用仍需明确构造各模型和环境，再按职责绑定到 context。

## 验证结果

- 原计算函数体逐字一致；主应用只有 context 构造和调用区域发生改变。
- 前后分别重新编译空气应用翻译单元；编译选项与 59 个链接输入一致。
  复用现有 CUDA/OpenMP 构建依赖，不用旧安装二进制冒充当前源码基准，
  也不覆盖生产安装。
- 独立 C++17 单元测试检查全部 49 个被转交参数的类型/引用或配置值、
  不可复制模型、const 绑定、磁场临时数组、重复调用、空 pool 和异常传播。
  ASan/UBSan 与 Release（保留断言，启用 -Werror）均通过。
- C7 对照应用通过保留旧接口的编译检查。
- 六组实际回归全部通过，每组前后各运行 2 个事件：
  CUDA 电子、OpenMP 电子、CUDA 光子、CUDA 正电子、
  CUDA 质子及原生标量电子。
- 66 个物理数据文件逐项精确一致，包含 longitudinal profile、能量沉积、
  粒子/反应记录、CoREAS 和 ZHS；两种射电输出均有非零信号且全部采样值一致。
- 物理统计计数、表标识、过程注册、能量账项及第二事件后端复用标记一致。
  实测墙钟计时允许变化，不能据此宣称整份输出目录逐字节相同。

初次基准生成了辅助表缓存，而候选程序命中缓存；物理数组仍完全一致，
但严格元数据检查发现 `aux_cache_hit` 不同。完整冷缓存记录已保留，
随后在同为热缓存的条件下重跑该基准案例；最终六组没有未解释的元数据差异。
未扩大忽略规则来掩盖这项差别。

测试条件：本地 RTX 4060 Laptop GPU，OpenMP 4 线程；
电子/光子/正电子 10 GeV，质子 100 GeV（强制首相互作用）；
seed=26092817，N=2，天顶角 25°、方位角 225°，
注入高度 5000 m、观测高度 1100 m，无薄化，内置 ring=1 天线，
正电子案例使用 both 穿界统计。CUDA 单线程主机调度，
驻留批次上限 4096，显存预算 5%。

## 复现与边界

项目测试已注册为 `testKokkosAirShowerContext`，不需要 GPU 即可运行。
回归程序和全部原始证据位于：

`/home/yuhanglu/21CMA/analysis_jobs/air_shower_context_20260928/`

- `baseline/build.json`、`candidate/build.json`：编译命令、源码及依赖校验值。
- `regression.py`、`regression.json`：测试矩阵与逐项结果。
- `runs/`：保留的前后输出。
- `initial_cache_mismatch.json`、`runs/cold_cache_baseline/`：首次缓存状态不一致的记录。

这次验证是同后端、同种子的应用接口等价性检查，不是跨后端逐位一致性证明，
也不是新一轮完整物理统计验收。未运行四卡生产规模回归，
未修改多卡协调器，未部署服务器、覆盖 install 或推送 GitHub。
