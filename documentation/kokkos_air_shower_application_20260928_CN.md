# 空气簇射主应用的 Kokkos 适配层

第二轮封装把固定绑定从逐事例分支移到了库初始化阶段。主应用不再逐例填写
`cuda_event` 或展开 `KokkosAirShowerContext`，而使用：

```cpp
auto EAS = accelerated_application.cascade(
    tracking, sequence, output, stack, stackInspect, thinning,
    static_cast<std::uint64_t>(i_shower), primaryTotalEnergy, maxWeight,
    diagnostics_before);
configure_forced_primary(EAS);
EAS.run();
```

## 分工

- `KokkosAirShowerApplication.hpp`：一次绑定几何、输出、模型和诊断对象；从实际
  cut 对象及应用选项读取固定配置，创建每例的轻量级 facade。
- `KokkosAirShowerBindings.hpp`：非拥有型模型角色、拥有数值的诊断快照，及
  `forceInteraction()` / `forceDecay()` / `run()` 接口。
- `KokkosAirShowerContext.hpp`：仍是到原 runner 的显式参数映射，不参与输运。
- `KokkosAirShowerRunner.hpp`：沿用已有输运、CPU 回退、射电和报告过程；保留
  旧接口供 `c8_corsika7_compare` 等调用者使用。
- `KokkosRunSession`：继续管理跨事例后端重用与输出，生命周期未改变。

几何、模型、输出仍需在主应用中明确配置，与原生 `Cascade` 一样；适配层没有
通过全局变量、无类型容器或改变物理默认值来隐藏这些依赖。

## 行为与生命周期约束

模型和 writer 不复制，不重建；它们及 session 必须比适配层活得更久。
每例返回的 `EAS` 必须在 tracking、sequence、stack、inspector、thinning
销毁前执行。诊断快照在原事件边界捕获，并按值保存，不因后续计数改变而变化。
固定配置从已解析且在事件循环中不再改变的 CLI 读取；能量抽样、max-weight
计算、物理过程顺序、首粒子设置、随机流与输出 start/end 顺序保持原样。

多卡协调器、frontier 切分/导入、设备内核及合并算法没有修改。
多卡 worker 仍通过同一个应用入口和原 runner 的 frontier hooks 执行。

## 验证位置

隔离的构建、源文件快照和回归输出：
`analysis_jobs/air_shower_facade_20260928/`（工作区目录）。

`regression.json` 包含 10 组、每版本每组 2 例的结果：原生 CPU，OpenMP
电子/正电子，CUDA 电子/光子/正电子/质子，薄化，随机初级能量及强制 μ 衰变。
110 个物理数据文件逐项完全一致；CoREAS 和 ZHS 非零波形也完全一致。
仅耗时等既有比较器明确列出的非物理诊断允许不同；物理飞行时间不豁免。
两次构建使用同一批缓存后端库，验证的是应用封装行为，不是全新物理实现的验收。

`testKokkosAirShowerBindings` 不依赖 GPU；检查模型引用、快照值、读取顺序、
空 pool、强制首相互作用/衰变以及异常透传，覆盖 ASan/UBSan 与 Release。
四张 L20 实测也已通过，见 `four_gpu_comparison.json`：每个版本 2 例光子、
1 例质子，三个事件的 prefix、四个 worker 和 merged 均逐项一致；frontier
及其分区哈希也一致。测试在生产第 89 例完成后执行，随后恢复原队列。

生产安装与 GitHub 不由这次接口重构自动替换。
