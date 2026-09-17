# 独立跨介质 Kokkos 山体射电模块

2026-09-13：山体介质参数统一由[材料模型](mountain_material_models_CN.md)提供，
包括折射率和电场衰减长度。旧场景的分散参数需要按该文迁移；SiO₂ 默认已更新为文件中的二氧化硅基准（n=√5）。

模块位于 `corsika/modules/radio/interface/`，CMake 目标为
`CORSIKA8::InterfaceRadio`。它复用 mountain 的有限轨迹 ZHS、Snell/Fresnel、
射线管扩散和连续时间区间矩的计算方法，独立实现设备侧求解与累加。
没有修改原有空气射电累加器、空气应用或空气输运核。

山体应用启用射电后默认同时计算 CoREAS 和 ZHS，不提供算法选择项。
两套算法共用输入轨迹和光学配置，分别输出结果。端点传播、辅助矩数组与适用范围见
[山体 CoREAS 说明](interface_coreas_CN.md)。旧的 `--radio-algorithm` 和 `radio.algorithm` 需要移除。

## 接口和设备生命周期

`Config` 提供两侧光学介质、观测点、采样窗、精度控制和设备内存预算。
介质可指定均匀折射率、径向折射率表和电场振幅衰减长度。编号 0 表示外围介质，
1 表示嵌入介质，不要求分别为空气和岩石。几何支持均匀介质、无限平面和闭合三角网格。

`KokkosAccumulator<ExecutionSpace, true>` 借用调用者已经上传的索引 BVH，
自己持有共用的折射率表、观测点、CPU 输入暂存区，以及两算法各自的统计量和矩数组。
它不创建或销毁 Kokkos runtime，也不依赖空气应用。调用者须保证借用网格的生命周期。

```cpp
using Radio = corsika::radio::interface::KokkosAccumulator<ExecutionSpace, true>;
Radio radio(config, geometry.deviceView(), face_count, batch_capacity, execution);
radio.accumulate(device_tracks, count, execution);  // 已在设备上的带电轨迹
radio.accumulateCpu(cpu_tracks, execution);         // CPU 产生的物理轨迹
auto result = radio.download(execution);           // 结束时下载一次
corsika::radio::interface::writeResult(result.coreas, output_directory + "/CoREAS");
corsika::radio::interface::writeResult(result.zhs, output_directory + "/ZHS");
```

底层仍保留单算法实例用于独立数值对照；山体应用固定使用上述双算法实例。

`Track` 使用 SI 单位，包含起终点、起终时间、以基本电荷为单位的带符号电荷、
权重、历史编号和来源介质。每条轨迹须位于同一介质，穿越界面的轨迹由输运模块切分。
轨迹是直线段；弯曲运动依靠输运步长离散。磁场属于输运介质配置，射电模块读取磁场
作用后的轨迹，不再次对粒子施加磁场。

山体 `InterfaceEmSession` 在完成电子/正电子物理输运之后、顶点和队列记账之前，
直接把该步写入设备射电轨迹缓冲区。CUDA 将输入通过设备内复制追加到 8192 条容量的
待算队列，满队列或最终结束时对同一批轨迹分别启动 `interface_radio_coreas_endpoints`
和 `interface_radio_zhs_intervals`。这不改变输运批次、
随机数或物理轨迹。CPU 输运的 μ、τ、带电强子等通过连续过程回调分批上传，
再进入单独的设备待算队列，以保留来源计数。CPU 来源只上传一次，供两算法共用；
OpenMP 直接处理输入批次。
GPU 已完成的轨迹和指定 CPU 顶点回退不重复贡献。

CUDA 下，输运队列、轨迹、几何和射电区间矩保留在 GPU；OpenMP 下使用相同模板的
OpenMP 执行空间。CPU 仍协调波前、处理指定回退和最终输出。这不是单个永久运行的
CUDA kernel。最终 FFT 和 CSV 写出在 CPU，运行中的射电矩数组不逐波前传回 CPU。

当前射电核按“轨迹 × 观测点”并行，透射候选仍需枚举三角面。CUDA 合并小输运批次
以增加并行量；大型 DEM 的透射枚举仍可能限制吞吐。
验收资源记录保留完整耗时，不能据此宣称 CUDA 比 256 线程 OpenMP 更快。

## 山体应用启用方式

编译时用 `C8_INTERFACE_EXECUTION_SPACE=CUDA` 或 `OPENMP` 选择独立接口模块的执行空间，
底层 Kokkos 包必须已启用该后端。山体命令示例：

```sh
c8_terrain_cascade --scene scene.yaml --output new_output \
  --em-backend kokkos --em-scheduler resident --radio \
  --threads 256 --device-memory-MiB 512 [原有初级粒子和输运参数]
```

也可在场景中写 `radio.enabled: true`。`--radio` 目前要求 `--em-backend kokkos`；
`batched` 参考调度器也可累加，用于与常驻调度器核对。

```yaml
radio:
  enabled: true
  observers:
    - name: diagnostic_overhead
      position_enu_m: [0, 0, 1000]
  air_index_model: native       # 或 uniform
  uniform_air_index: 1.000327  # 只在 uniform 模式使用
  index_table_step_m: 10
  optical_integration_samples: 64
  samples: 65536
  sample_rate_GHz: 1
  start_ns: -1000
  moment_order: 12
  subdivision_frequency_GHz: 2
  fraunhofer_limit: 0.025
  mesh_maximum_segment_m: 0.1
  maximum_subdivision_depth: 20
  memory_MiB: 384
  # rock_attenuation_length_m: 100
```

新场景适配器从实际原生五层大气对象采样密度，并统一采用海平面密度参考：
`n(h) = 1 + (n_sea - 1) ρ(h) / ρ_sea`。这避免原有逐层独立归一化属性把常密度顶层
重新设为海平面折射率。修正仅限新适配器，原有空气类保持不动。
在层边界两侧保留采样点，避免把模型跳变涂抹到普通插值单元中。
`radial_integration_breaks_m` 指定已知层边界：光程积分先在
光路与这些球面的所有交点处分段，然后使用复合八点 Gauss 积分。
`optical_integration_samples` 是每段的采样控制数，向上取整到 8 的倍数。
岩石折射率读取场景原有的 `rock_refractive_index`。
场景适配器的观测点必须位于山体外；底层通用接口也支持介质 1 中的观测点。

`memory_MiB` 是射电子模块预算，`--device-memory-MiB` 是包括射电在内的会话总预算；
超预算会拒绝启动。三份矩数组合计为 `3 × 8 × samples × observers × 6 × (moment_order+1)` 字节，
另外还有几何、物理表、队列和暂存区；CUDA 射电预算包括两份 `8192 × sizeof(Track)`
待算队列。该预算不包含 CUDA 上下文、运行时和 kernel 栈的开销。
独立射电 PSR 验收用 `run_interface_radio_guard.py` 另行监测实际显存增量，默认上限
768 MiB（且不超过显卡总容量的 25%）；原 EM 诊断保护器保持不动。
根据轨迹及传播时延选择采样窗，不能只覆盖初级粒子时间。

## 物理模型与数值控制

- CoREAS 使用共同中点光路的成对端点电场矩及近切伦科夫辅助矩，详见端点模块说明。
- ZHS 使用有限轨迹的连续检测时间区间矢势面积及单元矩，最终
  `E(f) = -i 2πf A(f)`。切伦科夫分母接近零时仍累加有限面积，不除以小数。
- 单次透射求解有限面上的 Snell 路径，应用 s/p Fresnel 系数、功率通量归一化、
  原 mountain 的射线管 Jacobian 和振幅衰减。共面共享边使用唯一面归属。
- 求解前用源和观测点在面上的投影作保守候选筛选：正折射率的平面 Snell 解必在
  两个投影之间，整段落在三角形某一半平面之外时可直接排除该面。筛选保留边容差和
  浮点余量；设备测试逐面对照未筛选求解，检查状态、折射点、时延和传递矩阵一致。
  共面共享边的唯一归属通过已有 BVH 查询折射点附近的面，包围盒扩展覆盖边容差及
  浮点余量，避免每条有效光路都重新遍历整座山体。
- 直达和透射的两段光路分别检查完整 BVH，忽略光学端点附近的交点，保留中途遮挡。
  直达轨迹在三角形阴影边界处切分；透射轨迹即使中点没有合法路径也按
  `mesh_maximum_segment_m` 细分。后者是空间积分分辨率，细于该分辨率的透射可见性变化
  需要继续收紧参数并检查收敛。
- `fraunhofer_limit` 同时限制 `L/R` 和相位曲率项
  `2π f L² (1-cos²θ)/(c R)`。因此轴向长轨迹也会细分。
- 折射率表沿直线光路积分。变折射率时用区间两端的光学时延计算检测时间宽度，
  不把均匀介质的局部导数当作一般变折射率的精确导数。

此模块是**直线光路、至多一次透射的几何光学模型**。它不包含弯曲大气射线、
反射、绕射、多次穿越、频散、天线和电子学响应。CoREAS 和 ZHS 使用同一组物理近似。
同介质直线穿过山体时，该直达支路被遮挡，不自动增加两次透射支路。
多面体棱边附近可能没有合法单面折射解，不能把“无此 GO 支路”解释为包含绕射的完整场。

不能通过缩短采样窗静默丢弃信号。越窗、非法轨迹、非有限数、光路求解失败或细分达到
上限都使输出明确失败；只有成功最终下载后才产生完整射电结果。
设备错误码 1–6 分别表示非有限累加、越出采样窗、非法轨迹、缺少界面、光路无效或
未收敛、细分达到上限。射电源的速度检查允许浮点数值误差，不接受超光速物理轨迹。

## 输出与复现

`output/radio/CoREAS/` 和 `output/radio/ZHS/` 分别保存：

- `config.json`：光学介质、完整折射率表、观测点及采样/积分配置；
- `moments.bin`：原始双精度数组，布局为 `(阶数, 观测点×2介质×3分量, 时间单元)`；
- `field.csv`：绝对时间、总 Ex/Ey/Ez，以及外围/嵌入来源分别贡献；
- `spectrum.csv`：包含绝对时间原点相位的总复频谱。

CoREAS 目录另有 `regularized_moments.bin`。CoREAS 主矩单位为 V·s/m，
辅助矩及 ZHS 原始矩单位为 V·s²/m；两算法电场单位均为 V/m，频谱单位为 V·s/m。
时间波形的 Nyquist 项投影为实数；连续复频谱保留原值。没有做幅度拟合、峰值对齐或带通。
已存在的射电输出目录不会被覆盖。`terrain_run.yaml` 的 `radio_result` 保存去重后的源计数、
共享批次数（`wavefronts`）、两个内核的总启动次数（`radio_kernel_launches`）、联合设备内存和最终化次数。
`CoREAS`、`ZHS` 子项分别保存各算法的源数、直达/透射/遮挡路径数、批次数及输出目录。
顶层 `moment_arrays=3`、`downloads=1`。CUDA 的射电批次数可以少于输运波前数。

所有编译和运行均在 PSR 进行，本地仅编辑和读取产物。验证脚本在 `validation/terrain/`：
`analyze_interface_radio_waveforms.py` 使用独立沿轨迹积分核对绝对复频谱，
`check_interface_radio_dem_visibility.py` 独立核对真实 DEM 的遮挡，
`run_interface_radio_pair_acceptance.py` 核对默认双算、冻结输运输出、带电源计数及原有单算法参考结果，
`analyze_interface_radio_showers.py` 输出真实级联诊断图，
`analyze_interface_radio_pair_residency.py` 检查默认双算的 CUPTI 实际 GPU kernel 与缓冲区复制。
`validate_interface_radio_atmosphere.py` 用独立密度公式验证共同海平面归一化，
`validate_interface_radio_optical.py` 用独立高分辨率积分验证径向光程。
`radio_consumer/` 仅链接导出的 `CORSIKA8::InterfaceRadio`，已在 PSR 分别运行 OpenMP 和
CUDA 实例，验证通用模块不依赖空气应用或输运库。

完整验收数据和诊断图目录：
`/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-interface-resident-20260911/interface-radio/`。
目录保留了早期失败及被排除的旧对象构建记录；应以最终验收清单列出的运行版本为准。
