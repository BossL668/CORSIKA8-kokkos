# 山体跨介质 CoREAS 端点模块

这是独立 `CORSIKA8::InterfaceRadio` 中的端点算法，入口是
`corsika/modules/radio/interface/CoreasEndpoints.hpp`。原空气 `CoREAS.hpp`、
`CoREAS.inl` 和 `accelerator/radio/` 保持不变。山体应用启用射电后默认同时计算 CoREAS 和 ZHS。

## 使用

山体应用使用 `--em-backend kokkos --em-scheduler resident --radio`，
或者在场景中配置：

```yaml
radio:
  enabled: true
  coreas_cherenkov_threshold: 0.001
```

没有算法选择开关。旧的 `--radio-algorithm` 和 `radio.algorithm` 配置需要移除。
两套算法使用同一批轨迹、观测点及光学配置，分别写入 `radio/CoREAS/` 和 `radio/ZHS/`。

使用 `C8_INTERFACE_EXECUTION_SPACE=CUDA` 或 `OPENMP` 的构建。
光学介质、观测点、采样窗和内存配置见 [跨介质射电说明](interface_kokkos_radio_CN.md)。
CoREAS 的 `moment_order` 至少为 1，默认 12；更高阶数降低分数采样的截断误差。

## 与 ZHS 一致的物理近似

同一组轨迹、介质、传播路径和精度条件下，CoREAS 与 ZHS 应在受控数值误差内一致。
这里采用 `mountain/cpp/c8_mountain_corsika.cpp` 中 `MountainCoREAS` 的共同中点远场光路方法。
如果只保留辐射项，却分别使用两个端点的距离、方向和传递振幅，缺少的近场项会使低频不相消；
这种直接迁移空气端点公式的做法不适用于本模块的 ZHS 远场电流模型。

每个已接受的子段使用同一个中点光路、偏振传递矩阵和到达时间线性化。
令 `A_area=C*T(beta_perp)*dt`，`C=q*weight/(4*pi*epsilon0*c)`，
`Delta_tau` 是带符号的到达时间跨度，则 CoREAS 累积

`J_start=-A_area/Delta_tau`，`J_stop=+A_area/Delta_tau`，

对应时间为 `tau_mid-Delta_tau/2` 和 `tau_mid+Delta_tau/2`。
均匀介质使用 `Delta_tau=(1-n_source*beta·k_emit)*dt`；径向折射率使用与 ZHS 相同的
光学时间差。负 Doppler 的时间顺序和端点符号均保留。

采用 `exp(-i*omega*t)` 约定，端点对的频谱可直接写成
`E(omega)=-i*omega*A_area*exp(-i*omega*tau_mid)*sinc(omega*Delta_tau/2)`，
其中 `sinc(x)=sin(x)/x`。这正是 ZHS 有限区间矢势求导后的结果；
当 `Delta_tau` 趋近零时，`sinc` 趋近 1，物理贡献仍有限。
两种设备实现分别累积电场端点矩和矢势区间矩，其差别应随矩阶数增加而收敛。

几何细分、可见区间、有限三角面、Fresnel 偏振传递、射线管扩散和衰减与 ZHS 一致。
穿界轨迹先按输运介质切分，各段采用自己的介质响应。磁场已经作用在输入的输运轨迹中，
射电模块不会再次推进粒子。

普通 CoREAS 是两项独立端点冲量的计算。若两端落在同一采样格，直接计算多项式差商，
避免大数相减，并使零阶矩在每对端点内严格相消；不人为移动到达时间。
接近零到达时间跨度时，采用有限电流极限并存入辅助矩数组，不丢弃切伦科夫方向的贡献。
`endpoint_contributions` 统计普通端点；`regularized_pairs` 统计用到稳定极限的轨迹—观测点对。
`boundary_endpoints` 记录透射子段中位于界面上的原始轨迹端点。

理论依据：[James 等，端点形式](https://arxiv.org/abs/1007.4146)，
[Huege 等，CoREAS 近切伦科夫处理](https://arxiv.org/abs/1301.2132)。

## 设备常驻与输出

CUDA 带电输运轨迹通过设备内复制进入 8192 条容量的射电队列；CPU 产生的带电来源按批上传。
CoREAS 端点电场矩、有限电流辅助矩及 ZHS 矢势矩均在 GPU 上累计，结束时各下载一次，
最终 FFT 和文件输出在 CPU。两算法共用源队列、CPU 上传暂存区和光学数据，每批分别启动两个内核。
顶层 `downloads=1` 表示一次最终化，`moment_arrays=3` 表示总共三份数组；
`radio_result.CoREAS.moment_arrays=2`，`radio_result.ZHS.moment_arrays=1`。
联合内存预算在分配前检查，不能只按一套算法估算。

`radio/CoREAS/moments.bin` 存储电场冲量矩，单位 V·s/m；同目录的 `regularized_moments.bin` 存储辅助矢势矩，
单位 V·s²/m。合成关系为
`E(f)=FFT(endpoint_moments)-i*2*pi*f*FFT(regularized_moments)`。
只对辅助矢势部分求导一次。频谱保留绝对时间相位。
零频相消由成对源实现，并单独验证，不靠输出时强制清零掩盖差异。

## 范围与验收

传播是直线光路、直达或一次透射的几何光学模型；没有反射、绕射、倏逝波、多次穿界、
弯曲大气射线或天线响应，不能称为完整 Ginzburg–Frank 过渡辐射。

2026-09-12 的 c4 物理验收在两个算法分别运行的版本上完成，全部使用 PSR 的 256 个 CPU 核
和 T400 GPU；本地没有构建或测试。以下数据用于物理实现的历史对照：

- 72 次固定轨迹运行：18 类场景 × 两后端 × 两算法。CoREAS/ZHS 复数频谱最大相对 L2
  为 `2.546e-12`；相对独立电流积分最大 L2 为 `4.618e-4`。零频相消通过。
- 16 组完整簇射：默认 12 阶矩下，CoREAS/ZHS 最大频谱 L2 为 `1.434e-8`，
  最大波形 L2 为 `1.433e-8`。轨迹、光学配置和采样条件一致，没有拟合幅度或平移时间。
- P8/P12/P16/P20 收敛检查通过。原空气 CoREAS 的 18 组对照在均匀空气退化条件下进行，
  并分别核对原有采样格误差及有限距离近似误差，未把空气内核用于穿界。
- 单算法 ZHS、光路和输运回归通过。CUPTI 确认 CoREAS 两份矩数组各在最后下载一次，设备轨迹以 D2D
  进入射电队列；中微子案例的 28,954 条 CPU 带电来源上传量恰为 `3,011,216` 字节。
- PSR 隔离源码中 244 个受保护空气文件的 SHA256 与任务前一致。

初版 c2 分别求两个物理端点光路，虽然通过自身公式参考，却未通过本模块 CoREAS/ZHS
低频一致性检查，已被上述共同光路实现替换，不能作为已验收版本使用。
完整随机簇射在各后端内使用相同轨迹比较；两后端的直接数值一致性由固定轨迹验证。

原始命令、二进制哈希、对照图与完整 CUPTI 记录见
[c4 验收报告](../../build/interface-coreas-validation-20260912/results-c4/README.md)。

默认双算的队列回归由 `coreas_consumer/paired.cpp` 检查：设备满批及尾批、CPU 来源交错、
联合预算、重复最终化保护，并逐数组对照各算法的独立实例。
`run_interface_radio_pair_acceptance.py` 对照既有完整簇射参考；
`analyze_interface_radio_pair_residency.py` 检查两个 CUDA 内核、共用上传缓冲和三份矩数组的最终下载。
本次记录见 [默认双算验收报告](../../build/interface-radio-pair-validation-20260912/results-p1/README.md)。
