# 山体材料模型

`c8_terrain_cascade` 的 shower 与射电共同读取一个材料模型。基准按你提供的
[materials.yaml](materials.yaml) 中二氧化硅基准（`S0_SILICA_TRANSPORT_PROXY`）定义；默认折射率为 √5。
原始 [material.md](material.md) 和 [materials.yaml](materials.yaml) 保留为参数来源，
它们本身不是运行卡；其中推荐花岗岩的建议不会改变你指定的 SiO₂ 默认值。

| 选择名 | 模型 | 密度 kg/m³ | 折射率 n | 电场衰减长度 m |
|---|---|---:|---:|---:|
| `SiO2` / `S0_SILICA_TRANSPORT_PROXY` | SiO₂，O:Si 核数比 2:1 | 2650 | √5 | 100 |
| `Limestone` / `Calcite` / `L0_LIMESTONE_CALCITE_PROXY` | 致密 CaCO₃，C:O:Ca=1:3:1 | 2710 | √6 | 14.47648 |
| `Granite` / `G0_GRANITE_REFERENCE` | 花岗岩参考混合物：H、C、O、Na、Mg、Al、Si、K、Ca、Fe | 2729 | √5 | 86.85890 |

这里的“石灰岩”是致密方解石代理，“花岗岩”是指定混合物；不存在适用于所有
碳酸盐或硅酸盐的唯一参数。单一代表同位素、均匀密度、各向同性、μᵣ=1 是当前假设。
折射率和衰减长度是文件给定的研究常数，不是 21CMA 山体实测值。

材料名称和图例写明成分，S0/L0/G0 仅是来源文件的兼容编号。
花岗岩没有单一化学式，当前采用以下核数分数，不能把这张表当作氧化物质量分数：

| 核种 | 核数分数 | 核种 | 核数分数 |
|---|---:|---|---:|
| H-1 | 0.027122 | Al-27 | 0.062783 |
| C-12 | 0.000502 | Si-28 | 0.205927 |
| O-16 | 0.607735 | K-39 | 0.013938 |
| Na-23 | 0.025866 | Ca-40 | 0.018960 |
| Mg-24 | 0.018081 | Fe-56 | 0.019086 |

## 怎么选择

在已有完整场景中，仅替换材料部分，地形、坐标、天线和 shower 命令继续使用原配置：

```yaml
geometry:
  # 保留 kind、mesh_path、boundary_padding_m、rock_reference_enu_m 等几何项
  material:
    preset: Calcite  # Limestone / dense calcite (CaCO3)
```

或者使用独立文件，路径相对于场景文件：

```yaml
geometry:
  material_file: /path/to/configs/mountain/materials/limestone.yaml
```

现成的 [SiO₂](../configs/mountain/materials/silica.yaml)、
[石灰岩](../configs/mountain/materials/limestone.yaml)、
[花岗岩](../configs/mountain/materials/granite.yaml) 和
[参数变化示例](../configs/mountain/materials/custom_example.yaml) 可直接选用。
`material` 与 `material_file` 二选一。射电仍按原方式启用，默认一起计算 CoREAS 与 ZHS。

**迁移旧场景时，删除 `geometry.rock_density_g_cm3`、`geometry.rock_refractive_index`、
`geometry.attenuation_length_m` 和 `radio.rock_attenuation_length_m`，改用材料卡。**
尤其旧的 `n=2` 不会因为文件写着 SiO₂ 就自动变成 √5。
旧标量材料名仍兼容显式密度/n/射电衰减覆盖，并在输出标记 `legacy_material_overrides`；
新材料映射或材料文件与旧字段冲突时直接报错。旧几何字段 `attenuation_length_m`
从来不是本程序有效的射电输入，现在明确拒绝，避免误以为它已生效。
地形准备程序生成的新版 native 场景已改为选择 `SiO2`，不再带入旧 n=2。

## 怎么增加模型

材料可以继承预设，只覆盖密度或射电常数。改变组分则必须同时指定电离模型：

```yaml
preset: Calcite
id: MY_CALCITE_MODEL
description: Custom calcite (CaCO3) model
provenance: Replace with measurement or model provenance
transport:
  density_kg_m3: 2710
  composition_basis: nucleus_number_fraction
  nuclei:
    - {element: C, Z: 6, A: 12, number_fraction: 0.2}
    - {element: O, Z: 8, A: 16, number_fraction: 0.6}
    - {element: Ca, Z: 20, A: 40, number_fraction: 0.2}
  ionisation:
    model: nist_density_scaled
    reference_medium: Calcite
radio:
  refractive_index: 2.449489742783178
  field_attenuation_length_m: 14.476482730108396
magnetic_field_enu_T: [0, 0, 0]
```

核数分数必须归一化，不能把质量分数直接填进去。支持不超过 16 个核种；
水和冰的旧选择名也保留为兼容入口，研究其射电时需补充适用频段的物理参数。
任意新配方可选 `bragg_sternheimer`，或提供 `explicit`
电离参数：`I_eV, Cbar, x0, x1, a, m, delta0, reference_density_kg_m3`。
`Cbar` 取正号约定；PROPOSAL 构造时按原接口转成 `C=-Cbar`。
显式参数的参考密度必须等于选定密度。材料卡未知字段、错误组分和暂不支持的频散选项会报错。

## 这些参数实际接到了哪里

| 参数 | shower | 射电 |
|---|---|---|
| 核种与核数分数 | 核靶集合、QGSJet、PROPOSAL 元素过程、散射/LPM；FLUKA 入口按 Z 映射天然元素靶 | 通过实际 shower 轨迹改变源 |
| 密度 | 克深、相互作用距离、连续能损、CPU/设备环境及 LPM 局部密度 | 不再通过空气关系式猜测岩石 n |
| 平均激发能与密度效应 | CPU PROPOSAL 与其导出的 Kokkos 表；山体强子连续能损 | 通过轨迹改变源 |
| n | 环境模型中的折射率 | 源介质、Cherenkov 相位、Snell/Fresnel、光程时延 |
| L_E | 不改变粒子输运 | 沿每段介质路径乘 `exp(-l/L_E)`，是电场幅度，不是功率长度 |
| B_ENU | 山体 CPU 跟踪与设备环境，默认零场 | 通过带电粒子轨迹改变辐射 |

SiO₂ 和 CaCO₃ 采用仓库 NIST 化合物的 I、Sternheimer 参数，按选定密度移动密度效应曲线。
花岗岩采用电子数加权 `ln I = Σ(xᵢZᵢ ln Iᵢ)/Σ(xᵢZᵢ)`，再构建凝聚态 Sternheimer 近似，
不是借用二氧化硅或 standard rock 的电离参数。这是一项有明确来源的近似，不能替代矿物样品测量。
密度效应近似的公式与适用定义参见 [Geant4 Physics Reference Manual](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/electron_incident/ionisation/eion.html)。

检查发现旧 `BetheBlochPDG` 写死了氮气参数。新增的山体 `MaterialHadronLoss` 使用材料的
I、Z/A 和密度效应计算带电强子的主要碰撞能损，并去掉原模块经验性的空气 `bE` 项。
这是主要 Bethe 项近似，不宣称覆盖低能壳层、Barkas 或重离子有效电荷修正；
强相互作用仍由 FLUKA/QGSJet 处理，μ/τ 能损仍交给 PROPOSAL。
本次没有改动原空气应用、空气过程类和空气射电模块。

花岗岩加大气共有 12 种元素。实际模拟发现原 FLUKA 初始化入口的区域数组上限为 10，
不能直接给每个元素分配一个区域。新增的山体适配器用一个辅助混合物初始化所有元素，
随后取出其纯元素靶编号；shower 的靶抽样仍由 CORSIKA 按真实材料组分完成，
**辅助混合物的权重不参与 shower**。不超过 10 个靶时继续调用原入口。
该适配器单独编译，使用已安装 FLUKA 的匹配头文件；原空气 FLUKA 包装和外部库均未修改。
默认从 `$FLUPRO/flukapro` 读取 `(DIMPAR)`、`(FLKMAT)`、`(FLKCMP)`；
也可用 CMake 的 `C8_TERRAIN_FLUKA_INCLUDE_DIR` 指定匹配版本的目录。

还需区分核种与元素：原 FLUKA `STPXYZ` 接口只接收 Z，不接收 A，因此其低能强子过程
使用天然同位素元素靶。材料卡指定的代表 A 用于 PROPOSAL、质量／核靶等 CORSIKA 输入，
不能据此宣称 FLUKA 也执行了同样的纯同位素计算。新入口保留这个已明确记录的近似。

每次输出 `terrain_run.yaml` 中的 `resolved_material`、`proposal_materials`、
`material_tables` 可核对输入、实际 CPU 计算器和设备表。
`resolved_material` 是完整参数快照，可单独保存为材料文件重放。
其中 `description` 给出材料名称，`transport.nuclei` 逐项给出元素符号、Z、A 和核数分数；
组分的数值定义以 Z、A 和 `number_fraction` 为准，`element` 是便于阅读的标注。
组分或输运参数变化会更换表/辅助数据指纹；只变 n 或 L_E 不更换电磁输运表。

C++ 接口位于 [MaterialModel.hpp](../corsika/modules/transport/MaterialModel.hpp)：
`materialPreset()` 选择模型，`makeModel<Interface>()` 构造环境属性。
使用 `interfaces::transportData(properties)` 读取完整输运参数；
旧 `getMedium()` 枚举仅用于兼容，无法表达任意混合物。
构造原有 PROPOSAL 过程时使用
[MaterialProposalEnvironment](../corsika/modules/transport/MaterialProposalEnvironment.hpp) 视图，
由它将完整材料数据交给原计算器；山体应用已经完成这一接线。

当前山体应用仍是一块均匀材料的 DEM 加外部大气；本接口支持**逐次选择不同材料**，
尚不把一个 DEM 内部分割成多个矿层。灰色衰减、直线光路/Snell 接口的原有传播范围也没有扩大。
密度或组分变化要重新跑 shower；n/L_E 变化要重新积累每条轨迹的射电，不能只缩放总波形。
SiO₂ 基准的 n 和输运密度效应、山体强子能损均已更新，先前 double-bang 图属于旧版本结果。

## 验证记录

测试和算例仅在 PSR 上执行，OpenMP 使用 256 线程；CUDA 使用服务器 GPU。
脚本位于 [validation/terrain/material_models](../validation/terrain/material_models)。
OpenMP、CUDA 各 10 项 C++ 检查通过，Python 配置／地形检查 41 项通过。
四份独立材料文件通过实际场景加载；旧 n=2 与新材料卡冲突时按预期拒绝。
FLUKA 扩展入口与原入口分别在独立进程初始化：10 种元素、4 种入射强子、
3 个动能共 120 个截面点，以及同种子下的 3 个质子—氧核碰撞事件，输出逐行一致。
加入该适配器后，两种后端重新编译并通过材料／环境检查，输运和射电内核代码未变。

原生 PROPOSAL 与可移植 LPM 实现共比较 1,800 个点：SiO₂ 240 点、CaCO₃ 360 点、
花岗岩十元素混合物 1,200 点；涵盖光子成对与电子轫致辐射、各组分、
1 GeV–100 PeV、五个能量分数和三个密度比例。最大相对差约 `3.9e-16`。
这是原生计算器与可移植实现的主机数值对照，CUDA 实际执行另外由运行跟踪检查。

最终 19 个真实输运算例全部完成：三种材料各在 OpenMP／CUDA 下进行
10 GeV 电子射电开／关对照（12 个），100 GeV 质子（6 个），另加 CaCO₃ 的纯 CPU
PROPOSAL 电子对照（1 个）。每组射电开／关的输运 CSV 完全一致，材料错配为零，
能量账最大未解释相对残差约 `7.8e-16`。两种后端导出的材料表和辅助表指纹相同。

| 材料 | 原生／可移植 LPM 最大相对差 | CUDA CoREAS／ZHS 频谱相对 L2 差 |
|---|---:|---:|
| 二氧化硅 SiO₂ | 3.87e-16 | 6.11e-9 |
| 石灰岩／方解石 CaCO₃ | 2.99e-16 | 1.13e-8 |
| 花岗岩十元素混合物 | 2.81e-16 | 3.67e-9 |

GPU 跟踪记录中，三种材料分别有 1034、606、1220 次输运内核执行，
均有真实的 CoREAS 和 ZHS GPU 内核，常驻粒子队列分配在 CUDA 内存中；
三份射电累加数组各在结束时回传一次。控制状态和诊断记录仍有回传，不能称为“零传输”。

这些算例使用 1100 ns 输运时间窗，终止时仍存活的粒子计入能量账；
它们验证材料接线、数值一致性与设备执行，**不等于完整 shower 收敛、矿物样品验证，
也不是新的 100 PeV double-bang 结果**。图中各材料只有一个电子事件，不能用于统计幅度排序。

图和简短读图说明见 [PPT 格式 Markdown](reports/material_models_20260913/SLIDES_CN.md)。
最终模拟清单与数值结果见 [validation.json](reports/material_models_20260913/validation.json)，
原生对照数据和测试日志保存在同目录的 `evidence/`。
完整模拟、GPU 跟踪和独立构建保留在 PSR：
`/data/yhlu/CorsikaData/corsika_validation_results/beta5_material_models_20260913`。
