# beta5 山体中微子独立应用

完整 21CMA 场景的当前缺项、真实 CPU/OpenMP/CUDA 测试以及与原 mountain 的差别，见
[2026-09-08 逐项复核](mountain_21cma_integration_audit_CN.md)。当前不是 DEM 嵌入标准大气的完整应用。

真实非凸 DEM、80 站地理坐标及五层大气的新增迁移和独立测试入口见
[真实山体迁移与构建](terrain_native_scene_build_CN.md)；不要用本页的有限凸体配置替代真实场景。

`c8_mountain_neutrino` 是独立的有限山体示例：在均匀 SiO₂ 凸山体中产生或传播
中微子 CC 事例，随后由标量 CPU 或 beta5 Kokkos 输运级联，并可计算山体内部
observer 的 CoREAS/ZHS 信号。它不修改 `c8_air_shower` 的参数、模型装配和大气模块。

**这是限定物理范围的集成示例，不是完整的地球/山体中微子探测率程序。**
下面列明适用范围；功能接线完成不能替代跨后端的大样本验收。

## 1. 计算链路和范围

```text
YAML：凸山体、均匀 SiO₂、初级、cut/thinning、内部 observer
                         │
                几何/物理范围预检
                         │
       ┌─────────────────┴─────────────────┐
       │ natural_cc                        │ forced_vertex_cc
       │ 在岩石弦上抽取 CC 距离            │ 在指定内点强制一次 CC
       └─────────────────┬─────────────────┘
                         │
           CPU：CTW2011 CC 率 + Pythia 末态
                         │
           ┌─────────────┴────────────────────┐
           │ CPU 强子/衰变及不支持的末态      │ EM/支持的轻子
           │ FLUKA + QGSJet-II.04              │ proposal 或 Kokkos
           └─────────────┬────────────────────┘
                         │
          内部轨迹 → profile / 沉积 / CoREAS / ZHS
                         │
                到达山体边界 → escaped
```

- 几何支持 box、四面体、**闭合凸**三角 OBJ；设备快照最多 32 个三角面平面记录。
  非凸 DEM、带洞或多山体拼接会拒绝，不取凸包代替原地形。
- 当前介质是均匀 SiO₂：Si-28:O-16 原子数比 1:2，示例密度 `2.2 g/cm³`。
  折射率和密度可配置，但没有深度分层、磁场或光学色散。
- 首版应用接受 **νe/反νe**，中微子初级能区限制为 **`1e4–1e12 GeV`**。
  其他味初级明确拒绝，须先补充相应带电轻子衰变/再生验收。截面使用 CTW2011 的等比例质子/中子靶
  CC 参数化；核靶按自由静止核子冲量近似处理。核子类型按靶核 Z/A 抽取。
- **不包含 NC、核屏蔽、完整 τ 再生或入射前穿越地球的传播。** Pythia 末态使用
  `PhaseSpace:Q2Min = 25 GeV²`；低 Q² 末态未验收。这是总 CC 截面与受限末态模型的
  明确限制，不能据此声称完整中微子动力学精度。
- `forced_vertex_cc` 是“已在此处发生 CC 反应”的条件事例；不输出通量归一化权重、
  山体相互作用概率权重或探测有效面积。
- 山体外是**吸收/逃逸边界**：粒子出山后记录，不继续跑空气 shower。逃逸能量不是
  沉积能量；当前总能量账本尚不宣称完成核静质量交换的完整闭合认证。
- 射电仅支持**内部 observer、均匀介质直达传播**，延迟按 `nR/c` 计算；不包含
  山体表面折射、反射、衰减或山外天线。不能把此输出视为 21CMA 已接收到的信号。

## 2. 编译

先按[项目中文 README](../README_CN.md)准备 Conda、编译器、授权 FLUKA 和对应的
Kokkos 工具链。新应用与 beta5 共用这些依赖，不需要另一个旧 mountain 项目。

以下在 `corsika8_kokkos_beta5/` **源码目录**执行。选一种后端，不必全部编译：

```bash
# 无 GPU：独立 OpenMP
C8_BUILD_JOBS=2 bash tools/build_kokkos.sh openmp \
  -DWITH_FLUKA=ON -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON

# NVIDIA：独立 Kokkos CUDA
C8_BUILD_JOBS=2 bash tools/build_kokkos.sh cuda \
  -DWITH_FLUKA=ON -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON
```

应用链接 FLUKA；未启用 `WITH_FLUKA` 的配置会跳过此目标，不会静默改成别的低能
强子模型。本示例高能模型为 QGSJet-II.04，**不是空气应用常用的 SIBYLL 配置**。

安装后分别得到：

```text
install/openmp/bin/c8_mountain_neutrino
install/cuda/bin/c8_mountain_neutrino
install/<backend>/share/corsika/mountain/*.yaml
```

山体应用当前直接调用各后端程序；不要将空气程序统一入口的 `--backend` 参数复制过来。
组合 CUDA/OpenMP 构建通过 `--kokkos-execution` 选实例，仍遵守组合程序对可用 NVIDIA
runtime 的既有限制。HIP/SYCL 的源码分支需要目标设备编译和验收，本指南不声称已验证。

## 3. 从快速示例开始

下面仍在源码目录运行，`../install` 指项目容器目录下的安装。每次用不同输出目录，
程序不会覆盖已有结果。CLI seed 默认 `67101`；示例命令显式写出便于复现。

### 3.1 只查几何，不初始化物理或加速器

```bash
../install/openmp/bin/c8_mountain_neutrino \
  --config configs/mountain/photon_1gev_box.yaml --geometry-only
```

成功时打印配置和岩石弦长。这个模式不运行 shower、不创建模拟输出，也不表示
所请求的加速器 runtime 已验证可用。

### 3.2 1 GeV 光子：CPU 与 Kokkos 的快速入口

[photon_1gev_box.yaml](../configs/mountain/photon_1gev_box.yaml) 使用 20 m 边长的
SiO₂ box，初级位置 `(0,0,-5) m`、方向 `+z`，observer 在 `(3,0,0) m`，不在初级轴线上。

```bash
# 标量 CPU PROPOSAL；使用独立 OpenMP 程序，但不初始化 Kokkos 后端
../install/openmp/bin/c8_mountain_neutrino \
  --config configs/mountain/photon_1gev_box.yaml \
  --em-backend proposal -s 67101 -f mountain_photon_cpu

# 多核 CPU 同时承担 Kokkos EM 与内部射电；不访问 GPU
../install/openmp/bin/c8_mountain_neutrino \
  --config configs/mountain/photon_1gev_box.yaml \
  --em-backend kokkos --kokkos-execution openmp --kokkos-num-threads 4 \
  -s 67101 -f mountain_photon_openmp

# NVIDIA GPU 同时承担 Kokkos EM 与内部射电，CPU 单线程调度
../install/cuda/bin/c8_mountain_neutrino \
  --config configs/mountain/photon_1gev_box.yaml \
  --em-backend kokkos --kokkos-execution cuda \
  -s 67101 -f mountain_photon_cuda
```

小能量示例用于检查通路，不用于判断 GPU 极限加速比。同 seed 不意味着标量 CPU 与
Kokkos 必须生成逐点相同的独立 shower；应结合逐过程 oracle 和统计样本比较。

### 3.3 强制 CC 顶点：保证本例确有 shower

```bash
../install/cuda/bin/c8_mountain_neutrino \
  --config configs/mountain/forced_nue_cc_box.yaml \
  --em-backend kokkos --kokkos-execution cuda \
  -s 67101 -f mountain_forced_nue_cuda
```

[forced_nue_cc_box.yaml](../configs/mountain/forced_nue_cc_box.yaml) 为 10 TeV νe，在
`(0,0,-5) m` 条件强制 CC；Pythia 生成末态，之后才进行真实输运。它不是把一个 electron
手工当作中微子全部末态。CPU 强子、衰变与指定过程 fallback 仍然保留。

### 3.4 自然中微子通过四面体

```bash
../install/openmp/bin/c8_mountain_neutrino \
  --config configs/mountain/natural_nue_cc_tetrahedron.yaml \
  --em-backend proposal -s 67101 -f mountain_natural_nue_cpu
```

[natural_nue_cc_tetrahedron.yaml](../configs/mountain/natural_nue_cc_tetrahedron.yaml)
从山体外沿 `+z` 入射：入口之前只计算几何飞行时间，不模拟空气。程序在真实岩石弦内
按 CC 率抽取相互作用；10 TeV 中微子穿过几十米岩石，**大多数单次模拟应没有 CC 反应**，
只记录中微子逃逸，射电信号为零也可能完全正常。不要删除这些事例只保留“有信号”样本。

## 4. 参数和 YAML 单位

CLI 的后端选择覆盖 EM 和射电，不能混用 OpenMP EM 与 GPU radio：

| CLI | 默认值 / 含义 |
|---|---|
| `--config FILE` | 必填，下面的 YAML |
| `-f, --output DIR` | `mountain_output`；必须是新目录 |
| `-s, --seed` | `67101` |
| `--em-backend proposal\|kokkos` | `proposal` |
| `--kokkos-execution` | 空：使用编译后端；可显式选 `openmp/cuda/hip/sycl` |
| `--kokkos-num-threads` | `0`：runtime 决定；GPU 模式禁止大于 1 |
| `--kokkos-device` | `0` |
| `--gpu-min-batch` | `4096` |
| `--gpu-resident-batch-limit` | `0`：使用 beta5 的内存预算策略 |
| `--gpu-memory-fraction` | `0.7`，是上限，不要求显存一直占满 |
| `--gpu-aux-cache-dir` | 可选的可写辅助缓存位置 |
| `--geometry-only` | 配置/几何检查，不运行物理 |

YAML 中坐标、box 半边长和 OBJ `scale_m` 用**米**；方向会归一化。`energy_GeV` 是
初级**总能量**，`emcut_GeV/hadcut_GeV/mucut_GeV/taucut_GeV` 是输运动能 cut。
默认 EM cut 为 0.5 MeV，其他 cut 为 0.3 GeV；`emthin` 默认 `1e-6`，`max_weight: 0`
表示沿用 `0.5 × emthin × Eprimary[GeV]` 的自动值，不等于显式最大权重 100。

射电 `start_ns/duration_ns` 单位为 ns，`sample_rate_GHz` 是 GHz。示例设置
`[-10,3990) ns`、1 GHz，留出相对于米级传播较宽的窗口。运行后仍需确认首尾没有截断；
加宽时间窗属于显式配置，不通过删除端点或平滑来隐藏截断。所有 observer 必须在山体内。

每次进程仅跑一个 shower，目前没有 `-N` 批量参数。批量任务应逐事件使用新目录和明确
seed，另保存二进制/配置哈希；不要把不同版本的结果直接当成同版本系综。

## 5. 原生 PROPOSAL 缓存

Kokkos 路径固定使用 `proposal-native`。程序直接从本事件的介质和 calculator 导出
PROPOSAL 原生样条，不要求手工 `.c8emrt` 文件，也不需要先跑完整的介质制表工具。

**首次**遇到当前 SiO₂、密度、cut 和依赖配置时，PROPOSAL 自身缓存及必要的辅助数据
可能自动生成，因此冷启动可能明显较慢。之后兼容缓存可复用；这不是“任何介质永远
不需要计算新表”。辅助缓存默认由 beta5 管理，可用 `--gpu-aux-cache-dir PATH` 指定。
缓存损坏、版本或覆盖范围不匹配等门禁仍有效，不能用更换表路径或旧数据静默绕过。

## 6. 输出与验收

以实际输出为准，主要文件为：

- `mountain_run.yaml`：配置、弦长、后端、时间、CC 顶点审计及最终 `complete`。
- `mountain/profile.csv`：沿初级方向的深度（m）、沉积能量与各 PDG 加权轨迹穿越数。
- `mountain/escaped_particles.csv`：出山粒子的能量、权重、位置、时间和方向。
- `CoREAS/`、`ZHS/`：现有 observer/writer 生成的内部时域波形。

只有 `complete: true` 且必需输出可完整读取，才算该次运行完成；它本身不是物理
统计一致性声明。局部测试包含凸几何解析 oracle、中微子率/目标/真实 Pythia 末态、
CPU/Kokkos 均匀介质射电。大样本山体 shower 的 profile、能量账本和波形验收需另做。

几何 API 详见[有限凸山体几何接口](mountain_geometry_CN.md)。
