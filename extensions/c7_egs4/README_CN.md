# C7 EGS4 C++ / Kokkos 电磁扩展

这是 C7 电磁算法的原生 C++ 实验实现，不是对 C7 Fortran 电磁程序的调用包装。电子、正电子和光子由本模块推进；强子模型与 μ 子物理由 C8 现有模块提供。默认 PROPOSAL 后端不变。

现在可以在原主应用 `c8_air_shower` 中选用 EGS4。旧的独立入口 `c8_egs4_shower` 和 `egs4_test_schedule` 保留供历史测试、正在运行的任务复现使用，不是新接口所需的额外程序。Python 只用于测试和参考数据生成，运行和多卡调度不需要 Python。

## 主应用统一接口

在原有构建配置中启用以下选项，然后照常编译 `c8_air_shower`：

```bash
-DCORSIKA_ENABLE_EGS4=ON
```

构建默认使用项目自带的 `resources/c7_egs4/EGSDAT6_.4`，不再依赖 C7 安装路径，也不需要手动指定表路径。表内容在**构建时**内嵌到二进制，运行时无需 `--table` 或外部表文件。输出 `native_egs4/config.yaml` 记录表的 SHA256，方便核对；生成的含表源码仍留在构建目录，不提交到仓库。

项目内的表逐字节复制自 CORSIKA 7.8050 的 `run/EGSDAT6_.4`（AIR-NTP），SHA256 为 `148c56f0f6397faf0f4e23d9a20b2d754f73bae02d644d1c3506153e2e8c990d`。来源说明及上游 `COPYING` 副本位于 `resources/c7_egs4/`。保留 `CORSIKA_EGS4_TABLE` 作为高级 CMake 覆盖选项；如果旧构建缓存还记录外部路径，用 `cmake -S <源码目录> -B <构建目录> -U CORSIKA_EGS4_TABLE` 切回项目默认值。

可选 `-DCORSIKA_EGS4_BUILD_TESTS=ON` 会构建 `test_c8_egs4_embedded`；用 `ctest --test-dir <构建目录> -R c8_egs4_embedded_tables --output-on-failure` 核对内嵌表与原文件及同种子输运的一致性。

对已有生产命令，只需将 `--em-backend kokkos` 换成 `--em-backend egs4`。唯一新增的算法参数是可选的 `--egs4-stepfc`，默认 `1`：

```bash
c8_air_shower --em-backend egs4 \
  -A 56 -Z 26 -E 215400 -z 60 -N 10 -s 1931 \
  --antenna-file /path/to/antennas.txt -f /path/to/new-output
```

`-A/-Z/-p/-E`、方向、高度、磁场/大气、天线、cuts、薄化、profile 穿界方式、输出均来自主应用的公共配置。不再设置 `--fe-reference`、`--muon-backend`、`--profile-backend`、`--queue-capacity` 或 `--frontier-batch`；μ 子复用 PROPOSAL/Kokkos，profile 自动在所选设备累加，批次和显存预算沿用 `--gpu-min-batch`、`--gpu-resident-batch-limit`、`--gpu-memory-fraction`。射电默认使用同一个 Kokkos 执行后端，也可显式选原有 `--radio-backend cpu`。OpenMP 构建无需显卡，CUDA 构建用 CUDA；双端构建沿用 `--kokkos-execution` 选择 CUDA 或 OpenMP（暂不支持 EGS4 的协同 `cuda-openmp` 执行模式）。

多卡沿用原生协调入口，在同一命令补上 `--devices 0,1,2,3 --gpu-memory-fraction 0.9` 即可。协调器把**尚未经过电磁输运**的粒子交给 EGS4 worker，避免 CPU 前缀混入 PROPOSAL 电磁输运。不是另开 Python 调度器。

注意：统一参数表示复用主应用设置，不是暗中套用旧 `--fe-reference` 的冻结几何。旧参考事例的磁场、大气、天线和输出深度网格不能假定与主应用默认值相同；历史结果须按原配置复现。本次入口回归验证也不替代 Fe500 的统计验收。

## 目录内容

| 文件 | 职责 |
| --- | --- |
| `Egs4Tables*`、`Egs4Model.hpp` | 外部 EGS4 表读取、设备副本及物理查询 |
| `Egs4Interactions*`、`Egs4Radiative*`、`Egs4Scattering*` | 末态、辐射过程及散射 |
| `Egs4Transport*`、`Egs4PhotonTransport*`、`Egs4Curved*` | 连续输运、步长、磁场与大气几何 |
| `Egs4C8Air*`、`Egs4C8Session*`、`Egs4C8Router*` | C8 对接、设备常驻队列及粒子路由 |
| `Egs4C8DeviceProfile.hpp`、`Egs4C8DeviceRadio.hpp` | 设备端 profile、沉积与射电累加 |
| `Egs4C8Thinning.hpp`、`Egs4C8Frontier.hpp`、`Egs4MultiGpuIO*` | 薄化、前沿分区与多卡结果合并 |
| `Egs4ProposalMuonBackend.hpp` | 复用 PROPOSAL μ 子算法；产生的电子继续交给本电磁模块 |
| `OverheadOptions.hpp` | 已测过的可选效率优化，默认全部关闭 |
| `test_*`、`prepare_*_oracle.py` | 模块、完整小事例、跨事例复现与独立 C7 参考测试 |

`SOURCE_MANIFEST.json` 记录最初收集快照的来源和文件哈希，不是后续修改的工作树校验和。项目资源目录仅保留上述冻结的 EGS4 表；构建产物、原始 shower、完整 C7 源码及 FLUKA 数据不随本目录分发。C8 其余第三方依赖仍须按原构建要求准备。

## 构建

先构建仓库原有 C8/Kokkos 及其依赖。以下命令从仓库根目录执行；占位路径须替换成本机路径，构建目录不能与现有生产构建共用。

仅构建电磁模块及基本测试：

```bash
cmake -S extensions/c7_egs4 -B build/egs4-openmp \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_DIR=/path/to/openmp-install/lib/cmake/dependencies
cmake --build build/egs4-openmp -j 2
ctest --test-dir build/egs4-openmp --output-on-failure
```

构建完整 shower 和多卡协调器，在同一配置命令中补充：

```bash
  -DEGS4_CPP_C8_OUTPUT_TEST=ON \
  -DEGS4_CPP_C8_SHOWER=ON \
  -DEGS4_C8_INSTALL=/path/to/c8-install \
  -DEGS4_PYTHIA_ROOT=/path/to/pythia8 \
  -DEGS4_SIBYLL_STATIC_LIBRARY=/path/to/libSIBYLL23d_static.a \
  -DEGS4_CPP_STABLE_MUON_IONIZATION=ON \
  -DEGS4_SHOWER_TEST_PYTHON=/path/to/python-with-numpy-pyarrow \
  -DEGS4_SHOWER_TEST_FLUPRO=/path/to/fluka
```

完整构建需要现有 C8 的 SIBYLL、FLUKA、QGSJet、Sophia、Tauola、Pythia、PROPOSAL 和 Arrow 等依赖。不会安装或替换生产二进制。`STABLE_MUON_IONIZATION` 对小角度方向构造使用已测试的数值稳定实现；只在本构建生成头文件覆盖层，不改 C8 源文件。跨事例输出重置同样使用构建目录内覆盖层。

CUDA 使用新的构建目录、对应 GPU 架构的 CUDA Kokkos 安装，并补充 `-DEGS4_CPP_CUDA=ON`、`-DCMAKE_CXX_COMPILER=/path/to/nvcc_wrapper` 和必要的 `-DCUDAToolkit_ROOT=/path/to/cuda`。Kokkos 安装的 GPU 架构必须适合目标设备。

独立 C7 参考测试可额外启用 `EGS4_CPP_REFERENCE_TESTS=ON`，并提供 `EGS4_C7_SOURCE`（`corsika.F`）和 `EGS4_C7_SCATTER_REFERENCE`（已审核的散射 oracle，CMake 校验哈希）。参考程序单独运行，不链接为电磁后端。可选 `EGS4_STEP_REFERENCE_FILE` 指向冻结的 `C7StepTables.hpp`，用于逐系数核对；不提供时仍做设备/主机一致性和边界测试，但不会声称完成冻结系数比对。

## 历史独立测试入口（保留复现）

先通过 `--help` 检查本机可用选项，再从小事例验证。以下沿用 Fe-56、215400 GeV、60°、110 根天线、STEPFC=1 的参考配置，需要此前已核对的天线坐标文件：

```bash
export FLUPRO=/path/to/fluka
export OMP_NUM_THREADS=1
build/egs4-cuda/egs4_test_schedule \
  --worker "$PWD/build/egs4-cuda/c8_egs4_shower" \
  --table "$PWD/resources/c7_egs4/EGSDAT6_.4" \
  --fe-reference --antenna-file /path/to/fe_reference_antennas.txt \
  --stepfc 1 --devices 0,1,2,3 --gpu-memory-fraction 0.9 \
  --muon-backend kokkos --radio-backend kokkos --profile-backend kokkos \
  --queue-workspace reuse --queue-capacity 1000000 --frontier-batch 65536 \
  --persistent --event-seeds 2609295000,2609295001 \
  --timeout 1800 -f /path/to/new-output
```

跨事例模式保留模型、设备表和队列，但重置本事例的输出。C8 主机随机流跨事例连续推进，因此复现后续事例需要相同 campaign 前缀，不能只取其中一个 seed 单独重跑。显存比例是预算，不保证每一时刻占用达到该比例。

## 优化与验证边界

保留 `EGS4_OVERHEAD_MASK=0..511` 的已有可选优化，包括大气列深缓存、减少队列清零、随机流准备复用、空天线快路径、稀疏主机输出、查询复用、散射三角函数和坐标复用。默认 `0`；`511` 为全部开启。`EGS4_AUDIT_NO_ANTENNAS` 仅供无射电计时诊断，正常生产不要设置。高能事例保持最新修复：不以累计 300 万个标量步作为整个 shower 的终止条件，仍保留队列及超时检查。

先前四 L20 的 500 例完成了输出完整性检查，不代表所有物理差异消失：μ 电离电子保留，且与原 C7 的算法/模型组合并非完全一致。效率优化只做过较小样本检查；本次打包重编译的结果另见 `VALIDATION_CN.md`，不得用先前 Fe500 的结论替代新组合的统计验证。
