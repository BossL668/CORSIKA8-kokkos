# C7 EGS4 C++ / Kokkos 电磁扩展

这是 C7 电磁算法的原生 C++ 实验实现，不是对 C7 Fortran 电磁程序的调用包装。电子、正电子和光子由本模块推进；强子模型与 μ 子物理由 C8 现有模块提供，μ 子可选择 PROPOSAL/Kokkos 路径。模块单独配置构建，不改变仓库原有应用和默认 PROPOSAL 后端。

当前可运行入口是 `c8_egs4_shower` 和原生 C++ 多卡协调器 `egs4_test_schedule`。**尚未接入主应用 `c8_air_shower` 的后端选择开关**；未启用的接入草稿没有收入本目录。Python 只用于测试和参考数据生成，生产调度不需要 Python。

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

`SOURCE_MANIFEST.json` 记录收集来源和文件哈希。构建产物、原始 shower、完整 C7 源码、EGSDAT 数据及 FLUKA 数据不随本目录分发；须自行提供合法安装。

## 构建

先构建仓库原有 C8/Kokkos 及其依赖。以下命令从仓库根目录执行；占位路径须替换成本机路径，构建目录不能与现有生产构建共用。

仅构建电磁模块及基本测试：

```bash
cmake -S extensions/c7_egs4 -B build/egs4-openmp \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_DIR=/path/to/openmp-install/lib/cmake/dependencies \
  -DEGS4_TABLE=/path/to/EGSDAT6_.4
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

## 四卡 Fe 运行示例

先通过 `--help` 检查本机可用选项，再从小事例验证。以下沿用 Fe-56、215400 GeV、60°、110 根天线、STEPFC=1 的参考配置，需要此前已核对的天线坐标文件：

```bash
export FLUPRO=/path/to/fluka
export OMP_NUM_THREADS=1
build/egs4-cuda/egs4_test_schedule \
  --worker "$PWD/build/egs4-cuda/c8_egs4_shower" \
  --table /path/to/EGSDAT6_.4 \
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
