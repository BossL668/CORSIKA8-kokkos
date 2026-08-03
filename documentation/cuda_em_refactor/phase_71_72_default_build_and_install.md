# Phase 71–72：默认后端、CPU-only 与安装导出

## 1. 当前全树回归

CUDA Release build：

```text
all target    PASS
CTest         32/32 PASS
```

覆盖 framework、media、stack、modules、output、全部 GPU host/device tests
以及 copyright gate。

CPU-only build：

```text
build directory            corsika8_gpu_refactor_build_clean
CORSIKA_ENABLE_CUDA        OFF
build type                 RelWithDebInfo
all target                 PASS
CTest                      10/10 PASS
```

CPU-only 全树仍会构建不依赖 CUDA 的 table serialization 与
`gpu_em_tablegen`，但不构建/链接 CUDA runtime 或 `CORSIKA8GpuEm`。

Python validation：

```text
40/40 PASS
```

## 2. 默认后端运行等价

使用相同 10 GeV electron、seed 171001，分别：

```text
省略 --em-backend
显式 --em-backend proposal
```

输出：

```text
/tmp/c8_phase71_default_backend_v1
/tmp/c8_phase71_explicit_proposal_v1
```

结果：

- 文件集合相同；
- 23 个科学输出文件逐字节相同；
- 两边都没有 `gpu_em/summary.yaml`；
- 只有 `config.yaml` 中命令/路径，以及 wall-time summary 不同。

因此当前可执行文件省略选项时确实保持原标量 PROPOSAL 路径。

## 3. 安装包 dependency 修复

`CORSIKA8::CORSIKA8GpuEm` 是静态库，内部 Molière 插值实现对
`CubicInterpolation::CubicInterpolation` 的 private link 在最终消费程序链接
时仍然需要。原安装配置导出了该 target 名，却没有先查找 dependency。

`corsikaConfig.cmake.in` 现在在 CUDA 包中执行：

```cmake
find_dependency(CUDAToolkit 12)
find_dependency(CubicInterpolation REQUIRED)
```

独立 downstream 工程位于：

```text
validation/gpu_em/install_smoke
```

它只通过安装包：

```cmake
find_package(corsika CONFIG REQUIRED)
target_link_libraries(app PRIVATE CORSIKA8::CORSIKA8GpuEm)
```

完成配置、编译、链接并运行，证明导出 target 的 dependency closure 完整。

## 4. 安装应用可重定位 RPATH

原 application RPATH 固定为配置时的：

```text
${CMAKE_INSTALL_PREFIX}/lib
```

使用 `cmake --install --prefix NEW_PREFIX` 后，程序仍寻找旧前缀并报告：

```text
libCONEXsibyll.so => not found
```

现在安装 RPATH 为：

```text
$ORIGIN/../lib
```

安装到：

```text
/tmp/c8_phase72_install_v3
```

后验证：

```text
RUNPATH              $ORIGIN/../lib
libCONEXsibyll.so    NEW_PREFIX/lib/libCONEXsibyll.so
c8_air_shower --help PASS，无 LD_LIBRARY_PATH
GPU CLI options      present
```

安装内容包含：

```text
lib/libCORSIKA8GpuEm.a
lib/libCORSIKA8GpuEmTables.a
bin/c8_air_shower
bin/gpu_em_tablegen
include/corsika/gpu/...
lib/cmake/corsika/corsikaTargets*.cmake
```

这关闭了“只在源码 build tree 内可用”的生产化缺口。
