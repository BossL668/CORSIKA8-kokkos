# 原生多 GPU 入口

CUDA 和 CUDA/OpenMP 构建直接在 **同一个 c8_air_shower 二进制**中提供
统一的 `--device` 参数：一个编号走单卡，多个编号走多卡。
不需要填写 JSON，也不调用 Python 协调器。

## 构建

在源码目录使用原有组合版构建脚本：

~~~bash
C8_BUILD_JOBS=1 bash tools/build_kokkos_dual.sh -DWITH_FLUKA=ON
~~~

FLUKA 仍需按主 README 配置已有的授权安装。CUDA 构建默认启用
CORSIKA_ENABLE_NATIVE_MULTIGPU；OpenMP-only 构建不依赖该模块。
运行阶段的前沿分配、进程管理、Parquet/YAML 读取和结果合并都使用 C++。
Conan 和构建时的代码生成工具仍按原项目要求安装。

## 使用

~~~bash
cd ../install/cuda-openmp/bin
./c8_air_shower -p 2212 -E 1e5 -N 10 -s 20260924 \
  --device 0,1,2,3 --gpu-memory-fraction 0.90 \
  --ring 1 --antenna-file /dev/null -f demo_multigpu
~~~

- `--device 0`：直接运行单卡；`--device 0,1,2,3` 或 `--device 0 1 2 3`：同一例 shower 使用四卡。
  编号与 `nvidia-smi` 一致，也接受完整 GPU UUID；两种路径均按 UUID 选择实际设备。
- -N：依次运行的 shower 数量；每例使用全部选中的卡。种子为 -s 加从 0 开始的事件序号。
- --gpu-memory-fraction：每个 worker 的预算，默认 0.5，可显式设置到 0.9。
- --multigpu-timeout：每个前缀/worker 的超时秒数，默认 7200；很慢的高能模拟可调大。
- --multigpu-frontier-energy：可选的调度根粒子能量上限（GeV），默认自动选择；它不是物理 cut。

`--device` 自动选择 Kokkos CUDA 输运和射电。显式给出的
--em-backend kokkos --radio-backend kokkos --kokkos-execution cuda 也接受。
也可用 `--em-backend egs4` 选择 EGS4 电磁后端。
不传 `--device` 时保持原有行为；CPU-only 用户仍用原 OpenMP 构建，不传 GPU 编号。
旧的 `--devices` 保留为兼容别名，但只传一个编号时现在同样走直接单卡路径，
不再创建单 worker 的协调目录。`--kokkos-device` 仅保留给旧脚本及内部 worker，
其含义仍是可见设备内的 CUDA 序号；不要与 `--device`/`--devices` 混用。

固定初级能量、物理 cut、磁场、大气、薄化和天线等参数继续传给原空气程序。
不支持在此路径同时选择能谱抽样、强制初级反应或 --compress。
输出目录需为新目录。

## 计算与输出

以下说明针对选择多个 GPU 的路径；单卡仍使用普通应用的输出结构。
一个 CPU 前缀导出互不重叠的 EM/μ 根粒子，按未加权能量代理分配一次；
每张卡在独立进程内推进自己的子簇、CPU 回退和随机流。子进程重新执行同一个
c8_air_shower，协调进程不创建 Kokkos/CUDA 上下文。每个 worker 有独立工作目录，
结束时间独立记录。全部成功后按固定 worker 顺序合并。

每例保留 prefix/、worker_*/、前沿及分片哈希、PARTITION.json、
TIMING.json，最终数据在 merged/。-N 1 直接写入指定目录；
-N > 1 为每例创建 seed_<seed>/，顶层保留总计时和批次清单。

profile/沉积以及**带符号**的 CoREAS、ZHS 场在完全相同的网格上以补偿
float64 加法合并；地面粒子拼接，首次相互作用只取全局前缀记录。
同种子划分顺序和 SHA256 worker 种子派生与旧 Python 协调器一致。
COMPLETE 只在正常退出、输出有限值/网格/配置、前沿消费和溢出检查通过后生成。
中断/超时只结束本次创建的子进程，保留已输出的数据。

COMPLETE.json 表示流程完成，不自动声明新的统计物理验收通过。
旧的 c8_air_shower_multigpu worker 和 Python 队列可继续用于既有任务；
新命令不经过这些 Python 文件。

## 验证

独立 CPU 测试使用 C++ fixture worker，验证真实 Parquet 合并、四分片与空分片、
符号场、首反应记录、重复 history、配置/网格不一致、非有限值、重复 GPU 别名、
失败退出、超时清理和连续多例。fixture 不用于宣称四张真实显卡的性能。

~~~bash
cmake -S validation/accelerator/native_multigpu -B ../build/native-multigpu-tests \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/../build/cuda-openmp/deps/conan_toolchain.cmake"
cmake --build ../build/native-multigpu-tests -j2
ctest --test-dir ../build/native-multigpu-tests --output-on-failure
~~~

已完成本地单卡/OpenMP 回归，以及四 L20 的两例 1 TeV 质子回归。
四卡第一例与旧协调器、旧 worker 的七类输出逐项一致；完整条件和边界见
[验证记录](../validation/accelerator/native_multigpu/VALIDATION_CN.md)。
同目录的 compare_reference.py 仅用于可选测试（对照旧 Python 协调器），
需要至少两张 GPU，不是新入口的运行依赖。原生协调器不调用 Python。
本次功能回归不替代大样本统计验收或单卡/四卡缩放测试。

### 统一 device 参数回归（2026-10-05）

- C++ 参数测试覆盖单编号、逗号/空格列表、UUID、兼容别名、重复编号、
  重复物理设备和参数冲突；旧启动器 26 项测试通过。
- OpenMP/CUDA 均编译通过。0.1 GeV 光子、固定种子、每次两例的小样本中，
  Kokkos 与 EGS4 的新旧入口及单卡别名共 70 项 Parquet 文件对照完全相同，
  包含 profile、沉积、CoREAS/ZHS 等输出；CUDA 在本地 RTX 4060 实测。
- C++ 四 worker fixture 验证三种列表写法、连续两例、合并结果和每个 worker
  在独立 UUID 掩码内使用序号 0。
- shower 前后对照使用提交 037d7a04 加本次 CLI 修改的隔离源码，复用对应的
  未修改物理库，避免混入工作区其他正在进行的物理修改；当前工作区协调器另行编译测试。
  本次未重新跑四 L20 实机性能测试，也未替换远端生产二进制。重编译后新参数才生效。
