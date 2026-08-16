# `c8_ice_cascade`：独立的均匀冰 CUDA 应用

## 1. 设计边界

`c8_ice_cascade` 是 beta4 CUDA 树中的独立可执行程序。它不通过
`c8_air_shower --environment ...` 切换介质，也不共享空气程序的环境 CLI：

- `applications/c8_air_shower.cpp` 保持空气专用源码；
- `applications/c8_ice_cascade.cpp` 固定构造均匀 `WaterIce`；
- 两者是不同 CMake target、不同 translation unit 和不同二进制；
- 冰应用没有 `--environment`、IGRF 模型或地磁年份选项；
- 冰应用只在 `CORSIKA_ENABLE_CUDA=ON` 时构建。

程序名是物理合同的一部分：今后的冰/水 Askaryan CUDA 事例使用
`c8_ice_cascade`，空气 shower 继续使用 `c8_air_shower`。

## 2. 介质与传播合同

CPU 环境固定为：

```text
medium              = Medium::WaterIce
nuclear composition = H:O = 2:1（数目比）
density              = --density-g-cm3，默认 0.919 g/cm3
refractive index     = --refractive-index，默认 1.78
magnetic field       = 0
```

CUDA 启动时执行 fail-closed 检查：

1. rate table 的 `medium_name` 必须严格为 `water_ice`；
2. PROPOSAL medium hash、H/O 组分 hash、PDG 和数目分数必须与 CPU
   `Medium::WaterIce` 一致；
3. bremsstrahlung 与 photon-pair LPM reference density 必须与 CORSIKA
   WaterIce 基准一致；
4. device environment 固定使用单层 homogeneous spherical snapshot，medium ID
   为 `Medium::WaterIce`；
5. 非均匀或跨介质的 CPU radio path 会被冰专用 propagator 拒绝。

均匀冰中的解析光路为直线：

\[
t_{\rm prop}=\frac{nR}{c},\qquad
\hat{\boldsymbol k}=\frac{\boldsymbol x_{\rm obs}-\boldsymbol x_{\rm src}}{R}.
\]

CPU CoREAS/ZHS 使用定义在 `c8_ice_cascade.cpp` 内的
`HomogeneousIceRadioPropagator`。它分别查询源点与 observer 的折射率并要求两者在
`1e-12` 相对容差内相等；该实现不依赖普通 particle 的 `getNode()`，因此同时兼容
标量 CORSIKA particle 与 CUDA compact radio-track adapter。CUDA radio 使用同一个
Environment 生成常数折射率传播表和相同 observer snapshot。

## 3. 构建

```bash
export FLUPRO=/home/yuhanglu/fluka
export FLUFOR=/usr/bin/gfortran

cmake -S corsika8_gpu_refactor_beta4 \
      -B corsika8_gpu_refactor_build_cuda
cmake --build corsika8_gpu_refactor_build_cuda \
      --target c8_ice_cascade -j2
```

当前已构建二进制：

```text
corsika8_gpu_refactor_build_cuda/applications/c8_ice_cascade
SHA-256: b8196ff3c2c5c33aaed0b91b4874da44356bfaad8c9080f0e7c3b855697db08c
```

该 hash 对应首次完成编译的版本；后续源码改动或重新链接时必须重新记录，不应把它当作
永久 release hash。

## 4. 运行接口

典型的 EM + radio 双 GPU 命令为：

```bash
FLUPRO=/home/yuhanglu/fluka FLUFOR=/usr/bin/gfortran \
corsika8_gpu_refactor_build_cuda/applications/c8_ice_cascade \
  -p 11 -E 100 -N 1 -z 0 -a 0 -s 62002 \
  -f /path/to/output \
  --density-g-cm3 0.919 \
  --refractive-index 1.78 \
  --observation-level 0 \
  --injection-height 2500 \
  --emcut 0.0005 --hadcut 0.3 --mucut 0.3 --taucut 0.3 \
  --antenna-file /path/to/observers.txt \
  --em-backend cuda \
  --radio-backend cuda \
  --gpu-min-batch 4096 \
  --gpu-resident-cross-species true \
  --gpu-table-cache /path/to/water_ice_table.c8emrt
```

`--help` 合同检查已通过：包含 `--density-g-cm3`、`--refractive-index`、
`--em-backend` 和 `--radio-backend`，不包含 `--environment` 或地磁配置。
EM 和 radio 后端默认均为 `cuda`；CPU/PROPOSAL 仅作为显式选择的参考路径保留。

## 5. 验收状态

- 独立 CMake target：通过；
- Release + CUDA + FLUKA 编译/链接：通过；
- 空气源码隔离：`git diff -- applications/c8_air_shower.cpp` 为空；
- CLI 隔离：通过；
- 100 GeV smoke、1 TeV 安全门和 100 GeV--1 PeV 分级能量扫描：通过；
- 1 TeV batch 扫描的本机推荐值：`4096`，不再使用 `1`；
- 三个 100 TeV 全事件能量账本：通过，最大相对闭合残差 `1.672e-6`；
- 100 TeV 原始射电输出：`1 ns × 4096`，profile 为 `10 g/cm2`；
- queue overflow、host spill、radio fixed-point overflow：全为零；
- framework/module 定向测试：`64/160` assertions 全通过。

旧的 1 EeV `gpu-min-batch=1` 作业已终止并标为不完整，不能作为物理结果。1 PeV 无
thinning 事例虽已完成，但主机 page churn/协调开销仍明显；在解决性能门和 thinning
收敛前，不直接重启 1 EeV。

## 6. 全事件能量账本

当前应用不再只报告 device-only EM 账本。每个 shower 同时审计：

- GPU 介质电子静质量输入、沉积、cut 静质量、observation 与 escape；
- CPU `ParticleCut` 的逐 PDG 动能/静质量和 cut 中微子动能；
- FLUKA、SIBYLL、SOPHIA/photo-hadronic 的 projectile、target、secondary 四动量；
- 未显式返回的类时核残余/局域能量。

FLUKA 材料按自然元素抽样，因此低能顶点由最终态净重子数和净电荷推回实际靶同位素，
避免把 `18O` 事例按名义 `16O` 记账。SOPHIA 的宽共振先在其离壳四动量下衰变，再在 COM
系投影到 CORSIKA 名义质量壳并强制 `sum p=0`、`sum E=sqrt(s)`；稳定的长寿命粒子仍由
CORSIKA 输运。

正式 100 TeV ensemble 为 seed `65501--65503`、`gpu-min-batch=4096`、CUDA EM + CUDA
radio、无 thinning。最大负顶点残差 `2.028 MeV`，超过 `1e-9 GeV` 容差的类空残差为
零。大正残差保持为独立核残余项，不计入可见沉积。

运行与分析脚本位于相邻 `corsika8-mountain` 项目：

```text
scripts/run_beta4_ice_100tev_energy_ledger_validation.py
scripts/analyze_beta4_ice_100tev_energy_ledger_validation.py
docs/BETA4_GPU_ICE_100TEV_ENERGY_LEDGER_VALIDATION_CN.md
```

本门只覆盖均匀冰、100 TeV、当前无 thinning 过程链；CoREAS 深冰、firn/界面、EeV 与
ARZ 绝对归一化不在这项通过声明内。新的冰 CUDA 事例仍不得使用空气入口。
