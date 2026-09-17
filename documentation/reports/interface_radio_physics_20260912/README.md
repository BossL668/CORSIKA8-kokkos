# beta5 山体跨介质射电报告

已完成 **36 页、16:9 Marp 幻灯片 Markdown**，附 **18 张诊断图**。本次补充编译、数值比较、绘图和导出均在 PSR 上进行；本地只编辑、查看和归档。没有改动生产物理或原空气模块。

- [Markdown 讲稿](REPORT_SLIDES_CN.md)：可编辑源文件，含公式、源码引用与讲述备注。
- [PDF](REPORT_SLIDES_CN.pdf)：便于阅读、打印和检查版面。
- [PPTX](REPORT_SLIDES_CN.pptx)：可直接演示；Marp 导出将每页内容渲染为图像，逐项编辑请修改 Markdown 后重新导出。
- [HTML 幻灯片](REPORT_SLIDES_CN.html)：与 `figures/` 保持相对位置。
- [数值指标](data/report_metrics.json)、[图片来源](data/figure_manifest.json)、[版本追溯](data/source_provenance.json)、[交付检查](data/report_audit.json)。

## 内容与结论范围

| 页码 | 内容 |
| --- | --- |
| 1–6 | 版本、物理效应、当前未实现的效应 |
| 7–14 | CoREAS / ZHS、共同光路适配、切伦科夫极限、连续矩、几何与光学积分 |
| 15–22 | 验证层次及实际 RadioPropa / TMM 对照 |
| 23–25 | 固定轨迹独立积分、精度收敛、原空气 CoREAS 对照 |
| 26–30 | 完整电子与强制 CC 中微子簇射、波形、复频谱及源分解 |
| 31–33 | GPU 常驻执行证据、结论边界、后续物理完善 |
| 34–36 | 复现索引、文献、启用方式与内存 |

新增光学探针直接调用当前 beta5 的 `Propagation.hpp`；公开库侧独立调用固定版本 RadioPropa 和 TMM，没有把 beta5 公式替换进公开库：

- 642 组角度 / 介质组合的全反射分类一致；透射方向的最大正弦绝对差约 `3.28e-15`。
- TMM 的 s/p 电场透射系数最大相对差约 `6.00e-15`；独立 RadioPropa 射线束面积雅可比最大相对差约 `3.79e-10`。
- 固定 RadioPropa 版本的纯 s 偏振振幅存在最高约 **7.97%** 的差异，报告展示了差异与对应偏振基实现，没有通过拟合消掉。
- 强线性折射率梯度示例中，beta5 直线积分与弯曲射线的到达时间差约 **1.23 ns**，路径最大偏离约 **10.75 m**。这是有意设置的模型边界实验，不是实际大气误差估计。

这些外部比较验证的是光学子问题，**没有完成外部库对整个簇射发射场的独立验证**。CoREAS / ZHS 共用光学近似和奇异极限处理，因此二者一致还需结合独立电流积分与收敛证据解读。中微子完整簇射案例使用 `--emthin 0.1`，用于功能与一致性检查，不作为收敛后的物理预报。

报告复用 c4 固定源 / 收敛验收及 p1 默认双算的实际 OpenMP / CUDA 簇射和 CUPTI 记录。此次没有重新运行 GPU 簇射。当前发射核心与 c4 相同，双算接入源码与 p1 相同，244 个受保护空气源码文件哈希保持一致；详见版本追溯 JSON。

## 数据与源码

- `figures/`：18 张 PNG；14 张本次新绘图另附矢量 PDF。4 张既有 PSR 验收图按原来源保留。
- `data/*.csv`：此次 beta5 / RadioPropa 原始光学探针输出。
- `data/*waveforms.npz`：实际簇射的时间、频率、两算法场 / 复频谱和介质分量数组，便于重画。
- `data/external_arrays.npz`：外部比较数组，含有效角度掩码。
- `data/c4_*.json`、`data/pair_*.json`、`data/scene_*.yaml`：关键历史验收结果及场景；完整运行命令见 `report_metrics.json`。
- `data/layout/`：最终 PDF 的版面检查和缩略图。
- `tools/`：报告专用 C++ 探针、Python 分析和 shell 运行 / 导出脚本。
- `SHA256SUMS`：交付文件校验清单；不含清单自身。

PSR 的原始工作根目录：

```text
/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-interface-resident-20260911/
├── source/                           # 被探针调用的 beta5 源码
├── interface-coreas-20260912/         # c4 物理验证与空气算法对照
├── interface-radio-pair-20260912/     # p1 默认双算及实际 CUPTI trace
└── interface-radio-report-20260912/  # 本次外部验证、绘图和导出
```

原始大型轨迹、矩数组与完整 trace 保留在上述 PSR 目录，没有全部复制进报告包。Markdown 的生产源码相对链接针对本地仓库中的 `documentation/reports/interface_radio_physics_20260912` 位置；单独分发报告包时，图表仍可阅读，生产源码链接需对应仓库。

旧 mountain 的 `external_validation/radiopropa_crosscheck.cpp` 被复用为梯度探针，平面折射、射线束与 beta5 光学探针为本次新增；旧项目的研究能力没有被记作 beta5 已接入的功能。

## 在 PSR 复现

以下命令应在 **PSR** 执行。既有工作目录已经具备依赖和产物；重画依赖相邻的 c4 / p1 原始验收目录。所有亲和范围为 CPU 0–255，OpenMP 线程数为 256；小型数值探针本身不保证能利用全部核心。

```bash
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads
export OPENBLAS_NUM_THREADS=1
c8_report=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-interface-resident-20260911/interface-radio-report-20260912
cd "$c8_report"
```

外部依赖未打包进交付文件。需要重新安装时，使用本报告固定版本：

```bash
mkdir -p external data figures
c8_rp_rev=544a2d6c4e284e3d724cb741dc481245a0f633d7
curl -fL "https://codeload.github.com/nu-radio/RadioPropa/tar.gz/$c8_rp_rev" \
  -o external/radio-report-public-radiopropa.tar.gz
tar -xzf external/radio-report-public-radiopropa.tar.gz -C external
taskset -c 0-255 cmake \
  -S "external/RadioPropa-$c8_rp_rev/radiopropa" \
  -B external/build-radiopropa -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/c++" \
  -DENABLE_TESTING=OFF -DENABLE_PYTHON=OFF -DENABLE_HDF5=OFF -DENABLE_GIT=OFF
taskset -c 0-255 cmake --build external/build-radiopropa --parallel 256
PIP_REQUIRE_VIRTUALENV=false python -m pip install --target external/python --no-deps tmm==0.2.0
```

该提交与旧 mountain 对照所用 RadioPropa 提交一致，不代表最新版本。哈希和逐文件未修改检查结果写入 `source_provenance.json`。

重新运行光学探针、分析与绘图：

```bash
bash tools/run_optics_psr.sh
taskset -c 0-255 python tools/make_report_figures.py
taskset -c 0-255 python tools/collect_evidence.py
```

绘图脚本需要此环境已有的 NumPy、SciPy、Matplotlib 和 Pillow；生产探针使用 C++17，公开库探针通过动态链接实际 RadioPropa 库运行。完整簇射的发射结果读取 p1 原始 CSV；复现全套图需要原验收目录，单独交付包主要提供可直接阅读及二次作图的图表和压缩数组。

## 在 PSR 导出幻灯片

本报告使用 Marp CLI **4.5.1** / Marp Core **4.4.0**、Chromium 和 Noto Sans CJK SC。PSR 已有 `/snap/bin/chromium`。获取 Marp：

```bash
curl -fL https://github.com/marp-team/marp-cli/releases/download/v4.5.1/marp-cli-v4.5.1-linux.tar.gz \
  -o tools/radio-report-marp.tar.gz
tar -xzf tools/radio-report-marp.tar.gz -C tools
bash tools/render_psr.sh
taskset -c 0-255 python tools/audit_report.py
```

渲染器通过 `tools/chromium_cpu.sh` 关闭 GPU 加速。版面检查确认 PDF / PPTX 均为 36 页、所有图像链接存在及 PDF 文字不超出页面边界；同时人工检查了缩略图。自动边界检查不能替代对图表内容和公式的阅读。

## 公开来源

- [RadioPropa 固定源码](https://github.com/nu-radio/RadioPropa/tree/544a2d6c4e284e3d724cb741dc481245a0f633d7) 与 [论文](https://arxiv.org/abs/1810.01780)。
- [TMM 官方源码](https://github.com/sbyrnes321/tmm) 与 [算法说明](https://arxiv.org/abs/1603.02720)。
- 端点、CoREAS、ZHS 的原始文献见幻灯片第 35 页；各外部主张旁保留直接来源链接。

