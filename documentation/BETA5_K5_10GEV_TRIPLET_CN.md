# k5：beta5 / 原单核 CPU 的 10 GeV 三后端验收

本轮是实验部署和统计脚本工作，未修改输运、随机数、射电或制表算法。
**尚不能将已启动任务写成“2000 × 3 验收通过”**：正式结果以服务器
`validation_plots/analysis_complete.json` 和报告为准。

## 数据集

```text
final_beta2scalar2000_beta5cuda2000_openmp2000_proton_10GeV_vertical_emthin1e-6_igrf14_2027_k5_radio20us_v2
```

- 服务器原始数据：`~/21CMA/corsika_validation_results/<上述目录>/`。
- 本地仅同步报告、图片和小型统计表：`D:\CorsikaData\corsika_validation_results\<上述目录>\validation_plots\`。
- 三臂：原 beta2 标量 PROPOSAL + CPU CoREAS/ZHS；beta5 Kokkos-CUDA；beta5 Kokkos-OpenMP。
- 各 2000 例，10 GeV **总能量**质子、垂直入射。没有使用 1 GeV 质子，因为其约 62 MeV 动能低于默认 hadron cut，容易形成无诊断价值的退化样本。
- emthin=1e-6、max-weight=0（交给原默认规则）、EM cut=0.5 MeV、hadron/muon/tau cut=0.3 GeV。
- FLUKA + SIBYLL-2.3d，转换能量 `10^1.9 GeV`，IGRF14/2027，原 81 天线位置。
- 三臂使用 80 个独立批次，种子 2026090601–2026090680，每批 `-N25`。
  同批初始种子相同，不等于 CPU/GPU 第 k 个 shower 的随机流相同；不是 decision-tape replay。
- 原 CPU 每个进程单核，同时运行 8 个独立批次；OpenMP 单个 16 线程团队；CUDA 使用 A100 + Serial 主机调度。
  没有在一个 shower 内混合 OpenMP 与 GPU。

## 必须加宽低能射电窗口

最初沿用高能比较的 400 ns 窗口。三臂 `-N1/-N25` 都有非零 EM profile，
但 CoREAS/ZHS 数组全零、GPU radio contributions=0。
加宽至 20 us 的原 CPU 测试实测最早非零信号约在原窗口起点后 5076 ns，
CoREAS 峰值绝对场约 1.75e-11 V/m。

原因是应用按从注入点以光速传播到天线的时间开启窗口，低能质子/次级的迟到不可忽略。
因此正式三臂统一使用 `--radio-window-duration-ns 20000`，仍为 1 GHz 采样、10 ns pretrigger。
不更改粒子物理，不选择“波形好看”的种子；早期窄窗目录 `..._k5_v1` 单独保留，不混入正式验收。
窄窗 scalar 完成了 2000 例，不能据其零波形宣布射电算法一致。
20 us 仍是有限观察窗口；不能将窗口内的比较声称为所有迟到辐射的积分验证。

宽窗三臂 `-N1/-N25` 已通过。额外逐列比较同一后端的窄窗与宽窗 25 例：
profile、parent profile、dE/dX、地面粒子及 interactions 共 15 张表全部一致。
这证明本次窗口调整没有改变这批 shower 的粒子演化；记录见
`window_change_physics_regression.json`。

## 部署、保护和运行

- beta5 CUDA/OpenMP 在 k5 原机独立 Release 构建，安装至 `install/cuda` / `install/openmp`。
  CUDA 只链接 CUDA runtime，不链接 libgomp；OpenMP 链接 libgomp，不链接 CUDA runtime。
- k5 旧原版目录实际为 beta1，默认高度和 ZHS 装配不同，不作为此轮参考。
  参考使用从 dirac 复制的已验证 beta2 CPU 安装，二进制哈希写入 manifest。
- 原 CPU 安装迁移的路径处理：设置 `CORSIKA_DATA` 到已安装的 `share/corsika/data`；
  IGRF14.COF 从同一安装的 `share/corsika/GeoMag` 复制到 data/GeoMag；
  编译期 Pythia XML 旧路径用符号链接指向安装中的相同 XML。
  不重编译原 CPU、不替换模型数据。IGRF14 SHA-256：
  `02fc7f4573ef453158eec55d2e5b7f4a25df98ca2475a20463f40b8f2bbfebcf`。
- CUDA 13 / A100 的真实内核运行已经验证；nvidia-smi 的 NVML 用户库/内核驱动不匹配仍可能使其不可用，未改驱动。
- 每个子进程 RSS 上限 8 GiB、服务器可用内存下限 32 GiB、单批诊断超时 900 s、磁盘剩余下限 15 GiB。
  触发门禁则停止，不替换失败种子；每 25 例退出进程释放内存。
- 使用 systemd 用户服务，不依赖临时终端：

  ```bash
  systemctl --user status c8-k5-10gev-triplet2000v2
  ```

  服务顺序执行：scalar 2000 → CUDA 2000 → OpenMP 2000 → 完整性复查 → 统计绘图。
  已完成批次可按原种子续跑；失败/中断的批次要求人工检查、归档后原种子重跑，不覆盖输出。
- 复现脚本存放在数据目录 `analysis_support/`，本地源副本在 beta5 的 `build/audit/k5_lowenergy_2000/`。

## 统计与图

- 全部纵向 EM、hadron、muon、带 parent 的生产 profile 和 dE/dX；不按 Xmax 对齐。
- 每例标量直方图，Xmax 是 EM 网格 bin 最大值，不是低能不稳定的 Gaisser–Hillas 拟合。
  零 EM 事件的 Xmax 未定义，单独统计比例，不删除这些事件的其他观测量。
- CoREAS/ZHS：逐天线、带正负号的 Ex/Ey/Ez 系综平均时域波形；阴影为均值 95% CI。
  不做峰值对齐、包络、宽度拟合、场强归一化或时间窗边缘删点。
- 以独立种子批次为重采样单位，避免把同批多 shower、粒子或相邻时间 bins 当作独立样本。
  9999 次批次 bootstrap 用于标量；曲线使用批次均值 t 区间及 Holm 多重检验。
- “未检出差异”和“95% 相对差异 CI 完全在 ±1% 内”分别报告；不能互相替代。
- 10 GeV 主要覆盖低能 FLUKA 以及低能 EM/radio；薄化阈值低于 EM cut，基本是未薄化样本。
  不声称验收了高能 LPM、UHE 强子反应或高显存负载性能。

## 图片自动回传

本地 `c8-k5-10gev-figures.timer` 每三分钟检查最终分析完成标记，完成后只 rsync 图片、Markdown、JSON、CSV，
不下载原始 Parquet 或大的 NPZ。成功后停止 timer。SSH 断线或 WSL 关机后可能需要重新连接 k5 并重启 timer；
服务器模拟与统计服务不依赖这条回传连接。
