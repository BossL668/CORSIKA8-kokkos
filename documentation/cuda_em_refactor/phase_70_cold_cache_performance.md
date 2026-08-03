# Phase 70：逐轮冷缓存正式性能验收

## 1. 为什么需要新的 runner 语义

原性能 runner 的 `--skip-cache-warmup` 只表示“不主动预热”。如果
`TABLE.moliere-initial-v1.c8cache` 已经存在，该运行仍然是热缓存，不能作为
冷启动证据。

现在增加：

```text
--cache-mode warm
--cache-mode as-is
--cache-mode cold-first
--cache-mode cold-each
```

- `warm`：正式计时前确保派生 Molière 插值缓存存在；
- `as-is`：保留当前状态，兼容旧 `--skip-cache-warmup`；
- `cold-first`：要求开始时缓存不存在，只让第一轮冷启动；
- `cold-each`：要求开始时缓存不存在，并在每个后续 CUDA repetition 前删除
  上一轮由本次 benchmark 生成的派生缓存。

冷模式如果发现 benchmark 开始前已经存在缓存，会直接拒绝运行，不会删除用户
预存文件。正式冷测把不可变 production rate table 复制到独立 `/tmp` 目录；
所以 runner 只处理自己产生的派生缓存。

每轮报告：

```text
present_before_prepare
removed_before_run
present_before_run
present_after_run
cache_bytes_after_run
```

两项单元测试验证预存缓存不会被删除，以及 `cold-each` 只删除上一轮自己产生的
缓存。Python validation tests 更新为 40/40 通过。

## 2. 正式配置

执行：

```bash
python validation/gpu_em/run_performance_acceptance.py \
  --executable \
    /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --table \
    /tmp/c8_phase70_cold_assets_v1/production_v9_1e-3_1EeV.c8emrt \
  --output-root \
    /tmp/c8_phase70_cold_performance_1PeV_5rep_v1 \
  --energy-gev 1000000 \
  --events 1 \
  --repetitions 5 \
  --seed 170001 \
  --em-thinning 1e-4 \
  --maximum-weight 100 \
  --gpu-min-batch 64 \
  --cache-mode cold-each \
  --require-release-build \
  --minimum-speedup 5
```

共同条件：

```text
build                       Release
CUDA architecture           sm_89
primary                     electron
energy                      1 PeV
EM cut                      0.5 MeV
EM thinning                 1e-4
maximum weight              100
CPU numerical threads       1
process repetitions         5
```

CPU/CUDA 运行顺序逐轮交替，且不并行重叠。

## 3. 冷缓存真实性

五轮均满足：

```text
present_before_run     false
present_after_run      true
cache_bytes_after_run  539032
```

第 0 轮开始时目录中没有缓存；第 1–4 轮均先确认上一轮生成的缓存存在，再删除并
由新的 CUDA 进程重建。若任何一轮未生成缓存，runner 会 hard fail。

## 4. 结果

原始样本：

| repetition | CPU external / s | CUDA external / s | CPU shower / ms | CUDA shower / ms |
|---:|---:|---:|---:|---:|
| 0 | 86.643 | 10.688 | 83734.6 | 6618.7 |
| 1 | 87.873 | 10.619 | 84935.4 | 6615.6 |
| 2 | 87.797 | 10.595 | 84871.8 | 6598.4 |
| 3 | 86.605 | 10.579 | 83565.2 | 6599.4 |
| 4 | 85.763 | 10.739 | 82819.3 | 6617.6 |

ratio-of-medians：

| 指标 | CPU median | CUDA median | speedup |
|---|---:|---:|---:|
| external wall | 86.643 s | 10.619 s | **8.1591x** |
| common shower timing | 83734.6 ms | 6615.6 ms | **12.6571x** |

五个逐轮 external speedup 的最小值仍为 7.9860x；正式状态：

```text
status                   passed
minimum required speedup 5
```

与 Phase 49 热缓存结果比较：

| 模式 | external speedup | shower timing speedup |
|---|---:|---:|
| hot | 9.1949x | 13.7978x |
| cold-each | 8.1591x | 12.6571x |

因此即使把每次派生 Molière 表重建计入，1 PeV 纯 EM 端到端加速仍明显超过
原计划的 5x 门限。
