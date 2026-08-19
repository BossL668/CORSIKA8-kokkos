# CUDA/FLUKA 混合级联文档索引

本目录保存当前生产接口、物理边界、验收结论以及按实施顺序记录的阶段日志。
新用户应先阅读“当前文档”；`phase_*` 文件主要用于追踪设计决策、回归证据和
历史性能，不应单独作为当前生产能力声明。

## 当前文档

| 文档 | 内容 |
|---|---|
| [命令行参数参考](cli_reference.md) | 新增 executable、表格工具、replay、FLUKA worker 和验收 runner 的参数、默认值与约束 |
| [生产使用指南](cuda_em_backend_user_guide.md) | 构建、物理表、运行、fallback、输出和生产流程 |
| [最终验收报告](final_acceptance_report.md) | 当前验收矩阵和仍需保留的限制 |
| [原始计划逐条审计](original_plan_requirement_audit.md) | 实施计划要求与当前实现的映射 |
| [CoREAS/ZHS 在线累计](coreas_zhs_online_accumulation.md) | 射电计算与粒子输运的执行时序 |
| [PROPOSAL 与 CUDA 插值](proposal_and_cuda_interpolation.md) | 两类插值的来源、误差和物理含义 |
| [介质 YAML 与自动制表](phase_92_medium_yaml_content_addressed_table_preparation.md) | 介质 schema、规范化哈希、缓存查找、自动生成和 10¹⁹ eV 示例 |
| [验证工具](../../validation/gpu_em/README.md) | 性能、系综、radio、replay 和 scaling-law 脚本 |

## 文档约定

- 每个 Markdown 文件只使用一个一级标题。
- 命令使用带语言标识的 fenced code block，例如 `bash`、`text`、`yaml`。
- 参数名、文件名、类名和字段名使用反引号。
- 数值必须带单位；能量同时注明 MeV、GeV 或 eV，距离注明 m，时间注明 ns 或 s。
- “通过”只表示文档中明确给出的配置、样本数、artifact identity 和门禁。
- production、validation/debug 和 experimental 参数必须明确区分。
- 绝对本地路径只作为历史证据；可迁移命令使用环境变量或占位路径。
- 新结论写入当前文档；阶段日志保持为不可回写的实施记录。

## 阶段记录

### 阶段 1–13：标量拆分、表格和第一条 GPU 路由

- [阶段 1：标量单粒子推进器](phase_01_scalar_cascade_stepper.md)
- [阶段 2：CPU-only HybridCascade](phase_02_cpu_only_hybrid_cascade.md)
- [阶段 3：CUDA wavefront 基础设施](phase_03_cuda_wavefront_infrastructure.md)
- [阶段 4：toy 混合路由](phase_04_toy_hybrid_route.md)
- [阶段 5：PROPOSAL 相互作用拆分](phase_05_proposal_interaction_split.md)
- [阶段 6：版本化相互作用率表](phase_06_proposal_rate_tables.md)
- [阶段 7：随机能损 inverse CDF](phase_07_inverse_cdf_tables.md)
- [阶段 8：rate table 上传与设备查询](phase_08_cuda_rate_table_upload.md)
- [阶段 9：GPU 相互作用选择](phase_09_interaction_selection.md)
- [阶段 10：photon-pair 末态](phase_10_photon_pair_final_state.md)
- [阶段 11：photon-pair LPM](phase_11_photon_pair_lpm.md)
- [阶段 12：球形大气 photon transport](phase_12_spherical_photon_transport.md)
- [阶段 13A：可复用 device workspace](phase_13_device_workspace.md)
- [阶段 13B：常驻 photon pipeline](phase_13b_resident_photon_pipeline.md)

### 阶段 14–24：电磁物理和生产输出

- [阶段 14：Molière、磁场和物理路由](phase_14_moliere_magnetic_physical_hybrid.md)
- [阶段 15：精确 fallback、Compton 和 photoelectric](phase_15_exact_fallback_compton_photoelectric.md)
- [阶段 16：positron annihilation](phase_16_gpu_positron_annihilation.md)
- [阶段 17：discrete ionization](phase_17_gpu_discrete_ionization.md)
- [阶段 18：常驻 charged-lepton cascade](phase_18_resident_lepton_cascade.md)
- [阶段 19：range-safe resident router](phase_19_range_safe_resident_router.md)
- [阶段 20：electron-pair production](phase_20_gpu_electron_pair_production.md)
- [阶段 21：GPU EM thinning](phase_21_gpu_em_thinning.md)
- [阶段 22：生产应用、streaming output 和严格失败状态](phase_22_production_application_output.md)
- [阶段 23：ParticleCut 和小批次调度](phase_23_particle_cut_and_small_batch_scheduler.md)
- [阶段 24：Molière 插值和 endpoint summary](phase_24_moliere_interpolation_and_endpoint_summary.md)

### 阶段 25–38：常驻队列、radio 和 kernel 优化

- [阶段 25：resident GPU CoREAS/ZHS](phase_25_resident_gpu_radio.md)
- [阶段 26：resident profile 和 workspace safety](phase_26_resident_profile_and_workspace_safety.md)
- [阶段 27：cross-species device queues](phase_27_resident_cross_species_queues.md)
- [阶段 28：融合 charged-secondary summary](phase_28_fused_charged_secondary_summary.md)
- [阶段 29：延迟 lepton final state](phase_29_deferred_lepton_final_state_summary.md)
- [阶段 30：延迟 photon final state](phase_30_deferred_photon_final_state_summary.md)
- [阶段 31：selection count 驱动 transport](phase_31_selection_transport_device_count.md)
- [阶段 32：transport/vertex device counts](phase_32_transport_vertex_device_counts.md)
- [阶段 33：UHE 验收和外部模型限制](phase_33_uhe_acceptance_and_external_model_limits.md)
- [阶段 34：lepton pipeline profiling 和 Epair rejection](phase_34_lepton_pipeline_profiling_and_epair_rejection.md)
- [阶段 35：Molière Newton 初值](phase_35_moliere_newton_initialization.md)
- [阶段 36：rate 共享和 endpoint fusion](phase_36_vertex_rate_and_endpoint_fusion.md)
- [阶段 37：brems LPM 设备预计算](phase_37_brems_lpm_preparation.md)
- [阶段 38：Molière mixture 初值](phase_38_moliere_mixture_initial_guess.md)

### 阶段 39–53：性能、能量账本和系综验收

- [阶段 39：纯 EM 性能验收](phase_39_pure_em_performance_acceptance.md)
- [阶段 40：系综验证和 tail budget](phase_40_ensemble_validation_and_tail_budget.md)
- [阶段 41：严格能量账本](phase_41_strict_energy_ledger.md)
- [阶段 42：五次重复性能门禁](phase_42_five_repetition_performance_gate.md)
- [阶段 43：相同轨迹 radio 验收](phase_43_identical_track_radio_acceptance.md)
- [阶段 44：过程序列兼容性门禁](phase_44_process_sequence_compatibility_gate.md)
- [阶段 45：稳定 wavefront bucketing](phase_45_stable_wavefront_bucketing.md)
- [阶段 46：低能 CPU spill](phase_46_low_energy_cpu_spill.md)
- [阶段 47：强子混合和 fallback provenance](phase_47_hadronic_hybrid_and_named_fallback_provenance.md)
- [阶段 48：1 PeV、200+200 事例](phase_48_1pev_200_event_acceptance_result.md)
- [阶段 49：Release 和五次性能验收](phase_49_release_build_and_five_run_performance_acceptance.md)
- [阶段 50：CPU step 方向等价与复验](phase_50_cpu_step_direction_equivalence_and_1pev_revalidation.md)
- [阶段 51：1 TeV photon 正式系综](phase_51_photon_1tev_formal_ensemble.md)
- [阶段 52：可复用 backend 生命周期](phase_52_reusable_backend_lifecycle.md)
- [阶段 53：80° 倾斜 shower](phase_53_inclined_shower_acceptance.md)

### 阶段 54–73：参数边界、UHE、强子和安装

- [阶段 54：minimum batch 诊断](phase_54_minimum_batch_diagnostic.md)
- [阶段 55：5 MeV cut](phase_55_cut5_table_and_acceptance.md)
- [阶段 56：50 MeV cut](phase_56_cut50_table_and_acceptance.md)
- [阶段 57：大气边界和 strict fallback](phase_57_boundary_guard_and_strict_fallback.md)
- [阶段 58：50 MeV 扩展系综](phase_58_cut50_expanded_ensemble.md)
- [阶段 59：Epair fallback 授权](phase_59_epair_fallback_authorization.md)
- [阶段 60：无 thinning 倾斜系综](phase_60_unthinned_ensemble.md)
- [阶段 61–64：UHE 正式系综](phase_61_64_uhe_formal_ensembles.md)
- [阶段 65–69：radio 生命周期和强子系综](phase_65_69_radio_and_hadronic_ensembles.md)
- [阶段 70：冷缓存性能](phase_70_cold_cache_performance.md)
- [阶段 71–72：默认构建和安装导出](phase_71_72_default_build_and_install.md)
- [阶段 73：运行时 fallback 边界回归](phase_73_runtime_fallback_boundary_regression.md)

阶段编号 74 没有独立文档；后续编号沿用实施时的实验编号，未重新编号。

### 阶段 75–83：无损 fallback、IGRF13 和原版 CPU/CUDA 对照

- [阶段 75：无损 photohadronic fallback 和 provenance](phase_75_lossless_photohadronic_fallback_and_validation_provenance.md)
- [阶段 76：lazy fallback 和性能复验](phase_76_lazy_fallback_and_current_build_performance.md)
- [阶段 77：threshold photoproduction](phase_77_threshold_photoproduction_and_final_performance.md)
- [阶段 78：终端观测面和 SOPHIA threshold](phase_78_terminal_observation_sophia_threshold_and_final_matrix.md)
- [阶段 79：IGRF13、CUDA replay 和 radio statistics](phase_79_igrf13_cuda_replay_and_radio_statistics.md)
- [阶段 80：原版 CPU/CUDA shower 和 radio](phase_80_original_cpu_cuda_validation.md)
- [阶段 81：统计复验和当前结论](phase_81_cpu_cuda_validation_sampling_and_verdict.md)
- [阶段 82：exact legacy decision replay](phase_82_exact_legacy_event_cuda_decision_replay.md)
- [阶段 83：shower feature distributions](phase_83_shower_feature_distribution_validation.md)

### 阶段 84–91：FLUKA 进程池、μ 子和配置驱动验收

- [阶段 84：恢复 FLUKA 和固定 seed baseline](phase_84_fluka_restore_hadronic_work_classes_and_fixed_seed_baseline.md)
- [阶段 85：进程隔离 FLUKA batch worker](phase_85_process_isolated_fluka_batch_worker.md)
- [阶段 86：生产 FLUKA process pool](phase_86_production_fluka_classified_process_pool.md)
- [阶段 87：CUDA μ 子和性能](phase_87_cuda_muon_config_adapter_performance_and_current_acceptance.md)
- [阶段 88：确定性强子调度和 radio/profile fusion](phase_88_deterministic_hadronic_scheduler_atomic_profile_radio_fusion_and_emthin1e-6.md)
- [阶段 89：cost-triggered batches 和 bulk IPC](phase_89_cost_triggered_hadronic_batches_and_bulk_ipc.md)
- [阶段 90：配置驱动 1 PeV 验收](phase_90_config_driven_1pev_cpu_cuda_acceptance.md)
- [阶段 91：1 PeV thinning 边界、radio 和能量账本](phase_91_1pev_emthin_boundary_full_radio_and_energy_ledger.md)

### 阶段 92：介质配置和内容寻址制表

- [阶段 92：介质 YAML、规范化哈希与自动物理表准备](phase_92_medium_yaml_content_addressed_table_preparation.md)

### 阶段 93–102：边界、profile、cut 与 beta4 CPU/GPU 输出对齐

- [阶段 93：磁场层边界和 500 例验收](phase_93_magnetic_layer_guard_and_500_event_acceptance.md)
- [阶段 94：μ 子 parent production profile 修复](phase_94_muon_parent_production_profile_fix.md)
- [阶段 95：FLUKA parent `SIGALRM` 修复](phase_95_fluka_parent_sigalrm_fix.md)
- [阶段 96：PROPOSAL cut 拆分和 profile reset 修复](phase_96_split_proposal_cut_and_profile_reset_fix.md)
- [阶段 97：beta4 连续输运与 CPU `Step` 弦轨迹对齐](phase_97_beta4_cpu_step_chord_grammage.md)
- [阶段 98：CUDA 观测平面与 CPU 几何对齐](phase_98_beta4_cpu_observation_plane_alignment.md)
- [阶段 99：GPU 首相互作用与 CPU `InteractionWriter` 对齐](phase_99_beta4_gpu_first_interaction_writer_alignment.md)
- [阶段 100：GPU 10 ms `ParticleCut` 与 CPU 时间语义对齐](phase_100_beta4_particle_cut_time_alignment.md)
- [阶段 101：强制初级顶点与全过程序列兼容性门禁](phase_101_beta4_forced_primary_and_process_compatibility_gate.md)
- [阶段 102：beta4 与 beta2 的 10 PeV CUDA 性能对比](phase_102_beta4_beta2_10pev_performance.md)

### 阶段 103–113：beta4 再审计、replay、系综与 CUDA profiling/优化

- [阶段 103：GPU/CPU 路径再审计和 100 TeV campaign](phase_103_beta4_gpu_cpu_path_reaudit_and_100tev_campaign.md)
- [阶段 104：CPU 500 / CUDA 500 验收](phase_104_beta4_cpu500_cuda500_acceptance_CN.md)
- [阶段 105：逐过程 CPU/GPU 路径复验](phase_105_beta4_cpu_gpu_process_path_revalidation_CN.md)
- [阶段 106：强制 CPU/CUDA shower replay](phase_106_beta4_forced_cpu_cuda_shower_replay_CN.md)
- [阶段 107：相同随机数第一次分叉诊断](phase_107_beta4_same_random_first_divergence_CN.md)
- [阶段 108：CPU 1000 / CUDA 1000 验收](phase_108_beta4_cpu1000_cuda1000_acceptance_CN.md)
- [阶段 109：EM profile 不确定度预算](phase_109_em_profile_uncertainty_budget_CN.md)
- [阶段 110：P0 CUDA 计时语义与等待分解](phase_110_beta4_p0_cuda_timing_semantics_CN.md)
- [阶段 111：P1 radio track 预计算与 observer tiling](phase_111_beta4_p1_radio_track_precompute_observer_tiling_CN.md)
- [阶段 112：P2 radio 8×32 tile 优化](phase_112_beta4_p2_radio_8x32_tiling_CN.md)
- [阶段 113：P2 生产显存回退与 100 TeV 性能验收](phase_113_beta4_p2_production_memory_fallback_and_100tev_acceptance_CN.md)
