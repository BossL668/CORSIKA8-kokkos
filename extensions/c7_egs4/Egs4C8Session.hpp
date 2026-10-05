#pragma once
#include "Egs4MemoryBudget.hpp"
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/Philox.hpp>
#include <corsika/accelerator/em/common/EmThinning.hpp>
#include <corsika/accelerator/radio/common/Types.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace c7_egs4 { struct Tables; }
namespace c7_egs4::application {
// Host-facing B experimental API. C8 quantity/geometry/output headers remain
// in the host translation unit; native kernels/tables live in the session TU.
// No Fortran backend and no PROPOSAL-compatible capability impersonation.
// Create, run and destroy sessions on the main host thread. An existing
// application-owned Kokkos runtime is borrowed, never finalized by a session.
// Otherwise it is initialized once and retained until the last session dies;
// Kokkos cannot be reinitialized after that finalization.
struct Configuration {
  ::corsika::gpu::em::EnvironmentSnapshot environment;
  double earth_radius_m{},sterncor{10.},electron_mass_MeV{.51099895};
  double electron_total_cut_MeV{1.},photon_cut_MeV{.5},stepfc{1.};
  double electronuclear_threshold_MeV{152.},photonuclear_threshold_MeV{152.},muon_pair_threshold_MeV{422.};
  std::uint64_t seed{},shower_id{},first_child_id{};
  int queue_capacity{2048},maximum_waves{20000};
  bool resolve_rare_vertices{false};
  ::corsika::gpu::em::EmThinningConfig thinning{};
  ::corsika::gpu::radio::GpuRadioConfig radio{};
  ::corsika::gpu::em::GpuEmConfig::ProfileProjection profile{};
  bool reuse_queue_workspace{false};
  bool split_rare_kernels{false};
  bool retain_across_showers{false};
  bool resume_retained_shower{false}; // only seed/shower_id may change on this path
  // Isolated B opt-ins. Rho/omega/phi can decay natively; the caller still
  // needs hadron/muon transport and a many-hadron generator. No substitution.
  bool native_photonuclear_vertices{false};
  bool native_prompt_rho_decay{false};
  bool native_prompt_resonance_decay{false};
  double muon_mass_MeV{105.6583755},charged_pion_mass_MeV{139.57039},proton_mass_MeV{938.27208816};
  double neutron_mass_MeV{939.56542052},neutral_pion_mass_MeV{134.9768};
  double phi_mass_MeV{1019.461},omega_mass_MeV{782.65},rho_mass_MeV{775.26};
  double charged_kaon_mass_MeV{493.677},long_kaon_mass_MeV{497.611},short_kaon_mass_MeV{497.611},eta_mass_MeV{547.862};
  double average_atomic_weight{14.543};
  double air_composition[3]{.7847,.2105,.0048};
};
struct PromptDecayRecord {
  ::corsika::gpu::em::EmParticleState parent;
  ::corsika::gpu::em::EmParticleState daughters[3];
  int count{};
  double polarization_cosine[3]{},polarization_azimuth[3]{};
};
struct OutputCallbacks {
  std::function<void(::corsika::gpu::em::EmStepRecord const&)> step;
  // Alternative to individual step callbacks; native EM only, exported once.
  std::function<void(::corsika::gpu::em::GpuProfileResult const&)> profile;
  std::function<void(::corsika::gpu::em::RadioTrackRecord const&)> radio;
  // Called once, before the ordinary C8 end-of-shower serialization. CoREAS
  // contains E; ZHS contains A, differentiated by the existing C8 writer.
  std::function<void(::corsika::gpu::radio::GpuRadioWaveforms const&,std::uint64_t)> radio_waveforms;
  std::function<void(::corsika::gpu::em::ObservationRecord const&)> observation;
  // C7 early termination is separate until its deposition-bin conventions
  // are integrated. Never silently drop it or put it into continuous dE/dX.
  std::function<void(double weighted_energy_GeV,unsigned reason)> discarded;
  // Genealogy only, not another transport/deposit/radio contribution. Required
  // when prompt native meson decay is enabled so the intermediate ID is not lost.
  std::function<void(PromptDecayRecord const&)> prompt_decay;
};
struct RunStatistics {
  std::string execution_space;
  // children counts allocated histories, including recorded transient mesons.
  std::uint64_t waves{},children{},steps{},radio_tracks{},observations{},discards{};
  std::uint64_t injections{},host_requests{};
  std::uint64_t prompt_decays{};
  std::uint64_t thinning_hillas{},thinning_statistical{},thinning_removed{};
  double target_rest_energy_GeV{}; // weighted medium rest energy used by native nuclear vertices
  std::string radio_execution_space{"CPU callbacks"};
  ::corsika::gpu::radio::GpuRadioStatistics radio{};
  std::uint64_t radio_projection_batches{};
  double radio_kernel_ms{},radio_projection_wall_ms{};
  std::string profile_execution_space{"CPU callbacks"};
  std::uint64_t profile_steps{},profile_fixed_point_overflows{},profile_invalid_records{};
  std::uint64_t host_output_records{},host_output_bytes{};
  std::uint64_t queue_workspace_allocations{},queue_workspace_reuses{};
  std::uint64_t queue_processed_records{},queue_peak_active{};
  double queue_advance_wall_ms{}; // host elapsed, includes allocation/sync, not CUDA-only kernel time
  std::uint64_t queue_append_calls{},queue_append_records{},queue_append_copied_active_records{};
  double queue_append_wall_ms{};
  std::uint64_t queue_split_rare_waves{},queue_split_rare_records{};
};
enum class HostRequestKind {
  muon_transport, photonuclear, hadron_transport, c7_vector_meson_decay, photonuclear_many_hadrons
};
struct HostRequest {
  HostRequestKind kind{};
  ::corsika::gpu::em::EmParticleState particle;
  bool virtual_photon{}; // vertex originated from ELNUCL (also true on its generated hadrons)
  int target_atomic_number{};
  double polarization_cosine{},polarization_azimuth{};
  int photonuclear_branch{},target_pdg{};
  // Many-hadron requests carry the already selected SDPM air nucleus in
  // target_pdg. Neither branch nor target is to be resampled by the host.
  ::corsika::gpu::em::RandomNumberKey random_key,hadron_random_key;
};
struct WaveResult {
  std::vector<HostRequest> host_requests;
  std::size_t active_particles{};
  double target_rest_energy_GeV{}; // counted once per native vertex, already weighted
};
class Session {
  struct Impl;std::unique_ptr<Impl> impl_;
public:
  explicit Session(std::string const& table_path);
  // Production builds supply the table bytes at build time, not per shower.
  explicit Session(::c7_egs4::Tables tables);
  ~Session();
  Session(Session const&)=delete;Session& operator=(Session const&)=delete;
  // Query the selected CUDA device after runtime initialization. The estimate
  // includes simultaneous old/after/next queues, vertex/output buffers and
  // scan offsets; host-only builds reject this GPU-specific operation.
  GpuMemoryBudget gpuMemoryBudget(double fraction,::corsika::gpu::radio::GpuRadioConfig const& radio={})const;
  RunStatistics run(Configuration const&,std::vector<::corsika::gpu::em::EmParticleState> const&,
      OutputCallbacks const&,bool host_reference=false);
  // Incremental API for the real C8 HybridCascade router. Native state is
  // retained between waves; append() accepts NEW CPU-stack histories only.
  void begin(Configuration const&,OutputCallbacks const&,bool host_reference=false);
  // Reuse the exact initialized geometry/physics/storage. No new configuration
  // is accepted: only the event keys and consumers are replaced.
  void resume(std::uint64_t seed,std::uint64_t shower_id,OutputCallbacks const&);
  bool hasRetainedShower()const;
  void append(std::vector<::corsika::gpu::em::EmParticleState> const&);
  std::size_t activeParticles()const;
  RunStatistics progressStatistics()const;
  WaveResult advance(std::uint64_t first_reserved_child,std::uint64_t reserved_count);
  // Only verifies that the EM queue drained. The caller must also consume all
  // returned host requests and drain its hadronic stack before shower end.
  RunStatistics finishEMQueue();
};
} // namespace c7_egs4::application
