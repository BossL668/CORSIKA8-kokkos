/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/egs4/MemoryBudget.hpp>
#include <corsika/kokkos/egs4/Data.hpp>
#include <corsika/kokkos/em/random/Philox.hpp>
#include <corsika/kokkos/em/algorithms/EmThinning.hpp>
#include <corsika/kokkos/radio/RadioConfig.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace corsika::kokkos::egs4 {
  struct Tables;
}
namespace corsika::kokkos::egs4::application {
  // Host-facing EGS4 session. C8 quantity/geometry/output headers remain
  // in the host translation unit; native kernels/tables live in the session TU.
  // No Fortran backend and no PROPOSAL-compatible capability impersonation.
  // Create, run and destroy sessions on the main host thread. An existing
  // application-owned Kokkos runtime is borrowed, never finalized by a session.
  // Otherwise it is initialized once and retained until the last session dies;
  // Kokkos cannot be reinitialized after that finalization.
  struct Configuration {
    ::corsika::kokkos::egs4::AtmosphereData environment_;
    double earthRadiusM_{}, sterncor_{10.}, electronMassMeV_{.51099895};
    double electronTotalCutMeV_{1.}, photonCutMeV_{.5}, stepfc_{1.};
    double electronuclearThresholdMeV_{152.}, photonuclearThresholdMeV_{152.},
        muonPairThresholdMeV_{422.};
    std::uint64_t seed_{}, showerId_{}, firstChildId_{};
    int queueCapacity_{2048}, maximumWaves_{20000};
    bool resolveRareVertices_{false};
    ::corsika::kokkos::EmThinningConfig thinning_{};
    ::corsika::kokkos::radio::RadioConfig radio_{};
    ::corsika::kokkos::ProfileProjectionConfig profile_{};
    bool reuseQueueWorkspace_{false};
    bool splitRareKernels_{false};
    bool retainAcrossShowers_{false};
    bool resumeRetainedShower_{false}; // only seed/shower_id may change on this path
    // Optional native rare vertices. Rho/omega/phi can decay natively; the caller still
    // needs hadron/muon transport and a many-hadron generator. No substitution.
    bool nativePhotonuclearVertices_{false};
    bool nativePromptRhoDecay_{false};
    bool nativePromptResonanceDecay_{false};
    double muonMassMeV_{105.6583755}, chargedPionMassMeV_{139.57039},
        protonMassMeV_{938.27208816};
    double neutronMassMeV_{939.56542052}, neutralPionMassMeV_{134.9768};
    double phiMassMeV_{1019.461}, omegaMassMeV_{782.65}, rhoMassMeV_{775.26};
    double chargedKaonMassMeV_{493.677}, longKaonMassMeV_{497.611},
        shortKaonMassMeV_{497.611}, etaMassMeV_{547.862};
    double averageAtomicWeight_{14.543};
    double airComposition_[3]{.7847, .2105, .0048};
  };
  struct PromptDecayRecord {
    ::corsika::kokkos::ParticleState parent_;
    ::corsika::kokkos::ParticleState daughters_[3];
    int count_{};
    double polarizationCosine_[3]{}, polarizationAzimuth_[3]{};
  };
  struct OutputCallbacks {
    std::function<void(::corsika::kokkos::StepRecord const&)> step_;
    // Alternative to individual step callbacks; native EM only, exported once.
    std::function<void(::corsika::kokkos::ProfileResult const&)> profile_;
    std::function<void(::corsika::kokkos::RadioTrackRecord const&)> radio_;
    // Called once, before the ordinary C8 end-of-shower serialization. CoREAS
    // contains E; ZHS contains A, differentiated by the existing C8 writer.
    std::function<void(::corsika::kokkos::radio::RadioWaveforms const&, std::uint64_t)>
        radioWaveforms_;
    std::function<void(::corsika::kokkos::ObservationRecord const&)> observation_;
    // C7 early termination is separate until its deposition-bin conventions
    // are integrated. Never silently drop it or put it into continuous dE/dX.
    std::function<void(double weighted_energy_GeV, unsigned reason)> discarded_;
    // Genealogy only, not another transport/deposit/radio contribution. Required
    // when prompt native meson decay is enabled so the intermediate ID is not lost.
    std::function<void(PromptDecayRecord const&)> promptDecay_;
  };
  struct RunStatistics {
    std::string executionSpace_;
    // children counts allocated histories, including recorded transient mesons.
    std::uint64_t waves_{}, children_{}, steps_{}, radioTracks_{}, observations_{},
        discards_{};
    std::uint64_t injections_{}, hostRequests_{};
    std::uint64_t promptDecays_{};
    std::uint64_t thinningHillas_{}, thinningStatistical_{}, thinningRemoved_{};
    double targetRestEnergyGeV_{}; // weighted medium rest energy used by native nuclear
                                   // vertices
    std::string radioExecutionSpace_{"CPU callbacks"};
    ::corsika::kokkos::radio::RadioStatistics radio_{};
    std::uint64_t radioProjectionBatches_{};
    double radioKernelMs_{}, radioProjectionWallMs_{};
    std::string profileExecutionSpace_{"CPU callbacks"};
    std::uint64_t profileSteps_{}, profileFixedPointOverflows_{},
        profileInvalidRecords_{};
    std::uint64_t hostOutputRecords_{}, hostOutputBytes_{};
    std::uint64_t queueWorkspaceAllocations_{}, queueWorkspaceReuses_{};
    std::uint64_t queueProcessedRecords_{}, queuePeakActive_{};
    double queueAdvanceWallMs_{}; // host elapsed, includes allocation/sync, not CUDA-only
                                  // kernel time
    std::uint64_t queueAppendCalls_{}, queueAppendRecords_{},
        queueAppendCopiedActiveRecords_{};
    double queueAppendWallMs_{};
    std::uint64_t queueSplitRareWaves_{}, queueSplitRareRecords_{};
  };
  enum class HostRequestKind {
    muon_transport,
    photonuclear,
    hadron_transport,
    c7_vector_meson_decay,
    photonuclear_many_hadrons
  };
  struct HostRequest {
    HostRequestKind kind_{};
    ::corsika::kokkos::ParticleState particle_;
    bool virtualPhoton_{}; // vertex originated from ELNUCL (also true on its generated
                           // hadrons)
    int targetAtomicNumber_{};
    double polarizationCosine_{}, polarizationAzimuth_{};
    int photonuclearBranch_{}, targetPdg_{};
    // Many-hadron requests carry the already selected SDPM air nucleus in
    // target_pdg. Neither branch nor target is to be resampled by the host.
    ::corsika::kokkos::RandomNumberKey randomKey_, hadronRandomKey_;
  };
  struct WaveResult {
    std::vector<HostRequest> hostRequests_;
    std::size_t activeParticles_{};
    double targetRestEnergyGeV_{}; // counted once per native vertex, already weighted
  };
  class Session {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit Session(std::string const& table_path);
    // Production builds supply the table bytes at build time, not per shower.
    explicit Session(::corsika::kokkos::egs4::Tables tables);
    ~Session();
    /** C7 EGSIN2 correction associated with the loaded table cutoff. */
    double sternheimerCorrection() const;
    Session(Session const&) = delete;
    Session& operator=(Session const&) = delete;
    // Query the selected CUDA device after runtime initialization. The estimate
    // includes simultaneous old/after/next queues, vertex/output buffers and
    // scan offsets; host-only builds reject this GPU-specific operation.
    GpuMemoryBudget gpuMemoryBudget(
        double fraction, ::corsika::kokkos::radio::RadioConfig const& radio = {}) const;
    RunStatistics run(Configuration const&,
                      std::vector<::corsika::kokkos::ParticleState> const&,
                      OutputCallbacks const&, bool host_reference = false);
    // Incremental API for the real C8 HybridCascade router. Native state is
    // retained between waves; append() accepts NEW CPU-stack histories only.
    void begin(Configuration const&, OutputCallbacks const&, bool host_reference = false);
    // Reuse the exact initialized geometry/physics/storage. No new configuration
    // is accepted: only the event keys and consumers are replaced.
    void resume(std::uint64_t seed, std::uint64_t shower_id, OutputCallbacks const&);
    bool hasRetainedShower() const;
    void append(std::vector<::corsika::kokkos::ParticleState> const&);
    std::size_t activeParticles() const;
    RunStatistics progressStatistics() const;
    WaveResult advance(std::uint64_t first_reserved_child, std::uint64_t reserved_count);
    // Only verifies that the EM queue drained. The caller must also consume all
    // returned host requests and drain its hadronic stack before shower end.
    RunStatistics finishEMQueue();
  };
} // namespace corsika::kokkos::egs4::application
