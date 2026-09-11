#pragma once
#include <corsika/geometry/terrain/FlatTerrainData.hpp>
#include <corsika/geometry/interfaces/MaterialInterface.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
#include <memory>
#include <string>

namespace corsika::terrain {
  namespace em = gpu::em;
  // Immutable calculator bank; its index is not a geometric region ID.
  // Composition comes from actual calculators, never from a geometry name.
  struct TerrainEmMaterial {
    em::tables::ProposalNativeTableSet table;
    em::tables::ProposalNativeAuxData auxiliary;
    em::EnvironmentSnapshot environment;
  };
  struct TerrainEmConfig {
    int threads{1}, device{0};
    std::size_t batch_size{64};
    std::size_t maximum_device_bytes{128u*1024u*1024u};
    double emcut_MeV{0.5};
    std::uint64_t seed{}, shower_id{};
    em::EmThinningConfig thinning;
    bool energy_ledger{false}; // read-only diagnostic; no additional random draws
    double maximum_step_m{HUGE_VAL}, transport_window_s{HUGE_VAL};
    interfaces::MaterialInterface interface;
  };
  enum class TerrainEmOutcome : std::uint32_t { Continuation, Children, Cut, Escape, Fallback, Error, WindowEscape };
  struct TerrainEmStep {
    em::EmParticleState start{}, end{}, children[3]{};
    em::ProposalFallbackEvent fallback{};
    TerrainEmOutcome outcome{TerrainEmOutcome::Error};
    std::uint32_t child_count{}, has_track{}, crossed_material{}, error{};
    std::int32_t process_id{};
    double distance_m{}, grammage_g_cm2{}, deposited_GeV{}, binding_energy_GeV{};
    double unthinned_secondary_total_GeV{}, weighted_thinning_delta_GeV{};
    // Actual transport midpoint, captured before scattering/final-state changes.
    double path_midpoint_m[3]{};
  };
  // EmParticleState.medium_id denotes the logical region within this session;
  // interface.material(region) selects the immutable physics bank. Internal
  // atmospheric layer IDs must not replace logical interface ownership.
  // Bounded, synchronous validation pipeline. Device geometry/tables are retained;
  // bounded track batches return to the host. This is not the production resident
  // scheduler and does not yet implement cross-interface radio propagation.
  class TerrainEmSession {
   public:
    TerrainEmSession(FlatTerrainData const&, std::vector<TerrainEmMaterial> const&, TerrainEmConfig const&);
    ~TerrainEmSession();
    TerrainEmSession(TerrainEmSession const&) = delete;
    TerrainEmSession& operator=(TerrainEmSession const&) = delete;
    std::vector<TerrainEmStep> advance(std::vector<em::EmParticleState> const&, std::uint64_t first_child_history);
    std::size_t deviceBytes() const;
    std::string executionSpace() const;
   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };
}
