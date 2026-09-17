#pragma once
#include <corsika/geometry/terrain/FlatTerrainData.hpp>
#include <corsika/geometry/terrain/DemCoverageData.hpp>
#include <corsika/geometry/interfaces/MaterialInterface.hpp>
#include <corsika/modules/radio/interface/Types.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
#include <memory>
#include <string>

namespace corsika::interfaces {
  namespace em = gpu::em;
  // Immutable calculator bank; its index is not a geometric region ID.
  // Composition comes from actual calculators, never from a geometry name.
  struct EmMaterial {
    em::tables::ProposalNativeTableSet table;
    em::tables::ProposalNativeAuxData auxiliary;
    em::EnvironmentSnapshot environment;
  };
  struct EmConfig {
    int threads{1}, device{0};
    std::size_t batch_size{64};
    // Zero uses batch_size. Otherwise decouple CPU staging/interleave from
    // the wider resident front; the FIFO and per-particle physics are unchanged.
    std::size_t resident_wavefront_capacity{0};
    // Zero keeps the stateless reference path; otherwise bounds the resident FIFO.
    std::size_t resident_capacity{0};
    // Bounded device output ledger, downloaded only at a cascade checkpoint.
    std::size_t resident_record_capacity{4096};
    std::size_t maximum_device_bytes{128u*1024u*1024u};
    double emcut_MeV{0.5};
    std::uint64_t seed{}, shower_id{};
    em::EmThinningConfig thinning;
    bool energy_ledger{false}; // read-only diagnostic; no additional random draws
    double maximum_step_m{HUGE_VAL}, transport_window_s{HUGE_VAL};
    interfaces::MaterialInterface interface;
    terrain::coverage::Data coverage;
    corsika::radio::interface::Config radio;
  };
  enum class EmOutcome : std::uint32_t { Continuation, Children, Cut, Escape, Fallback, Error, WindowEscape, DomainEscape };
  struct EmStep {
    em::EmParticleState start{}, end{}, children[3]{};
    em::ProposalFallbackEvent fallback{};
    EmOutcome outcome{EmOutcome::Error};
    std::uint32_t child_count{}, has_track{}, crossed_material{}, error{};
    std::uint32_t domain_edge{UINT32_MAX};
    std::int32_t process_id{};
    double distance_m{}, grammage_g_cm2{}, deposited_GeV{}, binding_energy_GeV{};
    double unthinned_secondary_total_GeV{}, weighted_thinning_delta_GeV{};
    // Actual transport midpoint, captured before scattering/final-state changes.
    double path_midpoint_m[3]{};
  };
} // namespace corsika::interfaces
