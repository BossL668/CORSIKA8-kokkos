/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Fixed production 41cdc543: original function bodies and record names.
// Namespace/include/annotation adapters only. Existing independent query oracles
// supply the original continuous and discrete calculations, never receiver code.
#include "../continuous/ContinuousReference.hpp"
#include "../discrete/DiscreteReference.hpp"
#include <corsika/detail/framework/random/random_iterator/detail/Random123/philox.h>
namespace stochastic_reference {
  using continuous_reference::EmParticleState;
  using continuous_reference::ProposalFallbackEvent;
  using continuous_reference::ProposalFallbackReason;
  using continuous_reference::proposalFallbackReason;
  namespace accelerator::scalar_constants {
    inline constexpr double SpeedOfLightMPerS = 299792458.;
  }
  struct PhiloxCounter {
    std::uint32_t words[4]{};
  };
  struct PhiloxKey {
    std::uint32_t words[2]{};
  };
  struct RandomNumberKey {
    std::uint64_t seed{};
    std::uint64_t shower_id{};
    std::uint64_t history_id{};
    std::uint64_t step_id{};
    std::uint32_t process_id{};
    std::uint64_t draw_id{};
  };
  namespace detail {
    CORSIKA_HOST_DEVICE inline std::uint64_t splitmix64(std::uint64_t value) {
      value += 0x9e3779b97f4a7c15ULL;
      value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
      value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31U);
    }
  } // namespace detail
  CORSIKA_HOST_DEVICE inline PhiloxCounter philox4x32_10(PhiloxCounter const& counter,
                                                         PhiloxKey const& key) {
    random_iterator_r123::Philox4x32::ctr_type r123_counter{
        {counter.words[0], counter.words[1], counter.words[2], counter.words[3]}};
    random_iterator_r123::Philox4x32::key_type r123_key{{key.words[0], key.words[1]}};
    auto const output = random_iterator_r123::Philox4x32{}(r123_counter, r123_key);
    return {{output.v[0], output.v[1], output.v[2], output.v[3]}};
  }

  CORSIKA_HOST_DEVICE inline PhiloxCounter randomWords(RandomNumberKey const& input) {
    PhiloxCounter const counter{{
        static_cast<std::uint32_t>(input.history_id),
        static_cast<std::uint32_t>(input.history_id >> 32U),
        static_cast<std::uint32_t>(input.step_id),
        static_cast<std::uint32_t>(input.step_id >> 32U),
    }};

    auto key_material = detail::splitmix64(input.seed);
    key_material = detail::splitmix64(key_material ^ input.shower_id);
    key_material =
        detail::splitmix64(key_material ^ static_cast<std::uint64_t>(input.process_id));
    key_material = detail::splitmix64(key_material ^ input.draw_id);
    PhiloxKey const key{{
        static_cast<std::uint32_t>(key_material),
        static_cast<std::uint32_t>(key_material >> 32U),
    }};
    return philox4x32_10(counter, key);
  }

  CORSIKA_HOST_DEVICE inline double uniformOpen01(RandomNumberKey const& input,
                                                  std::uint32_t lane = 0) {
    auto const words = randomWords(input);
    auto const value = words.words[lane & 3U];
    return (static_cast<double>(value) + 0.5) * (1. / 4294967296.);
  }
  inline constexpr std::uint32_t RandomDomainVersion = 2;
  inline constexpr std::uint32_t InteractionDistanceRandomProcessId = 0x454d0001U;
  inline constexpr std::uint32_t InteractionColumnRandomProcessId = 0x454d0002U;
  inline constexpr std::uint32_t ProposalSelectionRandomProcessId = 0x454d0004U;
  inline constexpr std::uint64_t InteractionDistanceDrawId = 0;
  inline constexpr std::uint64_t InteractionColumnDrawId = 0;
  inline constexpr std::uint64_t ProposalSelectionDrawId = 0;
  inline constexpr std::uint64_t InteractionLossDrawId = 0;
  inline constexpr std::uint32_t ContinuousScatteringRandomProcessId = 0x454d0003U;
  inline constexpr std::uint64_t MoliereFirstAngleDrawId = 0;
  inline constexpr std::uint64_t MoliereSecondAngleDrawId = 1;
  inline constexpr std::uint64_t MoliereAzimuthDrawId = 2;
  inline constexpr std::uint32_t MuonDecayRandomProcessId = 0x4d554445U;
  inline constexpr std::uint64_t MuonDecayDrawId = 0;
  inline constexpr double MuonMeanLifetimeS = 2.196981e-6;
  inline constexpr double MuonDecaySpeedOfLightMPerS =
      accelerator::scalar_constants::SpeedOfLightMPerS;
  constexpr bool randomProcessDomainsAreUnique() {
    constexpr std::uint32_t domains[]{
        InteractionDistanceRandomProcessId, InteractionColumnRandomProcessId,
        ProposalSelectionRandomProcessId, ContinuousScatteringRandomProcessId,
        MuonDecayRandomProcessId};
    for (std::uint32_t i = 0; i < 5; ++i)
      for (std::uint32_t j = i + 1; j < 5; ++j)
        if (domains[i] == domains[j]) return false;
    return true;
  }
  enum class EmPid : std::int32_t {
    Electron = 11,
    Positron = -11,
    MuonMinus = 13,
    MuonPlus = -13,
    Photon = 22,
  };
  inline constexpr double ParticleCutMaximumTimeS = 10.e-3;
  CORSIKA_HOST_DEVICE inline constexpr bool isElectronOrPositronPid(
      std::int32_t const pid) {
    return pid == static_cast<std::int32_t>(EmPid::Electron) ||
           pid == static_cast<std::int32_t>(EmPid::Positron);
  }
  CORSIKA_HOST_DEVICE inline constexpr bool isMuonPid(std::int32_t const pid) {
    return pid == static_cast<std::int32_t>(EmPid::MuonMinus) ||
           pid == static_cast<std::int32_t>(EmPid::MuonPlus);
  }
  CORSIKA_HOST_DEVICE inline constexpr bool isChargedLeptonPid(std::int32_t const pid) {
    return isElectronOrPositronPid(pid) || isMuonPid(pid);
  }
  CORSIKA_HOST_DEVICE inline constexpr bool exceedsParticleCutTime(double const time_s) {
    return time_s > ParticleCutMaximumTimeS;
  }
  inline constexpr std::int32_t BremsProcessId = 1000000002;
  inline constexpr std::int32_t IonizationProcessId = 1000000003;
  inline constexpr std::int32_t ElectronPairProcessId = 1000000004;
  inline constexpr std::int32_t PhotonuclearProcessId = 1000000005;
  inline constexpr std::int32_t MuonPairProcessId = 1000000006;
  inline constexpr std::int32_t HadronsProcessId = 1000000007;
  inline constexpr std::int32_t WeakInteractionProcessId = 1000000009;
  inline constexpr std::int32_t ComptonProcessId = 1000000010;
  inline constexpr std::int32_t DecayProcessId = 1000000011;
  inline constexpr std::int32_t AnnihilationProcessId = 1000000012;
  inline constexpr std::int32_t PhotonPairProcessId = 1000000013;
  inline constexpr std::int32_t PhotonPairFinalStateProcessId = 2000000013;
  inline constexpr std::int32_t PhotoproductionProcessId = 1000000014;
  inline constexpr std::int32_t PhotonMuonPairProcessId = 1000000015;
  inline constexpr std::int32_t PhotonMuonPairFinalStateProcessId = 2000000015;
  inline constexpr std::int32_t PhotoelectricProcessId = 1000000016;
  inline constexpr double NativeSelectionReplayEndpointWidth = 0.;
  enum class GpuProcessCapability : std::int32_t {
    PhotonPair = 0,
    Bremsstrahlung = 1,
    Compton = 2,
    Photoelectric = 3,
    Annihilation = 4,
    Ionization = 5,
    ElectronPair = 6,
    CpuOnlyFinalState = 7,
    GpuFinalStateNotImplemented = 8,
    PhotonMuonPair = 9,
  };
  inline char const* gpuEmProcessName(std::int32_t process_id) noexcept {
    switch (process_id) {
      case BremsProcessId:
        return "bremsstrahlung";
      case IonizationProcessId:
        return "ionization";
      case ElectronPairProcessId:
        return "electron_pair_production";
      case PhotonuclearProcessId:
        return "photonuclear";
      case MuonPairProcessId:
        return "lepton_muon_pair";
      case HadronsProcessId:
        return "hadrons";
      case WeakInteractionProcessId:
        return "weak_interaction";
      case ComptonProcessId:
        return "compton";
      case DecayProcessId:
        return "decay";
      case AnnihilationProcessId:
        return "annihilation";
      case PhotonPairProcessId:
        return "photon_pair_production";
      case PhotonPairFinalStateProcessId:
        return "photon_pair_final_state";
      case PhotoproductionProcessId:
        return "photoproduction";
      case PhotonMuonPairProcessId:
        return "photon_muon_pair";
      case PhotoelectricProcessId:
        return "photoelectric";
      default:
        return "unknown";
    }
  }

  CORSIKA_HOST_DEVICE inline GpuProcessCapability gpuProcessCapability(
      std::int32_t pid, std::int32_t process_id) {
    if (pid == static_cast<std::int32_t>(EmPid::Photon) &&
        process_id == PhotonMuonPairProcessId) {
      return GpuProcessCapability::PhotonMuonPair;
    }
    if (pid == static_cast<std::int32_t>(EmPid::Photon) &&
        process_id == PhotonPairProcessId) {
      return GpuProcessCapability::PhotonPair;
    }
    if ((pid == static_cast<std::int32_t>(EmPid::Electron) ||
         pid == static_cast<std::int32_t>(EmPid::Positron)) &&
        process_id == BremsProcessId) {
      return GpuProcessCapability::Bremsstrahlung;
    }
    if (pid == static_cast<std::int32_t>(EmPid::Photon) &&
        process_id == ComptonProcessId) {
      return GpuProcessCapability::Compton;
    }
    if (pid == static_cast<std::int32_t>(EmPid::Photon) &&
        process_id == PhotoelectricProcessId) {
      return GpuProcessCapability::Photoelectric;
    }
    if (pid == static_cast<std::int32_t>(EmPid::Positron) &&
        process_id == AnnihilationProcessId) {
      return GpuProcessCapability::Annihilation;
    }
    if (isChargedLeptonPid(pid) && process_id == IonizationProcessId) {
      return GpuProcessCapability::Ionization;
    }
    if (isChargedLeptonPid(pid) && process_id == ElectronPairProcessId) {
      // No native rho CDF table: complete e±/mu± pair sharing with CPU
      // PROPOSAL at the selected vertex, without device rejection sampling.
      return GpuProcessCapability::CpuOnlyFinalState;
    }
    if (isMuonPid(pid) && process_id == BremsProcessId) {
      return GpuProcessCapability::Bremsstrahlung;
    }
    switch (process_id) {
      case PhotonuclearProcessId:
      case MuonPairProcessId:
      case HadronsProcessId:
      case WeakInteractionProcessId:
      case DecayProcessId:
      case PhotoproductionProcessId:
        return GpuProcessCapability::CpuOnlyFinalState;
      default:
        return GpuProcessCapability::GpuFinalStateNotImplemented;
    }
  }

  CORSIKA_HOST_DEVICE inline ProposalFallbackReason processFallbackReason(
      GpuProcessCapability capability) {
    return capability == GpuProcessCapability::CpuOnlyFinalState
               ? ProposalFallbackReason::CpuOnlyProcess
               : ProposalFallbackReason::GpuProcessNotImplemented;
  }

  CORSIKA_HOST_DEVICE inline bool proposalNativeSelectionRequiresReplay(
      std::int32_t pid, std::int32_t process_id, double residual_quantile) {
    auto const capability = gpuProcessCapability(pid, process_id);
    if (capability == GpuProcessCapability::CpuOnlyFinalState) return true;
    // BKK consumes the photon exactly (v=1); an endpoint residual quantile
    // needs no host replay. Its independent secondary x is sampled later.
    if (capability == GpuProcessCapability::PhotonMuonPair) return false;
    // An endpoint replay is a numerical completion mechanism for processes
    // already covered by either a declared GPU final state or a declared
    // CPU-only final state.  It must never turn an unregistered process into
    // an implicit CPU fallback merely because its quantile lies near an
    // endpoint.
    if (capability == GpuProcessCapability::GpuFinalStateNotImplemented) return false;
    return residual_quantile < NativeSelectionReplayEndpointWidth ||
           residual_quantile > 1. - NativeSelectionReplayEndpointWidth;
  }
  namespace tables {
    using namespace continuous_reference::tables;
    enum class TableQueryKind : std::uint32_t {
      Rate = 0,
      TotalRate = 1,
      LossFraction = 2,
      ContinuousDedx = 3,
      ContinuousRange = 4,
      ContinuousEnergy = 5,
      EnergyAfterContinuousLoss = 6,
    };
    struct TableQuery {
      TableQueryKind kind{TableQueryKind::Rate};
      std::int32_t pdg_id{};
      std::int32_t process_id{};
      std::uint64_t component_hash{};
      double energy_MeV{};
      double quantile{};
    };
    // Value-only namespace bridge between the two previously frozen old oracles.
    CORSIKA_HOST_DEVICE inline TableQueryResult convert(
        discrete_reference::TableQueryResult r) {
      return {static_cast<TableLookupStatus>(r.status), r.reserved, r.value};
    }
    CORSIKA_HOST_DEVICE inline discrete_reference::NativePhysicsView discrete_view(
        NativePhysicsView const& view) {
      discrete_reference::NativePhysicsView r;
      r.proposal_native = view.proposal_native;
      return r;
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryRate(NativePhysicsView const& view,
                                                          std::int32_t pid,
                                                          std::int32_t process,
                                                          std::uint64_t component,
                                                          double energy) {
      return convert(discrete_reference::queryRate(discrete_view(view), pid, process,
                                                   component, energy));
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryTotalRate(
        NativePhysicsView const& view, std::int32_t pid, double energy) {
      return convert(
          discrete_reference::queryTotalRate(discrete_view(view), pid, energy));
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryLossFraction(
        NativePhysicsView const& view, std::int32_t pid, std::int32_t process,
        std::uint64_t component, double energy, double quantile) {
      return convert(discrete_reference::queryLossFraction(
          discrete_view(view), pid, process, component, energy, quantile));
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult executeTableQuery(
        NativePhysicsView const& view, TableQuery const& query) {
      if (query.kind == TableQueryKind::Rate) {
        return queryRate(view, query.pdg_id, query.process_id, query.component_hash,
                         query.energy_MeV);
      }
      if (query.kind == TableQueryKind::TotalRate) {
        return queryTotalRate(view, query.pdg_id, query.energy_MeV);
      }
      if (query.kind == TableQueryKind::LossFraction) {
        return queryLossFraction(view, query.pdg_id, query.process_id,
                                 query.component_hash, query.energy_MeV, query.quantile);
      }
      if (query.kind == TableQueryKind::ContinuousDedx) {
        return queryContinuousDedx(view, query.pdg_id, query.energy_MeV);
      }
      if (query.kind == TableQueryKind::ContinuousRange) {
        return queryContinuousRange(view, query.pdg_id, query.energy_MeV);
      }
      if (query.kind == TableQueryKind::ContinuousEnergy) {
        return queryContinuousEnergy(view, query.pdg_id, query.energy_MeV);
      }
      if (query.kind == TableQueryKind::EnergyAfterContinuousLoss) {
        return queryEnergyAfterContinuousLoss(view, query.pdg_id, query.energy_MeV,
                                              query.quantile);
      }
      return {TableLookupStatus::InvalidQueryKind, 0, 0.};
    }
  } // namespace tables
  CORSIKA_HOST_DEVICE inline bool proposalFallbackRequiresSelectedLoss(
      ProposalFallbackReason reason) {
    return reason == ProposalFallbackReason::NativeSelectionReplay ||
           reason == ProposalFallbackReason::InverseCdfUnavailable ||
           reason == ProposalFallbackReason::LossEnergyOutOfRange ||
           reason == ProposalFallbackReason::LossQuantileOutOfRange;
  }

  CORSIKA_HOST_DEVICE inline bool proposalFallbackRequiresInteractionVertex(
      ProposalFallbackReason reason) {
    return proposalFallbackRequiresSelectedLoss(reason) ||
           reason == ProposalFallbackReason::CpuOnlyProcess ||
           reason == ProposalFallbackReason::EpairRejectionEnvelopeExceeded ||
           reason == ProposalFallbackReason::EpairRejectionTrialsExhausted;
  }

  inline char const* proposalFallbackReasonName(ProposalFallbackReason reason) noexcept {
    switch (reason) {
      case ProposalFallbackReason::UnsupportedParticle:
        return "unsupported_particle";
      case ProposalFallbackReason::UnsupportedMedium:
        return "unsupported_medium";
      case ProposalFallbackReason::UnsupportedGeometry:
        return "unsupported_geometry";
      case ProposalFallbackReason::ParticleTableMissing:
        return "particle_table_missing";
      case ProposalFallbackReason::ProcessComponentMissing:
        return "process_component_missing";
      case ProposalFallbackReason::RateEnergyOutOfRange:
        return "rate_energy_out_of_range";
      case ProposalFallbackReason::InverseCdfUnavailable:
        return "inverse_cdf_unavailable";
      case ProposalFallbackReason::LossEnergyOutOfRange:
        return "loss_energy_out_of_range";
      case ProposalFallbackReason::LossQuantileOutOfRange:
        return "loss_quantile_out_of_range";
      case ProposalFallbackReason::InvalidTableQuery:
        return "invalid_table_query";
      case ProposalFallbackReason::DeviceMemorySpill:
        return "device_memory_spill";
      case ProposalFallbackReason::ZeroTotalRate:
        return "zero_total_rate";
      case ProposalFallbackReason::CpuOnlyProcess:
        return "cpu_only_process";
      case ProposalFallbackReason::GpuProcessNotImplemented:
        return "gpu_process_not_implemented";
      case ProposalFallbackReason::InvalidFinalState:
        return "invalid_final_state";
      case ProposalFallbackReason::LpmParametersUnavailable:
        return "lpm_parameters_unavailable";
      case ProposalFallbackReason::InvalidMassDensity:
        return "invalid_mass_density";
      case ProposalFallbackReason::ContinuousEnergyOutOfRange:
        return "continuous_energy_out_of_range";
      case ProposalFallbackReason::ContinuousRangeOutOfRange:
        return "continuous_range_out_of_range";
      case ProposalFallbackReason::TransportCutReached:
        return "transport_cut_reached";
      case ProposalFallbackReason::MoliereParametersUnavailable:
        return "moliere_parameters_unavailable";
      case ProposalFallbackReason::MoliereSamplingFailed:
        return "moliere_sampling_failed";
      case ProposalFallbackReason::MagneticTransportFailed:
        return "magnetic_transport_failed";
      case ProposalFallbackReason::MagneticBoundaryFailed:
        return "magnetic_boundary_failed";
      case ProposalFallbackReason::AtmosphereGrammageFailed:
        return "atmosphere_grammage_failed";
      case ProposalFallbackReason::AtmosphereInverseGrammageFailed:
        return "atmosphere_inverse_grammage_failed";
      case ProposalFallbackReason::AtmosphereVertexLookupFailed:
        return "atmosphere_vertex_lookup_failed";
      case ProposalFallbackReason::EpairRejectionEnvelopeExceeded:
        return "epair_rejection_envelope_exceeded";
      case ProposalFallbackReason::EpairRejectionTrialsExhausted:
        return "epair_rejection_trials_exhausted";
      case ProposalFallbackReason::NativeSelectionReplay:
        return "native_selection_replay";
    }
    return "unknown";
  }

  CORSIKA_HOST_DEVICE inline ProposalFallbackEvent makeTableFallbackEvent(
      EmParticleState const& particle, tables::TableQuery const& query,
      tables::TableQueryResult const& result, std::uint64_t random_draw_id,
      std::uint64_t input_index = 0) {
    ProposalFallbackEvent event{};
    event.particle = particle;
    event.input_index = input_index;
    event.process_id = query.process_id;
    event.reason = proposalFallbackReason(result.status);
    event.component_hash = query.component_hash;
    event.random_draw_id = random_draw_id;
    event.diagnostic_status = static_cast<std::int32_t>(result.status);
    event.diagnostic_value0 = query.energy_MeV;
    event.diagnostic_value1 = query.quantile;
    event.diagnostic_value2 = result.value;
    return event;
  }
} // namespace stochastic_reference
