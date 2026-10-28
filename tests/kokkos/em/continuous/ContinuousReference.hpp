/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Independent fixed-Git oracle. Only namespaces, includes and annotations are adapted.
// The underlying spline/root implementation is the separately audited 3A old oracle.
#include "../tables/ProposalNativeReference.hpp"
#include <corsika/framework/core/HostDevice.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace continuous_reference {
  struct alignas(16) EmParticleState {
    std::int32_t pid{};
    std::int32_t medium_id{};
    std::uint32_t generation{};
    std::uint32_t reserved{};

    double energy_GeV{};
    double position_m[3]{};
    double direction[3]{};
    double time_s{};
    double weight{1.};

    std::uint64_t history_id{};
    std::uint64_t parent_history_id{};
    std::uint64_t step_id{};
    // Independent of medium_id: distinct logical nodes may share one model.
    // -3 denotes legacy/unassigned, -1 denotes outside the exported tree.
    std::int32_t environment_node{-3};
  };
  enum class ProposalFallbackReason : std::int32_t {
    UnsupportedParticle = 0,
    UnsupportedMedium = 1,
    UnsupportedGeometry = 2,
    ParticleTableMissing = 3,
    ProcessComponentMissing = 4,
    RateEnergyOutOfRange = 5,
    InverseCdfUnavailable = 6,
    LossEnergyOutOfRange = 7,
    LossQuantileOutOfRange = 8,
    InvalidTableQuery = 9,
    DeviceMemorySpill = 10,
    ZeroTotalRate = 11,
    CpuOnlyProcess = 12,
    GpuProcessNotImplemented = 13,
    InvalidFinalState = 14,
    LpmParametersUnavailable = 15,
    InvalidMassDensity = 16,
    ContinuousEnergyOutOfRange = 17,
    ContinuousRangeOutOfRange = 18,
    TransportCutReached = 19,
    MoliereParametersUnavailable = 20,
    MoliereSamplingFailed = 21,
    MagneticTransportFailed = 22,
    MagneticBoundaryFailed = 23,
    AtmosphereGrammageFailed = 24,
    AtmosphereInverseGrammageFailed = 25,
    AtmosphereVertexLookupFailed = 26,
    // The bounded device rejection sampler found a production point above
    // its precomputed Epair envelope.  The interaction type, target and v
    // remain valid, so CPU PROPOSAL may generate only this specified final
    // state.  Keep this separate from InvalidFinalState: the latter is a
    // numerical/physics defect and must not be silently retried.
    EpairRejectionEnvelopeExceeded = 27,
    // The device selected the process/component with the original PROPOSAL
    // uniform, but the conditional interval is too ill-conditioned for the
    // last few host/device interpolation ULPs to guarantee the requested v
    // tolerance.  CPU PROPOSAL replays that same uniform, verifies the
    // selected identity, and completes the final state after the wavefront.
    NativeSelectionReplay = 28,
    // The selected interaction and v are valid; finish rho on the CPU,
    // never run a numerical inverse integral in a device exhaustion branch.
    EpairRejectionTrialsExhausted = 29,
  };
  struct ProposalFallbackEvent {
    EmParticleState particle{};
    // A selected final state is legal only after transport reached its vertex.
    // Process/component hashes alone do not establish that position and time
    // have advanced. Generic pre-transport failures keep this flag zero.
    std::uint32_t interaction_vertex_reached{};
    std::uint64_t input_index{};
    std::int32_t process_id{};
    ProposalFallbackReason reason{ProposalFallbackReason::InvalidTableQuery};
    // Optional transport diagnostics. These fields are populated only for
    // generic numerical/geometry failures; physics-process fallbacks leave
    // them zero. Keeping the first failing state self-describing avoids
    // silently hiding a rare device transport defect behind a scalar retry.
    std::int32_t diagnostic_status{};
    std::uint32_t diagnostic_reserved{};
    double diagnostic_value0{};
    double diagnostic_value1{};
    double diagnostic_value2{};
    std::uint64_t component_hash{};
    std::uint64_t medium_hash{};
    std::uint64_t interaction_hash{};
    double energy_fraction{};
    double selection_uniform{};
    double loss_quantile{};
    double final_state_uniform{};
    // Counter-key provenance. selection_uniform/random_* identify the inner
    // PROPOSAL SampleLoss draw; outer_acceptance_* identify CORSIKA's
    // independent interaction-process acceptance draw.
    double outer_acceptance_uniform{};
    std::uint32_t random_process_id{};
    std::uint32_t outer_acceptance_random_process_id{};
    std::uint64_t random_draw_id{};
    std::uint64_t outer_acceptance_draw_id{};
    std::uint64_t final_state_draw_id{};
  };
  enum class EmInteractionStatus : std::int32_t {
    Selected = 0,
    NoDiscreteInteraction = 1,
    /**
     * A charged particle reached the sampled interaction grammage after
     * continuous energy loss. Process/component/v must be sampled again at
     * the vertex energy before final-state generation.
     */
    RequiresReselection = 2,
    /**
     * Only the total rate, interaction grammage and process threshold have
     * been sampled. Charged-particle process/component/v selection is
     * deferred until after continuous loss at the vertex.
     */
    DistanceSampled = 3,
    /**
     * The state is already below the configured kinetic-energy cut. The
     * transport kernel emits a zero-length terminal record and deposits the
     * remaining kinetic energy without querying an out-of-domain rate.
     */
    ParticleCut = 4,
    // A stopped positron: kinetic energy has been deposited by transport;
    // two rest-energy photons must still be materialized, without rate/CDF
    // queries and without thinning the already weighted parent again.
    AtRestAnnihilation = 5,
  };
  struct EmInteractionRecord {
    EmParticleState particle{};
    // A photon can select a CPU-only loss/endpoint before transport, but must
    // still compete its sampled flight against geometry and cuts. Negative
    // means no completion is pending; otherwise this is a ProposalFallbackReason.
    std::int32_t deferred_photon_fallback_reason{-1};
    std::int32_t deferred_photon_table_status{};
    std::uint32_t interaction_vertex_reached{};
    std::uint64_t input_index{};
    std::int32_t process_id{};
    EmInteractionStatus status{EmInteractionStatus::Selected};
    std::uint64_t component_hash{};
    double total_rate_cm2_per_g{};
    double vertex_total_rate_cm2_per_g{};
    double interaction_grammage_g_per_cm2{};
    // Local density at the selected interaction vertex. It is intentionally
    // stored in the record because LPM suppression is evaluated after the
    // final-state energy split has been sampled.
    double mass_density_g_per_cm3{};
    // Rest mass resolved from the versioned continuous-energy table. Keeping
    // it with the selected interaction lets a shared charged-lepton
    // final-state kernel distinguish e± from μ± without hard-coded host
    // dispatch or an additional device-table lookup.
    double particle_mass_GeV{};
    // μ± decay competes with interaction, continuous and geometric limits.
    // The sampled lab-frame distance is retained across the selection and
    // transport kernels so a CPU decay module can be forced at the exact GPU
    // vertex without resampling the already selected lifetime.
    double decay_distance_m{std::numeric_limits<double>::infinity()};
    double decay_uniform{};
    std::uint64_t decay_draw_id{};
    double energy_fraction{};
    double distance_uniform{};
    double process_uniform{};
    double loss_quantile{};
    std::uint64_t distance_draw_id{};
    std::uint64_t process_draw_id{};
    std::uint64_t loss_draw_id{};
    // Proposal-native reproduces the second, independent scalar PROPOSAL draw
    // and derives loss_quantile from the selected rate interval. Legacy
    // c8emrt leaves these fields zero and keeps its existing loss draw.
    double proposal_selection_uniform{};
    std::uint32_t process_random_process_id{};
    std::uint32_t proposal_selection_random_process_id{};
    std::uint64_t proposal_selection_draw_id{};
  };

  inline constexpr double TransportElectronMassGeV = 5.109989e-4;
  inline constexpr double TransportMuonMassGeV = 1.056584e-1;
  CORSIKA_HOST_DEVICE inline double transportMassGeV(std::int32_t const pid) {
    if (pid == 11 || pid == -11) return TransportElectronMassGeV;
    if (pid == 13 || pid == -13) return TransportMuonMassGeV;
    return 0.;
  }
  namespace tables {
    using namespace native_table_reference;
    enum class TableLookupStatus : std::uint32_t {
      Success = 0,
      ParticleNotFound = 1,
      ColumnNotFound = 2,
      RateEnergyOutOfRange = 3,
      InverseCdfUnavailable = 4,
      LossEnergyOutOfRange = 5,
      LossQuantileOutOfRange = 6,
      NonFiniteInput = 7,
      InvalidQueryKind = 8,
      InvalidTableView = 9,
      ContinuousParticleNotFound = 10,
      ContinuousEnergyOutOfRange = 11,
      ContinuousRangeOutOfRange = 12,
      TransportCutReached = 13,
    };
    struct TableQueryResult {
      TableLookupStatus status{TableLookupStatus::InvalidTableView};
      std::uint32_t reserved{};
      double value{};
    };
    struct NativePhysicsView {
      double energy_cut_MeV{};
      double em_transport_cut_MeV{};
      ProposalNativeDeviceView proposal_native{};
      std::uint32_t physics_source{1u};
      unsigned long long* native_inverse_counters{};
      double muon_transport_cut_MeV{};
    };
    CORSIKA_HOST_DEVICE inline std::uint32_t packNativeInverseIterations(
        std::uint32_t newton, std::uint32_t bisection) {
      auto const bounded_newton = newton > 0xffffu ? 0xffffu : newton;
      auto const bounded_bisection = bisection > 0xffffu ? 0xffffu : bisection;
      return bounded_newton | (bounded_bisection << 16u);
    }
    CORSIKA_HOST_DEVICE inline std::uint32_t nativeNewtonIterations(
        TableQueryResult const& result) {
      return result.reserved & 0xffffu;
    }
    CORSIKA_HOST_DEVICE inline std::uint32_t nativeBisectionIterations(
        TableQueryResult const& result) {
      return result.reserved >> 16u;
    }
    namespace detail {
      CORSIKA_HOST_DEVICE inline bool finiteValue(double value) {
#if defined(__CUDA_ARCH__)
        return ::isfinite(value);
#else
        return std::isfinite(value);
#endif
      }
      CORSIKA_HOST_DEVICE inline TableQueryResult fromProposalNative(
          NativeQueryResult const& result, bool continuous = false) {
        switch (result.status) {
          case NativeQueryStatus::Success:
            return {TableLookupStatus::Success,
                    packNativeInverseIterations(result.newton_iterations,
                                                result.bisection_iterations),
                    result.value};
          case NativeQueryStatus::ParticleNotFound:
            return {continuous ? TableLookupStatus::ContinuousParticleNotFound
                               : TableLookupStatus::ParticleNotFound,
                    packNativeInverseIterations(result.newton_iterations,
                                                result.bisection_iterations),
                    0.};
          case NativeQueryStatus::ColumnNotFound:
            return {TableLookupStatus::ColumnNotFound,
                    packNativeInverseIterations(result.newton_iterations,
                                                result.bisection_iterations),
                    0.};
          case NativeQueryStatus::EnergyOutOfRange:
            return {continuous ? TableLookupStatus::ContinuousEnergyOutOfRange
                               : TableLookupStatus::RateEnergyOutOfRange,
                    packNativeInverseIterations(result.newton_iterations,
                                                result.bisection_iterations),
                    0.};
          case NativeQueryStatus::QuantileOutOfRange:
            return {TableLookupStatus::LossQuantileOutOfRange,
                    packNativeInverseIterations(result.newton_iterations,
                                                result.bisection_iterations),
                    0.};
          case NativeQueryStatus::UnsupportedKinematics:
          case NativeQueryStatus::RootNotConverged:
            return {TableLookupStatus::InverseCdfUnavailable,
                    packNativeInverseIterations(result.newton_iterations,
                                                result.bisection_iterations),
                    0.};
          case NativeQueryStatus::NonFiniteInput:
            return {TableLookupStatus::NonFiniteInput,
                    packNativeInverseIterations(result.newton_iterations,
                                                result.bisection_iterations),
                    0.};
          case NativeQueryStatus::InvalidView:
            return {TableLookupStatus::InvalidTableView,
                    packNativeInverseIterations(result.newton_iterations,
                                                result.bisection_iterations),
                    0.};
        }
        return {TableLookupStatus::InvalidTableView,
                packNativeInverseIterations(result.newton_iterations,
                                            result.bisection_iterations),
                0.};
      }
    } // namespace detail
    inline constexpr double ContinuousCutSafetyFactor = 0.9999;
    CORSIKA_HOST_DEVICE inline TableQueryResult queryContinuousDedx(
        NativePhysicsView const& view, std::int32_t pdg_id, double energy_MeV) {
      return detail::fromProposalNative(
          queryProposalNativeDedx(view.proposal_native, pdg_id, energy_MeV), true);
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryContinuousMass(
        NativePhysicsView const& view, std::int32_t pdg_id) {
      auto const* utility = native_detail::findUtility(view.proposal_native, pdg_id);
      if (!utility) return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
      if (!detail::finiteValue(utility->particle_mass_MeV) ||
          !(utility->particle_mass_MeV > 0.))
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      return {TableLookupStatus::Success, 0, utility->particle_mass_MeV};
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryContinuousTransportMass(
        NativePhysicsView const& view, std::int32_t pdg_id) {
      auto const native_mass = queryContinuousMass(view, pdg_id);
      if (native_mass.status != TableLookupStatus::Success) return native_mass;
      auto const transport_mass_MeV = 1000. * transportMassGeV(pdg_id);
      if (!(transport_mass_MeV > 0.))
        return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
      return {TableLookupStatus::Success, 0, transport_mass_MeV};
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryContinuousTransportCut(
        NativePhysicsView const& view, std::int32_t pdg_id) {
      auto const cut = (pdg_id == 13 || pdg_id == -13) ? view.muon_transport_cut_MeV
                                                       : view.em_transport_cut_MeV;
      if (!detail::finiteValue(cut) || !(cut > 0.))
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      return {TableLookupStatus::Success, 0, cut};
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryContinuousMinimumEnergy(
        NativePhysicsView const& view, std::int32_t pdg_id) {
      auto const* utility = native_detail::findUtility(view.proposal_native, pdg_id);
      if (!utility) return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
      // PROPOSAL's native displacement interpolant starts at the particle
      // mass. The scalar ContinuousProcess applies its safety factor to the
      // TOTAL cut energy, not just the kinetic cut. This endpoint is only a
      // step limiter: callers must query the user cut separately rather than
      // invert this expression to decide ParticleCut membership.
      auto const transport_cut = queryContinuousTransportCut(view, pdg_id);
      if (transport_cut.status != TableLookupStatus::Success) return transport_cut;
      auto const minimum = (1000. * transportMassGeV(pdg_id) + transport_cut.value) *
                           ContinuousCutSafetyFactor;
      if (!detail::finiteValue(utility->lower_energy_limit_MeV) ||
          utility->lower_energy_limit_MeV < utility->particle_mass_MeV ||
          !detail::finiteValue(minimum) || !(minimum > utility->particle_mass_MeV) ||
          minimum < utility->lower_energy_limit_MeV ||
          minimum > utility->spline.axis.high)
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      return {TableLookupStatus::Success, 0, minimum};
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryContinuousRange(
        NativePhysicsView const& view, std::int32_t pdg_id, double energy_MeV) {
      return detail::fromProposalNative(
          queryProposalNativeRange(view.proposal_native, pdg_id, energy_MeV), true);
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryContinuousEnergy(
        NativePhysicsView const& view, std::int32_t pdg_id, double range_g_per_cm2) {
      auto result = detail::fromProposalNative(
          queryProposalNativeEnergy(view.proposal_native, pdg_id, range_g_per_cm2), true);
      if (result.status == TableLookupStatus::ContinuousEnergyOutOfRange)
        result.status = TableLookupStatus::ContinuousRangeOutOfRange;
      return result;
    }
    CORSIKA_HOST_DEVICE inline TableQueryResult queryEnergyAfterContinuousLoss(
        NativePhysicsView const& view, std::int32_t pdg_id, double initial_energy_MeV,
        double grammage_g_per_cm2) {
      if (!detail::finiteValue(grammage_g_per_cm2) || grammage_g_per_cm2 < 0.) {
        return {TableLookupStatus::NonFiniteInput, 0, 0.};
      }
      auto const initial_range = queryContinuousRange(view, pdg_id, initial_energy_MeV);
      if (initial_range.status != TableLookupStatus::Success) { return initial_range; }
      if (grammage_g_per_cm2 >= initial_range.value && grammage_g_per_cm2 != 0.) {
        return {TableLookupStatus::TransportCutReached, 0, 0.};
      }
      return detail::fromProposalNative(
          queryProposalNativeEnergyAfterContinuousLoss(
              view.proposal_native, pdg_id, initial_energy_MeV, grammage_g_per_cm2),
          true);
    }
  } // namespace tables
  CORSIKA_HOST_DEVICE inline ProposalFallbackReason proposalFallbackReason(
      tables::TableLookupStatus status) {
    using tables::TableLookupStatus;
    switch (status) {
      case TableLookupStatus::ParticleNotFound:
        return ProposalFallbackReason::ParticleTableMissing;
      case TableLookupStatus::ColumnNotFound:
        return ProposalFallbackReason::ProcessComponentMissing;
      case TableLookupStatus::RateEnergyOutOfRange:
        return ProposalFallbackReason::RateEnergyOutOfRange;
      case TableLookupStatus::InverseCdfUnavailable:
        return ProposalFallbackReason::InverseCdfUnavailable;
      case TableLookupStatus::LossEnergyOutOfRange:
        return ProposalFallbackReason::LossEnergyOutOfRange;
      case TableLookupStatus::LossQuantileOutOfRange:
        return ProposalFallbackReason::LossQuantileOutOfRange;
      case TableLookupStatus::ContinuousParticleNotFound:
        return ProposalFallbackReason::ParticleTableMissing;
      case TableLookupStatus::ContinuousEnergyOutOfRange:
        return ProposalFallbackReason::ContinuousEnergyOutOfRange;
      case TableLookupStatus::ContinuousRangeOutOfRange:
        return ProposalFallbackReason::ContinuousRangeOutOfRange;
      case TableLookupStatus::TransportCutReached:
        return ProposalFallbackReason::TransportCutReached;
      case TableLookupStatus::Success:
      case TableLookupStatus::NonFiniteInput:
      case TableLookupStatus::InvalidQueryKind:
      case TableLookupStatus::InvalidTableView:
        return ProposalFallbackReason::InvalidTableQuery;
    }
    return ProposalFallbackReason::InvalidTableQuery;
  }
  CORSIKA_HOST_DEVICE inline ProposalFallbackEvent makeProcessFallbackEvent(
      EmInteractionRecord const& record, ProposalFallbackReason reason) {
    ProposalFallbackEvent event{};
    event.particle = record.particle;
    event.interaction_vertex_reached = record.interaction_vertex_reached;
    event.input_index = record.input_index;
    event.process_id = record.process_id;
    event.reason = reason;
    event.component_hash = record.component_hash;
    event.energy_fraction = record.energy_fraction;
    auto const native_selection = record.proposal_selection_random_process_id != 0u;
    event.selection_uniform =
        native_selection ? record.proposal_selection_uniform : record.process_uniform;
    event.loss_quantile = record.loss_quantile;
    event.outer_acceptance_uniform = record.process_uniform;
    event.outer_acceptance_random_process_id = record.process_random_process_id;
    event.outer_acceptance_draw_id = record.process_draw_id;
    event.random_process_id =
        native_selection ? record.proposal_selection_random_process_id : 0u;
    event.random_draw_id =
        native_selection ? record.proposal_selection_draw_id : record.loss_draw_id;
    return event;
  }
  namespace step {
    /** Match the scalar Cascade continuous-loss step limiter. */
    inline constexpr double MaximumContinuousRelativeLoss = 0.10;

    struct LeptonContinuousPreparation {
      double mass_MeV{};
      double minimum_energy_MeV{};
      double transport_cut_MeV{};
      double maximum_grammage_g_per_cm2{};
      std::uint32_t particle_cut{};
      std::uint32_t fallback_flag{};
      continuous_reference::ProposalFallbackEvent fallback{};
    };

    struct LeptonContinuousLoss {
      double final_energy_MeV{};
      double deposited_energy_GeV{};
      std::uint32_t transport_cut_reached{};
      std::uint32_t fallback_flag{};
      continuous_reference::ProposalFallbackEvent fallback{};
    };

    CORSIKA_HOST_DEVICE inline bool leptonContinuousFinite(double const value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline continuous_reference::ProposalFallbackEvent
    makeLeptonTransportFallback(
        continuous_reference::EmInteractionRecord const& interaction,
        continuous_reference::ProposalFallbackReason const reason) {
      auto event = continuous_reference::makeProcessFallbackEvent(interaction, reason);
      event.input_index = interaction.input_index;
      return event;
    }

    CORSIKA_HOST_DEVICE inline continuous_reference::ProposalFallbackEvent
    makeLeptonTableFallback(continuous_reference::EmInteractionRecord const& interaction,
                            continuous_reference::tables::TableLookupStatus const status,
                            double const query_stage = 0.,
                            double const query_value = 0.) {
      auto event = makeLeptonTransportFallback(
          interaction, continuous_reference::proposalFallbackReason(status));
      event.diagnostic_status = static_cast<std::int32_t>(status);
      event.diagnostic_value0 = query_stage;
      event.diagnostic_value1 = query_value;
      return event;
    }

    /**
     * Resolve the particle mass/cut and the maximum grammage allowed by the
     * scalar 10% continuous-loss limiter.  Both native CUDA and Kokkos call
     * this exact function, so proposal-native range interpolation and all
     * endpoint decisions have one implementation.
     */
    CORSIKA_HOST_DEVICE inline LeptonContinuousPreparation prepareLeptonContinuousStep(
        continuous_reference::tables::NativePhysicsView const& table,
        continuous_reference::EmInteractionRecord const& interaction) {
      using namespace continuous_reference;
      using namespace continuous_reference::tables;
      LeptonContinuousPreparation output{};
      auto const& start = interaction.particle;
      auto const mass = queryContinuousTransportMass(table, start.pid);
      auto const minimum_energy = queryContinuousMinimumEnergy(table, start.pid);
      if (mass.status != TableLookupStatus::Success) {
        output.fallback = makeLeptonTableFallback(interaction, mass.status, 1.);
        output.fallback_flag = 1;
        return output;
      }
      if (minimum_energy.status != TableLookupStatus::Success) {
        output.fallback = makeLeptonTableFallback(interaction, minimum_energy.status, 2.);
        output.fallback_flag = 1;
        return output;
      }
      output.mass_MeV = mass.value;
      output.minimum_energy_MeV = minimum_energy.value;
      auto const initial_energy_MeV = start.energy_GeV * 1000.;
      if (!(initial_energy_MeV > mass.value)) {
        output.fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::InvalidFinalState);
        output.fallback_flag = 1;
        return output;
      }
      auto const transport_cut = queryContinuousTransportCut(table, start.pid);
      output.transport_cut_MeV = transport_cut.value;
      if (transport_cut.status != TableLookupStatus::Success ||
          !leptonContinuousFinite(output.transport_cut_MeV) ||
          !(output.transport_cut_MeV > 0.)) {
        output.fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::InvalidTableQuery);
        output.fallback.diagnostic_value0 = mass.value;
        output.fallback.diagnostic_value1 = minimum_energy.value;
        output.fallback.diagnostic_value2 = output.transport_cut_MeV;
        output.fallback_flag = 1;
        return output;
      }
      if (interaction.status == EmInteractionStatus::ParticleCut ||
          initial_energy_MeV - mass.value < output.transport_cut_MeV) {
        output.particle_cut = 1;
        return output;
      }

      auto const initial_range =
          queryContinuousRange(table, start.pid, initial_energy_MeV);
      if (initial_range.status != TableLookupStatus::Success) {
        output.fallback = makeLeptonTableFallback(interaction, initial_range.status, 3.,
                                                  initial_energy_MeV);
        output.fallback_flag = 1;
        return output;
      }
      auto const relative_limit =
          (1. - MaximumContinuousRelativeLoss) * initial_energy_MeV;
      auto const target_energy_MeV =
          relative_limit > minimum_energy.value ? relative_limit : minimum_energy.value;
      auto const target_range = queryContinuousRange(table, start.pid, target_energy_MeV);
      if (target_range.status != TableLookupStatus::Success) {
        output.fallback = makeLeptonTableFallback(interaction, target_range.status, 4.,
                                                  target_energy_MeV);
        output.fallback_flag = 1;
        return output;
      }
      output.maximum_grammage_g_per_cm2 = initial_range.value - target_range.value;
      auto const range_scale = initial_range.value > 1. ? initial_range.value : 1.;
      if (output.maximum_grammage_g_per_cm2 < 0. &&
          output.maximum_grammage_g_per_cm2 >
              -64. * 2.22044604925031308085e-16 * range_scale) {
        output.maximum_grammage_g_per_cm2 = 0.;
      }
      if (!leptonContinuousFinite(output.maximum_grammage_g_per_cm2) ||
          output.maximum_grammage_g_per_cm2 < 0.) {
        output.fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::InvalidTableQuery);
        output.fallback.diagnostic_value0 = initial_range.value;
        output.fallback.diagnostic_value1 = target_range.value;
        output.fallback.diagnostic_value2 = output.maximum_grammage_g_per_cm2;
        output.fallback_flag = 1;
      }
      return output;
    }

    /** Evaluate the post-step range inversion and scalar ParticleCut state. */
    CORSIKA_HOST_DEVICE inline LeptonContinuousLoss evaluateLeptonContinuousLoss(
        continuous_reference::tables::NativePhysicsView const& table,
        continuous_reference::EmInteractionRecord const& interaction,
        LeptonContinuousPreparation const& preparation,
        double const traversed_grammage_g_per_cm2, bool const continuous_limit_won) {
      using namespace continuous_reference;
      using namespace continuous_reference::tables;
      LeptonContinuousLoss output{};
      auto const initial_energy_MeV = interaction.particle.energy_GeV * 1000.;
      auto const final_energy = queryEnergyAfterContinuousLoss(
          table, interaction.particle.pid, initial_energy_MeV,
          traversed_grammage_g_per_cm2);
      if (final_energy.status == TableLookupStatus::Success) {
        output.final_energy_MeV = final_energy.value;
      } else if (final_energy.status == TableLookupStatus::TransportCutReached) {
        output.final_energy_MeV = preparation.minimum_energy_MeV;
        output.transport_cut_reached = 1;
      } else {
        output.fallback = makeLeptonTableFallback(interaction, final_energy.status, 5.,
                                                  traversed_grammage_g_per_cm2);
        output.fallback_flag = 1;
        return output;
      }
      auto const actual_loss_MeV = initial_energy_MeV - output.final_energy_MeV;
      if (continuous_limit_won && !(actual_loss_MeV > 0.)) {
        output.fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::InvalidTableQuery);
        output.fallback.diagnostic_value0 = initial_energy_MeV;
        output.fallback.diagnostic_value1 = output.final_energy_MeV;
        output.fallback.diagnostic_value2 = traversed_grammage_g_per_cm2;
        output.fallback_flag = 1;
        return output;
      }
      output.deposited_energy_GeV = actual_loss_MeV / 1000.;
      return output;
    }

  } // namespace step
} // namespace continuous_reference
