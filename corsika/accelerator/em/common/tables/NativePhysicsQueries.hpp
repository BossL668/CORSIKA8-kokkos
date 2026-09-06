/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/common/PhotonPairKinematics.hpp>
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/accelerator/em/common/tables/PhysicsConstants.hpp>

#define CORSIKA_GPU_TABLE_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em::tables {

  /**
   * Non-throwing lookup result used by both host validation and CUDA kernels.
   *
   * Every non-success value is an explicit hybrid-scheduler fallback reason.
   * Device queries never clamp an energy or quantile to the stored domain.
   */
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

  struct TableQueryResult {
    TableLookupStatus status{TableLookupStatus::InvalidTableView};
    std::uint32_t reserved{};
    double value{};
  };

  /**
   * Result of walking the process/component columns in their canonical
   * PROPOSAL order.  This small POD is shared by the photon selector and the
   * charged-lepton vertex selector so neither kernel needs to know whether
   * the backing storage is a legacy c8emrt table or native PROPOSAL splines.
   */
  struct RateColumnSelectionResult {
    TableLookupStatus status{TableLookupStatus::InvalidTableView};
    std::int32_t process_id{};
    std::uint64_t component_hash{};
    double total_rate{};
    // Proposal-native only: (-sampled_rate)/selected_rate after the exact
    // Interaction::SampleLoss subtraction. Legacy c8emrt selectors leave it
    // zero and keep their independent loss draw unchanged.
    double residual_quantile{};
    std::uint32_t selected{};
  };

  CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t
  packNativeInverseIterations(std::uint32_t newton,
                              std::uint32_t bisection) {
    auto const bounded_newton = newton > 0xffffu ? 0xffffu : newton;
    auto const bounded_bisection =
        bisection > 0xffffu ? 0xffffu : bisection;
    return bounded_newton | (bounded_bisection << 16u);
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t
  nativeNewtonIterations(TableQueryResult const& result) {
    return result.reserved & 0xffffu;
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t
  nativeBisectionIterations(TableQueryResult const& result) {
    return result.reserved >> 16u;
  }

  struct NativePhysicsView {
    double energy_cut_MeV{};
    double em_transport_cut_MeV{};
    ProposalNativeDeviceView proposal_native{};
    std::uint32_t physics_source{1u};
    unsigned long long* native_inverse_counters{};
    double muon_transport_cut_MeV{};
  };
  static_assert(std::is_trivially_copyable_v<NativePhysicsView>);
  namespace detail {
CORSIKA_GPU_TABLE_HOST_DEVICE inline bool finiteValue(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult fromProposalNative(
        NativeQueryResult const& result, bool continuous = false) {
      switch (result.status) {
      case NativeQueryStatus::Success:
        return {TableLookupStatus::Success,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations),
                result.value};
      case NativeQueryStatus::ParticleNotFound:
        return {continuous ? TableLookupStatus::ContinuousParticleNotFound
                           : TableLookupStatus::ParticleNotFound,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::ColumnNotFound:
        return {TableLookupStatus::ColumnNotFound,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::EnergyOutOfRange:
        return {continuous ? TableLookupStatus::ContinuousEnergyOutOfRange
                           : TableLookupStatus::RateEnergyOutOfRange,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::QuantileOutOfRange:
        return {TableLookupStatus::LossQuantileOutOfRange,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::UnsupportedKinematics:
      case NativeQueryStatus::RootNotConverged:
        return {TableLookupStatus::InverseCdfUnavailable,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::NonFiniteInput:
        return {TableLookupStatus::NonFiniteInput,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::InvalidView:
        return {TableLookupStatus::InvalidTableView,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      }
      return {TableLookupStatus::InvalidTableView,
              packNativeInverseIterations(
                  result.newton_iterations,
                  result.bisection_iterations), 0.};
    }
  } // namespace detail
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryRate(
      NativePhysicsView const& view, std::int32_t pdg_id,
      std::int32_t process_id, std::uint64_t component_hash,
      double energy_MeV) {
      return detail::fromProposalNative(queryProposalNativeRate(
          view.proposal_native, pdg_id, process_id, component_hash,
          energy_MeV));
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryTotalRate(
      NativePhysicsView const& view, std::int32_t pdg_id,
      double energy_MeV) {
      return detail::fromProposalNative(queryProposalNativeTotalRate(
          view.proposal_native, pdg_id, energy_MeV));
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline RateColumnSelectionResult
  selectRateColumnByThreshold(
      NativePhysicsView const& view, std::int32_t pdg_id,
      double energy_MeV, double threshold) {
    if (!detail::finiteValue(energy_MeV) ||
        !detail::finiteValue(threshold) || threshold < 0.) {
      return {TableLookupStatus::NonFiniteInput, 0, 0, 0., 0., 0};
    }

    double cumulative = 0.;
    std::int32_t selected_process = 0;
    std::uint64_t selected_component = 0;
    std::int32_t last_positive_process = 0;
    std::uint64_t last_positive_component = 0;
    double selected_residual_quantile = 0.;
    bool found_particle = false;
    bool selected = false;
    bool have_positive = false;

    
      if (!native_detail::validView(view.proposal_native)) {
        return {TableLookupStatus::InvalidTableView, 0, 0, 0., 0., 0};
      }
      auto const range = native_detail::dndxParticleRange(
          view.proposal_native, pdg_id);
      for (std::uint32_t index = range.begin;
           index < range.end; ++index) {
        auto const* selected_column = native_detail::selectionDndx(
            view.proposal_native, index);
        if (!selected_column || selected_column->pdg_id != pdg_id)
          return {TableLookupStatus::InvalidTableView, 0, 0,
                  cumulative, 0., 0};
        auto const& column = *selected_column;
        found_particle = true;
        NativeQueryResult native_rate{
            NativeQueryStatus::Success, 0, 0, 0.};
        if (!proposalNativeRateBelowThreshold(column, energy_MeV))
          native_rate = queryProposalNativeRateForColumn(
              view.proposal_native, column, energy_MeV);
        auto const rate = detail::fromProposalNative(native_rate);
        if (rate.status != TableLookupStatus::Success) {
          // Preserve the exact descriptor identity for device-side failure
          // diagnostics.  The caller can then distinguish a bad column from
          // a malformed aggregate query without repeating the traversal.
          return {rate.status, column.process_id, column.component_hash,
                  cumulative, 0., 0};
        }
        if (rate.value > 0.) {
          have_positive = true;
          last_positive_process = column.process_id;
          last_positive_component = column.component_hash;
        }
        cumulative += rate.value;
        if (!selected && threshold < cumulative) {
          selected = true;
          selected_process = column.process_id;
          selected_component = column.component_hash;
          selected_residual_quantile =
              rate.value > 0. ? (cumulative - threshold) / rate.value : 0.;
        }
      }
    

    if (!found_particle) {
      return {TableLookupStatus::ParticleNotFound, 0, 0, 0., 0., 0};
    }
    if (!detail::finiteValue(cumulative)) {
      return {TableLookupStatus::InvalidTableView, 0, 0, 0., 0., 0};
    }
    // Guard the last representable boundary exactly as the legacy selectors
    // did.  This is only reachable through floating-point summation drift.
    if (!selected && have_positive && threshold < cumulative) {
      selected = true;
      selected_process = last_positive_process;
      selected_component = last_positive_component;
    }
    return {TableLookupStatus::Success, selected_process,
            selected_component, cumulative, selected_residual_quantile,
            selected ? 1u : 0u};
  }
CORSIKA_GPU_TABLE_HOST_DEVICE inline RateColumnSelectionResult
  selectRateColumnByUniform(NativePhysicsView const& view,
                            std::int32_t pdg_id, double energy_MeV,
                            double uniform) {
    if (!detail::finiteValue(uniform) || uniform < 0. || uniform >= 1.)
      return {TableLookupStatus::NonFiniteInput, 0, 0, 0., 0., 0};
    
    auto const total = queryProposalNativeSelectionTotalRate(
        view.proposal_native, pdg_id, energy_MeV);
    auto const converted_total = detail::fromProposalNative(total);
    if (converted_total.status != TableLookupStatus::Success)
      return {converted_total.status, 0, 0, 0., 0., 0};
    if (!(converted_total.value > 0.))
      return {TableLookupStatus::Success, 0, 0,
              converted_total.value, 0., 0};

    auto sampled_rate = uniform * converted_total.value;
    auto const range = native_detail::dndxParticleRange(
        view.proposal_native, pdg_id);
    for (auto position = range.begin; position < range.end; ++position) {
      auto const* column = native_detail::selectionDndx(
          view.proposal_native, position);
      if (!column || column->pdg_id != pdg_id)
        return {TableLookupStatus::InvalidTableView, 0, 0,
                converted_total.value, 0., 0};
      NativeQueryResult native_rate{NativeQueryStatus::Success, 0, 0, 0.};
      if (!proposalNativeRateBelowThreshold(*column, energy_MeV))
        native_rate = queryProposalNativeRateForColumn(
            view.proposal_native, *column, energy_MeV);
      auto const rate = detail::fromProposalNative(native_rate);
      if (rate.status != TableLookupStatus::Success)
        return {rate.status, column->process_id, column->component_hash,
                converted_total.value, 0., 0};
      sampled_rate -= rate.value;
      if (sampled_rate < 0.) {
        auto const quantile = (-sampled_rate) / rate.value;
        if (!detail::finiteValue(quantile) || quantile < 0. || quantile > 1.)
          return {TableLookupStatus::InvalidTableView, column->process_id,
                  column->component_hash, converted_total.value, 0., 0};
        return {TableLookupStatus::Success, column->process_id,
                column->component_hash, converted_total.value, quantile, 1u};
      }
    }
    return {TableLookupStatus::Success, 0, 0,
            converted_total.value, 0., 0};
  }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryLossFraction(
      NativePhysicsView const& view, std::int32_t pdg_id,
      std::int32_t process_id, std::uint64_t component_hash,
      double energy_MeV, double quantile) {
      return detail::fromProposalNative(queryProposalNativeLossFraction(
          view.proposal_native, pdg_id, process_id, component_hash,
          energy_MeV, quantile));
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryContinuousDedx(
      NativePhysicsView const& view, std::int32_t pdg_id,
      double energy_MeV) {
      return detail::fromProposalNative(queryProposalNativeDedx(
          view.proposal_native, pdg_id, energy_MeV), true);
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryContinuousMass(
      NativePhysicsView const& view, std::int32_t pdg_id) {
      auto const* utility = native_detail::findUtility(
          view.proposal_native, pdg_id);
      if (!utility)
        return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
      if (!detail::finiteValue(utility->particle_mass_MeV) ||
          !(utility->particle_mass_MeV > 0.))
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      return {TableLookupStatus::Success, 0,
              utility->particle_mass_MeV};
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult
  queryContinuousTransportMass(
      NativePhysicsView const& view, std::int32_t pdg_id) {
      auto const native_mass = queryContinuousMass(view, pdg_id);
      if (native_mass.status != TableLookupStatus::Success) return native_mass;
      auto const transport_mass_MeV = 1000. * transportMassGeV(pdg_id);
      if (!(transport_mass_MeV > 0.))
        return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
      return {TableLookupStatus::Success, 0, transport_mass_MeV};
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult
  queryContinuousTransportCut(
      NativePhysicsView const& view, std::int32_t pdg_id) {
      auto const cut = (pdg_id == 13 || pdg_id == -13)
                           ? view.muon_transport_cut_MeV
                           : view.em_transport_cut_MeV;
      if (!detail::finiteValue(cut) || !(cut > 0.))
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      return {TableLookupStatus::Success, 0, cut};
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult
  queryContinuousMinimumEnergy(
      NativePhysicsView const& view, std::int32_t pdg_id) {
      auto const* utility = native_detail::findUtility(
          view.proposal_native, pdg_id);
      if (!utility)
        return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
      // PROPOSAL's native displacement interpolant starts at the particle
      // mass. The scalar ContinuousProcess applies its safety factor to the
      // TOTAL cut energy, not just the kinetic cut. This endpoint is only a
      // step limiter: callers must query the user cut separately rather than
      // invert this expression to decide ParticleCut membership.
      auto const transport_cut = queryContinuousTransportCut(view, pdg_id);
      if (transport_cut.status != TableLookupStatus::Success)
        return transport_cut;
      auto const minimum =
          (1000. * transportMassGeV(pdg_id) + transport_cut.value) *
          ContinuousCutSafetyFactor;
      if (!detail::finiteValue(utility->lower_energy_limit_MeV) ||
          utility->lower_energy_limit_MeV <
              utility->particle_mass_MeV ||
          !detail::finiteValue(minimum) ||
          !(minimum > utility->particle_mass_MeV) ||
          minimum < utility->lower_energy_limit_MeV ||
          minimum > utility->spline.axis.high)
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      return {TableLookupStatus::Success, 0, minimum};
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryContinuousRange(
      NativePhysicsView const& view, std::int32_t pdg_id,
      double energy_MeV) {
      return detail::fromProposalNative(queryProposalNativeRange(
          view.proposal_native, pdg_id, energy_MeV), true);
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryContinuousEnergy(
      NativePhysicsView const& view, std::int32_t pdg_id,
      double range_g_per_cm2) {
      auto result = detail::fromProposalNative(queryProposalNativeEnergy(
          view.proposal_native, pdg_id, range_g_per_cm2), true);
      if (result.status == TableLookupStatus::ContinuousEnergyOutOfRange)
        result.status = TableLookupStatus::ContinuousRangeOutOfRange;
      return result;
    }
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult
  queryEnergyAfterContinuousLoss(
      NativePhysicsView const& view, std::int32_t pdg_id,
      double initial_energy_MeV, double grammage_g_per_cm2) {
    if (!detail::finiteValue(grammage_g_per_cm2) ||
        grammage_g_per_cm2 < 0.) {
      return {TableLookupStatus::NonFiniteInput, 0, 0.};
    }
    auto const initial_range =
        queryContinuousRange(view, pdg_id, initial_energy_MeV);
    if (initial_range.status != TableLookupStatus::Success) {
      return initial_range;
    }
    if (grammage_g_per_cm2 >= initial_range.value &&
        grammage_g_per_cm2 != 0.) {
      return {TableLookupStatus::TransportCutReached, 0, 0.};
    }
      return detail::fromProposalNative(
          queryProposalNativeEnergyAfterContinuousLoss(
              view.proposal_native, pdg_id, initial_energy_MeV,
              grammage_g_per_cm2),
          true);
}
CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult executeTableQuery(
      NativePhysicsView const& view, TableQuery const& query) {
    if (query.kind == TableQueryKind::Rate) {
      return queryRate(view, query.pdg_id, query.process_id,
                       query.component_hash, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::TotalRate) {
      return queryTotalRate(view, query.pdg_id, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::LossFraction) {
      return queryLossFraction(
          view, query.pdg_id, query.process_id, query.component_hash,
          query.energy_MeV, query.quantile);
    }
    if (query.kind == TableQueryKind::ContinuousDedx) {
      return queryContinuousDedx(
          view, query.pdg_id, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::ContinuousRange) {
      return queryContinuousRange(
          view, query.pdg_id, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::ContinuousEnergy) {
      return queryContinuousEnergy(
          view, query.pdg_id, query.energy_MeV);
    }
    if (query.kind ==
        TableQueryKind::EnergyAfterContinuousLoss) {
      return queryEnergyAfterContinuousLoss(
          view, query.pdg_id, query.energy_MeV, query.quantile);
    }
    return {TableLookupStatus::InvalidQueryKind, 0, 0.};
  }
} // namespace corsika::gpu::em::tables
#undef CORSIKA_GPU_TABLE_HOST_DEVICE
