/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "../tables/ProposalNativeReference.hpp"

// Independent immutable Git oracle: 41cdc543429b82a216857ad458f83deb0f210a5f.
// Only namespace, includes and host/device annotations are adapted.
namespace discrete_reference {
  using namespace native_table_reference;

  inline constexpr std::int32_t PhotonPairProcessId = 1000000013;
  inline constexpr std::int32_t PhotonPairFinalStateProcessId = 2000000013;
  inline constexpr std::int32_t PhotonMuonPairProcessId = 1000000015;
  inline constexpr std::int32_t PhotonMuonPairFinalStateProcessId = 2000000015;
  // PROPOSAL 7.6.2 ME converted from MeV to GeV.
  inline constexpr double ElectronMassGeV = 0.0005109989461;
  inline constexpr double ElectronMassMeV = 1000. * ElectronMassGeV;
  inline constexpr double PhotonPairThresholdMeV = 2. * ElectronMassMeV;

  namespace photon_pair_kinematics_detail {

    CORSIKA_HOST_DEVICE inline bool finite(double value) {
      auto constexpr maximum = 1.7976931348623157e308;
      return value == value && value <= maximum && value >= -maximum;
    }

  } // namespace photon_pair_kinematics_detail

  /**
   * Convert the kinematically normalized pair coordinate
   *
   *   s = (rho - m/E) / (1 - 2m/E)
   *
   * back to PROPOSAL's electron-energy fraction rho.  Storing s instead of
   * rho removes the collapsing [m/E, 1-m/E] interval at pair threshold.
   */
  CORSIKA_HOST_DEVICE inline bool decodePhotonPairNormalizedSplit(
      double parent_energy_GeV, double normalized_split, double& split_fraction) {
    if (!photon_pair_kinematics_detail::finite(parent_energy_GeV) ||
        !photon_pair_kinematics_detail::finite(normalized_split) ||
        !(parent_energy_GeV > 2. * ElectronMassGeV)) {
      return false;
    }
    auto constexpr tolerance = 64. * 2.2204460492503131e-16;
    if (normalized_split < -tolerance || normalized_split > 1. + tolerance) {
      return false;
    }
    auto const bounded =
        normalized_split < 0. ? 0. : (normalized_split > 1. ? 1. : normalized_split);
    auto const lower = ElectronMassGeV / parent_energy_GeV;
    auto const width = 1. - 2. * lower;
    split_fraction = lower + width * bounded;
    return photon_pair_kinematics_detail::finite(split_fraction) &&
           split_fraction >= lower && split_fraction <= 1. - lower;
  }

  CORSIKA_HOST_DEVICE inline bool encodePhotonPairNormalizedSplit(
      double parent_energy_GeV, double split_fraction, double& normalized_split) {
    if (!photon_pair_kinematics_detail::finite(parent_energy_GeV) ||
        !photon_pair_kinematics_detail::finite(split_fraction) ||
        !(parent_energy_GeV > 2. * ElectronMassGeV)) {
      return false;
    }
    auto const lower = ElectronMassGeV / parent_energy_GeV;
    auto const width = 1. - 2. * lower;
    auto const upper = 1. - lower;
    auto constexpr tolerance = 64. * 2.2204460492503131e-16;
    if (!(width > 0.) || split_fraction < lower - tolerance ||
        split_fraction > upper + tolerance) {
      return false;
    }
    auto const bounded = split_fraction < lower
                             ? lower
                             : (split_fraction > upper ? upper : split_fraction);
    normalized_split = (bounded - lower) / width;
    return photon_pair_kinematics_detail::finite(normalized_split) &&
           normalized_split >= 0. && normalized_split <= 1.;
  }

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
  CORSIKA_HOST_DEVICE inline TableQueryResult queryRate(NativePhysicsView const& view,
                                                        std::int32_t pdg_id,
                                                        std::int32_t process_id,
                                                        std::uint64_t component_hash,
                                                        double energy_MeV) {
    return detail::fromProposalNative(queryProposalNativeRate(
        view.proposal_native, pdg_id, process_id, component_hash, energy_MeV));
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult queryTotalRate(
      NativePhysicsView const& view, std::int32_t pdg_id, double energy_MeV) {
    return detail::fromProposalNative(
        queryProposalNativeTotalRate(view.proposal_native, pdg_id, energy_MeV));
  }

  CORSIKA_HOST_DEVICE inline RateColumnSelectionResult selectRateColumnByUniform(
      NativePhysicsView const& view, std::int32_t pdg_id, double energy_MeV,
      double uniform) {
    if (!detail::finiteValue(uniform) || uniform < 0. || uniform >= 1.)
      return {TableLookupStatus::NonFiniteInput, 0, 0, 0., 0., 0};

    auto const total =
        queryProposalNativeSelectionTotalRate(view.proposal_native, pdg_id, energy_MeV);
    auto const converted_total = detail::fromProposalNative(total);
    if (converted_total.status != TableLookupStatus::Success)
      return {converted_total.status, 0, 0, 0., 0., 0};
    if (!(converted_total.value > 0.))
      return {TableLookupStatus::Success, 0, 0, converted_total.value, 0., 0};

    auto sampled_rate = uniform * converted_total.value;
    auto const range = native_detail::dndxParticleRange(view.proposal_native, pdg_id);
    for (auto position = range.begin; position < range.end; ++position) {
      auto const* column = native_detail::selectionDndx(view.proposal_native, position);
      if (!column || column->pdg_id != pdg_id)
        return {TableLookupStatus::InvalidTableView, 0, 0, converted_total.value, 0., 0};
      NativeQueryResult native_rate{NativeQueryStatus::Success, 0, 0, 0.};
      if (!proposalNativeRateBelowThreshold(*column, energy_MeV))
        native_rate =
            queryProposalNativeRateForColumn(view.proposal_native, *column, energy_MeV);
      auto const rate = detail::fromProposalNative(native_rate);
      if (rate.status != TableLookupStatus::Success)
        return {rate.status,
                column->process_id,
                column->component_hash,
                converted_total.value,
                0.,
                0};
      sampled_rate -= rate.value;
      if (sampled_rate < 0.) {
        auto const quantile = (-sampled_rate) / rate.value;
        if (!detail::finiteValue(quantile) || quantile < 0. || quantile > 1.)
          return {TableLookupStatus::InvalidTableView,
                  column->process_id,
                  column->component_hash,
                  converted_total.value,
                  0.,
                  0};
        return {TableLookupStatus::Success, column->process_id, column->component_hash,
                converted_total.value,      quantile,           1u};
      }
    }
    return {TableLookupStatus::Success, 0, 0, converted_total.value, 0., 0};
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult queryLossFraction(
      NativePhysicsView const& view, std::int32_t pdg_id, std::int32_t process_id,
      std::uint64_t component_hash, double energy_MeV, double quantile) {
    if (process_id == PhotonMuonPairFinalStateProcessId && pdg_id == 22) {
      return detail::fromProposalNative(queryProposalNativeLossFraction(
          view.proposal_native, pdg_id, PhotonMuonPairProcessId, component_hash,
          energy_MeV, quantile, true));
    }
    if (process_id == PhotonPairFinalStateProcessId && pdg_id == 22) {
      auto result = detail::fromProposalNative(queryProposalNativeLossFraction(
          view.proposal_native, pdg_id, PhotonPairProcessId, component_hash, energy_MeV,
          quantile, true));
      if (result.status != TableLookupStatus::Success) return result;
      double normalized = 0.;
      if (!encodePhotonPairNormalizedSplit(energy_MeV / 1000., result.value, normalized))
        return {TableLookupStatus::InverseCdfUnavailable, result.reserved, 0.};
      result.value = normalized;
      return result;
    }
    return detail::fromProposalNative(queryProposalNativeLossFraction(
        view.proposal_native, pdg_id, process_id, component_hash, energy_MeV, quantile));
  }
} // namespace discrete_reference
