/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include "DiscreteReference.hpp"
#include "../tables/ProposalNativeQueryTestData.hpp"
#include <corsika/kokkos/em/tables/DiscretePhysicsQueries.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace discrete_test {
  namespace current = corsika::kokkos;
  namespace reference = discrete_reference;

  enum class Operation : std::uint32_t {
    Rate,
    Total,
    Select,
    Loss,
    Encode,
    Decode,
    Count
  };
  struct Query {
    Operation operation_{};
    std::int32_t pid_{};
    std::int32_t process_{};
    std::uint64_t component_{};
    double energy_{};
    double argument_{};
    bool invalidView_{};
  };
  struct Outcome {
    std::uint32_t status_{};
    std::uint32_t iterations_{};
    std::int32_t process_{};
    std::uint64_t component_{};
    double value_{};
    double total_{};
    double residual_{};
    std::uint32_t selected_{};
  };

  CORSIKA_HOST_DEVICE inline Outcome outcome(current::TableQueryResult result) {
    return {static_cast<std::uint32_t>(result.status_),
            result.reserved_,
            0,
            0,
            result.value_,
            0.,
            0.,
            0};
  }
  CORSIKA_HOST_DEVICE inline Outcome outcome(reference::TableQueryResult result) {
    return {static_cast<std::uint32_t>(result.status),
            result.reserved,
            0,
            0,
            result.value,
            0.,
            0.,
            0};
  }
  CORSIKA_HOST_DEVICE inline Outcome outcome(current::RateColumnSelectionResult result) {
    return {static_cast<std::uint32_t>(result.status_),
            0,
            result.processId_,
            result.componentHash_,
            0.,
            result.totalRate_,
            result.residualQuantile_,
            result.selected_};
  }
  CORSIKA_HOST_DEVICE inline Outcome outcome(
      reference::RateColumnSelectionResult result) {
    return {static_cast<std::uint32_t>(result.status),
            0,
            result.process_id,
            result.component_hash,
            0.,
            result.total_rate,
            result.residual_quantile,
            result.selected};
  }
  CORSIKA_HOST_DEVICE inline Outcome evaluate(current::ProposalNativeDeviceView view,
                                              Query const& query) {
    if (query.invalidView_) view = {};
    switch (query.operation_) {
      case Operation::Rate:
        return outcome(current::query_discrete_rate(view, query.pid_, query.process_,
                                                    query.component_, query.energy_));
      case Operation::Total:
        return outcome(
            current::query_discrete_total_rate(view, query.pid_, query.energy_));
      case Operation::Select:
        return outcome(current::select_rate_column_by_uniform(
            view, query.pid_, query.energy_, query.argument_));
      case Operation::Loss:
        return outcome(current::query_discrete_loss_fraction(
            view, query.pid_, query.process_, query.component_, query.energy_,
            query.argument_));
      case Operation::Encode: {
        double value = -7.;
        bool const valid = current::encode_photon_pair_normalized_split(
            query.energy_, query.argument_, value);
        return {valid ? 1u : 0u, 0, 0, 0, value, 0., 0., 0};
      }
      case Operation::Decode: {
        double value = -7.;
        bool const valid = current::decode_photon_pair_normalized_split(
            query.energy_, query.argument_, value);
        return {valid ? 1u : 0u, 0, 0, 0, value, 0., 0., 0};
      }
      case Operation::Count:
        break;
    }
    return {};
  }
  CORSIKA_HOST_DEVICE inline Outcome expected(reference::ProposalNativeDeviceView table,
                                              Query const& query) {
    reference::NativePhysicsView view{};
    view.proposal_native =
        query.invalidView_ ? reference::ProposalNativeDeviceView{} : table;
    switch (query.operation_) {
      case Operation::Rate:
        return outcome(reference::queryRate(view, query.pid_, query.process_,
                                            query.component_, query.energy_));
      case Operation::Total:
        return outcome(reference::queryTotalRate(view, query.pid_, query.energy_));
      case Operation::Select:
        return outcome(reference::selectRateColumnByUniform(
            view, query.pid_, query.energy_, query.argument_));
      case Operation::Loss:
        return outcome(reference::queryLossFraction(view, query.pid_, query.process_,
                                                    query.component_, query.energy_,
                                                    query.argument_));
      case Operation::Encode: {
        double value = -7.;
        bool const valid = reference::encodePhotonPairNormalizedSplit(
            query.energy_, query.argument_, value);
        return {valid ? 1u : 0u, 0, 0, 0, value, 0., 0., 0};
      }
      case Operation::Decode: {
        double value = -7.;
        bool const valid = reference::decodePhotonPairNormalizedSplit(
            query.energy_, query.argument_, value);
        return {valid ? 1u : 0u, 0, 0, 0, value, 0., 0., 0};
      }
      case Operation::Count:
        break;
    }
    return {};
  }

  inline std::vector<Query> queries(current::ProposalNativeTableSet const& table) {
    std::vector<Query> result;
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const infinity = std::numeric_limits<double>::infinity();
    std::array<double, 13> const quantiles{
        -1.,  0.,          std::nextafter(0., 1.), 1.e-12, .001, .25, .5,
        .999, 1. - 1.e-12, std::nextafter(1., 0.), 1.,     1.1,  nan};
    for (auto const& column : table.dndxColumns_) {
      bool const spline = column.rateModel_ == current::NativeRateModel::BicubicSpline;
      double const low = spline ? column.spline_.energyAxis_.low_
                                : std::max(1., std::max(column.lowerEnergyLimitMeV_,
                                                        column.particleMassMeV_));
      double const high = spline ? column.spline_.energyAxis_.high_ : 1.e8;
      double const middle = std::sqrt(low * high);
      for (double energy :
           {low, std::nextafter(low, 0.), std::nextafter(low, infinity), middle, high,
            std::nextafter(high, infinity), -1., 0., nan, infinity}) {
        result.push_back({Operation::Rate, column.pdgId_, column.processId_,
                          column.componentHash_, energy, .5, false});
      }
      for (double energy : {low, middle, high}) {
        for (double quantile : quantiles) {
          Query query{Operation::Loss,
                      column.pdgId_,
                      column.processId_,
                      column.componentHash_,
                      energy,
                      quantile,
                      false};
          result.push_back(query);
          if (column.pdgId_ == 22 &&
              (column.processId_ == current::PhotonPairProcessId ||
               column.processId_ == current::PhotonMuonPairProcessId)) {
            query.process_ += 1000000000;
            result.push_back(query);
          }
        }
      }
      // These auxiliary identities must never add a rate column.
      if (column.pdgId_ == 22 && (column.processId_ == current::PhotonPairProcessId ||
                                  column.processId_ == current::PhotonMuonPairProcessId))
        result.push_back({Operation::Rate, 22, column.processId_ + 1000000000,
                          column.componentHash_, middle, .5, false});
    }
    native_query_test::HostReference const owner(table);
    reference::NativePhysicsView original{};
    original.proposal_native = owner.view(table);
    for (auto const& column : table.totalRateColumns_) {
      for (double energy : {column.spline_.axis_.low_, 1.e3, 1.e5, 1.e7,
                            column.spline_.axis_.high_, -1., 0., nan, infinity}) {
        result.push_back({Operation::Total, column.pdgId_, 0, 0, energy, .5, false});
        for (double uniform : quantiles)
          result.push_back(
              {Operation::Select, column.pdgId_, 0, 0, energy, uniform, false});
      }
      // Probe both adjacent representable numbers at every ordered column boundary.
      for (double energy : {1.e3, 1.e5, 1.e7}) {
        auto const total = reference::queryProposalNativeSelectionTotalRate(
            original.proposal_native, column.pdgId_, energy);
        if (total.status != reference::NativeQueryStatus::Success || !(total.value > 0.))
          continue;
        double cumulative = 0.;
        auto const range = reference::native_detail::dndxParticleRange(
            original.proposal_native, column.pdgId_);
        for (auto position = range.begin; position < range.end; ++position) {
          auto const* item =
              reference::native_detail::selectionDndx(original.proposal_native, position);
          auto const rate = reference::queryProposalNativeRateForColumn(
              original.proposal_native, *item, energy);
          if (rate.status != reference::NativeQueryStatus::Success) continue;
          cumulative += rate.value;
          double const boundary = cumulative / total.value;
          for (double uniform : {std::nextafter(boundary, 0.), boundary,
                                 std::nextafter(boundary, infinity)})
            result.push_back(
                {Operation::Select, column.pdgId_, 0, 0, energy, uniform, false});
        }
      }
    }
    for (auto operation :
         {Operation::Rate, Operation::Total, Operation::Select, Operation::Loss}) {
      for (bool invalid : {false, true}) {
        for (std::int32_t pid : {11, 22, 999}) {
          for (double energy : {1.e5, nan})
            result.push_back({operation, pid, -1, 0, energy, .5, invalid});
        }
      }
    }
    for (double energy : {0., 2. * current::ElectronMassGeV,
                          std::nextafter(2. * current::ElectronMassGeV, infinity), .0011,
                          1., 1.e8, nan, infinity}) {
      for (double argument : quantiles) {
        result.push_back({Operation::Encode, 22, 0, 0, energy, argument, false});
        result.push_back({Operation::Decode, 22, 0, 0, energy, argument, false});
      }
      if (std::isfinite(energy) && energy > 2. * current::ElectronMassGeV) {
        double const lower = current::ElectronMassGeV / energy;
        double const tolerance = 64. * 2.2204460492503131e-16;
        for (double fraction :
             {lower, 1. - lower, lower - tolerance, lower - 2. * tolerance,
              1. - lower + tolerance, 1. - lower + 2. * tolerance})
          result.push_back({Operation::Encode, 22, 0, 0, energy, fraction, false});
      }
    }
    return result;
  }
} // namespace discrete_test
