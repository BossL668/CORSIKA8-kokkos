/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "ProposalNativeReference.hpp"
#include <corsika/kokkos/em/tables/ProposalNativeQueries.hpp>
#include <corsika/kokkos/em/tables/ProposalNativeTable.hpp>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace native_query_test {
  namespace current = corsika::kokkos;
  namespace reference = native_table_reference;

  inline reference::NativeQueryResult convert(current::NativeQueryResult const& source) {
    reference::NativeQueryResult result{};
    result.status = static_cast<reference::NativeQueryStatus>(source.status_);
    result.newton_iterations = source.newtonIterations_;
    result.bisection_iterations = source.bisectionIterations_;
    result.value = source.value_;
    return result;
  }

  inline reference::NativeAxisDescriptor convert(
      current::NativeAxisDescriptor const& source) {
    reference::NativeAxisDescriptor result{};
    result.type = static_cast<reference::NativeAxisType>(source.type_);
    result.nodes = source.nodes_;
    result.low = source.low_;
    result.high = source.high_;
    result.step = source.step_;
    return result;
  }

  inline reference::NativeBicubicDescriptor convert(
      current::NativeBicubicDescriptor const& source) {
    reference::NativeBicubicDescriptor result{};
    result.energy_axis = convert(source.energyAxis_);
    result.loss_axis = convert(source.lossAxis_);
    result.rows = source.rows_;
    result.columns = source.columns_;
    result.coefficient_offset = source.coefficientOffset_;
    result.polynomial_coefficient_offset = source.polynomialCoefficientOffset_;
    return result;
  }

  inline reference::NativeCubicDescriptor convert(
      current::NativeCubicDescriptor const& source) {
    reference::NativeCubicDescriptor result{};
    result.axis = convert(source.axis_);
    result.value_transform = convert(source.valueTransform_);
    result.coefficient_offset = source.coefficientOffset_;
    result.coefficient_count = source.coefficientCount_;
    result.has_value_transform = source.hasValueTransform_;
    return result;
  }

  inline reference::NativeDndxColumn convert(current::NativeDndxColumn const& source) {
    reference::NativeDndxColumn result{};
    result.pdg_id = source.pdgId_;
    result.process_id = source.processId_;
    result.medium_hash = source.mediumHash_;
    result.component_hash = source.componentHash_;
    result.cross_section_hash = source.crossSectionHash_;
    result.table_hash = source.tableHash_;
    result.loss_transform =
        static_cast<reference::NativeLossTransform>(source.lossTransform_);
    result.kinematic_model =
        static_cast<reference::NativeKinematicModel>(source.kinematicModel_);
    result.rate_model = static_cast<reference::NativeRateModel>(source.rateModel_);
    result.spline = convert(source.spline_);
    result.component_weight = source.componentWeight_;
    result.particle_mass_MeV = source.particleMassMeV_;
    result.component_nuclear_charge = source.componentNuclearCharge_;
    result.component_atomic_mass = source.componentAtomicMass_;
    result.component_average_nucleon_mass_MeV = source.componentAverageNucleonMassMeV_;
    result.direct_rate_factor = source.directRateFactor_;
    result.lower_energy_limit_MeV = source.lowerEnergyLimitMeV_;
    result.absolute_cut_MeV = source.absoluteCutMeV_;
    result.relative_cut = source.relativeCut_;
    result.medium_mean_excitation_energy_MeV = source.mediumMeanExcitationEnergyMeV_;
    result.selection_ordinal = source.selectionOrdinal_;
    result.reserved = source.reserved_;
    return result;
  }

  inline reference::NativeTotalRateColumn convert(
      current::NativeTotalRateColumn const& source) {
    reference::NativeTotalRateColumn result{};
    result.pdg_id = source.pdgId_;
    result.reserved = source.reserved_;
    result.medium_hash = source.mediumHash_;
    result.table_hash = source.tableHash_;
    result.spline = convert(source.spline_);
    result.lower_energy_limit_MeV = source.lowerEnergyLimitMeV_;
    return result;
  }

  inline reference::NativeDedxColumn convert(current::NativeDedxColumn const& source) {
    reference::NativeDedxColumn result{};
    result.pdg_id = source.pdgId_;
    result.process_id = source.processId_;
    result.medium_hash = source.mediumHash_;
    result.cross_section_hash = source.crossSectionHash_;
    result.component_index = source.componentIndex_;
    result.table_hash = source.tableHash_;
    result.spline = convert(source.spline_);
    result.component_weight = source.componentWeight_;
    result.lower_energy_limit_MeV = source.lowerEnergyLimitMeV_;
    result.accumulation_ordinal = source.accumulationOrdinal_;
    result.reserved = source.reserved_;
    return result;
  }

  inline reference::NativeUtilityColumn convert(
      current::NativeUtilityColumn const& source) {
    reference::NativeUtilityColumn result{};
    result.pdg_id = source.pdgId_;
    result.medium_hash = source.mediumHash_;
    result.table_hash = source.tableHash_;
    result.spline = convert(source.spline_);
    result.particle_mass_MeV = source.particleMassMeV_;
    result.lower_energy_limit_MeV = source.lowerEnergyLimitMeV_;
    result.reverse = source.reverse_;
    return result;
  }

  inline reference::NativePhysicalConstants convert(
      current::NativePhysicalConstants const& source) {
    reference::NativePhysicalConstants result{};
    result.electron_mass_MeV = source.electronMassMeV_;
    result.charged_pion_mass_MeV = source.chargedPionMassMeV_;
    result.fine_structure_constant = source.fineStructureConstant_;
    result.classical_electron_radius_cm = source.classicalElectronRadiusCm_;
    result.pi = source.pi_;
    result.sqrt_e = source.sqrtE_;
    result.muon_mass_MeV = source.muonMassMeV_;
    return result;
  }

  inline reference::NativeInteractionIdentity convert(
      current::NativeInteractionIdentity const& source) {
    reference::NativeInteractionIdentity result{};
    result.pdg_id = source.pdgId_;
    result.reserved = source.reserved_;
    result.medium_hash = source.mediumHash_;
    result.interaction_hash = source.interactionHash_;
    return result;
  }

  // A separate metadata owner makes mapping explicit; no aliasing cast is used.
  struct HostReference {
    std::vector<reference::NativeDndxColumn> dndxColumns_;
    std::vector<reference::NativeTotalRateColumn> totalRateColumns_;
    std::vector<std::uint32_t> selectionColumnIndices_;
    std::vector<reference::NativeDedxColumn> dedxColumns_;
    std::vector<std::uint32_t> dedxAccumulationIndices_;
    std::vector<reference::NativeUtilityColumn> utilityColumns_;
    explicit HostReference(current::ProposalNativeTableSet const& source) {
      for (auto const& value : source.dndxColumns_)
        dndxColumns_.push_back(convert(value));
      for (auto const& value : source.totalRateColumns_)
        totalRateColumns_.push_back(convert(value));
      selectionColumnIndices_ = source.selectionColumnIndices_;
      for (auto const& value : source.dedxColumns_)
        dedxColumns_.push_back(convert(value));
      dedxAccumulationIndices_ = source.dedxAccumulationIndices_;
      for (auto const& value : source.utilityColumns_)
        utilityColumns_.push_back(convert(value));
    }
    reference::ProposalNativeDeviceView view(
        current::ProposalNativeTableSet const& source) const {
      reference::ProposalNativeDeviceView result{};
      result.dndx_columns = dndxColumns_.data();
      result.total_rate_columns = totalRateColumns_.data();
      result.selection_column_indices = selectionColumnIndices_.data();
      result.dedx_columns = dedxColumns_.data();
      result.dedx_accumulation_indices = dedxAccumulationIndices_.data();
      result.utility_columns = utilityColumns_.data();
      result.bicubic_values = source.bicubicValues_.data();
      result.bicubic_derivative_energy = source.bicubicDerivativeEnergy_.data();
      result.bicubic_derivative_loss = source.bicubicDerivativeLoss_.data();
      result.bicubic_mixed_derivative = source.bicubicMixedDerivative_.data();
      result.bicubic_polynomial_coefficients =
          source.bicubicPolynomialCoefficients_.data();
      result.cubic_values = source.cubicValues_.data();
      result.cubic_node_derivatives = source.cubicNodeDerivatives_.data();
      result.dndx_column_count = static_cast<std::uint32_t>(dndxColumns_.size());
      result.total_rate_column_count =
          static_cast<std::uint32_t>(totalRateColumns_.size());
      result.selection_column_index_count =
          static_cast<std::uint32_t>(selectionColumnIndices_.size());
      result.dedx_column_count = static_cast<std::uint32_t>(dedxColumns_.size());
      result.dedx_accumulation_index_count =
          static_cast<std::uint32_t>(dedxAccumulationIndices_.size());
      result.utility_column_count = static_cast<std::uint32_t>(utilityColumns_.size());
      result.bicubic_coefficient_count = source.bicubicValues_.size();
      result.bicubic_polynomial_coefficient_count =
          source.bicubicPolynomialCoefficients_.size();
      result.cubic_coefficient_count = source.cubicValues_.size();
      result.constants = convert(source.constants_);
      return result;
    }
  };

  template <typename TMemorySpace>
  struct UploadedReference {
    template <typename TValue>
    using array_type = Kokkos::View<TValue*, TMemorySpace>;
    array_type<reference::NativeDndxColumn> dndxColumns_;
    array_type<reference::NativeTotalRateColumn> totalRateColumns_;
    array_type<std::uint32_t> selectionColumnIndices_;
    array_type<reference::NativeDedxColumn> dedxColumns_;
    array_type<std::uint32_t> dedxAccumulationIndices_;
    array_type<reference::NativeUtilityColumn> utilityColumns_;
    array_type<double> bicubicValues_;
    array_type<double> bicubicDerivativeEnergy_;
    array_type<double> bicubicDerivativeLoss_;
    array_type<double> bicubicMixedDerivative_;
    array_type<double> bicubicPolynomialCoefficients_;
    array_type<double> cubicValues_;
    array_type<double> cubicNodeDerivatives_;
    reference::NativePhysicalConstants constants_{};
    template <typename TValue>
    static array_type<TValue> upload(std::vector<TValue> const& source) {
      array_type<TValue> result("independent_native_table_array", source.size());
      auto mirror = Kokkos::create_mirror_view(result);
      for (std::size_t i = 0; i < source.size(); ++i) mirror(i) = source[i];
      Kokkos::deep_copy(result, mirror);
      return result;
    }
    explicit UploadedReference(current::ProposalNativeTableSet const& source) {
      HostReference const records(source);
      dndxColumns_ = upload(records.dndxColumns_);
      totalRateColumns_ = upload(records.totalRateColumns_);
      selectionColumnIndices_ = upload(records.selectionColumnIndices_);
      dedxColumns_ = upload(records.dedxColumns_);
      dedxAccumulationIndices_ = upload(records.dedxAccumulationIndices_);
      utilityColumns_ = upload(records.utilityColumns_);
      bicubicValues_ = upload(source.bicubicValues_);
      bicubicDerivativeEnergy_ = upload(source.bicubicDerivativeEnergy_);
      bicubicDerivativeLoss_ = upload(source.bicubicDerivativeLoss_);
      bicubicMixedDerivative_ = upload(source.bicubicMixedDerivative_);
      bicubicPolynomialCoefficients_ = upload(source.bicubicPolynomialCoefficients_);
      cubicValues_ = upload(source.cubicValues_);
      cubicNodeDerivatives_ = upload(source.cubicNodeDerivatives_);
      constants_ = convert(source.constants_);
    }
    reference::ProposalNativeDeviceView view() const {
      reference::ProposalNativeDeviceView result{};
      result.dndx_columns = dndxColumns_.data();
      result.total_rate_columns = totalRateColumns_.data();
      result.selection_column_indices = selectionColumnIndices_.data();
      result.dedx_columns = dedxColumns_.data();
      result.dedx_accumulation_indices = dedxAccumulationIndices_.data();
      result.utility_columns = utilityColumns_.data();
      result.bicubic_values = bicubicValues_.data();
      result.bicubic_derivative_energy = bicubicDerivativeEnergy_.data();
      result.bicubic_derivative_loss = bicubicDerivativeLoss_.data();
      result.bicubic_mixed_derivative = bicubicMixedDerivative_.data();
      result.bicubic_polynomial_coefficients = bicubicPolynomialCoefficients_.data();
      result.cubic_values = cubicValues_.data();
      result.cubic_node_derivatives = cubicNodeDerivatives_.data();
      result.dndx_column_count = static_cast<std::uint32_t>(dndxColumns_.extent(0));
      result.total_rate_column_count =
          static_cast<std::uint32_t>(totalRateColumns_.extent(0));
      result.selection_column_index_count =
          static_cast<std::uint32_t>(selectionColumnIndices_.extent(0));
      result.dedx_column_count = static_cast<std::uint32_t>(dedxColumns_.extent(0));
      result.dedx_accumulation_index_count =
          static_cast<std::uint32_t>(dedxAccumulationIndices_.extent(0));
      result.utility_column_count = static_cast<std::uint32_t>(utilityColumns_.extent(0));
      result.bicubic_coefficient_count = bicubicValues_.extent(0);
      result.bicubic_polynomial_coefficient_count =
          bicubicPolynomialCoefficients_.extent(0);
      result.cubic_coefficient_count = cubicValues_.extent(0);
      result.constants = constants_;
      return result;
    }
  };

  enum class Operation : std::uint32_t {
    BelowThreshold,
    Cumulative,
    ColumnRate,
    Rate,
    TotalRate,
    SelectionRate,
    Dedx,
    LossFraction,
    Range,
    EnergyAfterLoss,
    Energy,
    Count
  };

  struct Query {
    Operation operation_{};
    std::uint32_t column_{};
    std::int32_t pid_{};
    std::int32_t process_{};
    std::uint64_t component_{};
    double energy_{};
    double argument_{};
    bool photonSplit_{};
    bool invalidView_{};
    bool crossDevice_{};
  };

  struct Outcome {
    std::uint32_t status_{};
    std::uint32_t newton_{};
    std::uint32_t bisection_{};
    double value_{};
  };

  // Copy returned results by value: the oracle never borrows a temporary result.
  KOKKOS_INLINE_FUNCTION Outcome outcome(current::NativeQueryResult value) {
    return {static_cast<std::uint32_t>(value.status_), value.newtonIterations_,
            value.bisectionIterations_, value.value_};
  }
  KOKKOS_INLINE_FUNCTION Outcome outcome(reference::NativeQueryResult value) {
    return {static_cast<std::uint32_t>(value.status), value.newton_iterations,
            value.bisection_iterations, value.value};
  }

  KOKKOS_INLINE_FUNCTION Outcome evaluate(current::ProposalNativeDeviceView view,
                                          Query const& query) {
    auto const columns = view.getDndxColumns();
    if (query.invalidView_) view = {};
    switch (query.operation_) {
      case Operation::BelowThreshold:
        return {0, 0, 0,
                current::proposal_native_rate_below_threshold(columns[query.column_],
                                                              query.energy_)
                    ? 1.
                    : 0.};
      case Operation::Cumulative:
        return outcome(current::query_proposal_native_cumulative_rate(
            view, columns[query.column_], query.energy_, query.argument_));
      case Operation::ColumnRate:
        return outcome(current::query_proposal_native_rate_for_column(
            view, columns[query.column_], query.energy_));
      case Operation::Rate:
        return outcome(current::query_proposal_native_rate(
            view, query.pid_, query.process_, query.component_, query.energy_));
      case Operation::TotalRate:
        return outcome(
            current::query_proposal_native_total_rate(view, query.pid_, query.energy_));
      case Operation::SelectionRate:
        return outcome(current::query_proposal_native_selection_total_rate(
            view, query.pid_, query.energy_));
      case Operation::Dedx:
        return outcome(
            current::query_proposal_native_dedx(view, query.pid_, query.energy_));
      case Operation::LossFraction:
        return outcome(current::query_proposal_native_loss_fraction(
            view, query.pid_, query.process_, query.component_, query.energy_,
            query.argument_, query.photonSplit_));
      case Operation::Range:
        return outcome(
            current::query_proposal_native_range(view, query.pid_, query.energy_));
      case Operation::EnergyAfterLoss:
        return outcome(current::query_proposal_native_energy_after_continuous_loss(
            view, query.pid_, query.energy_, query.argument_));
      case Operation::Energy:
        return outcome(
            current::query_proposal_native_energy(view, query.pid_, query.argument_));
      case Operation::Count:
        break;
    }
    return {};
  }

  KOKKOS_INLINE_FUNCTION Outcome expected(reference::ProposalNativeDeviceView view,
                                          Query const& query) {
    auto const* columns = view.dndx_columns;
    if (query.invalidView_) view = {};
    switch (query.operation_) {
      case Operation::BelowThreshold:
        return {0, 0, 0,
                reference::proposalNativeRateBelowThreshold(columns[query.column_],
                                                            query.energy_)
                    ? 1.
                    : 0.};
      case Operation::Cumulative:
        return outcome(reference::queryProposalNativeCumulativeRate(
            view, columns[query.column_], query.energy_, query.argument_));
      case Operation::ColumnRate:
        return outcome(reference::queryProposalNativeRateForColumn(
            view, columns[query.column_], query.energy_));
      case Operation::Rate:
        return outcome(reference::queryProposalNativeRate(
            view, query.pid_, query.process_, query.component_, query.energy_));
      case Operation::TotalRate:
        return outcome(
            reference::queryProposalNativeTotalRate(view, query.pid_, query.energy_));
      case Operation::SelectionRate:
        return outcome(reference::queryProposalNativeSelectionTotalRate(view, query.pid_,
                                                                        query.energy_));
      case Operation::Dedx:
        return outcome(
            reference::queryProposalNativeDedx(view, query.pid_, query.energy_));
      case Operation::LossFraction:
        return outcome(reference::queryProposalNativeLossFraction(
            view, query.pid_, query.process_, query.component_, query.energy_,
            query.argument_, query.photonSplit_));
      case Operation::Range:
        return outcome(
            reference::queryProposalNativeRange(view, query.pid_, query.energy_));
      case Operation::EnergyAfterLoss:
        return outcome(reference::queryProposalNativeEnergyAfterContinuousLoss(
            view, query.pid_, query.energy_, query.argument_));
      case Operation::Energy:
        return outcome(
            reference::queryProposalNativeEnergy(view, query.pid_, query.argument_));
      case Operation::Count:
        break;
    }
    return {};
  }

  // All inputs, including range-derived grammages, are prepared once on the host.
  // The independent production evaluator, never the migrated query, prepares ranges.
  inline std::vector<Query> queries(current::ProposalNativeTableSet const& table) {
    HostReference const owner(table);
    auto const view = owner.view(table);
    std::vector<Query> result;
    auto const nan = std::numeric_limits<double>::quiet_NaN();
    auto const infinity = std::numeric_limits<double>::infinity();
    for (std::uint32_t index = 0; index < table.dndxColumns_.size(); ++index) {
      auto const& column = table.dndxColumns_[index];
      bool const spline = column.rateModel_ == current::NativeRateModel::BicubicSpline;
      auto const low = spline ? column.spline_.energyAxis_.low_
                              : std::max(1., std::max(column.lowerEnergyLimitMeV_,
                                                      column.particleMassMeV_));
      auto const high = spline ? column.spline_.energyAxis_.high_ : 1.e8;
      auto const middle = std::sqrt(low * high);
      std::array<double, 10> const energies{low,
                                            std::nextafter(low, 0.),
                                            std::nextafter(low, infinity),
                                            middle,
                                            high,
                                            std::nextafter(high, infinity),
                                            -1.,
                                            0.,
                                            nan,
                                            infinity};
      for (auto energy : energies) {
        Query query{Operation::BelowThreshold,
                    index,
                    column.pdgId_,
                    column.processId_,
                    column.componentHash_,
                    energy,
                    0.,
                    false,
                    false,
                    false};
        result.push_back(query);
        query.operation_ = Operation::ColumnRate;
        query.crossDevice_ = energy == middle;
        result.push_back(query);
        query.operation_ = Operation::Rate;
        result.push_back(query);
        if (spline) {
          query.operation_ = Operation::Cumulative;
          for (double coordinate : {0., .125, .5, .875, 1.}) {
            query.argument_ = coordinate;
            // Near-zero CDF differences are not a well-conditioned relative check.
            query.crossDevice_ = energy == middle && coordinate == 1.;
            result.push_back(query);
          }
        }
      }
      for (double energy : {middle, high}) {
        for (double quantile :
             {-0.1, 0., 1.e-10, .01, .5, .99, 1. - 1.e-10, 1., 1.1, nan}) {
          Query query{Operation::LossFraction,
                      index,
                      column.pdgId_,
                      column.processId_,
                      column.componentHash_,
                      energy,
                      quantile,
                      false,
                      false,
                      false};
          result.push_back(query);
          query.photonSplit_ = true;
          result.push_back(query);
        }
      }
    }
    for (auto const& column : table.totalRateColumns_) {
      auto const low = column.spline_.axis_.low_;
      auto const high = column.spline_.axis_.high_;
      for (double energy : {low, std::nextafter(low, 0.), std::sqrt(low * high), high,
                            std::nextafter(high, infinity), -1., nan, infinity}) {
        for (auto operation :
             {Operation::TotalRate, Operation::SelectionRate, Operation::Dedx})
          result.push_back({operation, 0, column.pdgId_, 0, 0, energy, 0., false, false,
                            energy == std::sqrt(low * high)});
      }
    }
    for (auto const& column : table.utilityColumns_) {
      auto const low = std::max(column.lowerEnergyLimitMeV_, column.spline_.axis_.low_);
      auto const high = column.spline_.axis_.high_;
      for (double energy :
           {low, std::nextafter(low, 0.), low * 1.01, std::sqrt(low * high), high,
            std::nextafter(high, infinity), nan}) {
        Query query{Operation::Range, 0,  column.pdgId_, 0,     0,
                    energy,           0., false,         false, false};
        result.push_back(query);
        auto const range =
            reference::queryProposalNativeRange(view, column.pdgId_, energy);
        if (range.status == reference::NativeQueryStatus::Success) {
          for (double fraction : {0., .01, .1, .5, 1., 1.1}) {
            query.argument_ = range.value * fraction;
            query.operation_ = Operation::EnergyAfterLoss;
            result.push_back(query);
            query.operation_ = Operation::Energy;
            result.push_back(query);
          }
        }
        for (double bad : {-1., nan, infinity}) {
          query.argument_ = bad;
          query.operation_ = Operation::EnergyAfterLoss;
          result.push_back(query);
          query.operation_ = Operation::Energy;
          result.push_back(query);
        }
      }
    }
    // Missing identifiers and default views exercise each entry point's own guard order.
    if (!table.dndxColumns_.empty()) {
      for (auto operation : {Operation::Cumulative, Operation::ColumnRate})
        result.push_back({operation, 0, 0, 0, 0, nan, .5, false, true, false});
    }
    for (auto operation :
         {Operation::Rate, Operation::TotalRate, Operation::SelectionRate,
          Operation::Dedx, Operation::LossFraction, Operation::Range,
          Operation::EnergyAfterLoss, Operation::Energy}) {
      for (bool invalid : {false, true}) {
        for (double energy : {1000., nan}) {
          result.push_back(
              {operation, 0, 123456, 123456, 0, energy, .5, false, invalid, false});
        }
      }
    }
    return result;
  }

  template <typename TScalar, typename TState>
  void compare(Outcome const& actual, Outcome const& expectedValue, TScalar const& scalar,
               TState const& state, bool iterations = true) {
    state(actual.status_, expectedValue.status_);
    if (iterations) {
      state(actual.newton_, expectedValue.newton_);
      state(actual.bisection_, expectedValue.bisection_);
    }
    scalar(actual.value_, expectedValue.value_);
  }
} // namespace native_query_test
