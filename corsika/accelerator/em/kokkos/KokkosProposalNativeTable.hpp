/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <corsika/accelerator/em/common/tables/ProposalNativeQueries.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  namespace table = gpu::em::tables;

  KOKKOS_INLINE_FUNCTION table::NativeQueryResult executeNativeQuery(
      table::ProposalNativeDeviceView const& view,
      table::ProposalNativeQuery const& query) {
    switch (query.kind) {
    case table::ProposalNativeQueryKind::Rate:
      return table::queryProposalNativeRate(
          view, query.pdg_id, query.process_id, query.component_hash,
          query.energy_MeV);
    case table::ProposalNativeQueryKind::TotalRate:
      return table::queryProposalNativeTotalRate(
          view, query.pdg_id, query.energy_MeV);
    case table::ProposalNativeQueryKind::CumulativeRate: {
      auto const* column = table::native_detail::findDndx(
          view, query.pdg_id, query.process_id, query.component_hash);
      if (!column) {
        return {table::NativeQueryStatus::ColumnNotFound, 0, 0, 0.};
      }
      return table::queryProposalNativeCumulativeRate(
          view, *column, query.energy_MeV, query.argument);
    }
    case table::ProposalNativeQueryKind::LossFraction:
      return table::queryProposalNativeLossFraction(
          view, query.pdg_id, query.process_id, query.component_hash,
          query.energy_MeV, query.argument);
    case table::ProposalNativeQueryKind::ContinuousDedx:
      return table::queryProposalNativeDedx(
          view, query.pdg_id, query.energy_MeV);
    case table::ProposalNativeQueryKind::ContinuousRange:
      return table::queryProposalNativeRange(
          view, query.pdg_id, query.energy_MeV);
    case table::ProposalNativeQueryKind::ContinuousEnergy:
      return table::queryProposalNativeEnergy(
          view, query.pdg_id, query.argument);
    case table::ProposalNativeQueryKind::ContinuousEnergyAfterLoss:
      return table::queryProposalNativeEnergyAfterContinuousLoss(
          view, query.pdg_id, query.energy_MeV, query.argument);
    }
    return {table::NativeQueryStatus::InvalidView, 0, 0, 0.};
  }

  KOKKOS_INLINE_FUNCTION table::ProposalNativeSelectionResult
  executeNativeSelection(
      table::ProposalNativeDeviceView const& view,
      table::ProposalNativeSelectionQuery const& query) {
    using table::NativeQueryStatus;
    if (!table::native_detail::validView(view)) {
      return {NativeQueryStatus::InvalidView, 0, 0, 0., 0., 0., 0};
    }
    if (!table::native_detail::finite(query.energy_MeV) ||
        !table::native_detail::finite(query.threshold) ||
        query.threshold < 0.) {
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0., 0., 0., 0};
    }
    auto const range =
        table::native_detail::dndxParticleRange(view, query.pdg_id);
    if (range.begin == range.end) {
      return {NativeQueryStatus::ParticleNotFound, 0, 0, 0., 0., 0., 0};
    }

    double cumulative = 0.;
    double boundary_cumulative = 0.;
    std::int32_t process = 0;
    std::uint64_t component = 0;
    double residual_quantile = 0.;
    bool selected = false;
    for (auto index = range.begin; index < range.end; ++index) {
      auto const* selected_column =
          table::native_detail::selectionDndx(view, index);
      if (!selected_column || selected_column->pdg_id != query.pdg_id) {
        return {NativeQueryStatus::InvalidView, 0, 0, cumulative,
                boundary_cumulative, 0., 0};
      }
      auto const& column = *selected_column;
      table::NativeQueryResult rate{NativeQueryStatus::Success, 0, 0, 0.};
      if (!table::proposalNativeRateBelowThreshold(
              column, query.energy_MeV)) {
        rate = table::queryProposalNativeRateForColumn(
            view, column, query.energy_MeV);
      }
      if (rate.status != NativeQueryStatus::Success) {
        return {rate.status, column.process_id, column.component_hash,
                cumulative, boundary_cumulative, 0., 0};
      }
      cumulative += rate.value;
      if (column.process_id == query.boundary_process_id &&
          column.component_hash == query.boundary_component_hash) {
        boundary_cumulative = cumulative;
      }
      if (!selected && query.threshold < cumulative) {
        selected = true;
        process = column.process_id;
        component = column.component_hash;
        residual_quantile = rate.value > 0.
                                ? (cumulative - query.threshold) / rate.value
                                : 0.;
      }
    }
    return {NativeQueryStatus::Success, process, component, cumulative,
            boundary_cumulative, residual_quantile, selected ? 1u : 0u};
  }

  template <class ExecutionSpace>
  struct NativeQueryKernel {
    table::ProposalNativeDeviceView table_view{};
    Kokkos::View<table::ProposalNativeQuery const*,
                 typename ExecutionSpace::memory_space>
        queries;
    Kokkos::View<table::NativeQueryResult*,
                 typename ExecutionSpace::memory_space>
        results;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index) const {
      results(index) = executeNativeQuery(table_view, queries(index));
    }
  };

  template <class ExecutionSpace>
  struct NativeSelectionKernel {
    table::ProposalNativeDeviceView table_view{};
    Kokkos::View<table::ProposalNativeSelectionQuery const*,
                 typename ExecutionSpace::memory_space>
        queries;
    Kokkos::View<table::ProposalNativeSelectionResult*,
                 typename ExecutionSpace::memory_space>
        results;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index) const {
      results(index) = executeNativeSelection(table_view, queries(index));
    }
  };

  /**
   * Immutable, execution-space-native copy of PROPOSAL's own interpolants.
   * The owning Views differ by backend; the POD view consumed by physics is
   * intentionally identical to native CUDA's ProposalNativeDeviceView.
   */
  template <class ExecutionSpace>
  class KokkosProposalNativeTable {
  public:
    using execution_space = ExecutionSpace;
    using memory_space = typename ExecutionSpace::memory_space;

    static std::size_t projectedDeviceBytes(
        table::ProposalNativeTableSet const& source) {
      return table::proposalNativeTableBytes(source);
    }

    void initialize(
        table::ProposalNativeTableSet const& source,
        std::size_t const maximum_bytes =
            std::numeric_limits<std::size_t>::max()) {
      if (initialized_) {
        throw std::logic_error("Kokkos PROPOSAL native table is already initialized");
      }
      table::validateProposalNativeTable(source);
      auto const calculated_hash = table::calculateProposalNativeHash(source);
      if (calculated_hash != source.content_hash) {
        throw std::invalid_argument(
            "PROPOSAL native-table content hash does not match its payload");
      }
      auto const requested = projectedDeviceBytes(source);
      if (requested > maximum_bytes) {
        throw std::runtime_error(
            "PROPOSAL native table exceeds the configured Kokkos memory budget");
      }

      try {
        dndx_ = upload(source.dndx_columns, "proposal_native_dndx");
        total_rate_ = upload(source.total_rate_columns, "proposal_native_total_rate");
        selection_indices_ = upload(
            source.selection_column_indices, "proposal_native_selection_order");
        dedx_ = upload(source.dedx_columns, "proposal_native_dedx");
        dedx_accumulation_indices_ = upload(
            source.dedx_accumulation_indices, "proposal_native_dedx_order");
        utility_ = upload(source.utility_columns, "proposal_native_utility");
        bicubic_values_ = upload(source.bicubic_values, "proposal_native_bicubic_values");
        bicubic_denergy_ = upload(
            source.bicubic_derivative_energy, "proposal_native_bicubic_denergy");
        bicubic_dloss_ = upload(
            source.bicubic_derivative_loss, "proposal_native_bicubic_dloss");
        bicubic_mixed_ = upload(
            source.bicubic_mixed_derivative, "proposal_native_bicubic_mixed");
        bicubic_polynomial_ = upload(
            source.bicubic_polynomial_coefficients,
            "proposal_native_bicubic_polynomial");
        cubic_values_ = upload(source.cubic_values, "proposal_native_cubic_values");
        cubic_derivatives_ = upload(
            source.cubic_node_derivatives, "proposal_native_cubic_derivatives");
        execution_.fence("upload PROPOSAL native table");

        view_ = {
            dndx_.data(), total_rate_.data(), selection_indices_.data(),
            dedx_.data(), dedx_accumulation_indices_.data(), utility_.data(),
            static_cast<std::uint32_t>(source.dndx_columns.size()),
            static_cast<std::uint32_t>(source.total_rate_columns.size()),
            static_cast<std::uint32_t>(source.selection_column_indices.size()),
            static_cast<std::uint32_t>(source.dedx_columns.size()),
            static_cast<std::uint32_t>(source.dedx_accumulation_indices.size()),
            static_cast<std::uint32_t>(source.utility_columns.size()),
            bicubic_values_.data(), bicubic_denergy_.data(),
            bicubic_dloss_.data(), bicubic_mixed_.data(),
            source.bicubic_values.size(), bicubic_polynomial_.data(),
            source.bicubic_polynomial_coefficients.size(), cubic_values_.data(),
            cubic_derivatives_.data(), source.cubic_values.size(),
            source.constants};
        hash_ = source.content_hash;
        bytes_ = requested;
        initialized_ = true;
      } catch (...) {
        reset();
        throw;
      }
    }

    void reset() noexcept {
      dndx_ = {};
      total_rate_ = {};
      selection_indices_ = {};
      dedx_ = {};
      dedx_accumulation_indices_ = {};
      utility_ = {};
      bicubic_values_ = {};
      bicubic_denergy_ = {};
      bicubic_dloss_ = {};
      bicubic_mixed_ = {};
      bicubic_polynomial_ = {};
      cubic_values_ = {};
      cubic_derivatives_ = {};
      view_ = {};
      hash_ = {};
      bytes_ = 0;
      initialized_ = false;
    }

    bool initialized() const noexcept { return initialized_; }
    std::size_t deviceBytes() const noexcept { return bytes_; }

    table::Sha256Digest const& sourceContentHash() const {
      requireInitialized();
      return hash_;
    }

    table::ProposalNativeDeviceView deviceView() const {
      requireInitialized();
      return view_;
    }

    std::vector<table::NativeQueryResult> queryForValidation(
        std::vector<table::ProposalNativeQuery> const& queries) const {
      requireInitialized();
      using QueryView = Kokkos::View<table::ProposalNativeQuery*, memory_space>;
      using ResultView = Kokkos::View<table::NativeQueryResult*, memory_space>;
      QueryView device_queries("proposal_native_queries", queries.size());
      ResultView device_results("proposal_native_results", queries.size());
      copyToDevice(queries, device_queries);
      Kokkos::parallel_for(
          "c8_kokkos_proposal_native_query",
          Kokkos::RangePolicy<ExecutionSpace>(execution_, 0, queries.size()),
          NativeQueryKernel<ExecutionSpace>{view_, device_queries, device_results});
      execution_.fence("query PROPOSAL native table");
      return copyToHost(device_results);
    }

    std::vector<table::ProposalNativeSelectionResult> selectForValidation(
        std::vector<table::ProposalNativeSelectionQuery> const& queries) const {
      requireInitialized();
      using QueryView =
          Kokkos::View<table::ProposalNativeSelectionQuery*, memory_space>;
      using ResultView =
          Kokkos::View<table::ProposalNativeSelectionResult*, memory_space>;
      QueryView device_queries("proposal_native_selection_queries", queries.size());
      ResultView device_results("proposal_native_selection_results", queries.size());
      copyToDevice(queries, device_queries);
      Kokkos::parallel_for(
          "c8_kokkos_proposal_native_selection",
          Kokkos::RangePolicy<ExecutionSpace>(execution_, 0, queries.size()),
          NativeSelectionKernel<ExecutionSpace>{
              view_, device_queries, device_results});
      execution_.fence("select PROPOSAL native process");
      return copyToHost(device_results);
    }

  private:
    template <class T>
    using View = Kokkos::View<T*, memory_space>;

    template <class T>
    View<T> upload(std::vector<T> const& source, std::string const& label) {
      View<T> destination(label, source.size());
      auto mirror = Kokkos::create_mirror_view(destination);
      for (std::size_t i = 0; i < source.size(); ++i) mirror(i) = source[i];
      Kokkos::deep_copy(execution_, destination, mirror);
      return destination;
    }

    template <class T, class DeviceView>
    void copyToDevice(std::vector<T> const& source, DeviceView const& destination) const {
      auto mirror = Kokkos::create_mirror_view(destination);
      for (std::size_t i = 0; i < source.size(); ++i) mirror(i) = source[i];
      Kokkos::deep_copy(execution_, destination, mirror);
    }

    template <class DeviceView>
    auto copyToHost(DeviceView const& source) const
        -> std::vector<typename DeviceView::non_const_value_type> {
      auto mirror = Kokkos::create_mirror_view(source);
      Kokkos::deep_copy(execution_, mirror, source);
      execution_.fence("download PROPOSAL native validation output");
      std::vector<typename DeviceView::non_const_value_type> output(source.extent(0));
      for (std::size_t i = 0; i < output.size(); ++i) output[i] = mirror(i);
      return output;
    }

    void requireInitialized() const {
      if (!initialized_) {
        throw std::logic_error("Kokkos PROPOSAL native table is not initialized");
      }
    }

    ExecutionSpace execution_{};
    View<table::NativeDndxColumn> dndx_{};
    View<table::NativeTotalRateColumn> total_rate_{};
    View<std::uint32_t> selection_indices_{};
    View<table::NativeDedxColumn> dedx_{};
    View<std::uint32_t> dedx_accumulation_indices_{};
    View<table::NativeUtilityColumn> utility_{};
    View<double> bicubic_values_{};
    View<double> bicubic_denergy_{};
    View<double> bicubic_dloss_{};
    View<double> bicubic_mixed_{};
    View<double> bicubic_polynomial_{};
    View<double> cubic_values_{};
    View<double> cubic_derivatives_{};
    table::ProposalNativeDeviceView view_{};
    table::Sha256Digest hash_{};
    std::size_t bytes_{};
    bool initialized_{};
  };

} // namespace corsika::accelerator::em::kokkos_detail
