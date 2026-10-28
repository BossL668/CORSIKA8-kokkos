/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <stdexcept>

namespace corsika::kokkos {

  template <typename TExecutionSpace>
  std::size_t NativeTableStorage<TExecutionSpace>::projectedDeviceBytes(
      ProposalNativeTableSet const& source) {
    return proposal_native_table_bytes(source);
  }

  template <typename TExecutionSpace>
  void NativeTableStorage<TExecutionSpace>::initialize(
      ProposalNativeTableSet const& source, std::size_t maximumBytes) {
    if (initialized_) {
      throw std::logic_error("Kokkos PROPOSAL native table is already initialized");
    }
    validate_proposal_native_table(source);
    auto const calculatedHash = calculate_proposal_native_hash(source);
    if (calculatedHash != source.contentHash_) {
      throw std::invalid_argument(
          "PROPOSAL native-table content hash does not match its payload");
    }
    auto const requested = projectedDeviceBytes(source);
    if (requested > maximumBytes) {
      throw std::runtime_error(
          "PROPOSAL native table exceeds the configured Kokkos memory budget");
    }

    try {
      dndx_ = upload(source.dndxColumns_, "proposal_native_dndx");
      totalRate_ = upload(source.totalRateColumns_, "proposal_native_total_rate");
      selectionIndices_ =
          upload(source.selectionColumnIndices_, "proposal_native_selection_order");
      dedx_ = upload(source.dedxColumns_, "proposal_native_dedx");
      dedxAccumulationIndices_ =
          upload(source.dedxAccumulationIndices_, "proposal_native_dedx_order");
      utility_ = upload(source.utilityColumns_, "proposal_native_utility");
      bicubicValues_ = upload(source.bicubicValues_, "proposal_native_bicubic_values");
      bicubicDerivativeEnergy_ =
          upload(source.bicubicDerivativeEnergy_, "proposal_native_bicubic_denergy");
      bicubicDerivativeLoss_ =
          upload(source.bicubicDerivativeLoss_, "proposal_native_bicubic_dloss");
      bicubicMixedDerivative_ =
          upload(source.bicubicMixedDerivative_, "proposal_native_bicubic_mixed");
      bicubicPolynomial_ = upload(source.bicubicPolynomialCoefficients_,
                                  "proposal_native_bicubic_polynomial");
      cubicValues_ = upload(source.cubicValues_, "proposal_native_cubic_values");
      cubicDerivatives_ =
          upload(source.cubicNodeDerivatives_, "proposal_native_cubic_derivatives");
      execution_.fence("upload PROPOSAL native table");

      view_ = {NativeArrayView<NativeDndxColumn, std::uint32_t>{dndx_},
               NativeArrayView<NativeTotalRateColumn, std::uint32_t>{totalRate_},
               NativeArrayView<std::uint32_t, std::uint32_t>{selectionIndices_},
               NativeArrayView<NativeDedxColumn, std::uint32_t>{dedx_},
               NativeArrayView<std::uint32_t, std::uint32_t>{dedxAccumulationIndices_},
               NativeArrayView<NativeUtilityColumn, std::uint32_t>{utility_},
               NativeArrayView<double>{bicubicValues_},
               NativeArrayView<double>{bicubicDerivativeEnergy_},
               NativeArrayView<double>{bicubicDerivativeLoss_},
               NativeArrayView<double>{bicubicMixedDerivative_},
               NativeArrayView<double>{bicubicPolynomial_},
               NativeArrayView<double>{cubicValues_},
               NativeArrayView<double>{cubicDerivatives_},
               source.constants_};
      hash_ = source.contentHash_;
      bytes_ = requested;
      initialized_ = true;
    } catch (...) {
      reset();
      throw;
    }
  }

  template <typename TExecutionSpace>
  void NativeTableStorage<TExecutionSpace>::reset() noexcept {
    dndx_ = {};
    totalRate_ = {};
    selectionIndices_ = {};
    dedx_ = {};
    dedxAccumulationIndices_ = {};
    utility_ = {};
    bicubicValues_ = {};
    bicubicDerivativeEnergy_ = {};
    bicubicDerivativeLoss_ = {};
    bicubicMixedDerivative_ = {};
    bicubicPolynomial_ = {};
    cubicValues_ = {};
    cubicDerivatives_ = {};
    view_ = {};
    hash_ = {};
    bytes_ = 0;
    initialized_ = false;
  }

  template <typename TExecutionSpace>
  bool NativeTableStorage<TExecutionSpace>::isInitialized() const noexcept {
    return initialized_;
  }

  template <typename TExecutionSpace>
  std::size_t NativeTableStorage<TExecutionSpace>::getDeviceBytes() const noexcept {
    return bytes_;
  }

  template <typename TExecutionSpace>
  sha256_digest_type const& NativeTableStorage<TExecutionSpace>::getSourceContentHash()
      const {
    requireInitialized();
    return hash_;
  }

  template <typename TExecutionSpace>
  ProposalNativeDeviceView NativeTableStorage<TExecutionSpace>::getDeviceView() const {
    requireInitialized();
    return view_;
  }

  template <typename TExecutionSpace>
  template <typename TValue>
  auto NativeTableStorage<TExecutionSpace>::upload(std::vector<TValue> const& source,
                                                   std::string const& label)
      -> array_view_type<TValue> {
    array_view_type<TValue> destination(label, source.size());
    auto mirror = Kokkos::create_mirror_view(destination);
    for (std::size_t index = 0; index < source.size(); ++index)
      mirror(index) = source[index];
    Kokkos::deep_copy(execution_, destination, mirror);
    return destination;
  }

  template <typename TExecutionSpace>
  void NativeTableStorage<TExecutionSpace>::requireInitialized() const {
    if (!initialized_) {
      throw std::logic_error("Kokkos PROPOSAL native table is not initialized");
    }
  }

} // namespace corsika::kokkos
