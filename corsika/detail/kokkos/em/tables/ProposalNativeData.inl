/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TValue, typename TSize>
  CORSIKA_HOST_DEVICE NativeArrayView<TValue, TSize>::NativeArrayView() {}

  template <typename TValue, typename TSize>
  template <typename TContainer>
  NativeArrayView<TValue, TSize>::NativeArrayView(TContainer const& values)
      : data_(values.data())
      , size_(static_cast<size_type>(values.size())) {}

  template <typename TValue, typename TSize>
  CORSIKA_HOST_DEVICE NativeArrayView<TValue, TSize>::NativeArrayView(
      value_type const* data, size_type size)
      : data_(data)
      , size_(size) {}

  template <typename TValue, typename TSize>
  CORSIKA_HOST_DEVICE auto NativeArrayView<TValue, TSize>::getSize() const -> size_type {
    return size_;
  }

  template <typename TValue, typename TSize>
  CORSIKA_HOST_DEVICE bool NativeArrayView<TValue, TSize>::hasStorage() const {
    return data_ != nullptr;
  }

  template <typename TValue, typename TSize>
  CORSIKA_HOST_DEVICE auto NativeArrayView<TValue, TSize>::operator[](
      size_type index) const -> value_type const& {
    return data_[index];
  }

  CORSIKA_HOST_DEVICE inline ProposalNativeDeviceView::ProposalNativeDeviceView() {}

  CORSIKA_HOST_DEVICE inline ProposalNativeDeviceView::ProposalNativeDeviceView(
      NativeArrayView<NativeDndxColumn, std::uint32_t> const& dndxColumns,
      NativeArrayView<NativeTotalRateColumn, std::uint32_t> const& totalRateColumns,
      NativeArrayView<std::uint32_t, std::uint32_t> const& selectionColumnIndices,
      NativeArrayView<NativeDedxColumn, std::uint32_t> const& dedxColumns,
      NativeArrayView<std::uint32_t, std::uint32_t> const& dedxAccumulationIndices,
      NativeArrayView<NativeUtilityColumn, std::uint32_t> const& utilityColumns,
      NativeArrayView<double, std::uint64_t> const& bicubicValues,
      NativeArrayView<double, std::uint64_t> const& bicubicDerivativeEnergy,
      NativeArrayView<double, std::uint64_t> const& bicubicDerivativeLoss,
      NativeArrayView<double, std::uint64_t> const& bicubicMixedDerivative,
      NativeArrayView<double, std::uint64_t> const& bicubicPolynomialCoefficients,
      NativeArrayView<double, std::uint64_t> const& cubicValues,
      NativeArrayView<double, std::uint64_t> const& cubicNodeDerivatives,
      NativePhysicalConstants const& constants)
      : dndxColumns_(dndxColumns.data_)
      , totalRateColumns_(totalRateColumns.data_)
      , selectionColumnIndices_(selectionColumnIndices.data_)
      , dedxColumns_(dedxColumns.data_)
      , dedxAccumulationIndices_(dedxAccumulationIndices.data_)
      , utilityColumns_(utilityColumns.data_)
      , dndxColumnCount_(dndxColumns.size_)
      , totalRateColumnCount_(totalRateColumns.size_)
      , selectionColumnIndexCount_(selectionColumnIndices.size_)
      , dedxColumnCount_(dedxColumns.size_)
      , dedxAccumulationIndexCount_(dedxAccumulationIndices.size_)
      , utilityColumnCount_(utilityColumns.size_)
      , bicubicValues_(bicubicValues.data_)
      , bicubicDerivativeEnergy_(bicubicDerivativeEnergy.data_)
      , bicubicDerivativeLoss_(bicubicDerivativeLoss.data_)
      , bicubicMixedDerivative_(bicubicMixedDerivative.data_)
      , bicubicCoefficientCount_(bicubicValues.size_)
      , bicubicPolynomialCoefficients_(bicubicPolynomialCoefficients.data_)
      , bicubicPolynomialCoefficientCount_(bicubicPolynomialCoefficients.size_)
      , cubicValues_(cubicValues.data_)
      , cubicNodeDerivatives_(cubicNodeDerivatives.data_)
      , cubicCoefficientCount_(cubicValues.size_)
      , constants_(constants) {}

  CORSIKA_HOST_DEVICE inline NativeArrayView<NativeDndxColumn, std::uint32_t>
  ProposalNativeDeviceView::getDndxColumns() const {
    return {dndxColumns_, dndxColumnCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<NativeTotalRateColumn, std::uint32_t>
  ProposalNativeDeviceView::getTotalRateColumns() const {
    return {totalRateColumns_, totalRateColumnCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<std::uint32_t, std::uint32_t>
  ProposalNativeDeviceView::getSelectionColumnIndices() const {
    return {selectionColumnIndices_, selectionColumnIndexCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<NativeDedxColumn, std::uint32_t>
  ProposalNativeDeviceView::getDedxColumns() const {
    return {dedxColumns_, dedxColumnCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<std::uint32_t, std::uint32_t>
  ProposalNativeDeviceView::getDedxAccumulationIndices() const {
    return {dedxAccumulationIndices_, dedxAccumulationIndexCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<NativeUtilityColumn, std::uint32_t>
  ProposalNativeDeviceView::getUtilityColumns() const {
    return {utilityColumns_, utilityColumnCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<double, std::uint64_t>
  ProposalNativeDeviceView::getBicubicValues() const {
    return {bicubicValues_, bicubicCoefficientCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<double, std::uint64_t>
  ProposalNativeDeviceView::getBicubicDerivativeEnergy() const {
    return {bicubicDerivativeEnergy_, bicubicCoefficientCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<double, std::uint64_t>
  ProposalNativeDeviceView::getBicubicDerivativeLoss() const {
    return {bicubicDerivativeLoss_, bicubicCoefficientCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<double, std::uint64_t>
  ProposalNativeDeviceView::getBicubicMixedDerivative() const {
    return {bicubicMixedDerivative_, bicubicCoefficientCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<double, std::uint64_t>
  ProposalNativeDeviceView::getBicubicPolynomialCoefficients() const {
    return {bicubicPolynomialCoefficients_, bicubicPolynomialCoefficientCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<double, std::uint64_t>
  ProposalNativeDeviceView::getCubicValues() const {
    return {cubicValues_, cubicCoefficientCount_};
  }

  CORSIKA_HOST_DEVICE inline NativeArrayView<double, std::uint64_t>
  ProposalNativeDeviceView::getCubicNodeDerivatives() const {
    return {cubicNodeDerivatives_, cubicCoefficientCount_};
  }

  CORSIKA_HOST_DEVICE inline NativePhysicalConstants const&
  ProposalNativeDeviceView::getConstants() const {
    return constants_;
  }

} // namespace corsika::kokkos
