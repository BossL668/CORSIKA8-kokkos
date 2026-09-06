/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "KokkosProposalNativeOracleDevice.hpp"

#include <Kokkos_Core.hpp>

#include <stdexcept>

#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>

namespace corsika::accelerator::em::testing {

  class KokkosProposalNativeOracleDevice::Impl {
  public:
    kokkos_detail::KokkosProposalNativeTable<Kokkos::DefaultExecutionSpace>
        table;
  };

  KokkosProposalNativeOracleDevice::KokkosProposalNativeOracleDevice()
      : impl_(std::make_unique<Impl>()) {}
  KokkosProposalNativeOracleDevice::~KokkosProposalNativeOracleDevice() =
      default;
  KokkosProposalNativeOracleDevice::KokkosProposalNativeOracleDevice(
      KokkosProposalNativeOracleDevice&&) noexcept = default;
  KokkosProposalNativeOracleDevice&
  KokkosProposalNativeOracleDevice::operator=(
      KokkosProposalNativeOracleDevice&&) noexcept = default;

  void KokkosProposalNativeOracleDevice::initialize(
      gpu::em::tables::ProposalNativeTableSet const& source, int const device,
      std::size_t const maximum_bytes) {
    if (device != 0) {
      throw std::invalid_argument(
          "the Kokkos oracle device is selected when Kokkos initializes; "
          "the validation facade accepts device index zero only");
    }
    impl_->table.initialize(source, maximum_bytes);
  }

  std::size_t KokkosProposalNativeOracleDevice::deviceBytes() const noexcept {
    return impl_->table.deviceBytes();
  }

  std::vector<gpu::em::tables::NativeQueryResult>
  KokkosProposalNativeOracleDevice::queryForValidation(
      std::vector<gpu::em::tables::ProposalNativeQuery> const& queries) const {
    return impl_->table.queryForValidation(queries);
  }

  std::vector<gpu::em::tables::ProposalNativeSelectionResult>
  KokkosProposalNativeOracleDevice::selectForValidation(
      std::vector<gpu::em::tables::ProposalNativeSelectionQuery> const&
          queries) const {
    return impl_->table.selectForValidation(queries);
  }

} // namespace corsika::accelerator::em::testing
