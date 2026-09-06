/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "KokkosProposalNativeTableTestDriver.hpp"

#include <Kokkos_Core.hpp>

#include <corsika/accelerator/em/kokkos/KokkosMoliereInterpolation.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhysicsContext.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileProjection.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentMemoryBudget.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>

namespace corsika::accelerator::em::testing {

  KokkosProposalNativeQueryOutput runKokkosProposalNativeQueries(
      gpu::em::tables::ProposalNativeTableSet const& source,
      std::vector<gpu::em::tables::ProposalNativeQuery> const& queries) {
    kokkos_detail::KokkosProposalNativeTable<Kokkos::DefaultExecutionSpace>
        table;
    table.initialize(source);
    return {table.queryForValidation(queries), table.sourceContentHash(),
            table.deviceBytes()};
  }

  KokkosStaticDeviceMemoryProjectionOutput
  projectKokkosStaticDeviceMemoryForValidation(
      gpu::em::tables::ProposalNativeTableSet const& source,
      gpu::em::GpuEmConfig const& config) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    KokkosStaticDeviceMemoryProjectionOutput result{};
    result.proposal_native_table =
        kokkos_detail::KokkosProposalNativeTable<
            ExecutionSpace>::projectedDeviceBytes(source);
    result.moliere_interpolation =
        kokkos_detail::KokkosMoliereInterpolation<
            ExecutionSpace>::projectedDeviceBytes();
    result.profile_projection =
        kokkos_detail::KokkosProfileProjection<
            ExecutionSpace>::projectedDeviceBytes(config.profile_projection);
    result.profile_accumulator =
        kokkos_detail::KokkosProfileAccumulator<
            ExecutionSpace>::projectedDeviceBytes(config.profile_projection);
    result.radio_accumulator = radio::kokkos_detail::KokkosRadioAccumulator<
        ExecutionSpace>::projectedDeviceBytes(config.radio);
    result.physics_context = kokkos_detail::KokkosPhysicsContext<
        ExecutionSpace>::projectedDeviceBytes();
    result.total_bytes = kokkos_detail::checkedMemoryAdd(
        result.proposal_native_table, result.moliere_interpolation);
    result.total_bytes = kokkos_detail::checkedMemoryAdd(
        result.total_bytes, result.profile_projection);
    result.total_bytes = kokkos_detail::checkedMemoryAdd(
        result.total_bytes, result.profile_accumulator);
    result.total_bytes = kokkos_detail::checkedMemoryAdd(
        result.total_bytes, result.radio_accumulator);
    result.total_bytes = kokkos_detail::checkedMemoryAdd(
        result.total_bytes, result.physics_context);
    return result;
  }

} // namespace corsika::accelerator::em::testing
