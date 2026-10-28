/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <class TExecutionSpace>
  PhotonTransportBatchResult transport_photons_batch(
      EnvironmentBorrowedView const environment, TransportGeometryConfig const geometry,
      std::vector<InteractionRecord> const& input, TExecutionSpace const& execution) {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using policy_type = Kokkos::RangePolicy<TExecutionSpace>;
    using outcome_type = PhotonTransportOutcome;
    PhotonTransportBatchResult result{};
    result.inputInteractions_ = input.size();
    if (input.empty()) return result;
    if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error("Kokkos photon transport exceeds 32-bit stable offsets");
    }

    Kokkos::View<InteractionRecord*, memory_space_type> interactions(
        "c8_kokkos_photon_interactions", input.size());
    Kokkos::View<outcome_type*, memory_space_type> outcomes(
        "c8_kokkos_photon_transport_outcomes", input.size());
    Kokkos::View<std::uint32_t*, memory_space_type> flags(
        "c8_kokkos_photon_transport_fallback_flags", input.size());
    Kokkos::View<std::uint32_t*, memory_space_type> offsets(
        "c8_kokkos_photon_transport_fallback_offsets", input.size());
    Kokkos::View<PhotonTransportRecord*, memory_space_type> records(
        "c8_kokkos_photon_transport_records", input.size());
    Kokkos::View<FallbackRequest*, memory_space_type> fallbacks(
        "c8_kokkos_photon_transport_fallbacks", input.size());
    auto hostInput = Kokkos::create_mirror_view(interactions);
    for (std::size_t i = 0; i < input.size(); ++i) hostInput(i) = input[i];
    Kokkos::deep_copy(execution, interactions, hostInput);

    Kokkos::parallel_for(
        "c8_kokkos_transport_photons", policy_type(execution, 0, input.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const outcome =
              transport_photon(environment, geometry, interactions(index));
          outcomes(index) = outcome;
          flags(index) = outcome.fallbackFlag_;
        });
    std::uint32_t fallbackCount{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_photon_transport_fallbacks",
        policy_type(execution, 0, input.size()),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index) = update;
          update += flags(index);
        },
        fallbackCount);
    Kokkos::parallel_for(
        "c8_kokkos_compact_photon_transport", policy_type(execution, 0, input.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const fallbackOffset = offsets(index);
          if (flags(index) != 0u) {
            fallbacks(fallbackOffset) = outcomes(index).fallback_;
          } else {
            records(index - fallbackOffset) = outcomes(index).record_;
          }
        });
    execution.fence("complete Kokkos photon transport");

    auto hostRecords = Kokkos::create_mirror_view(records);
    auto hostFallbacks = Kokkos::create_mirror_view(fallbacks);
    Kokkos::deep_copy(execution, hostRecords, records);
    Kokkos::deep_copy(execution, hostFallbacks, fallbacks);
    execution.fence("download Kokkos photon transport");
    result.records_.resize(input.size() - fallbackCount);
    result.fallbackEvents_.resize(fallbackCount);
    for (std::size_t i = 0; i < result.records_.size(); ++i) {
      result.records_[i] = hostRecords(i);
    }
    for (std::size_t i = 0; i < result.fallbackEvents_.size(); ++i) {
      result.fallbackEvents_[i] = hostFallbacks(i);
    }
    return result;
  }
} // namespace corsika::kokkos
