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
#include <vector>

#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  template <class ExecutionSpace>
  gpu::em::PhotonTransportBatchResult transportPhotons(
      gpu::em::EnvironmentSnapshot const environment,
      std::vector<gpu::em::EmInteractionRecord> const& input,
      ExecutionSpace const& execution = {}) {
    using Memory = typename ExecutionSpace::memory_space;
    using Policy = Kokkos::RangePolicy<ExecutionSpace>;
    using Outcome = accelerator::em::detail::PhotonTransportOutcome;
    gpu::em::PhotonTransportBatchResult result{};
    result.input_interactions = input.size();
    if (input.empty()) return result;
    if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "Kokkos photon transport exceeds 32-bit stable offsets");
    }

    Kokkos::View<gpu::em::EmInteractionRecord*, Memory> interactions(
        "c8_kokkos_photon_interactions", input.size());
    Kokkos::View<Outcome*, Memory> outcomes(
        "c8_kokkos_photon_transport_outcomes", input.size());
    Kokkos::View<std::uint32_t*, Memory> flags(
        "c8_kokkos_photon_transport_fallback_flags", input.size());
    Kokkos::View<std::uint32_t*, Memory> offsets(
        "c8_kokkos_photon_transport_fallback_offsets", input.size());
    Kokkos::View<gpu::em::PhotonTransportRecord*, Memory> records(
        "c8_kokkos_photon_transport_records", input.size());
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory> fallbacks(
        "c8_kokkos_photon_transport_fallbacks", input.size());
    auto host_input = Kokkos::create_mirror_view(interactions);
    for (std::size_t i = 0; i < input.size(); ++i) host_input(i) = input[i];
    Kokkos::deep_copy(execution, interactions, host_input);

    Kokkos::parallel_for(
        "c8_kokkos_transport_photons", Policy(execution, 0, input.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const outcome = accelerator::em::detail::transportPhoton(
              environment, interactions(index));
          outcomes(index) = outcome;
          flags(index) = outcome.fallback_flag;
        });
    std::uint32_t fallback_count{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_photon_transport_fallbacks",
        Policy(execution, 0, input.size()),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index) = update;
          update += flags(index);
        },
        fallback_count);
    Kokkos::parallel_for(
        "c8_kokkos_compact_photon_transport",
        Policy(execution, 0, input.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const fallback_offset = offsets(index);
          if (flags(index) != 0u) {
            fallbacks(fallback_offset) = outcomes(index).fallback;
          } else {
            records(index - fallback_offset) = outcomes(index).record;
          }
        });
    execution.fence("complete Kokkos photon transport");

    auto host_records = Kokkos::create_mirror_view(records);
    auto host_fallbacks = Kokkos::create_mirror_view(fallbacks);
    Kokkos::deep_copy(execution, host_records, records);
    Kokkos::deep_copy(execution, host_fallbacks, fallbacks);
    execution.fence("download Kokkos photon transport");
    result.records.resize(input.size() - fallback_count);
    result.fallback_events.resize(fallback_count);
    for (std::size_t i = 0; i < result.records.size(); ++i) {
      result.records[i] = host_records(i);
    }
    for (std::size_t i = 0; i < result.fallback_events.size(); ++i) {
      result.fallback_events[i] = host_fallbacks(i);
    }
    return result;
  }

} // namespace corsika::accelerator::em::kokkos_detail
