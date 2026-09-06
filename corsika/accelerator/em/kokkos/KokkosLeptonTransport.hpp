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

#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/accelerator/em/detail/MoliereStep.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  /**
   * Complete charged-lepton propagation up to, but not including, the split
   * Moliere stage.  The physics function is shared verbatim with native CUDA;
   * Kokkos owns only queue storage and stable compaction.
   */
  template <class ExecutionSpace>
  gpu::em::LeptonTransportBatchResult transportLeptons(
      gpu::em::tables::NativePhysicsView const physics,
      gpu::em::EnvironmentSnapshot const environment,
      gpu::em::MoliereSnapshot const electron_moliere,
      gpu::em::MoliereSnapshot const muon_moliere,
      gpu::em::MoliereInterpolationView const moliere_interpolation,
      bool const muon_moliere_available,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      std::vector<gpu::em::EmInteractionRecord> const& input,
      ExecutionSpace const& execution = {}) {
    using Memory = typename ExecutionSpace::memory_space;
    using Policy = Kokkos::RangePolicy<ExecutionSpace>;
    gpu::em::LeptonTransportBatchResult result{};
    result.input_interactions = input.size();
    if (input.empty()) return result;
    if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "Kokkos lepton transport exceeds 32-bit stable offsets");
    }

    auto const count = input.size();
    Kokkos::View<gpu::em::EmInteractionRecord*, Memory> interactions(
        "c8_kokkos_lepton_transport_input", count);
    Kokkos::View<gpu::em::LeptonTransportRecord*, Memory> raw_records(
        "c8_kokkos_lepton_transport_raw_records", count);
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory> raw_fallbacks(
        "c8_kokkos_lepton_transport_raw_fallbacks", count);
    Kokkos::View<std::uint32_t*, Memory> flags(
        "c8_kokkos_lepton_transport_flags", count);
    Kokkos::View<std::uint32_t*, Memory> offsets(
        "c8_kokkos_lepton_transport_offsets", count);
    Kokkos::View<gpu::em::LeptonTransportRecord*, Memory> records(
        "c8_kokkos_lepton_transport_records", count);
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory> fallbacks(
        "c8_kokkos_lepton_transport_fallbacks", count);

    auto host_input = Kokkos::create_mirror_view(interactions);
    for (std::size_t i = 0; i < count; ++i) host_input(i) = input[i];
    Kokkos::deep_copy(execution, interactions, host_input);
    Kokkos::parallel_for(
        "c8_kokkos_transport_leptons",
        Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          flags(index) = detail::transportLepton(
              physics, true, environment, interactions(index),
              raw_records(index), raw_fallbacks(index));
        });
    if (electron_moliere.component_count <= 4u) {
      Kokkos::parallel_for(
          "c8_kokkos_apply_moliere_air", Policy(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            flags(index) = detail::applyMoliereScatteringStage<4>(
                electron_moliere, muon_moliere, moliere_interpolation,
                muon_moliere_available, random_seed, shower_id,
                raw_records(index), raw_fallbacks(index), flags(index));
          });
    } else {
      Kokkos::parallel_for(
          "c8_kokkos_apply_moliere_general", Policy(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            flags(index) = detail::applyMoliereScatteringStage<
                gpu::em::MaxMoliereComponents>(
                electron_moliere, muon_moliere, moliere_interpolation,
                muon_moliere_available, random_seed, shower_id,
                raw_records(index), raw_fallbacks(index), flags(index));
          });
    }
    std::uint32_t fallback_count{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_transport_fallbacks",
        Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index) = update;
          update += flags(index) == 1u ? 1u : 0u;
        },
        fallback_count);
    Kokkos::parallel_for(
        "c8_kokkos_compact_lepton_transport", Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const fallback_offset = offsets(index);
          if (flags(index) == 1u) {
            fallbacks(fallback_offset) = raw_fallbacks(index);
          } else {
            records(index - fallback_offset) = raw_records(index);
          }
        });
    execution.fence("complete Kokkos lepton transport");

    auto host_records = Kokkos::create_mirror_view(records);
    auto host_fallbacks = Kokkos::create_mirror_view(fallbacks);
    Kokkos::deep_copy(execution, host_records, records);
    Kokkos::deep_copy(execution, host_fallbacks, fallbacks);
    execution.fence("download Kokkos lepton transport");
    result.records.resize(count - fallback_count);
    result.fallback_events.resize(fallback_count);
    for (std::size_t i = 0; i < result.records.size(); ++i)
      result.records[i] = host_records(i);
    for (std::size_t i = 0; i < result.fallback_events.size(); ++i)
      result.fallback_events[i] = host_fallbacks(i);
    return result;
  }

} // namespace corsika::accelerator::em::kokkos_detail
