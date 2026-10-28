/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <class TExecutionSpace>
  LeptonTransportBatchResult transport_leptons_batch(
      InteractionPhysicsView const physics, EnvironmentBorrowedView const environment,
      TransportGeometryConfig const geometry, MoliereSnapshot const electronMoliere,
      MoliereSnapshot const muonMoliere,
      MoliereInterpolationView const moliereInterpolation,
      bool const muonMoliereAvailable, std::uint64_t const randomSeed,
      std::uint64_t const showerId, std::vector<InteractionRecord> const& input,
      TExecutionSpace const& execution) {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using policy_type = Kokkos::RangePolicy<TExecutionSpace>;
    LeptonTransportBatchResult result{};
    result.inputInteractions_ = input.size();
    if (input.empty()) return result;
    if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error("Kokkos lepton transport exceeds 32-bit stable offsets");
    }

    auto const count = input.size();
    Kokkos::View<InteractionRecord*, memory_space_type> interactions(
        "c8_kokkos_lepton_transport_input", count);
    Kokkos::View<LeptonTransportRecord*, memory_space_type> rawRecords(
        "c8_kokkos_lepton_transport_raw_records", count);
    Kokkos::View<FallbackRequest*, memory_space_type> rawFallbacks(
        "c8_kokkos_lepton_transport_raw_fallbacks", count);
    Kokkos::View<std::uint32_t*, memory_space_type> flags(
        "c8_kokkos_lepton_transport_flags", count);
    Kokkos::View<std::uint32_t*, memory_space_type> offsets(
        "c8_kokkos_lepton_transport_offsets", count);
    Kokkos::View<LeptonTransportRecord*, memory_space_type> records(
        "c8_kokkos_lepton_transport_records", count);
    Kokkos::View<FallbackRequest*, memory_space_type> fallbacks(
        "c8_kokkos_lepton_transport_fallbacks", count);

    auto hostInput = Kokkos::create_mirror_view(interactions);
    for (std::size_t i = 0; i < count; ++i) hostInput(i) = input[i];
    Kokkos::deep_copy(execution, interactions, hostInput);
    Kokkos::parallel_for(
        "c8_kokkos_transport_leptons", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          flags(index) = transport_lepton(physics.getContinuousView(), true, environment,
                                          geometry, interactions(index),
                                          rawRecords(index), rawFallbacks(index));
        });
    if (electronMoliere.componentCount_ <= 4u) {
      Kokkos::parallel_for(
          "c8_kokkos_apply_moliere_air", policy_type(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            flags(index) = apply_moliere_scattering_stage<4>(
                electronMoliere, muonMoliere, moliereInterpolation, muonMoliereAvailable,
                randomSeed, showerId, rawRecords(index), rawFallbacks(index),
                flags(index));
          });
    } else {
      Kokkos::parallel_for(
          "c8_kokkos_apply_moliere_general", policy_type(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            flags(index) = apply_moliere_scattering_stage<MaxMoliereComponents>(
                electronMoliere, muonMoliere, moliereInterpolation, muonMoliereAvailable,
                randomSeed, showerId, rawRecords(index), rawFallbacks(index),
                flags(index));
          });
    }
    std::uint32_t fallbackCount{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_transport_fallbacks", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index) = update;
          update += flags(index) == 1u ? 1u : 0u;
        },
        fallbackCount);
    Kokkos::parallel_for(
        "c8_kokkos_compact_lepton_transport", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const fallbackOffset = offsets(index);
          if (flags(index) == 1u) {
            fallbacks(fallbackOffset) = rawFallbacks(index);
          } else {
            records(index - fallbackOffset) = rawRecords(index);
          }
        });
    execution.fence("complete Kokkos lepton transport");

    auto hostRecords = Kokkos::create_mirror_view(records);
    auto hostFallbacks = Kokkos::create_mirror_view(fallbacks);
    Kokkos::deep_copy(execution, hostRecords, records);
    Kokkos::deep_copy(execution, hostFallbacks, fallbacks);
    execution.fence("download Kokkos lepton transport");
    result.records_.resize(count - fallbackCount);
    result.fallbackEvents_.resize(fallbackCount);
    for (std::size_t i = 0; i < result.records_.size(); ++i)
      result.records_[i] = hostRecords(i);
    for (std::size_t i = 0; i < result.fallbackEvents_.size(); ++i)
      result.fallbackEvents_[i] = hostFallbacks(i);
    return result;
  }
} // namespace corsika::kokkos
