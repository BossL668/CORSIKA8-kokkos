/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/detail/ProfileProjectionStep.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>

namespace corsika::accelerator::em::detail {

  inline constexpr double ProfileSignedIntegerLimit = 0x1p63;
  inline constexpr long long ProfileSignedIntegerMaximum =
      9223372036854775807LL;
  inline constexpr long long ProfileSignedIntegerMinimum =
      (-9223372036854775807LL - 1LL);

  C8_ACCELERATOR_INLINE_FUNCTION inline bool profileFinite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double profileAbs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fabs(value);
#else
    return std::fabs(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double profileFloor(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::floor(value);
#else
    return std::floor(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double profileCeil(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::ceil(value);
#else
    return std::ceil(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline long long profileRound(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return static_cast<long long>(::rint(value));
#else
    return static_cast<long long>(std::rint(value));
#endif
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline bool checkedProfileAtomicAdd(
      long long* address, long long increment) {
    auto const previous = AtomicOperations::add(address, increment);
    return !((increment > 0 &&
              previous > ProfileSignedIntegerMaximum - increment) ||
             (increment < 0 &&
              previous < ProfileSignedIntegerMinimum - increment));
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void addProfileFixedPoint(
      long long* address, double contribution, double scale,
      gpu::em::detail::DeviceProfileCounters* counters) {
    auto const scaled = contribution * scale;
    if (!profileFinite(scaled) ||
        profileAbs(scaled) >= ProfileSignedIntegerLimit ||
        !checkedProfileAtomicAdd<AtomicOperations>(address,
                                                   profileRound(scaled)))
      AtomicOperations::add(&counters->fixed_point_overflows, 1ULL);
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateParticleProfile(
      gpu::em::detail::DeviceProfileAccumulator const& accumulator,
      std::int32_t pid, double start_grammage, double end_grammage,
      double weight) {
    using namespace gpu::em;
    if (start_grammage == end_grammage || accumulator.bins == 0) return;
    auto const first_value =
        profileCeil(start_grammage / accumulator.bin_width_g_per_cm2);
    auto const last_value =
        profileFloor(end_grammage / accumulator.bin_width_g_per_cm2);
    if (!(first_value >= 0.) || !(last_value >= first_value) ||
        first_value >= static_cast<double>(accumulator.bins))
      return;
    auto const first = static_cast<std::size_t>(first_value);
    auto const candidate_last = static_cast<std::size_t>(last_value);
    auto const last = candidate_last < accumulator.bins
                          ? candidate_last
                          : accumulator.bins - 1;
    long long* profile = nullptr;
    if (pid == static_cast<std::int32_t>(EmPid::Photon))
      profile = accumulator.photons;
    else if (pid == static_cast<std::int32_t>(EmPid::Electron))
      profile = accumulator.electrons;
    else if (pid == static_cast<std::int32_t>(EmPid::Positron))
      profile = accumulator.positrons;
    else if (pid == static_cast<std::int32_t>(EmPid::MuonMinus))
      profile = accumulator.muons_minus;
    else if (pid == static_cast<std::int32_t>(EmPid::MuonPlus))
      profile = accumulator.muons_plus;
    if (profile == nullptr) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    for (auto bin = first; bin <= last; ++bin)
      addProfileFixedPoint<AtomicOperations>(
          profile + bin, weight, accumulator.weight_scale,
          accumulator.counters);
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateMuonParentProduction(
      gpu::em::detail::DeviceProfileAccumulator const& accumulator,
      double grammage, double parent_weight) {
    if (accumulator.bins == 0 ||
        accumulator.muon_parent_productions == nullptr)
      return;
    auto const bin_value =
        profileFloor(grammage / accumulator.bin_width_g_per_cm2);
    if (!profileFinite(bin_value)) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    if (bin_value < 0.) return;
    auto const candidate = static_cast<std::size_t>(bin_value);
    auto const bin =
        candidate < accumulator.bins ? candidate : accumulator.bins - 1;
    addProfileFixedPoint<AtomicOperations>(
        accumulator.muon_parent_productions + bin, parent_weight,
        accumulator.weight_scale, accumulator.counters);
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateEnergyProfile(
      gpu::em::detail::DeviceProfileAccumulator const& accumulator,
      long long* energy_loss, double start_grammage, double end_grammage,
      double weighted_deposit_GeV) {
    if (!(weighted_deposit_GeV > 0.) || accumulator.bins == 0) return;
    AtomicOperations::add(&accumulator.counters->deposited_steps, 1ULL);
    if (start_grammage > end_grammage) {
      auto const temporary = start_grammage;
      start_grammage = end_grammage;
      end_grammage = temporary;
    }
    auto const delta = end_grammage - start_grammage;
    auto const maximum_bin = static_cast<int>(accumulator.bins - 1);
    auto first = static_cast<int>(
        start_grammage / accumulator.bin_width_g_per_cm2);
    first = first < 0 ? 0 : first;
    first = first > maximum_bin ? maximum_bin : first;
    if (delta < accumulator.energy_loss_threshold_g_per_cm2) {
      addProfileFixedPoint<AtomicOperations>(
          energy_loss + first, weighted_deposit_GeV,
          accumulator.energy_scale, accumulator.counters);
      return;
    }
    auto last = static_cast<int>(
        end_grammage / accumulator.bin_width_g_per_cm2);
    last = last < 0 ? 0 : last;
    last = last > maximum_bin ? maximum_bin : last;
    auto const density = weighted_deposit_GeV / delta;
    if (first == last) {
      addProfileFixedPoint<AtomicOperations>(
          energy_loss + first, density * delta, accumulator.energy_scale,
          accumulator.counters);
      return;
    }
    addProfileFixedPoint<AtomicOperations>(
        energy_loss + first,
        density * (1. + static_cast<double>(first)) *
                accumulator.bin_width_g_per_cm2 -
            density * start_grammage,
        accumulator.energy_scale, accumulator.counters);
    addProfileFixedPoint<AtomicOperations>(
        energy_loss + last,
        density * end_grammage - density * static_cast<double>(last) *
                                     accumulator.bin_width_g_per_cm2,
        accumulator.energy_scale, accumulator.counters);
    for (auto bin = first + 1; bin < last; ++bin)
      addProfileFixedPoint<AtomicOperations>(
          energy_loss + bin,
          density * accumulator.bin_width_g_per_cm2,
          accumulator.energy_scale, accumulator.counters);
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateTerminalEnergy(
      gpu::em::detail::DeviceProfileAccumulator const& accumulator,
      double weighted_total_energy_GeV, bool observed) {
    if (!(weighted_total_energy_GeV >= 0.) ||
        !profileFinite(weighted_total_energy_GeV)) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    addProfileFixedPoint<AtomicOperations>(
        observed ? &accumulator.counters->weighted_observed_total_energy
                 : &accumulator.counters->weighted_escaped_total_energy,
        weighted_total_energy_GeV, accumulator.energy_scale,
        accumulator.counters);
  }

  template <class AtomicOperations, class FinalStateRecord>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateProfileThinning(
      FinalStateRecord const& record,
      gpu::em::detail::DeviceProfileCounters* counters) {
    using namespace gpu::em;
    auto const status = static_cast<EmThinningStatus>(record.thinning_status);
    if (status == EmThinningStatus::Hillas)
      AtomicOperations::add(&counters->thinning_hillas_vertices, 1ULL);
    else if (status == EmThinningStatus::Statistical)
      AtomicOperations::add(&counters->thinning_statistical_vertices, 1ULL);
    else if (status != EmThinningStatus::NotApplied)
      AtomicOperations::add(&counters->invalid_records, 1ULL);
    auto const original_multiplicity =
        record.process_id == PhotoelectricProcessId
            ? 1U
            : record.process_id == ElectronPairProcessId ? 3U : 2U;
    if (record.secondary_count > original_multiplicity)
      AtomicOperations::add(&counters->invalid_records, 1ULL);
    else
      AtomicOperations::add(
          &counters->thinning_particles_discarded,
          static_cast<unsigned long long>(original_multiplicity -
                                          record.secondary_count));
  }

  template <class TransportRecord>
  C8_ACCELERATOR_INLINE_FUNCTION inline TransportRecord const*
  findProfileTransportRecord(TransportRecord const* records,
                             std::size_t count,
                             std::uint64_t input_index) {
    std::size_t first = 0;
    std::size_t last = count;
    while (first < last) {
      auto const middle = first + (last - first) / 2;
      if (records[middle].input_index < input_index)
        first = middle + 1;
      else
        last = middle;
    }
    if (first >= count || records[first].input_index != input_index)
      return nullptr;
    return records + first;
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulatePhotonProfileStep(
      gpu::em::detail::DeviceProfileProjection const& projection,
      gpu::em::detail::DeviceProfileAccumulator const& accumulator,
      gpu::em::PhotonTransportRecord const& record) {
    using namespace gpu::em;
    auto const start = projectProfileGrammage(projection, record.start.position_m);
    auto const end = projectProfileGrammage(projection, record.end.position_m);
    accumulateParticleProfile<AtomicOperations>(
        accumulator, record.start.pid, start, end, record.start.weight);
    accumulateEnergyProfile<AtomicOperations>(
        accumulator, accumulator.energy_loss, start, end,
        record.cut_deposited_energy_GeV * record.start.weight);
    AtomicOperations::add(&accumulator.counters->steps, 1ULL);
    if (record.limit == PhotonTransportLimit::ParticleCut) {
      AtomicOperations::add(&accumulator.counters->photon_cuts, 1ULL);
      if (record.observation_surface_reached_before_cut != 0U)
        accumulateTerminalEnergy<AtomicOperations>(
            accumulator, record.end.energy_GeV * record.start.weight, true);
    } else if (record.limit == PhotonTransportLimit::ObservationSurface) {
      accumulateTerminalEnergy<AtomicOperations>(
          accumulator, record.end.energy_GeV * record.start.weight, true);
    } else if (record.limit == PhotonTransportLimit::EscapedEnvironment) {
      accumulateTerminalEnergy<AtomicOperations>(
          accumulator, record.end.energy_GeV * record.start.weight, false);
    }
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulatePhotonProfileFinalState(
      gpu::em::detail::DeviceProfileProjection const& projection,
      gpu::em::detail::DeviceProfileAccumulator const& accumulator,
      gpu::em::PhotonTransportRecord const* transport_records,
      std::size_t transport_count,
      gpu::em::PhotonPairFinalStateRecord const& record) {
    using namespace gpu::em;
    accumulateProfileThinning<AtomicOperations>(record, accumulator.counters);
    auto const atomic_electron_process =
        record.process_id == ComptonProcessId ||
        record.process_id == PhotoelectricProcessId;
    if (!atomic_electron_process) return;
    auto const* transport = findProfileTransportRecord(
        transport_records, transport_count, record.input_index);
    if (transport == nullptr ||
        transport->start.history_id != record.parent_history_id ||
        transport->interaction.process_id != record.process_id) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    addProfileFixedPoint<AtomicOperations>(
        &accumulator.counters->weighted_medium_rest_mass_input,
        ElectronMassGeV * transport->start.weight, accumulator.energy_scale,
        accumulator.counters);
    if (record.process_id != PhotoelectricProcessId) return;
    if (!profileFinite(record.energy_split_fraction) ||
        record.energy_split_fraction < 0. ||
        record.energy_split_fraction > 1.) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    auto const start =
        projectProfileGrammage(projection, transport->start.position_m);
    auto const end =
        projectProfileGrammage(projection, transport->end.position_m);
    auto const deposit = transport->end.energy_GeV *
                         (1. - record.energy_split_fraction) *
                         transport->start.weight;
    accumulateEnergyProfile<AtomicOperations>(
        accumulator, accumulator.energy_loss, start, end, deposit);
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateLeptonProfileStep(
      gpu::em::detail::DeviceProfileProjection const& projection,
      gpu::em::detail::DeviceProfileAccumulator const& accumulator,
      gpu::em::LeptonTransportRecord const& record) {
    using namespace gpu::em;
    auto const start = projectProfileGrammage(projection, record.start.position_m);
    auto const end = projectProfileGrammage(projection, record.end.position_m);
    accumulateParticleProfile<AtomicOperations>(
        accumulator, record.start.pid, start, end, record.start.weight);
    accumulateEnergyProfile<AtomicOperations>(
        accumulator,
        isMuonPid(record.start.pid) ? accumulator.muon_energy_loss
                                    : accumulator.energy_loss,
        start, end,
        (record.continuous_deposited_energy_GeV +
         record.cut_deposited_energy_GeV) *
            record.start.weight);
    AtomicOperations::add(&accumulator.counters->steps, 1ULL);
    auto const limit = static_cast<std::int32_t>(record.limit);
    if (limit < 0 || limit >= 8)
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
    else
      AtomicOperations::add(&accumulator.counters->lepton_limits[limit], 1ULL);
    if (record.limit == LeptonTransportLimit::ParticleCut) {
      addProfileFixedPoint<AtomicOperations>(
          &accumulator.counters->weighted_cut_rest_mass_energy,
          (isMuonPid(record.start.pid) ? MuonMassGeV : ElectronMassGeV) *
              record.start.weight,
          accumulator.energy_scale, accumulator.counters);
      if (record.observation_surface_reached_before_cut != 0U)
        accumulateTerminalEnergy<AtomicOperations>(
            accumulator, record.end.energy_GeV * record.start.weight, true);
    } else if (record.limit == LeptonTransportLimit::ObservationSurface) {
      accumulateTerminalEnergy<AtomicOperations>(
          accumulator, record.end.energy_GeV * record.start.weight, true);
    } else if (record.limit == LeptonTransportLimit::EscapedEnvironment) {
      accumulateTerminalEnergy<AtomicOperations>(
          accumulator, record.end.energy_GeV * record.start.weight, false);
    }
    AtomicOperations::add(&accumulator.counters->moliere_trials, 1ULL);
    AtomicOperations::add(&accumulator.counters->moliere_newton_iterations,
                          static_cast<unsigned long long>(
                              record.multiple_scattering_iterations));
    AtomicOperations::maximum(
        &accumulator.counters->moliere_max_newton_iterations,
        static_cast<unsigned long long>(record.multiple_scattering_iterations));
    if (record.multiple_scattering_applied != 0)
      AtomicOperations::add(&accumulator.counters->moliere_deflections, 1ULL);
    else if (record.multiple_scattering_status ==
             static_cast<std::uint32_t>(MoliereStatus::NoDeflection))
      AtomicOperations::add(&accumulator.counters->moliere_zero_deflections,
                            1ULL);
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateLeptonProfileFinalState(
      gpu::em::detail::DeviceProfileProjection const& projection,
      gpu::em::detail::DeviceProfileAccumulator const& accumulator,
      gpu::em::LeptonTransportRecord const* transport_records,
      std::size_t transport_count,
      gpu::em::BremsFinalStateRecord const& record) {
    using namespace gpu::em;
    accumulateProfileThinning<AtomicOperations>(record, accumulator.counters);
    auto const atomic_electron_process =
        record.process_id == IonizationProcessId ||
        record.process_id == AnnihilationProcessId;
    if (!atomic_electron_process) return;
    auto const* transport = findProfileTransportRecord(
        transport_records, transport_count, record.input_index);
    if (transport == nullptr ||
        transport->start.history_id != record.parent_history_id) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    if (record.process_id == IonizationProcessId &&
        isMuonPid(transport->start.pid)) {
      auto const vertex_grammage =
          projectProfileGrammage(projection, transport->end.position_m);
      accumulateMuonParentProduction<AtomicOperations>(
          accumulator, vertex_grammage, transport->start.weight);
    }
    addProfileFixedPoint<AtomicOperations>(
        &accumulator.counters->weighted_medium_rest_mass_input,
        ElectronMassGeV * transport->start.weight, accumulator.energy_scale,
        accumulator.counters);
  }

} // namespace corsika::accelerator::em::detail
