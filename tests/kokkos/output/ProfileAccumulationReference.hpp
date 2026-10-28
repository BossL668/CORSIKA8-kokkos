/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Frozen production 41cdc543; no receiver accumulation function is called.
#include <tests/kokkos/output/ProfileProjectionReference.hpp>
#include <tests/kokkos/em/finalstate/FinalStateReference.hpp>
#include <Kokkos_Core.hpp>
namespace profile_accumulation_reference {
  using namespace profile_projection_reference;
  using continuous_reference::TransportElectronMassGeV;
  using continuous_reference::TransportMuonMassGeV;
  using selection_reference::isNativeMuonPairCutDeposit;
  using stochastic_reference::EmPid;
  using stochastic_reference::isMuonPid;
  using PhotonPairFinalStateRecord = final_state_reference::PhotonFinalStateRecord;
  using BremsFinalStateRecord = final_state_reference::BremsFinalStateRecord;
  using final_state_reference::EmThinningStatus;
  enum class MoliereStatus : std::uint32_t {
    Success = 0,
    NoDeflection = 1,
    InvalidSnapshot = 2,
    InvalidInput = 3,
    NonConvergent = 4,
    NonFiniteResult = 5,
  };
  inline constexpr double ElectronMassGeV = 0.0005109989461;
  inline constexpr std::uint64_t AtRestAnnihilationPolarDrawId = 3;
  struct DeviceProfileCounters {
    unsigned long long steps{};
    unsigned long long deposited_steps{};
    unsigned long long photon_cuts{};
    unsigned long long lepton_limits[8]{};
    unsigned long long moliere_trials{};
    unsigned long long moliere_deflections{};
    unsigned long long moliere_zero_deflections{};
    unsigned long long moliere_newton_iterations{};
    unsigned long long moliere_max_newton_iterations{};
    unsigned long long thinning_hillas_vertices{};
    unsigned long long thinning_statistical_vertices{};
    unsigned long long thinning_particles_discarded{};
    unsigned long long fixed_point_overflows{};
    unsigned long long invalid_records{};
    long long weighted_medium_rest_mass_input{};
    long long weighted_cut_rest_mass_energy{};
    long long weighted_observed_total_energy{};
    long long weighted_escaped_total_energy{};
    long long weighted_unwritten_photoelectric_binding_energy{};
    long long weighted_observation_cut_overlap_energy{};
    long long weighted_mass_convention_correction{};
    long long weighted_native_muon_pair_cut_energy{};
  };
  struct DeviceProfileAccumulator {
    long long* photons{};
    long long* electrons{};
    long long* positrons{};
    long long* muons_minus{};
    long long* muons_plus{};
    long long* muon_parent_productions{};
    long long* energy_loss{};
    long long* muon_energy_loss{};
    DeviceProfileCounters* counters{};
    std::size_t bins{};
    double bin_width_g_per_cm2{};
    ProfileCrossingMode crossing_mode{ProfileCrossingMode::Both};
    double energy_loss_threshold_g_per_cm2{};
    double weight_scale{};
    double inverse_weight_scale{};
    double energy_scale{};
    double inverse_energy_scale{};
  };
  inline constexpr double ProfileSignedIntegerLimit = 0x1p63;
  inline constexpr long long ProfileSignedIntegerMaximum = 9223372036854775807LL;
  inline constexpr long long ProfileSignedIntegerMinimum = (-9223372036854775807LL - 1LL);
  struct KokkosProfileAtomicOperations {
    KOKKOS_INLINE_FUNCTION static long long add(long long* address, long long value) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
      // CUDA has a native unsigned 64-bit atomic add, while the signed
      // overload selected through Kokkos/desul is implemented as a CAS retry
      // loop.  Two's-complement signed addition has the same bit result as
      // unsigned addition, and the returned pre-add bit pattern preserves the
      // overflow check in checkedProfileAtomicAdd().  This deliberately
      // mirrors the established native-CUDA profile projection path.
      auto const previous = ::atomicAdd(reinterpret_cast<unsigned long long*>(address),
                                        static_cast<unsigned long long>(value));
      return static_cast<long long>(previous);
#else
      return Kokkos::atomic_fetch_add(address, value);
#endif
    }

    KOKKOS_INLINE_FUNCTION static unsigned long long add(unsigned long long* address,
                                                         unsigned long long value) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
      return ::atomicAdd(address, value);
#else
      return Kokkos::atomic_fetch_add(address, value);
#endif
    }

    KOKKOS_INLINE_FUNCTION static void maximum(unsigned long long* address,
                                               unsigned long long value) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
      ::atomicMax(address, value);
#else
      Kokkos::atomic_max(address, value);
#endif
    }
  };
  CORSIKA_HOST_DEVICE inline bool profileFinite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }
  CORSIKA_HOST_DEVICE inline double profileAbs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fabs(value);
#else
    return std::fabs(value);
#endif
  }
  CORSIKA_HOST_DEVICE inline double profileFloor(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::floor(value);
#else
    return std::floor(value);
#endif
  }
  CORSIKA_HOST_DEVICE inline double profileCeil(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::ceil(value);
#else
    return std::ceil(value);
#endif
  }
  CORSIKA_HOST_DEVICE inline long long profileRound(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return static_cast<long long>(::rint(value));
#else
    return static_cast<long long>(std::rint(value));
#endif
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline bool checkedProfileAtomicAdd(long long* address,
                                                          long long increment) {
    auto const previous = AtomicOperations::add(address, increment);
    return !((increment > 0 && previous > ProfileSignedIntegerMaximum - increment) ||
             (increment < 0 && previous < ProfileSignedIntegerMinimum - increment));
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void addProfileFixedPoint(long long* address,
                                                       double contribution, double scale,
                                                       DeviceProfileCounters* counters) {
    auto const scaled = contribution * scale;
    if (!profileFinite(scaled) || profileAbs(scaled) >= ProfileSignedIntegerLimit ||
        !checkedProfileAtomicAdd<AtomicOperations>(address, profileRound(scaled)))
      AtomicOperations::add(&counters->fixed_point_overflows, 1ULL);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulateParticleProfile(
      DeviceProfileAccumulator const& accumulator, std::int32_t pid,
      double start_grammage, double end_grammage, double weight) {
    using namespace stochastic_reference;
    auto const crossings = profile_projection_reference::longitudinalProfileBins(
        start_grammage / accumulator.bin_width_g_per_cm2,
        end_grammage / accumulator.bin_width_g_per_cm2, accumulator.bins,
        accumulator.crossing_mode);
    if (crossings.begin == crossings.end) return;
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
    for (auto bin = crossings.begin; bin < crossings.end; ++bin)
      addProfileFixedPoint<AtomicOperations>(
          profile + bin, weight, accumulator.weight_scale, accumulator.counters);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulateMuonParentProduction(
      DeviceProfileAccumulator const& accumulator, double grammage,
      double parent_weight) {
    if (accumulator.bins == 0 || accumulator.muon_parent_productions == nullptr) return;
    auto const bin_value = profileFloor(grammage / accumulator.bin_width_g_per_cm2);
    if (!profileFinite(bin_value)) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    if (bin_value < 0.) return;
    auto const candidate = static_cast<std::size_t>(bin_value);
    auto const bin = candidate < accumulator.bins ? candidate : accumulator.bins - 1;
    addProfileFixedPoint<AtomicOperations>(accumulator.muon_parent_productions + bin,
                                           parent_weight, accumulator.weight_scale,
                                           accumulator.counters);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulateEnergyProfile(
      DeviceProfileAccumulator const& accumulator, long long* energy_loss,
      double start_grammage, double end_grammage, double weighted_deposit_GeV) {
    if (!(weighted_deposit_GeV > 0.) || accumulator.bins == 0) return;
    AtomicOperations::add(&accumulator.counters->deposited_steps, 1ULL);
    if (start_grammage > end_grammage) {
      auto const temporary = start_grammage;
      start_grammage = end_grammage;
      end_grammage = temporary;
    }
    auto const delta = end_grammage - start_grammage;
    auto const maximum_bin = static_cast<int>(accumulator.bins - 1);
    auto first = static_cast<int>(start_grammage / accumulator.bin_width_g_per_cm2);
    first = first < 0 ? 0 : first;
    first = first > maximum_bin ? maximum_bin : first;
    if (delta < accumulator.energy_loss_threshold_g_per_cm2) {
      addProfileFixedPoint<AtomicOperations>(energy_loss + first, weighted_deposit_GeV,
                                             accumulator.energy_scale,
                                             accumulator.counters);
      return;
    }
    auto last = static_cast<int>(end_grammage / accumulator.bin_width_g_per_cm2);
    last = last < 0 ? 0 : last;
    last = last > maximum_bin ? maximum_bin : last;
    auto const density = weighted_deposit_GeV / delta;
    if (first == last) {
      addProfileFixedPoint<AtomicOperations>(energy_loss + first, density * delta,
                                             accumulator.energy_scale,
                                             accumulator.counters);
      return;
    }
    addProfileFixedPoint<AtomicOperations>(
        energy_loss + first,
        density * (1. + static_cast<double>(first)) * accumulator.bin_width_g_per_cm2 -
            density * start_grammage,
        accumulator.energy_scale, accumulator.counters);
    addProfileFixedPoint<AtomicOperations>(
        energy_loss + last,
        density * end_grammage -
            density * static_cast<double>(last) * accumulator.bin_width_g_per_cm2,
        accumulator.energy_scale, accumulator.counters);
    for (auto bin = first + 1; bin < last; ++bin)
      addProfileFixedPoint<AtomicOperations>(
          energy_loss + bin, density * accumulator.bin_width_g_per_cm2,
          accumulator.energy_scale, accumulator.counters);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulatePointEnergyProfile(
      DeviceProfileAccumulator const& accumulator, long long* energy_loss,
      double grammage, double weighted_deposit_GeV, bool native_signed_cut = false) {
    if ((!(weighted_deposit_GeV > 0.) &&
         !(native_signed_cut && weighted_deposit_GeV < 0.)) ||
        accumulator.bins == 0)
      return;
    // Mirror EnergyLossWriter::write(Point, ...), including a zero dX threshold.
    // Reusing the segment overload with equal endpoints divides by zero when
    // the configured continuous-track threshold is zero.
    auto const maximum_bin = static_cast<int>(accumulator.bins - 1);
    auto bin = static_cast<int>(grammage / accumulator.bin_width_g_per_cm2);
    bin = bin < 0 ? 0 : bin;
    bin = bin > maximum_bin ? maximum_bin : bin;
    AtomicOperations::add(&accumulator.counters->deposited_steps, 1ULL);
    addProfileFixedPoint<AtomicOperations>(energy_loss + bin, weighted_deposit_GeV,
                                           accumulator.energy_scale,
                                           accumulator.counters);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulateTerminalEnergy(
      DeviceProfileAccumulator const& accumulator, double weighted_total_energy_GeV,
      bool observed) {
    if (!(weighted_total_energy_GeV >= 0.) || !profileFinite(weighted_total_energy_GeV)) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    addProfileFixedPoint<AtomicOperations>(
        observed ? &accumulator.counters->weighted_observed_total_energy
                 : &accumulator.counters->weighted_escaped_total_energy,
        weighted_total_energy_GeV, accumulator.energy_scale, accumulator.counters);
  }
  template <class AtomicOperations, class FinalStateRecord>
  CORSIKA_HOST_DEVICE inline void accumulateProfileThinning(
      FinalStateRecord const& record, DeviceProfileCounters* counters) {
    using namespace stochastic_reference;
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
      AtomicOperations::add(&counters->thinning_particles_discarded,
                            static_cast<unsigned long long>(original_multiplicity -
                                                            record.secondary_count));
  }
  template <class TransportRecord>
  CORSIKA_HOST_DEVICE inline TransportRecord const* findProfileTransportRecord(
      TransportRecord const* records, std::size_t count, std::uint64_t input_index) {
    std::size_t first = 0;
    std::size_t last = count;
    while (first < last) {
      auto const middle = first + (last - first) / 2;
      if (records[middle].input_index < input_index)
        first = middle + 1;
      else
        last = middle;
    }
    if (first >= count || records[first].input_index != input_index) return nullptr;
    return records + first;
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulatePhotonProfileStep(
      DeviceProfileProjection const& projection,
      DeviceProfileAccumulator const& accumulator, PhotonTransportRecord const& record) {
    using namespace stochastic_reference;
    auto const start = projectProfileGrammage(projection, record.start.position_m);
    auto const end = projectProfileGrammage(projection, record.end.position_m);
    accumulateParticleProfile<AtomicOperations>(accumulator, record.start.pid, start, end,
                                                record.start.weight);
    accumulatePointEnergyProfile<AtomicOperations>(
        accumulator, accumulator.energy_loss, end,
        record.cut_deposited_energy_GeV * record.start.weight);
    AtomicOperations::add(&accumulator.counters->steps, 1ULL);
    if (record.limit == PhotonTransportLimit::ParticleCut) {
      AtomicOperations::add(&accumulator.counters->photon_cuts, 1ULL);
      if (record.observation_surface_reached_before_cut != 0U) {
        addProfileFixedPoint<AtomicOperations>(
            &accumulator.counters->weighted_observation_cut_overlap_energy,
            record.cut_deposited_energy_GeV * record.start.weight,
            accumulator.energy_scale, accumulator.counters);
        accumulateTerminalEnergy<AtomicOperations>(
            accumulator, record.end.energy_GeV * record.start.weight, true);
      }
    } else if (record.limit == PhotonTransportLimit::ObservationSurface) {
      accumulateTerminalEnergy<AtomicOperations>(
          accumulator, record.end.energy_GeV * record.start.weight, true);
    } else if (record.limit == PhotonTransportLimit::EscapedEnvironment) {
      accumulateTerminalEnergy<AtomicOperations>(
          accumulator, record.end.energy_GeV * record.start.weight, false);
    }
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulatePhotonProfileFinalState(
      [[maybe_unused]] DeviceProfileProjection const& projection,
      DeviceProfileAccumulator const& accumulator,
      PhotonTransportRecord const* transport_records, std::size_t transport_count,
      PhotonPairFinalStateRecord const& record) {
    using namespace stochastic_reference;
    accumulateProfileThinning<AtomicOperations>(record, accumulator.counters);
    addProfileFixedPoint<AtomicOperations>(
        &accumulator.counters->weighted_mass_convention_correction,
        record.weighted_mass_convention_correction_GeV, accumulator.energy_scale,
        accumulator.counters);
    auto const atomic_electron_process = record.process_id == ComptonProcessId ||
                                         record.process_id == PhotoelectricProcessId;
    if (!atomic_electron_process) return;
    auto const* transport = findProfileTransportRecord(transport_records, transport_count,
                                                       record.input_index);
    if (transport == nullptr || transport->start.history_id != record.parent_history_id ||
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
        record.energy_split_fraction < 0. || record.energy_split_fraction > 1.) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    auto const deposit = transport->end.energy_GeV * (1. - record.energy_split_fraction) *
                         transport->start.weight;
    // Preserve scalar PROPOSAL's legacy dE/dX definition. Its photoelectric
    // secondary has this binding energy removed, but InteractionModel does
    // not pass it to EnergyLossWriter. Track it as an explicit closure sink.
    addProfileFixedPoint<AtomicOperations>(
        &accumulator.counters->weighted_unwritten_photoelectric_binding_energy, deposit,
        accumulator.energy_scale, accumulator.counters);
  }
  template <class AtomicOperations, bool AccumulateMoliereIterationCounters = true>
  CORSIKA_HOST_DEVICE inline void accumulateLeptonProfileStep(
      DeviceProfileProjection const& projection,
      DeviceProfileAccumulator const& accumulator, LeptonTransportRecord const& record) {
    using namespace stochastic_reference;
    auto const native_cut = isNativeMuonPairCut(record.start);
    if (native_cut && (record.limit != LeptonTransportLimit::ParticleCut ||
                       record.continuous_deposited_energy_GeV != 0. ||
                       !isNativeMuonPairCutDeposit(
                           record.start.pid, record.start.reserved, record.end.energy_GeV,
                           record.cut_deposited_energy_GeV))) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    auto const start = projectProfileGrammage(projection, record.start.position_m);
    auto const end = projectProfileGrammage(projection, record.end.position_m);
    accumulateParticleProfile<AtomicOperations>(accumulator, record.start.pid, start, end,
                                                record.start.weight);
    auto* const energy_loss = isMuonPid(record.start.pid) ? accumulator.muon_energy_loss
                                                          : accumulator.energy_loss;
    accumulateEnergyProfile<AtomicOperations>(
        accumulator, energy_loss, start, end,
        record.continuous_deposited_energy_GeV * record.start.weight);
    accumulatePointEnergyProfile<AtomicOperations>(
        accumulator, energy_loss, end,
        record.cut_deposited_energy_GeV * record.start.weight, native_cut);
    if (native_cut)
      addProfileFixedPoint<AtomicOperations>(
          &accumulator.counters->weighted_native_muon_pair_cut_energy,
          record.cut_deposited_energy_GeV * record.start.weight, accumulator.energy_scale,
          accumulator.counters);
    AtomicOperations::add(&accumulator.counters->steps, 1ULL);
    auto const limit = static_cast<std::int32_t>(record.limit);
    if (limit < 0 || limit >= 8)
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
    else
      AtomicOperations::add(&accumulator.counters->lepton_limits[limit], 1ULL);
    if (record.limit == LeptonTransportLimit::ParticleCut) {
      if (record.interaction.status != EmInteractionStatus::AtRestAnnihilation)
        addProfileFixedPoint<AtomicOperations>(
            &accumulator.counters->weighted_cut_rest_mass_energy,
            transportMassGeV(record.start.pid) * record.start.weight,
            accumulator.energy_scale, accumulator.counters);
      if (record.observation_surface_reached_before_cut != 0U) {
        addProfileFixedPoint<AtomicOperations>(
            &accumulator.counters->weighted_observation_cut_overlap_energy,
            (record.cut_deposited_energy_GeV + transportMassGeV(record.start.pid)) *
                record.start.weight,
            accumulator.energy_scale, accumulator.counters);
        accumulateTerminalEnergy<AtomicOperations>(
            accumulator, record.end.energy_GeV * record.start.weight, true);
      }
    } else if (record.limit == LeptonTransportLimit::ObservationSurface) {
      accumulateTerminalEnergy<AtomicOperations>(
          accumulator, record.end.energy_GeV * record.start.weight, true);
    } else if (record.limit == LeptonTransportLimit::EscapedEnvironment) {
      accumulateTerminalEnergy<AtomicOperations>(
          accumulator, record.end.energy_GeV * record.start.weight, false);
    }
    AtomicOperations::add(&accumulator.counters->moliere_trials, 1ULL);
    if constexpr (AccumulateMoliereIterationCounters) {
      AtomicOperations::add(
          &accumulator.counters->moliere_newton_iterations,
          static_cast<unsigned long long>(record.multiple_scattering_iterations));
      AtomicOperations::maximum(
          &accumulator.counters->moliere_max_newton_iterations,
          static_cast<unsigned long long>(record.multiple_scattering_iterations));
    }
    if (record.multiple_scattering_applied != 0)
      AtomicOperations::add(&accumulator.counters->moliere_deflections, 1ULL);
    else if (record.multiple_scattering_status ==
             static_cast<std::uint32_t>(MoliereStatus::NoDeflection))
      AtomicOperations::add(&accumulator.counters->moliere_zero_deflections, 1ULL);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulateLeptonProfileFinalState(
      DeviceProfileProjection const& projection,
      DeviceProfileAccumulator const& accumulator,
      LeptonTransportRecord const* transport_records, std::size_t transport_count,
      BremsFinalStateRecord const& record) {
    using namespace stochastic_reference;
    accumulateProfileThinning<AtomicOperations>(record, accumulator.counters);
    addProfileFixedPoint<AtomicOperations>(
        &accumulator.counters->weighted_mass_convention_correction,
        record.weighted_mass_convention_correction_GeV, accumulator.energy_scale,
        accumulator.counters);
    auto const atomic_electron_process = record.process_id == IonizationProcessId ||
                                         record.process_id == AnnihilationProcessId;
    if (!atomic_electron_process) return;
    auto const* transport = findProfileTransportRecord(transport_records, transport_count,
                                                       record.input_index);
    if (transport == nullptr || transport->start.history_id != record.parent_history_id) {
      AtomicOperations::add(&accumulator.counters->invalid_records, 1ULL);
      return;
    }
    if (record.process_id == IonizationProcessId && isMuonPid(transport->start.pid)) {
      auto const vertex_grammage =
          projectProfileGrammage(projection, transport->end.position_m);
      accumulateMuonParentProduction<AtomicOperations>(accumulator, vertex_grammage,
                                                       transport->start.weight);
    }
    addProfileFixedPoint<AtomicOperations>(
        &accumulator.counters->weighted_medium_rest_mass_input,
        (record.process_id == AnnihilationProcessId &&
                 record.final_state_draw_id == AtRestAnnihilationPolarDrawId
             ? TransportElectronMassGeV
             : ElectronMassGeV) *
            transport->start.weight,
        accumulator.energy_scale, accumulator.counters);
  }
} // namespace profile_accumulation_reference
