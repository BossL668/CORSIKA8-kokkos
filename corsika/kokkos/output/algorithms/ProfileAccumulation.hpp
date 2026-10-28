/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/output/ProfileAccumulatorView.hpp>
#include <corsika/kokkos/output/algorithms/ProfileProjection.hpp>
#include <corsika/kokkos/em/PhotonFinalStateRecord.hpp>
#include <corsika/kokkos/em/LeptonFinalStateRecord.hpp>
#include <corsika/kokkos/em/ProcessCapabilities.hpp>
#include <corsika/kokkos/em/algorithms/EmThinning.hpp>
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>
#include <corsika/kokkos/em/models/MoliereScattering.hpp>
#include <corsika/kokkos/em/models/PhotonPairKinematics.hpp>
#include <corsika/kokkos/em/random/LeptonFinalStateRandomDomains.hpp>
#include <cmath>
#include <cstdint>

namespace corsika::kokkos {

  inline constexpr double ProfileSignedIntegerLimit = 0x1p63;
  inline constexpr long long ProfileSignedIntegerMaximum = 9223372036854775807LL;
  inline constexpr long long ProfileSignedIntegerMinimum = (-9223372036854775807LL - 1LL);
  /** Production accumulation, with no RNG, cuts, thinning selection or transport.
   * AtomicOperations supplies signed/unsigned add returning the pre-add value,
   * and unsigned maximum. Preserve rint rounding and fixed-point overflow flags.
   * Final-state inputs require transport records sorted by inputIndex_.
   * Callers supply valid finite projection/storage parameters as documented in
   * the views. Photon final-state bookkeeping keeps its unused projection
   * argument for the common call interface; binding energy is a closure sink.
   */
  CORSIKA_HOST_DEVICE inline bool profile_finite(double value);
  CORSIKA_HOST_DEVICE inline double profile_abs(double value);
  CORSIKA_HOST_DEVICE inline double profile_floor(double value);
  CORSIKA_HOST_DEVICE inline double profile_ceil(double value);
  CORSIKA_HOST_DEVICE inline long long profile_round(double value);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline bool checked_profile_atomic_add(long long* address,
                                                             long long increment);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void add_profile_fixed_point(long long* address,
                                                          double contribution,
                                                          double scale,
                                                          ProfileCounters* counters);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_particle_profile(
      ProfileAccumulatorView const& accumulator, std::int32_t pid, double start_grammage,
      double end_grammage, double weight);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_muon_parent_production(
      ProfileAccumulatorView const& accumulator, double grammage, double parent_weight);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_energy_profile(
      ProfileAccumulatorView const& accumulator, long long* energy_loss,
      double start_grammage, double end_grammage, double weighted_deposit_GeV);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_point_energy_profile(
      ProfileAccumulatorView const& accumulator, long long* energy_loss, double grammage,
      double weighted_deposit_GeV, bool native_signed_cut = false);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_terminal_energy(
      ProfileAccumulatorView const& accumulator, double weighted_total_energy_GeV,
      bool observed);
  template <class AtomicOperations, class FinalStateRecord>
  CORSIKA_HOST_DEVICE inline void accumulate_profile_thinning(
      FinalStateRecord const& record, ProfileCounters* counters);
  template <class TransportRecord>
  CORSIKA_HOST_DEVICE inline TransportRecord const* find_profile_transport_record(
      TransportRecord const* records, std::size_t count, std::uint64_t input_index);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_photon_profile_step(
      ProfileProjectionView const& projection, ProfileAccumulatorView const& accumulator,
      PhotonTransportRecord const& record);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_photon_profile_final_state(
      [[maybe_unused]] ProfileProjectionView const& projection,
      ProfileAccumulatorView const& accumulator,
      PhotonTransportRecord const* transport_records, std::size_t transport_count,
      PhotonFinalStateRecord const& record);
  template <class AtomicOperations, bool AccumulateMoliereIterationCounters = true>
  CORSIKA_HOST_DEVICE inline void accumulate_lepton_profile_step(
      ProfileProjectionView const& projection, ProfileAccumulatorView const& accumulator,
      LeptonTransportRecord const& record);
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_lepton_profile_final_state(
      ProfileProjectionView const& projection, ProfileAccumulatorView const& accumulator,
      LeptonTransportRecord const* transport_records, std::size_t transport_count,
      LeptonFinalStateRecord const& record);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/algorithms/ProfileAccumulation.inl>
