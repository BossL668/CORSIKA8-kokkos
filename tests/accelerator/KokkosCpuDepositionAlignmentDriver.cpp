/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "KokkosCpuDepositionAlignmentDriver.hpp"

#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>

namespace corsika::accelerator::em::testing {
  namespace {
    namespace em = corsika::gpu::em;
    namespace device = corsika::accelerator::em::detail;
    using Exec = Kokkos::DefaultExecutionSpace;
    using Memory = typename Exec::memory_space;
    using Atomic = kokkos_detail::KokkosProfileAtomicOperations;
    constexpr std::size_t Bins = 8;
    constexpr double Scale = 1.e15;

    struct Storage {
      Kokkos::View<double*, Memory> grammage;
      Kokkos::View<long long*, Memory> histogram{"cpu_alignment_histogram", 8 * Bins};
      Kokkos::View<em::detail::DeviceProfileCounters*, Memory> counters{
          "cpu_alignment_counters", 1};
      Kokkos::View<em::ProjectedEmStepRecord*, Memory> projected{"projected_cut", 1};
      em::detail::DeviceProfileProjection projection{};
      em::detail::DeviceProfileAccumulator accumulator{};

      Storage(std::vector<double> const& support, double axis_step, double threshold) {
        grammage = decltype(grammage)("cpu_alignment_axis", support.size());
        auto host = Kokkos::create_mirror_view(grammage);
        for (std::size_t i = 0; i < support.size(); ++i) host(i) = support[i];
        Kokkos::deep_copy(grammage, host);
        projection.axis_direction[0] = 1.;
        projection.axis_step_length_m = axis_step;
        projection.axis_grammage_g_per_cm2 = grammage.data();
        projection.axis_support_count = support.size();
        accumulator.photons = histogram.data();
        accumulator.electrons = histogram.data() + Bins;
        accumulator.positrons = histogram.data() + 2 * Bins;
        accumulator.muons_minus = histogram.data() + 3 * Bins;
        accumulator.muons_plus = histogram.data() + 4 * Bins;
        accumulator.muon_parent_productions = histogram.data() + 5 * Bins;
        accumulator.energy_loss = histogram.data() + 6 * Bins;
        accumulator.muon_energy_loss = histogram.data() + 7 * Bins;
        accumulator.counters = counters.data();
        accumulator.bins = Bins;
        accumulator.bin_width_g_per_cm2 = 10.;
        accumulator.weight_scale = Scale;
        accumulator.energy_scale = Scale;
        accumulator.inverse_weight_scale = 1. / Scale;
        accumulator.inverse_energy_scale = 1. / Scale;
        accumulator.energy_loss_threshold_g_per_cm2 = threshold;
      }
      CpuDepositionAlignmentResult download() const {
        CpuDepositionAlignmentResult result;
        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, histogram);
        result.energy.resize(Bins);
        for (std::size_t i = 0; i < Bins; ++i)
          result.energy[i] = static_cast<double>(host(6 * Bins + i) + host(7 * Bins + i)) / Scale;
        result.counters = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, counters)(0);
        result.projected = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, projected)(0);
        return result;
      }
    };
  }

  CpuDepositionAlignmentResult runCpuDepositionDeviceStep(
      std::vector<double> const& support, double axis_step,
      gpu::em::LeptonTransportRecord const& input, bool photon, double threshold) {
    Storage storage(support, axis_step, threshold);
    auto const projection = storage.projection;
    auto const accumulator = storage.accumulator;
    auto const projected = storage.projected;
    auto const lepton = input;
    em::PhotonTransportRecord gamma{};
    gamma.start = lepton.start;
    gamma.end = lepton.end;
    gamma.limit = em::PhotonTransportLimit::ParticleCut;
    gamma.cut_deposited_energy_GeV = lepton.cut_deposited_energy_GeV;
    gamma.observation_surface_reached_before_cut = lepton.observation_surface_reached_before_cut;
    Kokkos::parallel_for("cpu_deposition_alignment", 1, KOKKOS_LAMBDA(int) {
      if (photon) {
        device::accumulatePhotonProfileStep<Atomic>(projection, accumulator, gamma);
        projected(0) = device::projectPhotonStep(projection, gamma);
      } else {
        device::accumulateLeptonProfileStep<Atomic>(projection, accumulator, lepton);
        projected(0) = device::projectLeptonStep(projection, lepton);
      }
    });
    return storage.download();
  }

  CpuDepositionAlignmentResult runCpuPhotoelectricDeviceStep(
      std::vector<double> const& support, double axis_step,
      gpu::em::PhotonTransportRecord const& input,
      gpu::em::PhotonPairLpmSnapshot const& input_snapshot) {
    Storage storage(support, axis_step, 1.e-4);
    auto const projection = storage.projection;
    auto const accumulator = storage.accumulator;
    auto const transport = input;
    auto const snapshot = input_snapshot;
    Kokkos::View<double*, Memory> energies("photoelectric_energies", 1);
    Kokkos::parallel_for("photoelectric_scalar_oracle", 1, KOKKOS_LAMBDA(int) {
      double energy{}, fraction{};
      if (!device::photoelectricEnergy(snapshot, snapshot.components[0].component_hash,
                                       transport.end.energy_GeV, energy, fraction)) {
        energies(0) = -1.;
        return;
      }
      energies(0) = energy;
      em::PhotonPairFinalStateRecord final{};
      final.parent_history_id = transport.start.history_id;
      final.process_id = em::PhotoelectricProcessId;
      final.secondary_count = 1;
      final.energy_split_fraction = fraction;
      final.weighted_mass_convention_correction_GeV = transport.start.weight *
          (em::TransportElectronMassGeV - em::ElectronMassGeV);
      device::accumulatePhotonProfileFinalState<Atomic>(
          projection, accumulator, &transport, 1, final);
    });
    auto result = storage.download();
    result.photoelectron_energy_GeV =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, energies)(0);
    return result;
  }
} // namespace corsika::accelerator::em::testing
