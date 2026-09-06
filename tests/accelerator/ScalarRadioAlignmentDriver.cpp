/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "ScalarRadioAlignmentDriver.hpp"

#include <Kokkos_Core.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>
#include <stdexcept>

namespace scalar_radio_test {
  using Execution = Kokkos::DefaultExecutionSpace;
  namespace radio = corsika::accelerator::radio;
  namespace detail = radio::detail;

  void initialize(int& argc, char**& argv) { Kokkos::initialize(argc, argv); }
  void finalize() { Kokkos::finalize(); }

  std::array<double, 3> doppler() {
    Kokkos::View<double*, typename Execution::memory_space> values("doppler", 3);
    Kokkos::parallel_for("scalar_radio_zero_doppler_rescue", 1,
      KOKKOS_LAMBDA(int) {
        detail::Vec3 const beta{1., 0x1p-30, 0.};
        detail::Vec3 const emit{1., 0x1p-30, 0.};
        values(0) = detail::coReasDoppler(1., beta, emit);
        values(1) = detail::coReasDoppler(1., {1., 0., 0.}, {1., 0., 0.});
        values(2) = detail::coReasDoppler(1.0003, {0.7, 0.2, 0.}, {0.6, 0.8, 0.});
      });
    auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    return {result(0), result(1), result(2)};
  }

  std::vector<double> observerWindow(
      std::vector<double> const& times, detail::DeviceObserver observer) {
    Kokkos::View<double*, typename Execution::memory_space> device_times("window_times", times.size());
    auto host_times = Kokkos::create_mirror_view(device_times);
    for (std::size_t i = 0; i < times.size(); ++i) host_times(i) = times[i];
    Kokkos::deep_copy(device_times, host_times);
    auto const bins = observer.number_of_bins;
    Kokkos::View<double*, typename Execution::memory_space> data("window_samples", 3 * bins);
    Kokkos::View<detail::DeviceRadioCounters*, typename Execution::memory_space> counters("window_counters", 1);
    detail::DeviceWaveforms samples{};
    samples.floating_x = data.data(); samples.floating_y = data.data() + bins;
    samples.floating_z = data.data() + 2 * bins;
    Kokkos::parallel_for("scalar_observer_window", times.size(),
      KOKKOS_LAMBDA(int i) {
        detail::addSample<radio::kokkos_detail::KokkosRadioAtomicOperations>(
            observer, samples, device_times(i), {static_cast<double>(1 << i), 0., 0.},
            &counters(0).coreas_contributions, counters.data());
      });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
    std::vector<double> result(bins);
    for (std::size_t i = 0; i < bins; ++i) result[i] = host(i);
    return result;
  }

  corsika::gpu::radio::GpuRadioWaveforms project(
      corsika::gpu::radio::GpuRadioConfig const& config,
      corsika::gpu::em::LeptonTransportRecord const& record, bool tiled) {
    radio::kokkos_detail::KokkosRadioAccumulator<Execution> accumulator;
    accumulator.initialize(config);
    using Record = corsika::gpu::em::LeptonTransportRecord;
    // One physical track plus invalid padding crosses the GPU tiling
    // threshold without changing the scalar oracle or summation order.
    std::size_t const count = tiled ? 16384 : 1;
    Kokkos::View<Record*, typename Execution::memory_space> tracks("scalar_radio_track", count);
    auto host = Kokkos::create_mirror_view(tracks);
    host(0) = record;
    Kokkos::deep_copy(tracks, host);
    accumulator.accumulateLeptonTracks(tracks, count);
    auto waveforms = accumulator.download();
    accumulator.reset();
    accumulator.accumulateLeptonTracks(tracks, count);
    auto const repeated = accumulator.download();
    for (std::size_t i = 0; i < waveforms.coreas.size(); ++i)
      if (!(waveforms.coreas[i].x == repeated.coreas[i].x &&
            waveforms.coreas[i].y == repeated.coreas[i].y &&
            waveforms.coreas[i].z == repeated.coreas[i].z &&
            waveforms.zhs[i].x == repeated.zhs[i].x &&
            waveforms.zhs[i].y == repeated.zhs[i].y &&
            waveforms.zhs[i].z == repeated.zhs[i].z))
        throw std::runtime_error("radio reset/reuse changed identical track waveforms");
    return waveforms;
  }
} // namespace scalar_radio_test
