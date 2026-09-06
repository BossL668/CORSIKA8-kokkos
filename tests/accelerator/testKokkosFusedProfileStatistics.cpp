/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp>

namespace {

  namespace kokkos_detail = corsika::accelerator::em::kokkos_detail;
  namespace gpu_em = corsika::gpu::em;

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using Memory = typename ExecutionSpace::memory_space;
  using Counts = kokkos_detail::ResidentLeptonSourceCounts;
  using Record = gpu_em::LeptonTransportRecord;
  using Counter = gpu_em::detail::DeviceProfileCounters;
  using StatisticsFunctor =
      kokkos_detail::ResidentLeptonTransportStatisticsProfileFunctor<
          ExecutionSpace>;

  struct ExpectedCounters {
    std::uint64_t steps{};
    std::uint64_t iterations{};
    std::uint64_t maximum_iterations{};

    void add(std::uint32_t const value) {
      ++steps;
      iterations += value;
      maximum_iterations = std::max(
          maximum_iterations, static_cast<std::uint64_t>(value));
    }
  };

  struct TestStorage {
    Kokkos::View<std::uint64_t*, Memory> call_statistics{
        "fused_profile_call_statistics",
        kokkos_detail::LeptonCallStatisticCount};
    Kokkos::View<Counter*, Memory> counters{
        "fused_profile_counters", 1};
    Kokkos::View<double*, Memory> axis_grammage{
        "fused_profile_axis_grammage", 2};
    gpu_em::detail::DeviceProfileProjection projection{};
    gpu_em::detail::DeviceProfileAccumulator accumulator{};

    explicit TestStorage(ExecutionSpace const& execution) {
      auto host_axis = Kokkos::create_mirror_view(axis_grammage);
      host_axis(0) = 0.;
      host_axis(1) = 1.;
      Kokkos::deep_copy(execution, axis_grammage, host_axis);

      projection.axis_direction[0] = 1.;
      projection.axis_step_length_m = 1.;
      projection.axis_grammage_g_per_cm2 = axis_grammage.data();
      projection.axis_support_count = axis_grammage.extent(0);

      // The records below use identical start/end positions and zero energy
      // deposit.  Histogram pointers are consequently never dereferenced; the
      // test isolates only scalar profile counters and the fused reducer.
      accumulator.counters = counters.data();
      accumulator.bins = 1;
      accumulator.bin_width_g_per_cm2 = 1.;
      accumulator.energy_loss_threshold_g_per_cm2 = 1.;
      accumulator.weight_scale = 1.;
      accumulator.inverse_weight_scale = 1.;
      accumulator.energy_scale = 1.;
      accumulator.inverse_energy_scale = 1.;
      reset(execution);
    }

    void reset(ExecutionSpace const& execution) {
      Kokkos::deep_copy(execution, call_statistics, std::uint64_t{0});
      Kokkos::deep_copy(execution, counters, Counter{});
      execution.fence("reset fused profile statistics test storage");
    }
  };

  bool keepStep(std::size_t const source, bool const holes) {
    if (!holes) return true;
    // Deliberately non-contiguous source records exercise the equivalence
    // between counts(source).step and the stable packed step ledger.
    return source % 5 != 1 && source % 7 != 3;
  }

  std::uint32_t iterationCount(std::size_t const source,
                               std::uint32_t const wavefront) {
    return 1U + static_cast<std::uint32_t>(
                    (source * 17U + wavefront * 11U) % 61U);
  }

  Record makeRecord(std::size_t const source,
                    std::uint32_t const wavefront) {
    Record record{};
    record.start.pid = static_cast<std::int32_t>(gpu_em::EmPid::Electron);
    record.start.energy_GeV = 1.;
    record.start.weight = 1.;
    record.start.history_id =
        static_cast<std::uint64_t>(wavefront) * 1000000ULL + source;
    record.end = record.start;
    record.limit = gpu_em::LeptonTransportLimit::ContinuousStep;
    record.traversed_grammage_g_per_cm2 = 1.;
    record.multiple_scattering_iterations =
        iterationCount(source, wavefront);
    record.multiple_scattering_applied = source % 2 == 0 ? 1U : 0U;
    return record;
  }

  void runWavefront(TestStorage& storage, std::size_t const count,
                    bool const holes, std::uint32_t const wavefront,
                    ExpectedCounters& expected,
                    ExecutionSpace const& execution,
                    bool const mixed_zero_grammage = false) {
    std::vector<Record> host_transports(count);
    std::vector<Counts> host_counts(count);
    std::vector<Record> host_steps;
    host_steps.reserve(count);
    for (std::size_t source = 0; source < count; ++source) {
      auto record = makeRecord(source, wavefront);
      if (mixed_zero_grammage && source % 3 == 0) {
        record.traversed_grammage_g_per_cm2 = 0.;
        record.multiple_scattering_iterations = 0;
        record.multiple_scattering_applied = 0;
        record.multiple_scattering_status = static_cast<std::uint16_t>(
            gpu_em::MoliereStatus::NoDeflection);
      }
      host_transports[source] = record;
      if (!keepStep(source, holes)) continue;
      host_counts[source].step = 1;
      host_steps.push_back(record);
      expected.add(record.multiple_scattering_iterations);
    }

    Kokkos::View<Record*, Memory> transports(
        "fused_profile_transports", count);
    Kokkos::View<Counts*, Memory> counts("fused_profile_counts", count);
    Kokkos::View<Record*, Memory> steps(
        "fused_profile_packed_steps", host_steps.size());
    auto transports_host = Kokkos::create_mirror_view(transports);
    auto counts_host = Kokkos::create_mirror_view(counts);
    auto steps_host = Kokkos::create_mirror_view(steps);
    for (std::size_t source = 0; source < count; ++source) {
      transports_host(source) = host_transports[source];
      counts_host(source) = host_counts[source];
    }
    for (std::size_t index = 0; index < host_steps.size(); ++index)
      steps_host(index) = host_steps[index];
    Kokkos::deep_copy(execution, transports, transports_host);
    Kokkos::deep_copy(execution, counts, counts_host);
    Kokkos::deep_copy(execution, steps, steps_host);

    Kokkos::View<Record const*, Memory> const_transports = transports;
    Kokkos::View<Counts const*, Memory> const_counts = counts;
    Kokkos::parallel_reduce(
        "c8_test_kokkos_fused_lepton_profile_statistics",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
        StatisticsFunctor{{const_transports, const_counts,
                           storage.call_statistics},
                          storage.projection,
                          storage.accumulator,
                          steps,
                          host_steps.size()});
  }

  void requireState(TestStorage const& storage,
                    ExpectedCounters const& expected,
                    std::string const& context,
                    ExecutionSpace const& execution) {
    auto host_counters = Kokkos::create_mirror_view(storage.counters);
    auto host_statistics =
        Kokkos::create_mirror_view(storage.call_statistics);
    Kokkos::deep_copy(execution, host_counters, storage.counters);
    Kokkos::deep_copy(
        execution, host_statistics, storage.call_statistics);
    execution.fence("download fused profile statistics test state");

    auto const& counters = host_counters(0);
    auto const statistic = [&](std::size_t const index) {
      return host_statistics(
          kokkos_detail::LeptonCallTransportStatisticBase + index);
    };
    if (counters.steps != expected.steps ||
        counters.moliere_trials != expected.steps ||
        counters.moliere_newton_iterations != expected.iterations ||
        counters.moliere_max_newton_iterations !=
            expected.maximum_iterations ||
        counters.fixed_point_overflows != 0 ||
        counters.invalid_records != 0 ||
        statistic(kokkos_detail::LeptonTransportMoliereNewtonIterations) !=
            expected.iterations ||
        statistic(
            kokkos_detail::LeptonTransportMoliereMaximumNewtonIterations) !=
            expected.maximum_iterations ||
        statistic(kokkos_detail::LeptonTransportMoliereTrials) !=
            expected.steps) {
      throw std::runtime_error(
          context + ": fused Moliere profile/statistics counters differ");
    }
  }

  void runSingleWavefrontCases(TestStorage& storage,
                               ExecutionSpace const& execution) {
    constexpr std::array<std::size_t, 8> CountsToTest{
        0, 1, 31, 32, 33, 127, 128, 129};
    for (auto const count : CountsToTest) {
      storage.reset(execution);
      ExpectedCounters expected{};
      runWavefront(storage, count, false, 1, expected, execution);
      requireState(storage, expected,
                   "dense wavefront size " + std::to_string(count),
                   execution);
    }

    storage.reset(execution);
    ExpectedCounters expected{};
    runWavefront(storage, 129, true, 2, expected, execution);
    requireState(storage, expected, "wavefront with counts.step holes",
                 execution);

    storage.reset(execution);
    expected = {};
    runWavefront(storage, 129, true, 5, expected, execution, true);
    requireState(storage, expected,
                 "zero-grammage steps still count as Moliere trials",
                 execution);
  }

  void runCumulativeCase(TestStorage& storage,
                         ExecutionSpace const& execution) {
    storage.reset(execution);
    ExpectedCounters expected{};
    runWavefront(storage, 33, false, 3, expected, execution);
    requireState(storage, expected, "first cumulative wavefront", execution);
    runWavefront(storage, 129, true, 4, expected, execution);
    requireState(storage, expected, "second cumulative wavefront", execution);
  }

  struct ExpectedPhotonStatistics {
    std::uint64_t interactions{};
    std::uint64_t layer_boundaries{};
    std::uint64_t particle_cuts{};
    std::uint64_t newton_iterations{};
    std::uint64_t bisection_iterations{};
    std::uint64_t inverse_failures{};
  };

  gpu_em::PhotonTransportRecord makePhotonRecord(std::size_t const index) {
    gpu_em::PhotonTransportRecord record{};
    record.start.pid = static_cast<std::int32_t>(gpu_em::EmPid::Photon);
    record.start.energy_GeV = 1.;
    record.start.weight = 1.;
    record.start.history_id = 7000000ULL + index;
    record.end = record.start;
    switch (index % 5) {
      case 0:
        record.limit = gpu_em::PhotonTransportLimit::Interaction;
        break;
      case 1:
        record.limit = gpu_em::PhotonTransportLimit::LayerBoundary;
        break;
      case 2:
        record.limit = gpu_em::PhotonTransportLimit::ParticleCut;
        break;
      case 3:
        record.limit = gpu_em::PhotonTransportLimit::ObservationSurface;
        break;
      default:
        record.limit = gpu_em::PhotonTransportLimit::EscapedEnvironment;
        break;
    }
    return record;
  }

  void runPhotonPackedStatisticsCase(TestStorage& storage,
                                     std::size_t const source_count,
                                     std::size_t const step_count,
                                     bool const with_profile,
                                     ExecutionSpace const& execution) {
    using Selection =
        corsika::accelerator::em::detail::InteractionSelectionOutcome;
    if (step_count > source_count)
      throw std::invalid_argument(
          "photon packed-statistics test has more steps than sources");

    std::vector<Selection> host_selections(source_count);
    std::vector<gpu_em::PhotonTransportRecord> host_steps(step_count);
    ExpectedPhotonStatistics expected{};
    for (std::size_t source = 0; source < source_count; ++source) {
      auto& selection = host_selections[source];
      selection.native_newton_iterations =
          static_cast<std::uint32_t>((source * 3U + 1U) % 17U);
      selection.native_bisection_iterations =
          static_cast<std::uint32_t>((source * 5U + 2U) % 19U);
      selection.native_inverse_failures = source % 23 == 7 ? 1U : 0U;
      expected.newton_iterations += selection.native_newton_iterations;
      expected.bisection_iterations +=
          selection.native_bisection_iterations;
      expected.inverse_failures += selection.native_inverse_failures;
    }
    for (std::size_t index = 0; index < step_count; ++index) {
      auto const record = makePhotonRecord(index);
      host_steps[index] = record;
      expected.interactions +=
          record.limit == gpu_em::PhotonTransportLimit::Interaction;
      expected.layer_boundaries +=
          record.limit == gpu_em::PhotonTransportLimit::LayerBoundary;
      expected.particle_cuts +=
          record.limit == gpu_em::PhotonTransportLimit::ParticleCut;
    }

    Kokkos::View<Selection*, Memory> selections(
        "photon_packed_statistics_selections", source_count);
    Kokkos::View<gpu_em::PhotonTransportRecord*, Memory> steps(
        "photon_packed_statistics_steps", step_count);
    Kokkos::View<std::uint64_t*, Memory> statistics(
        "photon_packed_statistics", kokkos_detail::PhotonCallStatisticCount);
    auto selections_host = Kokkos::create_mirror_view(selections);
    auto steps_host = Kokkos::create_mirror_view(steps);
    for (std::size_t source = 0; source < source_count; ++source)
      selections_host(source) = host_selections[source];
    for (std::size_t index = 0; index < step_count; ++index)
      steps_host(index) = host_steps[index];
    Kokkos::deep_copy(execution, selections, selections_host);
    Kokkos::deep_copy(execution, steps, steps_host);
    Kokkos::deep_copy(execution, statistics, std::uint64_t{0});
    Kokkos::deep_copy(execution, storage.counters, Counter{});

    Kokkos::View<Selection const*, Memory> const_selections = selections;
    Kokkos::View<gpu_em::PhotonTransportRecord const*, Memory> const_steps =
        steps;
    if (with_profile) {
      Kokkos::parallel_reduce(
          "c8_test_kokkos_photon_packed_profile_statistics",
          Kokkos::RangePolicy<ExecutionSpace>(
              execution, 0, source_count),
          kokkos_detail::ResidentPhotonTransportStatisticsProfileFunctor<
              ExecutionSpace>{{const_selections, const_steps, step_count,
                                statistics},
                              storage.projection,
                              storage.accumulator});
    } else {
      Kokkos::parallel_reduce(
          "c8_test_kokkos_photon_packed_statistics",
          Kokkos::RangePolicy<ExecutionSpace>(
              execution, 0, source_count),
          kokkos_detail::ResidentPhotonTransportStatisticsFunctor<
              ExecutionSpace>{const_selections, const_steps, step_count,
                              statistics});
    }

    auto statistics_host = Kokkos::create_mirror_view(statistics);
    auto counters_host = Kokkos::create_mirror_view(storage.counters);
    Kokkos::deep_copy(execution, statistics_host, statistics);
    Kokkos::deep_copy(execution, counters_host, storage.counters);
    execution.fence("download photon packed statistics test state");
    auto const statistic = [&](std::size_t const index) {
      return statistics_host(
          kokkos_detail::PhotonCallTransportStatisticOffset + index);
    };
    if (statistic(kokkos_detail::PhotonTransportInteractionVertices) !=
            expected.interactions ||
        statistic(kokkos_detail::PhotonTransportLayerBoundaries) !=
            expected.layer_boundaries ||
        statistic(kokkos_detail::PhotonTransportParticleCuts) !=
            expected.particle_cuts ||
        statistic(kokkos_detail::PhotonSelectionNativeNewtonIterations) !=
            expected.newton_iterations ||
        statistic(kokkos_detail::PhotonSelectionNativeBisectionIterations) !=
            expected.bisection_iterations ||
        statistic(kokkos_detail::PhotonSelectionNativeInverseFailures) !=
            expected.inverse_failures)
      throw std::runtime_error(
          "packed photon transport statistics differ from scalar oracle");
    if (with_profile) {
      if (counters_host(0).steps != step_count ||
          counters_host(0).photon_cuts != expected.particle_cuts ||
          counters_host(0).invalid_records != 0)
        throw std::runtime_error(
            "packed photon profile counters differ from scalar oracle");
    } else if (counters_host(0).steps != 0 ||
               counters_host(0).photon_cuts != 0) {
      throw std::runtime_error(
          "statistics-only photon reduction modified profile counters");
    }
  }

  void runPhotonPackedStatisticsCases(TestStorage& storage,
                                      ExecutionSpace const& execution) {
    constexpr std::array<std::size_t, 8> CountsToTest{
        0, 1, 31, 32, 33, 127, 128, 129};
    for (auto const count : CountsToTest) {
      auto const packed = count == 0 ? 0 : count - count / 4;
      runPhotonPackedStatisticsCase(
          storage, count, packed, false, execution);
      runPhotonPackedStatisticsCase(
          storage, count, packed, true, execution);
    }
  }

  void runSignedAtomicFetchAddCase(ExecutionSpace const& execution) {
    constexpr std::size_t Count = 4096;
    Kokkos::View<long long*, Memory> value("signed_profile_atomic_value", 1);
    Kokkos::View<long long*, Memory> previous(
        "signed_profile_atomic_previous", Count);
    Kokkos::deep_copy(execution, value, 0LL);
    Kokkos::parallel_for(
        "c8_test_kokkos_signed_profile_atomic_add_positive",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, Count),
        KOKKOS_LAMBDA(std::size_t const index) {
          previous(index) = kokkos_detail::KokkosProfileAtomicOperations::add(
              value.data(), 1LL);
        });
    auto host_value = Kokkos::create_mirror_view(value);
    auto host_previous = Kokkos::create_mirror_view(previous);
    Kokkos::deep_copy(execution, host_value, value);
    Kokkos::deep_copy(execution, host_previous, previous);
    execution.fence("download positive signed profile atomic results");
    std::vector<long long> ordered_previous(Count);
    for (std::size_t index = 0; index < Count; ++index)
      ordered_previous[index] = host_previous(index);
    std::sort(ordered_previous.begin(), ordered_previous.end());
    if (host_value(0) != static_cast<long long>(Count))
      throw std::runtime_error("positive signed profile atomic sum differs");
    for (std::size_t index = 0; index < Count; ++index)
      if (ordered_previous[index] != static_cast<long long>(index))
        throw std::runtime_error(
            "positive signed profile atomic previous value differs");

    Kokkos::parallel_for(
        "c8_test_kokkos_signed_profile_atomic_add_negative",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, Count),
        KOKKOS_LAMBDA(std::size_t const index) {
          previous(index) = kokkos_detail::KokkosProfileAtomicOperations::add(
              value.data(), -1LL);
        });
    Kokkos::deep_copy(execution, host_value, value);
    Kokkos::deep_copy(execution, host_previous, previous);
    execution.fence("download negative signed profile atomic results");
    for (std::size_t index = 0; index < Count; ++index)
      ordered_previous[index] = host_previous(index);
    std::sort(ordered_previous.begin(), ordered_previous.end());
    if (host_value(0) != 0LL)
      throw std::runtime_error("negative signed profile atomic sum differs");
    for (std::size_t index = 0; index < Count; ++index)
      if (ordered_previous[index] != static_cast<long long>(index + 1))
        throw std::runtime_error(
            "negative signed profile atomic previous value differs");

    // Exercise pre-add bit patterns with the sign bit set.  This is the
    // important companion to the positive-domain check above: CUDA performs
    // the operation through unsigned long long atomicAdd and converts the
    // returned two's-complement bit pattern back to long long.
    Kokkos::parallel_for(
        "c8_test_kokkos_signed_profile_atomic_add_negative_domain",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, Count),
        KOKKOS_LAMBDA(std::size_t const index) {
          previous(index) = kokkos_detail::KokkosProfileAtomicOperations::add(
              value.data(), -1LL);
        });
    Kokkos::deep_copy(execution, host_value, value);
    Kokkos::deep_copy(execution, host_previous, previous);
    execution.fence("download negative-domain profile atomic results");
    for (std::size_t index = 0; index < Count; ++index)
      ordered_previous[index] = host_previous(index);
    std::sort(ordered_previous.begin(), ordered_previous.end());
    if (host_value(0) != -static_cast<long long>(Count))
      throw std::runtime_error(
          "negative-domain signed profile atomic sum differs");
    for (std::size_t index = 0; index < Count; ++index) {
      auto const expected = -static_cast<long long>(Count) + 1LL +
                            static_cast<long long>(index);
      if (ordered_previous[index] != expected)
        throw std::runtime_error(
            "negative-domain signed profile atomic previous value differs");
    }

    Kokkos::parallel_for(
        "c8_test_kokkos_signed_profile_atomic_add_from_negative_domain",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, Count),
        KOKKOS_LAMBDA(std::size_t const index) {
          previous(index) = kokkos_detail::KokkosProfileAtomicOperations::add(
              value.data(), 1LL);
        });
    Kokkos::deep_copy(execution, host_value, value);
    Kokkos::deep_copy(execution, host_previous, previous);
    execution.fence("download return-from-negative profile atomic results");
    for (std::size_t index = 0; index < Count; ++index)
      ordered_previous[index] = host_previous(index);
    std::sort(ordered_previous.begin(), ordered_previous.end());
    if (host_value(0) != 0LL)
      throw std::runtime_error(
          "return-from-negative signed profile atomic sum differs");
    for (std::size_t index = 0; index < Count; ++index) {
      auto const expected = -static_cast<long long>(Count) +
                            static_cast<long long>(index);
      if (ordered_previous[index] != expected)
        throw std::runtime_error(
            "return-from-negative signed profile atomic previous value "
            "differs");
    }

    // Stay strictly within the signed range while touching both boundaries.
    // Actual wraparound is intentionally not exercised on host backends,
    // where signed overflow is not a portable C++ operation.
    auto const signed_maximum = std::numeric_limits<long long>::max();
    auto const signed_minimum = std::numeric_limits<long long>::min();
    Kokkos::deep_copy(
        execution, value,
        signed_maximum - static_cast<long long>(Count));
    Kokkos::parallel_for(
        "c8_test_kokkos_signed_profile_atomic_add_maximum_boundary",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, Count),
        KOKKOS_LAMBDA(std::size_t const index) {
          previous(index) = kokkos_detail::KokkosProfileAtomicOperations::add(
              value.data(), 1LL);
        });
    Kokkos::deep_copy(execution, host_value, value);
    Kokkos::deep_copy(execution, host_previous, previous);
    execution.fence("download maximum-boundary profile atomic results");
    for (std::size_t index = 0; index < Count; ++index)
      ordered_previous[index] = host_previous(index);
    std::sort(ordered_previous.begin(), ordered_previous.end());
    if (host_value(0) != signed_maximum)
      throw std::runtime_error(
          "maximum-boundary signed profile atomic sum differs");
    for (std::size_t index = 0; index < Count; ++index) {
      auto const expected =
          signed_maximum - static_cast<long long>(Count) +
          static_cast<long long>(index);
      if (ordered_previous[index] != expected)
        throw std::runtime_error(
            "maximum-boundary signed profile atomic previous value differs");
    }

    Kokkos::deep_copy(
        execution, value,
        signed_minimum + static_cast<long long>(Count));
    Kokkos::parallel_for(
        "c8_test_kokkos_signed_profile_atomic_add_minimum_boundary",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, Count),
        KOKKOS_LAMBDA(std::size_t const index) {
          previous(index) = kokkos_detail::KokkosProfileAtomicOperations::add(
              value.data(), -1LL);
        });
    Kokkos::deep_copy(execution, host_value, value);
    Kokkos::deep_copy(execution, host_previous, previous);
    execution.fence("download minimum-boundary profile atomic results");
    for (std::size_t index = 0; index < Count; ++index)
      ordered_previous[index] = host_previous(index);
    std::sort(ordered_previous.begin(), ordered_previous.end());
    if (host_value(0) != signed_minimum)
      throw std::runtime_error(
          "minimum-boundary signed profile atomic sum differs");
    for (std::size_t index = 0; index < Count; ++index) {
      auto const expected =
          signed_minimum + 1LL + static_cast<long long>(index);
      if (ordered_previous[index] != expected)
        throw std::runtime_error(
            "minimum-boundary signed profile atomic previous value differs");
    }
  }

  void runFrontControlAliasCase(ExecutionSpace const& execution) {
    {
      kokkos_detail::KokkosResidentPhotonWorkspace<ExecutionSpace> workspace;
      constexpr std::size_t AliasSourceCount = 129;
      workspace.ensureCapacity(AliasSourceCount, execution);
      kokkos_detail::ResidentPhotonScanValue totals{};
      for (std::size_t column = 0;
           column < kokkos_detail::PhotonOffsetCount; ++column)
        totals.values[column] = 1000 + column;
      Kokkos::deep_copy(execution, workspace.scan_totals, totals);
      Kokkos::deep_copy(
          execution, workspace.interaction_count, std::uint64_t{73});
      Kokkos::deep_copy(execution, workspace.error, std::uint32_t{19});
      Kokkos::deep_copy(
          execution, workspace.host_front_control, workspace.front_control);
      execution.fence("download photon front-control alias test");
      auto const& control = workspace.host_front_control();
      for (std::size_t column = 0;
           column < kokkos_detail::PhotonOffsetCount; ++column)
        if (control.totals.values[column] != 1000 + column)
          throw std::runtime_error(
              "photon scan total does not alias front control");
      if (control.interaction_count != 73 ||
          control.materialization_error != 19)
        throw std::runtime_error(
            "photon scalar does not alias front control");

      Kokkos::deep_copy(execution, workspace.error, std::uint32_t{0});
      Kokkos::deep_copy(
          execution, workspace.host_front_control, workspace.front_control);
      execution.fence("download photon front-control reset test");
      auto const& reset_control = workspace.host_front_control();
      if (reset_control.interaction_count != 73 ||
          reset_control.materialization_error != 0)
        throw std::runtime_error(
            "photon error reset clobbered adjacent front control state");

      auto host_counts = Kokkos::create_mirror_view(workspace.counts);
      auto host_flags =
          Kokkos::create_mirror_view(workspace.interaction_flags);
      std::uint32_t expected_interactions = 0;
      for (std::size_t source = 0; source < AliasSourceCount; ++source) {
        auto& source_counts = host_counts(source);
        source_counts = {};
        source_counts.child = 2;
        source_counts.next = 1;
        source_counts.charged = 2;
        source_counts.fallback_stage =
            kokkos_detail::PhotonSelectionFallback;
        source_counts.observation = 1;
        source_counts.step = 1;
        source_counts.record = 1;
        host_flags(source) = source % 5 == 0 ? 1U : 0U;
        expected_interactions += host_flags(source) != 0;
      }
      Kokkos::deep_copy(execution, workspace.counts, host_counts);
      Kokkos::deep_copy(
          execution, workspace.interaction_flags, host_flags);
      Kokkos::parallel_scan(
          "c8_test_photon_scan_into_front_control_alias",
          Kokkos::RangePolicy<ExecutionSpace>(
              execution, 0, AliasSourceCount),
          kokkos_detail::ResidentPhotonScanFunctor<ExecutionSpace>{
              workspace.counts, workspace.offsets},
          workspace.scan_totals);
      Kokkos::parallel_scan(
          "c8_test_photon_compaction_into_front_control_alias",
          Kokkos::RangePolicy<ExecutionSpace>(
              execution, 0, AliasSourceCount),
          kokkos_detail::ResidentPhotonCompactionScanFunctor<ExecutionSpace>{
              workspace.interaction_flags, workspace.interaction_sources},
          workspace.interaction_count);
      Kokkos::deep_copy(
          execution, workspace.host_front_control, workspace.front_control);
      execution.fence("download photon scan alias result");
      auto const& scan_control = workspace.host_front_control();
      if (scan_control.totals.values[kokkos_detail::PhotonChildOffset] !=
              2 * AliasSourceCount ||
          scan_control.totals.values[kokkos_detail::PhotonNextOffset] !=
              AliasSourceCount ||
          scan_control.interaction_count != expected_interactions)
        throw std::runtime_error(
            "photon parallel scan did not target front-control aliases");
    }

    {
      kokkos_detail::KokkosResidentLeptonWorkspace<ExecutionSpace> workspace;
      constexpr std::size_t AliasSourceCount = 129;
      workspace.ensureQueueCapacity(AliasSourceCount, execution);
      workspace.ensureSourceCapacity(
          AliasSourceCount, AliasSourceCount, execution);
      kokkos_detail::ResidentLeptonScanValue totals{};
      for (std::size_t column = 0;
           column < kokkos_detail::LeptonOffsetCount; ++column)
        totals.values[column] = 2000 + column;
      Kokkos::deep_copy(execution, workspace.scan_totals, totals);
      Kokkos::deep_copy(
          execution, workspace.interaction_count, std::uint64_t{91});
      Kokkos::deep_copy(
          execution, workspace.vertex_interaction_count,
          std::uint64_t{73});
      Kokkos::deep_copy(execution, workspace.error, std::uint32_t{23});
      Kokkos::deep_copy(
          execution, workspace.host_front_control, workspace.front_control);
      execution.fence("download lepton front-control alias test");
      auto const& control = workspace.host_front_control();
      for (std::size_t column = 0;
           column < kokkos_detail::LeptonOffsetCount; ++column)
        if (control.totals.values[column] != 2000 + column)
          throw std::runtime_error(
              "lepton scan total does not alias front control");
      if (control.interaction_count != 91 ||
          control.vertex_interaction_count != 73 ||
          control.materialization_error != 23)
        throw std::runtime_error(
            "lepton scalar does not alias front control");

      Kokkos::deep_copy(execution, workspace.error, std::uint32_t{0});
      Kokkos::deep_copy(
          execution, workspace.host_front_control, workspace.front_control);
      execution.fence("download lepton front-control reset test");
      auto const& reset_control = workspace.host_front_control();
      if (reset_control.interaction_count != 91 ||
          reset_control.vertex_interaction_count != 73 ||
          reset_control.materialization_error != 0)
        throw std::runtime_error(
            "lepton error reset clobbered adjacent front control state");

      auto host_counts = Kokkos::create_mirror_view(workspace.counts);
      auto host_flags =
          Kokkos::create_mirror_view(workspace.interaction_flags);
      std::uint32_t expected_interactions = 0;
      for (std::size_t source = 0; source < AliasSourceCount; ++source) {
        auto& source_counts = host_counts(source);
        source_counts = {};
        source_counts.child = 3;
        source_counts.next = 3;
        source_counts.photon = 2;
        source_counts.fallback_stage =
            kokkos_detail::LeptonVertexFallback;
        source_counts.observation = 1;
        source_counts.decay = 1;
        source_counts.step = 1;
        source_counts.record = 1;
        host_flags(source) = source % 7 == 0 ? 1U : 0U;
        expected_interactions += host_flags(source) != 0;
      }
      Kokkos::deep_copy(execution, workspace.counts, host_counts);
      Kokkos::deep_copy(
          execution, workspace.interaction_flags, host_flags);
      Kokkos::parallel_scan(
          "c8_test_lepton_scan_into_front_control_alias",
          Kokkos::RangePolicy<ExecutionSpace>(
              execution, 0, AliasSourceCount),
          kokkos_detail::ResidentLeptonScanFunctor<ExecutionSpace>{
              workspace.counts, workspace.offsets},
          workspace.scan_totals);
      Kokkos::parallel_scan(
          "c8_test_lepton_compaction_into_front_control_alias",
          Kokkos::RangePolicy<ExecutionSpace>(
              execution, 0, AliasSourceCount),
          kokkos_detail::ResidentLeptonInteractionScanFunctor<ExecutionSpace>{
              workspace.interaction_flags, workspace.interaction_sources},
          workspace.interaction_count);
      Kokkos::deep_copy(
          execution, workspace.host_front_control, workspace.front_control);
      execution.fence("download lepton scan alias result");
      auto const& scan_control = workspace.host_front_control();
      if (scan_control.totals.values[kokkos_detail::LeptonChildOffset] !=
              3 * AliasSourceCount ||
          scan_control.totals.values[kokkos_detail::LeptonNextOffset] !=
              3 * AliasSourceCount ||
          scan_control.totals.values[
              kokkos_detail::LeptonVertexFallbackOffset] !=
              AliasSourceCount ||
          scan_control.totals.values[
              kokkos_detail::LeptonFinalStateFallbackOffset] != 0 ||
          scan_control.interaction_count != expected_interactions ||
          scan_control.vertex_interaction_count != 73)
        throw std::runtime_error(
            "lepton parallel scan did not target front-control aliases");
    }
  }

  void runPhotonFallbackStageOrderingCase(
      ExecutionSpace const& execution) {
    constexpr std::array<std::uint32_t, 12> Stages{
        kokkos_detail::PhotonFinalStateFallback,
        kokkos_detail::PhotonSelectionFallback,
        kokkos_detail::PhotonTransportFallback,
        kokkos_detail::PhotonNoFallback,
        kokkos_detail::PhotonSelectionFallback,
        kokkos_detail::PhotonFinalStateFallback,
        kokkos_detail::PhotonTransportFallback,
        kokkos_detail::PhotonSelectionFallback,
        kokkos_detail::PhotonNoFallback,
        kokkos_detail::PhotonFinalStateFallback,
        kokkos_detail::PhotonTransportFallback,
        kokkos_detail::PhotonSelectionFallback};
    constexpr std::size_t SourceCount = Stages.size();
    constexpr std::size_t FallbackCount = 10;
    using SourceCounts = kokkos_detail::ResidentPhotonSourceCounts;
    using ScanValue = kokkos_detail::ResidentPhotonScanValue;
    Kokkos::View<SourceCounts*, Memory> counts(
        "photon_stage_order_counts", SourceCount);
    Kokkos::View<std::uint64_t*[kokkos_detail::PhotonOffsetCount], Memory>
        offsets("photon_stage_order_offsets", SourceCount);
    Kokkos::View<ScanValue, Memory> totals("photon_stage_order_totals");
    Kokkos::View<std::size_t*, Memory> ordered_sources(
        "photon_stage_order_sources", FallbackCount);
    auto host_counts = Kokkos::create_mirror_view(counts);
    for (std::size_t source = 0; source < SourceCount; ++source)
      host_counts(source).fallback_stage = Stages[source];
    Kokkos::deep_copy(execution, counts, host_counts);
    Kokkos::parallel_scan(
        "c8_test_photon_stage_order_scan",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, SourceCount),
        kokkos_detail::ResidentPhotonScanFunctor<ExecutionSpace>{
            counts, offsets},
        totals);
    Kokkos::parallel_for(
        "c8_test_photon_stage_order_materialize",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, SourceCount),
        KOKKOS_LAMBDA(std::size_t const source) {
          auto const stage = counts(source).fallback_stage;
          if (stage == kokkos_detail::PhotonNoFallback) return;
          auto const output = kokkos_detail::photonFallbackOutputOffset(
              stage,
              offsets(source,
                      kokkos_detail::PhotonSelectionFallbackOffset),
              offsets(source,
                      kokkos_detail::PhotonTransportFallbackOffset),
              offsets(source,
                      kokkos_detail::PhotonFinalStateFallbackOffset),
              totals());
          ordered_sources(output) = source;
        });

    auto host_totals = Kokkos::create_mirror_view(totals);
    auto host_ordered = Kokkos::create_mirror_view(ordered_sources);
    Kokkos::deep_copy(execution, host_totals, totals);
    Kokkos::deep_copy(execution, host_ordered, ordered_sources);
    execution.fence("download photon stage-major fallback order");
    if (host_totals().values[
            kokkos_detail::PhotonSelectionFallbackOffset] != 4 ||
        host_totals().values[
            kokkos_detail::PhotonTransportFallbackOffset] != 3 ||
        host_totals().values[
            kokkos_detail::PhotonFinalStateFallbackOffset] != 3)
      throw std::runtime_error(
          "photon fallback stage totals differ from scalar oracle");
    constexpr std::array<std::size_t, FallbackCount> Expected{
        1, 4, 7, 11, 2, 6, 10, 0, 5, 9};
    for (std::size_t index = 0; index < FallbackCount; ++index)
      if (host_ordered(index) != Expected[index])
        throw std::runtime_error(
            "photon fallback output is not stage-major and source-stable");
  }

  void runLeptonFallbackStageOrderingCase(
      ExecutionSpace const& execution) {
    constexpr std::array<std::uint32_t, 15> Stages{
        kokkos_detail::LeptonFinalStateFallback,
        kokkos_detail::LeptonSelectionFallback,
        kokkos_detail::LeptonVertexFallback,
        kokkos_detail::LeptonTransportFallback,
        kokkos_detail::LeptonNoFallback,
        kokkos_detail::LeptonSelectionFallback,
        kokkos_detail::LeptonFinalStateFallback,
        kokkos_detail::LeptonVertexFallback,
        kokkos_detail::LeptonTransportFallback,
        kokkos_detail::LeptonSelectionFallback,
        kokkos_detail::LeptonNoFallback,
        kokkos_detail::LeptonFinalStateFallback,
        kokkos_detail::LeptonTransportFallback,
        kokkos_detail::LeptonVertexFallback,
        kokkos_detail::LeptonSelectionFallback};
    constexpr std::size_t SourceCount = Stages.size();
    constexpr std::size_t FallbackCount = 13;
    using SourceCounts = kokkos_detail::ResidentLeptonSourceCounts;
    using ScanValue = kokkos_detail::ResidentLeptonScanValue;
    Kokkos::View<SourceCounts*, Memory> counts(
        "lepton_stage_order_counts", SourceCount);
    Kokkos::View<std::uint64_t*[kokkos_detail::LeptonOffsetCount], Memory>
        offsets("lepton_stage_order_offsets", SourceCount);
    Kokkos::View<ScanValue, Memory> totals("lepton_stage_order_totals");
    Kokkos::View<std::size_t*, Memory> ordered_sources(
        "lepton_stage_order_sources", FallbackCount);
    auto host_counts = Kokkos::create_mirror_view(counts);
    for (std::size_t source = 0; source < SourceCount; ++source) {
      host_counts(source) = {};
      host_counts(source).fallback_stage = Stages[source];
    }
    Kokkos::deep_copy(execution, counts, host_counts);
    Kokkos::parallel_scan(
        "c8_test_lepton_stage_order_scan",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, SourceCount),
        kokkos_detail::ResidentLeptonScanFunctor<ExecutionSpace>{
            counts, offsets},
        totals);
    Kokkos::parallel_for(
        "c8_test_lepton_stage_order_materialize",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, SourceCount),
        KOKKOS_LAMBDA(std::size_t const source) {
          auto const stage = counts(source).fallback_stage;
          if (stage == kokkos_detail::LeptonNoFallback) return;
          auto const output = kokkos_detail::leptonFallbackOutputOffset(
              stage,
              offsets(source,
                      kokkos_detail::LeptonSelectionFallbackOffset),
              offsets(source,
                      kokkos_detail::LeptonTransportFallbackOffset),
              offsets(source,
                      kokkos_detail::LeptonVertexFallbackOffset),
              offsets(source,
                      kokkos_detail::LeptonFinalStateFallbackOffset),
              totals());
          ordered_sources(output) = source;
        });

    auto host_totals = Kokkos::create_mirror_view(totals);
    auto host_ordered = Kokkos::create_mirror_view(ordered_sources);
    Kokkos::deep_copy(execution, host_totals, totals);
    Kokkos::deep_copy(execution, host_ordered, ordered_sources);
    execution.fence("download lepton stage-major fallback order");
    if (host_totals().values[
            kokkos_detail::LeptonSelectionFallbackOffset] != 4 ||
        host_totals().values[
            kokkos_detail::LeptonTransportFallbackOffset] != 3 ||
        host_totals().values[
            kokkos_detail::LeptonVertexFallbackOffset] != 3 ||
        host_totals().values[
            kokkos_detail::LeptonFinalStateFallbackOffset] != 3)
      throw std::runtime_error(
          "lepton fallback stage totals differ from scalar oracle");
    constexpr std::array<std::size_t, FallbackCount> Expected{
        1, 5, 9, 14, 3, 8, 12, 2, 7, 13, 0, 6, 11};
    for (std::size_t index = 0; index < FallbackCount; ++index)
      if (host_ordered(index) != Expected[index])
        throw std::runtime_error(
            "lepton fallback output is not stage-major and source-stable");
  }

  void runResidentCompositeScanCase(ExecutionSpace const& execution) {
    constexpr std::size_t SourceCount = 129;
    {
      using SourceCounts = kokkos_detail::ResidentPhotonSourceCounts;
      using ScanValue = kokkos_detail::ResidentPhotonScanValue;
      constexpr std::array<std::uint32_t,
                           kokkos_detail::PhotonOffsetCount>
          PerSource{2, 1, 2, 1, 0, 0, 1, 1, 1};
      Kokkos::View<SourceCounts*, Memory> counts(
          "checked_photon_scan_counts", SourceCount);
      Kokkos::View<std::uint64_t*[kokkos_detail::PhotonOffsetCount], Memory>
          offsets("checked_photon_scan_offsets", SourceCount);
      Kokkos::View<ScanValue, Memory> total("checked_photon_scan_total");
      auto host_counts = Kokkos::create_mirror_view(counts);
      for (std::size_t source = 0; source < SourceCount; ++source) {
        auto& source_counts = host_counts(source);
        source_counts = {};
        source_counts.child = 2;
        source_counts.next = 1;
        source_counts.charged = 2;
        source_counts.fallback_stage =
            kokkos_detail::PhotonSelectionFallback;
        source_counts.observation = 1;
        source_counts.step = 1;
        source_counts.record = 1;
      }
      Kokkos::deep_copy(execution, counts, host_counts);
      Kokkos::parallel_scan(
          "c8_test_checked_photon_output_scan",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, SourceCount),
          kokkos_detail::ResidentPhotonScanFunctor<ExecutionSpace>{
              counts, offsets},
          total);
      auto host_offsets = Kokkos::create_mirror_view(offsets);
      auto host_total = Kokkos::create_mirror_view(total);
      Kokkos::deep_copy(execution, host_offsets, offsets);
      Kokkos::deep_copy(execution, host_total, total);
      execution.fence("download checked photon output scan");
      for (std::size_t source = 0; source < SourceCount; ++source)
        for (std::size_t column = 0;
             column < kokkos_detail::PhotonOffsetCount; ++column) {
          auto const expected = static_cast<std::uint64_t>(source) *
                                PerSource[column];
          if (host_offsets(source, column) != expected)
            throw std::runtime_error(
                "resident photon prefix differs from scalar oracle");
        }
      for (std::size_t column = 0;
           column < kokkos_detail::PhotonOffsetCount; ++column) {
        auto const expected = static_cast<std::uint64_t>(SourceCount) *
                              PerSource[column];
        if (host_total().values[column] != expected)
          throw std::runtime_error(
              "resident photon total differs from scalar oracle");
      }

      Kokkos::View<std::uint32_t*, Memory> flags(
          "checked_photon_interaction_flags", SourceCount);
      Kokkos::View<std::size_t*, Memory> sources(
          "checked_photon_interaction_sources", SourceCount);
      Kokkos::View<std::uint64_t, Memory> interaction_count(
          "checked_photon_interaction_count");
      auto host_flags = Kokkos::create_mirror_view(flags);
      std::size_t expected_count = 0;
      for (std::size_t source = 0; source < SourceCount; ++source) {
        host_flags(source) = source % 3 == 1 ? 1U : 0U;
        expected_count += host_flags(source) != 0;
      }
      Kokkos::deep_copy(execution, flags, host_flags);
      Kokkos::parallel_scan(
          "c8_test_checked_photon_interaction_scan",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, SourceCount),
          kokkos_detail::ResidentPhotonCompactionScanFunctor<ExecutionSpace>{
              flags, sources},
          interaction_count);
      auto host_sources = Kokkos::create_mirror_view(sources);
      auto host_interaction_count =
          Kokkos::create_mirror_view(interaction_count);
      Kokkos::deep_copy(execution, host_sources, sources);
      Kokkos::deep_copy(
          execution, host_interaction_count, interaction_count);
      execution.fence("download checked photon interaction scan");
      if (host_interaction_count() != expected_count)
        throw std::runtime_error(
            "checked photon interaction count differs");
      std::size_t compact = 0;
      for (std::size_t source = 0; source < SourceCount; ++source)
        if (host_flags(source) != 0 && host_sources(compact++) != source)
          throw std::runtime_error(
              "checked photon interaction order differs");
    }

    {
      using SourceCounts = kokkos_detail::ResidentLeptonSourceCounts;
      using ScanValue = kokkos_detail::ResidentLeptonScanValue;
      constexpr std::array<std::uint32_t,
                           kokkos_detail::LeptonOffsetCount>
          PerSource{3, 3, 2, 1, 0, 0, 0, 1, 1, 1, 1};
      Kokkos::View<SourceCounts*, Memory> counts(
          "checked_lepton_scan_counts", SourceCount);
      Kokkos::View<std::uint64_t*[kokkos_detail::LeptonOffsetCount], Memory>
          offsets("checked_lepton_scan_offsets", SourceCount);
      Kokkos::View<ScanValue, Memory> total("checked_lepton_scan_total");
      auto host_counts = Kokkos::create_mirror_view(counts);
      for (std::size_t source = 0; source < SourceCount; ++source) {
        auto& source_counts = host_counts(source);
        source_counts = {};
        source_counts.child = 3;
        source_counts.next = 3;
        source_counts.photon = 2;
        source_counts.fallback_stage =
            kokkos_detail::LeptonSelectionFallback;
        source_counts.observation = 1;
        source_counts.decay = 1;
        source_counts.step = 1;
        source_counts.record = 1;
      }
      Kokkos::deep_copy(execution, counts, host_counts);
      Kokkos::parallel_scan(
          "c8_test_checked_lepton_output_scan",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, SourceCount),
          kokkos_detail::ResidentLeptonScanFunctor<ExecutionSpace>{
              counts, offsets},
          total);
      auto host_offsets = Kokkos::create_mirror_view(offsets);
      auto host_total = Kokkos::create_mirror_view(total);
      Kokkos::deep_copy(execution, host_offsets, offsets);
      Kokkos::deep_copy(execution, host_total, total);
      execution.fence("download checked lepton output scan");
      for (std::size_t source = 0; source < SourceCount; ++source)
        for (std::size_t column = 0;
             column < kokkos_detail::LeptonOffsetCount; ++column) {
          auto const expected = static_cast<std::uint64_t>(source) *
                                PerSource[column];
          if (host_offsets(source, column) != expected)
            throw std::runtime_error(
                "resident lepton prefix differs from scalar oracle");
        }
      for (std::size_t column = 0;
           column < kokkos_detail::LeptonOffsetCount; ++column) {
        auto const expected = static_cast<std::uint64_t>(SourceCount) *
                              PerSource[column];
        if (host_total().values[column] != expected)
          throw std::runtime_error(
              "resident lepton total differs from scalar oracle");
      }

      Kokkos::View<std::uint32_t*, Memory> flags(
          "checked_lepton_interaction_flags", SourceCount);
      Kokkos::View<std::size_t*, Memory> sources(
          "checked_lepton_interaction_sources", SourceCount);
      Kokkos::View<std::uint64_t, Memory> interaction_count(
          "checked_lepton_interaction_count");
      auto host_flags = Kokkos::create_mirror_view(flags);
      std::size_t expected_count = 0;
      for (std::size_t source = 0; source < SourceCount; ++source) {
        host_flags(source) = source % 4 == 2 ? 1U : 0U;
        expected_count += host_flags(source) != 0;
      }
      Kokkos::deep_copy(execution, flags, host_flags);
      Kokkos::parallel_scan(
          "c8_test_checked_lepton_interaction_scan",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, SourceCount),
          kokkos_detail::ResidentLeptonInteractionScanFunctor<ExecutionSpace>{
              flags, sources},
          interaction_count);
      auto host_sources = Kokkos::create_mirror_view(sources);
      auto host_interaction_count =
          Kokkos::create_mirror_view(interaction_count);
      Kokkos::deep_copy(execution, host_sources, sources);
      Kokkos::deep_copy(
          execution, host_interaction_count, interaction_count);
      execution.fence("download checked lepton interaction scan");
      if (host_interaction_count() != expected_count)
        throw std::runtime_error(
            "checked lepton interaction count differs");
      std::size_t compact = 0;
      for (std::size_t source = 0; source < SourceCount; ++source)
        if (host_flags(source) != 0 && host_sources(compact++) != source)
          throw std::runtime_error(
              "checked lepton interaction order differs");
    }
  }

} // namespace

int main(int argc, char** argv) {
  try {
    Kokkos::ScopeGuard guard(argc, argv);
    ExecutionSpace execution{};
    TestStorage storage(execution);
    runSingleWavefrontCases(storage, execution);
    runCumulativeCase(storage, execution);
    runPhotonPackedStatisticsCases(storage, execution);
    runSignedAtomicFetchAddCase(execution);
    runFrontControlAliasCase(execution);
    runPhotonFallbackStageOrderingCase(execution);
    runLeptonFallbackStageOrderingCase(execution);
    runResidentCompositeScanCase(execution);
    std::cout << "Kokkos fused profile/statistics Moliere counters passed\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "testKokkosFusedProfileStatistics: " << error.what()
              << '\n';
    return 1;
  }
}
