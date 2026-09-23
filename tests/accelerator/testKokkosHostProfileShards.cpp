/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
// Real Kokkos OpenMP reducers and the production shared profile equations.
// No GPU initialization, approximate tolerance, or stochastic shower fixture.
#include <corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace em = corsika::gpu::em;
namespace detail = corsika::accelerator::em::detail;
namespace kd = corsika::accelerator::em::kokkos_detail;
using Ex = Kokkos::OpenMP;
using Counter = em::detail::DeviceProfileCounters;
using Record = em::LeptonTransportRecord;

struct CrossingSegment { double start, end, weight; int pid; };
struct CrossingAccumulator {
  Kokkos::View<CrossingSegment*, Kokkos::HostSpace> input;
  em::detail::DeviceProfileAccumulator accumulator;
  void operator()(std::size_t i) const {
    auto const s = input(i);
    detail::accumulateParticleProfile<kd::KokkosProfileAtomicOperations>(
        accumulator, s.pid, s.start, s.end, s.weight);
  }
};

void require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
template<class Exception, class F> void rejects(F&& f) {
  try { f(); } catch (Exception const&) { return; }
  throw std::runtime_error("Expected rejection did not occur");
}
void sameCounters(Counter const& a, Counter const& b) {
#define CHECK(f) require(a.f == b.f, "profile counter changed: " #f)
  CHECK(steps); CHECK(deposited_steps); CHECK(photon_cuts);
  for (unsigned i = 0; i < 8; ++i) CHECK(lepton_limits[i]);
  CHECK(moliere_trials); CHECK(moliere_deflections); CHECK(moliere_zero_deflections);
  CHECK(moliere_newton_iterations); CHECK(moliere_max_newton_iterations);
  CHECK(thinning_hillas_vertices); CHECK(thinning_statistical_vertices);
  CHECK(thinning_particles_discarded); CHECK(fixed_point_overflows); CHECK(invalid_records);
  CHECK(weighted_medium_rest_mass_input); CHECK(weighted_cut_rest_mass_energy);
  CHECK(weighted_observed_total_energy); CHECK(weighted_escaped_total_energy);
  CHECK(weighted_unwritten_photoelectric_binding_energy);
  CHECK(weighted_observation_cut_overlap_energy); CHECK(weighted_mass_convention_correction);
#undef CHECK
}

void sameOutput(em::GpuProfileResult const& a, em::GpuProfileResult const& b) {
#define CHECK(f) require(a.f == b.f, "decoded profile changed: " #f)
  CHECK(photons); CHECK(electrons); CHECK(positrons); CHECK(muons_minus);
  CHECK(muons_plus); CHECK(muon_parent_productions); CHECK(energy_loss_GeV);
  CHECK(muon_energy_loss_GeV); CHECK(weighted_deposited_energy_GeV);
  CHECK(steps); CHECK(deposited_steps); CHECK(particle_cuts);
  CHECK(fixed_point_overflows); CHECK(invalid_records);
  CHECK(weighted_medium_rest_mass_input_GeV); CHECK(weighted_cut_rest_mass_energy_GeV);
  CHECK(weighted_observed_total_energy_GeV); CHECK(weighted_escaped_total_energy_GeV);
  CHECK(weighted_unwritten_photoelectric_binding_energy_GeV);
  CHECK(weighted_observation_cut_overlap_energy_GeV); CHECK(weighted_mass_convention_correction_GeV);
#undef CHECK
}

struct Ledger {
  kd::KokkosProfileAccumulator<Ex> profile;
  Kokkos::View<std::uint64_t*, Kokkos::HostSpace> statistics{
      "sharded-profile-test-statistics", kd::LeptonCallStatisticCount};
  Ledger(em::GpuEmConfig::ProfileProjection const& config, Ex const& ex) {
    profile.initialize(config, ex);
    Kokkos::deep_copy(ex, statistics, std::uint64_t{0});
  }
};

em::GpuEmConfig::ProfileProjection config() {
  em::GpuEmConfig::ProfileProjection c;
  c.enabled = c.accumulate_on_device = true;
  c.output_bin_count = 128; c.output_bin_width_g_per_cm2 = 10.;
  c.energy_loss_threshold_g_per_cm2 = .01;
  c.fixed_point_weight_limit = 1.e12; c.fixed_point_energy_limit_GeV = 1.e14;
  c.axis_direction[2] = 1.; c.axis_step_length_m = 1280.;
  c.axis_grammage_g_per_cm2 = {0., 1280.};
  return c;
}

Record record(std::size_t i) {
  Record s{};
  s.start.pid = i % 4 == 0 ? 13 : i % 4 == 1 ? -13 : i % 4 == 2 ? 11 : -11;
  s.start.weight = .5 + .25 * (i % 5); s.start.energy_GeV = 1. + .01 * (i % 20);
  s.start.history_id = i + 1;
  s.start.position_m[2] = 400. + .03125 * (i % 512);
  s.end = s.start; s.end.position_m[2] += .25 + .0625 * (i % 8);
  s.end.energy_GeV -= .001; s.continuous_deposited_energy_GeV = .001;
  s.limit = static_cast<em::LeptonTransportLimit>(i % 8);
  if (s.limit == em::LeptonTransportLimit::ParticleCut) s.cut_deposited_energy_GeV = .01;
  s.observation_surface_reached_before_cut = (i % 16 == 0);
  s.traversed_grammage_g_per_cm2 = i % 3 ? .1 : 0.;
  s.multiple_scattering_applied = i % 2;
  s.multiple_scattering_status = static_cast<unsigned>(em::MoliereStatus::NoDeflection);
  s.multiple_scattering_iterations = i % 12;
  return s;
}

void exercise(int threads, std::size_t limited_shards = 0,
              corsika::ProfileCrossingMode mode = corsika::ProfileCrossingMode::Both) {
  Ex ex;
  require(ex.concurrency() == threads, "OpenMP thread count mismatch");
  auto c = config();
  c.crossing_mode = mode;
  Ledger reference(c, ex), sharded(c, ex), standalone(c, ex), plain(c, ex);
  auto const budget = limited_shards
      ? limited_shards * detail::HostProfileShards::bytesPerShard(c.output_bin_count)
      : detail::HostProfileShards::MaximumBytes;
  sharded.profile.enableCooperativeHostShards(threads, budget);
  standalone.profile.enableHostShards(threads, budget);
  auto const bytes = sharded.profile.deviceBytes();
  auto const* allocation = sharded.profile.hostShards().entries();
  auto const standalone_bytes = standalone.profile.deviceBytes();
  auto const* standalone_allocation = standalone.profile.hostShards().entries();
  require(sharded.profile.hostShards().count() >= static_cast<unsigned>(threads) ||
          threads == 1 || limited_shards != 0, "not enough shards with generous budget");
  require(sharded.profile.hostShards().bytes() <= detail::HostProfileShards::MaximumBytes,
          "host ledger exceeded independent cap");
  double grammage[]{0., 1280.};
  em::detail::DeviceProfileProjection projection{};
  projection.axis_direction[2] = 1.; projection.axis_step_length_m = 1280.;
  projection.axis_grammage_g_per_cm2 = grammage; projection.axis_support_count = 2;
  for (unsigned event = 0; event < 32; ++event) {
    if (event) {
      for (auto* l : {&reference, &sharded, &standalone, &plain}) {
        l->profile.reset(c.fixed_point_weight_limit * (event + 1),
                         c.fixed_point_energy_limit_GeV * (event + 1), ex);
        Kokkos::deep_copy(ex, l->statistics, std::uint64_t{0});
      }
    }
    // Includes empty front, non-contiguous source holes and mixed PID/cut cases.
    std::size_t const count = event == 0 ? 0 : event % 3 == 0 ? 4097 : 1024;
    Kokkos::View<Record*, Kokkos::HostSpace> transports("test-transports", count);
    Kokkos::View<kd::ResidentLeptonSourceCounts*, Kokkos::HostSpace> counts("test-counts", count);
    Kokkos::View<Record*, Kokkos::HostSpace> steps("test-packed-steps", count);
    std::size_t packed = 0;
    for (std::size_t i = 0; i < count; ++i) {
      transports(i) = record(i + event);
      if (i % 5 != 1 && i % 7 != 3) {
        counts(i).step = 1;
        steps(packed++) = transports(i);
      }
    }
    for (auto* l : {&reference, &sharded, &standalone, &plain}) {
      // A canonical contribution models final-state/photon writers which do
      // not use the new shards and must not be omitted or counted twice.
      l->profile.deviceView().counters->thinning_hillas_vertices = 7;
      for (unsigned wave = 0; wave < 3; ++wave)
        kd::enqueueLeptonProfileStatistics(
            Kokkos::RangePolicy<Ex>(ex, 0, count),
            kd::ResidentLeptonTransportStatisticsProfileFunctor<Ex>{
                {transports, counts, l->statistics}, projection,
                l->profile.deviceView(), steps, packed}, l->profile.hostShards());
    }
    auto a = reference.profile.downloadFixed("shower", detail::CooperativeEndpoint::OpenMP, ex);
    auto b = sharded.profile.downloadFixed("shower", detail::CooperativeEndpoint::OpenMP, ex);
    require(a.histograms == b.histograms, "sharded profile histogram differs");
    sameCounters(a.counters, b.counters);
    auto const original_output = plain.profile.download(ex);
    sameOutput(original_output, detail::decodeFixedProfile(a));
    sameOutput(original_output, standalone.profile.download(ex));
    rejects<std::logic_error>([&] { standalone.profile.download(ex); });
    rejects<std::logic_error>([&] { standalone.profile.downloadFixed("shower", detail::CooperativeEndpoint::OpenMP, ex); });
    rejects<std::logic_error>([&] { sharded.profile.download(ex); });
    for (std::size_t i = 0; i < reference.statistics.extent(0); ++i)
      require(reference.statistics(i) == sharded.statistics(i), "reduction statistics differ");
    require(b.counters.steps == 3 * packed && b.counters.thinning_hillas_vertices == 7,
            "lost/duplicate contributions");
    rejects<std::logic_error>([&] { sharded.profile.downloadFixed("shower", detail::CooperativeEndpoint::OpenMP, ex); });
    require(sharded.profile.deviceBytes() == bytes &&
            sharded.profile.hostShards().entries() == allocation, "per-event allocation growth");
    require(standalone.profile.deviceBytes() == standalone_bytes &&
            standalone.profile.hostShards().entries() == standalone_allocation,
            "standalone per-event allocation growth");
  }
  std::cout << "exact_32_showers=true threads=" << threads
            << " shards=" << sharded.profile.hostShards().count()
            << " bytes=" << sharded.profile.hostShards().bytes() << '\n';
}

void gates() {
  Ex ex; auto c = config(); Ledger ledger(c, ex);
  auto base = ledger.profile.deviceView();
  auto const per = detail::HostProfileShards::bytesPerShard(base.bins);
  {
    // Actual inclined-air profile: budget fits 130, but not 256 replicas.
    // Rounding down to 128 would introduce two unnecessary contended pairs.
    auto inclined = base; inclined.bins = 1263;
    detail::HostProfileShards shards;
    shards.configure(inclined, 130, detail::HostProfileShards::MaximumBytes);
    require(shards.count() == 130 && shards.bytes() <= detail::HostProfileShards::MaximumBytes,
            "unnecessary shard collisions on a non-power-of-two server");
  }
  for (auto workers : {0U, 1U, 2U, 20U, 130U, 256U, 10000U}) {
    for (auto budget : {std::size_t{0}, per, 3 * per, detail::HostProfileShards::MaximumBytes}) {
      detail::HostProfileShards shards; shards.configure(base, workers, budget);
      require(shards.bytes() <= budget && shards.bytes() <= detail::HostProfileShards::MaximumBytes &&
              shards.count() <= 256, "unbounded shard storage");
      rejects<std::logic_error>([&] { shards.configure(base, workers, budget); });
    }
  }
  rejects<std::length_error>([] { detail::HostProfileShards::bytesPerShard(std::numeric_limits<std::size_t>::max()); });
  rejects<std::length_error>([] { detail::HostProfileShards::bytesPerShard(0); });
  for (int failure = 0; failure < 6; ++failure) {
    Ledger value(c, ex); value.profile.enableCooperativeHostShards(4, 4 * per);
    auto const* shards = value.profile.hostShards().entries();
    if (failure == 0) { shards[0].accumulator.photons[0] = std::numeric_limits<long long>::max(); shards[1].accumulator.photons[0] = 1; }
    if (failure == 1) { shards[0].accumulator.photons[0] = std::numeric_limits<long long>::min(); shards[1].accumulator.photons[0] = -1; }
    if (failure == 2) { shards[0].accumulator.counters->steps = std::numeric_limits<unsigned long long>::max(); shards[1].accumulator.counters->steps = 1; }
    if (failure == 3) shards[0].accumulator.counters->invalid_records = 1;
    if (failure == 4) shards[0].accumulator.counters->fixed_point_overflows = 1;
    if (failure == 5) { shards[0].accumulator.counters->weighted_mass_convention_correction = std::numeric_limits<long long>::max(); shards[1].accumulator.counters->weighted_mass_convention_correction = 1; }
    rejects<std::runtime_error>([&] {
      if (failure % 2) value.profile.download(ex);
      else value.profile.downloadFixed("shower", detail::CooperativeEndpoint::OpenMP, ex);
    });
    rejects<std::logic_error>([&] { value.profile.download(ex); });
    rejects<std::logic_error>([&] { value.profile.downloadFixed("shower", detail::CooperativeEndpoint::OpenMP, ex); });
    value.profile.reset(c.fixed_point_weight_limit, c.fixed_point_energy_limit_GeV, ex);
    auto empty = value.profile.downloadFixed("next", detail::CooperativeEndpoint::OpenMP, ex);
    require(empty.counters.steps == 0 && empty.counters.invalid_records == 0 &&
            std::all_of(empty.histograms.begin(), empty.histograms.end(), [](auto x) { return x == 0; }),
            "failed event contaminated next event");
  }
  // Check final merge transactionality and identity/scale rejection directly.
  auto original = ledger.profile.downloadFixed("check", detail::CooperativeEndpoint::OpenMP, ex);
  for (int failure = 0; failure < 4; ++failure) {
    detail::HostProfileShards shards; shards.configure(base, 4, 4 * per);
    auto snapshot = original;
    if (failure == 0) {
      snapshot.histograms[0] = 9;
      shards.entries()[0].accumulator.photons[0] = std::numeric_limits<long long>::max();
    }
    if (failure == 1) snapshot.weight_units *= 2;
    if (failure == 2) snapshot.config.output_bin_width_g_per_cm2 *= 2;
    if (failure == 3) snapshot.histograms.pop_back();
    auto const before = snapshot;
    rejects<std::exception>([&] { shards.mergeInto(snapshot); });
    require(snapshot.histograms == before.histograms, "failed merge partially wrote output");
    sameCounters(snapshot.counters, before.counters);
    rejects<std::logic_error>([&] { shards.mergeInto(snapshot); });
  }
}

// Real OpenMP atomic accumulation compared to a brute-force plane oracle.
// The oracle deliberately has no floor/ceil or shared bin-helper calls.
void bidirectionalCrossings() {
  constexpr std::size_t bins = 32, samples = 1000000;
  Kokkos::View<CrossingSegment*, Kokkos::HostSpace> input("crossing-input", samples);
  Kokkos::View<long long*, Kokkos::HostSpace> counts("crossing-counts", 5 * bins);
  Kokkos::View<Counter*, Kokkos::HostSpace> counters("crossing-counters", 1);
  std::vector<long long> expected(5 * bins);
  int const pids[]{22, 11, -11, 13, -13};
  for (std::size_t i = 0; i < samples; ++i) {
    double a = static_cast<double>((i * 7919) % 201) * .25 - 8.;
    double b = static_cast<double>((i * 1543 + 3) % 201) * .25 - 8.;
    auto const weight = .5 + .25 * (i % 5);
    input(i) = {a * 5., b * 5., weight, pids[i % 5]};
    for (std::size_t j = 0; j < bins; ++j)
      if (std::min(a, b) < j && j <= std::max(a, b))
        expected[(i % 5) * bins + j] += static_cast<long long>(weight * 4.);
  }
  em::detail::DeviceProfileAccumulator v{};
  v.bins = bins; v.bin_width_g_per_cm2 = 5.; v.weight_scale = 4.;
  v.photons = counts.data(); v.electrons = counts.data() + bins;
  v.positrons = counts.data() + 2 * bins;
  v.muons_minus = counts.data() + 3 * bins;
  v.muons_plus = counts.data() + 4 * bins; v.counters = counters.data();
  Kokkos::parallel_for("bidirectional-profile-million", Kokkos::RangePolicy<Ex>(0, samples),
      CrossingAccumulator{input, v});
  Ex{}.fence();
  for (std::size_t i = 0; i < expected.size(); ++i)
    require(counts(i) == expected[i], "bidirectional plane oracle differs");
  require(counters(0).invalid_records == 0 && counters(0).fixed_point_overflows == 0,
          "invalid crossing accumulation");
  auto range = corsika::detail::longitudinalCrossingBins;
  for (double bad : {std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN()}) {
    require(range(bad, 5., bins).end == 0 && range(5., bad, bins).end == 0,
            "non-finite crossing was accepted");
  }
  require(range(1., 2., 0).end == 0, "empty profile accepted crossings");
  for (double edge : {0., 1., 5., 31., 32.}) {
    double const points[]{std::nextafter(edge, -INFINITY), edge,
                          std::nextafter(edge, INFINITY)};
    for (double a : points) for (double b : points) {
      auto const r = range(a, b, bins);
      for (std::size_t j = 0; j < bins; ++j)
        require((r.begin <= j && j < r.end) ==
                    (std::min(a, b) < j && j <= std::max(a, b)),
                "nextafter boundary ownership differs");
    }
  }
  auto const backwards = range(601. / 5., 589. / 5., 200);
  require(backwards.begin == 118 && backwards.end == 121,
          "regression: missed backwards photon at X=590,595,600");
  std::cout << "bidirectional_crossings_million=PASS\n";
}

int main(int argc, char** argv) try {
  int threads = argc == 2 ? std::stoi(argv[1]) : 4;
  require(threads >= 1 && threads <= 256, "expected 1..256 OpenMP threads");
#ifdef KOKKOS_ENABLE_CUDA
  require(!Kokkos::Cuda::impl_is_initialized(), "test must not initialize CUDA");
#endif
  Kokkos::OpenMP::impl_initialize(Kokkos::InitializationSettings().set_num_threads(threads));
  try { exercise(threads); exercise(threads, 2); exercise(threads, 3); gates();
        exercise(threads, 0, corsika::ProfileCrossingMode::Forward);
        exercise(threads, 0, corsika::ProfileCrossingMode::OriginalC8);
        bidirectionalCrossings(); }
  catch (...) { Kokkos::OpenMP::impl_finalize(); throw; }
  Kokkos::OpenMP::impl_finalize();
#ifdef KOKKOS_ENABLE_CUDA
  require(!Kokkos::Cuda::impl_is_initialized(), "host profile test initialized CUDA");
#endif
  std::cout << "profile_shard_gates=PASS (no GPU initialized)\n";
  return 0;
} catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
