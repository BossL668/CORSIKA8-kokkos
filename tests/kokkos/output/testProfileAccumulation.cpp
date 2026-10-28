/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <tests/kokkos/output/ProfileAccumulationChecks.hpp>
#include <limits>

using namespace profile_test;

namespace {
  void ensure_runtime() { static current::Runtime runtime; }

  std::vector<Case> scenarios() {
    std::vector<Case> q;
    for (double start : {-2., 0., .25, 1., 4., 8., 10.})
      for (double end : {-1., 0., 1., 3.5, 8., 12.})
        for (double weight : {0., .5, 1., 1.5}) {
          for (int pid : {22, 11, -11, 13, -13, 999})
            q.push_back({0, pid, 0, 0, start, end, weight, 2.});
          q.push_back({1, 0, 0, 0, start, end, weight, 2.});
          q.push_back({2, 0, 0, 1, start, end, weight, -.125});
          q.push_back({4, 0, 0, 0, start, end, weight, 2.});
        }
    for (int kind : {5, 6})
      for (int pid : {11, -11, 13, -13})
        for (int limit = 0; limit < 9; ++limit)
          for (int variant = 0; variant < 7; ++variant)
            q.push_back({kind, pid, limit, variant, 1., 5.5, 1.5, 2.});
    for (int kind : {7, 8})
      for (int process : {current::ComptonProcessId, current::PhotoelectricProcessId,
                          current::IonizationProcessId, current::AnnihilationProcessId,
                          current::ElectronPairProcessId, current::BremsProcessId})
        for (int multiplicity = 0; multiplicity <= 4; ++multiplicity)
          for (int variant = 0; variant <= 5; ++variant)
            for (double fraction : {-.1, 0., .75, 1., 1.1})
              q.push_back({kind, process, multiplicity, variant, fraction, 2., 1.5, 4.});
    for (double value :
         {-HUGE_VAL, -2., -.5, -0., 0., .5, 2., HUGE_VAL, static_cast<double>(NAN)}) {
      q.push_back({3, 0, 0, 0, 0., 0., 1., value});
      q.push_back({3, 0, 0, 1, 0., 0., 1., value});
      q.push_back({4, 0, 0, 0, 0., value, 1., 0.});
    }
    for (double value :
         {-HUGE_VAL, -0x1p63, -2.5, -1.5, -.5, 0., .5, 1.5, 2.5,
          std::nextafter(0x1p63, 0.), 0x1p63, HUGE_VAL, static_cast<double>(NAN)})
      q.push_back({9, 0, 0, 0, 0., 0., 1., value});
    return q;
  }

  void run_device(std::vector<Case> const& cases, bool aggregate, int mode,
                  double threshold) {
    using execution_space_type = transport_test::execution_space_type;
    using memory_space_type = transport_test::memory_space_type;
    auto inputs = transport_test::upload<Case>(cases);
    auto depths = transport_test::upload<double>({0., 4., 8., 12.});
    std::size_t const count = cases.size(), slots = aggregate ? 1 : count;
    Kokkos::View<long long**, Kokkos::LayoutRight, memory_space_type> actual(
        "profile values", slots, 64);
    Kokkos::View<long long**, Kokkos::LayoutRight, memory_space_type> expected(
        "frozen values", slots, 64);
    Kokkos::View<current::ProfileCounters*, memory_space_type> counters(
        "profile counters", slots);
    Kokkos::View<reference::DeviceProfileCounters*, memory_space_type> oldCounters(
        "frozen counters", slots);
    current::ProfileProjectionView p{{0., 0., 0.}, {0., 0., 1.}, 4., depths.data(), 4};
    reference::DeviceProfileProjection oldP{
        {0., 0., 0.}, {0., 0., 1.}, 4., depths.data(), 4};
    std::vector<long long> forward;
    for (bool reverse : {false, true}) {
      Kokkos::deep_copy(actual, 0LL);
      Kokkos::deep_copy(expected, 0LL);
      Kokkos::deep_copy(counters, current::ProfileCounters{});
      Kokkos::deep_copy(oldCounters, reference::DeviceProfileCounters{});
      Kokkos::parallel_for(
          "production profile accumulation comparison",
          Kokkos::RangePolicy<execution_space_type>(0, count),
          KOKKOS_LAMBDA(std::size_t index) {
            auto const i = reverse ? count - 1 - index : index;
            auto const slot = aggregate ? 0 : i;
            auto a = view(&actual(slot, 0), &counters(slot), mode, threshold);
            auto b = old_view(&expected(slot, 0), &oldCounters(slot), mode, threshold);
            current_driver::apply(inputs(i), a, p);
            reference_driver::apply(inputs(i), b, oldP);
          });
      execution_space_type().fence();
      auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, actual);
      auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
      auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, counters);
      auto d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, oldCounters);
      for (std::size_t row = 0; row < slots; ++row) {
        compare(c(row), d(row));
        for (std::size_t col = 0; col < 64; ++col) {
          REQUIRE(a(row, col) == b(row, col));
          if (reverse)
            REQUIRE(a(row, col) == forward[row * 64 + col]);
          else
            forward.push_back(a(row, col));
        }
      }
    }
  }

  template <long long Previous>
  struct FixedPreviousAtomic {
    CORSIKA_HOST_DEVICE static long long add(long long*, long long) { return Previous; }
  };
} // namespace

TEST_CASE("Production profile and closure accumulation on host and device",
          "[profile_accumulation]") {
  ensure_runtime();
#ifdef CORSIKA_EXPECT_CUDA
  STATIC_REQUIRE(std::is_same_v<transport_test::execution_space_type, Kokkos::Cuda>);
#endif
#ifdef CORSIKA_EXPECT_OPENMP
  STATIC_REQUIRE(std::is_same_v<transport_test::execution_space_type, Kokkos::OpenMP>);
#endif
  auto const inputs = scenarios();
  for (int mode : {0, 1, 2}) {
    for (auto const& q : inputs) {
      long long values[64]{}, oldValues[64]{};
      current::ProfileCounters c{};
      reference::DeviceProfileCounters d{};
      double depth[]{0., 4., 8., 12.};
      current::ProfileProjectionView p{{0., 0., 0.}, {0., 0., 1.}, 4., depth, 4};
      reference::DeviceProfileProjection oldP{{0., 0., 0.}, {0., 0., 1.}, 4., depth, 4};
      current_driver::apply(q, view(values, &c, mode, .25), p);
      reference_driver::apply(q, old_view(oldValues, &d, mode, .25), oldP);
      compare(c, d);
      for (int i = 0; i < 64; ++i) REQUIRE(values[i] == oldValues[i]);
    }
    run_device(inputs, false, mode, .25);
    // Avoid artificial integer overflow in concurrent aggregation. Extreme
    // fixed-point inputs are checked independently above and in the next test.
    auto bounded = inputs;
    bounded.erase(std::remove_if(bounded.begin(), bounded.end(),
                                 [](Case const& q) { return q.kind_ == 9; }),
                  bounded.end());
    run_device(bounded, true, mode, .25);
  }
  run_device({}, false, 0, .25);
  run_device({{2, 0, 0, 0, 0., 0., 1., 2.}, {2, 0, 0, 1, 0., 0., 1., -.125}}, false, 0,
             0.);
  std::ofstream("accumulation-counts.txt")
      << "inputs " << inputs.size()
      << "\ncrossing_modes 3\nreverse_order 2\nshared_atomic_histograms 8\n";
}

TEST_CASE("Profile known totals preserve energy conventions and diagnostic limits",
          "[profile_accumulation_known]") {
  ensure_runtime();
  long long values[64]{};
  current::ProfileCounters c{};
  auto a = view(values, &c, 0, .25);
  using Atomic = current::ProfileAtomicOperations;
  current::accumulate_particle_profile<Atomic>(a, 11, 1., 3., 1.5);
  REQUIRE(values[8 + 1] == 0);
  REQUIRE(values[8 + 2] == 1536);
  REQUIRE(values[8 + 3] == 1536);
  current::accumulate_energy_profile<Atomic>(a, a.energyLoss_, 2., 6., 8.);
  for (int i = 2; i < 6; ++i) REQUIRE(values[48 + i] == 2048);
  REQUIRE(values[48 + 6] == 0);
  a.energyLossThresholdGPerCm2_ = 0.;
  current::accumulate_point_energy_profile<Atomic>(a, a.energyLoss_, 0., -.125, true);
  REQUIRE(values[48] == -128);
  REQUIRE(c.depositedSteps_ == 2);
  REQUIRE(current::profile_histogram_bytes(8) == sizeof(values));
  REQUIRE(current::profile_histogram_bytes(0) == 0);
  double depth[]{0., 4., 8., 12.};
  current::ProfileProjectionView p{{0., 0., 0.}, {0., 0., 1.}, 4., depth, 4};
  current::ProfileCounters boundary{};
  long long bins[64]{};
  current_driver::apply({6, 11, 8, 0, 1., 3., 1., 2.}, view(bins, &boundary, 0, .25), p);
  REQUIRE(boundary.steps_ == 1);
  REQUIRE(boundary.invalidRecords_ == 0);
  REQUIRE(boundary.leptonMaterialBoundaries_ == 1);
  REQUIRE(bins[8 + 2] == 1024);
  current::ProfileCounters photo{};
  long long photoBins[64]{};
  current_driver::apply({7, current::PhotoelectricProcessId, 1, 0, .75, 2., 1., 4.},
                        view(photoBins, &photo, 0, .25), p);
  REQUIRE(photo.weightedUnwrittenPhotoelectricBindingEnergy_ == 1024);
  REQUIRE(photo.depositedSteps_ == 0);
  for (int i = 48; i < 64; ++i) REQUIRE(photoBins[i] == 0);
  REQUIRE(photo.thinningParticlesDiscarded_ == 0);
  current::ProfileCounters cut{};
  long long cutBins[64]{};
  // A native signed cut contributes negative kinetic energy to the muon
  // histogram and to its explicit closure counter, without changing the cut.
  current::LeptonTransportRecord r{};
  r.start_.pid_ = 13;
  r.start_.reserved_ = current::NativeMuonPairCutFlag;
  r.start_.generation_ = 1;
  r.start_.energyGeV_ = .1;
  r.start_.weight_ = 1.;
  r.end_ = r.start_;
  r.end_.positionM_[2] = 2.;
  r.limit_ = current::LeptonTransportLimit::ParticleCut;
  r.cutDepositedEnergyGeV_ = .1 - current::TransportMuonMassGeV;
  current::accumulate_lepton_profile_step<Atomic>(p, view(cutBins, &cut, 0, .25), r);
  REQUIRE(cut.invalidRecords_ == 0);
  REQUIRE(cut.weightedNativeMuonPairCutEnergy_ < 0);
  REQUIRE(cut.weightedNativeMuonPairCutEnergy_ == cutBins[56 + 2]);
  REQUIRE(cut.steps_ == 1);
  current::LeptonTransportRecord records[3]{};
  records[0].inputIndex_ = 0;
  records[1].inputIndex_ = 2;
  records[2].inputIndex_ = 4;
  REQUIRE(current::find_profile_transport_record(records, 3, 2) == records + 1);
  REQUIRE(current::find_profile_transport_record(records, 3, 3) == nullptr);
  REQUIRE(current::find_profile_transport_record(records, 0, 0) == nullptr);
}

TEST_CASE("Profile fixed-point rounding and overflow reporting remain production rules",
          "[profile_accumulation_fixed_point]") {
  using Atomic = current::ProfileAtomicOperations;
  REQUIRE(current::profile_round(.5) == 0);
  REQUIRE(current::profile_round(1.5) == 2);
  REQUIRE(current::profile_round(-.5) == 0);
  REQUIRE(current::profile_round(-1.5) == -2);
  long long target = 7;
  current::ProfileCounters counters{};
  for (double invalid : {static_cast<double>(NAN), HUGE_VAL, -HUGE_VAL, 0x1p63, -0x1p63})
    current::add_profile_fixed_point<Atomic>(&target, invalid, 1., &counters);
  REQUIRE(target == 7);
  REQUIRE(counters.fixedPointOverflows_ == 5);
  using High = FixedPreviousAtomic<std::numeric_limits<long long>::max()>;
  using Low = FixedPreviousAtomic<std::numeric_limits<long long>::min()>;
  REQUIRE_FALSE(current::checked_profile_atomic_add<High>(&target, 1));
  REQUIRE_FALSE(current::checked_profile_atomic_add<Low>(&target, -1));
  REQUIRE(current::checked_profile_atomic_add<High>(&target, -1));
  REQUIRE(current::checked_profile_atomic_add<Low>(&target, 1));
}
