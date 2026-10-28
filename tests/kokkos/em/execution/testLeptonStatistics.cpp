/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/em/execution/LeptonStatistics.hpp>
#include <corsika/kokkos/output/ProfileAccumulatorStorage.hpp>
#include <corsika/kokkos/output/ProfileProjectionStorage.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <vector>

namespace {
  namespace cp = corsika::kokkos;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = execution_space_type::memory_space;
  using policy_type = Kokkos::RangePolicy<execution_space_type>;
#ifdef CORSIKA_EXPECT_CUDA
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  void ensure_runtime() { static cp::Runtime runtime; }
  template <typename T>
  auto upload(std::vector<T> const& input) {
    Kokkos::View<T*, memory_space_type> view("statistics inputs", input.size());
    auto host = Kokkos::create_mirror_view(view);
    for (std::size_t i = 0; i < input.size(); ++i) host(i) = input[i];
    Kokkos::deep_copy(view, host);
    return view;
  }
  template <typename T>
  auto download(T const& view) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  }
  cp::ProfileProjectionConfig config() {
    cp::ProfileProjectionConfig c;
    c.enabled_ = c.accumulateOnDevice_ = true;
    c.axisStartPositionM_[2] = 10.;
    c.axisDirection_[2] = -1.;
    c.axisStepLengthM_ = 2.;
    c.axisGrammageGPerCm2_ = {0., 2., 4., 6., 8., 10.};
    c.outputBinCount_ = 10;
    c.outputBinWidthGPerCm2_ = 1.;
    c.fixedPointWeightLimit_ = c.fixedPointEnergyLimitGeV_ = 0x1p48;
    return c;
  }
  void compare(cp::FixedProfileSnapshot const& a, cp::FixedProfileSnapshot const& b) {
    REQUIRE(a.histograms_ == b.histograms_);
    REQUIRE(a.weightUnits_ == b.weightUnits_);
    REQUIRE(a.energyUnits_ == b.energyUnits_);
#define CHECK_COUNTER(name) REQUIRE(a.counters_.name == b.counters_.name)
    CHECK_COUNTER(steps_);
    CHECK_COUNTER(leptonMaterialBoundaries_);
    CHECK_COUNTER(depositedSteps_);
    CHECK_COUNTER(photonCuts_);
    for (unsigned i = 0; i < 8; ++i) CHECK_COUNTER(leptonLimits_[i]);
    CHECK_COUNTER(moliereTrials_);
    CHECK_COUNTER(moliereDeflections_);
    CHECK_COUNTER(moliereZeroDeflections_);
    CHECK_COUNTER(moliereNewtonIterations_);
    CHECK_COUNTER(moliereMaxNewtonIterations_);
    CHECK_COUNTER(thinningHillasVertices_);
    CHECK_COUNTER(thinningStatisticalVertices_);
    CHECK_COUNTER(thinningParticlesDiscarded_);
    CHECK_COUNTER(fixedPointOverflows_);
    CHECK_COUNTER(invalidRecords_);
    CHECK_COUNTER(weightedMediumRestMassInput_);
    CHECK_COUNTER(weightedCutRestMassEnergy_);
    CHECK_COUNTER(weightedObservedTotalEnergy_);
    CHECK_COUNTER(weightedEscapedTotalEnergy_);
    CHECK_COUNTER(weightedUnwrittenPhotoelectricBindingEnergy_);
    CHECK_COUNTER(weightedObservationCutOverlapEnergy_);
    CHECK_COUNTER(weightedMassConventionCorrection_);
    CHECK_COUNTER(weightedNativeMuonPairCutEnergy_);
#undef CHECK_COUNTER
  }
} // namespace

TEST_CASE("Lepton source counters preserve sums and maxima across fronts",
          "[lepton_statistics]") {
  ensure_runtime();
  execution_space_type ex;
  std::vector<cp::LeptonTransportRecord> inputs(16);
  std::vector<cp::ResidentLeptonSourceCounts> counts(16);
  for (unsigned i = 0; i < 8; ++i) {
    auto& step = inputs[2 * i];
    counts[2 * i].step_ = 1;
    step.limit_ = static_cast<cp::LeptonTransportLimit>(i);
    step.multipleScatteringIterations_ = i;
    step.multipleScatteringApplied_ = i % 2;
    step.multipleScatteringStatus_ = static_cast<std::uint16_t>(
        i % 2 ? cp::MoliereStatus::Success : cp::MoliereStatus::NoDeflection);
    // Completed zero-grammage steps still count as Moliere trials.
    inputs[2 * i + 1] = step;
    inputs[2 * i + 1].multipleScatteringIterations_ = 9999;
  }
  auto t = upload(inputs);
  auto c = upload(counts);
  std::vector<std::uint64_t> initial(cp::LeptonCallStatisticCount, 23);
  initial[cp::LeptonTransportMoliereMaximumNewtonIterations] = 9;
  auto output = upload(initial);
  cp::ResidentLeptonTransportStatisticsFunctor<execution_space_type> reducer{t, c,
                                                                             output};
  for (unsigned round = 0; round < 3; ++round) {
    Kokkos::parallel_reduce("transport counters", policy_type(ex, 0, 16), reducer);
    ex.fence();
    auto actual = download(output);
    auto expected = initial;
    unsigned const n = round + 1;
    expected[cp::LeptonTransportInteractionVertices] += n;
    expected[cp::LeptonTransportMoliereTrials] += 8 * n;
    expected[cp::LeptonTransportMoliereDeflections] += 4 * n;
    expected[cp::LeptonTransportMoliereZeroDeflections] += 4 * n;
    expected[cp::LeptonTransportMoliereNewtonIterations] += 28 * n;
    for (unsigned j = 0; j < 8; ++j) expected[cp::LeptonTransportLimitBase + j] += n;
    for (unsigned j = 0; j < expected.size(); ++j) REQUIRE(actual(j) == expected[j]);
  }
  auto previous = download(output);
  Kokkos::parallel_reduce("empty transport counters", policy_type(ex, 0, 0), reducer);
  ex.fence();
  auto after = download(output);
  for (unsigned j = 0; j < initial.size(); ++j) REQUIRE(after(j) == previous(j));
}

TEST_CASE("Lepton interaction counters use only selected compacted candidates",
          "[lepton_statistics]") {
  ensure_runtime();
  execution_space_type ex;
  std::vector<cp::LeptonVertexSelectionOutcome> vertices(10);
  std::vector<cp::LeptonFinalStateClassification> finals(10);
  for (unsigned i = 1; i < 10; ++i) vertices[i].interactionFlag_ = 1;
  finals[0].recordFlag_ = finals[0].childCount_ = 999;
  finals[2].recordFlag_ = finals[2].bremsFlag_ = 1;
  finals[2].childCount_ = 2;
  finals[3].recordFlag_ = finals[3].ionizationFlag_ = finals[3].childCount_ = 1;
  finals[3].parameters_.thinningStatus_ =
      static_cast<unsigned>(cp::EmThinningStatus::Hillas);
  finals[4].recordFlag_ = finals[4].annihilationFlag_ = 1;
  finals[4].parameters_.thinningStatus_ =
      static_cast<unsigned>(cp::EmThinningStatus::Statistical);
  finals[5].suppressionFlag_ = finals[5].bremsSuppressionFlag_ = 1;
  // Legacy diagnostics remain readable; these artificial records do not enable
  // device direct-pair generation or any rejection sampler.
  finals[6].suppressionFlag_ = finals[6].electronPairSuppressionFlag_ = 1;
  finals[7].recordFlag_ = finals[7].electronPairFlag_ = 1;
  finals[7].childCount_ = 2;
  finals[7].parameters_.processId_ = cp::ElectronPairProcessId;
  finals[7].parameters_.thinningStatus_ =
      static_cast<unsigned>(cp::EmThinningStatus::Hillas);
  finals[8].recordFlag_ = 1;
  finals[8].childCount_ = 4;
  finals[9].electronPairRejectionTrials_ = 17;
  finals[9].electronPairZeroWeightFlag_ = finals[9].electronPairRejectionFallbackFlag_ =
      1;
  finals[9].electronPairEnvelopeViolationFlag_ = 1;
  std::array<std::uint64_t, cp::LeptonInteractionStatisticCount> delta{
      2, 5, 9, 1, 1, 1, 1, 2, 1, 2, 1, 17, 1, 1, 1, 2, 1, 4};
  auto v = upload(vertices);
  auto f = upload(finals);
  auto output = upload(std::vector<std::uint64_t>(cp::LeptonCallStatisticCount, 31));
  for (unsigned round = 1; round <= 3; ++round) {
    Kokkos::parallel_reduce(
        "interaction counters", policy_type(ex, 0, vertices.size()),
        cp::ResidentLeptonInteractionStatisticsFunctor<execution_space_type>{v, f,
                                                                             output});
    ex.fence();
    auto actual = download(output);
    for (unsigned i = 0; i < cp::LeptonTransportStatisticCount; ++i)
      REQUIRE(actual(i) == 31);
    for (unsigned i = 0; i < delta.size(); ++i)
      REQUIRE(actual(cp::LeptonCallInteractionStatisticBase + i) ==
              31 + round * delta[i]);
  }
}

TEST_CASE("Fused lepton profile agrees with independent step and final-state passes",
          "[lepton_statistics]") {
  ensure_runtime();
  execution_space_type ex;
  std::array<int, 4> const pids{11, -11, 13, -13};
  for (bool sharded : {false, true}) {
    std::vector<cp::LeptonTransportRecord> sources(257), packed;
    std::vector<cp::ResidentLeptonSourceCounts> counts(sources.size());
    std::vector<cp::LeptonFinalStateRecord> records;
    std::vector<cp::LeptonVertexSelectionOutcome> vertices(173);
    std::vector<cp::LeptonFinalStateClassification> finals(vertices.size());
    std::uint64_t iterations = 0;
    for (unsigned i = 0; i < sources.size(); ++i) {
      if (i % 3 == 0) continue;
      auto& step = sources[i];
      step.inputIndex_ = i;
      step.start_.historyId_ = 1000 + i;
      step.start_.pid_ = pids[i % 4];
      step.start_.weight_ = 1. + (i % 4) * .25;
      step.start_.positionM_[2] = 9.;
      step.start_.energyGeV_ = 20.;
      step.end_ = step.start_;
      step.end_.positionM_[2] = i % 11 ? 2. : 9.;
      step.end_.energyGeV_ = 18.;
      step.continuousDepositedEnergyGeV_ = 2.;
      step.limit_ = static_cast<cp::LeptonTransportLimit>(i % 8);
      step.multipleScatteringIterations_ = i % 17;
      step.multipleScatteringApplied_ = i % 2;
      step.multipleScatteringStatus_ = static_cast<std::uint16_t>(
          i % 2 ? cp::MoliereStatus::Success : cp::MoliereStatus::NoDeflection);
      iterations += i % 17;
      counts[i].step_ = 1;
      packed.push_back(step);
      if (i % 5 != 0) continue;
      cp::LeptonFinalStateRecord record;
      record.inputIndex_ = i;
      record.parentHistoryId_ = step.start_.historyId_;
      record.processId_ = i % 2 ? cp::IonizationProcessId : cp::BremsProcessId;
      record.secondaryCount_ = 1;
      record.thinningStatus_ = static_cast<unsigned>(cp::EmThinningStatus::Hillas);
      record.weightedMassConventionCorrectionGeV_ = .125;
      records.push_back(record);
    }
    REQUIRE(records.size() < vertices.size());
    REQUIRE(packed.size() < sources.size());
    auto source = upload(sources);
    auto count = upload(counts);
    auto steps = upload(packed);
    auto rec = upload(records);
    auto vertex = upload(vertices);
    auto final = upload(finals);
    auto stats = upload(std::vector<std::uint64_t>(cp::LeptonCallStatisticCount));
    cp::ProfileProjectionStorage<execution_space_type> projection;
    projection.initialize(config(), ex);
    cp::ProfileAccumulatorStorage<execution_space_type> fused, separate;
    fused.initialize(config(), ex);
    separate.initialize(config(), ex);
    if (sharded) {
#ifdef KOKKOS_ENABLE_OPENMP
      if constexpr (std::is_same_v<execution_space_type, Kokkos::OpenMP>)
        fused.enableHostShards(4, 1 << 20);
      else
#endif
      {
        REQUIRE_THROWS_AS(fused.enableHostShards(4, 1 << 20), std::logic_error);
        continue;
      }
    }
    auto const p = projection.deviceView();
    for (unsigned round = 0; round < 3; ++round) {
      fused.requireOpenForAccumulation();
      cp::ResidentLeptonTransportStatisticsProfileFunctor<execution_space_type> transport{
          {source, count, stats}, p, fused.deviceView(), steps, packed.size()};
      cp::enqueue_lepton_profile_statistics(policy_type(ex, 0, sources.size()), transport,
                                            fused.hostShards());
      Kokkos::parallel_reduce(
          "fused final profile", policy_type(ex, 0, vertices.size()),
          cp::ResidentLeptonInteractionStatisticsProfileFunctor<execution_space_type>{
              {vertex, final, stats},
              p,
              fused.deviceView(),
              steps,
              packed.size(),
              rec,
              records.size()});
      separate.accumulateLepton(p, steps, packed.size(), rec, records.size(), ex);
    }
    auto actual = fused.downloadFixed("statistics", cp::CooperativeEndpoint::Cuda, ex);
    auto expected =
        separate.downloadFixed("statistics", cp::CooperativeEndpoint::Cuda, ex);
    compare(actual, expected);
    REQUIRE(actual.counters_.steps_ == 3 * packed.size());
    REQUIRE(actual.counters_.moliereNewtonIterations_ == 3 * iterations);
    REQUIRE(actual.counters_.moliereMaxNewtonIterations_ == 16);
    REQUIRE(actual.counters_.invalidRecords_ == 0);
    REQUIRE(actual.counters_.weightedMediumRestMassInput_ > 0);
    REQUIRE(std::any_of(actual.histograms_.begin(), actual.histograms_.end(),
                        [](auto n) { return n != 0; }));
    REQUIRE_THROWS_AS(fused.requireOpenForAccumulation(), std::logic_error);
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<execution_space_type, Kokkos::OpenMP>)
      REQUIRE(fused.hostShards().enabled() == sharded);
#endif
  }
}
