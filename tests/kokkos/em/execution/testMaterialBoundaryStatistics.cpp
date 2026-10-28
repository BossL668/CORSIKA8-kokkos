/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "MaterialBoundaryFixture.hpp"
#include <corsika/kokkos/em/execution/LeptonStatistics.hpp>
#include <corsika/kokkos/em/algorithms/LeptonTransportStep.hpp>
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/output/ProfileAccumulatorStorage.hpp>
#include <corsika/kokkos/output/ProfileProjectionStorage.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <limits>

namespace {
  namespace cp = corsika::kokkos;
  using execution_type = Kokkos::DefaultExecutionSpace;
  using memory_type = execution_type::memory_space;
  using policy_type = Kokkos::RangePolicy<execution_type>;
  void ensure_runtime() { static cp::Runtime runtime; }
  template <typename T>
  auto upload(std::vector<T> const& items) {
    Kokkos::View<T*, memory_type> out("boundary inputs", items.size());
    auto host = Kokkos::create_mirror_view(out);
    for (std::size_t i = 0; i < items.size(); ++i) host(i) = items[i];
    Kokkos::deep_copy(out, host);
    return out;
  }
  template <typename T>
  auto download(T const& view) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  }
  cp::ProfileProjectionConfig config() {
    cp::ProfileProjectionConfig c;
    c.enabled_ = c.accumulateOnDevice_ = true;
    c.axisStartPositionM_[2] = -.1;
    c.axisDirection_[2] = 1.;
    c.axisStepLengthM_ = .05;
    c.axisGrammageGPerCm2_ = {0., 5., 10.};
    c.outputBinCount_ = 10;
    c.outputBinWidthGPerCm2_ = 1.;
    c.fixedPointWeightLimit_ = c.fixedPointEnergyLimitGeV_ = 0x1p48;
    return c;
  }
  void compare(cp::FixedProfileSnapshot const& a, cp::FixedProfileSnapshot const& b) {
    REQUIRE(a.histograms_ == b.histograms_);
#define COUNTER(name) REQUIRE(a.counters_.name == b.counters_.name)
    COUNTER(steps_);
    COUNTER(depositedSteps_);
    COUNTER(photonCuts_);
    for (unsigned i = 0; i < 8; ++i) COUNTER(leptonLimits_[i]);
    COUNTER(leptonMaterialBoundaries_);
    COUNTER(moliereTrials_);
    COUNTER(moliereDeflections_);
    COUNTER(moliereZeroDeflections_);
    COUNTER(moliereNewtonIterations_);
    COUNTER(moliereMaxNewtonIterations_);
    COUNTER(thinningHillasVertices_);
    COUNTER(thinningStatisticalVertices_);
    COUNTER(thinningParticlesDiscarded_);
    COUNTER(fixedPointOverflows_);
    COUNTER(invalidRecords_);
    COUNTER(weightedMediumRestMassInput_);
    COUNTER(weightedCutRestMassEnergy_);
    COUNTER(weightedObservedTotalEnergy_);
    COUNTER(weightedEscapedTotalEnergy_);
    COUNTER(weightedUnwrittenPhotoelectricBindingEnergy_);
    COUNTER(weightedObservationCutOverlapEnergy_);
    COUNTER(weightedMassConventionCorrection_);
    COUNTER(weightedNativeMuonPairCutEnergy_);
#undef COUNTER
  }
} // namespace

TEST_CASE("Actual firn-interface steps retain profiles without invalid records",
          "[lepton_statistics][firn][material_boundary]") {
  ensure_runtime();
  execution_type ex;
  auto fixture = material_boundary_test::make_fixture();
  REQUIRE(fixture.models_.size() == 1);
  REQUIRE(fixture.models_[0].densityKind_ ==
          cp::EnvironmentModel::Density::ExponentialFirn);
  cp::EnvironmentStorage<execution_type> environment(fixture.view());
  cp::EnvironmentBorrowedView const geometry(environment.getDataView());
  // An exact linear range fixture isolates geometry/record accounting from
  // cross-section accuracy; all transport and firn integration code is real.
  std::vector<cp::NativeUtilityColumn> columns(4);
  std::vector<double> values, derivatives;
  std::array<int, 4> const pids{11, -11, 13, -13};
  for (unsigned i = 0; i < 4; ++i) {
    double const mass = i < 2 ? .51099895 : 105.6583755;
    auto& column = columns[i];
    column.pdgId_ = pids[i];
    column.particleMassMeV_ = column.lowerEnergyLimitMeV_ = mass;
    column.spline_.axis_ = {cp::NativeAxisType::Linear, 2, mass, 1000., 1000. - mass};
    column.spline_.coefficientOffset_ = 2 * i;
    column.spline_.coefficientCount_ = 2;
    values.insert(values.end(), {0., 1000. - mass});
    derivatives.insert(derivatives.end(), {1000. - mass, 1000. - mass});
  }
  auto utility = upload(columns);
  auto cubic = upload(values);
  auto derivative = upload(derivatives);
  cp::ProposalNativeDeviceView const native(
      {}, {}, {}, {}, {},
      cp::NativeArrayView<cp::NativeUtilityColumn, std::uint32_t>(utility), {}, {}, {},
      {}, {}, cp::NativeArrayView<double, std::uint64_t>(cubic),
      cp::NativeArrayView<double, std::uint64_t>(derivative), {});
  auto inputs = upload(fixture.inputs_);
  auto lengths = upload(fixture.distances_);
  std::size_t const n = fixture.inputs_.size();
  Kokkos::View<cp::LeptonTransportRecord*, memory_type> steps("firn transport", n);
  Kokkos::View<unsigned*, memory_type> flags("firn status", n);
  Kokkos::parallel_for(
      "firn surface transport", policy_type(ex, 0, n), KOKKOS_LAMBDA(std::size_t i) {
        cp::FallbackRequest fallback;
        cp::ContinuousPhysicsView const physics(native, .4, .4);
        flags(i) = cp::transport_lepton(physics, false, geometry, {}, inputs(i), steps(i),
                                        fallback, {lengths(i), true, true});
      });
  auto actual = download(steps);
  auto status = download(flags);
  std::vector<cp::LeptonTransportRecord> sources(2 * n);
  std::vector<cp::ResidentLeptonSourceCounts> sourceCounts(2 * n);
  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    REQUIRE(status(i) == 0);
    REQUIRE(actual(i).limit_ == cp::LeptonTransportLimit::MaterialBoundary);
    REQUIRE(std::abs(actual(i).distanceM_ - fixture.distances_[i]) < 1.e-12);
    REQUIRE(std::abs(actual(i).traversedGrammageGPerCm2_ - fixture.grammages_[i]) <
            1.e-10);
    REQUIRE(actual(i).continuousDepositedEnergyGeV_ > 0.);
    REQUIRE(actual(i).cutDepositedEnergyGeV_ == 0.);
    REQUIRE(actual(i).end_.energyGeV_ < actual(i).start_.energyGeV_);
    REQUIRE(actual(i).end_.historyId_ == actual(i).start_.historyId_);
    REQUIRE(actual(i).end_.stepId_ == actual(i).start_.stepId_ + 1);
    sources[2 * i] = actual(i);
    sourceCounts[2 * i].step_ = 1;
  }
  auto source = upload(sources);
  auto counts = upload(sourceCounts);
  cp::ProfileProjectionStorage<execution_type> projection;
  projection.initialize(config(), ex);
  auto const p = projection.deviceView();
  for (bool sharded : {false, true}) {
    cp::ProfileAccumulatorStorage<execution_type> fused, separate, decoded;
    fused.initialize(config(), ex);
    separate.initialize(config(), ex);
    decoded.initialize(config(), ex);
    if (sharded) {
#ifdef KOKKOS_ENABLE_OPENMP
      if constexpr (std::is_same_v<execution_type, Kokkos::OpenMP>)
        fused.enableHostShards(4, 1 << 20);
      else
#endif
      {
        REQUIRE_THROWS_AS(fused.enableHostShards(4, 1 << 20), std::logic_error);
        continue;
      }
    }
    auto stats = upload(std::vector<std::uint64_t>(cp::LeptonCallStatisticCount));
    Kokkos::View<cp::LeptonFinalStateRecord*, memory_type> records("no interactions", 0);
    for (unsigned round = 0; round < 3; ++round) {
      cp::ResidentLeptonTransportStatisticsProfileFunctor<execution_type> functor{
          {source, counts, stats}, p, fused.deviceView(), steps, n};
      cp::enqueue_lepton_profile_statistics(policy_type(ex, 0, sources.size()), functor,
                                            fused.hostShards());
      separate.accumulateLepton(p, steps, n, records, 0, ex);
      decoded.accumulateLepton(p, steps, n, records, 0, ex);
    }
    auto a = fused.downloadFixed("firn", cp::CooperativeEndpoint::Cuda, ex);
    auto b = separate.downloadFixed("firn", cp::CooperativeEndpoint::OpenMP, ex);
    compare(a, b);
    REQUIRE(a.counters_.leptonMaterialBoundaries_ == 3 * n);
    REQUIRE(a.counters_.invalidRecords_ == 0);
    for (auto v : a.counters_.leptonLimits_) REQUIRE(v == 0);
    auto total = download(stats);
    REQUIRE(total(cp::LeptonCallMaterialBoundaries) == 3 * n);
    for (unsigned i = 0; i < 8; ++i)
      REQUIRE(total(cp::LeptonTransportLimitBase + i) == 0);
    REQUIRE(total(cp::LeptonTransportMoliereTrials) == 3 * n);
    cp::CooperativeProfileMerge merge("firn", config());
    REQUIRE_NOTHROW(merge.commit(a));
    REQUIRE_NOTHROW(merge.commit(b));
    auto merged = merge.take();
    REQUIRE(merged.counters_.leptonMaterialBoundaries_ == 6 * n);
    auto const result = cp::decode_fixed_profile(merged);
    REQUIRE(result.leptonMaterialBoundaries_ == 6 * n);
    REQUIRE(result.invalidRecords_ == 0);
    REQUIRE(result.weightedDepositedEnergyGeV_ > 0.);
    REQUIRE(decoded.download(ex).leptonMaterialBoundaries_ == 3 * n);
  }
}

TEST_CASE("Material diagnostics preserve old columns and reject unknown limits",
          "[lepton_statistics][material_boundary]") {
  ensure_runtime();
  static_assert(cp::LeptonTransportStatisticCount == 14);
  static_assert(cp::LeptonCallInteractionStatisticBase == 14);
  static_assert(cp::LeptonCallMaterialBoundaries == 32);
  execution_type ex;
  cp::ProfileProjectionStorage<execution_type> projection;
  projection.initialize(config(), ex);
  for (int invalid : {-1, 9, 100}) {
    cp::ProfileAccumulatorStorage<execution_type> owner;
    owner.initialize(config(), ex);
    cp::LeptonTransportRecord step;
    step.start_.pid_ = step.end_.pid_ = 11;
    step.start_.weight_ = 1.;
    step.limit_ = static_cast<cp::LeptonTransportLimit>(invalid);
    auto steps = upload(std::vector<cp::LeptonTransportRecord>{step});
    Kokkos::View<cp::LeptonFinalStateRecord*, memory_type> records("empty", 0);
    owner.accumulateLepton(projection.deviceView(), steps, 1, records, 0, ex);
    Kokkos::View<cp::ProfileCounters*, memory_type, Kokkos::MemoryUnmanaged> counters(
        owner.deviceView().counters_, 1);
    auto result = download(counters);
    REQUIRE(result(0).invalidRecords_ == 1);
    REQUIRE(result(0).leptonMaterialBoundaries_ == 0);
    REQUIRE_THROWS_AS(owner.downloadFixed("invalid", cp::CooperativeEndpoint::Cuda, ex),
                      std::runtime_error);
  }
  cp::ProfileCounters a, b;
  a.leptonMaterialBoundaries_ = std::numeric_limits<unsigned long long>::max();
  b.leptonMaterialBoundaries_ = 1;
  REQUIRE_THROWS_AS(cp::merge_profile_counters(a, b), std::overflow_error);
}
