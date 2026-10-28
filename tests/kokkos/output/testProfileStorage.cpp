/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <tests/kokkos/output/ProfileStorageChecks.hpp>

using namespace profile_storage_test;

namespace {
  using execution_space_type = transport_test::execution_space_type;
  using memory_space_type = transport_test::memory_space_type;
  void ensure_runtime() { static current::Runtime runtime; }

  current::ProfileProjectionConfig config() {
    current::ProfileProjectionConfig c;
    c.enabled_ = true;
    c.accumulateOnDevice_ = true;
    c.axisStartPositionM_[2] = 10.;
    c.axisDirection_[2] = -1.;
    c.axisStepLengthM_ = 2.;
    c.axisGrammageGPerCm2_ = {0., 2., 4., 6., 8., 10.};
    c.outputBinCount_ = 10;
    c.outputBinWidthGPerCm2_ = 1.;
    c.fixedPointWeightLimit_ = 0x1p48;
    c.fixedPointEnergyLimitGeV_ = 0x1p48;
    return c;
  }

  current::FixedProfileSnapshot snapshot(std::string identity = "event") {
    auto c = config();
    return {identity,
            current::CooperativeEndpoint::Cuda,
            c,
            1. / (0x1p62 / c.fixedPointWeightLimit_),
            1. / (0x1p62 / c.fixedPointEnergyLimitGeV_),
            std::vector<std::int64_t>(80),
            {}};
  }

  reference::FixedProfileSnapshot old_snapshot(current::FixedProfileSnapshot const& a) {
    reference::FixedProfileSnapshot b;
    b.shower_identity = a.showerIdentity_;
    b.endpoint = static_cast<reference::CooperativeEndpoint>(a.endpoint_);
    b.config = convert(a.config_);
    b.weight_units = a.weightUnits_;
    b.energy_units = a.energyUnits_;
    b.histograms = a.histograms_;
    return b;
  }
} // namespace

TEST_CASE("Production memory budget and transactional integer merge",
          "[profile_storage]") {
  auto const maximum = std::numeric_limits<std::size_t>::max();
  REQUIRE(current::checked_memory_add(maximum - 1, 1) == maximum);
  REQUIRE(current::checked_memory_multiply(maximum, 0) == 0);
  REQUIRE_THROWS_AS(current::checked_memory_add(maximum, 1), std::overflow_error);
  REQUIRE_THROWS_AS(current::checked_memory_multiply(maximum, 2), std::overflow_error);
  current::MemoryProjectionBuilder a(100);
  reference::KokkosMemoryProjectionBuilder b(100);
  for (std::size_t size : {0, 20, 100, 200}) {
    a.replace(20, size);
    b.replace(20, size);
    REQUIRE(a.result().retainedBytes_ == b.result().retained_bytes);
    REQUIRE(a.result().transientPeakBytes_ == b.result().transient_peak_bytes);
  }
  a.merge(10, {20, 40});
  b.merge(10, {20, 40});
  a.appendDisjointGrowth(10, {20, 60});
  b.appendDisjointGrowth(10, {20, 60});
  REQUIRE(a.result().retainedBytes_ == b.result().retained_bytes);
  REQUIRE(a.result().transientPeakBytes_ == b.result().transient_peak_bytes);
  REQUIRE_THROWS_AS(a.replace(maximum, 1), std::logic_error);
  REQUIRE_THROWS_AS(a.appendDisjointGrowth(10, {9, 60}), std::logic_error);
  REQUIRE_THROWS_AS(a.appendDisjointGrowth(10, {20, 19}), std::logic_error);
  current::ResidentMemoryBudgetGate gate;
  REQUIRE(gate.permits(10, {20, 30}, "stage"));
  std::size_t limit = 30;
  gate.context_ = &limit;
  gate.callback_ = [](void* context, std::size_t, const current::MemoryProjection& q,
                      char const*) {
    return q.transientPeakBytes_ <= *static_cast<std::size_t*>(context);
  };
  REQUIRE(gate.permits(10, {20, 30}, "stage"));
  REQUIRE_FALSE(gate.permits(10, {20, 31}, "stage"));

  current::FixedAccumulatorLayout layout{"test", .5, 3};
  reference::FixedAccumulatorLayout oldLayout{"test", .5, 3};
  std::vector<std::int64_t> values{1, -2, 3}, old = values, delta{4, 5, -6};
  current::merge_fixed_accumulators(values, layout, delta, layout);
  reference::mergeFixedAccumulators(old, oldLayout, delta, oldLayout);
  REQUIRE(values == old);
  REQUIRE(values == std::vector<std::int64_t>{5, 3, -3});
  auto const original = values;
  delta[2] = std::numeric_limits<std::int64_t>::min();
  REQUIRE_THROWS_AS(current::merge_fixed_accumulators(values, layout, delta, layout),
                    std::overflow_error);
  REQUIRE(values == original);
  auto wrong = layout;
  wrong.identity_ = "different";
  REQUIRE_THROWS_AS(current::merge_fixed_accumulators(values, layout, delta, wrong),
                    std::invalid_argument);
  REQUIRE(values == original);
  current::ProfileCounters counters{}, addition{};
  counters.steps_ = std::numeric_limits<unsigned long long>::max();
  addition.steps_ = 1;
  REQUIRE_THROWS_AS(current::merge_profile_counters(counters, addition),
                    std::overflow_error);
  counters = {};
  addition = {};
  counters.weightedNativeMuonPairCutEnergy_ = std::numeric_limits<long long>::min();
  addition.weightedNativeMuonPairCutEnergy_ = -1;
  REQUIRE_THROWS_AS(current::merge_profile_counters(counters, addition),
                    std::overflow_error);
}

TEST_CASE("Production profile snapshot commit and host replicas", "[profile_storage]") {
  auto a = snapshot();
  auto b = old_snapshot(a);
  current::CooperativeProfileMerge merger("event", a.config_);
  reference::CooperativeProfileMerge oldMerger("event", b.config);
  REQUIRE_THROWS_AS(merger.take(), std::logic_error);
  for (std::size_t k = 0; k < a.histograms_.size(); ++k)
    a.histograms_[k] = b.histograms[k] = static_cast<std::int64_t>(k) - 20;
  a.counters_.moliereMaxNewtonIterations_ = b.counters.moliere_max_newton_iterations = 9;
  a.counters_.steps_ = b.counters.steps = 2;
  merger.commit(a);
  oldMerger.commit(b);
  REQUIRE_THROWS_AS(merger.commit(a), std::invalid_argument);
  auto bad = a;
  bad.endpoint_ = current::CooperativeEndpoint::OpenMP;
  bad.showerIdentity_ = "other";
  REQUIRE_THROWS_AS(merger.commit(bad), std::invalid_argument);
  a.endpoint_ = current::CooperativeEndpoint::OpenMP;
  b.endpoint = reference::CooperativeEndpoint::OpenMP;
  a.counters_.moliereMaxNewtonIterations_ = b.counters.moliere_max_newton_iterations = 3;
  merger.commit(a);
  oldMerger.commit(b);
  auto merged = merger.take();
  auto oldMerged = oldMerger.take();
  compare(merged, oldMerged);
  REQUIRE(merged.counters_.moliereMaxNewtonIterations_ == 9);
  REQUIRE(merged.counters_.steps_ == 4);
  compare(current::decode_fixed_profile(merged),
          reference::decodeFixedProfile(oldMerged));
  REQUIRE_THROWS_AS(merger.take(), std::logic_error);
  REQUIRE_THROWS_AS(merger.commit(a), std::logic_error);
  bad = snapshot();
  bad.counters_.invalidRecords_ = 1;
  REQUIRE_THROWS_AS(current::validate_fixed_profile(bad), std::runtime_error);
  bad = snapshot();
  bad.counters_.fixedPointOverflows_ = 1;
  REQUIRE_THROWS_AS(current::validate_fixed_profile(bad), std::runtime_error);
  bad = snapshot();
  bad.energyUnits_ *= 2;
  REQUIRE_THROWS_AS(current::validate_fixed_profile(bad), std::invalid_argument);

  long long bins[80]{}, oldBins[80]{};
  current::ProfileCounters counters{};
  reference::DeviceProfileCounters oldCounters{};
  current::ProfileAccumulatorView view{};
  reference::DeviceProfileAccumulator oldView{};
  view.bins_ = oldView.bins = 10;
  view.counters_ = &counters;
  oldView.counters = &oldCounters;
  view.weightScale_ = oldView.weight_scale = 16384.;
  view.energyScale_ = oldView.energy_scale = 16384.;
  view.inverseWeightScale_ = oldView.inverse_weight_scale = 1. / 16384.;
  view.inverseEnergyScale_ = oldView.inverse_energy_scale = 1. / 16384.;
  view.binWidthGPerCm2_ = oldView.bin_width_g_per_cm2 = 1.;
  view.energyLossThresholdGPerCm2_ = oldView.energy_loss_threshold_g_per_cm2 = 1.e-4;
  view.photons_ = bins;
  oldView.photons = oldBins;
  for (std::size_t workers : {0, 1, 2, 3, 130, 300}) {
    for (std::size_t budget : {0, 1024, 1 << 20, 1 << 24}) {
      current::HostProfileShards shards;
      reference::HostProfileShards oldShards;
      shards.configure(view, workers, budget);
      oldShards.configure(oldView, workers, budget);
      REQUIRE(shards.enabled() == oldShards.enabled());
      REQUIRE(shards.count() == oldShards.count());
      REQUIRE(shards.bytes() == oldShards.bytes());
      REQUIRE_THROWS_AS(shards.configure(view, workers, budget), std::logic_error);
      for (std::size_t k = 0; k < shards.count(); ++k) {
        auto const& s = shards.entries()[k].accumulator_;
        auto const& o = oldShards.entries()[k].accumulator;
        current::accumulate_particle_profile<current::ProfileAtomicOperations>(s, 22, 1.,
                                                                               4., .5);
        reference::accumulateParticleProfile<reference::KokkosProfileAtomicOperations>(
            o, 22, 1., 4., .5);
        s.counters_->steps_ = o.counters->steps = k;
      }
      auto actual = snapshot();
      auto expected = old_snapshot(actual);
      shards.mergeInto(actual);
      oldShards.mergeInto(expected);
      compare(actual, expected);
      if (shards.enabled()) REQUIRE_THROWS_AS(shards.mergeInto(actual), std::logic_error);
      shards.reset(view);
      oldShards.reset(oldView);
      actual = snapshot();
      expected = old_snapshot(actual);
      shards.mergeInto(actual);
      oldShards.mergeInto(expected);
      compare(actual, expected);
      REQUIRE(actual.histograms_ == std::vector<std::int64_t>(80));
    }
  }
}

TEST_CASE("Production profile storage lifecycle and actual execution kernels",
          "[profile_storage]") {
  ensure_runtime();
  using Owner = current::ProfileAccumulatorStorage<execution_space_type>;
  using OldOwner = reference::KokkosProfileAccumulator<execution_space_type>;
  STATIC_REQUIRE(std::is_same_v<typename Owner::host_staging_space_type,
                                typename OldOwner::host_staging_space>);
#ifdef CORSIKA_EXPECT_CUDA
  STATIC_REQUIRE(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#endif
#ifdef CORSIKA_EXPECT_OPENMP
  STATIC_REQUIRE(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  auto c = config();
  auto oldConfig = convert(c);
  Owner disabled;
  REQUIRE_FALSE(disabled.enabled());
  REQUIRE(Owner::projectedDeviceBytes({}) == 0);
  REQUIRE_THROWS_AS(disabled.download(), std::logic_error);
  auto invalid = c;
  invalid.outputBinCount_ = 0;
  REQUIRE_THROWS_AS(Owner::projectedDeviceBytes(invalid), std::invalid_argument);
  invalid = c;
  invalid.outputBinCount_ = std::numeric_limits<std::size_t>::max();
  REQUIRE_THROWS_AS(Owner::projectedDeviceBytes(invalid), std::overflow_error);
  REQUIRE(Owner::projectedDeviceBytes(c) ==
          OldOwner::projectedDeviceBytes(oldConfig) + sizeof(unsigned long long));

  for (int mode = 0; mode < 3; ++mode) {
    c.crossingMode_ = static_cast<corsika::ProfileCrossingMode>(mode);
    oldConfig = convert(c);
    current::ProfileProjectionStorage<execution_space_type> projection;
    reference::KokkosProfileProjection<execution_space_type> oldProjection;
    projection.initialize(c);
    oldProjection.initialize(oldConfig);
    execution_space_type().fence();
    REQUIRE(projection.deviceBytes() == oldProjection.deviceBytes());
    Owner owner;
    OldOwner oldOwner;
    owner.initialize(c);
    oldOwner.initialize(oldConfig);
    REQUIRE(owner.enabled());
    REQUIRE(owner.deviceBytes() == oldOwner.deviceBytes() + sizeof(unsigned long long));
    std::vector<current::PhotonTransportRecord> photons(128);
    std::vector<reference::PhotonTransportRecord> oldPhotons;
    std::vector<current::LeptonTransportRecord> leptons(128);
    std::vector<reference::LeptonTransportRecord> oldLeptons;
    std::vector<current::PhotonFinalStateRecord> photonFinal(128);
    std::vector<reference::PhotonPairFinalStateRecord> oldPhotonFinal(128);
    std::vector<current::LeptonFinalStateRecord> leptonFinal(128);
    std::vector<reference::BremsFinalStateRecord> oldLeptonFinal(128);
    for (std::size_t k = 0; k < photons.size(); ++k) {
      auto& p = photons[k];
      p.inputIndex_ = k;
      p.start_.historyId_ = k + 1;
      p.start_.pid_ = 22;
      p.start_.positionM_[2] = 9.;
      p.start_.energyGeV_ = 8.;
      p.start_.weight_ = .5;
      p.end_ = p.start_;
      p.end_.positionM_[2] = 3.;
      p.end_.energyGeV_ = 4.;
      p.cutDepositedEnergyGeV_ = .25;
      p.interaction_.processId_ = current::ComptonProcessId;
      oldPhotons.push_back(transport_test::convert(p));
      auto& l = leptons[k];
      l.inputIndex_ = k;
      l.start_ = p.start_;
      l.start_.pid_ = (k % 2) ? 13 : 11;
      l.end_ = p.end_;
      l.end_.pid_ = l.start_.pid_;
      l.continuousDepositedEnergyGeV_ = .5;
      l.cutDepositedEnergyGeV_ = .25;
      l.multipleScatteringIterations_ = k % 7;
      oldLeptons.push_back(transport_test::convert(l));
      auto& pf = photonFinal[k];
      auto& opf = oldPhotonFinal[k];
      pf.inputIndex_ = opf.input_index = k;
      pf.parentHistoryId_ = opf.parent_history_id = k + 1;
      pf.processId_ = opf.process_id = current::ComptonProcessId;
      pf.secondaryCount_ = opf.secondary_count = 2;
      pf.energySplitFraction_ = opf.energy_split_fraction = .5;
      pf.weightedMassConventionCorrectionGeV_ =
          opf.weighted_mass_convention_correction_GeV = .125;
      auto& lf = leptonFinal[k];
      auto& olf = oldLeptonFinal[k];
      lf.inputIndex_ = olf.input_index = k;
      lf.parentHistoryId_ = olf.parent_history_id = k + 1;
      lf.processId_ = olf.process_id = current::IonizationProcessId;
      lf.secondaryCount_ = olf.secondary_count = 2;
      lf.weightedMassConventionCorrectionGeV_ =
          olf.weighted_mass_convention_correction_GeV = .125;
    }
    auto pd = transport_test::upload(photons);
    auto opd = transport_test::upload(oldPhotons);
    auto ld = transport_test::upload(leptons);
    auto old = transport_test::upload(oldLeptons);
    auto pf = transport_test::upload(photonFinal);
    auto opf = transport_test::upload(oldPhotonFinal);
    auto lf = transport_test::upload(leptonFinal);
    auto olf = transport_test::upload(oldLeptonFinal);
    owner.accumulatePhoton(projection.deviceView(), pd, photons.size(), pf,
                           photonFinal.size());
    oldOwner.accumulatePhoton(oldProjection.deviceView(), opd, oldPhotons.size(), opf,
                              oldPhotonFinal.size());
    owner.accumulateLepton(projection.deviceView(), ld, leptons.size(), lf,
                           leptonFinal.size());
    oldOwner.accumulateLepton(oldProjection.deviceView(), old, oldLeptons.size(), olf,
                              oldLeptonFinal.size());
    auto a = owner.downloadFixed("event", current::CooperativeEndpoint::Cuda);
    auto b = oldOwner.downloadFixed("event", reference::CooperativeEndpoint::Cuda);
    compare(a, b);
    REQUIRE(a.counters_.steps_ == 256);
    REQUIRE(a.counters_.weightedMassConventionCorrection_ == 256 * 2048);
    REQUIRE(a.counters_.weightedMediumRestMassInput_ > 0);
    compare(current::decode_fixed_profile(a), reference::decodeFixedProfile(b));
    REQUIRE_THROWS_AS(owner.downloadFixed("event", current::CooperativeEndpoint::Cuda),
                      std::logic_error);
    REQUIRE_THROWS_AS(owner.requireOpenForAccumulation(), std::logic_error);
    REQUIRE_THROWS_AS(owner.reset(0., 1.), std::invalid_argument);
    owner.reset(c.fixedPointWeightLimit_, c.fixedPointEnergyLimitGeV_);
    oldOwner.reset(oldConfig.fixed_point_weight_limit,
                   oldConfig.fixed_point_energy_limit_GeV);
    owner.accumulateLepton(projection.deviceView(), ld, leptons.size(), {}, 0);
    oldOwner.accumulateLepton(oldProjection.deviceView(), old, oldLeptons.size(), {}, 0);
    auto decoded = owner.download();
    auto oldDecoded = oldOwner.download();
    compare(decoded, oldDecoded);
    REQUIRE(decoded.steps_ == 128);
    REQUIRE_THROWS_AS(owner.download(), std::logic_error);
    owner.reset(c.fixedPointWeightLimit_, c.fixedPointEnergyLimitGeV_);
    oldOwner.reset(oldConfig.fixed_point_weight_limit,
                   oldConfig.fixed_point_energy_limit_GeV);
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<execution_space_type, Kokkos::OpenMP>) {
      owner.enableHostShards(3, 1 << 20);
      oldOwner.enableHostShards(3, 1 << 20);
      REQUIRE(owner.hostShards().count() == 3);
      compare(owner.download(), oldOwner.download());
      REQUIRE_THROWS_AS(owner.requireOpenForAccumulation(), std::logic_error);
    } else
#endif
    {
      REQUIRE_THROWS_AS(owner.enableHostShards(3, 1 << 20), std::logic_error);
    }
  }

  // A failed fixed snapshot is sealed; the plain download retains the original
  // diagnostic-return contract instead of silently dropping invalid records.
  Owner owner;
  OldOwner oldOwner;
  owner.initialize(c);
  oldOwner.initialize(convert(c));
  auto view = owner.deviceView();
  auto oldView = oldOwner.deviceView();
  auto markInvalid = [&]() {
    Kokkos::parallel_for(
        "mark invalid profile ledger", Kokkos::RangePolicy<execution_space_type>(0, 1),
        KOKKOS_LAMBDA(int) {
          view.counters_->invalidRecords_ = 1;
          oldView.counters->invalid_records = 1;
        });
  };
  markInvalid();
  compare(owner.download(), oldOwner.download());
  owner.reset(c.fixedPointWeightLimit_, c.fixedPointEnergyLimitGeV_);
  oldOwner.reset(c.fixedPointWeightLimit_, c.fixedPointEnergyLimitGeV_);
  markInvalid();
  REQUIRE_THROWS_AS(owner.downloadFixed("event", current::CooperativeEndpoint::Cuda),
                    std::runtime_error);
  REQUIRE_THROWS_AS(oldOwner.downloadFixed("event", reference::CooperativeEndpoint::Cuda),
                    std::runtime_error);
  REQUIRE_THROWS_AS(owner.requireOpenForAccumulation(), std::logic_error);
  REQUIRE_THROWS_AS(owner.downloadFixed("event", current::CooperativeEndpoint::Cuda),
                    std::logic_error);
}
