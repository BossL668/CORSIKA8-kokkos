/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "LeptonCascadeTestFixture.hpp"
#include <fstream>

namespace {
  using namespace lepton_cascade_test;
  struct Coverage {
    std::size_t calls_{}, waves_{}, steps_{}, vertices_{}, photons_{}, fallbacks_{},
        decays_{}, observations_{};
    std::map<int, std::size_t> process_, pid_;
  } coverage;
  cp::LeptonCascadeResult compare(Fixture& f, std::vector<cp::ParticleState> const& p,
                                  Options options = {}) {
    std::optional<cp::FirstInteractionRecord> a, b;
    auto got = run(f, p, a, options);
    auto old = run(f, p, b, options, true);
    equal(got, old);
    photon_cascade_test::equal(a, b);
    ++coverage.calls_;
    coverage.waves_ += got.wavefronts_;
    coverage.steps_ += got.transportRecords_;
    coverage.vertices_ += got.interactionVertices_;
    coverage.photons_ += got.generatedPhotons_.size();
    coverage.fallbacks_ += got.fallbackEvents_.size();
    coverage.decays_ += got.decayCandidates_.size();
    coverage.observations_ += got.observations_.size();
    for (auto const& q : p) ++coverage.pid_[q.pid_];
    for (auto const& r : got.finalStateRecords_) ++coverage.process_[r.processId_];
    return got;
  }
} // namespace
TEST_CASE("Lepton cascade validates input and preserves unstaged spill",
          "[lepton_cascade]") {
  ensure_runtime();
  Fixture f;
  std::optional<cp::FirstInteractionRecord> first;
  REQUIRE(run(f, {}, first).completed_);
  auto p = particles(3);
  Options o;
  o.pendingInput_ = 1;
  REQUIRE_THROWS_AS(run(f, p, first, o), std::invalid_argument);
  o = {};
  o.first_ = 0;
  REQUIRE_THROWS_AS(run(f, p, first, o), std::invalid_argument);
  o = {};
  o.last_ = o.first_;
  REQUIRE_THROWS_AS(run(f, p, first, o), std::invalid_argument);
  o = {};
  o.waves_ = 0;
  REQUIRE_THROWS_AS(run(f, p, first, o), std::invalid_argument);
  o = {};
  o.minimum_ = 0;
  REQUIRE_THROWS_AS(run(f, p, first, o), std::invalid_argument);
  for (unsigned rejection : {1, 2, 3, 4, 5}) {
    Budget budget;
    budget.reject_ = rejection;
    o = {};
    o.budget_ = budget.gate();
    auto r = run(f, p, first, o);
    REQUIRE(r.workspaceLimitCheckpoint_);
    REQUIRE(r.completed_);
    REQUIRE(r.wavefronts_ == 0);
    REQUIRE(r.secondaryHistoryIdsUsed_ == 0);
    equal(r.cpuSpillParticles_, p);
    REQUIRE(r.fallbackEvents_.empty());
  }
  o = {};
  o.reserved_ = 2;
  auto r = run(f, p, first, o);
  REQUIRE(r.workspaceLimitCheckpoint_);
  REQUIRE_FALSE(r.completed_);
  equal(r.remainingLeptons_, p);
  o = {};
  o.last_ = FirstHistory + 8;
  r = run(f, p, first, o);
  REQUIRE(r.historyRangeExhausted_);
  REQUIRE(r.wavefronts_ == 0);
  equal(r.remainingLeptons_, p);
  for (std::size_t n : {1, 17, 257})
    for (int pid : {11, -11, 13, -13}) compare(f, particles(n, pid));
}
TEST_CASE("Lepton cascade pending prefix is consumed exactly once", "[lepton_cascade]") {
  ensure_runtime();
  Fixture f;
  pending_type queue;
  auto old = particles(5);
  auto fresh = particles(2, -11);
  auto data = upload(old);
  REQUIRE(queue.tryAppend(data, old.size(), 100, f.execution_));
  Options o;
  o.leptons_ = &queue;
  o.pendingInput_ = 3;
  Budget budget;
  budget.reject_ = 1;
  o.budget_ = budget.gate();
  std::optional<cp::FirstInteractionRecord> first;
  auto result = run(f, fresh, first, o);
  auto expected = std::vector<cp::ParticleState>(old.begin(), old.begin() + 3);
  expected.insert(expected.end(), fresh.begin(), fresh.end());
  equal(result.cpuSpillParticles_, expected);
  equal(queue.downloadPrefix(2, f.execution_),
        std::vector<cp::ParticleState>(old.begin() + 3, old.end()));
  REQUIRE(queue.size() == 2);
  o.budget_ = {};
  o.pendingInput_ = 2;
  result = run(f, {}, first, o);
  REQUIRE(queue.empty());
  REQUIRE(result.fallbackEvents_.size() == 2);
  equal(result.fallbackEvents_[0].particle_, old[3]);
  equal(result.fallbackEvents_[1].particle_, old[4]);
}
TEST_CASE("Lepton accumulation retains separate material-boundary accounting",
          "[lepton_cascade]") {
  ensure_runtime();
  execution_space_type ex;
  workspace_type w;
  w.ensureQueueCapacity(3, ex);
  w.ensureSourceCapacity(1, 3, ex);
  w.ensureOutputCapacity(1, 0, 0, 0, 0, 0, false, 3, 6, ex);
  Kokkos::deep_copy(ex, w.callStatistics_, std::uint64_t{0});
  cp::LeptonTransportRecord step;
  step.start_ = particles(1)[0];
  step.end_ = step.start_;
  step.limit_ = cp::LeptonTransportLimit::MaterialBoundary;
  auto host = Kokkos::create_mirror_view(w.transports_);
  host(0) = step;
  Kokkos::deep_copy(ex, w.transports_, host);
  auto counts = Kokkos::create_mirror_view(w.counts_);
  counts(0) = {};
  counts(0).step_ = 1;
  counts(0).next_ = 1;
  Kokkos::deep_copy(ex, w.counts_, counts);
  cp::ResidentLeptonFrontControl control;
  cp::enqueue_lepton_accumulation<execution_space_type>(w, 1, control, {}, nullptr,
                                                        nullptr, ex, 0);
  auto totals = download(w.callStatistics_);
  REQUIRE(totals(cp::LeptonCallMaterialBoundaries) == 1);
  for (unsigned i = 0; i < 8; ++i)
    REQUIRE(totals(cp::LeptonCallTransportStatisticBase + cp::LeptonTransportLimitBase +
                   i) == 0);
}
#ifdef CORSIKA_TEST_CASCADE_NATIVE
TEST_CASE("Native lepton cascades match frozen production control flow",
          "[lepton_cascade][native]") {
  ensure_runtime();
  auto tables = cp::test::make_native_tables();
  ModelOwner models;
  for (int pid : {11, -11, 13, -13}) {
    Fixture f(tables, models.models_, pid);
    for (double energy : {1., 1000., 1.e7}) {
      auto p = particles(17, pid, energy);
      p[1].energyGeV_ = std::abs(pid) == 13 ? .1056583755 : .00051099895;
      p[2].positionM_[2] = -899.999999;
      p[3].positionM_[2] = 999.999999;
      p[3].direction_[2] = 1.;
      p[4].timeS_ = .1;
      p[5].pid_ = 999;
      for (bool project : {false, true}) {
        Options o;
        o.project_ = project;
        o.air_ = !project;
        compare(f, p, o);
      }
    }
    for (int process :
         {cp::BremsProcessId, cp::IonizationProcessId, cp::ElectronPairProcessId}) {
      Fixture selected(tables, models.models_, pid, process);
      auto r = compare(selected, particles(257, pid));
      if (process == cp::ElectronPairProcessId) REQUIRE_FALSE(r.fallbackEvents_.empty());
    }
  }
  Fixture positron(tables, models.models_, -11, cp::AnnihilationProcessId, true);
  compare(positron, particles(257, -11, 1.));
  // Isolate the generic density-domain contract. Air cross-section tables
  // intentionally remain fixed; this is not a physical ice-shower fixture.
  Fixture firn(tables, models.models_);
  cp::EnvironmentNode nodes[2]{};
  nodes[0].childCount_ = 1;
  nodes[1].parentIndex_ = 0;
  nodes[1].modelIndex_ = 0;
  cp::EnvironmentShape shapes[2]{};
  shapes[0].kind_ = cp::EnvironmentShape::Kind::Universe;
  shapes[1].kind_ = cp::EnvironmentShape::Kind::Sphere;
  shapes[1].radiusM_ = 1000.;
  int children[1]{1};
  cp::EnvironmentModel density;
  density.mediumId_ = 0;
  density.densityKind_ = cp::EnvironmentModel::Density::ExponentialFirn;
  density.referenceDensityGPerCm3_ = .92;
  density.densityDeficitGPerCm3_ = .4;
  density.scaleLengthM_ = 20.;
  density.inverseScaleLengthPerM_ = .05;
  density.axis_[2] = 1.;
  firn.environment_ = cp::EnvironmentStorage<execution_space_type>(
      {nodes, shapes, children, nullptr, &density, nullptr, 2, 1, 0, 1, 0});
  cp::TransportGeometryConfig geometry;
  geometry.observation_.pointM_[2] = -900.;
  geometry.observation_.normal_[2] = 1.;
  auto const& m = models.models_;
  firn.context_.initialize(cp::InteractionPhysicsView(cp::ContinuousPhysicsView(
                               firn.tables_.getDeviceView(), .4, 10.)),
                           cp::EnvironmentBorrowedView(firn.environment_.getDataView()),
                           geometry, m.electron_, m.muon_, m.interpolation_, true, {},
                           m.lpm_, {}, firn.projection_, 1234567, 6789, firn.execution_);
  auto interior = particles(4);
  for (auto& p : interior) p.positionM_[2] = -200.;
  std::optional<cp::FirstInteractionRecord> interiorFirst;
  Options interiorOptions;
  interiorOptions.waves_ = 1;
  interiorOptions.air_ = false;
  auto inFirn = run(firn, interior, interiorFirst, interiorOptions);
  REQUIRE(inFirn.fallbackEvents_.empty());
  REQUIRE(inFirn.stepRecords_.size() == interior.size());
  for (auto const& step : inFirn.stepRecords_) {
    REQUIRE(step.distanceM_ > 0.);
    REQUIRE(step.traversedGrammageGPerCm2_ > 0.);
    REQUIRE(step.end_.positionM_[2] < step.start_.positionM_[2]);
  }
  // The production front does not supply an ExternalTransportBoundary.
  // A deliberately inconsistent sphere extending above the firn domain must
  // return the original particles to the host, not invent a material crossing.
  auto surface = particles(4);
  for (auto& p : surface) {
    p.positionM_[2] = -1.e-8;
    p.direction_[2] = 1.;
  }
  std::optional<cp::FirstInteractionRecord> surfaceFirst;
  Options surfaceOptions;
  surfaceOptions.waves_ = 1;
  surfaceOptions.air_ = false;
  auto interface = run(firn, surface, surfaceFirst, surfaceOptions);
  REQUIRE(interface.fallbackEvents_.size() == surface.size());
  REQUIRE(interface.processStatistics_.leptonMaterialBoundaries_ == 0);
  REQUIRE(interface.remainingLeptons_.empty());
  REQUIRE(interface.stepRecords_.empty());
  for (std::size_t i = 0; i < surface.size(); ++i) {
    auto const& fallback = interface.fallbackEvents_[i];
    CAPTURE(i, fallback.diagnosticStatus_, fallback.diagnosticValue0_);
    REQUIRE(fallback.reason_ == cp::FallbackReason::AtmosphereGrammageFailed);
    equal(fallback.particle_, surface[i]);
  }
  REQUIRE(coverage.waves_ > coverage.calls_);
  REQUIRE(coverage.photons_ > 0);
  REQUIRE(coverage.vertices_ > 0);
  REQUIRE(coverage.observations_ > 0);
  REQUIRE(coverage.process_[cp::BremsProcessId] > 0);
  REQUIRE(coverage.process_[cp::IonizationProcessId] > 0);
  REQUIRE(coverage.process_[cp::ElectronPairProcessId] == 0);
  std::ofstream counts("lepton-cascade-counts.txt");
  counts << "calls " << coverage.calls_ << "\nwaves " << coverage.waves_ << "\nsteps "
         << coverage.steps_ << "\nvertices " << coverage.vertices_ << "\nphotons "
         << coverage.photons_ << "\nfallbacks " << coverage.fallbacks_
         << "\nobservations " << coverage.observations_ << "\ndecays " << coverage.decays_
         << '\n';
  for (auto const& x : coverage.pid_)
    counts << "pid_" << x.first << ' ' << x.second << '\n';
  for (auto const& x : coverage.process_)
    counts << "process_" << x.first << ' ' << x.second << '\n';
}
TEST_CASE("Native lepton checkpoints can resume without redrawing completed fronts",
          "[lepton_cascade][native]") {
  ensure_runtime();
  auto tables = cp::test::make_native_tables();
  ModelOwner models;
  Fixture f(tables, models.models_);
  auto p = particles(17);
  std::optional<cp::FirstInteractionRecord> first, wholeFirst;
  Options o;
  o.waves_ = 3;
  auto whole = run(f, p, wholeFirst, o);
  std::vector<cp::LeptonTransportRecord> steps;
  std::vector<cp::ParticleState> photons;
  std::vector<cp::FallbackRequest> fallbacks;
  std::vector<cp::LeptonFinalStateRecord> finals;
  std::uint64_t used = 0;
  std::size_t waves = 0;
  for (int i = 0; i < 3 && !p.empty(); ++i) {
    o.waves_ = 1;
    o.first_ = FirstHistory + used;
    auto r = run(f, p, first, o);
    used += r.secondaryHistoryIdsUsed_;
    waves += r.wavefronts_;
    steps.insert(steps.end(), r.stepRecords_.begin(), r.stepRecords_.end());
    photons.insert(photons.end(), r.generatedPhotons_.begin(), r.generatedPhotons_.end());
    fallbacks.insert(fallbacks.end(), r.fallbackEvents_.begin(), r.fallbackEvents_.end());
    finals.insert(finals.end(), r.finalStateRecords_.begin(), r.finalStateRecords_.end());
    p = r.remainingLeptons_;
  }
  REQUIRE(waves == whole.wavefronts_);
  REQUIRE(used == whole.secondaryHistoryIdsUsed_);
  equal(steps, whole.stepRecords_);
  equal(photons, whole.generatedPhotons_);
  equal(fallbacks, whole.fallbackEvents_);
  equal(finals, whole.finalStateRecords_);
  equal(p, whole.remainingLeptons_);
  photon_cascade_test::equal(first, wholeFirst);
  o = {};
  o.minimum_ = 1000000;
  auto r = run(f, particles(17), first, o);
  REQUIRE(r.wavefronts_ == 1);
  REQUIRE(r.belowMinimumBatchCheckpoint_);
}
TEST_CASE("Lepton profile and radio ownership matches explicit record replay",
          "[lepton_cascade][native]") {
  ensure_runtime();
  auto tables = cp::test::make_native_tables();
  ModelOwner models;
  Fixture f(tables, models.models_);
  auto p = particles(17);
  std::optional<cp::FirstInteractionRecord> a, b, c;
  auto host = run(f, p, a);
  profile_type actual, expected;
  actual.initialize(f.profile_, f.execution_);
  expected.initialize(f.profile_, f.execution_);
  radio_type radio, replay;
  auto config = radio_config();
  radio.initialize(config, {}, f.execution_);
  replay.initialize(config, {}, f.execution_);
  Options o;
  o.profile_ = &actual;
  o.radio_ = &radio;
  auto resident = run(f, p, b, o);
  REQUIRE(resident.stepRecords_.empty());
  REQUIRE(resident.finalStateRecords_.empty());
  equal(host.remainingLeptons_, resident.remainingLeptons_);
  equal(host.generatedPhotons_, resident.generatedPhotons_);
  // inputIndex is local to a front. Preserve front boundaries when replaying
  // final-state records; concatenating fronts would alias unrelated indices.
  Options one;
  one.waves_ = 1;
  for (unsigned round = 0; round < 3 && !p.empty(); ++round) {
    auto batch = run(f, p, c, one);
    auto steps = upload(batch.stepRecords_);
    auto records = upload(batch.finalStateRecords_);
    expected.accumulateLepton(f.projection_, steps, batch.stepRecords_.size(), records,
                              batch.finalStateRecords_.size(), f.execution_);
    replay.accumulateLeptonTracks(steps, batch.stepRecords_.size(), f.execution_);
    f.execution_.fence();
    one.first_ += batch.secondaryHistoryIdsUsed_;
    p = batch.remainingLeptons_;
  }
  auto pa = actual.downloadFixed("cascade", cp::CooperativeEndpoint::Cuda, f.execution_);
  auto pe =
      expected.downloadFixed("cascade", cp::CooperativeEndpoint::Cuda, f.execution_);
  REQUIRE(pa.histograms_ == pe.histograms_);
  photon_cascade_test::equal(pa.counters_, pe.counters_);
  auto ra = radio.downloadFixed("cascade", cp::CooperativeEndpoint::Cuda, f.execution_);
  auto re = replay.downloadFixed("cascade", cp::CooperativeEndpoint::Cuda, f.execution_);
  REQUIRE(ra.coreas_ == re.coreas_);
  REQUIRE(ra.zhs_ == re.zhs_);
  REQUIRE(radio.statistics().leptonTracks_ == host.stepRecords_.size());
  REQUIRE(radio.statistics().coreasContributions_ > 0);
  REQUIRE(radio.statistics().zhsContributions_ > 0);
}
TEST_CASE("Generated photons use pending queue or lossless CPU spill",
          "[lepton_cascade][native]") {
  ensure_runtime();
  auto tables = cp::test::make_native_tables();
  ModelOwner models;
  Fixture f(tables, models.models_, -11, cp::AnnihilationProcessId);
  auto p = particles(257, -11, 1.);
  std::optional<cp::FirstInteractionRecord> a, b, c;
  auto host = run(f, p, a);
  REQUIRE_FALSE(host.generatedPhotons_.empty());
  pending_type queue;
  Options o;
  o.photons_ = &queue;
  auto pending = run(f, p, b, o);
  REQUIRE(pending.generatedPhotons_.empty());
  REQUIRE(pending.cpuSpillParticles_.empty());
  equal(queue.downloadPrefix(queue.size(), f.execution_), host.generatedPhotons_);
  pending_type limited;
  o.photons_ = &limited;
  o.pendingMaximum_ = 0;
  auto spill = run(f, p, c, o);
  REQUIRE(limited.empty());
  REQUIRE(spill.generatedPhotons_.empty());
  equal(spill.cpuSpillParticles_, host.generatedPhotons_);
  equal(spill.remainingLeptons_, host.remainingLeptons_);
  REQUIRE(spill.secondaryHistoryIdsUsed_ == host.secondaryHistoryIdsUsed_);
}
#endif
