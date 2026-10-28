/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/em/execution/PhotonFront.hpp>
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <tests/kokkos/em/finalstate/PhotonFinalStateChecks.hpp>
#include <tests/kokkos/em/transport/TransportChecks.hpp>
#ifdef CORSIKA_TEST_FRONT_NATIVE
#include <tests/kokkos/em/models/AuxiliaryModelFixture.hpp>
#include <tests/kokkos/em/tables/NativeTableFixture.hpp>
#endif
#include <array>
#include <iostream>
#include <map>

namespace {
  namespace cp = corsika::kokkos;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = execution_space_type::memory_space;
#ifdef CORSIKA_EXPECT_CUDA
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  void ensure_runtime() { static cp::Runtime runtime; }

  template <class T>
  auto upload(std::vector<T> const& values) {
    Kokkos::View<T*, memory_space_type> device("front test inputs", values.size());
    auto host = Kokkos::create_mirror_view(device);
    for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
    Kokkos::deep_copy(device, host);
    return device;
  }
  template <class T>
  auto download(T const& view) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  }
  template <class T>
  void equal(T const& a, T const& b) {
    photon_final_test::Payload x, y;
    photon_final_test::flatten(x, a);
    photon_final_test::flatten(y, b);
    photon_final_test::check(x, y);
  }
  struct Oracle {
    cp::InteractionSelectionOutcome selection_{};
    cp::PhotonTransportOutcome transport_{};
    cp::PhotonFinalStateClassification final_{};
    cp::PhotonFinalStateMaterialization output_{};
    cp::ProjectedStepRecord projected_{};
    unsigned stage_{};
    bool interaction_{};
  };
  struct Coverage {
    std::array<std::size_t, 4> fallback_{};
    std::map<int, std::size_t> processes_;
    std::size_t steps_{}, observations_{}, next_{}, charged_{}, snapshots_{}, fronts_{};
  };
  Coverage coverage;

  void run_front(cp::ProposalNativeTableSet const& table,
                 cp::PhotonPairLpmSnapshot const& lpm, std::size_t count, double energy,
                 bool badGeometry = false, bool project = true, int process = 0,
                 bool thin = false, double density = .001) {
    execution_space_type ex;
    cp::NativeTableStorage<execution_space_type> tables;
    cp::ProposalNativeDeviceView native;
    if (!table.dndxColumns_.empty()) {
      tables.initialize(table);
      native = tables.getDeviceView();
    }
    std::vector<std::uint32_t> indices = table.selectionColumnIndices_;
    if (process) {
      auto found = std::find_if(indices.begin(), indices.end(), [&](auto i) {
        return table.dndxColumns_[i].pdgId_ == 22 &&
               table.dndxColumns_[i].processId_ == process;
      });
      REQUIRE(found != indices.end());
      auto chosen = *found;
      // Controlled routing fixture: preserve the full index count and all native
      // polynomial data, redirect other photon entries to the selected channel.
      // Duplicate routing weights are deliberate; this is not a shower distribution test.
      for (auto& i : indices)
        if (table.dndxColumns_[i].pdgId_ == 22 &&
            table.dndxColumns_[i].processId_ != process)
          i = chosen;
    }
    auto selected = upload(indices);
    if (process) {
      native = {native.getDndxColumns(),
                native.getTotalRateColumns(),
                cp::NativeArrayView<std::uint32_t, std::uint32_t>{selected},
                native.getDedxColumns(),
                native.getDedxAccumulationIndices(),
                native.getUtilityColumns(),
                native.getBicubicValues(),
                native.getBicubicDerivativeEnergy(),
                native.getBicubicDerivativeLoss(),
                native.getBicubicMixedDerivative(),
                native.getBicubicPolynomialCoefficients(),
                native.getCubicValues(),
                native.getCubicNodeDerivatives(),
                native.getConstants()};
    }
    auto support = upload(std::vector<double>{0., 50., 100., 150., 200.});
    cp::ProfileProjectionView projection;
    projection.axisStartPositionM_[2] = 1000.;
    projection.axisDirection_[2] = -1.;
    projection.axisStepLengthM_ = 500.;
    projection.axisSupportCount_ = 5;
    projection.axisGrammageGPerCm2_ = support.data();
    cp::EnvironmentNode nodes[2]{};
    nodes[0].childCount_ = 1;
    nodes[1].parentIndex_ = 0;
    nodes[1].modelIndex_ = 0;
    cp::EnvironmentShape shapes[2]{};
    shapes[0].kind_ = cp::EnvironmentShape::Kind::Universe;
    shapes[1].kind_ = cp::EnvironmentShape::Kind::Sphere;
    shapes[1].radiusM_ = 1000.;
    std::int32_t children[1]{1};
    cp::EnvironmentModel model{};
    model.densityKind_ = cp::EnvironmentModel::Density::Homogeneous;
    model.referenceDensityGPerCm3_ = density;
    model.mediumId_ = 0;
    cp::EnvironmentStorage<execution_space_type> environment(
        {nodes, shapes, children, nullptr, &model, nullptr, 2, 1, 0, 1, 0});
    cp::PhysicsContext<execution_space_type> context;
    cp::TransportGeometryConfig geometry;
    geometry.observation_.pointM_[2] = -500.;
    geometry.observation_.normal_[2] = 1.;
    context.initialize(
        cp::InteractionPhysicsView(cp::ContinuousPhysicsView(native, .4, 10.)),
        badGeometry ? cp::EnvironmentBorrowedView{}
                    : cp::EnvironmentBorrowedView(environment.getDataView()),
        geometry, {}, {}, {}, false, lpm, {},
        thin ? cp::EmThinningConfig{1.e10, 3., 1, 1} : cp::EmThinningConfig{}, projection,
        1234567, 6789, ex);
    auto physics = context.getDeviceView();
    std::vector<cp::ParticleState> particles(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& p = particles[i];
      p.pid_ = 22;
      p.mediumId_ = 0;
      p.environmentNode_ = 1;
      p.energyGeV_ = energy;
      p.direction_[2] = -1.;
      p.weight_ = 1. + i % 3;
      p.historyId_ = (std::uint64_t{1} << 54) + i;
      p.parentHistoryId_ = (std::uint64_t{1} << 53) + i;
      p.stepId_ = (std::uint64_t{1} << 35) + i;
      p.generation_ = i == 0 ? 0 : 2;
      switch (i % 11) {
        case 1:
          p.energyGeV_ = 1.e-9;
          break;
        case 2:
          p.pid_ = 999;
          break;
        case 3:
          p.direction_[2] = NAN;
          break;
        case 4:
          p.positionM_[2] = -499.999999;
          break;
        case 5:
          p.positionM_[2] = 999.999999;
          p.direction_[2] = 1.;
          break;
        case 6:
          p.timeS_ = .1;
          break;
        case 7:
          p.energyGeV_ = NAN;
          break;
        default:
          break;
      }
    }
    auto input = upload(particles);
    Kokkos::View<Oracle*, memory_space_type> oracle("scalar photon oracle", count);
    // Preserve production kernel boundaries: a fused oracle can retain
    // intermediates and contract arithmetic across the stored-input boundary.
    // Output offsets and compaction are still computed independently below.
    Kokkos::parallel_for(
        "independent per-particle selection",
        Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          Oracle o;
          auto const& c = physics();
          o.selection_ = cp::select_discrete_interaction(c.physics_, input(i), i,
                                                         c.randomSeed_, c.showerId_);
          if (o.selection_.fallbackFlag_) o.stage_ = 1;
          oracle(i) = o;
        });
    Kokkos::parallel_for(
        "independent per-particle transport",
        Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          auto& o = oracle(i);
          if (o.stage_ != 0) return;
          auto const& c = physics();
          o.transport_ = cp::transport_photon(c.environment_, c.geometry_,
                                              o.selection_.interaction_);
          if (o.transport_.fallbackFlag_)
            o.stage_ = 2;
          else
            o.interaction_ =
                o.transport_.record_.limit_ == cp::PhotonTransportLimit::Interaction;
        });
    Kokkos::parallel_for(
        "independent per-particle classification",
        Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          auto& o = oracle(i);
          auto const& c = physics();
          if (o.interaction_) {
            o.final_ = cp::classify_photon_final_state(
                c.physics_.getContinuousView(), c.photonPairLpm_, c.thinning_,
                o.transport_.record_.interaction_, c.randomSeed_, c.showerId_);
            if (o.final_.fallbackFlag_) o.stage_ = 3;
          }
          if (!o.selection_.fallbackFlag_ && !o.transport_.fallbackFlag_)
            o.projected_ =
                cp::project_photon_step(c.profileProjection_, o.transport_.record_);
        });
    ex.fence("scalar classifications");
    auto expected = download(oracle);
    std::vector<std::uint64_t> offsets(count);
    std::uint64_t childrenTotal = 0;
    for (std::size_t i = 0; i < count; ++i) {
      offsets[i] = childrenTotal;
      childrenTotal += expected(i).final_.childCount_;
    }
    auto childOffsets = upload(offsets);
    std::uint64_t constexpr firstHistory = std::uint64_t{1} << 60;
    Kokkos::parallel_for(
        "independent aggregate materialization",
        Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          if (oracle(i).interaction_)
            oracle(i).output_ = cp::materialize_photon_final_state(
                oracle(i).transport_.record_.interaction_, oracle(i).final_,
                childOffsets(i), firstHistory);
        });
    ex.fence("scalar materialization");
    expected = download(oracle);
    cp::PhotonFrontWorkspace<execution_space_type> workspace;
    auto const capacity = std::max<std::size_t>(1, count);
    workspace.ensureCapacity(capacity, ex);
    workspace.ensureCandidateCapacity(count, ex);
    workspace.queue_.upload(particles, ex);
    Kokkos::deep_copy(ex, workspace.error_, 0u);
    Kokkos::deep_copy(ex, workspace.firstInteractionCandidates_, 0u);
    cp::enqueue_photon_front(physics, workspace.queue_.current().rawDeviceView(),
                             workspace.queue_.next().rawDeviceView(), workspace, count,
                             capacity, firstHistory, project, true, ex, 7);
    Kokkos::deep_copy(ex, workspace.hostFrontControl_, workspace.frontControl_);
    ex.fence("photon front checkpoint");
    workspace.queue_.markExecutionSynchronized();
    auto const control = workspace.hostFrontControl_();
    REQUIRE(control.materializationError_ == 0);
    REQUIRE(control.totals_.values_[cp::PhotonChildOffset] == childrenTotal);
    auto steps = download(workspace.steps_);
    auto records = download(workspace.records_);
    auto charged = download(workspace.charged_);
    auto fallbacks = download(workspace.fallbacks_);
    auto observations = download(workspace.observations_);
    auto candidates = download(workspace.firstInteractionCandidates_);
    auto snapshots = download(workspace.firstSnapshots_);
    auto compacted = download(workspace.interactionSources_);
    auto classifications = download(workspace.finalStates_);
    auto projected = download(workspace.projectedSteps_);
    std::vector<cp::FallbackRequest> fallbackGroups[4];
    std::vector<cp::ParticleState> next;
    std::size_t stepIndex = 0, recordIndex = 0, chargedIndex = 0, observedIndex = 0,
                interactionIndex = 0, snapshotCount = 0;
    for (std::size_t i = 0; i < count; ++i) {
      auto const& o = expected(i);
      INFO("source " << i << ", energy " << energy << ", count " << count << ", process "
                     << process << ", thin " << thin << ", density " << density);
      if (o.stage_ == 1)
        fallbackGroups[1].push_back(o.selection_.fallback_);
      else if (o.stage_ == 2)
        fallbackGroups[2].push_back(o.transport_.fallback_);
      else {
        transport_test::check(transport_test::flatten(steps(stepIndex)),
                              transport_test::flatten(o.transport_.record_));
        if (project) {
          auto const v = o.projected_;
          auto const& a = projected(stepIndex);
          REQUIRE(a.historyId_ == v.historyId_);
          REQUIRE(a.pid_ == v.pid_);
          REQUIRE(a.processId_ == v.processId_);
          REQUIRE(a.transportLimit_ == v.transportLimit_);
          REQUIRE(a.reserved_ == v.reserved_);
          REQUIRE(a.observationSurfaceReachedBeforeCut_ ==
                  v.observationSurfaceReachedBeforeCut_);
          REQUIRE(a.nativeMuonPairCut_ == v.nativeMuonPairCut_);
          REQUIRE(a.startGrammageGPerCm2_ == v.startGrammageGPerCm2_);
          REQUIRE(a.endGrammageGPerCm2_ == v.endGrammageGPerCm2_);
          REQUIRE(a.endEnergyGeV_ == v.endEnergyGeV_);
          REQUIRE(a.depositedEnergyGeV_ == v.depositedEnergyGeV_);
          REQUIRE(a.cutDepositedEnergyGeV_ == v.cutDepositedEnergyGeV_);
          REQUIRE(a.weight_ == v.weight_);
        }
        ++stepIndex;
        auto const& step = o.transport_.record_;
        if (step.limit_ == cp::PhotonTransportLimit::LayerBoundary)
          next.push_back(step.end_);
        if (step.limit_ == cp::PhotonTransportLimit::ObservationSurface ||
            step.limit_ == cp::PhotonTransportLimit::EscapedEnvironment ||
            (step.limit_ == cp::PhotonTransportLimit::ParticleCut &&
             step.observationSurfaceReachedBeforeCut_)) {
          equal(observations(observedIndex).particle_, step.end_);
          REQUIRE(observations(observedIndex).status_ ==
                  (step.limit_ == cp::PhotonTransportLimit::EscapedEnvironment
                       ? cp::ObservationStatus::EscapedEnvironment
                       : cp::ObservationStatus::ReachedObservationSurface));
          ++observedIndex;
        }
      }
      if (!o.interaction_) continue;
      REQUIRE(compacted(interactionIndex) == i);
      equal(classifications(interactionIndex), o.final_);
      ++interactionIndex;
      ++coverage.processes_[o.transport_.record_.interaction_.processId_];
      auto const& out = o.output_;
      REQUIRE(out.error_ == 0);
      if (out.hasFallback_) fallbackGroups[3].push_back(out.fallback_);
      if (out.hasContinuation_) next.push_back(out.continuation_.particle_);
      if (out.hasSuppression_) next.push_back(out.suppression_.particle_);
      if (out.hasRecord_) equal(records(recordIndex++), out.record_);
      for (unsigned j = 0; j < out.secondaryCount_; ++j)
        if (out.secondaries_[j].pid_ == 22)
          next.push_back(out.secondaries_[j]);
        else
          equal(charged(chargedIndex++), out.secondaries_[j]);
      if (out.hasFirstInteraction_) {
        // Production captures only secondaryCount_ valid entries in an
        // uninitialized reusable allocation; inactive slots are not output.
        auto const& a = snapshots(0);
        auto const& b = out.firstInteraction_;
        REQUIRE(a.processId_ == b.processId_);
        REQUIRE(a.secondaryCount_ == b.secondaryCount_);
        equal(a.parentAtVertex_, b.parentAtVertex_);
        for (unsigned j = 0; j < b.secondaryCount_; ++j)
          equal(a.secondaries_[j], b.secondaries_[j]);
        ++snapshotCount;
      }
    }
    REQUIRE(candidates(0) == snapshotCount);
    REQUIRE(control.interactionCount_ == interactionIndex);
    REQUIRE(control.totals_.values_[cp::PhotonStepOffset] == stepIndex);
    REQUIRE(control.totals_.values_[cp::PhotonRecordOffset] == recordIndex);
    REQUIRE(control.totals_.values_[cp::PhotonChargedOffset] == chargedIndex);
    REQUIRE(control.totals_.values_[cp::PhotonNextOffset] == next.size());
    REQUIRE(control.totals_.values_[cp::PhotonObservationOffset] == observedIndex);
    std::size_t fallbackIndex = 0;
    for (unsigned s = 1; s <= 3; ++s) {
      REQUIRE(control.totals_.values_[cp::PhotonSelectionFallbackOffset + s - 1] ==
              fallbackGroups[s].size());
      coverage.fallback_[s] += fallbackGroups[s].size();
      for (auto const& value : fallbackGroups[s])
        equal(fallbacks(fallbackIndex++), value);
    }
    workspace.queue_.commitNextAlreadySynchronized(next.size());
    auto actual = workspace.queue_.download(ex);
    REQUIRE(actual.size() == next.size());
    for (std::size_t i = 0; i < next.size(); ++i) equal(actual[i], next[i]);
    coverage.steps_ += stepIndex;
    coverage.observations_ += observedIndex;
    coverage.next_ += next.size();
    coverage.charged_ += chargedIndex;
    coverage.snapshots_ += snapshotCount;
    ++coverage.fronts_;
  }
} // namespace

TEST_CASE("Photon workspace growth preserves capacities and checkpoint aliases",
          "[photon_front]") {
  ensure_runtime();
  execution_space_type ex;
  cp::PhotonFrontWorkspace<execution_space_type> w;
  REQUIRE(w.deviceBytes() == 0);
  REQUIRE_THROWS_AS(w.ensureCapacity(0, ex), std::invalid_argument);
  REQUIRE_THROWS_AS(w.projectedCapacity(0), std::invalid_argument);
  for (std::size_t n : {1, 3, 256, 257, 17}) {
    auto const p = w.projectedCapacity(n);
    w.ensureCapacity(n, ex);
    REQUIRE(w.deviceBytes() == p.retainedBytes_);
    REQUIRE(p.transientPeakBytes_ >= p.retainedBytes_);
    auto const q = w.projectedCandidateCapacity(n);
    w.ensureCandidateCapacity(n, ex);
    REQUIRE(w.deviceBytes() == q.retainedBytes_);
    auto const* bytes = reinterpret_cast<unsigned char*>(w.frontControl_.data());
    REQUIRE(reinterpret_cast<unsigned char*>(w.scanTotals_.data()) ==
            bytes + offsetof(cp::ResidentPhotonFrontControl, totals_));
    REQUIRE(reinterpret_cast<unsigned char*>(w.interactionCount_.data()) ==
            bytes + offsetof(cp::ResidentPhotonFrontControl, interactionCount_));
    REQUIRE(reinterpret_cast<unsigned char*>(w.error_.data()) ==
            bytes + offsetof(cp::ResidentPhotonFrontControl, materializationError_));
    w.ensureHostOutputCapacity(n, n, n, n, n, n);
    REQUIRE(w.hostSteps_.extent(0) >= n);
  }
  REQUIRE_THROWS_AS(w.projectedCapacity(std::numeric_limits<std::size_t>::max()),
                    std::length_error);
}

TEST_CASE("Photon front retains early cuts and selection failures", "[photon_front]") {
  ensure_runtime();
  for (std::size_t count : {0, 1, 255, 256, 257}) run_front({}, {}, count, 1.);
}

#ifdef CORSIKA_TEST_FRONT_NATIVE
TEST_CASE("Real native photon channels preserve stable output and fallback ledgers",
          "[photon_front][native]") {
  ensure_runtime();
  auto const original = cp::test::make_native_tables();
  auto const lpm = cp::test::make_auxiliary_dataset().snapshots_.front().photon_;
  for (int process :
       {cp::PhotonPairProcessId, cp::ComptonProcessId, cp::PhotoelectricProcessId,
        cp::PhotonMuonPairProcessId, cp::PhotoproductionProcessId}) {
    for (double energy : {.002, 1., 1000., 1.e4})
      run_front(original, lpm, 257, energy, false, true, process);
  }
  run_front(original, lpm, 257, 10., true);
  run_front(original, lpm, 257, 10., false, false);
  run_front(original, lpm, 257, 1.e4, false, true, cp::PhotonPairProcessId, true);
  run_front(original, lpm, 257, 1.e4, false, true, cp::ComptonProcessId, true);
  run_front(original, lpm, 257, 1.e4, false, true, cp::PhotonPairProcessId, false, 1.e6);
  for (unsigned s = 1; s <= 3; ++s) {
    INFO("fallback stage " << s << ": " << coverage.fallback_[1] << ","
                           << coverage.fallback_[2] << "," << coverage.fallback_[3]);
    REQUIRE(coverage.fallback_[s] > 0);
  }
  for (int p : {cp::PhotonPairProcessId, cp::ComptonProcessId, cp::PhotoelectricProcessId,
                cp::PhotonMuonPairProcessId, cp::PhotoproductionProcessId})
    REQUIRE(coverage.processes_[p] > 0);
  REQUIRE(coverage.snapshots_ > 0);
  REQUIRE(coverage.charged_ > 0);
  REQUIRE(coverage.next_ > 0);
  REQUIRE(coverage.observations_ > 0);
  std::cout << "photon fronts=" << coverage.fronts_ << " steps=" << coverage.steps_
            << " next=" << coverage.next_ << " charged=" << coverage.charged_
            << " observations=" << coverage.observations_
            << " fallback=" << coverage.fallback_[1] << "," << coverage.fallback_[2]
            << "," << coverage.fallback_[3] << "\n";
}
#endif
