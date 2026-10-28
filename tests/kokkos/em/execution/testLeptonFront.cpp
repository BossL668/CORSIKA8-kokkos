/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/em/execution/LeptonFront.hpp>
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <tests/kokkos/em/finalstate/LeptonFinalStateChecks.hpp>
#include <tests/kokkos/em/transport/TransportChecks.hpp>
#ifdef CORSIKA_TEST_FRONT_NATIVE
#include <tests/kokkos/em/models/AuxiliaryModelFixture.hpp>
#include <tests/kokkos/em/tables/NativeTableFixture.hpp>
#include <tests/kokkos/em/moliere/MoliereQueryTestData.hpp>
#include <corsika/kokkos/em/execution/MoliereInterpolationStorage.hpp>
#endif
#include <array>
#include <iostream>
#include <map>
#include <iomanip>
#include <typeinfo>
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
    Kokkos::View<T*, memory_space_type> device("lepton front inputs", values.size());
    auto host = Kokkos::create_mirror_view(device);
    for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
    Kokkos::deep_copy(device, host);
    return device;
  }
  template <class T>
  auto download(T const& view) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  }
  std::string diagnosticContext;
  template <class T>
  void equal(T const& a, T const& b) {
    lepton_final_test::Payload x, y;
    lepton_final_test::flatten(x, a);
    lepton_final_test::flatten(y, b);
    for (unsigned i = 0; i < x.reals_; ++i) {
      if (x.real_[i] != y.real_[i] &&
          !(std::isnan(x.real_[i]) && std::isnan(y.real_[i]))) {
        std::cerr << "lepton mismatch " << diagnosticContext << " type "
                  << typeid(T).name() << " field " << i << std::setprecision(17)
                  << " actual " << x.real_[i] << " expected " << y.real_[i] << "\n";
        break;
      }
    }
    lepton_final_test::check(x, y);
  }
  struct Oracle {
    cp::InteractionSelectionOutcome selection_{};
    cp::LeptonTransportRecord transport_{};
    cp::FallbackRequest transportFallback_{};
    cp::LeptonVertexSelectionOutcome vertex_{};
    cp::LeptonFinalStateClassification final_{};
    cp::LeptonFinalStateMaterialization output_{};
    cp::ProjectedStepRecord projected_{};
    unsigned stage_{}, state_{};
    bool interaction_{};
  };
  struct Coverage {
    std::array<std::size_t, 5> fallback_{};
    std::map<int, std::size_t> process_, limit_, pid_;
    std::size_t steps_{}, photons_{}, next_{}, observations_{}, decays_{}, snapshots_{},
        fronts_{}, deflections_{};
    std::size_t electronPairFallbacks_{}, muonPairFallbacks_{}, suppressions_{};
  } coverage;
  struct Models {
    cp::MoliereSnapshot electron_{}, muon_{};
    cp::MoliereInterpolationView interpolation_{};
    cp::BremsLpmSnapshot lpm_{};
  };

  void run_front(cp::ProposalNativeTableSet const& table, Models const& models,
                 std::size_t count, int pid, double energy, int process = 0,
                 bool air = true, bool thin = false, bool badGeometry = false,
                 bool muonAvailable = true, bool project = true, double density = .001,
                 bool restrictHistory = false) {
    execution_space_type ex;
    cp::NativeTableStorage<execution_space_type> tables;
    cp::ProposalNativeDeviceView native;
    if (!table.dndxColumns_.empty()) {
      tables.initialize(table);
      native = tables.getDeviceView();
    }
    std::vector<std::uint32_t> indices = table.selectionColumnIndices_;
    if (process) {
      auto it = std::find_if(indices.begin(), indices.end(), [&](auto i) {
        return table.dndxColumns_[i].pdgId_ == pid &&
               table.dndxColumns_[i].processId_ == process;
      });
      REQUIRE(it != indices.end());
      auto chosen = *it;
      // Routing-only fixture. Preserve native polynomial values; repeated column
      // entries force rare channels and are not a cross-section distribution test.
      for (auto& i : indices)
        if (table.dndxColumns_[i].pdgId_ == pid) i = chosen;
    }
    auto selected = upload(indices);
    if (process)
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
    auto support = upload(std::vector<double>{0., 50., 100., 150., 200.});
    cp::ProfileProjectionView projection;
    projection.axisStartPositionM_[2] = 1000.;
    projection.axisDirection_[2] = -1.;
    projection.axisStepLengthM_ = 500.;
    projection.axisSupportCount_ = 5;
    projection.axisGrammageGPerCm2_ = support.data();
    cp::TransportGeometryConfig geometry;
    geometry.observation_.pointM_[2] = -500.;
    geometry.observation_.normal_[2] = 1.;
    cp::PhysicsContext<execution_space_type> context;
    context.initialize(
        cp::InteractionPhysicsView(cp::ContinuousPhysicsView(native, .4, 10.)),
        badGeometry ? cp::EnvironmentBorrowedView{}
                    : cp::EnvironmentBorrowedView(environment.getDataView()),
        geometry, models.electron_, models.muon_, models.interpolation_, muonAvailable,
        {}, models.lpm_,
        thin ? cp::EmThinningConfig{1.e10, 3., 1, 1} : cp::EmThinningConfig{}, projection,
        1234567, 6789, ex);
    auto physics = context.getDeviceView();
    std::vector<cp::ParticleState> particles(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& p = particles[i];
      p.pid_ = pid;
      p.mediumId_ = 0;
      p.environmentNode_ = 1;
      p.energyGeV_ = energy;
      p.direction_[2] = -1.;
      p.weight_ = 1. + i % 3;
      p.historyId_ = (std::uint64_t{1} << 54) + i;
      p.parentHistoryId_ = (std::uint64_t{1} << 53) + i;
      p.stepId_ = (std::uint64_t{1} << 35) + i;
      p.generation_ = i == 0 ? 0 : 2;
      switch (i % 17) {
        case 1:
          p.energyGeV_ = std::abs(pid) == 13 ? .1056583755 : .00051099895;
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
    Kokkos::View<Oracle*, memory_space_type> oracle("independent lepton outputs", count);
    // Same physics kernel boundaries; independent host scans/output accounting.
    Kokkos::parallel_for(
        "oracle selection", Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          Oracle o;
          auto const& c = physics();
          o.selection_ = cp::select_discrete_interaction(c.physics_, input(i), i,
                                                         c.randomSeed_, c.showerId_);
          if (o.selection_.fallbackFlag_) o.stage_ = 1;
          oracle(i) = o;
        });
    Kokkos::parallel_for(
        "oracle transport", Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          auto& o = oracle(i);
          if (o.stage_) return;
          auto const& c = physics();
          o.state_ = cp::transport_lepton(
              c.physics_.getContinuousView(), true, c.environment_, c.geometry_,
              o.selection_.interaction_, o.transport_, o.transportFallback_);
        });
    Kokkos::parallel_for(
        "oracle scattering", Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          auto& o = oracle(i);
          if (o.stage_) return;
          auto const& c = physics();
          if (air)
            o.state_ = cp::apply_moliere_scattering_stage<4>(
                c.electronMoliere_, c.muonMoliere_, c.moliereInterpolation_,
                c.muonMoliereAvailable_ != 0, c.randomSeed_, c.showerId_, o.transport_,
                o.transportFallback_, o.state_);
          else
            o.state_ = cp::apply_moliere_scattering_stage<cp::MaxMoliereComponents>(
                c.electronMoliere_, c.muonMoliere_, c.moliereInterpolation_,
                c.muonMoliereAvailable_ != 0, c.randomSeed_, c.showerId_, o.transport_,
                o.transportFallback_, o.state_);
          if (o.state_)
            o.stage_ = 2;
          else
            o.interaction_ =
                o.transport_.limit_ == cp::LeptonTransportLimit::InteractionCandidate ||
                o.transport_.interaction_.status_ ==
                    cp::InteractionStatus::AtRestAnnihilation;
        });
    Kokkos::parallel_for(
        "oracle vertex and classification",
        Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          auto& o = oracle(i);
          auto const& c = physics();
          if (!o.interaction_) return;
          o.vertex_ = cp::select_lepton_vertex(c.physics_, o.transport_.interaction_,
                                               c.randomSeed_, c.showerId_);
          if (o.vertex_.fallbackFlag_)
            o.stage_ = 3;
          else if (o.vertex_.interactionFlag_) {
            o.final_ = cp::classify_lepton_final_state(
                c.bremsLpm_, c.thinning_, o.vertex_.record_, c.randomSeed_, c.showerId_);
            if (o.final_.fallbackFlag_) o.stage_ = 4;
          }
        });
    ex.fence("oracle classification complete");
    auto expected = download(oracle);
    std::vector<std::uint64_t> offsets(count);
    std::uint64_t childTotal = 0;
    for (std::size_t i = 0; i < count; ++i) {
      offsets[i] = childTotal;
      childTotal += expected(i).final_.childCount_;
    }
    auto childOffsets = upload(offsets);
    std::uint64_t constexpr firstHistory = std::uint64_t{1} << 60;
    double const mass = .00051099895;
    // Fixed per-source slots avoid all production compaction/scans. Reuse only
    // the production scalar materializer and its SoA writer: changing the writer
    // template changes CUDA floating-point contraction in two-body directions.
    cp::ParticleSoA<memory_space_type> oracleNext("oracle next", count * 3);
    auto oracleNextRaw = oracleNext.rawDeviceView();
    auto oracleRecords = upload(std::vector<cp::LeptonFinalStateRecord>(count));
    auto oraclePhotons = upload(std::vector<cp::ParticleState>(count * 2));
    auto oracleFallbacks = upload(std::vector<cp::FallbackRequest>(count));
    auto oracleSnapshots = upload(std::vector<cp::FirstInteractionRecord>(1));
    auto oracleControl = upload(std::vector<std::uint32_t>(2));
    auto oracleWritten = upload(std::vector<std::uint32_t>(count * 2));
    auto recordRaw = cp::raw_device_view(oracleRecords);
    auto photonRaw = cp::raw_device_view(oraclePhotons);
    auto fallbackRaw = cp::raw_device_view(oracleFallbacks);
    auto snapshotRaw = cp::raw_device_view(oracleSnapshots);
    auto* oracleControlRaw = oracleControl.data();
    Kokkos::parallel_for(
        "oracle materialization", Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          auto& o = oracle(i);
          if (o.vertex_.interactionFlag_) {
            cp::lepton_front_detail::ResidentLeptonFinalStateWriter writer{
                oracleNextRaw,
                recordRaw,
                photonRaw,
                fallbackRaw,
                snapshotRaw,
                oracleControlRaw,
                oracleControlRaw + 1,
                i * 3,
                i * 2,
                i,
                i,
                1};
            cp::materialize_lepton_final_state_in_place(
                o.vertex_.record_, o.final_, childOffsets(i), firstHistory, mass, writer);
            oracleWritten(2 * i) = writer.nextWritten_;
            oracleWritten(2 * i + 1) = writer.photonWritten_;
          }
        });
    auto oracleNextAoS = upload(std::vector<cp::ParticleState>(count * 3));
    Kokkos::parallel_for(
        "oracle copy next", Kokkos::RangePolicy<execution_space_type>(ex, 0, count * 3),
        KOKKOS_LAMBDA(std::size_t i) { oracleNextAoS(i) = oracleNextRaw.load(i); });
    // Production projects transport endpoints in a separate kernel from final
    // state materialization; do not fuse their register/temporary lifetimes.
    Kokkos::parallel_for(
        "oracle projection", Kokkos::RangePolicy<execution_space_type>(ex, 0, count),
        KOKKOS_LAMBDA(std::size_t i) {
          auto& o = oracle(i);
          if (o.stage_ != 1 && o.stage_ != 2)
            o.projected_ =
                cp::project_lepton_step(physics().profileProjection_, o.transport_);
        });
    ex.fence("oracle outputs complete");
    expected = download(oracle);
    auto referenceNext = download(oracleNextAoS);
    auto referencePhotons = download(oraclePhotons);
    auto referenceRecords = download(oracleRecords);
    auto referenceFallbacks = download(oracleFallbacks);
    auto referenceSnapshots = download(oracleSnapshots);
    auto referenceWritten = download(oracleWritten);
    REQUIRE(download(oracleControl)(1) == 0);
    for (std::size_t i = 0; i < count; ++i) {
      auto& o = expected(i);
      if (!o.vertex_.interactionFlag_) continue;
      auto& out = o.output_;
      auto const& f = o.final_;
      out.hasFallback_ = f.fallbackFlag_;
      out.hasContinuation_ = f.continuationFlag_;
      out.hasSuppression_ = f.suppressionFlag_;
      out.hasRecord_ = f.recordFlag_ && !f.fallbackFlag_ && !f.continuationFlag_ &&
                       !f.suppressionFlag_;
      if (out.hasFallback_) out.fallback_ = referenceFallbacks(i);
      if (out.hasContinuation_) {
        out.continuation_ = o.vertex_.record_;
        equal(referenceNext(3 * i), out.continuation_.particle_);
      }
      if (out.hasSuppression_) {
        out.suppression_.particle_ = o.vertex_.record_.particle_;
        ++out.suppression_.particle_.stepId_;
        equal(referenceNext(3 * i), out.suppression_.particle_);
      }
      if (out.hasRecord_) {
        out.record_ = referenceRecords(i);
        out.secondaryCount_ = f.childCount_;
        REQUIRE(referenceWritten(2 * i) + referenceWritten(2 * i + 1) ==
                out.secondaryCount_);
        unsigned j = 0;
        for (unsigned k = 0; k < referenceWritten(2 * i); ++k)
          out.secondaries_[j++] = referenceNext(3 * i + k);
        for (unsigned k = 0; k < referenceWritten(2 * i + 1); ++k)
          out.secondaries_[j++] = referencePhotons(2 * i + k);
        // Lepton channels write charged children before photons; annihilation
        // writes two photons. Host assembly is independent of the device scan.
        if (o.vertex_.record_.particle_.generation_ == 0) {
          out.hasFirstInteraction_ = 1;
          out.firstInteraction_ = referenceSnapshots(0);
        }
      }
    }

    cp::LeptonFrontWorkspace<execution_space_type> w;
    auto capacity = std::max<std::size_t>(1, count * 3);
    w.ensureQueueCapacity(capacity, ex);
    if (count) w.ensureSourceCapacity(count, capacity, ex);
    w.ensureOutputCapacity(count, count, count * 2, count, count, count, project,
                           capacity, capacity * 2, ex);
    w.queue_.upload(particles, ex);
    Kokkos::deep_copy(ex, w.frontControl_, cp::ResidentLeptonFrontControl{});
    Kokkos::deep_copy(ex, w.firstInteractionCandidates_, 0u);
    cp::enqueue_lepton_front<execution_space_type>(
        physics, mass, air, w.queue_.current().rawDeviceView(), w, count, capacity,
        capacity * 2, firstHistory,
        restrictHistory ? firstHistory : std::numeric_limits<std::uint64_t>::max(),
        project, true, ex, 7);
    Kokkos::deep_copy(ex, w.hostFrontControl_, w.frontControl_);
    ex.fence("front checkpoint");
    w.queue_.markExecutionSynchronized();
    auto control = w.hostFrontControl_();
    REQUIRE(control.materializationError_ == 0);
    REQUIRE(control.totals_.values_[cp::LeptonChildOffset] == childTotal);
    if (restrictHistory && childTotal) {
      REQUIRE(download(w.firstInteractionCandidates_)(0) == 0);
      return;
    }
    auto steps = download(w.steps_);
    auto records = download(w.records_);
    auto photons = download(w.photons_);
    auto fallbacks = download(w.fallbacks_);
    auto observations = download(w.observations_);
    auto decays = download(w.decays_);
    auto compacted = download(w.interactionSources_);
    auto final = download(w.finalStates_);
    auto vertices = download(w.vertices_);
    auto projected = download(w.projectedSteps_);
    auto snapshots = download(w.firstSnapshots_);
    std::vector<cp::ParticleState> next;
    std::vector<cp::FallbackRequest> groups[5];
    std::size_t si = 0, ri = 0, pi = 0, oi = 0, di = 0, ii = 0, vi = 0, ni = 0;
    for (std::size_t i = 0; i < count; ++i) {
      diagnosticContext = std::to_string(pid) + "/" + std::to_string(energy) + "/" +
                          std::to_string(process) + "/source=" + std::to_string(i);
      auto const& o = expected(i);
      INFO("source " << i << " pid " << pid << " energy " << energy << " process "
                     << process << " air " << air);
      if (o.stage_ == 1)
        groups[1].push_back(o.selection_.fallback_);
      else if (o.stage_ == 2)
        groups[2].push_back(o.transportFallback_);
      else {
        transport_test::check(transport_test::flatten(steps(si)),
                              transport_test::flatten(o.transport_));
        if (project) {
          auto const& a = projected(si);
          auto const& b = o.projected_;
          REQUIRE(a.historyId_ == b.historyId_);
          REQUIRE(a.pid_ == b.pid_);
          REQUIRE(a.processId_ == b.processId_);
          REQUIRE(a.transportLimit_ == b.transportLimit_);
          REQUIRE(a.startGrammageGPerCm2_ == b.startGrammageGPerCm2_);
          REQUIRE(a.endGrammageGPerCm2_ == b.endGrammageGPerCm2_);
          REQUIRE(a.endEnergyGeV_ == b.endEnergyGeV_);
          REQUIRE(a.depositedEnergyGeV_ == b.depositedEnergyGeV_);
          REQUIRE(a.cutDepositedEnergyGeV_ == b.cutDepositedEnergyGeV_);
          REQUIRE(a.weight_ == b.weight_);
        }
        ++si;
        auto const& s = o.transport_;
        ++coverage.limit_[int(s.limit_)];
        ++coverage.pid_[s.start_.pid_];
        coverage.deflections_ += s.multipleScatteringApplied_ != 0;
        if (s.limit_ == cp::LeptonTransportLimit::ContinuousStep ||
            s.limit_ == cp::LeptonTransportLimit::LayerBoundary ||
            s.limit_ == cp::LeptonTransportLimit::MagneticStep)
          next.push_back(s.end_);
        if (s.limit_ == cp::LeptonTransportLimit::ObservationSurface ||
            s.limit_ == cp::LeptonTransportLimit::EscapedEnvironment ||
            (s.limit_ == cp::LeptonTransportLimit::ParticleCut &&
             s.observationSurfaceReachedBeforeCut_)) {
          equal(observations(oi).particle_, s.end_);
          REQUIRE(observations(oi).status_ ==
                  (s.limit_ == cp::LeptonTransportLimit::EscapedEnvironment
                       ? cp::ObservationStatus::EscapedEnvironment
                       : cp::ObservationStatus::ReachedObservationSurface));
          ++oi;
        }
        if (s.limit_ == cp::LeptonTransportLimit::DecayCandidate)
          equal(decays(di++), s.end_);
      }
      if (!o.interaction_) continue;
      REQUIRE(compacted(ii) == i);
      equal(vertices(ii).record_, o.vertex_.record_);
      REQUIRE(vertices(ii).interactionFlag_ == o.vertex_.interactionFlag_);
      REQUIRE(vertices(ii).continuationFlag_ == o.vertex_.continuationFlag_);
      REQUIRE(vertices(ii).fallbackFlag_ == o.vertex_.fallbackFlag_);
      if (o.vertex_.interactionFlag_) {
        equal(final(ii), o.final_);
        ++vi;
      }
      ++ii;
      if (o.stage_ == 3) {
        groups[3].push_back(o.vertex_.fallback_);
        // Production routes the two direct-pair channels at vertex selection,
        // before attempting their missing asymmetry distributions.
        if (o.vertex_.fallback_.processId_ == cp::ElectronPairProcessId) {
          REQUIRE(o.vertex_.fallback_.reason_ ==
                  cp::FallbackReason::NativeSelectionReplay);
          if (std::abs(pid) == 13)
            ++coverage.muonPairFallbacks_;
          else
            ++coverage.electronPairFallbacks_;
          REQUIRE(o.vertex_.interactionFlag_ == 0);
        }
        continue;
      }
      if (o.vertex_.continuationFlag_) {
        next.push_back(o.vertex_.record_.particle_);
        continue;
      }
      REQUIRE(o.vertex_.interactionFlag_ != 0);
      ++coverage.process_[o.vertex_.record_.processId_];
      auto const& out = o.output_;
      REQUIRE(out.error_ == 0);
      if (out.hasFallback_) {
        groups[4].push_back(out.fallback_);
        if (o.vertex_.record_.processId_ == cp::ElectronPairProcessId) {
          if (std::abs(pid) == 13)
            ++coverage.muonPairFallbacks_;
          else
            ++coverage.electronPairFallbacks_;
          REQUIRE(out.secondaryCount_ == 0);
          REQUIRE(o.final_.electronPairRejectionTrials_ == 0);
        }
      }
      if (out.hasContinuation_) next.push_back(out.continuation_.particle_);
      if (out.hasSuppression_) {
        next.push_back(out.suppression_.particle_);
        ++coverage.suppressions_;
      }
      if (out.hasRecord_) equal(records(ri++), out.record_);
      for (unsigned j = 0; j < out.secondaryCount_; ++j)
        if (out.secondaries_[j].pid_ == 22)
          equal(photons(pi++), out.secondaries_[j]);
        else
          next.push_back(out.secondaries_[j]);
      if (out.hasFirstInteraction_) {
        auto const& a = snapshots(0);
        auto const& b = out.firstInteraction_;
        REQUIRE(a.processId_ == b.processId_);
        REQUIRE(a.secondaryCount_ == b.secondaryCount_);
        equal(a.parentAtVertex_, b.parentAtVertex_);
        for (unsigned j = 0; j < b.secondaryCount_; ++j)
          equal(a.secondaries_[j], b.secondaries_[j]);
        ++ni;
      }
    }
    REQUIRE(control.interactionCount_ == ii);
    REQUIRE(control.vertexInteractionCount_ == vi);
    REQUIRE(control.totals_.values_[cp::LeptonNextOffset] == next.size());
    REQUIRE(control.totals_.values_[cp::LeptonPhotonOffset] == pi);
    REQUIRE(control.totals_.values_[cp::LeptonStepOffset] == si);
    REQUIRE(control.totals_.values_[cp::LeptonRecordOffset] == ri);
    REQUIRE(control.totals_.values_[cp::LeptonObservationOffset] == oi);
    REQUIRE(control.totals_.values_[cp::LeptonDecayOffset] == di);
    REQUIRE(download(w.firstInteractionCandidates_)(0) == ni);
    std::size_t fi = 0;
    for (unsigned s = 1; s <= 4; ++s) {
      REQUIRE(control.totals_.values_[cp::LeptonSelectionFallbackOffset + s - 1] ==
              groups[s].size());
      for (auto const& v : groups[s]) equal(fallbacks(fi++), v);
      coverage.fallback_[s] += groups[s].size();
    }
    w.queue_.commitNextAlreadySynchronized(next.size());
    auto actual = w.queue_.download(ex);
    REQUIRE(actual.size() == next.size());
    for (std::size_t i = 0; i < next.size(); ++i) {
      diagnosticContext = std::to_string(pid) + "/" + std::to_string(energy) + "/" +
                          std::to_string(process) + "/next=" + std::to_string(i);
      equal(actual[i], next[i]);
    }
    coverage.steps_ += si;
    coverage.photons_ += pi;
    coverage.next_ += next.size();
    coverage.observations_ += oi;
    coverage.decays_ += di;
    coverage.snapshots_ += ni;
    ++coverage.fronts_;
  }
} // namespace
TEST_CASE("Lepton workspace separates logical capacity and growing allocations",
          "[lepton_front]") {
  ensure_runtime();
  execution_space_type ex;
  cp::LeptonFrontWorkspace<execution_space_type> w;
  REQUIRE(w.deviceBytes() == 0);
  // Check the shared writer independently of both front and scalar physics.
  cp::ParticleState known;
  known.pid_ = 11;
  known.energyGeV_ = 7.;
  known.direction_[2] = -1.;
  known.positionM_[0] = 123.;
  known.timeS_ = 4.;
  known.weight_ = 9.;
  known.historyId_ = 12345;
  known.parentHistoryId_ = 6789;
  known.stepId_ = 9876;
  known.environmentNode_ = 2;
  known.mediumId_ = 3;
  auto light = known;
  light.pid_ = 22;
  light.weight_ = 2.;
  cp::ParticleSoA<memory_space_type> knownNext("writer contract", 3);
  auto knownNextRaw = knownNext.rawDeviceView();
  auto knownRecords = upload(std::vector<cp::LeptonFinalStateRecord>(1));
  auto knownPhotons = upload(std::vector<cp::ParticleState>(1));
  auto knownFallbacks = upload(std::vector<cp::FallbackRequest>(1));
  auto knownSnapshots = upload(std::vector<cp::FirstInteractionRecord>(1));
  auto knownControl = upload(std::vector<std::uint32_t>(4));
  auto kr = cp::raw_device_view(knownRecords);
  auto kp = cp::raw_device_view(knownPhotons);
  auto kf = cp::raw_device_view(knownFallbacks);
  auto ks = cp::raw_device_view(knownSnapshots);
  auto* kc = knownControl.data();
  Kokkos::parallel_for(
      "writer contract", Kokkos::RangePolicy<execution_space_type>(ex, 0, 1),
      KOKKOS_LAMBDA(int) {
        cp::lepton_front_detail::ResidentLeptonFinalStateWriter writer{
            knownNextRaw, kr, kp, kf, ks, kc, kc + 1, 0, 0, 0, 0, 1};
        cp::InteractionRecord interaction;
        interaction.particle_ = known;
        writer.continuation(interaction);
        writer.nextOffset_ = 1;
        writer.suppression(known, interaction, {});
        writer.nextOffset_ = 2;
        writer.secondary(0, known);
        writer.secondary(1, light);
        cp::FallbackRequest fallback;
        fallback.particle_ = known;
        writer.fallback(fallback);
        writer.record().parentHistoryId_ = known.historyId_;
        writer.finishRecord(2);
        writer.firstInteraction(known, cp::BremsProcessId, known, &light);
        auto invalid = known;
        invalid.pid_ = 999;
        writer.secondary(2, invalid);
        writer.fail(9); // Preserve the first error from the invalid species.
        kc[2] = writer.nextWritten_;
        kc[3] = writer.photonWritten_;
      });
  auto copied = upload(std::vector<cp::ParticleState>(3));
  Kokkos::parallel_for(
      "copy writer contract", Kokkos::RangePolicy<execution_space_type>(ex, 0, 3),
      KOKKOS_LAMBDA(int i) { copied(i) = knownNextRaw.load(i); });
  ex.fence();
  auto knownHost = download(copied);
  for (int i = 0; i < 3; ++i) equal(knownHost(i), known);
  equal(download(knownPhotons)(0), light);
  equal(download(knownFallbacks)(0).particle_, known);
  REQUIRE(download(knownRecords)(0).parentHistoryId_ == known.historyId_);
  auto snapshot = download(knownSnapshots)(0);
  equal(snapshot.parentAtVertex_, known);
  equal(snapshot.secondaries_[0], known);
  light.weight_ = known.weight_;
  equal(snapshot.secondaries_[1], light);
  REQUIRE(snapshot.processId_ == cp::BremsProcessId);
  REQUIRE(snapshot.secondaryCount_ == 2);
  auto control = download(knownControl);
  REQUIRE(control(0) == 1);
  REQUIRE(control(1) == 2);
  REQUIRE(control(2) == 1);
  REQUIRE(control(3) == 1);

  // Non-absorbing firn interface: retain the particle rather than treating the
  // added receiving-tree limit as a cut, observation, decay or interaction.
  cp::LeptonTransportRecord boundary;
  boundary.limit_ = cp::LeptonTransportLimit::MaterialBoundary;
  cp::ResidentLeptonSourceCounts boundaryCounts;
  cp::lepton_front_detail::classify_lepton_transport_limit(boundary, boundaryCounts);
  REQUIRE(boundaryCounts.step_ == 1);
  REQUIRE(boundaryCounts.next_ == 1);
  REQUIRE(boundaryCounts.observation_ == 0);
  REQUIRE(boundaryCounts.decay_ == 0);
  REQUIRE(boundaryCounts.fallbackStage_ == cp::LeptonNoFallback);
  REQUIRE_THROWS_AS(w.ensureQueueCapacity(0, ex), std::invalid_argument);
  REQUIRE_THROWS_AS(w.ensureSourceCapacity(0, 1, ex), std::length_error);
  REQUIRE_THROWS_AS(w.projectedSourceCapacity(2, 1), std::length_error);
  for (std::size_t n : {1, 3, 256, 257, 17}) {
    auto q = w.projectedQueueCapacity(n * 3);
    w.ensureQueueCapacity(n * 3, ex);
    REQUIRE(w.deviceBytes() == q.retainedBytes_);
    auto s = w.projectedSourceCapacity(n, 1000);
    w.ensureSourceCapacity(n, 1000, ex);
    REQUIRE(w.deviceBytes() == s.retainedBytes_);
    auto o = w.projectedOutputCapacity(n, n, n * 2, n, n, n, true, 1000, 2000);
    w.ensureOutputCapacity(n, n, n * 2, n, n, n, true, 1000, 2000, ex);
    REQUIRE(w.deviceBytes() == o.retainedBytes_);
    auto const* bytes = reinterpret_cast<unsigned char*>(w.frontControl_.data());
    REQUIRE(reinterpret_cast<unsigned char*>(w.scanTotals_.data()) ==
            bytes + offsetof(cp::ResidentLeptonFrontControl, totals_));
    REQUIRE(reinterpret_cast<unsigned char*>(w.interactionCount_.data()) ==
            bytes + offsetof(cp::ResidentLeptonFrontControl, interactionCount_));
    REQUIRE(reinterpret_cast<unsigned char*>(w.vertexInteractionCount_.data()) ==
            bytes + offsetof(cp::ResidentLeptonFrontControl, vertexInteractionCount_));
    REQUIRE(reinterpret_cast<unsigned char*>(w.error_.data()) ==
            bytes + offsetof(cp::ResidentLeptonFrontControl, materializationError_));
    w.ensureHostOutputCapacity(n, n, n, n * 2, n, n, n, 1000, 2000);
    REQUIRE(w.hostSteps_.extent(0) >= n);
  }
  REQUIRE_THROWS_AS(w.projectedOutputCapacity(1001, 0, 0, 0, 0, 0, false, 1000, 2000),
                    std::length_error);
}
TEST_CASE("Lepton front preserves cuts and unavailable-input failures",
          "[lepton_front]") {
  ensure_runtime();
  for (int pid : {11, -11, 13, -13})
    for (std::size_t n : {0, 1, 255, 256, 257}) run_front({}, {}, n, pid, 1.);
}
#ifdef CORSIKA_TEST_FRONT_NATIVE
TEST_CASE("Native lepton fronts preserve physics outputs and four fallback stages",
          "[lepton_front][native]") {
  ensure_runtime();
  auto table = cp::test::make_native_tables();
  auto aux = cp::test::make_auxiliary_dataset();
  auto moliere = cp::test::make_moliere_dataset();
  cp::MoliereInterpolationStorage<execution_space_type> interpolation;
  interpolation.initialize(moliere.snapshots_[0], moliere.cacheFile_);
  Models models{moliere.snapshots_[0], moliere.snapshots_[1],
                interpolation.getDeviceView(), aux.snapshots_.front().electron_};
  auto missingLpm = models;
  missingLpm.lpm_.electronMassMeV_ = 0.;
  run_front(table, missingLpm, 257, 11, 1000., cp::BremsProcessId);
  for (int pid : {11, -11, 13, -13}) {
    for (double energy : {1., 1000., 1.e7})
      for (bool air : {true, false}) run_front(table, models, 257, pid, energy, 0, air);
    for (int process :
         {cp::BremsProcessId, cp::IonizationProcessId, cp::ElectronPairProcessId})
      run_front(table, models, 1024, pid, 1000., process);
  }
  run_front(table, models, 257, -11, 1., cp::AnnihilationProcessId);
  run_front(table, models, 257, 11, 10., 0, true, false, true);
  run_front(table, models, 257, 13, 10., 0, true, false, false, false);
  run_front(table, models, 257, 11, 1.e4, cp::BremsProcessId, true, true);
  run_front(table, models, 257, -11, 1.e4, cp::AnnihilationProcessId, true, true);
  run_front(table, models, 257, 11, 1.e7, cp::BremsProcessId, true, false, false, true,
            true, 1.e6);
  run_front(table, models, 257, 11, 1000., 0, true, false, false, true, false);
  run_front(table, models, 257, 11, 1000., cp::BremsProcessId, true, false, false, true,
            true, .001, true);
  std::cout << "lepton fronts=" << coverage.fronts_ << " steps=" << coverage.steps_
            << " photons=" << coverage.photons_ << " continuations=" << coverage.next_
            << " observations=" << coverage.observations_
            << " decays=" << coverage.decays_ << " deflections=" << coverage.deflections_
            << " suppressions=" << coverage.suppressions_
            << " electronPairCPU=" << coverage.electronPairFallbacks_
            << " muonPairCPU=" << coverage.muonPairFallbacks_
            << " fallback=" << coverage.fallback_[1] << "," << coverage.fallback_[2]
            << "," << coverage.fallback_[3] << "," << coverage.fallback_[4] << "\n";
  for (auto const& x : coverage.process_)
    std::cout << "process=" << x.first << " count=" << x.second << "\n";
  for (unsigned s = 1; s <= 4; ++s) {
    INFO("stage " << s);
    REQUIRE(coverage.fallback_[s] > 0);
  }
  for (int pid : {11, -11, 13, -13}) REQUIRE(coverage.pid_[pid] > 0);
  REQUIRE(coverage.electronPairFallbacks_ > 0);
  REQUIRE(coverage.muonPairFallbacks_ > 0);
  REQUIRE(coverage.photons_ > 0);
  REQUIRE(coverage.next_ > 0);
  REQUIRE(coverage.observations_ > 0);
  REQUIRE(coverage.deflections_ > 0);
}
#endif
