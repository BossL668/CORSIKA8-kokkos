/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/kokkos/em/execution/PhotonCascade.hpp>
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <tests/kokkos/em/finalstate/PhotonFinalStateChecks.hpp>
#include <tests/kokkos/em/transport/TransportChecks.hpp>
#ifdef CORSIKA_TEST_CASCADE_NATIVE
#include <tests/kokkos/em/models/AuxiliaryModelFixture.hpp>
#include <tests/kokkos/em/tables/NativeTableFixture.hpp>
#endif
#include <algorithm>
#include <array>
#include <iostream>
#include <map>

namespace photon_cascade_test {
  namespace cp = corsika::kokkos;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = execution_space_type::memory_space;
#ifdef CORSIKA_EXPECT_CUDA
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  inline void ensure_runtime() { static cp::Runtime runtime; }
  template <class T>
  auto upload(std::vector<T> const& input) {
    Kokkos::View<T*, memory_space_type> result("cascade test data", input.size());
    auto host = Kokkos::create_mirror_view(result);
    for (std::size_t i = 0; i < input.size(); ++i) host(i) = input[i];
    Kokkos::deep_copy(result, host);
    return result;
  }
  template <class T>
  auto download(T const& v) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  }
  template <class T>
  void equal(T const& a, T const& b) {
    photon_final_test::Payload x, y;
    photon_final_test::flatten(x, a);
    photon_final_test::flatten(y, b);
    photon_final_test::check(x, y);
  }
  inline void equal(cp::PhotonTransportRecord const& a,
                    cp::PhotonTransportRecord const& b) {
    transport_test::check(transport_test::flatten(a), transport_test::flatten(b));
  }
  inline void equal(cp::ObservationRecord const& a, cp::ObservationRecord const& b) {
    equal(a.particle_, b.particle_);
    REQUIRE(a.status_ == b.status_);
    REQUIRE(a.reserved_ == b.reserved_);
  }
  inline void equal(std::optional<cp::FirstInteractionRecord> const& a,
                    std::optional<cp::FirstInteractionRecord> const& b) {
    REQUIRE(a.has_value() == b.has_value());
    if (!a) return;
    REQUIRE(a->processId_ == b->processId_);
    REQUIRE(a->secondaryCount_ == b->secondaryCount_);
    equal(a->parentAtVertex_, b->parentAtVertex_);
    for (unsigned i = 0; i < a->secondaryCount_; ++i)
      equal(a->secondaries_[i], b->secondaries_[i]);
  }
  inline void equal(cp::ProjectedStepRecord const&, cp::ProjectedStepRecord const&);
  template <class T>
  void equal(std::vector<T> const& a, std::vector<T> const& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      INFO("output index " << i);
      equal(a[i], b[i]);
    }
  }
  inline std::vector<cp::ParticleState> particles(std::size_t n, double energy = 1.) {
    std::vector<cp::ParticleState> result(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto& p = result[i];
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
    }
    return result;
  }
  struct Fixture {
    execution_space_type execution_;
    cp::NativeTableStorage<execution_space_type> tables_;
    cp::EnvironmentStorage<execution_space_type> environment_;
    cp::PhysicsContext<execution_space_type> context_;
    Kokkos::View<std::uint32_t*, memory_space_type> selection_;
    Kokkos::View<double*, memory_space_type> support_;
    cp::ProfileProjectionConfig profile_;
    cp::ProfileProjectionView projection_;
    explicit Fixture(cp::ProposalNativeTableSet const& table = {}, int process = 0,
                     double density = .001, cp::PhotonPairLpmSnapshot lpm = {},
                     bool thin = false) {
      cp::ProposalNativeDeviceView native;
      if (!table.dndxColumns_.empty()) {
        tables_.initialize(table);
        native = tables_.getDeviceView();
      }
      auto indices = table.selectionColumnIndices_;
      if (process) {
        auto found = std::find_if(indices.begin(), indices.end(), [&](auto i) {
          return table.dndxColumns_[i].pdgId_ == 22 &&
                 table.dndxColumns_[i].processId_ == process;
        });
        REQUIRE(found != indices.end());
        auto const chosen = *found;
        // Route rare channels without changing the native polynomials. This
        // intentionally changes channel weights and is not a cross-section test.
        for (auto& i : indices)
          if (table.dndxColumns_[i].pdgId_ == 22 &&
              table.dndxColumns_[i].processId_ != process)
            i = chosen;
        selection_ = upload(indices);
        native = {native.getDndxColumns(),
                  native.getTotalRateColumns(),
                  cp::NativeArrayView<std::uint32_t, std::uint32_t>{selection_},
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
      cp::EnvironmentNode nodes[2]{};
      nodes[0].childCount_ = 1;
      nodes[1].parentIndex_ = 0;
      nodes[1].modelIndex_ = 0;
      cp::EnvironmentShape shapes[2]{};
      shapes[0].kind_ = cp::EnvironmentShape::Kind::Universe;
      shapes[1].kind_ = cp::EnvironmentShape::Kind::Sphere;
      shapes[1].radiusM_ = 1000.;
      std::int32_t children[1]{1};
      cp::EnvironmentModel model;
      model.densityKind_ = cp::EnvironmentModel::Density::Homogeneous;
      model.referenceDensityGPerCm3_ = density;
      model.mediumId_ = 0;
      environment_ = cp::EnvironmentStorage<execution_space_type>(
          {nodes, shapes, children, nullptr, &model, nullptr, 2, 1, 0, 1, 0});
      profile_.enabled_ = true;
      profile_.accumulateOnDevice_ = true;
      profile_.axisStartPositionM_[2] = 1000.;
      profile_.axisDirection_[2] = -1.;
      profile_.axisStepLengthM_ = 500.;
      profile_.axisGrammageGPerCm2_ = {0., 50., 100., 150., 200.};
      profile_.outputBinCount_ = 64;
      profile_.outputBinWidthGPerCm2_ = 4.;
      profile_.fixedPointWeightLimit_ = 1.e6;
      profile_.fixedPointEnergyLimitGeV_ = 1.e8;
      support_ = upload(profile_.axisGrammageGPerCm2_);
      projection_.axisStartPositionM_[2] = 1000.;
      projection_.axisDirection_[2] = -1.;
      projection_.axisStepLengthM_ = 500.;
      projection_.axisSupportCount_ = 5;
      projection_.axisGrammageGPerCm2_ = support_.data();
      cp::TransportGeometryConfig geometry;
      geometry.observation_.pointM_[2] = -900.;
      geometry.observation_.normal_[2] = 1.;
      context_.initialize(
          cp::InteractionPhysicsView(cp::ContinuousPhysicsView(native, .4, 10.)),
          cp::EnvironmentBorrowedView(environment_.getDataView()), geometry, {}, {}, {},
          false, lpm, {},
          thin ? cp::EmThinningConfig{1.e10, 3., 1, 1} : cp::EmThinningConfig{},
          projection_, 1234567, 6789, execution_);
    }
  };
  struct Budget {
    unsigned calls_{};
    unsigned reject_{};
    std::vector<std::string> stages_;
    static bool permit(void* opaque, std::size_t, cp::MemoryProjection const&,
                       char const* stage) {
      auto& b = *static_cast<Budget*>(opaque);
      b.stages_.push_back(stage);
      return ++b.calls_ != b.reject_;
    }
    cp::ResidentMemoryBudgetGate gate() { return {this, &permit}; }
  };
  inline constexpr std::uint64_t FirstHistory = std::uint64_t{1} << 60;
  inline void equal(cp::ProjectedStepRecord const& a, cp::ProjectedStepRecord const& b) {
    REQUIRE(a.historyId_ == b.historyId_);
    REQUIRE(a.pid_ == b.pid_);
    REQUIRE(a.processId_ == b.processId_);
    REQUIRE(a.transportLimit_ == b.transportLimit_);
    REQUIRE(a.reserved_ == b.reserved_);
    REQUIRE(a.startGrammageGPerCm2_ == b.startGrammageGPerCm2_);
    REQUIRE(a.endGrammageGPerCm2_ == b.endGrammageGPerCm2_);
    REQUIRE(a.endEnergyGeV_ == b.endEnergyGeV_);
    REQUIRE(a.depositedEnergyGeV_ == b.depositedEnergyGeV_);
    REQUIRE(a.weight_ == b.weight_);
    REQUIRE(a.cutDepositedEnergyGeV_ == b.cutDepositedEnergyGeV_);
    REQUIRE(a.observationSurfaceReachedBeforeCut_ ==
            b.observationSurfaceReachedBeforeCut_);
    REQUIRE(a.nativeMuonPairCut_ == b.nativeMuonPairCut_);
  }
  inline void equal(cp::ProcessStatistics const& a, cp::ProcessStatistics const& b) {
    REQUIRE(a.gpuFinalStates_ == b.gpuFinalStates_);
    REQUIRE(a.physicalSecondariesGenerated_ == b.physicalSecondariesGenerated_);
    REQUIRE(a.photonPairFinalStates_ == b.photonPairFinalStates_);
    REQUIRE(a.photonMuonPairFinalStates_ == b.photonMuonPairFinalStates_);
    REQUIRE(a.bremsFinalStates_ == b.bremsFinalStates_);
    REQUIRE(a.comptonFinalStates_ == b.comptonFinalStates_);
    REQUIRE(a.photoelectricFinalStates_ == b.photoelectricFinalStates_);
    REQUIRE(a.annihilationFinalStates_ == b.annihilationFinalStates_);
    REQUIRE(a.ionizationFinalStates_ == b.ionizationFinalStates_);
    REQUIRE(a.electronPairFinalStates_ == b.electronPairFinalStates_);
    REQUIRE(a.photonPairLpmTrials_ == b.photonPairLpmTrials_);
    REQUIRE(a.photonPairLpmSuppressions_ == b.photonPairLpmSuppressions_);
    REQUIRE(a.bremsLpmTrials_ == b.bremsLpmTrials_);
    REQUIRE(a.bremsLpmSuppressions_ == b.bremsLpmSuppressions_);
    REQUIRE(a.electronPairLpmTrials_ == b.electronPairLpmTrials_);
    REQUIRE(a.electronPairLpmSuppressions_ == b.electronPairLpmSuppressions_);
    REQUIRE(a.electronPairRejectionTrials_ == b.electronPairRejectionTrials_);
    REQUIRE(a.electronPairZeroWeightSamples_ == b.electronPairZeroWeightSamples_);
    REQUIRE(a.electronPairRejectionFallbacks_ == b.electronPairRejectionFallbacks_);
    REQUIRE(a.electronPairEnvelopeViolations_ == b.electronPairEnvelopeViolations_);
    REQUIRE(a.thinningHillasVertices_ == b.thinningHillasVertices_);
    REQUIRE(a.thinningStatisticalVertices_ == b.thinningStatisticalVertices_);
    REQUIRE(a.thinningParticlesDiscarded_ == b.thinningParticlesDiscarded_);
    REQUIRE(a.moliereTrials_ == b.moliereTrials_);
    REQUIRE(a.moliereDeflections_ == b.moliereDeflections_);
    REQUIRE(a.moliereZeroDeflections_ == b.moliereZeroDeflections_);
    REQUIRE(a.moliereNewtonIterations_ == b.moliereNewtonIterations_);
    REQUIRE(a.moliereMaxNewtonIterations_ == b.moliereMaxNewtonIterations_);
    REQUIRE(a.reserved_ == b.reserved_);
    REQUIRE(a.leptonTransportLimits_ == b.leptonTransportLimits_);
  }
  inline void equal(cp::PhotonCascadeResult const& a, cp::PhotonCascadeResult const& b) {
    REQUIRE(a.inputParticles_ == b.inputParticles_);
    REQUIRE(a.wavefronts_ == b.wavefronts_);
    REQUIRE(a.peakResidentPhotons_ == b.peakResidentPhotons_);
    REQUIRE(a.selectedInteractions_ == b.selectedInteractions_);
    REQUIRE(a.interactionBearingWavefronts_ == b.interactionBearingWavefronts_);
    REQUIRE(a.transportRecords_ == b.transportRecords_);
    REQUIRE(a.interactionVertices_ == b.interactionVertices_);
    REQUIRE(a.layerBoundaries_ == b.layerBoundaries_);
    REQUIRE(a.particleCuts_ == b.particleCuts_);
    REQUIRE(a.lpmSuppressions_ == b.lpmSuppressions_);
    REQUIRE(a.wavefrontBucketingBatches_ == b.wavefrontBucketingBatches_);
    REQUIRE(a.wavefrontBucketingParticles_ == b.wavefrontBucketingParticles_);
    REQUIRE(a.wavefrontBucketingSmallBatches_ == b.wavefrontBucketingSmallBatches_);
    REQUIRE(a.wavefrontBucketingSmallParticles_ == b.wavefrontBucketingSmallParticles_);
    REQUIRE(a.nativeNewtonIterations_ == b.nativeNewtonIterations_);
    REQUIRE(a.nativeBisectionIterations_ == b.nativeBisectionIterations_);
    REQUIRE(a.nativeInverseFailures_ == b.nativeInverseFailures_);
    equal(a.processStatistics_, b.processStatistics_);
    REQUIRE(a.belowMinimumBatchCheckpoint_ == b.belowMinimumBatchCheckpoint_);
    REQUIRE(a.workspaceLimitCheckpoint_ == b.workspaceLimitCheckpoint_);
    REQUIRE(a.completed_ == b.completed_);
    equal(a.electromagneticSecondaries_, b.electromagneticSecondaries_);
    equal(a.cpuSpillParticles_, b.cpuSpillParticles_);
    equal(a.fallbackEvents_, b.fallbackEvents_);
    equal(a.observations_, b.observations_);
    equal(a.interactionRecords_, b.interactionRecords_);
    equal(a.stepRecords_, b.stepRecords_);
    equal(a.projectedStepRecords_, b.projectedStepRecords_);
    equal(a.finalStateRecords_, b.finalStateRecords_);
    equal(a.remainingPhotons_, b.remainingPhotons_);
  }
  inline void equal(cp::ProfileCounters const& a, cp::ProfileCounters const& b) {
    REQUIRE(a.steps_ == b.steps_);
    REQUIRE(a.depositedSteps_ == b.depositedSteps_);
    REQUIRE(a.photonCuts_ == b.photonCuts_);
    for (unsigned i = 0; i < 8; ++i) REQUIRE(a.leptonLimits_[i] == b.leptonLimits_[i]);
    REQUIRE(a.moliereTrials_ == b.moliereTrials_);
    REQUIRE(a.moliereDeflections_ == b.moliereDeflections_);
    REQUIRE(a.moliereZeroDeflections_ == b.moliereZeroDeflections_);
    REQUIRE(a.moliereNewtonIterations_ == b.moliereNewtonIterations_);
    REQUIRE(a.moliereMaxNewtonIterations_ == b.moliereMaxNewtonIterations_);
    REQUIRE(a.thinningHillasVertices_ == b.thinningHillasVertices_);
    REQUIRE(a.thinningStatisticalVertices_ == b.thinningStatisticalVertices_);
    REQUIRE(a.thinningParticlesDiscarded_ == b.thinningParticlesDiscarded_);
    REQUIRE(a.fixedPointOverflows_ == b.fixedPointOverflows_);
    REQUIRE(a.invalidRecords_ == b.invalidRecords_);
    REQUIRE(a.weightedMediumRestMassInput_ == b.weightedMediumRestMassInput_);
    REQUIRE(a.weightedCutRestMassEnergy_ == b.weightedCutRestMassEnergy_);
    REQUIRE(a.weightedObservedTotalEnergy_ == b.weightedObservedTotalEnergy_);
    REQUIRE(a.weightedEscapedTotalEnergy_ == b.weightedEscapedTotalEnergy_);
    REQUIRE(a.weightedUnwrittenPhotoelectricBindingEnergy_ ==
            b.weightedUnwrittenPhotoelectricBindingEnergy_);
    REQUIRE(a.weightedObservationCutOverlapEnergy_ ==
            b.weightedObservationCutOverlapEnergy_);
    REQUIRE(a.weightedMassConventionCorrection_ == b.weightedMassConventionCorrection_);
    REQUIRE(a.weightedNativeMuonPairCutEnergy_ == b.weightedNativeMuonPairCutEnergy_);
  }

} // namespace photon_cascade_test
