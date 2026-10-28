/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <tests/kokkos/em/execution/PhotonCascadeTestFixture.hpp>
#include <tests/kokkos/em/execution/LeptonCascadeProduction.hpp>
#include <tests/kokkos/em/finalstate/LeptonFinalStateChecks.hpp>
#ifdef CORSIKA_TEST_CASCADE_NATIVE
#include <tests/kokkos/em/moliere/MoliereQueryTestData.hpp>
#include <corsika/kokkos/em/execution/MoliereInterpolationStorage.hpp>
#endif

namespace lepton_cascade_test {
  namespace cp = corsika::kokkos;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = execution_space_type::memory_space;
  using workspace_type = cp::LeptonFrontWorkspace<execution_space_type>;
  using pending_type = cp::PendingParticleQueue<execution_space_type>;
  using profile_type = cp::ProfileAccumulatorStorage<execution_space_type>;
  using radio_type = cp::radio::RadioAccumulator<execution_space_type>;
  using photon_cascade_test::Budget;
  using photon_cascade_test::download;
  using photon_cascade_test::ensure_runtime;
  using photon_cascade_test::FirstHistory;
  using photon_cascade_test::upload;
  constexpr std::uint64_t LastHistory = FirstHistory + 10000000;
  template <class T>
  void equal(T const& a, T const& b) {
    lepton_final_test::Payload x, y;
    lepton_final_test::flatten(x, a);
    lepton_final_test::flatten(y, b);
    lepton_final_test::check(x, y);
  }
  inline void equal(cp::LeptonTransportRecord const& a,
                    cp::LeptonTransportRecord const& b) {
    transport_test::check(transport_test::flatten(a), transport_test::flatten(b));
  }
  inline void equal(cp::ProjectedStepRecord const& a, cp::ProjectedStepRecord const& b) {
    photon_cascade_test::equal(a, b);
  }
  inline void equal(cp::ObservationRecord const& a, cp::ObservationRecord const& b) {
    photon_cascade_test::equal(a, b);
  }
  inline void equal(cp::ProcessStatistics const& a, cp::ProcessStatistics const& b) {
    photon_cascade_test::equal(a, b);
    REQUIRE(a.leptonMaterialBoundaries_ == b.leptonMaterialBoundaries_);
  }
  template <class T>
  void equal(std::vector<T> const& a, std::vector<T> const& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      INFO("record " << i);
      equal(a[i], b[i]);
    }
  }
  struct Models {
    cp::MoliereSnapshot electron_{}, muon_{};
    cp::MoliereInterpolationView interpolation_{};
    cp::BremsLpmSnapshot lpm_{};
  };
#ifdef CORSIKA_TEST_CASCADE_NATIVE
  struct ModelOwner {
    cp::MoliereInterpolationStorage<execution_space_type> storage_;
    Models models_;
    ModelOwner() {
      auto aux = cp::test::make_auxiliary_dataset();
      auto data = cp::test::make_moliere_dataset();
      storage_.initialize(data.snapshots_[0], data.cacheFile_);
      models_ = {data.snapshots_[0], data.snapshots_[1], storage_.getDeviceView(),
                 aux.snapshots_.front().electron_};
    }
  };
#endif
  struct Fixture : photon_cascade_test::Fixture {
    Fixture(cp::ProposalNativeTableSet const& table = {}, Models const& models = {},
            int pid = 11, int process = 0, bool thin = false)
        : photon_cascade_test::Fixture(table) {
      cp::ProposalNativeDeviceView native;
      if (!table.dndxColumns_.empty()) native = tables_.getDeviceView();
      if (process) {
        auto indices = table.selectionColumnIndices_;
        auto chosen = std::find_if(indices.begin(), indices.end(), [&](auto i) {
          return table.dndxColumns_[i].pdgId_ == pid &&
                 table.dndxColumns_[i].processId_ == process;
        });
        REQUIRE(chosen != indices.end());
        auto selected = *chosen;
        // Route rare channels without altering their native polynomial data.
        // These fixtures test dispatch, not the relative channel probabilities.
        for (auto& i : indices)
          if (table.dndxColumns_[i].pdgId_ == pid) i = selected;
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
      cp::TransportGeometryConfig geometry;
      geometry.observation_.pointM_[2] = -900.;
      geometry.observation_.normal_[2] = 1.;
      context_.initialize(
          cp::InteractionPhysicsView(cp::ContinuousPhysicsView(native, .4, 10.)),
          cp::EnvironmentBorrowedView(environment_.getDataView()), geometry,
          models.electron_, models.muon_, models.interpolation_, true, {}, models.lpm_,
          thin ? cp::EmThinningConfig{1.e10, 3., 1, 1} : cp::EmThinningConfig{},
          projection_, 1234567, 6789, execution_);
    }
  };
  inline std::vector<cp::ParticleState> particles(std::size_t n, int pid = 11,
                                                  double energy = 1000.) {
    auto p = photon_cascade_test::particles(n, energy);
    for (auto& x : p) x.pid_ = pid;
    return p;
  }
  struct Options {
    std::size_t waves_{3}, minimum_{1}, pendingInput_{}, pendingMaximum_{1000000},
        reserved_{};
    std::uint64_t first_{FirstHistory}, last_{LastHistory};
    bool air_{true}, project_{}, diagnostics_{true};
    workspace_type* workspace_{};
    pending_type* leptons_{};
    pending_type* photons_{};
    profile_type* profile_{};
    radio_type* radio_{};
    cp::ResidentMemoryBudgetGate budget_{};
    cp::ResidentExecutionWait<execution_space_type>* wait_{};
  };
  inline cp::LeptonCascadeResult run(Fixture& f, std::vector<cp::ParticleState> const& p,
                                     std::optional<cp::FirstInteractionRecord>& first,
                                     Options o = {}, bool production = false) {
    if (production)
      return lepton_cascade_production::run_lepton_cascade<execution_space_type>(
          f.context_.getDeviceView(), .00051099895, o.air_, p, o.first_, o.waves_,
          o.last_, o.minimum_, first, o.project_, f.projection_, o.profile_, o.radio_,
          o.workspace_, o.leptons_, o.photons_, o.pendingInput_, o.pendingMaximum_,
          f.execution_, o.diagnostics_, 0, o.budget_, o.reserved_, o.wait_);
    return cp::run_lepton_cascade<execution_space_type>(
        f.context_.getDeviceView(), .00051099895, o.air_, p, o.first_, o.waves_, o.last_,
        o.minimum_, first, o.project_, f.projection_, o.profile_, o.radio_, o.workspace_,
        o.leptons_, o.photons_, o.pendingInput_, o.pendingMaximum_, f.execution_,
        o.diagnostics_, 0, o.budget_, o.reserved_, o.wait_);
  }
  inline cp::radio::RadioConfig radio_config() {
    cp::radio::RadioConfig c;
    c.enabled_ = true;
    c.propagation_.homogeneousRefractiveIndex_ = 1.0003;
    for (int i = 0; i < 2; ++i) {
      cp::radio::RadioObserverSnapshot o;
      o.positionM_[0] = 30. + 10 * i;
      o.positionM_[2] = -900.;
      o.durationS_ = 20.e-6;
      o.sampleRateHz_ = 16.e6;
      o.numberOfBins_ = 320;
      c.coreasObservers_.push_back(o);
      c.zhsObservers_.push_back(o);
    }
    return c;
  }
  inline void equal(cp::LeptonCascadeResult const& a, cp::LeptonCascadeResult const& b) {
    REQUIRE(a.inputParticles_ == b.inputParticles_);
    REQUIRE(a.wavefronts_ == b.wavefronts_);
    REQUIRE(a.peakResidentLeptons_ == b.peakResidentLeptons_);
    REQUIRE(a.selectedInteractions_ == b.selectedInteractions_);
    REQUIRE(a.interactionBearingWavefronts_ == b.interactionBearingWavefronts_);
    REQUIRE(a.vertexInteractionsSelected_ == b.vertexInteractionsSelected_);
    REQUIRE(a.vertexNoInteractionContinuations_ == b.vertexNoInteractionContinuations_);
    REQUIRE(a.finalStateBearingWavefronts_ == b.finalStateBearingWavefronts_);
    REQUIRE(a.transportRecords_ == b.transportRecords_);
    REQUIRE(a.interactionVertices_ == b.interactionVertices_);
    REQUIRE(a.lpmSuppressions_ == b.lpmSuppressions_);
    REQUIRE(a.wavefrontBucketingBatches_ == b.wavefrontBucketingBatches_);
    REQUIRE(a.wavefrontBucketingParticles_ == b.wavefrontBucketingParticles_);
    REQUIRE(a.wavefrontBucketingSmallBatches_ == b.wavefrontBucketingSmallBatches_);
    REQUIRE(a.wavefrontBucketingSmallParticles_ == b.wavefrontBucketingSmallParticles_);
    equal(a.processStatistics_, b.processStatistics_);
    REQUIRE(a.secondaryHistoryIdsUsed_ == b.secondaryHistoryIdsUsed_);
    REQUIRE(a.historyRangeExhausted_ == b.historyRangeExhausted_);
    REQUIRE(a.belowMinimumBatchCheckpoint_ == b.belowMinimumBatchCheckpoint_);
    REQUIRE(a.workspaceLimitCheckpoint_ == b.workspaceLimitCheckpoint_);
    REQUIRE(a.completed_ == b.completed_);
    equal(a.stepRecords_, b.stepRecords_);
    equal(a.projectedStepRecords_, b.projectedStepRecords_);
    equal(a.generatedPhotons_, b.generatedPhotons_);
    equal(a.cpuSpillParticles_, b.cpuSpillParticles_);
    equal(a.fallbackEvents_, b.fallbackEvents_);
    equal(a.observations_, b.observations_);
    equal(a.decayCandidates_, b.decayCandidates_);
    equal(a.interactionRecords_, b.interactionRecords_);
    equal(a.finalStateRecords_, b.finalStateRecords_);
    equal(a.remainingLeptons_, b.remainingLeptons_);
  }
} // namespace lepton_cascade_test
