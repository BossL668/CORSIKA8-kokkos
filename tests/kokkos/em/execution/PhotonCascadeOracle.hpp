/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include "PhotonCascadeTestFixture.hpp"
namespace photon_cascade_test {
  // The accepted front is the primitive under test in 3Y. This oracle controls
  // rounds, stable sorting, history offsets and output ownership independently
  // on the host; it does not call the new cascade or its statistics functors.
  inline cp::PhotonCascadeResult reference_loop(
      Fixture& f, std::vector<cp::ParticleState> current, std::size_t maximum,
      std::size_t minimum, std::optional<cp::FirstInteractionRecord>& first,
      bool project = false,
      cp::ProfileAccumulatorStorage<execution_space_type>* profile = nullptr,
      bool diagnostics = false, std::uint64_t firstHistory = FirstHistory) {
    cp::PhotonCascadeResult result;
    result.inputParticles_ = current.size();
    result.peakResidentPhotons_ = current.size();
    auto history = firstHistory;
    bool const capture = !first;
    unsigned captures = 0;
    for (std::size_t round = 0; round < maximum && !current.empty(); ++round) {
      if (round != 0 && current.size() < minimum) {
        result.belowMinimumBatchCheckpoint_ = true;
        break;
      }
      if (current.size() >= cp::MinimumWavefrontRadixSortSize) {
        auto input = upload(current);
        Kokkos::View<std::uint64_t*, memory_space_type> keys("reference keys",
                                                             current.size());
        Kokkos::parallel_for(
            "reference bucket keys",
            Kokkos::RangePolicy<execution_space_type>(f.execution_, 0, current.size()),
            KOKKOS_LAMBDA(std::size_t i) {
              keys(i) = cp::wavefront_bucket_key(input(i));
            });
        auto host = download(keys);
        std::vector<std::size_t> indices(current.size());
        for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;
        std::stable_sort(indices.begin(), indices.end(),
                         [&](auto a, auto b) { return host(a) < host(b); });
        auto unsorted = current;
        for (std::size_t i = 0; i < indices.size(); ++i)
          current[i] = unsorted[indices[i]];
        ++result.wavefrontBucketingBatches_;
        result.wavefrontBucketingParticles_ += current.size();
      } else {
        ++result.wavefrontBucketingSmallBatches_;
        result.wavefrontBucketingSmallParticles_ += current.size();
      }
      cp::PhotonFrontWorkspace<execution_space_type> w;
      w.ensureCapacity(current.size(), f.execution_);
      w.ensureCandidateCapacity(current.size(), f.execution_);
      w.queue_.upload(current, f.execution_);
      Kokkos::deep_copy(f.execution_, w.error_, 0u);
      Kokkos::deep_copy(f.execution_, w.firstInteractionCandidates_, 0u);
      cp::enqueue_photon_front(
          f.context_.getDeviceView(), w.queue_.current().rawDeviceView(),
          w.queue_.next().rawDeviceView(), w, current.size(), current.size(), history,
          project, capture, f.execution_, 0);
      f.execution_.fence("independent front complete");
      w.queue_.markExecutionSynchronized();
      auto control = download(w.frontControl_)();
      REQUIRE(control.materializationError_ == 0);
      auto const& totals = control.totals_.values_;
      auto selection = download(w.selections_);
      auto final = download(w.finalStates_);
      auto steps = download(w.steps_);
      auto records = download(w.records_);
      auto charged = download(w.charged_);
      auto fallbacks = download(w.fallbacks_);
      auto observations = download(w.observations_);
      auto projected = download(w.projectedSteps_);
      for (std::size_t i = 0; i < current.size(); ++i) {
        result.selectedInteractions_ += selection(i).fallbackFlag_ == 0;
        result.nativeNewtonIterations_ += selection(i).nativeNewtonIterations_;
        result.nativeBisectionIterations_ += selection(i).nativeBisectionIterations_;
        result.nativeInverseFailures_ += selection(i).nativeInverseFailures_;
      }
      result.interactionBearingWavefronts_ += control.interactionCount_ != 0;
      auto& stats = result.processStatistics_;
      for (std::size_t i = 0; i < control.interactionCount_; ++i) {
        auto const& c = final(i);
        result.lpmSuppressions_ += c.suppressionFlag_;
        stats.photonPairLpmSuppressions_ += c.suppressionFlag_;
        stats.photonPairLpmTrials_ += c.photonPairFlag_ + c.suppressionFlag_;
        if (c.recordFlag_) {
          stats.gpuFinalStates_ += c.recordFlag_;
          stats.physicalSecondariesGenerated_ += c.childCount_;
          stats.photonPairFinalStates_ += c.photonPairFlag_;
          stats.photonMuonPairFinalStates_ += c.photonMuonPairFlag_;
          stats.comptonFinalStates_ += c.comptonFlag_;
          stats.photoelectricFinalStates_ += c.photoelectricFlag_;
          auto status = static_cast<cp::EmThinningStatus>(c.parameters_.thinningStatus_);
          stats.thinningHillasVertices_ += status == cp::EmThinningStatus::Hillas;
          stats.thinningStatisticalVertices_ +=
              status == cp::EmThinningStatus::Statistical;
          unsigned original =
              c.parameters_.processId_ == cp::PhotoelectricProcessId ? 1 : 2;
          if (c.childCount_ <= original)
            stats.thinningParticlesDiscarded_ += original - c.childCount_;
        }
      }
      auto const nsteps = totals[cp::PhotonStepOffset],
                 nrecords = totals[cp::PhotonRecordOffset];
      for (std::size_t i = 0; i < nsteps; ++i) {
        auto const& step = steps(i);
        result.interactionVertices_ +=
            step.limit_ == cp::PhotonTransportLimit::Interaction;
        result.layerBoundaries_ += step.limit_ == cp::PhotonTransportLimit::LayerBoundary;
        result.particleCuts_ += step.limit_ == cp::PhotonTransportLimit::ParticleCut;
        if (project) result.projectedStepRecords_.push_back(projected(i));
        if (diagnostics || (!project && !profile)) result.stepRecords_.push_back(step);
        if (diagnostics && step.limit_ == cp::PhotonTransportLimit::Interaction)
          result.interactionRecords_.push_back(step.interaction_);
      }
      if (profile)
        profile->accumulatePhoton(f.projection_, w.steps_, nsteps, w.records_, nrecords,
                                  f.execution_);
      else
        for (std::size_t i = 0; i < nrecords; ++i)
          result.finalStateRecords_.push_back(records(i));
      for (std::size_t i = 0; i < totals[cp::PhotonChargedOffset]; ++i)
        result.electromagneticSecondaries_.push_back(charged(i));
      auto nf = totals[cp::PhotonSelectionFallbackOffset] +
                totals[cp::PhotonTransportFallbackOffset] +
                totals[cp::PhotonFinalStateFallbackOffset];
      for (std::size_t i = 0; i < nf; ++i) result.fallbackEvents_.push_back(fallbacks(i));
      for (std::size_t i = 0; i < totals[cp::PhotonObservationOffset]; ++i)
        result.observations_.push_back(observations(i));
      if (capture) {
        auto candidates = download(w.firstInteractionCandidates_)(0);
        captures += candidates;
        REQUIRE(captures <= 1);
        if (candidates) first = download(w.firstSnapshots_)(0);
      }
      result.transportRecords_ += nsteps;
      history += totals[cp::PhotonChildOffset];
      f.execution_.fence("independent profile complete");
      w.queue_.commitNextAlreadySynchronized(totals[cp::PhotonNextOffset]);
      current = w.queue_.download(f.execution_);
      ++result.wavefronts_;
      result.peakResidentPhotons_ = std::max(result.peakResidentPhotons_, current.size());
    }
    result.completed_ = current.empty();
    result.remainingPhotons_ = std::move(current);
    return result;
  }
} // namespace photon_cascade_test
