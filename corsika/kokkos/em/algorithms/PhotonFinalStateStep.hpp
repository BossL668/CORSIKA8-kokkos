/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/FirstInteractionRecord.hpp>
#include <corsika/kokkos/em/PhotonFinalStateRecord.hpp>
#include <corsika/kokkos/em/PhotonPairLpmSuppressionRecord.hpp>
#include <corsika/kokkos/em/algorithms/EmThinning.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCut.hpp>
#include <corsika/kokkos/em/FallbackRequest.hpp>
#include <corsika/kokkos/em/ProcessCapabilities.hpp>
#include <corsika/kokkos/em/models/PhotonPairKinematics.hpp>
#include <corsika/kokkos/em/models/PhotonPairLpm.hpp>
#include <corsika/kokkos/em/random/Philox.hpp>
#include <corsika/kokkos/em/random/PhotonFinalStateRandomDomains.hpp>
#include <cmath>
#include <cstdint>

namespace corsika::kokkos {

  /** Classification and materialization records for one photon vertex.
   * Values retain production field order and defaults. Energies are total GeV,
   * positions metres, time seconds, density g/cm^3 and weights dimensionless.
   * childCount_ counts retained children; firstInteraction_ holds the unthinned
   * generation-zero final state. No record owns output storage or table arrays.
   */
  struct PhotonFinalStateParameters {
    std::int32_t processId_{};
    std::uint32_t thinningStatus_{};
    std::uint32_t thinningKeepMask_{0x3U};
    double splitFraction_{};
    double splitUniform_{};
    std::uint64_t splitDrawId_{PhotonPairSplitDrawId};
    double azimuthUniform_{};
    double electronPolarUniform_{};
    double positronPolarUniform_{};
    double lpmSurvivalProbability_{};
    double lpmUniform_{};
    double thinningFirstUniform_{};
    double thinningSecondUniform_{};
    double thinningFirstWeight_{};
    double thinningSecondWeight_{};
    double secondaryMassGeV_{}; // BKK muon mass exported from PROPOSAL.
  };
  struct PhotonFinalStateClassification {
    PhotonFinalStateParameters parameters_{};
    FallbackRequest fallback_{};
    std::uint32_t childCount_{};
    std::uint32_t recordFlag_{};
    std::uint32_t fallbackFlag_{};
    std::uint32_t continuationFlag_{};
    std::uint32_t suppressionFlag_{};
    std::uint32_t photonPairFlag_{};
    std::uint32_t comptonFlag_{};
    std::uint32_t photoelectricFlag_{};
    std::uint32_t photonMuonPairFlag_{};
  };
  struct PhotonFinalStateMaterialization {
    PhotonFinalStateRecord record_{};
    ParticleState secondaries_[2]{};
    FallbackRequest fallback_{};
    InteractionRecord continuation_{};
    PhotonPairLpmSuppressionRecord suppression_{};
    FirstInteractionRecord firstInteraction_{};
    std::uint32_t secondaryCount_{};
    std::uint32_t hasRecord_{};
    std::uint32_t hasFallback_{};
    std::uint32_t hasContinuation_{};
    std::uint32_t hasSuppression_{};
    std::uint32_t hasFirstInteraction_{};
    std::uint32_t error_{};
  };
  /** Classify one production photon interaction before output allocation.
   * Borrowed tables must remain accessible until all calls finish. The interaction
   * supplies the selected channel/component at the reached vertex, density and
   * original random identities. No-reaction records continue unchanged; deferred
   * CPU channels require a reached vertex. Table failures retain their diagnostics.
   * Pair sharing uses the native CDF inverse; LPM and EM thinning follow production
   * order. Muon pairs bypass electron-pair LPM and EM thinning as in native C8.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline PhotonFinalStateClassification
  classify_photon_final_state(ContinuousPhysicsView const& table,
                              PhotonPairLpmSnapshot const& lpm_snapshot,
                              EmThinningConfig const& thinning,
                              InteractionRecord const& interaction,
                              std::uint64_t random_seed, std::uint64_t shower_id);
  /** Write a classified state into exclusive, already allocated output slots.
   * Writer provides fallback, continuation, suppression, firstInteraction, secondary,
   * record, finishRecord and fail methods (see the aggregate writer below).
   * child_offset is a stable exclusive-scan offset; first_history_id plus that
   * offset and the retained child index assigns history IDs. The caller must ensure
   * capacity and non-overflowing IDs. Suppression advances only the parent step ID.
   * No queue allocation, atomic append, CPU handler or transport runs here.
   */
  template <class Writer>
  CORSIKA_HOST_DEVICE inline void materialize_photon_final_state_in_place(
      InteractionRecord const& interaction,
      PhotonFinalStateClassification const& classification, std::uint64_t child_offset,
      std::uint64_t first_history_id, Writer& output);
  /** Aggregate-return interface with the same production writes and diagnostics.
   * Use the in-place interface in resident kernels to avoid a large thread-local
   * aggregate. This interface is useful for callers and independent validation.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline PhotonFinalStateMaterialization
  materialize_photon_final_state(InteractionRecord const& interaction,
                                 PhotonFinalStateClassification const& classification,
                                 std::uint64_t child_offset,
                                 std::uint64_t first_history_id);
  namespace photon_final_state_detail {
    CORSIKA_HOST_DEVICE inline bool photon_finite(double value);
    CORSIKA_HOST_DEVICE inline double photon_sqrt(double value);
    CORSIKA_HOST_DEVICE inline double photon_cos(double value);
    CORSIKA_HOST_DEVICE inline double photon_sin(double value);
    CORSIKA_HOST_DEVICE inline double photon_fabs(double value);
    CORSIKA_HOST_DEVICE inline double photon_fmod(double left, double right);
    CORSIKA_HOST_DEVICE inline double photon_maximum(double left, double right);
    CORSIKA_HOST_DEVICE inline double bounded_cosine(double value);
    CORSIKA_HOST_DEVICE inline double sauter_cosine(double energy_GeV, double uniform);
    CORSIKA_HOST_DEVICE inline bool compton_cosines(double energy_GeV, double v,
                                                    double& photon_cosine,
                                                    double& electron_cosine);
    CORSIKA_HOST_DEVICE inline bool photoelectric_energy(
        PhotonPairLpmSnapshot const& snapshot, std::uint64_t component_hash,
        double parent_energy_GeV, double& electron_total_energy_GeV,
        double& kinetic_fraction);
    CORSIKA_HOST_DEVICE inline void deflect_photon_child(double const input[3],
                                                         double cosine, double azimuth,
                                                         double output[3]);
    CORSIKA_HOST_DEVICE inline FallbackRequest invalid_photon_final_state(
        InteractionRecord const& interaction,
        PhotonFinalStateParameters const& parameters = {});
    CORSIKA_HOST_DEVICE inline bool apply_photon_two_child_thinning(
        EmThinningConfig const& thinning, ParticleState const& parent,
        std::int32_t process_id, double first_energy_GeV, double second_energy_GeV,
        std::uint64_t random_seed, std::uint64_t shower_id,
        PhotonFinalStateParameters& sample, std::uint32_t& child_count);
    CORSIKA_HOST_DEVICE inline void fill_first_interaction(
        ParticleState const& parent, std::int32_t process_id, ParticleState const& first,
        ParticleState const* second, PhotonFinalStateMaterialization& output);
  } // namespace photon_final_state_detail

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/PhotonFinalStateStep.inl>
