/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/LeptonFinalStateRecord.hpp>
#include <corsika/kokkos/em/BremsLpmSuppressionRecord.hpp>
#include <corsika/kokkos/em/FirstInteractionRecord.hpp>
#include <corsika/kokkos/em/ProcessCapabilities.hpp>
#include <corsika/kokkos/em/algorithms/EmThinning.hpp>
#include <corsika/kokkos/em/algorithms/PositronAtRest.hpp>
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>
#include <corsika/kokkos/em/models/BremsLpmPreparedSnapshot.hpp>
#include <corsika/kokkos/em/random/LeptonFinalStateRandomDomains.hpp>
#include <corsika/kokkos/em/random/Philox.hpp>
#include <cmath>
#include <cstdint>

namespace corsika::kokkos {

  /** Production lepton classification and output records.
   * Units and child offsets follow the shared records. Direct-pair diagnostic
   * fields remain layout-compatible; the classifier routes that process to CPU.
   */
  struct LeptonFinalStateParameters {
    std::int32_t processId_{};
    std::uint32_t thinningStatus_{};
    std::uint32_t thinningKeepMask_{0x3U};
    double finalStateUniform_{};
    double energySplitFraction_{};
    double azimuthUniform_{};
    double auxiliaryUniform_{};
    double lpmSurvivalProbability_{};
    double lpmUniform_{};
    double thinningFirstUniform_{};
    double thinningSecondUniform_{};
    double thinningFirstWeight_{};
    double thinningSecondWeight_{};
  };
  struct LeptonFinalStateClassification {
    LeptonFinalStateParameters parameters_{};
    FallbackRequest fallback_{};
    std::uint32_t childCount_{};
    std::uint32_t recordFlag_{};
    std::uint32_t fallbackFlag_{};
    std::uint32_t continuationFlag_{};
    std::uint32_t suppressionFlag_{};
    std::uint32_t bremsFlag_{};
    std::uint32_t annihilationFlag_{};
    std::uint32_t ionizationFlag_{};
    std::uint32_t electronPairFlag_{};
    std::uint32_t bremsSuppressionFlag_{};
    std::uint32_t electronPairSuppressionFlag_{};
    std::uint32_t electronPairRejectionTrials_{};
    std::uint32_t electronPairZeroWeightFlag_{};
    std::uint32_t electronPairRejectionFallbackFlag_{};
    std::uint32_t electronPairEnvelopeViolationFlag_{};
  };
  struct LeptonFinalStateMaterialization {
    LeptonFinalStateRecord record_{};
    ParticleState secondaries_[3]{};
    FallbackRequest fallback_{};
    InteractionRecord continuation_{};
    BremsLpmSuppressionRecord suppression_{};
    FirstInteractionRecord firstInteraction_{};
    std::uint32_t secondaryCount_{};
    std::uint32_t hasRecord_{};
    std::uint32_t hasFallback_{};
    std::uint32_t hasContinuation_{};
    std::uint32_t hasSuppression_{};
    std::uint32_t hasFirstInteraction_{};
    std::uint32_t error_{};
  };
  namespace lepton_final_state_detail {
    inline constexpr double LeptonTwoPi = 6.283185307179586476925286766559005768;
    inline constexpr double LeptonPi = 3.141592653589793238462643383279502884;
    inline constexpr double ProposalHalfPrecision = 1.e-5;
    // Legacy template default only; no rejection sampler is present or called.
    inline constexpr int EpairRejectionMaximumTrials = 256;
  } // namespace lepton_final_state_detail
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline bool lepton_finite(double value);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_sqrt(double value);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_log(double value);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_cos(double value);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_sin(double value);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_exp(double value);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_fabs(double value);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_fmod(double left, double right);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_pow(double base, double exponent);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_maximum(double left, double right);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double lepton_minimum(double left, double right);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline void deflect_lepton_child(double const input[3],
                                                         double cosine, double azimuth,
                                                         double output[3]);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline bool normalize_lepton_direction(double vector[3]);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double effective_loss_fraction(double energy_GeV,
                                                              double sampled_fraction);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double annihilation_function_term(double a1, double a2,
                                                                 double rho);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline int lepton_sign(double value);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline double annihilation_equation(double value, double a1,
                                                            double a2, double lower_term,
                                                            double upper_term,
                                                            double uniform);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline bool sample_annihilation_rho(double energy_GeV,
                                                            double mass_GeV,
                                                            double uniform, double& rho);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline FallbackRequest invalid_lepton_final_state(
        InteractionRecord const& interaction,
        LeptonFinalStateParameters const& parameters = {});
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline bool apply_lepton_two_child_thinning(
        EmThinningConfig const& thinning, ParticleState const& parent,
        std::int32_t process_id, double first_energy_GeV, double second_energy_GeV,
        std::uint64_t random_seed, std::uint64_t shower_id,
        LeptonFinalStateParameters& sample, std::uint32_t& child_count);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline BremsLpmResult prepared_brems_lpm_suppression_factor(
        BremsLpmSnapshot const& snapshot, BremsLpmPreparedSnapshot const& prepared,
        std::uint64_t component_hash, double energy_MeV, double v,
        double local_mass_density_g_per_cm3);
  }
  namespace lepton_final_state_detail {
    template <
        bool UsePreparedLpm,
        int EpairMaximumTrials = lepton_final_state_detail::EpairRejectionMaximumTrials>
    CORSIKA_HOST_DEVICE inline LeptonFinalStateClassification
    classify_lepton_final_state_impl(BremsLpmSnapshot const& lpm_snapshot,
                                     BremsLpmPreparedSnapshot const& prepared_lpm,
                                     EmThinningConfig const& thinning,
                                     InteractionRecord const& interaction,
                                     std::uint64_t random_seed, std::uint64_t shower_id);
  }
  /** Use the exported muon LPM parameters; never substitute the electron model.
   * CPU direct-pair routing does not require valid device muon auxiliary data.
   */
  template <
      int EpairMaximumTrials = lepton_final_state_detail::EpairRejectionMaximumTrials>
  CORSIKA_HOST_DEVICE inline LeptonFinalStateClassification
  classify_muon_radiative_final_state(BremsLpmSnapshot const& source,
                                      EmThinningConfig const& thinning,
                                      InteractionRecord const& interaction,
                                      std::uint64_t random_seed, std::uint64_t shower_id);
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline bool needs_muon_radiative_final_state(
        InteractionRecord const& interaction);
  }
  /** Classify a selected production lepton vertex with unchanged RNG/LPM/thinning.
   * Electron/positron and muon direct-pair vertices return CPU fallback records.
   * The prepared overload uses execution-space component terms; the unprepared
   * overload retains the direct production LPM formula. Neither owns queues.
   */
  CORSIKA_HOST_DEVICE inline LeptonFinalStateClassification classify_lepton_final_state(
      BremsLpmSnapshot const& lpm_snapshot, EmThinningConfig const& thinning,
      InteractionRecord const& interaction, std::uint64_t random_seed,
      std::uint64_t shower_id);
  /** Classify a selected production lepton vertex with unchanged RNG/LPM/thinning.
   * Electron/positron and muon direct-pair vertices return CPU fallback records.
   * The prepared overload uses execution-space component terms; the unprepared
   * overload retains the direct production LPM formula. Neither owns queues.
   */
  CORSIKA_HOST_DEVICE inline LeptonFinalStateClassification classify_lepton_final_state(
      BremsLpmSnapshot const& lpm_snapshot, BremsLpmPreparedSnapshot const& prepared_lpm,
      EmThinningConfig const& thinning, InteractionRecord const& interaction,
      std::uint64_t random_seed, std::uint64_t shower_id);
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline void capture_lepton_first_interaction(
        LeptonFinalStateMaterialization& output, ParticleState const& parent,
        std::int32_t process_id, ParticleState const& first,
        ParticleState const* second = nullptr, ParticleState const* third = nullptr);
  }
  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline void populate_lepton_thinning_record(
        LeptonFinalStateRecord& record, LeptonFinalStateParameters const& sample);
  }
  /** Write into exclusive slots allocated by the caller after classification.
   * Writer supplies fallback, continuation, suppression, firstInteraction,
   * secondary, record, finishRecord and fail methods. child_offset is an
   * exclusive-scan offset, and the caller ensures capacity and ID arithmetic.
   * Direct-pair output code is retained for production fidelity but cannot be
   * reached from the production classifier. No CPU handler runs here.
   */
  template <class Writer>
  CORSIKA_HOST_DEVICE inline void materialize_lepton_final_state_in_place(
      InteractionRecord const& interaction,
      LeptonFinalStateClassification const& classification, std::uint64_t child_offset,
      std::uint64_t first_history_id, double electron_mass_GeV, Writer& output);
  /** Aggregate interface with identical production writes and diagnostics.
   * Resident kernels use the in-place interface to avoid a large local object.
   */
  CORSIKA_HOST_DEVICE inline LeptonFinalStateMaterialization
  materialize_lepton_final_state(InteractionRecord const& interaction,
                                 LeptonFinalStateClassification const& classification,
                                 std::uint64_t child_offset,
                                 std::uint64_t first_history_id,
                                 double electron_mass_GeV);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/LeptonFinalStateStep.inl>
