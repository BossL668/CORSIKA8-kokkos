/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Immutable production 41cdc543. Namespace/include/annotation adaptation only.
#include "../selection/SelectionReference.hpp"
#include "../models/AuxiliaryModelReference.hpp"
namespace photon_final_reference {
  using namespace selection_reference;
  using corsika::kokkos::test::auxiliary_reference::MaxPhotonPairLpmComponents;
  using corsika::kokkos::test::auxiliary_reference::PhotonPairLpmSnapshot;
  using corsika::kokkos::test::auxiliary_reference::PhotonPairLpmStatus;
  using corsika::kokkos::test::auxiliary_reference::photonPairLpmSuppressionFactor;
  using discrete_reference::decodePhotonPairNormalizedSplit;
  using discrete_reference::ElectronMassGeV;
  namespace gpu {
    namespace em = ::photon_final_reference;
  }
  inline constexpr std::uint64_t PhotonPairSplitDrawId = 1;
  inline constexpr std::uint64_t PhotonPairAzimuthDrawId = 2;
  inline constexpr std::uint64_t PhotonPairElectronPolarDrawId = 3;
  inline constexpr std::uint64_t PhotonPairPositronPolarDrawId = 4;
  inline constexpr std::uint64_t PhotonPairLpmDrawId = 5;
  inline constexpr std::uint64_t PhotonPairAnalyticCandidateDrawIdBase = 0x20;
  inline constexpr std::uint64_t PhotonPairAnalyticAcceptanceDrawIdBase = 0x60;
  inline constexpr std::uint32_t PhotonPairAnalyticMaximumAttempts = 64;
  inline constexpr std::uint64_t ComptonAzimuthDrawId = 1;
  inline constexpr std::uint64_t PhotonMuonPairSplitDrawId = std::uint64_t{1} << 32;
  inline constexpr std::uint64_t PhotonMuonPairAzimuthDrawId =
      PhotonMuonPairSplitDrawId + 1;
  struct GpuFirstInteractionSnapshot {
    EmParticleState parent_at_vertex{};
    EmParticleState secondaries[3]{};
    std::int32_t process_id{};
    std::uint32_t secondary_count{};
  };
  struct PhotonFinalStateRecord {
    std::uint64_t input_index{};
    std::uint64_t parent_history_id{};
    std::uint64_t secondary_offset{};
    std::uint32_t secondary_count{};
    std::int32_t process_id{};
    double energy_split_fraction{};
    double split_uniform{};
    double azimuth_uniform{};
    double electron_polar_uniform{};
    double positron_polar_uniform{};
    double lpm_survival_probability{};
    double lpm_uniform{};
    std::uint64_t split_draw_id{};
    std::uint64_t azimuth_draw_id{};
    std::uint64_t electron_polar_draw_id{};
    std::uint64_t positron_polar_draw_id{};
    std::uint64_t lpm_draw_id{};
    std::uint32_t thinning_status{};
    std::uint32_t thinning_keep_mask{0x3U};
    double thinning_first_uniform{};
    double thinning_second_uniform{};
    std::uint64_t thinning_first_draw_id{};
    std::uint64_t thinning_second_draw_id{};
    double weighted_mass_convention_correction_GeV{};
  };
  struct BremsFinalStateRecord {
    std::uint64_t input_index{};
    std::uint64_t parent_history_id{};
    std::uint64_t secondary_offset{};
    std::uint32_t secondary_count{};
    std::int32_t process_id{};
    double photon_energy_fraction{};
    double final_state_uniform{};
    double azimuth_uniform{};
    double auxiliary_uniform{};
    double lpm_survival_probability{};
    double lpm_uniform{};
    std::uint64_t final_state_draw_id{};
    std::uint64_t azimuth_draw_id{};
    std::uint64_t auxiliary_draw_id{};
    std::uint64_t lpm_draw_id{};
    std::uint32_t thinning_status{};
    std::uint32_t thinning_keep_mask{0x3U};
    double thinning_first_uniform{};
    double thinning_second_uniform{};
    std::uint64_t thinning_first_draw_id{};
    std::uint64_t thinning_second_draw_id{};
    double weighted_mass_convention_correction_GeV{};
  };
  struct PhotonPairLpmSuppressionRecord {
    EmParticleState particle{};
    std::uint64_t input_index{};
    std::uint64_t component_hash{};
    double survival_probability{};
    double uniform{};
    std::uint64_t draw_id{};
  };
  struct BremsLpmSuppressionRecord {
    EmParticleState particle{};
    std::uint64_t input_index{};
    std::uint64_t component_hash{};
    double survival_probability{};
    double uniform{};
    std::uint64_t draw_id{};
  };
  using PhotonPairFinalStateRecord = PhotonFinalStateRecord;
  // Process-specific final-state draws occupy small consecutive IDs. Keep
  // thinning in a disjoint range so enabling it cannot perturb those streams.
  inline constexpr std::uint64_t EmThinningFirstDrawId = 0x100;
  inline constexpr std::uint64_t EmThinningSecondDrawId = 0x101;

  /**
   * Device representation of the existing CORSIKA EMThinning configuration.
   *
   * The current CORSIKA process applies only to one-to-two EM splittings.
   * threshold_GeV is the parent total-energy threshold, not a secondary cut.
   */
  struct EmThinningConfig {
    double threshold_GeV{};
    double maximum_weight{};
    std::uint32_t enabled{};
    std::uint32_t erase_zero_weight{1};
  };

  enum class EmThinningStatus : std::uint32_t {
    NotApplied = 0,
    Hillas = 1,
    Statistical = 2,
    InvalidInput = 3,
  };

  /**
   * keep_mask uses bit zero for child 0 and bit one for child 1. If
   * erase_zero_weight is false, both bits remain set and discarded particles
   * carry weight zero, matching EMThinning's multithin mode.
   */
  struct EmThinningResult {
    EmThinningStatus status{EmThinningStatus::InvalidInput};
    std::uint32_t keep_mask{};
    double first_weight{};
    double second_weight{};
  };

  static_assert(std::is_standard_layout_v<EmThinningConfig>);
  static_assert(std::is_trivially_copyable_v<EmThinningConfig>);
  static_assert(std::is_standard_layout_v<EmThinningResult>);
  static_assert(std::is_trivially_copyable_v<EmThinningResult>);

  namespace thinning_detail {

    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double minimum(double first, double second) {
#if defined(__CUDA_ARCH__)
      return ::fmin(first, second);
#else
      return std::fmin(first, second);
#endif
    }

  } // namespace thinning_detail

  /**
   * Apply the exact branching rules of
   * corsika/detail/modules/thinning/EMThinning.inl to an already generated
   * one-to-two electromagnetic final state.
   *
   * Random numbers are explicit so callers can use keyed Philox draws.
   * first_uniform is used for the Hillas choice or the first independent
   * statistical decision; second_uniform is used only for the second
   * statistical decision.
   */
  CORSIKA_HOST_DEVICE inline EmThinningResult applyEmThinning(
      EmThinningConfig const& config, double parent_energy_GeV, double parent_weight,
      double first_energy_GeV, double second_energy_GeV, double first_uniform,
      double second_uniform) {
    using namespace thinning_detail;
    if (!finite(parent_energy_GeV) || !(parent_energy_GeV > 0.) ||
        !finite(parent_weight) || parent_weight < 0. || !finite(first_energy_GeV) ||
        !(first_energy_GeV > 0.) || !finite(second_energy_GeV) ||
        !(second_energy_GeV > 0.) || !finite(first_uniform) || !(first_uniform > 0.) ||
        !(first_uniform < 1.) || !finite(second_uniform) || !(second_uniform > 0.) ||
        !(second_uniform < 1.)) {
      return {EmThinningStatus::InvalidInput, 0, 0., 0.};
    }

    auto unchanged = EmThinningResult{EmThinningStatus::NotApplied, 0x3U, parent_weight,
                                      parent_weight};
    if (config.enabled == 0) { return unchanged; }
    if (!finite(config.threshold_GeV) || !(config.threshold_GeV > 0.) ||
        !finite(config.maximum_weight) || !(config.maximum_weight > 0.)) {
      return {EmThinningStatus::InvalidInput, 0, 0., 0.};
    }

    // These are the same early exits and comparisons used by EMThinning.
    if (parent_weight >= config.maximum_weight ||
        parent_energy_GeV > config.threshold_GeV) {
      return unchanged;
    }

    auto const energy_sum = first_energy_GeV + second_energy_GeV;
    if (!finite(energy_sum) || !(energy_sum > 0.)) {
      return {EmThinningStatus::InvalidInput, 0, 0., 0.};
    }
    auto const first_probability = first_energy_GeV / energy_sum;
    auto const second_probability = second_energy_GeV / energy_sum;
    auto const first_factor = 1. / first_probability;
    auto const second_factor = 1. / second_probability;
    auto const maximum_factor = config.maximum_weight / parent_weight;

    EmThinningResult result{};
    if (first_factor <= maximum_factor && second_factor <= maximum_factor) {
      result.status = EmThinningStatus::Hillas;
      if (first_uniform <= first_probability) {
        result.keep_mask = 0x1U;
        result.first_weight = parent_weight * first_factor;
        result.second_weight = 0.;
      } else {
        result.keep_mask = 0x2U;
        result.first_weight = 0.;
        result.second_weight = parent_weight * second_factor;
      }
    } else {
      result.status = EmThinningStatus::Statistical;
      auto const first_limited_factor = minimum(first_factor, maximum_factor);
      auto const second_limited_factor = minimum(second_factor, maximum_factor);
      if (first_uniform * first_limited_factor <= 1.) {
        result.keep_mask |= 0x1U;
        result.first_weight = parent_weight * first_limited_factor;
      }
      if (second_uniform * second_limited_factor <= 1.) {
        result.keep_mask |= 0x2U;
        result.second_weight = parent_weight * second_limited_factor;
      }
    }

    if (config.erase_zero_weight == 0) {
      result.keep_mask = 0x3U;
    } else {
      // Scalar EMThinning erases by the final weight, not by the random
      // selection bit. A selected child of a zero-weight parent (or an
      // underflowed product) still has to be erased in non-multithin mode.
      if (result.first_weight == 0.) result.keep_mask &= ~0x1U;
      if (result.second_weight == 0.) result.keep_mask &= ~0x2U;
    }
    return result;
  }

  CORSIKA_HOST_DEVICE inline double proposalSecondaryTransportEnergyGeV(
      std::int32_t const pid, double const native_total_energy_GeV,
      double const native_mass_GeV) {
    return (native_total_energy_GeV - native_mass_GeV) + transportMassGeV(pid);
  }
  CORSIKA_HOST_DEVICE inline double weightedProposalMassCorrectionGeV(
      std::int32_t const pid, double const native_mass_GeV, double const weight) {
    return weight * (transportMassGeV(pid) - native_mass_GeV);
  }
  /**
   * Backend-neutral intermediate state for one photon interaction.
   *
   * Classification is deliberately separate from output allocation.  A
   * Kokkos parallel_scan and the native CUDA CUB scan can therefore assign
   * the same stable child offsets without atomically appending particles.
   */
  struct PhotonFinalStateParameters {
    std::int32_t process_id{};
    std::uint32_t thinning_status{};
    std::uint32_t thinning_keep_mask{0x3U};
    double split_fraction{};
    double split_uniform{};
    std::uint64_t split_draw_id{PhotonPairSplitDrawId};
    double azimuth_uniform{};
    double electron_polar_uniform{};
    double positron_polar_uniform{};
    double lpm_survival_probability{};
    double lpm_uniform{};
    double thinning_first_uniform{};
    double thinning_second_uniform{};
    double thinning_first_weight{};
    double thinning_second_weight{};
    double secondary_mass_GeV{}; // BKK muon mass exported from PROPOSAL.
  };

  struct PhotonFinalStateClassification {
    PhotonFinalStateParameters parameters{};
    gpu::em::ProposalFallbackEvent fallback{};
    std::uint32_t child_count{};
    std::uint32_t record_flag{};
    std::uint32_t fallback_flag{};
    std::uint32_t continuation_flag{};
    std::uint32_t suppression_flag{};
    std::uint32_t photon_pair_flag{};
    std::uint32_t compton_flag{};
    std::uint32_t photoelectric_flag{};
    std::uint32_t photon_muon_pair_flag{};
  };

  struct PhotonFinalStateMaterialization {
    gpu::em::PhotonPairFinalStateRecord record{};
    gpu::em::EmParticleState secondaries[2]{};
    gpu::em::ProposalFallbackEvent fallback{};
    gpu::em::EmInteractionRecord continuation{};
    gpu::em::PhotonPairLpmSuppressionRecord suppression{};
    gpu::em::GpuFirstInteractionSnapshot first_interaction{};
    std::uint32_t secondary_count{};
    std::uint32_t has_record{};
    std::uint32_t has_fallback{};
    std::uint32_t has_continuation{};
    std::uint32_t has_suppression{};
    std::uint32_t has_first_interaction{};
    std::uint32_t error{};
  };

  CORSIKA_HOST_DEVICE inline bool photonFinite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline double photonSqrt(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sqrt(value);
#else
    return std::sqrt(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline double photonCos(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::cos(value);
#else
    return std::cos(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline double photonSin(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sin(value);
#else
    return std::sin(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline double photonFabs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fabs(value);
#else
    return std::fabs(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline double photonFmod(double left, double right) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fmod(left, right);
#else
    return std::fmod(left, right);
#endif
  }

  CORSIKA_HOST_DEVICE inline double photonMaximum(double left, double right) {
    return left > right ? left : right;
  }

  CORSIKA_HOST_DEVICE inline double boundedCosine(double value) {
    return photonMaximum(-1., value < 1. ? value : 1.);
  }

  CORSIKA_HOST_DEVICE inline double sauterCosine(double energy_GeV, double uniform) {
    auto const mass_squared = gpu::em::ElectronMassGeV * gpu::em::ElectronMassGeV;
    auto const momentum =
        photonSqrt(photonMaximum(0., energy_GeV * energy_GeV - mass_squared));
    auto const coordinate = 2. * uniform - 1.;
    auto const energy_minus_momentum = mass_squared / (energy_GeV + momentum);
    auto const one_minus_cosine =
        energy_minus_momentum * (1. - coordinate) / (momentum * coordinate + energy_GeV);
    return boundedCosine(1. - one_minus_cosine);
  }

  CORSIKA_HOST_DEVICE inline bool comptonCosines(double energy_GeV, double v,
                                                 double& photon_cosine,
                                                 double& electron_cosine) {
    auto const scattered_energy = energy_GeV * (1. - v);
    auto const electron_momentum_squared =
        2. * v * energy_GeV * gpu::em::ElectronMassGeV + v * v * energy_GeV * energy_GeV;
    if (!photonFinite(energy_GeV) || !(energy_GeV > 0.) || !photonFinite(v) ||
        !(v > 0.) || !(v < 1.) || !(scattered_energy > 0.) ||
        !(electron_momentum_squared > 0.)) {
      return false;
    }
    photon_cosine = 1. - (v * gpu::em::ElectronMassGeV) / scattered_energy;
    electron_cosine = v * (energy_GeV + gpu::em::ElectronMassGeV) /
                      photonSqrt(electron_momentum_squared);
    auto constexpr tolerance = 1.e-12;
    if (!photonFinite(photon_cosine) || !photonFinite(electron_cosine) ||
        photon_cosine < -1. - tolerance || photon_cosine > 1. + tolerance ||
        electron_cosine < -1. - tolerance || electron_cosine > 1. + tolerance) {
      return false;
    }
    photon_cosine = boundedCosine(photon_cosine);
    electron_cosine = boundedCosine(electron_cosine);
    return true;
  }

  CORSIKA_HOST_DEVICE inline bool photoelectricEnergy(
      gpu::em::PhotonPairLpmSnapshot const& snapshot, std::uint64_t component_hash,
      double parent_energy_GeV, double& electron_total_energy_GeV,
      double& kinetic_fraction) {
    if (!photonFinite(parent_energy_GeV) || !(parent_energy_GeV > 0.) ||
        snapshot.component_count == 0 ||
        snapshot.component_count > gpu::em::MaxPhotonPairLpmComponents ||
        !photonFinite(snapshot.fine_structure_constant) ||
        !(snapshot.fine_structure_constant > 0.)) {
      return false;
    }
    double nuclear_charge = 0.;
    for (std::uint32_t index = 0; index < snapshot.component_count; ++index) {
      if (snapshot.components[index].component_hash == component_hash) {
        nuclear_charge = snapshot.components[index].nuclear_charge;
        break;
      }
    }
    if (!photonFinite(nuclear_charge) || !(nuclear_charge > 0.)) return false;
    auto const z_alpha = nuclear_charge * snapshot.fine_structure_constant;
    auto const binding_energy_GeV = z_alpha * z_alpha * gpu::em::ElectronMassGeV / 2.;
    auto const kinetic_energy_GeV = parent_energy_GeV - binding_energy_GeV;
    if (!photonFinite(binding_energy_GeV) || !photonFinite(kinetic_energy_GeV) ||
        !(kinetic_energy_GeV >= 0.)) {
      return false;
    }
    electron_total_energy_GeV = gpu::em::ElectronMassGeV + kinetic_energy_GeV;
    kinetic_fraction = kinetic_energy_GeV / parent_energy_GeV;
    return photonFinite(electron_total_energy_GeV) &&
           electron_total_energy_GeV >= gpu::em::ElectronMassGeV &&
           photonFinite(kinetic_fraction) && kinetic_fraction >= 0. &&
           kinetic_fraction <= 1.;
  }

  CORSIKA_HOST_DEVICE inline void deflectPhotonChild(double const input[3], double cosine,
                                                     double azimuth, double output[3]) {
    auto const transverse =
        photonSqrt(photonMaximum(0., input[0] * input[0] + input[1] * input[1]));
    double cosine_phi = 1.;
    double sine_phi = 0.;
    if (transverse > 0.) {
      cosine_phi = input[0] / transverse;
      sine_phi = input[1] / transverse;
    }
    auto const cosine_theta = input[2];
    auto const sine_theta = transverse;
    double const rotation_x[3]{cosine_theta * cosine_phi, cosine_theta * sine_phi,
                               -sine_theta};
    double const rotation_y[3]{-sine_phi, cosine_phi, 0.};
    auto const sine = photonSqrt(photonMaximum(0., (1. - cosine) * (1. + cosine)));
    auto const local_x = sine * photonCos(azimuth);
    auto const local_y = sine * photonSin(azimuth);
    auto local_z =
        photonSqrt(photonMaximum(0., 1. - local_x * local_x - local_y * local_y));
    if (cosine < 0.) local_z = -local_z;
    for (int axis = 0; axis < 3; ++axis) {
      output[axis] =
          local_z * input[axis] + local_x * rotation_x[axis] + local_y * rotation_y[axis];
    }
  }

  CORSIKA_HOST_DEVICE inline gpu::em::ProposalFallbackEvent invalidPhotonFinalState(
      gpu::em::EmInteractionRecord const& interaction,
      PhotonFinalStateParameters const& parameters = {}) {
    auto event = gpu::em::makeProcessFallbackEvent(
        interaction, gpu::em::ProposalFallbackReason::InvalidFinalState);
    if (parameters.split_uniform > 0.) {
      event.final_state_uniform = parameters.split_uniform;
      event.final_state_draw_id = parameters.split_draw_id;
    }
    return event;
  }

  CORSIKA_HOST_DEVICE inline bool applyPhotonTwoChildThinning(
      gpu::em::EmThinningConfig const& thinning, gpu::em::EmParticleState const& parent,
      std::int32_t process_id, double first_energy_GeV, double second_energy_GeV,
      std::uint64_t random_seed, std::uint64_t shower_id,
      PhotonFinalStateParameters& sample, std::uint32_t& child_count) {
    gpu::em::RandomNumberKey first_key{random_seed,
                                       shower_id,
                                       parent.history_id,
                                       parent.step_id,
                                       static_cast<std::uint32_t>(process_id),
                                       gpu::em::EmThinningFirstDrawId};
    auto second_key = first_key;
    second_key.draw_id = gpu::em::EmThinningSecondDrawId;
    sample.thinning_first_uniform = gpu::em::uniformOpen01(first_key);
    sample.thinning_second_uniform = gpu::em::uniformOpen01(second_key);
    auto const result = gpu::em::applyEmThinning(
        thinning, parent.energy_GeV, parent.weight, first_energy_GeV, second_energy_GeV,
        sample.thinning_first_uniform, sample.thinning_second_uniform);
    if (result.status == gpu::em::EmThinningStatus::InvalidInput) return false;
    sample.thinning_status = static_cast<std::uint32_t>(result.status);
    sample.thinning_keep_mask = result.keep_mask;
    sample.thinning_first_weight = result.first_weight;
    sample.thinning_second_weight = result.second_weight;
    child_count =
        (result.keep_mask & 0x1U ? 1U : 0U) + (result.keep_mask & 0x2U ? 1U : 0U);
    return true;
  }

  // Pair energy sharing always uses the native PROPOSAL cumulative table.
  CORSIKA_HOST_DEVICE inline PhotonFinalStateClassification classifyPhotonFinalState(
      gpu::em::tables::NativePhysicsView const& table,
      gpu::em::PhotonPairLpmSnapshot const& lpm_snapshot,
      gpu::em::EmThinningConfig const& thinning,
      gpu::em::EmInteractionRecord const& interaction, std::uint64_t random_seed,
      std::uint64_t shower_id) {
    static_assert(PhotonPairSplitDrawId < PhotonPairAnalyticCandidateDrawIdBase);
    static_assert(PhotonPairAnalyticCandidateDrawIdBase +
                      PhotonPairAnalyticMaximumAttempts <=
                  PhotonPairAnalyticAcceptanceDrawIdBase);
    static_assert(PhotonPairAnalyticAcceptanceDrawIdBase +
                      PhotonPairAnalyticMaximumAttempts <=
                  gpu::em::EmThinningFirstDrawId);
    PhotonFinalStateClassification result{};
    if (interaction.status == gpu::em::EmInteractionStatus::NoDiscreteInteraction) {
      result.continuation_flag = 1;
      return result;
    }
    if (interaction.status != gpu::em::EmInteractionStatus::Selected) {
      result.fallback = invalidPhotonFinalState(interaction);
      result.fallback_flag = 1;
      return result;
    }
    if (interaction.deferred_photon_fallback_reason >= 0) {
      auto const reason = static_cast<gpu::em::ProposalFallbackReason>(
          interaction.deferred_photon_fallback_reason);
      if (interaction.interaction_vertex_reached != 1 ||
          !gpu::em::proposalFallbackRequiresSelectedLoss(reason)) {
        result.fallback = invalidPhotonFinalState(interaction);
      } else {
        result.fallback = gpu::em::makeProcessFallbackEvent(interaction, reason);
        if (interaction.deferred_photon_table_status != 0) {
          result.fallback.diagnostic_status = interaction.deferred_photon_table_status;
          result.fallback.diagnostic_value0 = interaction.particle.energy_GeV * 1000.;
          result.fallback.diagnostic_value1 = interaction.loss_quantile;
          result.fallback.diagnostic_value2 = interaction.energy_fraction;
        }
      }
      result.fallback_flag = 1;
      return result;
    }
    auto const capability =
        gpu::em::gpuProcessCapability(interaction.particle.pid, interaction.process_id);
    if (capability != gpu::em::GpuProcessCapability::PhotonPair &&
        capability != gpu::em::GpuProcessCapability::PhotonMuonPair &&
        capability != gpu::em::GpuProcessCapability::Compton &&
        capability != gpu::em::GpuProcessCapability::Photoelectric) {
      result.fallback = gpu::em::makeProcessFallbackEvent(
          interaction, gpu::em::processFallbackReason(capability));
      result.fallback_flag = 1;
      return result;
    }
    auto const& parent = interaction.particle;
    if (capability == gpu::em::GpuProcessCapability::PhotonMuonPair) {
      auto& sample = result.parameters;
      sample.process_id = gpu::em::PhotonMuonPairProcessId;
      sample.secondary_mass_GeV = table.proposal_native.constants.muon_mass_MeV / 1000.;
      if (!photonFinite(parent.energy_GeV) || !photonFinite(parent.weight) ||
          parent.weight < 0. || !(sample.secondary_mass_GeV > 0.) ||
          parent.generation == 0xffffffffU || parent.step_id == 0xffffffffffffffffULL) {
        result.fallback = invalidPhotonFinalState(interaction);
        result.fallback_flag = 1;
        return result;
      }
      gpu::em::RandomNumberKey key{
          random_seed,
          shower_id,
          parent.history_id,
          parent.step_id,
          static_cast<std::uint32_t>(gpu::em::PhotonMuonPairProcessId),
          PhotonMuonPairSplitDrawId};
      sample.split_uniform = gpu::em::uniformOpen01(key);
      sample.split_draw_id = PhotonMuonPairSplitDrawId;
      auto const split = gpu::em::tables::executeTableQuery(
          table, {gpu::em::tables::TableQueryKind::LossFraction, parent.pid,
                  gpu::em::PhotonMuonPairFinalStateProcessId, interaction.component_hash,
                  parent.energy_GeV * 1000., sample.split_uniform});
      if (split.status != gpu::em::tables::TableLookupStatus::Success) {
        // A corrupt/missing/out-of-range native table is not a normal CPU
        // physics path. Fail closed with the original vertex for diagnosis.
        result.fallback = invalidPhotonFinalState(interaction, sample);
        result.fallback.diagnostic_status = static_cast<std::uint32_t>(split.status);
        result.fallback_flag = 1;
        return result;
      }
      sample.split_fraction = split.value; // x = E_mu+ / E_gamma.
      auto const minus = (1. - split.value) * parent.energy_GeV;
      auto const plus = split.value * parent.energy_GeV;
      if (!photonFinite(split.value) || !(minus > 0.) || !(plus > 0.)) {
        result.fallback = invalidPhotonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
      key.draw_id = PhotonMuonPairAzimuthDrawId;
      sample.azimuth_uniform = gpu::em::uniformOpen01(key);
      // EMThinning::doSecondaries returns when either child is non-EM.
      // Muons therefore inherit the parent weight even when EM thinning is on.
      // CheckForLPM likewise applies no electron-pair LPM veto to PhotoMuPair.
      sample.thinning_status =
          static_cast<std::uint32_t>(gpu::em::EmThinningStatus::NotApplied);
      sample.thinning_keep_mask = 3U;
      sample.thinning_first_weight = sample.thinning_second_weight = parent.weight;
      result.child_count = 2;
      result.record_flag = 1;
      result.photon_muon_pair_flag = 1;
      return result;
    }
    if (capability == gpu::em::GpuProcessCapability::Compton) {
      double photon_cosine = 0.;
      double electron_cosine = 0.;
      if (parent.generation == 0xffffffffU || parent.step_id == 0xffffffffffffffffULL ||
          !comptonCosines(parent.energy_GeV, interaction.energy_fraction, photon_cosine,
                          electron_cosine)) {
        result.fallback = invalidPhotonFinalState(interaction);
        result.fallback_flag = 1;
        return result;
      }
      auto& sample = result.parameters;
      sample.process_id = gpu::em::ComptonProcessId;
      sample.split_fraction = interaction.energy_fraction;
      sample.split_uniform = interaction.loss_quantile;
      gpu::em::RandomNumberKey const azimuth_key{
          random_seed,
          shower_id,
          parent.history_id,
          parent.step_id,
          static_cast<std::uint32_t>(gpu::em::ComptonProcessId),
          ComptonAzimuthDrawId};
      sample.azimuth_uniform = gpu::em::uniformOpen01(azimuth_key);
      if (!applyPhotonTwoChildThinning(
              thinning, parent, gpu::em::ComptonProcessId,
              parent.energy_GeV * (1. - sample.split_fraction),
              gpu::em::proposalSecondaryTransportEnergyGeV(
                  11,
                  gpu::em::ElectronMassGeV + parent.energy_GeV * sample.split_fraction,
                  gpu::em::ElectronMassGeV),
              random_seed, shower_id, sample, result.child_count)) {
        result.fallback = invalidPhotonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
      result.record_flag = 1;
      result.compton_flag = 1;
      return result;
    }
    if (capability == gpu::em::GpuProcessCapability::Photoelectric) {
      double electron_energy_GeV = 0.;
      double kinetic_fraction = 0.;
      if (parent.generation == 0xffffffffU ||
          !photonFinite(interaction.energy_fraction) ||
          photonFabs(interaction.energy_fraction - 1.) > 1.e-12 ||
          !photoelectricEnergy(lpm_snapshot, interaction.component_hash,
                               parent.energy_GeV, electron_energy_GeV,
                               kinetic_fraction)) {
        result.fallback = invalidPhotonFinalState(interaction);
        result.fallback_flag = 1;
        return result;
      }
      auto& sample = result.parameters;
      sample.process_id = gpu::em::PhotoelectricProcessId;
      sample.split_fraction = kinetic_fraction;
      sample.split_uniform = interaction.loss_quantile;
      result.child_count = 1;
      result.record_flag = 1;
      result.photoelectric_flag = 1;
      return result;
    }
    if (!photonFinite(parent.energy_GeV) ||
        parent.energy_GeV < 2. * gpu::em::ElectronMassGeV ||
        parent.generation == 0xffffffffU || parent.step_id == 0xffffffffffffffffULL) {
      result.fallback = invalidPhotonFinalState(interaction);
      result.fallback_flag = 1;
      return result;
    }

    auto& sample = result.parameters;
    sample.process_id = gpu::em::PhotonPairProcessId;
    // Independent CDF quantile; no candidate/acceptance rejection loop.
    gpu::em::RandomNumberKey const split_key{
        random_seed,
        shower_id,
        parent.history_id,
        parent.step_id,
        static_cast<std::uint32_t>(gpu::em::PhotonPairProcessId),
        PhotonPairSplitDrawId};
    sample.split_uniform = gpu::em::uniformOpen01(split_key);
    sample.split_draw_id = PhotonPairSplitDrawId;
    gpu::em::tables::TableQuery const split_query{
        gpu::em::tables::TableQueryKind::LossFraction,
        parent.pid,
        gpu::em::PhotonPairFinalStateProcessId,
        interaction.component_hash,
        parent.energy_GeV * 1000.,
        sample.split_uniform};
    auto const split = gpu::em::tables::executeTableQuery(table, split_query);
    if (split.status == gpu::em::tables::TableLookupStatus::Success) {
      auto const split_ready = gpu::em::decodePhotonPairNormalizedSplit(
          parent.energy_GeV, split.value, sample.split_fraction);
      if (!split_ready) {
        result.fallback = invalidPhotonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
    } else {
      auto event = gpu::em::makeTableFallbackEvent(
          parent, split_query, split, PhotonPairSplitDrawId, interaction.input_index);
      event.interaction_vertex_reached = interaction.interaction_vertex_reached;
      event.energy_fraction = interaction.energy_fraction;
      event.process_id = interaction.process_id;
      auto const native_selection =
          interaction.proposal_selection_random_process_id != 0u;
      event.selection_uniform = native_selection ? interaction.proposal_selection_uniform
                                                 : interaction.process_uniform;
      event.loss_quantile = interaction.loss_quantile;
      event.outer_acceptance_uniform = interaction.process_uniform;
      event.outer_acceptance_random_process_id = interaction.process_random_process_id;
      event.outer_acceptance_draw_id = interaction.process_draw_id;
      event.random_process_id =
          native_selection ? interaction.proposal_selection_random_process_id : 0u;
      event.random_draw_id = native_selection ? interaction.proposal_selection_draw_id
                                              : interaction.loss_draw_id;
      event.final_state_uniform = sample.split_uniform;
      event.final_state_draw_id = PhotonPairSplitDrawId;
      result.fallback = event;
      result.fallback_flag = 1;
      return result;
    }
    auto const electron_energy = parent.energy_GeV * sample.split_fraction;
    auto const positron_energy = parent.energy_GeV - electron_energy;
    if (!photonFinite(sample.split_fraction) || !(sample.split_fraction > 0.) ||
        !(sample.split_fraction < 1.) || electron_energy < gpu::em::ElectronMassGeV ||
        positron_energy < gpu::em::ElectronMassGeV) {
      result.fallback = invalidPhotonFinalState(interaction, sample);
      result.fallback_flag = 1;
      return result;
    }
    gpu::em::RandomNumberKey angle_key{
        random_seed,
        shower_id,
        parent.history_id,
        parent.step_id,
        static_cast<std::uint32_t>(gpu::em::PhotonPairProcessId),
        PhotonPairAzimuthDrawId};
    sample.azimuth_uniform = gpu::em::uniformOpen01(angle_key);
    angle_key.draw_id = PhotonPairElectronPolarDrawId;
    sample.electron_polar_uniform = gpu::em::uniformOpen01(angle_key);
    angle_key.draw_id = PhotonPairPositronPolarDrawId;
    sample.positron_polar_uniform = gpu::em::uniformOpen01(angle_key);
    auto const lpm = gpu::em::photonPairLpmSuppressionFactor(
        lpm_snapshot, interaction.component_hash, parent.energy_GeV * 1000.,
        sample.split_fraction, interaction.mass_density_g_per_cm3);
    if (lpm.status != gpu::em::PhotonPairLpmStatus::Success) {
      auto const reason = lpm.status == gpu::em::PhotonPairLpmStatus::InvalidInput
                              ? gpu::em::ProposalFallbackReason::InvalidMassDensity
                              : gpu::em::ProposalFallbackReason::LpmParametersUnavailable;
      result.fallback = gpu::em::makeProcessFallbackEvent(interaction, reason);
      result.fallback_flag = 1;
      return result;
    }
    sample.lpm_survival_probability = lpm.survival_probability;
    gpu::em::RandomNumberKey const lpm_key{
        random_seed,
        shower_id,
        parent.history_id,
        parent.step_id,
        static_cast<std::uint32_t>(gpu::em::PhotonPairProcessId),
        PhotonPairLpmDrawId};
    sample.lpm_uniform = gpu::em::uniformOpen01(lpm_key);
    if (sample.lpm_uniform > sample.lpm_survival_probability) {
      result.suppression_flag = 1;
      return result;
    }
    if (!applyPhotonTwoChildThinning(thinning, parent, gpu::em::PhotonPairProcessId,
                                     gpu::em::proposalSecondaryTransportEnergyGeV(
                                         11, electron_energy, gpu::em::ElectronMassGeV),
                                     gpu::em::proposalSecondaryTransportEnergyGeV(
                                         -11, positron_energy, gpu::em::ElectronMassGeV),
                                     random_seed, shower_id, sample,
                                     result.child_count)) {
      result.fallback = invalidPhotonFinalState(interaction, sample);
      result.fallback_flag = 1;
      return result;
    }
    result.record_flag = 1;
    result.photon_pair_flag = 1;
    return result;
  }

  CORSIKA_HOST_DEVICE inline void fillFirstInteraction(
      gpu::em::EmParticleState const& parent, std::int32_t process_id,
      gpu::em::EmParticleState const& first, gpu::em::EmParticleState const* second,
      PhotonFinalStateMaterialization& output) {
    if (parent.generation != 0) return;
    output.has_first_interaction = 1;
    output.first_interaction.parent_at_vertex = parent;
    output.first_interaction.process_id = process_id;
    output.first_interaction.secondary_count = second == nullptr ? 1U : 2U;
    output.first_interaction.secondaries[0] = first;
    output.first_interaction.secondaries[0].weight = parent.weight;
    if (second != nullptr) {
      output.first_interaction.secondaries[1] = *second;
      output.first_interaction.secondaries[1].weight = parent.weight;
    }
  }

  /** Compatibility sink for callers that still need the aggregate result. */
  struct PhotonFinalStateMaterializationWriter {
    PhotonFinalStateMaterialization& output;

    CORSIKA_HOST_DEVICE void fallback(gpu::em::ProposalFallbackEvent const& value) {
      output.fallback = value;
      output.has_fallback = 1;
    }

    CORSIKA_HOST_DEVICE void continuation(gpu::em::EmInteractionRecord const& value) {
      output.continuation = value;
      output.has_continuation = 1;
    }

    CORSIKA_HOST_DEVICE void suppression(gpu::em::EmParticleState const& parent,
                                         gpu::em::EmInteractionRecord const& interaction,
                                         PhotonFinalStateParameters const& sample) {
      output.suppression = {parent,
                            interaction.input_index,
                            interaction.component_hash,
                            sample.lpm_survival_probability,
                            sample.lpm_uniform,
                            PhotonPairLpmDrawId};
      output.has_suppression = 1;
    }

    CORSIKA_HOST_DEVICE void firstInteraction(gpu::em::EmParticleState const& parent,
                                              std::int32_t process_id,
                                              gpu::em::EmParticleState const& first,
                                              gpu::em::EmParticleState const* second) {
      fillFirstInteraction(parent, process_id, first, second, output);
    }

    CORSIKA_HOST_DEVICE void secondary(std::uint32_t index,
                                       gpu::em::EmParticleState const& particle) {
      output.secondaries[index] = particle;
    }

    CORSIKA_HOST_DEVICE gpu::em::PhotonPairFinalStateRecord& record() {
      return output.record;
    }

    CORSIKA_HOST_DEVICE void finishRecord(std::uint32_t secondary_count) {
      output.secondary_count = secondary_count;
      output.has_record = 1;
    }

    CORSIKA_HOST_DEVICE void fail(std::uint32_t error) { output.error = error; }
  };

  /**
   * Materialize directly into a caller-provided sink.
   *
   * The resident Kokkos path uses this overload to write the already-scanned
   * output slots without first constructing the 1.6 KiB aggregate result in a
   * per-thread CUDA stack frame.  The aggregate-return API below remains the
   * compatibility and validation interface.
   */
  template <class Writer>
  CORSIKA_HOST_DEVICE inline void materializePhotonFinalStateInPlace(
      gpu::em::EmInteractionRecord const& interaction,
      PhotonFinalStateClassification const& classification, std::uint64_t child_offset,
      std::uint64_t first_history_id, Writer& output) {
    if (classification.fallback_flag != 0) {
      output.fallback(classification.fallback);
      return;
    }
    if (classification.continuation_flag != 0) {
      output.continuation(interaction);
      return;
    }
    if (classification.suppression_flag != 0) {
      auto parent = interaction.particle;
      ++parent.step_id;
      auto const& sample = classification.parameters;
      output.suppression(parent, interaction, sample);
      return;
    }
    if (classification.record_flag == 0) return;

    auto const& parent = interaction.particle;
    auto const& sample = classification.parameters;
    std::uint32_t secondary_count = 0;
    auto const azimuth = sample.azimuth_uniform * 6.283185307179586476925286766559;
    if (sample.process_id == gpu::em::ComptonProcessId) {
      double photon_cosine = 0.;
      double electron_cosine = 0.;
      if (!comptonCosines(parent.energy_GeV, sample.split_fraction, photon_cosine,
                          electron_cosine)) {
        output.fail(1);
        return;
      }
      auto photon = parent;
      photon.energy_GeV = parent.energy_GeV * (1. - sample.split_fraction);
      photon.parent_history_id = parent.history_id;
      photon.generation = parent.generation + 1;
      photon.step_id = 0;
      photon.reserved = 0;
      photon.weight = sample.thinning_first_weight;
      deflectPhotonChild(parent.direction, photon_cosine, azimuth, photon.direction);
      auto electron = parent;
      electron.pid = static_cast<std::int32_t>(gpu::em::EmPid::Electron);
      electron.energy_GeV = gpu::em::proposalSecondaryTransportEnergyGeV(
          electron.pid,
          gpu::em::ElectronMassGeV + parent.energy_GeV * sample.split_fraction,
          gpu::em::ElectronMassGeV);
      electron.parent_history_id = parent.history_id;
      electron.generation = parent.generation + 1;
      electron.step_id = 0;
      electron.reserved = 0;
      electron.weight = sample.thinning_second_weight;
      deflectPhotonChild(parent.direction, electron_cosine,
                         photonFmod(azimuth + 3.1415926535897932384626433832795,
                                    6.283185307179586476925286766559),
                         electron.direction);
      output.firstInteraction(parent, gpu::em::ComptonProcessId, photon, &electron);
      if ((sample.thinning_keep_mask & 0x1U) != 0) {
        photon.history_id = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, photon);
      }
      if ((sample.thinning_keep_mask & 0x2U) != 0) {
        electron.history_id = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, electron);
      }
      output.record() = {interaction.input_index,
                         parent.history_id,
                         child_offset,
                         secondary_count,
                         gpu::em::ComptonProcessId,
                         sample.split_fraction,
                         interaction.loss_quantile,
                         sample.azimuth_uniform,
                         0.,
                         0.,
                         0.,
                         0.,
                         interaction.loss_draw_id,
                         ComptonAzimuthDrawId,
                         0,
                         0,
                         0};
    } else if (sample.process_id == gpu::em::PhotoelectricProcessId) {
      auto electron = parent;
      electron.pid = static_cast<std::int32_t>(gpu::em::EmPid::Electron);
      electron.energy_GeV = gpu::em::proposalSecondaryTransportEnergyGeV(
          electron.pid,
          gpu::em::ElectronMassGeV + parent.energy_GeV * sample.split_fraction,
          gpu::em::ElectronMassGeV);
      electron.parent_history_id = parent.history_id;
      electron.history_id = first_history_id + child_offset;
      electron.generation = parent.generation + 1;
      electron.step_id = 0;
      electron.reserved = 0;
      output.firstInteraction(parent, gpu::em::PhotoelectricProcessId, electron, nullptr);
      output.secondary(0, electron);
      secondary_count = 1;
      output.record() = {interaction.input_index,
                         parent.history_id,
                         child_offset,
                         1,
                         gpu::em::PhotoelectricProcessId,
                         sample.split_fraction,
                         interaction.loss_quantile,
                         0.,
                         0.,
                         0.,
                         0.,
                         0.,
                         interaction.loss_draw_id,
                         0,
                         0,
                         0,
                         0};
    } else if (sample.process_id == gpu::em::PhotonMuonPairProcessId) {
      // PROPOSAL BKK uses equal polar angles m_mu/E_gamma and opposite
      // azimuths. This is its existing direction approximation, not an exact
      // nuclear recoil solver. Keep its (E in MeV)/MMU operation order.
      auto const k = (parent.energy_GeV * 1000.) / (sample.secondary_mass_GeV * 1000.);
      auto cosine = photonCos(1. / k);
      if (cosine == -1.) cosine *= -1.;
      gpu::em::EmParticleState minus = parent, plus = parent;
      minus.pid = static_cast<std::int32_t>(gpu::em::EmPid::MuonMinus);
      plus.pid = static_cast<std::int32_t>(gpu::em::EmPid::MuonPlus);
      minus.energy_GeV = gpu::em::proposalSecondaryTransportEnergyGeV(
          minus.pid, (1. - sample.split_fraction) * parent.energy_GeV,
          sample.secondary_mass_GeV);
      plus.energy_GeV = gpu::em::proposalSecondaryTransportEnergyGeV(
          plus.pid, sample.split_fraction * parent.energy_GeV, sample.secondary_mass_GeV);
      minus.parent_history_id = plus.parent_history_id = parent.history_id;
      minus.generation = plus.generation = parent.generation + 1;
      minus.step_id = plus.step_id = 0;
      minus.reserved = plus.reserved = 0;
      if (minus.energy_GeV <= gpu::em::transportMassGeV(minus.pid))
        minus.reserved = gpu::em::NativeMuonPairCutFlag;
      if (plus.energy_GeV <= gpu::em::transportMassGeV(plus.pid))
        plus.reserved = gpu::em::NativeMuonPairCutFlag;
      minus.weight = sample.thinning_first_weight;
      plus.weight = sample.thinning_second_weight;
      deflectPhotonChild(parent.direction, cosine, azimuth, minus.direction);
      deflectPhotonChild(parent.direction, cosine,
                         photonFmod(azimuth + 3.1415926535897932384626433832795,
                                    6.283185307179586476925286766559),
                         plus.direction);
      output.firstInteraction(parent, sample.process_id, minus, &plus);
      if (sample.thinning_keep_mask & 0x1U) {
        minus.history_id = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, minus);
      }
      if (sample.thinning_keep_mask & 0x2U) {
        plus.history_id = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, plus);
      }
      output.record() = {interaction.input_index,
                         parent.history_id,
                         child_offset,
                         secondary_count,
                         sample.process_id,
                         sample.split_fraction,
                         sample.split_uniform,
                         sample.azimuth_uniform,
                         0.,
                         0.,
                         0.,
                         0.,
                         sample.split_draw_id,
                         PhotonMuonPairAzimuthDrawId,
                         0,
                         0,
                         0};
    } else {
      auto const electron_energy = parent.energy_GeV * sample.split_fraction;
      auto const positron_energy = parent.energy_GeV - electron_energy;
      auto const electron_cosine =
          sauterCosine(electron_energy, sample.electron_polar_uniform);
      auto const positron_cosine =
          sauterCosine(positron_energy, sample.positron_polar_uniform);
      auto electron = parent;
      electron.pid = static_cast<std::int32_t>(gpu::em::EmPid::Electron);
      electron.energy_GeV = gpu::em::proposalSecondaryTransportEnergyGeV(
          electron.pid, electron_energy, gpu::em::ElectronMassGeV);
      electron.parent_history_id = parent.history_id;
      electron.generation = parent.generation + 1;
      electron.step_id = 0;
      electron.reserved = 0;
      electron.weight = sample.thinning_first_weight;
      deflectPhotonChild(parent.direction, electron_cosine, azimuth, electron.direction);
      auto positron = parent;
      positron.pid = static_cast<std::int32_t>(gpu::em::EmPid::Positron);
      positron.energy_GeV = gpu::em::proposalSecondaryTransportEnergyGeV(
          positron.pid, positron_energy, gpu::em::ElectronMassGeV);
      positron.parent_history_id = parent.history_id;
      positron.generation = parent.generation + 1;
      positron.step_id = 0;
      positron.reserved = 0;
      positron.weight = sample.thinning_second_weight;
      deflectPhotonChild(parent.direction, positron_cosine,
                         photonFmod(azimuth + 3.1415926535897932384626433832795,
                                    6.283185307179586476925286766559),
                         positron.direction);
      output.firstInteraction(parent, gpu::em::PhotonPairProcessId, electron, &positron);
      if ((sample.thinning_keep_mask & 0x1U) != 0) {
        electron.history_id = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, electron);
      }
      if ((sample.thinning_keep_mask & 0x2U) != 0) {
        positron.history_id = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, positron);
      }
      output.record() = {interaction.input_index,
                         parent.history_id,
                         child_offset,
                         secondary_count,
                         gpu::em::PhotonPairProcessId,
                         sample.split_fraction,
                         sample.split_uniform,
                         sample.azimuth_uniform,
                         sample.electron_polar_uniform,
                         sample.positron_polar_uniform,
                         sample.lpm_survival_probability,
                         sample.lpm_uniform,
                         sample.split_draw_id,
                         PhotonPairAzimuthDrawId,
                         PhotonPairElectronPolarDrawId,
                         PhotonPairPositronPolarDrawId,
                         PhotonPairLpmDrawId};
    }
    auto& record = output.record();
    double kept_lepton_weight = 0.;
    if (sample.process_id == gpu::em::PhotoelectricProcessId) {
      kept_lepton_weight = parent.weight;
    } else if (sample.process_id == gpu::em::ComptonProcessId) {
      if (sample.thinning_keep_mask & 0x2U)
        kept_lepton_weight = sample.thinning_second_weight;
    } else {
      if (sample.thinning_keep_mask & 0x1U)
        kept_lepton_weight += sample.thinning_first_weight;
      if (sample.thinning_keep_mask & 0x2U)
        kept_lepton_weight += sample.thinning_second_weight;
    }
    record.weighted_mass_convention_correction_GeV =
        gpu::em::weightedProposalMassCorrectionGeV(
            sample.process_id == gpu::em::PhotonMuonPairProcessId ? 13 : 11,
            sample.process_id == gpu::em::PhotonMuonPairProcessId
                ? sample.secondary_mass_GeV
                : gpu::em::ElectronMassGeV,
            kept_lepton_weight);
    record.thinning_status = sample.thinning_status;
    record.thinning_keep_mask = sample.thinning_keep_mask;
    record.thinning_first_uniform = sample.thinning_first_uniform;
    record.thinning_second_uniform = sample.thinning_second_uniform;
    record.thinning_first_draw_id = gpu::em::EmThinningFirstDrawId;
    record.thinning_second_draw_id = gpu::em::EmThinningSecondDrawId;
    output.finishRecord(secondary_count);
  }

  CORSIKA_HOST_DEVICE inline PhotonFinalStateMaterialization materializePhotonFinalState(
      gpu::em::EmInteractionRecord const& interaction,
      PhotonFinalStateClassification const& classification, std::uint64_t child_offset,
      std::uint64_t first_history_id) {
    PhotonFinalStateMaterialization output{};
    PhotonFinalStateMaterializationWriter writer{output};
    materializePhotonFinalStateInPlace(interaction, classification, child_offset,
                                       first_history_id, writer);
    return output;
  }

} // namespace photon_final_reference
