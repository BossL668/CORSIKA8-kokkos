/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace corsika::kokkos {

  /**
   * Coarse projectile families that have measurably different hadronic-model costs.
   *
   * Charge conjugates intentionally share a class. The class is a scheduling key, not a
   * physics identity; the exact PID remains in the particle payload.
   */
  enum class HadronicSpeciesClass : std::uint8_t {
    Nucleon,
    Pion,
    Kaon,
    OtherMeson,
    OtherBaryon,
    Nucleus,
    OtherHadron
  };

  enum class HadronicModelClass : std::uint8_t { LowEnergy, HighEnergy };

  inline constexpr std::string_view hadronic_species_class_name(
      HadronicSpeciesClass const value);

  inline constexpr std::string_view hadronic_model_class_name(
      HadronicModelClass const value);

  /**
   * A homogeneous CPU/GPU work class.
   *
   * The model and energy bin separate the dominant interaction implementation and its
   * multiplicity scale. mass_number_bin separates nuclei whose final-state cost grows
   * with A. Exact particle properties are deliberately not stored here.
   */
  struct HadronicWorkKey {
    HadronicModelClass model_{HadronicModelClass::LowEnergy};
    HadronicSpeciesClass species_{HadronicSpeciesClass::OtherHadron};
    std::int16_t energyBin_{};
    std::uint8_t massNumberBin_{};

    friend constexpr bool operator==(HadronicWorkKey const& lhs,
                                     HadronicWorkKey const& rhs);

    friend constexpr bool operator<(HadronicWorkKey const& lhs,
                                    HadronicWorkKey const& rhs);
  };

  struct HadronicWorkClassifierConfig {
    double transitionEnergyGeV_{79.43282347242814};
    int energyBinsPerOctave_{2};
    double minimumEnergyGeV_{1.e-9};
  };

  /**
   * Deterministic relative cost model for class-balanced hadronic batches.
   *
   * Feeding measured wall-clock durations back into the live scheduler makes
   * a fixed-seed shower depend on OS timing noise.  These coefficients encode
   * the smooth species/energy trend measured with FLUKA 2025, while the
   * caller-supplied reference cost retains a portable overall scale.  The
   * estimate is a scheduling hint only and cannot change a physics request.
   */
  inline double deterministic_hadronic_interaction_cost(HadronicWorkKey const& key,
                                                        double const reference_cost);

  inline HadronicSpeciesClass classify_hadronic_species(Code const pid);

  inline std::uint8_t hadronic_mass_number_bin(Code const pid);

  inline void validate_hadronic_work_classifier_config(
      HadronicWorkClassifierConfig const& config);

  /**
   * Classify a hadron without consuming random numbers or modifying the particle.
   */
  inline std::optional<HadronicWorkKey> classify_hadronic_work(
      Code const pid, HEPEnergyType const kinetic_energy,
      HadronicWorkClassifierConfig const& config);

  inline double hadronic_energy_bin_lower_gev(HadronicWorkKey const& key,
                                              HadronicWorkClassifierConfig const& config);

  inline double hadronic_energy_bin_upper_gev(HadronicWorkKey const& key,
                                              HadronicWorkClassifierConfig const& config);

  struct HadronicWorkClassStatistics {
    std::uint64_t steps_{};
    double totalTimeMs_{};
    double squaredTimeMs2_{};
    double minimumTimeMs_{std::numeric_limits<double>::infinity()};
    double maximumTimeMs_{};

    void record(double const time_ms);

    double meanTimeMs() const;

    double standardDeviationMs() const;
  };

  /**
   * Deterministic, class-separated staging storage for a future parallel backend.
   *
   * Each returned batch contains one model/species/energy/A class. Batch size is
   * limited by an estimated cost budget, so expensive classes contain fewer particles
   * than cheap classes. FIFO order inside a class and explicit sequence IDs make the
   * result merge order reproducible.
   */
  template <typename TPayload>
  class ClassifiedHadronicWorkQueue {
  public:
    struct WorkItem {
      std::uint64_t sequenceId_{};
      double estimatedCost_{};
      TPayload payload_;
    };

    struct Batch {
      HadronicWorkKey key_{};
      double estimatedCost_{};
      std::vector<std::uint64_t> sequenceIds_;
      std::vector<TPayload> payloads_;
    };

    void enqueue(HadronicWorkKey const key, std::uint64_t const sequence_id,
                 double const estimated_cost, TPayload payload);

    bool empty() const noexcept;

    std::size_t size() const noexcept;

    std::size_t classCount() const noexcept;

    double pendingCost() const noexcept;

    Batch popBalancedBatch(double const target_cost, std::size_t const maximum_items);

  private:
    struct Bucket {
      double pendingCost_{};
      std::deque<WorkItem> items_;
    };

    std::map<HadronicWorkKey, Bucket> buckets_;
    std::size_t size_{};
    double pendingCost_{};
  };

  enum class HadronicQueueFlushTrigger : std::uint8_t { EstimatedCost, Capacity, Drain };

  inline constexpr std::string_view hadronic_queue_flush_trigger_name(
      HadronicQueueFlushTrigger const trigger);

  /**
   * Decide whether a live hadronic queue contains enough predicted work to
   * occupy every worker for approximately one target interval.
   *
   * minimum_pending_items remains a latency floor.  The normal trigger is
   * predicted total cost, not a raw particle count, because projectile classes
   * have different final-state costs.  The capacity trigger is a deterministic
   * liveness bound for a systematically underestimated cost model.  Drain is
   * deliberately not returned here: the cascade supplies it when no independent
   * CPU/GPU work remains.
   */
  inline std::optional<HadronicQueueFlushTrigger> decide_hadronic_queue_flush(
      std::size_t const pending_items, double const pending_cost,
      std::size_t const worker_count, std::size_t const minimum_pending_items,
      double const target_cost_per_worker, std::size_t const maximum_batch_items);

  /**
   * The deterministic assignment of homogeneous batches to one worker process.
   */
  template <typename TPayload>
  struct HadronicWorkerAssignment {
    using queue_type = ClassifiedHadronicWorkQueue<TPayload>;
    using batch_type = typename queue_type::Batch;

    std::size_t workerId_{};
    double estimatedCost_{};
    std::vector<batch_type> batches_;
  };

  /**
   * One IPC envelope per worker and flush.
   *
   * Classification remains explicit in HadronicWorkerAssignment::batches and
   * therefore still controls cost estimation and worker balancing. FLUKA,
   * however, evaluates every request independently and does not benefit from
   * receiving a separate socket header for each homogeneous micro-batch.
   * Coalescing the already assigned micro-batches preserves their deterministic
   * request order while avoiding thousands of 1--3 request IPC transactions.
   */
  template <typename TPayload>
  struct HadronicWorkerSuperBatch {
    std::size_t workerId_{};
    double estimatedCost_{};
    std::size_t classifiedBatches_{};
    std::vector<std::uint64_t> sequenceIds_;
    std::vector<TPayload> payloads_;
  };

  template <typename TPayload>
  std::vector<HadronicWorkerSuperBatch<TPayload>> coalesce_hadronic_worker_assignments(
      std::vector<HadronicWorkerAssignment<TPayload>>& assignments);

  /**
   * Drain a classified queue and balance its batches across worker processes.
   *
   * Batches remain homogeneous in model/species/energy/A.  Longest-processing-time
   * first assignment minimizes the expensive serial tail, while worker-id and
   * sequence-id tie breaks make the plan reproducible.  This function only plans work;
   * it never touches a CORSIKA Stack or calls a non-thread-safe event generator.
   */
  template <typename TPayload>
  std::vector<HadronicWorkerAssignment<TPayload>> plan_hadronic_worker_assignments(
      ClassifiedHadronicWorkQueue<TPayload>& queue, std::size_t const worker_count,
      double const target_batch_cost, std::size_t const maximum_batch_items);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/HadronicWorkQueue.inl>
