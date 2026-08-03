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

namespace corsika {

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

  enum class HadronicModelClass : std::uint8_t {
    LowEnergy,
    HighEnergy
  };

  inline constexpr std::string_view hadronicSpeciesClassName(
      HadronicSpeciesClass const value) {
    switch (value) {
      case HadronicSpeciesClass::Nucleon:
        return "nucleon";
      case HadronicSpeciesClass::Pion:
        return "pion";
      case HadronicSpeciesClass::Kaon:
        return "kaon";
      case HadronicSpeciesClass::OtherMeson:
        return "other_meson";
      case HadronicSpeciesClass::OtherBaryon:
        return "other_baryon";
      case HadronicSpeciesClass::Nucleus:
        return "nucleus";
      case HadronicSpeciesClass::OtherHadron:
        return "other_hadron";
    }
    return "unknown";
  }

  inline constexpr std::string_view hadronicModelClassName(
      HadronicModelClass const value) {
    switch (value) {
      case HadronicModelClass::LowEnergy:
        return "low_energy";
      case HadronicModelClass::HighEnergy:
        return "high_energy";
    }
    return "unknown";
  }

  /**
   * A homogeneous CPU/GPU work class.
   *
   * The model and energy bin separate the dominant interaction implementation and its
   * multiplicity scale. mass_number_bin separates nuclei whose final-state cost grows
   * with A. Exact particle properties are deliberately not stored here.
   */
  struct HadronicWorkKey {
    HadronicModelClass model{HadronicModelClass::LowEnergy};
    HadronicSpeciesClass species{HadronicSpeciesClass::OtherHadron};
    std::int16_t energy_bin{};
    std::uint8_t mass_number_bin{};

    friend constexpr bool operator==(HadronicWorkKey const& lhs,
                                     HadronicWorkKey const& rhs) {
      return std::tie(lhs.model, lhs.species, lhs.energy_bin,
                      lhs.mass_number_bin) ==
             std::tie(rhs.model, rhs.species, rhs.energy_bin,
                      rhs.mass_number_bin);
    }

    friend constexpr bool operator<(HadronicWorkKey const& lhs,
                                    HadronicWorkKey const& rhs) {
      return std::tie(lhs.model, lhs.species, lhs.energy_bin,
                      lhs.mass_number_bin) <
             std::tie(rhs.model, rhs.species, rhs.energy_bin,
                      rhs.mass_number_bin);
    }
  };

  struct HadronicWorkClassifierConfig {
    double transition_energy_GeV{79.43282347242814};
    int energy_bins_per_octave{2};
    double minimum_energy_GeV{1.e-9};
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
  inline double deterministicHadronicInteractionCost(
      HadronicWorkKey const& key, double const reference_cost) {
    if (!std::isfinite(reference_cost) ||
        reference_cost <= 0.) {
      throw std::invalid_argument(
          "Hadronic reference cost must be finite and positive");
    }

    auto const energy_bin = std::clamp(
        static_cast<int>(key.energy_bin), -4, 12);
    auto const positive_energy_bin =
        std::max(energy_bin, 0);
    double relative_cost = 1.;
    switch (key.species) {
      case HadronicSpeciesClass::Nucleon:
        relative_cost =
            0.80 +
            (energy_bin >= 0
                 ? 0.10 * positive_energy_bin
                 : 0.06 * energy_bin);
        break;
      case HadronicSpeciesClass::Pion:
        relative_cost =
            0.96 + 0.07 * positive_energy_bin;
        break;
      case HadronicSpeciesClass::Kaon:
        relative_cost =
            1.70 + 0.02 * positive_energy_bin;
        break;
      case HadronicSpeciesClass::OtherMeson:
        relative_cost =
            1.40 + 0.05 * positive_energy_bin;
        break;
      case HadronicSpeciesClass::OtherBaryon:
        relative_cost =
            1.20 + 0.07 * positive_energy_bin;
        break;
      case HadronicSpeciesClass::Nucleus:
        relative_cost =
            (1.20 +
             0.40 * static_cast<double>(
                        key.mass_number_bin)) *
            (1. + 0.04 * positive_energy_bin);
        break;
      case HadronicSpeciesClass::OtherHadron:
        relative_cost =
            1.20 + 0.05 * positive_energy_bin;
        break;
    }
    if (key.model == HadronicModelClass::HighEnergy) {
      relative_cost *= 1.50;
    }
    return reference_cost *
           std::max(relative_cost, 0.10);
  }

  inline HadronicSpeciesClass classifyHadronicSpecies(Code const pid) {
    if (is_nucleus(pid)) {
      return HadronicSpeciesClass::Nucleus;
    }
    if (pid == Code::Proton || pid == Code::AntiProton ||
        pid == Code::Neutron || pid == Code::AntiNeutron) {
      return HadronicSpeciesClass::Nucleon;
    }
    if (pid == Code::PiPlus || pid == Code::PiMinus ||
        pid == Code::Pi0) {
      return HadronicSpeciesClass::Pion;
    }
    if (is_kaon(pid)) {
      return HadronicSpeciesClass::Kaon;
    }

    // PDG hadron codes use the thousands digit for baryons. This also groups strange,
    // charm, and bottom baryons without coupling this scheduler to the generated Code
    // enumeration.
    auto const pdg =
        std::abs(static_cast<std::int64_t>(get_PDG(pid)));
    if (pdg >= 1000 && pdg < 1000000000) {
      return HadronicSpeciesClass::OtherBaryon;
    }
    if (pdg >= 100 && pdg < 1000) {
      return HadronicSpeciesClass::OtherMeson;
    }
    return HadronicSpeciesClass::OtherHadron;
  }

  inline std::uint8_t hadronicMassNumberBin(Code const pid) {
    if (!is_nucleus(pid)) {
      return 0;
    }
    auto mass_number = get_nucleus_A(pid);
    std::uint8_t bin = 0;
    while (mass_number > 1 && bin < 63) {
      mass_number = (mass_number + 1) / 2;
      ++bin;
    }
    return bin;
  }

  inline void validateHadronicWorkClassifierConfig(
      HadronicWorkClassifierConfig const& config) {
    if (!std::isfinite(config.transition_energy_GeV) ||
        config.transition_energy_GeV < 0.) {
      throw std::invalid_argument(
          "Hadronic work transition energy must be finite and non-negative");
    }
    if (config.energy_bins_per_octave <= 0 ||
        config.energy_bins_per_octave > 64) {
      throw std::invalid_argument(
          "Hadronic work energy bins per octave must be in [1, 64]");
    }
    if (!std::isfinite(config.minimum_energy_GeV) ||
        config.minimum_energy_GeV <= 0.) {
      throw std::invalid_argument(
          "Hadronic work minimum energy must be finite and positive");
    }
  }

  /**
   * Classify a hadron without consuming random numbers or modifying the particle.
   */
  inline std::optional<HadronicWorkKey> classifyHadronicWork(
      Code const pid, HEPEnergyType const kinetic_energy,
      HadronicWorkClassifierConfig const& config) {
    if (!is_hadron(pid)) {
      return std::nullopt;
    }
    validateHadronicWorkClassifierConfig(config);

    auto const kinetic_energy_GeV =
        std::max(kinetic_energy / 1_GeV, config.minimum_energy_GeV);
    auto const raw_bin = std::floor(
        std::log2(kinetic_energy_GeV) *
        static_cast<double>(config.energy_bins_per_octave));
    auto const bounded_bin = std::clamp(
        raw_bin,
        static_cast<double>(std::numeric_limits<std::int16_t>::min()),
        static_cast<double>(std::numeric_limits<std::int16_t>::max()));

    return HadronicWorkKey{
        kinetic_energy_GeV < config.transition_energy_GeV
            ? HadronicModelClass::LowEnergy
            : HadronicModelClass::HighEnergy,
        classifyHadronicSpecies(pid),
        static_cast<std::int16_t>(bounded_bin),
        hadronicMassNumberBin(pid)};
  }

  inline double hadronicEnergyBinLowerGeV(
      HadronicWorkKey const& key,
      HadronicWorkClassifierConfig const& config) {
    validateHadronicWorkClassifierConfig(config);
    return std::exp2(
        static_cast<double>(key.energy_bin) /
        static_cast<double>(config.energy_bins_per_octave));
  }

  inline double hadronicEnergyBinUpperGeV(
      HadronicWorkKey const& key,
      HadronicWorkClassifierConfig const& config) {
    validateHadronicWorkClassifierConfig(config);
    return std::exp2(
        static_cast<double>(key.energy_bin + 1) /
        static_cast<double>(config.energy_bins_per_octave));
  }

  struct HadronicWorkClassStatistics {
    std::uint64_t steps{};
    double total_time_ms{};
    double squared_time_ms2{};
    double minimum_time_ms{std::numeric_limits<double>::infinity()};
    double maximum_time_ms{};

    void record(double const time_ms) {
      if (!std::isfinite(time_ms) || time_ms < 0.) {
        throw std::invalid_argument(
            "Hadronic work time sample must be finite and non-negative");
      }
      ++steps;
      total_time_ms += time_ms;
      squared_time_ms2 += time_ms * time_ms;
      minimum_time_ms = std::min(minimum_time_ms, time_ms);
      maximum_time_ms = std::max(maximum_time_ms, time_ms);
    }

    double meanTimeMs() const {
      return steps == 0 ? 0. : total_time_ms / static_cast<double>(steps);
    }

    double standardDeviationMs() const {
      if (steps == 0) {
        return 0.;
      }
      auto const mean = meanTimeMs();
      return std::sqrt(std::max(
          0., squared_time_ms2 / static_cast<double>(steps) - mean * mean));
    }
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
      std::uint64_t sequence_id{};
      double estimated_cost{};
      TPayload payload;
    };

    struct Batch {
      HadronicWorkKey key{};
      double estimated_cost{};
      std::vector<std::uint64_t> sequence_ids;
      std::vector<TPayload> payloads;
    };

    void enqueue(HadronicWorkKey const key, std::uint64_t const sequence_id,
                 double const estimated_cost, TPayload payload) {
      if (!std::isfinite(estimated_cost) || estimated_cost <= 0.) {
        throw std::invalid_argument(
            "Hadronic work estimated cost must be finite and positive");
      }
      auto& bucket = buckets_[key];
      bucket.pending_cost += estimated_cost;
      pending_cost_ += estimated_cost;
      bucket.items.push_back(
          WorkItem{sequence_id, estimated_cost, std::move(payload)});
      ++size_;
    }

    bool empty() const noexcept {
      return size_ == 0;
    }

    std::size_t size() const noexcept {
      return size_;
    }

    std::size_t classCount() const noexcept {
      return buckets_.size();
    }

    double pendingCost() const noexcept {
      return pending_cost_;
    }

    Batch popBalancedBatch(double const target_cost,
                           std::size_t const maximum_items) {
      if (empty()) {
        throw std::logic_error(
            "Cannot pop a batch from an empty hadronic work queue");
      }
      if (!std::isfinite(target_cost) || target_cost <= 0.) {
        throw std::invalid_argument(
            "Hadronic batch target cost must be finite and positive");
      }
      if (maximum_items == 0) {
        throw std::invalid_argument(
            "Hadronic batch maximum item count must be positive");
      }

      // Largest-remaining-work first prevents an expensive class from becoming a serial
      // tail. std::map order provides a deterministic tie break.
      auto selected = buckets_.begin();
      for (auto it = std::next(buckets_.begin()); it != buckets_.end(); ++it) {
        if (it->second.pending_cost > selected->second.pending_cost) {
          selected = it;
        }
      }

      Batch batch;
      batch.key = selected->first;
      auto& bucket = selected->second;
      while (!bucket.items.empty() &&
             batch.payloads.size() < maximum_items) {
        auto& next = bucket.items.front();
        if (!batch.payloads.empty() &&
            batch.estimated_cost + next.estimated_cost > target_cost) {
          break;
        }
        batch.estimated_cost += next.estimated_cost;
        batch.sequence_ids.push_back(next.sequence_id);
        batch.payloads.push_back(std::move(next.payload));
        bucket.pending_cost -= next.estimated_cost;
        pending_cost_ -= next.estimated_cost;
        bucket.items.pop_front();
        --size_;
      }

      if (bucket.items.empty()) {
        buckets_.erase(selected);
      }
      if (size_ == 0) {
        // Avoid exposing a tiny negative residual after many floating-point
        // subtractions.  An empty queue has exactly zero pending work.
        pending_cost_ = 0.;
      }
      return batch;
    }

  private:
    struct Bucket {
      double pending_cost{};
      std::deque<WorkItem> items;
    };

    std::map<HadronicWorkKey, Bucket> buckets_;
    std::size_t size_{};
    double pending_cost_{};
  };

  enum class HadronicQueueFlushTrigger : std::uint8_t {
    EstimatedCost,
    Capacity,
    Drain
  };

  inline constexpr std::string_view
  hadronicQueueFlushTriggerName(
      HadronicQueueFlushTrigger const trigger) {
    switch (trigger) {
      case HadronicQueueFlushTrigger::EstimatedCost:
        return "estimated_cost";
      case HadronicQueueFlushTrigger::Capacity:
        return "capacity";
      case HadronicQueueFlushTrigger::Drain:
        return "drain";
    }
    return "unknown";
  }

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
  inline std::optional<HadronicQueueFlushTrigger>
  decideHadronicQueueFlush(
      std::size_t const pending_items,
      double const pending_cost,
      std::size_t const worker_count,
      std::size_t const minimum_pending_items,
      double const target_cost_per_worker,
      std::size_t const maximum_batch_items) {
    if (worker_count == 0 || minimum_pending_items == 0 ||
        maximum_batch_items == 0) {
      throw std::invalid_argument(
          "Hadronic flush worker and item limits must be positive");
    }
    if (!std::isfinite(pending_cost) || pending_cost < 0. ||
        !std::isfinite(target_cost_per_worker) ||
        target_cost_per_worker <= 0.) {
      throw std::invalid_argument(
          "Hadronic flush costs must be finite and non-negative/positive");
    }
    if (pending_items < minimum_pending_items) {
      return std::nullopt;
    }

    auto const target_total_cost =
        target_cost_per_worker *
        static_cast<double>(worker_count);
    if (!std::isfinite(target_total_cost)) {
      throw std::overflow_error(
          "Hadronic flush target cost overflowed");
    }
    if (pending_cost >= target_total_cost) {
      return HadronicQueueFlushTrigger::EstimatedCost;
    }

    auto const maximum_size =
        std::numeric_limits<std::size_t>::max();
    auto const capacity =
        maximum_batch_items >
                maximum_size / worker_count
            ? maximum_size
            : worker_count * maximum_batch_items;
    if (pending_items >=
        std::max(minimum_pending_items, capacity)) {
      return HadronicQueueFlushTrigger::Capacity;
    }
    return std::nullopt;
  }

  /**
   * The deterministic assignment of homogeneous batches to one worker process.
   */
  template <typename TPayload>
  struct HadronicWorkerAssignment {
    using queue_type = ClassifiedHadronicWorkQueue<TPayload>;
    using batch_type = typename queue_type::Batch;

    std::size_t worker_id{};
    double estimated_cost{};
    std::vector<batch_type> batches;
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
    std::size_t worker_id{};
    double estimated_cost{};
    std::size_t classified_batches{};
    std::vector<std::uint64_t> sequence_ids;
    std::vector<TPayload> payloads;
  };

  template <typename TPayload>
  std::vector<HadronicWorkerSuperBatch<TPayload>>
  coalesceHadronicWorkerAssignments(
      std::vector<HadronicWorkerAssignment<TPayload>>& assignments) {
    std::vector<HadronicWorkerSuperBatch<TPayload>> result;
    result.reserve(assignments.size());
    for (auto& assignment : assignments) {
      std::size_t item_count = 0;
      for (auto const& batch : assignment.batches) {
        if (batch.sequence_ids.size() != batch.payloads.size()) {
          throw std::logic_error(
              "Hadronic classified batch identity/payload sizes differ");
        }
        item_count += batch.payloads.size();
      }
      if (item_count == 0) {
        continue;
      }

      HadronicWorkerSuperBatch<TPayload> combined;
      combined.worker_id = assignment.worker_id;
      combined.estimated_cost = assignment.estimated_cost;
      combined.classified_batches = assignment.batches.size();
      combined.sequence_ids.reserve(item_count);
      combined.payloads.reserve(item_count);
      for (auto& batch : assignment.batches) {
        combined.sequence_ids.insert(
            combined.sequence_ids.end(),
            batch.sequence_ids.begin(),
            batch.sequence_ids.end());
        for (auto& payload : batch.payloads) {
          combined.payloads.push_back(std::move(payload));
        }
      }
      result.push_back(std::move(combined));
    }
    return result;
  }

  /**
   * Drain a classified queue and balance its batches across worker processes.
   *
   * Batches remain homogeneous in model/species/energy/A.  Longest-processing-time
   * first assignment minimizes the expensive serial tail, while worker-id and
   * sequence-id tie breaks make the plan reproducible.  This function only plans work;
   * it never touches a CORSIKA Stack or calls a non-thread-safe event generator.
   */
  template <typename TPayload>
  std::vector<HadronicWorkerAssignment<TPayload>>
  planHadronicWorkerAssignments(
      ClassifiedHadronicWorkQueue<TPayload>& queue,
      std::size_t const worker_count, double const target_batch_cost,
      std::size_t const maximum_batch_items) {
    if (worker_count == 0) {
      throw std::invalid_argument(
          "Hadronic worker count must be positive");
    }

    using Assignment = HadronicWorkerAssignment<TPayload>;
    using Batch = typename Assignment::batch_type;
    std::vector<Batch> batches;
    while (!queue.empty()) {
      batches.push_back(
          queue.popBalancedBatch(
              target_batch_cost, maximum_batch_items));
    }

    auto firstSequence = [](Batch const& batch) {
      return batch.sequence_ids.empty()
                 ? std::numeric_limits<std::uint64_t>::max()
                 : batch.sequence_ids.front();
    };
    std::stable_sort(
        batches.begin(), batches.end(),
        [&](Batch const& lhs, Batch const& rhs) {
          if (lhs.estimated_cost != rhs.estimated_cost) {
            return lhs.estimated_cost > rhs.estimated_cost;
          }
          if (!(lhs.key == rhs.key)) {
            return lhs.key < rhs.key;
          }
          return firstSequence(lhs) < firstSequence(rhs);
        });

    std::vector<Assignment> assignments(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      assignments[worker].worker_id = worker;
    }

    for (auto& batch : batches) {
      auto selected = assignments.begin();
      for (auto it = std::next(assignments.begin());
           it != assignments.end(); ++it) {
        if (it->estimated_cost < selected->estimated_cost) {
          selected = it;
        }
      }
      selected->estimated_cost += batch.estimated_cost;
      selected->batches.push_back(std::move(batch));
    }
    return assignments;
  }

} // namespace corsika
