/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos {
  inline constexpr std::string_view hadronic_species_class_name(
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
  inline constexpr std::string_view hadronic_model_class_name(
      HadronicModelClass const value) {
    switch (value) {
      case HadronicModelClass::LowEnergy:
        return "low_energy";
      case HadronicModelClass::HighEnergy:
        return "high_energy";
    }
    return "unknown";
  }
  inline double deterministic_hadronic_interaction_cost(HadronicWorkKey const& key,
                                                        double const reference_cost) {
    if (!std::isfinite(reference_cost) || reference_cost <= 0.) {
      throw std::invalid_argument("Hadronic reference cost must be finite and positive");
    }

    auto const energy_bin = std::clamp(static_cast<int>(key.energyBin_), -4, 12);
    auto const positive_energy_bin = std::max(energy_bin, 0);
    double relative_cost = 1.;
    switch (key.species_) {
      case HadronicSpeciesClass::Nucleon:
        relative_cost =
            0.80 + (energy_bin >= 0 ? 0.10 * positive_energy_bin : 0.06 * energy_bin);
        break;
      case HadronicSpeciesClass::Pion:
        relative_cost = 0.96 + 0.07 * positive_energy_bin;
        break;
      case HadronicSpeciesClass::Kaon:
        relative_cost = 1.70 + 0.02 * positive_energy_bin;
        break;
      case HadronicSpeciesClass::OtherMeson:
        relative_cost = 1.40 + 0.05 * positive_energy_bin;
        break;
      case HadronicSpeciesClass::OtherBaryon:
        relative_cost = 1.20 + 0.07 * positive_energy_bin;
        break;
      case HadronicSpeciesClass::Nucleus:
        relative_cost = (1.20 + 0.40 * static_cast<double>(key.massNumberBin_)) *
                        (1. + 0.04 * positive_energy_bin);
        break;
      case HadronicSpeciesClass::OtherHadron:
        relative_cost = 1.20 + 0.05 * positive_energy_bin;
        break;
    }
    if (key.model_ == HadronicModelClass::HighEnergy) { relative_cost *= 1.50; }
    return reference_cost * std::max(relative_cost, 0.10);
  }
  inline HadronicSpeciesClass classify_hadronic_species(Code const pid) {
    if (is_nucleus(pid)) { return HadronicSpeciesClass::Nucleus; }
    if (pid == Code::Proton || pid == Code::AntiProton || pid == Code::Neutron ||
        pid == Code::AntiNeutron) {
      return HadronicSpeciesClass::Nucleon;
    }
    if (pid == Code::PiPlus || pid == Code::PiMinus || pid == Code::Pi0) {
      return HadronicSpeciesClass::Pion;
    }
    if (is_kaon(pid)) { return HadronicSpeciesClass::Kaon; }

    // PDG hadron codes use the thousands digit for baryons. This also groups strange,
    // charm, and bottom baryons without coupling this scheduler to the generated Code
    // enumeration.
    auto const pdg = std::abs(static_cast<std::int64_t>(get_PDG(pid)));
    if (pdg >= 1000 && pdg < 1000000000) { return HadronicSpeciesClass::OtherBaryon; }
    if (pdg >= 100 && pdg < 1000) { return HadronicSpeciesClass::OtherMeson; }
    return HadronicSpeciesClass::OtherHadron;
  }
  inline std::uint8_t hadronic_mass_number_bin(Code const pid) {
    if (!is_nucleus(pid)) { return 0; }
    auto mass_number = get_nucleus_A(pid);
    std::uint8_t bin = 0;
    while (mass_number > 1 && bin < 63) {
      mass_number = (mass_number + 1) / 2;
      ++bin;
    }
    return bin;
  }
  inline void validate_hadronic_work_classifier_config(
      HadronicWorkClassifierConfig const& config) {
    if (!std::isfinite(config.transitionEnergyGeV_) || config.transitionEnergyGeV_ < 0.) {
      throw std::invalid_argument(
          "Hadronic work transition energy must be finite and non-negative");
    }
    if (config.energyBinsPerOctave_ <= 0 || config.energyBinsPerOctave_ > 64) {
      throw std::invalid_argument(
          "Hadronic work energy bins per octave must be in [1, 64]");
    }
    if (!std::isfinite(config.minimumEnergyGeV_) || config.minimumEnergyGeV_ <= 0.) {
      throw std::invalid_argument(
          "Hadronic work minimum energy must be finite and positive");
    }
  }
  inline std::optional<HadronicWorkKey> classify_hadronic_work(
      Code const pid, HEPEnergyType const kinetic_energy,
      HadronicWorkClassifierConfig const& config) {
    if (!is_hadron(pid)) { return std::nullopt; }
    validate_hadronic_work_classifier_config(config);

    auto const kinetic_energy_GeV =
        std::max(kinetic_energy / 1_GeV, config.minimumEnergyGeV_);
    auto const raw_bin = std::floor(std::log2(kinetic_energy_GeV) *
                                    static_cast<double>(config.energyBinsPerOctave_));
    auto const bounded_bin =
        std::clamp(raw_bin, static_cast<double>(std::numeric_limits<std::int16_t>::min()),
                   static_cast<double>(std::numeric_limits<std::int16_t>::max()));

    return HadronicWorkKey{
        kinetic_energy_GeV < config.transitionEnergyGeV_ ? HadronicModelClass::LowEnergy
                                                         : HadronicModelClass::HighEnergy,
        classify_hadronic_species(pid), static_cast<std::int16_t>(bounded_bin),
        hadronic_mass_number_bin(pid)};
  }
  inline double hadronic_energy_bin_lower_gev(
      HadronicWorkKey const& key, HadronicWorkClassifierConfig const& config) {
    validate_hadronic_work_classifier_config(config);
    return std::exp2(static_cast<double>(key.energyBin_) /
                     static_cast<double>(config.energyBinsPerOctave_));
  }
  inline double hadronic_energy_bin_upper_gev(
      HadronicWorkKey const& key, HadronicWorkClassifierConfig const& config) {
    validate_hadronic_work_classifier_config(config);
    return std::exp2(static_cast<double>(key.energyBin_ + 1) /
                     static_cast<double>(config.energyBinsPerOctave_));
  }
  inline constexpr std::string_view hadronic_queue_flush_trigger_name(
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
  inline std::optional<HadronicQueueFlushTrigger> decide_hadronic_queue_flush(
      std::size_t const pending_items, double const pending_cost,
      std::size_t const worker_count, std::size_t const minimum_pending_items,
      double const target_cost_per_worker, std::size_t const maximum_batch_items) {
    if (worker_count == 0 || minimum_pending_items == 0 || maximum_batch_items == 0) {
      throw std::invalid_argument(
          "Hadronic flush worker and item limits must be positive");
    }
    if (!std::isfinite(pending_cost) || pending_cost < 0. ||
        !std::isfinite(target_cost_per_worker) || target_cost_per_worker <= 0.) {
      throw std::invalid_argument(
          "Hadronic flush costs must be finite and non-negative/positive");
    }
    if (pending_items < minimum_pending_items) { return std::nullopt; }

    auto const target_total_cost =
        target_cost_per_worker * static_cast<double>(worker_count);
    if (!std::isfinite(target_total_cost)) {
      throw std::overflow_error("Hadronic flush target cost overflowed");
    }
    if (pending_cost >= target_total_cost) {
      return HadronicQueueFlushTrigger::EstimatedCost;
    }

    auto const maximum_size = std::numeric_limits<std::size_t>::max();
    auto const capacity = maximum_batch_items > maximum_size / worker_count
                              ? maximum_size
                              : worker_count * maximum_batch_items;
    if (pending_items >= std::max(minimum_pending_items, capacity)) {
      return HadronicQueueFlushTrigger::Capacity;
    }
    return std::nullopt;
  }
  template <typename TPayload>
  std::vector<HadronicWorkerSuperBatch<TPayload>> coalesce_hadronic_worker_assignments(
      std::vector<HadronicWorkerAssignment<TPayload>>& assignments) {
    std::vector<HadronicWorkerSuperBatch<TPayload>> result;
    result.reserve(assignments.size());
    for (auto& assignment : assignments) {
      std::size_t item_count = 0;
      for (auto const& batch : assignment.batches_) {
        if (batch.sequenceIds_.size() != batch.payloads_.size()) {
          throw std::logic_error(
              "Hadronic classified batch identity/payload sizes differ");
        }
        item_count += batch.payloads_.size();
      }
      if (item_count == 0) { continue; }

      HadronicWorkerSuperBatch<TPayload> combined;
      combined.workerId_ = assignment.workerId_;
      combined.estimatedCost_ = assignment.estimatedCost_;
      combined.classifiedBatches_ = assignment.batches_.size();
      combined.sequenceIds_.reserve(item_count);
      combined.payloads_.reserve(item_count);
      for (auto& batch : assignment.batches_) {
        combined.sequenceIds_.insert(combined.sequenceIds_.end(),
                                     batch.sequenceIds_.begin(),
                                     batch.sequenceIds_.end());
        for (auto& payload : batch.payloads_) {
          combined.payloads_.push_back(std::move(payload));
        }
      }
      result.push_back(std::move(combined));
    }
    return result;
  }
  template <typename TPayload>
  std::vector<HadronicWorkerAssignment<TPayload>> plan_hadronic_worker_assignments(
      ClassifiedHadronicWorkQueue<TPayload>& queue, std::size_t const worker_count,
      double const target_batch_cost, std::size_t const maximum_batch_items) {
    if (worker_count == 0) {
      throw std::invalid_argument("Hadronic worker count must be positive");
    }

    using assignment_type = HadronicWorkerAssignment<TPayload>;
    using batch_type = typename assignment_type::batch_type;
    std::vector<batch_type> batches;
    while (!queue.empty()) {
      batches.push_back(queue.popBalancedBatch(target_batch_cost, maximum_batch_items));
    }

    auto firstSequence = [](batch_type const& batch) {
      return batch.sequenceIds_.empty() ? std::numeric_limits<std::uint64_t>::max()
                                        : batch.sequenceIds_.front();
    };
    std::stable_sort(batches.begin(), batches.end(),
                     [&](batch_type const& lhs, batch_type const& rhs) {
                       if (lhs.estimatedCost_ != rhs.estimatedCost_) {
                         return lhs.estimatedCost_ > rhs.estimatedCost_;
                       }
                       if (!(lhs.key_ == rhs.key_)) { return lhs.key_ < rhs.key_; }
                       return firstSequence(lhs) < firstSequence(rhs);
                     });

    std::vector<assignment_type> assignments(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      assignments[worker].workerId_ = worker;
    }

    for (auto& batch : batches) {
      auto selected = assignments.begin();
      for (auto it = std::next(assignments.begin()); it != assignments.end(); ++it) {
        if (it->estimatedCost_ < selected->estimatedCost_) { selected = it; }
      }
      selected->estimatedCost_ += batch.estimatedCost_;
      selected->batches_.push_back(std::move(batch));
    }
    return assignments;
  }
  inline constexpr bool operator==(HadronicWorkKey const& lhs,
                                   HadronicWorkKey const& rhs) {
    return std::tie(lhs.model_, lhs.species_, lhs.energyBin_, lhs.massNumberBin_) ==
           std::tie(rhs.model_, rhs.species_, rhs.energyBin_, rhs.massNumberBin_);
  }
  inline constexpr bool operator<(HadronicWorkKey const& lhs,
                                  HadronicWorkKey const& rhs) {
    return std::tie(lhs.model_, lhs.species_, lhs.energyBin_, lhs.massNumberBin_) <
           std::tie(rhs.model_, rhs.species_, rhs.energyBin_, rhs.massNumberBin_);
  }
  inline void HadronicWorkClassStatistics::record(double const time_ms) {
    if (!std::isfinite(time_ms) || time_ms < 0.) {
      throw std::invalid_argument(
          "Hadronic work time sample must be finite and non-negative");
    }
    ++steps_;
    totalTimeMs_ += time_ms;
    squaredTimeMs2_ += time_ms * time_ms;
    minimumTimeMs_ = std::min(minimumTimeMs_, time_ms);
    maximumTimeMs_ = std::max(maximumTimeMs_, time_ms);
  }
  inline double HadronicWorkClassStatistics::meanTimeMs() const {
    return steps_ == 0 ? 0. : totalTimeMs_ / static_cast<double>(steps_);
  }
  inline double HadronicWorkClassStatistics::standardDeviationMs() const {
    if (steps_ == 0) { return 0.; }
    auto const mean = meanTimeMs();
    return std::sqrt(
        std::max(0., squaredTimeMs2_ / static_cast<double>(steps_) - mean * mean));
  }
  template <typename TPayload>
  inline void ClassifiedHadronicWorkQueue<TPayload>::enqueue(
      HadronicWorkKey const key, std::uint64_t const sequence_id,
      double const estimated_cost, TPayload payload) {
    if (!std::isfinite(estimated_cost) || estimated_cost <= 0.) {
      throw std::invalid_argument(
          "Hadronic work estimated cost must be finite and positive");
    }
    auto& bucket = buckets_[key];
    bucket.pendingCost_ += estimated_cost;
    pendingCost_ += estimated_cost;
    bucket.items_.push_back(WorkItem{sequence_id, estimated_cost, std::move(payload)});
    ++size_;
  }
  template <typename TPayload>
  inline bool ClassifiedHadronicWorkQueue<TPayload>::empty() const noexcept {
    return size_ == 0;
  }
  template <typename TPayload>
  inline std::size_t ClassifiedHadronicWorkQueue<TPayload>::size() const noexcept {
    return size_;
  }
  template <typename TPayload>
  inline std::size_t ClassifiedHadronicWorkQueue<TPayload>::classCount() const noexcept {
    return buckets_.size();
  }
  template <typename TPayload>
  inline double ClassifiedHadronicWorkQueue<TPayload>::pendingCost() const noexcept {
    return pendingCost_;
  }
  template <typename TPayload>
  inline typename ClassifiedHadronicWorkQueue<TPayload>::Batch
  ClassifiedHadronicWorkQueue<TPayload>::popBalancedBatch(
      double const target_cost, std::size_t const maximum_items) {
    if (empty()) {
      throw std::logic_error("Cannot pop a batch from an empty hadronic work queue");
    }
    if (!std::isfinite(target_cost) || target_cost <= 0.) {
      throw std::invalid_argument(
          "Hadronic batch target cost must be finite and positive");
    }
    if (maximum_items == 0) {
      throw std::invalid_argument("Hadronic batch maximum item count must be positive");
    }

    // Largest-remaining-work first prevents an expensive class from becoming a serial
    // tail. std::map order provides a deterministic tie break.
    auto selected = buckets_.begin();
    for (auto it = std::next(buckets_.begin()); it != buckets_.end(); ++it) {
      if (it->second.pendingCost_ > selected->second.pendingCost_) { selected = it; }
    }

    Batch batch;
    batch.key_ = selected->first;
    auto& bucket = selected->second;
    while (!bucket.items_.empty() && batch.payloads_.size() < maximum_items) {
      auto& next = bucket.items_.front();
      if (!batch.payloads_.empty() &&
          batch.estimatedCost_ + next.estimatedCost_ > target_cost) {
        break;
      }
      batch.estimatedCost_ += next.estimatedCost_;
      batch.sequenceIds_.push_back(next.sequenceId_);
      batch.payloads_.push_back(std::move(next.payload_));
      bucket.pendingCost_ -= next.estimatedCost_;
      pendingCost_ -= next.estimatedCost_;
      bucket.items_.pop_front();
      --size_;
    }

    if (bucket.items_.empty()) { buckets_.erase(selected); }
    if (size_ == 0) {
      // Avoid exposing a tiny negative residual after many floating-point
      // subtractions.  An empty queue has exactly zero pending work.
      pendingCost_ = 0.;
    }
    return batch;
  }
} // namespace corsika::kokkos
