/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4C8AirOutput.inl>
#include <corsika/detail/kokkos/egs4/Egs4C8Rare.inl>
#include <corsika/detail/kokkos/egs4/Egs4C8Thinning.inl>
#include <limits>
#include <set>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace corsika::kokkos::egs4::c8_adapter {
  // Native C++ EGS4 backend building block for C8. Curved coordinates, optical
  // clocks, RNG and all three species remain resident across atmospheric layers.
  // No PROPOSAL fallback or false hasProposalTable() compatibility flag.
  template <class TExec>
  class AirWavefront {
  public:
    using records_type = Kokkos::View<AirRecord*, typename TExec::memory_space>;
    using outcomes_type = Kokkos::View<AirOutcome*, typename TExec::memory_space>;
    using outputs_type = Kokkos::View<AirOutput*, typename TExec::memory_space>;
    using host_transfers_type =
        Kokkos::View<RareAirTransfer*, typename TExec::memory_space>;
    using offsets_type = Kokkos::View<int*, typename TExec::memory_space>;
    using thin_results_type =
        Kokkos::View<thinning_result_type*, typename TExec::memory_space>;
    struct Batch {
      records_type before_, after_;
      outcomes_type outcomes_;
      outputs_type outputs_;
      host_transfers_type host_;
      int created_{};
      thin_results_type thinning_;
    };
    KOKKOS_INLINE_FUNCTION static void clearTransfer(RareAirTransfer& h, unsigned mask) {
      if (!(mask & overhead::compact_clear)) {
        h = RareAirTransfer{};
        return;
      }
      h.valid_ = false;
      h.status_ = InteractionStatus::invalid_input;
      h.count_ = 0;
      h.created_ = 0;
      h.targetRestEnergyMeV_ = 0.;
      h.hasRhoDecay_ = false;
      h.hasResonanceDecay_ = false;
      h.resonanceChildren_ = 0;
    }
    struct Advance {
      ModelView model_;
      AirEnvironment env_;
      TransportSettings settings_;
      records_type before_, after_;
      outcomes_type outcomes_;
      outputs_type outputs_;
      host_transfers_type host_;
      bool resolveRare_;
      RareMaterial rare_;
      PhotonuclearPolicy nuclear_;
      thinning_config_type thinning_;
      thin_results_type decisions_;
      bool initializeInKernel_{};
      KOKKOS_INLINE_FUNCTION void operator()(int i, int& errors) const {
        if (initializeInKernel_) {
          after_(i) = before_(i);
          // Ordinary histories do not otherwise assign this slot. Clearing it
          // here avoids a separate deep-copy/fence without leaking old vertices.
          clearTransfer(host_(i), env_.overheadMask_);
        }
        auto r = advance_air_record(model_, after_(i), env_, settings_);
        // Freeze the transported segment BEFORE a stochastic vertex changes the
        // surviving lepton energy. It must not alter the preceding radio track.
        outputs_(i) = make_air_output(before_(i), after_(i), r, env_);
        if (resolveRare_ && r.action_ == AirAction::needs_host_channel) {
          host_(i) = resolve_rare_air_record(model_, after_(i), r, env_, settings_, rare_,
                                             nuclear_);
          errors += !host_(i).valid_;
        }
        outcomes_(i) = r;
        decisions_(i) = thin_air_vertex(thinning_, after_(i), r);
        errors += decisions_(i).status_ == thinning_status_type::InvalidInput;
        errors += r.action_ == AirAction::error || !outputs_(i).valid_;
      }
    };
    struct TransportCounts {
      int errors_{}, rare_{};
    };
    // The main transport kernel must not inherit the register/stack footprint of
    // infrequent photonuclear/resonance final states. Compact their indices only;
    // the physical resolver below is unchanged and still writes original slots.
    struct TransportOnly {
      using value_type = TransportCounts;
      ModelView model_;
      AirEnvironment env_;
      TransportSettings settings_;
      records_type before_, after_;
      outcomes_type outcomes_;
      outputs_type outputs_;
      host_transfers_type host_;
      thinning_config_type thinning_;
      thin_results_type decisions_;
      offsets_type rareIndices_, rareCursor_;
      KOKKOS_INLINE_FUNCTION void init(value_type& v) const { v = {}; }
      KOKKOS_INLINE_FUNCTION void join(value_type& a, value_type const& b) const {
        a.errors_ += b.errors_;
        a.rare_ += b.rare_;
      }
      KOKKOS_INLINE_FUNCTION void operator()(int i, value_type& count) const {
        after_(i) = before_(i);
        clearTransfer(host_(i), env_.overheadMask_);
        auto r = advance_air_record(model_, after_(i), env_, settings_);
        outputs_(i) = make_air_output(before_(i), after_(i), r, env_);
        outcomes_(i) = r;
        decisions_(i) = thin_air_vertex(thinning_, after_(i), r);
        count.errors_ += r.action_ == AirAction::error || !outputs_(i).valid_;
        count.errors_ += decisions_(i).status_ == thinning_status_type::InvalidInput;
        if (r.action_ == AirAction::needs_host_channel) {
          int const slot = Kokkos::atomic_fetch_add(&rareCursor_(0), 1);
          rareIndices_(slot) = i;
          ++count.rare_;
        }
      }
    };
    struct ResolveCompactedRare {
      ModelView model_;
      AirEnvironment env_;
      TransportSettings settings_;
      records_type after_;
      outcomes_type outcomes_;
      host_transfers_type host_;
      RareMaterial rare_;
      PhotonuclearPolicy nuclear_;
      thinning_config_type thinning_;
      thin_results_type decisions_;
      offsets_type indices_;
      KOKKOS_INLINE_FUNCTION void operator()(int slot, int& errors) const {
        int const i = indices_(slot);
        auto r = outcomes_(i);
        host_(i) = resolve_rare_air_record(model_, after_(i), r, env_, settings_, rare_,
                                           nuclear_);
        outcomes_(i) = r;
        decisions_(i) = thin_air_vertex(thinning_, after_(i), r);
        errors += !host_(i).valid_ || r.action_ == AirAction::error;
        errors += decisions_(i).status_ == thinning_status_type::InvalidInput;
      }
    };
    struct Offset {
      outcomes_type outcomes_;
      host_transfers_type host_;
      offsets_type offsets_;
      bool children_;
      thin_results_type decisions_;
      KOKKOS_INLINE_FUNCTION void operator()(int i, int& sum, bool final) const {
        if (final) offsets_(i) = sum;
        auto const& r = outcomes_(i);
        if (r.action_ == AirAction::replaced)
          for (int j = 0; j < r.history_.childCount_; ++j)
            sum += children_ || ((decisions_(i).keepMask_ >> j) & 1U);
        else
          sum += !children_ && r.action_ == AirAction::active ? 1 : 0;
        if (children_)
          sum += host_(i).created_;
        else
          for (int j = 0; j < host_(i).count_; ++j)
            sum += host_(i).vertices_[j].kind_ == HostVertexKind::native_em;
      }
    };
    // The two stable prefixes are integer sums over the same input order.
    // Combining them does not change thinning, IDs, RNG or daughter ordering.
    struct PrefixCounts {
      int live_{}, created_{};
    };
    struct CombinedOffset {
      using value_type = PrefixCounts;
      outcomes_type outcomes_;
      host_transfers_type host_;
      offsets_type offsets_, childOffsets_;
      thin_results_type decisions_;
      KOKKOS_INLINE_FUNCTION void init(value_type& v) const {
        v.live_ = 0;
        v.created_ = 0;
      }
      KOKKOS_INLINE_FUNCTION void join(value_type& a, value_type const& b) const {
        a.live_ += b.live_;
        a.created_ += b.created_;
      }
      KOKKOS_INLINE_FUNCTION void operator()(int i, value_type& sum, bool final) const {
        if (final) {
          offsets_(i) = sum.live_;
          childOffsets_(i) = sum.created_;
        }
        auto const& r = outcomes_(i);
        if (r.action_ == AirAction::replaced)
          for (int j = 0; j < r.history_.childCount_; ++j) {
            ++sum.created_;
            sum.live_ += (decisions_(i).keepMask_ >> j) & 1U;
          }
        else
          sum.live_ += r.action_ == AirAction::active;
        sum.created_ += host_(i).created_;
        for (int j = 0; j < host_(i).count_; ++j)
          sum.live_ += host_(i).vertices_[j].kind_ == HostVertexKind::native_em;
      }
    };
    struct Emit {
      records_type after_, next_;
      outcomes_type outcomes_;
      host_transfers_type host_;
      offsets_type offsets_, childOffsets_;
      AirEnvironment env_;
      std::uint64_t first_;
      thin_results_type decisions_;
      KOKKOS_INLINE_FUNCTION void operator()(int i, int& errors) const {
        auto const& r = outcomes_(i);
        auto const& parent = after_(i);
        int offset = offsets_(i);
        if (!assign_host_history_ids(host_(i), parent.metadata_,
                                     first_ + childOffsets_(i))) {
          ++errors;
          return;
        }
        if (r.action_ == AirAction::active)
          next_(offset++) = parent;
        else if (r.action_ == AirAction::replaced)
          for (int j = 0; j < r.history_.childCount_; ++j) {
            if (!((decisions_(i).keepMask_ >> j) & 1U)) continue;
            AirRecord child = parent;
            child.track_ = r.children_[j];
            auto& p = child.metadata_;
            if (p.generation_ == ~std::uint32_t{0} ||
                !export_air_particle(child.track_, env_, p)) {
              ++errors;
              continue;
            }
            child.exportedPosition_ = true;
            p.parentHistoryId_ = parent.metadata_.historyId_;
            p.historyId_ = first_ + childOffsets_(i) + j;
            p.stepId_ = 0;
            ++p.generation_;
            p.weight_ = j ? decisions_(i).secondWeight_ : decisions_(i).firstWeight_;
            child.random_.key_ = {parent.random_.key_.seed_,
                                  parent.random_.key_.showerId_,
                                  p.historyId_,
                                  0,
                                  NativeEgs4RandomDomain,
                                  0};
            child.hadronRandom_.key_ = {parent.hadronRandom_.key_.seed_,
                                        parent.hadronRandom_.key_.showerId_,
                                        p.historyId_,
                                        0,
                                        NativeEgs4HadronRandomDomain,
                                        0};
            next_(offset++) = child;
          }
        for (int j = 0; j < host_(i).count_; ++j) {
          auto const& v = host_(i).vertices_[j];
          if (v.kind_ != HostVertexKind::native_em) continue;
          // Preserve the C7 local/observer geometry of the vertex, not a fresh
          // Cartesian reimport. New daughters never inherit the nuclear clock.
          AirRecord child = parent;
          child.metadata_ = v.particle_;
          auto& p = child.track_.particle_;
          p.pdg_ = v.nativeSecondary_.pdg_;
          p.energyMeV_ = v.nativeSecondary_.energyMeV_;
          p.direction_ = v.nativeSecondary_.direction_;
          p.clock_ = {};
          p.photonClock_ = {};
          p.pendingChannel_ = Channel::invalid;
          child.random_ = v.random_;
          child.hadronRandom_ = v.hadronRandom_;
          if (p.pdg_ != 22 || !valid_track(p)) {
            ++errors;
            continue;
          }
          next_(offset++) = child;
        }
      }
    };

  private:
    DeviceModel<TExec> model_;
    AirEnvironment env_;
    TransportSettings settings_;
    int capacity_;
    bool resolveRare_;
    RareMaterial rare_;
    PhotonuclearPolicy nuclear_;
    thinning_config_type thinning_;
    bool reuseWorkspace_{};
    bool splitRareKernels_{};
    records_type activeStorage_, afterStorage_, nextStorage_;
    outcomes_type outcomesStorage_;
    outputs_type outputsStorage_;
    host_transfers_type hostStorage_;
    thin_results_type thinningStorage_;
    offsets_type offsetsStorage_, childrenStorage_;
    offsets_type rareIndicesStorage_, rareCursorStorage_;
    std::uint64_t workspaceAllocations_{}, workspaceReuses_{};
    std::uint64_t processedRecords_{}, peakActive_{};
    double advanceWallMs_{};
    std::uint64_t appendCalls_{}, appendRecords_{}, appendCopiedActiveRecords_{};
    double appendWallMs_{};
    std::uint64_t splitRareWaves_{}, splitRareRecords_{};
    template <class TView>
    TView workspace(TView& storage, char const* name, int n) {
      if (!reuseWorkspace_) {
        ++workspaceAllocations_;
        return TView(std::string(name), n);
      }
      if (storage.extent(0) < std::size_t(n)) {
        std::size_t grown = std::max<std::size_t>(1, storage.extent(0));
        while (grown < std::size_t(n))
          grown = std::min<std::size_t>(capacity_, grown * 2);
        storage = TView(std::string(name), grown);
        ++workspaceAllocations_;
      } else
        ++workspaceReuses_;
      return Kokkos::subview(storage, std::make_pair(std::size_t(0), std::size_t(n)));
    }
    records_type active_;
    std::uint64_t nextId_{};
    std::uint64_t seed_{}, shower_{};
    std::set<std::uint64_t> importedIds_;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> childRanges_;

  public:
    AirWavefront(Tables const& tables, ChannelThresholds thresholds, AirEnvironment env,
                 TransportSettings settings, int capacity, bool resolve_rare = false,
                 RareMaterial rare = {}, PhotonuclearPolicy nuclear = {},
                 thinning_config_type thinning = {}, bool reuse_workspace = false,
                 bool split_rare_kernels = false)
        : model_(tables, thresholds)
        , env_(env)
        , settings_(settings)
        , capacity_(capacity)
        , resolveRare_(resolve_rare)
        , rare_(rare)
        , nuclear_(nuclear)
        , thinning_(thinning)
        , reuseWorkspace_(reuse_workspace)
        , splitRareKernels_(split_rare_kernels) {
      validate_thinning(thinning);
      if (capacity < 1 || capacity > std::numeric_limits<int>::max() / 5 ||
          !valid_settings(model_.view(), settings))
        throw std::invalid_argument("Invalid native C8 EGS4 air queue settings");
      if (resolve_rare &&
          (!valid_rare_material(rare) || rare.electronMassMeV_ != thresholds.massMeV_))
        throw std::invalid_argument("Explicit C7 rare-channel material/masses required");
      if (!valid_photonuclear_policy(nuclear, rare) ||
          (nuclear.enabled_ && !resolve_rare))
        throw std::invalid_argument(
            "Native photonuclear vertices require explicit masses/tables and rare-vertex "
            "resolution");
    }
    void beginShower(std::vector<particle_state_type> const& input, std::uint64_t seed,
                     std::uint64_t shower, std::uint64_t first_child) {
      if (input.size() > std::size_t(capacity_))
        throw std::length_error("EGS4 input exceeds capacity");
      records_type candidate("egs4_air_input", input.size());
      auto host = Kokkos::create_mirror_view(candidate);
      std::set<std::uint64_t> ids;
      for (std::size_t i = 0; i < input.size(); ++i)
        if (input[i].historyId_ >= first_child ||
            !ids.insert(input[i].historyId_).second ||
            !import_air_record(input[i], env_, seed, shower, host(i)))
          throw std::invalid_argument("Invalid EGS4 C8 particle/ID/medium");
      Kokkos::deep_copy(candidate, host);
      active_ = candidate;
      activeStorage_ = candidate;
      nextId_ = first_child;
      seed_ = seed;
      shower_ = shower;
      importedIds_ = std::move(ids);
      childRanges_.clear();
    }
    // A completed event releases history/RNG identity but retains buffer capacity.
    void resetForShower(std::uint64_t seed, std::uint64_t shower) {
      if (activeCount()) throw std::logic_error("Cannot reset a nonempty EGS4 queue");
      seed_ = seed;
      shower_ = shower;
      nextId_ = 1;
      importedIds_.clear();
      childRanges_.clear();
      active_ =
          Kokkos::subview(activeStorage_, std::make_pair(std::size_t(0), std::size_t(0)));
      workspaceAllocations_ = workspaceReuses_ = processedRecords_ = peakActive_ = 0;
      appendCalls_ = appendRecords_ = appendCopiedActiveRecords_ = 0;
      splitRareWaves_ = splitRareRecords_ = 0;
      advanceWallMs_ = appendWallMs_ = 0.;
    }
    // New C8 stack particles only. Existing native continuations never pass
    // through this import: their optical clocks and random streams stay resident.
    void append(std::vector<particle_state_type> const& input) {
      if (input.empty()) return;
      auto const started = std::chrono::steady_clock::now();
      if (input.size() > std::size_t(capacity_) - activeCount())
        throw std::length_error("Native air append exceeds capacity");
      // The reference keeps its owning copy; the resident path checks only the
      // incoming batch against the accumulated set. Nodes are merged only after
      // every import and device copy succeeds, preserving failed-append behavior.
      auto ids = reuseWorkspace_ ? std::set<std::uint64_t>{} : importedIds_;
      auto next_id = nextId_;
      records_type added("egs4_air_added", input.size());
      auto host = Kokkos::create_mirror_view(added);
      for (std::size_t i = 0; i < input.size(); ++i) {
        auto id = input[i].historyId_;
        auto range =
            std::upper_bound(childRanges_.begin(), childRanges_.end(), id,
                             [](auto key, auto const& r) { return key < r.first; });
        bool native_id = range != childRanges_.begin() && id < std::prev(range)->second;
        if (id == 0 || id == ~std::uint64_t{0} || native_id ||
            (reuseWorkspace_ && importedIds_.count(id)) || !ids.insert(id).second ||
            !import_air_record(input[i], env_, seed_, shower_, host(i)))
          throw std::invalid_argument(
              "Invalid/duplicate native air injection; continuation must not be "
              "reimported");
        next_id = std::max(next_id, id + 1);
      }
      Kokkos::deep_copy(added, host);
      auto old = activeCount();
      records_type joined =
          reuseWorkspace_
              ? workspace(activeStorage_, "egs4_air_joined", int(old + input.size()))
              : records_type("egs4_air_joined", old + input.size());
      if (old && joined.data() != active_.data()) {
        Kokkos::deep_copy(Kokkos::subview(joined, std::make_pair(std::size_t(0), old)),
                          active_);
        appendCopiedActiveRecords_ += old;
      }
      Kokkos::deep_copy(Kokkos::subview(joined, std::make_pair(old, old + input.size())),
                        added);
      active_ = joined;
      if (reuseWorkspace_)
        importedIds_.merge(ids);
      else {
        activeStorage_ = joined;
        importedIds_ = std::move(ids);
      }
      nextId_ = next_id;
      ++appendCalls_;
      appendRecords_ += input.size();
      appendWallMs_ += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count();
    }
    std::size_t activeCount() const { return active_.extent(0); }
    std::uint64_t nextHistoryId() const { return nextId_; }
    std::uint64_t workspaceAllocations() const { return workspaceAllocations_; }
    std::uint64_t workspaceReuses() const { return workspaceReuses_; }
    std::uint64_t processedRecords() const { return processedRecords_; }
    std::uint64_t peakActive() const { return peakActive_; }
    double advanceWallMilliseconds() const { return advanceWallMs_; }
    std::uint64_t appendCalls() const { return appendCalls_; }
    std::uint64_t appendRecords() const { return appendRecords_; }
    std::uint64_t appendCopiedActiveRecords() const { return appendCopiedActiveRecords_; }
    double appendWallMilliseconds() const { return appendWallMs_; }
    std::uint64_t splitRareWaves() const { return splitRareWaves_; }
    std::uint64_t splitRareRecords() const { return splitRareRecords_; }
    records_type activeRecords() const { return active_; }
    Batch advance() { return advance(nextId_, ~std::uint64_t{0} - nextId_); }
    // In HybridCascade the CPU stack is the sole allocator. It reserves up to
    // five histories with omega/phi decay, four with prompt rho (including the transient
    // parent), three with native photonuclear generation, or two otherwise. Unused IDs
    // are harmless gaps, never reused or assigned concurrently by the GPU.
    Batch advance(std::uint64_t first_child, std::uint64_t reserved) {
      auto const start = std::chrono::steady_clock::now();
      if (first_child < nextId_ || reserved > ~std::uint64_t{0} - first_child)
        throw std::invalid_argument(
            "Overlapping/overflowing native child ID reservation");
      // With reuse enabled, Batch is a borrowed view valid until the next
      // advance/beginShower. The session consumes it completely before advancing.
      // Default standalone API retains its original owning/snapshot semantics.
      int n = int(activeCount());
      Batch b;
      b.before_ = active_;
      b.after_ = workspace(afterStorage_, "egs4_air_after", n);
      b.outcomes_ = workspace(outcomesStorage_, "egs4_air_outcomes", n);
      b.outputs_ = workspace(outputsStorage_, "egs4_air_outputs", n);
      b.host_ = workspace(hostStorage_, "egs4_air_host", n);
      b.thinning_ = workspace(thinningStorage_, "egs4_air_thinning", n);
      if (!n) return b;
      bool const split = splitRareKernels_ && resolveRare_;
      int errors = 0, rare_records = 0;
      if (split) {
        auto indices = workspace(rareIndicesStorage_, "egs4_air_rare_indices", n);
        auto cursor = workspace(rareCursorStorage_, "egs4_air_rare_cursor", 1);
        // Stream-ordered reset; the reduction below supplies the completion fence.
        Kokkos::deep_copy(TExec(), cursor, 0);
        TransportCounts counts;
        Kokkos::parallel_reduce("egs4_air_transport_only",
                                Kokkos::RangePolicy<TExec>(0, n),
                                TransportOnly{model_.view(), env_, settings_, b.before_,
                                              b.after_, b.outcomes_, b.outputs_, b.host_,
                                              thinning_, b.thinning_, indices, cursor},
                                counts);
        errors = counts.errors_;
        rare_records = counts.rare_;
        if (!errors && rare_records)
          Kokkos::parallel_reduce(
              "egs4_air_compacted_rare", Kokkos::RangePolicy<TExec>(0, rare_records),
              ResolveCompactedRare{model_.view(), env_, settings_, b.after_, b.outcomes_,
                                   b.host_, rare_, nuclear_, thinning_, b.thinning_,
                                   indices},
              errors);
      } else {
        if (!reuseWorkspace_) Kokkos::deep_copy(b.after_, b.before_);
        Kokkos::parallel_reduce(
            "egs4_air_advance", Kokkos::RangePolicy<TExec>(0, n),
            Advance{model_.view(), env_, settings_, b.before_, b.after_, b.outcomes_,
                    b.outputs_, b.host_, resolveRare_, rare_, nuclear_, thinning_,
                    b.thinning_, reuseWorkspace_},
            errors);
      }
      if (errors) {
        auto states = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.before_);
        auto outcomes =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.outcomes_);
        auto output =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.outputs_);
        auto transfers =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.host_);
        auto thin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.thinning_);
        std::ostringstream message;
        message << std::setprecision(17)
                << "Native air history failed; live queue/RNG not committed";
        for (int i = 0; i < n; ++i)
          if (outcomes(i).action_ == AirAction::error || !output(i).valid_ ||
              thin(i).status_ == thinning_status_type::InvalidInput ||
              (outcomes(i).action_ == AirAction::needs_host_channel &&
               !transfers(i).valid_)) {
            auto const& s = states(i);
            auto const& p = s.track_.particle_;
            message << "; history=" << s.metadata_.historyId_
                    << " step=" << s.metadata_.stepId_ << " pdg=" << p.pdg_
                    << " E_MeV=" << p.energyMeV_ << " region=" << s.region_
                    << " position_cm=" << p.positionCm_.x_ << "," << p.positionCm_.y_
                    << "," << p.positionCm_.z_ << " direction=" << p.direction_.x_ << ","
                    << p.direction_.y_ << "," << p.direction_.z_
                    << " action=" << int(outcomes(i).action_)
                    << " history_error=" << int(outcomes(i).history_.error_)
                    << " channel=" << int(outcomes(i).history_.channel_)
                    << " output_valid=" << output(i).valid_
                    << " random_draw=" << s.random_.key_.drawId_;
            break;
          }
        throw std::runtime_error(message.str());
      }
      auto offsets = workspace(offsetsStorage_, "egs4_air_offsets", n);
      auto children = workspace(childrenStorage_, "egs4_air_children", n);
      int count = 0;
      if (reuseWorkspace_) {
        PrefixCounts totals;
        Kokkos::parallel_scan(
            "egs4_air_combined_count", Kokkos::RangePolicy<TExec>(0, n),
            CombinedOffset{b.outcomes_, b.host_, offsets, children, b.thinning_}, totals);
        count = totals.live_;
        b.created_ = totals.created_;
      } else {
        Kokkos::parallel_scan("egs4_air_count", Kokkos::RangePolicy<TExec>(0, n),
                              Offset{b.outcomes_, b.host_, offsets, false, b.thinning_},
                              count);
        Kokkos::parallel_scan("egs4_air_child_count", Kokkos::RangePolicy<TExec>(0, n),
                              Offset{b.outcomes_, b.host_, children, true, b.thinning_},
                              b.created_);
      }
      if (count > capacity_)
        throw std::length_error(
            "Native air queue overflow; no children discarded; active=" +
            std::to_string(n) + " required=" + std::to_string(count) +
            " capacity=" + std::to_string(capacity_));
      if (std::uint64_t(b.created_) > reserved)
        throw std::length_error("Native air history ID reservation exhausted");
      auto next = workspace(nextStorage_, "egs4_air_next", count);
      errors = 0;
      Kokkos::parallel_reduce("egs4_air_emit", Kokkos::RangePolicy<TExec>(0, n),
                              Emit{b.after_, next, b.outcomes_, b.host_, offsets,
                                   children, env_, first_child, b.thinning_},
                              errors);
      if (errors)
        throw std::runtime_error("Native air child export failed; queue not committed");
      TExec().fence();
      active_ = next;
      if (reuseWorkspace_)
        std::swap(activeStorage_, nextStorage_);
      else
        activeStorage_ = next;
      nextId_ = first_child + b.created_;
      if (b.created_) childRanges_.emplace_back(first_child, nextId_);
      processedRecords_ += n;
      peakActive_ = std::max(peakActive_, std::uint64_t(n));
      splitRareWaves_ += split;
      splitRareRecords_ += rare_records;
      advanceWallMs_ += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      return b;
    }
  };
} // namespace corsika::kokkos::egs4::c8_adapter
