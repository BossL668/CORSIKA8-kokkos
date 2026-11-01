/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4C8AirWavefront.inl>
#include <corsika/kokkos/egs4/Session.hpp>

namespace corsika::kokkos::egs4::application {
  // Only exceptional/CPU-consumed records cross the device boundary. Scan in
  // input order to keep host callbacks, RNG/ID ownership and generator order.
  struct HostOutputRecord {
    c8_adapter::AirOutput output_;
    c8_adapter::RareAirTransfer transfer_;
    c8_adapter::AirAction action_{};
    double weight_{};
  };
  struct WaveCounts {
    std::uint64_t steps_{}, radio_{}, observations_{}, discards_{}, hillas_{},
        statistical_{}, removed_{}, unresolved_{}, needed_{};
  };
  template <class TBatch>
  struct CountWave {
    using value_type = WaveCounts;
    TBatch batch_;
    bool sparse_{}, profileDevice_{}, radioDevice_{};
    KOKKOS_INLINE_FUNCTION void init(value_type& c) const { c = {}; }
    KOKKOS_INLINE_FUNCTION void join(value_type& a, value_type const& b) const {
      a.steps_ += b.steps_;
      a.radio_ += b.radio_;
      a.observations_ += b.observations_;
      a.discards_ += b.discards_;
      a.hillas_ += b.hillas_;
      a.statistical_ += b.statistical_;
      a.removed_ += b.removed_;
      a.unresolved_ += b.unresolved_;
      a.needed_ += b.needed_;
    }
    KOKKOS_INLINE_FUNCTION void operator()(int i, value_type& c) const {
      auto const& r = batch_.outputs_(i);
      auto const& t = batch_.thinning_(i);
      c.steps_ += r.hasStep_;
      c.radio_ += r.hasRadio_;
      c.observations_ += r.hasObservation_;
      c.discards_ += r.discard_ != c8_adapter::AirDiscard::none;
      c.hillas_ += t.status_ == c8_adapter::thinning_status_type::Hillas;
      c.statistical_ += t.status_ == c8_adapter::thinning_status_type::Statistical;
      if (t.status_ != c8_adapter::thinning_status_type::NotApplied)
        c.removed_ += 2 - ((t.keepMask_ & 1U) != 0) - ((t.keepMask_ & 2U) != 0);
      c.unresolved_ +=
          batch_.outcomes_(i).action_ == c8_adapter::AirAction::needs_host_channel;
      if (sparse_) {
        auto const& h = batch_.host_(i);
        c.needed_ += (!profileDevice_ && r.hasStep_) || (!radioDevice_ && r.hasRadio_) ||
                     r.hasObservation_ || r.discard_ != c8_adapter::AirDiscard::none ||
                     h.count_ || h.targetRestEnergyMeV_ != 0. || h.hasRhoDecay_ ||
                     h.hasResonanceDecay_;
      }
    }
  };
  template <class TBatch, class TRecords>
  struct SelectHostOutput {
    TBatch batch_;
    TRecords records_;
    bool profileDevice_, radioDevice_;
    KOKKOS_INLINE_FUNCTION void operator()(int i, int& count, bool final) const {
      auto const& r = batch_.outputs_(i);
      auto const& h = batch_.host_(i);
      bool needed = (!profileDevice_ && r.hasStep_) || (!radioDevice_ && r.hasRadio_) ||
                    r.hasObservation_ || r.discard_ != c8_adapter::AirDiscard::none ||
                    h.count_ || h.targetRestEnergyMeV_ != 0. || h.hasRhoDecay_ ||
                    h.hasResonanceDecay_;
      if (!needed) return;
      if (final)
        records_(count) = {r, h, batch_.outcomes_(i).action_,
                           batch_.before_(i).metadata_.weight_};
      ++count;
    }
  };
  template <class TExec>
  class HostOutput {
    using records_type = Kokkos::View<HostOutputRecord*, typename TExec::memory_space>;
    records_type records_;
    bool sparse_{(overhead::options() & overhead::sparse_host) != 0};

  public:
    template <class TBatch>
    auto collect(TBatch const& batch, bool profile_device, bool radio_device,
                 RunStatistics& stats) {
      int const n = int(batch.outputs_.extent(0));
      WaveCounts counts;
      Kokkos::parallel_reduce(
          "egs4_output_counts", Kokkos::RangePolicy<TExec>(0, n),
          CountWave<TBatch>{batch, sparse_, profile_device, radio_device}, counts);
      if (counts.unresolved_)
        throw std::runtime_error(
            "Native EGS4 unresolved rare channel; enable native rare vertices");
      stats.steps_ += counts.steps_;
      stats.radioTracks_ += counts.radio_;
      stats.observations_ += counts.observations_;
      stats.discards_ += counts.discards_;
      stats.thinningHillas_ += counts.hillas_;
      stats.thinningStatistical_ += counts.statistical_;
      stats.thinningRemoved_ += counts.removed_;
      int count = 0;
      if (!sparse_ || counts.needed_) {
        if (records_.extent(0) < std::size_t(n))
          records_ = records_type("egs4_compact_host_output", n);
        Kokkos::parallel_scan("egs4_compact_host_output",
                              Kokkos::RangePolicy<TExec>(0, n),
                              SelectHostOutput<TBatch, records_type>{
                                  batch, records_, profile_device, radio_device},
                              count);
      }
      stats.hostOutputRecords_ += count;
      stats.hostOutputBytes_ +=
          count * sizeof(HostOutputRecord) + sizeof(WaveCounts) + sizeof(int);
      return Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{},
          Kokkos::subview(records_, std::make_pair(std::size_t(0), std::size_t(count))));
    }
  };
} // namespace corsika::kokkos::egs4::application
