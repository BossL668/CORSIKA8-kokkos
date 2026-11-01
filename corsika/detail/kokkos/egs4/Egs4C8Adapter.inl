/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4History.inl>
#include <corsika/kokkos/egs4/Data.hpp>
#include <corsika/kokkos/em/random/Philox.hpp>

namespace corsika::kokkos::egs4::c8_adapter {
  using particle_state_type = ::corsika::kokkos::ParticleState;
  // Axes of the local EGS frame expressed in the caller's C8 frame. They must
  // form a right-handed orthonormal basis. This conversion does NOT implement
  // the C7 curved-atmosphere frame rebasing or its time correction.
  struct Frame {
    Point originM_;
    Direction x_{1., 0., 0.}, y_{0., 1., 0.}, z_{0., 0., 1.};
  };
  KOKKOS_INLINE_FUNCTION double dot(Direction a, Direction b) {
    return a.x_ * b.x_ + a.y_ * b.y_ + a.z_ * b.z_;
  }
  KOKKOS_INLINE_FUNCTION Direction cross(Direction a, Direction b) {
    return {a.y_ * b.z_ - a.z_ * b.y_, a.z_ * b.x_ - a.x_ * b.z_,
            a.x_ * b.y_ - a.y_ * b.x_};
  }
  KOKKOS_INLINE_FUNCTION bool valid_frame(Frame const& f) {
    return finite(f.originM_.x_) && finite(f.originM_.y_) && finite(f.originM_.z_) &&
           ::fabs(dot(f.x_, f.x_) - 1.) < 1.e-12 &&
           ::fabs(dot(f.y_, f.y_) - 1.) < 1.e-12 &&
           ::fabs(dot(f.z_, f.z_) - 1.) < 1.e-12 && ::fabs(dot(f.x_, f.y_)) < 1.e-12 &&
           ::fabs(dot(f.x_, f.z_)) < 1.e-12 && ::fabs(dot(f.y_, f.z_)) < 1.e-12 &&
           ::fabs(dot(cross(f.x_, f.y_), f.z_) - 1.) < 1.e-12;
  }
  KOKKOS_INLINE_FUNCTION Direction to_local(Frame const& f, Direction d) {
    return {dot(d, f.x_), dot(d, f.y_), dot(d, f.z_)};
  }
  KOKKOS_INLINE_FUNCTION Direction to_global(Frame const& f, Direction d) {
    return {f.x_.x_ * d.x_ + f.y_.x_ * d.y_ + f.z_.x_ * d.z_,
            f.x_.y_ * d.x_ + f.y_.y_ * d.y_ + f.z_.y_ * d.z_,
            f.x_.z_ * d.x_ + f.y_.z_ * d.y_ + f.z_.z_ * d.z_};
  }
  // C8 carrier energy_GeV is TOTAL energy. Never subtract/rest-add a mass here.
  // Optical clocks are private native history state and cannot be reconstructed
  // from EmParticleState: keep the NativeRecord across every batch continuation.
  KOKKOS_INLINE_FUNCTION bool import_particle(particle_state_type const& p, Frame f,
                                              TrackState& result) {
    if (!valid_frame(f) || !finite(p.weight_) || p.weight_ < 0.) return false;
    Direction displacement{p.positionM_[0] - f.originM_.x_,
                           p.positionM_[1] - f.originM_.y_,
                           p.positionM_[2] - f.originM_.z_};
    auto pos = to_local(f, displacement);
    TrackState s;
    s.pdg_ = p.pid_;
    s.energyMeV_ = p.energyGeV_ * 1000.;
    s.timeS_ = p.timeS_;
    s.positionCm_ = {pos.x_ * 100., pos.y_ * 100., pos.z_ * 100.};
    s.direction_ = to_local(f, {p.direction_[0], p.direction_[1], p.direction_[2]});
    if (!valid_track(s)) return false;
    result = s;
    return true;
  }
  KOKKOS_INLINE_FUNCTION bool export_particle(TrackState state, Frame f,
                                              particle_state_type& result) {
    if (!valid_frame(f) || !valid_track(state)) return false;
    auto position = to_global(f, {state.positionCm_.x_ * .01, state.positionCm_.y_ * .01,
                                  state.positionCm_.z_ * .01});
    auto direction = to_global(f, state.direction_);
    result.pid_ = state.pdg_;
    result.energyGeV_ = state.energyMeV_ * .001;
    result.timeS_ = state.timeS_;
    result.positionM_[0] = position.x_ + f.originM_.x_;
    result.positionM_[1] = position.y_ + f.originM_.y_;
    result.positionM_[2] = position.z_ + f.originM_.z_;
    result.direction_[0] = direction.x_;
    result.direction_[1] = direction.y_;
    result.direction_[2] = direction.z_;
    return true; // metadata, weight and history IDs remain caller-owned
  }
  struct CounterStream {
    ::corsika::kokkos::RandomNumberKey key_;
    KOKKOS_INLINE_FUNCTION bool next(double& u) {
      if (key_.drawId_ == ~std::uint64_t{0}) return false;
      u = ::corsika::kokkos::uniform_open01(key_);
      ++key_.drawId_;
      return true;
    }
  };
  // Scope-local preparation: no larger resident record, no changed lanes/draw IDs.
  // The caller must not change seed/history/step/process during this one history step.
  struct PreparedCounterStream {
    CounterStream& stream_;
    bool enabled_;
    ::corsika::kokkos::PhiloxCounter counter_;
    std::uint64_t prefix_{};
    KOKKOS_INLINE_FUNCTION PreparedCounterStream(CounterStream& s, bool on)
        : stream_(s)
        , enabled_(on) {
      if (!on) return;
      auto const& k = s.key_;
      using ::corsika::kokkos::philox_detail::splitmix64;
      prefix_ = splitmix64(splitmix64(splitmix64(k.seed_) ^ k.showerId_) ^
                           std::uint64_t(k.processId_));
      counter_ = {{std::uint32_t(k.historyId_), std::uint32_t(k.historyId_ >> 32),
                   std::uint32_t(k.stepId_), std::uint32_t(k.stepId_ >> 32)}};
    }
    KOKKOS_INLINE_FUNCTION bool next(double& u) {
      if (!enabled_) return stream_.next(u);
      if (stream_.key_.drawId_ == ~std::uint64_t{0}) return false;
      auto m =
          ::corsika::kokkos::philox_detail::splitmix64(prefix_ ^ stream_.key_.drawId_);
      ::corsika::kokkos::PhiloxKey k{{std::uint32_t(m), std::uint32_t(m >> 32)}};
      auto words = ::corsika::kokkos::philox4x32_10(counter_, k);
      u = (double(words.words_[0]) + .5) * (1. / 4294967296.);
      ++stream_.key_.drawId_;
      return true;
    }
  };
  inline constexpr std::uint32_t NativeEgs4RandomDomain = 0x43474534U;
  // Per-history counterpart of C7 stream 1 (PTRANS/meson decays), separate
  // from electromagnetic stream 2. Never a process-global mutable RNG.
  inline constexpr std::uint32_t NativeEgs4HadronRandomDomain = 0x43474831U;
  struct NativeRecord {
    particle_state_type metadata_;
    TrackState state_;
    CounterStream random_;
  };
  KOKKOS_INLINE_FUNCTION bool import_record(particle_state_type const& p, Frame frame,
                                            std::uint64_t seed, std::uint64_t shower_id,
                                            NativeRecord& result) {
    NativeRecord candidate;
    if (!import_particle(p, frame, candidate.state_)) return false;
    candidate.metadata_ = p;
    candidate.random_.key_ = {
        seed, shower_id, p.historyId_, p.stepId_, NativeEgs4RandomDomain, 0};
    result = candidate;
    return true;
  }
  // Host-assigned, non-overlapping IDs are supplied explicitly, rather than an
  // atomic allocation whose result (and RNG stream) depends on GPU scheduling.
  KOKKOS_INLINE_FUNCTION bool child_record(NativeRecord const& parent, TrackState child,
                                           Frame frame, std::uint64_t id,
                                           NativeRecord& result) {
    if (id == parent.metadata_.historyId_ ||
        parent.metadata_.generation_ == ~std::uint32_t{0})
      return false;
    auto p = parent.metadata_;
    if (!export_particle(child, frame, p)) return false;
    p.parentHistoryId_ = parent.metadata_.historyId_;
    p.historyId_ = id;
    p.generation_ += 1;
    p.stepId_ = 0;
    NativeRecord candidate;
    if (!import_record(p, frame, parent.random_.key_.seed_, parent.random_.key_.showerId_,
                       candidate))
      return false;
    candidate.state_ = child;
    candidate.state_.clock_ = {};
    candidate.state_.photonClock_ = {};
    candidate.state_.pendingChannel_ = Channel::invalid;
    result = candidate;
    return true;
  }
  // Adapter-level one-step integration. The geometry object is a device-callable
  // functor BoundaryLimit(TrackState const&, double proposed_cm). It must use
  // the same local frame as region/field; there is no hidden geometry fallback.
  template <class TGeometry>
  KOKKOS_INLINE_FUNCTION HistoryOutcome
  advance_record(ModelView model, NativeRecord& record, Frame frame, AirRegion region,
                 MagneticField field, TransportSettings settings, TGeometry geometry) {
    HistoryOutcome invalid;
    invalid.particle_ = record.state_;
    if (!valid_frame(frame) || record.metadata_.stepId_ == ~std::uint64_t{0})
      return invalid;
    auto p = plan_history(model, record.state_, region, field, settings, record.random_);
    BoundaryLimit limit;
    if (p.action_ == PlanAction::transport) limit = geometry(p.particle_, p.proposedCm_);
    auto r = finish_history(model, p, region, field, settings, limit, record.random_);
    if (r.action_ == HistoryAction::error)
      return r; // RNG may be consumed: fatal, never retry with another model
    record.state_ = r.particle_;
    if (!export_particle(record.state_, frame, record.metadata_)) {
      r.action_ = HistoryAction::error;
      r.error_ = HistoryError::invalid_input;
      return r;
    }
    if (r.hasTrack_) ++record.metadata_.stepId_;
    return r;
  }
  static_assert(std::is_trivially_copyable_v<NativeRecord>);
} // namespace corsika::kokkos::egs4::c8_adapter
