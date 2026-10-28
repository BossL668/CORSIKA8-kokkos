/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <cstdint>

namespace counter_fixture {
  struct SharedView {};
  struct SharedModel {
    int calls_ = 0;
    void doInteraction(SharedView&, corsika::Code, corsika::Code,
                       corsika::FourMomentum const&, corsika::FourMomentum const&) {
      ++calls_;
    }
  };
  struct CounterSnapshot {
    std::uint64_t count_;
    std::uint64_t deferred_;
    int histogram_;
  };
  CounterSnapshot native_counter_other_tu(SharedModel&, corsika::FourMomentum const&,
                                          corsika::FourMomentum const&);
  CounterSnapshot deferred_counter_other_tu(SharedModel&, corsika::FourMomentum const&,
                                            corsika::FourMomentum const&);
} // namespace counter_fixture
