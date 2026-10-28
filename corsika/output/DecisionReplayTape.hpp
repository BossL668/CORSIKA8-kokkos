/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/output/DecisionReplayTypes.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika::diagnostics {

  /**
   * Why this tape exists
   * --------------------
   *
   * A production CUDA shower uses history-keyed Philox streams and a
   * wavefront scheduler.  A scalar CORSIKA shower uses sequential random
   * streams and a LIFO stack.  Equal seeds therefore do not identify equal
   * events across those two algorithms.
   *
   * This file format is the explicit exception: the scalar run records its
   * already selected transport segments, and a CUDA replay consumer is
   * required to use those exact segments.  It is a debug/validation oracle,
   * not an alternative source of production Monte Carlo events.
   */

  /**
   * Build the replay radio snapshot with the same nearest-bin flat-atmosphere
   * discretization used by TabulatedFlatAtmospherePropagator and the CUDA
   * radio backend.
   */
  template <typename TEnvironment, typename TCoreasDetector, typename TZhsDetector>
  ReplayRadioSnapshot make_replay_radio_snapshot(TEnvironment const& environment,
                                                 Point const& upper_limit,
                                                 Point const& lower_limit,
                                                 LengthType const step,
                                                 TCoreasDetector const& coreas_detector,
                                                 TZhsDetector const& zhs_detector);

  class DecisionReplayTape {
  public:
    static DecisionReplayTape& instance();

    void open(std::filesystem::path const& path, ReplayRadioSnapshot const& radio,
              std::uint64_t random_seed, std::uint64_t configured_showers);

    bool enabled() const noexcept;

    void beginShower(std::uint64_t shower) noexcept;

    template <typename TParticle>
    void recordTransportStep(Step<TParticle> const& step,
                             ReplayTransportLimit transport_limit,
                             ProcessReturn process_return);

    void close();

    static ReplayTapeData read(std::filesystem::path const& path);

  private:
    DecisionReplayTape() = default;
    ~DecisionReplayTape();

    std::fstream stream_{};
    ReplayTapeHeader header_{};
    std::uint64_t recordCount_{};
    std::uint64_t shower_{};
    std::uint64_t ordinal_{};
  };

} // namespace corsika::diagnostics

#include <corsika/detail/output/DecisionReplayTape.inl>
