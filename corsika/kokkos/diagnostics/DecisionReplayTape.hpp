/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/diagnostics/DecisionReplayTypes.hpp>
#include <corsika/output/DecisionReplayTape.hpp>

namespace corsika::kokkos::diagnostics {
  using ::corsika::diagnostics::DecisionReplayTape;
  using ::corsika::diagnostics::make_replay_radio_snapshot;
  namespace replay_tape_detail = ::corsika::diagnostics::replay_tape_detail;
} // namespace corsika::kokkos::diagnostics
