/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/output/DecisionReplayTypes.hpp>

namespace corsika::kokkos::diagnostics {
  using ::corsika::diagnostics::ReplayEndianMarker;
  using ::corsika::diagnostics::ReplayPropagationSnapshot;
  using ::corsika::diagnostics::ReplayRadioObserver;
  using ::corsika::diagnostics::ReplayRadioSnapshot;
  using ::corsika::diagnostics::ReplayTapeData;
  using ::corsika::diagnostics::ReplayTapeHeader;
  using ::corsika::diagnostics::ReplayTapeMagic;
  using ::corsika::diagnostics::ReplayTapeVersion;
  using ::corsika::diagnostics::ReplayTransportLimit;
  using ::corsika::diagnostics::ReplayTransportStep;
} // namespace corsika::kokkos::diagnostics
