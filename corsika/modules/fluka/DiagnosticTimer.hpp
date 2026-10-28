/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::fluka {
  /** Disable the legacy FLUKA diagnostic alarm immediately after initialization.
   * The generator, random bridge and floating-point environment are unchanged.
   */
  void disable_diagnostic_timer_after_initialization();
} // namespace corsika::fluka

#include <corsika/detail/modules/fluka/DiagnosticTimer.inl>
