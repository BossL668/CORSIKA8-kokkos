/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/radio/RadioTrackSegment.hpp>

namespace corsika::radio {
  /** Destination for deferred projection of an electron/positron step. */
  class RadioTrackSink {
  public:
    virtual ~RadioTrackSink() = default;
    virtual void append(RadioTrackSegment const& segment) = 0;
  };
} // namespace corsika::radio
