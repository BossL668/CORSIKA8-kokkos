/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/radio/RadioTrackSegment.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>
#include <corsika/kokkos/em/ParticleId.hpp>
#include <corsika/kokkos/data/ParticleState.hpp>
#include <stdexcept>

namespace corsika::kokkos::radio {
  /** Convert a completed native step to the radio projector input, not a new transport.
   */
  inline ::corsika::kokkos::LeptonTransportRecord make_host_radio_record(
      corsika::radio::RadioTrackSegment const& segment);
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/HostRadioRecord.inl>
