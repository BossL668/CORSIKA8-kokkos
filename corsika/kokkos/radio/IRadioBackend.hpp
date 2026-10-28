/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/radio/RadioWaveforms.hpp>
#include <cstddef>
#include <boost/core/span.hpp>

namespace corsika::kokkos::radio {
  /** Host dispatch interface for a configured radio calculation backend.
   * A backend owns its propagation model, geometry and observer data; this
   * interface does not require a height table or a particular material.
   * Records use the backend's configured common coordinate system and units.
   * enqueueRadioTracks must copy records before returning. downloadRadioWaveforms
   * completes pending work, including a partial final batch, before returning.
   * CoREAS field and ZHS potential samples retain RadioWaveforms' conventions.
   *
   * Calls execute on the host. Implementations may dispatch Kokkos kernels with
   * their own device-compatible propagation data, including complex geometries.
   * This is an extension contract, not an implementation of firn ray tracing.
   * Each instance is used by one shower owner; concurrent calls are not implied.
   */
  class IRadioBackend {
  public:
    virtual ~IRadioBackend() = default;
    /** Whether this backend accepts completed tracks for radio accumulation. */
    virtual bool gpuRadioEnabled() const noexcept = 0;
    /** Copy a borrowed range of completed tracks before returning.
     * Empty ranges are allowed. The backend must not retain the input view.
     * Boost span supplies the non-owning view on the C++17 release baseline.
     */
    virtual void enqueueRadioTracks(boost::span<LeptonTransportRecord const>) = 0;
    /** Finish pending batches and return the accumulated shower waveforms. */
    virtual RadioWaveforms downloadRadioWaveforms() = 0;
  };
} // namespace corsika::kokkos::radio
