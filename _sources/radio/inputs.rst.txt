Track collection and observer preparation
=========================================

Radio calculation consumes completed particle tracks and prepared observer
data. It does not advance the shower or resample a particle's motion. Tracks
can arrive directly from resident transport, from native C8 steps redirected
to a radio backend, or from a completed prefix-stage track file.

This page describes the standard track and observer interfaces at the source
revision recorded in :doc:`../radio`. The propagation model and numerical
accumulator determine how those inputs contribute to a waveform.

Input records and units
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Type
     - Data contract
   * - ``LeptonTransportRecord``
     - Completed start/end particle states. Positions are metres, times
       seconds, total energies GeV and statistical weights dimensionless.
       The standard radio projector accepts electron/positron records.
   * - ``RadioTrackSegment``
     - Native-step transfer record: start/end positions, directions, times,
       kinetic energies, weight and PDG code. It uses metres, seconds and
       **kinetic** GeV in a common coordinate system.
   * - ``RadioTrackKinematics``
     - Projection-ready endpoints, displacement, duration, dimensionless
       velocity :math:`\boldsymbol{\beta}`, track length and weighted charge
       factor. It contains no native particle or environment pointer.
   * - ``RadioObserverSnapshot``
     - Observer position, waveform start time, duration, sampling rate and
       bin count. Units are metres, seconds and Hz; observer order is retained.

``make_host_radio_record(segment)`` converts the native transfer record to
``LeptonTransportRecord``. It copies geometry, directions, time, weight and
PID, and adds the transport rest mass to each kinetic energy. It is a radio
adapter, not a reconstruction of all transport diagnostics or a new particle
step. It rejects species other than electrons and positrons. See
`HostRadioRecord.inl`_ and `RadioTrackSegment.hpp`_.

Completed-track kinematics
--------------------------

``make_radio_track_kinematics(record, track)`` reads the start/end positions
and times. For a positive duration and nonzero displacement it forms

.. math::

   \boldsymbol{\beta} =
   \frac{\boldsymbol{r}_{\mathrm{end}}-\boldsymbol{r}_{\mathrm{start}}}
        {c\,(t_{\mathrm{end}}-t_{\mathrm{start}})}.

It does not derive this velocity from the endpoint energy or replace the
completed displacement with the direction after multiple scattering.
Energies and stored directions remain available for track diagnostics.
The construction follows the completed straight track used by radio, rather
than separately integrating the two magnetic half-steps. The transport
conventions are described in :doc:`../transport/lepton-step`.

The source coefficient is the signed elementary charge times the radio
constant and ``record.start_.weight_``. Electron and positron signs differ;
the statistical weight must not be applied again when those contributions
are accumulated. Non-electron/positron records, zero-duration records and
zero-length records do not become valid tracks. Muon transport support does
not imply that muon tracks enter this projector. The implementation is in
`RadioProjection.inl`_.

The helper is not a complete input validator: it assumes that completed
transport and its adapters have provided meaningful numerical state. Its
boolean result identifies a usable projection track, not an independent
validation of the particle's earlier history.

Native-step collection
----------------------

The native ``RadioProcess::doContinuous`` normally calls its CoREAS or ZHS
implementation for an electron/positron step. With a track sink attached, it
exports the completed step instead and returns without executing the native
radio calculation. Empty observer collections return before either path.

The sink receives numerical coordinates in the coordinate system supplied
when it was attached. Positions and directions are transformed to that
system; unit-bearing time, length and kinetic-energy values are converted
explicitly. The native export is implemented in `RadioProcess.inl`_.

``ScopedRadioTrackSink(coreas, zhs, sink, coordinates)`` attaches this route
to both radio processes. Only one process captures the track: CoREAS when
its observer collection is nonempty, otherwise ZHS. Both enabled native
projections are bypassed because the common backend computes the requested
radio models from that single captured track. Capturing independently in
both processes would count each physical segment twice.

The scope rejects an already attached sink, restores the first attachment
if the second fails, and detaches both on destruction. The referenced radio
processes, sink and backend must remain alive for the full attachment scope.
`CascadeRunner.inl`_ keeps the collector alive through router finalization and
output writing; `ScopedRadioTrackSink.inl`_ implements the attachment rules.

Host dispatch and input lifetime
--------------------------------

``HostRadioCollector`` implements the native ``RadioTrackSink`` interface.
Its ``append(segment)`` converts one completed record and passes a borrowed
span to ``enqueueRadioTracks``. The general host interface is
``IRadioBackend``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Contract
   * - ``gpuRadioEnabled()``
     - Reports whether this backend accepts completed radio tracks. The
       historical name does not restrict the interface to CUDA or require
       that the selected Kokkos execution space is a GPU.
   * - ``enqueueRadioTracks(span)``
     - Copies the supplied records before returning. The backend must not
       retain the borrowed span; an empty range is allowed.
   * - ``downloadRadioWaveforms()``
     - Completes pending work, including a final partial track batch, and
       returns the accumulated shower waveforms.

These are host-side dispatch calls, not functions invoked by particle
kernels. An implementation can launch Kokkos work behind this interface.
The contract assumes one shower owner per backend instance; concurrent host
calls are not implied. `BorrowedRadioBackend` forwards calls to an existing
EM backend without owning it. See `IRadioBackend.hpp`_ and
`BorrowedRadioBackend.inl`_.

In the standard accumulator, ``enqueueHostLeptonTracks`` copies records into
a bounded host staging buffer. Filling that buffer triggers an upload and
projection. The upload is fenced before the host buffer can be reused;
device operations remain ordered on the same execution instance. A final
partial buffer is flushed during waveform download. Thus a per-track host
``append`` is not a per-track GPU launch or synchronization. These lifecycle
operations are implemented in `RadioAccumulator.inl`_.

Prefix-stage track files
------------------------

CPU prefix workers can capture completed tracks without initializing a GPU.
``RadioTrackSpoolWriter`` writes those records in a bounded, ordered transfer
format. ``complete()`` explicitly publishes the valid record count; destruction
after a failure leaves an incomplete marker that the reader rejects.

``RadioTrackSpoolReader`` validates the magic, version, record size, completed
count and exact file length. ``readBatch(tracks, maximum)`` clears and fills
the caller's vector with up to ``maximum`` records and returns that count.
The default is 4096, and accepted batch limits are 1 through 65536.

Version 1 uses a 32-byte header and 140-byte records with explicit
little-endian fields. It does not serialize C++ struct padding or pointers.
Record validation checks the electron/positron identity, finite values,
nonnegative weight and kinetic energies, and nondecreasing endpoint time.
Zero-length or zero-duration tracks can still be represented and are rejected
as projection tracks later. See `RadioTrackSpoolFormat.inl`_,
`RadioTrackSpoolWriter.inl`_ and `RadioTrackSpoolReader.inl`_.

The file contains track values, not observer geometry, propagation settings,
event ownership or an RNG checkpoint. Those belong to the run configuration
and coordinator. Each completed spool must be assigned to exactly one
consumer. In the standard accelerated runner, duplicate canonical input
paths are rejected and every accepted file is checked for complete
consumption. Replaying a track file calculates radio from already transported
segments; it does not rerun the shower. See :doc:`../execution/coordinator`.

Observer snapshots and coordinate consistency
---------------------------------------------

``snapshot_detail::make_observers(detector, coordinate_system)`` iterates the
native observer collection in its existing order and returns an owning
``std::vector<RadioObserverSnapshot>``. It copies positions in the requested
system and retains the actual native waveform bin count, rather than
recomputing that count from duration and sampling rate:

.. literalinclude:: ../../../corsika/detail/kokkos/radio/ObserverConversion.inl
   :language: cpp
   :lines: 19-27
   :lineno-match:

The atmospheric configuration builder uses
``environment.getCoordinateSystem()``. The homogeneous-medium builder takes
an explicit coordinate-system argument. The standard air-shower runner
attaches native track collection in ``rootCS``; its transport and observer
configuration must use that same system. The lower-level interfaces are not
intrinsically restricted to RootCS, but they contain no coordinate-system
pointer that could correct mismatched arrays during a kernel.

``RadioConfig`` owns the host observer vectors and propagation data.
Constructing it does not allocate device storage or project tracks.
Accumulator initialization validates and copies these values, then publishes
borrowed numerical views for execution. A ``RadioObserverView`` additionally
contains waveform offsets and accumulation scales assigned during that
preparation. Its backing arrays must remain accessible and alive until the
calling kernel finishes.

For atmospheric table propagation, observer heights must fall in valid table
bins. The configuration builder samples a vertical column, including a lower
1 km margin; the environment must cover that whole column. Generic export
from a material model does not prove that a straight atmospheric propagation
model is appropriate for that material. The homogeneous path instead requires
that the complete source-to-observer ray remains in the uniform medium.
Propagation-specific support is distinct from particle environment export.

Validation coverage
-------------------

* `testHostRadioCollector.cpp`_ checks native-step conversion, shared
  coordinates, kinetic-to-total energy, weights and times, e± filtering,
  single capture and restoration of the scalar route.
* `testRadioSinkScope.cpp`_ covers the four enabled-model combinations,
  nested attachment rejection and failed second attachment.
* `testRadioSpoolCompatibility.cpp`_ checks byte compatibility with the
  retained version-1 format. `testRadioTrackSpool.cpp`_ covers serialization
  and malformed/incomplete input handling.
* `testRadioSpoolReplay.cpp`_ compares direct collection with replay using
  several read-batch sizes and a final partial batch. This is a fixed-track
  radio comparison, not a comparison of independent showers.
* `RadioFixture.cpp`_ checks observer and atmospheric-snapshot conversion
  against the retained implementation, including fractional native durations
  and independent firn export. The firn export case does not validate a
  straight-ray propagation approximation in firn.

These references describe existing tests. Their presence does not establish
that every backend has run every test. End-to-end physics validation also
requires appropriate trajectories, propagation conditions and observables.

.. _HostRadioRecord.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/HostRadioRecord.inl
.. _RadioTrackSegment.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/modules/radio/RadioTrackSegment.hpp
.. _RadioProjection.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/algorithms/RadioProjection.inl
.. _RadioProcess.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/modules/radio/RadioProcess.inl
.. _CascadeRunner.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeRunner.inl
.. _ScopedRadioTrackSink.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/ScopedRadioTrackSink.inl
.. _IRadioBackend.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/IRadioBackend.hpp
.. _BorrowedRadioBackend.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/BorrowedRadioBackend.inl
.. _RadioAccumulator.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/RadioAccumulator.inl
.. _RadioTrackSpoolFormat.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/RadioTrackSpoolFormat.inl
.. _RadioTrackSpoolWriter.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/RadioTrackSpoolWriter.inl
.. _RadioTrackSpoolReader.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/RadioTrackSpoolReader.inl
.. _testHostRadioCollector.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/host/testHostRadioCollector.cpp
.. _testRadioSinkScope.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/host/testRadioSinkScope.cpp
.. _testRadioSpoolCompatibility.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/host/testRadioSpoolCompatibility.cpp
.. _testRadioTrackSpool.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/host/testRadioTrackSpool.cpp
.. _testRadioSpoolReplay.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/host/testRadioSpoolReplay.cpp
.. _RadioFixture.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/RadioFixture.cpp
