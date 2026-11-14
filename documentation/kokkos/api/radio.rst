Radio inputs, accumulation and propagation
==========================================

Radio interfaces consume completed tracks. Host interfaces manage input
lifetime and output completion, while numerical projection functions operate
on borrowed records and waveform arrays. The declarations below are generated
from the public headers at the baseline recorded in :doc:`../api`.

Read :doc:`../radio/inputs` for track collection and observer preparation,
:doc:`../radio/accumulation` for execution and accumulation, and
:doc:`../radio/propagation` for ordinary, geometrical-optics and full-wave
propagation. The separate propagation interfaces have different input and
output contracts.

Track conversion and observer configuration
-------------------------------------------

The native-step adapter converts a completed electron/positron segment into
the standard projector record. It is not a particle-transport operation.
Observer positions and track positions must share the configured coordinate
system; observer times use seconds and sampling rates use Hz. Constructing
``RadioConfig`` does not initialize an accumulator or transfer arrays.

Sources: `HostRadioRecord.hpp`_, `RadioObserverSnapshot.hpp`_ and
`RadioConfig.hpp`_.

.. doxygenfunction:: corsika::kokkos::radio::make_host_radio_record

.. doxygenstruct:: corsika::kokkos::radio::RadioObserverSnapshot
   :members:
   :undoc-members:

.. doxygenstruct:: corsika::kokkos::radio::RadioConfig
   :members:
   :undoc-members:

Host dispatch and resident accumulation
---------------------------------------

``IRadioBackend`` accepts borrowed host tracks and requires their contents to
be copied before submission returns. Download completes pending input,
including a final partial batch. Its historical ``gpuRadioEnabled`` name
reports whether the implementation accepts these records; it does not
require CUDA or even a GPU execution space.

``RadioAccumulator`` owns ordinary CoREAS/ZHS storage in one execution space.
Host submission copies input; resident submission borrows the supplied device
view. The runtime, accumulator and device input allocations must remain valid
until submitted work completes. Use the same execution instance for dependent
operations. The reference below lists consumer lifecycle methods, excluding
the public projection helpers exposed for compiler implementation constraints.

Sources: `IRadioBackend.hpp`_ and `RadioAccumulator.hpp`_.

.. doxygenclass:: corsika::kokkos::radio::IRadioBackend
   :members:
   :undoc-members:

.. doxygenclass:: corsika::kokkos::radio::RadioAccumulator
   :members: projectedDeviceBytes, initialize, enabled, reset, enqueueHostLeptonTracks, flushHostLeptonTracks, accumulateLeptonTracks, downloadFixed, download, statistics, deviceBytes, staticHostToDeviceBytes, reserveTrackCapacity, projectedTrackCapacity
   :undoc-members:

Returned samples
----------------

The ordinary CoREAS array contains electric-field samples; the ZHS array
contains vector-potential samples. Native output performs the required
adjacent-sample differentiation for ZHS. A component vector alone carries
neither observer position nor its sample grid: retain the matching observer
configuration and ordering. Firn accumulators below return their own spectral
records rather than this ordinary waveform container.

Sources: `RadioWaveform.hpp`_ and `RadioWaveforms.hpp`_.

.. doxygenstruct:: corsika::kokkos::radio::RadioWaveform
   :members:
   :undoc-members:

.. doxygenstruct:: corsika::kokkos::radio::RadioWaveforms
   :members:
   :undoc-members:

Numerical emission projection
-----------------------------

These functions add one prepared track's contribution to caller-owned
waveform storage through the supplied atomic-operation type. They do not
allocate an observer grid or synchronize a host download. Propagation and
waveform views must be accessible, initialized and alive throughout the
kernel. These ordinary straight-ray projection interfaces are not substitutes
for firn ray solving. Source: `RadioProjection.hpp`_.

.. doxygenfunction:: corsika::kokkos::radio::accumulate_co_reas

.. doxygenfunction:: corsika::kokkos::radio::accumulate_zhs

Firn propagation interfaces
---------------------------

The geometrical-optics accumulator accepts tracks, solves the configured
exponential-profile paths and accumulates the selected emission response.
The full-wave accumulator instead consumes subdivided segments and a matching
external response table. Both are host-facing interfaces with execution-space
work behind them; their factory signatures expose the required inputs.
Factories do not establish support for arbitrary optical profiles or geometry.

Runtime acquisition is lazy. If an EM runtime is borrowed, it must outlive the
accumulator. Use and destroy these interfaces on the coordinating host thread.
The propagation chapter specifies table coverage, branch handling, source
formalism and finalization, which remain necessary when interpreting a
downloaded spectrum. Sources: `IGeometricalOpticsAccumulator.hpp`_ and
`IFullWaveAccumulator.hpp`_.

.. doxygenclass:: corsika::kokkos::radio::firn::IGeometricalOpticsAccumulator
   :members:
   :undoc-members:

.. doxygenfunction:: corsika::kokkos::radio::firn::make_geometrical_optics_accumulator

.. doxygenclass:: corsika::kokkos::radio::firn::IFullWaveAccumulator
   :members:
   :undoc-members:

.. doxygenfunction:: corsika::kokkos::radio::firn::make_full_wave_accumulator

.. _HostRadioRecord.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/HostRadioRecord.hpp
.. _RadioObserverSnapshot.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/RadioObserverSnapshot.hpp
.. _RadioConfig.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/RadioConfig.hpp
.. _IRadioBackend.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/IRadioBackend.hpp
.. _RadioAccumulator.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/RadioAccumulator.hpp
.. _RadioWaveform.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/RadioWaveform.hpp
.. _RadioWaveforms.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/RadioWaveforms.hpp
.. _RadioProjection.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/algorithms/RadioProjection.hpp
.. _IGeometricalOpticsAccumulator.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/firn/IGeometricalOpticsAccumulator.hpp
.. _IFullWaveAccumulator.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/firn/IFullWaveAccumulator.hpp
