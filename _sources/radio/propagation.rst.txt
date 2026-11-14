Propagation models and medium-specific interfaces
=================================================

Particle transport and radio propagation use different representations of the
medium. The transport environment selects material nodes and boundaries. The
radio configuration supplies the optical model needed between an emission
segment and an observer. Exporting a transport environment therefore does not
automatically provide a radio propagator for every geometry.

The ordinary ``RadioAccumulator`` supports the flat-atmosphere and homogeneous
configurations described below. Firn propagation uses separate interfaces under
``corsika/kokkos/radio/firn/``. These interfaces share cascade track delivery,
but do not use the ordinary accumulator's propagation snapshot.

.. note::

   This page describes source support at `d5297c43
   <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.
   Supported configuration, numerical agreement and independent physical
   validation are separate requirements; the tests listed below establish
   specific numerical comparisons, not universal validity in complex media.

Supported configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Configuration
     - Implementation
     - Domain and application restrictions
   * - Flat atmosphere
     - ``RadioAccumulator`` with
       ``atmosphere::make_radio_config``.
     - Straight propagation using a sampled vertical refractivity profile.
       No general interface refraction or reflected paths.
   * - Homogeneous medium
     - ``RadioAccumulator`` with
       ``make_homogeneous_radio_config``.
     - Constant refractive index along the entire source–observer path.
   * - Firn geometrical optics
     - ``CoherentStratifiedGOBackend`` or
       ``IGeometricalOpticsAccumulator``.
     - Direct, turning and surface-reflected rays. The device implementation
       accepts the exponential optical profile exported by the firn application.
   * - Firn full-wave response
     - ``TabulatedCoherentFullWaveBackend`` or
       ``IFullWaveAccumulator``.
     - Precomputed Green-response data with explicit source-depth, distance,
       observer-depth and frequency coverage.
   * - Firn hybrid response
     - ``HybridCoherentFullWaveBackend``.
     - Host path only: use the table where covered, otherwise try geometrical
       optics. The Kokkos application path rejects this selection.

The ordinary configurations use the execution space selected by the Kokkos
build. For firn, ``--radio-backend`` and ``--firn-radio-model`` select the
implementation through ``CascadeExtension``; the emission formalism is a
separate choice.

Ordinary propagation configuration
----------------------------------

``atmosphere::make_radio_config`` reads the native environment through
``getContainingNode`` and ``getRefractiveIndex`` along a vertical column. It
stores sampled refractivity, cumulative integral data and boundary slopes in
``FlatAtmosphereRadioSnapshot``. Observer positions are converted into the same
coordinate system. ``RadioProjection`` uses this snapshot to calculate the
propagation quantities required by CoREAS and ZHS.

This export is specific to the flat-atmosphere model. It does not retain the
environment tree as an optical ray-tracing geometry. A more general transport
scene may be supported by the :doc:`environment representation <../environment>`
without being supported by this radio model.

``make_homogeneous_radio_config`` instead sets
``homogeneousRefractiveIndex_`` and exports the observers. It requires a finite
index of at least one. The caller must ensure that the ray remains in a
homogeneous optical domain; crossing an interface is not handled by setting
this field alone. Both configurations then use the ordinary accumulation path
described in :doc:`accumulation`.

Relevant definitions are in `FlatAtmosphereConfig.inl`_,
`HomogeneousRadioConfig.inl`_ and `RadioProjection.inl`_.

Firn application configuration
------------------------------

``c8_firn_cascade.cpp`` constructs a native firn environment and a matching
``ExponentialProfile``. The optical parameters are the deep refractive index,
index deficit, scale depth, surface height, lower depth bound and attenuation
coefficient. Source and observer coordinates for this radio interface are
local metres: north, west and height relative to the surface, positive upward.
Track times are seconds. The application rejects antennas outside its
pure-firn volume.

``OnlineRadioConfig`` contains the sampling interval, sample count, time origin,
source formalism, subdivision limits and diagnostic choices. The application
then constructs ``VariableIndexRadioProcess<AnyGreenBackend>`` with the
observers, host response object and, where selected, a dedicated accumulator.
``CascadeRadioAdapter`` delivers cascade tracks to that process. Its second
radio slot is inactive; the two slots do not imply two simultaneous firn
formalisms.

``--firn-radio-model`` accepts ``coherent-go``, ``full-wave`` or
``hybrid-full-wave``. ``--firn-radio-formalism`` accepts ``spectrum``, ``zhs``
or ``coreas``; the default is ``spectrum``. In this application, ``zhs`` and
``coreas`` require ``coherent-go``.

.. list-table::
   :header-rows: 1
   :widths: 22 24 27 27

   * - Propagation
     - Source calculation
     - ``--radio-backend cpu``
     - Kokkos radio selection
   * - ``coherent-go``
     - ``spectrum``
     - Host Green-response evaluation.
     - GO accumulator with ``DefaultExecutionSpace``.
   * - ``coherent-go``
     - ``zhs`` or ``coreas``
     - GO accumulator with ``DefaultHostExecutionSpace``.
     - GO accumulator with ``DefaultExecutionSpace``.
   * - ``full-wave``
     - ``spectrum``
     - Host table evaluation.
     - Full-wave accumulator with the selected Kokkos execution space.
   * - ``hybrid-full-wave``
     - ``spectrum``
     - Host table/GO selection.
     - Rejected; there is no implicit device-to-host hybrid fallback.

These decisions are implemented by
``CascadeExtension::firnGeometricalOpticsAccumulator`` and
``CascadeExtension::firnRadioAccumulator`` in `CascadeExtension.cpp`_.
The complete application assembly is in `c8_firn_cascade.cpp`_.

Geometrical-optics evaluation
-----------------------------

``ExponentialOpticalParameters`` is a numerical description of

.. math::

   n(z) = n_{\mathrm{deep}} - \Delta n
          \exp\!\left(\frac{z-z_{\mathrm{surface}}}{z_0}\right).

The device solver accepts these parameters, not an arbitrary host
``VerticalOpticalProfile`` object. It also uses a constant attenuation
coefficient. The existence of other host profile implementations does not
imply that they can be passed to this device solver.

``detail::go::RaySolver`` calculates path integrals and their derivatives
analytically for this exponential profile. It solves the remaining horizontal
range equation with bracketed Newton updates using analytic derivatives.
There is no repeated numerical quadrature in this device ray solver. The
host controls track subdivision and reads compact query summaries; the
execution space retains the detailed ray records.

The solver distinguishes direct, turning and surface-reflected branches.
Their records include travel time, optical and geometrical path lengths,
emission/arrival directions, geometrical spreading and attenuation. Reflection
uses complex Fresnel coefficients; branch contributions are combined
coherently rather than selecting only the earliest arrival. Disabling surface
reflection leaves turning paths enabled.

Geometrical optics has explicit limits. Near a turning-ray fold, the
``foldGuardTimeS_`` criterion can reject the configuration. A shadow region
can have no supported GO path. Branch-capacity overflow and invalid geometry
are also reported rather than silently dropping contributions. The CoREAS
path additionally checks endpoint branch consistency; a topology change
within a child track can require finer subdivision or a full-wave treatment.
The code does not automatically manufacture a full-wave response for such
cases.

The public contract and numerical solver are in
`IGeometricalOpticsAccumulator.hpp`_ and `DeviceStratifiedRaySolver.hpp`_.
Host coherent branch handling is in `CoherentStratifiedGOBackend.inl`_.

Track input and radio output
----------------------------

``IGeometricalOpticsAccumulator::append`` accepts
``GeometricalOpticsTrack``: start/end positions, start/end times and
charge in units of the elementary charge multiplied by particle weight.
The application supplies electron/positron tracks; the numerical record does
not itself define which species the application emits. ``flush`` processes
queued tracks;
``download`` flushes remaining work and returns spectra indexed by observer,
Cartesian component and frequency. ``reset`` clears event data for reuse.
``statistics`` reports query counts, allocated bytes and stage timings;
``paths`` exposes ray diagnostics for independent checks.

The formalism determines the accumulation operation:

* ``FiniteTrackSpectrum`` computes finite-track spectral contributions. Each
  observer–frequency task accumulates children in input order.
* ``ZHS`` deposits time-bin integrals of the vector potential. Finalization
  forms the electric field through the discrete time derivative and retains
  the known half-bin timing convention.
* ``CoREAS`` deposits endpoint contributions using the corresponding ray
  directions, propagation factors and endpoint treatment.

The time-domain branches use floating-point atomic accumulation; they do not
inherit the ordinary accumulator's fixed-point reproducibility contract.
Their downloaded waveforms are Fourier transformed into the common spectral
interface. ``VariableIndexRadioProcess`` then writes both spectra and
time-domain electric-field waveforms on the configured grid. Optional branch
outputs retain the individual propagation families. An adequate time window
must include all arrivals being compared, including reflected signals.

``IFullWaveAccumulator`` has analogous ``append``, ``flush``, ``download`` and
``reset`` operations, but takes pre-subdivided ``FullWaveChildSegment``
records and evaluates a tabulated response. Its observer–frequency tasks
accumulate segments in order without floating-point atomics.

Both dedicated accumulators acquire the Kokkos runtime lazily. When they
borrow an existing EM runtime, that runtime must outlive the accumulator.
Calls belong on the coordinating host thread. The interfaces are not
interchangeable with ``RadioAccumulator``; see
`GeometricalOpticsAccumulator.cpp`_, `IFullWaveAccumulator.hpp`_ and
`VariableIndexRadioProcess.inl`_ for the implementation boundaries.

Full-wave table contract
------------------------

The current coherent full-wave path consumes a schema-2 JSON manifest and
an NPZ data file. It does not generate the Green function during a shower.
The loader verifies the data hash and expected optical-profile fingerprint,
then checks axes, tensor layout and interpolation metadata.

The dataset specifies a cylindrical dyadic basis, source depths, horizontal
source–observer distances, frequency bands and **one observer depth**.
Queries must remain inside its coverage. ``covers`` checks source depth,
horizontal distance, observer depth and the requested frequency range.
Changing the antenna depths, profile or frequency grid may therefore require
a different table. Neither the host nor device full-wave implementation
provides arbitrary extrapolation.

The stored contributions are ``fullwave_direct_matched`` and
``fullwave_air_boundary``. The latter is a full-wave boundary response, not
simply a GO reflected ray. Accordingly,
``--firn-surface-reflection=false`` is rejected for full-wave selections:
the surface response is already part of the supplied dataset.

On the host, ``HybridCoherentFullWaveBackend`` uses the table when the query
is covered and otherwise evaluates GO. That alternative can still fail in a
shadow or fold region. A valid table, a valid GO path and a physically
validated combination are distinct requirements. See
`TabulatedCoherentFullWaveBackend.inl`_, `FullWaveAccumulator.cpp`_ and
`HybridCoherentFullWaveBackend.inl`_.

Extension and validation requirements
-------------------------------------

A new environment should first define which optical propagation model is
applicable. Reusing C8 scene construction is desirable, but exporting material
density alone does not establish a valid refractive-index profile or radio
boundary response. Source and observer coordinate conventions must agree
between the scene, track conversion and optical model.

For a new analytic profile, add explicit numerical parameters and a
device-callable evaluator/solver, then compare its paths with a suitable
independent reference. For a new full-wave dataset, establish its profile,
domain, phase convention, normalization and interpolation validity before
using it in a shower. A matching fingerprint and checksum establish data
identity, not physical accuracy.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Source test
     - Numerical coverage
   * - `testGeometricalOpticsAccumulator.cpp`_
     - Device/host path and spectrum comparisons; direct, reflected and
       turning paths; analytic-integral checks; observer-depth variations;
       batch partitioning, reflection selection and accumulator reuse.
   * - `testTimeDomainEmission.cpp`_
     - ZHS bin integration, near-cone limits, endpoint treatment,
       spectral/time-domain comparisons and selected homogeneous C8 references.
   * - `testFullWaveAccumulator.cpp`_
     - Host/device table projection, branch outputs, ordering, reset and
       invalid-domain handling. The default fixture is synthetic; an external
       table can be selected with ``CORSIKA_FIRN_RADIO_TEST_TABLE``.
   * - `testFirnRadio.cpp`_
     - Optical profiles, host propagation, table contracts, Fourier
       conventions and prefix-track replay interfaces.

These tests exercise the backend selected by their build. Their presence
does not establish that every GPU architecture has passed them, or that a
complete shower reproduces NuRadioMC or a published result. Full-shower
validation must additionally align the source physics, geometry, propagation
branches, bandwidth and time window, and inspect unfiltered waveforms.

.. _FlatAtmosphereConfig.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/atmosphere/FlatAtmosphereConfig.inl
.. _HomogeneousRadioConfig.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/ice/HomogeneousRadioConfig.inl
.. _RadioProjection.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/algorithms/RadioProjection.inl
.. _c8_firn_cascade.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/applications/c8_firn_cascade.cpp
.. _CascadeExtension.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/cascade/CascadeExtension.cpp
.. _IGeometricalOpticsAccumulator.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/firn/IGeometricalOpticsAccumulator.hpp
.. _DeviceStratifiedRaySolver.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/firn/DeviceStratifiedRaySolver.hpp
.. _CoherentStratifiedGOBackend.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/firn/CoherentStratifiedGOBackend.inl
.. _GeometricalOpticsAccumulator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/radio/firn/GeometricalOpticsAccumulator.cpp
.. _IFullWaveAccumulator.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/firn/IFullWaveAccumulator.hpp
.. _VariableIndexRadioProcess.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/firn/VariableIndexRadioProcess.inl
.. _TabulatedCoherentFullWaveBackend.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/firn/TabulatedCoherentFullWaveBackend.inl
.. _FullWaveAccumulator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/radio/firn/FullWaveAccumulator.cpp
.. _HybridCoherentFullWaveBackend.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/firn/HybridCoherentFullWaveBackend.inl
.. _testGeometricalOpticsAccumulator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/firn/testGeometricalOpticsAccumulator.cpp
.. _testTimeDomainEmission.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/firn/testTimeDomainEmission.cpp
.. _testFullWaveAccumulator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/firn/testFullWaveAccumulator.cpp
.. _testFirnRadio.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/firn/testFirnRadio.cpp
