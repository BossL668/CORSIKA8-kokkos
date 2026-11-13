Application data and lifetime contracts
=======================================

The run is connected with three calls immediately before the event loop. These
are actual calls from ``main``; the environment and models already exist here.

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: auto geometry =
   :end-at: coreas, zhs, heCounted, leIntCounted, output);
   :lineno-match:

``geometry`` describes the scene through references, ``defaults`` contains
configuration values, and ``execution`` retains the library driver across
events. None of these calls advances a particle.

geometry(): borrow the existing scene
-------------------------------------

``CascadeExtension::geometry(...)`` calls ``make_cascade_geometry(...)`` and
returns ``CascadeGeometry<...>`` by value. Copying that bundle copies references
to C8 objects; it does not copy the environment tree to a GPU.

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Argument in ``main``
     - Field in ``CascadeGeometry``
     - Meaning
   * - ``env``
     - ``environment_``
     - Reference to the native ``Environment<EnvironmentInterface>`` and its
       geometry/material tree.
   * - ``rootCS``
     - ``coordinates_``
     - Reference to the coordinate-system pointer used for numerical export.
   * - ``injectionPos``
     - ``injectionPosition_``
     - Native primary-injection point, retaining its units and coordinates.
   * - ``surface_``
     - ``surface_``
     - Native surface reference used by the propagation setup.
   * - ``step``
     - ``propagationStep_``
     - Radio propagation step, set to ``1_m`` in this application. It is not
       the particle's selected transport step.
   * - ``showerAxis``
     - ``showerAxis_``
     - Native shower axis used for longitudinal projections.
   * - ``dX``
     - ``depthStep_``
     - Profile bin width, ``10_g / square(1_cm)`` here. It is not a sampled
       interaction depth.

The factory also copies the observation height, horizontal core coordinates
and magnetic-field metadata from the previously configured
``RunEnvironmentConfig``. The actual device field is obtained from the native
environment's model export during ``prepare_cascade(...)``.

The storage contract is visible in the declaration:

.. literalinclude:: ../../../corsika/kokkos/application/cascade/CascadeContext.hpp
   :language: cpp
   :start-at: struct CascadeGeometry {
   :end-at: };
   :lineno-match:

The referenced scene objects must remain alive through transport, CPU fallback
and output. Changing a few fields in ``RunEnvironmentConfig`` does not replace
the actual C8 environment.

make_cascade_defaults(): copy run settings
------------------------------------------

``make_cascade_defaults(...)`` returns ``CascadeEventConfig``. Its inputs are:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Input
     - Values read or stored
   * - ``app``
     - Parsed thinning fraction, automatic-weight selection, high-energy
       hadronic model name and maximum magnetic deflection angle.
   * - ``seed``, ``eMax``, ``beamCode``
     - Run seed, maximum configured primary energy and primary particle code.
       ``eMax`` has the native ``HEPEnergyType``.
   * - ``cut``
     - Electron and muon kinetic-energy cuts, obtained with
       ``getElectronKineticECut()`` and ``getMuonKineticECut()``.
   * - ``prod_threshold``, ``heHadronModelThreshold``
     - Native energy quantities for secondary production and the hadronic
       model transition.
   * - ``multithin``
     - Stored as ``keepZeroWeightParticles_``.

This is a value object. It does not store a new primary particle or start a
shower. ``CascadeApplication::cascade(...)`` later copies it and fills the
event's ordinal, output ID, actual primary energy and maximum weight. That
distinction also supports a run whose primary energies are sampled over a
range.

execution(): bind models and outputs for the run
------------------------------------------------

``backend.execution(...)`` receives ``geometry`` and ``defaults`` by value,
then references to six existing model/counter objects and the output manager:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Arguments
     - Why the library needs them
   * - ``emCascade``, ``emContinuousProposal``
     - Native interaction and continuous-process instances. They provide
       PROPOSAL calculator views for table export and the corresponding native
       process connections.
   * - ``coreas``, ``zhs``
     - Radio process instances. Their ``observerCollection()`` methods supply
       the configured observer collections.
   * - ``heCounted``, ``leIntCounted``
     - Existing hadronic interaction counters used for event diagnostics.
   * - ``output``
     - Existing ``OutputManager&`` used to register run-level timing and any
       additional backend output.

It creates ``SequenceModels``, ``SequenceOutputs`` and ``CascadeMonitors``
bundles, then returns a ``CascadeModule<...>``. The returned module contains
its ``CascadeDriver``, timing writer and binding/configuration state. The
physical model objects and detector collections remain owned by the caller.

For example, the run-level model bundle contains references:

.. literalinclude:: ../../../corsika/kokkos/application/cascade/SequenceModels.hpp
   :language: cpp
   :start-at: struct SequenceModels {
   :end-at: };
   :lineno-match:

Event-specific processes and writers are obtained from the actual
``ProcessSequence`` when the event executes. The application therefore does
not pass every event-local object to ``execution(...)``. The
:doc:`event-execution` page follows this lookup.

Where host objects become device data
-------------------------------------

The main application passes C8 objects to a host-side interface. GPU kernels
receive numerical views and backend-managed storage later in the call chain:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Boundary
     - Returned data or state change
   * - ``prepare_cascade(...)``
     - Returns ``PreparedCascade<...>``. It owns a shared reference to an
       exported environment snapshot, provides its ``EnvironmentDataView`` and
       prepares transport, radio, profile and physics-requirement configuration.
       It does not transport particles or initialize the backend.
   * - ``nativeCalculatorViews()``
     - The interaction and continuous models provide native calculator views.
       The runner requests them on first backend initialization.
   * - ``beginProposalNative(...)``
     - Exports/loads native table data and initializes device storage on the
       first call; resets shower-local backend state on reuse. Returns
       ``SessionBeginResult``; see :ref:`application-backend-start`.

``PreparedCascade::environmentData_`` borrows
``PreparedCascade::environmentStorage_``. The original native environment also
remains necessary for CPU processing and node-aware fallback. A changed
environment is rejected when reusing an initialized ``EmRunSession``; create
a new session to upload a different scene.

Ownership rules for application authors
---------------------------------------

* Keep ``CLI::App`` alive for the extension, and keep ``backend`` alive for
  the returned ``execution`` module.
* Keep the native environment, coordinate system, geometry, model instances,
  counters, detectors and output manager alive while their bindings are used.
* Keep the event, stack, tracking object and sequence alive until the returned
  adapter's ``run()`` finishes. Its callbacks capture references.
* Treat run-configuration values and borrowed object state differently:
  copying a configuration struct does not create independent models or writers.

Source references
-----------------

The factories are in `CascadeSupport.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeSupport.inl>`_
and `CascadeExtension.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeExtension.inl>`_.
The receiving types are declared in `CascadeContext.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeContext.hpp>`_,
`CascadeEventConfig.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeEventConfig.hpp>`_
and `PreparedCascade.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/PreparedCascade.hpp>`_.
