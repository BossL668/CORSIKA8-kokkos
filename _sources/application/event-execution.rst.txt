Event execution and results
===========================

The run-level ``CascadeModule`` binds the application-owned models and
environment as described in :doc:`data-contracts`. For each event, ``main``
constructs the primary, tracking object, particle stack and process sequence,
then obtains an executable cascade adapter.

This page documents the event lifecycle from particle input to output
finalization. It distinguishes adapter construction from execution and the
per-event objects from the backend session retained across events. The
accelerated implementation described below is ``kokkos-proposal``; the EGS4
branch is identified separately.

.. _create-the-event-and-its-input:

Event initialization and particle input
---------------------------------------

At the start of the event loop, the application handles a worker request when
running as a resident worker, then creates the event scope:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: if (!workerSession.next(backend, execution, output, seed)) break;
   :end-at: auto event = execution.shower(workerSession.isActive() ? 1 : i_shower);
   :lineno-match:

``execution.shower(...)`` returns ``CascadeEvent<...>``. The event retains
references to the driver and geometry, records the event ordinal and captures
the initial diagnostic counters. Particle transport begins later, when
``EAS.run()`` invokes the selected execution path.

``main`` assembles the C8 process sequence for this event:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: // assemble the final process sequence
   :end-at: observationLevel, inter_writer, thinning, cut);
   :lineno-match:

After constructing the tracking object and clearing the stack, the application
supplies the primary:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: auto const primaryProperties =
   :end-at: primaryWriter.recordPrimary(primaryProperties);
   :lineno-match:

``addPrimary(TStack&, TPrimary const&)`` returns ``void`` and constructs the
event's ``CascadeParticleInput``. For a direct run, it calls
``stack.addParticle(primaryProperties)``. The tuple contains particle code,
kinetic energy, normalized direction, injection point and initial time, all
in their native C8 types. In a worker receiving a frontier, the same interface
imports or stages the assigned particles instead of adding a second primary.

.. _construct-the-runnable-adapter:

Cascade adapter construction and lifetime
-----------------------------------------

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: auto EAS =
   :end-at: EAS.run();
   :lineno-match:

The arguments to ``event.cascade(...)`` are:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Input
     - Role
   * - ``tracking``
     - Reference to the native tracking object for scalar propagation.
   * - ``sequence``
     - Reference to this event's configured process tree.
   * - ``output``
     - Reference to the output manager used by the selected cascade.
   * - ``stack``
     - Reference to the event's particle stack.
   * - ``primaryTotalEnergy``
     - This event's total primary energy, as ``HEPEnergyType``. It is distinct
       from the kinetic energy in ``primaryProperties``.
   * - ``maxWeight``
     - Dimensionless maximum thinning weight selected for this event.

The overload used by ``main`` obtains the inspector and thinning process from
``sequence`` with ``select_process<...>()``. It requires the earlier
``addPrimary(...)`` call, then returns a ``CascadeAdapter<...>`` wrapping the
driver's event runner.

``forceInteraction()`` and ``forceDecay()`` set flags in the adapter. For
native and accelerated event execution, these flags are applied to the
selected cascade when ``run()`` executes the stored callback:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/CascadeAdapter.inl
   :language: cpp
   :start-at: void CascadeAdapter<TRun>::run() {
   :end-before: } // namespace corsika::kokkos::application
   :lineno-match:

``event.cascade(...)`` constructs the adapter without advancing particles.
``EAS.run()`` executes its stored callback and returns ``void``; simulation
state and results are updated through the supplied objects. The event,
tracking object, sequence, output manager and stack must remain valid until
``run()`` completes. The adapter borrows these objects rather than owning
independent copies.

.. _select-the-native-or-accelerated-runner:

Execution-path selection
------------------------

``CascadeDriver::cascade(...)`` installs the following branch order in its
callback:

1. ``input.runPrefix(...)`` handles a configured CPU-prefix capture or
   splitting role. If it returns ``true``, that stage has completed and the
   callback returns. This completion refers to the assigned prefix work,
   not to the downstream shower execution.
2. With ``options_.emBackend_ == "proposal"``, construct and run a native C8
   ``Cascade`` using the original environment, tracking, sequence, output and
   stack.
3. Otherwise, construct the accelerated event adapter through
   ``accelerated_.cascade(...)`` and call its ``run()``.

The native branch constructs C8's ``Cascade`` directly:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/CascadeDriver.inl
   :language: cpp
   :start-at: Cascade EAS(geometry_.environment_, tracking, sequence, output, stack);
   :end-at: EAS.run();
   :lineno-match:

The variable ``EAS`` in ``main`` therefore holds a ``CascadeAdapter``; the
local ``EAS`` in this branch holds the native ``Cascade``. The adapter forwards
execution and forced-primary settings to the selected implementation.

This selection concerns the electromagnetic execution path. The Kokkos
execution space, such as OpenMP, CUDA or HIP, is configured separately;
selecting the accelerated branch does not by itself imply GPU execution.

.. _recover-the-actual-event-processes-and-writers:

Event process and output binding
--------------------------------

On the accelerated branch, ``CascadeApplication::cascade(...)`` copies the
run defaults and assigns the event's primary energy, maximum weight, ordinal
and output identifier. When its adapter runs, the callback resolves model
and writer references through ``complete_models(models_, sequence)`` and
``complete_outputs(outputs_, sequence)``.

``complete_models`` keeps the explicitly bound radio and electromagnetic
models, then borrows the neutrino, hadron-switch, decay-switch, continuous-loss,
longitudinal, production and cut processes from the event sequence.
``complete_outputs`` retrieves the associated writers:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/SequenceOutputs.inl
   :language: cpp
   :start-at: auto complete_outputs(SequenceOutputs<TC, TZ> const& outputs, TS& sequence) {
   :end-before: } // namespace corsika::kokkos::application
   :lineno-match:

These calls return ``CascadeModels<...>`` and ``CascadeOutputs<...>`` containing
references to existing objects. ``select_process`` resolves process types at
compile time and requires exactly one match. For a composite process such as
a hadronic switch, it retrieves the switch itself; the particle-dependent
choice of physics model remains part of that process's execution.

The resulting ``CascadeContext`` groups geometry, outputs, processes, execution
objects and diagnostics. For ``kokkos-proposal``, it is passed to
``run_cascade(session_, options_, event, context)``. If ``kokkos-egs4`` was
selected in a build with EGS4 support, the context instead enters
``egs4_->run(...)``. The session initialization below is specific to the
PROPOSAL-table path.

.. _application-backend-start:
.. _prepare-the-backend-and-start-transport:

Backend preparation and session reuse
-------------------------------------

The PROPOSAL runner exports the observation plane from the actual observation
process and prepares the host data:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/CascadeRunner.inl
   :language: cpp
   :start-at: auto const observation_plane =
   :end-at: showerCoreY_m, magnetic_field_T, dEdX, profile, &observation_plane);
   :lineno-match:

``prepare_cascade(...)`` returns ``PreparedCascade<...>``, which holds the
environment snapshot, numerical views, geometry and backend settings described
in :doc:`data-contracts`. This preparation step does not initialize the EM
backend.

The runner then defines a ``begin_backend`` callback. It passes this callback
to ``run_hybrid_cascade(...)``, which invokes it after validating the process
registry. On first use, the callback obtains native calculator views from
the interaction and continuous-loss models. It then initializes or reuses
the session through:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/CascadeRunner.inl
   :language: cpp
   :start-at: auto session_begin = runtime_session.beginProposalNative(
   :end-at: muon_transport_available = session_begin.muonTransportAvailable_;
   :lineno-match:

The six inputs are the numerical environment view, transport geometry,
backend configuration, physics requirements, native interaction-calculator
views and native continuous-calculator views. The return type is:

.. literalinclude:: ../../../corsika/kokkos/em/EmRunSession.hpp
   :language: cpp
   :start-at: struct SessionBeginResult {
   :end-at: };
   :lineno-match:

``backend_`` is a borrowed pointer to the session-owned ``EmBackend``.
``reused_`` indicates whether the session was already initialized, and
``muonTransportAvailable_`` reports its muon-transport capability.

On first use, the session exports and validates the native tables, prepares
auxiliary data and initializes the backend. On subsequent events, it retains
the backend and tables, checks environment and transport-geometry compatibility,
then calls ``beginShower(...)`` to reset event state. An incompatible environment
requires a new session. Reuse therefore preserves initialized resources while
the event configuration and particle state are updated.

Hybrid cascade execution
------------------------

``run_hybrid_cascade(...)`` connects the prepared data to the scheduling layer
in the following order:

1. Validate the process sequence against the accelerated process registry.
2. Invoke ``begin_backend`` to obtain the initialized or reused backend and
   configure radio collection when enabled.
3. Construct the CPU-fallback handler, ``CorsikaOutputSink`` and particle
   router, using the exported environment topology.
4. Construct ``HybridCascade`` with the original environment, tracking,
   process sequence, output manager and stack; apply the event configuration.
5. Execute ``cascade.run()``. After it returns, invoke the completion callback
   that records backend and cascade statistics.

Both ``run_hybrid_cascade(...)`` and ``run_cascade(...)`` return ``void``.
The :doc:`../execution/hybrid-loop` page describes particle scheduling,
CPU fallback and batch processing inside ``HybridCascade::run()``.

.. _where-the-results-go:

Output lifecycle
----------------

The execution interface updates the application's existing writers and
observer collections; it does not return a separate event-result object.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Result
     - Receiver
   * - Energy deposition and longitudinal profiles
     - The writers recovered from the event sequence, connected through
       ``CorsikaOutputSink``.
   * - Observation-level particles and interaction records
     - The existing observation and interaction writers.
   * - Radio waveforms
     - The bound CoREAS/ZHS observer collections and their output modules.
   * - Backend and cascade statistics
     - Run-session output, recorded by ``ShowerReportBuilder::record(...)``
       after accelerated execution.

``HybridCascade::run()`` starts the shower output, advances particles and
finalizes the router before ending the shower output. The completion callback
then records the execution statistics. After the selected runner returns,
``CascadeEvent`` completes any radio-track capture.

``main`` subsequently reads the writers, including ``dEdX.getEnergyLost()``
and ``observationLevel.getEnergyGround()`` for the energy budget. At the end
of the event loop, ``workerSession.close(output)`` finalizes the output library.
A resident worker also closes the preceding request's library when accepting
the next request. Worker coordination is described in
:doc:`../execution/coordinator`.

Failure handling
----------------

``CascadeDriver`` catches ``std::exception`` from the accelerated branch,
records the event as incomplete, closes the shower output if active, closes
the output library and throws ``CascadeAborted``. ``main`` catches that
exception and returns ``EXIT_FAILURE``. Partial output files therefore do not
establish successful event completion.

This handler surrounds the accelerated branch specifically; native execution
and CPU-prefix execution are outside that ``try`` block.

.. _source-references-and-existing-checks:

Implementation references and validation coverage
-------------------------------------------------

The execution path is implemented in `CascadeDriver.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeDriver.inl>`_,
`CascadeApplication.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeApplication.inl>`_,
`CascadeRunner.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeRunner.inl>`_
and `EmRunSession.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmRunSession.cpp>`_.

Scheduling and output finalization are implemented in `HybridCascadeRunner.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/HybridCascadeRunner.inl>`_,
`HybridCascade.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/HybridCascade.inl>`_
and `WorkerSession.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/WorkerSession.inl>`_.

Existing focused checks include `execution selection
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/application_support/ExecutionSelection.cpp>`_,
`bindings and adapter behavior
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/application_support/bindings/Bindings.cpp>`_,
`process selection
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/framework/process/testProcessSelection.cpp>`_
and `hybrid cascade behavior
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/scheduling/hybrid/testHybridCascade.cpp>`_.
These tests cover execution selection, adapter behavior, process resolution
and hybrid-cascade integration. Full-shower physics and performance comparisons
require separate validation.
