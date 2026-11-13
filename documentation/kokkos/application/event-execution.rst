Event execution and results
===========================

This page follows one event after the run-level ``CascadeModule`` has been
created. The primary, tracking object, stack and process sequence are still
constructed by ``main``.

Create the event and its input
------------------------------

At the start of the event loop, the application handles a worker request when
running as a resident worker, then creates the event scope:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: if (!workerSession.next(backend, execution, output, seed)) break;
   :end-at: auto event = execution.shower(workerSession.isActive() ? 1 : i_shower);
   :lineno-match:

For a direct run, ``execution.shower(i_shower)`` returns
``CascadeEvent<...>``. The event retains references to its driver and geometry,
records the ordinal and captures the pre-event diagnostic state. This call
does not transport the primary.

``main`` builds the ordinary process sequence:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: // assemble the final process sequence
   :end-at: observationLevel, inter_writer, thinning, cut);
   :lineno-match:

It then supplies the primary through the event:

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

Construct the runnable adapter
------------------------------

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

``forceInteraction()`` and ``forceDecay()`` set flags in the adapter. They are
applied to the selected cascade when ``run()`` executes the stored callback:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/CascadeAdapter.inl
   :language: cpp
   :start-at: void CascadeAdapter<TRun>::run() {
   :end-before: } // namespace corsika::kokkos::application
   :lineno-match:

The factory returns an object; ``EAS.run()`` returns ``void``. The simulation
results are written through the references supplied to that object.

Select the native or accelerated runner
---------------------------------------

``CascadeDriver::cascade(...)`` installs the following branch order in its
callback:

1. ``input.runPrefix(...)`` handles a CPU-prefix capture/splitting role. If it
   returns ``true``, that role has completed and the callback returns.
2. With ``options_.emBackend_ == "proposal"``, construct and run a native C8
   ``Cascade`` using the original environment, tracking, sequence, output and
   stack.
3. Otherwise, construct the accelerated event adapter through
   ``accelerated_.cascade(...)`` and call its ``run()``.

The native branch contains the familiar C8 construction:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/CascadeDriver.inl
   :language: cpp
   :start-at: Cascade EAS(geometry_.environment_, tracking, sequence, output, stack);
   :end-at: EAS.run();
   :lineno-match:

Thus the outer ``EAS`` in ``main`` is an adapter. This inner ``EAS`` is the
native ``Cascade`` selected by that adapter. They are separate local objects
with the same run/forced-primary operations.

Recover the actual event processes and writers
----------------------------------------------

On the accelerated branch, ``CascadeApplication::cascade(...)`` combines the
run defaults with event-specific energy, weight and IDs. Its callback then
calls ``complete_models(models_, sequence)`` and
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

These calls return reference bundles, ``CascadeModels<...>`` and
``CascadeOutputs<...>``. ``select_process`` matches process types at compile
time and requires exactly one match. It does not run a hadronic switch's
particle-dependent condition to select a physics outcome. A matching
composite retains its whole switch scope.

The resulting ``CascadeContext`` groups geometry, outputs, processes, execution
objects and diagnostics. For ``kokkos-proposal``, it is passed to
``run_cascade(session_, options_, event, context)``. An enabled EGS4 session
takes its separate ``egs4_->run(...)`` branch.

.. _application-backend-start:

Prepare the backend and start transport
---------------------------------------

The PROPOSAL runner exports the observation plane from the actual observation
process and prepares the host data:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/CascadeRunner.inl
   :language: cpp
   :start-at: auto const observation_plane =
   :end-at: showerCoreY_m, magnetic_field_T, dEdX, profile, &observation_plane);
   :lineno-match:

``prepare_cascade(...)`` returns ``PreparedCascade<...>`` as described in
:doc:`data-contracts`. Inside the runner's ``begin_backend`` callback, the
interaction and continuous models provide calculator views on the first
initialization. The session is then started or reused:

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

``backend_`` is a borrowed pointer to the session-owned ``EmBackend``;
``reused_`` reports whether it was already initialized; and
``muonTransportAvailable_`` reports the backend capability. On first use the
session exports tables and initializes the backend. On reuse it calls
``beginShower(...)`` to reset event state, after checking environment
compatibility.

Finally, ``run_hybrid_cascade(...)`` validates the process registry, obtains the
backend, constructs the CPU-fallback handler and output sink, and creates the
particle router and ``HybridCascade``. It calls ``cascade.run()`` and then the
completion/report callback. Both this function and ``run_cascade(...)`` return
``void``.

Where the results go
--------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Result
     - Receiver
   * - Particle propagation and interactions
     - The supplied stack, particle router and CPU/device processing paths.
   * - Energy deposition and longitudinal profiles
     - The writers recovered from the event sequence, connected through
       ``CorsikaOutputSink``.
   * - Observation-level particles and interaction records
     - The existing observation and interaction writers.
   * - Radio waveforms
     - The bound CoREAS/ZHS observer collections and their output modules.
   * - Backend statistics and completion
     - Run-session output, recorded by ``ShowerReportBuilder::record(...)``
       after accelerated execution.

After ``EAS.run()`` returns, ``main`` reads the same writers, for example
``dEdX.getEnergyLost()`` and ``observationLevel.getEnergyGround()`` for its energy
budget. The event wrapper also completes any radio-track capture after a
successful run.

An exception in the accelerated branch is recorded as incomplete. The driver
closes an active shower/output library and throws ``CascadeAborted``. A failed
run must therefore not be interpreted as a completed event merely because
some writer files exist.

Source references and existing checks
-------------------------------------

Follow `CascadeDriver.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeDriver.inl>`_,
`CascadeApplication.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeApplication.inl>`_,
`CascadeRunner.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeRunner.inl>`_
and `EmRunSession.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmRunSession.cpp>`_.

Existing focused checks include `execution selection
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/application_support/ExecutionSelection.cpp>`_,
`bindings and adapter behavior
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/application_support/bindings/Bindings.cpp>`_,
`process selection
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/framework/process/testProcessSelection.cpp>`_
and `hybrid cascade behavior
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/scheduling/hybrid/testHybridCascade.cpp>`_.
They cover interface contracts at different levels; none alone establishes
full-shower statistical agreement. This chapter documents their scope, without
reporting a new simulation campaign.
