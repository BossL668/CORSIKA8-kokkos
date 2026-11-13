The hybrid event loop
=====================

``HybridCascade`` combines the native C8 stack and scalar stepper with an EM
router. The loop itself runs on the host. Parallel particle work begins inside
the backend calls made by the router.

1. Construct the event's scheduling objects
-------------------------------------------

The :ref:`application handoff <application-source-flow>` ends in
``run_hybrid_cascade(...)`` in ``HybridCascadeRunner.inl``. It receives native
C8 objects, an environment snapshot, transport geometry and callbacks:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Input
     - How it is used
   * - ``environment``, ``tracking``, ``sequence``, ``output``, ``stack``
     - References used by ``HybridCascade`` and its native scalar stepper.
   * - ``environmentSnapshot``, ``geometry``, ``coordinateSystem``
     - Borrowed numerical environment data and native node mapping used by
       particle conversion, routing and CPU return paths.
   * - ``backendFactory``
     - The application's ``begin_backend`` callback. Initializes or reuses the
       EM session and returns a backend reference.
   * - ``fallbackFactory`` / ``outputSinkFactory``
     - Create the CPU completion handler and C8 output adapter for this event.
   * - ``configureCascade`` / ``onComplete``
     - Apply run controls before execution; collect the final report after it.

The function first validates the process registry against the actual C8
sequence. It then constructs the adapters and runs the event:

.. literalinclude:: ../../../corsika/detail/kokkos/scheduling/HybridCascadeRunner.inl
   :language: cpp
   :start-at: using tracking_type = std::remove_reference_t<TTracking>;
   :end-at: fallbackHandler);
   :lineno-match:

The function returns ``void``. Particle state changes and records go through
the stack, backend, output sink and writers; ``onComplete`` receives references
to those objects. The router, fallback handler and sink remain alive until
that callback returns. The native environment and its snapshot must remain
valid throughout execution.

2. Acquire and route a particle
-------------------------------

``CpuParticleScheduler::acquireNext()`` selects a non-erased, non-suspended
particle in stack order, starting at the end. It returns a ``ScheduledParticle``
containing the particle handle, history/parent identifiers, generation and the
step identifier from ``beginTransportStep()``. ``completeParticleStep()`` closes
that host scheduling operation; it does not mean that the particle has ended.

``HybridCascade::run()`` starts the output event, sets environment nodes and
checks whether the selected particle can be routed. An explicitly forced
interaction or decay is resolved on the scalar path before ordinary routing.
For a routable particle, the actual transfer is:

.. literalinclude:: ../../../corsika/detail/kokkos/scheduling/HybridCascade.inl
   :language: cpp
   :start-at: emRouter_->stage(scheduled.particle_, scheduled.historyId_,
   :end-at: advanceInterleavedRouter();
   :lineno-match:

``EmRouter::canRoute(particle, stepId)`` checks one-step CPU-return markers,
particle type, backend capability and the environment query.
``stage(...)`` converts the state to the numerical representation and stores
it in the host vector ``staged_``. Only after staging succeeds does the loop
erase the native stack entry. This transfers responsibility for advancing that
state to the router.

3. Advance scalar and device work
---------------------------------

Particles not routed use ``ScalarCascadeStepper::advance(particle)`` with the
native environment, tracking and process sequence. Surviving particles and
secondaries remain available through the C8 stack. An optional hadronic process
pool can suspend a selected particle while an interaction request is outstanding;
this is separate from the CPU-prefix coordinator discussed later.

For routed work, ``readyForScalarInterleave()`` allows the resident front to
advance before the scalar stack is exhausted. In the ordinary endpoint this
is triggered by a sufficiently large staging vector, pending resident photons
or leptons, or a forced small-batch flush. When the scalar stack empties,
``HybridCascade`` also advances any pending router work. Returned CPU work
has its environment nodes refreshed before further scalar steps.

This is host-controlled interleaving. It does not imply that a scalar C8 step
and a device wavefront are running concurrently, or that one GPU thread keeps
ownership of a particle for its entire lifetime. The optional independent
endpoint has additional scheduling callbacks; its policy is not shown in the
ordinary flow diagram.

4. Finish only after all work is consumed
-----------------------------------------

The loop tests three sources of pending work: the CPU scheduler, EM router and
deferred hadronic interactions. The router includes staged particles, resident
photon/lepton queues and deferred CPU completions. Cascade-equation hooks can
add more stack work, so their execution is also inside the outer loop.

At completion, ``EmRouter::endOfShower()`` checks that its queues are drained,
downloads enabled accumulators and passes results to the output sink.
``output_.endOfShower()`` closes the shower. Only then does the runner execute
its reporting callback. Emptying the native stack alone is not a completion
criterion.

Continue with :doc:`batches` for the backend arguments, returned records and
the distinction between a physical CPU completion and a small-batch CPU step.
