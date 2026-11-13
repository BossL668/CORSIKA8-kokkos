Batches, checkpoints and CPU returns
====================================

``EmRouter::advanceOneWavefrontAndReturn(stack)`` connects host staging to the
resident EM backend. It returns a count of CPU returns or node-refresh requests
for ``HybridCascade``. Device steps and generated particles are accounted for
separately in result records and statistics.

Dispatch thresholds and capacity
--------------------------------

The ordinary router can first return a small staged batch to the scalar stack
for further expansion. Otherwise, it limits the host input to the backend's
advertised capacity, keeps excess input in ``staged_`` and separates photons
from charged leptons. Resident queues can contribute additional input without
a host round trip.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Control
     - Meaning in this implementation
   * - ``minimumBatchSize()``
     - A particle-count threshold for dispatch and small-batch checkpoints.
       The resident loops check the threshold after at least one wavefront.
       Forced flushes and pending work can also run below the threshold.
   * - Maximum resident input / photon / lepton sizes
     - Capacity limits used to split host input and reserve workspace. An input
       that exceeds the limit remains queued for later processing.
   * - ``maximumWavefronts`` and history-ID limits
     - Bounds on a resident call before it returns a checkpoint. In the ordinary
       router the photon call requests up to 16 wavefronts; the lepton call
       requests up to 1024 and also supplies a reserved history-ID range.
   * - Static frontier import block
     - Limits roots imported from a coordinator-produced file into an empty
       local stack. It is independent of the resident batch capacity; see
       :ref:`execution-frontier-import`.

These counts do not specify CUDA/HIP threads per block. Execution policies
inside the transport kernels determine how particle indices are mapped to
threads.

Calling the backend
-------------------

The photon call in ``EmRouter.inl`` is:

.. literalinclude:: ../../../corsika/detail/kokkos/scheduling/EmRouter.inl
   :language: cpp
   :start-at: auto result = backend_.runResidentPhotonCascadeForValidation(
   :end-at: photons, firstHistory, ResidentPhotonWavefronts, backend_.minimumBatchSize());
   :lineno-match:

The charged-lepton call additionally supplies an exclusive history-ID limit:

.. literalinclude:: ../../../corsika/detail/kokkos/scheduling/EmRouter.inl
   :language: cpp
   :start-at: auto result = backend_.runResidentLeptonCascadeForValidation(
   :end-at: backend_.minimumBatchSize());
   :lineno-match:

The ``ForValidation`` names are retained compatibility spellings used by the
production router. ``IEmBackend.inl`` forwards them to ``runPhotonWavefront``
and ``runLeptonWavefront``. They are host calls, not virtual dispatch inside
each particle's GPU kernel.

``EmBackend.cpp`` forwards to the selected backend instance.
``BackendInstanceImplementation<TExecutionSpace>`` combines the supplied
``std::vector<ParticleState>`` with pending resident input, supplies shared
physics views, workspaces and accumulators, and calls ``run_photon_cascade``
or ``run_lepton_cascade``. These templates use the selected Kokkos execution
space, including OpenMP, CUDA or HIP in a matching build.

.. figure:: ../_static/execution-backend-flow.svg
   :alt: Router inputs pass through the host backend interface into resident photon or lepton loops, then returned records are consumed by the router.
   :width: 100%
   :target: ../_static/execution-backend-flow.svg

   The scheduling-to-transport boundary. Pending queues and shared workspaces
   can remain in the execution space between calls. Open the SVG for source links.
   :download:`SVG <../_static/execution-backend-flow.svg>` ·
   :download:`Editable DOT source <../_static/execution-backend-flow.dot>`.

Returned data and its consumers
-------------------------------

The calls return ``PhotonCascadeResult`` or ``LeptonCascadeResult`` by value.
``consumeResidentPhoton`` and ``consumeResidentLepton`` handle these records:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Returned data
     - Consumer / effect
   * - Remaining photons or leptons; generated particles
     - Re-enter host staging when returned. With resident cross-species queues
       enabled, eligible generated particles stay in device-accessible storage
       instead; they must not also be imported as returned children.
   * - ``fallbackEvents_``
     - ``returnFallbacks(...)`` either queues a specified CPU completion or
       returns the particle for a scalar step, subject to the configured error
       policy.
   * - ``cpuSpillParticles_``
     - ``returnMemorySpills(...)`` restores unfinished states to the native
       stack. Capacity pressure must not silently remove particles.
   * - Lepton ``decayCandidates_``
     - ``returnForcedDecays(...)`` marks the corresponding scalar step to
       execute a decay.
   * - Step, final-state, projected-step and observation records
     - Router recording functions pass the required contributions to
       ``CorsikaOutputSink``. Device-owned profile contributions are accumulated
       on their device path and must not be counted again on host replay.
   * - Checkpoint flags and counters
     - Record small batches, workspace limits or exhausted history-ID ranges.
       A resident call can stop while the shower still has work to do.

Three different reasons to return to the CPU
--------------------------------------------

**Small-batch expansion** is a scheduling choice. If there is no pending
resident work and the staging vector is below threshold, the router can
import it back to the C8 stack. A history/step marker makes ``canRoute`` reject
that next routing attempt so the scalar stepper actually advances it. A bounded
number of expansion rounds and a no-progress check can force a device flush.

**Specified reaction completion** preserves an already selected interaction.
The ordinary endpoint stores requests accepted by
``ProposalCpuFallbackHandler::canHandle(...)`` in ``deferredFallbackEvents_``.
Once host staging and the resident front drain, ``flushDeferredFallbacks`` calls
``handle(stack, event)`` to complete these vertices on the CPU. The request
contains the particle and selected interaction information. It is not a request
to resample an unrelated step. The new EM products can then return to the
device path; a dedicated flag avoids another small-batch expansion cycle.

**A generic scalar return, decay or memory spill** has its own handling and
history/step markers. Unexpected numerical or capability returns can be rejected
by ``failOnUnexpectedFallback`` rather than silently retried. The reaction-level
capabilities are defined in the physics layer; a low batch count alone does not
mean that a reaction is unsupported.

For maintenance, inspect both the producer of a returned state and its consumer.
Changing only the kernel or only the router can otherwise leave a state
unconsumed, execute a selected vertex twice or double-count an output record.
