Runtime and execution interfaces
================================

These interfaces are called on the host. They select or own execution
resources, submit numerical work and coordinate native completion. Their
presence in a public header does not make their methods device-callable.
Declarations are generated from the reference headers identified in
:doc:`../api`.

The lifecycle contracts are in :doc:`../backends/runtime`; queue and return
semantics are in :doc:`../execution/batches` and
:doc:`../execution/hybrid-loop`. Runtime configuration, particle scheduling
and process coordination are separate responsibilities.

Execution configuration and ownership
-------------------------------------

``RuntimeConfig`` selects a compiled backend, visible device, host threads
and tuning policy. Cooperative fields apply only to the explicitly supported
cooperative path; they do not independently enable a hybrid application.
``BackendRuntime`` owns the production backend's process lifetime, with a
restricted cooperative lease. The generic ``Runtime`` below has a different
sharing/borrowing contract and is not a replacement owner for
``BackendRuntime``.

Sources: `RuntimeConfig.hpp`_ and `BackendRuntime.hpp`_.

.. doxygenstruct:: corsika::kokkos::RuntimeConfig
   :members:
   :undoc-members:

.. doxygenclass:: corsika::kokkos::BackendRuntime
   :members:
   :undoc-members:

The generic runtime manages or borrows the build-selected default execution
space. Views and execution-space objects must be released before its final
owning reference. Its accessors report configuration; they do not advance
particles. Source: `Runtime.hpp`_.

.. doxygenclass:: corsika::kokkos::Runtime
   :members:
   :undoc-members:

Backend submission interface
----------------------------

``IEmBackend`` is the router's host-facing contract for resident wavefronts,
capacities, pending work, CPU returns and downloaded observables. A wavefront
call can advance multiple physical steps; it is not a one-particle kernel
launch. Input ownership, history allocation and completion semantics are
specified in :doc:`../execution/batches`.

The independent-subshower methods are opt-in host coordination capabilities.
They must not be interpreted as a persistent per-particle task queue inside
one GPU. See :doc:`../decisions/execution-models`.

Source: `IEmBackend.hpp`_.

.. doxygenclass:: corsika::kokkos::IEmBackend
   :members:
   :undoc-members:

``EmBackend`` owns the selected implementation. The members below cover its
initialization, validation entry points and runtime metadata; its production
submission operations implement ``IEmBackend`` above. Functions ending in
``ForValidation`` expose individual stages for controlled comparisons and
are not the full production scheduler.

Source: `EmBackend.hpp`_.

.. doxygenclass:: corsika::kokkos::EmBackend
   :members: EmBackend, initialize, selectInteractionsForValidation, transportPhotonsForValidation, transportLeptonsForValidation, selectLeptonVerticesForValidation, reselectLeptonInteractionsAtVertexForValidation, generatePhotonFinalStatesForValidation, generateLeptonFinalStatesForValidation, projectRadioForValidation, runtimeInfo
   :undoc-members:

Reusable production session
---------------------------

``EmRunSession`` exports and uploads native model data on initial use, then
resets event state for compatible subsequent showers. ``SessionBeginResult``
borrows the backend; the session owns it. Reuse is not general model/scene
reconfiguration: changed environment data is rejected and changed calculators
are not re-exported on the reuse branch. Follow the fresh-process and
runtime-ownership guidance in :doc:`../development/extensions`.

Source: `EmRunSession.hpp`_.

.. doxygenclass:: corsika::kokkos::EmRunSession
   :members:
   :undoc-members:

.. doxygenstruct:: corsika::kokkos::SessionBeginResult
   :members:
   :undoc-members:

Hybrid cascade coordination
---------------------------

``HybridCascade`` coordinates scalar C8 work with an optional EM router and
hadronic worker pool. The referenced environment, sequence, stack, output,
router and pool remain caller-owned. Deferred states stay suspended until
their corresponding results are committed. The scheduling and completion
rules are in :doc:`../execution/hybrid-loop`.

Source: `HybridCascade.hpp`_. The EM router itself is declared in
`EmRouter.hpp`_ and implemented in the matching ``.inl``; its data exchange
contract is described in :doc:`../execution/batches`.

.. doxygenclass:: corsika::kokkos::HybridCascade
   :members:
   :undoc-members:

Multi-process coordination
--------------------------

The coordinator operates before a local Kokkos context is created. It starts
worker binaries, assigns static frontiers and validates/merges closed outputs.
``CoordinatorOptions`` is the library configuration record; command-line
parsing can resolve different defaults before invoking ``run``. The
coordinator's ``run`` function returns no scientific result object: its
results are the output files and completion metadata described in
:doc:`../execution/coordinator` and :doc:`../usage/outputs`.

Source: `AirShowerCoordinator.hpp`_.

.. doxygenstruct:: corsika::kokkos::coordinator::CoordinatorOptions
   :members:
   :undoc-members:

.. doxygenfunction:: corsika::kokkos::coordinator::run

.. _RuntimeConfig.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/runtime/RuntimeConfig.hpp
.. _BackendRuntime.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/runtime/BackendRuntime.hpp
.. _Runtime.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/runtime/Runtime.hpp
.. _IEmBackend.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/execution/IEmBackend.hpp
.. _EmBackend.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/execution/EmBackend.hpp
.. _EmRunSession.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/EmRunSession.hpp
.. _HybridCascade.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/scheduling/HybridCascade.hpp
.. _EmRouter.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/scheduling/EmRouter.hpp
.. _AirShowerCoordinator.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/scheduling/AirShowerCoordinator.hpp
