Application integration
=======================

The air-shower application constructs the C8 environment, physics models,
observers and process sequence. The application integration layer binds these
objects to the selected execution path, invoked through ``EAS.run()``. This
chapter documents the component responsibilities, interface data contracts
and control flow from ``main`` to cascade execution.

.. note::

   Source baseline: `d5297c43
   <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.
   The following pages describe the current implementation. Code excerpts are
   included directly from repository files, with their original identifiers and
   source line numbers. Paths are relative to the project root.

.. _application-source-files:
.. _source-file-map:

Source organization
-------------------

The following files implement the application integration layer and its
dependencies. Paths are relative to the repository root; source links refer
to the revision specified above.

The public headers in ``corsika/kokkos/application/`` declare the application
interfaces and data types. Template implementations are under
``corsika/detail/kokkos/application/``, while compiled, non-template
implementations are under ``src/kokkos/application/``.

.. parsed-literal::

   applications/
   └── `c8_air_shower.cpp`_                 main(), C8 setup and event loop

   corsika/kokkos/application/
   ├── `ExecutionSelection.hpp`_            direct/coordinated execution interface
   └── cascade/
       ├── `CascadeExtensionOptions.hpp`_   CascadeExtension declaration
       ├── `CascadeExtension.hpp`_          CascadeModule and CascadeEvent declarations
       ├── `CascadeContext.hpp`_            geometry and context reference bundles
       ├── `CascadeEventConfig.hpp`_        run defaults and per-event settings
       ├── `RunEnvironmentConfig.hpp`_      site and run metadata
       ├── `PreparedCascade.hpp`_           prepared snapshot and backend settings
       ├── `SequenceModels.hpp`_            borrowed model references
       └── `SequenceOutputs.hpp`_           borrowed observer references

   corsika/detail/kokkos/application/cascade/
   ├── `CascadeExtension.inl`_              geometry(), execution(), shower(), cascade()
   ├── `CascadeSupport.inl`_                geometry and defaults factories
   ├── `CascadeAdapter.inl`_                run() and forced-primary flags
   ├── `CascadeDriver.inl`_                 prefix/native/accelerated path selection
   ├── `CascadeApplication.inl`_            event context and accelerated dispatch
   ├── `CascadePreparation.inl`_            prepare_cascade(): host data preparation
   ├── `CascadeRunner.inl`_                 run_cascade(): start the hybrid cascade
   ├── `SequenceModels.inl`_                complete_models(): recover event processes
   └── `SequenceOutputs.inl`_               complete_outputs(): recover event writers

   src/kokkos/application/cascade/
   ├── `CascadeExtension.cpp`_              option registration, prepare(), configure()
   ├── `CascadeSupport.cpp`_                select_execution() and runtime setup
   └── `CascadeRunSession.cpp`_             run-session state and output records

``CascadeExtension.hpp`` is the header included by ``main``. It includes
``CascadeExtensionOptions.hpp``, where the ``CascadeExtension`` class itself is
declared, and defines the ``CascadeModule`` and ``CascadeEvent`` templates.
There are no separate ``CascadeModule.hpp`` or ``CascadeEvent.hpp`` files in
this baseline.

Dependencies outside the application integration layer:

* `ProcessSelection.hpp <https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/framework/process/ProcessSelection.hpp>`_ in
  ``corsika/framework/process/`` provides ``select_process(...)``.
* `Cascade.hpp <https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/framework/core/Cascade.hpp>`_ in
  ``corsika/framework/core/`` declares the native C8 cascade.
* `EmRunSession.cpp <https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmRunSession.cpp>`_ in
  ``src/kokkos/em/`` implements ``beginProposalNative(...)``.
* `HybridCascadeRunner.inl <https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/HybridCascadeRunner.inl>`_ in
  ``corsika/detail/kokkos/scheduling/`` implements ``run_hybrid_cascade(...)``.

.. _c8_air_shower.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/applications/c8_air_shower.cpp
.. _ExecutionSelection.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/ExecutionSelection.hpp
.. _CascadeExtensionOptions.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeExtensionOptions.hpp
.. _CascadeExtension.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeExtension.hpp
.. _CascadeContext.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeContext.hpp
.. _CascadeEventConfig.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeEventConfig.hpp
.. _RunEnvironmentConfig.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/RunEnvironmentConfig.hpp
.. _PreparedCascade.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/PreparedCascade.hpp
.. _SequenceModels.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/SequenceModels.hpp
.. _SequenceOutputs.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/SequenceOutputs.hpp
.. _CascadeExtension.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeExtension.inl
.. _CascadeSupport.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeSupport.inl
.. _CascadeAdapter.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeAdapter.inl
.. _CascadeDriver.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeDriver.inl
.. _CascadeApplication.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeApplication.inl
.. _CascadePreparation.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadePreparation.inl
.. _CascadeRunner.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeRunner.inl
.. _SequenceModels.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/SequenceModels.inl
.. _SequenceOutputs.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/SequenceOutputs.inl
.. _CascadeExtension.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/cascade/CascadeExtension.cpp
.. _CascadeSupport.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/cascade/CascadeSupport.cpp
.. _CascadeRunSession.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/cascade/CascadeRunSession.cpp

.. _read-the-interface-in-execution-order:

Application lifecycle
---------------------

.. toctree::
   :maxdepth: 1

   application/setup
   application/data-contracts
   application/event-execution

These sections document execution selection and configuration, input and
lifetime contracts, and per-event execution on the native and accelerated paths.

.. figure:: _static/application-calls.svg
   :alt: Direct application execution binds C8 objects into CascadeModule, creates an event and CascadeAdapter, then calls EAS.run to update the stack and outputs.
   :width: 100%

   The direct application path. Multi-GPU and parallel-prefix coordination is
   selected before this setup; each worker then uses the application interfaces
   for its assigned work. :download:`Diagram source <_static/application-calls.dot>`.

.. _names-that-refer-to-different-responsibilities:

Interface components and responsibilities
-----------------------------------------

The identifiers below refer to distinct objects in ``main``. Their types
separate execution selection, application configuration, persistent bindings
and per-event state.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Identifier in ``main``
     - Type and responsibility
   * - ``executionMode``
     - ``ExecutionSelection``: selects direct execution or the
       process coordinator before model setup.
   * - ``backend``
     - ``CascadeExtension``: registers command-line options, configures library
       resources and constructs the application bindings.
   * - ``geometry`` / ``defaults``
     - ``CascadeGeometry<...>``: references to scene objects.
       ``CascadeEventConfig``: run defaults used to configure individual events.
   * - ``execution``
     - ``CascadeModule<...>``: retains the driver and application bindings
       across events.
   * - ``event``
     - ``CascadeEvent<...>``: manages the input and diagnostic state of one shower.
   * - ``EAS``
     - ``CascadeAdapter<...>``: returned by ``event.cascade(...)``. Its
       ``run()`` call executes the selected path; the native branch constructs
       a C8 ``Cascade`` internally.

.. _what-crosses-the-application-interface:

Interface data contracts
------------------------

The interface accepts native C8 objects and typed quantities, including the
environment, coordinate system, particle stack, process sequence, physics models
and output writers. Reference bundles provide access to application-owned
objects; configuration structures carry run defaults and per-event settings.
The accelerated execution path subsequently exports numerical environment
snapshots and model tables and transfers the required data to device storage.

Execution functions generally return ``void`` and record their results through
updates to the particle stack, run-session state and output writers.
:doc:`application/data-contracts` specifies the input types, returned binding
objects and lifetime requirements; :doc:`application/event-execution` documents
state updates and output handling during execution.

.. _maintenance-entry-points:

Implementation and maintenance references
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Maintenance scope
     - Implementation references
   * - Direct versus coordinated execution
     - ``ExecutionSelection.hpp`` and ``CascadeSupport.cpp``;
       see :doc:`application/setup`.
   * - New application inputs or lifetimes
     - ``CascadeExtensionOptions.hpp``, ``CascadeContext.hpp`` and
       ``CascadeSupport.inl``; see :doc:`application/data-contracts`.
   * - Processes recovered from the event sequence
     - ``SequenceModels.inl``, ``SequenceOutputs.inl`` and
       ``ProcessSelection.hpp``; see :doc:`application/event-execution`.
   * - Starting or reusing device execution
     - ``CascadeRunner.inl`` and ``EmRunSession.cpp``;
       see :ref:`application-backend-start`.

The library retains the earlier ``bind(...)`` and ``dispatch_cascade(...)``
helpers. The current air-shower entry point uses ``select_execution(...)``,
``geometry(...)`` and ``execution(...)``; the earlier helpers are outside this
application's call path.

.. _application-source-flow:
.. _follow-the-data-from-main-to-the-scheduling-module:

Control and data flow
---------------------

The diagram maps calls in ``main`` to implementation files and the scheduling
module. It shows the ``kokkos-proposal`` event path and the native ``proposal``
branch. Process coordination is selected before this path; a dedicated prefix
task can return earlier through ``input.runPrefix(...)``. The EGS4 session uses
the separate execution branch described in
:doc:`application/event-execution`.

.. figure:: _static/application-source-flow.svg
   :alt: File-level flow from c8_air_shower.cpp through CascadeExtension, CascadeDriver, CascadeApplication and CascadeRunner to HybridCascadeRunner and HybridCascade, with environment preparation and backend initialization callbacks.
   :width: 100%
   :target: _static/application-source-flow.svg

   Control flow proceeds from top to bottom. Solid arrows represent function
   calls; dashed arrows represent returned objects. Nodes identify implementation
   files and functions or callbacks. Green nodes indicate the scheduling-module
   boundary. Source links are available in the full-size SVG.
   :download:`SVG <_static/application-source-flow.svg>` /
   :download:`Editable DOT source <_static/application-source-flow.dot>`.

Except where a full path is shown, application ``.inl`` files in this figure
are under ``corsika/detail/kokkos/application/cascade/``. Both green boxes are
under ``corsika/detail/kokkos/scheduling/``.

The application passes ``begin_backend`` as a callback to
``run_hybrid_cascade(...)``. That function invokes it to obtain ``EmBackend&``
before constructing the router and starting ``HybridCascade::run()``. The
application therefore supplies the inputs and initialization callback; the
scheduling module controls this execution order. The :doc:`execution` chapter
documents the hybrid loop, batch contracts and process coordination.
