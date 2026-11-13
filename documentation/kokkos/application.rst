Application integration
=======================

The air-shower application creates the C8 environment, models, observers and
process sequence. The application interface binds those objects to an execution
path and exposes the event through ``EAS.run()``. This chapter follows that
connection from the call in ``main`` to the code that starts a cascade.

.. note::

   Source baseline: `d5297c43
   <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.
   The following pages describe the current implementation. Code excerpts are
   included directly from repository files, with their original identifiers and
   source line numbers. Paths are relative to the project root.

Read the interface in execution order
-------------------------------------

.. toctree::
   :maxdepth: 1

   application/setup
   application/data-contracts
   application/event-execution

The first page covers process selection and setup. The second follows the
arguments into their receiving objects. The third follows an individual event,
including the native and accelerated branches and the output they update.

.. figure:: _static/application-calls.svg
   :alt: Direct application execution binds C8 objects into CascadeModule, creates an event and CascadeAdapter, then calls EAS.run to update the stack and outputs.
   :width: 100%

   The direct application path. Multi-GPU and parallel-prefix coordination is
   selected before this setup; each worker then uses the application interfaces
   for its assigned work. :download:`Diagram source <_static/application-calls.dot>`.

Names that refer to different responsibilities
----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Name in ``main``
     - Concrete role
   * - ``executionMode``
     - An ``ExecutionSelection`` value. It selects direct execution or the
       process coordinator before model setup.
   * - ``backend``
     - A ``CascadeExtension`` object. It registers options, configures library
       resources and creates the application bindings. It is not a GPU kernel
       or an ``EmBackend`` instance.
   * - ``geometry`` / ``defaults``
     - A ``CascadeGeometry<...>`` bundle of references to scene objects and a
       ``CascadeEventConfig`` value containing run defaults.
   * - ``execution``
     - A ``CascadeModule<...>`` that retains the driver and bindings across
       events. It is distinct from ``executionMode``.
   * - ``event``
     - A ``CascadeEvent<...>`` for one shower, including its input and diagnostic
       bookkeeping.
   * - ``EAS``
     - A ``CascadeAdapter<...>`` returned by ``event.cascade(...)``. Its
       ``run()`` call executes the selected path; the native branch constructs
       a C8 ``Cascade`` internally.

What crosses the application interface
--------------------------------------

The application supplies native C++ objects and typed quantities: its environment,
coordinate system, particle stack, process sequence, model instances and output
writers. The binding structs mainly hold references to those objects. Numerical
environment snapshots, model-table export and device uploads occur later inside
the accelerated execution path.

Most execution functions return ``void`` and write through the supplied stack,
writers or run-session objects. An event is therefore not returned as one new
"shower result" value. The detailed contracts below distinguish returned objects
from state changes and output records.

Maintenance entry points
------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Change under consideration
     - Start reading here
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

The older ``bind(...)`` and ``dispatch_cascade(...)`` helpers remain in the
library. The air-shower ``main`` described here uses ``select_execution(...)``,
``geometry(...)`` and ``execution(...)`` instead. Their presence in the source
should not be mistaken for additional calls made by this entry point.
