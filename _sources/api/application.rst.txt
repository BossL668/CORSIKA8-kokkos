Application interfaces
======================

These host interfaces connect application-owned C8 objects to direct or
coordinated execution. Declarations and public members below are generated
from the headers at the reference baseline recorded in :doc:`../api`.
They describe the current integration API, not an upstream stability guarantee.

For the call sequence and argument meanings, read :doc:`../application/setup`,
:doc:`../application/data-contracts` and
:doc:`../application/event-execution`. The configuration records on this page
are C++ values; their default initializers are not a substitute for the
application's resolved command-line defaults.

Execution selection
-------------------

Selection occurs before model construction and Kokkos initialization. Direct
execution includes native C8, OpenMP and a single GPU. Coordinated execution
uses the separate process coordinator. The returned selection retains device
configuration but does not own a running device session.

Source: `ExecutionSelection.hpp`_.

.. doxygenclass:: corsika::kokkos::application::ExecutionSelection
   :members:
   :undoc-members:

.. doxygenfunction:: corsika::kokkos::application::select_execution

Application preparation and binding
-----------------------------------

``CascadeExtension`` registers options and creates run bindings around native
scene, model and output objects. The listed members cover the current
application path. The earlier positional ``bind`` interface remains in the
source declaration but is not used by the current air-shower entry point.
For scene-specific radio factories, see the full header and
:doc:`../radio/propagation`.

Source: `CascadeExtensionOptions.hpp`_.

.. doxygenclass:: corsika::kokkos::application::CascadeExtension
   :members: CascadeExtension, prepare, assignWorkerInput, configure, registerOutput, interaction, interactionCounter, energyLossWriter, longitudinalWriter, geometry, execution
   :undoc-members:

Run and event lifetimes
-----------------------

``CascadeModule`` retains backend hooks across events. It borrows the native
objects supplied during construction; those owners must outlive the module.
``shower`` creates event state, and ``CascadeEvent::cascade`` returns the
adapter on which the application calls ``EAS.run()``. The original native
``Cascade`` is constructed inside the native execution branch.

``CascadeEvent`` also controls input, output and diagnostic lifecycle. Its
constructor or destructor alone does not establish that a shower completed;
use the completion contracts in :doc:`../usage/outputs`.

Source: `CascadeExtension.hpp`_.

.. doxygenclass:: corsika::kokkos::application::CascadeModule
   :members:
   :undoc-members:

.. doxygenclass:: corsika::kokkos::application::CascadeEvent
   :members:
   :undoc-members:

Borrowed application context
----------------------------

``CascadeGeometry`` holds references to the scene and typed geometry objects,
alongside copied numerical configuration. ``CascadeContext`` groups geometry,
outputs, processes, execution and diagnostic bindings. Copying these bundles
does not transfer ownership of the native stack, models or writers. The native
objects and any borrowed diagnostic snapshots must remain valid during use.

Source: `CascadeContext.hpp`_. The component bundles and their argument map
are described in :doc:`../application/data-contracts`.

.. doxygenstruct:: corsika::kokkos::application::CascadeGeometry
   :members:
   :undoc-members:

.. doxygenstruct:: corsika::kokkos::application::CascadeContext
   :members:
   :undoc-members:

Run metadata and event configuration
------------------------------------

``RunEnvironmentConfig`` records site, field, observation and scene metadata.
It does not replace the native environment tree. ``CascadeEventConfig`` carries
physics/run settings used to configure events, including typed energies,
cuts, thinning and primary identity. Native typed quantities remain on this
host boundary; numerical conversion happens during preparation and transfer.

Sources: `RunEnvironmentConfig.hpp`_ and `CascadeEventConfig.hpp`_.

.. doxygenstruct:: corsika::kokkos::application::RunEnvironmentConfig
   :members:
   :undoc-members:

.. doxygenstruct:: corsika::kokkos::application::CascadeEventConfig
   :members:
   :undoc-members:

Prepared host inputs
--------------------

``PreparedCascade`` owns its exported environment through shared immutable
storage; its numerical view borrows that storage. The native environment is
still required for host tracking, CPU completion and output conversion.
Prepared data is not a device upload, nor is it refreshed by later mutations
of native objects. See :doc:`../environment/export` and
:doc:`../development/extensions` for preparation and reuse requirements.

Source: `PreparedCascade.hpp`_.

.. doxygenstruct:: corsika::kokkos::application::PreparedCascade
   :members:
   :undoc-members:

.. _ExecutionSelection.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/ExecutionSelection.hpp
.. _CascadeExtensionOptions.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeExtensionOptions.hpp
.. _CascadeExtension.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeExtension.hpp
.. _CascadeContext.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeContext.hpp
.. _RunEnvironmentConfig.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/RunEnvironmentConfig.hpp
.. _CascadeEventConfig.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeEventConfig.hpp
.. _PreparedCascade.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/PreparedCascade.hpp
