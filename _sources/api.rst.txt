API reference
=============

This reference presents the main integration interfaces directly from their
C++ headers. Doxygen extracts declarations and comments; Breathe renders them
in the guide. Function names, template parameters, argument types, defaults
and return types come from source rather than manually copied signatures.

.. note::

   Source baseline: `d5297c43
   <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.
   These are the implementation's current interfaces, not an upstream API
   proposal or a blanket compatibility guarantee. Available implementations
   depend on the configured targets and execution backends.

.. toctree::
   :maxdepth: 1

   api/application
   api/execution
   api/particles
   api/environment
   api/physics
   api/radio

Reference organization
----------------------

.. list-table::
   :header-rows: 1
   :widths: 29 38 33

   * - Interfaces
     - Main responsibility
     - Behavioral documentation
   * - :doc:`api/application`
     - Application bindings, execution selection and event configuration.
     - :doc:`application`
   * - :doc:`api/execution`
     - Runtime ownership, backend submission, run sessions and coordination.
     - :doc:`execution`, :doc:`backends/runtime`
   * - :doc:`api/particles`
     - Particle and step records, numerical conversion and storage.
     - :doc:`particles`
   * - :doc:`api/environment`
     - Native scene export, topology, resident data and numerical queries.
     - :doc:`environment`
   * - :doc:`api/physics`
     - Native table preparation, queries and particle transport stages.
     - :doc:`physics-data`, :doc:`transport`
   * - :doc:`api/radio`
     - Track input, observer/configuration data, projection and accumulation.
     - :doc:`radio`

The pages select the interfaces used to connect components and explain their
roles before the generated declarations. They do not reproduce every helper
or every diagnostic member. Source links provide the complete headers;
the component chapters describe call order, memory accessibility, units,
ownership and supported domains. The alphabetical index and site search
include the rendered API names.

Interpreting a declaration
--------------------------

A reference or pointer does not transfer ownership. Check the documented
lifetime of the referred object and whether the function is a host adapter,
a data transfer or a device-callable calculation. Similarly, a result record
may require checking a status before its numerical values are valid. A
header's presence does not establish that a particular optional target has
been built or validated on a device.

The API pages retain source comments and add links to the relevant component
contracts. Where a source declaration lacks detailed parameter documentation,
use those contracts and the implementation; no additional behavior is inferred
from the function name. In particular, validation batch wrappers and resident
production execution can expose similar calculations with different storage
and completion costs.

Building and updating the reference
-----------------------------------

The documentation build extracts the public ``corsika/kokkos`` headers without
compiling the application or its physics dependencies. It does not execute
the interfaces. Use the :download:`documentation build instructions
<BUILDING.rst>` for the tool requirements and commands.

The `generated-header manifest <api-inputs.json>`_ records the Doxygen version,
input-header hashes and extraction configuration hash for this build. Types
with their own reference entry link within the guide; additional implementation
types link to their source declarations where the reference can be resolved.

When adding or changing an interface, update its declaration-level contract
and the relevant API page together. Regenerate the reference and review
overloads, template signatures, public member visibility and links. The
source extraction and the semantic guide answer different questions; both
must remain consistent with the implementation.
