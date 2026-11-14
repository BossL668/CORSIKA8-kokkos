Particle records and conversion
===============================

These declarations define the numerical particle state, storage views and
native C8 transfer boundary. Signatures and member descriptions are generated
from the public headers at the API reference's source baseline. The transfer
operations are host functions; storage accessors marked for device use still
require accessible allocations and valid indices.

The behavioral descriptions are in :doc:`../particles/representation`,
:doc:`../particles/conversion` and :doc:`../particles/storage`.

Value records
-------------

``ParticleState`` stores total energy in GeV, positions in metres, time in
seconds and dimensionless directions in the selected common coordinate
system. Its history and logical-node fields are part of the transport state.
``StepRecord`` carries an output segment, with unweighted deposits; it does
not update a profile or apply the statistical weight by being copied.

Sources: `ParticleState.hpp`_ and `StepRecord.hpp`_.

.. doxygenstruct:: corsika::kokkos::ParticleState
   :members:
   :undoc-members:

.. doxygenstruct:: corsika::kokkos::StepRecord
   :members:
   :undoc-members:

Field storage and borrowed access
---------------------------------

``ParticleSoA`` owns field arrays in the requested memory space. Its raw view
borrows the same allocations and preserves vector-field strides. A raw view
neither transfers data nor retains the storage owner. Complete outstanding
uses before releasing or replacing that storage. Queue progression is a
separate responsibility described in :doc:`../particles/storage`.

Source: `ParticleSoA.hpp`_.

.. doxygenstruct:: corsika::kokkos::ParticleSoA
   :members:
   :undoc-members:

.. doxygenstruct:: corsika::kokkos::ParticleSoARawView
   :members:
   :undoc-members:

Physical-value conversion
-------------------------

These host functions convert values into or out of an existing record or
native particle. They do not allocate transport identities or establish the
logical-node mapping. The production transfer operations below cover those
additional responsibilities.

Source: `ParticleConversion.hpp`_.

.. doxygenfunction:: corsika::kokkos::export_particle_values

.. doxygenfunction:: corsika::kokkos::import_particle_values

Production state transfer
-------------------------

These templates use the topology associated with the current snapshot to
retain the native particle's logical node across the transfer. That topology
and its native scene must remain alive and unchanged. ``to_device_state``
constructs a numerical record; ``import_particle`` inserts a native stack
particle. Importing the logical node alone does not import physical values.

Source: `RouterParticleConversion.hpp`_. The complete identity, cut and stack
insertion contract is documented in :doc:`../particles/conversion`.

.. doxygenfunction:: corsika::kokkos::particle_transfer::to_device_state

.. doxygenfunction:: corsika::kokkos::particle_transfer::import_particle

.. doxygenfunction:: corsika::kokkos::particle_transfer::export_logical_node

.. doxygenfunction:: corsika::kokkos::particle_transfer::import_logical_node

.. _ParticleState.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/data/ParticleState.hpp
.. _StepRecord.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/data/StepRecord.hpp
.. _ParticleSoA.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/data/ParticleSoA.hpp
.. _ParticleConversion.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/data/ParticleConversion.hpp
.. _RouterParticleConversion.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/data/RouterParticleConversion.hpp
