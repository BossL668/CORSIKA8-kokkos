Host and backend conversion
===========================

Two host-side interfaces convert particle data. The general helpers update
physical values in an existing record or particle. The production router's
transfer functions additionally handle identity, logical nodes and native
stack insertion. Both use the numerical conventions in
:doc:`representation`; neither performs a transport step.

Value-conversion interface
--------------------------

`ParticleConversion.hpp`_ declares the following template functions, implemented
in `ParticleConversion.inl`_:

.. list-table::
   :header-rows: 1
   :widths: 32 35 33

   * - Function
     - Inputs
     - Result and mutation
   * - ``export_particle_values``
     - Native particle, common ``CoordinateSystemPtr``, existing
       ``ParticleState&``.
     - Returns ``void``. Replaces PID, total energy, position, direction,
       time and weight; preserves all metadata fields.
   * - ``import_particle_values``
     - ``ParticleState const&``, the record's coordinate system, existing
       native particle reference.
     - Returns ``void``. Updates physical values, converting total to kinetic
       energy; does not insert an entry or restore identity/node metadata.

The export's unit and coordinate conversion is:

.. literalinclude:: ../../../corsika/detail/kokkos/data/ParticleConversion.inl
   :language: cpp
   :start-at: auto const position = particle.getPosition().getCoordinates(coordinateSystem);
   :end-at: state.weight_ = weight;
   :lineno-match:

Both helpers require a non-null coordinate system. Import checks the PDG
conversion, rejects total energy below the C8 rest mass, and rejects a
non-unit weight when the destination has no weight setter. Those checks
precede physical-value mutation. Finite values and appropriate directions
remain caller preconditions; this is not a complete particle validator.

Complete production transfer
----------------------------

The following functions belong to ``corsika::kokkos::particle_transfer`` in
`RouterParticleConversion.hpp`_ and `RouterParticleConversion.inl`_. They
implement the production transfer directly rather than calling the general
helpers above.

``to_device_state``
~~~~~~~~~~~~~~~~~~~

The function receives a native particle, the common coordinate system,
explicit ``historyId``, ``parentHistoryId``, ``generation`` and ``stepId``
values, and an ``EnvironmentTopology``. It returns a ``ParticleState`` by value
without changing or removing the source particle.

It exports physical values, copies the supplied identity and calls
``export_logical_node``. That function maps ``particle.getNode()`` to the
exported tree's index and rejects a missing or unmapped node. It does not
repeat a position-containment search.

``mediumId_`` is not populated by this conversion function. After checking
the environment state, ``EmRouter::stage`` obtains it from the selected node's
model. The end of that staging operation is:

.. literalinclude:: ../../../corsika/detail/kokkos/scheduling/EmRouter.inl
   :language: cpp
   :start-at: state.mediumId_ = environment_.getModel(state.environmentNode_)->mediumId_;
   :end-at: staged_.push_back(state);
   :lineno-match:

The host loop erases the native stack entry after staging succeeds. See
:doc:`../execution/hybrid-loop` for the routing and ownership transition.

``import_particle``
~~~~~~~~~~~~~~~~~~~

This function receives the destination stack, a numerical record, its common
coordinate system and the matching topology. It converts the PDG code,
checks the rest-mass boundary, constructs C8 geometry values, inserts a new
stack entry and restores identity, weight and the native node pointer. It
returns the newly inserted particle handle.

The final insertion and metadata restoration are:

.. literalinclude:: ../../../corsika/detail/kokkos/data/RouterParticleConversion.inl
   :language: cpp
   :start-at: auto particle = stack.addParticle(std::make_tuple(code, totalEnergy - mass, direction,
   :end-at: return particle;
   :lineno-match:

This operation requires a valid record and matching topology. It is not a
transactional validator: node restoration and some metadata checks occur
after insertion. The value-only helper's pre-mutation checks must not be
assumed to apply to this complete-transfer function.

Logical nodes and environment lifetime
--------------------------------------

``EnvironmentTopology`` maps native node addresses to exported indices and
back. Import restores the recorded logical node, even when the position
could geometrically belong to another nested node. Recomputing the node from
position alone would discard the tracking state carried across a boundary.

The native environment and topology must remain valid throughout transfer.
Node indices have meaning only within the corresponding exported tree;
``mediumId_`` cannot reconstruct a native node. Coordinator frontier import
also checks the exported environment fingerprint before accepting logical
indices; see :ref:`execution-frontier-import`.

Transport identity and step state
---------------------------------

`TransportIdentityStackExtension.inl`_ assigns each primary a history ID and
assigns each secondary a new ID, its parent's ID and the next generation.
Copy and swap operations move the identity with the particle. Import restores
the supplied identity and advances the allocator when necessary to avoid
reusing an imported ID.

The step counter has a specific convention:

.. literalinclude:: ../../../corsika/detail/stack/TransportIdentityStackExtension.inl
   :language: cpp
   :lines: 151-159
   :lineno-match:

``beginTransportStep()`` increments the stored counter and returns its previous
value. ``CpuParticleScheduler::acquireNext`` carries that returned value in
the scheduled particle; the router receives it explicitly. Replacing it with
a later call to ``getStepId()`` would change the exported step identifier.

History IDs are not stack indices, queue indices or thread identifiers. They
support ancestry, step tracking and random-stream selection across routing
and compaction. Preserving them is necessary for reproducibility, but does
not by itself guarantee identical histories after changing the scheduling
or random-stream policy.

Return-path responsibilities
----------------------------

The caller determines whether a returned state requires a native step,
completion of a selected interaction, a forced decay or cut processing.
These policies are described in :doc:`../execution/batches`; conversion
does not make that decision.

In particular, a compatibility-tagged muon-pair product below its rest mass
must receive the existing native-cut handling before ordinary stack import.
``reserved_`` does not relax the importer's rest-mass check. Conversion also
does not normalize a direction, draw a random number or repair a physical
state; those actions belong to their explicitly defined transport or
validation paths.

.. _ParticleConversion.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/data/ParticleConversion.hpp
.. _ParticleConversion.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/data/ParticleConversion.inl
.. _RouterParticleConversion.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/data/RouterParticleConversion.hpp
.. _RouterParticleConversion.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/data/RouterParticleConversion.inl
.. _TransportIdentityStackExtension.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/stack/TransportIdentityStackExtension.inl
