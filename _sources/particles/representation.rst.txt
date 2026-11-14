Particle representation and numerical conventions
=================================================

C8 exposes particle state through a stack-backed interface. The Kokkos
backend uses numerical records and field arrays with explicit units and a
common coordinate system. Both representations describe the same physical
state, but their energy fields and storage interfaces differ.

Native particle interfaces
--------------------------

``ParticleInterface<TStackIterator>`` in `VectorStack.hpp`_ provides
``getPID()``, ``getKineticEnergy()``, ``getPosition()``, ``getDirection()``
and ``getTime()``. These accessors retrieve fields at the particle's current
stack index. ``getEnergy()`` returns kinetic energy plus rest mass in C8's
energy units; velocity and momentum are derived quantities.

``VectorStackImpl`` owns separate vectors of particle codes, kinetic energies,
directions, positions and times. ``CombinedStack`` composes this storage with
additional fields. In `SetupStack.hpp`_, ``setup::Stack<TEnvironment>`` includes
geometry-node and weight data, with optional full history. The
``setup::HybridStack<TEnvironment>`` used by the integrated air-shower
application adds stable transport identities.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Native field or interface
     - Meaning
   * - ``Code`` / ``getPDG()``
     - The internal C8 particle enumeration and its conversion to a PDG code.
   * - ``HEPEnergyType``, ``TimeType``
     - Quantities with compile-time dimensions, defined through C8's units
       library in `PhysicalUnits.hpp`_.
   * - ``Point``, ``DirectionVector``
     - Geometry values associated with a ``CoordinateSystemPtr``. Component
       queries can transform them into another coordinate system.
   * - ``getNode()``
     - A pointer to the particle's logical environment node, supplied by the
       geometry-node stack extension.
   * - ``getWeight()``
     - Statistical weight supplied by the weight stack extension.
   * - ``getTransportIdentity()``
     - History, parent, generation and step identifiers supplied by the
       transport-identity extension.

The particle handle refers to stack storage; copying that handle is not an
independent particle snapshot. ``ParticleState`` is a value record and can
outlive the native entry from which it was exported.

Numerical field contract
------------------------

The record definition is included directly from the checked source:

.. literalinclude:: ../../../corsika/kokkos/data/ParticleState.hpp
   :language: cpp
   :start-at: struct alignas(16) ParticleState {
   :end-at:   };
   :lineno-match:

The following conventions apply at the conversion boundary:

.. list-table::
   :header-rows: 1
   :widths: 31 69

   * - Field
     - Contract
   * - ``pid_``
     - Signed PDG code, not the integer value of C8's ``Code`` enumeration.
   * - ``energyGeV_``
     - **Total** energy in GeV. Native stack insertion expects kinetic energy,
       so import subtracts the C8 rest mass expressed in energy units.
   * - ``positionM_[3]``
     - Position in metres in the coordinate system supplied to conversion.
   * - ``direction_[3]``
     - Dimensionless direction components in that same system. Conversion
       copies the transformed components without normalizing them.
   * - ``timeS_``, ``weight_``
     - Time in seconds and dimensionless statistical weight. Value export
       uses weight one if the source interface has no weight getter.
   * - ``environmentNode_``
     - Logical node index in the exported tree. ``-3`` is legacy/unassigned;
       ``-1`` denotes outside the tree. Ordinary native-stack restoration
       requires an index mapped to an actual native node.
   * - ``mediumId_``
     - Medium/model identifier used by the backend. Distinct logical nodes
       may share a medium; this field cannot replace ``environmentNode_``.
   * - ``historyId_``, ``parentHistoryId_``, ``generation_``, ``stepId_``
     - Transport identity and ancestry. Zero is an unassigned history ID;
       zero parent ID denotes no parent.
   * - ``reserved_``
     - Existing compatibility tags. The field is copied with the state and
       must not be treated as spare storage for unrelated features.

Velocity, momentum, mass, charge, density and random-generator state are not
stored in this record. Transport obtains the required quantities from
particle properties, shared environment/physics data and its random-stream
inputs.

Units and coordinate transformations
------------------------------------

``getCoordinates(coordinateSystem)`` transforms a position;
``getComponents(coordinateSystem)`` transforms a direction. Dividing the
typed results by ``meter``, ``second`` or ``1_GeV`` supplies the numerical
values recorded above. This conversion leaves the source particle's
coordinate-system association unchanged.

The air-shower application passes ``rootCS = env.getCoordinateSystem()`` at
its `geometry setup <https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/applications/c8_air_shower.cpp#L407>`_
and backend handoff. The record itself neither selects an Earth-fixed frame
nor stores a coordinate-system identifier. Its consumers must use the same
frame as the exported environment and observer data. Changing the common
frame requires changing those exports consistently.

These are transfer units, not a requirement that every formula use SI units.
A numerical physics routine can perform an explicit conversion to the units
required by its tables or equations. Such conversions belong to that routine's
contract and must not silently change the record's units.

Device compatibility
--------------------

Typed quantities and C++ classes are not inherently incompatible with GPU
execution. A device routine needs device-callable operations and memory
accessible from its execution space. In the current C8 representation,
particle access reaches host stack storage, positions and directions refer to
a shared coordinate-system graph, and geometry-node fields contain native
node pointers. Copying a particle handle does not relocate those dependencies
or compile their operations for the device.
The coordinate ownership is defined in
`BaseVector.hpp <https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/framework/geometry/BaseVector.hpp>`_
and
`CoordinateSystem.hpp <https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/framework/geometry/CoordinateSystem.hpp>`_.

The existing adapter resolves these dependencies on the host. Kernels receive
numerical fields, logical indices and shared backend data instead. In
``ParticleSoA`` and ``ParticleSoARawView``, ``load`` and ``store`` are marked
``KOKKOS_INLINE_FUNCTION`` for execution-space use. See Kokkos's
`function annotations <https://kokkos.org/kokkos-core-wiki/API/core/macros-special/host_device_macros.html>`_
and `View programming guide <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html>`_
for the underlying execution and memory rules.

Reusing more native types in device code would require auditing their
operations and dependencies, not merely adding a device annotation to the
particle accessor. That is a potential interface change, not the current
implementation.

.. _VectorStack.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/stack/VectorStack.hpp
.. _SetupStack.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/setup/SetupStack.hpp
.. _PhysicalUnits.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/framework/core/PhysicalUnits.hpp
