Scene construction and snapshot export
======================================

The native environment is the source of scene information for both execution
paths. Export changes its representation, not the application's ownership of
the scene. The resulting arrays contain topology, shape parameters and material
parameters; they do not contain native object addresses or virtual-function
tables.

Native environment construction
-------------------------------

``Environment<T>`` owns the universe node and provides ``createNode`` for
constructing volumes. Each ``VolumeTreeNode`` owns its volume and child nodes,
shares its model object through a smart pointer, and holds non-owning parent
and exclusion references. The interfaces are defined in `Environment.hpp`_
and `VolumeTreeNode.hpp`_.

``addChild`` establishes ownership and the child's parent. In contrast,
``excludeOverlapWith`` records a directed, non-owning relationship: a volume
can be excluded from another without becoming its child. Reciprocal exclusions
must be specified explicitly. The order of children and exclusions matters
for point location and boundary selection.

A node can have no model, and several nodes can share one model object.
Neither export nor the numerical query interface implicitly inherits model
properties from a parent. A model-free universe is therefore valid topology,
but is not itself a transport medium.

Application handoff
-------------------

``CascadeExtension::geometry(env, coordinates, ...)`` retains the environment
reference and coordinate-system handle in the cascade inputs. When the
accelerated run is prepared, ``prepare_cascade`` receives those native objects.
The caller must keep them alive throughout the run. See
:doc:`../application/data-contracts` for the surrounding application interface.

The environment preparation helper constructs a snapshot from the complete
universe and the supplied common coordinate system:

.. literalinclude:: ../../../corsika/detail/kokkos/environment/EnvironmentPreparation.inl
   :language: cpp
   :lines: 11-17
   :lineno-match:

The production caller then checks the scene policy and binds a host view:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/CascadePreparation.inl
   :language: cpp
   :lines: 42-53
   :lineno-match:

Despite its name, ``prepared.environmentStorage_`` here owns a **host snapshot**
through a shared pointer. It is distinct from
``EnvironmentStorage<TExecutionSpace>``, which owns the backend arrays.
``bind_environment_data`` only exposes host addresses and counts; it does not
upload memory.

Topology and identity
---------------------

``EnvironmentTopology<TModelProperties>`` traverses the owning tree in preorder
and assigns integer indices. Root index ``0`` has parent ``-1``. Child and
exclusion lists retain their native ordering, including explicitly repeated
references. Every exclusion must refer to a node in the same ownership tree.
Malformed parent links, duplicate ownership and external exclusions are
rejected during construction.

Model indices identify native model objects. Shared objects receive one index;
distinct objects with equal values remain distinct. ``modelIndex_ == -1`` means
that the node has no model. The topology also keeps host-only mappings:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Interface
     - Result and contract
   * - ``getNodeIndex(nativeNode)``
     - Snapshot index for the supplied native node; an absent node maps to
       ``-1``. An unknown node is rejected.
   * - ``getNativeNode(index)``
     - Native node reference for an exported index; ``-1`` denotes no node.
       Other invalid indices are rejected.

Indices are local to one snapshot. They are not persistent identifiers across
tree edits. The mappings borrow native objects, so those objects must outlive
uses of the mappings. Particle export/import must use the matching topology.

Numerical records
-----------------

``EnvironmentSnapshot`` combines that topology with shape and model records.
``export_shape`` converts each volume to the common coordinates;
``export_model`` reads the native model's export interfaces. Both operate on
stored parameters rather than sampling or fitting the original functions.

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Record
     - Preserved information
   * - ``EnvironmentNode``
     - Parent and model indices, plus offsets and counts into ordered child
       and exclusion arrays. The shape array uses the same index as the node.
   * - ``EnvironmentShape``
     - Shape kind, center, dimensions, axes or normal. Lengths are in metres;
       vector components refer to the snapshot's common coordinate system.
   * - ``EnvironmentModel``
     - Density and refractivity kinds and parameters, uniform magnetic field
       in tesla, native medium identifier, and a composition range.
   * - ``EnvironmentComponent``
     - Native C8 nuclear ``Code`` and number fraction, in the supplied order.
       Fractions are neither renormalized nor merged during export.

Density is represented in g/cm³, geometry in metres and grammage in g/cm².
Model records retain the scale lengths, anchors, axes and coefficients needed
to evaluate density or refractive index at a position. An exponential density
is therefore stored as a function's parameters, not as one constant density
or a sampled three-dimensional grid.

``modelIndex_`` selects an exported model record. ``mediumId_`` identifies a
native physical medium; these are different index spaces. For refractivity,
the uniform form stores the refractive index itself, whereas the
Gladstone–Dale form stores the reference refractivity and density scaling.
See `EnvironmentModel.hpp`_ for the precise field conventions.

Device upload and ownership
---------------------------

``CascadeRunner`` passes ``prepared.environmentData_`` and
``prepared.geometry_`` to ``EmRunSession::beginProposalNative``. On first use,
the session initializes the EM backend. Its implementation constructs
``EnvironmentStorage<TExecutionSpace>`` and obtains a borrowed view of the
uploaded arrays. The storage constructor copies the six arrays:

.. literalinclude:: ../../../corsika/detail/kokkos/environment/EnvironmentStorage.inl
   :language: cpp
   :lines: 26-36
   :lineno-match:

Each upload allocates a Kokkos view, fills a host mirror and calls
``Kokkos::deep_copy``. ``getView()`` returns a managed read-only view sharing
the allocations. ``getDataView()`` returns borrowed pointers and counts;
``EnvironmentBorrowedView`` provides the query accessors around those pointers.
The memory accessibility of a data view depends on its owner: a host-bound
view does not become device-accessible merely by passing the struct to a kernel.

Finish pending execution before replacing or releasing the last allocation
owner, and release device owners before Kokkos finalization. Copying the
snapshot struct's bytes is not an interchange or serialization format.

Session reuse and scene changes
-------------------------------

``EmRunSession`` compares the exported numerical environment, observation plane
and magnetic-deflection limit with its initial configuration. A changed
environment in an initialized session raises an exception requesting a new
session; it is not silently re-uploaded. With matching configuration, the
backend retains allocations and begins a new shower.

Changes to native geometry or models require a new export and a compatible
backend session. Mutating the native scene alone does not update resident
arrays. ``make_environment_snapshot_key`` compares numerical fields rather
than native addresses; the separate snapshot fingerprint is a compatibility
tag, not a cryptographic integrity check.

Observation geometry
--------------------

The observation level is passed separately. ``CascadeRunner`` calls
``export_observation_plane(observationLevel.getPlane(), rootCS)`` and supplies
the resulting point and normal to preparation. The point is in metres and
the normal is dimensionless, both in the common coordinate system. The
normal is validated rather than silently normalized.

``TransportGeometryConfig::observation_`` stores this plane. It is not inserted
into the material tree and does not acquire a material model. Preparation's
horizontal-plane fallback is used only when no explicit observation plane
is supplied. Its distance competes with other step limits in transport.

Implementation references
-------------------------

* `EnvironmentTopology.inl`_: ordered flattening and native-index mappings.
* `EnvironmentSnapshot.inl`_: assembled export and lifetime relationships.
* `CascadeRunner.inl`_: actual observation plane and session handoff.
* `BackendInstanceImplementation.inl`_: owning backend allocation and borrowed view.
* `EmRunSession.cpp`_: initialization, reuse and environment-change rejection.

.. _Environment.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/media/Environment.hpp
.. _VolumeTreeNode.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/media/VolumeTreeNode.hpp
.. _EnvironmentTopology.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/EnvironmentTopology.inl
.. _EnvironmentSnapshot.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/EnvironmentSnapshot.inl
.. _EnvironmentModel.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentModel.hpp
.. _CascadeRunner.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeRunner.inl
.. _BackendInstanceImplementation.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/BackendInstanceImplementation.inl
.. _EmRunSession.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmRunSession.cpp
