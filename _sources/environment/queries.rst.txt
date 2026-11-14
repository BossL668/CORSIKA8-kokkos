Environment queries and boundary selection
==========================================

Transport reads the exported environment through a view. The same query
templates accept managed ``EnvironmentView`` storage or an
``EnvironmentBorrowedView`` over existing arrays. The caller supplies numerical
positions and directions in the snapshot's common coordinate system.

Point location and logical node state
-------------------------------------

``locate_node(tree, positionM, start)`` implements the native
``VolumeTreeNode::getContainingNode`` search order:

1. Reject a position outside the current volume.
2. Visit the first containing child, in the stored child order.
3. If no child contains the position, visit the first containing exclusion.
4. If neither applies, return the current node.

The numerical implementation is iterative and limits the number of hops to the
node count. It returns a nonnegative node index on success, ``-1`` for a point
outside the current volume, and ``-2`` for invalid indices or exhaustion of
that traversal bound. It does not climb to a parent or skip model-free nodes.
The implementations are in `EnvironmentSearch.inl`_ and
`VolumeTreeNode.inl`_.

Shape predicates retain the native boundary conventions: sphere and box
interiors use strict inequalities, while a separation plane includes its
surface. These comparisons are part of the contract; changing ``<`` to ``<=``
can change which node receives a boundary point.

``query_environment(environment, config, positionM, direction, logicalNode)``
adds the transport-facing contract. ``logicalNode == -3`` requests a search
from the root. Any other value is an assigned node identity and is used
without replacing it by geometric point location. The function then obtains
that node's model and local density. A missing model yields
``OutsideEnvironment``; properties are not inherited from the parent.

This distinction preserves the node assigned by a boundary crossing even
when the stored position lies on the interface. A particle's logical node and
its coordinates must therefore be transferred together, as described in
:doc:`../particles/conversion`.

Query interfaces
----------------

The functions below return status together with their result. They assume a
valid exported snapshot; they do not validate the entire graph on each call.
Position arguments use metres, and direction arguments are dimensionless unit
vectors in the same frame.

.. list-table::
   :header-rows: 1
   :widths: 29 35 36

   * - Interface
     - Additional input
     - Result
   * - ``query_environment``
     - Query configuration, optional direction and logical node.
     - ``EnvironmentLayerQuery``: status, node index and density in g/cm³.
   * - ``distance_to_environment_boundary``
     - Direction and logical node; straight-line intersection.
     - ``EnvironmentBoundaryQuery``: status, distance in metres and next node.
   * - ``environment_grammage``
     - Assigned node, starting point, direction and segment length.
     - ``EnvironmentGrammageQuery``: status and depth in g/cm².
   * - ``environment_arclength``
     - Assigned node, starting point, direction and target depth.
     - ``EnvironmentGrammageQuery``: status and length in metres.

``EnvironmentQueryConfig`` carries the maximum magnetic deflection. The query
contract requires it to be finite and within ``(0, pi/2)``. The lower-level
node search does not use this transport configuration. Result fields named
``radiusM_`` are retained for compatibility and are zero on the tree path.

Density, refractivity and grammage
----------------------------------

``density(model, positionM)`` evaluates the exported model parameters. It
selects the firn, homogeneous or exponential implementation by the stored
model kind:

.. literalinclude:: ../../../corsika/detail/kokkos/environment/ModelEvaluation.inl
   :language: cpp
   :lines: 61-77
   :lineno-match:

``refractive_index`` similarly selects the uniform, Gladstone–Dale or firn
form. Uniform magnetic-field components are stored directly in the model.
These operations do not call the native polymorphic model during a kernel.

``grammage`` and ``arclength`` operate on **one assigned model along a straight
segment**. They do not locate a new node, clip the segment at boundaries or
integrate across several materials. Transport must first obtain appropriate
step limits and use the model assigned to the resulting segment.

For the sliding-planar exponential model, the integration axis is fixed from
the starting position. This follows the native model's local planar
prescription; it is not an exact integral through a continuously changing
radial direction. The homogeneous and atmospheric exponential inverses use
their corresponding algebraic expressions.

The firn profile stores deep density and refractive index, their deficits,
surface position and exponential scale. Its line integral is analytic, but
the current grammage inverse uses a bounded bisection with 80 iterations.
Thus an analytic density law does not imply that every inverse query has a
closed-form implementation. See `ModelGrammage.inl`_ and `FirnModel.inl`_.

Ordered boundary selection
--------------------------

Point location and boundary selection answer different questions. Point
location identifies the volume containing a position. Boundary selection
compares intersections along a proposed trajectory and determines the next
logical node.

The current repository shares ``NextEnvironmentBoundaryBody.inl`` between
``corsika::next_environment_boundary`` and the Kokkos entry point. The latter
is a device-callable wrapper around that common body:

.. literalinclude:: ../../../corsika/detail/kokkos/environment/NextEnvironmentBoundary.inl
   :language: cpp
   :lines: 12-17
   :lineno-match:

The access object supplies intersections and ordered topology references.
Selection starts from the caller's limit and current node, then considers
the current volume's exit, child entries and exclusion entries. An exit
selects the parent; an entry selects the corresponding child or exclusion.
The implementation retains ``<=`` tie comparisons, the native child-entry
condition and a final clamp of negative selected parameters to zero.
Later candidates can therefore replace an equal earlier candidate. Sorting
or deduplicating exported relationships would alter this behavior.

Sharing the selection body avoids maintaining two versions of that ordering
logic. It does not make the native and numerical intersection solvers
identical: their representations, arithmetic and trajectory adapters remain
distinct.

Straight and curved trajectories
--------------------------------

``StraightBoundaryAccess`` uses numerical line intersections. With a unit
direction, its trajectory parameter is distance in metres.
``CurvedBoundaryAccess`` instead uses the charged-particle leapfrog trajectory
and a parameter in seconds. Its constructor receives the required kinematics
and magnetic field; it does not infer a particle species.

``lepton_boundary`` converts the candidate length to time using the particle
speed, calls the shared selection and converts the result back to length.
This length is the trajectory's step parameter, not the straight chord between
the updated endpoints. Curved separation-plane and sphere intersections use
quadratic and quartic equations respectively. When the trajectory is linear,
the adapter uses the straight intersection path.

Curved box intersections are not implemented in this snapshot path. Unsupported
shapes and incomplete intersection calculations set flags in the curved access
object; ``lepton_boundary`` propagates them through its ``failed_`` result.
A selected distance alone is insufficient to establish a successful query.
See :doc:`extension` for the support matrix.

Observation-plane intersection
------------------------------

The independent ``ObservationPlane`` participates in transport through its
own intersection operation, not through node-tree search.
``intersect_observation_plane_straight`` returns a forward distance and status
within the supplied distance limit. ``lepton_observation_distance`` evaluates
the plane against the leapfrog trajectory. It converts a strictly positive
entry time to distance and otherwise returns infinity; it does not add a
separate intersection-validity check or truncate the distance to another limit.

These functions do not create a material node or perform output processing.
The transport and observation processes consume the selected step and decide
how to handle the crossing.

Failure semantics
-----------------

``EnvironmentStatus`` distinguishes invalid snapshots or configurations,
nonfinite inputs, positions without a supported material, missing forward
intersections and out-of-range grammage. The legacy
``BelowObservationSurface`` value is unused on the node-tree path.

Preserve status and value together. An inverse failure can retain a nonfinite
raw result; it must not be substituted silently with zero. The specialized
``geometry_competition_grammage`` also retains positive infinity for specific
growing-exponential candidates during limit comparison. That sentinel is not
a valid traversed depth for a selected physical step.

These interfaces report results to transport; they do not independently select
a CPU fallback policy. The caller must interpret failure according to the
transport contract. Unsupported scene export fails during preparation rather
than automatically creating a per-particle CPU fallback.

Implementation references
-------------------------

* `EnvironmentQuery.hpp`_ and `EnvironmentQuery.inl`_: arguments, statuses and wrappers.
* `NextEnvironmentBoundaryBody.inl`_: common ordered boundary-selection body.
* `CurvedBoundaryAccess.inl`_: trajectory intersections and error flags.
* `EnvironmentLeptonBoundary.inl`_: particle-kinematic adapter and distance conversion.
* `ObservationPlane.inl`_: independent straight observation-plane intersection.

.. _EnvironmentSearch.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/EnvironmentSearch.inl
.. _VolumeTreeNode.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/media/VolumeTreeNode.inl
.. _EnvironmentQuery.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentQuery.hpp
.. _EnvironmentQuery.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/EnvironmentQuery.inl
.. _ModelGrammage.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/ModelGrammage.inl
.. _FirnModel.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/firn/FirnModel.inl
.. _NextEnvironmentBoundaryBody.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/framework/geometry/NextEnvironmentBoundaryBody.inl
.. _CurvedBoundaryAccess.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/CurvedBoundaryAccess.inl
.. _EnvironmentLeptonBoundary.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/EnvironmentLeptonBoundary.inl
.. _ObservationPlane.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/ObservationPlane.inl
