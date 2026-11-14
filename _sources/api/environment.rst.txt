Environment preparation and queries
===================================

The environment interfaces separate native-scene export, numerical storage
and per-query results. Host preparation accepts the existing C8 tree and
coordinate system; device queries operate on accessible exported arrays.
The declarations below are generated from the public headers. Supported
shapes, model kinds and shared search semantics are described in
:doc:`../environment/export`, :doc:`../environment/queries` and
:doc:`../environment/extension`.

Host snapshot and topology
--------------------------

The snapshot owns independent numerical records in one common coordinate
system. Its topology also retains borrowed mappings to native nodes and
models. Exporting the numerical arrays does not transfer those host pointers
to the device. Keep the native objects alive and unchanged while their
mappings are used; rebuild after changing the scene.

Sources: `EnvironmentSnapshot.hpp`_, `EnvironmentTopology.hpp`_ and
`EnvironmentPreparation.hpp`_.

.. doxygenclass:: corsika::kokkos::EnvironmentSnapshot
   :members:
   :undoc-members:

.. doxygenclass:: corsika::kokkos::EnvironmentTopology
   :members:
   :undoc-members:

.. doxygenfunction:: corsika::kokkos::prepare_environment_snapshot

Resident ownership and data views
---------------------------------

``bind_environment_data`` borrows host addresses from a snapshot.
``EnvironmentStorage`` uploads those arrays and owns the resident allocations.
The same ``EnvironmentDataView`` type can contain host or device addresses;
the type itself does not establish accessibility. Its owner must outlive all
operations using the view. See :doc:`../backends/device-code` for ordering and
memory-space requirements.

Sources: `EnvironmentDataView.hpp`_, `EnvironmentSnapshotData.hpp`_ and
`EnvironmentStorage.hpp`_.

.. doxygenstruct:: corsika::kokkos::EnvironmentDataView
   :members:
   :undoc-members:

.. doxygenfunction:: corsika::kokkos::bind_environment_data(EnvironmentSnapshot<TModelProperties> const&)

.. doxygenclass:: corsika::kokkos::EnvironmentStorage
   :members:
   :undoc-members:

Query configuration and results
-------------------------------

Environment queries return status with the numerical result. Preserve both:
a failure or out-of-range status may retain a nonfinite raw value. Node
indices refer to this snapshot, not to a medium identifier. Positions and
lengths use metres; density uses g/cm³ and grammage uses g/cm². The grammage
result record is also used for inverse length queries, as its declaration
specifies.

Source: `EnvironmentQuery.hpp`_.

.. doxygenstruct:: corsika::kokkos::EnvironmentQueryConfig
   :members:
   :undoc-members:

.. doxygenenum:: corsika::kokkos::EnvironmentStatus

.. doxygenstruct:: corsika::kokkos::EnvironmentLayerQuery
   :members:
   :undoc-members:

.. doxygenstruct:: corsika::kokkos::EnvironmentBoundaryQuery
   :members:
   :undoc-members:

.. doxygenstruct:: corsika::kokkos::EnvironmentGrammageQuery
   :members:
   :undoc-members:

Numerical queries
-----------------

These functions assume valid exported arrays. The logical-node argument
distinguishes an already assigned node from a request for point location.
The straight-boundary query does not include the independent observation
plane. Grammage and its inverse do not truncate their segment at a boundary;
step-limit competition remains the caller's responsibility.

.. doxygenfunction:: corsika::kokkos::query_environment

.. doxygenfunction:: corsika::kokkos::distance_to_environment_boundary

.. doxygenfunction:: corsika::kokkos::environment_grammage

.. doxygenfunction:: corsika::kokkos::environment_arclength

.. doxygenfunction:: corsika::kokkos::geometry_competition_grammage

The geometry-candidate function retains the production positive-infinity
comparison sentinel under its documented exponential-profile conditions.
That sentinel must not be consumed as the finite traversed depth of a
selected physical step. Curved boundary selection and the observation-plane
path are documented separately in :doc:`../environment/queries`.

.. _EnvironmentSnapshot.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentSnapshot.hpp
.. _EnvironmentTopology.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentTopology.hpp
.. _EnvironmentPreparation.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentPreparation.hpp
.. _EnvironmentDataView.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentDataView.hpp
.. _EnvironmentSnapshotData.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentSnapshotData.hpp
.. _EnvironmentStorage.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentStorage.hpp
.. _EnvironmentQuery.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentQuery.hpp
