Supported models and extension requirements
===========================================

The native scene can supply both CPU and Kokkos calculations without being
constructed twice. Device execution still requires supported representations
and callable operations. This page records the current implementation limits
and the work needed to extend them.

Device-access requirements
--------------------------

The native environment is a host object graph. ``VolumeTreeNode`` owns volumes
and children through smart pointers, keeps node links as references, and calls
polymorphic volume and model interfaces. Its coordinate-aware operations also
refer to native coordinate systems. Copying the top-level object would not
copy the pointed-to allocations or make its methods device-callable.

Three requirements must hold for a GPU query:

* Every referenced allocation must be accessible in that execution space and
  remain alive until the query completes.
* Every function reached by the query must have a device-callable implementation,
  including its numerical and coordinate operations.
* Dynamic dispatch, when used, must refer to valid device implementations.
  A host-constructed virtual-function table cannot be made device-callable
  by copying the object bytes.

GPUs can use classes and pointers; arrays are an implementation choice rather
than the only possible representation. Device-side virtual calls are possible
on some backends with additional construction and lifetime requirements, but
are not uniformly portable. Shared memory accessibility alone does not repair
host virtual-function dispatch. The `Kokkos virtual-function documentation`_
describes these constraints.

The current implementation keeps native scene construction on the host and
uses indices, tagged parameter records and device-callable functions for
queries. Kokkos views provide allocation and transfer facilities; Kokkos does
not automatically convert arbitrary C8 objects or functions into device code.
See the `Kokkos View documentation`_ for allocation and memory-space semantics.

Shape support
-------------

``export_shape`` recognizes exact native types. An unknown type, including a
subclass that might override a supported shape's behavior, is rejected instead
of being reduced silently to a base shape. Coordinates are transformed through
C8 before numerical parameters are exported.

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Shape
     - Point location and straight intersection
     - Curved intersection
   * - ``Universe``
     - Contains all points; no finite enclosing boundary.
     - No finite enclosing boundary.
   * - ``Sphere``
     - Supported.
     - Supported through the leapfrog sphere-intersection polynomial.
   * - ``Box``
     - Supported, including its transformed local axes.
     - Unsupported for a genuinely curved trajectory; the linear branch
       remains available.
   * - ``SeparationPlane``
     - Supported as a half-space.
     - Supported through the leapfrog plane intersection.

An arbitrary mesh or convex polyhedron is not an exported shape kind in this
tree interface. A separate compatibility implementation elsewhere in the
repository does not establish support in this path. The precise conversion
and intersection policies are in `ShapeConversion.inl`_ and
`CurvedBoundaryAccess.inl`_.

Model support
-------------

Each exported native model must provide density, uniform-field and refractivity
export interfaces, together with its medium and nuclear composition. These
interfaces expose parameters; they do not invoke numerical sampling to
approximate an otherwise unsupported function.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Numerical representation
   * - Density
     - Homogeneous, sliding-planar exponential, flat exponential or exponential
       firn.
   * - Magnetic field
     - One uniform three-component field per model, expressed in the common
       coordinate system.
   * - Refractive index
     - Uniform, Gladstone–Dale or exponential firn. The representation also
       carries an unavailable state; that state does not provide a usable
       refractive-index evaluation.
   * - Material composition
     - Native medium identifier and an ordered list of nuclear codes and
       number fractions.

Spatially varying magnetic fields and other density laws require additional
representations and evaluators. A linear-density or tabulated model cannot be
made compatible by relabeling its parameters as an exponential. Missing export
interfaces and unsupported kinds are rejected during snapshot construction.

The required native interfaces are ``IDensityModelExport``,
``IUniformMagneticFieldExport`` and ``IRefractivityModelExport``. If a derived
model overrides the underlying physical behavior, its exporter must describe
that behavior as well; inheriting an incompatible base exporter is insufficient.
`ModelConversion.inl`_ implements their conversion and validation.

Application and physics constraints
-----------------------------------

A successful snapshot export is necessary but not sufficient to run a new
scene. Current cascade preparation also applies these application policies:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Scene policy
     - Current requirement
   * - Atmosphere
     - Every exported model has medium ``AirDry1Atm``.
   * - Homogeneous ice
     - Exactly one exported model: ``WaterIce``, homogeneous density and
       uniform refractive index.
   * - Firn
     - Exactly one exported model: ``WaterIce``, exponential-firn density
       and exponential-firn refractive index.

These checks appear in `AtmospherePolicy.inl`_, `HomogeneousIcePolicy.inl`_
and `FirnPolicy.inl`_. Several nodes may share one model; the single-model
requirement is not a single-node requirement.

The physics tables must cover the material and requested processes. Radio
propagation also has its own medium and geometry requirements. Exporting a
new medium does not by itself supply cross sections, enable mixed-material
transport or validate its radio calculation. Those integrations must be
checked separately from environment queries.

Adding or changing an environment
---------------------------------

For a scene composed entirely of supported shapes and models, retain the
native C8 construction and export it through the existing interface. Check
the application's scene policy and physics/radio configuration before
assuming the new combination is runnable. New geometry need not imply a
new search algorithm.

For a new shape or model kind, the implementation work is:

1. Define the native semantics and export contract: units, coordinates,
   parameter domain, boundary conventions and ownership.
2. Extend the numerical record and exporter. A new shape needs containment
   and all required intersection operations. A new density law needs local
   evaluation, line-integrated grammage and its inverse; refractivity and
   field behavior require their corresponding evaluators.
3. Review snapshot transfer, memory accounting, numerical keys and frontier
   compatibility whenever records change. Extend session validation if new
   geometry configuration affects queries. Rebuild the snapshot and session
   after changing an active scene.
4. Connect the appropriate material tables and scene policy. Validate radio
   propagation separately when enabled. Keep these policies outside the
   generic node-search implementation.
5. Add native-versus-numerical tests for transformed coordinates, interface
   points, node ordering, inverse domains and failures. Run them on each
   supported execution space, followed by full-shower physics and performance
   regression tests for the intended application.

The existing firn implementation illustrates the separation: it adds exported
profile parameters and numerical evaluation while retaining the same topology,
storage and node-search interfaces. Its surface-domain and inverse-grammage
behavior require their own tests; using the common tree does not eliminate
model-specific validation.

Shared implementation and maintenance
-------------------------------------

The current boundary selector already shares one implementation body through
native and numerical access adapters. Point location retains equivalent
ordering in separate implementations, and model evaluation remains separate
from the host polymorphic methods. Changes to those separate implementations
must be reviewed together with their equivalence tests.

A further maintenance direction is to share small numerical functions beneath
the native and device adapters where their contracts match. Such a change
would require device-callable dependencies and explicit decisions about units,
rounding and domain handling. It is a development option, not a claim that
all current environment algorithms already have a single implementation.

The existing tests are mapped in :ref:`environment-validation`. Preserve
coverage of failure paths as well as successful queries, and distinguish
matching node decisions from numerical agreement of intersection distances.

.. _Kokkos virtual-function documentation: https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Kokkos-and-Virtual-Functions.html
.. _Kokkos View documentation: https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html
.. _ShapeConversion.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/ShapeConversion.inl
.. _CurvedBoundaryAccess.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/CurvedBoundaryAccess.inl
.. _ModelConversion.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/ModelConversion.inl
.. _AtmospherePolicy.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/atmosphere/AtmospherePolicy.inl
.. _HomogeneousIcePolicy.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/ice/HomogeneousIcePolicy.inl
.. _FirnPolicy.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/firn/FirnPolicy.inl
