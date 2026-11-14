Table representation and device storage
=======================================

``ProposalNativeTableSet`` owns the numerical export on the host.
``NativeTableStorage<TExecutionSpace>`` copies its query arrays into the
selected execution space's memory. Both expose those arrays through
``ProposalNativeDeviceView``, a read-only, non-owning interface used by the
native table queries. The uploaded data contain numerical records and
coefficients; no PROPOSAL calculator object is passed to a kernel.

This page describes the format-v7 representation at the chapter's source
baseline. Its validator requires PROPOSAL ``7.6.2`` and CubicInterpolation
``0.1.5``. The field definitions are in `ProposalNativeData.hpp`_ and the host
container is in `ProposalNativeTable.hpp`_.

Records, identities and units
-----------------------------

The table stores descriptors separately from their coefficient arrays. Each
descriptor identifies a slice through an offset and dimensions or a count.
A column can therefore retain its native interpolation coordinates without
requiring every process to use one common energy grid.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Record or array
     - Meaning
   * - ``NativeDndxColumn``
     - A stochastic process/component record, including particle and process
       IDs, identity hashes, rate and kinematic models, cut values, component
       parameters and a bicubic descriptor when the rate uses a spline.
   * - ``NativeTotalRateColumn``
     - The independently interpolated rate used by PROPOSAL's
       ``Interaction`` mean-free-path calculation, represented by a cubic
       descriptor for each exported projectile.
   * - ``NativeDedxColumn``
     - A continuous-loss component with its cubic descriptor, component
       weight, lower energy limit and accumulation identity.
   * - ``NativeUtilityColumn``
     - A displacement-integral cubic descriptor, particle mass, lower
       energy limit and the original integration-direction flag ``reverse_``.
   * - ``selectionColumnIndices_``
     - A permutation of the stochastic columns preserving
       ``Interaction::Rates()`` selection order.
   * - ``dedxAccumulationIndices_``
     - A permutation of the continuous-loss columns preserving the scalar
       cross-section/component summation order.
   * - ``NativePhysicalConstants``
     - Constants copied from the configured PROPOSAL calculator, including
       the electron, charged-pion and muon masses.

Stochastic descriptors are sorted by
``(pdgId_, processId_, componentHash_)`` for lookup. A validated table permits
one medium identity per projectile; medium is not an additional argument to
the device column lookup. The two permutations preserve the scalar operation
orders independently of canonical descriptor order. Replacing either order
with an arbitrary traversal can change a selection or floating-point sum.

The representation retains PROPOSAL units: energies and masses in MeV,
grammage in g/cm², rates in cm²/g and stopping power in MeV cm²/g. Upload
does not convert units. A transformed spline value is converted back by its
descriptor before a query returns the physical quantity.

Host-only metadata include dependency versions, ``stochasticCutMeV_``,
``interactionIdentities_``, the content hash and cache-hit diagnostics.
``NativeTableStorage`` retains the source hash for inspection, while the
query view contains the numerical arrays and constants needed for evaluation.
Cache-hit counts are excluded from the physics digest, so a cold build and
a warm cache load can identify the same numerical content.

Interpolation coordinates
-------------------------

``NativeAxisDescriptor`` stores ``type_``, ``nodes_``, ``low_``, ``high_``
and ``step_``. The endpoints describe the input domain; the transformation
maps its nodes to integer coordinates from zero through ``nodes_ - 1``.
Consequently, ``step_`` is a spacing in the chosen transformed coordinate,
not necessarily a spacing in MeV. These are the actual forward maps:

.. literalinclude:: ../../../corsika/detail/kokkos/em/tables/ProposalNativeQueries.inl
   :language: cpp
   :lines: 88-101
   :lineno-match:

The three axis types are ``Linear``, ``Exponential`` and
``ExponentialMinusOne``. In the last case the subtraction is ``log(2)``;
it ensures that ``low_`` maps to zero. Validation checks the endpoint maps
and their inverse against the node count and domain.

For a bicubic descriptor, ``energyAxis_`` acts on energy in MeV and
``lossAxis_`` acts on an already transformed loss coordinate. The validated
loss axis is linear. The separate ``NativeLossTransform`` maps a physical
loss fraction to that coordinate, using energy-dependent kinematic and cut
limits. Its choices are ``RelativeLog``,
``RelativeLogPowerOnePointFive`` and ``OneMinusLog``. Thus three quantities
must be kept distinct: a physical loss fraction, its transformed loss
coordinate, and a probability quantile supplied to an inverse query.

``NativeCubicDescriptor`` also carries an optional ``valueTransform_``.
When ``hasValueTransform_`` is set, interpolation produces a transformed
ordinate and the query applies the inverse value transformation. This
descriptor describes an invertible value map; its node count is not an
additional interpolation grid.

Coefficient layout
------------------

The exporter copies the original spline representation through the
read-only export interfaces. It does not sample a new grid of physics
function values. `NativeTableExporter.cpp`_ implements the following layout:

* ``bicubicValues_``, ``bicubicDerivativeEnergy_``,
  ``bicubicDerivativeLoss_`` and ``bicubicMixedDerivative_`` have the same
  length and share each descriptor's ``coefficientOffset_``. The element
  for energy row ``i`` and transformed-loss column ``j`` is at
  ``coefficientOffset_ + i + rows_ * j``. These are the native Hermite values
  and derivatives in the interpolation coordinates, not derivatives taken
  directly with respect to physical energy and loss fraction.
* ``bicubicPolynomialCoefficients_`` contains sixteen row-major coefficients
  per cell. Cells are ordered by ``energy + (rows_ - 1) * loss``, relative
  to ``polynomialCoefficientOffset_``. The host derives these coefficients
  from the retained Hermite arrays using the same Eigen matrix expression
  as CubicInterpolation. The descriptor type admits a missing-polynomial
  sentinel, but a validated format-v7 production table requires these
  coefficients for every bicubic column.
* ``cubicValues_`` and ``cubicNodeDerivatives_`` share a one-dimensional
  descriptor's ``coefficientOffset_`` and ``coefficientCount_``. Node values
  and first derivatives reproduce each interval of the original cardinal
  cubic B-spline through Hermite evaluation. If present, the value
  transformation is then reversed.

The derivative arrays and polynomial coefficients are part of the exported
physics content. Changing their layout or arithmetic order requires
validation against the original evaluator, even when the mathematical
interpolant appears equivalent.

Cumulative rates and probability normalization
----------------------------------------------

A bicubic stochastic table stores a cumulative rate as a function of energy
and transformed loss. Its upper endpoint is a rate, not necessarily one.
For these columns, ``query_proposal_native_cumulative_rate(...)`` evaluates
the raw spline, divides by ``componentWeight_`` and clips a negative result
to zero, following PROPOSAL's public interpolant behavior. The component
rate is obtained at transformed loss one. Direct-rate columns instead carry
a ``NativeRateModel`` and the parameters required by that evaluator.

Inverse queries receive a probability quantile in ``[0, 1]`` and normalize
against the appropriate endpoint during the query. For ordinary stochastic
loss, the target is calculated as
``(quantile * cumulative_rate_at_one) * componentWeight_`` and the solver
uses the raw, unclipped spline. The separate photon secondary-split query
uses ``quantile * raw_cumulative_at_one``. Preserving these multiplication
and division steps matters to the native numerical result; the stored
arrays are not rewritten into a normalized CDF.

A rate spline also does not by itself establish which physical variable
can be sampled. For example, photon pair production consumes the incident
photon with loss fraction one; a secondary energy split is a separate
query. The kinematic model in the column determines this distinction.

Finally, ``query_proposal_native_total_rate(...)`` uses the independent
``NativeTotalRateColumn`` spline. The sum used internally to select a
process/component is provided by
``query_proposal_native_selection_total_rate(...)`` in the recorded
selection order. These are distinct calculations and must not be
substituted for one another. See `ProposalNativeQueries.inl`_ for the
evaluation and normalization contracts.

Upload and allocation lifetime
------------------------------

``NativeTableStorage::initialize(source, maximumBytes)`` performs these
steps in order:

1. Reject an already initialized owner, validate the table structure and
   dependency identities, then recompute and compare its content hash.
2. Compare ``projectedDeviceBytes(source)`` with ``maximumBytes`` before
   allocating the query arrays.
3. Copy each host vector through a Kokkos mirror into an owning view in
   ``TExecutionSpace::memory_space`` and fence the storage's execution
   instance after the uploads.
4. Construct the borrowed query view and publish the source hash, byte
   count and initialized state. An exception during allocation or upload
   resets partial storage before it propagates.

The byte estimate counts descriptor arrays, both ordering arrays and all
coefficient arrays. It excludes host diagnostics, allocation overhead and
the rest of the backend's memory. It is a table-array budget, not a complete
GPU memory estimate.

After successful initialization, the uploaded arrays are independent of
the source host vectors. ``getDeviceView()`` and ``getSourceContentHash()``
reject access before initialization. The implementation is in
`NativeTableStorage.inl`_.

``ProposalNativeDeviceView`` contains pointers and counts plus constants
by value. Copying it neither copies the arrays nor extends their lifetime.
``make_proposal_native_host_view(table)`` similarly borrows the host vectors
without validation: the caller must retain a validated table and keep its
storage unchanged until queries finish. A view's addresses must be
accessible from the execution space that uses it. The view is not a
portable serialized table representation.

Copying a ``NativeTableStorage`` owner shares its immutable Kokkos
allocations. Resetting one copy leaves a previously acquired view usable
only while another owner retains all those allocations. ``reset()`` clears
the owner's view, hash and byte count and permits another initialization;
it does not fence kernels using a borrowed view. Complete those kernels
before releasing the last owner. Initialize Kokkos before constructing
storage and retain the runtime until all storage owners have been destroyed.

Existing checks
---------------

`testNativeTableStorage.cpp`_ checks rejection before initialization,
insufficient budgets, invalid data and hash mismatches; it also checks
shared ownership, reset, reinitialization and a kernel query through a
retained view. `testProposalNativeExport.cpp`_ compares the canonical export
with its reference, checks warm-cache content identity and rejects selected
corrupt records. `testProposalNativeQueriesDevice.cpp`_ compares queries
against the retained production reference on the build's actual
``Kokkos::DefaultExecutionSpace``. Comparisons within the selected execution space include statuses
and solver iterations; host/device comparisons cover selected rates with
a numerical tolerance. These tests define interface coverage, rather than
claiming that all GPU architectures have been exercised.

.. _ProposalNativeData.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/tables/ProposalNativeData.hpp
.. _ProposalNativeTable.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/tables/ProposalNativeTable.hpp
.. _NativeTableExporter.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/proposal/NativeTableExporter.cpp
.. _ProposalNativeQueries.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/tables/ProposalNativeQueries.inl
.. _NativeTableStorage.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/NativeTableStorage.inl
.. _testNativeTableStorage.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/tables/testNativeTableStorage.cpp
.. _testProposalNativeExport.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/tables/testProposalNativeExport.cpp
.. _testProposalNativeQueriesDevice.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/tables/testProposalNativeQueriesDevice.cpp
