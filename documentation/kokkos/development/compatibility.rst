Maintaining compatibility with C8 and dependencies
==================================================

Maintenance starts with an upstream change: a C8 interface evolves, a physics
model is corrected, or a dependency is upgraded. Determine which integration
points use that code, update those points, and verify the affected execution
paths. The objective is to keep native and accelerated results consistent
without maintaining another independent copy of the entire framework.

Following a C8 change
---------------------

Compare the old and new C8 behavior before changing the Kokkos adapter. An API
rename may only require a caller update; a new sampling rule, cut or particle
property may also require changes to numerical calculations and reference
results. Successful compilation alone does not distinguish these cases.

.. list-table::
   :header-rows: 1
   :widths: 24 44 32

   * - Upstream change
     - Kokkos integration points to review
     - Comparison to retain
   * - Particle state or coordinate interface
     - Particle conversion, numerical records and reconstruction of CPU
       particles. Review units, time and identity fields together.
     - Conversion round trip and CPU/device handoff with the same input state.
   * - Environment API or search behavior
     - Snapshot export, native-node mapping, shared search helpers and the
       affected density/intersection adapter.
     - The same geometry and query inputs through native and numerical paths.
   * - Physics model, cut or final-state behavior
     - Calculator preparation, table export, selection, device final state and
       CPU completion for that process.
     - Same-input process comparisons and appropriate statistical shower checks.
   * - Output schema or observer configuration
     - Record projection, waveform/profile merger and analysis readers.
     - Units, grids, event identity and complete direct/coordinated output.
   * - Build or dependency interface
     - CMake targets, native export bridge, backend factory and installed
       headers/dependencies.
     - Native-only and affected accelerated builds, plus an installed consumer.

The file locations and call chains are in :doc:`workflow` and the component
chapters. Update a shared numerical body once where one already exists.
Where native and Kokkos implementations are separate, make their relationship
explicit in the patch and run the comparison on both sides.

Updating PROPOSAL, Kokkos or the toolchain
------------------------------------------

The native table exporter depends on specific PROPOSAL interpolation data.
At this guide's baseline, `corsikaCheckProposalExport.cmake`_ requires
PROPOSAL 7.6.2 and CubicInterpolation 0.1.5, checks the export headers and runs a
compile/link probe. A version upgrade must therefore review the exporter,
interpolation definitions, validators and comparison tests together; changing
only the package version is not sufficient.

Kokkos and compiler updates principally affect compilation, execution-space
support, memory access and runtime behavior. Keep the numerical algorithms
unchanged while evaluating such an update unless a separate correction is
required. Build and execute the affected device tests on real hardware;
recognizing a backend name in CMake does not demonstrate that it works.
:doc:`../backends/build` records this baseline's language and dependency limits.

Versioned data and reused sessions
----------------------------------

When a change alters the meaning or layout of stored data, identify every
writer and reader. Decide explicitly whether old data remain valid, must be
regenerated or need a conversion. Relevant examples include exported tables,
environment fingerprints, worker particle lists and radio-track spools.

A persistent worker or an ``EmRunSession`` can reuse prepared data across
events. That reuse does not update changed physical inputs automatically.
Follow :doc:`extensions` when changing models, cuts or the scene. Update the
cache identity or session comparison when adding a parameter that affects
those prepared inputs.

The detailed reference below records current versions and their checks. It is
separate from the normal development workflow so contributors can consult
only the formats their change affects.

.. toctree::
   :maxdepth: 1

   formats

Verifying a maintained version
------------------------------

Use the existing tests in increasing scope. The :doc:`../validation/tests`
page gives target names and build conditions; :doc:`../validation/numerics`
describes numerical and statistical comparisons.

1. **Component:** exercise the changed calculation or conversion with fixed
   inputs, including relevant boundaries and failure results.
2. **Integration:** check native application use and the affected Kokkos path.
   Include CPU completion, event reuse or worker exchange if the change touches
   those interfaces.
3. **Shower:** compare the relevant particle profiles, energy accounting and
   radio outputs when physics, random histories or particle routing can change.
4. **Performance:** measure a matched workload when the change affects cost,
   separating initialization from repeated events and retaining the timing
   definitions used in the report.

A same-seed mismatch is not automatically a physics regression. Scheduling
and worker lifetime can change random-stream progression. State whether the
change must preserve fixed-input values, event histories or statistical
distributions, and choose the comparison accordingly. Do not weaken the
reference or tolerances only to make a candidate pass.

Diagnosing a regression
-----------------------

Keep the failing configuration and find the first stage that differs:
prepared inputs, table/query result, transported particle, CPU completion or
merged output. A small reproducible input at that stage is more useful than
starting with a complete-shower plot alone.

Retain the source revision, dependency versions, command line, hardware and
logs. When needed, use the process trace or scalar replay recording described
in :doc:`../validation/numerics`. Keep the original cache or frontier that
reproduces the problem before regenerating it. Diagnostics add work and output;
disable them for ordinary performance comparisons unless their cost is part
of the measurement.

Review and release record
-------------------------

A maintained change should state what was updated, which paths it affects,
what numerical or output differences are expected, and which configurations
were tested. Record any data that must be regenerated and any required change
to application or analysis code. Update source links, examples and API
explanations together with the implementation.

These are recommended review steps, not a claim that the repository's current
CI runs every configuration. The supported combinations must follow the builds
and runtime tests actually demonstrated for that release. Upstream CORSIKA's
review process determines the final acceptance requirements.

.. _corsikaCheckProposalExport.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/corsikaCheckProposalExport.cmake
