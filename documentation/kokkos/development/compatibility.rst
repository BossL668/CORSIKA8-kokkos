Compatibility, diagnostics and release review
=============================================

Compatibility has several independent surfaces: C++ consumers, installed
targets, numerical snapshots, worker messages, output readers and random
histories. Review the surface affected by a change and retain evidence from
its consumers. A successful compile does not establish that an old cache,
worker executable or analysis script still has the same meaning.

This page records the checks present at the chapter's pinned source baseline.
The release checklist below is a recommended review procedure, not a claim
that an existing CI pipeline performs every check. Test selection and
comparison criteria belong to :doc:`../validation/tests` and
:doc:`../validation/numerics`.

C++ interfaces and installed packages
-------------------------------------

Review public headers together with their CMake targets. A change to a
parameter, record field, enum value, default, unit or ownership rule can affect
callers even if the function name is unchanged. Templates also depend on the
installed ``corsika/detail/kokkos`` implementations included by public headers.
Both that directory and ``corsika/kokkos`` are installed by the
`Kokkos target definitions`_. Their presence in the installation does not
make every implementation detail an independently versioned API.

The ``CORSIKA8::`` target names, transitive dependencies, include paths,
compile features and backend definitions are part of the consumer contract.
For example, ``KokkosData`` supplies C++17 and include paths without linking
Kokkos; ``KokkosRuntime`` links Kokkos and Threads; ``EmRunSession`` is available
only with the native export bridge and production backend. Retain these
boundaries when adding dependencies. The complete build gates are described
in :doc:`../backends/build`.

The `package configuration`_ records whether native export and EGS4 were
compiled. Those are installation properties, not switches that a consumer
can use to add missing code. The top-level `package version rule`_ uses
``SameMajorVersion`` for CMake package selection. That rule alone establishes
neither a C++ ABI guarantee nor interoperability of numerical or wire formats.
Check an external consumer against the installed headers and imported targets
using the contract modules described in :doc:`../validation/tests`.

Native export has an additional dependency check. At both build and consumer
configuration, `corsikaCheckProposalExport.cmake`_ requires PROPOSAL ``7.6.2``
and CubicInterpolation ``0.1.5`` exactly, hashes the listed export headers and
compiles and links an executable using the export API. An installed consumer's
header digest must match the digest recorded by that installation. The check
clears its cached compile result before repeating the probe. Preserve the
matching patched dependency graph when rebuilding an application or consumer.

Numerical identities and cache invalidation
-------------------------------------------

These mechanisms answer different questions. A content hash identifies
numerical data; a cache key identifies the inputs used to obtain it; an
environment key determines whether a session can reuse its allocations.
Changing a filename or reusing a process does not recompute all three.

.. list-table::
   :header-rows: 1
   :widths: 24 43 33

   * - Surface at this baseline
     - Identity and checks
     - Consequence for a change
   * - Native table format ``7``
     - ``validate_proposal_native_table`` checks the format, exact dependency
       versions and numerical structure. Upload recomputes the content hash
       before accepting the table. The digest includes physics arrays and
       ordering metadata, but excludes cache-hit diagnostics.
     - Review descriptors, coefficient layout, selection order and hash
       serialization together. A table version is not a portable serialization
       of its pointer-bearing device view. See :doc:`../physics-data/storage`.
   * - Auxiliary cache format ``4``
     - The key hashes format and algorithm versions, dependency versions and
       sorted, deduplicated medium/projectile/stochastic-cut identities.
       The reader checks magic, version, key, payload bounds, exact file
       consumption, SHA-256 and payload validity.
     - Incompatible or corrupt reads cause regeneration under the per-key lock.
       The algorithm tag is ``proposal-native-aux-v4-muon-brems-v1``. Its raw
       payload layout is ABI- and byte-order-dependent; changes to generation
       semantics require reviewing the key as well as the payload.
   * - Moliere interpolation cache format ``1``
     - Checks magic, exact file and object sizes, byte equality of the generating
       snapshot, table checksum and finite table values.
     - Mismatches rebuild the derived table. The file suffix is
       ``.moliere-initial-v1.c8cache``; keep its version and validation aligned
       with changes to the interpolation algorithm or stored layout.
   * - Tuning format ``c8-kokkos-tuning-v1``
     - Matches backend, device name/index, architecture, driver/runtime/Kokkos/
       compiler versions, project revision, native-table hash and thread count.
       Accepted parameter fields must be positive.
     - Missing or nonmatching files are cache misses unless exact matching was
       requested, in which case they throw. Malformed files can throw even in
       optional mode. Preserve the cache when diagnosing a mismatch.
   * - Environment/session reuse
     - The numerical key includes topology, shapes, media, composition, fields
       and firn deficit parameters. The frontier fingerprint hashes that key;
       the session also compares geometry, including the observation plane
       and magnetic-deflection limit.
     - A changed environment is rejected during session reuse. Reuse updates
       shower seed/ID and thinning; it does not re-export physics or apply new
       transport cuts. Use a fresh process for changed models, cuts or scene,
       respecting the runtime lifetime described below.

Sources: `ProposalNativeTable.cpp`_, `ProposalNativeAux.cpp`_,
`MoliereInterpolation.cpp`_, `TuningCache.cpp`_,
`EnvironmentSnapshotKey.inl`_ and `EmRunSession.cpp`_. Cache locations,
publication and ownership are covered in :doc:`../physics-data/preparation`.

An ordinary production backend owns Kokkos initialization. A competing owner
is rejected, and releasing the final owning lease finalizes Kokkos; it cannot
then be reinitialized in the same process. Replacing an initialized session
is therefore not a general reconfiguration mechanism. See
:ref:`backend-runtime-owner` for ownership and cooperative-lease rules.

The tuning key's project revision comes from
``CORSIKA_KOKKOS_PROJECT_REVISION``, whose build default is ``unknown``.
Set it to the reviewed revision when building release artifacts; the default
does not automatically invalidate tuning data after a source change.

Files and worker protocols
--------------------------

Review readers and writers together, including their rejection paths. Format
checks have a defined scope; a version field does not automatically negotiate
older or newer formats.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Exchange
     - Current format and compatibility boundary
   * - Static EM frontier
     - ``C8_STATIC_FRONTIER_V1`` has no logical-node tree. ``V2`` includes the
       environment fingerprint and a logical node per particle. ``V3`` adds
       an expanded history limit of more than one and at most 255 blocks of
       :math:`2^{48}`. The reader accepts these specific
       headers, rejects trailing header fields and checks environment identity,
       particle identity and numerical state. Logical nodes are restored through
       the supplied topology. See `StaticFrontier.inl`_.
   * - CPU prefix frontier
     - ``C8_CPU_FRONTIER_V1`` is a separate format with an environment
       fingerprint. Its importer requires a nonempty matching tree and rejects
       malformed headers. It is not interchangeable with the static EM frontier.
       See `CpuFrontier.inl`_.
   * - Radio track spool
     - Version ``1`` uses a 32-byte header and 140-byte records, with explicit
       little-endian integers and IEEE-754 double bit patterns. The reader
       requires matching magic/version/record size, a completed count and exact
       file length; record validation checks particle kind and numerical values.
       The writer replaces the incomplete count when closed. See
       `RadioTrackSpoolFormat.inl`_ and `RadioTrackSpoolReader.inl`_.
   * - Hadronic worker requests/results
     - ``HadronicBatchProtocolVersion = 3`` uses a batch header, contiguous
       interaction-result headers and flattened secondaries for each response.
       Headers check magic, version and endian marker; the parent also checks
       batch/count, result version, sequence and success status. The transport
       sends native C++ record bytes and targets local workers built from the
       same source tree. These checks are not a general cross-toolchain ABI
       protocol. See `HadronicBatchProtocol.hpp`_ and `HadronicProcessPool.inl`_.
   * - Persistent application worker control
     - ``request_<n>.json`` and ``done_<n>.json`` have required fields but no
       explicit format-version negotiation. Requests validate the sequence
       number; non-skipped requests require a positive seed and input/output
       paths. The parent checks acknowledgement request, PID and return code.
       Changes must cover both the coordinator and worker, including skipped
       requests. See `WorkerSession.inl`_ and `AirShowerCoordinator.cpp`_.
   * - Decision replay tape
     - Version ``1`` checks magic, header/record sizes and an endian marker;
       reading rejects truncated data and trailing bytes. The stored native
       records, propagation arrays and observer snapshots support the diagnostic
       reader, rather than an arbitrary-platform interchange format. See
       `DecisionReplayTape.inl`_.

The execution sequence and ownership of these exchanges are described in
:doc:`../execution/coordinator` and :doc:`../execution/hybrid-loop`.

Outputs and random-stream policies
----------------------------------

Output compatibility includes column names and types, units, observer order,
grids, YAML/JSON keys and completion semantics. The coordinator's
`output merger`_ checks configurations, Arrow schemas and row/grid agreement
before adding profile or radio arrays. It preserves coordinate columns and
checks record schemas when concatenating particle output. Updating an output
producer therefore also requires reviewing the merger and analysis consumers.
See :doc:`../usage/outputs` for the products and completion checks.

Metadata versions cover particular parts of the output. The PROPOSAL session
configuration currently records
``backend_version: kokkos-em-v2-cpu-alignment-rng2`` and
``rng_domain_version: 2``; the shower report records
``timing_schema_version: 2``. These fields do not collectively version every
Parquet, YAML or JSON product. The report also exposes native-table and
auxiliary hashes, dependency versions and cache-hit diagnostics. Their
producers are `CascadeRunSession.cpp`_ and `ShowerReport.inl`_.

Random-number domains are part of the numerical contract. A change to
``RandomNumberKey``, process/draw IDs, history allocation or native fallback
replay can alter results without changing an output schema. The current
`RandomDomains.hpp`_ sets domain version ``2``. Persistent workers explicitly
record ``cpu_rng_policy: continue-worker-streams`` and
``gpu_rng_policy: request-seed-and-particle-history``. Record process lifetime,
event order and coordinator worker assignment alongside the seed; consult
:doc:`../validation/numerics` before asserting reproducibility across modes.

Diagnosing a regression
-----------------------

Build a reproducible case around the earliest differing boundary. Keep the
original failing artifacts before trying a fresh build, cache or process.

1. **Record the configuration.** Retain source and local-change identity,
   command line, CMake cache, Conan profiles/dependency graph, model data,
   compiler and Kokkos versions, architecture, device/driver, execution mode
   and thread settings. Include seeds, event order, worker lifetime, cuts,
   thinning, environment and radio configuration.
2. **Keep the evidence of failure.** Save exit status and complete logs,
   ``gpu_em`` configuration/summary, timing records and relevant coordinator
   requests, acknowledgements and completion markers. Retain the original
   frontier/spool/tape and cache hashes when those inputs reproduce the issue.
3. **Localize the first mismatch.** Distinguish package selection/linkage,
   export validation, upload/context construction, query/selection, transport,
   CPU fallback, worker exchange and output merging. A dependency digest
   mismatch is resolved at package construction; a frontier fingerprint
   mismatch is resolved at environment capture/import.
4. **Use the narrow diagnostic.** ``--cuda-replay-trace`` writes process-level
   CSV records and enables full step records. The trace is independent of
   ``OutputManager`` and can contain evidence from incomplete showers.
   ``--cuda-replay-tape-out`` requires ``--em-backend proposal`` and records
   scalar transport segments plus radio snapshots. These options are
   diagnostics with additional work and storage; record their use in timing
   comparisons. See `BackendCli.cpp`_ and `ReplayTrace.hpp`_.
5. **Select the matching check.** Header and installed-consumer contracts
   address includes/linkage; table/auxiliary tests address export and cache
   identity; backend/session tests address staging and warm reuse;
   process-pool/coordinator tests address transport and ordering; radio spool
   and replay tests address file compatibility. Use the registrations in
   :doc:`../validation/tests`, inspect ``ctest -N``, then retain the exact
   selection and output. An absent backend-gated test supplies no result.

Use identity fields to align records before comparing numerical values.
A trace row, replay round-trip or fixed input fixture establishes its own
boundary; complete-shower agreement needs the comparison protocol in
:doc:`../validation/numerics`. For runtime ownership, unavailable backends or
thread-policy failures, follow :doc:`../backends/runtime`; for a timing
regression, follow :doc:`../validation/performance`.

Dependency updates and release review
-------------------------------------

The current integration uses C++17 and accepts Kokkos ``4.x`` from ``4.7``
upwards while rejecting ``5.x`` at source configuration. The repository's
`Conan recipe`_ selects Kokkos ``4.7.03`` and revision-pinned patched PROPOSAL
and CubicInterpolation recipes for export builds. A dependency update must
revisit the compile/link probe, table validator, cache identities and numerical
references together. Updating only the package requirement leaves other
version checks in force.

Record the exact backend/toolchain combinations reviewed. The build resolver
recognizes OpenMP, CUDA, HIP, SYCL and combined CUDA/OpenMP configurations;
recognized configuration is distinct from a passing build or numerical result.
The supported device sets, helper prerequisites and test limitations are
documented in :doc:`../backends/build` and :doc:`../validation/tests`.

For a release candidate, the following review produces a concrete record:

* Identify the source revision, dependency recipes, model data, compiler,
  architecture and enabled physics/backend options; set the project revision
  used by runtime metadata and tuning keys.
* Review changed public headers, exported targets and defaults. Build the
  candidate and an installed consumer in each claimed configuration, including
  the scalar-only build if shared application code changed.
* Inventory affected table/cache versions, fingerprints and file/message
  formats. Exercise their relevant acceptance and rejection tests, cold
  generation and warm reuse. Record whether previous artifacts remain readable,
  must be regenerated or need an explicitly implemented conversion.
* Run the complete configured test selection after focused checks. Preserve
  test lists, logs and reports; state disabled options and untested hardware.
  Use numerical and physics comparisons appropriate to the changed behavior.
* Check representative direct and coordinated outputs when their paths changed,
  including schemas, completion state, worker failure handling and random-stream
  metadata. Assess timing using the same boundaries and cold/warm policy.
* Update interface and format documentation, command examples and release notes
  with the resulting behavior and required consumer changes. If removing an
  interface, specify the replacement, migration and any deprecation period
  explicitly in that change.

Source map
----------

The links below are pinned to ``d5297c43c041a22b43c65c322d9b51bc7ec8e397``.
They identify implementation checks; validation results must additionally name
the candidate build and hardware on which the checks were run.

.. _Kokkos target definitions: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/CMakeLists.txt
.. _package configuration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/corsikaConfig.cmake.in
.. _package version rule: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/CMakeLists.txt
.. _corsikaCheckProposalExport.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/corsikaCheckProposalExport.cmake
.. _ProposalNativeTable.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/tables/ProposalNativeTable.cpp
.. _ProposalNativeAux.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/proposal/ProposalNativeAux.cpp
.. _MoliereInterpolation.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/tables/MoliereInterpolation.cpp
.. _TuningCache.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/runtime/TuningCache.cpp
.. _EnvironmentSnapshotKey.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/EnvironmentSnapshotKey.inl
.. _EmRunSession.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmRunSession.cpp
.. _StaticFrontier.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/StaticFrontier.inl
.. _CpuFrontier.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/CpuFrontier.inl
.. _RadioTrackSpoolFormat.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/RadioTrackSpoolFormat.inl
.. _RadioTrackSpoolReader.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/RadioTrackSpoolReader.inl
.. _HadronicBatchProtocol.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/scheduling/HadronicBatchProtocol.hpp
.. _HadronicProcessPool.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/HadronicProcessPool.inl
.. _WorkerSession.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/WorkerSession.inl
.. _AirShowerCoordinator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/scheduling/AirShowerCoordinator.cpp
.. _DecisionReplayTape.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/output/DecisionReplayTape.inl
.. _output merger: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/scheduling/StaticFrontierIO.cpp
.. _CascadeRunSession.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/cascade/CascadeRunSession.cpp
.. _ShowerReport.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/ShowerReport.inl
.. _RandomDomains.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/random/RandomDomains.hpp
.. _BackendCli.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/BackendCli.cpp
.. _ReplayTrace.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/output/ReplayTrace.hpp
.. _Conan recipe: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/conanfile.py
