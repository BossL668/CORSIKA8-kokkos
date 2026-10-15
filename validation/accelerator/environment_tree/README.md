# Environment-tree regression tests

The air-shower `kokkos-proposal` setup exports its native C8 environment tree.
`corsika/geometry/NextEnvironmentBoundary.hpp` is shared by native tracking and
the portable snapshot. It preserves search order, tie handling, child/excluded
node conditions and the final negative-distance clamp. This is not a global
nearest-surface search. The particle's current logical node matters.

The exporter supports the shapes and density/field models explicitly implemented
in `EnvironmentTreeBuilder.hpp` and `EnvironmentModelSnapshot.hpp`; unsupported
models fail at initialization instead of being approximated. The air-shower
setup requires air media. Exporting a geometry does not add missing interaction
tables for other materials. The separate terrain and EGS4 paths are not changed
by this integration.

Curved sphere intersections use `EnvironmentStableQuartic.hpp`: scaled
factorization with stable quadratic roots and a fixed correction of the
auxiliary cubic where needed. There is no per-step residual scan or bisection
fallback. This changes some floating-point roots relative to the original C8
solver; identical boundary-search logic does not imply bitwise-identical paths.

## Running

Configure the normal project for OpenMP or CUDA first, then build explicitly:

```sh
cmake --build BUILD --target testEnvironmentTree testEnvironmentCurved \
  testEnvironmentLepton testEnvironmentPolynomial test_multigpu_frontier
OMP_NUM_THREADS=1 OMP_PROC_BIND=false ctest --test-dir BUILD \
  -R 'testEnvironment' --output-on-failure
BUILD/applications/test_multigpu_frontier
```

- `TreeGeometry.cpp`: native search oracle, ordered/nested/excluded geometry,
  environment-model export and logical node retention in device queues.
- `CurvedGeometry.cpp`: curved intersections and search comparisons.
- `LeptonGeometry.cpp`: chained transport, boundary transitions, CPU/device
  state round trips and environment-session compatibility.
- `PolynomialStress.cpp`: known-root quartics, including ill-conditioned cases.
- `OriginalIntersect.inl`: frozen independent native search oracle. Do not
  replace it with a call to the shared routine under test.

These tests run their device kernels in the configured execution space. CPU
coordinator mocks and single-device tests do not replace a real multi-GPU run.
Whole-shower acceptance must also check output integrity, deferred CPU work,
reaction coverage and the selected production cut policy.

## Static frontier compatibility

`C8_STATIC_FRONTIER_V2` carries an environment fingerprint and a logical node ID
per particle. Its fingerprint compares exported parameters and topology, not
host pointers. It is a compatibility tag, not a cryptographic archive checksum.
Partitioning retains the header and node IDs; workers reject mismatched trees.
The old V1 format remains readable for paths without a full tree, but is rejected
by the full-tree path: a rounded position alone cannot reliably restore the
original logical owner at a boundary. Regenerate old frontiers with the matching
prefix/worker version. Completed shower data are unaffected by this change.
