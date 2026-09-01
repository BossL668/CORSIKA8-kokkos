# Version-locked PROPOSAL native-table dependencies

The project keeps two small, read-only patches here instead of modifying the
Conan cache:

- `cubicinterpolation/0.1.5@c8gpu/stable` exports immutable axis descriptors
  and the coefficient arrays already held by `CubicSplines` and
  `BicubicSplines`.
- `proposal/7.6.2@c8gpu/stable` exposes those arrays together with process,
  component, cut, hash, utility-table metadata, cache-hit diagnostics, and a
  read-only live-interpolant evaluator used only by validation.  Every dE/dX
  export also carries its stable ordinal in PROPOSAL's underlying component
  vector; the ordinal is assigned before non-interpolant entries are skipped.

Neither patch changes table generation, Boost serialization, CPU evaluation,
or random-number use. `conan-install.sh` registers both recipes with
`conan export`; the ordinary project `conan install --build=missing` then
builds them with the same profile and options as the rest of CORSIKA.

The additions are read-only in the behavioural sense, but they are **not
binary-ABI compatible** with the unpatched libraries.  The patches add virtual
functions and data members to existing C++ classes.  Always use the distinct
`@c8gpu/stable` references and perform a clean rebuild of every consumer; do
not mix vanilla headers, patched libraries, or object files from different
recipe revisions.  The recipes default to static libraries, which confines
this ABI boundary to the build graph when all targets are rebuilt together.
The PROPOSAL recipe uses Conan's `recipe_revision_mode` for its patched
CubicInterpolation requirement, so a changed CubicInterpolation recipe
revision produces a different PROPOSAL package ID instead of reusing a binary
built against older class layouts.

The PROPOSAL recipe applies three patches in order.  The first is the minimal
export API, the second adds validation/cache-state access without changing
table construction, and the third exports the independent total-rate spline
used by `Interaction::MeanFreePath()`.  All three patches are expected to
apply cleanly to an unmodified PROPOSAL 7.6.2 source archive.

## Isolated-cache verification

The packages can be reproduced without using or modifying the normal Conan
cache.  Both recipes require Conan 2 because the dependency traits and package
ID modes used here are Conan 2 APIs.  A clean Release verification is:

```bash
export C8_CONAN_AUDIT_HOME="$(mktemp -d)"
export CONAN_HOME="$C8_CONAN_AUDIT_HOME"

conan profile detect --force
conan create third_party/conan/cubicinterpolation --build=missing \
  -s build_type=Release
conan create third_party/conan/proposal --build=missing \
  -s build_type=Release
```

Each recipe has a `test_package` consumer.  The CubicInterpolation consumer
checks ordinary spline evaluation and verifies that exporting coefficients
does not alter the interpolant.  The PROPOSAL consumer constructs an ordinary
interpolated Compton cross section, exercises the existing rate, continuous
loss, cumulative-rate, and stochastic-loss API, exports the native tables,
and verifies that the ordinary results remain unchanged.  Neither consumer
enables CUDA or links a CUDA runtime.

Both test packages also build a separate `ordinary_api` executable that uses
only the upstream public API.  For an audit against the unpatched ConanCenter
packages, configure the same test-package source with
`-DC8GPU_BUILD_EXPORT_TEST=OFF`; its hexadecimal floating-point output can then
be compared byte-for-byte with the patched build without including any export
API in the test translation unit.

The isolated audit was exercised with Conan 2.11.0, CMake 3.31.0, GCC 13.3.0,
the `libstdc++11` ABI, GNU C++17, and `Release` static libraries.  The source
archives are pinned by SHA-256 in each `conandata.yml`.  This makes the patched
source reproducible, but a future remote can still resolve a newer compatible
revision for a transitive version range.  Preserve the Conan profile and a
lockfile as well when long-term reproduction of the complete binary graph is
required.
