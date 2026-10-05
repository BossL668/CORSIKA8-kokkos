# AIR-NTP EGS4 table

`EGSDAT6_.4` is the unmodified table copied from the CORSIKA 7.8050
distribution, `run/EGSDAT6_.4`, on 2026-10-05. It is a regular project
file, not a link to a local CORSIKA 7 installation.

SHA256: `148c56f0f6397faf0f4e23d9a20b2d754f73bae02d644d1c3506153e2e8c990d`

The original distribution's `COPYING` is preserved as `COPYING.CORSIKA7`.
This table retains its upstream provenance; the project does not claim
authorship of the data or relicense it under the project's source license.

The main application embeds this table at build time when
`CORSIKA_ENABLE_EGS4=ON`. Neither a CORSIKA 7 installation nor a runtime
table argument is required. `CORSIKA_EGS4_TABLE` remains an optional CMake
override for dedicated table tests. Module layout and build instructions
are in [the EGS4 guide](../../documentation/egs4_CN.md).

An existing CMake cache retains a previously explicit table path. Use
`cmake -S <source> -B <build> -U CORSIKA_EGS4_TABLE` once to adopt the new
default.
