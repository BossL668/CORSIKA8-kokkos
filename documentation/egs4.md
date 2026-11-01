# Native EGS4 atmospheric backend

This optional backend restores the previously validated C++/Kokkos port of C7
EGS4. It transports electrons, positrons and photons with EGS4, rather than
using PROPOSAL for their electromagnetic interactions. Both OpenMP and CUDA
use the same implementation. The default backend selection is unchanged.

## Build and run

Use the normal dependency setup and backend build script, adding:

```bash
tools/build_kokkos.sh openmp -DCORSIKA_ENABLE_EGS4=ON
# Or, with a CUDA toolkit and a supported NVIDIA GPU:
tools/build_kokkos.sh cuda -DCORSIKA_ENABLE_EGS4=ON
```

For a manual CMake build, enable `CORSIKA_ENABLE_KOKKOS`,
`CORSIKA_ENABLE_PROPOSAL_EXPORT` and `CORSIKA_ENABLE_EGS4`, and select the
OpenMP or CUDA Kokkos backend. The PROPOSAL export bridge is required for
muon transport even when EGS4 handles the electromagnetic cascade.

Select `--em-backend kokkos-egs4` in the existing `c8_air_shower` binary.
Use `--radio-backend kokkos` for device radio accumulation or `cpu` for the
existing host writers. `--egs4-stepfc` selects the C7 EGS4 step factor; it is
not the PROPOSAL continuous-loss limit. The native binary remains the entry
point; there is no Python launcher.

The AIR-NTP table is embedded at build time and loaded into memory once per
session. The default is `resources/c7_egs4/EGSDAT6_.4`. For comparisons with
a C7 run using a different table, pass its absolute path explicitly:

```bash
tools/build_kokkos.sh cuda -DCORSIKA_ENABLE_EGS4=ON \
  -DCORSIKA_EGS4_TABLE=/absolute/path/to/EGSDAT6_.25
```

The session selects the C7 EGSIN2 Sternheimer correction from the loaded
table's photon threshold: `.4` uses 10 and `.25` uses 11. Unsupported
thresholds fail explicitly. HATCH's lower bounds on electron and photon
transport cuts remain active; requesting a lower cut does not extend the
table. The output records the embedded table SHA-256 and STEPFC.

## Module boundaries

| Location | Responsibility |
| --- | --- |
| `corsika/kokkos/egs4` | Public session, table, routing and atmosphere interfaces |
| `corsika/detail/kokkos/egs4` | EGS4 physics, queue kernels and host handoff implementation |
| `src/kokkos/egs4` | Table parser, embedded data and compiled OpenMP/CUDA session |
| `corsika/kokkos/application/cascade/atmosphere/Egs4RunSession.hpp` | Atmospheric application facade |
| `tests/kokkos/egs4` | Table/cut, environment and muon-handoff regression tests |

`CascadeApplication` selects the EGS4 session through its existing backend
branch. `applications/c8_air_shower.cpp` keeps its scene construction,
`make_sequence` and `EAS.run()` interface. No EGS4 interaction details are
added to the main application.

Muons keep the existing PROPOSAL transport and C8 decay/final-state handlers.
They advance one wave at a time. Newly produced electrons, positrons and
photons enter EGS4 before another PROPOSAL electromagnetic step. Selected
photonuclear requests retain their target and branch when passed to the C8
hadronic handler. Unsupported requests fail rather than dropping a channel.

## Environment and outputs

The adapter exports the **actual C8 environment tree**. The C7 EGS4 stepping
implementation requires the standard five nested AIR-NTP atmospheric layers,
four radial exponentials followed by a homogeneous top layer, a constant
magnetic field and an observation plane. It retains C7's four-exponential
transport convention and five-layer overburden calculation. It is not a
general replacement for the tree-based PROPOSAL transport backend.

Additional exclusions, custom compositions, non-air media and incompatible
geometry are rejected before propagation. C8 and C7 standard air compositions
are recognized; the historical EGS4 material constants are retained. Ice and
firn continue to use their existing backend paths.

Steps, energy deposition, longitudinal profiles, observations and radio tracks
use the shared output interfaces. The EGS4 report requires zero scalar C8
electromagnetic transport steps and a drained queue before marking the event
complete. The native EM **production profile** remains a known limitation of
the historical backend and is marked incomplete in the metadata; it must not
be mistaken for the supported longitudinal particle profile.

EGS4 does not make a C7/C8 timing comparison automatically equivalent. Match
the table, effective cuts, atmospheric parameters, field, hadronic models,
thinning, observer positions and radio algorithms explicitly.

Table provenance and the original C7 license are retained in
[`resources/c7_egs4`](../resources/c7_egs4/README.md).
