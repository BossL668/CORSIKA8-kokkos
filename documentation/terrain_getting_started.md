# Terrain applications: installation and first run

[中文](terrain_getting_started_CN.md) · [Main installation guide](../README.md)

This guide describes the published beta5 application interfaces, not an older
mountain checkout or an uncommitted radio prototype. First complete the main
README's empty environment, branch/submodule, Conan recipe and FLUKA setup
(sections 3–5). Python preparation alone does not build a C++ simulator.

## 1. Build the optional applications

The terrain shower needs the complete licensed FLUKA installation, including its
Fortran headers. The geometry validator alone does not require shower physics,
but the following complete workflow uses FLUKA:

```bash
cd "$HOME/corsika-21cma-kokkos-beta5/corsika8_kokkos_beta5"
conda activate corsika_venv
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
export FLUPRO="$HOME/fluka"
test -r "$FLUPRO/libflukahp.a"
test -r "$FLUPRO/flukapro/(FLKMAT)"
test -r "$FLUPRO/flukapro/(DIMPAR)"
test -r "$FLUPRO/flukapro/(FLKCMP)"
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh openmp \
  -DWITH_FLUKA=ON -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON \
  -DC8_INTERFACE_EXECUTION_SPACE=OPENMP
```

Stop if a file check fails. For a licensed installation with a different header
layout, supply `-DC8_TERRAIN_FLUKA_INCLUDE_DIR=/absolute/header/directory` and check
the three files there. Do not replace FLUKA with another model to bypass this
requirement. The helper creates its own Conan toolchain and pinned Pythia/TAUOLA
builds; no prebuilt `build/external` is assumed. Reconfiguring an existing OpenMP
build adds optional targets; do not reinstall over a running campaign's binary.

Check the installed applications, from the project container directory:

```bash
cd ..
export CORSIKA_DATA="$PWD/install/openmp/share/corsika/data"
install/openmp/bin/c8_terrain_environment --help
install/openmp/bin/c8_terrain_cascade --help
install/openmp/bin/c8_mountain_neutrino --help
```

`c8_mountain_neutrino` is the older finite convex-volume example. Use
`c8_terrain_cascade` below for a geographic DEM, atmospheric embedding and
material crossings. These are direct executables, not the air launcher.

### NVIDIA alternative

After the main README's CUDA Toolkit setup, substitute this build command:

```bash
cd "$HOME/corsika-21cma-kokkos-beta5/corsika8_kokkos_beta5"
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh cuda \
  -DWITH_FLUKA=ON -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON \
  -DC8_INTERFACE_EXECUTION_SPACE=CUDA
```

Then use `install/cuda/bin/...`, its matching `CORSIKA_DATA` and `--threads 1
--device 0`. Architecture/profile selection is the same as for air. The terrain
interface execution space is a **build-time** choice; the air application's
`--kokkos-execution cuda-openmp` is not a terrain switch. If using the combined
helper, explicitly select `C8_INTERFACE_EXECUTION_SPACE=OPENMP` or `CUDA` and use
`install/cuda-openmp`; this still does not enable terrain cooperation.

## 2. Prepare a small geographic scene

In the source directory, install the separate terrain Python package and check
its entry point. Its `pyproject.toml` supplies the geographic dependencies:

```bash
cd "$HOME/corsika-21cma-kokkos-beta5/corsika8_kokkos_beta5"
python -m pip install ./python
c8-terrain --help
c8-terrain --bounds 86.700 42.930 86.710 42.940 \
  --output "$HOME/CorsikaData/terrain/demo" --estimate-only
```

Bounds are west longitude, south latitude, east longitude, north latitude in
degrees. Estimate mode does not download. After checking the estimated size:

```bash
c8-terrain --bounds 86.700 42.930 86.710 42.940 \
  --output "$HOME/CorsikaData/terrain/demo" \
  --check-with ../install/openmp/bin/c8_terrain_environment
```

For the CUDA alternative replace the validator path with its installed prefix.
Initial use downloads DEM/geoid data; subsequent identical requests reuse
verified caches. Use `--cache PATH` for another disk or `--offline` only when the
required caches already exist. Change the output directory when changing the
scene; do not overwrite scientific products.

Without station input, preparation creates **one demonstration observer**, not
the real 21CMA array. Add `--antennas-csv stations.csv` for geographic station
data (`name,longitude_deg,latitude_deg`), or `--station-directory DIR` for the
supported original station-file format. Default heights use DEM + 1 m; measured
ellipsoidal/EGM96 heights require explicit settings. See the
[preparation and height-datum guide](terrain_preparation_workflow_CN.md) and
[editable request example](../configs/mountain/terrain_region_21cma.yaml).

Inspect `terrain/scene.yaml`, `terrain_manifest.yaml`, `observers.yaml`,
`native_environment_check.yaml` and the three preparation figures. The full
`terrain/` directory contains the mesh and relative scene references. Keep it
together when transferring a scene. ENU terrain coordinates are not the air
application's NWU antenna coordinates or the analysis Ex′/Ey′/Ez′ basis.

## 3. Run transport before adding radio

For this small generated scene the origin is at the region-centre surface.
Verify that in the manifest/geometry check; never reuse these injection
coordinates blindly for another scene. From the project container directory:

```bash
cd "$HOME/corsika-21cma-kokkos-beta5"
export CORSIKA_DATA="$PWD/install/openmp/share/corsika/data"
install/openmp/bin/c8_terrain_cascade \
  --scene "$HOME/CorsikaData/terrain/demo/terrain/scene.yaml" \
  --primary photon --energy-GeV 0.002 --seed 67101 \
  --position-m 0 0 1 --direction 0 0 -1 \
  --em-backend kokkos --threads 2 --batch 8 \
  --resident-capacity 256 --device-memory-MiB 512 \
  --output "$HOME/CorsikaData/terrain/photon_entry"
```

This is a low-cost 2 MeV entry/crossing example, not a shower or radio benchmark.
For a scalar comparison use `--em-backend proposal` and a new output name. For
a CUDA build use its installed executable, `--threads 1 --device 0`; do not
switch a CPU binary to GPU by changing a label. Both modes need their own
material-cache initialization on first use. A budget rejection must not be
worked around by suppressing checks.

Inspect `terrain_run.yaml`: completion, actual execution space, material
transitions, pending particles, unsupported-physics counts and any truncated
diagnostics. A process returning or producing a CSV does not by itself establish
complete transport. `--track-row-limit` limits recorded diagnostics;
`--transport-step-limit` is a failure budget; `--transport-window-ns` is an
optional physical-time truncation. Time-limit survivors are not deposited energy.
Use `--energy-ledger` for a separate, more expensive accounting diagnostic.

## 4. Optional interface radio and neutrinos

The DEM application now accepts `--radio` with `--em-backend kokkos`. It computes
both CoREAS and ZHS using the scene's `radio:` settings. This is separate from
air's `--radio-backend kokkos`. Prepare valid observers first. A small initial
test can repeat section 3 with `--radio` and a new output name; signals may be
negligible at 2 MeV. A scientific run needs an appropriate primary and explicit
observer/window/convergence checks, not merely a larger particle energy.

The published path accumulates moments on the selected device and reconstructs
waveforms on the host. Read the [radio setup and limits](interface_kokkos_radio_CN.md)
for `samples`, `sample_rate_GHz`, `start_ns`, `moment_order`, subdivision and
`memory_MiB`. The transport budget includes radio allocations; the separate radio
budget is not additional free memory. The path model has straight legs and at
most one transmitted interface, not arbitrary reflection, multiple crossings,
diffraction or full-wave propagation. Moment/truncation and optical convergence
must be tested; this guide does not certify waveform equivalence.

Neutrinos use `--primary nu_tau` (or another accepted flavor), a suitable
`--energy-GeV`, and a geometry-checked injection point/direction. Without force
flags interactions are sampled naturally; a short rock chord usually yields
no interaction. `--force-vertex-cc`/`--force-vertex-nc` instead require an in-rock
vertex and generate conditional cases, not event-rate samples. The default
channels are CC+NC; TAUOLA is the default tau decay provider. Refer to
[model coverage and original-module alignment](terrain_original_neutrino_alignment_CN.md).
Coverage, complete-weak-physics, event-polarization and depolarization requirement
flags fail when their requested physics is unavailable; they do not install
missing models. Do not describe a passing transport test as complete neutrino
physics validation.

## 5. Acceptance and portability

Start with the geometry test list and small CPU/OpenMP cases; GPU tests require
explicit resource permission. Production validation is separate from compilation.

```bash
ctest --test-dir build/openmp -N -R 'Terrain|Interface|Mountain'
```

Inspect the selected tests before running them with `--output-on-failure`.
The current scene is one finite closed material volume embedded in native
atmosphere, with supported material cards, not a general multi-volume navigator
or global solid Earth. The prepared-rock altitude/mesh limits still apply.
Check artificial bottom/side boundaries, physical-model coverage, recording
completeness, numerical convergence and CPU/accelerated comparisons for each
new scene. HIP/SYCL remain target-hardware acceptance work, not a promise made
by the existence of a build profile.
