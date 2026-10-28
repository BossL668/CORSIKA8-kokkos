# CORSIKA8-kokkos

[中文说明](README_CN.md) · [Application and backend guide](documentation/kokkos.rst)

CORSIKA 8 particle-cascade simulations with optional Kokkos electromagnetic
transport and radio calculation. The native `c8_air_shower`, `c8_ice_cascade`
and `c8_firn_cascade` applications share library interfaces for execution,
particle scheduling, environment conversion and output. This is an independent
research repository, not an official CORSIKA release.

## Latest changes (2026-10-28)

- Reorganize the code into shared Kokkos library modules, with public headers,
  inline implementations and compiled sources in their respective directories.
  Atmosphere, ice and firn scene adapters use the same directory structure.
- Simplify application connections through shared adapters while retaining the
  native process sequence and `EAS.run()` entry. Update the installation helpers
  for this layout and the native application binaries.

## Source layout

```text
applications/                  native air, ice and firn entry points
corsika/kokkos/                 public library headers
  application/cascade/         shared application adapters and runner
    atmosphere/                air scene policy
    ice/                       homogeneous ice scene policy
    firn/                      firn scene policy
  environment/                 C8 environment export and geometry search
  em/                          electromagnetic models and native table queries
  radio/                       shared radio support and medium-specific propagation
  scheduling/                  particle exchange and CPU/device coordination
corsika/detail/kokkos/          corresponding inline/template implementations
src/kokkos/                    compiled library implementations
externals/conan/               dependency recipes and backend profiles
tests/kokkos/                  subsystem tests
```

See [library responsibilities](corsika/kokkos/README.md). The application constructs
the environment, models and observers; `CascadeExtension` connects those objects
to the selected backend. Numerical implementations and scheduling stay in the
library. Public headers are `.hpp`; inline implementations are `.inl` under
`corsika/detail/`. There is no Python application launcher.

## Install on Linux

Use a C++17 compiler (GCC 11 or newer), a compatible Fortran compiler, CMake 3.24+
and Conan 2. NVIDIA builds additionally need a compatible CUDA toolkit/driver.
Kokkos is pinned to 4.7.03. HIP/SYCL require their respective toolchains and
matching profiles; their availability is not a claim of validation on every GPU.

Install the system development tools (Ubuntu example):

```bash
sudo apt-get install build-essential gfortran git python3-venv \
  autoconf automake libtool bison flex pkg-config patch curl unzip bzip2 xz-utils
python3 -m venv "$HOME/venvs/corsika8"
source "$HOME/venvs/corsika8/bin/activate"
python -m pip install 'conan>=2,<3' 'cmake>=3.24,<4' particle==0.25.1 numpy
conan profile detect
mkdir -p "$HOME/corsika8-workspace"
cd "$HOME/corsika8-workspace"
git clone --recursive --branch CORSIKA8-kokkos \
  https://github.com/BossL668/CORSIKA8-kokkos.git
cd CORSIKA8-kokkos
```

Run `conan profile detect` only when creating a new profile; retain an existing
configured compiler profile. `C8_BUILD_JOBS` defaults to **1** to limit memory use.
The helpers export the supplied dependency recipes, prepare the normal C8 data
directory (including IGRF14), and build/install all three application binaries.
Each backend has independent dependency descriptors, build and install paths.
IGRF14 coefficients are included in `resources/GeoMag/IGRF14.COF` and installed
automatically in the normal `GeoMag` data directory; no separate download is needed.

```bash
# CPU-only installation
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh openmp

# NVIDIA installation; detect a supported local GPU architecture
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh cuda

# Explicit architecture, including cross-compilation without a local GPU
C8_KOKKOS_PROFILE=cuda-volta70 bash tools/build_kokkos.sh cuda  # V100/V100S
# cuda-ada89: L20/RTX 40; cuda-ampere80: A100
# cuda-blackwell120: RTX 50, CUDA 12.8 or newer
```

Additional CMake options follow the backend name. For FLUKA, obtain a licensed
installation separately and set `FLUPRO` to the directory containing
`libflukahp.a`; the Fortran/compiler versions must be compatible:

```bash
export FLUPRO=/path/to/fluka
export FLUFOR=gfortran
bash tools/build_kokkos.sh openmp -DWITH_FLUKA=ON
```

Without `-DWITH_FLUKA=ON`, the configured native low-energy model is used.
Pythia and other native model requirements remain part of the C8 build. Avoid
reusing a build directory after changing compilers, backend, or dependency ABI.
The optional combined CUDA/OpenMP build is selected with
`tools/build_kokkos.sh cuda-openmp`; use a separate OpenMP build on CPU-only hosts.

Default workspace layout:

```text
corsika8-workspace/
├── CORSIKA8-kokkos/                 source
├── build/{openmp,cuda}/deps/        independent dependency descriptors
└── install/{openmp,cuda}/
    ├── bin/c8_air_shower
    ├── bin/c8_ice_cascade
    ├── bin/c8_firn_cascade
    └── share/corsika/data/
```

`C8_WORKSPACE` changes the parent of `build/` and `install/`. The helper accepts
`openmp,cuda` to build both sequentially. Explicit profiles live in
`externals/conan/profiles/`; `C8_KOKKOS_PROFILE`, `C8_KOKKOS_OPENMP_PROFILE` and
`C8_BUILD_PROFILE` override host/build profile selection.

For an existing externally installed Kokkos, use the manual CMake procedure in
the [backend guide](documentation/kokkos.rst). For scalar C8 only, prepare data,
run `conan-install.sh` without `--with-proposal-export`, and configure with
`CORSIKA_ENABLE_KOKKOS=OFF`.

## Run the native binary

From the workspace directory:

```bash
export CORSIKA_DATA="$PWD/install/openmp/share/corsika/data"
install/openmp/bin/c8_air_shower --pdg 2212 --energy 1e4 --nevent 1 \
  --seed 18641 --filename shower-openmp \
  --em-backend kokkos-proposal --kokkos-execution openmp \
  --kokkos-num-threads 1 --radio-backend kokkos

export CORSIKA_DATA="$PWD/install/cuda/share/corsika/data"
install/cuda/bin/c8_air_shower --pdg 2212 --energy 1e4 --nevent 1 \
  --seed 18641 --filename shower-cuda \
  --em-backend kokkos-proposal --kokkos-execution cuda --device 0 \
  --gpu-memory-fraction 0.90 --radio-backend kokkos
```

The default EM/radio selections are scalar PROPOSAL/CPU; select Kokkos explicitly.
Use each binary's `--help` for its scene options. Additional photons from positrons
removed at the tracking cut are disabled by default; in-flight annihilation is
retained. PROPOSAL prepares/reuses native tables automatically; no manually
prepared `.c8emrt` file is required.

For atmospheric showers, `--device 0,1,2,3` distributes one shower across four
GPUs. `--prefix-workers 16` enables the independent CPU prefix stage.
`--gpu-min-batch` and `C8_STATIC_FRONTIER_IMPORT_BATCH` control dispatch and
static import granularity. These scheduling settings do not change the physics
cuts. See the [backend and scene guide](documentation/kokkos.rst) for supported
combinations, ice/firn propagation, and limitations.

## Checks and updates

```bash
ctest --test-dir ../build/openmp --output-on-failure -j1
# CUDA tests require a working GPU; choose the corresponding build directory.
git pull --ff-only
git submodule update --init --recursive
bash tools/build_kokkos.sh openmp
```

Use new build/install directories for this code layout. Keep simulation data
outside the source/build directories, and use identical physical settings when
comparing versions. The library module guide documents supported environments
and radio modes; unsupported combinations are rejected explicitly.

## CORSIKA references and license

CORSIKA 8 is distributed under the [BSD 3-Clause license](LICENSE). Additional
models and dependencies retain their own licenses; FLUKA is not redistributed.
Please read [USING_COLLABORATING.md](USING_COLLABORATING.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

When using CORSIKA 8, cite [the framework paper](https://doi.org/10.1007/s41781-018-0013-0)
and [the radio paper](https://doi.org/10.1016/j.astropartphys.2024.103072), together
with the applicable physics models.

## Earlier updates

These entries retain the published timeline and describe the versions released
on their respective dates; use the instructions above for the current layout.

Changes (2026-10-23):

- Reduce global atomic traffic for GPU radio diagnostic counters by accumulating
  them locally and combining them within each team on tiled ZHS/CoREAS+ZHS work.
  CoREAS-only, small-track and sparse-observer paths retain their existing
  counter updates. No new command-line option is required.
- Radio formulas, propagation, waveform accumulation, physics cuts and thinning
  are unchanged. CPU/OpenMP radio execution retains its existing implementation.

Changes (2026-10-20):

- Installation fixes: use versioned Miniforge installer/checksum assets; add
  RTX 50-series compute-capability 12.0 profiles (`BLACKWELL120`, CUDA 12.8+).
- Preserve Conan's backend and architecture selection when configuring with
  a toolchain file, so the build flags and installed manifest agree.
- Fix compilation of read-only particle adapters when stopped-positron
  annihilation is disabled; its default and all physics algorithms are unchanged.
- Clarify FLUKA checks and first CUDA-run diagnostics. See the
  [installation feedback and fixes](documentation/installation_feedback_20261020_CN.md).

Changes (2026-10-19):

- Optional **static CPU-parallel prefix**: `--prefix-workers N` advances
  independent CPU subshowers before handing their combined EM/muon frontier
  to the selected GPUs. The CPU process count is independent of the GPU count;
  both one-GPU and multi-GPU continuation are supported with `kokkos-proposal`.
  The default remains the serial prefix; no dynamic redistribution is added.
- `--prefix-split-energy` controls the CPU splitting threshold, not a physics
  cut. `C8_STATIC_FRONTIER_IMPORT_BATCH` optionally controls static GPU-root
  import chunks independently of `--gpu-min-batch`; unset retains the old rule.
- CPU handoff preserves particle species (including nuclei and hadrons), kinetic
  energy, weights, transport identity and logical environment ownership. CPU
  and GPU descendants use disjoint history ranges; signed radio fields and
  profiles include the bootstrap and every completed worker exactly once.
- The coordinator no longer reserves the large shower stack frame, and SHA256
  file reads use a heap buffer. Added handoff, namespace and small-stack tests.
  Physics models, CDF algorithms, cuts, thinning and stopped-positron defaults
  are unchanged. Parallel splitting changes subsequent random histories and
  may be slower for low-energy events; it is not enabled automatically.

See [prefix options and examples](documentation/native_multigpu_CN.md#静态-cpu-并行前缀2026-10-19).

Changes (2026-10-15):

- The air-shower Kokkos path now exports C8's environment-node hierarchy,
  including parent/child and excluded-node relations, and shares C8's ordered
  boundary-search logic. The observation plane remains a separate step limit.
- Curved sphere intersections use stable quartic factorization; the parallel-field
  transport projection is corrected. No per-step residual scan or bisection
  fallback is introduced. This does not promise bitwise-identical trajectories.
- Logical environment-node IDs survive device queues, CPU fallback and static
  multi-GPU handoff. Full-tree runs require the new V2 frontier format; regenerate
  older V1 frontiers before using them with this path.
- Extra two-photon emission from positrons removed at the energy cut is now
  **disabled by default**, matching original C8, on both scalar and shared Kokkos
  cut paths. **In-flight positron annihilation remains enabled.** Rebuild both
  CPU and GPU binaries; do not mix samples with different stopped-cut policies.
- The existing native-CDF policy is retained: photon pair production and muon
  bremsstrahlung use native-table inversion on the accelerated path; electron
  and muon direct electron-pair final states use CPU PROPOSAL because their
  conditional asymmetry tables are unavailable. No pair-splitting rejection
  sampler or experimental CDF-derivative/inverse-loss optimization is enabled.

Application interfaces retained from the previous release:

- The air application selects EM physics with `--em-backend proposal`,
  `kokkos-proposal` or `kokkos-egs4`; execution hardware is selected separately.
- Unified `--device`: one GPU ID selects a single-GPU shower; multiple IDs
  select native C++ multi-GPU coordination for the same shower.
- The optional C++/Kokkos EGS4 backend is a regular CORSIKA module, with its
  AIR-NTP table bundled and embedded at build time. No external C7 installation
  or runtime table-path argument is required.
- Application bindings and run sessions encapsulate backend setup and reuse;
  the main air-shower application remains the entry point for both EM backends.

Tiny-batch CPU fallback and tail-repartitioning experiments are not production
defaults. Module-level speedups must not be interpreted as whole-shower speedups.

Air application update (2026-09-23): `c8_air_shower` now defaults to
`--profile-crossings original-c8` (forward crossings, closed endpoints).
`forward` selects forward half-open counts; `both` selects positive weighted
crossings in either direction. CPU writers and Kokkos CUDA/OpenMP share the
policy; transport, radio tracks, cuts and deposited energy are unchanged.
Rebuild and reinstall; do not mix profile cohorts using different conventions.
See [counting semantics and runnable examples](documentation/profile_crossings_CN.md).

Historical note (2026-09-20): an additional stopped-positron two-photon model
was introduced at the energy cut. Its implementation is retained, but the
2026-10-15 default disables it to match original C8. The
[original implementation note](documentation/stopped_positron_annihilation_20260920_CN.md)
describes the earlier enabled policy, not the current default.
