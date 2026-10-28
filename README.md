# CORSIKA8-kokkos

[中文说明](README_CN.md) · [Application and backend guide](documentation/kokkos.rst)

CORSIKA 8 air-shower simulations with optional Kokkos electromagnetic
transport and radio calculation. The native `c8_air_shower` application uses
library interfaces for execution, particle scheduling, environment conversion
and output. This is an independent
research repository, not an official CORSIKA release.

## Latest changes (2026-10-28)

- Reorganize the code into shared Kokkos library modules, with public headers,
  inline implementations and compiled sources in their respective directories.
- Simplify application connections through shared adapters while retaining the
  native process sequence and `EAS.run()` entry. Update the installation helpers
  for this layout and the native application binaries.

## Source layout

```text
applications/c8_air_shower.cpp  native air-shower entry point
corsika/kokkos/                 public library headers
  application/cascade/         shared application adapters and runner
    atmosphere/                air scene policy
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

The following steps start from a fresh Linux account (Ubuntu 22.04/24.04 or WSL2
examples). Choose **Miniconda/Miniforge or venv**, then **OpenMP or CUDA**; you do
not need to build OpenMP before CUDA. Both builds use **FLUKA by default**.
Prepare FLUKA before starting a dependency build. Commands run in Bash.

Requirements: a C++17 compiler (GCC 11+), compatible GFortran, CMake 3.24–3.x,
Conan 2 and Python build tools. Kokkos is pinned to 4.7.03. Allow tens of GiB for
source, dependency caches and builds, plus simulation data; multiple backends
need more space. On WSL check free space on the Windows drive holding the VHDX
as well as `df -h` and `free -h`. Start with **one compile job**.

### 1. System tools and FLUKA

On Ubuntu, install the host tools (no GPU driver is installed by this command):

```bash
sudo apt-get update
sudo apt-get install -y build-essential gfortran git curl ca-certificates \
  python3-venv python3-dev pkg-config autoconf automake libtool bison flex \
  patch unzip bzip2 xz-utils
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
```

Without sudo, use the site's compiler modules or ask the administrator for these
tools. Keep C++, Fortran and the Conan compiler profile consistent; this guide
uses system compilers even when Conda supplies the build tools.

**FLUKA is a separately licensed prerequisite, not downloaded by this script.**
Obtain a compatible Linux/GFortran package from the
[FLUKA collaboration](https://www.fluka.eu/Fluka/www/html/fluka.php?id=ins_run&mm2=3)
and complete its installation first. This C8 interface expects `libflukahp.a`
containing `ndmhep.o`; a differently packaged CERN FLUKA library is not a drop-in
replacement. Keep the complete installation and runtime data, not only the archive.
For an installation in `$HOME/opt/fluka`:

```bash
export FLUPRO="$HOME/opt/fluka"   # replace with your actual installation
export FLUFOR=gfortran
export GFORFLU="$FC"
test -r "$FLUPRO/libflukahp.a"
ar t "$FLUPRO/libflukahp.a" | grep -x ndmhep.o
# For a package supplying the standalone executable:
ldd "$FLUPRO/flukahp"
```

Fix missing files or `not found` libraries before continuing. Use a package
compatible with your GFortran version and retain `FLUPRO` when running C8.
`-DC8_FLUKALIB=/absolute/path/libflukahp.a` can override the link-library path;
it does not replace FLUKA's runtime data. The helper now stops early if the
requested library is absent or incompatible and passes `-DWITH_FLUKA=ON` by default.

If you deliberately want a learning build without FLUKA, append
`-DWITH_FLUKA=OFF` to **both the check and build commands** below. The air-shower
low-energy hadronic model then uses UrQMD. This changes the physics configuration.
The default high-energy model is SIBYLL-2.3d; other models remain selectable.

### 2. Choose one build-tool environment

Miniconda/Miniforge and Python `venv` are alternative ways to install the tools.
Python is still needed for Conan and build-time particle-property generation;
the shower application is a **native binary**, with no Python launcher.
Do not activate Conda and venv simultaneously.

#### Option A: Miniconda or an existing Conda installation

If Conda already works, skip the installer and create the environment below.
Otherwise, for **Linux x86-64** (including x86-64 WSL), download Miniconda:

```bash
mkdir -p "$HOME/Downloads"
cd "$HOME/Downloads"
curl -fLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh
```

Compare the checksum with the same file in the
[official installer index](https://repo.anaconda.com/miniconda/), then run:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
# With the default installation directory:
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

Follow the installer prompts; do not overwrite an existing Conda installation.
For ARM or other systems, select the matching installer from the
[installation guide](https://www.anaconda.com/docs/getting-started/installation).
[Miniforge](https://github.com/conda-forge/miniforge#install) is also suitable;
source its own `etc/profile.d/conda.sh` instead. The remaining commands are the same:

```bash
conda create -n corsika8 --override-channels -c conda-forge \
  python=3.11 pip cmake=3.31 ninja numpy=1.26.4 pyyaml lxml
conda activate corsika8
python -m pip install 'conan>=2,<3' particle==0.25.1 hepunits==2.4.1
```

Use another environment name if `corsika8` already exists. Installing Conda
compilers as well as the system compilers is unnecessary for this route.

#### Option B: Python venv, without Conda

```bash
python3 -m venv "$HOME/venvs/corsika8"
source "$HOME/venvs/corsika8/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'conan>=2,<3' 'cmake>=3.24,<4' ninja \
  particle==0.25.1 hepunits==2.4.1 numpy==1.26.4 pyyaml lxml
```

With either route, check the active tools:

```bash
python -c 'import particle, hepunits, numpy; print("Build Python packages OK")'
cmake --version
conan --version
"$CXX" --version
"$FC" --version
```

In each new terminal, reactivate the selected environment and restore the compiler
and FLUKA variables before building/running. `particle` and `hepunits` are build
requirements, not plotting packages. Optional analysis tools such as `matplotlib`
and `pyarrow` can be installed later.

### 3. Get the source and check Conan

```bash
mkdir -p "$HOME/corsika8-workspace"
cd "$HOME/corsika8-workspace"
git clone --recursive --branch CORSIKA8-kokkos \
  https://github.com/BossL668/CORSIKA8-kokkos.git
cd CORSIKA8-kokkos
git submodule status --recursive
conan remote list
```

A leading `-` in submodule status means it is missing: run
`git submodule update --init --recursive`. GitHub source ZIPs omit submodules;
use a recursive clone for these instructions. Pinned model data and CONEX must
be present, even if you are not using CONEX. IGRF14 is bundled in
`resources/GeoMag/IGRF14.COF` and automatically copied/installed into the standard
`GeoMag` data directory.

The helper creates a **missing** default Conan profile with `conan profile detect`;
it never overwrites an existing one. You can inspect/create it beforehand:

```bash
# Run only if no default profile exists:
conan profile path default || conan profile detect
conan profile show -pr:h default -pr:b default
```

Check that the compiler major version matches `g++ --version`. With GCC, use
`libstdc++11`; the helper requests C++17 for host dependencies. Existing profiles
must be reviewed after changing compilers. If ConanCenter is absent from the
remote list, add it once:

```bash
conan remote add conancenter https://center2.conan.io
```

The helper exports the recipes in `externals/conan/` and builds missing packages,
including Kokkos and PROPOSAL's native-table export interface. This recipe export
is a build operation, not physics-table generation. CMake also downloads/builds
Pythia and TAUOLA by default. You do not need another machine's `build/external`
or preinstalled model paths. Initial installation needs access to the dependency
and model download sites; do not mix old PROPOSAL libraries with new headers.

### 4. NVIDIA toolkit (CUDA users only)

OpenMP users skip this step. CUDA builds need a working NVIDIA driver **and** the
development toolkit containing `nvcc`. Check them separately:

```bash
nvidia-smi
nvcc --version
```

The CUDA version shown by `nvidia-smi` is driver capability, not the installed
compiler version. If a compatible toolkit already exists, use it. Otherwise
follow [NVIDIA's installation guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)
for the toolkit and supported host compiler. In the Conda environment above,
one toolkit option supporting the listed Volta through Blackwell targets is:

```bash
conda install --override-channels -c nvidia/label/cuda-12.8.1 -c conda-forge cuda
nvcc --version
```

This does not install a system driver. WSL uses the Windows NVIDIA driver; follow
[NVIDIA's WSL guidance](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
A runtime-only CUDA package does not replace `nvcc`. Do not mix headers and
libraries from different toolkit installations. V100/V100S need CUDA 12.x for
compilation; CUDA 13 removed these targets. RTX 50 needs CUDA 12.8 or newer.
The helper checks whether the selected `nvcc` supports the target before building.

### 5. Check, build and install

Run from the source directory, with the selected environment active and FLUKA
ready. `--check` prints the selected profile and checks prerequisites without
fetching dependencies, compiling, or installing. Select the backend you need:

```bash
# CPU-only: no NVIDIA driver/toolkit required
bash tools/build_kokkos.sh openmp --check
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh openmp

# NVIDIA: detect the local GPU architecture, then build and install
bash tools/build_kokkos.sh cuda --check
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh cuda
```

The full command performs **data preparation → Conan dependencies → CMake
configuration → compilation → installation**. It prints the binary and data
paths on completion. `C8_BUILD_JOBS=1` limits build concurrency, not simulation
threads. Increase it only when RAM permits. No `sudo make install` is needed.

CUDA auto-detection selects one of the supplied profiles:

| GPU example | Compute capability | Profile |
|---|---|---|
| V100 / V100S | 7.0 | `cuda-volta70` |
| T4 / RTX 20 | 7.5 | `cuda-turing75` |
| A100 | 8.0 | `cuda-ampere80` |
| A10 / RTX 30 | 8.6 | `cuda-ampere86` |
| L20 / RTX 40 | 8.9 | `cuda-ada89` |
| H100 | 9.0 | `cuda-hopper90` |
| RTX 50 | 12.0 | `cuda-blackwell120` |

On a mixed-GPU host, select the physical `nvidia-smi` index/UUID used as the
**build target**. This does not set the device used by a later simulation:

```bash
C8_CUDA_DEVICE=0 bash tools/build_kokkos.sh cuda
```

For a known target on a build host without that GPU, choose the profile explicitly;
a compatible CUDA toolkit is still required:

```bash
C8_KOKKOS_PROFILE=cuda-volta70 bash tools/build_kokkos.sh cuda
```

Unknown/mixed capabilities are rejected rather than guessed. GPU profiles here
inherit the default compiler profile. `C8_KOKKOS_PROFILE` also accepts a custom
profile path; `C8_KOKKOS_OPENMP_PROFILE` and `C8_BUILD_PROFILE` override CPU-host
and build-tool profiles. HIP/SYCL require their toolchains and explicit profiles;
the automatic GPU detection in this helper is for NVIDIA CUDA.

Default layout:

```text
corsika8-workspace/
├── CORSIKA8-kokkos/                 source
├── build/{openmp,cuda}/deps/        independent dependency descriptors
└── install/{openmp,cuda}/
    ├── bin/c8_air_shower            native executable
    └── share/corsika/data/          installed model/geomagnetic data
```

To put builds/installations on a larger disk, set `C8_WORKSPACE=/path/to/workspace`
in every helper invocation. Conan's separate cache is under `~/.conan2` by default;
set `CONAN_HOME=/path/to/conan-cache` **before** detecting profiles/installing
packages if that cache also needs relocating. Inspect disk space before a full
build. Use fresh backend directories after changing compiler, architecture or ABI.

`openmp,cuda` builds both independent installations in sequence. The optional
`cuda-openmp` backend builds one combined binary in a separate prefix; it still
requires a working NVIDIA runtime even when selecting OpenMP. A CPU-only server
should use the independent `openmp` build.

For manual CMake integration with an externally installed Kokkos, see the
[backend guide](documentation/kokkos.rst). Add `-DWITH_FLUKA=ON` to manual CMake
commands when using the default scientific configuration described here.

## Run the native binary

Start with one **10 GeV photon** to check installation and output; this is not a
performance or physics acceptance test. From the workspace directory, run the
example for the backend you installed. Keep the build environment activated and
`FLUPRO` set. The final output directory must not already exist:

```bash
cd "$HOME/corsika8-workspace"   # or your C8_WORKSPACE
mkdir -p results

# OpenMP installation
export CORSIKA_DATA="$PWD/install/openmp/share/corsika/data"
install/openmp/bin/c8_air_shower --help
install/openmp/bin/c8_air_shower --pdg 22 --energy 10 --nevent 1 \
  --seed 18641 --filename results/smoke-openmp \
  --em-backend kokkos-proposal --kokkos-execution openmp \
  --kokkos-num-threads 1 --radio-backend kokkos

# CUDA installation
export CORSIKA_DATA="$PWD/install/cuda/share/corsika/data"
install/cuda/bin/c8_air_shower --help
install/cuda/bin/c8_air_shower --pdg 22 --energy 10 --nevent 1 \
  --seed 18641 --filename results/smoke-cuda \
  --em-backend kokkos-proposal --kokkos-execution cuda --device 0 \
  --gpu-memory-fraction 0.70 --radio-backend kokkos
```

For either run, check that it exits successfully and inspect the output directory
and run log. Low-energy radio can be weak or empty; that alone is not an
installation failure. The first run may spend time preparing native PROPOSAL
tables and caches; subsequent runs reuse compatible data. No hand-prepared
`.c8emrt` file is needed. Retain the same environment for shared-library lookup:
`ldd install/cuda/bin/c8_air_shower` (or the OpenMP path) should show no `not found`.

The default EM/radio selections are scalar PROPOSAL/CPU; select Kokkos explicitly
as above. `--pdg 2212 --energy 1e4` instead requests a 10 TeV proton shower. Extra
photons from positrons removed at the tracking cut are disabled by default;
in-flight annihilation is retained. Use `--help` for physical settings and outputs.

For atmospheric showers, `--device 0,1,2,3` distributes one shower across four
GPUs. `--prefix-workers 16` enables the independent CPU prefix stage.
`--gpu-min-batch` and `C8_STATIC_FRONTIER_IMPORT_BATCH` control dispatch and
static import granularity. These scheduling settings do not change physics cuts.
See the [backend guide](documentation/kokkos.rst) for combinations and limitations.

## Checks, troubleshooting and updates

```bash
# From the source directory; use your actual build path if C8_WORKSPACE differs
ctest --test-dir ../build/openmp --output-on-failure -j1
# CUDA tests require a working GPU and the CUDA build directory.
```

| Symptom | What to check |
|---|---|
| `conan`, `cmake` or Python module missing | Reactivate the chosen Conda/venv environment; install the build packages above. |
| FLUKA missing or incompatible | Check `FLUPRO`, `libflukahp.a`, `ndmhep.o` and the package's GFortran compatibility. |
| `nvcc` missing / target unsupported | Install the full compatible toolkit; check `nvcc --version` separately from `nvidia-smi`. |
| Mixed GPU architectures | Use `C8_CUDA_DEVICE` or an explicit target profile; use a separate build for another target. |
| Conan downloads fail | Check the remote list and network/proxy settings; rerun the same command after fixing access. |
| `Killed`, OOM or disk-full error | Keep `C8_BUILD_JOBS=1`, check RAM, build/Conan-cache space and the Windows disk on WSL. |
| Runtime library `not found` | Activate the build environment; inspect `ldd`, without copying arbitrary libraries from another installation. |
| Data file / IGRF error | Set `CORSIKA_DATA` to the installed backend's `share/corsika/data`; check recursive submodules. |
| Output directory already exists | Choose a new `--filename`; do not overwrite a previous shower. |

To update an existing checkout, first review local edits with `git status`, then:

```bash
git pull --ff-only
git submodule update --init --recursive
bash tools/build_kokkos.sh openmp   # or the backend you installed
```

Use new build/install directories when changing compilers, architecture or the
dependency ABI. Keep simulation data outside source/build directories. Rebuilding
a backend does not require deleting unrelated simulation results.

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
