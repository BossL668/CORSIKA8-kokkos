# CORSIKA 8 GPU Hybrid Cascade

This repository is a research fork of CORSIKA 8 that adds an optional NVIDIA
CUDA backend for high-energy air-shower production. It is being developed for
21CMA cosmic-ray and radio-emission studies, with an emphasis on accelerating
individual proton showers while retaining the established CORSIKA 8 physics
modules.

The original scalar `Cascade + PROPOSAL` execution path remains available and
is the default. CUDA transport, CUDA radio projection, and the process-isolated
FLUKA pool are explicit opt-in features.

> **Project status:** this is an independent research fork. It has not been
> reviewed or accepted by the CORSIKA Collaboration. Use it for scientific
> production only within the supported and validated configurations described
> below.

> **Current branch recommendation (2026-08-09):** beta4 is the maintained
> beta2-derived validation branch. It preserves the beta2 scheduler and GPU
> performance model while repairing the audited CPU/GPU semantic differences:
> charged-particle endpoint-chord grammage and per-PID cuts, the locally flat
> observation plane, native-GPU first-interaction output, the scalar 10 ms
> physical-time cut, and forced-primary interaction/decay handling. CUDA
> startup now also rejects any unregistered process category before a shower
> begins.
> The separate beta3 branch adds experimental hadronic multiprocessing and is
> not the reference for this repair. The complete beta4 change and evidence
> index is given below.

[Chinese README](README_CN.md) ·
[GPU production guide](documentation/cuda_em_refactor/cuda_em_backend_user_guide.md) ·
[CLI reference](documentation/cuda_em_refactor/cli_reference.md) ·
[validation tools](validation/gpu_em/README.md) ·
[upstream CORSIKA 8](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika)

## What this fork adds

The fork separates scalar particle stepping from cascade scheduling and adds a
hybrid CPU/GPU execution path. The main additions are:

| Capability | Implementation |
|---|---|
| CUDA electromagnetic transport | Device-resident wavefront transport for photons, electrons, and positrons |
| CUDA charged-lepton transport | Table-enabled muon transport with CPU return for decay and unsupported final states |
| Hybrid scheduling | Stable routing between the original CPU stack and independent GPU structure-of-arrays queues |
| Physics tables | Versioned PROPOSAL-derived rate, inverse-CDF, continuous-loss, LPM, and scattering tables |
| Automatic table preparation | Material YAML normalization, content hashing, cache lookup, locked generation, and validation |
| Radio calculation | Selectable CPU or CUDA CoREAS/ZHS waveform projection for electromagnetic tracks |
| Low-energy hadronic throughput | Optional persistent multi-process FLUKA final-state pool |
| Reproducibility | History-addressed Philox random numbers and exact scalar decision-tape replay tools |
| Scientific audit | Fail-closed output metadata, timing, fallback counters, energy accounting, and ensemble-validation tools |

Unsupported or rare processes are never silently discarded. They are either
returned to a declared CPU physics module or treated as a hard failure.

### Execution model

```text
primary particle
      |
      v
CPU CORSIKA 8: primary hadrons, interactions, decays, unsupported final states
      |
      +---------------- gamma / e- / e+ ----------------+
      |                                                  |
      +------------- table-enabled mu- / mu+ --------+   |
                                                     v   v
                                                CUDA wavefronts
                                                     |
                    +--------------------------------+------------------+
                    |                                |                  |
                    v                                v                  v
             CPU final-state fallback       longitudinal/profile   radio tracks
                    |                         accumulation          |
                    v                                               v
             CORSIKA CPU stack                              CPU or CUDA CoREAS/ZHS
```

The `fluka-process` backend does not move hadronic physics to the GPU. It runs
the same low-energy FLUKA final-state calculation in persistent worker
processes and batches homogeneous requests through binary IPC.

## Beta4 repair and validation status

Beta4 was copied from beta2 so that the proven beta2 wavefront scheduler could
be retained while the scalar/CUDA physics contracts were audited independently.
The following changes are specific to beta4; none changes the default scalar
backend.

| Contract audited in beta4 | Resolution | Evidence |
|---|---|---|
| Continuous grammage in a magnetic step and particle-dependent transport cuts | Continuous loss and scattering now use the chord between the stored step endpoints, as scalar `Step::getStraightTrack()` does; electron and muon cuts are resolved independently | [Phase 97](documentation/cuda_em_refactor/phase_97_beta4_cpu_step_chord_grammage.md) |
| Observation geometry | GPU transport now intersects the same locally flat plane through the shower core as the scalar application; spherical surfaces remain atmosphere boundaries only | [Phase 98](documentation/cuda_em_refactor/phase_98_beta4_cpu_observation_plane_alignment.md) |
| First physical interaction output | An unthinned generation-zero GPU snapshot is replayed through the ordinary `InteractionWriter` | [Phase 99](documentation/cuda_em_refactor/phase_99_beta4_gpu_first_interaction_writer_alignment.md) |
| Delayed-particle cut | Photon and charged-lepton paths implement the scalar strict `timePost > 10 ms` physical-time condition without truncating the crossing step | [Phase 100](documentation/cuda_em_refactor/phase_100_beta4_particle_cut_time_alignment.md) |
| Forced primary and custom process semantics | A requested primary interaction or decay executes once before GPU routing; all six process categories are checked by a fail-closed compatibility registry | [Phase 101](documentation/cuda_em_refactor/phase_101_beta4_forced_primary_and_process_compatibility_gate.md) |
| Performance regression | Three paired 10 PeV proton runs put beta4 within about 1--3% of beta2 on the same RTX 4060 Laptop GPU | [Phase 102](documentation/cuda_em_refactor/phase_102_beta4_beta2_10pev_performance.md) |
| Final code-path audit and 100 TeV gate | No further production-blocking omitted or reordered contract was found; the canonical beta2 CPU 500-event reference was verified and the matching beta4 CUDA campaign was started | [Phase 103](documentation/cuda_em_refactor/phase_103_beta4_gpu_cpu_path_reaudit_and_100tev_campaign.md) |

The completed automated gate is 34/34 C++/CUDA tests and 241/241 Python
validation tests with the FLUKA runtime environment present. A 500-versus-500
10 GeV electron diagnostic also completed; its shower curves were statistically
consistent, while the radio signal was too close to the numerical floor to be
a sufficient high-energy radio-shape acceptance sample. The canonical 100 TeV
proton beta4 campaign is still in progress, so its partial sample must not be
reported as a final beta4 production result.

## Supported production contract

The current CUDA application is intended for the following configuration:

- NVIDIA CUDA 12.x on Linux or WSL2;
- a five-layer spherical dry-air atmosphere;
- the locally flat observation plane used by `c8_air_shower`, independent of
  the spherical atmosphere-layer boundaries;
- a uniform magnetic-field vector calculated by the host application;
- photon, electron, positron, and table-enabled muon transport;
- electromagnetic cuts, including the scalar `timePost > 10 ms` delayed-particle
  condition, and the existing CORSIKA 8 EM thinning algorithm;
- CPU or CUDA CoREAS/ZHS projection;
- CPU high-energy hadronic physics and FLUKA 2025 low-energy interactions;
- double-precision particle state, geometry, and physics-table calculations;
- deterministic same-machine CUDA repetition without `--use_fast_math`.

The reference `c8_air_shower` application defaults to `GeoMag/IGRF14.COF` and
the 21CMA site field for the year 2027 at latitude 42.5527 degrees, longitude
86.4153816422 degrees, and altitude 2680.444195 m. `--geomagnetic-model`
selects IGRF13 or IGRF14 and `--geomagnetic-year` selects the epoch. Both
scalar and CUDA paths receive the same resolved field vector, and the selected
model/year are stored in the output metadata. IGRF14 is bundled under
`resources/GeoMag` and is installed automatically; an existing
`CORSIKA_DATA/GeoMag/IGRF14.COF` takes precedence.

Rock, soil, ice, lunar regolith, mountains, and arbitrary three-dimensional
media are not end-to-end supported yet. Their material tables can be prepared,
but device geometry, grammage integration, medium-ID mapping, and dedicated
physics validation must also be implemented before such a table can be used in
`c8_air_shower`.

## Requirements

### Host software

- a working Miniconda, Anaconda, or compatible Conda installation;
- Linux or WSL2;
- CMake 3.24 or newer for a CUDA build;
- a C++17 and Fortran toolchain;
- Conan 2;
- Python 3.9 or newer for the validation utilities;
- Git with submodule support;
- NVIDIA CUDA toolkit 12.x, including `nvcc`;
- a host NVIDIA driver compatible with the selected CUDA toolkit;
- FLUKA and its data files for the production low-energy hadronic path.

The Conda recipe below installs the project-level tools and Python packages,
but it deliberately uses the native host GCC/G++/GFortran family. On a fresh
Ubuntu or WSL installation, install that host toolchain before creating the
Conda environment:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  gfortran \
  git \
  ca-certificates
```

On a managed server without `sudo`, load the site's mutually compatible GCC,
G++, GFortran, and CUDA modules instead. Do not mix a Conda C++ compiler with
an unrelated system Fortran compiler. The initial dependency build also needs
outbound HTTPS access to Conda channels, Conan Center, the private GitHub fork,
the public KIT submodules, and the bundled Pythia/Tauola source archives.

Check the machine before building:

```bash
nvidia-smi
nvidia-smi \
  --query-gpu=index,name,compute_cap,driver_version,memory.total \
  --format=csv
gcc --version
g++ --version
gfortran --version
make --version
git --version
ldd --version
free -h
df -h .
```

In WSL2, install the NVIDIA display driver on Windows. Install only the CUDA
toolkit inside WSL; a second Linux display driver is neither required nor
recommended.

The `CUDA Version` printed by `nvidia-smi` is the newest CUDA driver API the
installed driver can support; it is not the toolkit used to compile this
project. `nvcc --version` is the authoritative compiler check. A driver that
reports CUDA 13.x can run a binary built with the supported CUDA 12.6 toolkit.

### Reference Conda environment

The following commands start from a completely new environment. They include
the Python packages required by the validation scripts, including Parquet and
`pytest` support; these are not supplied by the C++ Arrow package that Conan
builds later.

If `conda activate` is not yet available in the current non-interactive or SSH
shell, initialize it without modifying the shell startup files:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

```bash
conda create -n corsika_venv \
  -y \
  -c conda-forge \
  python=3.9 \
  cmake=3.31 \
  conan=2.11 \
  pip \
  git \
  make \
  pkg-config \
  particle=0.25.1 \
  numpy pandas scipy matplotlib pyyaml pyarrow pytest

conda activate corsika_venv
conda config --env --set channel_priority strict

conda install \
  -y \
  -c nvidia/label/cuda-12.6.3 \
  -c conda-forge \
  cuda-nvcc=12.6.85 \
  cuda-cudart-dev=12.6.77 \
  cuda-cccl=12.6.77
```

The three CUDA packages above are the tested minimum for this project. If a
site requires the full toolkit and the historical NVIDIA label can still
resolve it, the following may be used instead of those three packages:

```bash
conda install \
  -y \
  -c nvidia/label/cuda-12.6.3 \
  -c conda-forge \
  cuda-toolkit=12.6.3
```

Do not install both alternatives merely to repair a failed solve. Remove the
failed transaction and use the tested minimum set. Verify that all commands
are coming from the intended environment and host toolchain:

```bash
test "$CONDA_DEFAULT_ENV" = corsika_venv

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAHOSTCXX="$CXX"

for program in "$CC" "$CXX" "$FC" "$CUDACXX" cmake conan python make git; do
  test -x "$(command -v "$program")" || {
    printf 'Missing required program: %s\n' "$program" >&2
    exit 1
  }
done

"$CC" --version | head -n 1
"$CXX" --version | head -n 1
"$FC" --version | head -n 1
"$CUDACXX" --version
cmake --version | head -n 1
conan --version
python -c 'import numpy, pandas, pyarrow, pytest, scipy, yaml; print("Python dependencies: OK")'
```

The explicit `/usr/bin` paths are the tested Ubuntu/WSL choice. Replace all
three together with the paths supplied by a cluster compiler module when
needed. GCC, G++, and GFortran must be from the same major toolchain family.

On a managed cluster, loading the site's compiler and CUDA modules is usually
preferable to installing the toolkit in Conda. Use the same compiler family
when Conan creates its profile and when CMake builds the project.

`conan-install.sh` creates the project profile after these variables have been
set. Do not create the profile with one compiler and later configure CMake with
another one.

### FLUKA

FLUKA is licensed separately and is not distributed by this repository.
Download a FLUKA binary and data package compatible with the machine's glibc
and Fortran runtime, build it, and expose the installation before configuring
CMake:

```bash
export C8_FLUKA_ROOT=/path/to/fluka
export FLUPRO="$C8_FLUKA_ROOT"
export FLUFOR=gfortran

test -f "$C8_FLUKA_ROOT/libflukahp.a"
```

`-DWITH_FLUKA=ON` is fail-closed: configuration stops if a valid FLUKA library
cannot be found. `-DWITH_FLUKA=OFF` remains available for deliberate UrQMD
experiments, but it is not the reference configuration used for the results in
this repository. `FLUPRO` and `FLUFOR` are also runtime variables: export them
again in every new login shell before starting `c8_air_shower`. A successful
link alone does not make the FLUKA data directory discoverable at runtime.

## Build from source

Run steps 1--4 in the same activated `corsika_venv` shell so the selected
compiler, CUDA architecture, FLUKA, and workspace variables remain defined. If
the login session changes, repeat the relevant `export` commands before
continuing; do not rely on variables inherited from an older build.

### 1. Create the workspace and clone the fork

The recommended layout places the source, out-of-source build, and installation
directories under one dedicated parent directory, for example
`~/corsika-21cma-cuda`:

```text
corsika-21cma-cuda/
├── corsika8_gpu_refactor/              # Git source tree
├── corsika8_gpu_refactor_build_cuda/   # CMake build tree
└── corsika8_gpu_refactor_install_cuda/ # Installed programs and resources
```

Create the parent first and clone the beta4 branch explicitly. The GitHub fork
is private, so the recommended command assumes that an SSH key with repository
access has already been added to the user's GitHub account:

> **Deployment note:** `cuda-em-icrc2025-beta4` currently exists as a local
> working branch and has not yet been published to the private GitHub remote.
> The clone commands below become valid after that branch and its reviewed
> changes are pushed. Until then, use this complete checkout or transfer the
> complete beta4 source tree; cloning beta2 does not include the repairs listed
> above.

```bash
export C8_WORKSPACE=~/corsika-21cma-cuda
mkdir -p "$C8_WORKSPACE"
cd "$C8_WORKSPACE"

ssh -T git@github.com

git clone \
  --branch cuda-em-icrc2025-beta4 \
  --single-branch \
  git@github.com:BossL668/corsika8-gpu-hybrid.git \
  corsika8_gpu_refactor
```

GitHub's successful SSH test reports that it does not provide shell access;
that message is expected. If SSH is unavailable, install and authenticate the
GitHub CLI, then clone the same branch:

```bash
conda install -y -c conda-forge gh
gh auth login
gh auth status

cd "$C8_WORKSPACE"
gh repo clone BossL668/corsika8-gpu-hybrid \
  corsika8_gpu_refactor \
  -- --branch cuda-em-icrc2025-beta4 --single-branch
```

Do not add `--recursive` to either GitHub clone command. The upstream project
stores relative submodule URLs intended for its KIT GitLab origin. Relative to
the GitHub fork they resolve to non-existent GitHub repositories. Override the
two URLs with their public upstream locations and then initialize them:

```bash
export C8_SOURCE="$C8_WORKSPACE/corsika8_gpu_refactor"

git -C "$C8_SOURCE" config \
  submodule.modules/data.url \
  https://gitlab.iap.kit.edu/AirShowerPhysics/corsika-data.git
git -C "$C8_SOURCE" config \
  submodule.modules/conex.url \
  https://gitlab.iap.kit.edu/AirShowerPhysics/cxroot.git

git -C "$C8_SOURCE" submodule update --init --recursive --jobs 8
git -C "$C8_SOURCE" submodule status --recursive

test -f "$C8_SOURCE/modules/data/CMakeLists.txt"
test -f "$C8_SOURCE/modules/conex/cxroot/CMakeLists.txt"
test "$(git -C "$C8_SOURCE" branch --show-current)" = \
  cuda-em-icrc2025-beta4
```

Define all three paths once and keep them unchanged throughout configuration,
compilation, installation, and validation:

```bash
export C8_SOURCE="$C8_WORKSPACE/corsika8_gpu_refactor"
export C8_BUILD="$C8_WORKSPACE/corsika8_gpu_refactor_build_cuda"
export C8_INSTALL="$C8_WORKSPACE/corsika8_gpu_refactor_install_cuda"
cd "$C8_SOURCE"
```

On another machine, change only `C8_WORKSPACE` to the desired absolute parent
path; preserve the three child-directory names shown above. Never place the
build tree inside the Git source tree, and never reuse a build tree copied from
another CUDA toolkit, compiler, or GPU architecture.

### 2. Select the CUDA architecture

Compile for the compute capability of the GPU that will run the program.
CMake writes capability 8.9 as architecture `89`:

```bash
export C8_CUDA_ARCHS="$(
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits |
  tr -d ' ' |
  sed 's/\.//g' |
  sort -u |
  paste -sd';' -
)"

printf 'CMake CUDA architectures: %s\n' "$C8_CUDA_ARCHS"
case "$C8_CUDA_ARCHS" in
  ''|*[!0-9\;]*)
    printf 'Could not determine a valid CUDA architecture.\n' >&2
    false
    ;;
esac
```

A homogeneous server should normally use one architecture. A binary intended
for several GPU generations may list multiple architectures separated by
semicolons. If `nvcc` does not recognize a new GPU architecture, upgrade the
CUDA toolkit instead of substituting an unrelated older architecture.

### 3. Install dependencies and configure a Release build

Confirm the recommended sibling-directory layout in the current shell:

```bash
export C8_WORKSPACE=~/corsika-21cma-cuda
export C8_SOURCE="$C8_WORKSPACE/corsika8_gpu_refactor"
export C8_BUILD="$C8_WORKSPACE/corsika8_gpu_refactor_build_cuda"
export C8_INSTALL="$C8_WORKSPACE/corsika8_gpu_refactor_install_cuda"

# Set this from available RAM as well as CPU count. Start lower on a laptop.
export C8_BUILD_JOBS=16
export C8_CONAN_JOBS="$C8_BUILD_JOBS"

# These must still refer to the toolchain used when the Conan profile is made.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAHOSTCXX="$CXX"

test -f "$C8_SOURCE/conanfile.py"
test -f "$C8_FLUKA_ROOT/libflukahp.a"

"$C8_SOURCE/conan-install.sh" \
  --source-directory "$C8_SOURCE" \
  --release

test -f "$C8_SOURCE/conan_cmake/conan_toolchain.cmake"
conan profile show -pr corsika8

cmake \
  -S "$C8_SOURCE" \
  -B "$C8_BUILD" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
  -DCONAN_CMAKE_DIR="$C8_SOURCE/conan_cmake" \
  -DCMAKE_TOOLCHAIN_FILE="$C8_SOURCE/conan_cmake/conan_toolchain.cmake" \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_CUDA=ON \
  "-DCMAKE_CUDA_ARCHITECTURES=$C8_CUDA_ARCHS" \
  -DWITH_FLUKA=ON \
  -DC8_FLUKALIB="$C8_FLUKA_ROOT/libflukahp.a" \
  -DCMAKE_INSTALL_PREFIX="$C8_INSTALL"
```

`Release` is required for performance measurements. The CUDA build deliberately
does not enable fast math. CMake caches compiler and CUDA selections on its
first configuration. If any of them must change, use a new empty build
directory instead of attempting to repair an old cache.

### 4. Build, install, and test

```bash
# testModules initializes FLUKA; fail here with a clear message if a new shell
# lost the runtime variables.
test -f "$FLUPRO/libflukahp.a"

cmake --build "$C8_BUILD" --parallel "$C8_BUILD_JOBS"
cmake --install "$C8_BUILD"

# The GPU subset is the fastest required hardware/build check.
ctest \
  --test-dir "$C8_BUILD" \
  -R '^testGpu' \
  --output-on-failure \
  --parallel "$C8_BUILD_JOBS"

# Run the complete C++/CUDA suite before scientific production.
ctest \
  --test-dir "$C8_BUILD" \
  --output-on-failure \
  --parallel "$C8_BUILD_JOBS"

cd "$C8_SOURCE"
python -m unittest discover \
  -s "$C8_SOURCE/validation/gpu_em/tests" \
  -p 'test_*.py'

for program in \
  c8_air_shower \
  gpu_em_table_prepare \
  gpu_em_tablegen \
  cuda_decision_replay \
  fluka_batch_worker; do
  test -x "$C8_INSTALL/bin/$program" || {
    printf 'Missing installed program: %s\n' "$program" >&2
    exit 1
  }
done

test -d "$C8_INSTALL/share/corsika/data/PROPOSAL"
test -f "$C8_INSTALL/share/corsika/GeoMag/IGRF14.COF"
test -f "$C8_INSTALL/share/corsika/media/air_dry_1_atm.yaml"

if ldd "$C8_INSTALL/bin/c8_air_shower" | grep -q 'not found'; then
  ldd "$C8_INSTALL/bin/c8_air_shower"
  false
fi

"$C8_INSTALL/bin/c8_air_shower" --help >/dev/null
```

CUDA translation units can require several GiB of host RAM during compilation.
Reduce `--parallel` on memory-limited systems.

### 5. Restore the runtime environment in a new shell

Conda activation and exported variables do not survive logout. Before table
preparation or shower production in a new shell, restore the same installation
and the FLUKA runtime directory:

```bash
conda activate corsika_venv

export C8_WORKSPACE=~/corsika-21cma-cuda
export C8_SOURCE="$C8_WORKSPACE/corsika8_gpu_refactor"
export C8_BUILD="$C8_WORKSPACE/corsika8_gpu_refactor_build_cuda"
export C8_INSTALL="$C8_WORKSPACE/corsika8_gpu_refactor_install_cuda"
export C8_FLUKA_ROOT=/path/to/fluka

export FLUPRO="$C8_FLUKA_ROOT"
export FLUFOR=gfortran
export CORSIKA_DATA="$C8_INSTALL/share/corsika/data"
export PATH="$C8_INSTALL/bin:$PATH"

test -x "$C8_INSTALL/bin/c8_air_shower"
test -f "$FLUPRO/libflukahp.a"
nvidia-smi
```

The installed binary contains a runpath to `$C8_INSTALL/lib/corsika`, so a
manual `LD_LIBRARY_PATH` is normally unnecessary. Adding unrelated system or
Conda library directories can instead make the Fortran/FLUKA runtime
inconsistent.

### 6. Run an installed end-to-end smoke check

The unit tests prove that CUDA kernels execute, but they do not initialize the
complete shower application. Before generating a high-energy production
table, prepare a small 100 GeV table and run one installed proton shower. The
first table request calls PROPOSAL and may take several minutes; repeating the
same request reuses the cache.

```bash
export C8_TABLE_CACHE="$C8_WORKSPACE/gpu_em_table_cache"
export C8_SMOKE_TABLE="$(
  "$C8_INSTALL/bin/gpu_em_table_prepare" \
    --medium-yaml "$C8_INSTALL/share/corsika/media/air_dry_1_atm.yaml" \
    --cache-dir "$C8_TABLE_CACHE" \
    --primary-energy-eV 1e11 \
    --energy-margin 1.05 \
    --em-cut-MeV 0.5 \
    --electron-transport-cut-MeV 0.5 \
    --muon-transport-cut-MeV 300 \
    --tolerance 5e-4 \
    --loss-tolerance 5e-4 \
    --nonmonotonic-loss-policy proposal-monotone \
    --print-path-only
)"

test -f "$C8_SMOKE_TABLE"

export C8_SMOKE_ANTENNAS="$C8_WORKSPACE/antennas_smoke.txt"
printf '100 0 0\n' > "$C8_SMOKE_ANTENNAS"

# The output path must not exist before c8_air_shower starts.
export C8_SMOKE_OUTPUT="$C8_WORKSPACE/smoke_$(date +%Y%m%d_%H%M%S)"

"$C8_INSTALL/bin/c8_air_shower" \
  --pdg 2212 \
  --energy 100 \
  --seed 40077 \
  --filename "$C8_SMOKE_OUTPUT" \
  --emthin 0 \
  --geomagnetic-model IGRF14 \
  --geomagnetic-year 2027 \
  --antenna-file "$C8_SMOKE_ANTENNAS" \
  --ring 0 \
  --em-backend cuda \
  --radio-backend cuda \
  --gpu-device 0 \
  --gpu-min-batch 128 \
  --gpu-memory-fraction 0.70 \
  --gpu-table-cache "$C8_SMOKE_TABLE" \
  --gpu-table-tolerance 5e-4 \
  --gpu-deterministic true \
  --gpu-resident-cross-species true

test -f "$C8_SMOKE_OUTPUT/summary.yaml"
test -f "$C8_SMOKE_OUTPUT/gpu_em/summary.yaml"
test -f "$C8_SMOKE_OUTPUT/CoREAS/observers.parquet"
test -f "$C8_SMOKE_OUTPUT/ZHS/observers.parquet"
grep -q 'complete: true' "$C8_SMOKE_OUTPUT/gpu_em/summary.yaml"
```

This check must exit with status zero and report non-zero photon/lepton GPU
steps and radio tracks in `gpu_em/summary.yaml`. It is a portability check, not
a replacement for a multi-seed CPU/CUDA physics-validation campaign.

#### Portability check: NVIDIA T400 4 GB

The workspace, configure, build, table, and run workflow was exercised on
2026-08-06 on an NVIDIA T400 4 GB (compute capability 7.5) with driver
595.71.05. The same beta2 source snapshot was copied to the server because the
private Git remote was not authenticated there; source checkout authentication
was therefore outside this test. An isolated Conda environment used Python
3.9.23, CMake 3.31.8, Conan 2.11, native GCC/G++/GFortran 13.3, and the minimal
CUDA 12.6.3 package set listed above. The Release CUDA+FLUKA build and install
completed with `CMAKE_CUDA_ARCHITECTURES=75`.

All 27 targeted GPU tests passed. They cover table loading, hybrid routing,
fallback handling, real CUDA execution, wavefront queues, process and
final-state sampling, LPM, thinning, multiple scattering, magnetic and
spherical-atmosphere transport, muons, photons, and CUDA radio projection.
A new generator-contract-0.18 dry-air table through 105 GeV was then produced
and loaded successfully; its measured maximum rate and inverse-CDF errors were
`9.996133e-4` and `8.541396e-4` for a requested tolerance of `1e-3`.

Finally, one 100 GeV proton smoke shower was run with SIBYLL, FLUKA, PROPOSAL,
IGRF14/2027, CUDA EM, CUDA radio, and 81 external antennas. It exited normally
in 6.94 s and wrote complete top-level, GPU, CoREAS, and ZHS output groups. The
GPU summary records 3,643 photon steps, 31,534 charged-lepton steps, 5,497 GPU
final states, 29,774 radio tracks, zero CPU fallback, and a 344.4 MiB peak
device allocation. This establishes build and execution portability to the
T400; it is deliberately a smoke test, not a CPU/CUDA physics-equivalence or
performance benchmark. Its dedicated CUDA energy ledger reports incomplete
coverage, and the ordinary total-energy budget differs by -3.51%, so this
low-energy event must not be quoted as a precision-closure validation.

Two practical issues exposed by this test are handled by the current source:
public CUDA headers remain visible to non-CUDA C++ consumers of the GPU
library, and Conda compatibility-sysroot `libm`/`librt` entries are replaced by
the matching native multiarch libraries. The Pythia 8.315 fetch also uses the
official GitLab release archive because the former `pythia.org/download`
archive endpoint now returns 404.

The main installed programs are:

| Program | Purpose |
|---|---|
| `c8_air_shower` | Scalar or hybrid CUDA shower simulation |
| `gpu_em_table_prepare` | Find or generate a content-addressed physics table |
| `gpu_em_tablegen` | Low-level PROPOSAL table generator |
| `cuda_decision_replay` | Replay an exact scalar decision tape on CUDA |
| `fluka_batch_worker` | Persistent FLUKA final-state worker |

### CPU-only compatibility build

CUDA is disabled by default. A CPU-only build does not require a CUDA toolkit:

```bash
export C8_CPU_BUILD="$C8_WORKSPACE/corsika8_gpu_refactor_build_cpu"

cmake \
  -S "$C8_SOURCE" \
  -B "$C8_CPU_BUILD" \
  -DCONAN_CMAKE_DIR="$C8_SOURCE/conan_cmake" \
  -DCMAKE_TOOLCHAIN_FILE="$C8_SOURCE/conan_cmake/conan_toolchain.cmake" \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_CUDA=OFF \
  -DWITH_FLUKA=ON \
  -DC8_FLUKALIB="$C8_FLUKA_ROOT/libflukahp.a"

cmake --build "$C8_CPU_BUILD" --parallel 16
```

## Prepare the CUDA physics table

CUDA showers require a versioned `.c8emrt` file. This file is different from
PROPOSAL's internal interpolation cache. `c8_air_shower` never creates a table
implicitly during a shower; prepare it first with `gpu_em_table_prepare`.

### Recommended automatic workflow

The preparation tool performs the complete workflow:

```text
material YAML
  -> canonical material representation
  -> material and request SHA-256
  -> compatible table lookup
  -> locked generation on a cache miss
  -> read-back validation
  -> manifest and resolved table path
```

Prepare the standard dry-air table for primaries up to \(10^{18}\) eV:

```bash
export C8_TABLE_CACHE="$(dirname "$C8_SOURCE")/corsika8-table-cache"

export C8_TABLE="$(
  "$C8_INSTALL/bin/gpu_em_table_prepare" \
    --medium-yaml "$C8_INSTALL/share/corsika/media/air_dry_1_atm.yaml" \
    --cache-dir "$C8_TABLE_CACHE" \
    --primary-energy-eV 1e18 \
    --em-cut-MeV 0.5 \
    --electron-transport-cut-MeV 0.5 \
    --muon-transport-cut-MeV 300 \
    --tolerance 5e-4 \
    --loss-tolerance 5e-4 \
    --print-path-only
)"

test -f "$C8_TABLE"
printf 'CUDA physics table: %s\n' "$C8_TABLE"
```

The first request may take a long time because the tool calls PROPOSAL and
adaptively validates the interpolation grids. Identical requests, and smaller
compatible energy requests, reuse the content-addressed cache.

`--em-cut-MeV` is the user-facing CORSIKA production/transport cut. The
preparation tool automatically applies the same standard-table selection rule
as the scalar PROPOSAL backend. Consequently, a requested 0.5 MeV cut is stored
as a 0.5 MeV transport cut but uses a 0.4 MeV stochastic PROPOSAL table. The
manifest records both values. This distinction is required for CPU/CUDA
agreement, most visibly for discrete muon ionization. Do not replace the
requested 0.5 MeV value by 0.4 MeV in the command above.

For a \(10^{19}\) eV primary, change only the requested primary energy:

```bash
export C8_TABLE_1E19="$(
  "$C8_INSTALL/bin/gpu_em_table_prepare" \
    --medium-yaml "$C8_INSTALL/share/corsika/media/air_dry_1_atm.yaml" \
    --cache-dir "$C8_TABLE_CACHE" \
    --primary-energy-eV 1e19 \
    --em-cut-MeV 0.5 \
    --electron-transport-cut-MeV 0.5 \
    --muon-transport-cut-MeV 300 \
    --tolerance 5e-4 \
    --loss-tolerance 5e-4 \
    --print-path-only
)"
```

The default energy margin is 1.05. A \(10^{19}\) eV request therefore creates
or resolves a table with an upper bound of at least
\(1.05\times10^{13}\) MeV. Tables are never extrapolated beyond their declared
energy range.

Useful preparation modes are:

| Option | Behavior |
|---|---|
| `--dry-run` | Print the normalized request without generating anything |
| `--lookup-only` | Return exit code 2 on a cache miss instead of generating |
| `--force` | Regenerate the exact content-addressed request |
| `--print-path-only` | Print only the resolved `.c8emrt` path |
| `--no-muons` | Prepare only photon/electron/positron tables |
| `--nonmonotonic-loss-policy proposal-monotone\|proposal-direct` | Select compact interpolant repair (default) or full direct-column rebuild |
| `--direct-loss-max-energy-points N` | Set the independent energy-node budget used only by `proposal-direct` |

### Validated dry-air table contract

Any downloaded table for `--emcut 0.0005` GeV must have a preparation manifest
containing both `em_cut_MeV: 0.5` and
`proposal_stochastic_cut_MeV: 0.4`. Tables made by an earlier preparation
contract that used 0.5 MeV for both values are intentionally rejected by the
application. Until a release asset with the split-cut manifest is available,
use the automatic preparation workflow above. Do not reuse the legacy
`production_v10_muons_1e-3_1EeV.c8emrt` asset: it was generated before the
scalar-compatible split-cut contract and is intentionally rejected.

The validated table contract is:

- `AirDry1Atm` composition;
- photon, electron, positron, negative-muon, and positive-muon tables;
- a 0.4 MeV PROPOSAL stochastic cut resolved from the 0.5 MeV CORSIKA
  electromagnetic transport cut;
- 300 MeV muon transport cut;
- total particle energies through \(10^{18}\) eV;
- maximum rate and inverse-CDF interpolation tolerance of \(5\times10^{-4}\)
  for the beta4 production-validation table. Tables at \(10^{-3}\) remain a
  supported lower-cost option, but they are not the same validation artifact.

The content-addressed lookup also checks the table-generator contract version.
A table created by an older generator is regenerated instead of being silently
accepted.

#### PROPOSAL interpolation and the argon bremsstrahlung column

PROPOSAL 7.6.2 can return a locally non-monotonic stochastic-loss inverse CDF
for electron or positron bremsstrahlung on the argon component of dry air. This
is not an argon-projectile effect. It is a numerical feature of the cached
PROPOSAL interpolation and its inverse solver: evaluating the same points with
PROPOSAL interpolation disabled, using direct numerical integration and root
finding, restores the expected monotonic behavior. The original scalar path
does not scan adjacent quantiles for this condition and therefore normally
uses the interpolated result as returned.

For the diagnostic dry-air column at \(E=623.7318908\) MeV, increasing the
quantile from 0.9859885644 to 0.9859897698 changed the interpolated loss
fraction from 0.8847353442 down to 0.8836791531 (a 0.119% reversal). The
non-interpolated calculation changed monotonically from 0.8592897295 to
0.8593004245 at the same two points. These numbers are recorded as a numerical
diagnostic, not as a new physical correction to argon.

Beta2, beta3, and beta4 expose two offline policies through
`--nonmonotonic-loss-policy`:

- `proposal-monotone` (default) starts from the same cached PROPOSAL
  interpolant as the scalar program and replaces only local decreases by the
  preceding cumulative maximum. Once a moving reversal is identified, its
  upper-quantile branch is excluded from energy refinement so the stored
  surface remains smooth across energy. It is the closer, compact comparison
  policy;
  the affected column is tagged `proposal_interpolated_monotone`. This policy
  deliberately treats the 0.119% reversal as interpolation noise, so the
  configured table tolerance describes approximation of the repaired
  monotone reference rather than strict pointwise reproduction of that raw
  reversal.
- `proposal-direct` rebuilds the complete affected column with PROPOSAL
  interpolation disabled. It is tagged `proposal_direct` and is validated
  against direct integration/root values. This is a useful physics-oriented
  diagnostic, but it is expensive: a 0.4 MeV--105 GeV dry-air test requested
  more than 50,000 energy nodes for this one column at \(10^{-3}\), so it is
  not the compact default. Its independent node budget is controlled by
  `--direct-loss-max-energy-points` (default 65536).

In the completed 0.4 MeV--105 GeV dry-air `proposal-monotone` test, the final
table was 3.69 MB. Its measured rate and repaired-reference inverse-CDF errors
were \(9.996\times10^{-4}\) and \(8.541\times10^{-4}\), respectively. The audit
log reported a maximum local monotone projection of 0.924% and a maximum raw
moving-branch deviation of 9.90%. The latter two values quantify the rejected
PROPOSAL numerical branch and are not included in the interpolation-error
claim; production CPU/CUDA ensemble validation remains necessary.

The same table completed 100 GeV electron smoke showers with CUDA EM and CUDA
radio in both beta2 and beta3. Neither run reported a bremsstrahlung/argon
selected-loss fallback. The remaining CPU returns were already-defined physics
paths (photoproduction, plus one beta3 ionization quantile-bound return), not
the PROPOSAL argon interpolation issue.

A post-repair timing check used 12 paired seeds, 100 GeV vertical electron
primaries, `emthin=1e-3`, a 0.5 MeV transport cut, IGRF14 at epoch 2027, and
CUDA EM plus CUDA radio on the same RTX 4060 Laptop GPU. After a clean Release
rebuild, mean/median in-shower times were 0.731/0.715 s for beta1,
0.684/0.594 s for beta2, and 0.737/0.670 s for beta3. Thus the beta3 mean was
within 0.8% of the beta1 mean, while beta2 was 6.5% lower. Mean sampled active
GPU utilization was 19.2%, 25.5%, and 24.2%,
respectively. The repaired beta2/beta3 runs contained no argon bremsstrahlung
fallback; only one and four low-frequency non-argon quantile-bound returns
remained across the 12 showers. This test supports removal of the previous
roughly twofold low-energy timing regression, but it is a performance check,
not a shower-observable validation sample.

Strict bit-level reproduction of the raw scalar reversal is a third,
algorithm-emulation task and would require porting PROPOSAL's interpolation
and inverse solver rather than representing it by a smooth CUDA table. Simple
multi-process parallelism over particles or columns does not remove the main
direct-policy cost because the dense refinement occurs inside one argon
bremsstrahlung column; parallel row sampling may be added later after a
thread-safety audit of PROPOSAL.

The current `gpu_em_tablegen` implementation is serial; setting
`OMP_NUM_THREADS` does not accelerate it. Table preparation is an offline cost,
and the content-addressed cache prevents the same material/cut/energy request
from being generated again. Run independent table requests as separate jobs
only when the server policy and available memory permit it.

Both implemented policies finish during table preparation and keep shower
transport on the GPU: argon never causes a runtime selected-loss CPU fallback
or fragments an EM wavefront. Rate and process/component selection tables are
unchanged. The selected policy and grid budget enter the content-addressed
request, and generator contract `0.18` prevents an older fallback-bearing table
from being silently reused.

### Custom material YAML

`configs/media/air_dry_1_atm.yaml` is the schema-1 reference. The table request
hash includes material composition, PROPOSAL parameters, cuts, energy range,
grid controls, and format version. Comments, whitespace, and component order do
not change the normalized hash; a physical material change does.

Generating a rock, ice, soil, or lunar-regolith table does not enable that
material in the current GPU atmosphere. See
[`gpu_em_tables/README.md`](gpu_em_tables/README.md) for the YAML schema and the
runtime-environment boundary.

## Run a shower

`c8_air_shower` uses GeV for particle energies, degrees for direction angles,
and metres for geometry. The output directory passed to `-f` must not already
exist.

### Scalar reference run

Omitting `--em-backend` selects the original scalar PROPOSAL path:

```bash
export C8_ANTENNAS=/absolute/path/to/antennas.txt
export C8_CPU_OUTPUT=/absolute/path/to/proton_100TeV_cpu

"$C8_INSTALL/bin/c8_air_shower" \
  -p 2212 \
  -E 1e5 \
  -N 1 \
  -z 0 \
  -a 0 \
  --geomagnetic-model IGRF14 \
  --geomagnetic-year 2027 \
  -s 10001 \
  -f "$C8_CPU_OUTPUT" \
  --emcut 0.0005 \
  --emthin 1e-6 \
  --max-weight 100 \
  --ring 0 \
  --antenna-file "$C8_ANTENNAS"
```

This is equivalent to explicitly passing `--em-backend proposal`.

### Minimal CUDA run

The minimum additional CUDA arguments are the backend selection and a
compatible physics table:

```bash
export C8_CUDA_OUTPUT=/absolute/path/to/proton_100TeV_cuda

"$C8_INSTALL/bin/c8_air_shower" \
  -p 2212 \
  -E 1e5 \
  -N 1 \
  -z 0 \
  -a 0 \
  --geomagnetic-model IGRF14 \
  --geomagnetic-year 2027 \
  -s 20001 \
  -f "$C8_CUDA_OUTPUT" \
  --emcut 0.0005 \
  --emthin 1e-6 \
  --max-weight 100 \
  --ring 0 \
  --antenna-file "$C8_ANTENNAS" \
  --em-backend cuda \
  --gpu-table-cache "$C8_TABLE"
```

This uses CUDA electromagnetic transport with the default CPU radio and scalar
low-energy hadronic backends.

### Fully accelerated reference run

Enable CUDA radio and the process-isolated FLUKA pool explicitly:

```bash
export C8_FULL_OUTPUT=/absolute/path/to/proton_100TeV_cuda_full

"$C8_INSTALL/bin/c8_air_shower" \
  -p 2212 \
  -E 1e5 \
  -N 1 \
  -z 0 \
  -a 0 \
  --geomagnetic-model IGRF14 \
  --geomagnetic-year 2027 \
  -s 20001 \
  -f "$C8_FULL_OUTPUT" \
  --emcut 0.0005 \
  --hadcut 0.3 \
  --mucut 0.3 \
  --taucut 0.3 \
  --emthin 1e-6 \
  --max-weight 100 \
  --ring 0 \
  --antenna-file "$C8_ANTENNAS" \
  --em-backend cuda \
  --gpu-device 0 \
  --gpu-min-batch 4096 \
  --gpu-memory-fraction 0.70 \
  --gpu-table-cache "$C8_TABLE" \
  --gpu-table-tolerance 5e-4 \
  --gpu-deterministic true \
  --gpu-resident-cross-species true \
  --radio-backend cuda \
  --gpu-radio-field-limit 1 \
  --hadronic-backend fluka-process \
  --hadronic-workers 4 \
  --hadronic-min-batch 64 \
  --hadronic-target-batch-ms 5 \
  --hadronic-max-batch 256
```

For the inclined \(10^{17}\) eV test configuration, change the energy and
direction while retaining the same cuts and thinning:

```text
-E 1e8 -z 47 -a 180 --emcut 0.0005 --emthin 1e-6
```

The explicit `--max-weight 100` in these examples is intentional. With a
100 TeV primary and `emthin=1e-6`, omitting the option selects the application
value `0.5 * emthin * E_primary[GeV] = 0.05`. Because the scalar `EMThinning`
guard returns when `parentWeight >= maxWeight`, a unit-weight history cannot
start thinning in that configuration. CPU/CUDA comparison commands must use
the same explicit value. The value `100` is an example production setting, not
a universal optimum; it controls the variance/speed trade-off and must be
recorded as part of the physics configuration.

### Antenna file

`--antenna-file` expects three whitespace-separated columns in the local NWU
coordinate system:

```text
North_m  West_m  Up_m
```

Use `--ring 0` to disable the generated star-shaped observer rings and use only
the supplied antennas. CPU and CUDA radio calculations consume the same
observer geometry.

## Essential command-line options

Run the compiled program with `--help` for the exact defaults and constraints:

```bash
"$C8_INSTALL/bin/c8_air_shower" --help
"$C8_INSTALL/bin/gpu_em_table_prepare" --help
"$C8_INSTALL/bin/gpu_em_tablegen" --help
"$C8_INSTALL/bin/cuda_decision_replay" --help
"$C8_INSTALL/bin/fluka_batch_worker" --help
```

### Shower configuration

| Option | Meaning |
|---|---|
| `-p, --pdg` | Primary PDG code; proton is 2212, photon is 22, electron is 11 |
| `-Z` and `-A` | Nuclear charge and mass number; mutually exclusive with `--pdg` |
| `-E, --energy` | Primary total energy in GeV |
| `-N, --nevent` | Number of showers in the output library |
| `-z, --zenith` | Zenith angle in degrees |
| `-a, --azimuth` | Azimuth angle in degrees |
| `--geomagnetic-model` | `IGRF13` or `IGRF14`; default `IGRF14` |
| `--geomagnetic-year` | IGRF epoch in the range 1900--2030; default 2027 |
| `-s, --seed` | Initial random seed |
| `-f, --filename` | New output-library path |
| `--emcut` | Photon/electron/positron kinetic-energy cut in GeV |
| `--hadcut`, `--mucut`, `--taucut` | Non-EM kinetic-energy cuts in GeV |
| `--emthin` | Fraction of primary energy at which EM thinning starts |
| `--max-weight` | Maximum EM thinning weight; zero selects the application's automatic value |
| `--antenna-file` | Observer positions in NWU metres |
| `--ring` | Number of generated observer rings; zero uses only the antenna file |

When comparing CPU and CUDA ensembles, keep all physical parameters identical.
The application records the resolved thinning behavior, including whether the
automatic maximum weight can activate from an initially unweighted history.

### CUDA transport

| Option | Default | Meaning |
|---|---:|---|
| `--em-backend proposal\|cuda` | `proposal` | Select scalar PROPOSAL or hybrid CUDA transport |
| `--gpu-device` | `0` | CUDA device index |
| `--gpu-min-batch` | `4096` | Minimum useful GPU front; smaller fronts receive bounded scalar expansion |
| `--gpu-memory-fraction` | `0.70` | Fraction of currently free device memory available to the backend |
| `--gpu-table-cache` | none | Required versioned `.c8emrt` file |
| `--gpu-table-tolerance` | `1e-3` | Maximum accepted table interpolation error |
| `--gpu-deterministic` | `true` | Enable history-addressed deterministic Philox random numbers |
| `--gpu-resident-cross-species` | `true` | Keep photon/lepton secondaries in persistent device queues |
| `--gpu-detailed-stage-timing` | off | Record CUDA stage, device-copy, and host-wait timing; profiling only |
| `--gpu-full-step-records` | off | Return full transport records; validation only |

### Radio projection

| Option | Default | Meaning |
|---|---:|---|
| `--radio-backend cpu\|cuda` | `cpu` | Select CPU or resident CUDA CoREAS/ZHS projection |
| `--gpu-radio-field-limit` | `1.0` V/m | Checked fixed-point waveform range; overflow is fatal |
| `--radio-sampling-rate-ghz` | `1.0` | Radio time-domain sampling rate |
| `--radio-window-duration-ns` | `400` ns | Observer time-window length |
| `--radio-pretrigger-ns` | `10` ns | Time before the geometric direct-arrival reference |
| `--gpu-radio-track-diagnostics` | off | Collect extra track diagnostics; validation only |

Use at least 10 GHz sampling when making precision 50--350 MHz CPU/CUDA
waveform comparisons. CPU and CUDA radio projection are accumulated while the
electromagnetic tracks are produced; neither backend waits to reconstruct the
particle tree after the complete shower.

The CUDA radio path automatically precomputes observer-independent track
kinematics once per input record and projects `4 tracks x 64 observers` per
two-dimensional shared-memory tile. The double-buffered track workspaces count
against the configured GPU memory budget. This optimization requires no extra
CLI option and preserves deterministic waveform output.

### FLUKA scheduling

| Option | Default | Meaning |
|---|---:|---|
| `--hadronic-backend scalar\|fluka-process` | `scalar` | Select in-process scalar FLUKA or persistent FLUKA workers |
| `--hadronic-workers` | `4` | Number of persistent FLUKA worker processes |
| `--hadronic-min-batch` | `64` | Parked vertices required before cost-based flushing |
| `--hadronic-target-batch-ms` | `5` ms | Target estimated work per homogeneous worker batch |
| `--hadronic-max-batch` | `256` | Maximum requests in one worker batch |
| `--hadronic-worker-executable` | sibling binary | Explicit path to `fluka_batch_worker` |
| `--cpu-detailed-step-timing` | off | Record detailed scalar phase timings |

The worker executable must match the `c8_air_shower` build. Keep it beside the
main executable unless an explicit path is supplied.

FLUKA installs a periodic quota timer when its in-process interaction object is
constructed. In `fluka-process` mode the CUDA parent disables that redundant
timer after the isolated workers have been started. The workers retain native
FLUKA initialization and physics. This avoids calling C stdio from a
`SIGALRM` handler in the multithreaded CUDA parent, which can otherwise
deadlock on a libc stream lock. The scalar backend is unchanged.

### Replay and diagnostics

| Option | Meaning |
|---|---|
| `--cuda-replay-trace PATH` | Write a process-level CSV trace |
| `--cuda-replay-tape-out PATH` | Record scalar transport segments and the radio observer snapshot |
| `cuda_decision_replay --tape ...` | Replay the exact recorded decisions and tracks on CUDA |

Production profiling options add synchronization or data-transfer overhead.
Keep them disabled when reporting performance.

## Output and failure semantics

Each run writes a CORSIKA output library. Important paths include:

| Path | Contents |
|---|---|
| `config.yaml` | Complete application command and top-level configuration |
| `summary.yaml` | Library completion status |
| `gpu_em/config.yaml` | GPU, table, medium, magnetic field, backend, and build provenance |
| `gpu_em/summary.yaml` | Particle counts, fallbacks, queue peaks, memory, timing, and completion status |
| `simulation_timing/summary.yaml` | Per-shower end-to-end wall time |
| `profile/` | Longitudinal particle profiles |
| `production_profile/` | Particle-production and parent profiles |
| `energyloss/` | Longitudinal energy deposition |
| `particles/` | Observation-level particle output when enabled |
| `CoREAS/` and `ZHS/` | Radio observer configuration and waveforms |

The CUDA path fails closed for incompatible tables, unsupported environment
identity, illegal process registration, NaN or negative energy, invalid PID,
queue overflow, radio fixed-point overflow, CUDA errors, or an inconsistent
energy ledger. It does not silently switch the whole shower back to scalar
PROPOSAL.

Declared rare final states and bounded low-energy memory spill are the only
normal CPU returns. Every fallback category and spill count is written to the
GPU summary.

`--force-interaction` and `--force-decay` retain the scalar `Cascade`
contract in CUDA mode. The next scheduled primary executes exactly one forced
scalar vertex before it is eligible for GPU routing; its secondaries then use
the normal hybrid route. The two requests are mutually exclusive. Executed
counts are recorded under `forced_primary` in `gpu_em/summary.yaml`.

The CUDA startup gate recursively checks every `ContinuousProcess`,
`SecondariesProcess`, `InteractionProcess`, `DecayProcess`,
`BoundaryCrossingProcess`, and `StackProcess` in the application sequence.
Each type must declare whether it is device-replaced, record-replayed,
CPU-deferred, inapplicable to routed EM, or diagnostic-only. An unregistered
type aborts startup instead of being silently skipped. The policy counts and
all six unregistered counts are written under `process_registry`.

Before accepting an output, check at least:

```text
summary.yaml exists
gpu_em/summary.yaml: complete: true
gpu_em/summary.yaml: status: complete
simulation_timing/summary.yaml: closed: true
no unexpected fallback, overflow, NaN, or energy-ledger failure
```

## Reproducibility and CPU/CUDA comparison

The CUDA backend addresses random draws with a key derived from the seed,
shower, particle history, step, process, and draw index. On the same GPU, with
the same binary, table, and configuration, a deterministic CUDA run must repeat.
Different GPU architectures are required to reproduce statistical distributions
but are not required to reproduce every floating-point bit.

The scalar and CUDA schedulers do not consume one shared global random stream in
the same order. Therefore, the same initial seed does not imply an identical
production shower between scalar CORSIKA 8 and the CUDA backend. Physics
equivalence is evaluated with independent ensembles.

For process-by-process debugging, record and replay the exact scalar decision
tape:

```bash
"$C8_INSTALL/bin/c8_air_shower" \
  -p 11 -E 1000 -N 1 -s 10001 \
  -f /absolute/path/to/scalar_replay_source \
  --cuda-replay-tape-out /absolute/path/to/event.c8rpt

"$C8_INSTALL/bin/cuda_decision_replay" \
  --tape /absolute/path/to/event.c8rpt \
  --output /absolute/path/to/cuda_replay_output \
  --device 0 \
  --deterministic true
```

Decision replay verifies identical recorded transport decisions and radio
tracks; it is not a replacement for independent production-ensemble tests.

## Validation and measured performance

The validation suite compares:

- electromagnetic, muonic, and hadronic longitudinal profiles;
- shower maximum and profile integrals;
- energy deposition and energy closure;
- observation-level spectra, counts, lateral distributions, and arrival times;
- CoREAS/ZHS geomagnetic pulse amplitude and width;
- deterministic repetition, table identity, fallbacks, and failure behavior;
- cold-cache and warm-cache end-to-end timing.

The largest completed production-scale reference comparison currently remains
the beta2-era 500 scalar versus 500 CUDA proton ensemble:

```text
primary: proton
energy: 100 TeV (1e5 GeV)
zenith: 0 degrees
azimuth: 0 degrees (irrelevant for vertical incidence)
EM cut: 0.5 MeV
emthin argument: 1e-6
max-weight argument: omitted (automatic value 0.05)
effective thinning from unit-weight histories: inactive
```

The main shower-component profiles and radio-pulse features show overall
statistical consistency with the public scalar CORSIKA 8 implementation at the
precision of the current samples. Detailed reports retain the status of each
individual acceptance gate rather than replacing them with a single pass/fail
claim. The `emthin=1e-6` label in this historical data set must not be
interpreted as proof that thinning activated: at 100 TeV the automatic maximum
weight is 0.05, so the scalar guard prevents a unit-weight history from entering
the thinning branch. Both comparison arms used the same behavior.

The historical turnaround measurements were:

- the 500 scalar showers had a mean wall time of 9,955 s per event;
- the 500 CUDA showers on an RTX 4060 Laptop GPU had a mean wall time of
  69.1 s per event;
- ten scalar \(10^{17}\) eV proton jobs at zenith 47 degrees, azimuth
  180 degrees, and EM thinning \(10^{-6}\) required 4.69--5.18 days per
  event, with a mean of 5.09 days;
- one complete CUDA event with the same primary energy, direction, cuts, and
  thinning completed in 3,204 s, or 53.4 minutes, on the RTX 4060 Laptop GPU.

The CPU and GPU production campaigns used different host computers. These
numbers document observed scientific turnaround and are not a controlled
same-host hardware benchmark.

Beta4-specific acceptance is tracked separately:

- phase 97 completed a 500-versus-500, 10 GeV electron diagnostic after the
  chord-grammage and per-PID-cut repair. Longitudinal curves were statistically
  consistent. Its near-floor radio pulses are retained as a numerical
  diagnostic, not as the final high-energy radio acceptance sample;
- phase 98 used an 80-degree geometry and same-track CPU/CUDA radio projection
  to verify the corrected observation plane. CoREAS and ZHS track-replay
  differences remained at about `1e-4` or below;
- phases 99--101 passed the first-interaction, 10 ms physical-time cut,
  forced-primary, and six-category process-registry gates;
- phase 102 found mean beta4/beta2 wall time `1.014` in three paired 10 PeV
  proton runs, with both versions sustaining the same high-utilization GPU
  regime;
- phase 103 accepted a new IGRF14/2027 beta2 scalar 500-event reference on the
  server and started the exactly matched beta4 CUDA 500-event campaign. This
  comparison remains **in progress**; partial-event statistics are not a final
  validation result.

Run the maintained acceptance drivers rather than comparing only one shower:

```bash
python "$C8_SOURCE/validation/gpu_em/run_physics_acceptance.py" \
  --executable "$C8_INSTALL/bin/c8_air_shower" \
  --table "$C8_TABLE" \
  --output-root /absolute/path/to/acceptance_output \
  --primary-pdg 2212 \
  --energy-gev 100000 \
  --zenith-deg 0 \
  --azimuth-deg 0 \
  --em-cut-gev 5e-4 \
  --em-thinning 1e-6 \
  --maximum-weight 100 \
  --events 500
```

Use `--maximum-weight 0` only when deliberately reproducing the historical
automatic-weight configuration, and record that thinning cannot activate at
100 TeV. For an active-thinning study, set the same explicit value greater than
one in both scalar and CUDA ensembles.

See [`validation/gpu_em/README.md`](validation/gpu_em/README.md) for replay,
energy-closure, ensemble, radio, scaling-law, and performance workflows.

## Performance tuning on a new GPU

Start from the reference configuration and tune scheduling parameters without
changing the physics configuration, seed, table, or cache state.

1. Build for the actual CUDA architecture in `Release` mode.
2. Warm the physics and Moliere sidecar caches before production timing.
3. Sweep `--gpu-min-batch` over values such as 64, 256, 1024, 4096, and 8192.
4. Repeat each point at least five times and compare the median end-to-end time.
5. Sweep `--hadronic-workers` separately over values such as 1, 2, 4, and 8.
6. Inspect fallback time, FLUKA worker utilization, queue peaks, transfer time,
   peak device memory, and spill/overflow counts.
7. Re-run physics validation after selecting the performance configuration.

The current RTX 4060 100 TeV proton workload favors a minimum CUDA batch near
4096. This is not a universal value; larger GPUs and different shower energies
may prefer another threshold.

One `c8_air_shower` process controls one CUDA device. A single shower is not
split across several GPUs. On a multi-GPU node, launch one independent process
per GPU with disjoint seeds and output directories:

```bash
CUDA_VISIBLE_DEVICES=0 c8_air_shower \
  ... --gpu-device 0 -s 20000001 -f /path/to/output_gpu0 &

CUDA_VISIBLE_DEVICES=1 c8_air_shower \
  ... --gpu-device 0 -s 30000001 -f /path/to/output_gpu1 &

wait
```

Reserve CPU cores for each process's FLUKA workers and host scheduler. Do not
run a competing scalar campaign while measuring CUDA performance.

## Current limitations

- NVIDIA CUDA is the only accelerator backend; HIP and SYCL are not supported.
- A single shower uses one GPU.
- The device environment is limited to the validated five-layer dry-air
  atmosphere and the locally flat observation-plane geometry used by
  `c8_air_shower`.
- Custom material tables do not by themselves add custom runtime geometry.
- High-energy hadronic interactions remain on the CPU.
- `fluka-process` is CPU multiprocessing, not GPU hadronic transport.
- Rare and unsupported final states return to explicit CPU generators.
- Same-seed scalar and production CUDA showers are not expected to be
  event-by-event identical.
- New media, energy ranges, cuts, GPU architectures, or performance settings
  require renewed validation.
- The fork is not an official CORSIKA Collaboration release.

## Repository layout

| Path | Purpose |
|---|---|
| `applications/c8_air_shower.cpp` | Main scalar/CUDA production application |
| `corsika/framework/core/HybridCascade.hpp` | CPU/GPU cascade scheduler |
| `corsika/gpu/` and `src/gpu/` | CUDA transport, tables, radio, and runtime |
| `applications/gpu_em_table_prepare.cpp` | Material hashing and automatic table preparation |
| `applications/gpu_em_tablegen.cpp` | Low-level PROPOSAL table generation |
| `applications/cuda_decision_replay.cpp` | Exact scalar-tape CUDA replay |
| `applications/fluka_batch_worker.cpp` | Process-isolated FLUKA worker |
| `configs/media/` | Canonical schema-1 material definitions |
| `gpu_em_tables/` | Table documentation and release hashes; large tables are not tracked |
| `validation/gpu_em/` | Physics, radio, reproducibility, and performance validation |
| `documentation/cuda_em_refactor/` | Architecture, CLI, evidence, and implementation history |

Generated shower libraries, build directories, installed FLUKA files,
PROPOSAL caches, and large `.c8emrt` tables are deliberately excluded from Git.

## Relationship to upstream CORSIKA 8

This repository extends the CORSIKA 8 framework; it is not a replacement for
the upstream project or its documentation. For the upstream installation,
physics modules, collaboration agreement, contribution rules, and current
releases, use the official resources:

- [CORSIKA 8 repository](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika)
- [CORSIKA 8 documentation](https://corsika-8.readthedocs.io/)
- [CORSIKA project website](https://www.iap.kit.edu/corsika/)
- [upstream contribution guidelines](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/blob/master/CONTRIBUTING.md)

The scalar path is intentionally retained so that the public CORSIKA 8
implementation remains available as the reference backend in the same source
tree.

## Citation and license

When using this fork, cite the CORSIKA 8 works requested by the upstream
project, including:

- *Towards a Next Generation of CORSIKA: A Framework for the Simulation of
  Particle Cascades in Astroparticle Physics*, Comput. Softw. Big Sci. 3
  (2019) 2, https://doi.org/10.1007/s41781-018-0013-0
- *Simulating radio emission from particle cascades with CORSIKA 8*,
  Astropart. Phys. 166 (2025) 103072,
  https://doi.org/10.1016/j.astropartphys.2024.103072

See [`LICENSE`](LICENSE), [`USING_COLLABORATING.md`](USING_COLLABORATING.md),
and [`CONTRIBUTING.md`](CONTRIBUTING.md) before redistributing or contributing
code. CORSIKA 8 and this fork are distributed under the BSD 3-Clause License.
