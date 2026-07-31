# CORSIKA 8 CUDA/FLUKA Hybrid Cascade Research Fork

> This repository is a research fork for 21CMA air-shower studies. The scalar
> CORSIKA 8 `Cascade + PROPOSAL` path remains the default. CUDA transport,
> CUDA radio projection, and the process-isolated FLUKA pool are opt-in and
> fail closed when their declared capabilities are not met.

[中文说明](README_CN.md) ·
[CUDA/FLUKA CLI reference](documentation/cuda_em_refactor/cli_reference.md) ·
[production guide](documentation/cuda_em_refactor/cuda_em_backend_user_guide.md) ·
[validation tools](validation/gpu_em/README.md) ·
[implementation history](documentation/cuda_em_refactor/README.md)

## Repository status

| Path | Purpose |
|---|---|
| `applications/c8_air_shower.cpp` | Scalar/CUDA production application |
| `corsika/gpu` and `src/gpu` | CUDA EM transport, tables, radio, and runtime |
| `corsika/framework/core/HybridCascade.hpp` | CPU/GPU cascade scheduler |
| `applications/gpu_em_tablegen.cpp` | Versioned PROPOSAL table generator |
| `applications/cuda_decision_replay.cpp` | Exact scalar-tape CUDA replay |
| `applications/fluka_batch_worker.cpp` | Isolated FLUKA final-state worker |
| `validation/gpu_em` | Reproducible performance and physics acceptance |
| `documentation/cuda_em_refactor` | Architecture, CLI, evidence, and phase logs |

The branch is intended for local research and has not been accepted or
reviewed by the upstream CORSIKA collaboration. Generated shower libraries,
PROPOSAL caches, `.c8emrt` production tables, build directories, and local
FLUKA installations are deliberately not stored in Git.

## Upstream framework

The purpose of CORSIKA 8 is to simulate any particle cascades in
astroparticle physics or astrophysical context. A lot of emphasis has been
put on modularity, flexibility, completeness, validation and
correctness. To boost computational efficiency, different techniques
are provided, like thinning or cascade equations. The aim is that
CORSIKA 8 remains the most comprehensive framework for simulating
particle cascades with stochastic and continuous processes.

The software makes extensive use of static design patterns and
compiler optimization. Thus, the most fundamental configuration
decisions of the user must be performed at compile time. At run time,
model parameters can still be changed.

CORSIKA 8 is by default released under the BSD 3-Clause License. See [license
file](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/blob/master/LICENSE)
which is part of every release and the source code.

If you use, or want to refer to, CORSIKA 8 please cite ["Towards a Next
Generation of CORSIKA: A Framework for the Simulation of Particle
Cascades in Astroparticle Physics", Comput. Softw. Big Sci. 3 (2019)
2](https://doi.org/10.1007/s41781-018-0013-0) as well as
["Simulating radio emission from particle cascades with CORSIKA 8", Astropart. Phys. 166 (2025)
103072](https://doi.org/10.1016/j.astropartphys.2024.103072).

We kindly ask (and require) any relevant improvement or addition to be offered or
contributed to the main CORSIKA 8 repository for the benefit of the
whole community.

CORSIKA 8 makes use of various third-party code, in particular interaction
models. Please check the [using and collaborating
agreement](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/blob/master/USING_COLLABORATING.md)
for further information on this topic.

If you plan to contribute to CORSIKA 8, please check the guidelines outlined here:
[coding
guidelines](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/blob/master/CONTRIBUTING.md). Code
that fails the review by the CORSIKA 8 author group must be improved
before it can be merged in the official code base. After your code has
been accepted and merged, you become a contributor of the CORSIKA 8
project (code author).

IMPORTANT: Before you contribute, you need to read and agree to the conditions set out in the
[using and collaborating
agreement](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/blob/master/USING_COLLABORATING.md).
The agreement can be discussed, and eventually improved if necessary.


## Get in contact
  * Join our chat threads using Mattermost via this [invite link](https://mattermost.hzdr.de/signup_user_complete/?id=xtdd8jyt6trbiezt71gaz3z4ge&md=link&sbr=su). Click the `GitLab` button, then `Sign in with Helmholtz ID`. You will be able to make an account by either finding your institution, or using your e.g. ORCID, GitHub, or Google account.
  * Connect to https://gitlab.iap.kit.edu, register yourself and join the "Air Shower Physics" group. Write to us on Mattermost (in the User Questions channel), or directly contact one of the [steering comittee members](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/-/wikis/Steering-Committee) in case there are problems with that.
  * Connect to corsika-devel@lists.kit.edu (self-register at
    https://www.lists.kit.edu/sympa/subscribe/corsika-devel) to get in
    touch with the project.


## Installation

CORSIKA 8 is tested regularly at least on `gcc11.0.0` and `clang-14.0.0`.

### Prerequisites

You will also need:

- Python 3 (supported versions are Python >= 3.6), with pip
- cmake > 3.4
- git
- g++, gfortran, binutils, make
- optional: FLUKA (see below)


On a bare Ubuntu machine, just add:
``` shell
sudo apt-get install python3 python3-pip cmake g++ gfortran git doxygen graphviz
```

### Creating a virtual environment and Conan

It is recommended that you install CORSIKA 8 and its dependencies within a python3 virtual environment.
To do so, you can run the following.
``` shell
# Create the environment using your native python3 binary
python3 -m venv /path/to/new/virtual/environment/corsika-8
# Load the environment (should be run each time you open a new terminal)
source /path/to/new/virtual/environment/corsika-8/bin/activate

```

You will need to load the environment each time that you open a new terminal.

CORSIKA 8 uses the [conan](https://conan.io/) package manager to
manage our dependencies. This branch uses the Conan 2 interface; its current
reference environment is built with Conan 2.11.0.
**Note**: if you are NOT using a virtual environment, you may want to use the `pip install --user` flag.

``` shell
pip install conan particle==0.25.1 numpy
```

### Enabling FLUKA support

For legal reasons we do not distribute/bundle FLUKA together with CORSIKA 8.
As FLUKA is the standard low-energy hadronic interaction model for CORSIKA 8, you have to download
it separately from (http://www.fluka.org/), which requires registering there as FLUKA user.
The following should be done *before* compiling CORSIKA 8:

 1. Note your system's version of gfortran (`gfortran --version`) and glibc (`ldd --version`)
 2. Download the FLUKA __binary__, ensuring that it matches the versions you found above
 3. Download the FLUKA __data file__ (will be named something similar to __fluka20xy.z-data.tar.gz__).
 4. Un-tar the files that you downloaded using `tar -xf <filename>`
 5. Set environmental variables `export FLUFOR=gfortran` and `export FLUPRO=<path to where you unzipped the files>`. Note that the `FLUPRO` directory should contain __libflukahp.a__. Both of these variables will have to be set every time you open a new terminal.
 6. Go to the `FLUPRO` directory and run `make`. This will compile an exe, __flukahp__, in your current directory.
 7. Follow the normal steps to compile CORSIKA 8 (see below).

When you later install CORSIKA 8, you should see a message during the __cmake__ step indicating the FLUKA was correctly found.

``` shell
libflukahp.a found in directory <some location here> via FLUPRO environment variable
FLUKA support is enabled.
```

### Compiling CORSIKA 8

Once Conan is installed and FLUKA provided, follow these steps to download and install CORSIKA 8:

``` shell
cd ./top/directory/for/corsika/installation
git clone --recursive git@gitlab.iap.kit.edu:AirShowerPhysics/corsika.git
# Or for https: git clone --recursive https://gitlab.iap.kit.edu/AirShowerPhysics/corsika.git
mkdir corsika-build
cd corsika-build
../corsika/conan-install.sh --source-directory ../corsika --release-with-debug
# conan-install.sh takes required options from command line to install dependencies for 'Debug', 'Release' and 'RelWithDebInfo' builds.
../corsika/corsika-cmake.sh -c "-DCMAKE_BUILD_TYPE="RelWithDebInfo" -DWITH_FLUKA=ON -DCMAKE_INSTALL_PREFIX=../corsika-install"
make -j4  #The number should match the number of available cores on your machine
make install
```

## Alternate installation using docker containers

There are docker containers prepared that bring all the environment and packages you need to run CORSIKA. See [docker hub](https://hub.docker.com/repository/docker/corsika/devel) for a complete overview.

### Prerequisites

You only need docker, e.g. on Ubuntu: `sudo apt-get install docker` and of course root access.

### Compiling

Follow these steps to download and install CORSIKA 8, master development version
```shell
cd ./top/directory/for/corsika/installation
git clone --recursive git@gitlab.iap.kit.edu:AirShowerPhysics/corsika.git
sudo docker run -v $PWD:/corsika -it corsika/devel:clang-8 /bin/bash
mkdir corsika-build
cd corsika-build
../corsika/conan-install.sh --source-directory ../corsika --release-with-debug
# conan-install.sh takes required options from command line to install dependencies for 'Debug', 'Release' and 'RelWithDebInfo' builds.
../corsika/corsika-cmake.sh -c "-DCMAKE_BUILD_TYPE="RelWithDebInfo" -DWITH_FLUKA=ON -DCMAKE_INSTALL_PREFIX=../corsika-install"
make -j4  #The number should match the number of available cores on your machine
make install
```

## Running Unit Tests

To run the unit tests, do the following.

```shell
cd ./corsika-build
ctest -j4  #The number should match the number of available cores on your machine
```

## CUDA research branch: additions and deployment

This research branch adds an optional NVIDIA CUDA path without changing the
default scalar CORSIKA 8 execution:

```text
CORSIKA_ENABLE_CUDA=OFF  -> original Cascade + scalar PROPOSAL
CORSIKA_ENABLE_CUDA=ON   -> also build the optional HybridCascade CUDA path
```

The scalar executable remains available after a CUDA build. At run time,
omitting `--em-backend` is still equivalent to `--em-backend proposal`.

### What this branch adds

The main additions relative to the original project are:

- `ScalarCascadeStepper`, which separates one scalar transport step from the
  original LIFO cascade scheduler without making the CORSIKA stack concurrent;
- `HybridCascade` and stable CPU/GPU routing for photons, electrons,
  positrons and table-enabled muon transport;
- the compiled `CORSIKA8GpuEm` CUDA library, with resident photon/lepton
  queues, history-addressed Philox random numbers, spherical-atmosphere
  tracking, magnetic transport, EM interactions, thinning and profile
  accumulation;
- versioned PROPOSAL-derived `.c8emrt` rate/inverse-CDF/continuous-loss tables
  and the `gpu_em_tablegen` tool;
- explicit, named CPU fallback for unsupported or non-EM final states instead
  of silently dropping a physical process;
- optional resident CUDA CoREAS/ZHS waveform accumulation
  (`--radio-backend cuda`) and the original CPU radio path
  (`--radio-backend cpu`);
- exact legacy decision-tape replay through `cuda_decision_replay`, intended
  for step-by-step debugging rather than production sampling;
- an optional process-isolated FLUKA pool (`fluka_batch_worker`) for low-energy
  hadronic final states in a CUDA HybridCascade run;
- fail-closed output metadata, energy-ledger checks, performance runners and
  CPU/CUDA statistical-acceptance tools under `validation/gpu_em`.

The current device environment supports the five-layer spherical dry-air
snapshot, its medium IDs, the configured uniform magnetic field and spherical
observation surface. General mountain geometry, lunar regolith, ice and
arbitrary three-dimensional media require new device geometry, tables and a
separate physics validation; they are not enabled merely by changing an input
geometry.

The current `c8_air_shower` reference environment reads the repository's
`GeoMag/IGRF13.COF` and fixes the 21CMA field to year 2025, latitude
42.5527 degrees, longitude 86.4153816422 degrees and altitude 2680.444195 m.
Both backends receive this same field and the CUDA output records those values.

The production division of work is:

```text
CPU: primary hadrons, taus, decay final states and unsupported/rare final states
GPU: gamma/e-/e+ transport and table-enabled mu-/mu+ transport
CPU or GPU: CoREAS/ZHS projection, selected with --radio-backend
CPU FLUKA workers: optional batched low-energy hadronic final states
```

Muon decay vertices and rare muon discrete interactions selected during GPU
transport return to the exact CPU final-state generators. Muon tracks are not
added to CoREAS/ZHS because the scalar reference radio process also observes
only electron/positron tracks.

More detail about physical coverage and failure semantics is in
[the CUDA EM production guide](documentation/cuda_em_refactor/cuda_em_backend_user_guide.md).
The implementation history and evidence index are in
[the CUDA refactor documentation](documentation/cuda_em_refactor/).

#### EM-thinning maximum-weight caveat

When `--max-weight` is not specified (or is zero), both the original CPU
application and this branch calculate

```text
maxWeight = 0.5 * emthin * E_primary[GeV]
```

`EMThinning` returns immediately while `parentWeight >= maxWeight`. Since an
unweighted history starts at weight 1, automatic `maxWeight` must exceed 1
before thinning can start from it. For example, at `E_primary=1e5 GeV` and
`emthin=1e-6`, the threshold is `0.1 GeV` but `maxWeight=0.05`, so EM thinning
does not activate. At `1e6 GeV` the automatic value is still only `0.5`.
Specify a physically appropriate `--max-weight > 1` when thinning is intended,
and use the same explicit value for CPU/CUDA validation. The application writes
`can_activate_from_unit_weight` to the summary and warns when the configured
combination cannot activate.

### Deploying on a new NVIDIA server

The following procedure is intended for a clean Linux server or a Linux
environment managed by an HPC scheduler. WSL2 also works, but its NVIDIA driver
is installed on Windows; do not install a second Linux display driver inside
WSL. The CUDA toolkit may be installed system-wide, loaded as an HPC module or
installed in the Conda environment.

#### 1. Inspect the new machine

Before configuring CMake, record the hardware and toolchain:

```bash
nvidia-smi
nvidia-smi \
  --query-gpu=index,name,compute_cap,driver_version,memory.total \
  --format=csv
nvcc --version
cmake --version
c++ --version
gfortran --version
ldd --version
```

The build requires:

- a working NVIDIA driver visible through `nvidia-smi`;
- CUDA toolkit 12.x, including `nvcc` and the CUDA development libraries;
- CMake 3.24 or newer when `CORSIKA_ENABLE_CUDA=ON`;
- C++17 and CUDA C++17 compilers;
- a C++/Fortran/glibc combination compatible with the selected FLUKA binary;
- Conan 2 and the normal CORSIKA 8 build prerequisites.

The driver is a host requirement. Installing only `cuda-toolkit` in Conda
cannot repair an absent or incompatible NVIDIA driver.

#### 2. Prepare an environment

One reproducible reference environment is:

```bash
conda create -n corsika_venv \
  -c conda-forge \
  python=3.9 \
  cmake=3.31 \
  conan=2.11 \
  particle=0.25.1 \
  numpy pandas scipy matplotlib pyyaml

conda activate corsika_venv

conda install \
  -c nvidia/label/cuda-12.6.3 \
  cuda-toolkit=12.6.3
```

The versions above reproduce the current development environment; they are not
a claim that a newer CUDA 12.x or Conan 2.x release is invalid. On a managed
server, it is often preferable to load the site's CUDA/compiler modules and
install only Python/Conan in Conda:

```bash
module load cuda/12.6
module load gcc/12
conda activate corsika_venv
```

Use one compiler family consistently when Conan creates its profile and when
CMake builds CORSIKA. Confirm the selected programs before installing
dependencies:

```bash
which c++
which gfortran
which nvcc
conan profile detect --force
```

#### 3. Install and expose FLUKA

FLUKA is not distributed by this repository. Download a licensed FLUKA binary
and data package matching the server's glibc and Fortran runtime, then build it
as described in the main FLUKA instructions above. A production CUDA +
hadronic build should expose:

```bash
export C8_FLUPRO=/path/to/fluka
export FLUPRO="$C8_FLUPRO"
export FLUFOR=gfortran

test -f "$C8_FLUPRO/libflukahp.a"
```

`-DWITH_FLUKA=ON` is deliberately fail-closed in this branch: configuration
stops if a valid FLUKA library is not found. A successful build produces both
`c8_air_shower` and `fluka_batch_worker`. Keep those installed executables
together unless `--hadronic-worker-executable` is set explicitly.

#### 4. Select the CUDA architecture

Compile for the compute capability of the GPU that will run the program. CMake
uses the capability without a decimal point, so capability `8.9` becomes
architecture `89`:

```bash
export C8_CUDA_ARCHS="$(
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
  sed 's/\.//g' |
  sort -u |
  paste -sd';' -
)"

printf 'CMake CUDA architectures: %s\n' "$C8_CUDA_ARCHS"
```

For a homogeneous server, a single architecture gives the shortest build and
an architecture-specific binary. For a deployment image shared by different
GPU generations, keep all unique architectures separated by semicolons.
If `nvcc` reports that a newly released architecture is unknown, the toolkit
is too old for that GPU and must be upgraded; substituting an unrelated older
architecture is not a valid performance build.

This branch does not use `--use_fast_math`. Do not add it during migration:
that changes the numerical contract and invalidates the existing validation.

#### 5. Create Release Conan dependencies

Use a fresh source, build and install directory. Do not reuse a CMake cache
created with another CUDA toolkit, compiler, source tree or GPU architecture.

```bash
export C8_SOURCE=/path/to/corsika8_gpu_refactor
export C8_BUILD=/path/to/corsika8_gpu_refactor_build_cuda
export C8_INSTALL=/path/to/corsika8_gpu_refactor_install_cuda

"$C8_SOURCE/conan-install.sh" \
  --source-directory "$C8_SOURCE" \
  --release
```

The script creates `conan_cmake/conan_toolchain.cmake` and
`corsika-cmake.sh` in `C8_SOURCE`. The explicit CMake command below uses that
toolchain directly, which also safely preserves a semicolon-separated
multi-architecture list. `Release` is required for performance measurements;
a `RelWithDebInfo` or `Debug` timing must not be reported as the production GPU
speed.

#### 6. Configure, build and install

```bash
mkdir -p "$C8_BUILD"

cmake \
  -S "$C8_SOURCE" \
  -B "$C8_BUILD" \
  -DCONAN_CMAKE_DIR="$C8_SOURCE/conan_cmake" \
  -DCMAKE_TOOLCHAIN_FILE="$C8_SOURCE/conan_cmake/conan_toolchain.cmake" \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_CUDA=ON \
  "-DCMAKE_CUDA_ARCHITECTURES=$C8_CUDA_ARCHS" \
  -DWITH_FLUKA=ON \
  -DC8_FLUKALIB="$C8_FLUPRO/libflukahp.a" \
  -DCMAKE_INSTALL_PREFIX="$C8_INSTALL"

cmake --build "$C8_BUILD" --parallel 16
cmake --install "$C8_BUILD"
```

CUDA translation units can use several GiB of host RAM during compilation.
Choose `--parallel` from both core count and available memory; using every
logical core on a many-core but memory-limited node can be slower or trigger
the OOM killer.

The expected production executables are:

```bash
test -x "$C8_BUILD/applications/c8_air_shower"
test -x "$C8_BUILD/applications/gpu_em_tablegen"
test -x "$C8_BUILD/applications/cuda_decision_replay"
test -x "$C8_BUILD/applications/fluka_batch_worker"
```

The default CPU-only build remains available and does not require CUDA:

```bash
export C8_CPU_BUILD=/path/to/corsika8_build_cpu
mkdir -p "$C8_CPU_BUILD"
cmake \
  -S "$C8_SOURCE" \
  -B "$C8_CPU_BUILD" \
  -DCONAN_CMAKE_DIR="$C8_SOURCE/conan_cmake" \
  -DCMAKE_TOOLCHAIN_FILE="$C8_SOURCE/conan_cmake/conan_toolchain.cmake" \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_CUDA=OFF \
  -DWITH_FLUKA=ON \
  -DC8_FLUKALIB="$C8_FLUPRO/libflukahp.a"
cmake --build "$C8_CPU_BUILD" --parallel 16
```

#### 7. Verify the migrated build

Run all compiled tests before a physics job:

```bash
ctest \
  --test-dir "$C8_BUILD" \
  --output-on-failure \
  --parallel 16

cd "$C8_SOURCE"
python -m unittest discover \
  -s validation/gpu_em/tests \
  -p 'test_*.py'
```

Also preserve a machine record:

```bash
cmake -LAH -N "$C8_BUILD" |
  grep -E 'CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CORSIKA_ENABLE_CUDA|WITH_FLUKA'

sha256sum \
  "$C8_BUILD/applications/c8_air_shower" \
  "$C8_BUILD/applications/fluka_batch_worker"
```

Passing tests on a new architecture establishes implementation integrity, not
physics equivalence by itself. The same-GPU deterministic regression should
repeat exactly for the same CUDA configuration; a different GPU architecture
is required to reproduce statistical distributions, not necessarily every
floating-point bit.

### Physics tables on the new server

The `.c8emrt` table is a versioned physics artifact, not a GPU-specific
compile artifact. Copy the validated table and its recorded SHA-256 to the new
machine:

```bash
export C8_TABLE=/path/to/production_v10_muons_1e-3_1EeV.c8emrt
test -f "$C8_TABLE"
sha256sum "$C8_TABLE"
```

Do not regenerate it merely because the GPU model changed. Generate a new table
only when the PROPOSAL configuration, media, cut, supported energy range or
table schema changes, and then repeat table and shower validation. A basic
generator invocation is documented in
[the production guide](documentation/cuda_em_refactor/cuda_em_backend_user_guide.md#4-物理表).

The first CUDA run may create a checked Molière interpolation sidecar beside
the table. Treat that as cold-cache initialization. Production benchmarks
must state whether they use a cold or warm cache and must not mix the two.

### Production run on one GPU

The following example enables all current accelerated production paths:

```bash
export C8_OUTPUT=/path/to/output/proton_100TeV_cuda
export C8_ANTENNAS=/path/to/antennas.txt

"$C8_BUILD/applications/c8_air_shower" \
  -p 2212 \
  -E 100000 \
  -N 50 \
  -f "$C8_OUTPUT" \
  --seed 10200001 \
  --zenith 0 \
  --azimuth 0 \
  --emcut 0.0005 \
  --emthin 1e-6 \
  --hadcut 0.3 \
  --mucut 0.3 \
  --taucut 0.3 \
  --ring 0 \
  --antenna-file "$C8_ANTENNAS" \
  --em-backend cuda \
  --gpu-device 0 \
  --gpu-min-batch 4096 \
  --gpu-memory-fraction 0.70 \
  --gpu-table-cache "$C8_TABLE" \
  --gpu-table-tolerance 1e-3 \
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

`--radio-backend cpu` retains the original CPU CoREAS/ZHS projection while
still using CUDA EM transport. `--radio-backend cuda` enables the resident
fixed-point CUDA radio accumulator. An overflow of
`--gpu-radio-field-limit` is a hard error; do not solve it by ignoring an
incomplete output.

Profiling switches such as `--gpu-detailed-stage-timing`,
`--gpu-full-step-records` and `--gpu-radio-track-diagnostics` add collection
or synchronization overhead. Keep them off for the final performance number
and enable them only while diagnosing a bottleneck.

### Adapting performance to a different server

The current values are good starting points, not universal constants. Change
one scheduling control at a time and keep the physics configuration, seed,
table and cache state fixed.

| Control | Starting point | What to measure |
|---|---:|---|
| `--gpu-min-batch` | 4096 | end-to-end median time, scalar front expansion, kernel occupancy |
| `--gpu-memory-fraction` | 0.70 | peak device bytes, spill/overflow counts |
| `--hadronic-workers` | 4 | FLUKA worker utilization and CPU fallback critical path |
| `--hadronic-min-batch` | 64 | parked vertices versus IPC batch count |
| `--hadronic-target-batch-ms` | 5 | latency/throughput balance |
| `--hadronic-max-batch` | 256 | load balance and maximum worker latency |

For a new GPU, first sweep `--gpu-min-batch` over values such as
`64, 256, 1024, 4096, 8192`. Use at least five repetitions, report the median
and include host-to-device transfer and CPU fallback time. A larger GPU does
not automatically prefer the smallest launch threshold. The RTX 4060
100 TeV path used for current development favored 4096 over 64, while older
small-shower tests had a different optimum.

For a new many-core CPU, sweep FLUKA workers independently, for example
`1, 2, 4, 8`, while monitoring physical cores, memory bandwidth and worker
batch imbalance. Do not run a separate CPU ensemble concurrently with the
CUDA performance measurement: it changes both FLUKA and GPU host-side timing.

The formal timing runner is:

```bash
python validation/gpu_em/run_performance_acceptance.py \
  --executable "$C8_BUILD/applications/c8_air_shower" \
  --table "$C8_TABLE" \
  --output-root /path/to/performance_output \
  --energy-gev 1000000 \
  --events 1 \
  --repetitions 5 \
  --cache-mode warm \
  --require-release-build \
  --minimum-speedup 5
```

Inspect `gpu_em/summary.yaml` and the performance manifest rather than relying
only on wall time. Important counters include GPU/CPU particle counts,
wavefronts, named fallback counts, queue peaks, spill/overflow, peak device
memory and kernel/transfer/CPU-fallback/host-postprocess time.

### Multi-GPU nodes

One `c8_air_shower` process currently controls one CUDA device. A single
shower is not divided across multiple GPUs. For ensemble production, launch
one independent process per GPU with disjoint seeds and output directories:

```bash
CUDA_VISIBLE_DEVICES=0 \
  "$C8_BUILD/applications/c8_air_shower" \
  ... --gpu-device 0 --seed 20000001 -f /path/to/output/gpu0 &

CUDA_VISIBLE_DEVICES=1 \
  "$C8_BUILD/applications/c8_air_shower" \
  ... --gpu-device 0 --seed 30000001 -f /path/to/output/gpu1 &

wait
```

With `CUDA_VISIBLE_DEVICES` each process sees its assigned physical GPU as
logical device zero. Reserve enough CPU cores for each process's FLUKA workers
and host-side scheduler. On NUMA servers, bind each process and its memory to
the CPU socket closest to the selected GPU only after checking the topology
with `nvidia-smi topo -m`.

### Minimum validation after hardware migration

A new server is ready for scientific production only after all of the
following:

1. Release CUDA build and CTest pass.
2. FLUKA is discovered, `fluka_batch_worker` starts and no unintended UrQMD
   substitution occurs.
3. The production `.c8emrt` SHA-256 and table tolerance are verified.
4. A fixed-seed CUDA run repeats on that machine.
5. CPU/CUDA shower distributions pass for representative primary, energy,
   zenith, cut and thinning configurations.
6. CoREAS/ZHS geomagnetic amplitude and pulse-width distributions pass for the
   actual antenna layout.
7. Cold-cache and warm-cache performance are recorded separately.
8. Output reports `complete: true`, no illegal fallback, no overflow and a
   closed energy ledger where the strict ledger applies.

Use `validation/gpu_em/run_physics_acceptance.py` for independent CPU/CUDA
ensembles and
`validation/gpu_em/analyze_geomagnetic_pulse_distributions.py` for the radio
amplitude/width comparison. Their detailed contracts and other validation
tools are documented in
[validation/gpu_em/README.md](validation/gpu_em/README.md).

### Common migration failures

- `nvidia-smi` fails: fix the host/WSL driver before changing CMake.
- `nvcc` is missing: load or install the CUDA toolkit; the driver alone is not
  a compiler.
- CMake rejects the GPU architecture: use a toolkit that recognizes the new
  compute capability and recreate the build directory.
- CMake finds CUDA from one installation and `nvcc` from another: cleanly
  configure with one toolkit in `PATH`/`CUDAToolkit_ROOT`.
- `WITH_FLUKA=ON` cannot find `libflukahp.a`: verify `C8_FLUKALIB` or `FLUPRO`
  and the FLUKA binary's compiler/runtime compatibility.
- `fluka-process` cannot find its worker: install or copy
  `fluka_batch_worker` beside `c8_air_shower`, or pass its explicit path.
- a copied CMake cache references the old server: create a new out-of-source
  build instead of editing absolute cache paths.
- the CUDA table is rejected: compare table hash, schema, cut, energy range,
  media and requested tolerance; do not silently fall back to CPU.
- performance is unexpectedly poor: verify `Release`, warm-cache state,
  architecture-specific compilation, exclusive GPU use, batch threshold,
  FLUKA worker count and that diagnostic flags are disabled.

## Running applications and examples

### Standard applications

Applications for standard use-cases are located in the `applications` directory.
These are example scripts that can be used directly or slightly modified for your use case.
See [applications/README.md] for more.
The applications are compiled automatically after running `make` and will appear your `corsika-build/bin` directory.
After running `make install` the binaries will also be copied into your `corsika-install/bin` directory as well.


For example, from inside your `corsika-install/bin` directory, run
```shell
c8_air_shower --pdg 2212 -E 1e5 -f my_shower
```
This will run a vertical 100 TeV proton shower and will create and put the output into `./my_shower`.


### Building the examples

Unlike the applications, the examples must be compiled as a second step.
From your top corsika directory, (the one that includes `corsika-build` and `corsika-install`) run
```shell
export CONAN_DEPENDENCIES=$PWD/corsika-install/lib/cmake/dependencies
cmake -DCMAKE_TOOLCHAIN_FILE=${CONAN_DEPENDENCIES}/conan_toolchain.cmake -DCMAKE_PREFIX_PATH=${CONAN_DEPENDENCIES} -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_BUILD_TYPE=RelWithDebInfo -Dcorsika_DIR=$PWD/corsika-build -DWITH_FLUKA=ON -S $PWD/corsika/examples -B $PWD/corsika-build-examples
cd corsika-build-examples
make -j4 #The number should match the number of available cores on your machine
```

You can run the examples by using the binaries in `corsika-build-examples/bin/`.
For example:
```shell
corsika-build-examples/bin/known_particles
```
This will print out all of the particles that are known by CORSIKA.


### Generating doxygen documentation

To generate the documentation, you need doxygen and graphviz. If you work with
the docker corsika/devel containers this is already included.
Otherwise, e.g. on Ubuntu machines, do:
```shell
sudo apt-get install doxygen graphviz
```
Switch to the `corsika-build` directory and do
```shell
make docs
make install
```
open with firefox:
```shell
firefox ../corsika-install/share/corsika/doc/html/index.html
```
