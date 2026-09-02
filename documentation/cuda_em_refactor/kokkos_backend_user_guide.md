# Kokkos EM and radio backend

This guide describes the experimental performance-portable backend in beta4.
It is independent of the established native CUDA backend and does not alter the
default scalar CORSIKA 8 path.

## Execution contract

One binary contains exactly one accelerated execution space:

| Build product | Particle/radio execution | Host scheduling | Explicitly absent |
|---|---|---|---|
| `kokkos-openmp` | Kokkos OpenMP | OpenMP | CUDA, HIP, SYCL |
| `kokkos-cuda` | Kokkos CUDA | Serial | OpenMP, HIP, SYCL |
| `kokkos-hip` | Kokkos HIP | Serial | OpenMP, CUDA, SYCL |
| `kokkos-sycl` | Kokkos SYCL | Serial | OpenMP, CUDA, HIP |

OpenMP and a GPU are never used together for one shower. A GPU run therefore
rejects `--kokkos-num-threads > 1` and `--hadronic-workers > 1`. MPI is not part
of this backend; a future MPI layer may distribute independent showers without
sharing particle queues across ranks.

The pre-existing paths remain available in their own builds:

```text
--em-backend proposal --radio-backend cpu       scalar reference
--em-backend cuda     --radio-backend cuda      native NVIDIA CUDA
--em-backend kokkos   --radio-backend kokkos    selected Kokkos space
```

Kokkos accepts only `--gpu-physics-source proposal-native`. It exports the
read-only spline state from the live PROPOSAL 7.6.2 calculators, uploads or maps
it once, and reuses it across showers in one process. It does not consume a
`.c8emrt` table.

## Source layout

- `corsika/accelerator/em/IAcceleratedEmBackend.hpp`: host-wavefront interface;
- `corsika/accelerator/em/NativeCudaBackendAdapter.hpp`: native-CUDA adapter;
- `corsika/accelerator/em/KokkosEmBackend.hpp`: Kokkos PIMPL boundary;
- `corsika/accelerator/em/PhysicalAcceleratedEmRouter.hpp`: device-independent
  router name; the old CUDA name remains a compatibility alias;
- `corsika/accelerator/em/detail/AcceleratedHybridCascadeRunner.hpp`: generic
  HybridCascade assembly;
- `corsika/accelerator/em/kokkos/`: proposal-native device view, deterministic
  SoA queues, photon/lepton transport and final states, profile projection and
  fixed-point accumulation;
- `corsika/accelerator/radio/detail/`: shared CoREAS/ZHS projection equations;
- `corsika/accelerator/radio/kokkos/`: Kokkos observer tiling and fixed-point
  waveform accumulator;
- `src/accelerator/em/kokkos/`: compiled runtime, session, backend and tuning
  cache implementation;
- `applications/c8_kokkos_tune.cpp`: device-specific tuning utility.

Virtual dispatch occurs only at a host wavefront boundary. Particle kernels use
the compile-time execution space and shared beta4 physics POD/functions.
History IDs and Philox keys do not depend on thread completion order. Secondary
slots and compaction order use flag + exclusive scan + scatter.

## Version-locked dependency setup

The examples assume an activated `corsika_venv`, a valid FLUKA installation,
Conan 2, CMake 3.24 or newer, and separate sibling directories. Export the
three local recipes once:

```bash
export C8_SOURCE="$PWD"

conan export "$C8_SOURCE/third_party/conan/cubicinterpolation"
conan export "$C8_SOURCE/third_party/conan/proposal"
conan export "$C8_SOURCE/dependencies/kokkos"
```

The Kokkos recipe is fixed to 4.7.03. The patched PROPOSAL and
CubicInterpolation recipes expose read-only interpolation state; they do not
change the scalar calculation.

### OpenMP-only

```bash
export C8_DEPS="$C8_SOURCE/../deps-kokkos-openmp"
export C8_BUILD="$C8_SOURCE/../build-kokkos-openmp"
export C8_INSTALL="$C8_SOURCE/../install-kokkos-openmp"

conan install "$C8_SOURCE" \
  -pr:h "$C8_SOURCE/dependencies/kokkos/profiles/openmp" \
  -pr:b default \
  --lockfile="$C8_SOURCE/dependencies/kokkos/locks/openmp.lock" \
  --output-folder="$C8_DEPS" \
  --build=missing \
  -s:h build_type=Release

cmake -S "$C8_SOURCE" -B "$C8_BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$C8_DEPS/conan_toolchain.cmake" \
  -DCONAN_CMAKE_DIR="$C8_DEPS" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_KOKKOS=ON \
  -DCORSIKA_KOKKOS_BACKEND=OPENMP \
  -DCORSIKA_ENABLE_CUDA=OFF \
  -DCMAKE_INSTALL_PREFIX="$C8_INSTALL"

cmake --build "$C8_BUILD" --parallel
cmake --install "$C8_BUILD"
```

### Kokkos CUDA for an RTX 40-series device

This profile selects Kokkos `ADA89` and CMake CUDA architecture 89. Use a new
profile/build directory for a different GPU architecture.

```bash
export C8_DEPS="$C8_SOURCE/../deps-kokkos-cuda"
export C8_BUILD="$C8_SOURCE/../build-kokkos-cuda"
export C8_INSTALL="$C8_SOURCE/../install-kokkos-cuda"

conan install "$C8_SOURCE" \
  -pr:h "$C8_SOURCE/dependencies/kokkos/profiles/cuda-ada89" \
  -pr:b default \
  --lockfile="$C8_SOURCE/dependencies/kokkos/locks/cuda-ada89.lock" \
  --output-folder="$C8_DEPS" \
  --build=missing \
  -s:h build_type=Release

cmake -S "$C8_SOURCE" -B "$C8_BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$C8_DEPS/conan_toolchain.cmake" \
  -DCONAN_CMAKE_DIR="$C8_DEPS" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_KOKKOS=ON \
  -DCORSIKA_KOKKOS_BACKEND=CUDA \
  -DCORSIKA_KOKKOS_ARCHITECTURE=ADA89 \
  -DCORSIKA_KOKKOS_CUDA_ARCHITECTURES=89 \
  -DCORSIKA_ENABLE_CUDA=OFF \
  -DCMAKE_INSTALL_PREFIX="$C8_INSTALL"

cmake --build "$C8_BUILD" --parallel
cmake --install "$C8_BUILD"
```

HIP and SYCL use the matching profile/lockfile in `dependencies/kokkos/` and
their own dependency, build and install directories. They must be configured
on a machine whose compiler and runtime support the selected device. This beta4
checkout has not yet completed hardware acceptance for those two variants.

## Runtime

The Kokkos-specific options are:

```text
--kokkos-num-threads N       OpenMP threads; GPU builds accept only 0 or 1
--kokkos-device N            device ordinal in a GPU build
--kokkos-tuning-cache PATH   exact tuning record to load
--kokkos-require-tuning      fail if the record is absent or mismatched
```

Minimal OpenMP execution:

```bash
OMP_NUM_THREADS=16 install-kokkos-openmp/bin/c8_air_shower \
  --em-backend kokkos \
  --radio-backend kokkos \
  --gpu-physics-source proposal-native \
  --kokkos-num-threads 16
```

Minimal NVIDIA execution:

```bash
install-kokkos-cuda/bin/c8_air_shower \
  --em-backend kokkos \
  --radio-backend kokkos \
  --gpu-physics-source proposal-native \
  --kokkos-device 0
```

The application refuses a Kokkos `.c8emrt` request, a backend mismatch, GPU
plus multiple hadronic workers, unsupported native spline metadata, an
insufficient table energy domain, or a required tuning-cache mismatch. It does
not silently switch to scalar or native CUDA.

## Tuning

First perform a short untuned run and read the 64-digit native table hash from
the shower metadata. Then tune the exact build/device/table combination:

```bash
install-kokkos-openmp/bin/c8_kokkos_tune \
  --backend openmp \
  --threads 16 \
  --proposal-table-hash HASH \
  --output ~/.cache/corsika8/kokkos/openmp-16.tune

install-kokkos-cuda/bin/c8_kokkos_tune \
  --backend cuda \
  --device 0 \
  --proposal-table-hash HASH \
  --output ~/.cache/corsika8/kokkos/ada89-device0.tune
```

Each candidate has one warm-up and five timed repetitions by default. The
cache key contains backend, device, architecture, driver/runtime, Kokkos and
compiler versions, project revision, thread count and proposal-native hash.
Tracked edits add a source-diff digest to a dirty development build key.

The current tuner applies radio team/track/observer tiling. Its batch value is
a workspace/capacity hint only: it deliberately does not change the router's
minimum wavefront checkpoint, because doing so would change history allocation
and the Philox stream. `device_queues` is fixed to one in this first version.

## Tests and current boundary

Run the portable-backend gates with:

```bash
ctest --test-dir "$C8_BUILD" -R 'Kokkos' --output-on-failure
```

The implemented tests cover exact backend selection, dependency boundaries,
stable queue/scan behavior, proposal-native host/device evaluation, tuning
cache matching, photon/lepton transport, profile and CoREAS/ZHS projection.
OpenMP and CUDA full-application smoke runs have completed. Native CUDA's radio
projection regression also remains green after extracting shared equations.
An aggregate one-million-query proposal-native run passes on both OpenMP and
CUDA; this is distinct from the still-pending one-million samples for every
individual PID/process/component column.

The installed target is checked with the minimal downstream project in
`tests/gpu/kokkos_install_consumer`. The installed package restores Kokkos
before importing CORSIKA targets and restores CUDA Toolkit targets only for a
Kokkos-CUDA installation.

This is not yet a production declaration for the Kokkos backend. Remaining
release gates are the specified million-point process matrix, fixed-decision
replay, 500/2000-event shower ensembles, 1--16-thread OpenMP scaling, and
100 TeV/1 PeV same-host performance comparison. Native CUDA remains the NVIDIA
production reference until those gates pass.
