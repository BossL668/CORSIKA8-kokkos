# Version-locked Kokkos package

This directory contains the project Conan recipe for unmodified Kokkos 4.7.03
sources, used by the portable electromagnetic and radio backends. Choose one
package configuration per build tree:

- `openmp`: OpenMP only, with no GPU runtime;
- `cuda`, `hip`, or `sycl`: the selected GPU backend plus the required Serial
  host execution space, with OpenMP disabled;
- `cuda_openmp`: CUDA, OpenMP and Serial in one package, for the optional
  combined air application. It requires a working NVIDIA runtime even when the
  application selects OpenMP. This is not the CPU-only deployment package.

Export the recipe once into the active Conan cache:

```bash
conan export dependencies/kokkos --user=c8gpu --channel=stable
```

The root CORSIKA Conan options then select the package variant. GPU architecture
names use Kokkos spelling, for example `ADA89` for an RTX 4060.

[`profiles/`](profiles) contains independent-backend profiles and the combined
`cuda-openmp-ada89` example. Root Conan option `kokkos_backend=cuda_openmp`
corresponds to CMake `CORSIKA_KOKKOS_BACKEND=CUDA_OPENMP`. The independent helper
can detect supported NVIDIA architectures; the combined helper defaults to Ada
and needs `C8_KOKKOS_DUAL_PROFILE` for other targets.

[`locks/`](locks) preserves historical beta4 recipe revisions, not a current
beta5 release lock. The build helpers do not consume these locks. Export the
current patched recipes and record the actual resolved dependency graph for
reproduction; do not blindly apply a historical lock to newer recipes. Never
mix generated dependency/toolchain directories between variants.

The HIP and SYCL profiles are concrete starting points for ROCm 6.x/LLVM 18
and oneAPI 2025. Their compiler executable and Conan compiler version must be
updated together when the destination uses another toolkit release.

Current validation status:

- OpenMP and CUDA are compiled and exercised in this repository;
- HIP and SYCL have versioned profiles and fail-closed CMake selection, but
  require a matching ROCm or oneAPI build host before they can be called
  validated;
- independent GPU packages use Serial host control and disable OpenMP; the
  combined package is an explicit exception, not automatic cooperation;
- the OpenMP-only package contains no GPU execution space.

Start with the [English installation guide](../../README.md) or
[中文教程](../../README_CN.md), including all three recipe exports. A profile is
not a driver installer and does not establish physics or performance acceptance.
