# Version-locked Kokkos package

This directory contains the unmodified Kokkos 4.7.03 Conan recipe used by the
portable electromagnetic and radio backends. Each package contains exactly one
execution backend:

- `openmp`: OpenMP only, with no GPU runtime;
- `cuda`, `hip`, or `sycl`: the selected GPU backend plus the required Serial
  host execution space, with OpenMP disabled.

Export the recipe once into the active Conan cache:

```bash
conan export dependencies/kokkos --user=c8gpu --channel=stable
```

The root CORSIKA Conan options then select the package variant. GPU architecture
names use Kokkos spelling, for example `ADA89` for an RTX 4060.

`profiles/` contains one host profile for each supported build product and
`locks/` records the exact dependency recipe revisions used for the initial
beta4 implementation.  The four lockfiles intentionally have the same Conan
dependency graph: backend and architecture are package options supplied by the
corresponding profile and are not duplicated in Conan's graph lock.  Never mix
the generated dependency/toolchain directories between variants.

The HIP and SYCL profiles are concrete starting points for ROCm 6.x/LLVM 18
and oneAPI 2025. Their compiler executable and Conan compiler version must be
updated together when the destination uses another toolkit release.

Current validation status:

- OpenMP and CUDA are compiled and exercised in this repository;
- HIP and SYCL have versioned profiles and fail-closed CMake selection, but
  require a matching ROCm or oneAPI build host before they can be called
  validated;
- a GPU package always enables Kokkos Serial for host control and disables
  Kokkos OpenMP; the OpenMP package contains no GPU execution space.
