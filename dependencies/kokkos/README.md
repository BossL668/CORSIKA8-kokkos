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
