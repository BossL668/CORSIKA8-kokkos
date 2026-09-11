# beta5 geographic terrain tools

From the beta5 source root in your Python/conda environment:

```bash
python -m pip install ./python
c8-terrain --bounds 86.700 42.930 86.710 42.940 --output ./my-region
```

This downloads or reuses verified DEM/geoid caches, prepares the watertight mesh
and an ENU scene for the existing native five-layer atmosphere application.
With no antenna input, a **demonstration observer** 1 m above the terrain at the
region centre is created; it is not the measured 21CMA array.

Use `--antennas-csv FILE` for decimal-degree geodetic stations, or
`--station-directory DIR` for the original measured 21CMA station-centre files.
Use `--config FILE` for explicit height/datum policies and mesh/material settings.
`--offline` refuses network requests and `--estimate-only` writes nothing.

```bash
c8-terrain --config configs/mountain/terrain_region_21cma.yaml \
  --check-with ../build/mountain-openmp/applications/c8_terrain_environment
```

`--run --application /path/to/c8_terrain_cascade -- ...` forwards exact application
arguments, including `--em-backend kokkos`, after scene preparation. Injection,
energy and output remain explicit. This is transport/geometry preparation, not
completion of cross-interface radio or a global solid-Earth model.

See `documentation/terrain_preparation_workflow_CN.md` in the source tree for the
full guide, limitations, tests and original module mapping.
