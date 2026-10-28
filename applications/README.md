# CORSIKA 8 Applications

This directory contains standard applications which are typical for astroparticle physics solutions.
Select the environment, physics models and propagation mode required for the study;
the applicable limitations are documented in the backend guide.

For example, `c8_air_shower.cpp` would be a similar binary to what would be built by CORSIKA 7 and will simulate
air showers in a curved atmosphere.

`c8_ice_cascade.cpp` provides the homogeneous-ice application, and
`c8_firn_cascade.cpp` provides the stratified firn application. They share the
library adapters in `corsika/kokkos/application/cascade`, with scene policies
under `atmosphere`, `ice` and `firn`. Applications retain the native process
sequence and `EAS.run()` entry; backend implementation belongs in the library.

All three native binaries are installed in the selected prefix's `bin` directory.
See [installation](../README.md) and [backend and scene options](../documentation/kokkos.rst).
