# Kokkos library layout

The application selects a scene and constructs the native CORSIKA environment.
Common particle transport and scheduling do not depend on an application name.
Only scene-specific preparation and propagation live in the subdirectories below.

| Location | Responsibility | Used by |
| --- | --- | --- |
| `application/cascade/` | Application binding, event lifetime, output hooks and shared runner | Air, ice, firn |
| `application/cascade/SouthPoleSite.hpp` | South Pole site and coordinate convention | Ice, firn |
| `application/cascade/atmosphere/` | Validate the air composition and prepare atmospheric observations | Air shower |
| `application/cascade/ice/` | Validate the homogeneous water-ice scene | Uniform ice |
| `application/cascade/firn/` | Validate the exported exponential firn model | Firn |
| `environment/` | Export the C8 tree, shapes and media; search nodes and boundaries; evaluate density and grammage | All compatible environments |
| `environment/firn/` | Evaluate the exported firn density and its grammage integral/inverse | Firn |
| `em/` | Native PROPOSAL table access, continuous and discrete EM processes, execution backends | All supported materials |
| `radio/` and `radio/algorithms/` | Track records, observer conversion, accumulators and shared projection infrastructure | Air and uniform ice; common interfaces for further propagation models |
| `radio/atmosphere/` | Atmospheric refractivity and propagation preparation | Air shower |
| `radio/ice/` | Homogeneous-index radio configuration | Uniform ice |
| `radio/firn/` | Optical profiles, stratified rays, coherent branch responses, full-wave table access, spectra and waveforms | Firn |
| `scheduling/`, `data/`, `output/` | Particle exchange, queues, common records and output conversion | Air, ice, firn |
| `diagnostics/`, `runtime/` | Instrumentation and execution-space support | Shared; individual diagnostics may require a specific scene |

Public declarations are in this tree (`.hpp`). Inline/template implementations
mirror their paths under `corsika/detail/kokkos/` (`.inl`). Compiled implementations
are under `src/kokkos/`. Tests follow the same subsystem division under
`tests/kokkos/`. No application implementation helpers belong in `applications/`.

Examples:

- `radio/ice/HomogeneousRadioConfig.hpp` is implemented in
  `corsika/detail/kokkos/radio/ice/HomogeneousRadioConfig.inl`.
- `environment/firn/FirnModel.hpp` is implemented in
  `corsika/detail/kokkos/environment/firn/FirnModel.inl`.
- `radio/firn/StratifiedRaySolver.hpp` is implemented in
  `corsika/detail/kokkos/radio/firn/StratifiedRaySolver.inl`.

Native medium classes, including `ExponentialFirn`, remain in `corsika/media/`:
they are also usable by the scalar C8 path and are not Kokkos-only physics.
The generic exporter retains the environment's parent, child and excluded-node
relationships. Applications do not replace that tree with a sphere-layer list.

`FlatAtmosphereRadioSnapshot` currently also carries the homogeneous-propagation
mode used by ice. Its historical name does not make it an air-only interface;
it stays in the common radio directory. Similarly, the existing
`AirShowerCoordinator` is the shared process coordinator, not a second firn
scheduler. Do not copy either implementation into scene folders.

## Firn integration scope

The firn application uses the shared scalar/Kokkos EM runner and a dedicated
variable-index radio adapter. `--radio-backend cpu` selects the archived host
response algorithms. `--radio-backend kokkos --firn-radio-model full-wave`
selects `radio/firn/IFullWaveAccumulator.hpp`, implemented by the Kokkos library
in `src/kokkos/radio/firn/FullWaveAccumulator.cpp`.

The device implementation preserves schema-2 full-wave interpolation, phase
restoration and ordered accumulation of both direct and boundary branches.
It accepts bounded host-staged track batches; it is not yet a device-to-device
track connection. Phase subdivision, diagnostics and final inverse FFT remain
on the host. Uniform-ice radio is never substituted for firn radio.

Full-wave execution requires a matching external response table and rejects
out-of-domain queries. `--firn-radio-model coherent-go --radio-backend kokkos`
uses `IGeometricalOpticsAccumulator.hpp` and `DeviceStratifiedRaySolver.hpp`
for the existing exponential-profile GO solver on the device, followed by
ordered finite-track spectrum accumulation. The profile is exported exactly;
quadrature, root iterations, branch selection and phases remain those of the
CPU reference. Different observer depths are supported without a response table.
Hybrid GO/table radio remains a CPU mode; deferred prefix
radio replay and multi-process firn merging are not yet connected and are
rejected explicitly. Files under `kokkos/` do not imply every operation is a
kernel. The applications retain native C8 node search and process sequences.
