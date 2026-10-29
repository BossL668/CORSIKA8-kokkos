Optional Kokkos backend
=======================

The optional Kokkos implementation uses the existing ``c8_air_shower`` binary.
The application constructs the environment, models, observers and output writers.
Library adapters connect these objects to the selected execution backend while
retaining the process sequence, event loop and ``EAS.run()`` interface.

The build uses C++17 and Kokkos 4.7. Enabling
Kokkos does not change the default EM backend:
without an explicit backend/device selection, the application uses scalar
PROPOSAL and CPU radio.

Build configurations
--------------------

The ordinary build has ``CORSIKA_ENABLE_KOKKOS=OFF`` and does not require Kokkos.
The optional build requires a Kokkos 4.7 package. Use separate build
and install directories for OpenMP and CUDA, and build Kokkos with a compiler
compatible with the CORSIKA dependency graph. The recommended helpers prepare
the matching dependency graph and install all native applications::

   bash tools/build_kokkos.sh openmp
   bash tools/build_kokkos.sh cuda

The helpers enable FLUKA by default. Prepare a compatible licensed installation
and set ``FLUPRO`` first; ``-DWITH_FLUKA=OFF`` explicitly selects a different
low-energy model. Use ``bash tools/build_kokkos.sh cuda --check`` to check the
toolchain and automatically selected GPU profile without building. Full Conda
and venv setup instructions are in the root README.

Set ``C8_BUILD_JOBS`` to control build concurrency (default 1). For V100/V100S,
``C8_KOKKOS_PROFILE=cuda-volta70`` selects the target explicitly. The commands
below are the manual alternative for an existing external Kokkos installation.
Run ``bash tools/prepare_data.sh`` first to populate the standard data directory.

``CORSIKA_ENABLE_KOKKOS=ON`` enables the Kokkos library components. The complete
``kokkos-proposal`` air-shower path also requires
``CORSIKA_ENABLE_PROPOSAL_EXPORT=ON`` and the read-only native-table export
extensions in PROPOSAL and CubicInterpolation. Ordinary PROPOSAL libraries cannot
be combined with the patched export headers. Configuration checks both the
export API and the dependency header identity.

After preparing the ordinary CORSIKA build prerequisites and a Conan profile,
export the two dependency recipes from the project root::

   conan export externals/conan/cubicinterpolation
   conan export externals/conan/proposal

The top-level recipe pins these recipe revisions. Generate a separate dependency
graph for the enabled backend, for example::

   conan install . --output-folder=build/openmp/deps --build=missing \
       -s build_type=Release -s compiler.cppstd=17 \
       -o with_proposal_export=True

With an OpenMP-enabled Kokkos installation, configure and build using the
generated toolchain::

   cmake -S . -B build/openmp \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_TOOLCHAIN_FILE="$PWD/build/openmp/deps/conan_toolchain.cmake" \
       -DCORSIKA_ENABLE_KOKKOS=ON \
       -DCORSIKA_ENABLE_PROPOSAL_EXPORT=ON \
       -DCORSIKA_KOKKOS_BACKEND=OPENMP \
       -DKokkos_DIR=/path/to/kokkos-openmp/lib64/cmake/Kokkos \
       -DCMAKE_INSTALL_PREFIX="$PWD/install/openmp"
   cmake --build build/openmp --parallel 1
   ctest --test-dir build/openmp --output-on-failure
   cmake --install build/openmp

``Kokkos_DIR`` must point to the actual installation; some installations use
``lib`` instead of ``lib64``. The optional physics generators retain the build
requirements described in the main installation guide.

For a CUDA build, generate the dependency graph under ``build/cuda/deps`` and
use a Kokkos installation with CUDA and Serial host execution. Select
``CORSIKA_KOKKOS_BACKEND=CUDA``, the CUDA toolchain and the architecture of the
target GPU. For example, L20 uses ``CMAKE_CUDA_ARCHITECTURES=89``. The main
application uses the host C++ compiler; device implementation files use NVCC.
Use ``build/cuda`` and ``install/cuda`` for this configuration. Do not reuse the
OpenMP CMake cache or request OpenMP threads from the CUDA/Serial runtime.

Running the existing application
--------------------------------

Each install prefix contains its own native binary; there is no Python launcher.
Supply the same physical options and antenna file when comparing backends.
For example, a 100 GeV proton with the application's default environment::

   export CORSIKA_DATA="$PWD/install/openmp/share/corsika/data"
   install/openmp/bin/c8_air_shower --filename shower-native \
       --pdg 2212 --energy 100 --nevent 2 --seed 18641 \
       --em-backend proposal --radio-backend cpu

   install/openmp/bin/c8_air_shower --filename shower-openmp \
       --pdg 2212 --energy 100 --nevent 2 --seed 18641 \
       --em-backend kokkos-proposal --kokkos-execution openmp \
       --kokkos-num-threads 1 --radio-backend kokkos

   export CORSIKA_DATA="$PWD/install/cuda/share/corsika/data"
   install/cuda/bin/c8_air_shower --filename shower-cuda \
       --pdg 2212 --energy 100 --nevent 2 --seed 18641 \
       --em-backend kokkos-proposal --kokkos-execution cuda \
       --device 0 --gpu-memory-fraction 0.70 --radio-backend kokkos

Set ``CORSIKA_DATA`` to the absolute data directory of the selected installation.
This uses C8's existing data-location override: installing a binary does not
replace the source-data path compiled into it. Child processes inherit this
setting. Dependencies and model libraries still require their normal runtime
configuration.

Physical defaults such as the observation height, cuts and atmosphere are not
changed by these backend options. ``--device`` selects a physical NVIDIA GPU
index or UUID. A list such as ``--device 0,1,2,3`` invokes the C++ coordinator
and distributes each shower across those devices. ``--prefix-workers`` enables
the independent CPU prefix stage; it can also be used with one GPU. The
coordinator launches the same installed binary and merges the per-worker
outputs. It does not require a separate application or Python runtime launcher.

With ``P > 0`` CPU prefix workers, ``--prefix-split-energy 0`` resolves to
``primary_energy / (96 * P)`` in GeV. A positive explicit value overrides it.
This total-energy threshold controls when produced particles are saved for
static CPU partitioning; it is independent of the number of GPUs and
``--multigpu-frontier-energy``. The default GPU handoff threshold remains
``primary_energy / (8 * number_of_selected_GPUs)``. No CPU splitting threshold
is used for a serial prefix. These thresholds change task grouping, and may
change subsequent random histories, but do not change the physics cuts.
The resolved values are recorded as ``cpu_prefix_split_energy_effective_GeV``
and ``em_frontier_energy_effective_GeV`` in the event's ``CONFIG.json``.
The automatic coefficient was tuned with a high-energy atmospheric proton
prefix; use an explicit threshold to tune other energies, primaries or hardware.

``--gpu-min-batch`` controls GPU dispatch and small-batch checkpoints.
The ``C8_STATIC_FRONTIER_IMPORT_BATCH`` environment variable controls how many
staged particles a worker imports at a time in the static coordinator path.
These are scheduling controls; they do not specify CUDA thread counts.
``--gpu-memory-fraction`` budgets a
fraction of currently available device memory, rather than promising that the
whole process will occupy exactly that fraction of physical memory.

Application integration
-----------------------

``CascadeExtension`` registers backend options and validates the selected
configuration. ``bind`` borrows the objects already constructed by the
application; those objects must outlive the run. ``CascadeEvent`` manages the
per-event input and diagnostic lifetime. ``CascadeDriver`` selects the native
``Cascade`` or the accelerated runner.

The accelerated runner exports the environment, obtains the native PROPOSAL
tables, routes supported particle work, completes CPU fallback requests and
returns records to the native output writers. An explicit process registry
describes how each member of the process sequence participates. Adding a new
physical process to a sequence requires a corresponding integration policy;
it must not be silently omitted from accelerated transport.

Public headers are under ``corsika/kokkos``; template and inline definitions are
mirrored under ``corsika/detail/kokkos``; compiled implementations are under
``src/kokkos``. The application helpers are organized as follows:

* ``application/cascade`` contains the shared assembly facade, event configuration,
  model bindings, run sessions, diagnostics and reporting. All three applications use
  ``CascadeExtension`` and the same scheduler, CPU fallback and output machinery.
* ``application/cascade/atmosphere`` contains the air-material policy.
* ``application/cascade/ice`` contains the homogeneous-water-ice scene policy.
* ``application/cascade/firn`` contains the exponential-firn scene policy.
  The three scene directories are peers. Ice and firn share
  ``application/cascade/SouthPoleSite.hpp`` for the site convention.
  Material and propagation choices belong to the scene adapter rather than to
  the shared scheduler.
* Backend options, command-line registration and device selection remain in
  ``application`` because they are shared across applications.

The same relative directories are used for public headers, inline definitions
and compiled implementations where present. The independent native entry points are
``applications/c8_air_shower.cpp``, ``applications/c8_ice_cascade.cpp`` and
``applications/c8_firn_cascade.cpp``. Runtime selection, scheduling, environment
conversion, EM models and radio interfaces retain their own library modules.

Environment and material scope
------------------------------

``prepare_environment_snapshot`` exports the actual C8 root tree, including
parent/child links, ordered exclusion links, shapes and medium models. Host
node pointers are represented by indices in device data. Geometry and model
parameters use an explicitly chosen common coordinate system and fixed units.
Density functions are represented by their model kind and parameters, not by a
single density sampled at the injection point.

The general snapshot supports uniform density, the registered exponential
profiles and exponential firn. Shape conversion and containment support
Universe, spheres, oriented boxes and separation planes. Straight intersections
support these exported shapes; curved magnetic intersections do not yet support
every exported shape, notably boxes. Unsupported model exporters and shapes
must not be replaced silently with an air or spherical approximation.

The native table exporter accepts supported PROPOSAL calculators for the
configured medium. A session currently requires one medium physics definition:
composition, mean excitation energy and density-effect parameters must agree,
although the mass density may vary. This permits separate air, ice or rock
configurations at the library level; it is not a claim that arbitrary mixed
materials are supported by one air-shower session.

The complete air application applies its explicit air-material check in
``application/cascade/atmosphere``. Its height-dependent radio configuration is in
``radio/atmosphere``. A non-air application must supply the appropriate scene,
physics setup and propagation backend. Removing the air check alone is not a
valid implementation of an ice or mountain simulation.

Homogeneous ice application
---------------------------

``c8_ice_cascade`` builds the environment and native process sequence, then uses
exactly the same ``CascadeExtension`` interface and ``EAS.run()`` call as the air
application. Backend defaults remain scalar PROPOSAL and CPU radio. OpenMP and
CUDA are selected with the same options shown above; the current ice adapter
supports ``proposal`` and ``kokkos-proposal``, not the air-specific EGS4 adapter
or atmospheric diagnostic replay writer.

The shared ``SouthPoleSite`` convention is 90 degrees south, with a reference
surface elevation of 2835 m above C8's mean Earth radius. Longitude zero defines
the otherwise arbitrary horizontal axes at the pole. The common coordinates are
Earth-centred polar NWU: north along Earth +x, west along -y, and up along -z.
All particle and observer data are converted to those same coordinates.
The magnetic field is zero. The reference site elevation follows the
`NSF South Pole station description <https://www.nsf.gov/od/opp/ail/amundson-scott-south-pole-station>`_.

This first migration preserves the historical **bulk homogeneous ice** problem:
a distant spherical enclosure contains uniform water ice, density 0.919 g/cm3
and refractive index 1.78 by default. The reference surface defines depth; it
is **not** an ice--air interface. No surface refraction, reflection or firn ray
tracing is introduced by this application. The C8 environment tree, exported
material model, node search and observation plane remain the transport geometry.
The homogeneous radio propagator uses a single direct path with delay n R/c,
without attenuation. Whole rays must remain inside the uniform convex volume.

``--injection-height`` and ``--observation-level`` use metres above mean Earth
radius, as in the air application; their defaults, 2635 and 2585 m, correspond
to 200 and 250 m below the reference surface. ``--surface-altitude``,
``--density-g-cm3``, ``--refractive-index`` and ``--profile-bin-g-cm2`` configure
the scene. Antenna-file columns are north, west and altitude in metres, with
the third column also measured above mean Earth radius, not depth below ice.
For example, after configuring model-data paths for the selected installation::

   printf '10 0 2635\n0 10 2635\n' > ice-antennas.txt
   install/cuda/bin/c8_ice_cascade --filename ice-electron \
       --pdg 11 --energy 1 --nevent 2 --seed 27102751 --emthin 0 \
       --antenna-file ice-antennas.txt \
       --em-backend kokkos-proposal --radio-backend kokkos \
       --kokkos-execution cuda --device 0 --gpu-memory-fraction 0.70

Firn application and module boundaries
--------------------------------------

``c8_firn_cascade`` uses the same ``CascadeExtension`` binding and native process
sequence as ``c8_ice_cascade``. It selects a planar exponential water-ice density
and refractive-index profile, with zero magnetic field at the same South Pole
site. The native model and its exported snapshot use the same parameters::

   n(z) = n_deep - delta_n exp(z/H)
   rho(z) = rho_deep - rho_deep delta_n/(n_deep-1) exp(z/H)

Here z is height relative to the surface, not distance from the Earth's centre.
Defaults are rho_deep=0.917 g/cm3, n_deep=1.78, delta_n=0.423 and H=77 m.
The application constructs a pure-firn box from the surface to 5000 m depth,
with a horizontal half-width of 10000 m. The C8 environment tree determines
containment and exit. There is no air transport outside this domain; this is not
an air-to-ice shower calculation. The box avoids treating an oriented surface
plane as a closed volume. The source, observation plane and antennas must lie
inside the configured volume.

Hadronic water-ice showers require a low-energy model supporting both hydrogen
and oxygen. Use a FLUKA-enabled build for these runs. The release's default
UrQMD cross-section table covers nitrogen, oxygen and argon only; the application
does not remove hydrogen or replace its interactions to bypass that limitation.

The radio model uses separate local surface-centred NWU coordinates. Every C8
track is explicitly converted into those coordinates before evaluating a Green
response. Antenna-file columns remain north, west and altitude above mean Earth
radius, matching the ice application. The source and observation altitude
options retain that convention too.

``--firn-radio-model`` selects ``coherent-go``, ``full-wave`` or
``hybrid-full-wave``. The coherent GO model retains direct, reflected and turning
branches, including their polarization and phase; caustics outside its validity
are rejected. Full-wave modes require a schema-v2 manifest supplied through
``--firn-radio-table``. Profile fingerprints, hashes and table domains are checked;
strict full-wave mode does not extrapolate. The explicitly selected hybrid uses
ordinary GO outside the certified table domain. Radio response tables are
external inputs, separate from automatically prepared native PROPOSAL EM tables.

Use ``--radio-backend cpu`` for the host reference or
``--radio-backend kokkos --firn-radio-model full-wave`` for the Kokkos full-wave
accumulator. The latter runs the archived schema-2 interpolation, phase restoration
and direct/boundary spectrum accumulation on the build-selected execution space.
It accepts bounded host-staged track batches. Phase subdivision, diagnostics and
final inverse FFT remain on the host; this is not yet a device-to-device track
connection. No atmospheric or uniform-ice radio projector is used for firn.

``--radio-backend kokkos --firn-radio-model coherent-go`` selects device ray
solving and ordered coherent spectrum accumulation without a full-wave table.
It exports the exact exponential optical-profile parameters and retains the
CPU solver's Simpson quadrature, bracketing scans, root iterations, grazing
cutoffs, Fresnel factors, branch phases and finite-track sinc. Observers can
have different depths. The existing host subdivision controller and final FFT
are retained; geometry is computed once per child/observer and reused across
frequencies. The workspace is bounded independently of the event length.
No-path, fold-guard, invalid-geometry and branch-capacity failures stop the
radio event explicitly. This is ordinary GO, not a replacement for full-wave
propagation in shadow/caustic regions. The device profile currently supports
the application's exponential, frequency-independent attenuation model;
other CPU profiles are not silently approximated by it.

The GO spectrum uses ``exp(-i omega T)`` for delay and ``exp(+i omega t)``
for time synthesis. Total internal reflection therefore uses the exterior
vertical wavenumber with negative imaginary part, so the evanescent field
decays away from the surface. The point-source turning phase is zero before
a ray-tube caustic and ``+pi/2`` after one caustic. A depth turning point alone
does not imply a caustic; the sign of ``dX/dp`` distinguishes the two branches
in this monotonic profile. The isolated ``+/-pi/4`` stationary-phase factors
must not be used as complete Green-function phases.

For exponential firn, both implementations use an analytic turning-leg
``dX/dp`` for geometric spreading. This avoids a clipped finite-difference
stencil near a horizontal receiving ray. Other CPU profiles retain a bounded
numerical derivative. These fixes apply to ordinary GO only; agreement with
another GO implementation does not validate the wave field at a shadow edge
or a caustic. In particular, an ARZ charge-excess source and individual C8
tracks are different source models, even when their longitudinal profiles agree.

Kokkos full-wave radio requires a matching response manifest and rejects
out-of-domain queries. Hybrid GO/table execution remains a host mode. Multi-process
firn merging and deferred prefix radio replay are still rejected explicitly.
Radio output is registered once as ``firn_radio``; the shared runner's two radio
slots do not produce duplicate CoREAS/ZHS outputs.

``--firn-samples`` must be a power of two (default 4096);
``--radio-sampling-rate-ghz`` and ``--firn-time-origin-ns`` define the Fourier
window. Choose a window containing the physical arrival times: a Fourier window
is periodic, not an automatically centred pulse window. For a short verification::

   printf '10 0 2635\n' > firn-antennas.txt
   install/cuda/bin/c8_firn_cascade --filename firn-electron \
       --pdg 11 --energy 0.1 --nevent 1 --seed 27102761 --emthin 0 \
       --antenna-file firn-antennas.txt --emcut 0.0005 \
       --em-backend kokkos-proposal --kokkos-execution cuda \
       --radio-backend kokkos --firn-radio-model coherent-go --gpu-memory-fraction 0.70

For full-wave device radio, provide the response table matching the profile, source and
observer domain. The archived 100 MeV South Pole table uses a profile floor of
-1000 m and observers at -100 m; a bounded example is::

   printf '563 0 2735\n' > firn-fullwave-antennas.txt
   install/cuda/bin/c8_firn_cascade --filename firn-fullwave \
       --pdg 11 --energy 0.01 --nevent 2 --seed 27102783 --emthin 0 \
       --surface-altitude 2835 --injection-height 2621.9 \
       --observation-level 2571.9 --firn-depth 1000 --ring 0 --emcut 0.0005 \
       --antenna-file firn-fullwave-antennas.txt --firn-samples 2048 \
       --firn-time-origin-ns 2000 --firn-radio-model full-wave \
       --firn-radio-table /path/to/southpole_fullwave_100mev_v3.yaml \
       --em-backend kokkos-proposal --radio-backend kokkos \
       --kokkos-execution cuda --device 0 --gpu-memory-fraction 0.70

The ``firn_radio`` summary records the actual execution space, batches, transfers,
allocated device bytes, table-domain failures and completed-kernel timings. Its
``device_direct_input`` field remains false for this host-staged integration.

``corsika/kokkos/README.md`` maps the shared, atmospheric, ice and firn folders.
Inline implementations mirror this layout under ``corsika/detail/kokkos/``;
applications own scene configuration and native processes, not device scheduling.

Radio extension contract
------------------------

``radio::IRadioBackend`` is a host dispatch interface. A concrete backend owns
its propagation model, observer data and geometry; the interface does not
require an atmospheric height table or a particular medium.

``enqueueRadioTracks`` must copy the input records before returning, since the
caller may immediately reuse its storage. Records use the backend's configured
common coordinates and units. ``downloadRadioWaveforms`` must finish pending
work, including the final partial batch, before returning. CoREAS field and
ZHS potential conventions are defined by ``RadioWaveforms``. One shower owner
uses each instance; concurrent calls are not implied by this contract.

The present air runner connects the atmospheric implementation through
``BorrowedRadioBackend``. Other implementations may dispatch Kokkos kernels
using different propagation data, including complex medium geometry. This
extension point does not itself implement firn ray tracing, reflection or
transmission at material interfaces; those require a concrete backend and
their own physical validation.

Validation boundaries
---------------------

Run the complete CTest suite for the configured backend and test the installed
binary. Environment conversion tests, sampling tests, same-input transport
tests, complete showers and multi-process handoff checks verify different
contracts. Passing one does not substitute for the others. CPU/GPU or different
prefix-worker configurations can change random histories, so whole-shower
physics comparisons require statistical tests rather than bitwise equality.
