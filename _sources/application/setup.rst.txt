Execution selection and setup
=============================

This page covers the calls made before the per-event loop. ``air`` is the
application's namespace alias for ``corsika::kokkos::application``. The two
headers exposing the entry interfaces are included in ``c8_air_shower.cpp``:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: #include <corsika/kokkos/application/cascade/CascadeExtension.hpp>
   :end-at: #include <corsika/kokkos/application/ExecutionSelection.hpp>
   :lineno-match:

1. Select process organization
------------------------------

The application still enters through ``main``. Before constructing models, it
inspects whether the command line requires coordinated execution:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: int main(int argc, char** argv) try {
   :end-at: CLI::App app{"Simulate standard (downgoing) showers with CORSIKA 8."};
   :lineno-match:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Inputs, return and effect
   * - ``select_execution(argc, argv)``
     - Reads the original argument vector. Returns ``ExecutionSelection``,
       containing parsed device identifiers and a coordination flag. It does
       not initialize physics models, random streams or Kokkos.
   * - ``isCoordinated()``
     - Returns ``bool``. In a coordinator-enabled build, multiple selected GPUs
       or an explicit ``--prefix-workers`` option select this branch.
   * - ``runCoordinator(argc, argv, true)``
     - Passes the original arguments to the coordinator and returns its integer
       exit status. ``true`` declares that this application implements resident
       worker sessions. The coordinator manages subprocesses; it does not
       construct the parent application's C8 models.
   * - ``devices()``
     - Returns ``DeviceSelection const&``. The direct path passes it to backend
       option validation below.

Native C8, OpenMP and a single GPU without a parallel prefix use the direct
path. An invalid device selection is reported through ``CascadeAborted``.
Calling ``runCoordinator`` on a direct selection throws ``std::logic_error``.

Implementation: `ExecutionSelection.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/ExecutionSelection.hpp>`_
and `CascadeSupport.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/cascade/CascadeSupport.cpp>`_.

2. Register and validate backend options
----------------------------------------

The extension is constructed while the application is registering CLI options:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: air::CascadeExtension backend{app};
   :end-at: air::CascadeExtension backend{app};
   :lineno-match:

``CascadeExtension(CLI::App&, CascadeScene)`` retains a reference to the parser
and owns its ``BackendOptions``, worker options and runtime holder. The scene
defaults to ``CascadeScene::Atmosphere``. Its constructor calls
``add_backend_cli_options(...)`` and ``workers_.addOptions(...)``; it does not
construct the application's atmosphere or physics models.

After parsing, the application validates the combined selection:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: CLI11_PARSE(app, argc, argv);
   :end-at: if (!backend.prepare(executionMode.devices(), argv[0])) return EXIT_FAILURE;
   :lineno-match:

``prepare(DeviceSelection const&, char const*)`` receives the selected devices
and executable path. It reads already-parsed application options and delegates
to ``CascadeWorkerOptions::prepare(...)``. That function resolves device options,
checks worker-input combinations and calls ``validate_backend_cli_options(...)``.
It returns ``true`` on success or ``false`` for a rejected configuration; some
exceptional failures can propagate. ``main`` stops when validation returns false.

This separates checking a requested backend from actually creating the device
backend. The latter happens during execution, as described in
:ref:`application-backend-start`.

3. Configure run resources
--------------------------

After constructing the environment and shower geometry, ``main`` supplies a
``RunEnvironmentConfig``:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: air::RunEnvironmentConfig run_environment{
   :end-at: backend.configure(run_environment);
   :lineno-match:

The positional fields in this call are the magnetic-model name and year, site
latitude/longitude/altitude, the field components in tesla, the magnetic
deflection limit, the observation-plane point in metres, and the antenna-file
path. The struct's final ``scene_`` field keeps its atmosphere default here.

``configure(RunEnvironmentConfig const&)`` returns ``void``. It checks scene
compatibility, copies this configuration into the extension and calls
``CascadeRuntime::configure(...)``. In a Kokkos-enabled build, the runtime
creates a ``CascadeRunSession``. Its underlying ``EmBackend`` remains
uninitialized until ``beginProposalNative(...)``.

This configuration contains settings and metadata. The actual C8 environment
tree is passed separately through ``geometry(...)``; the next page follows
that reference.

4. Connect writers and model adapters
-------------------------------------

The extension provides several small factories used during the normal C8 setup:

.. list-table::
   :header-rows: 1
   :widths: 37 63

   * - Call in ``main``
     - Inputs and result
   * - ``registerOutput(output)``
     - Receives ``OutputManager&``; returns ``void``. Registers the available
       accelerated run output as ``gpu_em``.
   * - ``energyLossWriter(showerAxis, dX)``
     - Returns an ``EnergyLossWriter`` for the supplied axis and depth binning,
       with the configured short-track deposit convention. ``main`` registers
       this writer as ``energyloss``.
   * - ``longitudinalWriter(showerAxis, dX)``
     - Returns a ``LongitudinalWriter`` using ``--profile-crossings``. ``main``
       registers it as ``profile`` and supplies it to ``LongitudinalProfile``.
   * - ``interaction(env, sophia, sibyll->getHadronInteractionModel(), heHadronModelThreshold)``
     - Returns ``proposal::AirShowerInteraction``. It receives the environment,
       low- and high-energy photonuclear models and their transition energy.
   * - ``interactionCounter(leIntModel)``
     - Returns a counter around the supplied low-energy hadronic model. With
       FLUKA enabled, the library uses ``make_fluka_interaction_counter(...)``;
       otherwise it constructs ``InteractionCounter{model}``.

These factories produce objects used by the application and its process
sequence. Their existence does not imply that the associated calculation runs
on a GPU.

Implementation: `CascadeExtensionOptions.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeExtensionOptions.hpp>`_,
`CascadeExtension.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/cascade/CascadeExtension.cpp>`_
and `CascadeExtension.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeExtension.inl>`_.
