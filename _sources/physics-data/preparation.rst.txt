Model selection, preparation and reuse
======================================

The application constructs the native C8 physics models before the accelerated
backend starts. The PROPOSAL bridge then reads their existing calculators and
creates owned numerical data for the backend. Model construction, disk-cache
reuse and reuse of an initialized backend are separate stages.

Application configuration and native calculators
------------------------------------------------

In `c8_air_shower.cpp`_, the environment is a five-layer ``USStdBK`` atmosphere
with ``Medium::AirDry1Atm``. The application chooses the low- and high-energy
hadronic models and their transition energy, then supplies those objects to
``backend.interaction(env, sophia, sibyll->getHadronInteractionModel(), threshold)``.
This returns the application's ``proposal::AirShowerInteraction`` adapter.
``proposal::ContinuousProcess(env, dEdX)`` supplies continuous losses for the
non-hadronic branch; hadrons use ``BetheBlochPDG``. Selecting a Kokkos execution
backend does not itself select a new set of PROPOSAL parametrizations.

The four tracking cuts are kinetic-energy thresholds. Their minimum becomes
the production threshold passed to the native PROPOSAL setup:

.. literalinclude:: ../../../applications/c8_air_shower.cpp
   :language: cpp
   :start-at: // tell proposal that we are interested in all energy losses above the particle cut
   :end-at: set_energy_production_threshold(Code::TauPlus, prod_threshold);
   :lineno-match:

The tracking cut and the PROPOSAL stochastic-loss cut have different roles.
``ProposalProcessBase::getOptimizedEmCut(code)`` reads the particle's production
threshold and selects the largest configured table-cut value no greater than
that threshold. If none qualifies, it uses the threshold itself. The configured
values are 1000, 100, 20, 10, 3, 1, 0.4, 0.25, 0.15 and 0.05 MeV. With the
application defaults, the EM tracking cut is 0.5 MeV and the stochastic cut is
0.4 MeV. The ordinary relative cut is ``v_cut = 0.01``; ionization uses a
relative cut of 1, and photon interactions are stochastic only.

``ProposalProcessBase(env)`` walks every environment node with model
properties. It combines the node's nuclear composition with the ionization
and reference-density parameters from ``mediumData(prop.getMedium())`` to
construct ``PROPOSAL::Medium`` values, indexed by composition hash. It records
the resolved cuts and calls the cross-section builders for each medium and
each native tracked species. ``InteractionModel`` and ``ContinuousProcess``
then construct and retain their calculators, indexed by composition hash and
particle code. Continuous calculators are omitted for photons. This native
setup includes tau leptons; native-table export selects only photons,
electrons, positrons and both muon charges.

The concrete cross-section choices are in `ProposalProcessBase.hpp`_.
``InteractionModel::makeCalculator(code, compositionHash, energyCut)`` returns
the owned secondaries, interaction and LPM-helper objects plus the process
list. ``ContinuousProcess::buildCalculator(code, compositionHash)`` returns
``void`` and stores displacement and ``MoliereInterpol`` scattering objects.
The environment and thresholds therefore need to be established before these
owners are constructed. See `ProposalProcessBase.inl`_, `InteractionModel.inl`_
and `ContinuousProcess.inl`_ for this construction order.

Calculator views and exported data
----------------------------------

The application passes its models to ``backend.execution(...)``. The execution
object retains access to them; ``run_cascade`` obtains calculator views only
when the runtime session has no backend:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/CascadeRunner.inl
   :language: cpp
   :start-at: std::vector<proposal::NativeInteractionCalculatorView> native_views;
   :end-at: native_views, native_continuous_views);
   :lineno-match:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Interface
     - Inputs, return and ownership
   * - ``nativeCalculatorViews() const``
     - Returns a vector of ``NativeInteractionCalculatorView`` or
       ``NativeContinuousCalculatorView`` values. Each carries particle,
       medium, cut and mass information and borrows const calculator references.
       Creating the views neither constructs another calculator nor advances
       a PROPOSAL random stream. The scalar owners must remain alive.
   * - ``export_proposal_native_tables(interactions, continuous)``
     - Reads the two view vectors and returns an owning
       ``ProposalNativeTableSet`` by value. It copies exported axes,
       coefficients, process/component identities and numerical constants.
       The resulting arrays do not borrow the calculators.
   * - ``validate_proposal_native_requirements(table, requirements)``
     - Returns ``void`` or throws. It requires total-rate data for all five
       routed species and displacement data for charged leptons, and checks
       the common interpolation domain against the transport cuts and maximum
       configured primary energy.
   * - ``EmRunSession::beginProposalNative(...)``
     - Receives the environment view, geometry, backend configuration, physics
       requirements and calculator views. On first use it owns the exported
       table and creates the backend. It returns ``SessionBeginResult`` with
       a borrowed backend pointer, a reuse flag and a muon-transport capability
       flag. The session owns that backend.

``prepare_cascade`` constructs ``PhysicsRequirements`` in MeV using the
configured maximum primary energy, the EM and muon tracking cuts and
``optimized_proposal_energy_cut(productionThreshold)``. The domain validator
uses total energy for charged leptons, including the rest mass and the
continuous-cut safety factor. The supplied stochastic-cut field is not
compared against the exported cut by this validator; the exporter separately
checks consistency among positive interaction-calculator cuts.

Export accepts one medium physics definition: composition, excitation energy
and density-effect parameters must agree. Layers may differ in mass density.
For each supported particle, it selects the calculator with the smallest
medium hash as the deterministic representative. Local density remains part
of transport and LPM evaluation. Unsupported particle views are skipped;
unsupported process/parametrization/particle combinations throw. This is a
specific bridge for the implemented models, not a general PROPOSAL plugin API.

The bridge uses the patched read-only export API from PROPOSAL 7.6.2 and
CubicInterpolation 0.1.5, as fixed in the `PROPOSAL recipe`_. It reads live
calculator exports rather than parsing their private disk files. Bicubic
coefficient expansion changes the storage layout without sampling another
physics table. Direct Heitler annihilation, Sauter photoeffect and
HeckC7Shadowing photoproduction rates instead carry parameters for their
implemented formulae. See `NativeCalculatorView.hpp`_,
`NativeTableExporter.cpp`_ and `NativeRequirements.cpp`_.

Auxiliary models and cache management
-------------------------------------

The native table set is only part of the numerical physics state.
``load_or_create_proposal_native_aux(interactions, directory)`` returns an
owning ``ProposalNativeAuxData``. On a cache miss it copies photon-pair and
bremsstrahlung LPM parameters from the existing helper objects and builds
electron and, when present, muon Moliere snapshots from the medium, particle
masses and PROPOSAL constants/series coefficients. These snapshots own fixed
component and coefficient arrays; they are not borrowed scattering objects.

There is no general export of ``PROPOSAL::SecondariesCalculator`` into the
device context. Supported device final states use their numerical model
implementations and the exported constants/component data. In particular,
direct-pair energy-asymmetry data are absent from the auxiliary cache, and
those final states retain their CPU path. The scalar secondaries calculators
remain owned by the native interaction model.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Data
     - Location and identity
     - Miss, mismatch or failure
   * - PROPOSAL interpolation files
     - ``ProposalProcessBase`` sets
       ``PROPOSAL::InterpolationSettings::TABLES_PATH`` to
       ``corsika_data("PROPOSAL")``. ``CORSIKA_DATA`` overrides the compiled
       data directory. Table creation/loading belongs to PROPOSAL.
     - Native construction can build tables when they are unavailable on
       disk. The bridge exports the resulting calculator state. Exported
       cache counters record availability before the builder ran; they do
       not measure the whole application's cold-start time.
   * - Auxiliary model snapshots
     - ``<key>.c8emaux`` under ``--gpu-aux-cache-dir``, or
       ``$XDG_CACHE_HOME/corsika8/gpu-em-aux``, falling back to
       ``$HOME/.cache/corsika8/gpu-em-aux``. The key includes format/algorithm
       and dependency versions, medium physics identities, particle IDs and
       stochastic cuts.
     - Absent, incompatible or corrupt files trigger regeneration under a
       per-key lock, followed by validation and atomic publication. Locking,
       generation and I/O failures propagate. Default environment paths must
       be absolute. This version-four format is ABI- and byte-order-dependent.
   * - Derived Moliere interpolation
     - The auxiliary filename plus ``.moliere-initial-v1.c8cache``. The file
       includes the generating snapshot and validation metadata. It stores
       Moliere-function polynomials and an inverse-sampling initial-guess grid.
     - Missing, corrupt or snapshot-mismatched files are rebuilt. Unlike the
       auxiliary snapshot writer, this cache can fall back to the generated
       in-memory result when the lock or file publication fails.

The auxiliary cache key is calculated before checking the cold-generation
helpers. A valid hit can therefore be read without live LPM helpers; a cold
generation requires photon-pair and electron bremsstrahlung helpers, and a
muon bremsstrahlung helper when muon views are supplied. The same-medium
restrictions also apply to cold auxiliary generation. See
`ProposalNativeAux.cpp`_ and `MoliereInterpolation.cpp`_.

An initialized ``EmRunSession`` is a fourth, in-memory lifetime: it keeps the
export and device allocations across showers without reopening these files.
Changing files on disk does not refresh a running session.

Context publication and session reuse
-------------------------------------

``BackendInstanceImplementation::initialize`` validates the table structure
and content hash, copies a host table, uploads environment arrays, and fills
``NativeTableStorage`` and ``MoliereInterpolationStorage``. Their uploads
complete before the context is published. ``InteractionPhysicsView`` combines
the native-table view with the EM and muon transport cuts.

The context call shows the independent inputs that meet at this boundary:

.. literalinclude:: ../../../corsika/detail/kokkos/em/execution/BackendInstanceImplementation.inl
   :language: cpp
   :start-at: impl_->physicsContext_.initialize(
   :end-at: impl_->config_.showerId_, impl_->execution_);
   :lineno-match:

``PhysicsContext::initialize(...)`` returns ``void``. It copies a numerical
``PhysicsContextData`` record to one rank-zero device allocation and fences
that publication. The record contains value snapshots for Moliere and LPM,
borrowed views of the uploaded tables/environment/profile arrays, geometry,
thinning settings and random keys. The backend owns the referenced allocations;
retaining ``getDeviceView()`` alone does not extend their lifetime. See
`BackendInstanceImplementation.inl`_ and `PhysicsContext.hpp`_.

On subsequent ``beginProposalNative`` calls, the session compares the
numerical environment snapshot and geometry key, including the observation
plane and magnetic-deflection limit. An equivalent snapshot at different
host addresses is accepted; changed numerical content throws. The successful
reuse branch calls ``beginShower`` to update random seed, shower ID and
thinning, and to reset shower-local counters/output state. It does not export
calculators again, revalidate the supplied physics requirements, or apply new
transport cuts from ``BackendConfig``. Create a new session when changing
models, cuts, required energy coverage or environment. The existing scalar
owners must also be reconstructed when their construction inputs change.

Missing data and initialization failures
----------------------------------------

Unavailable disk files and unavailable numerical physics are distinct cases.
Export rejects null required calculator references, missing total-rate or
displacement interpolation, malformed nonempty coefficient exports and direct
processes without an implemented representation. An empty continuous-loss
export is valid for a process with no interpolated continuous contribution.
Domain validation rejects requested transport energies outside the supported
common range. These preparation failures throw; this layer does not synthesize
replacement physics or automatically turn the whole run into a scalar run.

Failure recovery depends on the stage. A requirements failure leaves the
exported host table but no backend, so a corrected call can retry. If backend
construction has succeeded but its ``initialize`` call throws, the session
retains that backend pointer. ``EmRunSession::isInitialized()`` tests this
pointer, so it can return true after such a failure; it is not a transactional
success marker. The environment key has not been committed, and an ordinary
retry is rejected. Discard that session and construct another. These states
are explicit in `EmRunSession.cpp`_ and the ``early-failure``/``late-failure``
cases in `Session.cpp`_.

.. _c8_air_shower.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/applications/c8_air_shower.cpp
.. _ProposalProcessBase.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/modules/proposal/ProposalProcessBase.hpp
.. _ProposalProcessBase.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/modules/proposal/ProposalProcessBase.inl
.. _InteractionModel.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/modules/proposal/InteractionModel.inl
.. _ContinuousProcess.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/modules/proposal/ContinuousProcess.inl
.. _NativeCalculatorView.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/modules/proposal/NativeCalculatorView.hpp
.. _NativeTableExporter.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/proposal/NativeTableExporter.cpp
.. _NativeRequirements.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/proposal/NativeRequirements.cpp
.. _PROPOSAL recipe: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/externals/conan/proposal/conanfile.py
.. _ProposalNativeAux.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/proposal/ProposalNativeAux.cpp
.. _MoliereInterpolation.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/tables/MoliereInterpolation.cpp
.. _BackendInstanceImplementation.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/BackendInstanceImplementation.inl
.. _PhysicsContext.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/execution/PhysicsContext.hpp
.. _EmRunSession.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmRunSession.cpp
.. _Session.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/session/Session.cpp
