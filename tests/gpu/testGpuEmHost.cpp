/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/ProcessSequenceCompatibility.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/framework/process/BoundaryCrossingProcess.hpp>
#include <corsika/framework/process/DecayProcess.hpp>
#include <corsika/framework/process/InteractionProcess.hpp>
#include <corsika/framework/process/SecondariesProcess.hpp>
#include <corsika/framework/process/StackProcess.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>

namespace {

  using namespace corsika::gpu::em;

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  bool equal(PhiloxCounter const& a, PhiloxCounter const& b) {
    for (std::size_t i = 0; i < 4; ++i) {
      if (a.words[i] != b.words[i]) {
        return false;
      }
    }
    return true;
  }

  void testParticleState() {
    static_assert(std::is_trivially_copyable_v<EmParticleState>);
    static_assert(std::is_standard_layout_v<EmParticleState>);
    static_assert(alignof(EmParticleState) == 16);

    EmParticleState state{};
    state.pid = static_cast<std::int32_t>(EmPid::Photon);
    state.energy_GeV = 1.e8;
    state.position_m[2] = 100000.;
    state.direction[2] = -1.;
    state.history_id = 17;

    require(state.pid == 22, "photon PID is not the PDG value");
    require(state.energy_GeV == 1.e8, "particle energy changed");
    require(state.position_m[2] == 100000., "particle position changed");
    require(state.direction[2] == -1., "particle direction changed");
    require(state.history_id == 17, "particle history ID changed");
  }

  void testReferenceVector() {
    PhiloxCounter const counter{};
    PhiloxKey const key{};
    auto const result = philox4x32_10(counter, key);

    require(result.words[0] == 0x6627e8d5U,
            "Philox reference word 0 differs");
    require(result.words[1] == 0xe169c58dU,
            "Philox reference word 1 differs");
    require(result.words[2] == 0xbc57ac4cU,
            "Philox reference word 2 differs");
    require(result.words[3] == 0x9b00dbd8U,
            "Philox reference word 3 differs");
  }

  void testIdentityKey() {
    RandomNumberKey const original{1234, 7, 91, 12, 4, 2};
    auto const expected = randomWords(original);
    require(equal(randomWords(original), expected),
            "identical random keys are not reproducible");

    auto changed = original;
    changed.seed++;
    require(!equal(randomWords(changed), expected), "seed is absent from RNG key");
    changed = original;
    changed.shower_id++;
    require(!equal(randomWords(changed), expected),
            "shower ID is absent from RNG key");
    changed = original;
    changed.history_id++;
    require(!equal(randomWords(changed), expected),
            "history ID is absent from RNG key");
    changed = original;
    changed.step_id++;
    require(!equal(randomWords(changed), expected),
            "step ID is absent from RNG key");
    changed = original;
    changed.process_id++;
    require(!equal(randomWords(changed), expected),
            "process ID is absent from RNG key");
    changed = original;
    changed.draw_id++;
    require(!equal(randomWords(changed), expected),
            "draw ID is absent from RNG key");

    for (std::uint64_t draw = 0; draw < 64; ++draw) {
      changed = original;
      changed.draw_id = draw;
      auto const value = uniformOpen01(changed);
      require(value > 0., "uniform random value reached zero");
      require(value < 1., "uniform random value reached one");
    }
  }

  void testDefaults() {
    GpuEmConfig const config{};
    ProposalTableSet const table_descriptor{};
    require(config.device == 0, "default CUDA device changed");
    require(config.min_batch_size == 4096, "default batch size changed");
    require(config.memory_fraction == 0.70, "default memory fraction changed");
    require(config.table_tolerance == 1.e-3, "default table tolerance changed");
    require(config.deterministic, "deterministic mode is not the default");
    require(
        table_descriptor.format_version ==
            corsika::gpu::em::tables::RateTableFormatVersion,
        "default PROPOSAL table format differs from the current rate-table schema");
    require(MaxAtmosphereLayers == 5, "environment schema is not five-layer");
  }

  void testScalarProposalCutResolution() {
    using namespace corsika;
    using corsika::proposal::optimized_proposal_energy_cut;
    require(
        optimized_proposal_energy_cut(0.5_MeV) == 0.4_MeV,
        "0.5 MeV CORSIKA cut no longer resolves to the scalar 0.4 MeV "
        "PROPOSAL table");
    require(
        optimized_proposal_energy_cut(1_MeV) == 1_MeV,
        "an exact standard PROPOSAL cut was changed");
    require(
        optimized_proposal_energy_cut(0.01_MeV) == 0.01_MeV,
        "a cut below the standard PROPOSAL cache grid was changed");
  }

  void testInteractionRecords() {
    static_assert(std::is_trivially_copyable_v<EmInteractionRecord>);
    static_assert(std::is_standard_layout_v<EmInteractionRecord>);
    static_assert(std::is_trivially_copyable_v<ProposalFallbackEvent>);

    EmInteractionRecord record{};
    record.status = EmInteractionStatus::Selected;
    record.component_hash = 0xfedcba9876543210ULL;
    record.input_index = 17;
    require(record.status == EmInteractionStatus::Selected,
            "interaction status changed");
    require(record.component_hash == 0xfedcba9876543210ULL,
            "interaction component hash was truncated");
    require(record.input_index == 17,
            "interaction input identity changed");

    ProposalFallbackEvent fallback{};
    fallback.reason =
        ProposalFallbackReason::LossQuantileOutOfRange;
    fallback.component_hash = record.component_hash;
    require(fallback.reason ==
                ProposalFallbackReason::LossQuantileOutOfRange,
            "fallback reason changed");
    require(fallback.component_hash == record.component_hash,
            "fallback component hash was truncated");

    static_assert(
        std::is_trivially_copyable_v<PhotonPairFinalStateRecord>);
    static_assert(
        std::is_standard_layout_v<PhotonPairFinalStateRecord>);
    PhotonPairFinalStateRecord final_state{};
    final_state.secondary_count = 2;
    final_state.secondary_offset = 41;
    require(final_state.secondary_count == 2 &&
                final_state.secondary_offset == 41,
            "photon-pair final-state record changed");
  }

  void testProcessCapabilities() {
    auto const photon = static_cast<std::int32_t>(EmPid::Photon);
    auto const electron =
        static_cast<std::int32_t>(EmPid::Electron);
    require(gpuProcessCapability(photon, PhotonPairProcessId) ==
                GpuProcessCapability::PhotonPair,
            "photon pair production is not GPU enabled");
    require(gpuProcessCapability(photon, ComptonProcessId) ==
                GpuProcessCapability::Compton,
            "photon Compton scattering is not GPU enabled");
    require(gpuProcessCapability(electron, ComptonProcessId) ==
                GpuProcessCapability::GpuFinalStateNotImplemented,
            "non-photon Compton interaction was accepted by the GPU");
    require(
        gpuProcessCapability(photon, PhotoelectricProcessId) ==
            GpuProcessCapability::Photoelectric,
        "photon photoelectric absorption is not GPU enabled");
    require(
        gpuProcessCapability(electron, PhotoelectricProcessId) ==
            GpuProcessCapability::GpuFinalStateNotImplemented,
        "non-photon photoelectric interaction was accepted by the GPU");
    require(gpuProcessCapability(electron, PhotonPairProcessId) ==
                GpuProcessCapability::GpuFinalStateNotImplemented,
            "non-photon photopair was accepted by the GPU");
    require(gpuProcessCapability(photon, PhotoproductionProcessId) ==
                GpuProcessCapability::CpuOnlyFinalState,
            "photoproduction is not forced to CPU");
    require(gpuProcessCapability(photon, PhotonMuonPairProcessId) ==
                GpuProcessCapability::CpuOnlyFinalState,
            "photon-induced muon pairs are not forced to CPU");
    require(gpuProcessCapability(electron, BremsProcessId) ==
                GpuProcessCapability::Bremsstrahlung,
            "electron bremsstrahlung GPU capability is missing");
    require(
        gpuProcessCapability(
            static_cast<std::int32_t>(EmPid::Positron),
            BremsProcessId) ==
            GpuProcessCapability::Bremsstrahlung,
        "positron bremsstrahlung GPU capability is missing");
    require(
        gpuProcessCapability(
            static_cast<std::int32_t>(EmPid::Positron),
            AnnihilationProcessId) ==
            GpuProcessCapability::Annihilation,
        "positron annihilation GPU capability is missing");
    require(
        gpuProcessCapability(electron, AnnihilationProcessId) ==
            GpuProcessCapability::GpuFinalStateNotImplemented,
        "electron annihilation was incorrectly accepted by the GPU");
    require(
        gpuProcessCapability(electron, IonizationProcessId) ==
            GpuProcessCapability::Ionization &&
            gpuProcessCapability(
                static_cast<std::int32_t>(EmPid::Positron),
                IonizationProcessId) ==
                GpuProcessCapability::Ionization,
        "electron/positron ionization GPU capability is missing");
    require(
        gpuProcessCapability(electron, ElectronPairProcessId) ==
                GpuProcessCapability::ElectronPair &&
            gpuProcessCapability(
                static_cast<std::int32_t>(EmPid::Positron),
                ElectronPairProcessId) ==
                GpuProcessCapability::ElectronPair,
        "electron/positron pair-production GPU capability is missing");
    require(processFallbackReason(
                GpuProcessCapability::CpuOnlyFinalState) ==
                ProposalFallbackReason::CpuOnlyProcess,
            "CPU-only process fallback reason differs");
    require(
        proposalNativeSelectionRequiresReplay(
            photon, PhotoproductionProcessId, 0.5) &&
            proposalNativeSelectionRequiresReplay(
                photon, ComptonProcessId, 1.e-8) &&
            !proposalNativeSelectionRequiresReplay(
                electron, ComptonProcessId, 1.e-8),
        "native endpoint replay did not fail closed for an unsupported "
        "particle/process combination");
    require(
        std::string(gpuEmProcessName(PhotoproductionProcessId)) ==
            "photoproduction" &&
            std::string(gpuEmProcessName(PhotonMuonPairProcessId)) ==
                "photon_muon_pair" &&
            std::string(gpuEmProcessName(-1)) == "unknown",
        "human-readable GPU process provenance differs");
    require(
        std::string(proposalFallbackReasonName(
                        ProposalFallbackReason::CpuOnlyProcess)) ==
                "cpu_only_process" &&
            std::string(proposalFallbackReasonName(
                        ProposalFallbackReason::LossQuantileOutOfRange)) ==
                "loss_quantile_out_of_range",
        "human-readable fallback reason provenance differs");
  }

  class RegisteredContinuous final
      : public corsika::ContinuousProcess<RegisteredContinuous> {};

  class UnknownContinuous final
      : public corsika::ContinuousProcess<UnknownContinuous> {};

  class RegisteredSecondaries final
      : public corsika::SecondariesProcess<RegisteredSecondaries> {};

  class UnknownSecondaries final
      : public corsika::SecondariesProcess<UnknownSecondaries> {};

  class RegisteredInteraction final
      : public corsika::InteractionProcess<RegisteredInteraction> {};

  class UnknownInteraction final
      : public corsika::InteractionProcess<UnknownInteraction> {};

  class RegisteredDecay final
      : public corsika::DecayProcess<RegisteredDecay> {};

  class UnknownDecay final
      : public corsika::DecayProcess<UnknownDecay> {};

  class RegisteredBoundary final
      : public corsika::BoundaryCrossingProcess<RegisteredBoundary> {};

  class UnknownBoundary final
      : public corsika::BoundaryCrossingProcess<UnknownBoundary> {};

  class RegisteredStack final
      : public corsika::StackProcess<RegisteredStack> {
  public:
    RegisteredStack()
        : corsika::StackProcess<RegisteredStack>(1) {}
  };

  class UnknownStack final
      : public corsika::StackProcess<UnknownStack> {
  public:
    UnknownStack()
        : corsika::StackProcess<UnknownStack>(1) {}
  };

  void testProcessSequenceCompatibility() {
    using ReplacedRegistration =
        GpuEmStepProcessRegistration<
            RegisteredContinuous,
            GpuEmStepProcessPolicy::ReplacedOnDevice>;
    using UnknownRegistration =
        GpuEmStepProcessRegistration<
            UnknownContinuous,
            GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>;
    using PartialRegistry =
        GpuEmStepProcessRegistry<ReplacedRegistration>;
    using CompleteRegistry =
        GpuEmStepProcessRegistry<
            ReplacedRegistration, UnknownRegistration>;

    RegisteredContinuous registered;
    UnknownContinuous unknown;
    auto nested = corsika::make_sequence(
        registered, corsika::make_sequence(unknown));
    using Sequence = decltype(nested);

    static_assert(
        PartialRegistry::
            unregisteredContinuousProcessCount<Sequence>() == 1);
    static_assert(!PartialRegistry::compatible<Sequence>());
    static_assert(CompleteRegistry::compatible<Sequence>());
    require(
        PartialRegistry::
                unregisteredContinuousProcessCount<Sequence>() == 1,
        "unknown continuous process was not discovered recursively");
    bool rejected = false;
    try {
      PartialRegistry::validateOrThrow<Sequence>();
    } catch (std::runtime_error const& error) {
      rejected =
          std::string(error.what()).find(
              "1 ContinuousProcess") != std::string::npos;
    }
    require(
        rejected,
        "unknown continuous process did not cause a CUDA startup rejection");
    CompleteRegistry::validateOrThrow<Sequence>();
    require(
        CompleteRegistry::registrationCount() == 2 &&
            CompleteRegistry::replacedOnDeviceCount() == 1 &&
            CompleteRegistry::
                    replayedFromDeviceRecordCount() == 1,
        "process registry policy counts differ");

    using WholeSequenceRegistration =
        GpuEmStepProcessRegistration<
            Sequence,
            GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>;
    using WholeSequenceRegistry =
        GpuEmStepProcessRegistry<WholeSequenceRegistration>;
    static_assert(WholeSequenceRegistry::compatible<Sequence>());
    require(
        WholeSequenceRegistry::compatible<Sequence>(),
        "explicitly registered process-sequence wrapper was not accepted");

    auto switched = corsika::make_select(
        [](auto const&) { return true; }, registered, unknown);
    using SwitchedSequence = decltype(switched);
    static_assert(
        PartialRegistry::
            unregisteredContinuousProcessCount<SwitchedSequence>() == 1);
    static_assert(!PartialRegistry::compatible<SwitchedSequence>());
    static_assert(CompleteRegistry::compatible<SwitchedSequence>());
    require(
        PartialRegistry::
                unregisteredContinuousProcessCount<SwitchedSequence>() == 1,
        "unknown continuous process inside SwitchProcessSequence was not "
        "discovered recursively");
    bool switchedRejected = false;
    try {
      PartialRegistry::validateOrThrow<SwitchedSequence>();
    } catch (std::runtime_error const& error) {
      switchedRejected =
          std::string(error.what()).find(
              "1 ContinuousProcess") != std::string::npos;
    }
    require(
        switchedRejected,
        "unknown continuous process inside SwitchProcessSequence did not "
        "cause a CUDA startup rejection");

    using RegisteredSecondariesRegistration =
        GpuEmStepProcessRegistration<
            RegisteredSecondaries,
            GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>;
    using SecondariesRegistry =
        GpuEmStepProcessRegistry<RegisteredSecondariesRegistration>;
    RegisteredSecondaries registered_secondaries;
    UnknownSecondaries unknown_secondaries;
    auto secondary_sequence = corsika::make_sequence(
        registered_secondaries, unknown_secondaries);
    using SecondarySequence = decltype(secondary_sequence);
    static_assert(
        SecondariesRegistry::
            unregisteredSecondariesProcessCount<SecondarySequence>() == 1);
    static_assert(!SecondariesRegistry::compatible<SecondarySequence>());
    bool secondaries_rejected = false;
    try {
      SecondariesRegistry::validateOrThrow<SecondarySequence>();
    } catch (std::runtime_error const& error) {
      secondaries_rejected =
          std::string(error.what()).find(
              "1 SecondariesProcess") != std::string::npos;
    }
    require(
        secondaries_rejected,
        "unknown secondaries process did not cause a CUDA startup rejection");

    using RegisteredInteractionRegistration =
        GpuEmStepProcessRegistration<
            RegisteredInteraction,
            GpuEmStepProcessPolicy::ReplacedOnDevice>;
    using RegisteredDecayRegistration =
        GpuEmStepProcessRegistration<
            RegisteredDecay,
            GpuEmStepProcessPolicy::DeferredToCpu>;
    using RegisteredBoundaryRegistration =
        GpuEmStepProcessRegistration<
            RegisteredBoundary,
            GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>;
    using RegisteredStackRegistration =
        GpuEmStepProcessRegistration<
            RegisteredStack,
            GpuEmStepProcessPolicy::DiagnosticOnly>;
    using CategoryRegistry = GpuEmStepProcessRegistry<
        RegisteredInteractionRegistration,
        RegisteredDecayRegistration,
        RegisteredBoundaryRegistration,
        RegisteredStackRegistration>;

    RegisteredInteraction registered_interaction;
    UnknownInteraction unknown_interaction;
    RegisteredDecay registered_decay;
    UnknownDecay unknown_decay;
    RegisteredBoundary registered_boundary;
    UnknownBoundary unknown_boundary;
    RegisteredStack registered_stack;
    UnknownStack unknown_stack;
    auto category_sequence = corsika::make_sequence(
        registered_interaction, unknown_interaction,
        registered_decay, unknown_decay,
        registered_boundary, unknown_boundary,
        registered_stack, unknown_stack);
    using CategorySequence = decltype(category_sequence);
    static_assert(
        CategoryRegistry::
            unregisteredInteractionProcessCount<CategorySequence>() == 1);
    static_assert(
        CategoryRegistry::
            unregisteredDecayProcessCount<CategorySequence>() == 1);
    static_assert(
        CategoryRegistry::
            unregisteredBoundaryProcessCount<CategorySequence>() == 1);
    static_assert(
        CategoryRegistry::
            unregisteredStackProcessCount<CategorySequence>() == 1);
    static_assert(!CategoryRegistry::compatible<CategorySequence>());
    bool categories_rejected = false;
    try {
      CategoryRegistry::validateOrThrow<CategorySequence>();
    } catch (std::runtime_error const& error) {
      auto const message = std::string{error.what()};
      categories_rejected =
          message.find("1 InteractionProcess") != std::string::npos &&
          message.find("1 DecayProcess") != std::string::npos &&
          message.find("1 BoundaryCrossingProcess") != std::string::npos &&
          message.find("1 StackProcess") != std::string::npos;
    }
    require(
        categories_rejected,
        "unknown interaction/decay/boundary/stack processes did not cause "
        "a CUDA startup rejection");

    using CompleteCategoryRegistry = GpuEmStepProcessRegistry<
        RegisteredInteractionRegistration,
        GpuEmStepProcessRegistration<
            UnknownInteraction,
            GpuEmStepProcessPolicy::InapplicableToRoutedEm>,
        RegisteredDecayRegistration,
        GpuEmStepProcessRegistration<
            UnknownDecay,
            GpuEmStepProcessPolicy::DeferredToCpu>,
        RegisteredBoundaryRegistration,
        GpuEmStepProcessRegistration<
            UnknownBoundary,
            GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
        RegisteredStackRegistration,
        GpuEmStepProcessRegistration<
            UnknownStack,
            GpuEmStepProcessPolicy::DiagnosticOnly>>;
    static_assert(
        CompleteCategoryRegistry::compatible<CategorySequence>());
    CompleteCategoryRegistry::validateOrThrow<CategorySequence>();
    require(
        CompleteCategoryRegistry::replacedOnDeviceCount() == 1 &&
            CompleteCategoryRegistry::replayedFromDeviceRecordCount() == 2 &&
            CompleteCategoryRegistry::deferredToCpuCount() == 2 &&
            CompleteCategoryRegistry::inapplicableToRoutedEmCount() == 1 &&
            CompleteCategoryRegistry::diagnosticOnlyCount() == 2,
        "extended process registry policy counts differ");
  }

} // namespace

int main() {
  try {
    testParticleState();
    testReferenceVector();
    testIdentityKey();
    testDefaults();
    testScalarProposalCutResolution();
    testInteractionRecords();
    testProcessCapabilities();
    testProcessSequenceCompatibility();
  } catch (std::exception const& error) {
    std::cerr << "GPU EM host validation failed after " << checks
              << " checks: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "GPU EM host validation passed: " << checks << " checks\n";
  return EXIT_SUCCESS;
}
