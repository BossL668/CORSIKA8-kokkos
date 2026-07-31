/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <SetupTestEnvironment.hpp>

#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/gpu/em/ProposalCpuFallbackHandler.hpp>
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/setup/SetupStack.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

  using namespace corsika;
  using namespace corsika::gpu::em;
  using FallbackStack =
      setup::HybridStack<DummyEnvironment>;

  std::size_t checks{};

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  class DummyHadronicModel {
  public:
    explicit DummyHadronicModel(HEPEnergyType threshold)
        : threshold_(threshold) {}

    template <typename TSecondaryView>
    void doInteraction(
        TSecondaryView& view, Code, Code,
        FourMomentum const& projectile,
        FourMomentum const&) {
      auto const energy =
          projectile.getTimeLikeComponent();
      auto const& coordinate_system =
          view.getProjectile()
              .getMomentum()
              .getCoordinateSystem();
      for (int index = 0; index < 2; ++index) {
        view.addSecondary(std::make_tuple(
            Code::PiPlus, energy / 2,
            DirectionVector{
                coordinate_system, {1., 0., 0.}}));
      }
    }

    bool isValid(
        Code, Code, HEPEnergyType energy) const {
      return energy >= threshold_;
    }

  private:
    HEPEnergyType threshold_;
  };

  class NoOpSecondaries {
  public:
    template <typename TSecondaryView>
    void doSecondaries(TSecondaryView&) {
      ++calls_;
    }

    std::uint64_t calls() const { return calls_; }

  private:
    std::uint64_t calls_{};
  };

  ProposalFallbackEvent makeFallback(
      proposal::ProposalInteractionRecord const& record,
      EmParticleState const& particle) {
    ProposalFallbackEvent event{};
    event.particle = particle;
    event.process_id =
        static_cast<std::int32_t>(record.type);
    event.reason =
        ProposalFallbackReason::CpuOnlyProcess;
    event.component_hash = record.component_hash;
    event.medium_hash = record.context.medium_hash;
    event.interaction_hash =
        record.interaction_hash;
    event.energy_fraction = record.v_loss;
    event.selection_uniform =
        record.selection_uniform;
    event.random_draw_id = 2;
    return event;
  }

} // namespace

int main() {
  try {
    logging::set_level(logging::level::warn);
    RNGManager<>::getInstance()
        .registerRandomStream("proposal");
    auto [environment_ptr, coordinate_system_ptr,
          environment_node] =
        setup::testing::setup_environment(
            Code::Oxygen);
    auto& environment = *environment_ptr;
    auto const coordinate_system =
        *coordinate_system_ptr;
    set_energy_production_threshold(
        Code::Electron, 0.5_MeV);
    set_energy_production_threshold(
        Code::Positron, 0.5_MeV);
    DummyHadronicModel hadronic_low(100_MeV);
    DummyHadronicModel hadronic_high(10_GeV);
    proposal::InteractionModel proposal_model(
        environment, hadronic_low, hadronic_high,
        80_GeV);

    FallbackStack sampling_stack;
    auto sampling_particle =
        sampling_stack.addParticle(std::make_tuple(
            Code::Electron,
            10_GeV,
            DirectionVector{
                coordinate_system, {1., 0., 0.}},
            Point{
                coordinate_system, 1_m, 2_m, 3_m},
            4_ns));
    sampling_particle.setNode(environment_node);
    auto const rates = proposal_model.getRateTable(
        sampling_particle, Code::Electron);
    auto sample_cpu_only =
        [&](auto& model,
            proposal::ProposalRateTable const& rate_table) {
          auto selected = model.sampleInteraction(
              sampling_particle, Code::Electron,
              rate_table, 0.375);
          auto found =
              gpuProcessCapability(
                  static_cast<std::int32_t>(
                      EmPid::Electron),
                  static_cast<std::int32_t>(
                      selected.type)) ==
              GpuProcessCapability::CpuOnlyFinalState;
          auto cumulative = 0.;
          for (auto const& entry :
               rate_table.entries()) {
            if (!found && entry.rate > 0.) {
              auto const selection_uniform =
                  (cumulative + 0.5 * entry.rate) /
                  rate_table.totalRate();
              auto candidate = model.sampleInteraction(
                  sampling_particle, Code::Electron,
                  rate_table, selection_uniform);
              if (gpuProcessCapability(
                      static_cast<std::int32_t>(
                          EmPid::Electron),
                      static_cast<std::int32_t>(
                          candidate.type)) ==
                  GpuProcessCapability::
                      CpuOnlyFinalState) {
                selected = std::move(candidate);
                found = true;
              }
            }
            cumulative += entry.rate;
          }
          if (!found) {
            throw std::runtime_error(
                "PROPOSAL did not expose a CPU-only interaction for the fallback test");
          }
          return selected;
        };
    auto record = sample_cpu_only(
        proposal_model, rates);
    require(
        record.type !=
            PROPOSAL::InteractionType::Undefined,
        "PROPOSAL did not expose a CPU-only interaction for the fallback test");

    auto state = router_detail::toDeviceState(
        sampling_particle, coordinate_system,
        42, 7, 3, 9);
    auto const event = makeFallback(record, state);
    require(
        hasSpecifiedProposalFinalState(event),
        "sampled PROPOSAL record did not form a specified GPU fallback");

    NoOpSecondaries process_list;
    using Handler = ProposalCpuFallbackHandler<
        FallbackStack,
        decltype(proposal_model),
        NoOpSecondaries,
        DummyEnvironment>;
    Handler handler{
        proposal_model, process_list, environment,
        coordinate_system, 12345, 6};
    require(
        handler.canHandle(event),
        "CPU-only PROPOSAL process was not authorized");
    auto invalid_final_state_event = event;
    invalid_final_state_event.reason =
        ProposalFallbackReason::InvalidFinalState;
    require(
        hasSpecifiedProposalFinalState(
            invalid_final_state_event) &&
            !permitsSpecifiedProposalCpuFinalState(
                invalid_final_state_event) &&
            !handler.canHandle(invalid_final_state_event),
        "syntactically complete invalid final state was silently authorized");

    auto epair_envelope_event = event;
    epair_envelope_event.reason =
        ProposalFallbackReason::EpairRejectionEnvelopeExceeded;
    epair_envelope_event.process_id =
        static_cast<std::int32_t>(
            PROPOSAL::InteractionType::Epair);
    require(
        permitsSpecifiedProposalCpuFinalState(
            epair_envelope_event) &&
            handler.canHandle(epair_envelope_event),
        "specified Epair envelope fallback was rejected");
    epair_envelope_event.process_id =
        static_cast<std::int32_t>(
            PROPOSAL::InteractionType::Brems);
    require(
        !permitsSpecifiedProposalCpuFinalState(
            epair_envelope_event) &&
            !handler.canHandle(epair_envelope_event),
        "Epair envelope reason authorized a different process");

    FallbackStack output_stack;
    // Production rate tables carry PROPOSAL::Medium::GetHash(), not the
    // CORSIKA NuclearComposition key used by InteractionModel::calc_.  The
    // handler must resolve the latter from the imported vertex node.
    auto table_hash_event = event;
    table_hash_event.medium_hash ^= 0x5a5a5a5aU;
    require(
        table_hash_event.medium_hash !=
            event.medium_hash &&
            hasSpecifiedProposalFinalState(table_hash_event),
        "fallback fixture did not create distinct medium hash domains");
    handler.handle(output_stack, table_hash_event);
    require(
        handler.statistics().specified_interactions == 1 &&
            handler.statistics().generated_secondaries > 0,
        "specified PROPOSAL fallback statistics differ");
    require(
        process_list.calls() == 1,
        "specified PROPOSAL fallback skipped doSecondaries");
    require(
        output_stack.getEntries() ==
            handler.statistics().generated_secondaries,
        "specified PROPOSAL fallback left its consumed parent or lost secondaries");
    for (auto&& particle : output_stack) {
      require(
          particle.getEnergy() >=
              get_mass(particle.getPID()),
          "specified PROPOSAL fallback generated an invalid secondary energy");
    }

    // The GPU inverse-CDF tables deliberately exclude numerically delicate
    // tails. In that case process and component are already selected, while
    // v must be completed by the exact CPU cross section using the retained
    // loss quantile. In particular, the event's default energy_fraction=0
    // must never be mistaken for a valid e-pair final state.
    proposal::ProposalRateEntry const* epair_entry =
        nullptr;
    for (auto const& entry : rates.entries()) {
      if (entry.type ==
              PROPOSAL::InteractionType::Epair &&
          entry.rate > 0.) {
        epair_entry = &entry;
        break;
      }
    }
    require(
        epair_entry != nullptr,
        "PROPOSAL fixture did not expose an electron-pair rate");
    auto unresolved_loss_event = event;
    unresolved_loss_event.process_id =
        static_cast<std::int32_t>(
            PROPOSAL::InteractionType::Epair);
    unresolved_loss_event.component_hash =
        epair_entry->component_hash;
    unresolved_loss_event.reason =
        ProposalFallbackReason::LossQuantileOutOfRange;
    unresolved_loss_event.energy_fraction = 0.;
    unresolved_loss_event.loss_quantile = 0.99;
    require(
        !hasSpecifiedProposalFinalState(
            unresolved_loss_event) &&
            hasResolvableProposalSelectedLoss(
                unresolved_loss_event),
        "inverse-CDF tail fallback was misclassified as a complete final state");

    NoOpSecondaries resolving_process_list;
    Handler resolving_handler{
        proposal_model, resolving_process_list,
        environment, coordinate_system, 12345, 6};
    require(
        resolving_handler.canHandle(
            unresolved_loss_event),
        "CPU fallback handler rejected a resolvable selected loss");
    FallbackStack resolving_output_stack;
    resolving_handler.handle(
        resolving_output_stack,
        unresolved_loss_event);
    require(
        resolving_handler.statistics()
                    .specified_interactions ==
                1 &&
            resolving_handler.statistics()
                    .completed_selected_losses ==
                1 &&
            resolving_handler.statistics()
                    .generated_secondaries >
                0 &&
            resolving_process_list.calls() == 1 &&
            resolving_output_stack.getEntries() ==
                resolving_handler.statistics()
                    .generated_secondaries,
        "CPU completion of a GPU-selected loss did not produce a final state");
    for (auto&& particle : resolving_output_stack) {
      require(
          std::isfinite(
              particle.getEnergy() / 1_GeV) &&
              particle.getEnergy() >=
                  get_mass(particle.getPID()),
          "selected-loss completion generated a non-finite or invalid secondary");
    }

    // CORSIKA normally resolves a requested 0.5 MeV production threshold to
    // its cached 0.4 MeV PROPOSAL calculator. Emulate a GPU table made with a
    // distinct exact cut and prove the fallback cache can decode that record
    // without replacing the scalar calculator above.
    set_energy_production_threshold(
        Code::Electron, 0.037_MeV);
    proposal::InteractionModel exact_cut_model(
        environment, hadronic_low, hadronic_high,
        80_GeV);
    auto const exact_rates =
        exact_cut_model.getRateTable(
            sampling_particle, Code::Electron);
    auto const exact_record = sample_cpu_only(
        exact_cut_model, exact_rates);
    require(
        exact_record.interaction_hash !=
            record.interaction_hash,
        "exact-cut fixture did not produce a distinct calculator hash");
    auto exact_event = makeFallback(
        exact_record, state);
    exact_event.medium_hash ^= 0x5a5a5a5aU;
    NoOpSecondaries exact_process_list;
    Handler exact_handler{
        proposal_model, exact_process_list,
        environment, coordinate_system, 12345, 6,
        0.037_MeV};
    FallbackStack exact_output_stack;
    exact_handler.handle(
        exact_output_stack, exact_event);
    require(
        exact_handler.statistics().specified_interactions ==
                1 &&
            exact_handler.statistics().generated_secondaries >
                0 &&
            exact_process_list.calls() == 1 &&
            exact_output_stack.getEntries() ==
                exact_handler.statistics()
                    .generated_secondaries,
        "exact-cut specified PROPOSAL fallback did not complete");

    auto broken = event;
    ++broken.interaction_hash;
    FallbackStack broken_stack;
    bool rejected = false;
    try {
      handler.handle(broken_stack, broken);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    require(
        rejected,
        "specified PROPOSAL fallback accepted a mismatched calculator hash");

    std::cout
        << "GPU specified PROPOSAL fallback passed "
        << checks << " checks and generated "
        << handler.statistics().generated_secondaries
        << " secondaries\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr
        << "GPU specified PROPOSAL fallback failed after "
        << checks << " checks: " << error.what()
        << '\n';
    return 1;
  }
}
