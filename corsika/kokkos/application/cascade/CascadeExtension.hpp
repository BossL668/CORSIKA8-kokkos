/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

/* clang-format off */
// Include before boost/type_traits, as required by InteractionCounter.
#include <corsika/framework/process/InteractionCounter.hpp>
/* clang-format on */

#include <corsika/kokkos/application/cascade/CascadeExtensionOptions.hpp>
#include <corsika/kokkos/application/cascade/CascadeDriver.hpp>
#include <corsika/kokkos/diagnostics/DecisionReplayTape.hpp>
#include <corsika/output/ReplayTrace.hpp>
#include <corsika/output/SimulationTiming.hpp>
#include <corsika/modules/proposal/AirShowerInteraction.hpp>
#include <corsika/modules/writers/EnergyLossWriter.hpp>
#include <corsika/modules/writers/LongitudinalWriter.hpp>
#include <exception>
#include <type_traits>

#ifdef WITH_FLUKA
#include <corsika/kokkos/scheduling/FlukaInteractionDeferral.hpp>
#endif

namespace corsika::kokkos::application {
  namespace detail {
    template <typename TEnvironment>
    struct EnvironmentMedium;
    template <typename TMedium>
    struct EnvironmentMedium<Environment<TMedium>> {
      using medium_type = TMedium;
    };
  } // namespace detail

  /** Event lifetime hooks; the caller retains the official process sequence. */
  template <typename TDriver, typename TGeometry, typename TMonitors>
  class CascadeEvent {
    using medium_type =
        typename detail::EnvironmentMedium<std::remove_reference_t<decltype(
            std::declval<TGeometry>().environment_)>>::medium_type;
    struct ConfigurePrimary {
      template <typename T>
      void operator()(T&) const {}
    };

  public:
    CascadeEvent(TDriver&, BackendOptions const&, CascadeWorkerOptions const&,
                 TGeometry const&, TMonitors, std::uint64_t, int, double);
    CascadeEvent(CascadeEvent const&) = delete;
    CascadeEvent& operator=(CascadeEvent const&) = delete;
    CascadeEvent(CascadeEvent&&) = delete;
    CascadeEvent& operator=(CascadeEvent&&) = delete;
    ~CascadeEvent();
    template <typename TStack, typename TPrimary>
    void addPrimary(TStack&, TPrimary const&);
    template <typename TTracking, typename TSequence, typename TOutput, typename TStack,
              typename TInspector, typename TThinning>
    auto cascade(TTracking&, TSequence&, TOutput&, TStack&, TInspector&, TThinning&,
                 HEPEnergyType, double);

  private:
    TDriver& driver_;
    BackendOptions const& options_;
    CascadeWorkerOptions const& workers_;
    TGeometry const& geometry_;
    TMonitors monitors_;
    std::uint64_t ordinal_;
    double thinningFraction_;
    decltype(std::declval<TDriver&>().captureRadio()) radio_;
    decltype(std::declval<TDriver&>().captureDiagnostics(
        std::declval<TMonitors>().photoHigh_.statistics(),
        std::declval<TMonitors>().photoLow_.statistics())) before_;
    std::optional<CascadeParticleInput<medium_type>> input_;
    ConfigurePrimary configure_;
    int exceptions_;
  };

  /** Bind once to caller-owned native models and outputs; own only backend hooks. */
  template <typename TG, typename TO, typename TM, typename TD>
  class CascadeModule {
    using driver_type = CascadeDriver<TG, TO, TM, TD>;

  public:
    template <typename TOutput>
    CascadeModule(CascadeRuntime&, BackendOptions const&, CascadeWorkerOptions const&,
                  CascadeEventConfig, TG, TO, TM, TD, TOutput&, int);
    CascadeModule(CascadeModule const&) = delete;
    CascadeModule& operator=(CascadeModule const&) = delete;
    CascadeModule(CascadeModule&&) = delete;
    CascadeModule& operator=(CascadeModule&&) = delete;
    ~CascadeModule();
    auto shower(std::uint64_t ordinal);

  private:
    BackendOptions const& options_;
    CascadeWorkerOptions const& workers_;
    TG geometry_;
    TD monitors_;
    CascadeEventConfig defaults_;
    int events_;
    SimulationTiming timing_;
    std::optional<driver_type> driver_;
  };

} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadeExtension.inl>
