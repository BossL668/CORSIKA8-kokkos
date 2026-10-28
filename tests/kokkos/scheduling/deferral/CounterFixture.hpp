/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include "ProductionInteractionCounter.hpp"
#include <corsika/kokkos/scheduling/FlukaInteractionDeferral.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <stdexcept>
#include <vector>

namespace counter_fixture {
  using namespace corsika;
  namespace reference = corsika::production_counter_reference;
  namespace production = corsika::production_stepper_reference;

  struct Secondary {
    Code pid_;
    HEPEnergyType energy_;
    double weight_;
    MomentumVector momentum_;
    Code getPID() const { return pid_; }
    HEPEnergyType getEnergy() const { return energy_; }
    double getWeight() const { return weight_; }
    MomentumVector const& getMomentum() const { return momentum_; }
  };
  struct Projectile {
    double weight_;
    double getWeight() const { return weight_; }
  };
  struct View {
    Projectile projectile_{2.};
    std::vector<Secondary> secondaries_;
    auto begin() const { return secondaries_.begin(); }
    auto end() const { return secondaries_.end(); }
    Projectile const& getProjectile() const { return projectile_; }
  };
  struct Process {
    unsigned calls_ = 0;
    bool fail_ = false;
    std::vector<Secondary> output_;
    CrossSectionType getCrossSection(Code, Code, FourMomentum const&,
                                     FourMomentum const&) const {
      return 73_mb;
    }
    template <typename TView>
    void doInteraction(TView& view, Code, Code, FourMomentum const&,
                       FourMomentum const&) {
      ++calls_;
      if (fail_) throw std::runtime_error("controlled model failure");
      view.secondaries_ = output_;
    }
  };
  struct WorkerProcess : Process {
    static constexpr auto hadronic_worker_model = production::HadronicWorkerModel::Fluka;
  };
  struct Inputs {
    Code projectile_ = Code::Proton;
    Code target_ = Code::Nitrogen;
    FourMomentum projectileP4_;
    FourMomentum targetP4_;
    explicit Inputs()
        : projectileP4_(100_GeV, MomentumVector(get_root_CoordinateSystem(),
                                                {0_GeV, 0_GeV, 99_GeV}))
        , targetP4_(get_mass(Code::Nitrogen),
                    MomentumVector(get_root_CoordinateSystem(), {0_GeV, 0_GeV, 0_GeV})) {}
    template <typename TCounter, typename TView>
    void invoke(TCounter& counter, TView& view) const {
      counter.doInteraction(view, projectile_, target_, projectileP4_, targetP4_);
    }
  };

  bool header_counter_first(Code, Code, FourMomentum const&, FourMomentum const&);
  bool header_adapter_first(Code, Code, FourMomentum const&, FourMomentum const&);
} // namespace counter_fixture
