/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::egs4::application {

  // The supplied photon model is C8's HadronicPhotonModel, NOT its PROPOSAL
  // interaction selector. EGS4 has already selected the vertex and air nucleus.
  // C8 may select a nucleon inside that nucleus for its LE model, exactly as in
  // its existing implementation. No EM rate, free path or branch is resampled.
  // Arrival is an explicit application hook: bool(particle, request). It handles
  // host transport cuts/output for already-generated native daughters; returning
  // false asks this handler to erase the particle. It must not erase it itself.
  // C7 polarization scalars are available in request but are not C8 global spin.
  template <class TStack, class TEnvironment, class TPhotonModel, class TSecondaries,
            class TArrival>
  class HostHandler {
    using topology_type =
        EnvironmentTopology<typename TEnvironment::BaseNodeType::IModelProperties>;
    topology_type const& topology_;
    TEnvironment const& environment_;
    ::corsika::CoordinateSystemPtr coordinates_;
    TPhotonModel& photons_;
    TSecondaries& secondaries_;
    TArrival& arrival_;
    HostStatistics statistics_{};
    std::unordered_set<std::uint64_t> consumed_;
    bool failed_{};

  public:
    HostHandler(TEnvironment const& environment, topology_type const& topology,
                ::corsika::CoordinateSystemPtr coordinates, TPhotonModel& photons,
                TSecondaries& secondaries, TArrival& arrival)
        : topology_(topology)
        , environment_(environment)
        , coordinates_(std::move(coordinates))
        , photons_(photons)
        , secondaries_(secondaries)
        , arrival_(arrival) {
      if (!coordinates_)
        throw std::invalid_argument("Missing EGS4 host coordinate system");
    }
    bool canHandle(HostRequest const& r) const {
      using namespace ::corsika;
      auto const& p = r.particle_;
      if (failed_ || !p.historyId_ || consumed_.count(p.historyId_) ||
          !std::isfinite(p.energyGeV_) || !std::isfinite(p.weight_) || p.weight_ <= 0. ||
          !std::isfinite(p.timeS_))
        return false;
      double n = 0.;
      for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(p.positionM_[i]) || !std::isfinite(p.direction_[i]))
          return false;
        n += p.direction_[i] * p.direction_[i];
      }
      if (std::abs(n - 1.) > 1.e-8) return false;
      try {
        auto code = convert_from_PDG(static_cast<PDGCode>(p.pid_));
        if (p.energyGeV_ < get_mass(code) / 1_GeV) return false;
        if (r.kind_ == HostRequestKind::muon_transport)
          return code == Code::MuMinus || code == Code::MuPlus;
        if (r.kind_ == HostRequestKind::hadron_transport) return is_hadron(code);
        if (r.kind_ == HostRequestKind::photonuclear_many_hadrons) {
          // Current native SDPM selector supports the actual AIR-NTP nuclei only.
          int z =
              r.targetPdg_ == 1000070140
                  ? 7
                  : r.targetPdg_ == 1000080160 ? 8 : r.targetPdg_ == 1000180400 ? 18 : 0;
          return code == Code::Photon && r.photonuclearBranch_ == 4 && z != 0 &&
                 r.targetAtomicNumber_ == z;
        }
      } catch (std::exception const&) { return false; }
      // Unresolved PIGEN/RHO/RESDEC is not silently substituted with a C8 decay.
      return false;
    }
    std::size_t handle(TStack& stack, HostRequest const& r) {
      using namespace ::corsika;
      if (!canHandle(r))
        throw std::invalid_argument("Unsupported or duplicate EGS4 host request");
      // A generator/output exception can leave a partial stack; prohibit replay
      // instead of generating the same selected vertex twice.
      try {
        auto state = r.particle_;
        Point const position(coordinates_, state.positionM_[0] * 1_m,
                             state.positionM_[1] * 1_m, state.positionM_[2] * 1_m);
        auto node = environment_.getUniverse()->getContainingNode(position);
        if (!node || !node->hasModelProperties())
          throw std::runtime_error("EGS4 host vertex is outside the C8 medium");
        state.environmentNode_ = topology_.getNodeIndex(*node);
        auto p =
            particle_transfer::import_particle(stack, state, coordinates_, topology_);
        std::size_t returned = 0;
        if (r.kind_ == HostRequestKind::photonuclear_many_hadrons) {
          typename TStack::stack_view_type view(p);
          auto target = convert_from_PDG(static_cast<PDGCode>(r.targetPdg_));
          auto E = r.particle_.energyGeV_ * 1_GeV;
          FourMomentum photon(
              E, DirectionVector{coordinates_,
                                 {r.particle_.direction_[0], r.particle_.direction_[1],
                                  r.particle_.direction_[2]}} *
                     E);
          auto status =
              photons_.doHadronicPhotonInteraction(view, coordinates_, photon, target);
          if (status != ProcessReturn::Ok && status != ProcessReturn::Interacted)
            throw std::runtime_error(
                "C8 selected photonuclear final state did not complete");
          statistics_.generatedSecondaries_ += view.getSize();
          secondaries_.doSecondaries(view);
          returned = view.getSize();
          statistics_.retainedSecondaries_ += returned;
          p.erase();
          ++statistics_.selectedPhotonuclear_;
        } else {
          bool keep = arrival_(p, r);
          if (keep) {
            returned = 1;
            if (r.kind_ == HostRequestKind::muon_transport)
              ++statistics_.transportedMuons_;
            else
              ++statistics_.transportedHadrons_;
          } else {
            p.erase();
            ++statistics_.arrivalCuts_;
          }
        }
        consumed_.insert(r.particle_.historyId_);
        return returned;
      } catch (...) {
        failed_ = true;
        throw;
      }
    }
    HostStatistics const& statistics() const { return statistics_; }
  };
} // namespace corsika::kokkos::egs4::application
