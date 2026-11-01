/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/egs4/Data.hpp>
#include <corsika/kokkos/em/execution/LeptonCascadeResult.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace corsika::kokkos::egs4::application {
  // Optional host-only diagnostic of the experiment's existing handoff records.
  // Never draws random numbers or changes the records sent to either backend.
  class MuonAudit {
    using particle_type = ::corsika::kokkos::ParticleState;
    std::ofstream out_;
    void vector(double const* v) {
      out_ << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
    }
    void state(particle_type const& p) {
      out_ << "{\"pid\":" << p.pid_ << ",\"history\":" << p.historyId_
           << ",\"parent\":" << p.parentHistoryId_ << ",\"step\":" << p.stepId_
           << ",\"generation\":" << p.generation_ << ",\"medium\":" << p.mediumId_
           << ",\"energy\":" << p.energyGeV_ << ",\"position\":";
      vector(p.positionM_);
      out_ << ",\"direction\":";
      vector(p.direction_);
      out_ << ",\"time\":" << p.timeS_ << ",\"weight\":" << p.weight_ << '}';
    }

  public:
    MuonAudit() {
      auto path = std::getenv("C8_EGS4_MUON_AUDIT");
      if (!path || !*path) return;
      if (std::filesystem::exists(path))
        throw std::runtime_error("Muon audit file already exists");
      out_.exceptions(std::ios::badbit | std::ios::failbit);
      out_.open(path);
      out_ << std::setprecision(17);
    }
    void particle(char const* kind, particle_type const& p) {
      if (!out_.is_open()) return;
      out_ << "{\"kind\":\"" << kind << "\",\"state\":";
      state(p);
      out_ << "}\n";
    }
    void wave(::corsika::kokkos::LeptonCascadeResult const& result) {
      if (!out_.is_open()) return;
      for (auto const& r : result.stepRecords_) {
        out_ << "{\"kind\":\"step\",\"start\":";
        state(r.start_);
        out_ << ",\"end\":";
        state(r.end_);
        out_ << ",\"limit\":" << int(r.limit_)
             << ",\"process\":" << r.interaction_.processId_
             << ",\"distance\":" << r.distanceM_
             << ",\"grammage\":" << r.traversedGrammageGPerCm2_
             << ",\"deposit\":" << r.continuousDepositedEnergyGeV_
             << ",\"scatter_angle\":" << r.multipleScatteringAngleRad_
             << ",\"scatter_u\":[" << r.multipleScatteringFirstUniform_ << ','
             << r.multipleScatteringSecondUniform_ << ','
             << r.multipleScatteringAzimuthUniform_
             << "],\"selection_u\":" << r.interaction_.proposalSelectionUniform_
             << ",\"loss_quantile\":" << r.interaction_.lossQuantile_ << "}\n";
      }
      for (auto const& e : result.fallbackEvents_) {
        out_ << "{\"kind\":\"fallback\",\"state\":";
        state(e.particle_);
        out_ << ",\"reason\":" << int(e.reason_) << ",\"process\":" << e.processId_
             << ",\"vertex_reached\":" << e.interactionVertexReached_
             << ",\"component_hash\":" << e.componentHash_
             << ",\"interaction_hash\":" << e.interactionHash_
             << ",\"energy_fraction\":" << e.energyFraction_
             << ",\"selection_u\":" << e.selectionUniform_
             << ",\"loss_quantile\":" << e.lossQuantile_
             << ",\"final_state_u\":" << e.finalStateUniform_
             << ",\"draw\":" << e.randomDrawId_
             << ",\"final_state_draw\":" << e.finalStateDrawId_ << "}\n";
      }
    }
  };
} // namespace corsika::kokkos::egs4::application
