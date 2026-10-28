/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  inline char const* em_process_name(std::int32_t process_id) noexcept {
    switch (process_id) {
      case BremsProcessId:
        return "bremsstrahlung";
      case IonizationProcessId:
        return "ionization";
      case ElectronPairProcessId:
        return "electron_pair_production";
      case PhotonuclearProcessId:
        return "photonuclear";
      case MuonPairProcessId:
        return "lepton_muon_pair";
      case HadronsProcessId:
        return "hadrons";
      case WeakInteractionProcessId:
        return "weak_interaction";
      case ComptonProcessId:
        return "compton";
      case DecayProcessId:
        return "decay";
      case AnnihilationProcessId:
        return "annihilation";
      case PhotonPairProcessId:
        return "photon_pair_production";
      case PhotonPairFinalStateProcessId:
        return "photon_pair_final_state";
      case PhotoproductionProcessId:
        return "photoproduction";
      case PhotonMuonPairProcessId:
        return "photon_muon_pair";
      case PhotoelectricProcessId:
        return "photoelectric";
      default:
        return "unknown";
    }
  }

  CORSIKA_HOST_DEVICE inline ProcessCapability process_capability(
      std::int32_t pid, std::int32_t process_id) {
    if (pid == static_cast<std::int32_t>(EmPid::Photon) &&
        process_id == PhotonMuonPairProcessId) {
      return ProcessCapability::PhotonMuonPair;
    }
    if (pid == static_cast<std::int32_t>(EmPid::Photon) &&
        process_id == PhotonPairProcessId) {
      return ProcessCapability::PhotonPair;
    }
    if ((pid == static_cast<std::int32_t>(EmPid::Electron) ||
         pid == static_cast<std::int32_t>(EmPid::Positron)) &&
        process_id == BremsProcessId) {
      return ProcessCapability::Bremsstrahlung;
    }
    if (pid == static_cast<std::int32_t>(EmPid::Photon) &&
        process_id == ComptonProcessId) {
      return ProcessCapability::Compton;
    }
    if (pid == static_cast<std::int32_t>(EmPid::Photon) &&
        process_id == PhotoelectricProcessId) {
      return ProcessCapability::Photoelectric;
    }
    if (pid == static_cast<std::int32_t>(EmPid::Positron) &&
        process_id == AnnihilationProcessId) {
      return ProcessCapability::Annihilation;
    }
    if (is_charged_lepton_pid(pid) && process_id == IonizationProcessId) {
      return ProcessCapability::Ionization;
    }
    if (is_charged_lepton_pid(pid) && process_id == ElectronPairProcessId) {
      // No native rho CDF table: complete e±/mu± pair sharing with CPU
      // PROPOSAL at the selected vertex, without device rejection sampling.
      return ProcessCapability::CpuOnlyFinalState;
    }
    if (is_muon_pid(pid) && process_id == BremsProcessId) {
      return ProcessCapability::Bremsstrahlung;
    }
    switch (process_id) {
      case PhotonuclearProcessId:
      case MuonPairProcessId:
      case HadronsProcessId:
      case WeakInteractionProcessId:
      case DecayProcessId:
      case PhotoproductionProcessId:
        return ProcessCapability::CpuOnlyFinalState;
      default:
        return ProcessCapability::GpuFinalStateNotImplemented;
    }
  }

  CORSIKA_HOST_DEVICE inline FallbackReason process_fallback_reason(
      ProcessCapability capability) {
    return capability == ProcessCapability::CpuOnlyFinalState
               ? FallbackReason::CpuOnlyProcess
               : FallbackReason::GpuProcessNotImplemented;
  }

  CORSIKA_HOST_DEVICE inline bool proposal_native_selection_requires_replay(
      std::int32_t pid, std::int32_t process_id, double residual_quantile) {
    auto const capability = process_capability(pid, process_id);
    if (capability == ProcessCapability::CpuOnlyFinalState) return true;
    // BKK consumes the photon exactly (v=1); an endpoint residual quantile
    // needs no host replay. Its independent secondary x is sampled later.
    if (capability == ProcessCapability::PhotonMuonPair) return false;
    // An endpoint replay is a numerical completion mechanism for processes
    // already covered by either a declared GPU final state or a declared
    // CPU-only final state.  It must never turn an unregistered process into
    // an implicit CPU fallback merely because its quantile lies near an
    // endpoint.
    if (capability == ProcessCapability::GpuFinalStateNotImplemented) return false;
    return residual_quantile < NativeSelectionReplayEndpointWidth ||
           residual_quantile > 1. - NativeSelectionReplayEndpointWidth;
  }

} // namespace corsika::kokkos
