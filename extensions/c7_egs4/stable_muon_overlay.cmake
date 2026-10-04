# Compile-time, experiment-only overlay. Never write into the production tree.
set(muon_step_relative corsika/accelerator/em/detail/LeptonFinalStateStep.hpp)
file(READ "${EGS4_C8_SOURCE}/${muon_step_relative}" muon_step_source)
set(muon_original [=[      deflectLeptonChild(parent.direction, outgoing_cosine, azimuth,
                         outgoing_direction);
      deflectLeptonChild(parent.direction, delta_cosine,
                         leptonFmod(azimuth + LeptonPi, LeptonTwoPi),
                         delta_direction);]=])
set(muon_stable [=[      if (isMuonPid(parent.pid)) {
        auto const angles = ::c7_egs4::application::stableMuonIonizationAngles(
            parent.energy_GeV, parent.energy_GeV * v, lepton_mass_GeV, electron_mass_GeV);
        ::c7_egs4::application::deflectWithSine(parent.direction, angles.muon_sine,
            angles.muon_cosine, azimuth, outgoing_direction);
        ::c7_egs4::application::deflectWithSine(parent.direction, angles.electron_sine,
            angles.electron_cosine, leptonFmod(azimuth + LeptonPi, LeptonTwoPi), delta_direction);
      } else {
        deflectLeptonChild(parent.direction, outgoing_cosine, azimuth, outgoing_direction);
        deflectLeptonChild(parent.direction, delta_cosine,
            leptonFmod(azimuth + LeptonPi, LeptonTwoPi), delta_direction);
      }]=])
string(FIND "${muon_step_source}" "${muon_original}" muon_anchor)
if(muon_anchor LESS 0)
  message(FATAL_ERROR "Muon ionization source changed: re-audit the isolated overlay")
endif()
string(REPLACE "${muon_original}" "${muon_stable}" muon_step_source "${muon_step_source}")
string(REPLACE "#pragma once" "#pragma once\n#include \"StableMuonIonization.hpp\"" muon_step_source "${muon_step_source}")
set(muon_overlay_root "${CMAKE_CURRENT_BINARY_DIR}/stable_muon_overlay")
file(MAKE_DIRECTORY "${muon_overlay_root}/corsika/accelerator/em/detail")
set(muon_overlay_previous "")
if(EXISTS "${muon_overlay_root}/${muon_step_relative}")
  file(READ "${muon_overlay_root}/${muon_step_relative}" muon_overlay_previous)
endif()
if(NOT muon_overlay_previous STREQUAL muon_step_source)
  file(WRITE "${muon_overlay_root}/${muon_step_relative}" "${muon_step_source}")
endif()
target_include_directories(egs4_proposal_muon BEFORE PUBLIC "${muon_overlay_root}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(c8_egs4_shower PRIVATE EGS4_STABLE_MUON_IONIZATION=1)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${EGS4_C8_SOURCE}/${muon_step_relative}")
