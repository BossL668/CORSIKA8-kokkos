/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

namespace corsika::epos {

  enum class EposCode : int32_t;
  using EposCodeIntType = std::underlying_type<EposCode>::type;

  /**
   * These are the possible projectile for which Epos knows the cross section.
   */
  enum class EposXSClass : int8_t {
    CannotInteract = 0,
    Baryon = 2,
    Pion = 1,
    Kaon = 3,
    Charm = 4,
  };
  using EposXSClassIntType = std::underlying_type<EposXSClass>::type;

#include <corsika/modules/epos/Generated.inc>

  unsigned int constexpr get_nucleus_A(EposCode const eposId) {
    // 100ZZZAAA0 -> std. pdg code
    EposCodeIntType const eposPdg = static_cast<EposCodeIntType>(eposId);
    return int(eposPdg / 10) % 1000;
  }
  unsigned int constexpr get_nucleus_Z(EposCode const eposId) {
    // 100ZZZAAA0 -> std. pdg code
    EposCodeIntType const eposPdg = static_cast<EposCodeIntType>(eposId);
    return int(eposPdg / 10000) % 1000;
  }

  EposCode constexpr convertToEpos(Code const code) {
    return corsika2epos[static_cast<CodeIntType>(code)];
  }

  Code constexpr convertFromEpos(EposCode const eposId) {
    EposCodeIntType const s = static_cast<EposCodeIntType>(eposId);
    // if nucleus (pdg-id)
    if (s >= 1000000000) {
      return get_nucleus_code(get_nucleus_A(eposId), get_nucleus_Z(eposId));
    }
    auto const corsikaCode = epos2corsika[s - minEpos];
    if (corsikaCode == Code::Unknown) {
      throw std::runtime_error(std::string("EPOS/CORSIKA conversion of ")
                                   .append(std::to_string(s))
                                   .append(" impossible"));
    }
    return corsikaCode;
  }

  int constexpr convertToEposRaw(Code const code) {
    return static_cast<int>(convertToEpos(code));
  }

  int constexpr getEposXSCode(Code const code) {
    if (is_nucleus(code)) { return static_cast<EposXSClassIntType>(EposXSClass::Baryon); }
    return static_cast<EposXSClassIntType>(
        corsika2eposXStype[static_cast<CodeIntType>(code)]);
  }

  bool constexpr canInteract(Code const code) {
    return is_nucleus(code) || getEposXSCode(code) > 0;
  }

  HEPMassType getEposMass(Code const);

  PDGCode getEposPDGId(Code const);

} // namespace corsika::epos

#include <corsika/detail/modules/epos/ParticleConversion.inl>
