/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

#include <sibyll2.3d-public.hpp>

#include <string>

namespace corsika::sibyll {

  enum class SibyllCode : int8_t;
  using SibyllCodeIntType = std::underlying_type<SibyllCode>::type;

  /**
     These are the possible projectile for which Sibyll knows the cross section
   */
  enum class SibyllXSClass : int8_t {
    CrossSectionUnknown = 0,
    Baryon = 1,
    Pion = 2,
    Kaon = 3,
  };
  using SibyllXSClassIntType = std::underlying_type<SibyllXSClass>::type;

#include <corsika/modules/sibyll/Generated.inc>

  SibyllCode constexpr convertToSibyll(Code const pCode) {
    return corsika2sibyll[static_cast<CodeIntType>(pCode)];
  }

  Code constexpr convertFromSibyll(SibyllCode const pCode) {
    auto const s = static_cast<SibyllCodeIntType>(pCode);
    auto const corsikaCode = sibyll2corsika[s - minSibyll];
    if (corsikaCode == Code::Unknown) {
      throw std::runtime_error(std::string("SIBYLL/CORSIKA conversion of ")
                                   .append(std::to_string(s))
                                   .append(" impossible"));
    }
    return corsikaCode;
  }

  int constexpr convertToSibyllRaw(Code const code) {
    return static_cast<int>(convertToSibyll(code));
  }

  // find which cross section to use for this particle. maps to either proton, pion or
  // kaon or unknown
  int constexpr getSibyllXSCode(Code const code) {
    if (is_nucleus(code))
      return static_cast<SibyllXSClassIntType>(SibyllXSClass::CrossSectionUnknown);
    return static_cast<SibyllXSClassIntType>(
        corsika2sibyllXStype[static_cast<CodeIntType>(code)]);
  }

  // find if interaction can be generated.
  bool constexpr canInteract(Code const pCode) {
    if (is_nucleus(pCode)) return false;
    return caninteract[static_cast<CodeIntType>(pCode)];
  }

  HEPMassType getSibyllMass(Code const);

} // namespace corsika::sibyll

#include <corsika/detail/modules/sibyll/ParticleConversion.inl>
