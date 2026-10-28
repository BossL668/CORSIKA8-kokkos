/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

#include <sophia.hpp>

#include <string>

namespace corsika::sophia {

  enum class SophiaCode : int8_t;
  using SophiaCodeIntType = std::underlying_type<SophiaCode>::type;

#include <corsika/modules/sophia/Generated.inc>

  SophiaCode constexpr convertToSophia(Code const pCode) {
    return corsika2sophia[static_cast<CodeIntType>(pCode)];
  }

  Code constexpr convertFromSophia(SophiaCode const pCode) {
    auto const s = static_cast<SophiaCodeIntType>(pCode);
    auto const corsikaCode = sophia2corsika[s - minSophia];
    if (corsikaCode == Code::Unknown) {
      throw std::runtime_error(std::string("SOPHIA/CORSIKA conversion of ")
                                   .append(std::to_string(s))
                                   .append(" impossible"));
    }
    return corsikaCode;
  }

  int constexpr convertToSophiaRaw(Code const code) {
    return static_cast<int>(convertToSophia(code));
  }

  bool constexpr canInteract(Code const pCode) {
    return (pCode == Code::Photon ? true : false);
  }

  HEPMassType getSophiaMass(Code const);

} // namespace corsika::sophia

#include <corsika/detail/modules/sophia/ParticleConversion.inl>
