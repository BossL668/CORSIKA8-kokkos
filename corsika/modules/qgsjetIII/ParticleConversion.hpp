/*
 * (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>

#include <string>
#include <cstdint>

namespace corsika::qgsjetIII {

  /**
   * These are the possible secondaries produced by QGSJetII.
   */
  enum class QgsjetIIICode : int8_t;
  using QgsjetIIICodeIntType = std::underlying_type<QgsjetIIICode>::type;

  /**
   * These are the possible projectile for which QGSJetII knwos cross section.
   */
  enum class QgsjetIIIXSClass : int8_t {
    CannotInteract = 0,
    LightMesons = 1,
    Baryons = 2,
    Kaons = 3,
  };
  using QgsjetIIIXSClassIntType = std::underlying_type<QgsjetIIIXSClass>::type;

  /**
   *  These are the only possible projectile types in QGSJetII.
   */
  enum class QgsjetIIIHadronType : int8_t {
    UndefinedType = 0,
    PiPlusType = +1,
    PiMinusType = -1,
    ProtonType = +2,
    AntiProtonType = -2,
    NeutronType = +3,
    AntiNeutronType = -3,
    KaonPlusType = +4,
    KaonMinusType = -4,
    Kaon0LType = +5,
    Kaon0SType = -5,
    // special codes, not in QGSJetII
    NucleusType = 100,
    NeutralLightMesonType = 101,
  };
  using QgsjetIIIHadronTypeIntType = std::underlying_type<QgsjetIIIHadronType>::type;
} // namespace corsika::qgsjetIII

// include automatically generated code:
#include <corsika/modules/qgsjetIII/Generated.inc>

namespace corsika::qgsjetIII {

  QgsjetIIICode constexpr convertToQgsjetIII(Code const pCode) {
    return corsika2qgsjetIII[static_cast<CodeIntType>(pCode)];
  }

  Code constexpr convertFromQgsjetIII(QgsjetIIICode const code) {
    auto const codeInt = static_cast<QgsjetIIICodeIntType>(code);
    auto const corsikaCode = qgsjetIII2corsika[codeInt - minQgsjetIII];
    if (corsikaCode == Code::Unknown) {
      throw std::runtime_error(std::string("QGSJETIII/CORSIKA conversion of pCodeInt=")
                                   .append(std::to_string(codeInt))
                                   .append(" impossible"));
    }
    return corsikaCode;
  }

  QgsjetIIICodeIntType constexpr convertToQgsjetIIIRaw(Code const code) {
    return static_cast<QgsjetIIICodeIntType>(convertToQgsjetIII(code));
  }

  QgsjetIIIXSClass constexpr getQgsjetIIIXSCode(Code const code) {
    return is_nucleus(code) ? QgsjetIIIXSClass::Baryons
                            : corsika2qgsjetIIIXStype[static_cast<CodeIntType>(code)];
  }

  QgsjetIIIXSClassIntType constexpr getQgsjetIIIXSCodeRaw(Code const code) {
    return is_nucleus(code)
               ? static_cast<QgsjetIIIXSClassIntType>(QgsjetIIIXSClass::Baryons)
               : static_cast<QgsjetIIIXSClassIntType>(getQgsjetIIIXSCode(code));
  }

  bool constexpr canInteract(Code const code) {
    if (is_nucleus(code)) return true;
    return getQgsjetIIIXSCode(code) != QgsjetIIIXSClass::CannotInteract;
  }

  QgsjetIIIHadronType constexpr getQgsjetIIIHadronType(Code const code) {
    if (is_nucleus(code)) return QgsjetIIIHadronType::NucleusType;
    return corsika2qgsjetIIIHadronType[static_cast<CodeIntType>(code)];
  }

} // namespace corsika::qgsjetIII
