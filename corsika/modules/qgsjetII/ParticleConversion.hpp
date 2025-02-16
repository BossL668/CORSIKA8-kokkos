/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>

#include <cstdint>
#include <string>

namespace corsika::qgsjetII {

  /**
   * These are the possible secondaries produced by QGSJetII.
   */
  enum class QgsjetIICode : int8_t;
  using QgsjetIICodeIntType = std::underlying_type<QgsjetIICode>::type;

  /**
   * These are the possible projectile for which QGSJetII knwos cross section.
   */
  enum class QgsjetIIXSClass : int8_t {
    CannotInteract = 0,
    LightMesons = 1,
    Baryons = 2,
    Kaons = 3,
  };
  using QgsjetIIXSClassIntType = std::underlying_type<QgsjetIIXSClass>::type;

  /**
   *  These are the only possible projectile types in QGSJetII.
   */
  enum class QgsjetIIHadronType : int8_t {
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
  using QgsjetIIHadronTypeIntType = std::underlying_type<QgsjetIIHadronType>::type;
} // namespace corsika::qgsjetII

// include automatically generated code:
#include <corsika/modules/qgsjetII/Generated.inc>

namespace corsika::qgsjetII {

  QgsjetIICode constexpr convertToQgsjetII(Code const pCode) {
    return corsika2qgsjetII[static_cast<CodeIntType>(pCode)];
  }

  Code constexpr convertFromQgsjetII(QgsjetIICode const code) {
    auto const codeInt = static_cast<QgsjetIICodeIntType>(code);
    auto const corsikaCode = qgsjetII2corsika[codeInt - minQgsjetII];
    if (corsikaCode == Code::Unknown) {
      throw std::runtime_error(std::string("QGSJETII/CORSIKA conversion of pCodeInt=")
                                   .append(std::to_string(codeInt))
                                   .append(" impossible"));
    }
    return corsikaCode;
  }

  QgsjetIICodeIntType constexpr convertToQgsjetIIRaw(Code const code) {
    return static_cast<QgsjetIICodeIntType>(convertToQgsjetII(code));
  }

  QgsjetIIXSClass constexpr getQgsjetIIXSCode(Code const code) {
    return is_nucleus(code) ? QgsjetIIXSClass::Baryons
                            : corsika2qgsjetIIXStype[static_cast<CodeIntType>(code)];
  }

  QgsjetIIXSClassIntType constexpr getQgsjetIIXSCodeRaw(Code const code) {
    return is_nucleus(code)
               ? static_cast<QgsjetIIXSClassIntType>(QgsjetIIXSClass::Baryons)
               : static_cast<QgsjetIIXSClassIntType>(getQgsjetIIXSCode(code));
  }

  bool constexpr canInteract(Code const code) {
    if (is_nucleus(code)) return true;
    return getQgsjetIIXSCode(code) != QgsjetIIXSClass::CannotInteract;
  }

  QgsjetIIHadronType constexpr getQgsjetIIHadronType(Code const code) {
    if (is_nucleus(code)) return QgsjetIIHadronType::NucleusType;
    return corsika2qgsjetIIHadronType[static_cast<CodeIntType>(code)];
  }

} // namespace corsika::qgsjetII
