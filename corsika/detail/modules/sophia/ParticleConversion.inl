/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>

#include <sophia.hpp>

namespace corsika::sophia {

  inline HEPMassType getSophiaMass(Code const pCode) {
    if (is_nucleus(pCode)) throw std::runtime_error("Not defined for nuclei.");
    auto sCode = convertToSophiaRaw(pCode);
    if (sCode == 0)
      throw std::runtime_error("getSophiaMass: unknown particle!");
    else
      return sqrt(get_sophia_mass2(sCode)) * 1_GeV;
  }
} // namespace corsika::sophia