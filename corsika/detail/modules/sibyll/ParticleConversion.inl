/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <sibyll2.3d-public.hpp>

namespace corsika::sibyll {

  inline HEPMassType getSibyllMass(Code const pCode) {
    if (is_nucleus(pCode)) throw std::runtime_error("Not defined for nuclei.");
    auto sCode = convertToSibyllRaw(pCode);
    if (sCode == 0)
      throw std::runtime_error("getSibyllMass: unknown particle!");
    else
      return sqrt(::sibyll23d::get_sibyll_mass2(sCode)) * 1_GeV;
  }
} // namespace corsika::sibyll
