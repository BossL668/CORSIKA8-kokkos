/*
 * (c) Copyright 2023 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <fmt/core.h>

#include <string>

namespace corsika::fluka {

  enum class FLUKACode : int;
  using FLUKACodeIntType = std::underlying_type<FLUKACode>::type;

#include <corsika/modules/fluka/Generated.inc>

  FLUKACode constexpr convertToFluka(Code const c8id) {
    if (is_nucleus(c8id)) {
      throw std::runtime_error{"nucleus conversion to FLUKA not implemented"};
    }

    FLUKACode const flukaID = corsika2fluka[static_cast<corsika::CodeIntType>(c8id)];
    if (flukaID == FLUKACode::Unknown) {
      throw std::runtime_error{
          fmt::format("no correspondig FLUKA id for {}", get_name(c8id)).c_str()};
    }

    return flukaID;
  }

  int constexpr convertToFlukaRaw(Code const code) {
    return static_cast<FLUKACodeIntType>(convertToFluka(code));
  }

  inline bool const canInteract(Code const code) {
    if (is_nucleus(code)) return false; // nuclei support not yet implemented
    return flukaCanInteract[static_cast<CodeIntType>(code)];
  }
} // namespace corsika::fluka
