/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace corsika::gpu::em::tables {

  using Sha256Digest = std::array<std::uint8_t, 32>;

  Sha256Digest sha256(std::uint8_t const* data, std::size_t size);

  inline Sha256Digest sha256(std::vector<std::uint8_t> const& data) {
    return sha256(data.data(), data.size());
  }

  std::string toHex(Sha256Digest const& digest);

} // namespace corsika::gpu::em::tables
