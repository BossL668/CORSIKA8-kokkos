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

namespace corsika::kokkos {

  using sha256_digest_type = std::array<std::uint8_t, 32>;

  /** Calculate the digest of a host byte vector without changing its contents. */
  [[nodiscard]] sha256_digest_type sha256(std::vector<std::uint8_t> const& data);

  /** Encode a digest as lowercase hexadecimal. */
  [[nodiscard]] std::string to_hex(sha256_digest_type const& digest);

} // namespace corsika::kokkos
