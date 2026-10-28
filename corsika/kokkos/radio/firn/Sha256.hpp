/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace corsika::kokkos::radio::firn {
  /** Sha256 from the validated stratified-radio implementation. */
  class Sha256 {
  public:
    void update(unsigned char const* input, std::size_t length);
    std::array<unsigned char, 32> finish();

  private:
    static std::uint32_t rotateRight(std::uint32_t value, unsigned count);
    void transform();

    std::array<unsigned char, 64> block_{};
    std::size_t blockLength_{0};
    std::uint64_t bitLength_{0};
    std::array<std::uint32_t, 8> state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                        0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                        0x1f83d9abu, 0x5be0cd19u};
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/Sha256.inl>

#include <corsika/detail/kokkos/radio/firn/Sha256Common.inl>
