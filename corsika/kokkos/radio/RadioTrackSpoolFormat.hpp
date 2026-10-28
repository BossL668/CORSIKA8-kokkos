/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/radio/RadioTrackSink.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace corsika::kokkos::radio::radio_spool {

  inline constexpr std::array<char, 16> magic{'C', '8', '_', 'R', 'A', 'D', 'I', 'O',
                                              '_', 'T', 'R', 'A', 'C', 'K', 'S', '\0'};
  inline constexpr std::uint32_t version = 1;
  inline constexpr std::uint32_t recordBytes = 17 * 8 + 4;
  inline constexpr std::uint64_t headerBytes = 32;
  inline constexpr std::uint64_t incomplete = std::numeric_limits<std::uint64_t>::max();

  inline void require(bool ok, char const* message);

  template <typename UInt>
  void write_integer(std::ostream& out, UInt value);

  template <typename UInt>
  UInt read_integer(std::istream& in);

  inline void write_double(std::ostream& out, double value);

  inline double read_double(std::istream& in);

  inline void validate(corsika::radio::RadioTrackSegment const& track);

  inline void write_track(std::ostream& out,
                          corsika::radio::RadioTrackSegment const& track);

  inline corsika::radio::RadioTrackSegment read_track(std::istream& in);

} // namespace corsika::kokkos::radio::radio_spool

#include <corsika/detail/kokkos/radio/RadioTrackSpoolFormat.inl>
