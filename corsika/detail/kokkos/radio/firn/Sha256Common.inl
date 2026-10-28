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

  inline std::string sha256_file(std::filesystem::path const& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot hash file: " + path.string());
    Sha256 hash;
    std::array<unsigned char, 1 << 16> buffer{};
    while (stream) {
      stream.read(reinterpret_cast<char*>(buffer.data()),
                  static_cast<std::streamsize>(buffer.size()));
      if (stream.gcount() > 0) {
        hash.update(buffer.data(), static_cast<std::size_t>(stream.gcount()));
      }
    }
    if (!stream.eof()) throw std::runtime_error("failed while hashing file");
    auto const digest = hash.finish();
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char value : digest) {
      output << std::setw(2) << static_cast<unsigned int>(value);
    }
    return output.str();
  }

} // namespace corsika::kokkos::radio::firn
