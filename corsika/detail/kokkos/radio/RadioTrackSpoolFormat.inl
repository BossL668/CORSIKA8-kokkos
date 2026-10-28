/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::radio_spool {
  inline void require(bool ok, char const* message) {
    if (!ok) throw std::runtime_error(message);
  }

  template <typename UInt>
  void write_integer(std::ostream& out, UInt value) {
    std::array<char, sizeof(UInt)> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<char>((value >> (8 * i)) & 0xff);
    out.write(bytes.data(), bytes.size());
    require(bool(out), "Cannot write radio track spool");
  }

  template <typename UInt>
  UInt read_integer(std::istream& in) {
    std::array<unsigned char, sizeof(UInt)> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    require(bool(in), "Truncated radio track spool");
    UInt value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i)
      value |= static_cast<UInt>(bytes[i]) << (8 * i);
    return value;
  }

  inline void write_double(std::ostream& out, double value) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    write_integer(out, bits);
  }

  inline double read_double(std::istream& in) {
    auto const bits = read_integer<std::uint64_t>(in);
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  inline void validate(corsika::radio::RadioTrackSegment const& track) {
    require(track.pdg_ == 11 || track.pdg_ == -11, "Non-electron radio spool record");
    for (auto const* coordinates : {track.startPositionM_, track.endPositionM_,
                                    track.startDirection_, track.endDirection_})
      for (unsigned axis = 0; axis < 3; ++axis)
        require(std::isfinite(coordinates[axis]), "Non-finite radio spool coordinates");
    require(std::isfinite(track.startTimeS_) && std::isfinite(track.endTimeS_) &&
                std::isfinite(track.startKineticEnergyGeV_) &&
                std::isfinite(track.endKineticEnergyGeV_) && std::isfinite(track.weight_),
            "Non-finite radio spool scalar");
    require(track.weight_ >= 0. && track.startKineticEnergyGeV_ >= 0. &&
                track.endKineticEnergyGeV_ >= 0. && track.endTimeS_ >= track.startTimeS_,
            "Invalid radio spool energy, weight or time");
  }

  inline void write_track(std::ostream& out,
                          corsika::radio::RadioTrackSegment const& track) {
    validate(track);
    // Explicit little-endian fields: no struct padding, pointers or host ABI.
    write_integer(out, static_cast<std::uint32_t>(track.pdg_));
    for (auto const* coordinates : {track.startPositionM_, track.endPositionM_,
                                    track.startDirection_, track.endDirection_})
      for (unsigned axis = 0; axis < 3; ++axis) write_double(out, coordinates[axis]);
    for (auto value : {track.startTimeS_, track.endTimeS_, track.startKineticEnergyGeV_,
                       track.endKineticEnergyGeV_, track.weight_})
      write_double(out, value);
  }

  inline corsika::radio::RadioTrackSegment read_track(std::istream& in) {
    corsika::radio::RadioTrackSegment track{};
    auto const pdg = read_integer<std::uint32_t>(in);
    require(pdg == 11 || pdg == static_cast<std::uint32_t>(-11),
            "Non-electron radio spool record");
    track.pdg_ = pdg == 11 ? 11 : -11;
    for (auto* coordinates : {track.startPositionM_, track.endPositionM_,
                              track.startDirection_, track.endDirection_})
      for (unsigned axis = 0; axis < 3; ++axis) coordinates[axis] = read_double(in);
    track.startTimeS_ = read_double(in);
    track.endTimeS_ = read_double(in);
    track.startKineticEnergyGeV_ = read_double(in);
    track.endKineticEnergyGeV_ = read_double(in);
    track.weight_ = read_double(in);
    validate(track);
    return track;
  }
} // namespace corsika::kokkos::radio::radio_spool
