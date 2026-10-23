/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/modules/radio/RadioTrack.hpp>

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

namespace corsika::accelerator::radio {

  namespace radio_spool_detail {
    inline constexpr std::array<char, 16> magic{
        'C', '8', '_', 'R', 'A', 'D', 'I', 'O', '_', 'T', 'R', 'A', 'C', 'K', 'S', '\0'};
    inline constexpr std::uint32_t version = 1;
    inline constexpr std::uint32_t recordBytes = 17 * 8 + 4;
    inline constexpr std::uint64_t headerBytes = 32;
    inline constexpr std::uint64_t incomplete = std::numeric_limits<std::uint64_t>::max();

    inline void require(bool ok, char const* message) {
      if (!ok) throw std::runtime_error(message);
    }

    template <typename UInt>
    void writeInteger(std::ostream& out, UInt value) {
      std::array<char, sizeof(UInt)> bytes{};
      for (std::size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<char>((value >> (8 * i)) & 0xff);
      out.write(bytes.data(), bytes.size());
      require(bool(out), "Cannot write radio track spool");
    }

    template <typename UInt>
    UInt readInteger(std::istream& in) {
      std::array<unsigned char, sizeof(UInt)> bytes{};
      in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      require(bool(in), "Truncated radio track spool");
      UInt value = 0;
      for (std::size_t i = 0; i < bytes.size(); ++i)
        value |= static_cast<UInt>(bytes[i]) << (8 * i);
      return value;
    }

    inline void writeDouble(std::ostream& out, double value) {
      static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
      std::uint64_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      writeInteger(out, bits);
    }

    inline double readDouble(std::istream& in) {
      auto const bits = readInteger<std::uint64_t>(in);
      double value;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    }

    inline void validate(corsika::radio::RadioTrackSegment const& track) {
      require(track.pdg == 11 || track.pdg == -11, "Non-electron radio spool record");
      for (auto const* coordinates : {track.start_position_m, track.end_position_m,
                                      track.start_direction, track.end_direction})
        for (unsigned axis = 0; axis < 3; ++axis)
          require(std::isfinite(coordinates[axis]), "Non-finite radio spool coordinates");
      require(std::isfinite(track.start_time_s) && std::isfinite(track.end_time_s) &&
                  std::isfinite(track.start_kinetic_energy_GeV) &&
                  std::isfinite(track.end_kinetic_energy_GeV) &&
                  std::isfinite(track.weight),
              "Non-finite radio spool scalar");
      require(track.weight >= 0. && track.start_kinetic_energy_GeV >= 0. &&
                  track.end_kinetic_energy_GeV >= 0. &&
                  track.end_time_s >= track.start_time_s,
              "Invalid radio spool energy, weight or time");
    }

    inline void writeTrack(std::ostream& out, corsika::radio::RadioTrackSegment const& track) {
      validate(track);
      // Explicit little-endian fields: no struct padding, pointers or host ABI.
      writeInteger(out, static_cast<std::uint32_t>(track.pdg));
      for (auto const* coordinates : {track.start_position_m, track.end_position_m,
                                      track.start_direction, track.end_direction})
        for (unsigned axis = 0; axis < 3; ++axis) writeDouble(out, coordinates[axis]);
      for (auto value : {track.start_time_s, track.end_time_s,
                         track.start_kinetic_energy_GeV, track.end_kinetic_energy_GeV,
                         track.weight})
        writeDouble(out, value);
    }

    inline corsika::radio::RadioTrackSegment readTrack(std::istream& in) {
      corsika::radio::RadioTrackSegment track{};
      auto const pdg = readInteger<std::uint32_t>(in);
      require(pdg == 11 || pdg == static_cast<std::uint32_t>(-11),
              "Non-electron radio spool record");
      track.pdg = pdg == 11 ? 11 : -11;
      for (auto* coordinates : {track.start_position_m, track.end_position_m,
                                track.start_direction, track.end_direction})
        for (unsigned axis = 0; axis < 3; ++axis) coordinates[axis] = readDouble(in);
      track.start_time_s = readDouble(in);
      track.end_time_s = readDouble(in);
      track.start_kinetic_energy_GeV = readDouble(in);
      track.end_kinetic_energy_GeV = readDouble(in);
      track.weight = readDouble(in);
      validate(track);
      return track;
    }
  } // namespace radio_spool_detail

  /** Closed, bounded-memory handoff of already transported CPU segments.
   * All positions/directions use the application's common root coordinate
   * system; meters, seconds and GeV are explicit in the segment fields.
   * This format transfers tracks only: it never advances a particle or draws
   * random numbers. The coordinator keeps all worker geometry options equal.
   */
  class RadioTrackSpoolWriter final : public corsika::radio::RadioTrackSink {
    std::ofstream file_;
    std::uint64_t count_{};
    bool completed_{};
  public:
    explicit RadioTrackSpoolWriter(std::filesystem::path const& path) {
      using namespace radio_spool_detail;
      require(!std::filesystem::exists(path), "Radio track spool already exists");
      file_.open(path, std::ios::binary | std::ios::out);
      require(bool(file_), "Cannot create radio track spool");
      file_.write(magic.data(), magic.size());
      writeInteger(file_, version);
      writeInteger(file_, recordBytes);
      writeInteger(file_, incomplete);
    }
    void append(corsika::radio::RadioTrackSegment const& track) override {
      using namespace radio_spool_detail;
      require(!completed_, "Cannot append to a completed radio track spool");
      require(count_ < (incomplete - headerBytes) / recordBytes,
              "Radio track spool size overflow");
      writeTrack(file_, track);
      ++count_;
    }
    void complete() {
      using namespace radio_spool_detail;
      require(!completed_, "Radio track spool completed twice");
      // Only the explicit successful completion publishes a valid count.
      // Destruction after an exception leaves the incomplete sentinel intact.
      file_.flush();
      require(bool(file_), "Cannot flush radio track spool");
      file_.seekp(24);
      writeInteger(file_, count_);
      file_.close();
      require(bool(file_), "Cannot close radio track spool");
      completed_ = true;
    }
    std::uint64_t count() const { return count_; }
  };

  class RadioTrackSpoolReader {
    std::ifstream file_;
    std::uint64_t count_{};
    std::uint64_t consumed_{};
  public:
    explicit RadioTrackSpoolReader(std::filesystem::path const& path) {
      using namespace radio_spool_detail;
      file_.open(path, std::ios::binary);
      require(bool(file_), "Cannot read radio track spool");
      std::array<char, 16> actual{};
      file_.read(actual.data(), actual.size());
      require(bool(file_) && actual == magic, "Invalid radio track spool magic");
      require(readInteger<std::uint32_t>(file_) == version,
              "Unsupported radio track spool version");
      require(readInteger<std::uint32_t>(file_) == recordBytes,
              "Invalid radio track spool record size");
      count_ = readInteger<std::uint64_t>(file_);
      require(count_ != incomplete && count_ <= (incomplete - headerBytes) / recordBytes,
              "Incomplete or oversized radio track spool");
      require(std::filesystem::file_size(path) == headerBytes + count_ * recordBytes,
              "Radio track spool length does not match its record count");
    }
    std::size_t readBatch(std::vector<corsika::radio::RadioTrackSegment>& tracks,
                          std::size_t maximum = 4096) {
      using namespace radio_spool_detail;
      require(maximum > 0 && maximum <= 65536, "Invalid radio spool read batch");
      tracks.clear();
      auto const amount = static_cast<std::size_t>(
          std::min<std::uint64_t>(maximum, count_ - consumed_));
      tracks.reserve(amount);
      for (std::size_t i = 0; i < amount; ++i) tracks.push_back(readTrack(file_));
      consumed_ += amount;
      return amount;
    }
    std::uint64_t count() const { return count_; }
    std::uint64_t consumed() const { return consumed_; }
    bool finished() const { return consumed_ == count_; }
  };
} // namespace corsika::accelerator::radio
