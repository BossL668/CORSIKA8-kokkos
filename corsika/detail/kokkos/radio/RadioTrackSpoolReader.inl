/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  inline RadioTrackSpoolReader::RadioTrackSpoolReader(std::filesystem::path const& path) {
    using namespace radio_spool;
    file_.open(path, std::ios::binary);
    require(bool(file_), "Cannot read radio track spool");
    std::array<char, 16> actual{};
    file_.read(actual.data(), actual.size());
    require(bool(file_) && actual == magic, "Invalid radio track spool magic");
    require(read_integer<std::uint32_t>(file_) == version,
            "Unsupported radio track spool version");
    require(read_integer<std::uint32_t>(file_) == recordBytes,
            "Invalid radio track spool record size");
    count_ = read_integer<std::uint64_t>(file_);
    require(count_ != incomplete && count_ <= (incomplete - headerBytes) / recordBytes,
            "Incomplete or oversized radio track spool");
    require(std::filesystem::file_size(path) == headerBytes + count_ * recordBytes,
            "Radio track spool length does not match its record count");
  }

  inline std::size_t RadioTrackSpoolReader::readBatch(
      std::vector<corsika::radio::RadioTrackSegment>& tracks, std::size_t maximum) {
    using namespace radio_spool;
    require(maximum > 0 && maximum <= 65536, "Invalid radio spool read batch");
    tracks.clear();
    auto const amount =
        static_cast<std::size_t>(std::min<std::uint64_t>(maximum, count_ - consumed_));
    tracks.reserve(amount);
    for (std::size_t i = 0; i < amount; ++i) tracks.push_back(read_track(file_));
    consumed_ += amount;
    return amount;
  }

  inline std::uint64_t RadioTrackSpoolReader::count() const { return count_; }

  inline std::uint64_t RadioTrackSpoolReader::consumed() const { return consumed_; }

  inline bool RadioTrackSpoolReader::finished() const { return consumed_ == count_; }
} // namespace corsika::kokkos::radio
