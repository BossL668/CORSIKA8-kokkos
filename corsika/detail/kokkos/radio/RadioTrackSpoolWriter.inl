/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  inline RadioTrackSpoolWriter::RadioTrackSpoolWriter(std::filesystem::path const& path) {
    using namespace radio_spool;
    require(!std::filesystem::exists(path), "Radio track spool already exists");
    file_.open(path, std::ios::binary | std::ios::out);
    require(bool(file_), "Cannot create radio track spool");
    file_.write(magic.data(), magic.size());
    write_integer(file_, version);
    write_integer(file_, recordBytes);
    write_integer(file_, incomplete);
  }

  inline void RadioTrackSpoolWriter::append(
      corsika::radio::RadioTrackSegment const& track) {
    using namespace radio_spool;
    require(!completed_, "Cannot append to a completed radio track spool");
    require(count_ < (incomplete - headerBytes) / recordBytes,
            "Radio track spool size overflow");
    write_track(file_, track);
    ++count_;
  }

  inline void RadioTrackSpoolWriter::complete() {
    using namespace radio_spool;
    require(!completed_, "Radio track spool completed twice");
    // Only the explicit successful completion publishes a valid count.
    // Destruction after an exception leaves the incomplete sentinel intact.
    file_.flush();
    require(bool(file_), "Cannot flush radio track spool");
    file_.seekp(24);
    write_integer(file_, count_);
    file_.close();
    require(bool(file_), "Cannot close radio track spool");
    completed_ = true;
  }

  inline std::uint64_t RadioTrackSpoolWriter::count() const { return count_; }
} // namespace corsika::kokkos::radio
