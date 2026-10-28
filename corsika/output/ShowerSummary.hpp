/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace corsika {

  /** Disk-backed, ordered shower_N mapping. Only the current event can be
   * modified. flush() releases its YAML graph, not just its logical length.
   * snapshot() intentionally materializes the whole library for explicit API
   * consumers; OutputManager uses writeSummary() instead.
   *
   * The last entry may be rewritten (e.g. complete -> incomplete on failure).
   * This preserves unique YAML keys without retaining earlier events in RAM.
   * The journal is private to this object and lives beside the output, never
   * in a global /tmp cache. Failed writes throw before releasing the record.
   */
  class ShowerSummary {
  public:
    ShowerSummary() = default;
    // Geometry adapters copy their (normally uninitialized) output objects.
    // If already opened, make an independent journal, never share ownership.
    ShowerSummary(ShowerSummary const& other);
    ShowerSummary& operator=(ShowerSummary const& other);

    ~ShowerSummary();

    void open(boost::filesystem::path const& directory);

    YAML::Node event(unsigned int id);

    void flush() const;

    YAML::Node snapshot() const;

    void writeSummary(boost::filesystem::path const& destination) const;

    bool hasResidentRecord() const noexcept;

  private:
    void swap(ShowerSummary& other);
    boost::filesystem::path journal_;
    mutable YAML::Node current_;
    std::uintmax_t lastOffset_{};
    unsigned int lastId_{};
    bool hasEvent_{};
  };
} // namespace corsika

#include <corsika/detail/output/ShowerSummary.inl>
