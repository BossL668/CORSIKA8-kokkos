/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
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
    ShowerSummary(ShowerSummary const& other)
        : current_(YAML::Clone(other.current_)), last_offset_(other.last_offset_),
          last_id_(other.last_id_), has_event_(other.has_event_) {
      if (!other.journal_.empty()) {
        journal_ = other.journal_.parent_path() /
                   boost::filesystem::unique_path(".summary-%%%%-%%%%.yaml");
        boost::filesystem::copy_file(other.journal_, journal_);
      }
    }
    ShowerSummary& operator=(ShowerSummary const& other) {
      ShowerSummary copy(other);
      swap(copy);
      return *this;
    }

    ~ShowerSummary() {
      if (!journal_.empty()) {
        boost::system::error_code ignored;
        boost::filesystem::remove(journal_, ignored);
      }
    }

    void open(boost::filesystem::path const& directory) {
      auto path = directory / boost::filesystem::unique_path(".summary-%%%%-%%%%.yaml");
      std::ofstream stream(path.string(), std::ios::binary);
      stream.exceptions(std::ios::badbit | std::ios::failbit);
      stream.close();
      ShowerSummary fresh;
      fresh.journal_ = std::move(path);
      swap(fresh);
    }

    YAML::Node event(unsigned int id) {
      if (has_event_ && id < last_id_)
        throw std::logic_error("ShowerSummary cannot rewrite an earlier shower");
      if (!has_event_ || id != last_id_) {
        flush();
        last_offset_ = journal_.empty() ? 0 : boost::filesystem::file_size(journal_);
        last_id_ = id;
        has_event_ = true;
        current_.reset(YAML::Node(YAML::NodeType::Map));
      } else if (current_.IsNull()) {
        std::ifstream stream(journal_.string(), std::ios::binary);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream.seekg(static_cast<std::streamoff>(last_offset_));
        std::string text{std::istreambuf_iterator<char>(stream), {}};
        current_.reset(YAML::Load(text));
      }
      return current_["shower_" + std::to_string(id)];
    }

    void flush() const {
      if (current_.IsNull()) return;
      if (journal_.empty())
        throw std::logic_error("ShowerSummary requires startOfLibrary before flush");
      YAML::Emitter emitter;
      emitter << current_;
      if (!emitter.good()) throw std::runtime_error(emitter.GetLastError());
      std::string const text = std::string(emitter.c_str()) + "\n";
      std::fstream stream(journal_.string(), std::ios::in | std::ios::out | std::ios::binary);
      stream.exceptions(std::ios::badbit | std::ios::failbit);
      stream.seekp(static_cast<std::streamoff>(last_offset_));
      stream.write(text.data(), static_cast<std::streamsize>(text.size()));
      stream.flush();
      stream.close();
      boost::filesystem::resize_file(journal_, last_offset_ + text.size());
      current_.reset();
    }

    YAML::Node snapshot() const {
      if (!has_event_) return YAML::Node();
      // Some diagnostic callers inspect an uninitialized output. No disk
      // operation is necessary until a library has actually been opened.
      if (journal_.empty()) return YAML::Clone(current_);
      flush();
      return YAML::LoadFile(journal_.string());
    }

    void writeSummary(boost::filesystem::path const& destination) const {
      flush();
      if (!has_event_) return;
      auto temporary = destination;
      temporary += ".tmp";
      boost::filesystem::copy_file(journal_, temporary,
                                   boost::filesystem::copy_options::overwrite_existing);
      boost::filesystem::rename(temporary, destination);
    }

    bool hasResidentRecord() const noexcept { return !current_.IsNull(); }

  private:
    void swap(ShowerSummary& other) {
      using std::swap;
      swap(journal_, other.journal_);
      auto current = current_;
      current_.reset(other.current_);
      other.current_.reset(current);
      swap(last_offset_, other.last_offset_);
      swap(last_id_, other.last_id_);
      swap(has_event_, other.has_event_);
    }
    boost::filesystem::path journal_;
    mutable YAML::Node current_;
    std::uintmax_t last_offset_{};
    unsigned int last_id_{};
    bool has_event_{};
  };
} // namespace corsika
