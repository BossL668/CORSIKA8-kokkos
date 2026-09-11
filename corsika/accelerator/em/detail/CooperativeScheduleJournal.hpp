/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/accelerator/em/detail/CooperativeScheduling.hpp>
#include <istream>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>

namespace corsika::accelerator::em::detail {

  enum class CooperativeAction : unsigned { Dispatch, Checkpoint, Migration, Commit, Failure };

  struct CooperativeScheduleRecord {
    CooperativeAction action{};
    CooperativeEndpoint endpoint{};
    std::uint64_t batch{};
    std::uint64_t particles{};
    CooperativeHistoryRange history{};
    // SHA-256 of canonical input states, supplied by the coordinator. The
    // journal does not reassign IDs or recompute any particle's random draws.
    std::string state_sha256;
  };

  inline bool scheduleDigestValid(std::string const& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
  }
  inline void validateScheduleRecord(CooperativeScheduleRecord const& record) {
    if (static_cast<unsigned>(record.action) > static_cast<unsigned>(CooperativeAction::Failure) ||
        static_cast<unsigned>(record.endpoint) > 1 || !record.batch ||
        !record.history.first || record.history.limit <= record.history.first ||
        !scheduleDigestValid(record.state_sha256))
      throw std::invalid_argument("invalid cooperative schedule record");
  }

  /** Streaming, bounded-memory diagnostic of scheduler decisions.
   * This is not a physics decision tape: it checks the order/identity of
   * submitted states, not their per-process random draws or floating results.
   * Truncated, unfinished, or different-configuration files fail closed.
   */
  class CooperativeScheduleWriter {
   public:
    CooperativeScheduleWriter(std::ostream& output, std::string const& configuration_sha256)
        : output_(output) {
      if (!scheduleDigestValid(configuration_sha256))
        throw std::invalid_argument("invalid cooperative schedule configuration hash");
      write("C8COOP 1 " + configuration_sha256 + "\n");
    }
    void append(CooperativeScheduleRecord const& record) {
      if (finished_) throw std::logic_error("cooperative schedule already closed");
      validateScheduleRecord(record);
      if (records_ == UINT64_MAX) throw std::overflow_error("schedule ordinal overflow");
      std::ostringstream line;
      line.imbue(std::locale::classic());
      line << "R " << records_ << ' ' << static_cast<unsigned>(record.action) << ' '
           << static_cast<unsigned>(record.endpoint) << ' ' << record.batch << ' '
           << record.particles << ' ' << record.history.first << ' ' << record.history.limit
           << ' ' << record.state_sha256 << '\n';
      write(line.str());
      ++records_;
    }
    void finish() {
      if (finished_) throw std::logic_error("cooperative schedule already closed");
      write("END " + std::to_string(records_) + "\n");
      output_.flush();
      if (!output_) throw std::runtime_error("cooperative schedule flush failed");
      finished_ = true;
    }
    std::uint64_t records() const noexcept { return records_; }
   private:
    void write(std::string const& text) {
      output_.write(text.data(), static_cast<std::streamsize>(text.size()));
      if (!output_) throw std::runtime_error("cooperative schedule write failed");
    }
    std::ostream& output_;
    std::uint64_t records_{};
    bool finished_{};
  };

  class CooperativeScheduleReader {
   public:
    CooperativeScheduleReader(std::istream& input, std::string const& configuration_sha256)
        : input_(input) {
      if (!scheduleDigestValid(configuration_sha256) ||
          line() != "C8COOP 1 " + configuration_sha256)
        throw std::runtime_error("cooperative schedule version/configuration mismatch");
    }
    CooperativeScheduleRecord next() {
      if (finished_) throw std::logic_error("cooperative replay already finished");
      if (failed_) throw std::logic_error("cooperative replay previously failed");
      try {
        auto const text = line();
        // Unsigned stream extraction accepts negative input modulo 2^N.
        // No negative value is valid in this unsigned-only format.
        if (text.find('-') != std::string::npos)
          throw std::runtime_error("negative cooperative schedule field");
        std::istringstream entry(text);
        entry.imbue(std::locale::classic());
        char tag{};
        std::uint64_t ordinal{};
        unsigned action{}, endpoint{};
        CooperativeScheduleRecord record;
        if (!(entry >> tag >> ordinal >> action >> endpoint >> record.batch
                    >> record.particles >> record.history.first
                    >> record.history.limit >> record.state_sha256) ||
            tag != 'R' || ordinal != records_ || !(entry >> std::ws).eof())
          throw std::runtime_error("invalid cooperative schedule at record " +
                                   std::to_string(records_));
        record.action = static_cast<CooperativeAction>(action);
        record.endpoint = static_cast<CooperativeEndpoint>(endpoint);
        validateScheduleRecord(record);
        if (records_ == UINT64_MAX) throw std::overflow_error("schedule ordinal overflow");
        ++records_;
        return record;
      } catch (...) {
        failed_ = true;
        throw;
      }
    }
    void expect(CooperativeScheduleRecord const& actual) {
      validateScheduleRecord(actual);
      auto const ordinal = records_;
      auto const expected = next();
      char const* field = nullptr;
      if (actual.action != expected.action) field = "action";
      else if (actual.endpoint != expected.endpoint) field = "endpoint";
      else if (actual.batch != expected.batch) field = "batch";
      else if (actual.particles != expected.particles) field = "particles";
      else if (actual.history.first != expected.history.first ||
               actual.history.limit != expected.history.limit) field = "history";
      else if (actual.state_sha256 != expected.state_sha256) field = "state_sha256";
      if (field) {
        failed_ = true;
        throw std::runtime_error("first cooperative schedule divergence at record " +
                                 std::to_string(ordinal) + ": " + field);
      }
    }
    void finish() {
      if (finished_) throw std::logic_error("cooperative replay already finished");
      if (failed_) throw std::logic_error("cooperative replay previously failed");
      if (line() != "END " + std::to_string(records_) || input_.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("unfinished or trailing cooperative schedule");
      finished_ = true;
    }
   private:
    // Never let a malformed/oversized record grow an unbounded host string.
    std::string line() {
      std::string result;
      for (unsigned i = 0; i < 512; ++i) {
        auto const c = input_.get();
        if (c == std::char_traits<char>::eof())
          throw std::runtime_error("truncated cooperative schedule");
        if (c == '\n') return result;
        result.push_back(static_cast<char>(c));
      }
      throw std::runtime_error("oversized cooperative schedule record");
    }
    std::istream& input_;
    std::uint64_t records_{};
    bool finished_{};
    bool failed_{};
  };
} // namespace corsika::accelerator::em::detail
